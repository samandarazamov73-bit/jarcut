"""JarCut — локальный автоматический дубляж видео через Vertex AI Gemini 3.1.

Запуск: python server.py
Открыть: http://localhost:8000
"""

import base64
import difflib
import importlib.util
import json
import math
import os
import re
import struct
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()


def bounded_env_seconds(name: str, default: float, minimum: float, maximum: float) -> float:
    """Read a timeout setting without allowing invalid values to break startup."""
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    return max(minimum, min(value, maximum))


# Vertex AI credentials. GEMINI_API_KEY remains a backward-compatible alias so
# existing local .env files keep working when that value is actually a Vertex key.
API_KEY = (os.getenv("VERTEX_API_KEY") or os.getenv("GEMINI_API_KEY", "")).strip()
VERTEX_MODE = os.getenv("VERTEX_MODE", "auto").strip().lower()
VERTEX_PROJECT_ID = os.getenv("VERTEX_PROJECT_ID", "").strip()
VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", "global").strip() or "global"
TRANSCRIBE_MODEL = os.getenv(
    "VERTEX_TRANSCRIBE_MODEL",
    os.getenv("GEMINI_TRANSCRIBE_MODEL", "gemini-3.1-pro-preview"),
)
TTS_MODEL = os.getenv(
    "VERTEX_TTS_MODEL",
    os.getenv("GEMINI_TTS_MODEL", "gemini-3.1-flash-tts-preview"),
)
ALIGNMENT_MODE = os.getenv("ALIGNMENT_MODE", "hybrid").strip().lower()
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small").strip() or "small"
ALIGNMENT_LOW_CONFIDENCE = 0.45
SPEAKER_VERIFICATION = os.getenv("SPEAKER_VERIFICATION", "auto").strip().lower()
SPEAKER_EMBEDDING_MODEL = os.getenv(
    "SPEAKER_EMBEDDING_MODEL", "speechbrain/spkrec-ecapa-voxceleb"
).strip() or "speechbrain/spkrec-ecapa-voxceleb"
VERTEX_TRANSCRIBE_TIMEOUT_SECONDS = bounded_env_seconds(
    "VERTEX_TRANSCRIBE_TIMEOUT_SECONDS", 240.0, 60.0, 600.0
)

BASE = Path(__file__).resolve().parent
UPLOADS = BASE / "uploads"
EXPORTS = BASE / "exports"
for directory in (UPLOADS, EXPORTS):
    directory.mkdir(exist_ok=True)

app = FastAPI(title="JarCut", version="4.0")

LANGS = {
    "uz": "Uzbek (O'zbek tili)", "ru": "Russian", "en": "English",
    "tr": "Turkish", "kk": "Kazakh", "ky": "Kyrgyz", "tg": "Tajik",
    "ar": "Arabic", "de": "German", "fr": "French", "es": "Spanish",
    "ko": "Korean", "ja": "Japanese",
}

VOICES = {
    "Kore", "Puck", "Charon", "Fenrir", "Aoede", "Leda", "Orus",
    "Zephyr", "Callirrhoe", "Autonoe", "Enceladus", "Iapetus", "Umbriel",
    "Algieba", "Despina", "Erinome", "Algenib", "Rasalgethi", "Laomedeia",
    "Achernar", "Alnilam", "Schedar", "Gacrux", "Pulcherrima", "Achird",
    "Zubenelgenubi", "Vindemiatrix", "Sadachbia", "Sadaltager", "Sulafat",
}

VOICE_POOLS = {
    "masculine": ["Charon", "Orus", "Fenrir", "Iapetus", "Algenib", "Rasalgethi", "Enceladus"],
    "feminine": ["Kore", "Aoede", "Leda", "Autonoe", "Callirrhoe", "Despina", "Erinome", "Laomedeia"],
    "young": ["Puck", "Zephyr", "Achird", "Pulcherrima", "Gacrux"],
    "neutral": ["Puck", "Zephyr", "Algieba", "Umbriel", "Achernar", "Alnilam"],
}

MIME_TYPES = {
    ".mp4": "video/mp4", ".webm": "video/webm", ".mov": "video/quicktime",
    ".mkv": "video/x-matroska", ".avi": "video/x-msvideo",
    ".mp3": "audio/mpeg", ".wav": "audio/wav", ".m4a": "audio/mp4",
    ".ogg": "audio/ogg", ".aac": "audio/aac",
}


@app.get("/")
async def root() -> FileResponse:
    return FileResponse(str(BASE / "index.html"))


def resolved_vertex_mode() -> str:
    if VERTEX_MODE in {"express", "standard"}:
        return VERTEX_MODE
    if VERTEX_MODE == "auto":
        return "standard" if VERTEX_PROJECT_ID else "express"
    raise ValueError(
        "VERTEX_MODE должен быть express, standard или auto "
        f"(сейчас: {VERTEX_MODE or 'пусто'})"
    )


def resolved_alignment_mode() -> str:
    if ALIGNMENT_MODE not in {"off", "fast", "hybrid", "precise"}:
        raise ValueError(
            "ALIGNMENT_MODE должен быть off, fast, hybrid или precise "
            f"(сейчас: {ALIGNMENT_MODE or 'пусто'})"
        )
    return ALIGNMENT_MODE


def resolved_speaker_verification_mode() -> str:
    if SPEAKER_VERIFICATION not in {"off", "auto", "on"}:
        raise ValueError(
            "SPEAKER_VERIFICATION должен быть off, auto или on "
            f"(сейчас: {SPEAKER_VERIFICATION or 'пусто'})"
        )
    return SPEAKER_VERIFICATION


def speaker_verification_available() -> bool:
    return all(importlib.util.find_spec(name) is not None for name in ("speechbrain", "torch"))


def vertex_model_url(model: str) -> str:
    """Build a Vertex AI Express or Standard GenerateContent endpoint."""
    mode = resolved_vertex_mode()
    if mode == "express":
        return f"https://aiplatform.googleapis.com/v1/publishers/google/models/{model}:generateContent"

    if not VERTEX_PROJECT_ID:
        raise ValueError("Для VERTEX_MODE=standard укажите VERTEX_PROJECT_ID в .env")
    if VERTEX_LOCATION == "global":
        host = "https://aiplatform.googleapis.com"
    else:
        host = f"https://{VERTEX_LOCATION}-aiplatform.googleapis.com"
    return (
        f"{host}/v1/projects/{VERTEX_PROJECT_ID}/locations/{VERTEX_LOCATION}"
        f"/publishers/google/models/{model}:generateContent"
    )


def vertex_headers() -> dict[str, str]:
    return {"Content-Type": "application/json", "x-goog-api-key": API_KEY}


def vertex_public_config() -> dict[str, Any]:
    alignment_mode = resolved_alignment_mode()
    speaker_verification_mode = resolved_speaker_verification_mode()
    alignment_dependency_available = importlib.util.find_spec("faster_whisper") is not None
    identity_available = speaker_verification_available()
    return {
        "backend": "Vertex AI",
        "mode": resolved_vertex_mode(),
        "project": VERTEX_PROJECT_ID or "express-mode",
        "location": VERTEX_LOCATION if resolved_vertex_mode() == "standard" else "global",
        "transcribeModel": TRANSCRIBE_MODEL,
        "ttsModel": TTS_MODEL,
        "transcribeTimeoutSeconds": VERTEX_TRANSCRIBE_TIMEOUT_SECONDS,
        "alignmentMode": alignment_mode,
        "alignmentAvailable": alignment_mode == "off" or alignment_dependency_available,
        "alignmentDependencyAvailable": alignment_dependency_available,
        "whisperModel": WHISPER_MODEL if alignment_mode in {"hybrid", "precise"} else "disabled",
        "speakerVerificationMode": speaker_verification_mode,
        "speakerVerificationAvailable": identity_available,
        "speakerEmbeddingModel": (
            SPEAKER_EMBEDDING_MODEL
            if speaker_verification_mode != "off" and identity_available
            else "disabled"
        ),
    }


def api_error(response: httpx.Response, action: str) -> HTTPException:
    """Translate Vertex AI errors into actionable messages without exposing the key."""
    try:
        payload = response.json()
        error = payload.get("error", {})
        message = error.get("message", response.text[:300])
        reasons = [
            str(item.get("reason", ""))
            for item in error.get("details", [])
            if isinstance(item, dict)
        ]
    except Exception:
        message = response.text[:300]
        reasons = []

    mode = resolved_vertex_mode()
    if response.status_code in (401, 403):
        if mode == "express":
            help_steps = [
                "Проверьте, что VERTEX_API_KEY — ключ Vertex AI Express Mode, а не Google AI Studio.",
                "Если ключ ограничен, разрешите ему Vertex AI API (aiplatform.googleapis.com).",
                "Для локального Python-сервера не используйте ограничение Website/HTTP referrers.",
                "После изменения .env полностью перезапустите run.bat.",
            ]
        else:
            help_steps = [
                "Включите Vertex AI API: https://console.cloud.google.com/apis/library/aiplatform.googleapis.com",
                "Проверьте, что Vertex API key привязан к service account с доступом Vertex AI User.",
                "Проверьте VERTEX_PROJECT_ID и убедитесь, что ключ создан в этом же Google Cloud проекте.",
                "В ограничениях ключа разрешите Vertex AI API (aiplatform.googleapis.com).",
                "Проверьте billing, доступ к моделям и VERTEX_LOCATION; затем перезапустите run.bat.",
            ]
        return HTTPException(
            status_code=response.status_code,
            detail={
                "code": "VERTEX_PERMISSION_DENIED",
                "message": f"Vertex AI запретил {action}: {message}",
                "help": help_steps,
                "action": action,
                "reasons": reasons,
            },
        )

    if response.status_code == 400 and ("API key" in message or "api key" in message.lower()):
        return HTTPException(
            status_code=400,
            detail={
                "code": "INVALID_VERTEX_API_KEY",
                "message": message,
                "help": [
                    "Вставьте действующий Vertex AI API key в VERTEX_API_KEY без кавычек и пробелов.",
                    "Документация: https://cloud.google.com/vertex-ai/generative-ai/docs/start/api-keys",
                ],
                "action": action,
            },
        )

    if response.status_code == 404:
        return HTTPException(
            status_code=404,
            detail={
                "code": "VERTEX_MODEL_NOT_AVAILABLE",
                "message": f"Vertex AI не нашёл модель или endpoint: {message}",
                "help": [
                    f"Проверьте доступ проекта к {TRANSCRIBE_MODEL} и {TTS_MODEL}.",
                    "Для Standard Mode попробуйте VERTEX_LOCATION=global и проверьте VERTEX_PROJECT_ID.",
                ],
                "action": action,
            },
        )

    return HTTPException(
        status_code=response.status_code,
        detail={"code": "VERTEX_ERROR", "message": message, "help": [], "action": action},
    )


@app.get("/api/diagnostics")
async def diagnostics() -> dict[str, Any]:
    """Test the exact Vertex GenerateContent method JarCut uses."""
    try:
        public_config = vertex_public_config()
    except ValueError as exc:
        message = str(exc)
        alignment_error = "ALIGNMENT_MODE" in message
        speaker_error = "SPEAKER_VERIFICATION" in message
        return {
            "backend": "Vertex AI",
            "mode": VERTEX_MODE or "invalid",
            "project": VERTEX_PROJECT_ID or "not-set",
            "location": VERTEX_LOCATION,
            "transcribeModel": TRANSCRIBE_MODEL,
            "ttsModel": TTS_MODEL,
            "alignmentMode": ALIGNMENT_MODE or "invalid",
            "alignmentAvailable": importlib.util.find_spec("faster_whisper") is not None,
            "checkMethod": "GenerateContent",
            "ok": False,
            "code": (
                "ALIGNMENT_CONFIG_ERROR"
                if alignment_error
                else "SPEAKER_CONFIG_ERROR"
                if speaker_error
                else "VERTEX_CONFIG_ERROR"
            ),
            "message": message,
            "help": [
                "Укажите ALIGNMENT_MODE=off, fast, hybrid или precise в .env."
                if alignment_error
                else "Укажите SPEAKER_VERIFICATION=off, auto или on в .env."
                if speaker_error
                else "Укажите VERTEX_MODE=express, standard или auto в .env."
            ],
        }

    base_result = {
        **public_config,
        "checkMethod": "GenerateContent",
    }
    if not API_KEY or API_KEY in {"ваш_ключ_сюда", "ваш_vertex_ключ"}:
        return {
            **base_result,
            "ok": False,
            "code": "NO_KEY",
            "message": "В .env не вставлен VERTEX_API_KEY.",
            "help": ["Вставьте ключ: VERTEX_API_KEY=ваш_vertex_ключ и перезапустите run.bat."],
        }
    try:
        endpoint = vertex_model_url(TRANSCRIBE_MODEL)
    except ValueError as exc:
        return {
            **base_result,
            "ok": False,
            "code": "VERTEX_CONFIG_ERROR",
            "message": str(exc),
            "help": ["Добавьте VERTEX_PROJECT_ID в .env или переключите VERTEX_MODE=express."],
        }

    payload = {
        "contents": [{"role": "user", "parts": [{"text": "Reply with exactly OK."}]}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 128},
    }
    try:
        async with httpx.AsyncClient(timeout=45) as client:
            response = await client.post(endpoint, headers=vertex_headers(), json=payload)
    except httpx.RequestError as exc:
        return {
            **base_result,
            "ok": False,
            "code": "VERTEX_UNREACHABLE",
            "message": f"Vertex AI недоступен: {exc}",
            "help": ["Проверьте интернет, VPN/прокси и нажмите «Проверить» ещё раз."],
            "retryable": True,
        }

    if response.status_code != 200:
        exc = api_error(response, "GenerateContent")
        return {**base_result, "ok": False, **exc.detail}

    return {
        **base_result,
        "ok": True,
        "keyValid": True,
        "generateContentAvailable": True,
        "ttsUnchecked": True,
        "message": f"Vertex AI {TRANSCRIBE_MODEL}: GenerateContent работает. TTS проверится при первой озвучке.",
        "help": [],
    }


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> dict[str, Any]:
    suffix = Path(file.filename or "video.mp4").suffix.lower() or ".mp4"
    name = f"{uuid.uuid4().hex[:12]}{suffix}"
    content = await file.read()
    (UPLOADS / name).write_bytes(content)
    return {"ok": True, "filename": name, "url": f"/uploads/{name}", "size": len(content)}


def extract_audio(source: Path, target: Path) -> bool:
    """Create a compact mono audio track for reliable transcription and diarization."""
    command = [
        "ffmpeg", "-y", "-i", str(source), "-vn", "-ac", "1", "-ar", "16000",
        "-codec:a", "libmp3lame", "-b:a", "32k", str(target),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=600)
        return result.returncode == 0 and target.exists() and target.stat().st_size > 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _mark_overlap_segments(segments: list[dict[str, Any]]) -> None:
    """Mark Gemini overlap and group it for optional Whisper disambiguation."""
    count = len(segments)
    parents = list(range(count))

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = root(left), root(right)
        if left_root != right_root:
            parents[right_root] = left_root

    overlap_pairs: list[tuple[int, int]] = []
    for segment in segments:
        segment["is_overlap"] = False
        segment["gemini_overlap_candidate"] = False

    for index, segment in enumerate(segments):
        start = float(segment.get("start", 0))
        end = float(segment.get("end", start))
        speaker_id = str(segment.get("speaker_id", ""))
        for other_index in range(index + 1, count):
            other = segments[other_index]
            if str(other.get("speaker_id", "")) == speaker_id:
                continue
            overlap = min(end, float(other.get("end", 0))) - max(
                start, float(other.get("start", 0))
            )
            if overlap >= 0.08:
                segment["is_overlap"] = True
                other["is_overlap"] = True
                overlap_pairs.append((index, other_index))
                union(index, other_index)

    groups: dict[int, list[int]] = {}
    involved = {index for pair in overlap_pairs for index in pair}
    for index in involved:
        groups.setdefault(root(index), []).append(index)

    for group_number, indices in enumerate(
        sorted(groups.values(), key=lambda items: min(items))
    ):
        ordered = sorted(
            indices,
            key=lambda item: (
                float(segments[item].get("start", 0)),
                float(segments[item].get("end", 0)),
            ),
        )
        window_start = max(
            0.0, min(float(segments[item].get("start", 0)) for item in ordered) - 0.6
        )
        window_end = max(float(segments[item].get("end", 0)) for item in ordered) + 0.6
        for position, index in enumerate(ordered):
            segment = segments[index]
            segment.update({
                "gemini_overlap_candidate": True,
                "gemini_overlap_group_id": group_number,
                "transition_window_start": round(window_start, 3),
                "transition_window_end": round(window_end, 3),
                "speaker_transition_detected": position > 0,
                "speaker_transition_reference": position + 1 < len(ordered),
            })


def run_silero_vad(source: Path) -> tuple[Any | None, list[dict[str, float]], str | None]:
    """Decode media and run the Silero v6 ONNX model bundled with faster-whisper."""
    try:
        from faster_whisper.audio import decode_audio
        from faster_whisper.vad import VadOptions, get_speech_timestamps
    except ImportError:
        return None, [], "faster-whisper не установлен"

    try:
        audio = decode_audio(str(source), sampling_rate=16000)
        timestamps = get_speech_timestamps(
            audio,
            VadOptions(
                threshold=0.5,
                min_speech_duration_ms=80,
                min_silence_duration_ms=100,
                speech_pad_ms=15,
                max_speech_duration_s=30,
            ),
            sampling_rate=16000,
        )
        regions = [
            {"start": item["start"] / 16000, "end": item["end"] / 16000}
            for item in timestamps
            if item["end"] > item["start"]
        ]
        return audio, regions, None
    except Exception as exc:
        return None, [], f"Silero VAD не запустился: {exc}"


def _match_components(segment: dict[str, Any], region: dict[str, float]) -> tuple[float, float, float]:
    g_start, g_end = float(segment["start"]), float(segment["end"])
    v_start, v_end = region["start"], region["end"]
    intersection = max(0.0, min(g_end, v_end) - max(g_start, v_start))
    union = max(g_end, v_end) - min(g_start, v_start)
    iou = intersection / union if union > 0 else 0.0
    start_score = max(0.0, 1.0 - abs(g_start - v_start) / 1.5)
    g_duration, v_duration = max(0.1, g_end - g_start), max(0.1, v_end - v_start)
    duration_score = max(0.0, 1.0 - abs(g_duration - v_duration) / max(g_duration, v_duration))
    return iou, start_score, duration_score


def _region_match_score(segment: dict[str, Any], region: dict[str, float]) -> float:
    iou, start_score, duration_score = _match_components(segment, region)
    overlap_penalty = 2.0 if segment.get("is_overlap") else 0.0
    return 3.0 * iou + 2.0 * start_score + 1.5 * duration_score + 0.5 - overlap_penalty


def _assign_regions_monotonic(
    segments: list[dict[str, Any]], regions: list[dict[str, float]]
) -> list[int | None]:
    """Assign ordered segments to non-decreasing VAD regions with reuse penalties."""
    if not segments or not regions:
        return [None] * len(segments)

    count = len(regions)
    previous = [
        _region_match_score(segments[0], region) - 0.25 * index
        for index, region in enumerate(regions)
    ]
    parents: list[list[int]] = []

    for segment in segments[1:]:
        current = [-math.inf] * count
        row_parents = [0] * count
        for region_index, region in enumerate(regions):
            best_score, best_previous = -math.inf, 0
            for previous_index in range(region_index + 1):
                reuse_penalty = 1.0 if previous_index == region_index else 0.0
                skipped_penalty = 0.25 * max(0, region_index - previous_index - 1)
                candidate = previous[previous_index] - reuse_penalty - skipped_penalty
                if candidate > best_score:
                    best_score, best_previous = candidate, previous_index
            current[region_index] = best_score + _region_match_score(segment, region)
            row_parents[region_index] = best_previous
        previous = current
        parents.append(row_parents)

    region_index = max(range(count), key=lambda index: previous[index])
    assignments = [region_index]
    for row in reversed(parents):
        region_index = row[region_index]
        assignments.append(region_index)
    assignments.reverse()
    return assignments


def _energy_split_boundaries(
    audio: Any,
    region: dict[str, float],
    group: list[dict[str, Any]],
    sampling_rate: int = 16000,
) -> list[float] | None:
    """Find speech onsets after local energy valleys inside one shared VAD region."""
    try:
        import numpy as np
    except ImportError:
        return None

    if len(group) < 2:
        return []
    start_sample = max(0, int(region["start"] * sampling_rate))
    end_sample = min(len(audio), int(region["end"] * sampling_rate))
    signal = audio[start_sample:end_sample]
    frame, hop = int(0.02 * sampling_rate), int(0.01 * sampling_rate)
    if len(signal) < frame * 3:
        return None

    energies = np.array([
        float(np.sqrt(np.mean(signal[offset:offset + frame] ** 2) + 1e-12))
        for offset in range(0, len(signal) - frame + 1, hop)
    ])
    boundaries: list[float] = []
    for left, right in zip(group, group[1:]):
        expected = (float(left["end"]) + float(right["start"])) / 2
        search_start = max(region["start"] + 0.08, expected - 0.45)
        search_end = min(region["end"] - 0.08, expected + 0.45)
        first = max(0, int((search_start - region["start"]) * sampling_rate / hop))
        last = min(len(energies), int((search_end - region["start"]) * sampling_rate / hop) + 1)
        if last - first < 3:
            return None
        local = energies[first:last]
        valley_relative = int(np.argmin(local))
        valley_index = first + valley_relative
        median = float(np.median(local))
        valley = float(energies[valley_index])
        depth_db = 20 * math.log10((median + 1e-9) / (valley + 1e-9))
        if depth_db < 6.0:
            return None
        rise_threshold = valley + 0.35 * max(0.0, median - valley)
        onset_index = valley_index
        while onset_index + 1 < last and energies[onset_index] < rise_threshold:
            onset_index += 1
        boundary = region["start"] + onset_index * hop / sampling_rate
        if boundaries and boundary - boundaries[-1] < 0.08:
            return None
        boundaries.append(boundary)
    return boundaries


def _alignment_confidence(
    segment: dict[str, Any],
    region: dict[str, float],
    shared: bool,
    split_succeeded: bool,
) -> float:
    iou, start_score, duration_score = _match_components(segment, region)
    confidence = 0.45 * iou + 0.35 * start_score + 0.20 * duration_score
    if segment.get("is_overlap"):
        confidence *= 0.55
    if shared:
        confidence *= 0.85 if split_succeeded else 0.55
    if start_score < 0.2 or iou < 0.05:
        confidence = min(confidence, 0.35)
    return round(max(0.0, min(1.0, confidence)), 3)


def align_segments_with_vad(
    segments: list[dict[str, Any]], audio: Any, regions: list[dict[str, float]]
) -> list[dict[str, Any]]:
    # Overlapping voices cannot be separated by VAD, so exclude them from the
    # monotonic path rather than letting them distort neighboring assignments.
    normal_indices = [index for index, segment in enumerate(segments) if not segment.get("is_overlap")]
    normal_segments = [segments[index] for index in normal_indices]
    normal_assignments = _assign_regions_monotonic(normal_segments, regions)
    assignments: list[int | None] = [None] * len(segments)
    for index, assignment in zip(normal_indices, normal_assignments):
        assignments[index] = assignment

    weak_vad_indices: set[int] = set()
    for index, assignment in enumerate(assignments):
        if assignment is None:
            continue
        iou, start_score, _ = _match_components(segments[index], regions[assignment])
        if iou < 0.03 and start_score < 0.25:
            assignments[index] = None
            weak_vad_indices.add(index)

    # If a region already has a genuine overlap match, do not let a nearby
    # zero-overlap segment join and split that region away from the real line.
    provisional_groups: dict[int, list[int]] = {}
    for index, assignment in enumerate(assignments):
        if assignment is not None:
            provisional_groups.setdefault(assignment, []).append(index)
    for assignment, indices in provisional_groups.items():
        if len(indices) < 2:
            continue
        overlap_by_index = {
            index: _match_components(segments[index], regions[assignment])[0]
            for index in indices
        }
        if any(iou >= 0.03 for iou in overlap_by_index.values()):
            for index, iou in overlap_by_index.items():
                if iou < 0.03:
                    assignments[index] = None
                    weak_vad_indices.add(index)

    grouped: dict[int, list[int]] = {}
    for index, assignment in enumerate(assignments):
        if assignment is not None:
            grouped.setdefault(assignment, []).append(index)

    for index, segment in enumerate(segments):
        segment["gemini_start"] = float(segment["start"])
        segment["gemini_end"] = float(segment["end"])
        segment["manual_offset"] = 0.0
        assignment = assignments[index]
        if assignment is None:
            is_overlap = bool(segment.get("is_overlap"))
            warning = "overlap" if is_overlap else "weak_vad_match" if index in weak_vad_indices else "no_vad_match"
            segment.update({
                "auto_start": segment["gemini_start"],
                "auto_end": segment["gemini_end"],
                "alignment_method": "gemini_fallback",
                "alignment_confidence": 0.2 if is_overlap else 0.0,
                "alignment_warning": warning,
            })
            continue

        region = regions[assignment]
        indices = grouped[assignment]
        group = [segments[item] for item in indices]
        position = indices.index(index)
        shared = len(group) > 1
        speaker_ids = {str(item.get("speaker_id", "")) for item in group}
        cross_speaker_shared = shared and len(speaker_ids) > 1
        previous_in_group = group[position - 1] if position > 0 else None
        next_in_group = group[position + 1] if position + 1 < len(group) else None
        transition_from_previous = bool(
            previous_in_group
            and str(previous_in_group.get("speaker_id", ""))
            != str(segment.get("speaker_id", ""))
        )
        transition_to_next = bool(
            next_in_group
            and str(next_in_group.get("speaker_id", ""))
            != str(segment.get("speaker_id", ""))
        )
        segment.update({
            "matched_vad_region_id": assignment,
            "matched_vad_start": round(float(region["start"]), 3),
            "matched_vad_end": round(float(region["end"]), 3),
            "shared_vad_cross_speaker": cross_speaker_shared,
            "speaker_transition_detected": transition_from_previous,
            "speaker_transition_reference": transition_to_next,
        })

        # VAD only detects speech activity, not who is speaking. A shared region
        # containing different speaker IDs must never be split by an energy dip:
        # that dip may be an inhale or pause inside the previous speaker's line.
        # Preserve Gemini timing until ordered Whisper text anchors confirm the turn.
        if cross_speaker_shared:
            segment.update({
                "auto_start": segment["gemini_start"],
                "auto_end": segment["gemini_end"],
                "alignment_method": "gemini_fallback_speaker_transition",
                "alignment_confidence": 0.0,
                "alignment_warning": (
                    "speaker_boundary_pending"
                    if transition_from_previous
                    else "speaker_transition_shared_region"
                ),
            })
            continue

        boundaries = _energy_split_boundaries(audio, region, group) if shared else []
        split_succeeded = boundaries is not None

        if not shared:
            auto_start, auto_end = region["start"], region["end"]
        elif split_succeeded:
            edges = [region["start"], *boundaries, region["end"]]
            auto_start, auto_end = edges[position], edges[position + 1]
        else:
            group_start = min(float(item["start"]) for item in group)
            group_end = max(float(item["end"]) for item in group)
            span = max(0.1, group_end - group_start)
            scale = (region["end"] - region["start"]) / span
            auto_start = region["start"] + (float(segment["start"]) - group_start) * scale
            auto_end = region["start"] + (float(segment["end"]) - group_start) * scale

        auto_start = max(0.0, round(auto_start, 3))
        auto_end = max(auto_start + 0.2, round(auto_end, 3))
        confidence = _alignment_confidence(segment, region, shared, split_succeeded)
        warning = None
        if shared:
            warning = "split_region" if split_succeeded else "split_uncertain"
        segment.update({
            "auto_start": auto_start,
            "auto_end": auto_end,
            "alignment_method": "silero",
            "alignment_confidence": confidence,
            "alignment_warning": warning,
        })

    return segments


def _normalize_words(text: str) -> list[str]:
    return [
        token
        for token in re.sub(r"[^\w\s]", " ", str(text or "").casefold(), flags=re.UNICODE).split()
        if token
    ]


def merge_alignment_windows(segments: list[dict[str, Any]], precise_all: bool = False) -> list[dict[str, Any]]:
    candidates = [
        segment for segment in segments
        if segment.get("gemini_overlap_candidate")
        or (
            not segment.get("is_overlap")
            and (
                precise_all
                or float(segment.get("alignment_confidence", 0)) < ALIGNMENT_LOW_CONFIDENCE
            )
        )
    ]
    candidates.sort(key=lambda item: item["gemini_start"])
    windows: list[dict[str, Any]] = []
    for segment in candidates:
        if segment.get("gemini_overlap_candidate"):
            start = max(0.0, float(segment.get("transition_window_start", segment["gemini_start"] - 0.6)))
            end = float(segment.get("transition_window_end", segment["gemini_end"] + 0.6))
        elif segment.get("shared_vad_cross_speaker"):
            start = max(0.0, float(segment.get("matched_vad_start", segment["gemini_start"])) - 0.3)
            end = float(segment.get("matched_vad_end", segment["gemini_end"])) + 0.3
        else:
            start = max(0.0, float(segment["gemini_start"]) - 0.6)
            end = float(segment["gemini_end"]) + 0.6
        if windows and start - windows[-1]["end"] <= 1.0:
            windows[-1]["end"] = max(windows[-1]["end"], end)
            windows[-1]["segment_ids"].append(id(segment))
        else:
            windows.append({"start": start, "end": end, "segment_ids": [id(segment)]})
    return windows


_WHISPER_INSTANCE: Any = None


def _get_whisper_model() -> Any:
    global _WHISPER_INSTANCE
    if _WHISPER_INSTANCE is None:
        from faster_whisper import WhisperModel
        model_cache = BASE / ".models"
        model_cache.mkdir(exist_ok=True)
        _WHISPER_INSTANCE = WhisperModel(
            WHISPER_MODEL,
            device="cpu",
            compute_type="int8",
            cpu_threads=max(1, min(4, os.cpu_count() or 1)),
            download_root=str(model_cache),
        )
    return _WHISPER_INSTANCE


def run_whisper_for_windows(source: Path, windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not windows:
        return []
    model = _get_whisper_model()
    clip_timestamps: list[float] = []
    for window in windows:
        clip_timestamps.extend([window["start"], window["end"]])
    transcription, _ = model.transcribe(
        str(source),
        word_timestamps=True,
        clip_timestamps=clip_timestamps,
        beam_size=3,
        condition_on_previous_text=False,
        vad_filter=False,
    )
    words: list[dict[str, Any]] = []
    for item in transcription:
        for word in item.words or []:
            normalized = _normalize_words(word.word)
            if normalized:
                for token in normalized:
                    words.append({
                        "word": token,
                        "start": float(word.start),
                        "end": float(word.end),
                        "probability": float(word.probability or 0),
                    })
    words.sort(key=lambda item: (item["start"], item["end"]))
    for index, word in enumerate(words):
        word["index"] = index
    return words


def _word_span_candidates(
    segment: dict[str, Any],
    words: list[dict[str, Any]],
    min_word_index: int = 0,
    region_start: float | None = None,
    region_end: float | None = None,
    use_gemini_window: bool = True,
) -> list[dict[str, Any]]:
    """Return contiguous Whisper spans with independent text/probability evidence."""
    if use_gemini_window:
        window_start = max(
            float(segment["gemini_start"]) - 1.5,
            region_start if region_start is not None else -math.inf,
        )
        window_end = min(
            float(segment["gemini_end"]) + 1.5,
            region_end if region_end is not None else math.inf,
        )
    else:
        window_start = region_start if region_start is not None else -math.inf
        window_end = region_end if region_end is not None else math.inf

    target = _normalize_words(str(segment.get("original") or ""))
    if not target:
        return []
    candidates = [
        (int(word.get("index", index)), word)
        for index, word in enumerate(words)
        if int(word.get("index", index)) >= min_word_index
        and word["end"] >= window_start
        and word["start"] <= window_end
    ]
    if not candidates:
        return []

    target_text = " ".join(target)
    expected_start = float(segment["gemini_start"])
    matches: list[dict[str, Any]] = []
    # Short turns need tight length bounds so a candidate cannot absorb the
    # previous speaker's trailing words. Longer phrases tolerate proportional
    # omissions from Whisper without opening an excessively wide window.
    slack = 1 if len(target) <= 4 else max(2, math.ceil(len(target) * 0.25))
    min_length, max_length = max(1, len(target) - slack), len(target) + slack
    for start_index in range(len(candidates)):
        for length in range(min_length, max_length + 1):
            chosen = candidates[start_index:start_index + length]
            if len(chosen) != length:
                continue
            indices = [item[0] for item in chosen]
            if any(right != left + 1 for left, right in zip(indices, indices[1:])):
                continue
            chosen_words = [item[1] for item in chosen]
            similarity = difflib.SequenceMatcher(
                None, target_text, " ".join(word["word"] for word in chosen_words)
            ).ratio()
            proximity = max(0.0, 1.0 - abs(chosen_words[0]["start"] - expected_start) / 1.5)
            probability = sum(word["probability"] for word in chosen_words) / len(chosen_words)
            score = 0.70 * similarity + 0.15 * proximity + 0.15 * probability
            first_token_similarity = difflib.SequenceMatcher(
                None, target[0], chosen_words[0]["word"]
            ).ratio()
            last_token_similarity = difflib.SequenceMatcher(
                None, target[-1], chosen_words[-1]["word"]
            ).ratio()
            candidate = {
                "score": score,
                "text_similarity": similarity,
                "first_token_similarity": first_token_similarity,
                "last_token_similarity": last_token_similarity,
                "probability": probability,
                "length_ratio": min(len(target), len(chosen_words)) / max(len(target), len(chosen_words)),
                "start": float(chosen_words[0]["start"]),
                "end": float(chosen_words[-1]["end"]),
                "start_index": indices[0],
                "end_index": indices[-1],
            }
            matches.append(candidate)
    return sorted(matches, key=lambda item: (item["start_index"], -item["score"]))


def _best_word_span(
    segment: dict[str, Any],
    words: list[dict[str, Any]],
    min_word_index: int = 0,
    region_start: float | None = None,
    region_end: float | None = None,
) -> dict[str, Any] | None:
    matches = _word_span_candidates(
        segment,
        words,
        min_word_index=min_word_index,
        region_start=region_start,
        region_end=region_end,
    )
    return max(matches, key=lambda item: item["score"], default=None)


def _best_word_sequence(
    segment: dict[str, Any],
    words: list[dict[str, Any]],
    min_word_start: float = -math.inf,
) -> tuple[float, float, float] | None:
    """Backward-compatible tuple wrapper used by generic low-confidence alignment."""
    eligible_indices = [
        int(word.get("index", index))
        for index, word in enumerate(words)
        if float(word["start"]) >= min_word_start - 0.001
    ]
    min_word_index = min(eligible_indices) if eligible_indices else len(words)
    match = _best_word_span(segment, words, min_word_index=min_word_index)
    if match is None:
        return None
    return match["score"], match["start"], match["end"]


TRANSITION_MIN_TEXT_SIMILARITY = 0.78
TRANSITION_MIN_WORD_PROBABILITY = 0.55
TRANSITION_MAX_ADVANCE_SECONDS = 0.15
TRANSITION_MAX_DELAY_SECONDS = 1.0
TRANSITION_SAFETY_MARGIN_SECONDS = 0.04
TRANSITION_TRUE_OVERLAP_SECONDS = 0.10

SPEAKER_REFERENCE_MIN_SECONDS = 1.0
SPEAKER_REFERENCE_MAX_SECONDS = 3.0
SPEAKER_SCAN_WINDOW_SECONDS = 1.0
SPEAKER_SCAN_STEP_SECONDS = 0.25
SPEAKER_MIN_COSINE = 0.25
SPEAKER_MIN_MARGIN = 0.08
SPEAKER_MAX_REFERENCE_SIMILARITY = 0.78
_SPEAKER_ENCODER_INSTANCE: Any = None


def _get_speaker_encoder() -> Any:
    global _SPEAKER_ENCODER_INSTANCE
    if _SPEAKER_ENCODER_INSTANCE is None:
        from speechbrain.inference.classifiers import EncoderClassifier

        model_cache = BASE / ".models" / "speechbrain-ecapa"
        model_cache.mkdir(parents=True, exist_ok=True)
        _SPEAKER_ENCODER_INSTANCE = EncoderClassifier.from_hparams(
            source=SPEAKER_EMBEDDING_MODEL,
            savedir=str(model_cache),
            run_opts={"device": "cpu"},
        )
    return _SPEAKER_ENCODER_INSTANCE


def _bounded_audio_interval(
    audio: Any,
    start: float,
    end: float,
    sampling_rate: int = 16000,
) -> tuple[float, float] | None:
    duration = len(audio) / sampling_rate
    start = max(0.0, min(float(start), duration))
    end = max(start, min(float(end), duration))
    return (start, end) if end - start >= SPEAKER_REFERENCE_MIN_SECONDS else None


def _speaker_embedding(
    encoder: Any,
    audio: Any,
    start: float,
    end: float,
    sampling_rate: int = 16000,
) -> Any | None:
    interval = _bounded_audio_interval(audio, start, end, sampling_rate)
    if interval is None:
        return None
    try:
        import numpy as np
        import torch

        first = int(interval[0] * sampling_rate)
        last = int(interval[1] * sampling_rate)
        samples = np.asarray(audio[first:last], dtype=np.float32)
        if samples.size < int(SPEAKER_REFERENCE_MIN_SECONDS * sampling_rate):
            return None
        # Reject silence/near-silence before it can become a misleading identity
        # reference. decode_audio returns normalized float samples.
        rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
        peak = float(np.max(np.abs(samples)))
        if not np.isfinite(rms) or rms < 0.001 or peak < 0.005:
            return None
        waveform = torch.from_numpy(samples).unsqueeze(0)
        with torch.inference_mode():
            embedding = encoder.encode_batch(waveform)
        return embedding.detach().float().cpu().reshape(-1)
    except Exception:
        return None


def _embedding_cosine(left: Any, right: Any) -> float:
    try:
        import torch

        return float(torch.nn.functional.cosine_similarity(
            left.reshape(1, -1), right.reshape(1, -1), dim=1
        ).item())
    except Exception:
        return -1.0


def _fallback_reference_interval(
    segments: list[dict[str, Any]],
    speaker_id: str,
    excluded_ids: set[int],
) -> tuple[float, float] | None:
    candidates: list[tuple[float, float, float]] = []
    for segment in segments:
        if id(segment) in excluded_ids or str(segment.get("speaker_id", "")) != speaker_id:
            continue
        if segment.get("is_overlap") or segment.get("shared_vad_cross_speaker"):
            continue
        if segment.get("alignment_method") not in {"silero", "whisper"}:
            continue
        confidence = float(segment.get("alignment_confidence", 0))
        if confidence < 0.55:
            continue
        start = float(segment.get("auto_start", segment.get("start", 0)))
        end = float(segment.get("auto_end", segment.get("end", start)))
        if end - start < SPEAKER_REFERENCE_MIN_SECONDS:
            continue
        candidates.append((confidence, start, end))
    if not candidates:
        return None
    _, start, end = max(candidates, key=lambda item: (item[0], item[2] - item[1]))
    return start, min(end, start + SPEAKER_REFERENCE_MAX_SECONDS)


def _transition_reference_intervals(
    previous: dict[str, Any],
    current: dict[str, Any],
    segments: list[dict[str, Any]],
) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    # References must come from the ordered Whisper spans that established this
    # boundary, not from the inaccurate Gemini overlap timestamps under repair.
    try:
        previous_word_start = float(current["previous_reference_word_start"])
        previous_word_end = float(current["previous_reference_word_end"])
        current_word_start = float(current["next_reference_word_start"])
        current_word_end = float(current["next_reference_word_end"])
    except (KeyError, TypeError, ValueError):
        previous_word_start = previous_word_end = 0.0
        current_word_start = current_word_end = 0.0

    previous_interval = None
    clean_previous_end = min(previous_word_end, current_word_start - 0.10)
    if clean_previous_end - previous_word_start >= SPEAKER_REFERENCE_MIN_SECONDS:
        previous_interval = (
            max(previous_word_start, clean_previous_end - SPEAKER_REFERENCE_MAX_SECONDS),
            clean_previous_end,
        )

    current_interval = None
    clean_current_start = max(current_word_start, previous_word_end + 0.10)
    if current_word_end - clean_current_start >= SPEAKER_REFERENCE_MIN_SECONDS:
        current_interval = (
            clean_current_start,
            min(current_word_end, clean_current_start + SPEAKER_REFERENCE_MAX_SECONDS),
        )

    excluded = {id(previous), id(current)}
    if previous_interval is None:
        previous_interval = _fallback_reference_interval(
            segments, str(previous.get("speaker_id", "")), excluded
        )
    if current_interval is None:
        current_interval = _fallback_reference_interval(
            segments, str(current.get("speaker_id", "")), excluded
        )
    return previous_interval, current_interval


def _speaker_transition_groups(
    segments: list[dict[str, Any]],
) -> dict[tuple[str, int], list[dict[str, Any]]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for segment in segments:
        region_id = segment.get("matched_vad_region_id")
        overlap_group_id = segment.get("gemini_overlap_group_id")
        if segment.get("shared_vad_cross_speaker") and isinstance(region_id, int):
            groups.setdefault(("vad", region_id), []).append(segment)
        elif segment.get("gemini_overlap_candidate") and isinstance(overlap_group_id, int):
            groups.setdefault(("gemini_overlap", overlap_group_id), []).append(segment)
    for group in groups.values():
        group.sort(key=lambda item: (float(item["gemini_start"]), float(item["gemini_end"])))
    return groups


def _verify_transition_pair_with_embeddings(
    encoder: Any,
    audio: Any,
    segments: list[dict[str, Any]],
    previous: dict[str, Any],
    current: dict[str, Any],
) -> bool:
    reference_intervals = _transition_reference_intervals(previous, current, segments)
    if reference_intervals[0] is None or reference_intervals[1] is None:
        current["speaker_verification"] = "reference_unavailable"
        return False

    previous_reference = _speaker_embedding(encoder, audio, *reference_intervals[0])
    current_reference = _speaker_embedding(encoder, audio, *reference_intervals[1])
    if previous_reference is None or current_reference is None:
        current["speaker_verification"] = "reference_embedding_failed"
        return False

    reference_similarity = _embedding_cosine(previous_reference, current_reference)
    current["speaker_reference_similarity"] = round(reference_similarity, 3)
    if reference_similarity >= SPEAKER_MAX_REFERENCE_SIMILARITY:
        current["speaker_verification"] = "references_not_separable"
        return False

    candidate = float(current.get("auto_start", current["gemini_start"]))
    scan_start = max(
        float(previous["gemini_start"]),
        candidate - 2.0,
    )
    scan_end = min(
        float(current["gemini_end"]),
        candidate + 3.0,
    )
    centers: list[dict[str, float | str | None]] = []
    center = scan_start + SPEAKER_SCAN_WINDOW_SECONDS / 2
    while center + SPEAKER_SCAN_WINDOW_SECONDS / 2 <= scan_end + 1e-6:
        label: str | None = None
        previous_similarity = -1.0
        current_similarity = -1.0
        embedding = _speaker_embedding(
            encoder,
            audio,
            center - SPEAKER_SCAN_WINDOW_SECONDS / 2,
            center + SPEAKER_SCAN_WINDOW_SECONDS / 2,
        )
        if embedding is not None:
            previous_similarity = _embedding_cosine(embedding, previous_reference)
            current_similarity = _embedding_cosine(embedding, current_reference)
            if (
                previous_similarity >= SPEAKER_MIN_COSINE
                and previous_similarity - current_similarity >= SPEAKER_MIN_MARGIN
            ):
                label = "previous"
            elif (
                current_similarity >= SPEAKER_MIN_COSINE
                and current_similarity - previous_similarity >= SPEAKER_MIN_MARGIN
            ):
                label = "current"
        # Keep failed/ambiguous positions so two list neighbors always mean two
        # physically consecutive 250 ms scan steps.
        centers.append({
            "time": center,
            "previous": previous_similarity,
            "current": current_similarity,
            "label": label,
        })
        center += SPEAKER_SCAN_STEP_SECONDS

    transition: tuple[float, float] | None = None
    for index in range(1, len(centers) - 1):
        previous_item = centers[index - 1]
        item = centers[index]
        next_item = centers[index + 1]
        if (
            previous_item["label"] == "previous"
            and item["label"] == "current"
            and next_item["label"] == "current"
        ):
            transition = (float(previous_item["time"]), float(item["time"]))
            break
    if transition is None:
        current["speaker_verification"] = "voice_change_unconfirmed"
        return False

    last_previous, first_current = transition
    # The beginning of a current-dominant one-second window can still contain
    # the prior voice. Use the end of the immediately preceding previous-dominant
    # window as a conservative no-early-start floor.
    boundary_floor = max(
        0.0,
        last_previous + SPEAKER_SCAN_WINDOW_SECONDS / 2,
    )
    old_start = float(current.get("auto_start", current["gemini_start"]))
    new_start = max(old_start, boundary_floor)
    current.update({
        "auto_start": round(new_start, 3),
        "auto_end": round(max(new_start + 0.2, float(current["auto_end"])), 3),
        "speaker_verification": "ecapa_confirmed",
        "speaker_boundary_floor": round(boundary_floor, 3),
        "speaker_previous_last_center": round(last_previous, 3),
        "speaker_current_first_center": round(first_current, 3),
    })
    if new_start > old_start + 0.02:
        current["alignment_method"] = "whisper_ecapa_speaker_boundary"
        current["alignment_warning"] = None
    return True


def verify_speaker_transitions_with_embeddings(
    audio: Any,
    segments: list[dict[str, Any]],
) -> tuple[int, str | None]:
    mode = resolved_speaker_verification_mode()
    if mode == "off" or audio is None:
        return 0, None
    if not speaker_verification_available():
        warning = "SpeechBrain ECAPA не установлен; используется Whisper speaker fallback"
        return 0, warning if mode == "on" else None
    try:
        encoder = _get_speaker_encoder()
    except Exception as exc:
        return 0, f"ECAPA speaker verification недоступен: {exc}"

    verified = 0
    attempted = 0
    embedding_failed = False
    for group in _speaker_transition_groups(segments).values():
        for previous, current in zip(group, group[1:]):
            if str(previous.get("speaker_id", "")) == str(current.get("speaker_id", "")):
                continue
            if current.get("alignment_method") != "whisper_speaker_boundary":
                continue
            # Aggregate ECAPA embeddings cannot disprove simultaneous speech.
            # Never move an original Gemini overlap candidate; ordered Whisper
            # timing remains its overlap-safe primary correction.
            if current.get("gemini_overlap_candidate"):
                current["speaker_verification"] = "overlap_safe_text_only"
                continue
            attempted += 1
            if _verify_transition_pair_with_embeddings(
                encoder, audio, segments, previous, current
            ):
                verified += 1
            elif current.get("speaker_verification") == "reference_embedding_failed":
                embedding_failed = True
    warning = None
    if mode == "on" and attempted and embedding_failed:
        warning = "ECAPA не смог извлечь speaker embeddings; сохранён Whisper fallback"
    return verified, warning


def _select_ordered_transition_spans(
    group: list[dict[str, Any]],
    words: list[dict[str, Any]],
    region_start: float,
    region_end: float,
) -> list[dict[str, Any] | None]:
    """Globally assign earliest valid non-overlapping spans across one VAD group."""
    candidate_lists: list[list[dict[str, Any]]] = []
    for segment in group:
        unique: dict[tuple[int, int], dict[str, Any]] = {}
        for match in _word_span_candidates(
            segment,
            words,
            region_start=region_start,
            region_end=region_end,
            use_gemini_window=False,
        ):
            if (
                match["text_similarity"] < TRANSITION_MIN_TEXT_SIMILARITY
                or match["probability"] < TRANSITION_MIN_WORD_PROBABILITY
                or match["length_ratio"] < 0.75
                or match["first_token_similarity"] < 0.60
                or match["last_token_similarity"] < 0.60
            ):
                continue
            key = (int(match["start_index"]), int(match["end_index"]))
            if key not in unique or match["score"] > unique[key]["score"]:
                unique[key] = match
        candidate_lists.append(sorted(
            unique.values(),
            key=lambda item: (item["start_index"], item["end_index"], -item["score"]),
        ))

    # State objective: match as many group segments as possible, then prefer the
    # earliest complete ordered assignment, then the strongest aggregate score.
    states: dict[int, tuple[list[dict[str, Any] | None], tuple[int, int, float]]] = {
        -1: ([], (0, 0, 0.0))
    }
    for candidates in candidate_lists:
        next_states: dict[int, tuple[list[dict[str, Any] | None], tuple[int, int, float]]] = {}
        for last_end, (chosen, objective) in states.items():
            options: list[dict[str, Any] | None] = [None, *candidates]
            for match in options:
                if match is not None and int(match["start_index"]) <= last_end:
                    continue
                new_end = last_end if match is None else int(match["end_index"])
                new_objective = objective
                if match is not None:
                    new_objective = (
                        objective[0] + 1,
                        objective[1] - int(match["start_index"]),
                        objective[2] + float(match["score"]),
                    )
                candidate_state = ([*chosen, match], new_objective)
                existing = next_states.get(new_end)
                if existing is None or candidate_state[1] > existing[1]:
                    next_states[new_end] = candidate_state
        states = next_states

    if not states:
        return [None] * len(group)
    return max(states.values(), key=lambda item: item[1])[0]


def refine_speaker_transitions(
    segments: list[dict[str, Any]], words: list[dict[str, Any]]
) -> int:
    """Anchor uncertain cross-speaker turns without moving the prior line."""
    groups = _speaker_transition_groups(segments)

    refined = 0
    for group_key, group in groups.items():
        is_gemini_overlap_group = group_key[0] == "gemini_overlap"
        group.sort(key=lambda item: (float(item["gemini_start"]), float(item["gemini_end"])))
        if is_gemini_overlap_group:
            region_start = min(float(item["transition_window_start"]) for item in group)
            region_end = max(float(item["transition_window_end"]) for item in group)
        else:
            region_start = min(float(item["matched_vad_start"]) for item in group) - 0.3
            region_end = max(float(item["matched_vad_end"]) for item in group) + 0.3
        selected = _select_ordered_transition_spans(
            group, words, region_start=region_start, region_end=region_end
        )
        spans = {id(segment): match for segment, match in zip(group, selected)}

        for previous, current in zip(group, group[1:]):
            if str(previous.get("speaker_id", "")) == str(current.get("speaker_id", "")):
                continue
            previous_span = spans.get(id(previous))
            current_span = spans.get(id(current))
            if previous_span is None or current_span is None:
                current.update({
                    "auto_start": float(current["gemini_start"]),
                    "auto_end": float(current["gemini_end"]),
                    "alignment_method": "gemini_fallback_speaker_transition",
                    "alignment_confidence": 0.0,
                    "alignment_warning": "speaker_boundary_not_found_low_confidence",
                    "boundary_source": "gemini_fallback",
                    "overlap_resolution": (
                        "unverified_preserved" if is_gemini_overlap_group else None
                    ),
                })
                continue

            if (
                is_gemini_overlap_group
                and float(previous_span["end"]) - float(current_span["start"])
                >= TRANSITION_TRUE_OVERLAP_SECONDS
            ):
                current.update({
                    "auto_start": float(current["gemini_start"]),
                    "auto_end": float(current["gemini_end"]),
                    "alignment_method": "whisper_confirmed_overlap",
                    "alignment_confidence": round(min(
                        float(previous_span["score"]), float(current_span["score"])
                    ), 3),
                    "alignment_warning": "true_overlap_preserved",
                    "boundary_source": "overlapping_whisper_word_spans",
                    "overlap_resolution": "confirmed_preserved",
                })
                continue

            boundary = float(previous_span["end"]) + TRANSITION_SAFETY_MARGIN_SECONDS
            anchored_start = max(float(current_span["start"]), boundary)
            gemini_start = float(current["gemini_start"])
            if anchored_start < gemini_start - TRANSITION_MAX_ADVANCE_SECONDS:
                current.update({
                    "auto_start": gemini_start,
                    "auto_end": float(current["gemini_end"]),
                    "alignment_method": "gemini_fallback_speaker_transition",
                    "alignment_confidence": 0.3,
                    "alignment_warning": "whisper_start_too_early_vs_gemini",
                    "boundary_source": "gemini_fallback",
                    "overlap_resolution": (
                        "unverified_preserved" if is_gemini_overlap_group else None
                    ),
                })
                continue
            if (
                not is_gemini_overlap_group
                and anchored_start > gemini_start + TRANSITION_MAX_DELAY_SECONDS
            ):
                current.update({
                    "auto_start": gemini_start,
                    "auto_end": float(current["gemini_end"]),
                    "alignment_method": "gemini_fallback_speaker_transition",
                    "alignment_confidence": 0.3,
                    "alignment_warning": "whisper_start_too_late_vs_gemini",
                    "boundary_source": "gemini_fallback",
                })
                continue

            confidence = min(
                0.95,
                0.55
                + 0.25 * min(previous_span["text_similarity"], current_span["text_similarity"])
                + 0.20 * min(previous_span["probability"], current_span["probability"]),
            )
            current.update({
                "auto_start": round(max(0.0, anchored_start), 3),
                "auto_end": round(max(anchored_start + 0.2, float(current["gemini_end"])), 3),
                "alignment_method": "whisper_speaker_boundary",
                "alignment_confidence": round(confidence, 3),
                "alignment_warning": None,
                "boundary_source": "ordered_whisper_word_spans",
                "previous_reference_word_start": round(float(previous_span["start"]), 3),
                "previous_reference_word_end": round(float(previous_span["end"]), 3),
                "next_reference_word_start": round(float(current_span["start"]), 3),
                "next_reference_word_end": round(float(current_span["end"]), 3),
                "previous_text_similarity": round(float(previous_span["text_similarity"]), 3),
                "next_text_similarity": round(float(current_span["text_similarity"]), 3),
                "overlap_resolution": (
                    "ordered_non_overlap" if is_gemini_overlap_group else None
                ),
            })
            refined += 1
    return refined


def refine_segments_with_whisper(
    segments: list[dict[str, Any]], words: list[dict[str, Any]], precise_all: bool = False
) -> int:
    refined = 0
    consumed_until = -math.inf
    candidates = sorted(
        (
            segment for segment in segments
            if not segment.get("is_overlap")
            and not segment.get("shared_vad_cross_speaker")
            and not segment.get("gemini_overlap_candidate")
            and (
                precise_all
                or float(segment.get("alignment_confidence", 0)) < ALIGNMENT_LOW_CONFIDENCE
            )
        ),
        key=lambda item: (float(item["gemini_start"]), float(item["gemini_end"])),
    )
    for segment in candidates:
        match = _best_word_sequence(segment, words, min_word_start=consumed_until)
        if match is None or match[0] < 0.55:
            continue
        score, start, end = match
        segment.update({
            "auto_start": round(max(0.0, start), 3),
            "auto_end": round(max(start + 0.2, end), 3),
            "alignment_method": "whisper",
            "alignment_confidence": round(min(0.95, score), 3),
            "alignment_warning": None if score >= 0.7 else "whisper_low_text_match",
        })
        consumed_until = end
        refined += 1
    return refined


def apply_hybrid_alignment(source: Path, segments: list[dict[str, Any]]) -> dict[str, Any]:
    mode = resolved_alignment_mode()
    _mark_overlap_segments(segments)
    for segment in segments:
        segment.setdefault("gemini_start", float(segment["start"]))
        segment.setdefault("gemini_end", float(segment["end"]))
        segment.setdefault("manual_offset", 0.0)
        segment.setdefault("auto_start", float(segment["start"]))
        segment.setdefault("auto_end", float(segment["end"]))
        segment.setdefault("alignment_method", "gemini_fallback")
        segment.setdefault("alignment_confidence", 0.0)
        segment.setdefault("alignment_warning", "alignment_disabled" if mode == "off" else None)

    if mode == "off":
        return {
            "mode": mode,
            "vadRegions": 0,
            "whisperRefined": 0,
            "speakerTransitionsRefined": 0,
            "speakerTransitionsVerified": 0,
            "speakerVerificationWarning": None,
            "warning": None,
        }

    audio, regions, warning = run_silero_vad(source)
    if audio is not None and regions:
        align_segments_with_vad(segments, audio, regions)

    whisper_refined = 0
    transition_refined = 0
    speaker_verified = 0
    speaker_warning: str | None = None
    if mode in {"hybrid", "precise"} and importlib.util.find_spec("faster_whisper") is not None:
        windows = merge_alignment_windows(segments, precise_all=mode == "precise")
        if windows:
            try:
                words = run_whisper_for_windows(source, windows)
                transition_refined = refine_speaker_transitions(segments, words)
                generic_refined = refine_segments_with_whisper(
                    segments, words, precise_all=mode == "precise"
                )
                whisper_refined = transition_refined + generic_refined
                speaker_verified, speaker_warning = verify_speaker_transitions_with_embeddings(
                    audio, segments
                )
            except Exception as exc:
                warning = f"Precise fallback недоступен: {exc}"

    if speaker_warning:
        warning = f"{warning}; {speaker_warning}" if warning else speaker_warning

    for segment in segments:
        segment["start"] = float(segment["auto_start"]) + float(segment.get("manual_offset", 0))
        segment["end"] = float(segment["auto_end"]) + float(segment.get("manual_offset", 0))
        segment["effective_start"] = segment["start"]
        segment["effective_end"] = segment["end"]

    return {
        "mode": mode,
        "vadRegions": len(regions),
        "whisperRefined": whisper_refined,
        "speakerTransitionsRefined": transition_refined,
        "speakerTransitionsVerified": speaker_verified,
        "speakerVerificationWarning": speaker_warning,
        "warning": warning,
    }


def letter_count(text: Any) -> int:
    """Count Unicode letters while ignoring spaces, punctuation, digits, and apostrophes."""
    apostrophes = {"'", "’", "‘", "ʻ", "ʼ", "`"}
    return sum(
        character.isalpha() and character not in apostrophes
        for character in str(text or "")
    )


def normalize_segment_prosody(segment: dict[str, Any]) -> None:
    allowed_pace = {"slow", "normal", "fast"}
    allowed_energy = {"low", "medium", "high", "whisper"}
    allowed_pitch = {"flat", "rising", "falling", "rising_falling", "variable"}
    pace = str(segment.get("pace") or "normal").lower()
    energy = str(segment.get("energy") or "medium").lower()
    pitch = str(segment.get("pitch_tendency") or "variable").lower()
    segment["pace"] = pace if pace in allowed_pace else "normal"
    segment["energy"] = energy if energy in allowed_energy else "medium"
    segment["pitch_tendency"] = pitch if pitch in allowed_pitch else "variable"
    segment["intonation_contour"] = str(segment.get("intonation_contour") or "natural").strip()[:200]
    segment["delivery_instruction"] = str(
        segment.get("delivery_instruction") or "Perform naturally and preserve the original emotion."
    ).strip()[:300]
    emphasized = segment.get("emphasized_words")
    segment["emphasized_words"] = [str(item).strip() for item in emphasized[:8] if str(item).strip()] if isinstance(emphasized, list) else []
    pauses = segment.get("pauses")
    normalized_pauses = []
    if isinstance(pauses, list):
        for pause in pauses[:6]:
            if not isinstance(pause, dict):
                continue
            try:
                word_index = max(0, int(pause.get("after_word_index", 0)))
                duration_ms = max(100, min(800, int(pause.get("duration_ms", 150))))
            except (TypeError, ValueError):
                continue
            normalized_pauses.append({"after_word_index": word_index, "duration_ms": duration_ms})
    segment["pauses"] = normalized_pauses


def professional_prompt(target_language: str, max_extra_letters: int | None = None) -> str:
    length_rule = ""
    if max_extra_letters is not None:
        length_rule = f"""
STRICT DUBBING-LENGTH RULE:
- For every segment, count Unicode alphabetic letters only; ignore spaces, punctuation, apostrophes, and digits.
- The translated text MUST contain no more than original letter count + {max_extra_letters} letters.
- Prefer a natural shorter Uzbek expression instead of speaking faster.
- Preserve the complete meaning, but remove filler and choose concise synonyms.
- Verify the letter count yourself before returning JSON. This rule is mandatory for every segment.
"""

    return f"""You are a senior dubbing director, dialogue editor, translator, and speaker-diarization expert.
Listen to the supplied audio and create a professional dubbing script translated into {target_language}.
{length_rule}
Identify every DISTINCT audible speaker by voice, not by visual appearance. Keep one stable speaker_id for the same person throughout the whole recording. Treat narrators and off-screen speakers as separate speakers. If people talk over each other, return separate overlapping segments. Do not merge different speakers.

Return exactly one JSON object with this structure:
{{
  "speakers": [
    {{
      "id": "S1",
      "label": "Speaker 1",
      "voice_character": "masculine|feminine|young|neutral",
      "delivery": "short description of pitch, age impression, energy and emotion"
    }}
  ],
  "segments": [
    {{
      "start": 0.50,
      "end": 2.30,
      "speaker_id": "S1",
      "original": "exact original speech",
      "translated": "natural dubbing translation",
      "emotion": "neutral|happy|sad|angry|surprised|fearful|skeptical|sarcastic|urgent|calm|whispering|serious",
      "pace": "slow|normal|fast",
      "energy": "low|medium|high|whisper",
      "pitch_tendency": "flat|rising|falling|rising_falling|variable",
      "pauses": [{{"after_word_index": 2, "duration_ms": 180}}],
      "emphasized_words": ["translated word to stress"],
      "intonation_contour": "short description of the pitch movement",
      "delivery_instruction": "one concise instruction for a dubbing actor"
    }}
  ]
}}

Requirements:
- Accurate start/end timestamps in seconds.
- Capture every spoken phrase and audible speaker.
- Translate for natural professional dubbing, not word-for-word.
- Keep translated speech short enough to fit the original time window.
- Preserve intent, politeness, humor and emotion.
- Analyze prosody for EACH segment, not only for the speaker: pace, energy, pitch movement, pauses longer than 150 ms, and emphasis.
- emphasized_words must contain words from the translated dialogue, not the original language.
- delivery_instruction must be one or two concise sentences describing how to perform this translated line naturally.
- Never infer identity or actual gender; voice_character only describes audible vocal presentation for voice casting.
- Return JSON only. If there is no speech, return {{"speakers":[],"segments":[]}}."""


async def shorten_overlong_translations(
    segments: list[dict[str, Any]],
    target_language: str,
    max_extra_letters: int,
) -> int:
    """Ask Vertex to shorten translations until every measurable line fits its budget."""
    adjusted_indices: set[int] = set()
    endpoint = vertex_model_url(TRANSCRIBE_MODEL)

    async with httpx.AsyncClient(timeout=180) as client:
        for _ in range(3):
            jobs = []
            for index, segment in enumerate(segments):
                original = str(segment.get("original") or "").strip()
                translated = str(segment.get("translated") or "").strip()
                original_letters = letter_count(original)
                max_letters = original_letters + max_extra_letters
                if translated and letter_count(translated) > max_letters:
                    jobs.append({
                        "index": index,
                        "original": original,
                        "current_translation": translated,
                        "max_letters": max_letters,
                    })

            if not jobs:
                return len(adjusted_indices)

            correction_prompt = f"""You are a professional {target_language} dubbing editor.
Rewrite every current_translation below into natural spoken {target_language} while preserving its complete meaning and emotion.
Each result MUST have no more than max_letters Unicode alphabetic letters. Spaces, punctuation, apostrophes, and digits do not count.
Use concise synonyms and remove filler. Never split a line, omit an item, or change its index.
Count letters before answering.

Return JSON only in this exact format:
{{"segments":[{{"index":0,"translated":"short natural translation"}}]}}

INPUT:
{json.dumps(jobs, ensure_ascii=False)}"""
            payload = {
                "contents": [{"role": "user", "parts": [{"text": correction_prompt}]}],
                "generationConfig": {
                    "temperature": 0.1,
                    "responseMimeType": "application/json",
                },
            }
            try:
                response = await client.post(endpoint, headers=vertex_headers(), json=payload)
            except httpx.RequestError as exc:
                raise HTTPException(
                    502,
                    {
                        "code": "VERTEX_LENGTH_CHECK_UNREACHABLE",
                        "message": f"Не удалось сократить длинные реплики через Vertex AI: {exc}",
                        "help": ["Повторите дубляж через минуту."],
                    },
                ) from exc

            if response.status_code != 200:
                raise api_error(response, "translation_length_correction")

            try:
                text = response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                if text.startswith("```"):
                    text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                parsed = json.loads(text)
                if not isinstance(parsed, dict):
                    raise ValueError("корень ответа должен быть JSON-объектом")
                updates = parsed.get("segments")
                if not isinstance(updates, list) or any(not isinstance(item, dict) for item in updates):
                    raise ValueError("segments должен быть массивом JSON-объектов")
            except Exception as exc:
                raise HTTPException(
                    502,
                    {
                        "code": "BAD_LENGTH_CORRECTION_JSON",
                        "message": f"Vertex AI вернул неверный ответ при сокращении реплик: {exc}",
                        "help": ["Повторите дубляж."],
                    },
                ) from exc

            requested = {job["index"] for job in jobs}
            for update in updates:
                try:
                    index = int(update.get("index"))
                except (TypeError, ValueError):
                    continue
                candidate = str(update.get("translated") or "").strip()
                if index in requested and candidate:
                    current = str(segments[index].get("translated") or "")
                    if letter_count(candidate) < letter_count(current):
                        segments[index]["translated"] = candidate
                        adjusted_indices.add(index)

    remaining = []
    for index, segment in enumerate(segments):
        original_letters = letter_count(segment.get("original"))
        translated_letters = letter_count(segment.get("translated"))
        if translated_letters > original_letters + max_extra_letters:
            remaining.append(index + 1)

    if remaining:
        raise HTTPException(
            422,
            {
                "code": "TRANSLATION_TOO_LONG",
                "message": (
                    "Vertex AI не смог достаточно сократить реплики: "
                    + ", ".join(map(str, remaining[:10]))
                ),
                "help": [
                    "Нажмите «Дублировать заново»: сервер ещё раз подберёт более короткие формулировки.",
                    f"Лимит для узбекского перевода: оригинал + {max_extra_letters} букв.",
                ],
            },
        )

    return len(adjusted_indices)


def assign_speaker_voices(speakers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    used: set[str] = set()
    all_voices = sorted(VOICES)
    for index, speaker in enumerate(speakers):
        character = str(speaker.get("voice_character", "neutral")).lower()
        if character not in VOICE_POOLS:
            character = "neutral"
        preferred = next((candidate for candidate in VOICE_POOLS[character] if candidate not in used), None)
        fallback = next((candidate for candidate in all_voices if candidate not in used), None)
        voice = preferred or fallback or VOICE_POOLS[character][index % len(VOICE_POOLS[character])]
        reused = voice in used
        used.add(voice)
        speaker["voice_character"] = character
        speaker["voice"] = voice
        speaker["voice_reused"] = reused
        speaker.setdefault("id", f"S{index + 1}")
        speaker.setdefault("label", f"Спикер {index + 1}")
        speaker.setdefault("delivery", "natural")
    return speakers


@app.post("/api/translate")
async def translate(request: Request) -> dict[str, Any]:
    """Transcribe, translate and diarize all audible speakers."""
    if not API_KEY or API_KEY in {"ваш_ключ_сюда", "ваш_vertex_ключ"}:
        raise HTTPException(500, {"code": "NO_KEY", "message": "Вставьте VERTEX_API_KEY в .env", "help": []})

    body = await request.json()
    source = UPLOADS / Path(body.get("filename", "")).name
    requested_language_code = str(body.get("lang", "uz"))
    language_code = requested_language_code if requested_language_code in LANGS else "uz"
    target_language = LANGS[language_code]
    max_extra_letters = 8 if language_code == "uz" else None
    if not source.exists():
        raise HTTPException(404, "Файл не найден")

    with tempfile.TemporaryDirectory(prefix="jarcut_") as temp_dir:
        compact_audio = Path(temp_dir) / "speech.mp3"
        if extract_audio(source, compact_audio):
            media_path, mime = compact_audio, "audio/mpeg"
        else:
            media_path = source
            mime = MIME_TYPES.get(source.suffix.lower(), "video/mp4")

        prompt = professional_prompt(target_language, max_extra_letters)
        max_inline_bytes = 14 * 1024 * 1024
        media_size = media_path.stat().st_size
        if media_size > max_inline_bytes:
            raise HTTPException(
                status_code=413,
                detail={
                    "code": "AUDIO_TOO_LARGE",
                    "message": (
                        f"Аудиодорожка после сжатия занимает {media_size / 1024 / 1024:.1f} МБ. "
                        "Для безопасной inline-отправки Vertex AI нужно не более 14 МБ."
                    ),
                    "help": [
                        "Разделите очень длинное видео на части и обработайте их отдельно.",
                        "Убедитесь, что ffmpeg установлен: без него сервер отправляет исходное видео.",
                    ],
                },
            )

        media_part = {
            "inlineData": {
                "mimeType": mime,
                "data": base64.b64encode(media_path.read_bytes()).decode("ascii"),
            }
        }
        payload = {
            "contents": [{"role": "user", "parts": [media_part, {"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json",
            },
        }
        try:
            endpoint = vertex_model_url(TRANSCRIBE_MODEL)
        except ValueError as exc:
            raise HTTPException(
                500,
                {"code": "VERTEX_CONFIG_ERROR", "message": str(exc), "help": []},
            ) from exc

        try:
            timeout = httpx.Timeout(
                connect=30.0,
                read=VERTEX_TRANSCRIBE_TIMEOUT_SECONDS,
                write=120.0,
                pool=30.0,
            )
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    endpoint,
                    headers=vertex_headers(),
                    json=payload,
                )
        except httpx.ReadTimeout as exc:
            timeout_minutes = VERTEX_TRANSCRIBE_TIMEOUT_SECONDS / 60
            raise HTTPException(
                504,
                {
                    "code": "VERTEX_TIMEOUT",
                    "message": (
                        "Vertex AI не завершил распознавание вовремя "
                        f"(лимит ожидания ответа {timeout_minutes:g} мин.)."
                    ),
                    "help": [
                        "Повторите попытку: временная задержка Vertex AI обычно проходит.",
                        "Если видео длинное, разделите его на более короткие части.",
                        "При необходимости измените VERTEX_TRANSCRIBE_TIMEOUT_SECONDS в .env (60–600).",
                    ],
                    "retryable": True,
                },
            ) from exc
        except httpx.TimeoutException as exc:
            raise HTTPException(
                504,
                {
                    "code": "VERTEX_TIMEOUT",
                    "message": "Истекло время подключения к Vertex AI или отправки аудио.",
                    "help": [
                        "Проверьте интернет, VPN/прокси и повторите попытку.",
                        "Если видео длинное, разделите его на более короткие части.",
                    ],
                    "retryable": True,
                },
            ) from exc
        except httpx.RequestError as exc:
            raise HTTPException(
                502,
                {
                    "code": "VERTEX_UNREACHABLE",
                    "message": f"Не удалось связаться с Vertex AI: {exc}",
                    "help": ["Проверьте интернет, VPN/прокси и повторите через минуту."],
                },
            ) from exc

    if response.status_code != 200:
        raise api_error(response, "transcription")

    try:
        text = response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        result = json.loads(text)
        if isinstance(result, list):
            result = {"speakers": [], "segments": result}
    except Exception as exc:
        raise HTTPException(500, {"code": "BAD_VERTEX_JSON", "message": f"Не удалось разобрать сценарий Vertex AI: {exc}", "help": []})

    raw_speakers = result.get("speakers") or []
    segments = result.get("segments") or []
    length_adjusted = 0
    if max_extra_letters is not None:
        length_adjusted = await shorten_overlong_translations(
            segments,
            target_language,
            max_extra_letters,
        )

    # Never pass model-generated IDs into inline browser handlers. Normalize all IDs.
    speakers: list[dict[str, Any]] = []
    id_map: dict[str, str] = {}
    for item in raw_speakers:
        old_id = str(item.get("id") or f"speaker_{len(id_map) + 1}")
        if old_id in id_map:
            continue
        safe_id = f"S{len(id_map) + 1}"
        id_map[old_id] = safe_id
        normalized = dict(item)
        normalized["id"] = safe_id
        speakers.append(normalized)

    for segment in segments:
        old_id = str(segment.get("speaker_id") or "speaker_1")
        if old_id not in id_map:
            safe_id = f"S{len(id_map) + 1}"
            id_map[old_id] = safe_id
            speakers.append({
                "id": safe_id,
                "label": f"Спикер {len(speakers) + 1}",
                "voice_character": "neutral",
                "delivery": "natural",
            })
        segment["speaker_id"] = id_map[old_id]
        segment["start"] = max(0.0, float(segment.get("start", 0)))
        segment["end"] = max(segment["start"] + 0.2, float(segment.get("end", segment["start"] + 2)))
        normalize_segment_prosody(segment)
        if max_extra_letters is not None:
            original_letters = letter_count(segment.get("original"))
            segment["original_letters"] = original_letters
            segment["translated_letters"] = letter_count(segment.get("translated"))
            segment["max_translated_letters"] = original_letters + max_extra_letters

    segments.sort(key=lambda item: (float(item.get("start", 0)), float(item.get("end", 0))))
    alignment = apply_hybrid_alignment(source, segments)
    speakers = assign_speaker_voices(speakers)
    return {
        "ok": True,
        "model": TRANSCRIBE_MODEL,
        "speakers": speakers,
        "segments": segments,
        "lengthAdjusted": length_adjusted,
        "maxExtraLetters": max_extra_letters,
        "alignment": alignment,
    }


def pcm_to_wav(pcm: bytes, rate: int = 24000) -> bytes:
    block_align = 2
    byte_rate = rate * block_align
    return (
        b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, rate, byte_rate, block_align, 16)
        + b"data" + struct.pack("<I", len(pcm)) + pcm
    )


def parse_pcm_rate(mime: str) -> int:
    for part in mime.split(";"):
        if "rate=" in part:
            try:
                return int(part.split("=", 1)[1])
            except ValueError:
                pass
    return 24000


def trim_tts_leading_silence(source: Path) -> Path:
    """Remove model-added silence so audible speech starts at the segment timestamp."""
    synced = source.with_name(f"{source.stem}_synced.wav")
    command = [
        "ffmpeg", "-y", "-i", str(source),
        "-af", (
            "silenceremove=start_periods=1:start_duration=0.01:"
            "start_threshold=-55dB:start_silence=0.015"
        ),
        "-c:a", "pcm_s16le", str(synced),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        if result.returncode != 0 or not synced.exists() or synced.stat().st_size <= 44:
            synced.unlink(missing_ok=True)
            return source

        source_duration = probe_duration(source)
        synced_duration = probe_duration(synced)
        removed_duration = source_duration - synced_duration
        max_safe_removal = min(1.0, max(0.12, source_duration * 0.35))
        if (
            source_duration <= 0
            or synced_duration <= 0.05
            or removed_duration < -0.05
            or removed_duration > max_safe_removal
        ):
            synced.unlink(missing_ok=True)
            return source

        source.unlink(missing_ok=True)
        return synced
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        synced.unlink(missing_ok=True)
        return source


async def synthesize(
    client: httpx.AsyncClient,
    text: str,
    voice: str,
    emotion: str,
    delivery: str,
    target_duration: float,
    pace: str = "normal",
    energy: str = "medium",
    pitch_tendency: str = "variable",
    intonation_contour: str = "natural",
    emphasized_words: list[str] | None = None,
    pauses: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    voice = voice if voice in VOICES else "Kore"
    target_duration = max(0.4, min(float(target_duration or 3), 30.0))
    emphasis = ", ".join((emphasized_words or [])[:8]) or "none"
    pause_directions = "; ".join(
        f"after translated word {int(item.get('after_word_index', 0)) + 1}, make a {int(item.get('duration_ms', 150))} ms pause"
        for item in (pauses or [])[:6]
        if isinstance(item, dict)
    ) or "no special internal pauses"
    prompt = (
        "Perform the translated dialogue like a professional film dubbing actor. "
        f"Line-specific direction: {delivery or 'natural'}. "
        f"Emotion: {emotion or 'neutral'}; pace: {pace}; energy: {energy}; "
        f"pitch tendency: {pitch_tendency}; intonation contour: {intonation_contour}. "
        f"Emphasize these translated words naturally: {emphasis}. Pause plan: {pause_directions}. "
        f"Finish naturally in approximately {target_duration:.1f} seconds. "
        "Start speaking immediately with no introductory pause or leading silence. "
        "Instructions describe acting style, not words to read aloud. Speak only this dialogue: " + text
    )
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}},
        },
    }
    try:
        endpoint = vertex_model_url(TTS_MODEL)
        response = await client.post(
            endpoint,
            headers=vertex_headers(),
            json=payload,
        )
    except ValueError as exc:
        return {
            "ok": False,
            "error": {"code": "VERTEX_CONFIG_ERROR", "message": str(exc), "help": []},
        }
    except httpx.RequestError as exc:
        return {
            "ok": False,
            "error": {
                "code": "VERTEX_UNREACHABLE",
                "message": f"Не удалось связаться с Vertex AI: {exc}",
                "help": ["Проверьте интернет, VPN/прокси и повторите через минуту."],
            },
        }
    if response.status_code != 200:
        exc = api_error(response, "tts")
        return {"ok": False, "error": exc.detail}

    try:
        parts = response.json()["candidates"][0]["content"]["parts"]
        inline = next(part["inlineData"] for part in parts if "inlineData" in part)
        raw = base64.b64decode(inline["data"])
        mime = inline.get("mimeType", "audio/wav")
        if "pcm" in mime.lower() or "l16" in mime.lower():
            raw = pcm_to_wav(raw, parse_pcm_rate(mime))
        suffix = ".wav" if ("wav" in mime.lower() or "pcm" in mime.lower() or "l16" in mime.lower()) else ".mp3"
        filename = f"voice_{uuid.uuid4().hex[:10]}{suffix}"
        source_path = UPLOADS / filename
        source_path.write_bytes(raw)
        synced_path = trim_tts_leading_silence(source_path)
        audio_duration = probe_duration(synced_path)
        duration_ratio = audio_duration / target_duration if target_duration > 0 and audio_duration > 0 else 1.0
        recommended_rate = min(1.08, duration_ratio) if duration_ratio > 1.0 else 1.0
        return {
            "ok": True,
            "url": f"/uploads/{synced_path.name}",
            "filename": synced_path.name,
            "voice": voice,
            "leadingSilenceTrimmed": synced_path != source_path,
            "audioDuration": round(audio_duration, 3),
            "targetDuration": round(target_duration, 3),
            "recommendedRate": round(recommended_rate, 4),
            "durationWarning": "too_long" if duration_ratio > 1.08 else None,
        }
    except Exception as exc:
        return {"ok": False, "error": {"code": "BAD_TTS_RESPONSE", "message": str(exc), "help": []}}


@app.post("/api/voice")
async def voice(request: Request) -> dict[str, Any]:
    """Synthesize one or more segments, each with its speaker-specific voice."""
    if not API_KEY or API_KEY in {"ваш_ключ_сюда", "ваш_vertex_ключ"}:
        raise HTTPException(500, {"code": "NO_KEY", "message": "Вставьте VERTEX_API_KEY в .env", "help": []})

    body = await request.json()
    output = []
    async with httpx.AsyncClient(timeout=180) as client:
        for segment in body.get("segments", []):
            text = str(segment.get("text", "")).strip()
            if not text:
                output.append({"id": segment.get("id"), "ok": False, "error": {"message": "Пустой текст"}})
                continue
            result = await synthesize(
                client=client,
                text=text,
                voice=str(segment.get("voice") or "Kore"),
                emotion=str(segment.get("emotion") or "neutral"),
                delivery=str(segment.get("delivery_instruction") or segment.get("delivery") or "natural"),
                target_duration=float(segment.get("duration") or 3),
                pace=str(segment.get("pace") or "normal"),
                energy=str(segment.get("energy") or "medium"),
                pitch_tendency=str(segment.get("pitch_tendency") or "variable"),
                intonation_contour=str(segment.get("intonation_contour") or "natural"),
                emphasized_words=segment.get("emphasized_words") if isinstance(segment.get("emphasized_words"), list) else [],
                pauses=segment.get("pauses") if isinstance(segment.get("pauses"), list) else [],
            )
            result["id"] = segment.get("id")
            output.append(result)
    return {"ok": True, "model": TTS_MODEL, "results": output}


def probe_duration(path: Path) -> float:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
            capture_output=True, text=True, timeout=30,
        )
        return max(0.0, float(result.stdout.strip())) if result.returncode == 0 else 0.0
    except (ValueError, FileNotFoundError, subprocess.TimeoutExpired):
        return 0.0


def has_audio_stream(path: Path) -> bool:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=index", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=30,
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def atempo_chain(factor: float) -> str:
    factors: list[float] = []
    factor = max(0.05, min(factor, 20.0))
    while factor > 2.0:
        factors.append(2.0)
        factor /= 2.0
    while factor < 0.5:
        factors.append(0.5)
        factor /= 0.5
    factors.append(factor)
    return ",".join(f"atempo={item:.5f}" for item in factors)


@app.post("/api/export")
async def export(request: Request) -> dict[str, Any]:
    """Mix speaker-specific clips, fitting each clip into its original timing window."""
    body = await request.json()
    video = UPLOADS / Path(body.get("video", "")).name
    segments = body.get("segments", [])
    original_volume = max(0.0, min(float(body.get("originalVolume", 0.15)), 1.0))
    if not video.exists():
        raise HTTPException(404, "Видео не найдено")

    valid_segments = []
    inputs = ["-i", str(video)]
    for segment in segments:
        audio = UPLOADS / Path(segment.get("audio", "")).name
        if audio.exists():
            inputs.extend(["-i", str(audio)])
            valid_segments.append((segment, audio))
    if not valid_segments:
        raise HTTPException(400, "Нет готовых реплик для экспорта")

    video_duration = probe_duration(video)
    if video_duration <= 0:
        return {"ok": False, "error": "ffprobe не смог определить длительность видео. Проверьте установку ffmpeg/ffprobe и формат файла."}
    dubbed_specs: list[tuple[int, float, float, float]] = []
    output_duration = video_duration
    for input_index, (segment, audio) in enumerate(valid_segments, start=1):
        start = max(0.0, float(segment.get("start", 0)))
        requested_target = max(0.2, float(segment.get("end", start + 2)) - start)
        source_duration = probe_duration(audio)
        speed = min(1.08, max(1.0, source_duration / requested_target)) if source_duration > 0 else 1.0
        # Never slow short clips or accelerate long clips by more than 8%. If a
        # generated line is still long, let it finish naturally instead of clipping it.
        target = max(requested_target, source_duration / speed) if source_duration > 0 else requested_target
        output_duration = max(output_duration, start + target)
        dubbed_specs.append((input_index, start, target, speed))

    filters: list[str] = []
    if has_audio_stream(video):
        filters.append(
            f"[0:a]volume={original_volume},apad,atrim=duration={output_duration:.3f}[original]"
        )
    else:
        filters.append(
            f"anullsrc=r=48000:cl=stereo,atrim=duration={output_duration:.3f}[original]"
        )

    dubbed_labels = []
    for input_index, start, target, speed in dubbed_specs:
        label = f"dub{input_index}"
        filters.append(
            f"[{input_index}:a]{atempo_chain(speed)},atrim=duration={target:.3f},"
            f"adelay={int(start * 1000)}|{int(start * 1000)}[{label}]"
        )
        dubbed_labels.append(f"[{label}]")

    filters.append(
        f"[original]{''.join(dubbed_labels)}amix=inputs={len(dubbed_labels) + 1}:"
        f"duration=longest:normalize=0:dropout_transition=0,atrim=duration={output_duration:.3f}[final]"
    )

    tail_extension = max(0.0, output_duration - video_duration)
    video_options = ["-c:v", "copy"]
    if tail_extension > 0.01:
        video_options = [
            "-vf", f"tpad=stop_mode=clone:stop_duration={tail_extension:.3f}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        ]

    output = EXPORTS / f"dub_{uuid.uuid4().hex[:8]}.mp4"
    command = [
        "ffmpeg", "-y", *inputs, "-filter_complex", ";".join(filters),
        "-map", "0:v:0", "-map", "[final]", *video_options, "-c:a", "aac",
        "-b:a", "192k", "-movflags", "+faststart", "-t", f"{output_duration:.3f}", str(output),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=1200)
        if result.returncode != 0:
            return {"ok": False, "error": result.stderr[-800:]}
    except FileNotFoundError:
        return {"ok": False, "error": "ffmpeg/ffprobe не установлен"}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "Экспорт занял больше 20 минут"}

    return {"ok": True, "url": f"/exports/{output.name}", "filename": output.name}


app.mount("/uploads", StaticFiles(directory=str(UPLOADS)), name="uploads")
app.mount("/exports", StaticFiles(directory=str(EXPORTS)), name="exports")

if __name__ == "__main__":
    import uvicorn

    config = vertex_public_config()
    print("\nJarCut v4 Vertex AI → http://localhost:8000")
    print(f"Режим Vertex: {config['mode']}")
    if config["mode"] == "standard":
        print(f"Проект/регион: {config['project']} / {config['location']}")
    print(f"Распознавание: {TRANSCRIBE_MODEL} (таймаут {VERTEX_TRANSCRIBE_TIMEOUT_SECONDS:g} сек.)")
    print(f"Озвучка: {TTS_MODEL}")
    alignment_status = config["alignmentMode"] if config["alignmentAvailable"] else "недоступна (установите faster-whisper)"
    if config["alignmentAvailable"] and config["alignmentMode"] in {"hybrid", "precise"}:
        alignment_status += f" / Whisper {config['whisperModel']}"
    print(f"Синхронизация: {alignment_status}")
    if config["speakerVerificationMode"] == "off":
        speaker_status = "выключена"
    elif config["speakerVerificationAvailable"]:
        speaker_status = f"{config['speakerVerificationMode']} / ECAPA {config['speakerEmbeddingModel']}"
    else:
        speaker_status = (
            f"{config['speakerVerificationMode']} / Whisper fallback "
            "(необязательно: python -m pip install speechbrain)"
        )
    print(f"Проверка голоса: {speaker_status}\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
