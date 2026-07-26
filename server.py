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
    alignment_dependency_available = importlib.util.find_spec("faster_whisper") is not None
    return {
        "backend": "Vertex AI",
        "mode": resolved_vertex_mode(),
        "project": VERTEX_PROJECT_ID or "express-mode",
        "location": VERTEX_LOCATION if resolved_vertex_mode() == "standard" else "global",
        "transcribeModel": TRANSCRIBE_MODEL,
        "ttsModel": TTS_MODEL,
        "alignmentMode": alignment_mode,
        "alignmentAvailable": alignment_mode == "off" or alignment_dependency_available,
        "alignmentDependencyAvailable": alignment_dependency_available,
        "whisperModel": WHISPER_MODEL if alignment_mode in {"hybrid", "precise"} else "disabled",
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
            "code": "ALIGNMENT_CONFIG_ERROR" if alignment_error else "VERTEX_CONFIG_ERROR",
            "message": message,
            "help": [
                "Укажите ALIGNMENT_MODE=off, fast, hybrid или precise в .env."
                if alignment_error
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
    """Mark meaningful cross-speaker overlap while ignoring timestamp rounding noise."""
    for index, segment in enumerate(segments):
        start = float(segment.get("start", 0))
        end = float(segment.get("end", start))
        speaker_id = str(segment.get("speaker_id", ""))
        segment["is_overlap"] = any(
            other_index != index
            and str(other.get("speaker_id", "")) != speaker_id
            and min(end, float(other.get("end", 0)))
            - max(start, float(other.get("start", 0))) >= 0.08
            for other_index, other in enumerate(segments)
        )


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
        shared = len(group) > 1
        boundaries = _energy_split_boundaries(audio, region, group) if shared else []
        split_succeeded = boundaries is not None

        if not shared:
            auto_start, auto_end = region["start"], region["end"]
        elif split_succeeded:
            position = indices.index(index)
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
        if not segment.get("is_overlap")
        and (precise_all or float(segment.get("alignment_confidence", 0)) < ALIGNMENT_LOW_CONFIDENCE)
    ]
    candidates.sort(key=lambda item: item["gemini_start"])
    windows: list[dict[str, Any]] = []
    for segment in candidates:
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
                words.append({
                    "word": normalized[0],
                    "start": float(word.start),
                    "end": float(word.end),
                    "probability": float(word.probability or 0),
                })
    return words


def _best_word_sequence(
    segment: dict[str, Any],
    words: list[dict[str, Any]],
    min_word_start: float = -math.inf,
) -> tuple[float, float, float] | None:
    target = _normalize_words(str(segment.get("original") or ""))
    if not target:
        return None
    window_start = float(segment["gemini_start"]) - 1.5
    window_end = float(segment["gemini_end"]) + 1.5
    candidates = [
        word for word in words
        if word["start"] >= min_word_start - 0.001
        and word["end"] >= window_start
        and word["start"] <= window_end
    ]
    if not candidates:
        return None

    target_text = " ".join(target)
    expected_start = float(segment["gemini_start"])
    best: tuple[float, float, float] | None = None
    min_length, max_length = max(1, len(target) - 3), len(target) + 4
    for start_index in range(len(candidates)):
        for length in range(min_length, max_length + 1):
            chosen = candidates[start_index:start_index + length]
            if not chosen:
                continue
            ratio = difflib.SequenceMatcher(
                None, target_text, " ".join(word["word"] for word in chosen)
            ).ratio()
            proximity = max(0.0, 1.0 - abs(chosen[0]["start"] - expected_start) / 1.5)
            probability = sum(word["probability"] for word in chosen) / len(chosen)
            score = 0.70 * ratio + 0.15 * proximity + 0.15 * probability
            if best is None or score > best[0]:
                best = (score, chosen[0]["start"], chosen[-1]["end"])
    return best


def refine_segments_with_whisper(
    segments: list[dict[str, Any]], words: list[dict[str, Any]], precise_all: bool = False
) -> int:
    refined = 0
    consumed_until = -math.inf
    candidates = sorted(
        (
            segment for segment in segments
            if not segment.get("is_overlap")
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
        return {"mode": mode, "vadRegions": 0, "whisperRefined": 0, "warning": None}

    audio, regions, warning = run_silero_vad(source)
    if audio is not None and regions:
        align_segments_with_vad(segments, audio, regions)

    whisper_refined = 0
    if mode in {"hybrid", "precise"} and importlib.util.find_spec("faster_whisper") is not None:
        windows = merge_alignment_windows(segments, precise_all=mode == "precise")
        if windows:
            try:
                words = run_whisper_for_windows(source, windows)
                whisper_refined = refine_segments_with_whisper(
                    segments, words, precise_all=mode == "precise"
                )
            except Exception as exc:
                warning = f"Precise fallback недоступен: {exc}"

    for segment in segments:
        segment["start"] = float(segment["auto_start"]) + float(segment.get("manual_offset", 0))
        segment["end"] = float(segment["auto_end"]) + float(segment.get("manual_offset", 0))
        segment["effective_start"] = segment["start"]
        segment["effective_end"] = segment["end"]

    return {
        "mode": mode,
        "vadRegions": len(regions),
        "whisperRefined": whisper_refined,
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
            async with httpx.AsyncClient(timeout=600) as client:
                response = await client.post(
                    endpoint,
                    headers=vertex_headers(),
                    json=payload,
                )
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
    print(f"Распознавание: {TRANSCRIBE_MODEL}")
    print(f"Озвучка: {TTS_MODEL}")
    alignment_status = config["alignmentMode"] if config["alignmentAvailable"] else "недоступна (установите faster-whisper)"
    if config["alignmentAvailable"] and config["alignmentMode"] in {"hybrid", "precise"}:
        alignment_status += f" / Whisper {config['whisperModel']}"
    print(f"Синхронизация: {alignment_status}\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
