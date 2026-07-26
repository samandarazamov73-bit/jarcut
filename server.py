"""JarCut — локальный автоматический дубляж видео на Gemini 3.1.

Запуск: python server.py
Открыть: http://localhost:8000
"""

import asyncio
import base64
import json
import os
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

API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
API_ROOT = "https://generativelanguage.googleapis.com"
TRANSCRIBE_MODEL = os.getenv("GEMINI_TRANSCRIBE_MODEL", "gemini-3.1-pro-preview")
TTS_MODEL = os.getenv("GEMINI_TTS_MODEL", "gemini-3.1-flash-tts-preview")

BASE = Path(__file__).resolve().parent
UPLOADS = BASE / "uploads"
EXPORTS = BASE / "exports"
for directory in (UPLOADS, EXPORTS):
    directory.mkdir(exist_ok=True)

app = FastAPI(title="JarCut", version="3.0")

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


def gemini_headers() -> dict[str, str]:
    return {"Content-Type": "application/json", "x-goog-api-key": API_KEY}


def api_error(response: httpx.Response, action: str) -> HTTPException:
    """Translate Gemini errors to short, actionable messages without exposing the key."""
    try:
        payload = response.json()
        message = payload.get("error", {}).get("message", response.text[:300])
    except Exception:
        message = response.text[:300]

    if response.status_code == 403:
        detail = {
            "code": "GEMINI_KEY_BLOCKED",
            "message": f"Gemini запретил запрос: {message}",
            "help": [
                "Откройте Google Cloud Console → APIs & Services → Credentials.",
                "Выберите ваш API key → API restrictions → Restrict key.",
                "Разрешите Generative Language API и сохраните изменения.",
                "Если ключ из AI Studio всё равно заблокирован — создайте новый ключ в новом Google Cloud проекте с включённым billing.",
                "Перезапустите run.bat после изменения .env.",
            ],
            "action": action,
        }
        return HTTPException(status_code=403, detail=detail)

    if response.status_code == 404:
        return HTTPException(
            status_code=404,
            detail={
                "code": "MODEL_NOT_AVAILABLE",
                "message": f"Модель недоступна для этого ключа: {message}",
                "help": [
                    f"Проверьте доступ к {TRANSCRIBE_MODEL} и {TTS_MODEL} в Google AI Studio.",
                    "Preview-модели могут требовать billing или быть недоступны в вашем регионе.",
                ],
                "action": action,
            },
        )

    return HTTPException(
        status_code=response.status_code,
        detail={"code": "GEMINI_ERROR", "message": message, "help": [], "action": action},
    )


@app.get("/api/diagnostics")
async def diagnostics() -> dict[str, Any]:
    """Check the key and show which Gemini 3.1 models this key can see."""
    if not API_KEY or API_KEY == "ваш_ключ_сюда":
        return {
            "ok": False,
            "code": "NO_KEY",
            "message": "В .env не вставлен GEMINI_API_KEY.",
            "models": [],
        }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(f"{API_ROOT}/v1beta/models", headers={"x-goog-api-key": API_KEY})
    except httpx.RequestError as exc:
        return {
            "ok": False,
            "code": "GEMINI_UNREACHABLE",
            "message": f"Сервер Google Gemini недоступен: {exc}",
            "help": ["Проверьте интернет, VPN/прокси и повторите через минуту."],
            "models": [],
            "transcribeModel": TRANSCRIBE_MODEL,
            "ttsModel": TTS_MODEL,
        }

    if response.status_code != 200:
        exc = api_error(response, "diagnostics")
        return {
            "ok": False,
            **exc.detail,
            "transcribeModel": TRANSCRIBE_MODEL,
            "ttsModel": TTS_MODEL,
        }

    models = [item.get("name", "").replace("models/", "") for item in response.json().get("models", [])]
    models_31 = sorted(model for model in models if "3.1" in model)
    transcribe_available = TRANSCRIBE_MODEL in models
    tts_available = TTS_MODEL in models
    missing = []
    if not transcribe_available:
        missing.append(TRANSCRIBE_MODEL)
    if not tts_available:
        missing.append(TTS_MODEL)
    ready = not missing
    return {
        "ok": ready,
        "keyValid": True,
        "transcribeAvailable": transcribe_available,
        "ttsAvailable": tts_available,
        "transcribeModel": TRANSCRIBE_MODEL,
        "ttsModel": TTS_MODEL,
        "models": models_31,
        "message": "Gemini 3.1 готов" if ready else "Ключ работает, но нет доступа: " + ", ".join(missing),
        "help": [] if ready else [
            "Откройте модели в Google AI Studio и проверьте доступ Preview для вашего проекта.",
            "Для Preview-моделей может потребоваться подключённый billing и поддерживаемый регион.",
        ],
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
        "-codec:a", "libmp3lame", "-b:a", "64k", str(target),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=600)
        return result.returncode == 0 and target.exists() and target.stat().st_size > 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


async def upload_gemini_file(client: httpx.AsyncClient, path: Path, mime: str) -> str:
    """Upload larger audio through the resumable Gemini Files API."""
    start_headers = {
        "x-goog-api-key": API_KEY,
        "X-Goog-Upload-Protocol": "resumable",
        "X-Goog-Upload-Command": "start",
        "X-Goog-Upload-Header-Content-Length": str(path.stat().st_size),
        "X-Goog-Upload-Header-Content-Type": mime,
        "Content-Type": "application/json",
    }
    start = await client.post(
        f"{API_ROOT}/upload/v1beta/files",
        headers=start_headers,
        json={"file": {"displayName": path.name}},
    )
    if start.status_code not in (200, 201):
        raise api_error(start, "file_upload")

    upload_url = start.headers.get("X-Goog-Upload-URL")
    if not upload_url:
        raise HTTPException(500, "Gemini не вернул адрес загрузки файла")

    upload = await client.post(
        upload_url,
        headers={
            "X-Goog-Upload-Offset": "0",
            "X-Goog-Upload-Command": "upload, finalize",
            "Content-Length": str(path.stat().st_size),
        },
        content=path.read_bytes(),
    )
    if upload.status_code not in (200, 201):
        raise api_error(upload, "file_upload")

    file_info = upload.json().get("file", {})
    uri = file_info.get("uri", "")
    name = file_info.get("name", "")
    state = file_info.get("state", "ACTIVE")
    for _ in range(60):
        if state == "ACTIVE":
            return uri
        if state == "FAILED":
            raise HTTPException(500, "Gemini не смог обработать загруженное аудио")
        if not name:
            break
        await asyncio.sleep(2)
        status = await client.get(f"{API_ROOT}/v1beta/{name}", headers={"x-goog-api-key": API_KEY})
        if status.status_code != 200:
            raise api_error(status, "file_processing")
        file_info = status.json()
        uri = file_info.get("uri", uri)
        state = file_info.get("state", "")

    raise HTTPException(504, "Gemini слишком долго обрабатывает аудио")


def professional_prompt(target_language: str) -> str:
    return f"""You are a senior dubbing director, dialogue editor, translator, and speaker-diarization expert.
Listen to the supplied audio and create a professional dubbing script translated into {target_language}.

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
      "emotion": "neutral|happy|sad|angry|excited|whispering|serious"
    }}
  ]
}}

Requirements:
- Accurate start/end timestamps in seconds.
- Capture every spoken phrase and audible speaker.
- Translate for natural professional dubbing, not word-for-word.
- Keep translated speech short enough to fit the original time window.
- Preserve intent, politeness, humor and emotion.
- Never infer identity or actual gender; voice_character only describes audible vocal presentation for voice casting.
- Return JSON only. If there is no speech, return {{"speakers":[],"segments":[]}}."""


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
    if not API_KEY or API_KEY == "ваш_ключ_сюда":
        raise HTTPException(500, {"code": "NO_KEY", "message": "Вставьте Gemini API key в .env", "help": []})

    body = await request.json()
    source = UPLOADS / Path(body.get("filename", "")).name
    target_language = LANGS.get(body.get("lang", "uz"), LANGS["uz"])
    if not source.exists():
        raise HTTPException(404, "Файл не найден")

    with tempfile.TemporaryDirectory(prefix="jarcut_") as temp_dir:
        compact_audio = Path(temp_dir) / "speech.mp3"
        if extract_audio(source, compact_audio):
            media_path, mime = compact_audio, "audio/mpeg"
        else:
            media_path = source
            mime = MIME_TYPES.get(source.suffix.lower(), "video/mp4")

        prompt = professional_prompt(target_language)
        async with httpx.AsyncClient(timeout=600) as client:
            if media_path.stat().st_size <= 15 * 1024 * 1024:
                media_part = {
                    "inlineData": {
                        "mimeType": mime,
                        "data": base64.b64encode(media_path.read_bytes()).decode("ascii"),
                    }
                }
            else:
                uri = await upload_gemini_file(client, media_path, mime)
                if not uri:
                    raise HTTPException(500, "Gemini File API не вернул URI")
                media_part = {"fileData": {"mimeType": mime, "fileUri": uri}}

            payload = {
                "contents": [{"parts": [media_part, {"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.1,
                    "responseMimeType": "application/json",
                },
            }
            try:
                response = await client.post(
                    f"{API_ROOT}/v1beta/models/{TRANSCRIBE_MODEL}:generateContent",
                    headers=gemini_headers(),
                    json=payload,
                )
            except httpx.RequestError as exc:
                raise HTTPException(
                    502,
                    {
                        "code": "GEMINI_UNREACHABLE",
                        "message": f"Не удалось связаться с Google Gemini: {exc}",
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
        raise HTTPException(500, {"code": "BAD_GEMINI_JSON", "message": f"Не удалось разобрать сценарий: {exc}", "help": []})

    raw_speakers = result.get("speakers") or []
    segments = result.get("segments") or []

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

    speakers = assign_speaker_voices(speakers)
    return {
        "ok": True,
        "model": TRANSCRIBE_MODEL,
        "speakers": speakers,
        "segments": segments,
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


async def synthesize(
    client: httpx.AsyncClient,
    text: str,
    voice: str,
    emotion: str,
    delivery: str,
    target_duration: float,
) -> dict[str, Any]:
    voice = voice if voice in VOICES else "Kore"
    target_duration = max(0.4, min(float(target_duration or 3), 30.0))
    prompt = (
        "Perform the following translated line like a professional film dubbing actor. "
        f"Voice direction: {delivery or 'natural'}, emotion: {emotion or 'neutral'}. "
        f"Finish naturally in approximately {target_duration:.1f} seconds. "
        "Speak only the dialogue; do not read these instructions. Dialogue: " + text
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}},
        },
    }
    try:
        response = await client.post(
            f"{API_ROOT}/v1beta/models/{TTS_MODEL}:generateContent",
            headers=gemini_headers(),
            json=payload,
        )
    except httpx.RequestError as exc:
        return {
            "ok": False,
            "error": {
                "code": "GEMINI_UNREACHABLE",
                "message": f"Не удалось связаться с Google Gemini: {exc}",
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
        (UPLOADS / filename).write_bytes(raw)
        return {"ok": True, "url": f"/uploads/{filename}", "filename": filename, "voice": voice}
    except Exception as exc:
        return {"ok": False, "error": {"code": "BAD_TTS_RESPONSE", "message": str(exc), "help": []}}


@app.post("/api/voice")
async def voice(request: Request) -> dict[str, Any]:
    """Synthesize one or more segments, each with its speaker-specific voice."""
    if not API_KEY or API_KEY == "ваш_ключ_сюда":
        raise HTTPException(500, {"code": "NO_KEY", "message": "Вставьте Gemini API key в .env", "help": []})

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
                delivery=str(segment.get("delivery") or "natural"),
                target_duration=float(segment.get("duration") or 3),
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
    filters: list[str] = []
    safe_video_duration = video_duration
    if has_audio_stream(video):
        filters.append(
            f"[0:a]volume={original_volume},apad,atrim=duration={safe_video_duration:.3f}[original]"
        )
    else:
        filters.append(
            f"anullsrc=r=48000:cl=stereo,atrim=duration={safe_video_duration:.3f}[original]"
        )

    dubbed_labels = []
    for input_index, (segment, audio) in enumerate(valid_segments, start=1):
        start = max(0.0, float(segment.get("start", 0)))
        target = max(0.25, float(segment.get("end", start + 2)) - start)
        source_duration = probe_duration(audio)
        speed = source_duration / target if source_duration > 0 else 1.0
        label = f"dub{input_index}"
        filters.append(
            f"[{input_index}:a]{atempo_chain(speed)},atrim=duration={target:.3f},"
            f"adelay={int(start * 1000)}|{int(start * 1000)}[{label}]"
        )
        dubbed_labels.append(f"[{label}]")

    filters.append(
        f"[original]{''.join(dubbed_labels)}amix=inputs={len(dubbed_labels) + 1}:"
        f"duration=longest:normalize=0:dropout_transition=0,atrim=duration={safe_video_duration:.3f}[final]"
    )

    output = EXPORTS / f"dub_{uuid.uuid4().hex[:8]}.mp4"
    command = [
        "ffmpeg", "-y", *inputs, "-filter_complex", ";".join(filters),
        "-map", "0:v:0", "-map", "[final]", "-c:v", "copy", "-c:a", "aac",
        "-b:a", "192k", "-movflags", "+faststart", "-t", f"{safe_video_duration:.3f}", str(output),
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

    print("\nJarCut → http://localhost:8000")
    print(f"Распознавание: {TRANSCRIBE_MODEL}")
    print(f"Озвучка: {TTS_MODEL}\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
