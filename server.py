"""
JarCut — Авто-дубляж видео.

Поставил видео → сам распознал речь → перевёл → озвучил как професс. дублёр → скачал.

ЗАПУСК:  python server.py
ОТКРОЙ:  http://localhost:8000
"""

import os, json, uuid, subprocess, base64
from pathlib import Path

import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY", "")
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
MODEL_PRO = "gemini-3.1-pro"                  # распознавание + перевод
MODEL_TTS = "gemini-3.1-flash-tts-preview"    # озвучка

BASE = Path(__file__).parent
UPLOADS, EXPORTS = BASE / "uploads", BASE / "exports"
for d in (UPLOADS, EXPORTS):
    d.mkdir(exist_ok=True)

app = FastAPI()

LANGS = {
    "uz": "Uzbek (O'zbek tili)", "ru": "Russian", "en": "English", "tr": "Turkish",
    "kk": "Kazakh", "ky": "Kyrgyz", "tg": "Tajik", "ar": "Arabic",
    "de": "German", "fr": "French", "es": "Spanish", "ko": "Korean", "ja": "Japanese",
}


@app.get("/")
async def root():
    return FileResponse(str(BASE / "index.html"))


# ─── 1. Загрузка видео ───────────────────────────────────────────────────────

@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    name = f"{uuid.uuid4().hex[:10]}{Path(file.filename).suffix.lower()}"
    (UPLOADS / name).write_bytes(await file.read())
    return {"ok": True, "filename": name, "url": f"/uploads/{name}"}


# ─── 2. Распознать речь + перевести (Gemini 3.1 Pro) ─────────────────────────

@app.post("/api/translate")
async def translate(request: Request):
    """Слушает видео, распознаёт речь, переводит. Возвращает реплики с таймингами."""
    if not API_KEY:
        raise HTTPException(500, "GEMINI_API_KEY не задан в .env")

    body = await request.json()
    path = UPLOADS / body.get("filename", "")
    lang = LANGS.get(body.get("lang", "uz"), "Uzbek")

    if not path.exists():
        raise HTTPException(404, "Файл не найден")

    mime = {".mp4": "video/mp4", ".webm": "video/webm", ".mov": "video/quicktime",
            ".mkv": "video/x-matroska", ".avi": "video/x-msvideo",
            ".mp3": "audio/mp3", ".wav": "audio/wav", ".m4a": "audio/mp4"}.get(path.suffix.lower(), "video/mp4")

    prompt = f"""You are a professional dubbing translator. Listen to this video and transcribe ALL spoken speech, then translate it into {lang}.

Return a JSON array. One item per spoken phrase:
[{{"start": 0.5, "end": 2.3, "original": "...", "translated": "..."}}]

Rules:
- start/end in seconds (float), accurate to the speech
- Translate NATURALLY for dubbing — like a professional voice actor would say it, not word-by-word
- Keep the translated phrase roughly the same spoken length as the original so it fits the timing
- Preserve tone and emotion of the speaker
- Capture EVERY phrase, skip nothing
- Return ONLY the JSON array, no markdown
- If no speech: []"""

    data_b64 = base64.b64encode(path.read_bytes()).decode()
    payload = {
        "contents": [{"parts": [
            {"inlineData": {"mimeType": mime, "data": data_b64}},
            {"text": prompt},
        ]}],
        "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"},
    }

    async with httpx.AsyncClient(timeout=300) as client:
        r = await client.post(f"{BASE_URL}/{MODEL_PRO}:generateContent?key={API_KEY}", json=payload)

    if r.status_code != 200:
        raise HTTPException(r.status_code, f"Gemini: {r.text[:300]}")

    try:
        text = r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        segments = json.loads(text)
    except Exception as e:
        raise HTTPException(500, f"Не удалось разобрать ответ: {e}")

    return {"ok": True, "segments": segments}


# ─── 3. Озвучка одной реплики (Gemini 3.1 Flash TTS) ─────────────────────────

async def _tts_one(client: httpx.AsyncClient, text: str, voice: str) -> dict:
    """Генерирует голос для одной реплики в стиле профессионального дублёра."""
    # Инструкция стиля — Gemini TTS понимает natural language промпты
    styled = f"Say this naturally, like a professional voice actor dubbing a movie: {text}"

    payload = {
        "contents": [{"parts": [{"text": styled}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}},
        },
    }

    r = await client.post(f"{BASE_URL}/{MODEL_TTS}:generateContent?key={API_KEY}", json=payload)
    if r.status_code != 200:
        return {"ok": False, "error": f"HTTP {r.status_code}"}

    parts = r.json()["candidates"][0]["content"]["parts"]
    inline = next((p["inlineData"] for p in parts if "inlineData" in p), None)
    if not inline:
        return {"ok": False, "error": "нет аудио"}

    raw = base64.b64decode(inline["data"])
    mime = inline.get("mimeType", "audio/wav")
    ext = ".wav" if "wav" in mime or "pcm" in mime else ".mp3"
    fname = f"v_{uuid.uuid4().hex[:8]}{ext}"

    # Gemini отдаёт raw PCM — оборачиваем в WAV-контейнер, чтобы играло в браузере
    if "pcm" in mime or "L16" in mime:
        raw = _pcm_to_wav(raw)

    (UPLOADS / fname).write_bytes(raw)
    return {"ok": True, "url": f"/uploads/{fname}", "filename": fname}


def _pcm_to_wav(pcm: bytes, rate: int = 24000, channels: int = 1, bits: int = 16) -> bytes:
    """Добавляет WAV-заголовок к raw PCM (Gemini TTS отдаёт 24kHz mono 16-bit)."""
    import struct
    block = channels * bits // 8
    header = b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt " + \
        struct.pack("<IHHIIHH", 16, 1, channels, rate, rate * block, block, bits) + \
        b"data" + struct.pack("<I", len(pcm))
    return header + pcm


@app.post("/api/voice")
async def voice(request: Request):
    """Озвучивает список реплик. Принимает: {segments:[{id,text}], voice:"Kore"}"""
    if not API_KEY:
        raise HTTPException(500, "GEMINI_API_KEY не задан в .env")

    body = await request.json()
    segments = body.get("segments", [])
    v = body.get("voice", "Kore")
    out = []

    async with httpx.AsyncClient(timeout=120) as client:
        for s in segments:
            text = (s.get("text") or "").strip()
            if not text:
                out.append({"id": s.get("id"), "ok": False, "error": "пусто"})
                continue
            try:
                res = await _tts_one(client, text, v)
            except Exception as e:
                res = {"ok": False, "error": str(e)[:100]}
            res["id"] = s.get("id")
            out.append(res)

    return {"ok": True, "results": out}


# ─── 4. Экспорт: видео + дубляж (ffmpeg) ─────────────────────────────────────

@app.post("/api/export")
async def export(request: Request):
    """Накладывает дубляж на видео, приглушая оригинальный звук."""
    body = await request.json()
    video = UPLOADS / body.get("video", "")
    segs = body.get("segments", [])
    orig_vol = float(body.get("originalVolume", 0.15))

    if not video.exists():
        raise HTTPException(404, "Видео не найдено")
    if not segs:
        raise HTTPException(400, "Нет озвученных реплик")

    out = EXPORTS / f"dub_{uuid.uuid4().hex[:6]}.mp4"
    inputs, filters, n = ["-i", str(video)], [], 0

    for s in segs:
        af = UPLOADS / s.get("audio", "")
        if not af.exists():
            continue
        n += 1
        inputs += ["-i", str(af)]
        ms = int(float(s.get("start", 0)) * 1000)
        filters.append(f"[{n}:a]adelay={ms}|{ms}[a{n}]")

    if not n:
        raise HTTPException(400, "Аудиофайлы не найдены")

    labels = "".join(f"[a{i}]" for i in range(1, n + 1))
    fc = (f"[0:a]volume={orig_vol}[orig];" + ";".join(filters) +
          f";[orig]{labels}amix=inputs={n + 1}:duration=first:normalize=0[out]")

    cmd = ["ffmpeg", "-y", *inputs, "-filter_complex", fc,
           "-map", "0:v", "-map", "[out]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", str(out)]

    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        if r.returncode != 0:
            return {"ok": False, "error": r.stderr[-400:]}
    except FileNotFoundError:
        return {"ok": False, "error": "ffmpeg не установлен. Скачайте: ffmpeg.org"}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "ffmpeg таймаут"}

    return {"ok": True, "url": f"/exports/{out.name}"}


app.mount("/uploads", StaticFiles(directory=str(UPLOADS)), name="uploads")
app.mount("/exports", StaticFiles(directory=str(EXPORTS)), name="exports")


if __name__ == "__main__":
    import uvicorn
    print("\n  JarCut → http://localhost:8000\n")
    if not API_KEY:
        print("  [!] Вставьте GEMINI_API_KEY в файл .env")
        print("      Ключ: https://aistudio.google.com/apikey\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
