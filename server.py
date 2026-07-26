"""
JarCut — локальный видеоредактор с TTS озвучкой.

ЗАПУСК:
    python server.py

Откроется на http://localhost:8000
Нужен файл .env с ключом GEMINI_API_KEY (рядом с этим файлом).
"""

import os, json, uuid, subprocess, base64
from pathlib import Path
from datetime import datetime

import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

# ─── Конфиг ──────────────────────────────────────────────────────────────────

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_TTS_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-tts:generateContent"

BASE = Path(__file__).parent
UPLOADS = BASE / "uploads"
PROJECTS = BASE / "projects"
EXPORTS = BASE / "exports"
for d in [UPLOADS, PROJECTS, EXPORTS]:
    d.mkdir(exist_ok=True)

app = FastAPI()

# ─── Главная страница (отдаёт index.html) ────────────────────────────────────

@app.get("/")
async def root():
    return FileResponse(str(BASE / "index.html"))

# ─── Загрузка файлов ─────────────────────────────────────────────────────────

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix
    name = f"{uuid.uuid4().hex[:8]}{ext}"
    path = UPLOADS / name
    path.write_bytes(await file.read())
    return {"ok": True, "filename": name, "url": f"/uploads/{name}"}

# ─── TTS (Gemini) — генерация нескольких вариантов ────────────────────────────

@app.post("/api/tts")
async def tts(request: Request):
    if not GEMINI_API_KEY:
        raise HTTPException(500, "GEMINI_API_KEY не задан в .env")

    body = await request.json()
    text = body.get("text", "").strip()
    if not text:
        raise HTTPException(400, "Нужен текст")

    voices = ["Kore", "Puck", "Charon", "Fenrir", "Aoede", "Leda", "Orus", "Zephyr"]
    n = min(body.get("variants", 3), len(voices))
    results = []

    async with httpx.AsyncClient(timeout=60) as client:
        for i in range(n):
            vid = uuid.uuid4().hex[:8]
            voice = voices[i]
            payload = {
                "contents": [{"parts": [{"text": text}]}],
                "generationConfig": {
                    "responseModalities": ["AUDIO"],
                    "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}}
                }
            }
            try:
                r = await client.post(f"{GEMINI_TTS_URL}?key={GEMINI_API_KEY}", json=payload)
                if r.status_code != 200:
                    results.append({"id": vid, "voice": voice, "status": "error", "error": f"HTTP {r.status_code}"})
                    continue
                data = r.json()
                parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                audio = next((p["inlineData"] for p in parts if "inlineData" in p), None)
                if not audio:
                    results.append({"id": vid, "voice": voice, "status": "error", "error": "Нет аудио"})
                    continue
                raw = base64.b64decode(audio["data"])
                mime = audio.get("mimeType", "audio/wav")
                ext = ".wav" if "wav" in mime else ".mp3" if "mp3" in mime else ".ogg"
                fname = f"tts_{vid}{ext}"
                (UPLOADS / fname).write_bytes(raw)
                results.append({"id": vid, "voice": voice, "status": "ok", "url": f"/uploads/{fname}", "mime": mime})
            except Exception as e:
                results.append({"id": vid, "voice": voice, "status": "error", "error": str(e)[:100]})

    return {"ok": True, "variants": results}

# ─── Проекты (сохранение/загрузка) ───────────────────────────────────────────

@app.post("/api/save")
async def save(request: Request):
    body = await request.json()
    name = "".join(c for c in body.get("name", "project") if c.isalnum() or c in "-_ ") or "project"
    (PROJECTS / f"{name}.json").write_text(
        json.dumps({"name": name, "saved": datetime.now().isoformat(), "data": body.get("data", {})},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True}

@app.get("/api/load/{name}")
async def load(name: str):
    f = PROJECTS / f"{name}.json"
    if not f.exists():
        raise HTTPException(404, "Не найден")
    return {"ok": True, "project": json.loads(f.read_text("utf-8"))}

@app.get("/api/projects")
async def projects():
    files = sorted(PROJECTS.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
    return {"ok": True, "list": [{"name": f.stem} for f in files]}

# ─── Экспорт (ffmpeg) ────────────────────────────────────────────────────────

@app.post("/api/export")
async def export(request: Request):
    body = await request.json()
    video = UPLOADS / body.get("video", "")
    markers = body.get("markers", [])
    if not video.exists():
        raise HTTPException(404, "Видео не найдено")
    if not markers:
        raise HTTPException(400, "Нет маркеров")

    out = EXPORTS / f"export_{uuid.uuid4().hex[:6]}.mp4"
    inputs = ["-i", str(video)]
    filters = []
    for i, m in enumerate(markers):
        af = UPLOADS / m.get("audio", "")
        if not af.exists(): continue
        inputs += ["-i", str(af)]
        ms = int(m.get("start", 0) * 1000)
        filters.append(f"[{i+1}:a]adelay={ms}|{ms}[a{i}]")
    if not filters:
        raise HTTPException(400, "Нет валидных аудио")
    labels = "".join(f"[a{i}]" for i in range(len(filters)))
    full = ";".join(filters) + f";{labels}amix=inputs={len(filters)}:duration=longest[mix]"
    cmd = ["ffmpeg", "-y", *inputs, "-filter_complex", full, "-map", "0:v", "-map", "[mix]", "-c:v", "copy", "-c:a", "aac", "-shortest", str(out)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            return {"ok": False, "error": r.stderr[-300:]}
    except FileNotFoundError:
        return {"ok": False, "error": "ffmpeg не установлен"}
    return {"ok": True, "url": f"/exports/{out.name}"}

# ─── Статика ─────────────────────────────────────────────────────────────────

app.mount("/uploads", StaticFiles(directory=str(UPLOADS)), name="uploads")
app.mount("/exports", StaticFiles(directory=str(EXPORTS)), name="exports")

# ─── Запуск ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 45)
    print("  JarCut запущен → http://localhost:8000")
    print("=" * 45 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
