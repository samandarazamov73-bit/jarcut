"""
JarCut v2 — Видеоредактор с авто-переводом и озвучкой.

Workflow:
1. Загрузил видео
2. Нажал "Перевести" → Gemini 3.1 Pro слушает видео, распознаёт речь,
   переводит на нужный язык, возвращает реплики с таймингами
3. Редактируешь текст если надо
4. Нажал "Озвучить" → Gemini 3.1 Flash TTS генерирует голос
5. Регулируешь громкость оригинала
6. Экспорт через ffmpeg

ЗАПУСК:  python server.py
ОТКРОЙ:  http://localhost:8000
"""

import os, json, uuid, subprocess, base64, tempfile
from pathlib import Path
from datetime import datetime

import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

# ─── Конфиг ──────────────────────────────────────────────────────────────────

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY", "")
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

# Модели Gemini 3.1
MODEL_PRO = "gemini-3.1-pro"           # для транскрипции + перевода
MODEL_TTS = "gemini-3.1-flash-tts-preview"  # для озвучки

BASE = Path(__file__).parent
UPLOADS = BASE / "uploads"
PROJECTS = BASE / "projects"
EXPORTS = BASE / "exports"
for d in [UPLOADS, PROJECTS, EXPORTS]:
    d.mkdir(exist_ok=True)

app = FastAPI()

# ─── Главная страница ─────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return FileResponse(str(BASE / "index.html"))

# ─── Загрузка файлов ─────────────────────────────────────────────────────────

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    name = f"{uuid.uuid4().hex[:10]}{ext}"
    content = await file.read()
    (UPLOADS / name).write_bytes(content)
    return {"ok": True, "filename": name, "url": f"/uploads/{name}", "size": len(content)}

# ─── Загрузка файла в Gemini File API (для больших видео) ─────────────────────

async def upload_to_gemini(filepath: Path, mime_type: str) -> str:
    """Загружает файл в Gemini Files API, возвращает file URI."""
    url = f"https://generativelanguage.googleapis.com/upload/v1beta/files?key={API_KEY}"
    
    async with httpx.AsyncClient(timeout=300) as client:
        # Сначала создаём upload
        file_size = filepath.stat().st_size
        
        # Для файлов < 20MB отправляем inline
        if file_size < 20 * 1024 * 1024:
            return None  # будем отправлять inline
        
        # Для больших файлов используем File API
        headers = {
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(file_size),
            "X-Goog-Upload-Header-Content-Type": mime_type,
            "Content-Type": "application/json"
        }
        
        meta = {"file": {"displayName": filepath.name}}
        resp = await client.post(url, headers=headers, json=meta)
        
        if resp.status_code != 200:
            return None
            
        upload_url = resp.headers.get("X-Goog-Upload-URL")
        if not upload_url:
            return None
        
        # Загружаем данные
        data = filepath.read_bytes()
        headers2 = {
            "Content-Length": str(file_size),
            "X-Goog-Upload-Offset": "0",
            "X-Goog-Upload-Command": "upload, finalize"
        }
        resp2 = await client.post(upload_url, headers=headers2, content=data)
        
        if resp2.status_code == 200:
            result = resp2.json()
            return result.get("file", {}).get("uri")
    
    return None


# ─── ТРАНСКРИПЦИЯ + ПЕРЕВОД (Gemini 3.1 Pro) ─────────────────────────────────

@app.post("/api/transcribe")
async def transcribe(request: Request):
    """
    Принимает: { "filename": "video.mp4", "targetLang": "uz" }
    Gemini 3.1 Pro смотрит видео, распознаёт речь, переводит.
    Возвращает: { "segments": [{ "start": 0.5, "end": 2.1, "original": "...", "translated": "..." }] }
    """
    if not API_KEY:
        raise HTTPException(500, "GEMINI_API_KEY не задан в .env")

    body = await request.json()
    filename = body.get("filename", "")
    target_lang = body.get("targetLang", "uz")
    
    filepath = UPLOADS / filename
    if not filepath.exists():
        raise HTTPException(404, f"Файл {filename} не найден")

    # Определяем MIME тип
    ext = filepath.suffix.lower()
    mime_map = {".mp4": "video/mp4", ".webm": "video/webm", ".mov": "video/quicktime",
                ".avi": "video/x-msvideo", ".mkv": "video/x-matroska", ".mp3": "audio/mp3",
                ".wav": "audio/wav", ".ogg": "audio/ogg", ".m4a": "audio/mp4"}
    mime_type = mime_map.get(ext, "video/mp4")

    # Определяем язык для промпта
    lang_names = {
        "uz": "Uzbek (O'zbek tili)", "ru": "Russian (Русский)", "en": "English",
        "tr": "Turkish (Türkçe)", "ko": "Korean (한국어)", "ja": "Japanese (日本語)",
        "de": "German (Deutsch)", "fr": "French (Français)", "es": "Spanish (Español)",
        "ar": "Arabic (العربية)", "zh": "Chinese (中文)", "hi": "Hindi (हिन्दी)",
        "kk": "Kazakh (Қазақ)", "ky": "Kyrgyz (Кыргыз)", "tg": "Tajik (Тоҷикӣ)"
    }
    target_name = lang_names.get(target_lang, target_lang)

    # Промпт для Gemini
    prompt = f"""Listen carefully to this video/audio. Transcribe ALL spoken speech and translate it to {target_name}.

Return the result as a JSON array. Each element represents one phrase/sentence with timing:

[
  {{"start": 0.5, "end": 2.3, "original": "original text in source language", "translated": "translated text in {target_name}"}},
  {{"start": 3.1, "end": 5.8, "original": "...", "translated": "..."}},
  ...
]

Rules:
- "start" and "end" are in seconds (float)
- Capture EVERY spoken phrase, don't skip anything
- Keep timing accurate to the speech
- Translate naturally, not word-by-word
- Return ONLY valid JSON array, no markdown, no explanation
- If no speech is detected, return empty array []"""

    # Собираем запрос
    file_size = filepath.stat().st_size
    
    if file_size < 20 * 1024 * 1024:
        # Inline (base64) для файлов < 20MB
        file_data = base64.b64encode(filepath.read_bytes()).decode()
        contents = [{
            "parts": [
                {"inlineData": {"mimeType": mime_type, "data": file_data}},
                {"text": prompt}
            ]
        }]
    else:
        # Используем File API для больших файлов
        file_uri = await upload_to_gemini(filepath, mime_type)
        if not file_uri:
            raise HTTPException(500, "Не удалось загрузить файл в Gemini (слишком большой)")
        contents = [{
            "parts": [
                {"fileData": {"mimeType": mime_type, "fileUri": file_uri}},
                {"text": prompt}
            ]
        }]

    payload = {
        "contents": contents,
        "generationConfig": {
            "temperature": 0.2,
            "responseMimeType": "application/json"
        }
    }

    url = f"{BASE_URL}/{MODEL_PRO}:generateContent?key={API_KEY}"

    async with httpx.AsyncClient(timeout=180) as client:
        resp = await client.post(url, json=payload, headers={"Content-Type": "application/json"})

    if resp.status_code != 200:
        error_text = resp.text[:300]
        raise HTTPException(resp.status_code, f"Gemini ошибка: {error_text}")

    data = resp.json()
    
    # Парсим ответ
    try:
        candidates = data.get("candidates", [])
        text = candidates[0]["content"]["parts"][0]["text"]
        # Убираем возможные markdown обёртки
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
        segments = json.loads(text)
    except (IndexError, KeyError, json.JSONDecodeError) as e:
        raise HTTPException(500, f"Не удалось распарсить ответ Gemini: {str(e)[:200]}")

    return {"ok": True, "segments": segments}


# ─── TTS (Gemini 3.1 Flash TTS) ──────────────────────────────────────────────

@app.post("/api/tts")
async def tts(request: Request):
    """
    Генерация голоса через Gemini 3.1 Flash TTS.
    
    Принимает: { "text": "...", "voice": "Kore", "lang": "uz" }
    Возвращает: { "ok": true, "url": "/uploads/tts_xxx.wav" }
    """
    if not API_KEY:
        raise HTTPException(500, "GEMINI_API_KEY не задан в .env")

    body = await request.json()
    text = body.get("text", "").strip()
    voice = body.get("voice", "Kore")
    
    if not text:
        raise HTTPException(400, "Нужен текст")

    payload = {
        "contents": [{"parts": [{"text": text}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {
                    "prebuiltVoiceConfig": {"voiceName": voice}
                }
            }
        }
    }

    url = f"{BASE_URL}/{MODEL_TTS}:generateContent?key={API_KEY}"

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(url, json=payload, headers={"Content-Type": "application/json"})

    if resp.status_code != 200:
        error_text = resp.text[:300]
        return {"ok": False, "error": f"HTTP {resp.status_code}: {error_text}"}

    data = resp.json()
    
    try:
        parts = data["candidates"][0]["content"]["parts"]
        audio_part = next((p["inlineData"] for p in parts if "inlineData" in p), None)
        if not audio_part:
            return {"ok": False, "error": "Нет аудио в ответе"}
        
        raw = base64.b64decode(audio_part["data"])
        mime = audio_part.get("mimeType", "audio/wav")
        ext = ".wav" if "wav" in mime else ".mp3" if "mp3" in mime else ".ogg"
        fname = f"tts_{uuid.uuid4().hex[:8]}{ext}"
        (UPLOADS / fname).write_bytes(raw)
        
        return {"ok": True, "url": f"/uploads/{fname}", "filename": fname, "mime": mime, "size": len(raw)}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}


# ─── TTS BATCH (озвучить все сегменты разом) ──────────────────────────────────

@app.post("/api/tts-batch")
async def tts_batch(request: Request):
    """
    Озвучивает массив сегментов.
    Принимает: { "segments": [{"text": "...", "id": "seg_0"}], "voice": "Kore" }
    """
    if not API_KEY:
        raise HTTPException(500, "GEMINI_API_KEY не задан в .env")

    body = await request.json()
    segments = body.get("segments", [])
    voice = body.get("voice", "Kore")
    
    results = []
    
    async with httpx.AsyncClient(timeout=60) as client:
        for seg in segments:
            text = seg.get("text", "").strip()
            seg_id = seg.get("id", "")
            
            if not text:
                results.append({"id": seg_id, "ok": False, "error": "Пустой текст"})
                continue

            payload = {
                "contents": [{"parts": [{"text": text}]}],
                "generationConfig": {
                    "responseModalities": ["AUDIO"],
                    "speechConfig": {
                        "voiceConfig": {
                            "prebuiltVoiceConfig": {"voiceName": voice}
                        }
                    }
                }
            }

            url = f"{BASE_URL}/{MODEL_TTS}:generateContent?key={API_KEY}"
            
            try:
                resp = await client.post(url, json=payload, headers={"Content-Type": "application/json"})
                
                if resp.status_code != 200:
                    results.append({"id": seg_id, "ok": False, "error": f"HTTP {resp.status_code}"})
                    continue

                data = resp.json()
                parts = data["candidates"][0]["content"]["parts"]
                audio_part = next((p["inlineData"] for p in parts if "inlineData" in p), None)
                
                if not audio_part:
                    results.append({"id": seg_id, "ok": False, "error": "Нет аудио"})
                    continue
                
                raw = base64.b64decode(audio_part["data"])
                mime = audio_part.get("mimeType", "audio/wav")
                ext = ".wav" if "wav" in mime else ".mp3" if "mp3" in mime else ".ogg"
                fname = f"tts_{uuid.uuid4().hex[:8]}{ext}"
                (UPLOADS / fname).write_bytes(raw)
                
                results.append({"id": seg_id, "ok": True, "url": f"/uploads/{fname}", "filename": fname})
            except Exception as e:
                results.append({"id": seg_id, "ok": False, "error": str(e)[:100]})
    
    return {"ok": True, "results": results}


# ─── Проекты ─────────────────────────────────────────────────────────────────

@app.post("/api/save")
async def save(request: Request):
    body = await request.json()
    name = "".join(c for c in body.get("name", "project") if c.isalnum() or c in "-_ ") or "project"
    (PROJECTS / f"{name}.json").write_text(
        json.dumps({"name": name, "saved": datetime.now().isoformat(), "data": body.get("data", {})},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True}

@app.get("/api/projects")
async def projects():
    files = sorted(PROJECTS.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
    return {"ok": True, "list": [{"name": f.stem} for f in files]}

@app.get("/api/load/{name}")
async def load(name: str):
    f = PROJECTS / f"{name}.json"
    if not f.exists():
        raise HTTPException(404, "Не найден")
    return {"ok": True, "project": json.loads(f.read_text("utf-8"))}


# ─── Экспорт (ffmpeg с регулировкой громкости) ───────────────────────────────

@app.post("/api/export")
async def export(request: Request):
    """
    Экспорт видео с наложенной озвучкой.
    
    Принимает: {
        "video": "filename.mp4",
        "segments": [{"audio": "tts_xxx.wav", "start": 1.5}],
        "originalVolume": 0.3,   // 0.0-1.0 громкость оригинала
        "outputName": "result"
    }
    """
    body = await request.json()
    video_file = body.get("video", "")
    segments = body.get("segments", [])
    orig_volume = body.get("originalVolume", 0.3)
    output_name = body.get("outputName", f"export_{uuid.uuid4().hex[:6]}")

    video_path = UPLOADS / video_file
    if not video_path.exists():
        raise HTTPException(404, "Видео не найдено")
    if not segments:
        raise HTTPException(400, "Нет сегментов для экспорта")

    output_path = EXPORTS / f"{output_name}.mp4"

    # Строим ffmpeg команду
    inputs = ["-i", str(video_path)]
    filter_parts = []
    valid_count = 0

    for i, seg in enumerate(segments):
        audio_file = seg.get("audio", "")
        start = seg.get("start", 0)
        audio_path = UPLOADS / audio_file
        if not audio_path.exists():
            continue
        inputs += ["-i", str(audio_path)]
        delay_ms = int(start * 1000)
        valid_count += 1
        filter_parts.append(f"[{valid_count}:a]adelay={delay_ms}|{delay_ms}[a{valid_count}]")

    if not filter_parts:
        raise HTTPException(400, "Нет валидных аудио файлов")

    # Фильтр: приглушаем оригинальный звук + миксуем TTS
    orig_vol_filter = f"[0:a]volume={orig_volume}[orig]"
    tts_labels = "".join(f"[a{i+1}]" for i in range(len(filter_parts)))
    mix_inputs = f"[orig]{tts_labels}"
    mix_filter = f"{mix_inputs}amix=inputs={len(filter_parts)+1}:duration=first:dropout_transition=2[final]"
    
    full_filter = orig_vol_filter + ";" + ";".join(filter_parts) + ";" + mix_filter

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", full_filter,
        "-map", "0:v",
        "-map", "[final]",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        str(output_path)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            return {"ok": False, "error": result.stderr[-500:]}
    except FileNotFoundError:
        return {"ok": False, "error": "ffmpeg не установлен! Установите: https://ffmpeg.org"}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "Таймаут ffmpeg (>10 мин)"}

    return {"ok": True, "url": f"/exports/{output_name}.mp4", "filename": f"{output_name}.mp4"}


# ─── Статика ─────────────────────────────────────────────────────────────────

app.mount("/uploads", StaticFiles(directory=str(UPLOADS)), name="uploads")
app.mount("/exports", StaticFiles(directory=str(EXPORTS)), name="exports")

# ─── Запуск ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print()
    print("  ╔═══════════════════════════════════════════╗")
    print("  ║    JarCut v2 — Видеоредактор             ║")
    print("  ║    http://localhost:8000                  ║")
    print("  ╚═══════════════════════════════════════════╝")
    print()
    if not API_KEY:
        print("  ⚠️  ВНИМАНИЕ: GEMINI_API_KEY не задан в .env!")
        print("     Получить ключ: https://aistudio.google.com/apikey")
        print()
    uvicorn.run(app, host="0.0.0.0", port=8000)
