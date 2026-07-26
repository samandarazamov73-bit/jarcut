"""
CapCut-Lite — локальный сервер видеоредактора.
Запуск: uvicorn server:app --reload --port 8000
Или:   python server.py

Что делает:
1. Отдаёт статику (фронтенд) из папки static/
2. Принимает загрузку видео/аудио → uploads/
3. Проксирует запросы к Gemini TTS API (ключ хранится в .env)
4. Сохраняет/загружает проект (JSON) → projects/
5. Экспорт: склейка видео + аудио через ffmpeg → exports/
"""

import os
import json
import uuid
import subprocess
import base64
from pathlib import Path
from datetime import datetime

import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from dotenv import load_dotenv

# ─── Конфигурация ───────────────────────────────────────────────────────────

load_dotenv()  # читаем .env из корня проекта

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_TTS_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-tts:generateContent"

BASE_DIR = Path(__file__).parent
UPLOADS_DIR = BASE_DIR / "uploads"
PROJECTS_DIR = BASE_DIR / "projects"
EXPORTS_DIR = BASE_DIR / "exports"

# Создаём папки если нет
for d in [UPLOADS_DIR, PROJECTS_DIR, EXPORTS_DIR]:
    d.mkdir(exist_ok=True)

# ─── Приложение ─────────────────────────────────────────────────────────────

app = FastAPI(title="CapCut-Lite", version="0.1.0")


# ─── 1. Загрузка файлов ─────────────────────────────────────────────────────

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Загружает видео или аудиофайл в uploads/.
    Возвращает путь, по которому файл доступен.
    """
    # Генерим уникальное имя чтобы не было конфликтов
    ext = Path(file.filename).suffix
    filename = f"{uuid.uuid4().hex}{ext}"
    filepath = UPLOADS_DIR / filename

    content = await file.read()
    filepath.write_bytes(content)

    return {
        "ok": True,
        "filename": filename,
        "url": f"/uploads/{filename}",
        "size": len(content)
    }


# ─── 2. TTS через Gemini API ────────────────────────────────────────────────

@app.post("/api/tts")
async def generate_tts(request: Request):
    """
    Генерирует несколько вариантов озвучки текста через Gemini TTS.
    
    Тело запроса (JSON):
    {
        "text": "Текст для озвучки",
        "variants": 3,                    // сколько вариантов (по умолчанию 3)
        "language": "ru-RU"               // язык (по умолчанию ru-RU)
    }
    
    Возвращает массив вариантов: [{id, audio_url, voice_name, status}, ...]
    """
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY не задан в .env")

    body = await request.json()
    text = body.get("text", "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Поле 'text' обязательно")

    num_variants = body.get("variants", 3)
    language = body.get("language", "ru-RU")

    # Разные голоса для вариантов (Gemini TTS поддерживает несколько)
    voices = ["Kore", "Puck", "Charon", "Fenrir", "Aoede", "Leda", "Orus", "Zephyr"]

    results = []

    async with httpx.AsyncClient(timeout=60.0) as client:
        for i in range(min(num_variants, len(voices))):
            voice_name = voices[i]
            variant_id = uuid.uuid4().hex[:8]

            # Формируем запрос к Gemini TTS API
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": text}
                        ]
                    }
                ],
                "generationConfig": {
                    "responseModalities": ["AUDIO"],
                    "speechConfig": {
                        "voiceConfig": {
                            "prebuiltVoiceConfig": {
                                "voiceName": voice_name
                            }
                        }
                    }
                }
            }

            try:
                resp = await client.post(
                    f"{GEMINI_TTS_URL}?key={GEMINI_API_KEY}",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )

                if resp.status_code != 200:
                    results.append({
                        "id": variant_id,
                        "voice_name": voice_name,
                        "status": "error",
                        "error": f"API вернул {resp.status_code}: {resp.text[:200]}"
                    })
                    continue

                data = resp.json()

                # Извлекаем аудио из ответа Gemini
                candidates = data.get("candidates", [])
                if not candidates:
                    results.append({
                        "id": variant_id,
                        "voice_name": voice_name,
                        "status": "error",
                        "error": "Нет candidates в ответе"
                    })
                    continue

                parts = candidates[0].get("content", {}).get("parts", [])
                audio_part = None
                for part in parts:
                    if "inlineData" in part:
                        audio_part = part["inlineData"]
                        break

                if not audio_part:
                    results.append({
                        "id": variant_id,
                        "voice_name": voice_name,
                        "status": "error",
                        "error": "Нет аудио в ответе"
                    })
                    continue

                # Сохраняем аудио в файл
                audio_bytes = base64.b64decode(audio_part["data"])
                mime_type = audio_part.get("mimeType", "audio/wav")
                ext = ".wav" if "wav" in mime_type else ".mp3" if "mp3" in mime_type else ".ogg"
                audio_filename = f"tts_{variant_id}{ext}"
                audio_path = UPLOADS_DIR / audio_filename
                audio_path.write_bytes(audio_bytes)

                results.append({
                    "id": variant_id,
                    "voice_name": voice_name,
                    "status": "ready",
                    "audio_url": f"/uploads/{audio_filename}",
                    "mime_type": mime_type,
                    "size": len(audio_bytes)
                })

            except httpx.TimeoutException:
                results.append({
                    "id": variant_id,
                    "voice_name": voice_name,
                    "status": "error",
                    "error": "Таймаут запроса к Gemini API"
                })
            except Exception as e:
                results.append({
                    "id": variant_id,
                    "voice_name": voice_name,
                    "status": "error",
                    "error": str(e)
                })

    return {"ok": True, "variants": results}


# ─── 3. Сохранение / загрузка проекта ───────────────────────────────────────

@app.post("/api/project/save")
async def save_project(request: Request):
    """
    Сохраняет проект (JSON с раскладкой маркеров, таймингами, аудио-ссылками).
    
    Тело запроса:
    {
        "name": "мой_проект",     // имя проекта
        "data": { ... }           // вся структура проекта
    }
    """
    body = await request.json()
    name = body.get("name", "untitled")
    data = body.get("data", {})

    # Безопасное имя файла
    safe_name = "".join(c for c in name if c.isalnum() or c in "-_ ").strip() or "untitled"
    filename = f"{safe_name}.json"
    filepath = PROJECTS_DIR / filename

    # Добавляем метаданные
    project = {
        "name": name,
        "savedAt": datetime.now().isoformat(),
        "data": data
    }

    filepath.write_text(json.dumps(project, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"ok": True, "filename": filename, "path": str(filepath)}


@app.get("/api/project/load/{name}")
async def load_project(name: str):
    """Загружает сохранённый проект по имени."""
    safe_name = "".join(c for c in name if c.isalnum() or c in "-_ ").strip()
    filepath = PROJECTS_DIR / f"{safe_name}.json"

    if not filepath.exists():
        raise HTTPException(status_code=404, detail=f"Проект '{name}' не найден")

    project = json.loads(filepath.read_text(encoding="utf-8"))
    return {"ok": True, "project": project}


@app.get("/api/project/list")
async def list_projects():
    """Список всех сохранённых проектов."""
    files = list(PROJECTS_DIR.glob("*.json"))
    projects = []
    for f in sorted(files, key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            projects.append({
                "name": data.get("name", f.stem),
                "filename": f.name,
                "savedAt": data.get("savedAt", "")
            })
        except Exception:
            pass
    return {"ok": True, "projects": projects}


# ─── 4. Экспорт (склейка через ffmpeg) ──────────────────────────────────────

@app.post("/api/export")
async def export_project(request: Request):
    """
    Экспорт: накладывает аудио-маркеры на видео через ffmpeg.
    
    Тело запроса:
    {
        "video": "filename.mp4",          // файл из uploads/
        "markers": [
            {
                "audio": "tts_abc123.wav", // файл из uploads/
                "startTime": 2.5           // секунда начала
            },
            ...
        ],
        "outputName": "result"            // имя выходного файла (без расширения)
    }
    """
    body = await request.json()
    video_file = body.get("video", "")
    markers = body.get("markers", [])
    output_name = body.get("outputName", f"export_{uuid.uuid4().hex[:6]}")

    video_path = UPLOADS_DIR / video_file
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Видео '{video_file}' не найдено")

    # Формируем команду ffmpeg для наложения аудио
    # Стратегия: все аудио-дорожки смешиваются с задержкой (adelay)
    if not markers:
        raise HTTPException(status_code=400, detail="Нет маркеров для экспорта")

    output_path = EXPORTS_DIR / f"{output_name}.mp4"

    # Строим complex filter для ffmpeg
    inputs = ["-i", str(video_path)]
    filter_parts = []
    
    for idx, marker in enumerate(markers):
        audio_file = marker.get("audio", "")
        start_time = marker.get("startTime", 0)
        audio_path = UPLOADS_DIR / audio_file

        if not audio_path.exists():
            continue

        inputs.extend(["-i", str(audio_path)])
        # adelay принимает миллисекунды
        delay_ms = int(start_time * 1000)
        filter_parts.append(f"[{idx + 1}:a]adelay={delay_ms}|{delay_ms}[a{idx}]")

    if not filter_parts:
        raise HTTPException(status_code=400, detail="Нет валидных аудио-маркеров")

    # Смешиваем все аудио-дорожки
    audio_labels = "".join(f"[a{i}]" for i in range(len(filter_parts)))
    mix_filter = f"{audio_labels}amix=inputs={len(filter_parts)}:duration=longest[mixed]"
    
    # Если у видео есть свой звук, добавляем его
    full_filter = ";".join(filter_parts) + ";" + mix_filter

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", full_filter,
        "-map", "0:v",
        "-map", "[mixed]",
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        str(output_path)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            return {
                "ok": False,
                "error": f"ffmpeg ошибка: {result.stderr[-500:]}"
            }
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "ffmpeg таймаут (>5 мин)"}
    except FileNotFoundError:
        return {"ok": False, "error": "ffmpeg не найден. Установите ffmpeg."}

    return {
        "ok": True,
        "output": f"/exports/{output_name}.mp4",
        "path": str(output_path)
    }


# ─── 5. Раздача файлов ──────────────────────────────────────────────────────

# Отдаём загруженные файлы (видео, аудио)
app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")

# Отдаём экспортированные файлы
app.mount("/exports", StaticFiles(directory=str(EXPORTS_DIR)), name="exports")

# Главная страница
@app.get("/")
async def root():
    return FileResponse(str(BASE_DIR / "static" / "index.html"))

# Статика (CSS, JS)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


# ─── Запуск ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print("=" * 50)
    print("  CapCut-Lite запущен!")
    print("  Откройте: http://localhost:8000")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8000)
