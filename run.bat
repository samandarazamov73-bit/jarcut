@echo off
chcp 65001 >nul
echo.
echo   JarCut v2 — Видеоредактор с авто-переводом
echo   ════════════════════════════════════════════
echo.
cd /d "%~dp0"
if not exist .env (
    echo GEMINI_API_KEY=ваш_ключ_сюда> .env
    echo   [!] Создан файл .env
    echo   [!] Вставьте туда ваш Gemini API ключ!
    echo   [!] Получить: https://aistudio.google.com/apikey
    echo.
    pause
)
pip install -q fastapi uvicorn httpx python-dotenv python-multipart 2>nul
echo.
echo   Откройте: http://localhost:8000
echo   Ctrl+C — остановить
echo.
python server.py
pause
