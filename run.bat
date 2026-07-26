@echo off
chcp 65001 >nul
echo.
echo   JarCut — Видеоредактор
echo   ═══════════════════════
echo.
cd /d "%~dp0"
if not exist .env (
    echo GEMINI_API_KEY=ваш_ключ_сюда> .env
    echo  Создан файл .env — вставьте туда ваш Gemini ключ!
    echo  Получить: https://aistudio.google.com/apikey
    echo.
    pause
)
pip install -q fastapi uvicorn httpx python-dotenv python-multipart 2>nul
echo.
echo  Открывайте: http://localhost:8000
echo  Ctrl+C чтобы остановить
echo.
python server.py
pause
