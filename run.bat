@echo off
chcp 65001 >nul
title JarCut
echo.
echo   JarCut — автоматический дубляж
echo   =================================
echo.
cd /d "%~dp0"

if not exist .env (
    echo GEMINI_API_KEY=ваш_ключ_сюда> .env
    echo   Создан файл .env.
    echo   Вставьте в него Gemini API key и снова запустите run.bat.
    echo   https://aistudio.google.com/apikey
    echo.
    pause
    exit /b 1
)

python --version >nul 2>&1
if errorlevel 1 (
    echo   ОШИБКА: Python не установлен или не добавлен в PATH.
    echo   Установите Python 3.11+: https://python.org
    pause
    exit /b 1
)

echo   Проверяю зависимости...
python -m pip install -q fastapi uvicorn httpx python-dotenv python-multipart
if errorlevel 1 (
    echo   ОШИБКА: не удалось установить Python-зависимости.
    pause
    exit /b 1
)

echo   Сервер: http://localhost:8000
echo   Для остановки нажмите Ctrl+C.
echo.
start "" cmd /c "timeout /t 2 /nobreak >nul & start http://localhost:8000"
python server.py
pause
