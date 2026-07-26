@echo off
chcp 65001 >nul
title JarCut
echo.
echo   JarCut — автоматический дубляж
echo   =================================
echo.
cd /d "%~dp0"

if not exist .env (
    (
        echo # Vertex AI Express Mode ^(для Standard Mode измените настройки ниже^)
        echo # Standard Mode: project ID + Vertex key bound to a service account
        echo VERTEX_API_KEY=ваш_vertex_ключ
        echo VERTEX_MODE=express
        echo VERTEX_PROJECT_ID=
        echo VERTEX_LOCATION=global
    )> .env
    echo   Создан файл .env для Vertex AI.
    echo   Вставьте Vertex API key в VERTEX_API_KEY и снова запустите run.bat.
    echo   https://cloud.google.com/vertex-ai/generative-ai/docs/start/api-keys
    echo.
    echo   Standard Mode: VERTEX_MODE=standard, VERTEX_PROJECT_ID и ключ, привязанный к service account.
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
