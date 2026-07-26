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
        echo VERTEX_TRANSCRIBE_TIMEOUT_SECONDS=240
        echo ALIGNMENT_MODE=hybrid
        echo WHISPER_MODEL=small
        echo SPEAKER_VERIFICATION=auto
        echo SPEAKER_EMBEDDING_MODEL=speechbrain/spkrec-ecapa-voxceleb
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
python -m pip install -q fastapi uvicorn httpx python-dotenv python-multipart faster-whisper
if errorlevel 1 (
    echo   ОШИБКА: не удалось установить Python-зависимости.
    pause
    exit /b 1
)

findstr /I /R /C:"^[ ]*SPEAKER_VERIFICATION[ ]*=[ ]*off[ ]*$" .env >nul
if errorlevel 1 (
    python -c "import speechbrain, torch" >nul 2>&1
    if errorlevel 1 (
        if exist .speechbrain-install-failed (
            echo   SpeechBrain ранее не установился; использую Whisper fallback.
            echo   Для новой попытки удалите .speechbrain-install-failed.
        ) else (
            echo   Устанавливаю необязательную проверку голосов SpeechBrain ECAPA...
            python -m pip install -q speechbrain
            if errorlevel 1 (
                type nul > .speechbrain-install-failed
                echo   ПРЕДУПРЕЖДЕНИЕ: SpeechBrain не установился. JarCut продолжит с Whisper fallback.
                echo   Для Python 3.14 может потребоваться дождаться совместимой версии SpeechBrain.
            ) else (
                python -c "import speechbrain, torch" >nul 2>&1
                if errorlevel 1 (
                    type nul > .speechbrain-install-failed
                    echo   ПРЕДУПРЕЖДЕНИЕ: SpeechBrain установился, но не загружается. Использую Whisper fallback.
                    echo   Для Python 3.14 может потребоваться дождаться совместимой версии SpeechBrain.
                ) else (
                    if exist .speechbrain-install-failed del /q .speechbrain-install-failed
                )
            )
        )
    ) else (
        if exist .speechbrain-install-failed del /q .speechbrain-install-failed
    )
) else (
    echo   Проверка голосов SpeechBrain выключена в .env.
)

echo   Сервер: http://localhost:8000
echo   Для остановки нажмите Ctrl+C.
echo.
start "" cmd /c "timeout /t 2 /nobreak >nul & start http://localhost:8000"
python server.py
pause
