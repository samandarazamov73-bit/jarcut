@echo off
chcp 65001 >nul
REM ═══════════════════════════════════════════════════════════════
REM CapCut-Lite — Запуск сервера (Windows)
REM Просто дважды кликните по этому файлу
REM ═══════════════════════════════════════════════════════════════

echo ╔══════════════════════════════════════════╗
echo ║       CapCut-Lite — Видеоредактор        ║
echo ╚══════════════════════════════════════════╝
echo.

REM Переходим в папку скрипта
cd /d "%~dp0"

REM Проверяем .env
if not exist .env (
    echo ⚠️  Файл .env не найден!
    echo    Создаю из шаблона...
    copy .env.example .env >nul
    echo.
    echo ❗ Откройте файл .env и вставьте ваш GEMINI_API_KEY
    echo    Получить ключ: https://aistudio.google.com/apikey
    echo.
    pause
)

REM Проверяем Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python не найден! Установите Python: https://python.org
    echo    При установке поставьте галочку "Add Python to PATH"
    pause
    exit /b 1
)

REM Устанавливаем зависимости
echo 📦 Проверяю зависимости...
pip install -q -r requirements.txt 2>nul

echo.
echo 🚀 Запускаю сервер...
echo    Откройте в браузере: http://localhost:8000
echo    Для остановки нажмите Ctrl+C или закройте это окно
echo.

REM Запуск сервера
python server.py

pause
