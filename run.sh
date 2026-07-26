#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# CapCut-Lite — Запуск сервера (Linux / macOS)
# Просто дважды кликните или выполните: ./run.sh
# ═══════════════════════════════════════════════════════════════

echo "╔══════════════════════════════════════════╗"
echo "║       CapCut-Lite — Видеоредактор        ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# Переходим в папку скрипта (чтобы работало откуда угодно)
cd "$(dirname "$0")"

# Проверяем .env
if [ ! -f .env ]; then
    echo "⚠️  Файл .env не найден!"
    echo "   Создаю из шаблона..."
    cp .env.example .env
    echo ""
    echo "❗ Откройте файл .env и вставьте ваш GEMINI_API_KEY"
    echo "   Получить ключ: https://aistudio.google.com/apikey"
    echo ""
    read -p "Нажмите Enter когда вставите ключ..."
fi

# Проверяем Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 не найден! Установите Python: https://python.org"
    read -p "Нажмите Enter для выхода..."
    exit 1
fi

# Устанавливаем зависимости (если ещё не установлены)
echo "📦 Проверяю зависимости..."
pip3 install -q -r requirements.txt 2>/dev/null || pip install -q -r requirements.txt 2>/dev/null

echo ""
echo "🚀 Запускаю сервер..."
echo "   Откройте в браузере: http://localhost:8000"
echo "   Для остановки нажмите Ctrl+C"
echo ""

# Запуск сервера
python3 server.py || python server.py
