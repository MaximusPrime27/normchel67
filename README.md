# MathSolver 9–11 + Groq

Telegram-бот для математики и русского языка 9–11 классов.

## Возможности
- программа 9–11 классов, ОГЭ/ЕГЭ;
- SymPy для точных вычислений;
- OpenCV + Tesseract для улучшенного OCR;
- Groq AI для объяснений;
- qwen/qwen3.8-27b для прямого анализа фото;
- работает и без Groq API, но тогда сложные задачи ограничены.

## Render Environment
Обязательные:
BOT_TOKEN
WEBHOOK_SECRET
BASE_URL

Для ИИ:
GROQ_API_KEY
GROQ_TEXT_MODEL=openai/gpt-oss-120b
GROQ_VISION_MODEL=qwen/qwen3.8-27b

После добавления ключа сделай Deploy latest commit.

Проверка:
https://YOUR-SERVICE.onrender.com/health

С подключённым Groq должно быть:
{"ok":true,"ai_enabled":true}
