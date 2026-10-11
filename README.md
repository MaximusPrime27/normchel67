# MathSolver Telegram Bot

Telegram-бот для 9–11 классов:

- математика;
- русский язык;
- ОГЭ;
- ЕГЭ;
- решение по фото;
- выбор области фото;
- память предыдущего решения;
- OpenAI Responses API.

## 1. Создай Telegram-бота

В Telegram открой `@BotFather`.

Выполни:

```text
/newbot
```

Сохрани токен.

## 2. Получи OpenAI API key

Создай API key в своей OpenAI API учётной записи.

Не добавляй токены в исходный код и GitHub.

## 3. Переменные окружения

Нужны:

```text
BOT_TOKEN=...
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-5.5
```

Если выбранная модель недоступна твоему API-проекту, замени `OPENAI_MODEL` на модель,
которая доступна в твоём аккаунте и поддерживает изображения.

## 4. Запуск локально

```bash
pip install -r requirements.txt
cp .env.example .env
python -m bot.main
```

## 5. GitHub

Загрузи ВСЕ файлы этого проекта в корень нового репозитория.

Токены в GitHub не загружай.

## 6. Render

Создай Background Worker / Docker service из GitHub-репозитория.

Добавь Environment Variables:

- BOT_TOKEN
- OPENAI_API_KEY
- OPENAI_MODEL

Команда запуска уже задана в Dockerfile.

## Команды

- `/start`
- `/help`
- `/clear`
- `/mode`

После решения пользователь может писать:
- `почему?`
- `объясни второй шаг`
- `реши другим способом`

Бот продолжит предыдущий контекст.
