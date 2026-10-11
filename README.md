# MathSolver Telegram Bot FREE

Полностью бесплатная версия без OpenAI API.

Функции:
- математика 9–11;
- русский язык;
- ОГЭ/ЕГЭ;
- OCR фото на русском и английском;
- выбор области фото;
- webhook для бесплатного Render Web Service.

## Render

Создай **Web Service**, не Background Worker.

Runtime: Docker  
Plan: Free

Environment Variables:

```text
BOT_TOKEN=токен от BotFather
WEBHOOK_SECRET=любая длинная случайная строка
BASE_URL=https://ИМЯ-СЕРВИСА.onrender.com
```

После первого успешного Deploy Telegram webhook установится автоматически.

## Проверка

Открой:

```text
https://ИМЯ-СЕРВИСА.onrender.com/health
```

Должно показать:

```json
{"ok":true}
```

После этого отправь `/start` своему Telegram-боту.

## Ограничение

Это локальный решатель без большой языковой модели. Он хорошо работает с
выражениями, уравнениями, базовыми школьными формулами и встроенными правилами
русского, но сложные произвольные текстовые задачи понимает хуже, чем ИИ.
