# MathSolver V7 Admin Pro

Версия V7 добавляет всё, кроме режима учителя и групповых заданий.

## Главное

- PostgreSQL-ready база;
- SQLite остаётся fallback по умолчанию;
- Groq primary AI;
- резервный OpenAI-compatible AI;
- мини-уроки;
- тренировки;
- проверка ответов;
- проверка решения по фото;
- ОГЭ/ЕГЭ варианты с баллами;
- слабые темы ученика;
- профиль и статистика;
- история;
- избранное;
- кэш;
- журнал ошибок;
- веб-админка.

## Веб-админка

Добавь в Render:

```text
ADMIN_WEB_USER=admin
ADMIN_WEB_PASSWORD=придумай_длинный_пароль
```

После Deploy открой:

```text
https://ТВОЙ-СЕРВИС.onrender.com/admin
```

Браузер спросит логин и пароль.

Разделы:
- Обзор
- Пользователи
- Ошибки
- Темы
- Экзамены

Никому не передавай ADMIN_WEB_PASSWORD.

## PostgreSQL

Код уже поддерживает PostgreSQL.

Если есть внешний PostgreSQL URL, просто замени:

```text
DATABASE_URL=sqlite:///data/mathsolver.db
```

на:

```text
DATABASE_URL=postgresql+psycopg://USER:PASSWORD@HOST:PORT/DBNAME
```

Код менять не нужно.

## Резервный AI

Можно оставить пустым.

При необходимости:

```text
FALLBACK_API_BASE=https://provider.example/v1
FALLBACK_API_KEY=...
FALLBACK_MODEL=...
```

Если Groq временно падает, бот попробует резервный сервис.

## Проверка после Deploy

В Telegram:

```text
/status
/version
```

Должно быть:

```text
Версия: V7 Admin Pro
```

В браузере:

```text
/health
/admin
```
