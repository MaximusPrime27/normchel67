# MathSolver V8.1 Free Check

Добавлена оплата Telegram Stars.

## Тарифы

- 30 дней — 100 ⭐
- 90 дней — 250 ⭐
- 180 дней — 450 ⭐
- 360 дней — 900 ⭐
- Навсегда — 1500 ⭐

Это разовые покупки. Автосписания нет.

## Что открывает Premium

- полные тренировочные варианты ОГЭ/ЕГЭ;
- статистика слабых тем.

Проверка решения ученика по фото теперь бесплатная.

Обычное решение задач, русский язык, обучение и тренировки остаются доступными бесплатно.

## Telegram команды

/premium — тарифы и статус Premium  
/paysupport — поддержка по оплате  
/status — статус бота и Premium

## Новые переменные Render

PAY_SUPPORT_CONTACT=@ваш_username
PREMIUM_GATES_ENABLED=true

`PAY_SUPPORT_CONTACT` лучше обязательно заполнить настоящим контактом поддержки.

## Платежи

Бот:
1. отправляет invoice в валюте XTR;
2. проверяет pre_checkout_query;
3. ждёт successful_payment;
4. сохраняет telegram_payment_charge_id;
5. только после successful_payment выдаёт Premium.

Один и тот же charge ID повторно не активирует подписку.

## Админка

В `/admin` добавлены:
- количество платежей;
- сумма полученных Stars;
- раздел «Платежи».

Админ-команда возврата:

/refund USER_ID TELEGRAM_PAYMENT_CHARGE_ID

Она вызывает Telegram refundStarPayment.

## После Deploy

/status

Должно показать:

Версия: V8.1 Free Check

Затем нажми:

💎 Premium

Для реального теста выбери самый дешёвый тариф 30 дней — 100 ⭐.
