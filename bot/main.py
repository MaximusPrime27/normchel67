import asyncio
import base64
import io
import logging
import os
import sqlite3
import time
from typing import Optional

from aiogram import Bot, Dispatcher, F, Router
from aiogram.enums import ChatAction
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    BufferedInputFile,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    Message,
    ReplyKeyboardMarkup,
)
from dotenv import load_dotenv
from openai import AsyncOpenAI
from PIL import Image

load_dotenv()

BOT_TOKEN = os.environ.get("BOT_TOKEN", "").strip()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.5").strip()

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN is not set")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")

logging.basicConfig(level=logging.INFO)

bot = Bot(BOT_TOKEN)
dp = Dispatcher()
router = Router()
dp.include_router(router)

client = AsyncOpenAI(api_key=OPENAI_API_KEY)

DB_PATH = "data/bot.db"
os.makedirs("data", exist_ok=True)

db = sqlite3.connect(DB_PATH, check_same_thread=False)
db.execute("""
CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY,
    mode TEXT NOT NULL DEFAULT 'auto',
    previous_response_id TEXT
)
""")
db.commit()

last_request = {}
photo_cache = {}

MAIN_KB = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="🧮 Математика"), KeyboardButton(text="🇷🇺 Русский язык")],
        [KeyboardButton(text="🎓 ОГЭ"), KeyboardButton(text="🏆 ЕГЭ")],
        [KeyboardButton(text="📷 Решить по фото"), KeyboardButton(text="🧠 Авто-режим")],
        [KeyboardButton(text="🗑 Очистить память"), KeyboardButton(text="ℹ️ Помощь")],
    ],
    resize_keyboard=True,
)

CROP_KB = InlineKeyboardMarkup(
    inline_keyboard=[
        [
            InlineKeyboardButton(text="Верх", callback_data="crop:top"),
            InlineKeyboardButton(text="Центр", callback_data="crop:center"),
            InlineKeyboardButton(text="Низ", callback_data="crop:bottom"),
        ],
        [
            InlineKeyboardButton(text="Левая часть", callback_data="crop:left"),
            InlineKeyboardButton(text="Правая часть", callback_data="crop:right"),
        ],
        [InlineKeyboardButton(text="Всё фото", callback_data="crop:full")],
    ]
)

MODE_PROMPTS = {
    "auto": """Ты учебный помощник для школьника 9–11 классов РФ.
Сам определи, относится запрос к математике, русскому языку, ОГЭ или ЕГЭ.
Решай по шагам, проверяй ответ и объясняй понятным школьным языком.
Не пропускай существенные преобразования. Если на фото несколько заданий, раздели ответ по номерам.""",

    "math": """Ты репетитор по математике 9–11 классов РФ.
Решай алгебру, геометрию, тригонометрию, функции, производные, вероятность,
текстовые задачи, ОГЭ и профильный/базовый ЕГЭ.
Показывай дано/ход решения/ответ. Проверяй вычисления. Для уравнений и неравенств
указывай ОДЗ, когда она нужна. Не усложняй объяснение без необходимости.""",

    "russian": """Ты репетитор по русскому языку 9–11 классов РФ.
Помогай с орфографией, пунктуацией, морфологией, синтаксисом, лексикой,
ударениями, ОГЭ и ЕГЭ. Объясняй правило и почему именно оно применяется.
Если пользователь просит исправить текст, сначала дай исправленный вариант,
затем кратко перечисли ошибки и правила.""",

    "oge": """Ты репетитор по ОГЭ для 9 класса РФ по математике и русскому языку.
Определи предмет по заданию. Дай решение в формате, удобном для подготовки к ОГЭ:
краткий алгоритм, решение, ответ и типичная ошибка.""",

    "ege": """Ты репетитор по ЕГЭ для 10–11 классов РФ по математике и русскому языку.
Определи предмет и уровень. Для математики показывай строгий ход решения;
для русского объясняй правило и логику выбора ответа.
В конце обязательно дай итоговый ответ в экзаменационном формате."""
}

def get_user(user_id: int):
    row = db.execute(
        "SELECT mode, previous_response_id FROM users WHERE user_id = ?",
        (user_id,),
    ).fetchone()
    if not row:
        db.execute("INSERT INTO users(user_id) VALUES(?)", (user_id,))
        db.commit()
        return "auto", None
    return row[0], row[1]

def set_mode(user_id: int, mode: str):
    db.execute(
        """INSERT INTO users(user_id, mode)
           VALUES(?, ?)
           ON CONFLICT(user_id) DO UPDATE SET mode=excluded.mode""",
        (user_id, mode),
    )
    db.commit()

def set_previous(user_id: int, response_id: Optional[str]):
    db.execute(
        """INSERT INTO users(user_id, previous_response_id)
           VALUES(?, ?)
           ON CONFLICT(user_id) DO UPDATE SET previous_response_id=excluded.previous_response_id""",
        (user_id, response_id),
    )
    db.commit()

def rate_limited(user_id: int) -> bool:
    now = time.monotonic()
    last = last_request.get(user_id, 0)
    if now - last < 1.5:
        return True
    last_request[user_id] = now
    return False

def crop_image(data: bytes, preset: str) -> bytes:
    im = Image.open(io.BytesIO(data)).convert("RGB")
    w, h = im.size

    if preset == "top":
        box = (0, 0, w, int(h * 0.55))
    elif preset == "center":
        box = (0, int(h * 0.2), w, int(h * 0.8))
    elif preset == "bottom":
        box = (0, int(h * 0.45), w, h)
    elif preset == "left":
        box = (0, 0, int(w * 0.6), h)
    elif preset == "right":
        box = (int(w * 0.4), 0, w, h)
    else:
        box = (0, 0, w, h)

    out = im.crop(box)
    buf = io.BytesIO()
    out.save(buf, format="JPEG", quality=92)
    return buf.getvalue()

async def ask_ai(
    user_id: int,
    text: str,
    image_bytes: Optional[bytes] = None,
    continue_context: bool = True,
) -> str:
    mode, previous_response_id = get_user(user_id)

    content = [{"type": "input_text", "text": text or "Реши задание на изображении."}]

    if image_bytes is not None:
        b64 = base64.b64encode(image_bytes).decode("ascii")
        content.append({
            "type": "input_image",
            "image_url": f"data:image/jpeg;base64,{b64}",
            "detail": "high",
        })

    kwargs = {
        "model": OPENAI_MODEL,
        "instructions": MODE_PROMPTS.get(mode, MODE_PROMPTS["auto"]),
        "input": [{"role": "user", "content": content}],
        "max_output_tokens": 2200,
    }

    if continue_context and previous_response_id:
        kwargs["previous_response_id"] = previous_response_id

    response = await client.responses.create(**kwargs)
    set_previous(user_id, response.id)
    return response.output_text.strip() or "Не удалось получить текстовый ответ."

async def send_long(message: Message, text: str):
    limit = 3900
    while text:
        chunk = text[:limit]
        if len(text) > limit:
            split = chunk.rfind("\n")
            if split > 1000:
                chunk = chunk[:split]
        await message.answer(chunk)
        text = text[len(chunk):].lstrip()

@router.message(CommandStart())
async def start(message: Message):
    get_user(message.from_user.id)
    await message.answer(
        "Привет! Я MathSolver 📚\n\n"
        "Умею решать математику и русский язык 9–11 классов, "
        "помогать с ОГЭ/ЕГЭ и разбирать задания по фото.\n\n"
        "Выбери режим или просто отправь задачу.",
        reply_markup=MAIN_KB,
    )

@router.message(Command("help"))
@router.message(F.text == "ℹ️ Помощь")
async def help_cmd(message: Message):
    await message.answer(
        "Как пользоваться:\n\n"
        "• отправь текст задачи — я решу её;\n"
        "• отправь фото — предложу выбрать область и разберу её;\n"
        "• после решения можно писать «почему?», «объясни шаг 2», «реши иначе»;\n"
        "• /clear — забыть текущую задачу;\n"
        "• /mode — показать текущий режим;\n\n"
        "Режимы: математика, русский, ОГЭ, ЕГЭ и авто."
    )

@router.message(Command("clear"))
@router.message(F.text == "🗑 Очистить память")
async def clear_cmd(message: Message):
    set_previous(message.from_user.id, None)
    await message.answer("Контекст очищен. Следующая задача начнёт новый диалог.")

@router.message(Command("mode"))
async def mode_cmd(message: Message):
    mode, _ = get_user(message.from_user.id)
    await message.answer(f"Текущий режим: {mode}")

@router.message(F.text == "🧮 Математика")
async def mode_math(message: Message):
    set_mode(message.from_user.id, "math")
    set_previous(message.from_user.id, None)
    await message.answer("Режим «Математика» включён. Отправь задачу или фото.")

@router.message(F.text == "🇷🇺 Русский язык")
async def mode_russian(message: Message):
    set_mode(message.from_user.id, "russian")
    set_previous(message.from_user.id, None)
    await message.answer("Режим «Русский язык» включён. Отправь задание или текст.")

@router.message(F.text == "🎓 ОГЭ")
async def mode_oge(message: Message):
    set_mode(message.from_user.id, "oge")
    set_previous(message.from_user.id, None)
    await message.answer("Режим ОГЭ включён. Отправь задание.")

@router.message(F.text == "🏆 ЕГЭ")
async def mode_ege(message: Message):
    set_mode(message.from_user.id, "ege")
    set_previous(message.from_user.id, None)
    await message.answer("Режим ЕГЭ включён. Отправь задание.")

@router.message(F.text == "🧠 Авто-режим")
async def mode_auto(message: Message):
    set_mode(message.from_user.id, "auto")
    set_previous(message.from_user.id, None)
    await message.answer("Авто-режим включён. Я сам определю предмет.")

@router.message(F.text == "📷 Решить по фото")
async def photo_hint(message: Message):
    await message.answer("Отправь фотографию задания. После этого выбери область для анализа.")

@router.message(F.photo)
async def photo_handler(message: Message):
    user_id = message.from_user.id
    photo = message.photo[-1]
    file = await bot.get_file(photo.file_id)
    buf = io.BytesIO()
    await bot.download_file(file.file_path, destination=buf)
    photo_cache[user_id] = buf.getvalue()

    await message.answer_photo(
        BufferedInputFile(photo_cache[user_id], filename="task.jpg"),
        caption="Выбери область изображения. Если всё задание помещается целиком — нажми «Всё фото».",
        reply_markup=CROP_KB,
    )

@router.callback_query(F.data.startswith("crop:"))
async def crop_callback(callback: CallbackQuery):
    user_id = callback.from_user.id
    data = photo_cache.get(user_id)
    if not data:
        await callback.answer("Фото уже не найдено. Отправь его снова.", show_alert=True)
        return

    preset = callback.data.split(":", 1)[1]
    cropped = crop_image(data, preset)
    photo_cache[user_id] = cropped

    await callback.answer("Область выбрана")
    await callback.message.answer_photo(
        BufferedInputFile(cropped, filename="crop.jpg"),
        caption="Анализирую выбранную область…",
    )

    try:
        await bot.send_chat_action(callback.message.chat.id, ChatAction.TYPING)
        answer = await ask_ai(
            user_id,
            "Распознай задание на фото и реши его. Если это русский язык — объясни правило. "
            "Если математика — покажи решение по шагам и итоговый ответ.",
            image_bytes=cropped,
            continue_context=False,
        )
        await send_long(callback.message, answer)
    except Exception as exc:
        logging.exception("AI image request failed")
        await callback.message.answer(
            "Не удалось обработать фото через ИИ. Проверь OPENAI_API_KEY и доступность модели.\n"
            f"Техническая ошибка: {type(exc).__name__}"
        )

@router.message(F.text)
async def text_handler(message: Message):
    user_id = message.from_user.id
    text = message.text.strip()

    if text.startswith("/"):
        return

    if rate_limited(user_id):
        await message.answer("Слишком быстро 🙂 Отправь следующий запрос через пару секунд.")
        return

    if len(text) > 12000:
        await message.answer("Сообщение слишком длинное. Сократи его или отправь задачу частями.")
        return

    try:
        await bot.send_chat_action(message.chat.id, ChatAction.TYPING)
        answer = await ask_ai(user_id, text)
        await send_long(message, answer)
    except Exception as exc:
        logging.exception("AI text request failed")
        await message.answer(
            "Не удалось получить ответ от ИИ. Проверь OPENAI_API_KEY, баланс API и название модели.\n"
            f"Техническая ошибка: {type(exc).__name__}"
        )

async def main():
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
