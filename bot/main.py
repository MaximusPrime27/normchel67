
import io
import math
import os
import re
import secrets
from typing import Dict, Tuple

import pytesseract
from PIL import Image
from sympy import Eq, Symbol, solve, sympify, sqrt, pi, E
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)

from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    BufferedInputFile,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    Message,
    ReplyKeyboardMarkup,
    Update,
)
from fastapi import FastAPI, Header, HTTPException, Request
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
BASE_URL = os.getenv("BASE_URL", "").strip().rstrip("/")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "").strip()

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN is not set")
if not BASE_URL:
    raise RuntimeError("BASE_URL is not set")
if not WEBHOOK_SECRET:
    WEBHOOK_SECRET = secrets.token_urlsafe(32)

bot = Bot(BOT_TOKEN)
dp = Dispatcher()
router = Router()
dp.include_router(router)
app = FastAPI(title="MathSolver Telegram Bot FREE")

photo_cache: Dict[int, bytes] = {}
mode_cache: Dict[int, str] = {}

MAIN_KB = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="🧮 Математика"), KeyboardButton(text="🇷🇺 Русский язык")],
        [KeyboardButton(text="🎓 ОГЭ"), KeyboardButton(text="🏆 ЕГЭ")],
        [KeyboardButton(text="📷 Решить по фото"), KeyboardButton(text="ℹ️ Помощь")],
    ],
    resize_keyboard=True,
)

CROP_KB = InlineKeyboardMarkup(
    inline_keyboard=[
        [
            InlineKeyboardButton(text="⬆ Верх", callback_data="crop:top"),
            InlineKeyboardButton(text="🎯 Центр", callback_data="crop:center"),
            InlineKeyboardButton(text="⬇ Низ", callback_data="crop:bottom"),
        ],
        [
            InlineKeyboardButton(text="⬅ Левая", callback_data="crop:left"),
            InlineKeyboardButton(text="➡ Правая", callback_data="crop:right"),
        ],
        [InlineKeyboardButton(text="🖼 Всё фото", callback_data="crop:full")],
    ]
)

TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)
x = Symbol("x")

def set_mode(user_id: int, mode: str):
    mode_cache[user_id] = mode

def get_mode(user_id: int) -> str:
    return mode_cache.get(user_id, "auto")

def crop_image(data: bytes, preset: str) -> bytes:
    image = Image.open(io.BytesIO(data)).convert("RGB")
    w, h = image.size
    boxes = {
        "top": (0, 0, w, int(h * 0.58)),
        "center": (0, int(h * 0.18), w, int(h * 0.82)),
        "bottom": (0, int(h * 0.42), w, h),
        "left": (0, 0, int(w * 0.62), h),
        "right": (int(w * 0.38), 0, w, h),
        "full": (0, 0, w, h),
    }
    cropped = image.crop(boxes.get(preset, boxes["full"]))
    buf = io.BytesIO()
    cropped.save(buf, "JPEG", quality=92)
    return buf.getvalue()

def ocr_image(data: bytes) -> str:
    image = Image.open(io.BytesIO(data)).convert("RGB")
    image.thumbnail((2200, 2200))
    try:
        text = pytesseract.image_to_string(image, lang="rus+eng", config="--psm 6")
    except Exception:
        text = pytesseract.image_to_string(image, config="--psm 6")
    return text.strip()

def normalize_math(s: str) -> str:
    return (
        s.replace("×", "*")
         .replace("·", "*")
         .replace("÷", "/")
         .replace("−", "-")
         .replace(",", ".")
         .replace("√", "sqrt")
         .replace("π", "pi")
    )

def solve_math(text: str) -> str:
    raw = text.strip()
    s = normalize_math(raw)

    # quadratic form ax^2+bx+c=0 and general single equation
    if "=" in s:
        left, right = s.split("=", 1)
        try:
            lhs = parse_expr(left, transformations=TRANSFORMS, local_dict={"x": x, "sqrt": sqrt, "pi": pi, "e": E})
            rhs = parse_expr(right, transformations=TRANSFORMS, local_dict={"x": x, "sqrt": sqrt, "pi": pi, "e": E})
            roots = solve(Eq(lhs, rhs), x)
            if roots:
                lines = [
                    "🧮 Решение уравнения",
                    f"{raw}",
                    "",
                    f"Переносим всё в одну сторону:",
                    f"{lhs - rhs} = 0",
                    "",
                    "Ответ:",
                    ", ".join([f"x = {r}" for r in roots]),
                ]
                return "\n".join(lines)
            return "🧮 Уравнение разобрано, но действительных/символьных корней в простом виде не найдено."
        except Exception:
            pass

    # common geometry keywords
    low = raw.lower()
    nums = [float(v.replace(",", ".")) for v in re.findall(r"-?\d+(?:[.,]\d+)?", raw)]

    if "площад" in low and "круг" in low and nums:
        r = nums[0]
        return f"📐 Площадь круга\nS = πr²\nS = π·{r}² ≈ {math.pi*r*r:.4g}\n\nОтвет: {math.pi*r*r:.4g}"

    if ("гипотен" in low or "прямоугольн" in low) and len(nums) >= 2:
        a, b = nums[:2]
        c = math.hypot(a, b)
        return f"📐 Теорема Пифагора\nc = √(a²+b²)\nc = √({a}²+{b}²) = {c:.4g}\n\nОтвет: {c:.4g}"

    if "процент" in low and len(nums) >= 2:
        a, p = nums[:2]
        val = a * p / 100
        return f"📊 {p:g}% от {a:g}\n{a:g}·{p:g}/100 = {val:g}\n\nОтвет: {val:g}"

    # plain expression
    try:
        expr = parse_expr(
            s,
            transformations=TRANSFORMS,
            local_dict={"x": x, "sqrt": sqrt, "pi": pi, "e": E},
        )
        if not expr.free_symbols:
            val = expr.evalf()
            return f"🧮 Вычисление\n{raw}\n\nОтвет: {val}"
    except Exception:
        pass

    return (
        "Я не смог однозначно разобрать эту задачу локальным решателем.\n\n"
        "Попробуй:\n"
        "• уравнение: `2x + 5 = 17`\n"
        "• выражение: `(2+3)^2/5 + sqrt(16)`\n"
        "• написать условие короче и указать, что нужно найти.\n\n"
        "Сложные текстовые задачи без ИИ эта бесплатная версия решает ограниченно."
    )

STRESS = {
    "звонит": "звонИт",
    "позвонит": "позвонИт",
    "баловать": "баловАть",
    "красивее": "красИвее",
    "торты": "тОрты",
    "договор": "договОр",
    "договоры": "договОры",
    "каталог": "каталОг",
    "километр": "киломЕтр",
    "жалюзи": "жалюзИ",
    "цемент": "цемЕнт",
    "облегчить": "облегчИть",
    "включит": "включИт",
    "начала": "началА",
    "начала́": "началА",
    "приняла": "принялА",
    "заняла": "занялА",
    "создала": "создалА",
    "клала": "клАла",
}

INTRO = ["конечно", "вероятно", "возможно", "кажется", "например", "следовательно", "во-первых", "во-вторых"]
SUBORD = ["когда", "если", "потому что", "чтобы", "хотя", "который", "поскольку", "так как"]

def solve_russian(text: str) -> str:
    t = text.strip()
    low = t.lower()
    result = []

    # stress: exact single word
    if re.fullmatch(r"[а-яёА-ЯЁ-]+", t):
        if low in STRESS:
            return f"🇷🇺 Ударение\nПравильно: {STRESS[low]}"

    # -тся/-ться
    if "тся" in low or "ться" in low:
        result.append(
            "• -ТСЯ / -ТЬСЯ: задай вопрос к глаголу. "
            "«что делает/сделает?» → -тся; «что делать/сделать?» → -ться."
        )

    # не
    if re.search(r"\bне\s+[а-яё]+", low):
        result.append(
            "• НЕ: проверь часть речи, наличие противопоставления с «а» "
            "и возможность подобрать синоним без НЕ."
        )

    # пре-/при-
    if re.search(r"\b(пре|при)[а-яё]+", low):
        result.append(
            "• ПРЕ-/ПРИ-: ПРИ- обычно означает приближение, присоединение, близость "
            "или неполноту действия; ПРЕ- — «очень» или значение, близкое к ПЕРЕ-."
        )

    # нн
    if "нн" in low:
        result.append(
            "• Н/НН: определи часть речи и способ образования слова; "
            "для причастий проверь приставку, зависимые слова и вид глагола."
        )

    for word in INTRO:
        if re.search(rf"\b{re.escape(word)}\b", low):
            result.append(
                f"• «{word}» может быть вводным словом и тогда обычно обособляется запятыми."
            )

    for word in SUBORD:
        if word in low:
            result.append(
                f"• Найдено «{word}»: возможно, это часть СПП. "
                "Найди грамматические основы и проверь запятую между частями."
            )
            break

    if re.search(r"\b(а|но)\b", low):
        result.append(
            "• Союз «а/но»: перед ним часто ставится запятая между однородными членами "
            "или частями сложного предложения."
        )

    if not result:
        result.append(
            "Явных типичных конструкций не найдено.\n"
            "Для проверки: определи часть речи, найди грамматические основы, "
            "затем проверь орфограммы и границы частей предложения."
        )

    return "🇷🇺 Разбор\n\n" + "\n\n".join(result)

def exam_hint(text: str, level: str) -> str:
    subject = "русский язык" if re.search(r"[а-яё]{4,}", text.lower()) and any(k in text.lower() for k in ["запят", "слово", "удар", "орф", "предлож"]) else "математика"
    if subject == "русский язык":
        base = solve_russian(text)
    else:
        base = solve_math(text)
    return f"🎓 Режим {level}\nПредмет: {subject}\n\n{base}\n\nТипичная ошибка: спешить и не проверять условие/правило перед ответом."

def solve_by_mode(user_id: int, text: str) -> str:
    mode = get_mode(user_id)
    if mode == "math":
        return solve_math(text)
    if mode == "russian":
        return solve_russian(text)
    if mode == "oge":
        return exam_hint(text, "ОГЭ")
    if mode == "ege":
        return exam_hint(text, "ЕГЭ")

    low = text.lower()
    russian_markers = ["запят", "орф", "удар", "слово", "предлож", "нн", "тся", "ться", "пунктуац"]
    if any(k in low for k in russian_markers):
        return solve_russian(text)
    return solve_math(text)

@router.message(CommandStart())
async def start(message: Message):
    await message.answer(
        "📚 MathSolver FREE\n\n"
        "Полностью бесплатная версия без платного ИИ.\n"
        "Есть математика, русский, ОГЭ/ЕГЭ и распознавание фото.\n\n"
        "Выбери режим или отправь задание.",
        reply_markup=MAIN_KB,
    )

@router.message(Command("help"))
@router.message(F.text == "ℹ️ Помощь")
async def help_cmd(message: Message):
    await message.answer(
        "ℹ️ Возможности бесплатной версии:\n\n"
        "🧮 выражения и многие уравнения\n"
        "📐 базовая геометрия\n"
        "📊 проценты\n"
        "🇷🇺 основные правила русского\n"
        "🎓 режимы ОГЭ/ЕГЭ\n"
        "📷 OCR текста с фотографии\n\n"
        "Сложные произвольные задачи без ИИ могут потребовать более точной записи условия."
    )

@router.message(F.text == "🧮 Математика")
async def math_mode(message: Message):
    set_mode(message.from_user.id, "math")
    await message.answer("🧮 Режим математики включён. Отправь пример, уравнение или условие.")

@router.message(F.text == "🇷🇺 Русский язык")
async def rus_mode(message: Message):
    set_mode(message.from_user.id, "russian")
    await message.answer("🇷🇺 Режим русского языка включён. Отправь слово, предложение или задание.")

@router.message(F.text == "🎓 ОГЭ")
async def oge_mode(message: Message):
    set_mode(message.from_user.id, "oge")
    await message.answer("🎓 Режим ОГЭ включён.")

@router.message(F.text == "🏆 ЕГЭ")
async def ege_mode(message: Message):
    set_mode(message.from_user.id, "ege")
    await message.answer("🏆 Режим ЕГЭ включён.")

@router.message(F.text == "📷 Решить по фото")
async def photo_hint(message: Message):
    await message.answer("📷 Отправь фотографию. Я предложу выбрать область и распознаю текст.")

@router.message(F.photo)
async def photo_handler(message: Message):
    photo = message.photo[-1]
    tg_file = await bot.get_file(photo.file_id)
    buf = io.BytesIO()
    await bot.download_file(tg_file.file_path, destination=buf)
    photo_cache[message.from_user.id] = buf.getvalue()

    await message.answer_photo(
        BufferedInputFile(photo_cache[message.from_user.id], filename="task.jpg"),
        caption="✂️ Выбери часть фото для распознавания:",
        reply_markup=CROP_KB,
    )

@router.callback_query(F.data.startswith("crop:"))
async def crop_callback(callback: CallbackQuery):
    data = photo_cache.get(callback.from_user.id)
    if not data:
        await callback.answer("Фото не найдено. Отправь его снова.", show_alert=True)
        return

    preset = callback.data.split(":", 1)[1]
    cropped = crop_image(data, preset)
    photo_cache[callback.from_user.id] = cropped
    await callback.answer("Распознаю…")

    text = ocr_image(cropped)
    await callback.message.answer_photo(
        BufferedInputFile(cropped, filename="crop.jpg"),
        caption="✅ Выбранная область",
    )

    if not text:
        await callback.message.answer(
            "Не удалось распознать текст. Попробуй более чёткое фото или другую область."
        )
        return

    await callback.message.answer(f"🔎 Распознано:\n\n{text[:3500]}")
    answer = solve_by_mode(callback.from_user.id, text)
    await callback.message.answer(answer[:4000])

@router.message(F.text)
async def text_handler(message: Message):
    text = message.text.strip()
    if text.startswith("/"):
        return
    answer = solve_by_mode(message.from_user.id, text)
    await message.answer(answer[:4000])

@app.get("/")
async def health():
    return {"ok": True, "service": "MathSolver Telegram Bot FREE"}

@app.get("/health")
async def health2():
    return {"ok": True}

@app.on_event("startup")
async def startup():
    webhook_url = f"{BASE_URL}/telegram/{WEBHOOK_SECRET}"
    await bot.set_webhook(
        webhook_url,
        secret_token=WEBHOOK_SECRET,
        drop_pending_updates=False,
    )

@app.on_event("shutdown")
async def shutdown():
    await bot.session.close()

@app.post("/telegram/{secret}")
async def telegram_webhook(
    secret: str,
    request: Request,
    x_telegram_bot_api_secret_token: str | None = Header(default=None),
):
    if secret != WEBHOOK_SECRET:
        raise HTTPException(status_code=403, detail="bad secret")
    if x_telegram_bot_api_secret_token != WEBHOOK_SECRET:
        raise HTTPException(status_code=403, detail="bad telegram secret")

    data = await request.json()
    update = Update.model_validate(data, context={"bot": bot})
    await dp.feed_update(bot, update)
    return {"ok": True}
