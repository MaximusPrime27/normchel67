
import asyncio
import base64
import hashlib
import io
import json
import logging
import math
import os
import re
import secrets
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import cv2
import httpx
import numpy as np
import pytesseract
from PIL import Image
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request
from groq import AsyncGroq, APIConnectionError, APIStatusError, RateLimitError
from sqlalchemy import (
    Boolean, Column, DateTime, Integer, String, Text, create_engine, delete,
    func, select
)
from sqlalchemy.orm import declarative_base, sessionmaker
from sympy import E, Eq, Symbol, diff, expand, factor, integrate, pi, simplify, solve, sqrt
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
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

load_dotenv()

APP_VERSION = "V6 Ultimate"

# -----------------------------
# CONFIG
# -----------------------------
BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
BASE_URL = os.getenv("BASE_URL", "").strip().rstrip("/")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "").strip()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_TEXT_MODEL = os.getenv("GROQ_TEXT_MODEL", "openai/gpt-oss-120b").strip()
GROQ_VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "qwen/qwen3.8-27b").strip()

# Optional OpenAI-compatible fallback provider:
# e.g. OpenRouter/Together/another compatible service.
FALLBACK_API_BASE = os.getenv("FALLBACK_API_BASE", "").strip().rstrip("/")
FALLBACK_API_KEY = os.getenv("FALLBACK_API_KEY", "").strip()
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "").strip()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///data/mathsolver.db").strip()
ADMIN_IDS = {
    int(v) for v in os.getenv("ADMIN_IDS", "").split(",")
    if v.strip().isdigit()
}

REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "55"))
MAX_CONCURRENT_AI = int(os.getenv("MAX_CONCURRENT_AI", "4"))
USER_COOLDOWN = float(os.getenv("USER_COOLDOWN_SECONDS", "1.2"))
MAX_HISTORY = int(os.getenv("MAX_HISTORY_MESSAGES", "8"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "2600"))
MAX_IMAGE_MB = float(os.getenv("MAX_IMAGE_MB", "7"))

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN is not set")
if not BASE_URL:
    raise RuntimeError("BASE_URL is not set")
if not WEBHOOK_SECRET:
    WEBHOOK_SECRET = secrets.token_urlsafe(32)

# -----------------------------
# LOGGING
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("mathsolver")

# -----------------------------
# DATABASE
# -----------------------------
if DATABASE_URL.startswith("sqlite:///"):
    db_file = DATABASE_URL.removeprefix("sqlite:///")
    os.makedirs(os.path.dirname(db_file) or ".", exist_ok=True)

engine_kwargs = {"pool_pre_ping": True}
if DATABASE_URL.startswith("sqlite"):
    engine_kwargs["connect_args"] = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, **engine_kwargs)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
Base = declarative_base()

class UserState(Base):
    __tablename__ = "user_state"
    user_id = Column(Integer, primary_key=True)
    grade = Column(Integer, nullable=False, default=11)
    mode = Column(String(32), nullable=False, default="auto")
    pending_type = Column(String(64), nullable=True)
    pending_payload = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

class History(Base):
    __tablename__ = "history"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, index=True, nullable=False)
    role = Column(String(16), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, server_default=func.now())

class Favorite(Base):
    __tablename__ = "favorites"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, index=True, nullable=False)
    task = Column(Text, nullable=False)
    answer = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now())

class Stat(Base):
    __tablename__ = "stats"
    user_id = Column(Integer, primary_key=True)
    solved = Column(Integer, nullable=False, default=0)
    photos = Column(Integer, nullable=False, default=0)
    correct_checks = Column(Integer, nullable=False, default=0)
    wrong_checks = Column(Integer, nullable=False, default=0)
    ai_errors = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, server_default=func.now())

class CacheEntry(Base):
    __tablename__ = "cache"
    key = Column(String(64), primary_key=True)
    value = Column(Text, nullable=False)
    created_at = Column(DateTime, server_default=func.now())

def init_db():
    Base.metadata.create_all(engine)

def ensure_user(uid: int):
    with SessionLocal() as s:
        state = s.get(UserState, uid)
        if not state:
            s.add(UserState(user_id=uid, grade=11, mode="auto"))
        stat = s.get(Stat, uid)
        if not stat:
            s.add(Stat(user_id=uid))
        s.commit()

def get_state(uid: int) -> UserState:
    ensure_user(uid)
    with SessionLocal() as s:
        return s.get(UserState, uid)

def update_state(uid: int, **kwargs):
    ensure_user(uid)
    with SessionLocal() as s:
        row = s.get(UserState, uid)
        for k, v in kwargs.items():
            setattr(row, k, v)
        s.commit()

def set_pending(uid: int, kind: Optional[str], payload: Optional[dict] = None):
    update_state(
        uid,
        pending_type=kind,
        pending_payload=json.dumps(payload, ensure_ascii=False) if payload else None,
    )

def get_pending(uid: int) -> Tuple[Optional[str], dict]:
    st = get_state(uid)
    payload = {}
    if st.pending_payload:
        try:
            payload = json.loads(st.pending_payload)
        except Exception:
            payload = {}
    return st.pending_type, payload

def add_history(uid: int, role: str, content: str):
    with SessionLocal() as s:
        s.add(History(user_id=uid, role=role, content=content[:12000]))
        s.commit()
        ids = s.execute(
            select(History.id)
            .where(History.user_id == uid)
            .order_by(History.id.desc())
            .offset(30)
        ).scalars().all()
        if ids:
            s.execute(delete(History).where(History.id.in_(ids)))
            s.commit()

def get_history(uid: int, limit: int = MAX_HISTORY) -> List[dict]:
    with SessionLocal() as s:
        rows = s.execute(
            select(History)
            .where(History.user_id == uid)
            .order_by(History.id.desc())
            .limit(limit)
        ).scalars().all()
    rows.reverse()
    return [{"role": r.role, "content": r.content} for r in rows]

def clear_history(uid: int):
    with SessionLocal() as s:
        s.execute(delete(History).where(History.user_id == uid))
        s.commit()

def add_favorite(uid: int, task: str, answer: str):
    with SessionLocal() as s:
        s.add(Favorite(user_id=uid, task=task[:8000], answer=(answer or "")[:12000]))
        s.commit()

def list_favorites(uid: int, limit: int = 10):
    with SessionLocal() as s:
        return s.execute(
            select(Favorite)
            .where(Favorite.user_id == uid)
            .order_by(Favorite.id.desc())
            .limit(limit)
        ).scalars().all()

def bump_stat(uid: int, field: str, amount: int = 1):
    ensure_user(uid)
    with SessionLocal() as s:
        row = s.get(Stat, uid)
        setattr(row, field, int(getattr(row, field) or 0) + amount)
        s.commit()

def get_stats(uid: int) -> Stat:
    ensure_user(uid)
    with SessionLocal() as s:
        return s.get(Stat, uid)

def cache_get(key: str) -> Optional[str]:
    with SessionLocal() as s:
        row = s.get(CacheEntry, key)
        return row.value if row else None

def cache_put(key: str, value: str):
    with SessionLocal() as s:
        row = s.get(CacheEntry, key)
        if row:
            row.value = value
        else:
            s.add(CacheEntry(key=key, value=value[:16000]))
        s.commit()

# -----------------------------
# TELEGRAM / AI
# -----------------------------
bot = Bot(BOT_TOKEN)
dp = Dispatcher()
router = Router()
dp.include_router(router)
app = FastAPI(title=f"MathSolver {APP_VERSION}")

groq = AsyncGroq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
ai_semaphore = asyncio.Semaphore(MAX_CONCURRENT_AI)
last_request_at: Dict[int, float] = defaultdict(float)
photo_cache: Dict[int, bytes] = {}
last_task_cache: Dict[int, str] = {}
last_answer_cache: Dict[int, str] = {}

x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

# -----------------------------
# UI
# -----------------------------
MAIN_KB = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="▶️ Старт")],
        [KeyboardButton(text="🧮 Математика"), KeyboardButton(text="🇷🇺 Русский язык")],
        [KeyboardButton(text="🎓 ОГЭ"), KeyboardButton(text="🏆 ЕГЭ")],
        [KeyboardButton(text="📷 Решить по фото"), KeyboardButton(text="✅ Проверить решение")],
        [KeyboardButton(text="📖 Учиться"), KeyboardButton(text="📝 Тренировка")],
        [KeyboardButton(text="🎒 Выбрать класс"), KeyboardButton(text="👤 Профиль")],
        [KeyboardButton(text="⭐ Избранное"), KeyboardButton(text="📚 История")],
        [KeyboardButton(text="🧹 Сбросить контекст"), KeyboardButton(text="ℹ️ Помощь")],
    ],
    resize_keyboard=True,
)

CLASS_KB = InlineKeyboardMarkup(
    inline_keyboard=[[
        InlineKeyboardButton(text="9 класс", callback_data="class:9"),
        InlineKeyboardButton(text="10 класс", callback_data="class:10"),
        InlineKeyboardButton(text="11 класс", callback_data="class:11"),
    ]]
)

SUBJECT_KB = InlineKeyboardMarkup(
    inline_keyboard=[[
        InlineKeyboardButton(text="🧮 Математика", callback_data="subject:math"),
        InlineKeyboardButton(text="🇷🇺 Русский", callback_data="subject:rus"),
    ]]
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

ANSWER_ACTIONS = InlineKeyboardMarkup(
    inline_keyboard=[
        [
            InlineKeyboardButton(text="🟢 Проще", callback_data="act:simpler"),
            InlineKeyboardButton(text="🎯 Только ответ", callback_data="act:short"),
        ],
        [
            InlineKeyboardButton(text="📘 Подробнее", callback_data="act:detail"),
            InlineKeyboardButton(text="🔁 Другим способом", callback_data="act:another"),
        ],
        [
            InlineKeyboardButton(text="📝 Похожее задание", callback_data="act:similar"),
            InlineKeyboardButton(text="✅ Проверить мой ответ", callback_data="act:check"),
        ],
        [
            InlineKeyboardButton(text="📚 Что повторить", callback_data="act:review"),
            InlineKeyboardButton(text="⭐ В избранное", callback_data="act:favorite"),
        ],
    ]
)

TOPICS = {
    "math": {
        9: ["Квадратные уравнения", "Неравенства", "Функции и графики", "Прогрессии", "Планиметрия", "Вероятность"],
        10: ["Тригонометрия", "Степени и корни", "Функции", "Векторы", "Стереометрия", "Вероятность"],
        11: ["Производная", "Интеграл", "Тригонометрия", "Параметры", "Стереометрия", "Экономические задачи"],
    },
    "rus": {
        9: ["Орфография", "Пунктуация", "СПП/ССП/БСП", "Н/НН", "НЕ/НИ", "ОГЭ: изложение и сочинение"],
        10: ["Орфография", "Морфология", "Синтаксис", "Пунктуация", "Лексика и паронимы", "Текст и стили речи"],
        11: ["Орфография ЕГЭ", "Пунктуация ЕГЭ", "Грамматические нормы", "Средства выразительности", "Работа с текстом", "Сочинение ЕГЭ"],
    },
}

def topics_keyboard(subject: str, grade: int, prefix: str) -> InlineKeyboardMarkup:
    rows = []
    topics = TOPICS[subject][grade]
    for i in range(0, len(topics), 2):
        row = []
        for j in range(i, min(i + 2, len(topics))):
            row.append(InlineKeyboardButton(
                text=topics[j],
                callback_data=f"{prefix}:{subject}:{j}"
            ))
        rows.append(row)
    return InlineKeyboardMarkup(inline_keyboard=rows)

# -----------------------------
# PROMPTS
# -----------------------------
CURRICULUM = """
Ты работаешь по школьной программе России 9–11 классов.

Математика:
алгебраические выражения; степени и корни; рациональные выражения;
линейные, квадратные, рациональные, иррациональные, показательные,
логарифмические и тригонометрические уравнения и неравенства; системы;
модули; параметры; функции и графики; прогрессии; тригонометрия;
производная; касательная; экстремумы; первообразная и интеграл;
планиметрия; стереометрия; координаты и векторы; комбинаторика;
вероятность; статистика; текстовые и экономические задачи; ОГЭ и ЕГЭ.

Русский:
орфоэпия; лексика; паронимы; фразеология; морфемика; словообразование;
орфография; корни; приставки; суффиксы; окончания; ПРЕ-/ПРИ-; Н/НН;
НЕ/НИ; -ТСЯ/-ТЬСЯ; слитное, раздельное и дефисное написание;
морфология; синтаксис; грамматическая основа; однородные и обособленные
члены; вводные слова; обращения; ССП, СПП, БСП; прямая речь; пунктуация;
текст; стили и типы речи; выразительные средства; сочинение; ОГЭ/ЕГЭ.
"""

SYSTEM_PROMPT = f"""
Ты MathSolver — сильный, но понятный учебный помощник.

{CURRICULUM}

ФОРМАТ:
- Только обычный читаемый текст Telegram.
- Не используй Markdown-заголовки, таблицы Markdown, кодовые блоки или LaTeX.
- Не используй \\frac, \\sqrt, \\cdot, \\pm, \\[, \\], $...$.
- Используй обычные символы: √, ², ³, ·, ±, ≤, ≥, ≠, π, Δ.
- Дроби: (a+b)/(c+d).
- Коротко и понятно.
- Для математики: Решение → необходимые шаги → Ответ.
- Для русского: Правило → Разбор → Ответ.
- Если есть проверенный локальный результат SymPy, считай его приоритетным.
- Не выдумывай данные.
- При неоднозначном фото честно укажи, что не читается.
- Для ОГЭ/ЕГЭ соблюдай школьный формат.
"""

# -----------------------------
# TEXT CLEANING
# -----------------------------
def split_text(text: str, limit: int = 3900) -> List[str]:
    if len(text) <= limit:
        return [text]
    out = []
    while text:
        if len(text) <= limit:
            out.append(text)
            break
        cut = text.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = limit
        out.append(text[:cut])
        text = text[cut:].lstrip()
    return out

def clean_text(text: str) -> str:
    if not text:
        return ""
    s = str(text).replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"```[a-zA-Z0-9_-]*\n?", "", s).replace("```", "")
    s = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", s)
    s = s.replace("**", "").replace("__", "").replace("~~", "")
    for token in (r"\[", r"\]", r"\(", r"\)"):
        s = s.replace(token, "")
    s = s.replace("$", "")
    mapping = {
        r"\cdot": "·", r"\times": "×", r"\div": "÷", r"\pm": "±",
        r"\leq": "≤", r"\geq": "≥", r"\neq": "≠", r"\approx": "≈",
        r"\pi": "π", r"\Delta": "Δ", r"\left": "", r"\right": "",
    }
    for a, b in mapping.items():
        s = s.replace(a, b)

    frac = re.compile(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
    for _ in range(8):
        new = frac.sub(lambda m: f"({m.group(1)})/({m.group(2)})", s)
        if new == s:
            break
        s = new

    sq = re.compile(r"\\sqrt\s*\{([^{}]+)\}")
    for _ in range(5):
        new = sq.sub(lambda m: f"√({m.group(1)})", s)
        if new == s:
            break
        s = new

    super_map = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
    sub_map = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    s = re.sub(r"\^\{([0-9]+)\}", lambda m: m.group(1).translate(super_map), s)
    s = re.sub(r"_\{([0-9,]+)\}", lambda m: "".join(
        ch.translate(sub_map) if ch.isdigit() else ch for ch in m.group(1)
    ), s)
    s = re.sub(r"_([0-9])", lambda m: m.group(1).translate(sub_map), s)
    s = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", s)
    s = s.replace("{", "(").replace("}", ")")
    s = re.sub(r"\\([A-Za-z]+)", r"\1", s)
    s = re.sub(r"(?m)^\s*[-*]\s+", "• ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip()

async def send_plain(message: Message, text: str, actions: bool = False):
    cleaned = clean_text(text) or "Не удалось сформировать ответ."
    parts = split_text(cleaned)
    for i, part in enumerate(parts):
        await message.answer(
            part,
            reply_markup=ANSWER_ACTIONS if actions and i == len(parts) - 1 else None
        )

# -----------------------------
# RATE LIMIT
# -----------------------------
def is_rate_limited(uid: int) -> bool:
    now = time.monotonic()
    if now - last_request_at[uid] < USER_COOLDOWN:
        return True
    last_request_at[uid] = now
    return False

# -----------------------------
# EXACT MATH
# -----------------------------
def normalize_math(s: str) -> str:
    return (
        s.replace("×", "*")
        .replace("·", "*")
        .replace("÷", "/")
        .replace("−", "-")
        .replace(",", ".")
        .replace("π", "pi")
    )

def pexpr(s: str):
    s = normalize_math(s)
    s = re.sub(r"√\s*\(([^)]+)\)", r"sqrt(\1)", s)
    s = re.sub(r"√\s*([0-9A-Za-z.]+)", r"sqrt(\1)", s)
    return parse_expr(
        s,
        transformations=TRANSFORMS,
        local_dict={"x": x, "y": y, "z": z, "sqrt": sqrt, "pi": pi, "e": E},
    )

def exact_math_hint(text: str) -> Optional[str]:
    raw = text.strip()
    s = normalize_math(raw)
    low = raw.lower()

    if ";" in s and s.count("=") >= 2:
        try:
            equations = []
            for part in s.split(";"):
                if "=" in part:
                    a, b = part.split("=", 1)
                    equations.append(Eq(pexpr(a), pexpr(b)))
            vars_used = sorted(
                set().union(*(e.free_symbols for e in equations)),
                key=lambda q: q.name,
            )
            ans = solve(equations, vars_used, dict=True)
            if ans:
                return "Проверенный SymPy-результат системы: " + "; ".join(map(str, ans))
        except Exception:
            pass

    m = re.search(r"(?:производн\w*|d/dx)\s*[:=]?\s*(.+)", low)
    if m:
        try:
            return f"Проверенный результат: производная = {diff(pexpr(m.group(1)), x)}"
        except Exception:
            pass

    m = re.search(r"(?:интеграл|первообразн\w*)\s*[:=]?\s*(.+)", low)
    if m:
        try:
            return f"Проверенный результат: первообразная = {integrate(pexpr(m.group(1)), x)} + C"
        except Exception:
            pass

    for phrase, fn, label in [
        ("разложи на множители", factor, "разложение"),
        ("раскрой скобки", expand, "раскрытие"),
        ("упрости", simplify, "упрощение"),
    ]:
        if phrase in low:
            try:
                expr = pexpr(low.split(phrase, 1)[1].strip(" :"))
                return f"Проверенный результат ({label}): {fn(expr)}"
            except Exception:
                pass

    if s.count("=") == 1:
        try:
            a, b = s.split("=", 1)
            lhs, rhs = pexpr(a), pexpr(b)
            vars_used = sorted((lhs - rhs).free_symbols, key=lambda q: q.name)
            if vars_used:
                v = vars_used[0]
                roots = solve(Eq(lhs, rhs), v)
                if roots:
                    return "Проверенные корни: " + ", ".join(f"{v} = {r}" for r in roots)
        except Exception:
            pass

    try:
        expr = pexpr(s)
        if not expr.free_symbols:
            val = simplify(expr)
            return f"Проверенное значение: {val}; приближённо {val.evalf(12)}"
    except Exception:
        pass

    return None

# -----------------------------
# OCR
# -----------------------------
def compress_image(data: bytes, max_side: int = 1800) -> bytes:
    im = Image.open(io.BytesIO(data)).convert("RGB")
    im.thumbnail((max_side, max_side))
    b = io.BytesIO()
    im.save(b, "JPEG", quality=90, optimize=True)
    return b.getvalue()

def crop_image(data: bytes, preset: str) -> bytes:
    im = Image.open(io.BytesIO(data)).convert("RGB")
    w, h = im.size
    boxes = {
        "top": (0, 0, w, int(h * 0.58)),
        "center": (0, int(h * 0.18), w, int(h * 0.82)),
        "bottom": (0, int(h * 0.42), w, h),
        "left": (0, 0, int(w * 0.62), h),
        "right": (int(w * 0.38), 0, w, h),
        "full": (0, 0, w, h),
    }
    out = im.crop(boxes.get(preset, boxes["full"]))
    b = io.BytesIO()
    out.save(b, "JPEG", quality=94)
    return b.getvalue()

def preprocess_variants(data: bytes):
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return []

    h, w = img.shape[:2]
    scale = max(1.0, min(3.0, 2000 / max(h, w)))
    if scale > 1.05:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    img = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    contrast = clahe.apply(img)
    adaptive = cv2.adaptiveThreshold(
        contrast, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 35, 11
    )
    _, otsu = cv2.threshold(
        contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return [contrast, adaptive, otsu]

def ocr_image(data: bytes) -> str:
    candidates = []
    for img in preprocess_variants(data):
        pil = Image.fromarray(img)
        for psm in (6, 11, 12):
            try:
                text = pytesseract.image_to_string(
                    pil, lang="rus+eng",
                    config=f"--oem 3 --psm {psm}"
                ).strip()
                if text:
                    score = sum(ch.isalnum() for ch in text)
                    candidates.append((score, text))
            except Exception:
                pass
    return max(candidates, key=lambda q: q[0])[1] if candidates else ""

# -----------------------------
# AI ROUTING
# -----------------------------
def mode_line(uid: int) -> str:
    st = get_state(uid)
    return f"Режим: {st.mode}. Класс: {st.grade}."

async def fallback_chat(messages: List[dict]) -> Optional[str]:
    if not (FALLBACK_API_BASE and FALLBACK_API_KEY and FALLBACK_MODEL):
        return None
    url = FALLBACK_API_BASE
    if not url.endswith("/chat/completions"):
        url += "/chat/completions"
    payload = {
        "model": FALLBACK_MODEL,
        "messages": messages,
        "temperature": 0.15,
        "max_tokens": MAX_OUTPUT_TOKENS,
    }
    headers = {"Authorization": f"Bearer {FALLBACK_API_KEY}"}
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]

async def ai_chat(messages: List[dict], model: Optional[str] = None) -> str:
    errors = []
    async with ai_semaphore:
        if groq:
            for delay in (0.0, 1.2, 2.8):
                if delay:
                    await asyncio.sleep(delay)
                try:
                    response = await asyncio.wait_for(
                        groq.chat.completions.create(
                            model=model or GROQ_TEXT_MODEL,
                            messages=messages,
                            temperature=0.15,
                            max_completion_tokens=MAX_OUTPUT_TOKENS,
                        ),
                        timeout=REQUEST_TIMEOUT,
                    )
                    text = response.choices[0].message.content or ""
                    if text.strip():
                        return clean_text(text)
                except (RateLimitError, APIConnectionError, asyncio.TimeoutError) as e:
                    errors.append(type(e).__name__)
                except APIStatusError as e:
                    errors.append(f"APIStatusError:{getattr(e, 'status_code', '?')}")
                    if getattr(e, "status_code", 500) < 500 and getattr(e, "status_code", 500) != 429:
                        break
                except Exception as e:
                    errors.append(type(e).__name__)
                    break

        try:
            fallback = await fallback_chat(messages)
            if fallback:
                return clean_text(fallback)
        except Exception as e:
            errors.append(f"fallback:{type(e).__name__}")

    raise RuntimeError("AI unavailable: " + ",".join(errors[-4:]))

def make_cache_key(uid: int, text: str, hint: Optional[str], context: List[dict]) -> str:
    st = get_state(uid)
    material = json.dumps({
        "grade": st.grade,
        "mode": st.mode,
        "text": text,
        "hint": hint,
        "context": context[-4:],
        "model": GROQ_TEXT_MODEL,
    }, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()

async def solve_text(uid: int, text: str, use_history: bool = True) -> str:
    hint = exact_math_hint(text)
    hist = get_history(uid, MAX_HISTORY) if use_history else []
    key = make_cache_key(uid, text, hint, hist)
    cached = cache_get(key)
    if cached:
        return cached

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": mode_line(uid)},
    ]
    messages.extend(hist)
    prompt = text
    if hint:
        prompt += "\n\nПРОВЕРЕННЫЙ ЛОКАЛЬНЫЙ РЕЗУЛЬТАТ:\n" + hint
    messages.append({"role": "user", "content": prompt})

    try:
        answer = await ai_chat(messages)
        add_history(uid, "user", text)
        add_history(uid, "assistant", answer)
        bump_stat(uid, "solved")
        cache_put(key, answer)
        return answer
    except Exception:
        bump_stat(uid, "ai_errors")
        log.exception("solve_text failed")
        if hint:
            return "Решение:\n" + clean_text(hint) + "\n\nОтвет: см. точный результат выше."
        return (
            "Сейчас ИИ временно недоступен, а локальный решатель не смог "
            "уверенно разобрать задачу. Попробуй ещё раз через несколько секунд."
        )

async def solve_image(uid: int, data: bytes, ocr: str, check_solution: bool = False) -> str:
    if not groq:
        if ocr:
            return await solve_text(uid, ocr, use_history=False)
        return "Не удалось распознать фото."

    compressed = compress_image(data)
    b64 = base64.b64encode(compressed).decode()
    special = (
        "Проверь именно решение ученика. Найди первый неверный шаг, объясни ошибку "
        "и покажи правильное продолжение. Не ограничивайся только финальным ответом."
        if check_solution else
        "Реши задание с изображения."
    )
    prompt = (
        SYSTEM_PROMPT + "\n\n" + mode_line(uid) + "\n\n" +
        special +
        "\nOCR-подсказка (может ошибаться):\n" + (ocr or "(пусто)")
    )
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{b64}"
            }},
        ],
    }]
    try:
        answer = await ai_chat(messages, model=GROQ_VISION_MODEL)
        bump_stat(uid, "photos")
        return answer
    except Exception:
        bump_stat(uid, "ai_errors")
        if ocr:
            return await solve_text(uid, ocr, use_history=False)
        return "Не удалось проанализировать изображение. Попробуй более чёткое фото."

# -----------------------------
# COMMON ACTIONS
# -----------------------------
async def answer_and_remember(message: Message, uid: int, task: str, answer: str):
    last_task_cache[uid] = task
    last_answer_cache[uid] = answer
    await send_plain(message, answer, actions=True)

async def generate_lesson(uid: int, subject: str, topic: str) -> str:
    grade = get_state(uid).grade
    prompt = (
        f"Составь короткий мини-урок для {grade} класса.\n"
        f"Предмет: {'математика' if subject == 'math' else 'русский язык'}.\n"
        f"Тема: {topic}.\n"
        "Формат: 1) простое правило/идея; 2) один разобранный пример; "
        "3) три частые ошибки; 4) одно маленькое задание для самопроверки. "
        "Не давай ответ к заданию сразу."
    )
    return await solve_text(uid, prompt, use_history=False)

async def generate_training(uid: int, subject: str, topic: str) -> Tuple[str, str]:
    grade = get_state(uid).grade
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"Создай одно тренировочное задание для {grade} класса. "
            f"Предмет: {subject}. Тема: {topic}. "
            "Ответ верни строго двумя блоками:\n"
            "ЗАДАНИЕ: ...\n"
            "ОТВЕТ: ...\n"
            "Без дополнительных комментариев."
        )},
    ]
    raw = await ai_chat(messages)
    m1 = re.search(r"ЗАДАНИЕ:\s*(.+?)(?=\nОТВЕТ:|$)", raw, re.S | re.I)
    m2 = re.search(r"ОТВЕТ:\s*(.+)$", raw, re.S | re.I)
    task = clean_text(m1.group(1).strip()) if m1 else clean_text(raw)
    answer = clean_text(m2.group(1).strip()) if m2 else ""
    return task, answer

# -----------------------------
# START / INFO
# -----------------------------
@router.message(CommandStart())
@router.message(F.text == "▶️ Старт")
async def start(message: Message):
    ensure_user(message.from_user.id)
    st = get_state(message.from_user.id)
    await message.answer(
        f"📚 MathSolver {APP_VERSION}\n\n"
        f"🎒 Класс: {st.grade}\n"
        f"🤖 ИИ: {'подключён' if groq else 'резервный/локальный режим'}\n\n"
        "Я могу решать задачи, проверять домашнюю работу, обучать по темам, "
        "давать тренировки и разбирать ошибки.",
        reply_markup=MAIN_KB,
    )

@router.message(Command("status"))
async def status_cmd(message: Message):
    uid = message.from_user.id
    st = get_state(uid)
    stat = get_stats(uid)
    await message.answer(
        f"🟢 Бот работает\n"
        f"Версия: {APP_VERSION}\n"
        f"Класс: {st.grade}\n"
        f"Режим: {st.mode}\n"
        f"Groq: {'да' if groq else 'нет'}\n"
        f"Резервный AI: {'да' if FALLBACK_API_KEY else 'нет'}\n"
        f"Решено: {stat.solved}\n"
        f"Фото: {stat.photos}"
    )

@router.message(Command("version"))
async def version_cmd(message: Message):
    await message.answer(APP_VERSION)

@router.message(Command("help"))
@router.message(F.text == "ℹ️ Помощь")
async def help_cmd(message: Message):
    await message.answer(
        "Что умеет бот:\n\n"
        "• решать математику 9–11 классов;\n"
        "• разбирать русский язык;\n"
        "• ОГЭ/ЕГЭ;\n"
        "• решать задания по фото;\n"
        "• проверять решение ученика по фото;\n"
        "• объяснять проще/подробнее/другим способом;\n"
        "• давать похожие задачи;\n"
        "• проверять твой ответ;\n"
        "• проводить мини-уроки и тренировки;\n"
        "• хранить профиль, историю и избранное.\n\n"
        "Команды: /status, /version, /reset, /profile"
    )

# -----------------------------
# MODES / CLASS
# -----------------------------
@router.message(F.text == "🧮 Математика")
async def math_mode(message: Message):
    update_state(message.from_user.id, mode="math")
    clear_history(message.from_user.id)
    await message.answer("🧮 Режим математики включён.")

@router.message(F.text == "🇷🇺 Русский язык")
async def rus_mode(message: Message):
    update_state(message.from_user.id, mode="russian")
    clear_history(message.from_user.id)
    await message.answer("🇷🇺 Режим русского языка включён.")

@router.message(F.text == "🎓 ОГЭ")
async def oge_mode(message: Message):
    update_state(message.from_user.id, mode="oge")
    clear_history(message.from_user.id)
    await message.answer("🎓 Режим ОГЭ включён.")

@router.message(F.text == "🏆 ЕГЭ")
async def ege_mode(message: Message):
    update_state(message.from_user.id, mode="ege")
    clear_history(message.from_user.id)
    await message.answer("🏆 Режим ЕГЭ включён.")

@router.message(F.text == "🎒 Выбрать класс")
async def class_menu(message: Message):
    await message.answer("Выбери класс:", reply_markup=CLASS_KB)

@router.callback_query(F.data.startswith("class:"))
async def class_callback(callback: CallbackQuery):
    grade = int(callback.data.split(":")[1])
    if grade not in (9, 10, 11):
        await callback.answer("Ошибка", show_alert=True)
        return
    update_state(callback.from_user.id, grade=grade)
    clear_history(callback.from_user.id)
    await callback.answer(f"{grade} класс выбран")
    await callback.message.answer(f"🎒 Выбран {grade} класс.")

# -----------------------------
# PROFILE / HISTORY / FAVORITES
# -----------------------------
@router.message(Command("profile"))
@router.message(F.text == "👤 Профиль")
async def profile(message: Message):
    uid = message.from_user.id
    st = get_state(uid)
    stat = get_stats(uid)
    fav_count = len(list_favorites(uid, 100))
    await message.answer(
        "👤 Профиль ученика\n\n"
        f"Класс: {st.grade}\n"
        f"Режим: {st.mode}\n"
        f"Решено задач: {stat.solved}\n"
        f"Фото разобрано: {stat.photos}\n"
        f"Верных проверок: {stat.correct_checks}\n"
        f"Ошибочных ответов: {stat.wrong_checks}\n"
        f"Избранных задач: {fav_count}"
    )

@router.message(F.text == "📚 История")
async def history_handler(message: Message):
    hist = get_history(message.from_user.id, 10)
    user_items = [h["content"] for h in hist if h["role"] == "user"]
    if not user_items:
        await message.answer("📚 История пока пустая.")
        return
    lines = ["📚 Последние задачи:"]
    for i, item in enumerate(user_items[-5:], 1):
        lines.append(f"{i}. {clean_text(item).replace(chr(10), ' ')[:150]}")
    await message.answer("\n".join(lines))

@router.message(F.text == "⭐ Избранное")
async def favorites_handler(message: Message):
    rows = list_favorites(message.from_user.id, 10)
    if not rows:
        await message.answer("⭐ Избранных задач пока нет.")
        return
    text = ["⭐ Последние избранные:"]
    for i, row in enumerate(rows, 1):
        text.append(f"{i}. {clean_text(row.task).replace(chr(10), ' ')[:150]}")
    await message.answer("\n".join(text))

@router.message(Command("reset"))
@router.message(F.text == "🧹 Сбросить контекст")
async def reset_handler(message: Message):
    clear_history(message.from_user.id)
    set_pending(message.from_user.id, None)
    photo_cache.pop(message.from_user.id, None)
    await message.answer("🧹 Контекст очищен.")

# -----------------------------
# STUDY / TRAINING
# -----------------------------
@router.message(F.text == "📖 Учиться")
async def study_menu(message: Message):
    set_pending(message.from_user.id, "choose_study_subject")
    await message.answer("Выбери предмет для мини-урока:", reply_markup=SUBJECT_KB)

@router.message(F.text == "📝 Тренировка")
async def training_menu(message: Message):
    set_pending(message.from_user.id, "choose_training_subject")
    await message.answer("Выбери предмет для тренировки:", reply_markup=SUBJECT_KB)

@router.callback_query(F.data.startswith("subject:"))
async def subject_callback(callback: CallbackQuery):
    uid = callback.from_user.id
    subject = callback.data.split(":")[1]
    pending, _ = get_pending(uid)
    grade = get_state(uid).grade

    if pending == "choose_study_subject":
        set_pending(uid, "choose_study_topic", {"subject": subject})
        await callback.message.answer(
            "Выбери тему:",
            reply_markup=topics_keyboard(subject, grade, "study")
        )
    elif pending == "choose_training_subject":
        set_pending(uid, "choose_training_topic", {"subject": subject})
        await callback.message.answer(
            "Выбери тему:",
            reply_markup=topics_keyboard(subject, grade, "train")
        )
    else:
        await callback.answer("Сначала выбери режим «Учиться» или «Тренировка».", show_alert=True)
        return
    await callback.answer()

@router.callback_query(F.data.startswith("study:"))
async def study_topic_callback(callback: CallbackQuery):
    uid = callback.from_user.id
    _, subject, idx = callback.data.split(":")
    topic = TOPICS[subject][get_state(uid).grade][int(idx)]
    await callback.answer("Готовлю мини-урок…")
    lesson = await generate_lesson(uid, subject, topic)
    set_pending(uid, None)
    await send_plain(callback.message, lesson)

@router.callback_query(F.data.startswith("train:"))
async def train_topic_callback(callback: CallbackQuery):
    uid = callback.from_user.id
    _, subject, idx = callback.data.split(":")
    topic = TOPICS[subject][get_state(uid).grade][int(idx)]
    await callback.answer("Готовлю задание…")
    try:
        task, answer = await generate_training(uid, subject, topic)
    except Exception:
        await callback.message.answer("Не удалось создать тренировку. Попробуй ещё раз.")
        return
    set_pending(uid, "training_answer", {
        "task": task,
        "answer": answer,
        "subject": subject,
        "topic": topic,
    })
    await callback.message.answer(
        f"📝 Тренировка: {topic}\n\n{task}\n\n"
        "Отправь свой ответ обычным сообщением."
    )

# -----------------------------
# PHOTO / HOMEWORK CHECK
# -----------------------------
@router.message(F.text == "📷 Решить по фото")
async def solve_photo_mode(message: Message):
    set_pending(message.from_user.id, "awaiting_photo", {"check_solution": False})
    await message.answer("📷 Отправь фото задания.")

@router.message(F.text == "✅ Проверить решение")
async def check_photo_mode(message: Message):
    set_pending(message.from_user.id, "awaiting_photo", {"check_solution": True})
    await message.answer(
        "✅ Отправь фото своего решения. Я постараюсь найти первый неверный шаг."
    )

@router.message(F.photo)
async def photo_handler(message: Message):
    uid = message.from_user.id
    if is_rate_limited(uid):
        await message.answer("⏳ Подожди секунду.")
        return

    file = await bot.get_file(message.photo[-1].file_id)
    b = io.BytesIO()
    await bot.download_file(file.file_path, destination=b)
    data = b.getvalue()
    if len(data) > MAX_IMAGE_MB * 1024 * 1024:
        data = compress_image(data, 1600)

    photo_cache[uid] = data
    await message.answer_photo(
        BufferedInputFile(data, filename="task.jpg"),
        caption="✂️ Выбери область:",
        reply_markup=CROP_KB,
    )

@router.callback_query(F.data.startswith("crop:"))
async def crop_callback(callback: CallbackQuery):
    uid = callback.from_user.id
    data = photo_cache.get(uid)
    if not data:
        await callback.answer("Фото не найдено.", show_alert=True)
        return

    preset = callback.data.split(":")[1]
    cropped = crop_image(data, preset)
    await callback.answer("Распознаю и анализирую…")
    ocr = await asyncio.to_thread(ocr_image, cropped)
    pending, payload = get_pending(uid)
    check_solution = bool(payload.get("check_solution")) if pending == "awaiting_photo" else False

    answer = await solve_image(uid, cropped, ocr, check_solution=check_solution)
    task_label = ocr or ("Проверка решения по фото" if check_solution else "Задание по фото")
    last_task_cache[uid] = task_label
    last_answer_cache[uid] = answer
    set_pending(uid, None)

    if ocr:
        await send_plain(callback.message, "🔎 Распознано:\n" + ocr[:2200])
    await send_plain(callback.message, answer, actions=True)

# -----------------------------
# ANSWER ACTIONS
# -----------------------------
@router.callback_query(F.data.startswith("act:"))
async def action_callback(callback: CallbackQuery):
    uid = callback.from_user.id
    action = callback.data.split(":")[1]
    task = last_task_cache.get(uid)
    answer = last_answer_cache.get(uid)

    if not task or not answer:
        await callback.answer("Последнее решение не найдено.", show_alert=True)
        return

    if action == "favorite":
        add_favorite(uid, task, answer)
        await callback.answer("Добавлено в избранное ⭐")
        return

    if action == "check":
        set_pending(uid, "check_user_answer", {"task": task, "solution": answer})
        await callback.answer()
        await callback.message.answer("✅ Отправь свой ответ или своё решение текстом.")
        return

    prompts = {
        "simpler": "Объясни это решение значительно проще и короче.",
        "short": "Дай только итоговый ответ без объяснений.",
        "detail": "Объясни решение подробнее, не пропуская важные шаги.",
        "another": "Реши эту же задачу другим корректным способом.",
        "similar": (
            "Создай одно похожее задание того же уровня. "
            "Не решай его и не показывай ответ."
        ),
        "review": (
            "Скажи, какие 2–4 темы или правила нужно повторить ученику, "
            "чтобы уверенно решать такие задания. Коротко."
        ),
    }

    await callback.answer("Готовлю…")
    request = (
        prompts[action] +
        "\n\nИсходная задача:\n" + task +
        "\n\nПредыдущее решение:\n" + answer
    )
    result = await solve_text(uid, request, use_history=False)
    if action != "similar":
        last_answer_cache[uid] = result
    await send_plain(callback.message, result, actions=(action != "similar"))

# -----------------------------
# TEXT HANDLER / PENDING FLOWS
# -----------------------------
async def check_student_answer(uid: int, payload: dict, user_text: str) -> str:
    task = payload.get("task", "")
    solution = payload.get("solution", payload.get("answer", ""))
    prompt = (
        "Проверь ответ ученика. Сначала скажи «Верно» или «Есть ошибка». "
        "Если ошибка есть, кратко объясни где. Не решай всё заново без необходимости.\n\n"
        f"Задача:\n{task}\n\nЭталон/предыдущее решение:\n{solution}\n\n"
        f"Ответ ученика:\n{user_text}"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    return await ai_chat(messages)

@router.message(F.text)
async def text_handler(message: Message):
    uid = message.from_user.id
    text = message.text.strip()

    if text.startswith("/"):
        return
    if text in {
        "▶️ Старт", "🧮 Математика", "🇷🇺 Русский язык", "🎓 ОГЭ", "🏆 ЕГЭ",
        "📷 Решить по фото", "✅ Проверить решение", "📖 Учиться", "📝 Тренировка",
        "🎒 Выбрать класс", "👤 Профиль", "⭐ Избранное", "📚 История",
        "🧹 Сбросить контекст", "ℹ️ Помощь"
    }:
        return
    if is_rate_limited(uid):
        await message.answer("⏳ Подожди секунду.")
        return

    pending, payload = get_pending(uid)

    if pending == "training_answer":
        try:
            result = await check_student_answer(uid, payload, text)
            low = result.lower()
            if "верно" in low and "неверно" not in low and "ошиб" not in low:
                bump_stat(uid, "correct_checks")
            else:
                bump_stat(uid, "wrong_checks")
            set_pending(uid, None)
            await send_plain(message, result)
        except Exception:
            await message.answer("Не удалось проверить ответ. Попробуй ещё раз.")
        return

    if pending == "check_user_answer":
        try:
            result = await check_student_answer(uid, payload, text)
            low = result.lower()
            if "верно" in low and "неверно" not in low and "ошиб" not in low:
                bump_stat(uid, "correct_checks")
            else:
                bump_stat(uid, "wrong_checks")
            set_pending(uid, None)
            await send_plain(message, result)
        except Exception:
            await message.answer("Не удалось проверить ответ.")
        return

    answer = await solve_text(uid, text)
    await answer_and_remember(message, uid, text, answer)

# -----------------------------
# ADMIN
# -----------------------------
@router.message(Command("admin"))
async def admin_cmd(message: Message):
    if message.from_user.id not in ADMIN_IDS:
        await message.answer("Команда недоступна.")
        return

    with SessionLocal() as s:
        users = s.execute(select(func.count(UserState.user_id))).scalar_one()
        solved = s.execute(select(func.sum(Stat.solved))).scalar() or 0
        photos = s.execute(select(func.sum(Stat.photos))).scalar() or 0
        errors = s.execute(select(func.sum(Stat.ai_errors))).scalar() or 0

    await message.answer(
        "🛠 Админ-панель\n\n"
        f"Пользователей: {users}\n"
        f"Решений: {solved}\n"
        f"Фото: {photos}\n"
        f"AI-сбоев: {errors}"
    )

# -----------------------------
# HEALTH / WEBHOOK
# -----------------------------
@app.get("/")
async def root():
    return {
        "ok": True,
        "service": f"MathSolver {APP_VERSION}",
        "ai_enabled": bool(groq or FALLBACK_API_KEY),
        "groq_enabled": bool(groq),
        "fallback_enabled": bool(FALLBACK_API_KEY),
        "database": "postgres" if DATABASE_URL.startswith("postgres") else "sqlite",
    }

@app.get("/health")
async def health():
    try:
        with SessionLocal() as s:
            s.execute(select(1))
        db_ok = True
    except Exception:
        db_ok = False
    return {
        "ok": db_ok,
        "version": APP_VERSION,
        "ai_enabled": bool(groq or FALLBACK_API_KEY),
        "database_ok": db_ok,
    }

@app.get("/ready")
async def ready():
    return {
        "ready": bool(BOT_TOKEN and BASE_URL and WEBHOOK_SECRET),
        "version": APP_VERSION,
    }

@app.on_event("startup")
async def startup():
    init_db()
    await bot.set_webhook(
        f"{BASE_URL}/telegram/{WEBHOOK_SECRET}",
        secret_token=WEBHOOK_SECRET,
        drop_pending_updates=False,
        allowed_updates=dp.resolve_used_update_types(),
    )
    log.info("Started %s", APP_VERSION)

@app.on_event("shutdown")
async def shutdown():
    await bot.session.close()

@app.post("/telegram/{secret}")
async def webhook(
    secret: str,
    request: Request,
    x_telegram_bot_api_secret_token: Optional[str] = Header(default=None),
):
    if secret != WEBHOOK_SECRET:
        raise HTTPException(status_code=403, detail="bad secret")
    if x_telegram_bot_api_secret_token != WEBHOOK_SECRET:
        raise HTTPException(status_code=403, detail="bad telegram secret")

    data = await request.json()
    update = Update.model_validate(data, context={"bot": bot})
    await dp.feed_update(bot, update)
    return {"ok": True}
