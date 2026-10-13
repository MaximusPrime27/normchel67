
import asyncio
import base64
import hashlib
import html
import io
import json
import logging
import math
import os
import re
import secrets
import time
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import cv2
import httpx
import numpy as np
import pytesseract
from PIL import Image
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request, Form, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
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
    LabeledPrice,
    Message,
    PreCheckoutQuery,
    ReplyKeyboardMarkup,
    Update,
)

load_dotenv()

APP_VERSION = "V10.3 Photo UX"

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

ADMIN_WEB_USER = os.getenv("ADMIN_WEB_USER", "admin").strip()
ADMIN_WEB_PASSWORD = os.getenv("ADMIN_WEB_PASSWORD", "").strip()

PAY_SUPPORT_CONTACT = os.getenv("PAY_SUPPORT_CONTACT", "").strip()
PREMIUM_GATES_ENABLED = os.getenv("PREMIUM_GATES_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}

PREMIUM_PLANS = {
    "30": {"days": 30, "stars": 100, "label": "30 дней"},
    "90": {"days": 90, "stars": 250, "label": "90 дней"},
    "180": {"days": 180, "stars": 450, "label": "180 дней"},
    "360": {"days": 360, "stars": 900, "label": "360 дней"},
    "life": {"days": None, "stars": 1500, "label": "Навсегда"},
}

REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "55"))
MAX_CONCURRENT_AI = int(os.getenv("MAX_CONCURRENT_AI", "4"))
USER_COOLDOWN = float(os.getenv("USER_COOLDOWN_SECONDS", "1.2"))
MAX_HISTORY = int(os.getenv("MAX_HISTORY_MESSAGES", "8"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "2600"))
MAX_IMAGE_MB = float(os.getenv("MAX_IMAGE_MB", "7"))
MAX_IMAGE_SIDE = int(os.getenv("MAX_IMAGE_SIDE", "1400"))
FAST_PHOTO_MODE = os.getenv("FAST_PHOTO_MODE", "true").strip().lower() in {"1", "true", "yes", "on"}

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




class PremiumAccess(Base):
    __tablename__ = "premium_access"
    user_id = Column(Integer, primary_key=True)
    lifetime = Column(Boolean, nullable=False, default=False)
    expires_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

class StarPayment(Base):
    __tablename__ = "star_payments"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, index=True, nullable=False)
    plan_key = Column(String(16), nullable=False)
    stars = Column(Integer, nullable=False)
    currency = Column(String(8), nullable=False, default="XTR")
    invoice_payload = Column(String(128), nullable=False)
    telegram_payment_charge_id = Column(String(256), nullable=False, unique=True, index=True)
    provider_payment_charge_id = Column(String(256), nullable=True)
    status = Column(String(32), nullable=False, default="paid")
    created_at = Column(DateTime, server_default=func.now())


class TopicProgress(Base):
    __tablename__ = "topic_progress"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, index=True, nullable=False)
    subject = Column(String(16), nullable=False)
    topic = Column(String(160), nullable=False)
    attempts = Column(Integer, nullable=False, default=0)
    correct = Column(Integer, nullable=False, default=0)
    wrong = Column(Integer, nullable=False, default=0)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

class ErrorLog(Base):
    __tablename__ = "error_log"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, index=True, nullable=True)
    source = Column(String(64), nullable=False)
    error_type = Column(String(128), nullable=False)
    message = Column(Text, nullable=False)
    created_at = Column(DateTime, server_default=func.now())

class ExamSession(Base):
    __tablename__ = "exam_sessions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, index=True, nullable=False)
    exam_type = Column(String(16), nullable=False)
    subject = Column(String(16), nullable=False)
    score = Column(Integer, nullable=False, default=0)
    max_score = Column(Integer, nullable=False, default=0)
    completed = Column(Boolean, nullable=False, default=False)
    payload = Column(Text, nullable=True)
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




def utcnow_naive():
    return datetime.now(timezone.utc).replace(tzinfo=None)

def get_premium(uid: int):
    with SessionLocal() as s:
        row = s.get(PremiumAccess, uid)
        if not row:
            return {
                "active": False,
                "lifetime": False,
                "expires_at": None,
            }
        now = utcnow_naive()
        active = bool(row.lifetime or (row.expires_at and row.expires_at > now))
        return {
            "active": active,
            "lifetime": bool(row.lifetime),
            "expires_at": row.expires_at,
        }

def premium_text(uid: int) -> str:
    info = get_premium(uid)
    if info["lifetime"]:
        return "💎 Premium: навсегда"
    if info["active"] and info["expires_at"]:
        return "💎 Premium до " + info["expires_at"].strftime("%d.%m.%Y")
    return "⚪ Premium не активен"

def grant_premium(uid: int, plan_key: str):
    plan = PREMIUM_PLANS[plan_key]
    now = utcnow_naive()
    with SessionLocal() as s:
        row = s.get(PremiumAccess, uid)
        if not row:
            row = PremiumAccess(user_id=uid, lifetime=False, expires_at=None)
            s.add(row)

        if plan["days"] is None:
            row.lifetime = True
            row.expires_at = None
        elif not row.lifetime:
            start = now
            if row.expires_at and row.expires_at > now:
                start = row.expires_at
            row.expires_at = start + timedelta(days=int(plan["days"]))
        s.commit()

def payment_exists(charge_id: str) -> bool:
    with SessionLocal() as s:
        return s.execute(
            select(StarPayment).where(
                StarPayment.telegram_payment_charge_id == charge_id
            )
        ).scalar_one_or_none() is not None

def save_star_payment(
    uid: int,
    plan_key: str,
    stars: int,
    payload: str,
    charge_id: str,
    provider_charge_id: Optional[str],
):
    with SessionLocal() as s:
        s.add(StarPayment(
            user_id=uid,
            plan_key=plan_key,
            stars=stars,
            currency="XTR",
            invoice_payload=payload,
            telegram_payment_charge_id=charge_id,
            provider_payment_charge_id=provider_charge_id,
            status="paid",
        ))
        s.commit()

def require_premium_flag(uid: int) -> bool:
    if not PREMIUM_GATES_ENABLED:
        return True
    return bool(get_premium(uid)["active"])


def log_error(user_id: Optional[int], source: str, exc: Exception):
    try:
        with SessionLocal() as s:
            s.add(ErrorLog(
                user_id=user_id,
                source=source[:64],
                error_type=type(exc).__name__[:128],
                message=str(exc)[:4000],
            ))
            s.commit()
    except Exception:
        log.exception("Failed to persist error log")

def update_topic_progress(uid: int, subject: str, topic: str, correct: bool):
    with SessionLocal() as s:
        row = s.execute(
            select(TopicProgress).where(
                TopicProgress.user_id == uid,
                TopicProgress.subject == subject,
                TopicProgress.topic == topic,
            )
        ).scalar_one_or_none()
        if not row:
            row = TopicProgress(
                user_id=uid, subject=subject, topic=topic,
                attempts=0, correct=0, wrong=0
            )
            s.add(row)
        row.attempts += 1
        if correct:
            row.correct += 1
        else:
            row.wrong += 1
        s.commit()

def weak_topics(uid: int, limit: int = 5):
    with SessionLocal() as s:
        rows = s.execute(
            select(TopicProgress)
            .where(TopicProgress.user_id == uid, TopicProgress.attempts > 0)
        ).scalars().all()
    rows.sort(key=lambda r: (r.correct / max(r.attempts, 1), -r.attempts))
    return rows[:limit]

def create_exam_session(uid: int, exam_type: str, subject: str, payload: dict, max_score: int):
    with SessionLocal() as s:
        row = ExamSession(
            user_id=uid,
            exam_type=exam_type,
            subject=subject,
            score=0,
            max_score=max_score,
            completed=False,
            payload=json.dumps(payload, ensure_ascii=False),
        )
        s.add(row)
        s.commit()
        s.refresh(row)
        return row.id

def finish_exam_session(session_id: int, score: int):
    with SessionLocal() as s:
        row = s.get(ExamSession, session_id)
        if row:
            row.score = score
            row.completed = True
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
        [KeyboardButton(text="📚 Предметы")],
        [KeyboardButton(text="🧮 Математика"), KeyboardButton(text="🇷🇺 Русский язык")],
        [KeyboardButton(text="🎓 ОГЭ"), KeyboardButton(text="🏆 ЕГЭ")],
        [KeyboardButton(text="📷 Решить по фото"), KeyboardButton(text="✅ Проверить решение — бесплатно")],
        [KeyboardButton(text="📖 Учиться"), KeyboardButton(text="📝 Тренировка")],
        [KeyboardButton(text="🧪 Вариант ОГЭ/ЕГЭ"), KeyboardButton(text="📉 Слабые темы")],
        [KeyboardButton(text="💎 Premium"), KeyboardButton(text="👤 Профиль")],
        [KeyboardButton(text="🎒 Выбрать класс"), KeyboardButton(text="📚 История")],
        [KeyboardButton(text="⭐ Избранное")],
        [KeyboardButton(text="🧹 Сбросить контекст"), KeyboardButton(text="ℹ️ Помощь")],
    ],
    resize_keyboard=True,
)



PREMIUM_KB = InlineKeyboardMarkup(
    inline_keyboard=[
        [
            InlineKeyboardButton(text="30 дней — 100 ⭐", callback_data="buy:30"),
            InlineKeyboardButton(text="90 дней — 250 ⭐", callback_data="buy:90"),
        ],
        [
            InlineKeyboardButton(text="180 дней — 450 ⭐", callback_data="buy:180"),
            InlineKeyboardButton(text="360 дней — 900 ⭐", callback_data="buy:360"),
        ],
        [
            InlineKeyboardButton(text="Навсегда — 1500 ⭐", callback_data="buy:life"),
        ],
    ]
)

EXAM_TYPE_KB = InlineKeyboardMarkup(
    inline_keyboard=[[
        InlineKeyboardButton(text="🎓 ОГЭ", callback_data="examtype:oge"),
        InlineKeyboardButton(text="🏆 ЕГЭ", callback_data="examtype:ege"),
    ]]
)

EXAM_SUBJECT_KB = InlineKeyboardMarkup(
    inline_keyboard=[
        [
            InlineKeyboardButton(text="🧮 Математика", callback_data="examsubject:math"),
            InlineKeyboardButton(text="🇷🇺 Русский", callback_data="examsubject:russian"),
        ],
        [
            InlineKeyboardButton(text="⚛️ Физика", callback_data="examsubject:physics"),
            InlineKeyboardButton(text="🧪 Химия", callback_data="examsubject:chemistry"),
        ],
        [
            InlineKeyboardButton(text="🧬 Биология", callback_data="examsubject:biology"),
            InlineKeyboardButton(text="💻 Информатика", callback_data="examsubject:informatics"),
        ],
        [
            InlineKeyboardButton(text="🌍 География", callback_data="examsubject:geography"),
            InlineKeyboardButton(text="🏛 История", callback_data="examsubject:history"),
        ],
        [
            InlineKeyboardButton(text="⚖️ Обществознание", callback_data="examsubject:social"),
            InlineKeyboardButton(text="🇬🇧 Английский", callback_data="examsubject:english"),
        ],
        [
            InlineKeyboardButton(text="📖 Литература", callback_data="examsubject:literature"),
        ],
    ]
)


SUBJECTS = {
    "math": {"label": "🧮 Математика", "mode": "math"},
    "russian": {"label": "🇷🇺 Русский язык", "mode": "russian"},
    "physics": {"label": "⚛️ Физика", "mode": "physics"},
    "chemistry": {"label": "🧪 Химия", "mode": "chemistry"},
    "biology": {"label": "🧬 Биология", "mode": "biology"},
    "informatics": {"label": "💻 Информатика", "mode": "informatics"},
    "geography": {"label": "🌍 География", "mode": "geography"},
    "history": {"label": "🏛 История", "mode": "history"},
    "social": {"label": "⚖️ Обществознание", "mode": "social"},
    "english": {"label": "🇬🇧 Английский", "mode": "english"},
    "literature": {"label": "📖 Литература", "mode": "literature"},
}

SUBJECT_MENU_KB = InlineKeyboardMarkup(
    inline_keyboard=[
        [
            InlineKeyboardButton(text="🧮 Математика", callback_data="setsubject:math"),
            InlineKeyboardButton(text="🇷🇺 Русский", callback_data="setsubject:russian"),
        ],
        [
            InlineKeyboardButton(text="⚛️ Физика", callback_data="setsubject:physics"),
            InlineKeyboardButton(text="🧪 Химия", callback_data="setsubject:chemistry"),
        ],
        [
            InlineKeyboardButton(text="🧬 Биология", callback_data="setsubject:biology"),
            InlineKeyboardButton(text="💻 Информатика", callback_data="setsubject:informatics"),
        ],
        [
            InlineKeyboardButton(text="🌍 География", callback_data="setsubject:geography"),
            InlineKeyboardButton(text="🏛 История", callback_data="setsubject:history"),
        ],
        [
            InlineKeyboardButton(text="⚖️ Обществознание", callback_data="setsubject:social"),
            InlineKeyboardButton(text="🇬🇧 Английский", callback_data="setsubject:english"),
        ],
        [
            InlineKeyboardButton(text="📖 Литература", callback_data="setsubject:literature"),
        ],
    ]
)

STUDY_SUBJECT_KB = InlineKeyboardMarkup(
    inline_keyboard=[
        [
            InlineKeyboardButton(text="🧮 Математика", callback_data="subject:math"),
            InlineKeyboardButton(text="🇷🇺 Русский", callback_data="subject:russian"),
        ],
        [
            InlineKeyboardButton(text="⚛️ Физика", callback_data="subject:physics"),
            InlineKeyboardButton(text="🧪 Химия", callback_data="subject:chemistry"),
        ],
        [
            InlineKeyboardButton(text="🧬 Биология", callback_data="subject:biology"),
            InlineKeyboardButton(text="💻 Информатика", callback_data="subject:informatics"),
        ],
        [
            InlineKeyboardButton(text="🌍 География", callback_data="subject:geography"),
            InlineKeyboardButton(text="🏛 История", callback_data="subject:history"),
        ],
        [
            InlineKeyboardButton(text="⚖️ Обществознание", callback_data="subject:social"),
            InlineKeyboardButton(text="🇬🇧 Английский", callback_data="subject:english"),
        ],
        [
            InlineKeyboardButton(text="📖 Литература", callback_data="subject:literature"),
        ],
    ]
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
        [InlineKeyboardButton(text="📸 Решить всё фото", callback_data="crop:full")],
        [InlineKeyboardButton(text="✂️ Выбрать область", callback_data="crop:choose")],
    ]
)

CROP_DETAIL_KB = InlineKeyboardMarkup(
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
    "russian": {
        9: ["Орфография", "Пунктуация", "СПП/ССП/БСП", "Н/НН", "НЕ/НИ", "ОГЭ: изложение и сочинение"],
        10: ["Орфография", "Морфология", "Синтаксис", "Пунктуация", "Лексика и паронимы", "Текст и стили речи"],
        11: ["Орфография ЕГЭ", "Пунктуация ЕГЭ", "Грамматические нормы", "Средства выразительности", "Работа с текстом", "Сочинение ЕГЭ"],
    },
    "physics": {
        9: ["Механика", "Законы Ньютона", "Работа и энергия", "Импульс", "Электричество", "Оптика"],
        10: ["Кинематика", "Динамика", "МКТ", "Термодинамика", "Электростатика", "Постоянный ток"],
        11: ["Магнитное поле", "Электромагнитная индукция", "Колебания и волны", "Оптика", "Квантовая физика", "Атом и ядро"],
    },
    "chemistry": {
        9: ["Строение атома", "Периодический закон", "Химическая связь", "Классы неорганических веществ", "Уравнения реакций", "Расчёты по уравнениям"],
        10: ["Органическая химия", "Углеводороды", "Спирты и фенолы", "Альдегиды и кислоты", "Углеводы", "Амины и аминокислоты"],
        11: ["Окислительно-восстановительные реакции", "Электролиз", "Равновесие", "Растворы", "Цепочки превращений", "Расчётные задачи"],
    },
    "biology": {
        9: ["Клетка", "Ткани", "Органы и системы", "Генетика", "Эволюция", "Экология"],
        10: ["Цитология", "Обмен веществ", "Размножение", "Генетика", "Селекция", "Биосфера"],
        11: ["Молекулярная биология", "Генетика человека", "Эволюция", "Экология", "Анатомия", "ЕГЭ: анализ заданий"],
    },
    "informatics": {
        9: ["Информация и кодирование", "Системы счисления", "Логика", "Алгоритмы", "Таблицы", "ОГЭ: программирование"],
        10: ["Алгоритмы", "Python", "Логические выражения", "Системы счисления", "Массивы", "Компьютерные сети"],
        11: ["Python", "Динамическое программирование", "Графы", "Теория игр", "Логика", "ЕГЭ: задачи 24–27"],
    },
    "geography": {
        9: ["Россия: природа", "Население России", "Хозяйство России", "Регионы России", "Карты", "ОГЭ: задания по картам"],
        10: ["Политическая карта мира", "Природные ресурсы", "Население мира", "Мировое хозяйство", "Регионы мира", "Глобальные проблемы"],
        11: ["География России", "Экономическая география", "Демография", "Мировое хозяйство", "Карты и статистика", "ЕГЭ: анализ данных"],
    },
    "history": {
        9: ["Россия XIX века", "Реформы", "Общественные движения", "Культура XIX века", "Первая мировая", "Революции"],
        10: ["Древняя Русь", "Московское государство", "Россия XVII века", "Пётр I", "XVIII век", "Культура"],
        11: ["Россия XX века", "СССР", "Великая Отечественная война", "Послевоенный СССР", "Перестройка", "Современная Россия"],
    },
    "social": {
        9: ["Человек и общество", "Экономика", "Социальная сфера", "Политика", "Право", "ОГЭ: практика"],
        10: ["Общество", "Духовная культура", "Экономика", "Социальные отношения", "Политика", "Право"],
        11: ["Экономика", "Социология", "Политология", "Конституционное право", "Гражданское право", "ЕГЭ: задания второй части"],
    },
    "english": {
        9: ["Present/Past/Future", "Modal verbs", "Passive Voice", "Reported Speech", "Vocabulary", "ОГЭ: письмо и чтение"],
        10: ["Tenses", "Conditionals", "Gerund/Infinitive", "Articles", "Word formation", "Writing"],
        11: ["Advanced grammar", "Vocabulary", "Reading", "Listening strategies", "Writing", "ЕГЭ: устная и письменная части"],
    },
    "literature": {
        9: ["Древнерусская литература", "Классицизм и сентиментализм", "Пушкин", "Лермонтов", "Гоголь", "Анализ эпизода и сочинение"],
        10: ["Гончаров", "Тургенев", "Островский", "Достоевский", "Толстой", "Чехов"],
        11: ["Серебряный век", "Булгаков", "Шолохов", "Пастернак", "Солженицын", "ЕГЭ: анализ и сочинение"],
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


Физика:
механика; молекулярная физика; термодинамика; электродинамика;
колебания и волны; оптика; квантовая, атомная и ядерная физика;
расчётные задачи; ОГЭ/ЕГЭ.

Химия:
строение атома; периодический закон; химическая связь; неорганическая
и органическая химия; ОВР; электролиз; растворы; химическое равновесие;
цепочки превращений; расчётные задачи; ОГЭ/ЕГЭ.

Биология:
цитология; анатомия; генетика; эволюция; экология; ботаника; зоология;
молекулярная биология; физиология человека; ОГЭ/ЕГЭ.

Информатика:
информация и кодирование; системы счисления; логика; алгоритмы; Python;
таблицы; базы данных; сети; графы; теория игр; ОГЭ/ЕГЭ.

География:
карты; физическая и социально-экономическая география; Россия;
население; мировое хозяйство; природные ресурсы; статистические данные;
ОГЭ/ЕГЭ.

История:
история России с древнейших времён до современности; всемирная история;
хронология; личности; события; причины и последствия; культура; ОГЭ/ЕГЭ.

Обществознание:
человек и общество; духовная культура; экономика; социальные отношения;
политика; право; Конституция РФ; ОГЭ/ЕГЭ.

Английский язык:
грамматика; лексика; словообразование; чтение; письмо; перевод;
объяснение ошибок; школьные задания; ОГЭ/ЕГЭ.

Литература:
история русской литературы; теория литературы; жанры; композиция;
сюжет; конфликт; система образов; авторская позиция; тема и идея;
художественные средства; анализ эпизода и стихотворения; сравнительный
анализ; аргументация; сочинение; ОГЭ/ЕГЭ.
"""


EXPERT_PROMPTS = {
    "math": """
Ты эксперт по школьной математике 9–11 классов.
Главное: точность, проверка вычислений, понятные шаги.
Для уравнений и систем проверяй найденные корни подстановкой, где это уместно.
Для геометрии явно указывай используемую теорему.
Формат: Решение → Проверка/обоснование → Ответ.
""",
    "russian": """
Ты эксперт по русскому языку 9–11 классов.
Объясняй правило, показывай конкретное место ошибки и почему выбран именно этот вариант.
Для пунктуации объясняй каждую спорную запятую.
Для сочинений оценивай структуру, логику, речевые и грамматические ошибки.
Формат: Правило → Разбор → Ответ.
""",
    "physics": """
Ты эксперт по школьной физике 9–11 классов.
Всегда отслеживай единицы измерения и перевод в СИ.
Если задача расчётная: выпиши Дано, формулы, подстановку чисел и итог с единицами.
Проверяй размерность результата.
Если данных не хватает, прямо скажи об этом.
Формат: Дано → Формулы → Решение → Проверка единиц → Ответ.
""",
    "chemistry": """
Ты эксперт по школьной химии 9–11 классов.
Проверяй коэффициенты в уравнениях реакций, заряды, валентности и формулы веществ.
В расчётных задачах показывай количество вещества, молярные массы и пропорции.
Для ОВР при необходимости показывай изменение степеней окисления.
Формат: Что происходит → Уравнение/формулы → Расчёт → Ответ.
""",
    "biology": """
Ты эксперт по школьной биологии 9–11 классов.
Опирайся на точные биологические термины, но объясняй простым языком.
Для генетики показывай генотипы, гаметы, логику скрещивания и вывод.
Для анатомии связывай строение с функцией.
Формат: Ключевая идея → Объяснение → Вывод/ответ.
""",
    "informatics": """
Ты эксперт по школьной информатике 9–11 классов.
Для алгоритмических задач сначала объясняй идею, затем шаги.
Для систем счисления и логики показывай промежуточные преобразования.
Для программирования давай короткий понятный код на Python только если это реально нужно.
Проверяй граничные случаи.
Формат: Идея → Алгоритм → Решение → Ответ.
""",
    "geography": """
Ты эксперт по школьной географии 9–11 классов.
Связывай природные условия, население и хозяйство причинно-следственно.
Для карт и статистики объясняй, как был сделан вывод.
Не выдумывай численные показатели, если они не даны.
Формат: Факт → Причина/связь → Вывод → Ответ.
""",
    "history": """
Ты эксперт по истории России и всемирной истории для 9–11 классов.
Всегда различай дату, событие, причину, повод и последствия.
Для спорных трактовок отделяй установленный факт от интерпретации.
Для заданий ЕГЭ показывай причинно-следственные связи и роль личности.
Формат: Контекст → Факты → Причины/последствия → Ответ.
""",
    "social": """
Ты эксперт по обществознанию 9–11 классов.
Используй школьные определения понятий и не подменяй их бытовыми формулировками.
Для права опирайся на общие принципы законодательства без выдуманных статей.
Для экономики и политики показывай причинно-следственные связи.
Формат: Понятие/норма → Разбор ситуации → Вывод → Ответ.
""",
    "english": """
You are an expert school English tutor for grades 9–11.
Explain grammar in Russian unless the task explicitly asks for English-only output.
For grammar tasks: identify the rule, explain the error, then give the correct form.
For writing: check grammar, vocabulary, structure and naturalness.
For translation: preserve meaning, not just word-for-word structure.
Format: Rule/idea → Analysis → Correct answer.
""",
    "literature": """
Ты эксперт по русской литературе 9–11 классов.
Анализируй произведение через тему, идею, конфликт, композицию, систему образов и авторскую позицию.
Для стихотворений обращай внимание на лирического героя, мотивы, композицию, размер и художественные средства, если это уместно.
Для сочинений помогай сформулировать тезис, подобрать аргументы и выстроить логичный вывод.
Не выдумывай точные цитаты: если не уверен в формулировке, перескажи смысл без кавычек.
Для ОГЭ/ЕГЭ отвечай в школьном экзаменационном формате.
Формат: Тезис → Анализ → Аргументы → Вывод.
""",
}

def expert_prompt_for_mode(mode: str) -> str:
    return EXPERT_PROMPTS.get(mode, "")

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
- Для русского и английского: Правило → Разбор → Ответ.
- Для физики и химии: Дано → Формула/реакция → Решение → Ответ с единицами.
- Для биологии, истории, обществознания и географии: краткий вывод → объяснение → ответ.
- Для литературы: Тезис → анализ текста/образа → аргументы → вывод; не выдумывай цитаты.
- Для информатики: алгоритм → решение → ответ; код только если он действительно нужен.
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
def compress_image(data: bytes, max_side: int = MAX_IMAGE_SIDE) -> bytes:
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


def ocr_image_quick(data: bytes) -> str:
    """Single-pass OCR used only as a fallback in fast photo mode."""
    try:
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return ""
        h, w = img.shape[:2]
        scale = max(1.0, min(2.0, 1400 / max(h, w)))
        if scale > 1.05:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return pytesseract.image_to_string(
            Image.fromarray(img),
            lang="rus+eng",
            config="--oem 3 --psm 6",
        ).strip()
    except Exception:
        return ""


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

async def ai_chat(messages: List[dict], model: Optional[str] = None, clean_output: bool = True) -> str:
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
                        return clean_text(text) if clean_output else text
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
                return clean_text(fallback) if clean_output else fallback
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

    st = get_state(uid)
    expert = expert_prompt_for_mode(st.mode)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": mode_line(uid)},
    ]
    if expert:
        messages.append({"role": "system", "content": expert})
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
    except Exception as e:
        bump_stat(uid, "ai_errors")
        log_error(uid, "solve_text", e)
        log.exception("solve_text failed")
        if hint:
            return "Решение:\n" + clean_text(hint) + "\n\nОтвет: см. точный результат выше."
        return (
            "Сейчас ИИ временно недоступен, а локальный решатель не смог "
            "уверенно разобрать задачу. Попробуй ещё раз через несколько секунд."
        )

async def solve_image(uid: int, data: bytes, ocr: str = "", check_solution: bool = False) -> str:
    # FAST PATH:
    # 1) send compressed image directly to vision model
    # 2) only run OCR if vision fails
    compressed = compress_image(data, MAX_IMAGE_SIDE)

    special = (
        "Проверь именно решение ученика. Найди первый неверный шаг, объясни ошибку "
        "и покажи правильное продолжение. Не ограничивайся только финальным ответом."
        if check_solution else
        "Реши задание с изображения. Сначала самостоятельно прочитай текст с фото."
    )

    st = get_state(uid)
    expert = expert_prompt_for_mode(st.mode)

    if groq:
        b64 = base64.b64encode(compressed).decode()
        prompt = (
            SYSTEM_PROMPT + "\n\n" + mode_line(uid) + "\n\n" +
            (expert + "\n\n" if expert else "") +
            special +
            "\n\nВажно: анализируй изображение напрямую. "
            "Не требуй предварительного OCR. Если небольшой фрагмент не читается, "
            "скажи только что именно не удалось разобрать."
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
            if answer and len(answer.strip()) >= 10:
                bump_stat(uid, "photos")
                return answer
        except Exception as e:
            bump_stat(uid, "ai_errors")
            log_error(uid, "solve_image_vision", e)

    # FALLBACK ONLY: quick OCR
    if not ocr:
        ocr = await asyncio.to_thread(ocr_image_quick, compressed)

    if ocr:
        fallback_prompt = (
            ("Проверь решение ученика по распознанному тексту:\n" if check_solution
             else "Реши задание по распознанному тексту:\n")
            + ocr
        )
        return await solve_text(uid, fallback_prompt, use_history=False)

    return (
        "Не удалось уверенно прочитать фото. "
        "Попробуй сфотографировать ближе, без наклона и при хорошем освещении."
    )

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
        f"Предмет: {SUBJECTS.get(subject, {'label': subject})['label']}.\n" 
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
            f"Предмет: {SUBJECTS.get(subject, {'label': subject})['label']}. Тема: {topic}. "
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
        f"🤖 ИИ: {'подключён' if groq else 'резервный/локальный режим'}\n"
        f"{premium_text(message.from_user.id)}\n\n"
        "Я могу помогать по 11 школьным предметам, решать задачи, проверять домашнюю работу, обучать по темам, "
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
        f"Фото: {stat.photos}\n"
        f"{premium_text(uid)}\n"
        f"Экспертный режим: {'да' if expert_prompt_for_mode(st.mode) else 'общий'}"
    )

@router.message(Command("version"))
async def version_cmd(message: Message):
    await message.answer(APP_VERSION)

@router.message(Command("help"))
@router.message(F.text == "ℹ️ Помощь")
async def help_cmd(message: Message):
    await message.answer(
        "Что умеет бот:\n\n"
        "• математика и русский язык;\n"
        "• физика, химия, биология;\n"
        "• информатика и география;\n"
        "• история и обществознание;\n"
        "• английский язык;\n"
        "• литература;\n"
        "• ОГЭ/ЕГЭ;\n"
        "• быстро решать задания по фото;\n"
        "• бесплатно проверять решение ученика по фото;\n"
        "• объяснять проще/подробнее/другим способом;\n"
        "• давать похожие задачи;\n"
        "• проверять твой ответ;\n"
        "• проводить мини-уроки и тренировки;\n"
        "• создавать варианты ОГЭ/ЕГЭ с баллами;\n"
        "• показывать слабые темы;\n"
        "• хранить профиль, историю и избранное.\n\n"
        "Команды: /status, /version, /reset, /profile, /premium, /paysupport, /expert"
    )


# -----------------------------
# TELEGRAM STARS / PREMIUM
# -----------------------------
@router.message(Command("premium"))
@router.message(F.text == "💎 Premium")
async def premium_menu(message: Message):
    uid = message.from_user.id
    ensure_user(uid)
    status = premium_text(uid)
    await message.answer(
        "💎 MathSolver Premium\n\n"
        f"{status}\n\n"
        "Тарифы:\n"
        "• 30 дней — 100 Stars\n"
        "• 90 дней — 250 Stars\n"
        "• 180 дней — 450 Stars\n"
        "• 360 дней — 900 Stars\n"
        "• Навсегда — 1500 Stars\n\n"
        "Premium открывает расширенные функции: "
        "проверку решения по фото, варианты ОГЭ/ЕГЭ и статистику слабых тем.\n\n"
        "Оплата разовая — автоматического списания нет.",
        reply_markup=PREMIUM_KB,
    )

@router.callback_query(F.data.startswith("buy:"))
async def buy_premium(callback: CallbackQuery):
    uid = callback.from_user.id
    plan_key = callback.data.split(":", 1)[1]
    plan = PREMIUM_PLANS.get(plan_key)
    if not plan:
        await callback.answer("Неизвестный тариф.", show_alert=True)
        return

    if get_premium(uid)["lifetime"]:
        await callback.answer("У тебя уже Premium навсегда.", show_alert=True)
        return

    payload = f"premium:{plan_key}"
    await callback.answer()
    await callback.message.answer_invoice(
        title=f"Premium — {plan['label']}",
        description=f"MathSolver Premium на {plan['label']}. Разовая покупка.",
        payload=payload,
        currency="XTR",
        prices=[LabeledPrice(
            label=f"Premium {plan['label']}",
            amount=int(plan["stars"]),
        )],
        provider_token="",
    )

@router.pre_checkout_query()
async def pre_checkout_handler(query: PreCheckoutQuery):
    payload = query.invoice_payload or ""
    try:
        prefix, plan_key = payload.split(":", 1)
        plan = PREMIUM_PLANS.get(plan_key)
        valid = (
            prefix == "premium"
            and plan is not None
            and query.currency == "XTR"
            and int(query.total_amount) == int(plan["stars"])
        )
    except Exception:
        valid = False

    if not valid:
        await query.answer(
            ok=False,
            error_message="Не удалось проверить тариф. Открой Premium и попробуй снова.",
        )
        return

    await query.answer(ok=True)

@router.message(F.successful_payment)
async def successful_payment_handler(message: Message):
    uid = message.from_user.id
    payment = message.successful_payment

    if payment.currency != "XTR":
        return

    payload = payment.invoice_payload or ""
    try:
        prefix, plan_key = payload.split(":", 1)
    except Exception:
        await message.answer("Платёж получен, но тариф не распознан. Напиши в /paysupport.")
        return

    plan = PREMIUM_PLANS.get(plan_key)
    if prefix != "premium" or not plan:
        await message.answer("Платёж получен, но тариф не распознан. Напиши в /paysupport.")
        return

    charge_id = payment.telegram_payment_charge_id

    # Idempotency: webhook/update can theoretically be delivered again.
    if payment_exists(charge_id):
        await message.answer(
            "✅ Этот платёж уже обработан.\n" + premium_text(uid)
        )
        return

    # Validate the final successful payment too.
    if int(payment.total_amount) != int(plan["stars"]):
        await message.answer(
            "Платёж получен с неожиданной суммой. Доступ автоматически не выдан. "
            "Напиши в /paysupport."
        )
        return

    try:
        save_star_payment(
            uid=uid,
            plan_key=plan_key,
            stars=int(payment.total_amount),
            payload=payload,
            charge_id=charge_id,
            provider_charge_id=getattr(payment, "provider_payment_charge_id", None),
        )
        grant_premium(uid, plan_key)
    except Exception as e:
        log_error(uid, "successful_payment", e)
        await message.answer(
            "Платёж успешен, но возникла ошибка активации. "
            "Не оплачивай повторно — напиши в /paysupport."
        )
        return

    await message.answer(
        "✅ Оплата прошла успешно!\n\n"
        f"Тариф: {plan['label']}\n"
        f"Списано: {plan['stars']} ⭐\n"
        f"{premium_text(uid)}"
    )

@router.message(Command("paysupport"))
async def pay_support(message: Message):
    contact = PAY_SUPPORT_CONTACT or "администратору бота"
    await message.answer(
        "💳 Поддержка по оплате\n\n"
        f"Связь: {contact}\n\n"
        "Если платёж прошёл, но Premium не появился, "
        "не оплачивай второй раз. Сообщи дату платежа и свой Telegram ID: "
        f"{message.from_user.id}"
    )



@router.message(F.text == "📚 Предметы")
async def subjects_menu(message: Message):
    await message.answer("Выбери предмет:", reply_markup=SUBJECT_MENU_KB)

@router.callback_query(F.data.startswith("setsubject:"))
async def set_subject_callback(callback: CallbackQuery):
    key = callback.data.split(":", 1)[1]
    item = SUBJECTS.get(key)
    if not item:
        await callback.answer("Неизвестный предмет.", show_alert=True)
        return
    update_state(callback.from_user.id, mode=item["mode"])
    clear_history(callback.from_user.id)
    await callback.answer("Предмет выбран")
    await callback.message.answer(
        f"{item['label']} — режим включён.\n"
        f"🎒 Класс: {get_state(callback.from_user.id).grade}\n\n"
        "Экспертный режим для этого предмета включён. Теперь отправь вопрос, задачу или фото."
    )



@router.message(Command("expert"))
async def expert_status(message: Message):
    st = get_state(message.from_user.id)
    label = next(
        (v["label"] for v in SUBJECTS.values() if v["mode"] == st.mode),
        st.mode,
    )
    enabled = bool(expert_prompt_for_mode(st.mode))
    await message.answer(
        f"🧠 Экспертный режим\n\n"
        f"Предмет: {label}\n"
        f"Статус: {'включён' if enabled else 'общий режим'}\n"
        f"Класс: {st.grade}"
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
        f"Избранных задач: {fav_count}\n"
        f"{premium_text(uid)}"
    )
    weak = weak_topics(uid, 3)
    if weak:
        await message.answer(
            "📉 Слабые темы:\n" +
            "\n".join(
                f"• {r.topic}: {round(100*r.correct/max(r.attempts,1))}%"
                for r in weak
            )
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
    await message.answer("Выбери предмет для мини-урока:", reply_markup=STUDY_SUBJECT_KB)

@router.message(F.text == "📝 Тренировка")
async def training_menu(message: Message):
    set_pending(message.from_user.id, "choose_training_subject")
    await message.answer("Выбери предмет для тренировки:", reply_markup=STUDY_SUBJECT_KB)

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


@router.message(F.text == "📉 Слабые темы")
async def weak_topics_handler(message: Message):
    if not require_premium_flag(message.from_user.id):
        await message.answer("💎 Статистика слабых тем доступна в Premium.", reply_markup=PREMIUM_KB)
        return
    rows = weak_topics(message.from_user.id, 6)
    if not rows:
        await message.answer(
            "📉 Пока недостаточно данных. Пройди несколько тренировок "
            "или вариантов ОГЭ/ЕГЭ."
        )
        return
    lines = ["📉 Темы, которые стоит повторить:", ""]
    for r in rows:
        accuracy = round(100 * r.correct / max(r.attempts, 1))
        lines.append(
            f"• {r.topic} — {accuracy}% верных "
            f"({r.correct}/{r.attempts})"
        )
    await message.answer("\n".join(lines))

@router.message(F.text == "🧪 Вариант ОГЭ/ЕГЭ")
async def exam_menu(message: Message):
    if not require_premium_flag(message.from_user.id):
        await message.answer("💎 Полные варианты ОГЭ/ЕГЭ доступны в Premium.", reply_markup=PREMIUM_KB)
        return
    set_pending(message.from_user.id, "choose_exam_type")
    await message.answer("Выбери экзамен:", reply_markup=EXAM_TYPE_KB)

@router.callback_query(F.data.startswith("examtype:"))
async def exam_type_callback(callback: CallbackQuery):
    exam_type = callback.data.split(":")[1]
    set_pending(callback.from_user.id, "choose_exam_subject", {"exam_type": exam_type})
    await callback.answer()
    await callback.message.answer("Выбери предмет:", reply_markup=EXAM_SUBJECT_KB)

@router.callback_query(F.data.startswith("examsubject:"))
async def exam_subject_callback(callback: CallbackQuery):
    uid = callback.from_user.id
    subject = callback.data.split(":")[1]
    pending, payload = get_pending(uid)
    if pending != "choose_exam_subject":
        await callback.answer("Сначала выбери экзамен.", show_alert=True)
        return
    exam_type = payload.get("exam_type", "oge")
    await callback.answer("Генерирую вариант…")
    try:
        items = await generate_exam_variant(uid, exam_type, subject)
        max_score = sum(int(i.get("points", 1)) for i in items)
        session_id = create_exam_session(
            uid, exam_type, subject,
            {"items": items, "exam_type": exam_type, "subject": subject},
            max_score,
        )
        set_pending(uid, "exam_answers", {
            "session_id": session_id,
            "items": items,
            "exam_type": exam_type,
            "subject": subject,
        })
        await send_plain(
            callback.message,
            format_exam_variant(items, exam_type, subject)
        )
    except Exception as e:
        log_error(uid, "generate_exam_variant", e)
        await callback.message.answer(
            "Не удалось создать вариант. Попробуй ещё раз через несколько секунд."
        )


# -----------------------------
# PHOTO / HOMEWORK CHECK
# -----------------------------
@router.message(F.text == "📷 Решить по фото")
async def solve_photo_mode(message: Message):
    set_pending(message.from_user.id, "awaiting_photo", {"check_solution": False})
    await message.answer("📷 Отправь фото задания. После загрузки можно сразу нажать «📸 Решить всё фото».")

@router.message(F.text == "✅ Проверить решение — бесплатно")
async def check_photo_mode(message: Message):
    set_pending(message.from_user.id, "awaiting_photo", {"check_solution": True})
    await message.answer(
        "✅ Отправь фото своего решения. После загрузки можно сразу нажать «📸 Решить всё фото»."
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
        data = compress_image(data, MAX_IMAGE_SIDE)

    photo_cache[uid] = data
    await message.answer_photo(
        BufferedInputFile(data, filename="task.jpg"),
        caption="📷 Что сделать с фото?",
        reply_markup=CROP_KB,
    )


@router.callback_query(F.data == "crop:choose")
async def crop_choose_callback(callback: CallbackQuery):
    uid = callback.from_user.id
    if uid not in photo_cache:
        await callback.answer("Фото не найдено.", show_alert=True)
        return
    await callback.answer()
    await callback.message.answer(
        "✂️ Выбери область изображения:",
        reply_markup=CROP_DETAIL_KB,
    )

@router.callback_query(F.data.startswith("crop:"))
async def crop_callback(callback: CallbackQuery):
    uid = callback.from_user.id
    data = photo_cache.get(uid)
    if not data:
        await callback.answer("Фото не найдено.", show_alert=True)
        return

    preset = callback.data.split(":")[1]
    if preset == "choose":
        await callback.answer()
        return
    cropped = crop_image(data, preset)
    await callback.answer("Анализирую фото…")
    pending, payload = get_pending(uid)
    check_solution = bool(payload.get("check_solution")) if pending == "awaiting_photo" else False

    # Vision-first: OCR is no longer run before AI.
    answer = await solve_image(uid, cropped, "", check_solution=check_solution)
    task_label = "Проверка решения по фото" if check_solution else "Задание по фото"
    last_task_cache[uid] = task_label
    last_answer_cache[uid] = answer
    set_pending(uid, None)

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
        "▶️ Старт", "📚 Предметы", "🧮 Математика", "🇷🇺 Русский язык", "🎓 ОГЭ", "🏆 ЕГЭ",
        "📷 Решить по фото", "✅ Проверить решение — бесплатно", "📖 Учиться", "📝 Тренировка",
        "🧪 Вариант ОГЭ/ЕГЭ", "📉 Слабые темы", "💎 Premium",
        "🎒 Выбрать класс", "👤 Профиль", "⭐ Избранное", "📚 История",
        "🧹 Сбросить контекст", "ℹ️ Помощь"
    }:
        return
    if is_rate_limited(uid):
        await message.answer("⏳ Подожди секунду.")
        return

    pending, payload = get_pending(uid)

    if pending == "exam_answers":
        try:
            result_text, score, max_score = await check_exam_answers(uid, payload, text)
            finish_exam_session(int(payload.get("session_id")), score)
            set_pending(uid, None)
            await send_plain(message, result_text)
        except Exception as e:
            log_error(uid, "check_exam_answers", e)
            await message.answer("Не удалось проверить вариант. Попробуй ещё раз.")
        return

    if pending == "training_answer":
        try:
            result = await check_student_answer(uid, payload, text)
            low = result.lower()
            is_correct = "верно" in low and "неверно" not in low and "ошиб" not in low
            if is_correct:
                bump_stat(uid, "correct_checks")
            else:
                bump_stat(uid, "wrong_checks")
            if payload.get("topic"):
                update_topic_progress(
                    uid,
                    payload.get("subject", "math"),
                    payload.get("topic", "Общая тема"),
                    is_correct,
                )
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



@router.message(Command("refund"))
async def refund_cmd(message: Message):
    if message.from_user.id not in ADMIN_IDS:
        await message.answer("Команда недоступна.")
        return

    parts = (message.text or "").split()
    if len(parts) != 3 or not parts[1].isdigit():
        await message.answer(
            "Использование:\n/refund USER_ID TELEGRAM_PAYMENT_CHARGE_ID"
        )
        return

    user_id = int(parts[1])
    charge_id = parts[2]

    try:
        ok = await bot.refund_star_payment(
            user_id=user_id,
            telegram_payment_charge_id=charge_id,
        )
        if ok:
            with SessionLocal() as s:
                row = s.execute(
                    select(StarPayment).where(
                        StarPayment.telegram_payment_charge_id == charge_id
                    )
                ).scalar_one_or_none()
                if row:
                    row.status = "refunded"
                    s.commit()
            await message.answer(
                "✅ Stars возвращены.\n"
                "Важно: срок Premium автоматически не уменьшается; "
                "при необходимости скорректируй доступ вручную в БД."
            )
        else:
            await message.answer("Telegram не подтвердил возврат.")
    except Exception as e:
        log_error(message.from_user.id, "refund_star_payment", e)
        await message.answer(f"Ошибка возврата: {type(e).__name__}")


# -----------------------------
# WEB ADMIN
# -----------------------------
security = HTTPBasic()

def require_admin(credentials: HTTPBasicCredentials = Depends(security)):
    if not ADMIN_WEB_PASSWORD:
        raise HTTPException(status_code=503, detail="ADMIN_WEB_PASSWORD is not configured")
    good_user = secrets.compare_digest(credentials.username, ADMIN_WEB_USER)
    good_pass = secrets.compare_digest(credentials.password, ADMIN_WEB_PASSWORD)
    if not (good_user and good_pass):
        raise HTTPException(
            status_code=401,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Basic"},
        )
    return True

ADMIN_CSS = """
<style>
body{font-family:Inter,Arial,sans-serif;background:#0f172a;color:#e5e7eb;margin:0}
.wrap{max-width:1180px;margin:auto;padding:24px}
h1,h2{margin:0 0 16px}
nav{display:flex;gap:10px;flex-wrap:wrap;margin:16px 0 24px}
nav a{color:#e5e7eb;text-decoration:none;background:#1e293b;padding:10px 14px;border-radius:10px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px;margin-bottom:24px}
.card{background:#111827;border:1px solid #334155;border-radius:14px;padding:18px}
.big{font-size:28px;font-weight:700}
table{width:100%;border-collapse:collapse;background:#111827;border-radius:12px;overflow:hidden}
th,td{text-align:left;padding:10px;border-bottom:1px solid #263244;vertical-align:top}
th{background:#1e293b}
small,.muted{color:#94a3b8}
.ok{color:#86efac}.bad{color:#fca5a5}
a{color:#93c5fd}
</style>
"""

def admin_page(title: str, body: str) -> str:
    return f"""<!doctype html><html><head><meta charset='utf-8'>
    <meta name='viewport' content='width=device-width,initial-scale=1'>
    <title>{title}</title>{ADMIN_CSS}</head><body><div class='wrap'>
    <h1>MathSolver — {title}</h1>
    <nav>
      <a href='/admin'>Обзор</a>
      <a href='/admin/users'>Пользователи</a>
      <a href='/admin/errors'>Ошибки</a>
      <a href='/admin/topics'>Темы</a>
      <a href='/admin/exams'>Экзамены</a>
      <a href='/admin/payments'>Платежи</a>
    </nav>{body}</div></body></html>"""

@app.get("/admin", response_class=HTMLResponse)
async def admin_dashboard(_: bool = Depends(require_admin)):
    with SessionLocal() as s:
        users = s.execute(select(func.count(UserState.user_id))).scalar_one()
        solved = s.execute(select(func.sum(Stat.solved))).scalar() or 0
        photos = s.execute(select(func.sum(Stat.photos))).scalar() or 0
        errors = s.execute(select(func.count(ErrorLog.id))).scalar_one()
        exams = s.execute(select(func.count(ExamSession.id))).scalar_one()
        completed = s.execute(
            select(func.count(ExamSession.id)).where(ExamSession.completed == True)
        ).scalar_one()
        payment_count = s.execute(
            select(func.count(StarPayment.id)).where(StarPayment.status == "paid")
        ).scalar_one()
        stars_total = s.execute(
            select(func.sum(StarPayment.stars)).where(StarPayment.status == "paid")
        ).scalar() or 0
    body = f"""
    <div class='cards'>
      <div class='card'><div class='muted'>Пользователи</div><div class='big'>{users}</div></div>
      <div class='card'><div class='muted'>Решено задач</div><div class='big'>{solved}</div></div>
      <div class='card'><div class='muted'>Фото</div><div class='big'>{photos}</div></div>
      <div class='card'><div class='muted'>Ошибки</div><div class='big'>{errors}</div></div>
      <div class='card'><div class='muted'>Варианты</div><div class='big'>{completed}/{exams}</div></div>
      <div class='card'><div class='muted'>Платежи</div><div class='big'>{payment_count}</div></div>
      <div class='card'><div class='muted'>Получено Stars</div><div class='big'>{stars_total} ⭐</div></div>
      <div class='card'><div class='muted'>Версия</div><div class='big'>{APP_VERSION}</div></div>
    </div>
    <div class='card'>
      <b>Состояние:</b>
      Groq: <span class='{"ok" if groq else "bad"}'>{"ON" if groq else "OFF"}</span> ·
      Fallback: <span class='{"ok" if FALLBACK_API_KEY else "muted"}'>{"ON" if FALLBACK_API_KEY else "OFF"}</span> ·
      DB: {"PostgreSQL" if DATABASE_URL.startswith(("postgres", "postgresql")) else "SQLite"}
    </div>
    """
    return HTMLResponse(admin_page("Админка", body))

@app.get("/admin/users", response_class=HTMLResponse)
async def admin_users(_: bool = Depends(require_admin)):
    with SessionLocal() as s:
        rows = s.execute(
            select(UserState, Stat)
            .join(Stat, Stat.user_id == UserState.user_id)
            .order_by(UserState.user_id.desc())
            .limit(300)
        ).all()
    tr = []
    for st, stat in rows:
        tr.append(
            f"<tr><td>{st.user_id}</td><td>{st.grade}</td><td>{st.mode}</td>"
            f"<td>{stat.solved}</td><td>{stat.photos}</td>"
            f"<td>{stat.correct_checks}</td><td>{stat.wrong_checks}</td>"
            f"<td>{stat.ai_errors}</td></tr>"
        )
    body = """<table><tr><th>User ID</th><th>Класс</th><th>Режим</th>
    <th>Решено</th><th>Фото</th><th>Верно</th><th>Ошибки ученика</th><th>AI-сбои</th></tr>""" + "".join(tr) + "</table>"
    return HTMLResponse(admin_page("Пользователи", body))

@app.get("/admin/errors", response_class=HTMLResponse)
async def admin_errors(_: bool = Depends(require_admin)):
    with SessionLocal() as s:
        rows = s.execute(
            select(ErrorLog).order_by(ErrorLog.id.desc()).limit(250)
        ).scalars().all()
    tr = []
    for e in rows:
        tr.append(
            f"<tr><td>{e.id}</td><td>{e.user_id or ''}</td><td>{e.source}</td>"
            f"<td>{html.escape(str(e.error_type))}</td><td>{html.escape(clean_text(e.message)[:500])}</td>"
            f"<td>{e.created_at or ''}</td></tr>"
        )
    body = """<table><tr><th>ID</th><th>User</th><th>Источник</th>
    <th>Тип</th><th>Сообщение</th><th>Время</th></tr>""" + "".join(tr) + "</table>"
    return HTMLResponse(admin_page("Ошибки", body))

@app.get("/admin/topics", response_class=HTMLResponse)
async def admin_topics(_: bool = Depends(require_admin)):
    with SessionLocal() as s:
        rows = s.execute(
            select(
                TopicProgress.subject,
                TopicProgress.topic,
                func.sum(TopicProgress.attempts),
                func.sum(TopicProgress.correct),
                func.sum(TopicProgress.wrong),
            )
            .group_by(TopicProgress.subject, TopicProgress.topic)
            .order_by(func.sum(TopicProgress.attempts).desc())
            .limit(200)
        ).all()
    tr = []
    for subject, topic, attempts, correct, wrong in rows:
        accuracy = round(100 * (correct or 0) / max(attempts or 1, 1))
        tr.append(
            f"<tr><td>{html.escape(str(subject))}</td><td>{html.escape(str(topic))}</td><td>{attempts}</td>"
            f"<td>{correct}</td><td>{wrong}</td><td>{accuracy}%</td></tr>"
        )
    body = """<table><tr><th>Предмет</th><th>Тема</th><th>Попытки</th>
    <th>Верно</th><th>Ошибки</th><th>Точность</th></tr>""" + "".join(tr) + "</table>"
    return HTMLResponse(admin_page("Темы", body))

@app.get("/admin/exams", response_class=HTMLResponse)
async def admin_exams(_: bool = Depends(require_admin)):
    with SessionLocal() as s:
        rows = s.execute(
            select(ExamSession).order_by(ExamSession.id.desc()).limit(250)
        ).scalars().all()
    tr = []
    for e in rows:
        tr.append(
            f"<tr><td>{e.id}</td><td>{e.user_id}</td><td>{e.exam_type.upper()}</td>"
            f"<td>{e.subject}</td><td>{e.score}/{e.max_score}</td>"
            f"<td>{'да' if e.completed else 'нет'}</td><td>{e.created_at or ''}</td></tr>"
        )
    body = """<table><tr><th>ID</th><th>User</th><th>Экзамен</th>
    <th>Предмет</th><th>Баллы</th><th>Завершён</th><th>Время</th></tr>""" + "".join(tr) + "</table>"
    return HTMLResponse(admin_page("Экзамены", body))



@app.get("/admin/payments", response_class=HTMLResponse)
async def admin_payments(_: bool = Depends(require_admin)):
    with SessionLocal() as s:
        rows = s.execute(
            select(StarPayment).order_by(StarPayment.id.desc()).limit(300)
        ).scalars().all()
    tr = []
    for p in rows:
        tr.append(
            f"<tr><td>{p.id}</td><td>{p.user_id}</td><td>{html.escape(p.plan_key)}</td>"
            f"<td>{p.stars} ⭐</td><td>{html.escape(p.status)}</td>"
            f"<td>{html.escape(p.telegram_payment_charge_id[:80])}</td>"
            f"<td>{p.created_at or ''}</td></tr>"
        )
    body = """<table><tr><th>ID</th><th>User</th><th>Тариф</th>
    <th>Stars</th><th>Статус</th><th>Telegram charge</th><th>Время</th></tr>""" + "".join(tr) + "</table>"
    return HTMLResponse(admin_page("Платежи", body))


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
        "database": "postgres" if DATABASE_URL.startswith(("postgres", "postgresql")) else "sqlite",
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
        "database": "postgres" if DATABASE_URL.startswith(("postgres", "postgresql")) else "sqlite",
        "admin_enabled": bool(ADMIN_WEB_PASSWORD),
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
