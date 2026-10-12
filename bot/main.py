
import asyncio, base64, io, math, os, re, secrets
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import pytesseract
from PIL import Image
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request
from groq import Groq
from sympy import Eq, Symbol, diff, integrate, pi, simplify, solve, sqrt, E
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations,
    implicit_multiplication_application, convert_xor,
)

from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    BufferedInputFile, CallbackQuery, InlineKeyboardButton,
    InlineKeyboardMarkup, KeyboardButton, Message,
    ReplyKeyboardMarkup, Update,
)

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
BASE_URL = os.getenv("BASE_URL", "").strip().rstrip("/")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_TEXT_MODEL = os.getenv("GROQ_TEXT_MODEL", "openai/gpt-oss-120b").strip()
GROQ_VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "qwen/qwen3.8-27b").strip()

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
app = FastAPI(title="MathSolver 9-11 Groq")

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

photo_cache: Dict[int, bytes] = {}
mode_cache: Dict[int, str] = {}

x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

MAIN_KB = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="🧮 Математика"), KeyboardButton(text="🇷🇺 Русский язык")],
        [KeyboardButton(text="🎓 ОГЭ"), KeyboardButton(text="🏆 ЕГЭ")],
        [KeyboardButton(text="📷 Решить по фото"), KeyboardButton(text="🤖 ИИ-режим")],
        [KeyboardButton(text="📚 Темы"), KeyboardButton(text="ℹ️ Помощь")],
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

SYSTEM = """
Ты — MathSolver, учебный помощник для 9–11 классов России по математике и русскому языку.

МАТЕМАТИКА:
алгебраические выражения; степени и корни; дроби; линейные, квадратные,
рациональные, иррациональные, показательные, логарифмические и
тригонометрические уравнения и неравенства; системы; модули; функции и графики;
прогрессии; тригонометрия; производная; касательная; экстремумы; интеграл;
планиметрия; стереометрия; координаты и векторы; вероятность; статистика;
комбинаторика; текстовые, экономические задачи и параметры; ОГЭ/ЕГЭ.

РУССКИЙ:
орфоэпия; лексика; паронимы; фразеология; морфемика и словообразование;
орфография; Н/НН; НЕ/НИ; -ТСЯ/-ТЬСЯ; приставки, корни, суффиксы и окончания;
морфология; синтаксис; грамматическая основа; однородные и обособленные члены;
вводные слова; обращения; ССП, СПП, БСП; пунктуация; текст, стили и типы речи;
средства выразительности; сочинение; задания ОГЭ/ЕГЭ.

Правила ответа:
1) Определи предмет и тему.
2) Объясняй по шагам простым школьным языком.
3) В математике используй точный локальный результат, если он дан ниже.
4) Показывай формулы и короткую проверку.
5) В русском называй правило и объясняй выбор.
6) Не выдумывай данные, которых нет в условии.
7) Если фото/OCR неоднозначны — скажи, что именно не читается.
8) В конце экзаменационной задачи пиши «Ответ: ...».
"""

def set_mode(uid:int, mode:str): mode_cache[uid] = mode
def get_mode(uid:int)->str: return mode_cache.get(uid, "auto")

def split_text(text:str, limit:int=3900):
    out=[]
    while len(text)>limit:
        cut=text.rfind("\n",0,limit)
        if cut<limit//2: cut=limit
        out.append(text[:cut]); text=text[cut:].lstrip()
    if text: out.append(text)
    return out

def normalize_math(s:str)->str:
    s=(s.replace("×","*").replace("·","*").replace("÷","/")
         .replace("−","-").replace(",",".").replace("π","pi"))
    s=re.sub(r"√\s*\(([^)]+)\)", r"sqrt(\1)", s)
    s=re.sub(r"√\s*([0-9A-Za-z.]+)", r"sqrt(\1)", s)
    return s

def pexpr(s:str):
    return parse_expr(
        normalize_math(s),
        transformations=TRANSFORMS,
        local_dict={"x":x,"y":y,"z":z,"sqrt":sqrt,"pi":pi,"e":E}
    )

def exact_math_hint(text:str)->Optional[str]:
    s=normalize_math(text.strip())
    low=text.lower()

    # systems separated by ;
    if ";" in s and s.count("=")>=2:
        try:
            eqs=[]
            for part in s.split(";"):
                if "=" in part:
                    a,b=part.split("=",1)
                    eqs.append(Eq(pexpr(a),pexpr(b)))
            vars_used=sorted(set().union(*(e.free_symbols for e in eqs)), key=lambda q:q.name)
            ans=solve(eqs, vars_used, dict=True)
            if ans:
                return "Точное решение системы: " + "; ".join(map(str, ans))
        except Exception:
            pass

    # derivative
    m=re.search(r"(?:производн\w*|d/dx)\s*[:=]?\s*(.+)",low)
    if m:
        try:
            return f"Точная производная: {diff(pexpr(m.group(1)),x)}"
        except Exception: pass

    # antiderivative
    m=re.search(r"(?:интеграл|первообразн\w*)\s*[:=]?\s*(.+)",low)
    if m:
        try:
            return f"Первообразная: {integrate(pexpr(m.group(1)),x)} + C"
        except Exception: pass

    # one equation
    if s.count("=")==1:
        try:
            a,b=s.split("=",1)
            lhs,rhs=pexpr(a),pexpr(b)
            vars_used=sorted((lhs-rhs).free_symbols,key=lambda q:q.name)
            if vars_used:
                v=vars_used[0]
                roots=solve(Eq(lhs,rhs),v)
                if roots:
                    return "Точное решение: " + ", ".join(f"{v} = {r}" for r in roots)
        except Exception:
            pass

    # plain numeric expression
    try:
        expr=pexpr(s)
        if not expr.free_symbols:
            val=simplify(expr)
            return f"Точное значение: {val}; приближённо: {val.evalf(12)}"
    except Exception:
        pass

    return None

def preprocess_for_ocr(data:bytes):
    arr=np.frombuffer(data,dtype=np.uint8)
    img=cv2.imdecode(arr,cv2.IMREAD_GRAYSCALE)
    if img is None: return []
    h,w=img.shape[:2]
    scale=max(1.0,min(3.0,2000/max(h,w)))
    if scale>1.05:
        img=cv2.resize(img,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
    img=cv2.fastNlMeansDenoising(img,None,10,7,21)
    clahe=cv2.createCLAHE(clipLimit=2.2,tileGridSize=(8,8))
    c=clahe.apply(img)
    a=cv2.adaptiveThreshold(c,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,35,11)
    _,o=cv2.threshold(c,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return [c,a,o]

def ocr_image(data:bytes)->str:
    candidates=[]
    for img in preprocess_for_ocr(data):
        pil=Image.fromarray(img)
        for psm in (6,11):
            try:
                t=pytesseract.image_to_string(pil,lang="rus+eng",
                    config=f"--oem 3 --psm {psm}").strip()
                if t:
                    score=sum(ch.isalnum() for ch in t)
                    candidates.append((score,t))
            except Exception: pass
    return max(candidates,key=lambda q:q[0])[1] if candidates else ""

def crop_image(data:bytes,preset:str)->bytes:
    im=Image.open(io.BytesIO(data)).convert("RGB")
    w,h=im.size
    boxes={
        "top":(0,0,w,int(h*.58)),
        "center":(0,int(h*.18),w,int(h*.82)),
        "bottom":(0,int(h*.42),w,h),
        "left":(0,0,int(w*.62),h),
        "right":(int(w*.38),0,w,h),
        "full":(0,0,w,h)
    }
    out=im.crop(boxes.get(preset,boxes["full"]))
    b=io.BytesIO(); out.save(b,"JPEG",quality=94)
    return b.getvalue()

def mode_prompt(uid:int)->str:
    return {
        "math":"Режим: математика.",
        "russian":"Режим: русский язык.",
        "oge":"Режим: ОГЭ. Оформляй как подготовку к ОГЭ.",
        "ege":"Режим: ЕГЭ. Оформляй как подготовку к ЕГЭ.",
        "ai":"Режим: ИИ. Сам определи предмет и реши максимально точно.",
        "auto":"Сам определи предмет и тему."
    }.get(get_mode(uid),"Сам определи предмет и тему.")

async def groq_text(uid:int,text:str,hint:Optional[str])->Optional[str]:
    if not groq_client: return None
    user = mode_prompt(uid) + "\n\nЗадание:\n" + text
    if hint:
        user += "\n\nПроверенный локальный символьный результат:\n" + hint

    def call():
        r=groq_client.chat.completions.create(
            model=GROQ_TEXT_MODEL,
            messages=[
                {"role":"system","content":SYSTEM},
                {"role":"user","content":user}
            ],
            temperature=0.2,
            max_completion_tokens=2800,
            reasoning_effort="medium",
        )
        return r.choices[0].message.content or ""

    try:
        return (await asyncio.to_thread(call)).strip()
    except Exception as e:
        return f"⚠️ Groq временно недоступен: {type(e).__name__}"

async def groq_image(uid:int,img:bytes,ocr:str)->Optional[str]:
    if not groq_client: return None
    b64=base64.b64encode(img).decode()
    prompt=(
        SYSTEM+"\n\n"+mode_prompt(uid)+
        "\nНа фото школьное задание. Прочитай само изображение. "
        "OCR ниже только подсказка и может ошибаться. Учитывай формулы, графики, "
        "таблицы, рисунки и геометрию. Реши по шагам.\n\nOCR:\n"+
        (ocr or "(пусто)")
    )

    def call():
        r=groq_client.chat.completions.create(
            model=GROQ_VISION_MODEL,
            messages=[{
                "role":"user",
                "content":[
                    {"type":"text","text":prompt},
                    {"type":"image_url","image_url":{
                        "url":f"data:image/jpeg;base64,{b64}"
                    }}
                ]
            }],
            temperature=0.2,
            max_completion_tokens=2800,
            reasoning_effort="medium",
        )
        return r.choices[0].message.content or ""

    try:
        return (await asyncio.to_thread(call)).strip()
    except Exception as e:
        return f"⚠️ Анализ фото Groq временно недоступен: {type(e).__name__}"

def local_russian(text:str)->str:
    low=text.lower()
    stress={
        "звонит":"звонИт","позвонит":"позвонИт","баловать":"баловАть",
        "красивее":"красИвее","торты":"тОрты","договор":"договОр",
        "каталог":"каталОг","километр":"киломЕтр","жалюзи":"жалюзИ"
    }
    w=re.sub(r"[^а-яё-]","",low)
    if w in stress: return "🇷🇺 Правильно: "+stress[w]
    notes=[]
    if "тся" in low or "ться" in low:
        notes.append("–ТСЯ/–ТЬСЯ: задай вопрос «что делает?» или «что делать?».")
    if "нн" in low:
        notes.append("Н/НН: определи часть речи и способ образования слова.")
    if any(k in low for k in ["конечно","возможно","например","следовательно"]):
        notes.append("Вводное слово обычно обособляется запятыми.")
    if not notes:
        notes.append("Для полного разбора этого задания лучше включить Groq AI.")
    return "🇷🇺 Локальный разбор:\n• "+"\n• ".join(notes)

async def solve_text(uid:int,text:str)->str:
    hint=exact_math_hint(text)
    if groq_client:
        ans=await groq_text(uid,text,hint)
        if ans: return ans

    if get_mode(uid)=="russian" or any(k in text.lower() for k in
        ["запят","орф","удар","пунктуац","предлож","нн","тся","ться"]):
        return local_russian(text)
    if hint:
        return "🧮 "+hint+"\n\n🤖 Groq AI не подключён."
    return "Не удалось уверенно решить локально. Подключи GROQ_API_KEY для ИИ-решения."

@router.message(CommandStart())
async def start(message:Message):
    status="✅ подключён" if groq_client else "⚪ не подключён"
    await message.answer(
        "📚 MathSolver 9–11 + Groq\n\n"
        "Математика, русский, ОГЭ/ЕГЭ, точные вычисления и решение по фото.\n"
        f"🤖 Groq AI: {status}",
        reply_markup=MAIN_KB
    )

@router.message(Command("help"))
@router.message(F.text=="ℹ️ Помощь")
async def help_cmd(message:Message):
    await message.answer(
        "Отправь задачу текстом или фото.\n\n"
        "Примеры:\n"
        "• 2x + 5 = 17\n"
        "• x+y=5; x-y=1\n"
        "• производная x^3+2x\n"
        "• интеграл 3x^2\n\n"
        "Для фото бот сначала улучшает изображение и OCR, затем Groq анализирует само фото."
    )

@router.message(F.text=="📚 Темы")
async def topics(message:Message):
    await message.answer(
        "📚 9–11 классы:\n\n"
        "🧮 Алгебра, уравнения/неравенства, функции, прогрессии, тригонометрия, "
        "производная, интеграл, планиметрия, стереометрия, векторы, вероятность, "
        "статистика, текстовые и экономические задачи, параметры, ОГЭ/ЕГЭ.\n\n"
        "🇷🇺 Орфография, орфоэпия, лексика, морфемика, морфология, синтаксис, "
        "пунктуация, сложные предложения, текст, выразительность, сочинение, ОГЭ/ЕГЭ."
    )

@router.message(F.text=="🧮 Математика")
async def m1(message:Message):
    set_mode(message.from_user.id,"math"); await message.answer("🧮 Режим математики включён.")

@router.message(F.text=="🇷🇺 Русский язык")
async def m2(message:Message):
    set_mode(message.from_user.id,"russian"); await message.answer("🇷🇺 Режим русского включён.")

@router.message(F.text=="🎓 ОГЭ")
async def m3(message:Message):
    set_mode(message.from_user.id,"oge"); await message.answer("🎓 Режим ОГЭ включён.")

@router.message(F.text=="🏆 ЕГЭ")
async def m4(message:Message):
    set_mode(message.from_user.id,"ege"); await message.answer("🏆 Режим ЕГЭ включён.")

@router.message(F.text=="🤖 ИИ-режим")
async def m5(message:Message):
    set_mode(message.from_user.id,"ai")
    await message.answer(
        f"🤖 Groq AI {'включён' if groq_client else 'не подключён'}. "
        + (f"Модель: {GROQ_TEXT_MODEL}" if groq_client else "Добавь GROQ_API_KEY в Render.")
    )

@router.message(F.text=="📷 Решить по фото")
async def ph(message:Message):
    await message.answer("📷 Отправь фото задания. Лучше без бликов и под прямым углом.")

@router.message(F.photo)
async def photo(message:Message):
    p=message.photo[-1]
    f=await bot.get_file(p.file_id)
    b=io.BytesIO()
    await bot.download_file(f.file_path,destination=b)
    photo_cache[message.from_user.id]=b.getvalue()
    await message.answer_photo(
        BufferedInputFile(b.getvalue(),filename="task.jpg"),
        caption="✂️ Выбери область задания:",
        reply_markup=CROP_KB
    )

@router.callback_query(F.data.startswith("crop:"))
async def crop_cb(callback:CallbackQuery):
    data=photo_cache.get(callback.from_user.id)
    if not data:
        await callback.answer("Фото не найдено.",show_alert=True); return
    preset=callback.data.split(":",1)[1]
    cropped=crop_image(data,preset)
    await callback.answer("Распознаю и решаю…")
    ocr=await asyncio.to_thread(ocr_image,cropped)

    if groq_client:
        ans=await groq_image(callback.from_user.id,cropped,ocr)
        if ocr:
            await callback.message.answer("🔎 OCR-подсказка:\n\n"+ocr[:2200])
        for part in split_text(ans or "Не удалось получить ответ."):
            await callback.message.answer(part)
        return

    if not ocr:
        await callback.message.answer("Не удалось распознать фото. Попробуй более чёткий снимок.")
        return

    await callback.message.answer("🔎 Распознано:\n\n"+ocr[:3000])
    ans=await solve_text(callback.from_user.id,ocr)
    for part in split_text(ans):
        await callback.message.answer(part)

@router.message(F.text)
async def text_msg(message:Message):
    if message.text.startswith("/"): return
    ans=await solve_text(message.from_user.id,message.text.strip())
    for part in split_text(ans):
        await message.answer(part)

@app.get("/")
async def root():
    return {
        "ok":True,
        "service":"MathSolver 9-11 Groq",
        "ai_enabled":bool(groq_client),
        "text_model":GROQ_TEXT_MODEL if groq_client else None,
        "vision_model":GROQ_VISION_MODEL if groq_client else None,
    }

@app.get("/health")
async def health():
    return {"ok":True,"ai_enabled":bool(groq_client)}

@app.on_event("startup")
async def startup():
    await bot.set_webhook(
        f"{BASE_URL}/telegram/{WEBHOOK_SECRET}",
        secret_token=WEBHOOK_SECRET,
        drop_pending_updates=False,
    )

@app.on_event("shutdown")
async def shutdown():
    await bot.session.close()

@app.post("/telegram/{secret}")
async def webhook(
    secret:str,
    request:Request,
    x_telegram_bot_api_secret_token: str|None = Header(default=None),
):
    if secret!=WEBHOOK_SECRET or x_telegram_bot_api_secret_token!=WEBHOOK_SECRET:
        raise HTTPException(status_code=403,detail="bad secret")
    data=await request.json()
    update=Update.model_validate(data,context={"bot":bot})
    await dp.feed_update(bot,update)
    return {"ok":True}
