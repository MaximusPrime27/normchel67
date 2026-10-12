import asyncio, base64, io, logging, os, re, secrets, time
from collections import defaultdict, deque
import cv2, numpy as np, pytesseract
from PIL import Image
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request
from groq import AsyncGroq, APIConnectionError, APIStatusError, RateLimitError
from sympy import Eq, Symbol, diff, integrate, pi, simplify, solve, sqrt, E
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application, convert_xor
from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import BufferedInputFile, InlineKeyboardButton, InlineKeyboardMarkup, KeyboardButton, ReplyKeyboardMarkup, Update

load_dotenv()
APP_VERSION = "Stable V5 Pro"
BOT_TOKEN=os.getenv('BOT_TOKEN','').strip(); BASE_URL=os.getenv('BASE_URL','').strip().rstrip('/'); WEBHOOK_SECRET=os.getenv('WEBHOOK_SECRET','').strip(); GROQ_API_KEY=os.getenv('GROQ_API_KEY','').strip()
TEXT_MODEL=os.getenv('GROQ_TEXT_MODEL','openai/gpt-oss-120b').strip(); VISION_MODEL=os.getenv('GROQ_VISION_MODEL','qwen/qwen3.8-27b').strip()
TIMEOUT=float(os.getenv('REQUEST_TIMEOUT','55')); MAX_CONCURRENT=int(os.getenv('MAX_CONCURRENT_AI','4')); COOLDOWN=float(os.getenv('USER_COOLDOWN_SECONDS','1.2')); MAX_TOKENS=int(os.getenv('MAX_OUTPUT_TOKENS','2600'))
if not BOT_TOKEN: raise RuntimeError('BOT_TOKEN is not set')
if not BASE_URL: raise RuntimeError('BASE_URL is not set')
if not WEBHOOK_SECRET: WEBHOOK_SECRET=secrets.token_urlsafe(32)
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s'); log=logging.getLogger('mathsolver')
bot=Bot(BOT_TOKEN); dp=Dispatcher(); router=Router(); dp.include_router(router); app=FastAPI(title='MathSolver Stable V3 Clean')
groq=AsyncGroq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None; sem=asyncio.Semaphore(MAX_CONCURRENT)
mode=defaultdict(lambda:'auto'); grade=defaultdict(lambda:11); photos={}; history=defaultdict(lambda:deque(maxlen=8)); last=defaultdict(float)
x,y,z=Symbol('x'),Symbol('y'),Symbol('z'); TR=standard_transformations+(implicit_multiplication_application,convert_xor)

MAIN=ReplyKeyboardMarkup(keyboard=[[KeyboardButton(text='🧮 Математика'),KeyboardButton(text='🇷🇺 Русский язык')],[KeyboardButton(text='🎓 ОГЭ'),KeyboardButton(text='🏆 ЕГЭ')],[KeyboardButton(text='📷 Решить по фото'),KeyboardButton(text='🤖 ИИ-режим')],[KeyboardButton(text='🎒 Выбрать класс'),KeyboardButton(text='📚 Темы')],[KeyboardButton(text='🧹 Сбросить контекст'),KeyboardButton(text='ℹ️ Помощь')]],resize_keyboard=True)
CLASS=InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text='9 класс',callback_data='class:9'),InlineKeyboardButton(text='10 класс',callback_data='class:10'),InlineKeyboardButton(text='11 класс',callback_data='class:11')]])
CROP=InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text='⬆ Верх',callback_data='crop:top'),InlineKeyboardButton(text='🎯 Центр',callback_data='crop:center'),InlineKeyboardButton(text='⬇ Низ',callback_data='crop:bottom')],[InlineKeyboardButton(text='⬅ Левая',callback_data='crop:left'),InlineKeyboardButton(text='➡ Правая',callback_data='crop:right')],[InlineKeyboardButton(text='🖼 Всё фото',callback_data='crop:full')]])
PROMPT="""Ты MathSolver, учебный помощник 9–11 классов России.

Математика: алгебра, уравнения и неравенства, системы, функции и графики,
прогрессии, тригонометрия, производная, интеграл, планиметрия, стереометрия,
координаты, векторы, вероятность, статистика, текстовые и экономические задачи,
параметры, ОГЭ/ЕГЭ.

Русский: орфоэпия, лексика, паронимы, морфемика, словообразование, орфография,
Н/НН, НЕ/НИ, -ТСЯ/-ТЬСЯ, морфология, синтаксис, ССП/СПП/БСП, пунктуация,
текст, стили речи, выразительные средства, сочинение, ОГЭ/ЕГЭ.

ФОРМАТ ОТВЕТА ДЛЯ TELEGRAM:
- Только обычный читаемый текст.
- НИКОГДА не используй Markdown-разметку: #, ##, ###, **, __, ``` и таблицы Markdown.
- НИКОГДА не используй LaTeX: запрещены \frac, \sqrt, \cdot, \pm, \[, \], \(, \), $...$.
- Пиши математические символы непосредственно: √, ², ³, ·, ÷, ±, ≤, ≥, ≠, π, Δ.
- Дроби записывай как (числитель)/(знаменатель), например (5+√7)/4.
- Степени желательно писать как x², x³; сложные степени допустимо x^(n+1).
- Не пиши слова «Шаг 1», «Шаг 2» без необходимости.
- Для обычной задачи формат:
  Решение:
  1. ...
  2. ...
  Ответ: ...
- Для простого примера можно ещё короче: вычисление → ответ.
- Для русского:
  Правило:
  ...
  Разбор:
  ...
  Ответ: ...
- Не повторяй условие целиком, если это не нужно.
- Не растягивай ответ: достаточно, чтобы ученик понял ход решения.
- Используй точный результат SymPy, если он передан.
- Не выдумывай данные.
- Для геометрии называй нужные теоремы.
- Для ОГЭ/ЕГЭ обязательно заканчивай строкой «Ответ: ...».
"""

def split_text(t,n=3900):
    out=[]
    while len(t)>n:
        k=t.rfind('\n',0,n); k=k if k>n//2 else n; out.append(t[:k]); t=t[k:].lstrip()
    if t: out.append(t)
    return out


def clean_telegram_text(text):
    """Convert common Markdown/LaTeX AI output into clean Telegram plain text."""
    if not text:
        return text

    s = str(text).replace("\r\n", "\n").replace("\r", "\n")

    # Code fences and markdown headings/emphasis.
    s = re.sub(r"```[a-zA-Z0-9_-]*\n?", "", s)
    s = s.replace("```", "")
    s = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", s)
    s = s.replace("**", "").replace("__", "").replace("~~", "")
    s = re.sub(r"(?m)^\s*---+\s*$", "", s)

    # Display/inline math delimiters.
    for token in (r"\[", r"\]", r"\(", r"\)"):
        s = s.replace(token, "")
    s = s.replace("$", "")

    # Common LaTeX operators and symbols.
    latex_map = {
        r"\cdot": "·",
        r"\times": "×",
        r"\div": "÷",
        r"\pm": "±",
        r"\mp": "∓",
        r"\leq": "≤",
        r"\le": "≤",
        r"\geq": "≥",
        r"\ge": "≥",
        r"\neq": "≠",
        r"\approx": "≈",
        r"\pi": "π",
        r"\Delta": "Δ",
        r"\infty": "∞",
        r"\rightarrow": "→",
        r"\Rightarrow": "⇒",
        r"\left": "",
        r"\right": "",
        r"\,": " ",
        r"\;": " ",
        r"\!": "",
    }
    for a, b in latex_map.items():
        s = s.replace(a, b)

    # Fractions. Repeat to handle several simple occurrences.
    frac_pat = re.compile(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
    for _ in range(8):
        new_s = frac_pat.sub(lambda m: f"({m.group(1)})/({m.group(2)})", s)
        if new_s == s:
            break
        s = new_s

    # Square roots.
    sqrt_pat = re.compile(r"\\sqrt\s*\{([^{}]+)\}")
    for _ in range(5):
        new_s = sqrt_pat.sub(lambda m: f"√({m.group(1)})", s)
        if new_s == s:
            break
        s = new_s

    # Superscripts/subscripts for common digits.
    super_map = str.maketrans("0123456789+-=()", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾")
    sub_map = str.maketrans("0123456789+-=()", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎")
    s = re.sub(
        r"\^\{([0-9+\-=()]+)\}",
        lambda m: m.group(1).translate(super_map),
        s,
    )
    s = re.sub(
        r"_\{([0-9+\-=(),]+)\}",
        lambda m: m.group(1).translate(sub_map),
        s,
    )
    s = re.sub(
        r"\^([23])\b",
        lambda m: {"2": "²", "3": "³"}[m.group(1)],
        s,
    )

    # Other frequent LaTeX commands.
    s = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\s*\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\operatorname\s*\{([^{}]*)\}", r"\1", s)

    # Any remaining braces are more readable without TeX grouping.
    s = s.replace("{", "(").replace("}", ")")

    # Remove leftover backslash before command-like fragments.
    s = re.sub(r"\\([A-Za-z]+)", r"\1", s)

    # Clean bullet markers generated by markdown, but keep readable list structure.
    s = re.sub(r"(?m)^\s*[-*]\s+", "• ", s)

    # Make common heading labels neat.
    s = re.sub(r"(?mi)^\s*решение\s*:?\s*$", "Решение:", s)
    s = re.sub(r"(?mi)^\s*ответ\s*:?\s*", "Ответ: ", s)
    s = re.sub(r"(?mi)^\s*проверка\s*:?\s*$", "Проверка:", s)
    s = re.sub(r"(?mi)^\s*правило\s*:?\s*$", "Правило:", s)
    s = re.sub(r"(?mi)^\s*разбор\s*:?\s*$", "Разбор:", s)

    # Common indexed variables: x_{1,2} -> x₁,₂ and x_1 -> x₁.
    digit_sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    s = re.sub(
        r"_\{([0-9,;]+)\}",
        lambda m: "".join(ch.translate(digit_sub) if ch.isdigit() else ch for ch in m.group(1)),
        s,
    )
    s = re.sub(
        r"_([0-9])",
        lambda m: m.group(1).translate(digit_sub),
        s,
    )

    # Convert simple TeX commands that may remain after malformed model output.
    s = s.replace("\\\\", "")
    s = re.sub(r"(?m)^\s*#{1,6}\s*", "", s)

    # Remove excessive blank lines/spaces.
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ \t]{2,}", " ", s)

    return s.strip()


async def send_plain(message, text):
    """Absolute final output barrier: nothing reaches Telegram before cleanup."""
    cleaned = clean_telegram_text(text or "")
    if not cleaned:
        cleaned = "Не удалось сформировать ответ."
    for chunk in split_text(cleaned):
        await message.answer(chunk)


def norm(s):
    s=s.replace('×','*').replace('·','*').replace('÷','/').replace('−','-').replace(',','.').replace('π','pi'); s=re.sub(r'√\s*\(([^)]+)\)',r'sqrt(\1)',s); s=re.sub(r'√\s*([0-9A-Za-z.]+)',r'sqrt(\1)',s); return s

def expr(s): return parse_expr(norm(s),transformations=TR,local_dict={'x':x,'y':y,'z':z,'sqrt':sqrt,'pi':pi,'e':E})

def exact_hint(text):
    s=norm(text.strip()); low=text.lower()
    if ';' in s and s.count('=')>=2:
        try:
            eqs=[]
            for p in s.split(';'):
                if '=' in p:
                    a,b=p.split('=',1); eqs.append(Eq(expr(a),expr(b)))
            vs=sorted(set().union(*(e.free_symbols for e in eqs)),key=lambda v:v.name); ans=solve(eqs,vs,dict=True)
            if ans:return 'Точное решение системы: '+'; '.join(map(str,ans))
        except Exception: pass
    m=re.search(r'(?:производн\w*|d/dx)\s*[:=]?\s*(.+)',low)
    if m:
        try:return f'Точная производная: {diff(expr(m.group(1)),x)}'
        except Exception:pass
    m=re.search(r'(?:интеграл|первообразн\w*)\s*[:=]?\s*(.+)',low)
    if m:
        try:return f'Первообразная: {integrate(expr(m.group(1)),x)} + C'
        except Exception:pass
    if s.count('=')==1:
        try:
            a,b=s.split('=',1); L,R=expr(a),expr(b); vs=sorted((L-R).free_symbols,key=lambda v:v.name)
            if vs:
                roots=solve(Eq(L,R),vs[0]);
                if roots:return 'Точное решение: '+', '.join(f'{vs[0]} = {r}' for r in roots)
        except Exception: pass
    try:
        e=expr(s)
        if not e.free_symbols:
            v=simplify(e); return f'Точное значение: {v}; приближённо: {v.evalf(12)}'
    except Exception: pass
    return None

def crop(data,preset):
    im=Image.open(io.BytesIO(data)).convert('RGB'); w,h=im.size; boxes={'top':(0,0,w,int(h*.58)),'center':(0,int(h*.18),w,int(h*.82)),'bottom':(0,int(h*.42),w,h),'left':(0,0,int(w*.62),h),'right':(int(w*.38),0,w,h),'full':(0,0,w,h)}; im=im.crop(boxes.get(preset,boxes['full'])); b=io.BytesIO(); im.save(b,'JPEG',quality=92); return b.getvalue()

def compress(data):
    im=Image.open(io.BytesIO(data)).convert('RGB'); im.thumbnail((1800,1800)); b=io.BytesIO(); im.save(b,'JPEG',quality=88,optimize=True); return b.getvalue()

def ocr(data):
    arr=np.frombuffer(data,np.uint8); im=cv2.imdecode(arr,cv2.IMREAD_GRAYSCALE)
    if im is None:return ''
    h,w=im.shape[:2]; scale=max(1.0,min(3.0,2000/max(h,w)))
    if scale>1.05: im=cv2.resize(im,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
    im=cv2.fastNlMeansDenoising(im,None,10,7,21); c=cv2.createCLAHE(2.2,(8,8)).apply(im); a=cv2.adaptiveThreshold(c,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,35,11); _,o=cv2.threshold(c,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cand=[]
    for v in (c,a,o):
        for psm in (6,11,12):
            try:
                t=pytesseract.image_to_string(Image.fromarray(v),lang='rus+eng',config=f'--oem 3 --psm {psm}').strip()
                if t:cand.append((sum(ch.isalnum() for ch in t),t))
            except Exception: pass
    return max(cand,key=lambda q:q[0])[1] if cand else ''

async def ai_call(messages,model):
    if not groq:return None
    async with sem:
        for i,delay in enumerate((0.8,1.8,3.5),1):
            try:
                r=await asyncio.wait_for(groq.chat.completions.create(model=model,messages=messages,temperature=.15,max_completion_tokens=MAX_TOKENS),timeout=TIMEOUT)
                t=(r.choices[0].message.content or '').strip()
                if t:return t
            except (RateLimitError,APIConnectionError,asyncio.TimeoutError): log.warning('temporary Groq error, attempt %s',i)
            except APIStatusError as e:
                log.warning('Groq status %s',getattr(e,'status_code','?'))
                if getattr(e,'status_code',500)<500 and getattr(e,'status_code',500)!=429:break
            except Exception: log.exception('Groq error'); break
            if i<3: await asyncio.sleep(delay)
    return None

async def solve_text(uid,text):
    hint=exact_hint(text); context=list(history[uid]); p=f'Режим: {mode[uid]}. Класс: {grade[uid]}. Формат: чистый текст Telegram, короткое понятное решение и ответ, без Markdown и LaTeX.\nЗадание: {text}'+(f'\nПроверенный локальный результат: {hint}' if hint else '')
    ans=await ai_call([{'role':'system','content':PROMPT}]+context+[{'role':'user','content':p}],TEXT_MODEL)
    if ans:
        ans=clean_telegram_text(ans)
        history[uid].append({'role':'user','content':text})
        history[uid].append({'role':'assistant','content':ans})
        return ans
    if hint:return clean_telegram_text('🧮 '+hint+'\n\n⚠️ ИИ временно недоступен, показан локальный точный результат.')
    return '⚠️ ИИ временно недоступен. Попробуй ещё раз через несколько секунд.'

async def solve_image(uid,data,text):
    b64=base64.b64encode(compress(data)).decode(); p=PROMPT+f'\nКласс: {grade[uid]}, режим: {mode[uid]}. Прочитай само фото. OCR ниже лишь подсказка и может ошибаться. Не выдумывай нечитабельные данные.\nOCR:\n{text or "(пусто)"}'
    ans=await ai_call([{'role':'user','content':[{'type':'text','text':p},{'type':'image_url','image_url':{'url':f'data:image/jpeg;base64,{b64}'}}]}],VISION_MODEL)
    return clean_telegram_text(ans) if ans else None

def limited(uid):
    now=time.monotonic(); yes=now-last[uid]<COOLDOWN; last[uid]=now; return yes


@router.message(F.text == "▶️ Старт")
async def start_button(message: Message):
    await start(message)

@router.message(CommandStart())
async def start(m): await m.answer(f'📚 MathSolver Stable V5 Pro\n🤖 AI: {"✅" if groq else "❌"}\n🎒 Класс: {grade[m.from_user.id]}',reply_markup=MAIN)
@router.message(Command('status'))
async def status(m):
    await m.answer(
        f'🟢 Бот работает\n'
        f'Версия: {APP_VERSION}\n'
        f'AI: {bool(groq)}\n'
        f'Класс: {grade[m.from_user.id]}\n'
        f'Режим: {mode[m.from_user.id]}\n'
        f'Контекст: {len(history[m.from_user.id])}'
    )
@router.message(Command('reset'))
@router.message(F.text=='🧹 Сбросить контекст')
async def reset(m): history[m.from_user.id].clear(); photos.pop(m.from_user.id,None); await m.answer('🧹 Контекст очищен.')
@router.message(Command('help'))
@router.message(F.text=='ℹ️ Помощь')
async def help_(m): await m.answer('Отправь задачу текстом или фото. /status — состояние, /reset — очистить контекст.')
@router.message(F.text=='🎒 Выбрать класс')
async def choose(m): await m.answer('Выбери класс:',reply_markup=CLASS)
@router.callback_query(F.data.startswith('class:'))
async def class_cb(c): grade[c.from_user.id]=int(c.data.split(':')[1]); history[c.from_user.id].clear(); await c.answer('Готово'); await c.message.answer(f'🎒 Выбран {grade[c.from_user.id]} класс.')
@router.message(F.text=='🧮 Математика')
async def math_mode(m): mode[m.from_user.id]='math'; history[m.from_user.id].clear(); await m.answer('🧮 Режим математики.')
@router.message(F.text=='🇷🇺 Русский язык')
async def rus_mode(m): mode[m.from_user.id]='russian'; history[m.from_user.id].clear(); await m.answer('🇷🇺 Режим русского языка.')
@router.message(F.text=='🎓 ОГЭ')
async def oge(m): mode[m.from_user.id]='oge'; history[m.from_user.id].clear(); await m.answer('🎓 Режим ОГЭ.')
@router.message(F.text=='🏆 ЕГЭ')
async def ege(m): mode[m.from_user.id]='ege'; history[m.from_user.id].clear(); await m.answer('🏆 Режим ЕГЭ.')
@router.message(F.text=='🤖 ИИ-режим')
async def aim(m): mode[m.from_user.id]='ai'; history[m.from_user.id].clear(); await m.answer('🤖 ИИ-режим включён.')
@router.message(F.text=='📚 Темы')
async def topics(m): await m.answer('📚 Покрытие: программа математики и русского 9–11 классов, ОГЭ и ЕГЭ.')
@router.message(F.text=='📷 Решить по фото')
async def photo_hint(m): await m.answer('📷 Отправь фото крупно, ровно и без бликов.')
@router.message(F.photo)
async def photo(m):
    uid=m.from_user.id
    if limited(uid):await m.answer('⏳ Подожди секунду.'); return
    f=await bot.get_file(m.photo[-1].file_id); b=io.BytesIO(); await bot.download_file(f.file_path,destination=b); photos[uid]=b.getvalue(); await m.answer_photo(BufferedInputFile(photos[uid],filename='task.jpg'),caption='✂️ Выбери область:',reply_markup=CROP)
@router.callback_query(F.data.startswith('crop:'))
async def crop_cb(c):
    uid=c.from_user.id; data=photos.get(uid)
    if not data:await c.answer('Фото не найдено',show_alert=True); return
    d=crop(data,c.data.split(':')[1]); await c.answer('Решаю…'); text=await asyncio.to_thread(ocr,d); ans=await solve_image(uid,d,text)
    if not ans and text: ans=await solve_text(uid,text)
    if text: await send_plain(c.message, '🔎 Распознано:\n' + text[:1800])
    await send_plain(c.message, ans or 'Не удалось прочитать фото. Попробуй более чёткий снимок.')
@router.message(F.text)
async def text(m):
    if m.text.startswith('/'):return
    if limited(m.from_user.id):await m.answer('⏳ Подожди секунду.'); return
    await send_plain(m, await solve_text(m.from_user.id,m.text.strip()))

@app.get('/')
async def root(): return {'ok':True,'service': APP_VERSION,'ai_enabled':bool(groq),'text_model':TEXT_MODEL if groq else None,'vision_model':VISION_MODEL if groq else None}
@app.get('/health')
async def health(): return {'ok':True,'ai_enabled':bool(groq)}
@app.get('/ready')
async def ready(): return {'ready':True,'ai_enabled':bool(groq)}
@app.on_event('startup')
async def startup(): await bot.set_webhook(f'{BASE_URL}/telegram/{WEBHOOK_SECRET}',secret_token=WEBHOOK_SECRET,allowed_updates=dp.resolve_used_update_types()); log.info('Webhook configured')
@app.on_event('shutdown')
async def shutdown(): await bot.session.close()
@app.post('/telegram/{secret}')
async def webhook(secret:str,request:Request,x_telegram_bot_api_secret_token:str|None=Header(default=None)):
    if secret!=WEBHOOK_SECRET or x_telegram_bot_api_secret_token!=WEBHOOK_SECRET: raise HTTPException(403,'bad secret')
    update=Update.model_validate(await request.json(),context={'bot':bot}); await dp.feed_update(bot,update); return {'ok':True}
