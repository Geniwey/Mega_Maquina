import os
import io
import re
import sys
import json
import time
import wave
import random
import asyncio
import logging
import unicodedata
import requests
import numpy as np
import pandas as pd
from groq import Groq
import edge_tts

# --- BLINDAJE ANTIALIAS PARA PIL ---
import PIL.Image
if not hasattr(PIL.Image, "ANTIALIAS"):
    try:
        PIL.Image.ANTIALIAS = PIL.Image.Resampling.LANCZOS
    except AttributeError:
        PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import (
    VideoFileClip, AudioFileClip, ImageClip, VideoClip, ColorClip,
    CompositeVideoClip, CompositeAudioClip, concatenate_videoclips,
)

# ---------------------------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------------------------
SHEET_ID = "10gJJCIlPzCHYEfPYPKgT3-xjUtghbIaYpdR87Da4JPQ"
CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv"
COLUMNAS_REQUERIDAS = ["NOMBRE_PRODUCTO", "PROBLEMAS_QUE_RESUELVE", "PALABRA_CLAVE_MANYCHAT", "ENLACE_HOTMART"]

ARCHIVO_JSON = "contenido_hoy.json"
ARCHIVO_VIDEO = "video_final.mp4"
ARCHIVO_TXT = "TEXTOS_PARA_REDES.txt"
CARPETA_PINES = "pines"

MARCA = ""  # Ej: "@tumarca"
VOZ = "es-ES-AlvaroNeural"
VELOCIDAD_VOZ = "+6%"
SEGUNDOS_POR_PLANO = 2.5
NUM_FONDOS = 4

MAX_REINTENTOS = 3
ESPERA_BASE_SEG = 5
TIMEOUT_CSV_SEG = 30
TIMEOUT_GROQ_SEG = 90

CLAVES_JSON_ESPERADAS = ["video_script", "tiktok_data", "ig_reel_data", "youtube_seo", "pinterest_pins", "linkedin_post"]

FUENTES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "DejaVuSans-Bold.ttf",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", stream=sys.stdout)
log = logging.getLogger("mega_maquina")
logging.getLogger("httpx").setLevel(logging.WARNING)

class ErrorFatal(Exception): pass

# ---------------------------------------------------------------------------
# 1. CSV
# ---------------------------------------------------------------------------
def limpiar_texto(valor) -> str:
    s = str(valor)
    s = "".join(ch for ch in s if unicodedata.category(ch) not in ("Cf", "Cc"))
    return s.replace("\u00a0", " ").strip()

def descargar_csv(url: str) -> pd.DataFrame:
    for intento in range(1, MAX_REINTENTOS + 1):
        try:
            resp = requests.get(url, timeout=TIMEOUT_CSV_SEG)
            resp.raise_for_status()
            return pd.read_csv(io.StringIO(resp.content.decode("utf-8")), sep=None, engine="python")
        except Exception as e:
            if intento < MAX_REINTENTOS: time.sleep(ESPERA_BASE_SEG)
    raise ErrorFatal("No se pudo leer el CSV.")

def validar_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: raise ErrorFatal("El DataFrame está vacío.")
    df.columns = [limpiar_texto(c) for c in df.columns]
    df = df[COLUMNAS_REQUERIDAS].dropna().copy()
    for col in COLUMNAS_REQUERIDAS: df[col] = df[col].map(limpiar_texto)
    df = df[(df[COLUMNAS_REQUERIDAS] != "").all(axis=1)]
    if df.empty: raise ErrorFatal("No quedan filas válidas.")
    return df

# ---------------------------------------------------------------------------
# 2. IA: RADAR DE MODELOS BLINDADO
# ---------------------------------------------------------------------------
EXCLUIR_EN_NOMBRE = ("whisper", "guard", "safeguard", "vision", "llava", "orpheus", "tts", "playai", "embed", "distil-whisper")
PREFERENCIAS = ["openai/gpt-oss-120b", "llama-3.3-70b-versatile", "qwen/qwen3", "openai/gpt-oss-20b",
                "llama-3.1-70b-versatile", "llama-3.1-8b-instant", "llama", "mixtral", "gemma"]
ERRORES_MODELO_NO_VALIDO = ("model_terms_required", "requires terms acceptance", "model_not_found",
                            "does not exist", "decommissioned", "not supported", "permission", "no access")

def obtener_modelos_candidatos(client: Groq) -> list:
    log.info("📡 Escaneando modelos en Groq...")
    activos = [m.id for m in client.models.list().data]
    texto = [m for m in activos if not any(x in m.lower() for x in EXCLUIR_EN_NOMBRE)]
    if not texto: raise ErrorFatal("Groq no devuelve modelos de texto válidos.")
    ordenados = []
    for pref in PREFERENCIAS:
        for m in texto:
            if pref in m.lower() and m not in ordenados: ordenados.append(m)
    resto = [m for m in texto if m not in ordenados]
    resto.sort(key=lambda m: "allam" in m.lower())
    return ordenados + resto

def extraer_json(texto: str) -> dict:
    texto = re.sub(r"<think>.*?</think>", "", texto, flags=re.DOTALL).strip()
    texto = re.sub(r"^```(?:json)?|```$", "", texto.strip(), flags=re.MULTILINE).strip()
    try: return json.loads(texto)
    except json.JSONDecodeError:
        ini, fin = texto.find("{"), texto.rfind("}")
        if ini == -1 or fin == -1: raise ValueError("La respuesta no contiene JSON.")
        return json.loads(texto[ini:fin + 1])

def generar_contenido(client: Groq, prompt: str, modelos: list):
    mensajes = [{"role": "system", "content": "Responde SOLO JSON válido."}, {"role": "user", "content": prompt}]
    for modelo in modelos:
        modo_minimo = False
        for intento in range(1, 4):
            try:
                log.info(f"🧠 Generando con {modelo} (intento {intento}, modo {'mínimo' if modo_minimo else 'normal'})...")
                kwargs = dict(messages=mensajes, model=modelo, temperature=0.7)
                if not modo_minimo:
                    kwargs["response_format"] = {"type": "json_object"}
                    kwargs["max_completion_tokens"] = 6000
                resp = client.chat.completions.create(**kwargs)
                datos = extraer_json(resp.choices[0].message.content or "")
                for k in CLAVES_JSON_ESPERADAS:
                    if k not in datos: raise ValueError(f"Falta clave JSON: {k}")
                return datos, modelo
            except Exception as e:
                msg = str(e).lower()
                if any(x in msg for x in ERRORES_MODELO_NO_VALIDO): break
                if ("400" in msg or "json" in msg or "format" in msg) and not modo_minimo:
                    modo_minimo = True; continue
                if "429" in msg or "rate" in msg: time.sleep(15); continue
                if intento < 3: time.sleep(ESPERA_BASE_SEG)
        log.warning(f"⏭️ Siguiente modelo tras fallar {modelo}.")
    raise ErrorFatal("Ningún modelo generó contenido.")

# ---------------------------------------------------------------------------
# 3. UTILIDADES PILLOW PARA VÍDEO Y PINES
# ---------------------------------------------------------------------------
def limpiar_para_imagen(texto: str) -> str:
    out = []
    for ch in str(texto):
        o = ord(ch)
        if o > 0xFFFF or 0x2600 <= o <= 0x27BF or 0x2B00 <= o <= 0x2BFF or o in (0xFE0F, 0x200D): continue
        out.append(ch)
    return re.sub(r"\s+", " ", "".join(out)).strip()

def cargar_fuente(tam: int):
    for ruta in FUENTES:
        try: return ImageFont.truetype(ruta, tam)
        except Exception: continue
    return ImageFont.load_default()

def ajustar_lineas(draw, texto: str, fuente, ancho_max: int) -> list:
    lineas, actual = [], ""
    for p in texto.split():
        prueba = f"{actual} {p}".strip()
        if draw.textlength(prueba, font=fuente) <= ancho_max or not actual: actual = prueba
        else: lineas.append(actual); actual = p
    if actual: lineas.append(actual)
    return lineas or [""]

def render_texto_rgba(texto, ancho, tam, color=(255, 255, 255, 255), stroke=0, fondo=None, pad=30, radio=40, color_stroke=(0, 0, 0, 255)):
    fuente = cargar_fuente(tam)
    tmp = ImageDraw.Draw(Image.new("RGBA", (ancho, 10)))
    lineas = ajustar_lineas(tmp, texto, fuente, ancho - 2 * pad - 2 * stroke)
    alto_linea = int(tam * 1.22)
    alto = len(lineas) * alto_linea + 2 * pad
    img = Image.new("RGBA", (ancho, alto), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if fondo: d.rounded_rectangle([0, 0, ancho - 1, alto - 1], radius=radio, fill=fondo)
    y = pad
    for linea in lineas:
        w = d.textlength(linea, font=fuente)
        d.text(((ancho - w) / 2, y), linea, font=fuente, fill=color, stroke_width=stroke, stroke_fill=color_stroke)
        y += alto_linea
    return np.array(img)

# ---------------------------------------------------------------------------
# 4. PINES DE PINTEREST (BLINDADOS)
# ---------------------------------------------------------------------------
PALETAS = [((10, 25, 60), (30, 90, 170)), ((15, 15, 25), (130, 40, 50)), ((8, 50, 60), (20, 140, 150))]

def degradado(w, h, c1, c2):
    g = np.linspace(0, 1, h).reshape(h, 1, 1)
    fila = np.array(c1).reshape(1, 1, 3) * (1 - g) + np.array(c2).reshape(1, 1, 3) * g
    return Image.fromarray(np.repeat(fila, w, axis=1).astype("uint8"), "RGB")

def crear_pin(texto: str, etiqueta: str, ruta: str, paleta):
    W, H = 1000, 1500
    img = degradado(W, H, *paleta).convert("RGBA")
    deco = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    dd = ImageDraw.Draw(deco)
    dd.ellipse([-250, -250, 450, 450], fill=(255, 255, 255, 18))
    dd.ellipse([600, 1000, 1300, 1700], fill=(255, 255, 255, 18))
    img = Image.alpha_composite(img, deco)
    d = ImageDraw.Draw(img)

    etiqueta = limpiar_para_imagen(etiqueta).upper()[:48]
    fe = cargar_fuente(38)
    d.text(((W - d.textlength(etiqueta, font=fe)) / 2, 150), etiqueta, font=fe, fill=(255, 196, 0, 255))
    d.rectangle([W / 2 - 60, 215, W / 2 + 60, 224], fill=(255, 196, 0, 255))

    texto = limpiar_para_imagen(texto).upper()
    tam = 130
    while True:
        fuente = cargar_fuente(tam)
        lineas = ajustar_lineas(d, texto, fuente, 840)
        alto_total = len(lineas) * int(tam * 1.2)
        if alto_total <= 880 or tam <= 56: break
        tam -= 6
    y = 270 + (900 - alto_total) / 2
    for linea in lineas:
        w = d.textlength(linea, font=fuente)
        d.text(((W - w) / 2, y), linea, font=fuente, fill=(255, 255, 255, 255), stroke_width=3, stroke_fill=(0, 0, 0, 120))
        y += int(tam * 1.2)

    fb = cargar_fuente(46)
    btn = "GUARDA ESTE PIN"
    wb = d.textlength(btn, font=fb)
    d.rounded_rectangle([W / 2 - wb / 2 - 50, 1230, W / 2 + wb / 2 + 50, 1320], radius=45, fill=(255, 196, 0, 255))
    d.text(((W - wb) / 2, 1246), btn, font=fb, fill=(0, 0, 0, 255))
    img.convert("RGB").save(ruta, "JPEG", quality=92)

def fabricar_pines(contenido: dict, nombre: str, problemas: str, enlace: str) -> list:
    os.makedirs(CARPETA_PINES, exist_ok=True)
    pines = contenido.get("pinterest_pins", [])
    
    # === EL BLINDAJE QUE FALTABA (Adiós Error de Slice) ===
    if isinstance(pines, dict):
        pines = list(pines.values()) if all(isinstance(v, dict) for v in pines.values()) else [pines]
    if not isinstance(pines, list):
        pines = []
    # ======================================================
        
    paletas = PALETAS[:]
    random.shuffle(paletas)
    salida = []
    
    for i, p in enumerate(pines[:3], start=1):
        if isinstance(p, str): p = {"text_on_image": p}
        if not isinstance(p, dict): continue
        texto = str(p.get("text_on_image") or p.get("texto") or p.get("text") or "").strip()
        if not texto: continue
        
        titulo = str(p.get("title") or p.get("titulo") or texto)[:100]
        desc = str(p.get("description") or p.get("descripcion") or f"{nombre}. {problemas[:150]}")[:480]
        archivo = os.path.join(CARPETA_PINES, f"pin_{i}.jpg")
        
        crear_pin(texto, nombre, archivo, paletas[(i - 1) % len(paletas)])
        salida.append({"archivo": archivo, "titulo": titulo, "descripcion": desc, "enlace": enlace})
        log.info(f"📌 Pin {i} creado: {archivo}")
        
    with open(os.path.join(CARPETA_PINES, "pines.json"), "w", encoding="utf-8") as f:
        json.dump(salida, f, ensure_ascii=False, indent=2)
    return salida

# ---------------------------------------------------------------------------
# 5. MÚSICA B2B GENERADA (CERO DERECHOS)
# ---------------------------------------------------------------------------
def generar_musica(duracion: float, ruta: str, sr: int = 22050) -> str:
    total = duracion + 1.0
    n = int(total * sr)
    t = np.arange(n) / sr
    acordes = [[220.00, 261.63, 329.63], [174.61, 220.00, 261.63], [261.63, 329.63, 392.00], [196.00, 246.94, 293.66]]
    seg, senal, idx, ini = 4.0, np.zeros(n), 0, 0.0
    while ini < total:
        i0, i1 = int(ini * sr), min(int((ini + seg + 0.5) * sr), n)
        tt = t[i0:i1]
        ac = acordes[idx % len(acordes)]
        env = np.clip(np.minimum((tt - ini + 0.5) / 1.0, ((ini + seg + 0.5) - tt) / 1.0), 0, 1)
        tono = sum(np.sin(2 * np.pi * f * tt) for f in ac) + 1.2 * np.sin(2 * np.pi * (ac[0] / 2) * tt)
        senal[i0:i1] += tono * env
        ini += seg; idx += 1
    senal = senal / (np.max(np.abs(senal)) or 1.0) * 0.12
    senal = senal * np.clip(t / 1.0, 0, 1) * np.clip((total - t) / 1.5, 0, 1)
    datos = (np.repeat(senal[:, None], 2, axis=1) * 32767).astype(np.int16)
    with wave.open(ruta, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(sr); w.writeframes(datos.tobytes())
    return ruta

# ---------------------------------------------------------------------------
# 6. FONDOS PEXELS MULTICÁMARA
# ---------------------------------------------------------------------------
def a_vertical(clip):
    clip = clip.resize(height=1920)
    if clip.w < 1080: clip = clip.resize(width=1080)
    return clip.crop(x_center=clip.w / 2, y_center=clip.h / 2, width=1080, height=1920)

def descargar_fondos_pexels(n: int) -> list:
    api_key = os.environ.get("PEXELS_API_KEY")
    if not api_key: return []
    consultas = ["cargo ship", "shipping containers", "logistics warehouse", "freight truck"]
    random.shuffle(consultas)
    rutas = []
    for consulta in consultas:
        if len(rutas) >= n: break
        try:
            r = requests.get("https://api.pexels.com/videos/search", headers={"Authorization": api_key},
                             params={"query": consulta, "orientation": "portrait", "per_page": 10}, timeout=30)
            r.raise_for_status()
            videos = r.json().get("videos", [])
            if not videos: continue
            v = random.choice(videos)
            archivos = [f for f in v.get("video_files", []) if f.get("file_type") == "video/mp4"]
            if not archivos: continue
            archivos.sort(key=lambda f: f.get("height", 0))
            buenos = [f for f in archivos if f.get("height", 0) >= 1080]
            elegido = buenos[0] if buenos else archivos[-1]
            ruta = f"temp_bg_{len(rutas)}.mp4"
            with requests.get(elegido["link"], stream=True, timeout=60) as resp:
                resp.raise_for_status()
                with open(ruta, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 256): f.write(chunk)
            rutas.append(ruta)
        except Exception: pass
    return rutas

def construir_fondo(rutas: list, duracion: float):
    base = []
    for r in rutas:
        try:
            c = VideoFileClip(r).without_audio()
            if c.duration > 0.5: base.append(a_vertical(c))
        except Exception: pass
    if not base: raise ErrorFatal("No hay fondos válidos.")
    segmentos, total, i = [], 0.0, 0
    while total < duracion:
        c = base[i % len(base)]
        d = min(SEGUNDOS_POR_PLANO, c.duration)
        inicio = min((i // len(base)) * SEGUNDOS_POR_PLANO, max(0.0, c.duration - d))
        segmentos.append(c.subclip(inicio, inicio + d))
        total += d; i += 1
    return concatenate_videoclips(segmentos, method="compose").subclip(0, duracion)

# ---------------------------------------------------------------------------
# 7. MONTAJE DE VÍDEO COMPLETO (HORMOZI STYLE)
# ---------------------------------------------------------------------------
async def generar_voz_y_tiempos(texto: str, archivo_salida: str) -> list:
    com = edge_tts.Communicate(texto, VOZ, rate=VELOCIDAD_VOZ, boundary="WordBoundary")
    eventos = []
    with open(archivo_salida, "wb") as f:
        async for chunk in com.stream():
            if chunk["type"] == "audio": f.write(chunk["data"])
            elif chunk["type"] in ("WordBoundary", "SentenceBoundary"):
                eventos.append((chunk["offset"] / 1e7, chunk["duration"] / 1e7, chunk["text"]))
    return eventos

def construir_subtitulos(eventos: list, texto: str, duracion: float) -> list:
    tokens = []
    for ini, dur, t in eventos:
        ws = t.split()
        if not ws: continue
        paso = dur / len(ws) if dur > 0 else 0.3
        for k, w in enumerate(ws): tokens.append((ini + k * paso, ini + (k + 1) * paso, w))
    
    if not tokens:
        ws = texto.split()
        paso = duracion / max(len(ws), 1)
        tokens = [(k * paso, (k + 1) * paso, w) for k, w in enumerate(ws)]
        
    grupos = []
    # AGRUPA DE 1 a 2 PALABRAS PARA RITMO FRENÉTICO
    for i in range(0, len(tokens), 2):
        b = tokens[i:i + 2]
        grupos.append([b[0][0], b[-1][1], " ".join(x[2] for x in b)])
        
    for j in range(len(grupos) - 1): grupos[j][1] = grupos[j + 1][0]
    if grupos: grupos[-1][1] = min(duracion, grupos[-1][1] + 0.3)
    return [tuple(g) for g in grupos]

def fabricar_video_mp4(script_texto: str, hook_texto: str, cta_texto: str):
    audio_path, musica_path = "temp_voice.mp3", "temp_music.wav"
    eventos = asyncio.run(generar_voz_y_tiempos(script_texto, audio_path))
    voz = AudioFileClip(audio_path)
    duracion = voz.duration + 0.6

    generar_musica(duracion, musica_path)
    musica = AudioFileClip(musica_path)
    audio_final = CompositeAudioClip([voz, musica]).set_duration(duracion)

    log.info("🎬 Descargando y procesando fondos...")
    rutas_bg = descargar_fondos_pexels(NUM_FONDOS)
    fondo = None
    if rutas_bg:
        try: fondo = construir_fondo(rutas_bg, duracion)
        except Exception: pass
    if fondo is None:
        fondo = ColorClip((1080, 1920), color=(15, 23, 42)).set_duration(duracion)

    capas = [fondo, ColorClip((1080, 1920), color=(0, 0, 0)).set_opacity(0.35).set_duration(duracion)]

    hook = limpiar_para_imagen(hook_texto).upper()
    if hook:
        arr = render_texto_rgba(hook, 960, 78, fondo=(0, 0, 0, 175), pad=34)
        capas.append(ImageClip(arr).set_start(0).set_duration(min(4.0, duracion)).set_position(("center", 260)))

    # Subtítulos Hormozi
    for i, (ini, fin, texto) in enumerate(construir_subtitulos(eventos, script_texto, duracion)):
        t = limpiar_para_imagen(texto).upper()
        if not t: continue
        # Intercala colores: Amarillo y Blanco
        color = (255, 196, 0, 255) if i % 4 == 0 else (255, 255, 255, 255)
        arr = render_texto_rgba(t, 980, 110, color=color, stroke=6, pad=20)
        capas.append(ImageClip(arr).set_start(ini).set_duration(fin - ini).set_position(("center", "center")))

    cta = limpiar_para_imagen(cta_texto)
    if cta:
        arr = render_texto_rgba(cta.upper(), 960, 66, color=(0, 0, 0, 255), fondo=(255, 196, 0, 240), pad=34)
        ini_cta = max(duracion - 4.5, 0)
        capas.append(ImageClip(arr).set_start(ini_cta).set_duration(duracion - ini_cta).set_position(("center", 1300)))

    final = CompositeVideoClip(capas, size=(1080, 1920)).set_duration(duracion).set_audio(audio_final)

    log.info("⚙️ Renderizando MP4 final multicámara...")
    final.write_videofile(ARCHIVO_VIDEO, fps=24, codec="libx264", audio_codec="aac", preset="ultrafast", threads=4, logger=None)

    for c in (voz, musica, final):
        try: c.close()
        except: pass
    for p in [audio_path, musica_path] + rutas_bg:
        if os.path.exists(p): os.remove(p)

# ---------------------------------------------------------------------------
# 8. TEXTOS PARA REDES
# ---------------------------------------------------------------------------
def escribir_txt(contenido: dict, enlace: str, pines: list):
    def g(d, k): return d.get(k, "") if isinstance(d, dict) else str(d)
    def hs(h): return " ".join(h) if isinstance(h, list) else str(h)

    tk, ig, yt = contenido.get("tiktok_data", {}), contenido.get("ig_reel_data", {}), contenido.get("youtube_seo", {})
    with open(ARCHIVO_TXT, "w", encoding="utf-8") as f:
        f.write("=== TIKTOK ===\n")
        f.write(f"{g(tk, 'caption')}\n{hs(g(tk, 'hashtags'))}\n\n")
        f.write("=== INSTAGRAM REEL ===\n")
        f.write(f"{g(ig, 'caption')}\n{hs(g(ig, 'hashtags'))}\n\n")
        f.write("=== YOUTUBE SHORTS ===\n")
        f.write(f"Título: {g(yt, 'title')}\nDescripción: {g(yt, 'description')}\n\n")
        f.write("=== LINKEDIN ===\n")
        
        lin = contenido.get('linkedin_post', '')
        if isinstance(lin, dict): lin = " ".join(str(v) for v in lin.values())
        elif isinstance(lin, list): lin = " ".join(str(x) for x in lin)
        f.write(f"{str(lin).strip()}\n\n")
        
        f.write("=== PINTEREST ===\n")
        for p in pines:
            f.write(f"[{p['archivo']}]\nTítulo: {p['titulo']}\nDescripción: {p['descripcion']}\nEnlace: {p['enlace']}\n\n")
        f.write(f"=== ENLACE DEL PRODUCTO ===\n{enlace}\n")

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    log.info("🚀 Arrancando la Mega Máquina de Contenido...")
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key: raise ErrorFatal("Falta GROQ_API_KEY en los Secrets.")

    df = validar_dataframe(descargar_csv(CSV_URL))
    prod = df.sample(n=1).iloc[0]
    nombre, problemas = prod["NOMBRE_PRODUCTO"], prod["PROBLEMAS_QUE_RESUELVE"]
    palabra_clave, enlace = prod["PALABRA_CLAVE_MANYCHAT"], prod["ENLACE_HOTMART"]

    client = Groq(api_key=api_key, timeout=TIMEOUT_GROQ_SEG, max_retries=0)
    modelos = obtener_modelos_candidatos(client)

    prompt = f"""
Actúa como un copywriter B2B experto en logística.
Vende este producto: '{nombre}'. Problemas que soluciona: '{problemas}'.
REGLAS: Cero niños, dolor de e-commerce real, usa jerga (Demurrage, DUA, Incoterms).

El video_script debe ser UN SOLO TEXTO continuo de 60 palabras max:
1) Gancho de dolor
2) Problema agravado
3) Beneficios
4) CTA: "Comenta '{palabra_clave}'".

Devuelve JSON estricto:
- video_script (string puro)
- hook_text (max 8 palabras)
- cta_text (max 8 palabras)
- tiktok_data (caption, hashtags)
- ig_reel_data (caption, hashtags)
- youtube_seo (title, description)
- pinterest_pins (array de 3 objetos con 'text_on_image' max 8 palabras)
- linkedin_post (texto estructurado)
"""

    contenido, mod = generar_contenido(client, prompt, modelos)
    
    with open(ARCHIVO_JSON, "w", encoding="utf-8") as f: json.dump(contenido, f, ensure_ascii=False, indent=2)

    guion_voz = limpiar_texto(contenido.get("video_script", ""))
    if isinstance(contenido.get("video_script"), dict): guion_voz = " ".join(str(v) for v in contenido.get("video_script").values())
    elif isinstance(contenido.get("video_script"), list): guion_voz = " ".join(str(x) for x in contenido.get("video_script"))
    
    if not guion_voz: guion_voz = f"Evita sobrecostes. Comenta {palabra_clave} y te ayudo con {nombre}."

    hook = limpiar_texto(contenido.get("hook_text", ""))
    cta = limpiar_texto(contenido.get("cta_text", "")) or f"Comenta {palabra_clave} y te lo envío"

    pines = []
    try: pines = fabricar_pines(contenido, nombre, problemas, enlace)
    except Exception as e: log.warning(f"⚠️ Error en pines: {e}")

    try: escribir_txt(contenido, enlace, pines)
    except Exception as e: log.warning(f"⚠️ Error en TXT: {e}")

    fabricar_video_mp4(guion_voz, hook, cta)
    log.info("🏁 Pipeline completado.")

if __name__ == "__main__":
    try: main()
    except Exception as e:
        log.error(f"❌ ERROR FATAL: {e}")
        sys.exit(1)
