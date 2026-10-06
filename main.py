import os
import io
import re
import sys
import json
import time
import random
import asyncio
import logging
import unicodedata
import requests
import numpy as np
import pandas as pd
from groq import Groq
import edge_tts

import PIL.Image
if not hasattr(PIL.Image, "ANTIALIAS"):
    try: PIL.Image.ANTIALIAS = PIL.Image.Resampling.LANCZOS
    except AttributeError: PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

from PIL import Image, ImageDraw, ImageFont, ImageEnhance
from moviepy.editor import (
    VideoFileClip, AudioFileClip, ImageClip, ColorClip, VideoClip,
    CompositeVideoClip, CompositeAudioClip, concatenate_videoclips,
)

# ---------------------------------------------------------------------------
# CONFIGURACIÓN OMNICANAL
# ---------------------------------------------------------------------------
SHEET_ID = "10gJJCIlPzCHYEfPYPKgT3-xjUtghbIaYpdR87Da4JPQ"
CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv"
COLUMNAS_REQUERIDAS = ["NOMBRE_PRODUCTO", "PROBLEMAS_QUE_RESUELVE", "PALABRA_CLAVE_MANYCHAT", "ENLACE_HOTMART"]

CARPETA_PINES = "pines"
SEGUNDOS_POR_PLANO = 2.5
NUM_FONDOS = 4

MAX_REINTENTOS = 3
ESPERA_BASE_SEG = 5
TIMEOUT_CSV_SEG = 30
TIMEOUT_GROQ_SEG = 90

CLAVES_JSON_ESPERADAS = [
    "video_script", "hook_text", "tiktok_caption", "ig_reel_caption", 
    "ig_post_image_text", "youtube_seo", "pinterest_pins", 
    "linkedin_story", "linkedin_pdf_slides"
]

FUENTES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "DejaVuSans-Bold.ttf",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", stream=sys.stdout)
log = logging.getLogger("agencia_360")
logging.getLogger("httpx").setLevel(logging.WARNING)

class ErrorFatal(Exception): pass

# ---------------------------------------------------------------------------
# 1. UTILIDADES Y LECTURA
# ---------------------------------------------------------------------------
def limpiar_texto(valor) -> str:
    s = "".join(ch for ch in str(valor) if unicodedata.category(ch) not in ("Cf", "Cc"))
    return s.replace("\u00a0", " ").strip()

def descargar_csv(url: str) -> pd.DataFrame:
    for i in range(3):
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.content.decode("utf-8")), sep=None, engine="python")
            df.columns = [limpiar_texto(c) for c in df.columns]
            df = df[COLUMNAS_REQUERIDAS].dropna().copy()
            for col in COLUMNAS_REQUERIDAS: df[col] = df[col].map(limpiar_texto)
            return df[(df[COLUMNAS_REQUERIDAS] != "").all(axis=1)]
        except Exception as e: 
            log.warning(f"⚠️ Error leyendo CSV: {e}")
            time.sleep(5)
    raise ErrorFatal("Fallo crítico leyendo CSV.")

# ---------------------------------------------------------------------------
# 2. IA: RADAR DE MODELOS BLINDADO
# ---------------------------------------------------------------------------
EXCLUIR_EN_NOMBRE = ("whisper", "guard", "safeguard", "vision", "llava", "orpheus", "tts", "playai", "embed", "distil-whisper")
PREFERENCIAS = [
    "openai/gpt-oss-20b", "openai/gpt-oss-120b", "llama-3.3-70b-versatile", 
    "qwen/qwen3", "llama-3.1-70b-versatile", "mixtral", "gemma"
]

def obtener_modelos_candidatos(client: Groq) -> list:
    log.info("📡 Escaneando modelos disponibles en Groq...")
    try: activos = [m.id for m in client.models.list().data]
    except Exception as e: raise ErrorFatal(f"Fallo en el radar: {e}")
    
    texto = [m for m in activos if not any(x in m.lower() for x in EXCLUIR_EN_NOMBRE)]
    if not texto: raise ErrorFatal("Groq no devuelve modelos de texto.")
    
    ordenados = []
    for pref in PREFERENCIAS:
        for m in texto:
            if pref in m.lower() and m not in ordenados: ordenados.append(m)
    return ordenados + [m for m in texto if m not in ordenados]

def extraer_json(texto: str) -> dict:
    t = re.sub(r"<think>.*?</think>", "", texto, flags=re.DOTALL).strip()
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.MULTILINE).strip()
    try: return json.loads(t)
    except json.JSONDecodeError:
        ini, fin = t.find("{"), t.rfind("}")
        if ini == -1 or fin == -1: raise ValueError("Sin JSON válido.")
        return json.loads(t[ini:fin + 1])

def generar_contenido(client: Groq, prompt: str):
    modelos = obtener_modelos_candidatos(client)
    for modelo in modelos:
        modo_minimo = False
        for intento in range(1, 3):
            try:
                log.info(f"🧠 Generando con {modelo}...")
                kwargs = dict(
                    messages=[{"role": "system", "content": "Responde SOLO JSON válido."}, {"role": "user", "content": prompt}],
                    model=modelo, temperature=0.7
                )
                if not modo_minimo:
                    kwargs["response_format"] = {"type": "json_object"}
                    kwargs["max_completion_tokens"] = 6000
                
                resp = client.chat.completions.create(**kwargs)
                datos = extraer_json(resp.choices[0].message.content or "")
                for k in CLAVES_JSON_ESPERADAS:
                    if k not in datos: raise ValueError(f"Falta clave: {k}")
                return datos
            except Exception as e:
                msg = str(e).lower()
                if ("400" in msg or "json" in msg or "format" in msg) and not modo_minimo:
                    modo_minimo = True; continue
                time.sleep(3)
    raise ErrorFatal("Colapso de la IA.")

# ---------------------------------------------------------------------------
# 3. UTILIDADES PILLOW
# ---------------------------------------------------------------------------
def cargar_fuente(tam: int):
    for ruta in FUENTES:
        try: return ImageFont.truetype(ruta, tam)
        except: continue
    return ImageFont.load_default()

def ajustar_lineas(draw, texto: str, fuente, ancho_max: int) -> list:
    lineas, actual = [], ""
    for p in str(texto).split():
        prueba = f"{actual} {p}".strip()
        if draw.textlength(prueba, font=fuente) <= ancho_max or not actual: actual = prueba
        else: lineas.append(actual); actual = p
    if actual: lineas.append(actual)
    return lineas or [""]

def render_texto_rgba(texto, ancho, tam, color=(255, 255, 255, 255), stroke=0, fondo=None, pad=30, radio=40):
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
        d.text(((ancho - w) / 2 + 4, y + 4), linea, font=fuente, fill=(0, 0, 0, 220))
        d.text(((ancho - w) / 2, y), linea, font=fuente, fill=color, stroke_width=stroke, stroke_fill=(0, 0, 0, 255))
        y += alto_linea
    return np.array(img)

# ---------------------------------------------------------------------------
# 4. GENERADORES VISUALES (PINES, IG POST, PDF LINKEDIN)
# ---------------------------------------------------------------------------
def generar_pines(contenido: dict):
    os.makedirs(CARPETA_PINES, exist_ok=True) # <--- AQUÍ ESTABA EL FALLO DE DIRECTORIO
    pines = contenido.get("pinterest_pins", [])
    if isinstance(pines, dict): pines = list(pines.values())
    if not isinstance(pines, list): pines = []
        
    for i, p in enumerate(pines[:3]):
        if isinstance(p, str): p = {"title": p}
        try:
            r = requests.get("https://api.pexels.com/v1/search?query=logistics container&orientation=portrait&per_page=5",
                             headers={"Authorization": os.environ.get("PEXELS_API_KEY")}, timeout=15)
            fotos = r.json().get("photos", [])
            url = random.choice(fotos)["src"]["large2x"] if fotos else None
            img = Image.open(io.BytesIO(requests.get(url, stream=True).content)).convert("RGB").resize((1080, 1920), Image.ANTIALIAS) if url else Image.new("RGB", (1080, 1920), (20, 30, 60))
        except:
            img = Image.new("RGB", (1080, 1920), (20, 30, 60))
            
        img = ImageEnhance.Brightness(img).enhance(0.4)
        draw = ImageDraw.Draw(img)
        
        titulo = str(p.get("title", p.get("text_on_image", "IMPORTACIÓN"))).upper()
        
        fuente = cargar_fuente(80)
        lineas = ajustar_lineas(draw, titulo, fuente, 900)
        y = 700 - (len(lineas) * 50)
        for linea in lineas:
            w = draw.textlength(linea, font=fuente)
            draw.text(((1080 - w)/2, y), linea, font=fuente, fill=(255,255,255), stroke_width=3, stroke_fill=(0,0,0))
            y += 100
            
        draw.rectangle([440, y+40, 640, y+50], fill=(255, 196, 0))
        img.save(os.path.join(CARPETA_PINES, f"pin_{i+1}.jpg"), quality=90)
        log.info(f"📌 Pin {i+1} generado con foto real.")

def generar_post_ig(contenido: dict):
    try:
        r = requests.get("https://api.pexels.com/v1/search?query=logistics container&orientation=square&per_page=5",
                         headers={"Authorization": os.environ.get("PEXELS_API_KEY")}, timeout=15)
        fotos = r.json().get("photos", [])
        url = random.choice(fotos)["src"]["large2x"] if fotos else None
        img = Image.open(io.BytesIO(requests.get(url, stream=True).content)).convert("RGB").resize((1080, 1080), Image.ANTIALIAS) if url else Image.new("RGB", (1080, 1080), (15, 23, 42))
    except:
        img = Image.new("RGB", (1080, 1080), (15, 23, 42))
        
    img = ImageEnhance.Brightness(img).enhance(0.35)
    draw = ImageDraw.Draw(img)
    texto_ig = str(contenido.get("ig_post_image_text", "TIP LOGÍSTICO")).upper()
    fuente = cargar_fuente(75)
    lineas = ajustar_lineas(draw, texto_ig, fuente, 900)
    y = 540 - (len(lineas) * 45)
    for linea in lineas:
        w = draw.textlength(linea, font=fuente)
        draw.text(((1080 - w)/2, y), linea, font=fuente, fill=(255,255,255), stroke_width=3, stroke_fill=(0,0,0))
        y += 90
    img.save("ig_post_estatico.jpg", quality=95)
    log.info("📷 Post estático IG generado.")

def generar_pdf_linkedin(contenido: dict):
    diapositivas = contenido.get("linkedin_pdf_slides", ["Slide 1", "Slide 2", "Slide 3"])
    if not isinstance(diapositivas, list): diapositivas = ["Slide 1", "Slide 2"]
    imagenes = []
    for i, texto in enumerate(diapositivas[:3]):
        img = Image.new("RGB", (1080, 1080), (240, 240, 240) if i > 0 else (10, 30, 60))
        color_texto = (30, 30, 30) if i > 0 else (255, 255, 255)
        draw = ImageDraw.Draw(img)
        fuente = cargar_fuente(65)
        lineas = ajustar_lineas(draw, str(texto).upper(), fuente, 900)
        y = 540 - (len(lineas) * 40)
        for linea in lineas:
            w = draw.textlength(linea, font=fuente)
            draw.text(((1080 - w)/2, y), linea, font=fuente, fill=color_texto)
            y += 80
        if i == len(diapositivas)-1:
            draw.rectangle([200, 850, 880, 950], fill=(255, 196, 0))
            w = draw.textlength("COMENTA PARA RECIBIR LA GUÍA", font=cargar_fuente(35))
            draw.text(((1080 - w)/2, 882), "COMENTA PARA RECIBIR LA GUÍA", font=cargar_fuente(35), fill=(0,0,0))
        imagenes.append(img)
    if imagenes:
        imagenes[0].save("linkedin_carrusel.pdf", save_all=True, append_images=imagenes[1:])
        log.info("💼 PDF LinkedIn generado.")

# ---------------------------------------------------------------------------
# 5. MONTAJE DE VÍDEO
# ---------------------------------------------------------------------------
async def generar_audio(texto: str, archivo: str) -> list:
    com = edge_tts.Communicate(texto, "es-ES-AlvaroNeural", rate="+6%", boundary="WordBoundary")
    eventos = []
    with open(archivo, "wb") as f:
        async for chunk in com.stream():
            if chunk["type"] == "audio": f.write(chunk["data"])
            elif chunk["type"] in ("WordBoundary", "SentenceBoundary"):
                eventos.append((chunk["offset"]/1e7, chunk["duration"]/1e7, chunk["text"]))
    return eventos

def descargar_fondos_pexels() -> list:
    api_key = os.environ.get("PEXELS_API_KEY")
    if not api_key: return []
    rutas = []
    try:
        q = random.choice(["cargo ship", "logistics warehouse", "shipping containers", "freight logistics"])
        r = requests.get(f"https://api.pexels.com/videos/search?query={q}&orientation=portrait&per_page=15",
                         headers={"Authorization": api_key}, timeout=30)
        for v in r.json().get("videos", [])[:NUM_FONDOS]:
            archivos = [f for f in v["video_files"] if f["file_type"] == "video/mp4" and f["height"] >= 1080]
            if not archivos: continue
            ruta = f"bg_{len(rutas)}.mp4"
            with requests.get(archivos[0]["link"], stream=True) as resp:
                with open(ruta, "wb") as f:
                    for chunk in resp.iter_content(1024*256): f.write(chunk)
            rutas.append(ruta)
    except: pass
    return rutas

def montar_video(script: str, hook: str, archivo_salida: str):
    audio_path = "temp_voice.mp3"
    eventos = asyncio.run(generar_audio(script, audio_path))
    voz = AudioFileClip(audio_path)
    dur = voz.duration + 0.5
    
    rutas = descargar_fondos_pexels()
    base = []
    for r in rutas:
        try:
            clip = VideoFileClip(r).without_audio().resize(height=1920).crop(x_center=960, width=1080)
            clip = clip.resize(lambda t: 1 + 0.03 * (t / clip.duration)) 
            base.append(clip)
        except: pass
        
    if base:
        segmentos, total, i = [], 0.0, 0
        while total < dur:
            c = base[i % len(base)]
            d = min(SEGUNDOS_POR_PLANO, c.duration)
            segmentos.append(c.subclip(0, d)); total += d; i += 1
        fondo = concatenate_videoclips(segmentos, method="compose").subclip(0, dur)
    else:
        fondo = ColorClip((1080, 1920), color=(15, 23, 42)).set_duration(dur)

    capas = [fondo, ColorClip((1080, 1920), color=(0,0,0)).set_opacity(0.35).set_duration(dur)]
    
    if hook:
        arr = render_texto_rgba(hook.upper(), 960, 60, (255,255,255,255), 4)
        capas.append(ImageClip(arr).set_start(0).set_duration(3.0).set_position(("center", 250)))

    tokens = []
    for ini, d_ev, t in eventos:
        ws = t.split()
        if not ws: continue
        paso = d_ev / len(ws) if d_ev > 0 else 0.3
        for k, w in enumerate(ws): tokens.append((ini + k*paso, ini + (k+1)*paso, w))
    
    grupos = []
    for i in range(0, len(tokens), 2):
        b = tokens[i:i+2]
        grupos.append([b[0][0], b[-1][1], " ".join(x[2] for x in b)])
    for j in range(len(grupos)-1): grupos[j][1] = grupos[j+1][0]
    if grupos: grupos[-1][1] = min(dur, grupos[-1][1] + 0.3)

    for i, (ini, fin, txt) in enumerate(grupos):
        t = limpiar_texto(txt).upper()
        if not t: continue
        c = (255, 196, 0, 255) if i % 3 == 0 else (255, 255, 255, 255)
        arr = render_texto_rgba(t, 980, 110, c, 6)
        capas.append(ImageClip(arr).set_start(ini).set_duration(fin-ini).set_position(("center", "center")))

    def make_barra(t):
        w_barra = int(1080 * (t / dur))
        img = Image.new("RGBA", (1080, 12), (0,0,0,0))
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, w_barra, 10], fill=(255, 196, 0, 255))
        return np.array(img)
    
    barra_clip = VideoClip(make_barra, duration=dur).set_position(("left", "top"))
    capas.append(barra_clip)

    final = CompositeVideoClip(capas, size=(1080, 1920)).set_duration(dur).set_audio(voz)
    final.write_videofile(archivo_salida, fps=24, codec="libx264", audio_codec="aac", preset="ultrafast", logger=None)
    voz.close(); final.close()
    for f in rutas:
        if os.path.exists(f): os.remove(f)

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    log.info("🚀 Arrancando Agencia 360 (Versión de Élite)...")
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key: raise ErrorFatal("Falta GROQ_API_KEY.")

    df = descargar_csv(CSV_URL)
    prod = df.sample(n=1).iloc[0]
    nombre, problemas, palabra = prod["NOMBRE_PRODUCTO"], prod["PROBLEMAS_QUE_RESUELVE"], prod["PALABRA_CLAVE_MANYCHAT"]
    enlace = prod["ENLACE_HOTMART"]

    prompt = f"""
    Copywriter B2B logística. Producto: '{nombre}'. Problema: '{problemas}'.
    Debes generar contenido de alta conversión en ESPAÑOL.
    
    JSON REQUERIDO EXACTO:
    - video_script (Español, max 60 palabras, dolor + solución + CTA 'Comenta {palabra}')
    - hook_text (max 6 palabras de impacto)
    - tiktok_caption (Texto para TikTok con hashtags)
    - ig_reel_caption (Texto para Instagram Reels con hashtags)
    - ig_post_image_text (Frase corta técnica para imagen estática de Instagram)
    - youtube_seo (Objeto con 'title' y 'description')
    - pinterest_pins (Array de 3 objetos, cada uno con 'title' y 'description' que incluya CTA)
    - linkedin_story (Post texto narrativo contando una anécdota real de la trinchera logística)
    - linkedin_pdf_slides (Array de 3 strings cortos con lecciones de logística para un carrusel)
    """
    
    client = Groq(api_key=api_key)
    contenido = generar_contenido(client, prompt)
    
    with open("TEXTOS_PARA_REDES.txt", "w", encoding="utf-8") as f:
        f.write(f"=== TIKTOK ===\n{contenido.get('tiktok_caption', '')}\n\n")
        f.write(f"=== INSTAGRAM REELS ===\n{contenido.get('ig_reel_caption', '')}\n\n")
        f.write(f"=== LINKEDIN STORY ===\n{contenido.get('linkedin_story', '')}\n\n")
        yt = contenido.get('youtube_seo', {})
        if isinstance(yt, dict):
            f.write(f"=== YOUTUBE SHORTS ===\nTítulo: {yt.get('title', '')}\nDescripción: {yt.get('description', '')}\n\n")
        f.write(f"=== ENLACE DIRECTO ===\n{enlace}\n")

    log.info("🖼️ Fabricando Pines, Post IG y PDF LinkedIn...")
    generar_pines(contenido)
    generar_post_ig(contenido)
    generar_pdf_linkedin(contenido)

    log.info("🎞️ Renderizando MP4 de Élite con Zoom y Barra de Progreso...")
    guion = str(contenido.get("video_script", f"Problemas logísticos. Comenta {palabra}"))
    hook = str(contenido.get("hook_text", ""))
    
    montar_video(guion, hook, "video_final.mp4")

    log.info("🏁 Operación completada con éxito.")

if __name__ == "__main__":
    try: main()
    except Exception as e:
        log.error(f"❌ ERROR FATAL: {e}")
        sys.exit(1)
