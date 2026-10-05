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
        except Exception: time.sleep(5)
    raise ErrorFatal("Fallo crítico leyendo CSV.")

# ---------------------------------------------------------------------------
# 2. IA: RADAR Y GENERACIÓN
# ---------------------------------------------------------------------------
def extraer_json(texto: str) -> dict:
    t = re.sub(r"<think>.*?</think>", "", texto, flags=re.DOTALL).strip()
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.MULTILINE).strip()
    try: return json.loads(t)
    except json.JSONDecodeError:
        ini, fin = t.find("{"), t.rfind("}")
        if ini == -1 or fin == -1: raise ValueError("Sin JSON.")
        return json.loads(t[ini:fin + 1])

def generar_contenido(client: Groq, prompt: str):
    modelos = ["llama-3.3-70b-versatile", "llama-3.1-70b-versatile", "mixtral-8x7b-32768", "llama3-8b-8192"]
    for modelo in modelos:
        for intento in range(3):
            try:
                log.info(f"🧠 Prompting {modelo}...")
                resp = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=modelo, temperature=0.7, response_format={"type": "json_object"}
                )
                return extraer_json(resp.choices[0].message.content or "")
            except Exception: time.sleep(5)
    raise ErrorFatal("Colapso de la IA.")

# ---------------------------------------------------------------------------
# 3. PEXELS Y DIBUJO (PILLOW)
# ---------------------------------------------------------------------------
def obtener_fondos_pexels(orientacion="portrait", num=3) -> list:
    api_key = os.environ.get("PEXELS_API_KEY")
    if not api_key: return []
    rutas = []
    try:
        q = random.choice(["cargo ship", "logistics warehouse", "shipping containers"])
        r = requests.get(f"https://api.pexels.com/videos/search?query={q}&orientation={orientacion}&per_page=15",
                         headers={"Authorization": api_key}, timeout=30)
        videos = r.json().get("videos", [])
        for v in videos[:num]:
            archivos = [f for f in v["video_files"] if f["file_type"] == "video/mp4" and f["height"] >= 1080]
            if not archivos: continue
            ruta = f"bg_{len(rutas)}.mp4"
            with requests.get(archivos[0]["link"], stream=True) as resp:
                with open(ruta, "wb") as f:
                    for chunk in resp.iter_content(1024*256): f.write(chunk)
            rutas.append(ruta)
    except Exception as e: log.warning(f"⚠️ Fallo Pexels: {e}")
    return rutas

def descargar_foto_pexels(orientacion="portrait") -> str:
    api_key = os.environ.get("PEXELS_API_KEY")
    if not api_key: return None
    try:
        r = requests.get(f"https://api.pexels.com/v1/search?query=logistics container&orientation={orientacion}&per_page=5",
                         headers={"Authorization": api_key}, timeout=30)
        fotos = r.json().get("photos", [])
        if fotos: return random.choice(fotos)["src"]["large2x"]
    except: return None
    return None

def cargar_fuente(tam: int):
    try: return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", tam)
    except: return ImageFont.load_default()

def dibujar_texto_centrado(draw, texto, w, y, fuente, color=(255,255,255), stroke=0):
    lineas, actual = [], ""
    for p in texto.split():
        if draw.textlength(actual + " " + p, font=fuente) < w - 100: actual += " " + p
        else: lineas.append(actual.strip()); actual = p
    lineas.append(actual.strip())
    
    for linea in lineas:
        ancho = draw.textlength(linea, font=fuente)
        draw.text(((w - ancho)/2, y), linea, font=fuente, fill=color, stroke_width=stroke, stroke_fill=(0,0,0))
        y += int(fuente.size * 1.2)
    return y

# ---------------------------------------------------------------------------
# 4. GENERADORES VISUALES (PINES, IG POST, LINKEDIN PDF)
# ---------------------------------------------------------------------------
def generar_pines(contenido: dict):
    os.makedirs(CARPETA_PINES, exist_ok=True)
    pines = contenido.get("pinterest_pins", [])[:3]
    for i, p in enumerate(pines):
        url = descargar_foto_pexels("portrait")
        if url:
            img = Image.open(requests.get(url, stream=True).raw).convert("RGB").resize((1080, 1920), Image.ANTIALIAS)
        else:
            img = Image.new("RGB", (1080, 1920), (20, 30, 60))
        
        img = ImageEnhance.Brightness(img).enhance(0.4)
        draw = ImageDraw.Draw(img)
        
        y = dibujar_texto_centrado(draw, p.get("title", "IMPORTACIÓN").upper(), 1080, 600, cargar_fuente(90), stroke=4)
        draw.rectangle([440, y+40, 640, y+50], fill=(255, 196, 0))
        img.save(os.path.join(CARPETA_PINES, f"pin_{i+1}.jpg"), quality=90)

def generar_post_ig(contenido: dict):
    url = descargar_foto_pexels("square")
    img = Image.open(requests.get(url, stream=True).raw).convert("RGB").resize((1080, 1080), Image.ANTIALIAS) if url else Image.new("RGB", (1080, 1080), (15, 23, 42))
    img = ImageEnhance.Brightness(img).enhance(0.3)
    draw = ImageDraw.Draw(img)
    dibujar_texto_centrado(draw, contenido.get("ig_post_image_text", "TIP LOGÍSTICO").upper(), 1080, 300, cargar_fuente(85), stroke=3)
    img.save("ig_post_estatico.jpg", quality=95)

def generar_pdf_linkedin(contenido: dict):
    diapositivas = contenido.get("linkedin_pdf_slides", ["Slide 1", "Slide 2", "Slide 3"])[:3]
    imagenes = []
    for i, texto in enumerate(diapositivas):
        img = Image.new("RGB", (1080, 1080), (240, 240, 240) if i > 0 else (10, 30, 60))
        color_texto = (30, 30, 30) if i > 0 else (255, 255, 255)
        draw = ImageDraw.Draw(img)
        dibujar_texto_centrado(draw, texto.upper(), 1080, 400, cargar_fuente(70), color=color_texto)
        if i == len(diapositivas)-1:
            draw.rectangle([200, 800, 880, 900], fill=(255, 196, 0))
            dibujar_texto_centrado(draw, "COMENTA PARA RECIBIR LA GUÍA", 1080, 820, cargar_fuente(40), color=(0,0,0))
        imagenes.append(img)
    
    if imagenes:
        imagenes[0].save("linkedin_carrusel.pdf", save_all=True, append_images=imagenes[1:])

# ---------------------------------------------------------------------------
# 5. MONTAJE DE VÍDEO (ESPAÑOL Y FRANCÉS)
# ---------------------------------------------------------------------------
async def generar_audio(texto: str, archivo: str, voz: str) -> list:
    com = edge_tts.Communicate(texto, voz, rate="+6%", boundary="WordBoundary")
    eventos = []
    with open(archivo, "wb") as f:
        async for chunk in com.stream():
            if chunk["type"] == "audio": f.write(chunk["data"])
            elif chunk["type"] in ("WordBoundary", "SentenceBoundary"):
                eventos.append((chunk["offset"]/1e7, chunk["duration"]/1e7, chunk["text"]))
    return eventos

def render_texto_rgba(texto, w, tam, color, stroke):
    fuente = cargar_fuente(tam)
    img = Image.new("RGBA", (w, int(tam*1.5)), (0,0,0,0))
    d = ImageDraw.Draw(img)
    ancho = d.textlength(texto, font=fuente)
    d.text(((w-ancho)/2, 10), texto, font=fuente, fill=color, stroke_width=stroke, stroke_fill=(0,0,0,255))
    return np.array(img)

def montar_video(script: str, hook: str, archivo_salida: str, voz_id: str, fondos_paths: list):
    audio_path = "temp.mp3"
    eventos = asyncio.run(generar_audio(script, audio_path, voz_id))
    voz = AudioFileClip(audio_path)
    dur = voz.duration + 0.5
    
    base = []
    for r in fondos_paths:
        try: base.append(VideoFileClip(r).without_audio().resize(height=1920).crop(x_center=960, width=1080))
        except: pass
        
    if base:
        segmentos, total, i = [], 0.0, 0
        while total < dur:
            c = base[i % len(base)]
            d = min(2.5, c.duration)
            segmentos.append(c.subclip(0, d)); total += d; i += 1
        fondo = concatenate_videoclips(segmentos, method="compose").subclip(0, dur)
    else:
        fondo = ColorClip((1080, 1920), color=(15, 23, 42)).set_duration(dur)

    capas = [fondo, ColorClip((1080, 1920), color=(0,0,0)).set_opacity(0.3).set_duration(dur)]
    
    if hook:
        arr = render_texto_rgba(hook.upper(), 960, 60, (255,255,255,255), 4)
        capas.append(ImageClip(arr).set_start(0).set_duration(3.0).set_position(("center", 250)))

    # Subtítulos Hormozi (1-2 palabras)
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

    final = CompositeVideoClip(capas, size=(1080, 1920)).set_duration(dur).set_audio(voz)
    final.write_videofile(archivo_salida, fps=24, codec="libx264", audio_codec="aac", preset="ultrafast", logger=None)
    voz.close(); final.close()
    if os.path.exists(audio_path): os.remove(audio_path)

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    log.info("🚀 Arrancando Agencia 360 Omnicanal...")
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key: raise ErrorFatal("Falta GROQ_API_KEY.")

    df = descargar_csv(CSV_URL)
    prod = df.sample(n=1).iloc[0]
    nombre, problemas, palabra = prod["NOMBRE_PRODUCTO"], prod["PROBLEMAS_QUE_RESUELVE"], prod["PALABRA_CLAVE_MANYCHAT"]

    prompt = f"""
    Copywriter B2B logística. Producto: '{nombre}'. Problema: '{problemas}'.
    Debes generar contenido para 5 redes sociales distintas.
    
    JSON REQUERIDO:
    - video_script_es (Español, max 60 palabras, dolor + solución + CTA 'Comenta {palabra}')
    - video_script_fr (Francés, traducción exacta del script español)
    - hook_es (max 6 palabras)
    - hook_fr (max 6 palabras francés)
    - tiktok_fr_caption (Texto para TikTok en francés con hashtags)
    - ig_reel_es_caption (Texto para Instagram en español con hashtags)
    - ig_post_image_text (Frase corta técnica para imagen estática de Instagram)
    - youtube_seo (title y description para Shorts en español)
    - pinterest_pins (Array de 3 objetos con 'title', 'description' y CTA)
    - linkedin_story (Post texto narrativo contando una anécdota real de la trinchera logística, sin hashtags excesivos)
    - linkedin_pdf_slides (Array de 3 strings cortos con lecciones de logística para el PDF)
    """
    
    client = Groq(api_key=api_key)
    contenido = generar_contenido(client, prompt)
    
    # 1. TEXTOS OMNICANAL
    with open("TEXTOS_PARA_REDES.txt", "w", encoding="utf-8") as f:
        f.write(f"=== TIKTOK FRANCIA ===\n{contenido.get('tiktok_fr_caption', '')}\n\n")
        f.write(f"=== IG YOUTUBE ESPAÑA ===\n{contenido.get('ig_reel_es_caption', '')}\n\n")
        f.write(f"=== LINKEDIN STORY ===\n{contenido.get('linkedin_story', '')}\n")

    # 2. IMÁGENES Y PDF
    log.info("🖼️ Fabricando Pines, Post IG y PDF LinkedIn...")
    generar_pines(contenido)
    generar_post_ig(contenido)
    generar_pdf_linkedin(contenido)

    # 3. VÍDEOS (ESPAÑA Y FRANCIA)
    log.info("🎬 Descargando fondos maestros de Pexels...")
    fondos = obtener_fondos_pexels()
    
    log.info("🇪🇸 Renderizando MP4 España...")
    montar_video(contenido.get("video_script_es", "Problemas de aduana. Comenta."), 
                 contenido.get("hook_es", ""), "video_ig_yt_es.mp4", "es-ES-AlvaroNeural", fondos)
                 
    log.info("🇫🇷 Renderizando MP4 Francia...")
    montar_video(contenido.get("video_script_fr", "Problèmes de douane. Commente."), 
                 contenido.get("hook_fr", ""), "video_tiktok_fr.mp4", "fr-FR-HenriNeural", fondos)

    for f in fondos:
        if os.path.exists(f): os.remove(f)

    log.info("🏁 Operación Agencia 360 finalizada.")

if __name__ == "__main__":
    main()
