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
    try:
        PIL.Image.ANTIALIAS = PIL.Image.Resampling.LANCZOS
    except AttributeError:
        PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

from PIL import Image, ImageDraw, ImageFont, ImageEnhance
from moviepy.editor import (
    VideoFileClip, AudioFileClip, ImageClip, ColorClip,
    CompositeVideoClip, concatenate_videoclips, TextClip
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

VOZ = "es-ES-AlvaroNeural"
VELOCIDAD_VOZ = "+8%"  # Voz un poco más rápida y dinámica
MAX_REINTENTOS = 3
ESPERA_BASE_SEG = 5
TIMEOUT_CSV_SEG = 30
TIMEOUT_GROQ_SEG = 90

CLAVES_JSON_ESPERADAS = ["video_script", "tiktok_data", "ig_reel_data", "youtube_seo", "pinterest_pins", "linkedin_post"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", stream=sys.stdout)
log = logging.getLogger("mega_maquina")
logging.getLogger("httpx").setLevel(logging.WARNING)

class ErrorFatal(Exception): pass

# ---------------------------------------------------------------------------
# 1. UTILIDADES Y CSV
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
# 2. IA: RADAR Y GENERACIÓN
# ---------------------------------------------------------------------------
PREFERENCIAS = ["llama-3.3-70b-versatile", "llama-3.1-70b-versatile", "mixtral-8x7b-32768", "llama3-8b-8192"]

def obtener_modelos(client: Groq) -> list:
    activos = [m.id for m in client.models.list().data]
    texto = [m for m in activos if not any(x in m.lower() for x in ["whisper", "guard", "vision"])]
    ordenados = [p for p in PREFERENCIAS if p in texto]
    return ordenados + [m for m in texto if m not in ordenados]

def generar_contenido(client: Groq, prompt: str, modelos: list):
    for modelo in modelos:
        modo_minimo = False
        for intento in range(1, 4):
            try:
                log.info(f"🧠 Generando con {modelo}...")
                kwargs = dict(messages=[{"role": "user", "content": prompt}], model=modelo, temperature=0.7)
                if not modo_minimo: kwargs["response_format"] = {"type": "json_object"}
                resp = client.chat.completions.create(**kwargs)
                texto = re.sub(r"^```(?:json)?|```$", "", resp.choices[0].message.content.strip(), flags=re.MULTILINE)
                ini, fin = texto.find("{"), texto.rfind("}")
                datos = json.loads(texto[ini:fin + 1])
                for k in CLAVES_JSON_ESPERADAS:
                    if k not in datos: raise ValueError(f"Falta clave: {k}")
                return datos, modelo
            except Exception as e:
                msg = str(e).lower()
                if any(x in msg for x in ["400", "json", "format"]) and not modo_minimo:
                    modo_minimo = True; continue
                if intento < 3: time.sleep(ESPERA_BASE_SEG)
    raise ErrorFatal("Ningún modelo generó contenido.")

# ---------------------------------------------------------------------------
# 3. PEXELS API: IMÁGENES Y VÍDEOS REALES
# ---------------------------------------------------------------------------
def obtener_url_pexels(tipo="video"):
    api_key = os.environ.get("PEXELS_API_KEY")
    if not api_key: raise ErrorFatal("¡NECESITAS LA PEXELS_API_KEY PARA TENER CALIDAD PROFESIONAL!")
    
    headers = {"Authorization": api_key}
    query = random.choice(["cargo ship", "logistics port", "shipping containers", "freight ship"])
    
    if tipo == "video":
        r = requests.get(f"https://api.pexels.com/videos/search?query={query}&orientation=portrait&per_page=10", headers=headers)
        videos = r.json().get("videos", [])
        if not videos: return None
        v = random.choice(videos)
        archivos = sorted([f for f in v["video_files"] if f["height"] >= 1080], key=lambda x: x["height"])
        return archivos[0]["link"] if archivos else v["video_files"][0]["link"]
    else:
        r = requests.get(f"https://api.pexels.com/v1/search?query={query}&orientation=portrait&per_page=10", headers=headers)
        fotos = r.json().get("photos", [])
        if not fotos: return None
        return random.choice(fotos)["src"]["large2x"]

# ---------------------------------------------------------------------------
# 4. PINES PROFESIONALES (FOTO REAL DE FONDO)
# ---------------------------------------------------------------------------
def fabricar_pines(contenido: dict):
    os.makedirs(CARPETA_PINES, exist_ok=True)
    pines = contenido.get("pinterest_pins", [])[:3]
    
    for i, p in enumerate(pines):
        texto = p.get("text_on_image", "LOGÍSTICA B2B")
        img_url = obtener_url_pexels(tipo="foto")
        
        # Descargar foto
        r = requests.get(img_url, stream=True)
        img = Image.open(r.raw).convert("RGB").resize((1000, 1500), Image.ANTIALIAS)
        
        # Oscurecer fondo (estética premium)
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(0.4)
        
        d = ImageDraw.Draw(img)
        
        # Tipografía
        try: font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 90)
        except: font = ImageFont.load_default()
        
        # Separar texto en varias líneas
        words = texto.upper().split()
        lines = []
        current = ""
        for w in words:
            if d.textlength(current + w, font=font) < 850: current += w + " "
            else: lines.append(current); current = w + " "
        lines.append(current)
        
        # Dibujar texto
        y = 500
        for line in lines:
            w = d.textlength(line, font=font)
            d.text(((1000-w)/2, y), line, font=font, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0,0,0))
            y += 110
            
        # Línea decorativa corporativa
        d.rectangle([400, y+50, 600, y+60], fill=(255, 196, 0))
            
        img.save(os.path.join(CARPETA_PINES, f"pin_{i+1}.jpg"), quality=95)
        log.info(f"✅ Pin {i+1} HD generado.")

# ---------------------------------------------------------------------------
# 5. VÍDEO CON SUBTÍTULOS ESTILO HORMOZI
# ---------------------------------------------------------------------------
async def generar_voz(texto: str, archivo: str) -> list:
    com = edge_tts.Communicate(texto, VOZ, rate=VELOCIDAD_VOZ)
    await com.save(archivo)

def generar_subtitulos_dinamicos(texto: str, duracion_total: float):
    # Ritmo rápido: 1 o 2 palabras por pantalla máximo
    palabras = texto.split()
    chunks = [" ".join(palabras[i:i+2]) for i in range(0, len(palabras), 2)]
    dur_chunk = duracion_total / len(chunks)
    
    clips = []
    for i, chunk in enumerate(chunks):
        # Texto gigantesco amarillo/blanco
        color = 'yellow' if i % 3 == 0 else 'white'
        txt = TextClip(chunk.upper(), fontsize=110, color=color, font='DejaVu-Sans-Bold', stroke_color='black', stroke_width=4, method='caption', size=(900, None))
        txt = txt.set_position(('center', 'center')).set_duration(dur_chunk).set_start(i * dur_chunk)
        clips.append(txt)
    return clips

def fabricar_video_mp4(script_texto: str):
    audio_path, bg_path = "temp_voice.mp3", "temp_bg.mp4"
    asyncio.run(generar_voz(script_texto, audio_path))
    audio_clip = AudioFileClip(audio_path)
    duracion = audio_clip.duration
    
    log.info("🎬 Descargando metraje HD de Pexels...")
    url_video = obtener_url_pexels("video")
    r = requests.get(url_video, stream=True)
    with open(bg_path, "wb") as f:
        for chunk in r.iter_content(1024*256): f.write(chunk)
        
    video_fondo = VideoFileClip(bg_path).resize(height=1920).crop(x_center=960, width=1080)
    
    # Buclear si es corto
    if video_fondo.duration < duracion:
        reps = int(duracion / video_fondo.duration) + 1
        video_fondo = concatenate_videoclips([video_fondo] * reps)
        
    video_fondo = video_fondo.subclip(0, duracion).set_audio(audio_clip)
    
    # Añadir oscurecimiento leve para que los subtítulos destaquen
    oscuro = ColorClip((1080, 1920), color=(0, 0, 0)).set_opacity(0.3).set_duration(duracion)
    
    clips_subs = generar_subtitulos_dinamicos(script_texto, duracion)
    
    log.info("⚙️ Renderizando MP4 Premium...")
    final = CompositeVideoClip([video_fondo, oscuro] + clips_subs)
    final.write_videofile(ARCHIVO_VIDEO, fps=24, codec="libx264", audio_codec="aac", preset="ultrafast", logger=None)
    
    final.close(); audio_clip.close(); video_fondo.close()
    for f in [audio_path, bg_path]: os.remove(f)

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    log.info("🚀 Arrancando la Mega Máquina B2B Premium...")
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key: raise ErrorFatal("Falta GROQ_API_KEY.")

    df = validar_dataframe(descargar_csv(CSV_URL))
    prod = df.sample(n=1).iloc[0]
    nombre, problemas, palabra = prod["NOMBRE_PRODUCTO"], prod["PROBLEMAS_QUE_RESUELVE"], prod["PALABRA_CLAVE_MANYCHAT"]

    client = Groq(api_key=api_key, max_retries=0)
    modelos = obtener_modelos(client)

    prompt = f"""
    Actúa como copywriter B2B experto en logística. Vende: '{nombre}'. Soluciona: '{problemas}'.
    El video_script debe durar máximo 30 segundos (unas 60 palabras).
    1) Gancho agresivo que frene el scroll.
    2) Problema con datos reales.
    3) Solución (tu producto).
    4) CTA: "Comenta la palabra {palabra}".
    Devuelve JSON estricto con: video_script (string puro), tiktok_data, ig_reel_data, youtube_seo, pinterest_pins (3 objetos con 'text_on_image' de máx 5 palabras), linkedin_post.
    """

    contenido, mod = generar_contenido(client, prompt, modelos)
    
    with open(ARCHIVO_JSON, "w", encoding="utf-8") as f: json.dump(contenido, f, indent=2)
    with open(ARCHIVO_TXT, "w", encoding="utf-8") as f:
        f.write(f"=== TIKTOK ===\n{contenido.get('tiktok_data', {}).get('caption', '')}\n\n")
        f.write(f"=== LINKEDIN ===\n{contenido.get('linkedin_post', '')}\n")
    
    guion = str(contenido.get("video_script", "")).strip()
    
    fabricar_pines(contenido)
    fabricar_video_mp4(guion)
    log.info("🏁 Pipeline completado. Contenido listo para publicar.")

if __name__ == "__main__":
    main()
