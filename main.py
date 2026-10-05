import os
import io
import sys
import json
import time
import asyncio
import logging
import subprocess
import unicodedata
import requests
import pandas as pd
from groq import Groq
import edge_tts

# --- BLINDAJE ANTIALIAS PARA MOVIEPY ---
import PIL.Image
if not hasattr(PIL.Image, 'ANTIALIAS'):
    try:
        PIL.Image.ANTIALIAS = PIL.Image.Resampling.LANCZOS
    except AttributeError:
        PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

from moviepy.editor import VideoFileClip, AudioFileClip, ImageClip, TextClip, CompositeVideoClip, concatenate_videoclips

# ---------------------------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------------------------
SHEET_ID = "10gJJCIlPzCHYEfPYPKgT3-xjUtghbIaYpdR87Da4JPQ"
CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv"
COLUMNAS_REQUERIDAS = ["NOMBRE_PRODUCTO", "PROBLEMAS_QUE_RESUELVE", "PALABRA_CLAVE_MANYCHAT", "ENLACE_HOTMART"]
ARCHIVO_JSON = "contenido_hoy.json"
ARCHIVO_TXT = "TEXTOS_PARA_REDES.txt"
ARCHIVO_VIDEO = "video_final.mp4"
MAX_REINTENTOS = 3
ESPERA_BASE_SEG = 5

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", stream=sys.stdout)
log = logging.getLogger("mega_maquina")

class ErrorFatal(Exception): pass

# ---------------------------------------------------------------------------
# 1. LECTURA Y VALIDACIÓN
# ---------------------------------------------------------------------------
def limpiar_texto(valor) -> str:
    s = str(valor)
    s = "".join(ch for ch in s if unicodedata.category(ch) not in ("Cf", "Cc"))
    return s.replace("\u00a0", " ").strip()

def descargar_csv(url: str) -> pd.DataFrame:
    for intento in range(1, MAX_REINTENTOS + 1):
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            return pd.read_csv(io.StringIO(resp.content.decode("utf-8")), sep=None, engine='python')
        except Exception as e:
            if intento < MAX_REINTENTOS: time.sleep(ESPERA_BASE_SEG)
    raise ErrorFatal("Fallo crítico al leer el CSV de Google Sheets.")

def validar_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: raise ErrorFatal("Excel vacío.")
    df.columns = [limpiar_texto(c) for c in df.columns]
    for c in COLUMNAS_REQUERIDAS:
        if c not in df.columns: raise ErrorFatal(f"Falta columna: {c}")
    df = df[COLUMNAS_REQUERIDAS].dropna().copy()
    for col in COLUMNAS_REQUERIDAS: df[col] = df[col].map(limpiar_texto)
    df = df[(df[COLUMNAS_REQUERIDAS] != "").all(axis=1)]
    if df.empty: raise ErrorFatal("Sin filas válidas.")
    return df

# ---------------------------------------------------------------------------
# 2. INTELIGENCIA ARTIFICIAL
# ---------------------------------------------------------------------------
def obtener_mejor_modelo(client: Groq) -> str:
    modelos_activos = [m.id for m in client.models.list().data]
    modelos_texto = [m for m in modelos_activos if "whisper" not in m.lower() and "guard" not in m.lower() and "vision" not in m.lower()]
    for pref in ["llama-3.3-70b-versatile", "llama-3.1-70b-versatile", "mixtral-8x7b-32768"]:
        if pref in modelos_texto: return pref
    return modelos_texto[0] if modelos_texto else "llama3-8b-8192"

def generar_contenido(client: Groq, prompt: str, modelo: str) -> dict:
    for _ in range(MAX_REINTENTOS):
        try:
            res = client.chat.completions.create(messages=[{"role": "user", "content": prompt}], model=modelo, response_format={"type": "json_object"}, temperature=0.7)
            return json.loads(res.choices[0].message.content)
        except Exception:
            time.sleep(ESPERA_BASE_SEG)
    raise ErrorFatal("Fallo en la IA.")

# ---------------------------------------------------------------------------
# 3. EXPORTAR TEXTOS PARA EL USUARIO
# ---------------------------------------------------------------------------
def crear_documento_textos(datos: dict):
    log.info("📝 Generando archivo de textos listos para copiar y pegar...")
    with open(ARCHIVO_TXT, "w", encoding="utf-8") as f:
        f.write("=========================================\n")
        f.write("📱 TIKTOK & INSTAGRAM REELS\n")
        f.write("=========================================\n")
        f.write(f"TÍTULO: {datos.get('tiktok_data', {}).get('caption', '')}\n")
        f.write(f"HASHTAGS: {datos.get('tiktok_data', {}).get('hashtags', '')}\n\n")
        
        f.write("=========================================\n")
        f.write("📍 PINTEREST\n")
        f.write("=========================================\n")
        for i, pin in enumerate(datos.get('pinterest_pins', [])):
            f.write(f"PIN {i+1}: {pin.get('text_on_image', '')}\n")
        f.write("\n")
        
        f.write("=========================================\n")
        f.write("💼 LINKEDIN (Post Profesional)\n")
        f.write("=========================================\n")
        f.write(f"{datos.get('linkedin_post', '')}\n\n")
        
        f.write("=========================================\n")
        f.write("▶️ YOUTUBE SHORTS\n")
        f.write("=========================================\n")
        f.write(f"TÍTULO: {datos.get('youtube_seo', {}).get('title', '')}\n")
        f.write(f"DESCRIPCIÓN: {datos.get('youtube_seo', {}).get('description', '')}\n")

# ---------------------------------------------------------------------------
# 4. FÁBRICA VISUAL PROFESIONAL
# ---------------------------------------------------------------------------
async def generar_voz(texto: str, archivo: str):
    await edge_tts.Communicate(texto, "es-ES-AlvaroNeural").save(archivo)

def generar_subtitulos(texto: str, duracion_total: float):
    # Rompe el guion en trozos cortos para leer en pantalla
    palabras = texto.split()
    chunks = [" ".join(palabras[i:i+4]) for i in range(0, len(palabras), 4)]
    dur_chunk = duracion_total / len(chunks)
    
    clips = []
    for i, chunk in enumerate(chunks):
        # Letras blancas, gordas, con contorno negro
        txt_clip = TextClip(chunk, fontsize=65, color='white', font='DejaVu-Sans-Bold', 
                            stroke_color='black', stroke_width=2.5, method='caption', size=(900, None))
        txt_clip = txt_clip.set_position(('center', 'center')).set_duration(dur_chunk).set_start(i * dur_chunk)
        clips.append(txt_clip)
    return clips

def fabricar_video_mp4(script_texto: str):
    audio_path = "temp_voice.mp3"
    bg_path = "temp_bg.mp4"
    img_path = "temp_bg.jpg"
    video_fondo = None
    
    asyncio.run(generar_voz(script_texto, audio_path))
    audio_clip = AudioFileClip(audio_path)
    duracion = audio_clip.duration
    
    # 1. Intentar descargar vídeo de barcos
    try:
        import yt_dlp
        opts = {'format': 'bestvideo[ext=mp4]/best', 'outtmpl': bg_path, 'quiet': True}
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.extract_info("ytsearch1:cargo ship container port drone aerial HD stock footage no text short", download=True)
        video_fondo = VideoFileClip(bg_path)
    except:
        # 2. PLAN B: Si falla, descarga foto de puerto y LE METE MOVIMIENTO (ZOOM)
        r = requests.get("https://images.unsplash.com/photo-1586528116311-ad8dd3c8310d?q=80&w=1280", stream=True)
        with open(img_path, "wb") as f:
            for chunk in r.iter_content(1024): f.write(chunk)
        # Aquí está la magia: Transforma la foto en un vídeo en movimiento (efecto dron)
        video_fondo = ImageClip(img_path).resize(lambda t: 1 + 0.015 * t).set_duration(duracion)

    # Ajustar tiempo del fondo
    if video_fondo.duration < duracion:
        reps = int(duracion / video_fondo.duration) + 1
        video_fondo = concatenate_videoclips([video_fondo] * reps)
    video_fondo = video_fondo.subclip(0, duracion).set_audio(audio_clip)
    
    # Añadir los subtítulos dinámicos por encima del vídeo
    clips_subtitulos = generar_subtitulos(script_texto, duracion)
    video_final = CompositeVideoClip([video_fondo] + clips_subtitulos)
    
    video_final.write_videofile(ARCHIVO_VIDEO, fps=24, codec="libx264", audio_codec="aac", preset="ultrafast", logger=None)
    
    # Limpieza
    audio_clip.close(); video_fondo.close(); video_final.close()
    for f in [audio_path, bg_path, img_path]:
        if os.path.exists(f): os.remove(f)

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    log.info("🚀 Arrancando la Mega Máquina Profesional...")
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key: raise ErrorFatal("Falta GROQ_API_KEY")

    df = validar_dataframe(descargar_csv(CSV_URL))
    prod = df.sample(n=1).iloc[0]
    
    nombre, problemas, palabra = prod["NOMBRE_PRODUCTO"], prod["PROBLEMAS_QUE_RESUELVE"], prod["PALABRA_CLAVE_MANYCHAT"]
    log.info(f"🎯 Producto: {nombre}")

    client = Groq(api_key=api_key, max_retries=0)
    modelo = obtener_mejor_modelo(client)

    prompt = f"""
    Actúa como copywriter B2B experto en logística. Vende: '{nombre}'. Soluciona: '{problemas}'.
    REGLAS: Dolor de e-commerce real, jerga (Demurrage, DUA, Incoterms), CTA pidiendo comentar '{palabra}'.
    Devuelve STRICTAMENTE JSON: video_script (texto puro sin diccionario), tiktok_data (caption, hashtags), ig_reel_data (caption, hashtags), youtube_seo (title, description), pinterest_pins (lista con text_on_image), linkedin_post.
    """
    
    contenido = generar_contenido(client, prompt, modelo)
    
    # Crear archivo de texto para el usuario
    crear_documento_textos(contenido)

    # Asegurar que el guion es texto puro para la voz
    guion = contenido.get("video_script", "")
    if isinstance(guion, dict): guion = " ".join(str(v) for v in guion.values())
    elif isinstance(guion, list): guion = " ".join(str(x) for x in guion)
    
    fabricar_video_mp4(str(guion).strip())
    log.info("🏁 Pipeline completado.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.error(f"❌ ERROR FATAL: {e}")
        sys.exit(1)
