import os
import io
import sys
import json
import time
import asyncio
import logging
import unicodedata
import requests
import pandas as pd
from groq import Groq
import edge_tts
from moviepy.editor import VideoFileClip, AudioFileClip, ColorClip

# ---------------------------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------------------------
SHEET_ID = "10gJJCIlPzCHYEfPYPKgT3-xjUtghbIaYpdR87Da4JPQ"
CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv"

COLUMNAS_REQUERIDAS = [
    "NOMBRE_PRODUCTO",
    "PROBLEMAS_QUE_RESUELVE",
    "PALABRA_CLAVE_MANYCHAT",
    "ENLACE_HOTMART",
]

ARCHIVO_JSON = "contenido_hoy.json"
ARCHIVO_VIDEO = "video_final.mp4"

MAX_REINTENTOS = 3
ESPERA_BASE_SEG = 5
TIMEOUT_CSV_SEG = 30
TIMEOUT_GROQ_SEG = 60

CLAVES_JSON_ESPERADAS = [
    "video_script",
    "tiktok_data",
    "ig_reel_data",
    "youtube_seo",
    "pinterest_pins",
    "linkedin_post",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("mega_maquina")

class ErrorFatal(Exception):
    pass

# ---------------------------------------------------------------------------
# 1. LECTURA Y VALIDACIÓN DEL CSV
# ---------------------------------------------------------------------------
def limpiar_texto(valor) -> str:
    s = str(valor)
    s = "".join(ch for ch in s if unicodedata.category(ch) not in ("Cf", "Cc"))
    s = s.replace("\u00a0", " ")
    return s.strip()

def descargar_csv(url: str) -> pd.DataFrame:
    for intento in range(1, MAX_REINTENTOS + 1):
        try:
            log.info(f"📥 Descargando base de datos B2B (intento {intento}/{MAX_REINTENTOS})...")
            resp = requests.get(url, timeout=TIMEOUT_CSV_SEG)
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.content.decode("utf-8")), sep=None, engine='python')
            log.info(f"✅ CSV descargado: {len(df)} filas crudas.")
            return df
        except Exception as e:
            log.warning(f"⚠️ Fallo leyendo CSV: {e}")
            if intento < MAX_REINTENTOS:
                time.sleep(ESPERA_BASE_SEG * (2 ** (intento - 1)))
    raise ErrorFatal("No se pudo leer el CSV.")

def validar_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ErrorFatal("El DataFrame está vacío.")

    df.columns = [limpiar_texto(c) for c in df.columns]

    for c in COLUMNAS_REQUERIDAS:
        if c not in df.columns:
            raise ErrorFatal(f"Falta columna obligatoria: '{c}'. Columnas detectadas: {list(df.columns)}")

    df = df[COLUMNAS_REQUERIDAS].dropna().copy()
    for col in COLUMNAS_REQUERIDAS:
        df[col] = df[col].map(limpiar_texto)

    df = df[(df[COLUMNAS_REQUERIDAS] != "").all(axis=1)]

    if df.empty:
        raise ErrorFatal("No quedan filas válidas tras la limpieza.")

    log.info(f"✅ {len(df)} fila(s) válida(s) tras validar.")
    return df

# ---------------------------------------------------------------------------
# 2. RADAR DE MODELOS (BLINDADO)
# ---------------------------------------------------------------------------
def obtener_mejor_modelo(client: Groq) -> str:
    log.info("📡 Escaneando modelos disponibles en Groq...")
    try:
        modelos_activos = [m.id for m in client.models.list().data]
        
        # Lista estricta, si no están, usaremos el comodín seguro.
        preferencias = [
            "llama-3.3-70b-versatile",
            "llama-3.1-70b-versatile",
            "llama3-70b-8192",
            "mixtral-8x7b-32768"
        ]
        
        for pref in preferencias:
            if pref in modelos_activos:
                log.info(f"⭐ Radar fijado en el mejor modelo: {pref}")
                return pref
                
        log.warning("⚠️ No se encontró la élite. Forzando modelo estándar seguro: llama3-8b-8192")
        return "llama3-8b-8192"
    except Exception as e:
        log.error(f"❌ Fallo en el radar de modelos: {e}. Forzando fallback.")
        return "llama3-8b-8192"

def generar_contenido(client: Groq, prompt: str, modelo_elegido: str) -> dict:
    for intento in range(1, MAX_REINTENTOS + 1):
        try:
            log.info(f"🧠 Generando copy con {modelo_elegido} (intento {intento}/{MAX_REINTENTOS})...")
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=modelo_elegido,
                response_format={"type": "json_object"},
                temperature=0.7,
            )
            texto = chat_completion.choices[0].message.content
            datos = json.loads(texto)
            for k in CLAVES_JSON_ESPERADAS:
                if k not in datos:
                    raise ValueError(f"Falta clave JSON: {k}")
            log.info("✅ JSON validado correctamente.")
            return datos
        except Exception as e:
            log.warning(f"⚠️ Error en Groq: {e}")
            if intento < MAX_REINTENTOS:
                time.sleep(ESPERA_BASE_SEG * (2 ** (intento - 1)))
    raise ErrorFatal("Groq falló tras varios reintentos.")

# ---------------------------------------------------------------------------
# 3. LA FÁBRICA VISUAL (ANTI-BLOQUEOS)
# ---------------------------------------------------------------------------
async def generar_voz_audio(texto: str, archivo_salida: str):
    log.info("🎙️ Sintetizando voz en off profesional (Edge TTS)...")
    comunicador = edge_tts.Communicate(texto, "es-ES-AlvaroNeural")
    await comunicador.save(archivo_salida)
    log.info("✅ Audio de voz generado.")

def fabricar_video_mp4(script_texto: str):
    audio_path = "temp_voice.mp3"
    asyncio.run(generar_voz_audio(script_texto, audio_path))
    
    log.info("🎬 Descargando fondo y renderizando vídeo MP4 con MoviePy...")
    
    bg_url = "https://assets.mixkit.co/videos/preview/mixkit-cargo-ship-in-the-sea-41584-large.mp4"
    bg_path = "temp_bg.mp4"
    
    audio_clip = AudioFileClip(audio_path)
    duracion = audio_clip.duration
    
    video_fondo = None
    
    try:
        # Intento de descarga con máscara de navegador (User-Agent) para evitar el 403
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        resp = requests.get(bg_url, headers=headers, stream=True, timeout=30)
        resp.raise_for_status()
        with open(bg_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        video_fondo = VideoFileClip(bg_path).subclip(0, min(duracion, 60))
        log.info("✅ Vídeo de fondo descargado con éxito.")
    except Exception as e:
        log.warning(f"⚠️ Mixkit bloqueó la descarga. Entrando en modo rescate... {e}")
        # MODO RESCATE: Si la web bloquea a GitHub, creamos un fondo oscuro (gris marengo) para que la máquina no colapse.
        video_fondo = ColorClip(size=(1080, 1920), color=(30, 30, 30), duration=duracion)
        log.info("✅ Fondo de color sólido generado por el modo rescate.")
        
    video_fondo = video_fondo.set_audio(audio_clip)
    
    video_fondo.write_videofile(
        ARCHIVO_VIDEO,
        fps=24,
        codec="libx264",
        audio_codec="aac",
        preset="ultrafast",
        logger=None
    )
    
    audio_clip.close()
    video_fondo.close()
    if os.path.exists(audio_path): os.remove(audio_path)
    if os.path.exists(bg_path): os.remove(bg_path)
    
    log.info(f"✅ ¡Vídeo fabricado con éxito: {ARCHIVO_VIDEO}!")

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    log.info("🚀 Arrancando la Mega Máquina (Fase Vídeo)...")
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ErrorFatal("Falta GROQ_API_KEY en los Secrets.")

    df = validar_dataframe(descargar_csv(CSV_URL))
    prod = df.sample(n=1).iloc[0]
    
    nombre = prod["NOMBRE_PRODUCTO"]
    problemas = prod["PROBLEMAS_QUE_RESUELVE"]
    palabra_clave = prod["PALABRA_CLAVE_MANYCHAT"]
    enlace = prod["ENLACE_HOTMART"]

    log.info(f"🎯 Producto seleccionado: {nombre}")

    client = Groq(api_key=api_key, timeout=TIMEOUT_GROQ_SEG, max_retries=0)
    
    mejor_modelo = obtener_mejor_modelo(client)

    prompt = f"""
Actúa como un copywriter B2B experto en logística y comercio internacional.
Vende este producto: '{nombre}'. Problemas que soluciona: '{problemas}'.
REGLAS: Cero niños, dolor de e-commerce/importadores real, usa jerga (Demurrage, DUA, Incoterms), CTA duro pidiendo comentar '{palabra_clave}'.
Devuelve estrictamente un JSON con estas claves:
video_script, tiktok_data (caption, hashtags), ig_reel_data (caption, hashtags), youtube_seo (title, description), pinterest_pins (array de objetos con text_on_image), linkedin_post.
"""

    contenido = generar_contenido(client, prompt, mejor_modelo)
    contenido["_meta"] = {"producto": nombre, "enlace": enlace, "modelo_usado": mejor_modelo}

    with open(ARCHIVO_JSON, "w", encoding="utf-8") as f:
        json.dump(contenido, f, ensure_ascii=False, indent=2)

    guion_voz = contenido["video_script"]
    fabricar_video_mp4(guion_voz)

    log.info("🏁 Pipeline completo de texto y vídeo finalizado.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.error(f"❌ ERROR: {e}")
        sys.exit(1)
