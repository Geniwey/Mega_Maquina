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

from moviepy.editor import VideoFileClip, AudioFileClip, ImageClip, concatenate_videoclips

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
# 2. RADAR DE MODELOS PROFESIONAL
# ---------------------------------------------------------------------------
def obtener_mejor_modelo(client: Groq) -> str:
    log.info("📡 Escaneando inteligencia artificial disponible...")
    try:
        modelos_activos = [m.id for m in client.models.list().data]
        
        modelos_texto = [
            m for m in modelos_activos 
            if "whisper" not in m.lower() 
            and "guard" not in m.lower() 
            and "vision" not in m.lower()
            and "llava" not in m.lower()
        ]
        
        if not modelos_texto:
            raise ErrorFatal("Groq no devuelve modelos de texto válidos.")

        preferencias = [
            "llama-3.3-70b-versatile",
            "llama-3.1-70b-versatile",
            "mixtral-8x7b-32768"
        ]
        for pref in preferencias:
            if pref in modelos_texto:
                log.info(f"⭐ Radar fijado en el modelo principal: {pref}")
                return pref
                
        llamas = [m for m in modelos_texto if 'llama' in m.lower()]
        if llamas:
            mejor_llama = sorted(llamas, key=lambda x: "70b" in x.lower(), reverse=True)[0]
            log.info(f"⭐ Usando alternativa Llama detectada: {mejor_llama}")
            return mejor_llama
            
        qwens = [m for m in modelos_texto if 'qwen' in m.lower()]
        if qwens:
            log.info(f"⭐ Usando motor Qwen de alta capacidad: {qwens[0]}")
            return qwens[0]
                
        log.warning(f"⚠️ Usando motor estándar: {modelos_texto[0]}")
        return modelos_texto[0]
        
    except Exception as e:
        raise ErrorFatal(f"Fallo en el radar de modelos: {e}")

def generar_contenido(client: Groq, prompt: str, modelo_elegido: str) -> dict:
    for intento in range(1, MAX_REINTENTOS + 1):
        try:
            log.info(f"🧠 Generando copy experto con {modelo_elegido} (intento {intento}/{MAX_REINTENTOS})...")
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
            log.info("✅ JSON transaccional validado correctamente.")
            return datos
        except Exception as e:
            log.warning(f"⚠️ Error en Groq: {e}")
            if intento < MAX_REINTENTOS:
                time.sleep(ESPERA_BASE_SEG * (2 ** (intento - 1)))
    raise ErrorFatal("Groq falló tras varios reintentos.")

# ---------------------------------------------------------------------------
# 3. LA FÁBRICA VISUAL (100% BLINDADA)
# ---------------------------------------------------------------------------
async def generar_voz_audio(texto: str, archivo_salida: str):
    log.info("🎙️ Sintetizando voz en off B2B (Edge TTS)...")
    comunicador = edge_tts.Communicate(texto, "es-ES-AlvaroNeural")
    await comunicador.save(archivo_salida)
    log.info("✅ Audio de voz corporativo generado.")

def fabricar_video_mp4(script_texto: str):
    audio_path = "temp_voice.mp3"
    asyncio.run(generar_voz_audio(script_
