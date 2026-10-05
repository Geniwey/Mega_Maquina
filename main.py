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

VOZ = "es-ES-AlvaroNeural"
VELOCIDAD_VOZ = "+6%"
SEGUNDOS_POR_PLANO = 2.5  # <--- RITMO TIKTOK: Cambia de vídeo cada 2.5 segundos
NUM_FONDOS = 4  # Cantidad de vídeos distintos a descargar

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
# 2. IA: RADAR DE MODELOS BLINDADO
# ---------------------------------------------------------------------------
def obtener_modelos_candidatos(client: Groq) -> list:
    activos = [m.id for m in client.models.list().data]
    texto = [m for m in activos if not any(x in m.lower() for x in ["whisper", "guard", "vision", "llava"])]
    pref = ["llama-3.3-70b-versatile", "llama-3.1-70b-versatile", "mixtral-8x7b-32768", "llama3-8b-8192"]
    ordenados = [p for p in pref if p in texto]
    return ordenados + [m for m in texto if m not in ordenados]

def extraer_json(texto: str) -> dict:
    texto = re.sub(r"<think>.*?</think>", "", texto, flags=re.DOTALL).strip()
    texto = re.sub(r"^```(?:json)?|```$", "", texto.strip(), flags=re.MULTILINE).strip()
    try: return json.loads(texto)
    except json.JSONDecodeError:
        ini, fin = texto.find("{"), texto.rfind("}")
        if ini == -1 or fin == -1: raise ValueError("La respuesta no contiene JSON.")
        return json.loads(texto[ini:fin + 1])

def generar_contenido(client: Groq, prompt: str, modelos: list):
    for modelo in modelos:
        modo_minimo = False
        for intento in range(1, 4):
            try:
                log.info(f"🧠 Generando con {modelo}...")
                kwargs = dict(messages=[{"role": "user", "content": prompt}], model=modelo, temperature=0.7)
                if not modo_minimo: kwargs["response_format"] = {"type": "json_object"}
                resp = client.chat.completions.create(**kwargs)
                datos = extraer_json(resp.choices[0].message.content or "")
                for k in CLAVES_JSON_ESPERADAS:
                    if k not in datos: raise ValueError(f"Falta clave JSON: {k}")
                log.info(f"✅ JSON validado con {modelo}.")
                return datos, modelo
            except Exception as e:
                msg = str(e).lower()
                if any(x in msg for x in ["400", "json", "format"]) and not modo_minimo:
                    modo_minimo = True; continue
                if intento < 3: time.sleep(ESPERA_BASE_SEG)
    raise ErrorFatal("Ningún modelo de Groq pudo generar el contenido.")

# ---------------------------------------------------------------------------
# 3. TEXTOS PARA IMAGEN Y PINES (PILLOW)
# ---------------------------------------------------------------------------
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
    d
