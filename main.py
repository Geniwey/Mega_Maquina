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
from moviepy.editor import VideoFileClip, AudioFileClip, VideoClip

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
            df = pd.read_csv(io.StringIO(resp.content.decode("utf-8")), sep=None, engine="python")
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
# 2. RADAR DE MODELOS (LECTURA EN VIVO + CADENA DE RESPALDO)
# ---------------------------------------------------------------------------
EXCLUIR_EN_NOMBRE = (
    "whisper", "guard", "safeguard", "vision", "llava",
    "orpheus", "tts", "playai", "embed", "distil-whisper",
)

PREFERENCIAS = [
    "openai/gpt-oss-120b",
    "llama-3.3-70b-versatile",
    "qwen/qwen3",
    "openai/gpt-oss-20b",
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
    "llama",
    "mixtral",
    "gemma",
]

ERRORES_MODELO_NO_VALIDO = (
    "model_terms_required",
    "requires terms acceptance",
    "model_not_found",
    "does not exist",
    "decommissioned",
    "not supported",
    "permission",
    "no access",
)


def obtener_modelos_candidatos(client: Groq) -> list:
    log.info("📡 Escaneando modelos disponibles HOY en Groq...")
    try:
        activos = [m.id for m in client.models.list().data]
    except Exception as e:
        raise ErrorFatal(f"Fallo en el radar de modelos: {e}")

    log.info(f"Modelos detectados online: {activos}")

    texto = [m for m in activos if not any(x in m.lower() for x in EXCLUIR_EN_NOMBRE)]
    if not texto:
        raise ErrorFatal("Groq no devuelve modelos de texto válidos.")

    ordenados = []
    for pref in PREFERENCIAS:
        for m in texto:
            if pref in m.lower() and m not in ordenados:
                ordenados.append(m)

    resto = [m for m in texto if m not in ordenados]
    resto.sort(key=lambda m: "allam" in m.lower())
    ordenados += resto

    log.info(f"⭐ Orden de modelos a probar: {ordenados}")
    return ordenados


def extraer_json(texto: str) -> dict:
    texto = re.sub(r"<think>.*?</think>", "", texto, flags=re.DOTALL).strip()
    texto = re.sub(r"^```(?:json)?|```$", "", texto.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(texto)
    except json.JSONDecodeError:
        ini, fin = texto.find("{"), texto.rfind("}")
        if ini == -1 or fin == -1:
            raise ValueError("La respuesta no contiene JSON.")
        return json.loads(texto[ini:fin + 1])


def generar_contenido(client: Groq, prompt: str, modelos: list):
    """Devuelve (datos, modelo_usado). Prueba cada modelo hasta que uno funcione."""
    for modelo in modelos:
        usar_json_mode = True
        for intento in range(1, MAX_REINTENTOS + 1):
            try:
                log.info(f"🧠 Generando copy con {modelo} (intento {intento}/{MAX_REINTENTOS})...")
                kwargs = dict(
                    messages=[
                        {"role": "system", "content": "Responde SOLO con un objeto JSON válido, sin texto extra."},
                        {"role": "user", "content": prompt},
                    ],
                    model=modelo,
                    temperature=0.7,
                    max_completion_tokens=6000,
                )
                if usar_json_mode:
                    kwargs["response_format"] = {"type": "json_object"}

                resp = client.chat.completions.create(**kwargs)
                datos = extraer_json(resp.choices[0].message.content or "")

                for k in CLAVES_JSON_ESPERADAS:
                    if k not in datos:
                        raise ValueError(f"Falta clave JSON: {k}")

                log.info(f"✅ JSON validado correctamente con {modelo}.")
                return datos, modelo

            except Exception as e:
                msg = str(e).lower()
                log.warning(f"⚠️ Error en Groq ({modelo}): {e}")

                if any(x in msg for x in ERRORES_MODELO_NO_VALIDO):
                    log.warning(f"⏭️ Modelo {modelo} descartado, probando el siguiente...")
                    break

                if "response_format" in msg and usar_json_mode:
                    log.warning("↩️ Este modelo no admite json_object; reintento sin él.")
                    usar_json_mode = False
                    continue

                if intento < MAX_REINTENTOS:
                    time.sleep(ESPERA_BASE_SEG * (2 ** (intento - 1)))

    raise ErrorFatal("Ningún modelo de Groq pudo generar el contenido.")


# ---------------------------------------------------------------------------
# 3. LA FÁBRICA VISUAL (ANTI-BLOQUEOS)
# ---------------------------------------------------------------------------
async def generar_voz_audio(texto: str, archivo_salida: str):
    log.info("🎙️ Sintetizando voz en off profesional (Edge TTS)...")
    comunicador = edge_tts.Communicate(texto, "es-ES-AlvaroNeural")
    await comunicador.save(archivo_salida)
    log.info("✅ Audio de voz generado.")


def a_vertical(clip, ancho=1080, alto=1920):
    """Escala y recorta cualquier vídeo a formato vertical 1080x1920."""
    clip = clip.resize(height=alto)
    if clip.w < ancho:
        clip = clip.resize(width=ancho)
    return clip.crop(x_center=clip.w / 2, y_center=clip.h / 2, width=ancho, height=alto)


def descargar_fondo_pexels(ruta: str):
    """Descarga un vídeo vertical de Pexels. Devuelve la ruta o None si no se puede."""
    api_key = os.environ.get("PEXELS_API_KEY")
    if not api_key:
        log.info("ℹ️ Sin PEXELS_API_KEY: se usará el fondo animado generado.")
        return None

    consultas = ["cargo ship", "shipping containers port", "logistics warehouse", "container terminal"]
    random.shuffle(consultas)

    for consulta in consultas:
        try:
            log.info(f"🔎 Buscando fondo en Pexels: '{consulta}'...")
            r = requests.get(
                "https://api.pexels.com/videos/search",
                headers={"Authorization": api_key},
                params={"query": consulta, "orientation": "portrait", "per_page": 15},
                timeout=30,
            )
            r.raise_for_status()
            videos = r.json().get("videos", [])
            random.shuffle(videos)

            for v in videos:
                candidatos = [
                    f for f in v.get("video_files", [])
                    if f.get("file_type") == "video/mp4"
                    and (f.get("height") or 0) >= (f.get("width") or 0)
                    and (f.get("height") or 0) >= 1080
                ]
                if not candidatos:
                    continue
                candidatos.sort(key=lambda f: f["height"])  # el más ligero que cumple 1080p
                enlace = candidatos[0]["link"]

                with requests.get(enlace, stream=True, timeout=60) as resp:
                    resp.raise_for_status()
                    with open(ruta, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=1024 * 256):
                            if chunk:
                                f.write(chunk)
                log.info("✅ Fondo descargado de Pexels.")
                return ruta
        except Exception as e:
            log.warning(f"⚠️ Pexels falló con '{consulta}': {e}")
    return None


def fondo_animado(duracion: float):
    """Fondo vertical animado generado por código (sin descargas, no se puede bloquear)."""
    w, h = 270, 480
    yy, xx = np.mgrid[0:h, 0:w]

    def make_frame(t):
        v = (np.sin(xx / 60 + t * 0.6) + np.sin(yy / 90 - t * 0.4) + np.sin((xx + yy) / 80 + t * 0.5)) / 3
        v = (v + 1) / 2
        r = 15 + 25 * v
        g = 30 + 60 * v
        b = 60 + 110 * v
        return np.dstack([r, g, b]).astype("uint8")

    return VideoClip(make_frame, duration=duracion).resize((1080, 1920))


def fabricar_video_mp4(script_texto: str):
    audio_path = "temp_voice.mp3"
    bg_path = "temp_bg.mp4"
    asyncio.run(generar_voz_audio(script_texto, audio_path))

    log.info("🎬 Preparando fondo y renderizando vídeo MP4 con MoviePy...")

    audio_clip = AudioFileClip(audio_path)
    duracion = audio_clip.duration

    video_fondo = None
    ruta = descargar_fondo_pexels(bg_path)
    if ruta:
        try:
            video_fondo = a_vertical(VideoFileClip(ruta).without_audio())
            video_fondo = video_fondo.loop(duration=duracion)
            log.info("✅ Fondo de Pexels adaptado a vertical.")
        except Exception as e:
            log.warning(f"⚠️ No se pudo procesar el fondo de Pexels: {e}")
            video_fondo = None

    if video_fondo is None:
        video_fondo = fondo_animado(duracion)
        log.info("✅ Fondo animado generado por código.")

    video_fondo = video_fondo.set_audio(audio_clip)

    video_fondo.write_videofile(
        ARCHIVO_VIDEO,
        fps=24,
        codec="libx264",
        audio_codec="aac",
        preset="ultrafast",
        logger=None,
    )

    audio_clip.close()
    video_fondo.close()
    for p in (audio_path, bg_path):
        if os.path.exists(p):
            os.remove(p)

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

    modelos = obtener_modelos_candidatos(client)

    prompt = f"""
Actúa como un copywriter B2B experto en logística y comercio internacional.
Vende este producto: '{nombre}'. Problemas que soluciona: '{problemas}'.
REGLAS: Cero niños, dolor de e-commerce/importadores real, usa jerga (Demurrage, DUA, Incoterms), CTA duro pidiendo comentar '{palabra_clave}'.
Devuelve estrictamente un JSON con estas claves:
video_script, tiktok_data (caption, hashtags), ig_reel_data (caption, hashtags), youtube_seo (title, description), pinterest_pins (array de objetos con text_on_image), linkedin_post.
"""

    contenido, modelo_usado = generar_contenido(client, prompt, modelos)
    contenido["_meta"] = {"producto": nombre, "enlace": enlace, "modelo_usado": modelo_usado}

    with open(ARCHIVO_JSON, "w", encoding="utf-8") as f:
        json.dump(contenido, f, ensure_ascii=False, indent=2)

    guion_voz = contenido["video_script"]
    if not isinstance(guion_voz, str):
        guion_voz = json.dumps(guion_voz, ensure_ascii=False)
    fabricar_video_mp4(guion_voz)

    log.info("🏁 Pipeline completo de texto y vídeo finalizado.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.error(f"❌ ERROR: {e}")
        sys.exit(1)
