import os
import io
import re
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
from moviepy.editor import VideoFileClip, AudioFileClip, ColorClip, vfx

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
DURACION_MAX_VIDEO_SEG = 60

CLAVES_JSON_ESPERADAS = [
    "video_script",
    "tiktok_data",
    "ig_reel_data",
    "youtube_seo",
    "pinterest_pins",
    "linkedin_post",
]

# Modelos de chat preferidos, en orden. Se pueden forzar con GROQ_MODEL en el workflow.
MODELOS_PREFERIDOS = [
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "llama-3.1-8b-instant",
    "moonshotai/kimi-k2-instruct",
    "qwen/qwen3-32b",
]

# Cualquier modelo cuyo id contenga estas palabras NO es un modelo de chat de texto
PALABRAS_EXCLUIDAS = (
    "whisper", "orpheus", "playai", "tts", "guard", "safeguard",
    "embed", "vision", "canopylabs", "speech", "audio", "compound",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("mega_maquina")
logging.getLogger("httpx").setLevel(logging.WARNING)


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
            df = pd.read_csv(
                io.StringIO(resp.content.decode("utf-8-sig")),
                sep=None,
                engine="python",
            )
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
            raise ErrorFatal(
                f"Falta columna obligatoria: '{c}'. Columnas detectadas: {list(df.columns)}"
            )

    df = df[COLUMNAS_REQUERIDAS].dropna().copy()
    for col in COLUMNAS_REQUERIDAS:
        df[col] = df[col].map(limpiar_texto)

    df = df[(df[COLUMNAS_REQUERIDAS] != "").all(axis=1)]

    if df.empty:
        raise ErrorFatal("No quedan filas válidas tras la limpieza.")

    log.info(f"✅ {len(df)} fila(s) válida(s) tras validar.")
    return df


# ---------------------------------------------------------------------------
# 2. SELECCIÓN ROBUSTA DE MODELOS + GENERACIÓN
# ---------------------------------------------------------------------------
def es_modelo_de_chat(model_id: str) -> bool:
    m = model_id.lower()
    return not any(p in m for p in PALABRAS_EXCLUIDAS)


def obtener_candidatos(client: Groq) -> list:
    """Devuelve una lista ordenada de modelos a probar. Nunca devuelve una lista vacía."""
    candidatos = []

    forzado = os.environ.get("GROQ_MODEL", "").strip()
    if forzado:
        candidatos.append(forzado)

    activos = []
    try:
        activos = [m.id for m in client.models.list().data]
        log.info(f"📡 Groq reporta {len(activos)} modelos activos.")
    except Exception as e:
        log.warning(f"⚠️ No se pudo listar modelos ({e}). Usando lista fija.")

    activos_chat = [m for m in activos if es_modelo_de_chat(m)]

    # 1) Preferidos que estén activos
    for pref in MODELOS_PREFERIDOS:
        if pref in activos_chat and pref not in candidatos:
            candidatos.append(pref)

    # 2) Cualquier otro modelo de chat activo
    for m in activos_chat:
        if m not in candidatos:
            candidatos.append(m)

    # 3) Lista fija como red de seguridad final
    for pref in MODELOS_PREFERIDOS:
        if pref not in candidatos:
            candidatos.append(pref)

    log.info(f"🧭 Orden de candidatos: {candidatos[:6]}{'...' if len(candidatos) > 6 else ''}")
    return candidatos


def extraer_json(texto: str) -> dict:
    """Extrae un objeto JSON de la respuesta aunque venga con <think>, ```json o texto extra."""
    texto = re.sub(r"<think>.*?</think>", "", texto, flags=re.DOTALL).strip()
    texto = re.sub(r"^```(?:json)?\s*|\s*```$", "", texto, flags=re.MULTILINE).strip()
    try:
        return json.loads(texto)
    except json.JSONDecodeError:
        inicio, fin = texto.find("{"), texto.rfind("}")
        if inicio != -1 and fin > inicio:
            return json.loads(texto[inicio:fin + 1])
        raise


def _llamar_modelo(client: Groq, modelo: str, prompt: str, usar_json_mode: bool) -> dict:
    kwargs = dict(
        messages=[{"role": "user", "content": prompt}],
        model=modelo,
        temperature=0.7,
    )
    if usar_json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(**kwargs)
    texto = resp.choices[0].message.content or ""
    datos = extraer_json(texto)
    if not isinstance(datos, dict):
        raise ValueError("La respuesta no es un objeto JSON.")
    return datos


def generar_contenido(client: Groq, prompt: str) -> tuple:
    """Prueba modelo tras modelo hasta que uno funcione. Devuelve (datos, modelo_usado)."""
    candidatos = obtener_candidatos(client)

    for modelo in candidatos:
        usar_json_mode = True
        for intento in range(1, MAX_REINTENTOS + 1):
            try:
                log.info(f"🧠 Generando copy con {modelo} (intento {intento}/{MAX_REINTENTOS})...")
                datos = _llamar_modelo(client, modelo, prompt, usar_json_mode)

                if not str(datos.get("video_script", "")).strip() and not isinstance(
                    datos.get("video_script"), (list, dict)
                ):
                    raise ValueError("JSON sin 'video_script' utilizable.")

                faltan = [k for k in CLAVES_JSON_ESPERADAS if k not in datos]
                if faltan:
                    log.warning(f"⚠️ Faltaban claves {faltan}; se rellenan vacías.")
                    for k in faltan:
                        datos[k] = "" if k in ("video_script", "linkedin_post") else {}

                log.info(f"✅ JSON válido obtenido con {modelo}.")
                return datos, modelo

            except Exception as e:
                status = getattr(e, "status_code", None)
                msg = str(e)
                log.warning(f"⚠️ {modelo} falló (status={status}): {msg[:250]}")

                # Errores de modelo no usable: pasar al siguiente SIN reintentar
                if status in (400, 401, 403, 404) and "response_format" not in msg and "json" not in msg.lower():
                    log.warning(f"⏭️ {modelo} descartado, probando el siguiente.")
                    break

                # Modelo que no soporta JSON mode: reintenta sin él
                if status == 400 and usar_json_mode:
                    log.info("↩️ Reintentando sin response_format JSON...")
                    usar_json_mode = False
                    continue

                # Errores transitorios (429, 5xx, timeout, JSON roto): backoff y reintento
                if intento < MAX_REINTENTOS:
                    time.sleep(ESPERA_BASE_SEG * (2 ** (intento - 1)))

    raise ErrorFatal("Ningún modelo de Groq pudo generar el contenido.")


# ---------------------------------------------------------------------------
# 3. LA FÁBRICA VISUAL
# ---------------------------------------------------------------------------
async def generar_voz_audio(texto: str, archivo_salida: str):
    log.info("🎙️ Sintetizando voz en off profesional (Edge TTS)...")
    comunicador = edge_tts.Communicate(texto, "es-ES-AlvaroNeural")
    await comunicador.save(archivo_salida)
    log.info("✅ Audio de voz generado.")


def descargar_fondo(bg_path: str) -> bool:
    bg_url = "https://assets.mixkit.co/videos/preview/mixkit-cargo-ship-in-the-sea-41584-large.mp4"
    for intento in range(1, MAX_REINTENTOS + 1):
        try:
            resp = requests.get(bg_url, stream=True, timeout=60)
            resp.raise_for_status()
            with open(bg_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            if os.path.getsize(bg_path) > 10_000:
                return True
        except Exception as e:
            log.warning(f"⚠️ Fallo descargando fondo (intento {intento}): {e}")
            time.sleep(2)
    return False


def script_a_texto(script) -> str:
    """El modelo a veces devuelve el guion como lista o dict en vez de string."""
    if isinstance(script, str):
        return script.strip()
    if isinstance(script, list):
        return " ".join(script_a_texto(x) for x in script).strip()
    if isinstance(script, dict):
        return " ".join(script_a_texto(v) for v in script.values()).strip()
    return str(script).strip()


def fabricar_video_mp4(script_texto: str):
    script_texto = script_a_texto(script_texto)
    if not script_texto:
        raise ErrorFatal("El guion de vídeo está vacío.")

    audio_path = "temp_voice.mp3"
    bg_path = "temp_bg.mp4"

    asyncio.run(generar_voz_audio(script_texto, audio_path))

    log.info("🎬 Preparando fondo y renderizando vídeo MP4 con MoviePy...")
    audio_clip = AudioFileClip(audio_path)
    duracion = min(audio_clip.duration, DURACION_MAX_VIDEO_SEG)
    audio_clip = audio_clip.subclip(0, duracion)

    if descargar_fondo(bg_path):
        fondo = VideoFileClip(bg_path)
        if fondo.duration < duracion:
            fondo = fondo.fx(vfx.loop, duration=duracion)
        else:
            fondo = fondo.subclip(0, duracion)
    else:
        log.warning("⚠️ No se pudo descargar el fondo; usando fondo de color sólido.")
        fondo = ColorClip(size=(1080, 1920), color=(10, 25, 50), duration=duracion)

    video = fondo.set_audio(audio_clip)
    video.write_videofile(
        ARCHIVO_VIDEO,
        fps=24,
        codec="libx264",
        audio_codec="aac",
        preset="ultrafast",
        logger=None,
    )

    audio_clip.close()
    video.close()
    fondo.close()
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

    log.info(f"🎯 Producto seleccionado:
