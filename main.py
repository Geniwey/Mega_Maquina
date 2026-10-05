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
SEGUNDOS_POR_PLANO = 5.0
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
# 1. CSV
# ---------------------------------------------------------------------------
def limpiar_texto(valor) -> str:
    s = str(valor)
    s = "".join(ch for ch in s if unicodedata.category(ch) not in ("Cf", "Cc"))
    return s.replace("\u00a0", " ").strip()


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
            raise ErrorFatal(f"Falta columna obligatoria: '{c}'. Detectadas: {list(df.columns)}")
    df = df[COLUMNAS_REQUERIDAS].dropna().copy()
    for col in COLUMNAS_REQUERIDAS:
        df[col] = df[col].map(limpiar_texto)
    df = df[(df[COLUMNAS_REQUERIDAS] != "").all(axis=1)]
    if df.empty:
        raise ErrorFatal("No quedan filas válidas tras la limpieza.")
    log.info(f"✅ {len(df)} fila(s) válida(s) tras validar.")
    return df


# ---------------------------------------------------------------------------
# 2. IA: RADAR DE MODELOS + CADENA DE RESPALDO
# ---------------------------------------------------------------------------
EXCLUIR_EN_NOMBRE = ("whisper", "guard", "safeguard", "vision", "llava", "orpheus", "tts", "playai", "embed", "distil-whisper")
PREFERENCIAS = ["openai/gpt-oss-120b", "llama-3.3-70b-versatile", "qwen/qwen3", "openai/gpt-oss-20b",
                "llama-3.1-70b-versatile", "llama-3.1-8b-instant", "llama", "mixtral", "gemma"]
ERRORES_MODELO_NO_VALIDO = ("model_terms_required", "requires terms acceptance", "model_not_found",
                            "does not exist", "decommissioned", "not supported", "permission", "no access")


def obtener_modelos_candidatos(client: Groq) -> list:
    log.info("📡 Escaneando modelos disponibles HOY en Groq...")
    try:
        activos = [m.id for m in client.models.list().data]
    except Exception as e:
        raise ErrorFatal(f"Fallo en el radar de modelos: {e}")
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
    """Prueba cada modelo. Ante un 400 reintenta en modo mínimo (sin json_mode ni límites)."""
    mensajes = [
        {"role": "system", "content": "Responde SOLO con un objeto JSON válido, sin texto extra ni markdown."},
        {"role": "user", "content": prompt},
    ]
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
                    if k not in datos:
                        raise ValueError(f"Falta clave JSON: {k}")
                log.info(f"✅ JSON validado con {modelo}.")
                return datos, modelo
            except Exception as e:
                msg = str(e)
                low = msg.lower()
                log.warning(f"⚠️ {modelo}: {msg[:500]}")
                if any(x in low for x in ERRORES_MODELO_NO_VALIDO):
                    log.warning(f"⏭️ {modelo} descartado.")
                    break
                if ("400" in low or "json" in low or "response_format" in low) and not modo_minimo:
                    modo_minimo = True
                    continue
                if "429" in low or "rate" in low:
                    time.sleep(15)
                    continue
                if intento < 3:
                    time.sleep(ESPERA_BASE_SEG)
        log.warning(f"⏭️ Pasando al siguiente modelo tras fallar {modelo}.")
    raise ErrorFatal("Ningún modelo de Groq pudo generar el contenido.")


# ---------------------------------------------------------------------------
# 3. UTILIDADES DE TEXTO E IMAGEN (PILLOW, SIN IMAGEMAGICK)
# ---------------------------------------------------------------------------
def limpiar_para_imagen(texto: str) -> str:
    out = []
    for ch in str(texto):
        o = ord(ch)
        if o > 0xFFFF or 0x2600 <= o <= 0x27BF or 0x2B00 <= o <= 0x2BFF or o in (0xFE0F, 0x200D):
            continue
        out.append(ch)
    return re.sub(r"\s+", " ", "".join(out)).strip()


def limpiar_guion_para_voz(texto: str) -> str:
    t = re.sub(r"\[.*?\]", " ", str(texto))
    t = t.replace("*", " ").replace("#", " ")
    return limpiar_para_imagen(t)


def texto_plano(valor) -> str:
    if isinstance(valor, dict):
        valor = " ".join(str(v) for v in valor.values())
    elif isinstance(valor, list):
        valor = " ".join(str(x) for x in valor)
    return str(valor or "").strip()


def cargar_fuente(tam: int):
    for ruta in FUENTES:
        try:
            return ImageFont.truetype(ruta, tam)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=tam)
    except TypeError:
        return ImageFont.load_default()


def ajustar_lineas(draw, texto: str, fuente, ancho_max: int) -> list:
    lineas, actual = [], ""
    for p in texto.split():
        prueba = f"{actual} {p}".strip()
        if draw.textlength(prueba, font=fuente) <= ancho_max or not actual:
            actual = prueba
        else:
            lineas.append(actual)
            actual = p
    if actual:
        lineas.append(actual)
    return lineas or [""]


def render_texto_rgba(texto, ancho, tam, color=(255, 255, 255, 255), stroke=0,
                      fondo=None, pad=30, radio=40, color_stroke=(0, 0, 0, 255)):
    fuente = cargar_fuente(tam)
    tmp = ImageDraw.Draw(Image.new("RGBA", (ancho, 10)))
    lineas = ajustar_lineas(tmp, texto, fuente, ancho - 2 * pad - 2 * stroke)
    alto_linea = int(tam * 1.22)
    alto = len(lineas) * alto_linea + 2 * pad
    img = Image.new("RGBA", (ancho, alto), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if fondo:
        d.rounded_rectangle([0, 0, ancho - 1, alto - 1], radius=radio, fill=fondo)
    y = pad
    for linea in lineas:
        w = d.textlength(linea, font=fuente)
        d.text(((ancho - w) / 2, y), linea, font=fuente, fill=color,
               stroke_width=stroke, stroke_fill=color_stroke)
        y += alto_linea
    return np.array(img)


# ---------------------------------------------------------------------------
# 4. VOZ + SUBTÍTULOS SINCRONIZADOS
# ---------------------------------------------------------------------------
async def generar_voz_y_tiempos(texto: str, archivo_salida: str) -> list:
    log.info("🎙️ Sintetizando voz en off B2B (Edge TTS)...")
    try:
        com = edge_tts.Communicate(texto, VOZ, rate=VELOCIDAD_VOZ, boundary="WordBoundary")
    except TypeError:
        com = edge_tts.Communicate(texto, VOZ, rate=VELOCIDAD_VOZ)
    eventos = []
    with open(archivo_salida, "wb") as f:
        async for chunk in com.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])
            elif chunk["type"] in ("WordBoundary", "SentenceBoundary"):
                eventos.append((chunk["offset"] / 1e7, chunk["duration"] / 1e7, chunk["text"]))
    log.info(f"✅ Audio generado ({len(eventos)} marcas de tiempo).")
    return eventos


def construir_subtitulos(eventos: list, texto: str, duracion: float) -> list:
    tokens = []
    for ini, dur, t in eventos:
        ws = t.split()
        if not ws:
            continue
        paso = dur / len(ws) if dur > 0 else 0.3
        for k, w in enumerate(ws):
            tokens.append((ini + k * paso, ini + (k + 1) * paso, w))
    if not tokens:
        ws = texto.split()
        paso = duracion / max(len(ws), 1)
        tokens = [(k * paso, (k + 1) * paso, w) for k, w in enumerate(ws)]
    grupos = []
    for i in range(0, len(tokens), 3):
        b = tokens[i:i + 3]
        grupos.append([b[0][0], b[-1][1], " ".join(x[2] for x in b)])
    for j in range(len(grupos) - 1):
        grupos[j][1] = grupos[j + 1][0]
    if grupos:
        grupos[-1][1] = min(duracion, grupos[-1][1] + 0.3)
    for g in grupos:
        if g[1] - g[0] < 0.15:
            g[1] = g[0] + 0.15
    return [tuple(g) for g in grupos]


# ---------------------------------------------------------------------------
# 5. MÚSICA GENERADA POR CÓDIGO
# ---------------------------------------------------------------------------
def generar_musica(duracion: float, ruta: str, sr: int = 22050) -> str:
    total = duracion + 1.0
    n = int(total * sr)
    t = np.arange(n) / sr
    acordes = [[220.00, 261.63, 329.63], [174.61, 220.00, 261.63],
               [261.63, 329.63, 392.00], [196.00, 246.94, 293.66]]
    seg, senal, idx, ini = 4.0, np.zeros(n), 0, 0.0
    while ini < total:
        i0, i1 = int(ini * sr), min(int((ini + seg + 0.5) * sr), n)
        tt = t[i0:i1]
        ac = acordes[idx % len(acordes)]
        env = np.clip(np.minimum((tt - ini + 0.5) / 1.0, ((ini + seg + 0.5) - tt) / 1.0), 0, 1)
        tono = sum(np.sin(2 * np.pi * f * tt) for f in ac) + 1.2 * np.sin(2 * np.pi * (ac[0] / 2) * tt)
        senal[i0:i1] += tono * env
        ini += seg
        idx += 1
    senal = senal / (np.max(np.abs(senal)) or 1.0) * 0.12
    senal = senal * np.clip(t / 1.0, 0, 1) * np.clip((total - t) / 1.5, 0, 1)
    datos = (np.repeat(senal[:, None], 2, axis=1) * 32767).astype(np.int16)
    with wave.open(ruta, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(datos.tobytes())
    return ruta


# ---------------------------------------------------------------------------
# 6. FONDOS (PEXELS + RESPALDO ANIMADO)
# ---------------------------------------------------------------------------
def a_vertical(clip, ancho=1080, alto=1920):
    clip = clip.resize(height=alto)
    if clip.w < ancho:
        clip = clip.resize(width=ancho)
    return clip.crop(x_center=clip.w / 2, y_center=clip.h / 2, width=ancho, height=alto)


def descargar_fondos_pexels(n: int) -> list:
    api_key = os.environ.get("PEXELS_API_KEY")
    if not api_key:
        log.info("ℹ️ Sin PEXELS_API_KEY: se usará el fondo animado generado.")
        return []
    consultas = ["cargo ship", "shipping containers port", "logistics warehouse",
                 "container terminal", "freight truck highway", "airplane cargo"]
    random.shuffle(consultas)
    candidatos, vistos = [], set()
    for consulta in consultas:
        if len(candidatos) >= n * 3:
            break
        try:
            log.info(f"🔎 Buscando fondos en Pexels: '{consulta}'...")
            r = requests.get("https://api.pexels.com/videos/search",
                             headers={"Authorization": api_key},
                             params={"query": consulta, "orientation": "portrait", "per_page": 15},
                             timeout=30)
            r.raise_for_status()
            for v in r.json().get("videos", []):
                if v["id"] not in vistos:
                    vistos.add(v["id"])
                    candidatos.append(v)
        except Exception as e:
            log.warning(f"⚠️ Pexels falló con '{consulta}': {e}")
    random.shuffle(candidatos)
    rutas = []
    for v in candidatos:
        if len(rutas) >= n:
            break
        archivos = [f for f in v.get("video_files", []) if f.get("file_type") == "video/mp4"
                    and (f.get("height") or 0) >= (f.get("width") or 0)]
        if not archivos:
            continue
        archivos.sort(key=lambda f: f["height"])
        buenos = [f for f in archivos if f["height"] >= 1080]
        elegido = buenos[0] if buenos else archivos[-1]
        ruta = f"temp_bg_{len(rutas)}.mp4"
        try:
            with requests.get(elegido["link"], stream=True, timeout=60) as resp:
                resp.raise_for_status()
                with open(ruta, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 256):
                        if chunk:
                            f.write(chunk)
            rutas.append(ruta)
            log.info(f"✅ Fondo {len(rutas)}/{n} descargado.")
        except Exception as e:
            log.warning(f"⚠️ No se pudo descargar un fondo: {e}")
    return rutas


def fondo_animado(duracion: float):
    w, h = 270, 480
    yy, xx = np.mgrid[0:h, 0:w]

    def make_frame(t):
        v = (np.sin(xx / 60 + t * 0.6) + np.sin(yy / 90 - t * 0.4) + np.sin((xx + yy) / 80 + t * 0.5)) / 3
        v = (v + 1) / 2
        return np.dstack([15 + 25 * v, 30 + 60 * v, 60 + 110 * v]).astype("uint8")

    return VideoClip(make_frame, duration=duracion).resize((1080, 1920))


def construir_fondo(rutas: list, duracion: float):
    base = []
    for r in rutas:
        try:
            c = VideoFileClip(r).without_audio()
            if c.duration and c.duration > 0.5:
                base.append(a_vertical(c))
        except Exception as e:
            log.warning(f"⚠️ Fondo inutilizable ({r}): {e}")
    if not base:
        raise ErrorFatal("Ningún fondo de Pexels se pudo procesar.")
    segmentos, total, i = [], 0.0, 0
    while total < duracion:
        c = base[i % len(base)]
        d = min(SEGUNDOS_POR_PLANO, c.duration)
        vuelta = i // len(base)
        inicio = min(vuelta * SEGUNDOS_POR_PLANO, max(0.0, c.duration - d))
        segmentos.append(c.subclip(inicio, inicio + d))
        total += d
        i += 1
    return concatenate_videoclips(segmentos, method="compose").subclip(0, duracion)


# ---------------------------------------------------------------------------
# 7. VÍDEO
# ---------------------------------------------------------------------------
def fabricar_video_mp4(script_texto: str, hook_texto: str, cta_texto: str):
    audio_path, musica_path, rutas_bg = "temp_voice.mp3", "temp_music.wav", []

    eventos = asyncio.run(generar_voz_y_tiempos(script_texto, audio_path))
    voz = AudioFileClip(audio_path)
    duracion = voz.duration + 0.6

    generar_musica(duracion, musica_path)
    musica = AudioFileClip(musica_path)
    audio_final = CompositeAudioClip([voz, musica]).set_duration(duracion)

    log.info("🎬 Preparando fondo y capas del vídeo...")
    rutas_bg = descargar_fondos_pexels(NUM_FONDOS)
    fondo = None
    if rutas_bg:
        try:
            fondo = construir_fondo(rutas_bg, duracion)
            log.info("✅ Fondo de Pexels montado con cambios de plano.")
        except Exception as e:
            log.warning(f"⚠️ {e}")
    if fondo is None:
        fondo = fondo_animado(duracion)
        log.info("✅ Fondo animado generado por código.")

    capas = [fondo, ColorClip((1080, 1920), color=(0, 0, 0)).set_opacity(0.35).set_duration(duracion)]

    hook = limpiar_para_imagen(hook_texto).upper()
    if hook:
        arr = render_texto_rgba(hook, 960, 78, fondo=(0, 0, 0, 175), pad=34)
        capas.append(ImageClip(arr).set_start(0).set_duration(min(4.0, duracion)).set_position(("center", 260)))

    if MARCA:
        arr = render_texto_rgba(limpiar_para_imagen(MARCA), 700, 44, stroke=3, pad=10)
        capas.append(ImageClip(arr).set_duration(duracion).set_position(("center", 140)))

    for ini, fin, texto in construir_subtitulos(eventos, script_texto, duracion):
        t = limpiar_para_imagen(texto).upper()
        if not t:
            continue
        arr = render_texto_rgba(t, 980, 96, stroke=8, pad=20)
        capas.append(ImageClip(arr).set_start(ini).set_duration(fin - ini).set_position(("center", 880)))

    cta = limpiar_para_imagen(cta_texto)
    if cta:
        arr = render_texto_rgba(cta.upper(), 960, 66, color=(0, 0, 0, 255), fondo=(255, 196, 0, 240), pad=34)
        inicio_cta = max(duracion - 4.5, 0)
        capas.append(ImageClip(arr).set_start(inicio_cta).set_duration(duracion - inicio_cta)
                     .set_position(("center", 1250)))

    final = CompositeVideoClip(capas, size=(1080, 1920)).set_duration(duracion).set_audio(audio_final)

    log.info("⚙️ Renderizando MP4 final vertical...")
    final.write_videofile(ARCHIVO_VIDEO, fps=24, codec="libx264", audio_codec="aac",
                          preset="veryfast", threads=4, ffmpeg_params=["-pix_fmt", "yuv420p"], logger=None)

    for c in (voz, musica, final):
        try:
            c.close()
        except Exception:
            pass
    for p in [audio_path, musica_path] + rutas_bg:
        if os.path.exists(p):
            os.remove(p)
    log.info(f"✅ ¡Vídeo fabricado con éxito: {ARCHIVO_VIDEO}!")


# ---------------------------------------------------------------------------
# 8. PINES DE PINTEREST (1000x1500, SIN DESCARGAS)
# ---------------------------------------------------------------------------
PALETAS = [((10, 25, 60), (30, 90, 170)), ((15, 15, 25), (130, 40, 50)),
           ((8, 50, 60), (20, 140, 150)), ((30, 20, 60), (110, 70, 190))]


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
        if alto_total <= 880 or tam <= 56:
            break
        tam -= 6
    y = 270 + (900 - alto_total) / 2
    for linea in lineas:
        w = d.textlength(linea, font=fuente)
        d.text(((W - w) / 2, y), linea, font=fuente, fill=(255, 255, 255, 255),
               stroke_width=3, stroke_fill=(0, 0, 0, 120))
        y += int(tam * 1.2)

    fb = cargar_fuente(46)
    btn = "GUARDA ESTE PIN"
    wb = d.textlength(btn, font=fb)
    d.rounded_rectangle([W / 2 - wb / 2 - 50, 1230, W / 2 + wb / 2 + 50, 1320], radius=45, fill=(255, 196, 0, 255))
    d.text(((W - wb) / 2, 1246), btn
