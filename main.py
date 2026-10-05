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
    d = ImageDraw.Draw(img)
    if fondo: d.rounded_rectangle([0, 0, ancho - 1, alto - 1], radius=radio, fill=fondo)
    y = pad
    for linea in lineas:
        w = d.textlength(linea, font=fuente)
        d.text(((ancho - w) / 2, y), linea, font=fuente, fill=color, stroke_width=stroke, stroke_fill=color_stroke)
        y += alto_linea
    return np.array(img)

def crear_pin(texto: str, etiqueta: str, ruta: str, paleta):
    W, H = 1000, 1500
    g = np.linspace(0, 1, H).reshape(H, 1, 1)
    fila = np.array(paleta[0]).reshape(1, 1, 3) * (1 - g) + np.array(paleta[1]).reshape(1, 1, 3) * g
    img = Image.fromarray(np.repeat(fila, W, axis=1).astype("uint8"), "RGB").convert("RGBA")
    
    deco = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    dd = ImageDraw.Draw(deco)
    dd.ellipse([-250, -250, 450, 450], fill=(255, 255, 255, 18))
    img = Image.alpha_composite(img, deco)
    d = ImageDraw.Draw(img)

    fe = cargar_fuente(38)
    d.text(((W - d.textlength(etiqueta[:48], font=fe)) / 2, 150), etiqueta[:48], font=fe, fill=(255, 196, 0, 255))
    d.rectangle([W / 2 - 60, 215, W / 2 + 60, 224], fill=(255, 196, 0, 255))

    tam = 130
    while True:
        fuente = cargar_fuente(tam)
        lineas = ajustar_lineas(d, texto, fuente, 840)
        if len(lineas) * int(tam * 1.2) <= 880 or tam <= 56: break
        tam -= 6
        
    y = 270 + (900 - len(lineas) * int(tam * 1.2)) / 2
    for linea in lineas:
        w = d.textlength(linea, font=fuente)
        d.text(((W - w) / 2, y), linea, font=fuente, fill=(255, 255, 255, 255), stroke_width=3, stroke_fill=(0, 0, 0, 120))
        y += int(tam * 1.2)

    fb = cargar_fuente(46)
    btn = "GUARDA ESTE PIN"
    wb = d.textlength(btn, font=fb)
    d.rounded_rectangle([W / 2 - wb / 2 - 50, 1230, W / 2 + wb / 2 + 50, 1320], radius=45, fill=(255, 196, 0, 255))
    d.text(((W - wb) / 2, 1246), btn, font=fb, fill=(0, 0, 0, 255))

    img.convert("RGB").save(ruta, "JPEG", quality=92)

def fabricar_pines(contenido: dict, nombre: str):
    os.makedirs(CARPETA_PINES, exist_ok=True)
    pines = contenido.get("pinterest_pins", [])
    paletas = [((10, 25, 60), (30, 90, 170)), ((15, 15, 25), (130, 40, 50)), ((8, 50, 60), (20, 140, 150))]
    for i, p in enumerate(pines[:3]):
        texto = p.get("text_on_image", "Importación")
        ruta = os.path.join(CARPETA_PINES, f"pin_{i+1}.jpg")
        crear_pin(texto, nombre, ruta, paletas[i % len(paletas)])
        log.info(f"📌 Pin {i+1} creado.")

# ---------------------------------------------------------------------------
# 4. FONDOS MÚLTIPLES DE PEXELS (CAMBIO DE PLANO CADA 2.5 SEG)
# ---------------------------------------------------------------------------
def descargar_fondos_pexels(n: int) -> list:
    api_key = os.environ.get("PEXELS_API_KEY")
    if not api_key:
        log.warning("⚠️ Sin PEXELS_API_KEY. Usando fondos generados por código.")
        return []
        
    consultas = ["cargo ship", "logistics port", "shipping containers", "freight ship", "warehouse forklift"]
    random.shuffle(consultas)
    rutas = []
    
    for consulta in consultas:
        if len(rutas) >= n: break
        try:
            log.info(f"🔎 Buscando clip en Pexels: '{consulta}'...")
            r = requests.get(f"https://api.pexels.com/videos/search?query={consulta}&orientation=portrait&per_page=10",
                             headers={"Authorization": api_key}, timeout=30)
            videos = r.json().get("videos", [])
            if not videos: continue
            
            v = random.choice(videos)
            archivos = [f for f in v["video_files"] if f["height"] >= 1080 and f["file_type"] == "video/mp4"]
            elegido = archivos[0] if archivos else v["video_files"][0]
            
            ruta = f"temp_bg_{len(rutas)}.mp4"
            with requests.get(elegido["link"], stream=True, timeout=60) as resp:
                with open(ruta, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 256):
                        if chunk: f.write(chunk)
            rutas.append(ruta)
            log.info(f"✅ Clip de Pexels descargado.")
        except Exception as e:
            log.warning(f"⚠️ Pexels falló con '{consulta}': {e}")
            
    return rutas

def a_vertical(clip):
    return clip.resize(height=1920).crop(x_center=clip.w / 2, y_center=1920 / 2, width=1080, height=1920)

def construir_fondo_multicamara(rutas: list, duracion: float):
    base = []
    for r in rutas:
        try:
            c = VideoFileClip(r).without_audio()
            if c.duration > 1.0: base.append(a_vertical(c))
        except: pass
        
    if not base: raise ErrorFatal("Ningún clip descargado sirve.")
    
    segmentos = []
    total = 0.0
    i = 0
    # BATIDORA DE CLIPS: Cambia de plano cada SEGUNDOS_POR_PLANO
    while total < duracion:
        c = base[i % len(base)]
        d = min(SEGUNDOS_POR_PLANO, c.duration)
        # Extraer un cacho del clip
        segmentos.append(c.subclip(0, d))
        total += d
        i += 1
        
    return concatenate_videoclips(segmentos, method="compose").subclip(0, duracion)

# ---------------------------------------------------------------------------
# 5. AUDIO Y SUBTÍTULOS HORMOZI
# ---------------------------------------------------------------------------
async def generar_voz_y_tiempos(texto: str, archivo_salida: str) -> list:
    com = edge_tts.Communicate(texto, VOZ, rate=VELOCIDAD_VOZ, boundary="WordBoundary")
    eventos = []
    with open(archivo_salida, "wb") as f:
        async for chunk in com.stream():
            if chunk["type"] == "audio": f.write(chunk["data"])
            elif chunk["type"] in ("WordBoundary", "SentenceBoundary"):
                eventos.append((chunk["offset"] / 1e7, chunk["duration"] / 1e7, chunk["text"]))
    return eventos

def construir_subtitulos(eventos: list, texto: str, duracion: float) -> list:
    tokens = []
    for ini, dur, t in eventos:
        ws = t.split()
        if not ws: continue
        paso = dur / len(ws) if dur > 0 else 0.3
        for k, w in enumerate(ws): tokens.append((ini + k * paso, ini + (k + 1) * paso, w))
    
    if not tokens:
        ws = texto.split()
        paso = duracion / max(len(ws), 1)
        tokens = [(k * paso, (k + 1) * paso, w) for k, w in enumerate(ws)]
        
    # AGRUPAR DE 1 A 2 PALABRAS (Ritmo rápido para retención)
    grupos = []
    for i in range(0, len(tokens), 2):
        b = tokens[i:i + 2]
        grupos.append([b[0][0], b[-1][1], " ".join(x[2] for x in b)])
        
    for j in range(len(grupos) - 1): grupos[j][1] = grupos[j + 1][0]
    if grupos: grupos[-1][1] = min(duracion, grupos[-1][1] + 0.3)
    return [tuple(g) for g in grupos]

# ---------------------------------------------------------------------------
# 6. MONTAJE DE VÍDEO FINAL
# ---------------------------------------------------------------------------
def fabricar_video_mp4(script_texto: str, hook_texto: str, cta_texto: str):
    audio_path = "temp_voice.mp3"
    eventos = asyncio.run(generar_voz_y_tiempos(script_texto, audio_path))
    voz = AudioFileClip(audio_path)
    duracion = voz.duration + 0.5

    log.info("🎬 Descargando y montando metraje multicámara...")
    rutas_bg = descargar_fondos_pexels(NUM_FONDOS)
    
    if rutas_bg:
        fondo = construir_fondo_multicamara(rutas_bg, duracion)
    else:
        # Fondo oscuro de emergencia si Pexels falla
        fondo = ColorClip((1080, 1920), color=(15, 23, 42)).set_duration(duracion)

    # Oscurecer fondo al 30% para que los subtítulos destaquen siempre
    oscuro = ColorClip((1080, 1920), color=(0, 0, 0)).set_opacity(0.3).set_duration(duracion)
    capas = [fondo, oscuro]

    # Subtítulos dinámicos en el centro
    for i, (ini, fin, texto) in enumerate(construir_subtitulos(eventos, script_texto, duracion)):
        t = texto.strip().upper()
        if not t: continue
        # Intercala colores para mayor impacto visual
        color = (255, 196, 0, 255) if i % 4 == 0 else (255, 255, 255, 255)
        arr = render_texto_rgba(t, 980, 110, color=color, stroke=6, pad=20)
        capas.append(ImageClip(arr).set_start(ini).set_duration(fin - ini).set_position(("center", "center")))

    # Gancho estático arriba
    if hook_texto:
        arr = render_texto_rgba(hook_texto.upper(), 960, 60, fondo=(0, 0, 0, 175), pad=25)
        capas.append(ImageClip(arr).set_start(0).set_duration(min(4.0, duracion)).set_position(("center", 250)))

    # CTA al final
    if cta_texto:
        arr = render_texto_rgba(cta_texto.upper(), 960, 66, color=(0, 0, 0, 255), fondo=(255, 196, 0, 240), pad=34)
        ini_cta = max(duracion - 4.5, 0)
        capas.append(ImageClip(arr).set_start(ini_cta).set_duration(duracion - ini_cta).set_position(("center", 1300)))

    final = CompositeVideoClip(capas, size=(1080, 1920)).set_duration(duracion).set_audio(voz)

    log.info("⚙️ Renderizando MP4 final...")
    final.write_videofile(ARCHIVO_VIDEO, fps=24, codec="libx264", audio_codec="aac", preset="ultrafast", logger=None)

    voz.close(); final.close()
    for p in [audio_path] + rutas_bg:
        if os.path.exists(p): os.remove(p)

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    log.info("🚀 Arrancando la Máquina Multicámara...")
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key: raise ErrorFatal("Falta GROQ_API_KEY")

    df = validar_dataframe(descargar_csv(CSV_URL))
    prod = df.sample(n=1).iloc[0]
    nombre, problemas, palabra = prod["NOMBRE_PRODUCTO"], prod["PROBLEMAS_QUE_RESUELVE"], prod["PALABRA_CLAVE_MANYCHAT"]

    client = Groq(api_key=api_key, max_retries=0)
    modelos = obtener_modelos_candidatos(client)

    prompt = f"""
    Copywriter B2B logística. Vende: '{nombre}'. Soluciona: '{problemas}'.
    Guion de UN SOLO TEXTO CONTINUO (60 palabras max):
    1) Gancho de dolor
    2) Problema agravado
    3) Beneficios
    4) CTA: Comenta '{palabra}'
    Devuelve JSON con: video_script, hook_text (max 6 palabras), cta_text, tiktok_data, ig_reel_data, youtube_seo, pinterest_pins, linkedin_post.
    """

    contenido, mod = generar_contenido(client, prompt, modelos)
    
    with open(ARCHIVO_JSON, "w", encoding="utf-8") as f: json.dump(contenido, f, indent=2)
    
    with open(ARCHIVO_TXT, "w", encoding="utf-8") as f:
        f.write(f"=== TIKTOK/REELS ===\n{contenido.get('tiktok_data', {}).get('caption', '')}\n\n")
        f.write(f"=== LINKEDIN ===\n{contenido.get('linkedin_post', '')}\n")
    
    guion = str(contenido.get("video_script", "")).strip()
    hook = str(contenido.get("hook_text", "")).strip()
    cta = str(contenido.get("cta_text", "")).strip()

    fabricar_pines(contenido, nombre)
    fabricar_video_mp4(guion, hook, cta)
    log.info("🏁 Pipeline completado.")

if __name__ == "__main__":
    main()
