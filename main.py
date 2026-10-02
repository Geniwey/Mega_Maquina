import os
import io
import sys
import json
import time
import logging

import requests
import pandas as pd
import groq
from groq import Groq

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

# llama-3.1-70b-versatile fue retirado por Groq. Se usa el sucesor oficial.
# Puedes cambiarlo sin tocar el código con la variable de entorno GROQ_MODEL.
MODELO = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

ARCHIVO_SALIDA = "contenido_hoy.json"

MAX_REINTENTOS = 3
ESPERA_BASE_SEG = 5          # 5s, 10s, 20s...
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

# ---------------------------------------------------------------------------
# LOGS
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("mega_maquina")


class ErrorFatal(Exception):
    """Error que impide continuar el pipeline."""


# Caracteres invisibles que se cuelan al copiar/pegar en Google Sheets
CARACTERES_INVISIBLES = ["\u2060", "\u200b", "\u200c", "\u200d", "\ufeff", "\u00a0"]


def limpiar_texto(valor) -> str:
    """Quita caracteres invisibles y espacios sobrantes."""
    texto = str(valor)
    for c in CARACTERES_INVISIBLES:
        texto = texto.replace(c, " " if c == "\u00a0" else "")
    return texto.strip()


# ---------------------------------------------------------------------------
# 1. LECTURA Y VALIDACIÓN DEL CSV
# ---------------------------------------------------------------------------
def descargar_csv(url: str) -> pd.DataFrame:
    """Descarga el CSV con timeout y reintentos."""
    ultimo_error = None
    for intento in range(1, MAX_REINTENTOS + 1):
        try:
            log.info(f"📥 Descargando base de datos B2B (intento {intento}/{MAX_REINTENTOS})...")
            resp = requests.get(url, timeout=TIMEOUT_CSV_SEG)
            resp.raise_for_status()

            contenido_tipo = resp.headers.get("Content-Type", "")
            if "text/html" in contenido_tipo.lower():
                raise ErrorFatal(
                    "Google devolvió HTML en vez de CSV. "
                    "Comprueba que la hoja es pública ('Cualquiera con el enlace')."
                )

            df = pd.read_csv(io.StringIO(resp.content.decode("utf-8")))
            log.info(f"✅ CSV descargado: {len(df)} filas, {len(df.columns)} columnas.")
            return df

        except ErrorFatal:
            raise
        except (requests.RequestException, pd.errors.ParserError,
                pd.errors.EmptyDataError, UnicodeDecodeError) as e:
            ultimo_error = e
            log.warning(f"⚠️ Fallo leyendo el CSV: {type(e).__name__}: {e}")
            if intento < MAX_REINTENTOS:
                espera = ESPERA_BASE_SEG * (2 ** (intento - 1))
                log.info(f"⏳ Reintentando en {espera}s...")
                time.sleep(espera)

    raise ErrorFatal(f"No se pudo leer el CSV tras {MAX_REINTENTOS} intentos: {ultimo_error}")


def validar_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Valida que el DataFrame no esté vacío y tenga las columnas requeridas."""
    if df is None or df.empty:
        raise ErrorFatal("El DataFrame está vacío.")

    df.columns = [limpiar_texto(c) for c in df.columns]

    faltantes = [c for c in COLUMNAS_REQUERIDAS if c not in df.columns]
    if faltantes:
        raise ErrorFatal(
            f"Faltan columnas obligatorias: {faltantes}. "
            f"Columnas encontradas: {list(df.columns)}"
        )

    extras = [c for c in df.columns if c not in COLUMNAS_REQUERIDAS]
    if extras:
        log.info(f"ℹ️ Columnas extra ignoradas: {extras}")

    df = df[COLUMNAS_REQUERIDAS].copy()
    antes = len(df)
    df = df.dropna(how="any")
    for col in COLUMNAS_REQUERIDAS:
        df[col] = df[col].map(limpiar_texto)
        df = df[df[col] != ""]
    descartadas = antes - len(df)
    if descartadas:
        log.warning(f"⚠️ {descartadas} filas descartadas por tener campos vacíos.")

    if df.empty:
        raise ErrorFatal("No quedan filas válidas tras la limpieza.")

    log.info(f"✅ Validación OK: {len(df)} productos utilizables.")
    return df


# ---------------------------------------------------------------------------
# 2. LLAMADA A GROQ CON REINTENTOS Y VALIDACIÓN DE JSON
# ---------------------------------------------------------------------------
def es_error_reintentable(e: Exception) -> bool:
    if isinstance(e, (groq.RateLimitError, groq.APITimeoutError, groq.APIConnectionError,
                      groq.InternalServerError)):
        return True
    if isinstance(e, groq.APIStatusError) and getattr(e, "status_code", 0) >= 500:
        return True
    return False


def validar_json(texto: str) -> dict:
    """Parsea y valida la estructura del JSON devuelto por Groq."""
    if not texto or not texto.strip():
        raise ValueError("Groq devolvió una respuesta vacía.")
    datos = json.loads(texto)
    if not isinstance(datos, dict):
        raise ValueError("El JSON no es un objeto (dict).")
    faltan = [k for k in CLAVES_JSON_ESPERADAS if k not in datos]
    if faltan:
        raise ValueError(f"Al JSON le faltan claves: {faltan}")
    return datos


def generar_contenido(client: Groq, prompt: str) -> dict:
    ultimo_error = None
    for intento in range(1, MAX_REINTENTOS + 1):
        try:
            log.info(f"🧠 Generando copy con {MODELO} (intento {intento}/{MAX_REINTENTOS})...")
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=MODELO,
                response_format={"type": "json_object"},
                temperature=0.7,
            )
            texto = chat_completion.choices[0].message.content
            datos = validar_json(texto)
            log.info("✅ JSON recibido y validado correctamente.")
            return datos

        except (json.JSONDecodeError, ValueError) as e:
            ultimo_error = e
            log.warning(f"⚠️ JSON inválido: {e}")

        except groq.APIError as e:
            ultimo_error = e
            if not es_error_reintentable(e):
                raise ErrorFatal(f"Error no recuperable de Groq ({type(e).__name__}): {e}")
            log.warning(f"⚠️ Error temporal de Groq ({type(e).__name__}): {e}")

        except (IndexError, AttributeError) as e:
            ultimo_error = e
            log.warning(f"⚠️ Respuesta de Groq con formato inesperado: {e}")

        if intento < MAX_REINTENTOS:
            espera = ESPERA_BASE_SEG * (2 ** (intento - 1))
            log.info(f"⏳ Reintentando en {espera}s...")
            time.sleep(espera)

    raise ErrorFatal(f"Groq falló tras {MAX_REINTENTOS} intentos. Último error: {ultimo_error}")


# ---------------------------------------------------------------------------
# 3. GUARDADO SEGURO
# ---------------------------------------------------------------------------
def guardar_json(datos: dict, ruta: str) -> None:
    """Escritura atómica: si algo falla, nunca queda un archivo a medias."""
    tmp = ruta + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(datos, f, ensure_ascii=False, indent=2)
        os.replace(tmp, ruta)
        log.info(f"💾 Guardado en '{ruta}' ({os.path.getsize(ruta)} bytes).")
    except OSError as e:
        raise ErrorFatal(f"No se pudo guardar '{ruta}': {e}")
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main() -> None:
    log.info("🚀 Arrancando la Mega Máquina...")

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ErrorFatal("Falta la variable de entorno GROQ_API_KEY (revisa los Secrets de GitHub).")

    # 1. LEER EL EXCEL INFINITO
    df = validar_dataframe(descargar_csv(CSV_URL))

    producto_hoy = df.sample(n=1).iloc[0]
    nombre = producto_hoy["NOMBRE_PRODUCTO"]
    problemas = producto_hoy["PROBLEMAS_QUE_RESUELVE"]
    palabra_clave = producto_hoy["PALABRA_CLAVE_MANYCHAT"]
    enlace = producto_hoy["ENLACE_HOTMART"]

    log.info(f"🎯 Producto a reventar hoy: {nombre}")

    # 2. CONECTAR CON GROQ (El Cerebro)
    # max_retries=0: los reintentos los controlamos nosotros para tener logs claros.
    client = Groq(api_key=api_key, timeout=TIMEOUT_GROQ_SEG, max_retries=0)

    prompt_nivel_dios = f"""
Actúa como un copywriter de respuesta directa y un transitario experto en logística marítima B2B.
Tu objetivo es vender el producto: '{nombre}'.
Este producto soluciona: '{problemas}'.

REGLAS INQUEBRANTABLES:
1. CERO NIÑOS: El gancho de los primeros 3 segundos del vídeo debe filtrar agresivamente. Empieza atacando un dolor de dueños de e-commerce o importadores (ej: pérdida de margen, mercancía bloqueada, sobrecostes sorpresa). No saludes, ve directo a la yugular.
2. LENGUAJE TÉCNICO PERO VISUAL: Usa términos reales (Demurrage, FOB vs CIF, Despacho, DUA, TARIC, Packing List) pero explica el dolor económico que causan.
3. SEO TRANSACCIONAL: Los títulos y descripciones para YouTube y Pinterest deben atacar búsquedas de gente que ya tiene un problema aduanero y quiere pagar para solucionarlo.
4. LLAMADA A LA ACCIÓN (CTA): Termina SIEMPRE exigiendo que comenten la palabra exacta '{palabra_clave}'.

Devuelve el resultado ESTRICTAMENTE en este formato JSON, sin añadir ningún otro texto fuera de las llaves:
{{
  "video_script": "Guion exacto para voz en off de 45-60 seg. Gancho brutal de filtro B2B, desarrollo del dolor y CTA directo.",
  "tiktok_data": {{
    "caption": "Título corto y agresivo para el algoritmo",
    "hashtags": "#ImportacionChina #Logistica #Incoterms #Aduanas #Ecommerce"
  }},
  "ig_reel_data": {{
    "caption": "Texto persuasivo detallando el problema técnico. Cierra con: 'Comenta la palabra {palabra_clave} y te envío el acceso directo por DM'.",
    "hashtags": "#AmazonFBA #Emprendimiento #Negocios #Flete"
  }},
  "youtube_seo": {{
    "title": "Título SEO largo y transaccional (Ej: Cómo evitar recargos Demurrage importando de China)",
    "description": "Descripción SEO enfocada en B2B. Cierre pidiendo el comentario."
  }},
  "pinterest_pins": [
    {{"text_on_image": "Frase lapidaria B2B para imagen 1 (Ej: El fraude del Incoterm CIF)"}},
    {{"text_on_image": "Frase lapidaria B2B para imagen 2 (Ej: Contenedor retenido en Valencia)"}},
    {{"text_on_image": "Frase lapidaria B2B para imagen 3 (Ej: Cómo evitar pagar Demurrage)"}}
  ],
  "linkedin_post": "Escribe un post de 3 párrafos contando una 'historia de guerra' real sobre un cliente que perdió miles de euros por un error en el Packing List o el BL. Tono 100% corporativo para CEOs. Cierra invitando a leer la guía comentando {palabra_clave}."
}}
"""

    contenido = generar_contenido(client, prompt_nivel_dios)

    # Metadatos útiles para el siguiente paso del pipeline (no alteran el resto del JSON)
    contenido["_meta"] = {
        "producto": nombre,
        "palabra_clave": palabra_clave,
        "enlace_hotmart": enlace,
        "modelo": MODELO,
    }

    guardar_json(contenido, ARCHIVO_SALIDA)
    log.info("🏁 Proceso completado con éxito.")


if __name__ == "__main__":
    try:
        main()
    except ErrorFatal as e:
        log.error(f"❌ ERROR FATAL: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        log.error("🛑 Interrumpido manualmente.")
        sys.exit(130)
    except Exception as e:  # red de seguridad final
        log.exception(f"💥 Error inesperado: {type(e).__name__}: {e}")
        sys.exit(1)
