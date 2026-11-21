import json
import logging
import os
import sys
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

# Cambia "unlisted" a "public" cuando verifiques que la primera subida sale perfecta
PRIVACIDAD = "unlisted"
ARCHIVO_VIDEO = "video_final.mp4"
ARCHIVO_METADATOS = "metadatos_redes.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("subir_youtube")


def obtener_cliente_youtube():
  client_id = os.environ.get("YT_CLIENT_ID")
  client_secret = os.environ.get("YT_CLIENT_SECRET")
  refresh_token = os.environ.get("YT_REFRESH_TOKEN")

  if not all([client_id, client_secret, refresh_token]):
    raise RuntimeError(
        "Faltan secretos de YouTube (YT_CLIENT_ID, YT_CLIENT_SECRET o"
        " YT_REFRESH_TOKEN)."
    )

  creds = Credentials(
      token=None,
      refresh_token=refresh_token,
      token_uri="https://oauth2.googleapis.com/token",
      client_id=client_id,
      client_secret=client_secret,
      scopes=["https://www.googleapis.com/auth/youtube.upload"],
  )
  return build("youtube", "v3", credentials=creds)


def preparar_textos() -> tuple[str, str, list[str]]:
  if not os.path.exists(ARCHIVO_METADATOS):
    raise FileNotFoundError(
        f"No existe {ARCHIVO_METADATOS}. Ejecuta primero main.py."
    )

  with open(ARCHIVO_METADATOS, "r", encoding="utf-8") as f:
    data = json.load(f)

  enlace = data.get("enlace_hotmart", "").strip()
  contenido = data.get("contenido", {})
  yt_seo = contenido.get("youtube_seo", {})

  if isinstance(yt_seo, dict):
    titulo_raw = str(
        yt_seo.get("title", "Cómo evitar sobrecostes CIF en importaciones")
    ).strip()
    desc_raw = str(yt_seo.get("description", "")).strip()
  else:
    titulo_raw = str(yt_seo).strip()[:90]
    desc_raw = ""

  # Asegura #Shorts en el título respetando el límite de 100 caracteres de YouTube
  if "#shorts" not in titulo_raw.lower():
    titulo = f"{titulo_raw[:91].strip()} #Shorts"
  else:
    titulo = titulo_raw[:100]

  bloques_desc = []
  if desc_raw:
    bloques_desc.append(desc_raw)
  if enlace and enlace not in desc_raw:
    bloques_desc.append(
        f"👉 Accede a la guía y auditoría completa aquí:\n{enlace}"
    )
  bloques_desc.append(
      "#Shorts #logistica #aduanas #importacion #comerciointernacional"
      " #incoterms"
  )

  descripcion = "\n\n".join(bloques_desc)
  tags = [
      "logistica",
      "aduanas",
      "importacion",
      "incoterm CIF",
      "comercio internacional",
      "importar de China",
      "CIFDEF",
  ]
  return titulo, descripcion, tags


def subir_short():
  if not os.path.exists(ARCHIVO_VIDEO):
    raise FileNotFoundError(f"No se encuentra el vídeo {ARCHIVO_VIDEO}.")

  titulo, descripcion, tags = preparar_textos()
  youtube = obtener_cliente_youtube()

  cuerpo = {
      "snippet": {
          "title": titulo,
          "description": descripcion,
          "tags": tags,
          "categoryId": "27",  # 27 = Educación / Formación
      },
      "status": {
          "privacyStatus": PRIVACIDAD,
          "selfDeclaredMadeForKids": False,
      },
  }

  log.info(f"📺 Subiendo Short a YouTube ({PRIVACIDAD}): '{titulo}'...")
  media = MediaFileUpload(
      ARCHIVO_VIDEO, mimetype="video/mp4", chunksize=1024 * 1024, resumable=True
  )
  request = youtube.videos().insert(
      part="snippet,status", body=cuerpo, media_body=media
  )

  response = None
  while response is None:
    status, response = request.next_chunk()
    if status:
      log.info(f"⏳ Progreso de subida: {int(status.progress() * 100)}%")

  video_id = response.get("id")
  log.info(
      f"✅ Vídeo subido con éxito. URL: https://youtube.com/shorts/{video_id}"
  )


if __name__ == "__main__":
  try:
    subir_short()
  except HttpError as e:
    log.error(f"❌ Error de API de YouTube: {e}")
    sys.exit(1)
  except Exception as e:
    log.error(f"❌ Error subiendo a YouTube: {e}")
    sys.exit(1)
