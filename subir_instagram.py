import glob
import json
import os
import sys
import time
import requests

GRAPH_API_VERSION = "v22.0"
BASE_URL = f"https://graph.facebook.com/{GRAPH_API_VERSION}"


def obtener_ruta_video() -> str:
    """Busca automáticamente el vídeo .mp4 generado más reciente en el repositorio."""
    ruta_env = os.getenv("VIDEO_PATH", "").strip()
    if ruta_env and os.path.exists(ruta_env):
        return ruta_env

    nombres_prioritarios = [
        "video_final.mp4",
        "final.mp4",
        "reel.mp4",
        "short.mp4",
        "output.mp4",
    ]
    for nombre in nombres_prioritarios:
        if os.path.exists(nombre):
            return nombre

    candidatos = [
        ruta
        for ruta in glob.glob("**/*.mp4", recursive=True)
        if not ruta.startswith((".git", "venv", ".venv", "env"))
    ]
    if not candidatos:
        print("ERROR: No se encontró ningún archivo .mp4 para subir a Instagram.")
        sys.exit(1)

    return max(candidatos, key=os.path.getmtime)


def construir_caption(ruta_metadatos: str = "metadatos_redes.json") -> str:
    """Lee metadatos_redes.json (si existe) y construye el texto del Reel."""
    if not os.path.exists(ruta_metadatos):
        print("AVISO: No se encontró metadatos_redes.json. Se usará un texto por defecto.")
        return "Nuevo vídeo disponible. #reels #actualidad #noticias"

    try:
        with open(ruta_metadatos, "r", encoding="utf-8") as f:
            datos = json.load(f)
    except Exception as e:
        print(f"AVISO: Error leyendo {ruta_metadatos} ({e}). Se usará texto por defecto.")
        return "Nuevo vídeo disponible. #reels #actualidad #noticias"

    # Soporta tanto estructura anidada (datos["instagram"]) como plana
    bloque = datos.get("instagram") if isinstance(datos.get("instagram"), dict) else datos

    caption_directo = (
        bloque.get("instagram_caption")
        or bloque.get("caption")
        or ""
    ).strip()

    if caption_directo:
        texto = caption_directo
    else:
        titulo = (bloque.get("titulo") or bloque.get("title") or "").strip()
        descripcion = (
            bloque.get("descripcion")
            or bloque.get("description")
            or ""
        ).strip()
        hashtags = bloque.get("hashtags") or ""
        if isinstance(hashtags, list):
            hashtags = " ".join(
                tag if str(tag).startswith("#") else f"#{tag}" for tag in hashtags
            )
        hashtags = str(hashtags).strip()

        partes = [p for p in [titulo, descripcion, hashtags] if p]
        texto = "\n\n".join(partes) if partes else "Nuevo vídeo disponible. #reels"

    # Límite máximo de Instagram: 2200 caracteres
    return texto[:2200]


def subir_reel_instagram(video_path: str, caption: str, ig_user_id: str, access_token: str) -> str:
    """Ejecuta el flujo oficial Resumable Upload de Instagram Graph API."""
    file_size = os.path.getsize(video_path)
    print(f"[1/4] Creando contenedor de Reel en Instagram ({video_path} - {file_size / (1024 * 1024):.2f} MB)...")

    init_url = f"{BASE_URL}/{ig_user_id}/media"
    init_payload = {
        "media_type": "REELS",
        "upload_type": "resumable",
        "caption": caption,
        "share_to_feed": "true",
        "access_token": access_token,
    }
    init_res = requests.post(init_url, data=init_payload, timeout=60)
    init_data = init_res.json()

    if init_res.status_code != 200 or "id" not in init_data or "uri" not in init_data:
        print(f"ERROR al crear contenedor en Meta: {json.dumps(init_data, ensure_ascii=False)}")
        sys.exit(1)

    container_id = init_data["id"]
    upload_uri = init_data["uri"]
    print(f"      Contenedor creado correctamente (ID: {container_id}).")

    print("[2/4] Subiendo archivo de vídeo binario a los servidores de Meta...")
    upload_headers = {
        "Authorization": f"OAuth {access_token}",
        "offset": "0",
        "file_size": str(file_size),
    }
    with open(video_path, "rb") as video_file:
        upload_res = requests.post(
            upload_uri,
            headers=upload_headers,
            data=video_file,
            timeout=300,
        )

    try:
        upload_data = upload_res.json()
    except Exception:
        upload_data = {"raw": upload_res.text}

    if upload_res.status_code != 200 or not upload_data.get("success"):
        print(f"ERROR al subir el archivo de vídeo: {upload_data}")
        sys.exit(1)

    print("      Vídeo transferido correctamente.")

    print("[3/4] Esperando a que Instagram procese el Reel...")
    status_url = f"{BASE_URL}/{container_id}"
    max_intentos = 30  # 30 * 8s = 4 minutos máximo

    for intento in range(1, max_intentos + 1):
        time.sleep(8)
        status_res = requests.get(
            status_url,
            params={
                "fields": "status_code,status",
                "access_token": access_token,
            },
            timeout=30,
        )
        status_data = status_res.json()
        status_code = status_data.get("status_code", "DESCONOCIDO")
        print(f"      Intento {intento}/{max_intentos} -> Estado: {status_code}")

        if status_code == "FINISHED":
            break
        if status_code in ("ERROR", "EXPIRED"):
            print(f"ERROR durante el procesamiento en Instagram: {json.dumps(status_data, ensure_ascii=False)}")
            sys.exit(1)
    else:
        print("ERROR: Tiempo de espera agotado mientras Instagram procesaba el vídeo.")
        sys.exit(1)

    print("[4/4] Publicando el Reel en el perfil...")
    publish_url = f"{BASE_URL}/{ig_user_id}/media_publish"
    publish_payload = {
        "creation_id": container_id,
        "access_token": access_token,
    }
    publish_res = requests.post(publish_url, data=publish_payload, timeout=60)
    publish_data = publish_res.json()

    if publish_res.status_code != 200 or "id" not in publish_data:
        print(f"ERROR al publicar el Reel: {json.dumps(publish_data, ensure_ascii=False)}")
        sys.exit(1)

    media_id = publish_data["id"]
    print(f"ÉXITO: Reel publicado en Instagram correctamente. Media ID: {media_id}")
    return media_id


def main():
    ig_user_id = os.getenv("IG_USER_ID", "").strip()
    access_token = os.getenv("IG_ACCESS_TOKEN", "").strip()

    if not ig_user_id or not access_token:
        print("ERROR: Faltan las variables de entorno IG_USER_ID o IG_ACCESS_TOKEN en GitHub Secrets.")
        sys.exit(1)

    video_path = obtener_ruta_video()
    caption = construir_caption("metadatos_redes.json")
    subir_reel_instagram(video_path, caption, ig_user_id, access_token)


if __name__ == "__main__":
    main()
