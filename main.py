import re  # <-- añadir junto a los demás imports

# ---------------------------------------------------------------------------
# 2. RADAR DE MODELOS (LECTURA EN VIVO + CADENA DE RESPALDO)
# ---------------------------------------------------------------------------
EXCLUIR_EN_NOMBRE = (
    "whisper", "guard", "safeguard", "vision", "llava",
    "orpheus", "tts", "playai", "embed", "distil-whisper",
)

# Orden de preferencia (se comparan como "contiene", así soporta versiones nuevas)
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

# Errores que significan "este modelo no sirve, pasa al siguiente"
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

    # El resto (por si cambian todos los nombres), con allam al final
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
                    break  # pasa al siguiente modelo

                if "response_format" in msg and usar_json_mode:
                    log.warning("↩️ Este modelo no admite json_object; reintento sin él.")
                    usar_json_mode = False
                    continue

                if intento < MAX_REINTENTOS:
                    time.sleep(ESPERA_BASE_SEG * (2 ** (intento - 1)))

    raise ErrorFatal("Ningún modelo de Groq pudo generar el contenido.")
