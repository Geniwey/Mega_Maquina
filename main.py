def obtener_mejor_modelo(client: Groq) -> str:
    log.info("📡 Escaneando modelos disponibles en Groq...")
    try:
        modelos_activos = [m.id for m in client.models.list().data]
        
        # Orden de preferencia: queremos el más listo primero
        preferencias = [
            "llama-3.3-70b-versatile",
            "llama-3.1-70b-versatile",
            "llama3-70b-8192",
            "mixtral-8x7b-32768",
            "llama-3.1-8b-instant",
            "llama3-8b-8192"
        ]
        
        for pref in preferencias:
            if pref in modelos_activos:
                log.info(f"⭐ Radar fijado en el mejor modelo disponible: {pref}")
                return pref
                
        # Filtro estricto: buscar modelos de generación y EVITAR prompt-guard y whisper
        modelos_generativos = [
            m for m in modelos_activos 
            if "prompt-guard" not in m.lower() and "whisper" not in m.lower() and "vision" not in m.lower()
        ]
        
        for m in modelos_generativos:
            if 'llama' in m.lower():
                log.warning(f"⚠️ Modelo preferido no encontrado. Usando alternativa Llama validada: {m}")
                return m
                
        if modelos_generativos:
             return modelos_generativos[0]
             
        # Fallback ultra seguro si la API devuelve algo inusual
        return "llama-3.3-70b-versatile"
