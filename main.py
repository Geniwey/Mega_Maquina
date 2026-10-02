import os
import pandas as pd
from groq import Groq

# 1. LEER EL EXCEL INFINITO
SHEET_ID = "10gJJCIlPzCHYEfPYPKgT3-xjUtghbIaYpdR87Da4JPQ" 
csv_url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv"

print("📥 Leyendo base de datos B2B...")
df = pd.read_csv(csv_url)

producto_hoy = df.sample(n=1).iloc[0]
nombre = producto_hoy['NOMBRE_PRODUCTO']
problemas = producto_hoy['PROBLEMAS_QUE_RESUELVE']
palabra_clave = producto_hoy['PALABRA_CLAVE_MANYCHAT']
enlace = producto_hoy['ENLACE_HOTMART']

print(f"🎯 Producto a reventar hoy: {nombre}")

# 2. CONECTAR CON GROQ (El Cerebro)
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

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

print("🧠 Generando copy Nivel Dios con Llama 3.1 70B...")
chat_completion = client.chat.completions.create(
    messages=[{"role": "user", "content": prompt_nivel_dios}],
    model="llama-3.1-70b-versatile",
    response_format={"type": "json_object"},
    temperature=0.7
)

resultado_json = chat_completion.choices[0].message.content
print("✅ JSON Generado con éxito:")
print(resultado_json)

with open("contenido_hoy.json", "w", encoding="utf-8") as f:
    f.write(resultado_json)
