import os
import random
import pandas as pd
from groq import Groq

# 1. LEER EL EXCEL INFINITO
# CAMBIA ESTE ID POR EL DE TU GOOGLE SHEETS
SHEET_ID = "10gJJCIlPzCHYEfPYPKgT3-xjUtghbIaYpdR87Da4JPQ 
csv_url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv"

print("📥 Leyendo base de datos de productos...")
df = pd.read_csv(csv_url)

# Elegir un producto al azar para hoy
producto_hoy = df.sample(n=1).iloc[0]
nombre = producto_hoy['NOMBRE_PRODUCTO']
problemas = producto_hoy['PROBLEMAS_QUE_RESUELVE']
palabra_clave = producto_hoy['PALABRA_CLAVE_MANYCHAT']
enlace = producto_hoy['ENLACE_HOTMART']

print(f"🎯 Producto seleccionado hoy: {nombre}")

# 2. CONECTAR CON GROQ (El Cerebro)
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

prompt_maestro = f"""
Actúa como un experto en comercio internacional y logística de importación desde China. 
Tu objetivo es vender este producto: '{nombre}'.
Este producto resuelve estos problemas: '{problemas}'.
La llamada a la acción final debe pedir que comenten la palabra '{palabra_clave}'.

Inventa un ángulo de venta agresivo y devuelve el contenido EXACTAMENTE en este formato JSON, sin añadir texto fuera del JSON:
{{
  "video_script": "Guion para voz en off de 45 seg. Gancho agresivo, desarrollo técnico y llamada a la acción.",
  "tiktok_data": {{
    "caption": "Título corto y agresivo",
    "hashtags": "#Importacion #Ecommerce #Aduanas #China"
  }},
  "ig_reel_data": {{
    "caption": "Texto persuasivo detallando el problema. Termina con: 'Comenta {palabra_clave} y te envío el acceso'.",
    "hashtags": "#Incoterms #AmazonFBA #Logistica"
  }},
  "pinterest_pins": [
    {{"text_on_image": "Frase muy corta y llamativa para imagen 1"}},
    {{"text_on_image": "Frase muy corta y llamativa para imagen 2"}},
    {{"text_on_image": "Frase muy corta y llamativa para imagen 3"}}
  ]
}}
"""

print("🧠 Generando contenido con Llama 3.1 70B...")
chat_completion = client.chat.completions.create(
    messages=[{"role": "user", "content": prompt_maestro}],
    model="llama-3.1-70b-versatile",
    response_format={"type": "json_object"},
    temperature=0.7
)

# 3. GUARDAR EL RESULTADO
resultado_json = chat_completion.choices[0].message.content
print("✅ JSON Generado con éxito:")
print(resultado_json)

# Aquí guardamos el JSON en un archivo para que el siguiente paso (montar vídeo) lo pueda leer
with open("contenido_hoy.json", "w", encoding="utf-8") as f:
    f.write(resultado_json)
