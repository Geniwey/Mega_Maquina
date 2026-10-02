name: El Motor de la Mega Maquina

on:
  workflow_dispatch: # Esto crea un boton para encenderlo a mano

jobs:
  arrancar_cerebro:
    runs-on: ubuntu-latest

    steps:
      - name: 📥 Extraer archivos
        uses: actions/checkout@v4

      - name: ⚙️ Preparar ordenador virtual (Python)
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: 📦 Instalar librerías
        run: |
          pip install pandas groq requests

      - name: 🚀 Ejecutar la Inteligencia Artificial
        env:
          GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
        run: |
          python main.py
