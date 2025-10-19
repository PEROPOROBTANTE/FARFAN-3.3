#!/bin/bash

echo "=== Iniciando instalación de dependencias para análisis de texto en español ==="

# Actualizar pip primero
echo "Actualizando pip..."
python -m pip install --upgrade pip

# Opción 1: Instalación básica
echo "Instalando dependencias desde requirements.txt..."
pip install -r requirements.txt

# Opción 2: Si hay problemas de memoria durante la instalación
# echo "Instalando dependencias sin usar caché..."
# pip install --no-cache-dir -r requirements.txt

# Opción 3: Si quieres forzar la reinstalación
# echo "Forzando reinstalación de dependencias..."
# pip install --force-reinstall -r requirements.txt

# Verificar instalación de modelos de spaCy
echo "Verificando instalación de modelos de spaCy..."
python -c "import spacy; print(f'Modelo pequeño disponible: {spacy.util.is_package(\"es_core_news_sm\")}'); print(f'Modelo grande disponible: {spacy.util.is_package(\"es_core_news_lg\")}')"

echo "=== Instalación completada ==="

