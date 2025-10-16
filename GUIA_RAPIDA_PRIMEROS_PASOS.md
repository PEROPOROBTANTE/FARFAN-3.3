# 🚀 GUÍA RÁPIDA - PRIMEROS PASOS

**Para el desarrollador que va a implementar las correcciones**

---

## ⏱️ ACCIÓN INMEDIATA (Próximas 2 horas)

### 1. Instalar Dependencias Completas

```bash
# Crear entorno virtual
python3 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install --upgrade pip
pip install -r requirements_complete.txt

# Instalar modelo spaCy español
python -m spacy download es_core_news_lg

# Descargar datos NLTK
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

**Tiempo estimado:** 15-30 minutos

### 2. Verificar Importaciones

```bash
# Ejecutar script de validación
python3 -c "
import sys
modules = [
    'policy_processor',
    'causal_proccesor', 
    'Analyzer_one',
    'contradiction_deteccion',
    'emebedding_policy',
    'financiero_viabilidad_tablas',
    'policy_segmenter',
    'semantic_chunking_policy'
]

print('Testing module imports...\n')
failed = []
for mod in modules:
    try:
        __import__(mod)
        print(f'✓ {mod}')
    except Exception as e:
        print(f'✗ {mod}: {str(e)[:80]}')
        failed.append((mod, str(e)))

print(f'\n{len(modules)-len(failed)}/{len(modules)} modules OK')
if failed:
    print('\nFailed modules:')
    for mod, err in failed:
        print(f'  - {mod}: {err[:100]}')
    sys.exit(1)
"
```

**Tiempo estimado:** 5 minutos

### 3. Verificar Estado del Sistema

```bash
# Probar que el orquestador inicia
python run_farfan.py --health
```

**Tiempo estimado:** 1 minuto

**Resultado esperado:**
```
System Health Check
Circuit Breaker: UNKNOWN
Plans Processed: 0
Questions Answered: 0
```

---

## 🔍 INVESTIGACIÓN CRÍTICA (Próximas 4 horas)

### **PRIORIDAD #1: Localizar Derek Beach**

#### Opción A: Buscar en el Repositorio

```bash
# Buscar referencias a derek en el historial
git log --all --oneline | grep -i derek
git log --all --oneline | grep -i beach

# Buscar en ramas
git branch -a | grep -i derek

# Buscar en tags
git tag | grep -i derek

# Buscar en issues (si aplica)
gh issue list --search "derek" --state all
gh issue list --search "beach" --state all
```

#### Opción B: Buscar en Archivos de Configuración

```bash
# Buscar referencias a derek_beach
grep -r "derek" . --include="*.md" --include="*.txt" --include="*.yaml"
grep -r "beach" . --include="*.md" --include="*.txt" --include="*.yaml"

# Revisar requirements antiguos
git log --all --full-history -- "*requirements*"
```

#### Opción C: Buscar en Repositorios Relacionados

- Revisar otros repositorios del usuario `kkkkknhh`
- Buscar "Derek Beach" + "CDAF Framework" en GitHub
- Revisar si hay paquetes Python públicos

#### Opción D: Contactar al Autor Original

Si el código fue desarrollado por otra persona:
- Revisar commits para encontrar contributors
- Enviar mensaje/email preguntando por Derek Beach module

### **Documenta tu hallazgo:**

Crear archivo `DEREK_BEACH_STATUS.md`:

```markdown
# Estado del Módulo Derek Beach

**Investigador:** [Tu nombre]
**Fecha:** [Fecha]

## Hallazgos:

[ ] Derek Beach existe en: [ubicación]
[ ] Derek Beach no existe - necesita implementación desde cero
[ ] Derek Beach existe pero no es compatible - necesita adaptación

## Próximos pasos:

[Describe qué hacer basado en tus hallazgos]
```

---

## 🔧 CORRECCIONES BÁSICAS (Día 1)

### Corregir Errores de Sintaxis

#### 1. Analyzer_one.py

```bash
# Ver el error específico
python -m py_compile Analyzer_one.py

# Abrir el archivo y buscar el error
# Típicamente será un problema de indentación o import
```

**Solución típica:**
```python
# Si el error es "name 'MunicipalAnalyzer' is not defined"
# Asegurarse de que la clase esté definida antes de usarse:

class MunicipalAnalyzer:
    def __init__(self):
        # ...
        pass

# Y que cualquier referencia a ella venga DESPUÉS de la definición
```

#### 2. contradiction_deteccion.py

```bash
# Encontrar paréntesis sin cerrar
python -m py_compile contradiction_deteccion.py

# Ver la línea específica del error
# Usar editor que marque paréntesis matching
```

**Tip:** Usa un editor con syntax highlighting (VS Code, PyCharm)

#### 3. semantic_chunking_policy.py

```bash
# Ver el error de indentación
python -m py_compile semantic_chunking_policy.py
```

**Solución típica:**
```python
# Si hay una función vacía:
def mi_funcion():
    pass  # ← Agregar esto

# O implementar algo básico:
def mi_funcion():
    return None
```

### Validar Todas las Correcciones

```bash
# Script que valida sintaxis de todos los archivos
for file in *.py; do
    echo "Checking $file..."
    python -m py_compile "$file" 2>&1
    if [ $? -eq 0 ]; then
        echo "✓ $file"
    else
        echo "✗ $file FAILED"
    fi
done
```

---

## 🧪 PRIMERA PRUEBA (Día 1, tarde)

### Crear un Plan de Prueba Simple

```bash
# Crear archivo de prueba
cat > test_plan.txt << 'EOF'
PLAN DE DESARROLLO MUNICIPAL 2024-2027

DIAGNÓSTICO:
La población del municipio es de 50,000 habitantes.
El desempleo está en 15%.
Hay 500 empresas registradas.

OBJETIVO ESTRATÉGICO:
Reducir el desempleo al 10% en 4 años.

PROGRAMAS:
1. Programa de Capacitación Laboral
   Presupuesto: $500,000
   Meta: Capacitar 1,000 personas

2. Programa de Emprendimiento
   Presupuesto: $300,000
   Meta: Crear 100 nuevas empresas

INDICADORES:
- Tasa de desempleo
- Número de personas capacitadas
- Número de empresas creadas
EOF
```

### Ejecutar Prueba Básica

```bash
# Intentar analizar el plan de prueba
python run_farfan.py --plan test_plan.txt --output test_output/

# Revisar logs
tail -100 logs/farfan_*.log

# Revisar salida
ls -la test_output/
```

**Resultados esperados:**
- Sistema no crashea
- Genera al menos archivos de salida vacíos/básicos
- Logs muestran progreso (aunque fallen algunos módulos)

---

## 📋 CHECKLIST DE VALIDACIÓN

Marca con `[x]` cuando completes cada tarea:

### Día 1:
- [ ] Dependencias instaladas (`requirements_complete.txt`)
- [ ] Modelo spaCy descargado (`es_core_news_lg`)
- [ ] 3 errores de sintaxis corregidos
- [ ] Todos los módulos importables (verificado con script)
- [ ] Estado de Derek Beach documentado
- [ ] Sistema inicia sin errores (`--health` funciona)
- [ ] Primera prueba ejecutada (aunque falle parcialmente)

### Día 2:
- [ ] Derek Beach: Decisión tomada (integrar/stub/implementar)
- [ ] Derek Beach: Solución implementada
- [ ] Todos los módulos importables E instanciables
- [ ] Al menos 1 pregunta completa su ejecución

### Día 3:
- [ ] Suite de pruebas básicas creada
- [ ] Pruebas de importación pasando (100%)
- [ ] Al menos 10 preguntas funcionando
- [ ] Documentación actualizada con hallazgos

---

## 🆘 PROBLEMAS COMUNES Y SOLUCIONES

### Error: "No module named 'sentence_transformers'"

```bash
pip install sentence-transformers
```

### Error: "No module named 'camelot'"

```bash
# Instalar dependencias del sistema primero
# Ubuntu/Debian:
sudo apt-get install ghostscript python3-tk

# Luego instalar camelot
pip install camelot-py[cv]
```

### Error: "Can't find model 'es_core_news_lg'"

```bash
python -m spacy download es_core_news_lg
```

### Error: "CUDA not available" (con PyTorch)

```bash
# Instalar versión CPU de PyTorch
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Error: "Memory Error" al ejecutar

```python
# En config.py, reducir workers:
MAX_PARALLEL_WORKERS = 2  # En lugar de 4 o 8
```

### Sistema muy lento

```python
# Opciones de optimización en config.py:
ENABLE_CACHE = True
CACHE_SIZE_MB = 1024  # Aumentar caché
ASYNC_EXECUTION = True  # Habilitar async
```

---

## 📞 CONTACTOS DE SOPORTE

Si te atascas:

1. **Revisar documentación completa:**
   - `ANALISIS_CRITICO_IMPLEMENTACION.md`
   - `ROADMAP_IMPLEMENTACION.md`
   - `EXECUTION_MAPPING_MASTER.md`

2. **Revisar logs detallados:**
   - `logs/farfan_*.log`
   - Buscar mensajes ERROR o WARNING

3. **Crear issue en GitHub:**
   - Incluir: versión Python, sistema operativo
   - Incluir: error completo con stack trace
   - Incluir: lo que intentabas hacer

---

## ✅ SEÑALES DE ÉXITO

**Sabrás que vas por buen camino cuando:**

1. ✅ `python run_farfan.py --health` ejecuta sin errores
2. ✅ Los 8 módulos importan sin errores
3. ✅ El sistema procesa al menos 1 pregunta completa
4. ✅ Generas tu primer reporte (aunque parcial)

**Celebra pequeñas victorias:**
- Cada módulo que logras importar
- Cada error de sintaxis corregido
- Cada pregunta que funciona

**Este es un proyecto complejo pero bien estructurado. ¡Tú puedes!**

---

**Tiempo total estimado para llegar a "sistema funcional básico": 2-3 días**

**Última actualización:** 16 de Octubre, 2025
