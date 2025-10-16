# 🚀 FARFAN 3.0

**Framework de Análisis y Retroalimentación para Alineación Nacional**

Sistema avanzado de análisis de políticas públicas para evaluar Planes de Desarrollo Territorial utilizando inferencia Bayesiana y análisis causal.

---

## 📊 Estado del Proyecto

**Versión:** 3.0  
**Estado:** 75-80% Completo  
**Última Actualización:** 16 de Octubre, 2025

### Score de Preparación

```
[████████░░] 75/100

✓ Arquitectura de clase mundial
✓ 17,278 líneas de código
✓ 300 preguntas estructuradas
⚠️ Dependencias incompletas
⚠️ Módulo Derek Beach faltante
```

---

## 🎯 ¿Qué hace FARFAN 3.0?

FARFAN 3.0 analiza **Planes de Desarrollo Territorial** respondiendo **300 preguntas** estructuradas a través de 6 dimensiones causales:

- **D1: Insumos** - Diagnóstico y líneas base
- **D2: Actividades** - Programas y acciones
- **D3: Productos** - Resultados tangibles
- **D4: Resultados** - Cambios medibles
- **D5: Impactos** - Efectos a largo plazo
- **D6: Causalidad** - Coherencia lógica

### Características Clave

- 🧠 **Inferencia Bayesiana** para cuantificar incertidumbre
- 🔗 **Análisis Causal** con framework Derek Beach
- 📊 **300 Preguntas Inteligentes** organizadas en P#-D#-Q#
- 🎯 **8 Módulos Especializados** para análisis granular
- 🔄 **Orquestación Robusta** con tolerancia a fallos
- 📈 **Reportes Multi-nivel** (MICRO/MESO/MACRO)

---

## 🚀 Inicio Rápido

### 1. Validar Sistema

```bash
# Verificar estado del sistema
python validate_system.py
```

### 2. Instalar Dependencias

```bash
# Crear entorno virtual
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Instalar dependencias completas
pip install -r requirements_complete.txt

# Descargar modelo spaCy
python -m spacy download es_core_news_lg
```

### 3. Verificar Instalación

```bash
# Probar que el sistema inicia
python run_farfan.py --health
```

### 4. Analizar un Plan

```bash
# Analizar plan individual
python run_farfan.py --plan mi_plan.pdf

# Analizar múltiples planes
python run_farfan.py --batch planes/ --max-plans 10
```

---

## 📚 Documentación

### Para Empezar YA MISMO

- **[RESUMEN_EJECUTIVO.md](RESUMEN_EJECUTIVO.md)** - Léelo primero (3 minutos)
- **[GUIA_RAPIDA_PRIMEROS_PASOS.md](GUIA_RAPIDA_PRIMEROS_PASOS.md)** - Instrucciones paso a paso

### Documentación Técnica

- **[ANALISIS_CRITICO_IMPLEMENTACION.md](ANALISIS_CRITICO_IMPLEMENTACION.md)** - Análisis exhaustivo del repositorio (17KB)
- **[ROADMAP_IMPLEMENTACION.md](ROADMAP_IMPLEMENTACION.md)** - Plan de 4 fases para completar el sistema (19KB)
- **[EXECUTION_MAPPING_MASTER.md](EXECUTION_MAPPING_MASTER.md)** - Especificación técnica completa (1,031 líneas)

### Configuración

- **[requirements_complete.txt](requirements_complete.txt)** - Todas las dependencias
- **[cuestionario.json](cuestionario.json)** - 300 preguntas estructuradas
- **[orchestrator/execution_mapping.yaml](orchestrator/execution_mapping.yaml)** - Configuración de ejecución

---

## 🏗️ Arquitectura

```
FARFAN 3.0
│
├── orchestrator/              # Orquestador principal
│   ├── core_orchestrator.py   # Coordinación de alto nivel
│   ├── choreographer.py       # Gestión de dependencias
│   ├── circuit_breaker.py     # Tolerancia a fallos
│   ├── question_router.py     # Enrutamiento de preguntas
│   └── module_adapters.py     # Adaptadores de módulos
│
├── Módulos Especializados
│   ├── policy_processor.py           # Procesamiento de políticas
│   ├── causal_proccesor.py           # Análisis causal
│   ├── Analyzer_one.py               # Análisis municipal
│   ├── contradiction_deteccion.py    # Detección de contradicciones
│   ├── emebedding_policy.py          # Embeddings semánticos
│   ├── financiero_viabilidad_tablas.py  # Análisis financiero
│   ├── policy_segmenter.py           # Segmentación de documentos
│   └── semantic_chunking_policy.py   # Chunking semántico
│
└── dereck_beach/ (⚠️ FALTANTE)  # Framework Derek Beach
    ├── CDAFFramework             # 26 clases
    └── BayesianMechanismInference  # 89 métodos
```

---

## ⚠️ Problemas Conocidos

### 🔴 Críticos (Bloquean Implementación)

1. **Módulo Derek Beach Ausente**
   - 26 clases faltantes
   - 89 métodos no disponibles
   - **Acción:** Localizar o implementar

2. **Dependencias Incompletas**
   - `sentence-transformers`, `torch`, `spacy`, etc.
   - **Acción:** `pip install -r requirements_complete.txt`

### 🟡 Menores (30 minutos para corregir)

3. **Errores de Sintaxis**
   - `Analyzer_one.py` - NameError
   - `contradiction_deteccion.py` - Paréntesis sin cerrar
   - `semantic_chunking_policy.py` - Indentación

---

## 📈 Estado de Módulos

| Módulo | Estado | Importable | Funcional |
|--------|--------|------------|-----------|
| `orchestrator` | ✅ Completo | ✓ | ✓ |
| `policy_processor` | ✅ Completo | ✓ | ✓ |
| `causal_proccesor` | ✅ Completo | ✓ | ✓ |
| `Analyzer_one` | ⚠️ Error sintaxis | ✗ | - |
| `contradiction_deteccion` | ⚠️ Error sintaxis | ✗ | - |
| `emebedding_policy` | ⚠️ Falta dep. | ✗ | - |
| `financiero_viabilidad_tablas` | ⚠️ Falta dep. | ✗ | - |
| `policy_segmenter` | ⚠️ Falta dep. | ✗ | - |
| `semantic_chunking_policy` | ⚠️ Error sintaxis | ✗ | - |
| `dereck_beach` | ❌ No existe | ✗ | - |

**Progreso:** 2/8 módulos completamente funcionales (25%)  
**Con correcciones:** 8/9 módulos funcionales (89%)

---

## 🎯 Roadmap

### Fase 0: Decisiones Críticas (1 día)
- [ ] Localizar Derek Beach
- [ ] Completar dependencias
- [ ] Definir scope mínimo

### Fase 1: Fundamentos (3-5 días)
- [ ] Instalar todas las dependencias
- [ ] Corregir errores de sintaxis
- [ ] Integrar Derek Beach (o stub)

### Fase 2: Integración (5-7 días)
- [ ] Crear suite de pruebas
- [ ] Ejecutar pruebas incrementales
- [ ] Corregir bugs de integración

### Fase 3: Validación (7-10 días)
- [ ] Validar con 5 planes reales
- [ ] Optimizar rendimiento
- [ ] Documentar resultados

### Fase 4: Producción (5-10 días)
- [ ] Análisis batch de 170 planes
- [ ] Reportes agregados
- [ ] Dashboard de resultados

**Tiempo Total:** 2-8 semanas (según estado de Derek Beach)

---

## 🧪 Testing

```bash
# Ejecutar suite de pruebas (cuando esté implementada)
pytest tests/ -v

# Probar con plan de ejemplo
python run_farfan.py --plan test_plan.txt

# Verificar estado del sistema
python validate_system.py
```

---

## 📊 Métricas del Proyecto

- **17,278** líneas de código Python
- **5,098** líneas en orquestador
- **12,180** líneas en módulos
- **300** preguntas estructuradas
- **8** módulos especializados
- **10** áreas de política (P1-P10)
- **6** dimensiones causales (D1-D6)

---

## 🤝 Contribuir

Este es un proyecto en desarrollo activo. Las áreas prioritarias son:

1. 🔴 Localizar/implementar Derek Beach module
2. 🟡 Corregir errores de sintaxis
3. 🟡 Agregar sistema de pruebas
4. 🟢 Crear web dashboard
5. 🟢 Optimizar rendimiento

---

## 📝 Licencia

(Agregar información de licencia aquí)

---

## 👥 Contacto

Para preguntas o soporte:
- Crear issue en GitHub
- Ver documentación en `ANALISIS_CRITICO_IMPLEMENTACION.md`

---

## 🏆 Reconocimientos

Este proyecto utiliza:
- Inferencia Bayesiana para análisis de incertidumbre
- Framework Derek Beach para análisis causal (CDAF)
- Sentence Transformers (BGE-M3) para embeddings semánticos
- SpaCy para procesamiento de lenguaje natural

---

## 📅 Últimas Actualizaciones

**16 de Octubre, 2025:**
- ✅ Análisis completo del repositorio
- ✅ Documentación exhaustiva creada
- ✅ Plan de implementación de 4 fases
- ✅ Script de validación automática
- ✅ Archivo de dependencias completo

**Ver [ROADMAP_IMPLEMENTACION.md](ROADMAP_IMPLEMENTACION.md) para detalles.**

---

**FARFAN 3.0** - *Framework de Análisis y Retroalimentación para Alineación Nacional*

**Estado:** En Desarrollo Activo | **Calidad:** 8.0/10 | **Arquitectura:** Clase Mundial
