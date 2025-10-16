# 🗺️ ROADMAP DE IMPLEMENTACIÓN - FARFAN 3.0

**Documento de Planificación Estratégica**  
**Fecha:** 16 de Octubre, 2025  
**Versión:** 1.0

---

## 🎯 OBJETIVO

Llevar FARFAN 3.0 desde su estado actual (75-80% completitud) a un **sistema production-ready** que pueda analizar 170 planes de desarrollo territorial con el cuestionario de 300 preguntas.

---

## 📊 SITUACIÓN ACTUAL

### Estado General: **75-80% COMPLETO**

**✅ Lo que funciona:**
- Arquitectura orquestador (100%)
- Sistema de enrutamiento (100%)
- Circuit breaker (100%)
- Documentación técnica (90%)
- 3 de 8 módulos core (policy_processor, causal_proccesor, financiero_viabilidad_tablas)

**❌ Lo que falta:**
- Módulo Derek Beach (crítico)
- Dependencias completas
- Corrección de errores de sintaxis
- Sistema de pruebas
- Web Dashboard

---

## 🚀 FASES DE IMPLEMENTACIÓN

### **FASE 0: EVALUACIÓN Y DECISIONES CRÍTICAS** (1 día)

**Objetivo:** Resolver incógnitas que bloquean planificación.

#### Tareas:

1. **Localizar Derek Beach Module** 🔴
   - [ ] Revisar historial de commits en busca de referencias
   - [ ] Buscar en repositorios relacionados del usuario
   - [ ] Revisar si hay issues/PRs que mencionen "derek" o "beach"
   - [ ] Contactar al autor original del código
   - [ ] **DECISIÓN:** ¿Existe el módulo? → Determina timeline

2. **Validar Entorno de Desarrollo**
   - [ ] Crear entorno virtual limpio
   - [ ] Instalar `requirements_complete.txt`
   - [ ] Documentar problemas de instalación
   - [ ] Resolver conflictos de dependencias

3. **Definir Scope Mínimo Viable**
   - [ ] ¿Cuántas preguntas DEBEN funcionar? (mínimo: 50/300)
   - [ ] ¿Qué dimensiones son prioritarias? (D1, D2, D6)
   - [ ] ¿Cuántos planes en la prueba piloto? (recomendado: 5-10)

**Entregables:**
- Documento de decisiones técnicas
- Plan de implementación detallado (este o actualizado)
- Entorno de desarrollo funcional

---

### **FASE 1: FUNDAMENTOS SÓLIDOS** (3-5 días)

**Objetivo:** Eliminar blockers técnicos básicos.

#### 1.1 Completar Dependencias (Día 1) 🔴

- [ ] Reemplazar `requirements.txt` con `requirements_complete.txt`
- [ ] Instalar todas las dependencias en entorno limpio
- [ ] Descargar modelo spaCy español: `es_core_news_lg`
- [ ] Descargar datos NLTK necesarios
- [ ] Probar importaciones de todos los módulos
- [ ] Documentar versiones exactas que funcionan

**Script de validación:**
```bash
#!/bin/bash
# validate_dependencies.sh

echo "Testing module imports..."
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

failed = []
for mod in modules:
    try:
        __import__(mod)
        print(f'✓ {mod}')
    except Exception as e:
        print(f'✗ {mod}: {e}')
        failed.append(mod)

if failed:
    print(f'\n{len(failed)} modules failed')
    sys.exit(1)
else:
    print('\n✓ All modules imported successfully')
"
```

#### 1.2 Corregir Errores de Sintaxis (Día 1-2) 🟡

**Analyzer_one.py:**
- [ ] Identificar la línea con `NameError`
- [ ] Corregir definición/import de `MunicipalAnalyzer`
- [ ] Validar con `python -m py_compile Analyzer_one.py`

**contradiction_deteccion.py:**
- [ ] Encontrar paréntesis sin cerrar
- [ ] Ejecutar linter: `pylint contradiction_deteccion.py`
- [ ] Corregir y validar

**semantic_chunking_policy.py:**
- [ ] Identificar funciones vacías
- [ ] Agregar `pass` o implementación stub
- [ ] Validar sintaxis

**Script de validación:**
```bash
#!/bin/bash
# validate_syntax.sh

for file in *.py; do
    echo "Checking $file..."
    python3 -m py_compile "$file"
    if [ $? -eq 0 ]; then
        echo "✓ $file"
    else
        echo "✗ $file FAILED"
        exit 1
    fi
done

echo "✓ All Python files have valid syntax"
```

#### 1.3 Resolver Derek Beach (Día 2-5) 🔴

**Opción A: Si existe y es compatible (2 días)**
- [ ] Integrar como submódulo Git o copiar código
- [ ] Verificar compatibilidad de APIs
- [ ] Actualizar imports en `module_adapters.py`
- [ ] Probar instanciación de clases principales
- [ ] Ejecutar métodos clave con datos de prueba

**Opción B: Si existe pero necesita adaptación (3-4 días)**
- [ ] Integrar código base
- [ ] Crear capa de adaptación
- [ ] Implementar interfaces requeridas
- [ ] Pruebas de integración

**Opción C: Si no existe - crear stub temporal (1 día)**
- [ ] Crear estructura de carpeta `dereck_beach/`
- [ ] Implementar stubs para 26 clases
- [ ] Retornar datos mock estructurados
- [ ] Documentar funcionalidad faltante
- [ ] Planificar implementación completa (Fase 3)

**Estructura del stub:**
```python
# dereck_beach/__init__.py

class CDAFFramework:
    def __init__(self, config_path, logger=None):
        self.logger = logger
        print("⚠️  Using STUB implementation of Derek Beach")
    
    def process_document(self, pdf_path_or_text, plan_name):
        return {
            "status": "stub",
            "causal_hierarchy": None,
            "mechanism_inferences": [],
            "financial_audit": {"stub": True},
            "confidence": 0.5,
            "message": "Derek Beach stub - implement real logic"
        }

# ... repetir para las otras 25 clases
```

**Entregables Fase 1:**
- ✅ Todos los módulos importables
- ✅ Cero errores de sintaxis
- ✅ Derek Beach stub o integrado
- ✅ Script de validación de entorno

---

### **FASE 2: INTEGRACIÓN Y PRUEBAS** (5-7 días)

**Objetivo:** Validar que el sistema funciona end-to-end.

#### 2.1 Crear Suite de Pruebas Básicas (Día 6-7)

**Estructura de tests:**
```
tests/
├── __init__.py
├── conftest.py                 # Fixtures compartidas
├── test_orchestrator.py        # Pruebas del orquestador
├── test_question_routing.py    # Pruebas de enrutamiento
├── test_modules_import.py      # Validar importaciones
├── test_single_question.py     # Ejecutar 1 pregunta
├── test_dimension_execution.py # Ejecutar 1 dimensión completa
└── fixtures/
    ├── sample_plan.txt         # Plan de prueba
    └── expected_results.json   # Resultados esperados
```

**Pruebas críticas:**

1. **test_orchestrator.py:**
```python
import pytest
from orchestrator import FARFANOrchestrator

def test_orchestrator_initialization():
    """Verifica que el orquestador se inicializa correctamente."""
    orch = FARFANOrchestrator()
    assert orch is not None
    assert orch.router is not None
    assert orch.choreographer is not None
    assert orch.circuit_breaker is not None

def test_system_health():
    """Verifica que el sistema reporta estado de salud."""
    orch = FARFANOrchestrator()
    health = orch.get_system_health()
    assert "circuit_breaker" in health
    assert "execution_stats" in health
```

2. **test_single_question.py:**
```python
def test_answer_single_question(sample_plan_text):
    """Prueba end-to-end de 1 pregunta."""
    orch = FARFANOrchestrator()
    
    result = orch.answer_single_question(
        question_id="P1-D1-Q1",
        plan_text=sample_plan_text,
        plan_name="Test Plan"
    )
    
    assert result is not None
    assert result["question_id"] == "P1-D1-Q1"
    assert "confidence" in result
    assert result["confidence"] >= 0.0
    assert result["confidence"] <= 1.0
    assert "evidence" in result
```

#### 2.2 Ejecutar Pruebas Incrementales (Día 8-9)

**Progresión de pruebas:**

1. **Nivel 1: Importaciones**
   ```bash
   pytest tests/test_modules_import.py -v
   ```
   - Objetivo: 100% de módulos importables

2. **Nivel 2: Componentes individuales**
   ```bash
   pytest tests/test_orchestrator.py -v
   pytest tests/test_question_routing.py -v
   ```
   - Objetivo: Cada componente funciona aislado

3. **Nivel 3: Una pregunta**
   ```bash
   pytest tests/test_single_question.py -v
   ```
   - Objetivo: Pipeline completo P1-D1-Q1

4. **Nivel 4: Una dimensión completa**
   ```bash
   pytest tests/test_dimension_execution.py::test_dimension_D1 -v
   ```
   - Objetivo: Todas las preguntas de D1 (50 preguntas)

5. **Nivel 5: Plan completo (300 preguntas)**
   ```bash
   pytest tests/test_full_plan.py -v --timeout=600
   ```
   - Objetivo: Análisis completo de 1 plan

#### 2.3 Corregir Bugs de Integración (Día 10-12)

Basándose en los resultados de las pruebas:

- [ ] Documentar cada fallo
- [ ] Priorizar por severidad
- [ ] Corregir bugs críticos primero
- [ ] Re-ejecutar pruebas después de cada corrección
- [ ] Actualizar tests si es necesario

**Estrategia de debugging:**
1. Aislar el módulo/método problemático
2. Crear test específico para reproducir
3. Usar debugger o logs detallados
4. Corregir código
5. Validar que no se rompió nada más

**Entregables Fase 2:**
- ✅ Suite de pruebas con >70% cobertura crítica
- ✅ Al menos 1 pregunta funcionando end-to-end
- ✅ Al menos 1 dimensión funcionando completa
- ✅ Documentación de bugs conocidos

---

### **FASE 3: OPTIMIZACIÓN Y VALIDACIÓN** (7-10 días)

**Objetivo:** Validar con datos reales y optimizar rendimiento.

#### 3.1 Validación con Planes Reales (Día 13-15)

**Plan piloto:**

1. **Seleccionar 5 planes de prueba:**
   - 1 plan excelente (caso ideal)
   - 2 planes buenos (casos normales)
   - 1 plan deficiente (caso problemático)
   - 1 plan atípico (edge case)

2. **Ejecutar análisis completo:**
   ```bash
   python run_farfan.py --plan plan1.pdf --output results/plan1/
   python run_farfan.py --plan plan2.pdf --output results/plan2/
   # ... etc
   ```

3. **Métricas a recopilar:**
   - Tiempo de ejecución total
   - Tiempo por pregunta (promedio, min, max)
   - Tasa de éxito por dimensión
   - Confianza promedio por dimensión
   - Errores/warnings generados

4. **Validación de resultados:**
   - [ ] ¿Las respuestas tienen sentido?
   - [ ] ¿La evidencia es relevante?
   - [ ] ¿Las puntuaciones son razonables?
   - [ ] ¿Hay contradicciones internas?

#### 3.2 Perfilado y Optimización (Día 16-18)

**Identificar cuellos de botella:**

```python
# profile_execution.py
import cProfile
import pstats
from orchestrator import FARFANOrchestrator

def profile_full_analysis():
    orch = FARFANOrchestrator()
    orch.analyze_single_plan("test_plan.pdf")

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    profile_full_analysis()
    profiler.disable()
    
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(50)
```

**Optimizaciones típicas:**

1. **Caché agresivo:**
   - Embeddings de texto
   - Resultados de módulos
   - Análisis semántico

2. **Paralelización:**
   - Módulos independientes en paralelo
   - Procesamiento de múltiples planes simultáneamente

3. **Lazy loading:**
   - Cargar modelos solo cuando se necesitan
   - Singleton pattern para modelos pesados

4. **Reducción de memoria:**
   - Procesar chunks de texto en lugar de todo el documento
   - Liberar recursos después de cada módulo

#### 3.3 Documentación de Usuario (Día 19-20)

Crear guías para usuarios finales:

**README_USUARIO.md:**
```markdown
# Guía de Usuario - FARFAN 3.0

## Instalación Rápida
1. Instalar dependencias: `pip install -r requirements_complete.txt`
2. Descargar modelo spaCy: `python -m spacy download es_core_news_lg`
3. Probar instalación: `python run_farfan.py --health`

## Uso Básico

### Analizar un plan:
\`\`\`bash
python run_farfan.py --plan mi_plan.pdf
\`\`\`

### Analizar múltiples planes:
\`\`\`bash
python run_farfan.py --batch carpeta_planes/ --max-plans 10
\`\`\`

## Interpretación de Resultados
...
```

**Entregables Fase 3:**
- ✅ 5 planes reales analizados exitosamente
- ✅ Reporte de performance (tiempos, cuellos de botella)
- ✅ Optimizaciones implementadas
- ✅ Documentación de usuario completa

---

### **FASE 4: PRODUCCIÓN Y ESCALAMIENTO** (5-10 días)

**Objetivo:** Preparar para analizar 170 planes.

#### 4.1 Análisis Batch Optimizado (Día 21-23)

**Implementar procesamiento eficiente:**

1. **Worker pool con límite de memoria:**
```python
# orchestrator/batch_processor.py

from multiprocessing import Pool, Manager
import psutil

def process_plan_with_memory_limit(plan_path, max_memory_gb=4):
    """Procesa plan con límite de memoria."""
    process = psutil.Process()
    
    # Verificar memoria disponible
    if process.memory_info().rss / 1024**3 > max_memory_gb:
        raise MemoryError("Memory limit exceeded")
    
    orch = FARFANOrchestrator()
    return orch.analyze_single_plan(plan_path)

def batch_process_plans(plan_paths, workers=4):
    """Procesa múltiples planes en paralelo."""
    with Pool(workers) as pool:
        results = pool.map(process_plan_with_memory_limit, plan_paths)
    return results
```

2. **Checkpoint y recuperación:**
   - Guardar progreso cada N planes
   - Reanudar desde último checkpoint en caso de fallo
   - Logging detallado de progreso

3. **Monitoreo en tiempo real:**
   - Dashboard simple con Flask
   - Estadísticas de progreso
   - Alertas de errores

#### 4.2 Prueba de Estrés (Día 24-25)

**Ejecutar análisis de 20-30 planes:**

```bash
python run_farfan.py --batch planes_prueba/ --max-plans 30 --workers 4
```

**Monitorear:**
- Uso de CPU
- Uso de memoria
- Uso de disco
- Tiempo total
- Tasa de éxito
- Errores/warnings

**Ajustar configuración según resultados:**
- Número de workers
- Tamaño de caché
- Timeout de módulos

#### 4.3 Análisis de 170 Planes (Día 26-30)

**Ejecución en producción:**

1. **Pre-validación:**
   - [ ] Verificar que todos los 170 PDFs son legibles
   - [ ] Estimar tiempo total (planes_prueba × escala)
   - [ ] Asegurar espacio en disco suficiente

2. **Ejecución por lotes:**
   ```bash
   # Lote 1: 50 planes
   python run_farfan.py --batch lote1/ --max-plans 50 --workers 4
   
   # Lote 2: 50 planes
   python run_farfan.py --batch lote2/ --max-plans 50 --workers 4
   
   # Lote 3: 50 planes
   python run_farfan.py --batch lote3/ --max-plans 50 --workers 4
   
   # Lote 4: 20 planes
   python run_farfan.py --batch lote4/ --max-plans 20 --workers 4
   ```

3. **Validación de resultados:**
   - [ ] ¿Todos los 170 planes procesados?
   - [ ] ¿Tasa de éxito aceptable? (>90%)
   - [ ] ¿Resultados consistentes?
   - [ ] ¿Reportes generados correctamente?

4. **Análisis agregado:**
   - Generar reporte comparativo de 170 planes
   - Rankings por dimensión
   - Identificar patrones comunes
   - Detectar outliers

**Entregables Fase 4:**
- ✅ 170 planes analizados exitosamente
- ✅ Reporte agregado de todos los planes
- ✅ Dashboard de resultados
- ✅ Documentación de proceso batch

---

## 📅 TIMELINE CONSOLIDADO

### **Escenario Optimista** (Derek Beach existe)

| Fase | Duración | Días Acumulados | Hitos |
|------|----------|----------------|-------|
| Fase 0 | 1 día | 1 | Decisiones críticas |
| Fase 1 | 3 días | 4 | Sistema sin errores |
| Fase 2 | 7 días | 11 | 1 dimensión funcional |
| Fase 3 | 7 días | 18 | 5 planes validados |
| Fase 4 | 5 días | **23 días** | **170 planes completos** |

**Total: ~3-4 semanas** ✅

### **Escenario Realista** (Derek Beach necesita adaptación)

| Fase | Duración | Días Acumulados | Hitos |
|------|----------|----------------|-------|
| Fase 0 | 2 días | 2 | Decisiones + evaluación |
| Fase 1 | 5 días | 7 | Derek Beach adaptado |
| Fase 2 | 10 días | 17 | Sistema integrado |
| Fase 3 | 10 días | 27 | Performance optimizado |
| Fase 4 | 8 días | **35 días** | **170 planes completos** |

**Total: ~5-6 semanas** ⚠️

### **Escenario Pesimista** (Derek Beach desde cero)

| Fase | Duración | Días Acumulados | Hitos |
|------|----------|----------------|-------|
| Fase 0 | 3 días | 3 | Análisis + decisión |
| Fase 1 | 15 días | 18 | Derek Beach implementado |
| Fase 2 | 15 días | 33 | Sistema completamente funcional |
| Fase 3 | 10 días | 43 | Validación exhaustiva |
| Fase 4 | 10 días | **53 días** | **170 planes completos** |

**Total: ~7-8 semanas** 🔴

---

## 🎯 CRITERIOS DE ÉXITO

### **Fase 1 (Fundamentos):**
- ✅ 100% módulos importables sin errores
- ✅ Derek Beach integrado o stub funcional
- ✅ Orquestador inicializa sin errores

### **Fase 2 (Integración):**
- ✅ Al menos 50 preguntas funcionando (1 dimensión)
- ✅ Tasa de éxito >80% en pruebas
- ✅ Suite de tests con >70% cobertura

### **Fase 3 (Validación):**
- ✅ 5 planes reales analizados exitosamente
- ✅ Tiempo de ejecución <30 min por plan
- ✅ Resultados validados por experto de dominio

### **Fase 4 (Producción):**
- ✅ 170 planes procesados con éxito >90%
- ✅ Reportes generados correctamente
- ✅ Sistema estable (sin crashes)
- ✅ Documentación completa

---

## ⚠️ RIESGOS Y MITIGACIONES

| Riesgo | Probabilidad | Impacto | Mitigación |
|--------|--------------|---------|------------|
| Derek Beach no existe | Media | **CRÍTICO** | Crear stubs → implementar gradualmente |
| Dependencias incompatibles | Baja | Alto | Usar entorno virtual, fijar versiones |
| Performance insuficiente | Media | Alto | Paralelización, caché, optimización |
| Calidad de resultados baja | Media | Alto | Validación con expertos, ajuste de umbrales |
| Errores en 170 planes | Media | Medio | Procesamiento por lotes, checkpoints |
| Recursos computacionales | Baja | Medio | Cloud computing, procesamiento distribuido |

---

## 📞 PUNTOS DE DECISIÓN CLAVE

### **Checkpoint 1 (Fin de Fase 0):**
**Pregunta:** ¿Existe Derek Beach?
- **SI:** Continuar con escenario optimista
- **NO:** Decidir entre stub temporal o implementación completa

### **Checkpoint 2 (Fin de Fase 1):**
**Pregunta:** ¿Todos los módulos importables?
- **SI:** Proceder a Fase 2
- **NO:** Resolver dependencias faltantes antes de continuar

### **Checkpoint 3 (Fin de Fase 2):**
**Pregunta:** ¿Al menos 1 dimensión funciona end-to-end?
- **SI:** Proceder a Fase 3
- **NO:** Revisar arquitectura de integración

### **Checkpoint 4 (Fin de Fase 3):**
**Pregunta:** ¿Performance aceptable para 170 planes?
- **SI:** Proceder a Fase 4
- **NO:** Optimizar o considerar reducir scope

---

## 📊 MÉTRICAS DE PROGRESO

**Actualizar semanalmente:**

```
SEMANA 1:
- Módulos importables: [ ] / 8
- Errores de sintaxis: [ ] / 3
- Derek Beach: [ ] Localizado [ ] Integrado [ ] Stub

SEMANA 2:
- Preguntas funcionando: [ ] / 300
- Dimensiones completas: [ ] / 6
- Cobertura de tests: [ ]%

SEMANA 3:
- Planes validados: [ ] / 5
- Tiempo promedio por plan: [ ] min
- Optimizaciones implementadas: [ ]

SEMANA 4:
- Planes procesados batch: [ ] / 170
- Tasa de éxito: [ ]%
- Reportes generados: [ ]
```

---

## 🏁 ENTREGABLES FINALES

Al completar este roadmap:

1. **Sistema FARFAN 3.0 Production-Ready**
   - 8 módulos funcionando
   - 300 preguntas operativas
   - Análisis de 170 planes

2. **Documentación Completa**
   - Manual de usuario
   - Documentación técnica
   - Guía de troubleshooting

3. **Reportes de Análisis**
   - 170 reportes individuales
   - 1 reporte agregado
   - Dashboard de visualización

4. **Repositorio Limpio**
   - Tests completos
   - CI/CD configurado
   - Sin deuda técnica crítica

---

**Este roadmap debe actualizarse cada semana con el progreso real.**

---

**Creado por:** GitHub Copilot  
**Fecha:** 16 de Octubre, 2025  
**Próxima revisión:** Una semana después de iniciar Fase 1
