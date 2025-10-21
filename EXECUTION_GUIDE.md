# FARFAN 3.0 Execution Guide

Complete step-by-step guide to install, validate, and execute the FARFAN 3.0 policy analysis pipeline.

---

## Step 1: Installation

Install all required dependencies using pip:

```bash
pip install -r requirements.txt
```

**Expected Output:**
```
Collecting spacy>=3.5.0
  Downloading spacy-3.5.0-cp310-cp310-macosx_11_0_arm64.whl (6.8 MB)
Collecting transformers>=4.26.0
  Downloading transformers-4.26.0-py3-none-any.whl (5.8 MB)
...
Successfully installed spacy-3.5.0 transformers-4.26.0 torch-1.13.1 ...
```

Download required spaCy language models:

```bash
python -m spacy download es_core_news_sm
python -m spacy download es_core_news_lg
```

**Expected Output:**
```
✔ Download and installation successful
You can now load the package via spacy.load('es_core_news_sm')
```

---

## Step 2: Verify Required Configuration Files

Confirm the presence of essential configuration files in the project root:

### Check for cuestionario.json

```bash
ls -l cuestionario.json
```

**Expected Output:**
```
-rw-r--r--  1 user  staff  45678 Oct 21 10:30 cuestionario.json
```

**Note:** If `cuestionario.json` is not in the root, check these alternate locations:
- `data/raw/cuestionario.json`
- `orchestrator/cuestionario.json`
- `config/cuestionario.json`

If the file is missing, the system will fail with:
```
FileNotFoundError: cuestionario.json not found at ./cuestionario.json
```

### Check for rubric_scoring.json

```bash
ls -l rubric_scoring.json
```

**Expected Output:**
```
-rw-r--r--  1 user  staff  12345 Oct 21 10:30 rubric_scoring.json
```

**Note:** If `rubric_scoring.json` is not found in the root, check:
- `data/raw/rubric_scoring.json`
- `config/rubric_scoring.json`

The rubric file defines scoring thresholds and weights for each question dimension.

---

## Step 3: Validate Installation

Run the integration test suite to ensure all components are properly configured:

```bash
python -m pytest tests/test_process_flow.py -v
```

**Expected Output:**
```
tests/test_process_flow.py::TestProcessFlowIntegration::test_mock_registry_initialization PASSED [ 10%]
tests/test_process_flow.py::TestProcessFlowIntegration::test_mock_adapter_method_execution PASSED [ 20%]
tests/test_process_flow.py::TestProcessFlowIntegration::test_question_metadata_extraction PASSED [ 30%]
tests/test_process_flow.py::TestProcessFlowIntegration::test_execution_chain_routing PASSED [ 40%]
tests/test_process_flow.py::TestProcessFlowIntegration::test_full_orchestration_flow_simulation PASSED [ 50%]
tests/test_process_flow.py::TestProcessFlowIntegration::test_adapter_receives_correct_metadata PASSED [ 60%]
tests/test_process_flow.py::TestProcessFlowIntegration::test_circuit_breaker_integration PASSED [ 70%]
tests/test_process_flow.py::TestProcessFlowIntegration::test_parallel_adapter_execution_simulation PASSED [ 80%]
tests/test_process_flow.py::TestProcessFlowIntegration::test_dependency_chain_execution PASSED [ 90%]
tests/test_process_flow.py::TestProcessFlowIntegration::test_error_recovery_flow PASSED [100%]

========================== 10 passed in 5.32s ==========================
```

**Alternative Validation:** Run all integration tests:

```bash
python -m pytest tests/integration/ -v
```

If tests fail, check:
- Python version (requires 3.10+)
- All dependencies installed correctly
- Configuration files present

---

## Step 4: Create Sample Plan File

Create a sample urban planning document for testing:

```bash
cat > plan.txt << 'EOF'
PLAN DE DESARROLLO URBANO SOSTENIBLE
Ciudad de Ejemplo - 2024-2030

DIAGNÓSTICO DE SITUACIÓN ACTUAL

La ciudad de Ejemplo enfrenta desafíos significativos en materia de movilidad urbana 
y acceso a servicios básicos. El diagnóstico identifica las siguientes problemáticas:

1. Congestión vehicular en el 65% de las vías principales
2. Déficit de 15,000 unidades habitacionales para población de bajos ingresos
3. Insuficiente cobertura de transporte público (solo 40% de la población tiene acceso)
4. Carencia de espacios verdes: 2.5 m² per cápita vs. estándar OMS de 9 m²

OBJETIVOS ESTRATÉGICOS

Objetivo 1: Mejorar la movilidad urbana sostenible
- Reducir congestión vehicular en 30% para 2028
- Aumentar uso de transporte público al 60% de viajes diarios
- Implementar 50 km de ciclovías protegidas

Objetivo 2: Aumentar oferta de vivienda asequible
- Construir 10,000 viviendas de interés social en 5 años
- Regularizar 5,000 viviendas informales existentes
- Implementar subsidios para 3,000 familias de bajos ingresos

Objetivo 3: Expandir infraestructura de servicios básicos
- Alcanzar 95% de cobertura en agua potable
- Implementar sistema de reciclaje en 100% de barrios
- Crear 10 nuevos parques urbanos (20 hectáreas totales)

CADENA CAUSAL Y TEORÍA DE CAMBIO

Si invertimos $50 millones en infraestructura de transporte público (INPUTS),
entonces construiremos 3 líneas de BRT y 100 paraderos modernos (ACTIVIDADES),
lo que permitirá transportar 200,000 pasajeros diarios adicionales (OUTPUTS),
resultando en 25% menos emisiones de CO2 y 20% reducción en tiempos de viaje (OUTCOMES),
y finalmente mejorando la calidad de vida urbana y productividad económica (IMPACTOS).

Supuestos críticos:
- Disponibilidad presupuestaria confirmada
- Aprobación de permisos ambientales en 6 meses
- Aceptación ciudadana del nuevo sistema (>70% en encuestas)
- Capacidad institucional para operar y mantener infraestructura

PRESUPUESTO Y FINANCIAMIENTO

Inversión total: $150 millones USD (2024-2030)

Desglose por programa:
- Movilidad sostenible: $50M (33%)
- Vivienda asequible: $60M (40%)
- Servicios básicos: $30M (20%)
- Fortalecimiento institucional: $10M (7%)

Fuentes de financiamiento:
- Gobierno federal: $80M (53%)
- Gobierno municipal: $40M (27%)
- Organismos multilaterales: $20M (13%)
- Sector privado: $10M (7%)

Cronograma de ejecución:
Fase 1 (2024-2025): Diseño y licitaciones - $30M
Fase 2 (2026-2028): Construcción principal - $90M
Fase 3 (2029-2030): Finalización y evaluación - $30M

INDICADORES Y MONITOREO

Indicadores de proceso:
- Porcentaje de avance físico mensual
- Ejecución presupuestaria trimestral
- Número de licitaciones completadas

Indicadores de resultado:
- Cobertura de transporte público (línea base: 40%, meta: 60%)
- Unidades habitacionales construidas (meta: 10,000)
- Toneladas de CO2 reducidas anualmente (meta: 50,000 ton)
- Satisfacción ciudadana (línea base: 45%, meta: 75%)

Sistema de monitoreo:
- Plataforma digital de seguimiento en tiempo real
- Reportes trimestrales a consejo ciudadano
- Evaluación de medio término en 2027
- Evaluación de impacto final en 2031

MECANISMOS DE PARTICIPACIÓN

El plan incorpora mecanismos participativos en todas sus fases:
- Consultas públicas en 50 barrios (ya realizadas)
- Consejo ciudadano con 30 representantes vecinales
- Presupuesto participativo: 5% del total ($7.5M)
- Plataforma digital para propuestas y seguimiento
- Audiencias públicas semestrales de rendición de cuentas
EOF
```

**Verify file creation:**

```bash
wc -l plan.txt
```

**Expected Output:**
```
      91 plan.txt
```

---

## Step 5: Execute Analysis

Run the FARFAN pipeline to analyze the plan:

```bash
python run_farfan.py --plan plan.txt --output results/
```

**Expected Output:**
```
2024-10-21 14:30:15 [INFO] orchestrator: ================================================================================
2024-10-21 14:30:15 [INFO] orchestrator: FARFAN 3.0 - Policy Analysis Orchestrator
2024-10-21 14:30:15 [INFO] orchestrator: World's First Causal Mechanism Analysis System
2024-10-21 14:30:15 [INFO] orchestrator: ================================================================================
2024-10-21 14:30:15 [INFO] orchestrator: Initializing components...
2024-10-21 14:30:16 [INFO] orchestrator: Mode: Single Plan Analysis
2024-10-21 14:30:16 [INFO] orchestrator: Plan: plan.txt
2024-10-21 14:30:18 [INFO] policy_processor: Extracting text from plan.txt
2024-10-21 14:30:18 [INFO] choreographer: Loading questions from cuestionario.json
2024-10-21 14:30:19 [INFO] choreographer: Loaded 300 questions from cuestionario
2024-10-21 14:30:19 [INFO] module_controller: Processing dimension D1: Baseline & Context
2024-10-21 14:30:22 [INFO] module_controller: Processing dimension D2: Activity Sequencing
2024-10-21 14:30:25 [INFO] module_controller: Processing dimension D3: Outcomes & Impact
2024-10-21 14:30:28 [INFO] module_controller: Processing dimension D4: Resource Allocation
2024-10-21 14:30:31 [INFO] module_controller: Processing dimension D5: Monitoring & Evaluation
2024-10-21 14:30:34 [INFO] module_controller: Processing dimension D6: Causal Mechanisms
2024-10-21 14:30:38 [INFO] report_assembly: Assembling final report for plan.txt
2024-10-21 14:30:39 [INFO] report_assembly: Report saved to results/plan_analysis.json

================================================================================
ANALYSIS COMPLETE
================================================================================
Plan: plan.txt
Classification: STRONG_CONVERGENCE
Overall Score: 7.8
Agenda Alignment: 8.2
Execution Time: 23.45s
Output Directory: results/
================================================================================
```

**Command Options:**

- `--plan`: Path to single plan file (PDF, TXT, or DOCX)
- `--output`: Directory for results (default: `./output`)
- `--debug`: Enable detailed logging
- `--workers`: Number of parallel workers (default: 4)

**Batch Processing Example:**

```bash
python run_farfan.py --batch plans_directory/ --max-plans 10 --output results/
```

---

## Step 6: Locate and Interpret Results

### Output Directory Structure

After successful execution, the output directory contains:

```bash
ls -l results/
```

**Expected Output:**
```
total 256
-rw-r--r--  1 user  staff   45678 Oct 21 14:30 plan_analysis.json
-rw-r--r--  1 user  staff   12345 Oct 21 14:30 execution_summary.json
-rw-r--r--  1 user  staff    8901 Oct 21 14:30 dimension_scores.json
-rw-r--r--  1 user  staff   23456 Oct 21 14:30 question_responses.json
-rw-r--r--  1 user  staff    5678 Oct 21 14:30 evidence_artifacts.json
```

### Main Analysis Report (plan_analysis.json)

View the complete analysis report:

```bash
cat results/plan_analysis.json
```

**Structure:**
```json
{
  "plan_metadata": {
    "plan_name": "plan.txt",
    "analysis_date": "2024-10-21T14:30:39",
    "execution_time_seconds": 23.45,
    "farfan_version": "3.0"
  },
  "macro_convergence": {
    "plan_classification": "STRONG_CONVERGENCE",
    "overall_score": 7.8,
    "agenda_alignment": 8.2,
    "confidence": 0.87
  },
  "dimension_scores": {
    "D1_baseline_context": {
      "score": 8.5,
      "weight": 0.15,
      "questions_answered": 10,
      "questions_total": 10
    },
    "D2_activity_sequencing": {
      "score": 7.2,
      "weight": 0.20,
      "questions_answered": 15,
      "questions_total": 15
    },
    ...
  },
  "question_responses": [
    {
      "question_id": "D1-Q1",
      "question_text": "¿El plan identifica claramente la situación actual?",
      "answer": "YES",
      "score": 9.0,
      "confidence": 0.92,
      "evidence": [
        {
          "type": "text_segment",
          "content": "La ciudad de Ejemplo enfrenta desafíos significativos...",
          "location": "page_1_paragraph_2",
          "relevance": 0.95
        }
      ],
      "feedback": "El diagnóstico identifica problemas específicos con datos cuantitativos."
    },
    ...
  ]
}
```

### Execution Summary (execution_summary.json)

View execution statistics:

```bash
cat results/execution_summary.json
```

**Structure:**
```json
{
  "execution_metadata": {
    "start_time": "2024-10-21T14:30:16",
    "end_time": "2024-10-21T14:30:39",
    "duration_seconds": 23.45,
    "status": "success"
  },
  "modules_executed": {
    "policy_processor": {
      "calls": 8,
      "successes": 8,
      "failures": 0,
      "avg_execution_time": 0.45
    },
    "teoria_cambio": {
      "calls": 12,
      "successes": 12,
      "failures": 0,
      "avg_execution_time": 1.23
    },
    ...
  },
  "questions_processed": {
    "total": 300,
    "answered": 287,
    "skipped": 13,
    "success_rate": 0.957
  },
  "circuit_breaker_status": {
    "triggered": false,
    "open_circuits": [],
    "overall_health": "healthy"
  }
}
```

### Question Responses Detail

Extract specific dimension results:

```bash
python -c "import json; data = json.load(open('results/plan_analysis.json')); print(json.dumps(data['dimension_scores'], indent=2))"
```

**Example Output:**
```json
{
  "D1_baseline_context": {
    "score": 8.5,
    "dimension_name": "Diagnóstico y Línea Base",
    "questions": [
      {
        "id": "D1-Q1",
        "score": 9.0,
        "answer": "YES",
        "feedback": "Diagnóstico claro con datos cuantitativos"
      },
      {
        "id": "D1-Q2",
        "score": 8.0,
        "answer": "PARTIAL",
        "feedback": "Identificación de problemas presente, falta priorización"
      }
    ]
  }
}
```

---

## Understanding the Analysis Results

### Score Interpretation

**Overall Score Scale:** 0-10
- **9.0-10.0:** Excelente - Plan cumple todos los criterios de manera ejemplar
- **7.0-8.9:** Bueno - Plan cumple mayoría de criterios con calidad
- **5.0-6.9:** Aceptable - Plan cumple criterios básicos, necesita mejoras
- **3.0-4.9:** Deficiente - Plan tiene deficiencias significativas
- **0.0-2.9:** Insuficiente - Plan no cumple criterios mínimos

### Plan Classification

- **STRONG_CONVERGENCE:** Plan altamente alineado con Nueva Agenda Urbana
- **MODERATE_CONVERGENCE:** Plan parcialmente alineado, requiere ajustes
- **WEAK_CONVERGENCE:** Plan con baja alineación, necesita rediseño
- **DIVERGENCE:** Plan no alineado con principios de la agenda

### Question Responses

Each question receives:
- **Score:** 0-10 numerical score
- **Answer:** YES, NO, PARTIAL, or NOT_FOUND
- **Confidence:** 0-1 statistical confidence level
- **Evidence:** Text segments supporting the answer
- **Feedback:** Qualitative explanation in Spanish

### Key Dimensions Evaluated

1. **D1 - Baseline & Context:** Diagnostic quality and problem identification
2. **D2 - Activity Sequencing:** Logical progression and causal chains
3. **D3 - Outcomes & Impact:** Expected results and success indicators
4. **D4 - Resource Allocation:** Budget and financing adequacy
5. **D5 - Monitoring:** Evaluation mechanisms and indicators
6. **D6 - Causal Mechanisms:** Theory of change and assumption testing

---

## Troubleshooting

### Common Issues

**Issue:** `FileNotFoundError: cuestionario.json not found`

**Solution:**
```bash
# Check if file exists in alternate location
find . -name "cuestionario.json"
# Create symlink if needed
ln -s data/raw/cuestionario.json cuestionario.json
```

---

**Issue:** `ModuleNotFoundError: No module named 'spacy'`

**Solution:**
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

---

**Issue:** `OSError: [E050] Can't find model 'es_core_news_lg'`

**Solution:**
```bash
python -m spacy download es_core_news_lg
python -m spacy validate
```

---

**Issue:** Analysis produces low scores for valid plan

**Possible Causes:**
- Plan language is not Spanish (system optimized for Spanish text)
- Plan lacks structured sections (headers, clear formatting)
- Missing key elements (budget, timeline, causal logic)

**Solution:** Review plan structure and ensure it includes:
- Clear problem diagnosis with quantitative data
- Explicit objectives and expected outcomes
- Budget breakdown and financing sources
- Timeline with milestones
- Monitoring indicators

---

## Advanced Usage

### Analyze Multiple Plans

```bash
python run_farfan.py --batch /path/to/plans/ --max-plans 50 --workers 8 --output batch_results/
```

### Health Check

```bash
python run_farfan.py --health
```

### Debug Mode

```bash
python run_farfan.py --plan plan.txt --output results/ --debug
```

---

## Next Steps

After successful execution:

1. Review detailed question responses in `results/question_responses.json`
2. Analyze dimension-specific scores to identify improvement areas
3. Use evidence artifacts to understand AI reasoning
4. Compare multiple plans using batch processing
5. Integrate results into decision-making workflows

For additional documentation:
- **API Integration:** See `API_IMPLEMENTATION_SUMMARY.md`
- **Architecture Details:** See `STRUCTURE_OVERVIEW.md`
- **Development Guide:** See `AGENTS.md`

---

## Support

For issues or questions:
- Check `logs/` directory for detailed execution traces
- Run validation tests: `pytest tests/integration/ -v`
- Review circuit breaker status in `execution_summary.json`
