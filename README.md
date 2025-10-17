# FARFAN 3.0 - Policy Analysis System

**Version:** 3.0.0  
**Framework:** Territorial Development Plan Evaluation  
**Compliance:** SIN_CARRETA Doctrine

---

## Overview

FARFAN 3.0 is a comprehensive policy analysis system designed to evaluate Territorial Development Plans (PDT) across 10 policy areas and 6 analytical dimensions, using a rigorous framework of 300 structured questions.

### Key Features

- **300 Question Framework:** Comprehensive evaluation across 60 dimensions (10 policy areas × 6 dimensions × 5 questions)
- **Multi-Level Reporting:** MICRO (individual questions), MESO (clusters), MACRO (overall convergence)
- **Deterministic Orchestration:** Reproducible, auditable analysis pipeline
- **Canonical Source Architecture:** Single source of truth for all questionnaire data

---

## Architecture

### Core Components

1. **Questionnaire Parser** (`orchestrator/questionnaire_parser.py`)
   - **Status:** ✅ Canonical source
   - **Purpose:** Single source of truth for all 300 questions, dimensions, rubrics, and weights
   - **Compliance:** SIN_CARRETA validated

2. **Question Router** (`orchestrator/question_router.py`)
   - Routes questions to appropriate analysis modules
   - Maps 300 questions to 8 processing components

3. **Execution Choreographer** (`orchestrator/choreographer.py`)
   - Manages module execution with dependency resolution
   - Hybrid parallel/sequential execution strategy

4. **Report Assembler** (`orchestrator/report_assembly.py`)
   - Generates MICRO/MESO/MACRO level reports
   - Doctoral-level explanations and convergence analysis

5. **Core Orchestrator** (`orchestrator/core_orchestrator.py`)
   - Main coordination engine
   - Integrates all components with fault tolerance

### Processing Modules

1. **Analyzer One** - Municipal analysis and baseline identification
2. **Causal Processor** - Causal chain analysis across all dimensions
3. **Contradiction Detector** - Logical consistency validation
4. **Derek Beach CDAF** - Causality and theory of change analysis
5. **Embedding Policy** - Semantic analysis and embeddings
6. **Financial Analyzer** - Budget and financial viability
7. **Policy Processor** - Industrial policy processing
8. **Policy Segmenter** - Document segmentation and extraction

---

## Questionnaire Structure

### Dimensions (D1-D6)

- **D1: Insumos (Inputs)** - Baseline identification, gap analysis, resource allocation
- **D2: Actividades (Activities)** - Activity design, mechanisms, causal links
- **D3: Productos (Products)** - Output indicators, budget alignment, traceability
- **D4: Resultados (Results)** - Outcome measurement, causal chains, monitoring
- **D5: Impactos (Impacts)** - Impact projections, proxy indicators, risk analysis
- **D6: Causalidad (Causality)** - Theory of change, causal logic, consistency

### Policy Areas (P1-P10) - Decálogo

1. **P1:** Women's rights and gender equality
2. **P2:** Violence prevention and conflict protection
3. **P3:** Children and adolescents
4. **P4:** Older adults
5. **P5:** Persons with disabilities
6. **P6:** Land, rural reform, and territorial planning
7. **P7:** Environment and sustainable development
8. **P8:** Education
9. **P9:** Health
10. **P10:** Migrants and vulnerable populations

---

## Data Flow and Traceability

### Canonical Source: cuestionario.json

**Location:** `/cuestionario.json`  
**Status:** ✅ Validated JSON (23,677 lines)  
**Version:** 2.0.0  
**Last Validated:** 2025-10-17

**IMPORTANT - Traceability Rule:**
> All questionnaire data MUST be accessed through `QuestionnaireParser`.  
> Direct loading of `cuestionario.json` is PROHIBITED for determinism and auditability.

### Data Access Pattern

```python
from orchestrator.questionnaire_parser import get_questionnaire_parser

# Get singleton parser instance
parser = get_questionnaire_parser()

# Access dimension data
dimension = parser.get_dimension("D1")
print(f"{dimension.name}: {dimension.description}")

# Access questions
questions = parser.get_questions_for_dimension("D1")
for q in questions:
    print(f"{q.id}: {q.text_template}")

# Get rubric levels
rubric = parser.get_rubric_for_question("D1-Q1")
print(rubric)  # {"EXCELENTE": 0.85, "BUENO": 0.70, ...}

# Generate all 300 questions
all_questions = parser.generate_all_questions()
print(f"Total: {len(all_questions)} questions")
```

### Contract Guarantees

1. **Immutability:** All parsed data structures are immutable (frozen dataclasses)
2. **Determinism:** Question generation is always in sorted order (P1-P10, D1-D6)
3. **Validation:** Structure validated on initialization (6 dimensions, 10 points, 30 base questions)
4. **Error Propagation:** Explicit exceptions, no silent failures
5. **Singleton Pattern:** Parser loaded once, cached for efficiency

---

## SIN_CARRETA Doctrine Compliance

### Core Principles

1. **Determinism:** All operations must be reproducible
2. **Contract Integrity:** Strict interfaces with type safety
3. **Explicit Failure:** No silent errors or warnings
4. **Single Source:** Canonical data with traceability
5. **Auditability:** Complete logging and versioning

### Implementation Status

| Component | Compliance | Verified |
|-----------|------------|----------|
| Questionnaire Parser | ✅ 100% | 2025-10-17 |
| Question Router | ✅ 100% | 2025-10-17 |
| Report Assembly | ✅ 100% | 2025-10-17 |
| Core Orchestrator | ✅ 100% | 2025-10-17 |

**Full compliance report:** See `CODE_FIX_REPORT.md`

---

## Usage

### Basic Analysis

```python
from orchestrator import FARFANOrchestrator
from pathlib import Path

# Initialize orchestrator
orchestrator = FARFANOrchestrator()

# Analyze a plan
results = orchestrator.analyze_single_plan(
    plan_path=Path("data/plan_desarrollo_municipal.pdf"),
    plan_name="PDM Municipio X",
    output_dir=Path("output/municipio_x")
)

# Access results
print(f"Overall Score: {results['macro_level'].overall_score}")
print(f"Classification: {results['macro_level'].plan_classification}")
```

### Module Initialization

```python
from orchestrator.config import CONFIG
from orchestrator.question_router import QuestionRouter
from orchestrator.choreographer import ExecutionChoreographer
from orchestrator.report_assembly import ReportAssembler

# Components automatically use QuestionnaireParser
router = QuestionRouter()
choreographer = ExecutionChoreographer()
assembler = ReportAssembler()
```

---

## Development Guidelines

### Adding New Modules

1. Register module in `orchestrator/config.py`
2. Define module dependencies in choreographer
3. Map module to relevant dimensions
4. **DO NOT** load `cuestionario.json` directly
5. Use `QuestionnaireParser` for all questionnaire data

### Modifying Questions

1. Edit `cuestionario.json` only
2. Maintain JSON structure (no schema changes without migration)
3. Validate JSON syntax before commit
4. Update version number in metadata
5. Document changes in `CODE_FIX_REPORT.md`

### Testing Requirements

1. Validate cuestionario.json syntax
2. Test parser initialization
3. Verify 300 questions generated
4. Check dimension and rubric access
5. Confirm orchestrator integration

---

## Project Structure

```
FARFAN-3.0/
├── cuestionario.json           # Canonical questionnaire source
├── orchestrator/
│   ├── questionnaire_parser.py # Canonical parser (MUST USE)
│   ├── question_router.py      # Question-to-module routing
│   ├── choreographer.py        # Execution orchestration
│   ├── report_assembly.py      # Multi-level reporting
│   ├── core_orchestrator.py    # Main coordination
│   ├── config.py               # System configuration
│   └── ...
├── Analyzer_one.py             # Municipal analyzer
├── causal_proccesor.py         # Causal processor
├── contradiction_deteccion.py  # Contradiction detector
├── dereck_beach.py             # CDAF framework
├── emebedding_policy.py        # Embedding analyzer
├── financiero_viabilidad_tablas.py  # Financial analyzer
├── policy_processor.py         # Policy processor
├── policy_segmenter.py         # Document segmenter
├── CODE_FIX_REPORT.md          # Compliance report
└── README.md                   # This file
```

---

## Dependencies

See `requirements.txt` for complete list. Key dependencies:

- Python 3.8+
- NumPy, Pandas, SciPy
- scikit-learn, PyTorch, Transformers
- NetworkX (for dependency graphs)
- ChromaDB, FAISS (vector databases)
- PDFPlumber, python-docx (document processing)

---

## Version History

### Version 3.0.0 (2025-10-17)

- ✅ Implemented QuestionnaireParser as canonical source
- ✅ Fixed cuestionario.json syntax error (line 23677)
- ✅ Updated all orchestration modules to use parser
- ✅ Achieved 100% SIN_CARRETA compliance
- ✅ Established deterministic orchestration
- ✅ Complete audit trail and traceability

### Version 2.0.0

- Initial 300-question framework
- Multi-dimensional analysis system
- Integration of 8 processing modules

---

## License

[License information to be added]

---

## Contact

For technical questions or contributions, refer to `CONTRIBUTING.md`.

---

## References

- **SIN_CARRETA Doctrine:** Internal compliance framework
- **Decálogo Framework:** 10 policy areas for territorial development
- **DNP Standards:** National Planning Department methodologies
- **CODE_FIX_REPORT.md:** Complete implementation audit

---

**Last Updated:** 2025-10-17  
**Status:** ✅ Production Ready  
**Compliance:** SIN_CARRETA Validated
