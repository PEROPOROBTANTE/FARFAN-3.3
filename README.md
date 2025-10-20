# FARFAN 3.0 - Framework for Automated Review and Formulation Analysis

Colombian Municipal Development Plans Analysis Framework

## Overview

FARFAN 3.0 is a comprehensive framework for analyzing Colombian municipal development plans using advanced NLP, causal inference, and policy analysis techniques. The system evaluates 300 questions across 6 dimensions (D1-D6) using 11 specialized adapters with 413+ analysis methods.

## Quick Start

### Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Basic Usage

```bash
# Analyze a single plan
python run_farfan.py --plan path/to/plan.pdf

# Batch analysis
python run_farfan.py --batch plans_directory/

# Health check
python run_farfan.py --health
```

## Architecture

FARFAN 3.0 uses a modular architecture with three core components:

1. **ModuleAdapterRegistry** - Manages 11 adapters with dependency injection
2. **QuestionRouter** - Maps 300 questions to execution chains via `execution_mapping.yaml`
3. **CircuitBreaker** - Provides fault tolerance with state-based failure tracking

See [docs/architecture.md](docs/architecture.md) for detailed architecture documentation.

## 11 Specialized Adapters

| Adapter | Methods | Purpose |
|---------|---------|---------|
| teoria_cambio | 51 | Theory of change, causal pathways |
| analyzer_one | 39 | Municipal development analysis |
| dereck_beach | 89 | CDAF causal deconstruction |
| embedding_policy | 37 | Semantic embeddings (P-D-Q notation) |
| semantic_chunking_policy | 18 | Document segmentation |
| contradiction_detection | 52 | Policy contradiction detection |
| financial_viability | 60 | Financial viability assessment |
| policy_processor | 34 | Industrial policy processing |
| policy_segmenter | 33 | Document segmentation |
| causal_processor | ~30 | Causal relationship processing |
| info_extractor | ~20 | Information extraction |

**Total: 413+ methods**

## Development

### Running Tests

```bash
# All tests
pytest test_*.py

# Specific test suites
python test_architecture_compilation.py
python test_orchestrator_integration.py
python test_circuit_breaker_integration.py
python test_question_router.py
```

### Linting

```bash
black *.py orchestrator/*.py
flake8 *.py orchestrator/*.py
isort *.py orchestrator/*.py
mypy *.py orchestrator/*.py
```

## Documentation

- **[Architecture Documentation](docs/architecture.md)** - System architecture, design patterns, API reference
- **[AGENTS.md](AGENTS.md)** - Development guide for contributors
- **[EXECUTION_MAPPING_MASTER.md](EXECUTION_MAPPING_MASTER.md)** - Question routing mappings

---

## ⚠️ MIGRATION GUIDE - Deprecated Methods

### Overview

During the adapter consolidation phase, several methods were renamed or reorganized. **Alias shims** have been added to maintain backward compatibility, but these are **deprecated** and will be removed in FARFAN 4.0.

### Deprecation Timeline

- **FARFAN 3.0** (Current): Deprecated methods available with warnings
- **FARFAN 3.5** (Q2 2024): Deprecated methods removed from documentation
- **FARFAN 4.0** (Q4 2024): Deprecated methods completely removed

### Old-to-New Method Mappings

#### teoria_cambio Module

| Old Method (DEPRECATED) | New Method | Adapter Class |
|------------------------|------------|---------------|
| `bayesian_engine.compute_score()` | `calculate_bayesian_confidence()` | BayesianEngineAdapter |
| `temporal_logic.validate_sequence()` | `validate_temporal_coherence()` | TemporalLogicAdapter |
| `causal_analysis.extract_chains()` | `extract_causal_chains()` | CausalAnalysisAdapter |
| `financial_trace.analyze_budget()` | `trace_financial_flows()` | FinancialTraceAdapter |

#### analyzer_one Module

| Old Method (DEPRECATED) | New Method | Adapter Class |
|------------------------|------------|---------------|
| `municipal_analyzer.evaluate()` | `analyze_municipal_context()` | AnalyzerOneAdapter |
| `policy_alignment.check()` | `assess_policy_alignment()` | AnalyzerOneAdapter |
| `dimension_scorer.score()` | `score_dimension()` | AnalyzerOneAdapter |

#### dereck_beach Module

| Old Method (DEPRECATED) | New Method | Adapter Class |
|------------------------|------------|---------------|
| `cdaf_analyzer.run_test()` | `apply_evidential_test()` | DerekBeachAdapter |
| `mechanism_evaluator.evaluate()` | `evaluate_causal_mechanism()` | DerekBeachAdapter |
| `beach_test.straw_in_wind()` | `apply_straw_test()` | DerekBeachAdapter |
| `beach_test.smoking_gun()` | `apply_smoking_gun_test()` | DerekBeachAdapter |

#### embedding_policy Module

| Old Method (DEPRECATED) | New Method | Adapter Class |
|------------------------|------------|---------------|
| `semantic_embedder.embed()` | `embed_text()` | EmbeddingPolicyAdapter |
| `similarity_calculator.compute()` | `calculate_semantic_similarity()` | EmbeddingPolicyAdapter |
| `pdq_parser.parse()` | `parse_pdq_notation()` | EmbeddingPolicyAdapter |

#### policy_processor Module

| Old Method (DEPRECATED) | New Method | Adapter Class |
|------------------------|------------|---------------|
| `text_processor.normalize()` | `normalize_unicode()` | PolicyProcessorAdapter |
| `pattern_matcher.match()` | `match_policy_patterns()` | PolicyProcessorAdapter |
| `evidence_scorer.score()` | `compute_evidence_score()` | PolicyProcessorAdapter |

#### policy_segmenter Module

| Old Method (DEPRECATED) | New Method | Adapter Class |
|------------------------|------------|---------------|
| `segmenter.split_document()` | `segment()` | PolicySegmenterAdapter |
| `boundary_detector.detect()` | `detect_section_boundaries()` | PolicySegmenterAdapter |

#### semantic_chunking_policy Module

| Old Method (DEPRECATED) | New Method | Adapter Class |
|------------------------|------------|---------------|
| `chunker.chunk_document()` | `chunk_by_semantics()` | SemanticChunkingPolicyAdapter |
| `bayesian_boundary.score()` | `score_boundary_confidence()` | SemanticChunkingPolicyAdapter |

#### contradiction_detection Module

| Old Method (DEPRECATED) | New Method | Adapter Class |
|------------------------|------------|---------------|
| `contradiction_finder.find()` | `detect_contradictions()` | ContradictionDetectionAdapter |
| `logical_validator.validate()` | `validate_logical_consistency()` | ContradictionDetectionAdapter |

#### financial_viability Module

| Old Method (DEPRECATED) | New Method | Adapter Class |
|------------------------|------------|---------------|
| `budget_analyzer.analyze()` | `analyze_budget_allocation()` | FinancialViabilityAdapter |
| `risk_assessor.assess()` | `assess_financial_risk()` | FinancialViabilityAdapter |
| `pdet_analyzer.analyze_pdet()` | `analyze_pdet_compliance()` | FinancialViabilityAdapter |

### Migration Examples

#### Before (DEPRECATED - will cause warnings)

```python
# Old style - DO NOT USE
from teoria_cambio import BayesianEngine

engine = BayesianEngine()
score = engine.compute_score(text, prior=0.5)  # DeprecationWarning
```

#### After (CURRENT)

```python
# New style - RECOMMENDED
result = module_registry.execute_module_method(
    module_name="teoria_cambio",
    method_name="calculate_bayesian_confidence",
    args=[text],
    kwargs={"prior_confidence": 0.5}
)
score = result.confidence
```

### Checking for Deprecated Usage

```bash
# Search for deprecated method calls in your code
grep -r "compute_score\|split_document\|run_test" your_code/

# Run with deprecation warnings enabled
python -W default::DeprecationWarning run_farfan.py --plan test.pdf
```

### Getting Help

If you need assistance migrating from deprecated methods:

1. Check [docs/architecture.md](docs/architecture.md) for current API documentation
2. Review `orchestrator/execution_mapping.yaml` for current method names
3. Examine `orchestrator/module_adapters.py` for adapter implementations

---

## Tech Stack

- **Language**: Python 3.10+
- **NLP**: spaCy (Spanish models), transformers, sentence-transformers, NLTK
- **ML/Data**: scikit-learn, torch, tensorflow, pandas, numpy
- **Graph Analysis**: networkx, pydot
- **Testing**: pytest
- **Linting**: black, flake8, isort, mypy

## Project Structure

```
FARFAN-3.0/
├── orchestrator/              # Core orchestration engine
│   ├── core_orchestrator.py   # Main orchestrator
│   ├── choreographer.py       # DAG-based execution
│   ├── question_router.py     # Question routing
│   ├── circuit_breaker.py     # Fault tolerance
│   ├── module_adapters.py     # Adapter registry
│   ├── report_assembly.py     # Report generation
│   └── execution_mapping.yaml # Question-to-method mappings
├── docs/                      # Documentation
│   └── architecture.md        # Architecture documentation
├── tests/                     # Test suites
├── Analyzer_one.py            # Municipal analysis adapter
├── teoria_cambio.py           # Theory of change adapter
├── dereck_beach.py            # CDAF adapter
├── emebedding_policy.py       # Semantic embedding adapter
├── policy_processor.py        # Policy processing adapter
├── policy_segmenter.py        # Document segmentation adapter
├── semantic_chunking_policy.py # Semantic chunking adapter
├── contradiction_deteccion.py # Contradiction detection adapter
├── financiero_viabilidad_tablas.py # Financial viability adapter
├── causal_proccesor.py        # Causal processing adapter
├── info_info.py               # Information extraction adapter
├── cuestionario.json          # 300 evaluation questions
├── run_farfan.py              # Main entry point
└── requirements.txt           # Python dependencies
```

## License

[Specify license here]

## Contributing

See [AGENTS.md](AGENTS.md) for development guidelines.

## Authors

FARFAN Integration Team

## Version

3.0.0 (2024)
