# FARFAN 3.0 - Development Guide

## Commands

### Setup
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Build
Not applicable (Python project)

### Lint
```bash
black *.py orchestrator/*.py
flake8 *.py orchestrator/*.py
isort *.py orchestrator/*.py
mypy *.py orchestrator/*.py
```

### Test
```bash
pytest test_*.py
python test_architecture_compilation.py
python test_orchestrator_integration.py
```

### Run
```bash
python run_farfan.py --plan <path> [--workers N]
python run_farfan.py --health
```

## Tech Stack
- **Language:** Python 3.10+
- **NLP:** spaCy (Spanish models), transformers, sentence-transformers, NLTK
- **ML/Data:** scikit-learn, torch, tensorflow, pandas, numpy
- **Testing:** pytest
- **Linting:** black, flake8, isort, mypy

## Architecture
- **orchestrator/**: Core orchestration engine with router, choreographer, circuit breaker, adapters, reports
- **Main modules:** Analyzer_one.py, policy_processor.py, causal_proccesor.py, contradiction_deteccion.py, etc.
- **Entry point:** run_farfan.py - Main entry point for single/batch plan analysis

## Code Style
- Follow PEP 8, use type hints
- Classes: PascalCase, functions/vars: snake_case
- Comprehensive docstrings for public APIs
- No logging of secrets/keys
- Spanish text in comments/docstrings is acceptable (domain-specific)
