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
python run_farfan.py --plan path/to/plan.pdf
python run_farfan.py --health
```

## Tech Stack
- **Language:** Python 3.11+
- **NLP:** spaCy (Spanish models), transformers, sentence-transformers
- **Data:** pandas, numpy, scikit-learn
- **Testing:** pytest
- **Linting:** black, flake8, mypy

## Architecture
- **orchestrator/**: Core orchestration engine with module adapters, choreographer, circuit breaker, question router
- **Main modules:** Analyzer_one.py, policy_processor.py, causal_proccesor.py, contradiction_deteccion.py, etc.
- **Entry point:** run_farfan.py

## Code Style
- Follow PEP 8 (enforced by black and flake8)
- Type hints required (checked by mypy)
- Spanish text in comments/docstrings is acceptable (domain-specific)
- Module files: lowercase_underscore.py
