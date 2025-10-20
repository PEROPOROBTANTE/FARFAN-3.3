# FARFAN 3.0 - Agent Guide

## Setup
```bash
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install --upgrade pip
pip install -r requirements.txt
```

## Commands
- **Build**: N/A (Python project, no build step)
- **Lint**: `flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics`
- **Test**: `pytest test_architecture_compilation.py test_orchestrator_integration.py -v`
- **Dev**: `python run_farfan.py --help` (see usage options below)

## Running FARFAN
```bash
# Single plan analysis
python run_farfan.py --plan path/to/plan.pdf

# Batch analysis
python run_farfan.py --batch plans_directory/ --workers 8

# System health check
python run_farfan.py --health
```

## Tech Stack
- **Language**: Python 3.10+
- **NLP**: spaCy (Spanish models), transformers, sentence-transformers, NLTK, Stanza
- **ML/Science**: PyTorch, TensorFlow, scikit-learn, NumPy, Pandas
- **Architecture**: Orchestrator pattern with circuit breaker, choreographer, and multi-level reporting

## Structure
- `orchestrator/`: Core orchestration engine (routing, execution, fault tolerance, reporting)
- `run_farfan.py`: Main CLI entry point
- Root-level modules: Policy processing, contradiction detection, financial analysis, semantic chunking

## Code Style
- Spanish comments/docstrings throughout (domain is Spanish policy analysis)
- Black formatting, isort for imports, type hints encouraged
- Functional decomposition with adapter pattern for module integration
