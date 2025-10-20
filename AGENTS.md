# FARFAN 3.0 - Agent Guide

## Initial Setup
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate (or venv\Scripts\activate on Windows)
pip install --upgrade pip
pip install -r requirements.txt
```

## Commands
- **Build**: N/A (Python project, no build step/compilation needed)
- **Lint**: `flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics`
- **Format**: `black . && isort .`
- **Tests**: `pytest test_architecture_compilation.py test_orchestrator_integration.py -v`
- **Run**: `python run_farfan.py --plan path/to/plan.pdf` or `python run_farfan.py --batch plans/ --workers 8`
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
- **Language**: Python 3.10+ (tested on 3.11)
- **NLP**: spaCy (es_core_news_sm/lg, Spanish models), transformers, BERT models for Spanish text, sentence-transformers, NLTK, Stanza
- **ML/Science**: PyTorch, TensorFlow, sentence-transformers, scikit-learn, NumPy, Pandas
- **Architecture**: Orchestrator pattern with circuit breaker, choreographer, 9 analysis modules, and multi-level reporting

## Structure
- `orchestrator/`: Core orchestration engine (routing, execution, fault tolerance, reporting)
- Root modules: 8-9 specialized analyzers (causal, contradiction, financial, policy processing, semantic chunking)
- `run_farfan.py`: Main CLI entry point for single/batch plan analysis

## Code Style
- Follow PEP 8, use type hints, docstrings for all public methods
- Spanish comments/docstrings throughout (domain is Spanish policy analysis)
- Format with black (line length 100), sort imports with isort
- Modules use dataclasses, avoid mutable defaults, prefer composition over inheritance
- Functional decomposition with adapter pattern for module integration
