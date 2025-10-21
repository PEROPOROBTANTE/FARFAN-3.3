# AGENTS.md - FARFAN 3.0 Development Guide

## Commands

### Setup
```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download es_core_news_sm es_core_news_lg
```

### Build
```bash
pip install -e .
```

### Lint
```bash
black src/ tests/ && flake8 src/ tests/ && isort src/ tests/
```

### Test
```bash
pytest  # All tests
pytest -m unit  # Unit tests only
pytest -m "not slow"  # Skip slow tests
```

### Dev Server
N/A - Pipeline runs via `python run_farfan.py --plan <path>`

## Tech Stack
- **Language**: Python 3.10+
- **NLP**: spaCy (Spanish models), transformers, sentence-transformers, NLTK, stanza
- **ML**: scikit-learn, PyTorch 1.13.1, TensorFlow 2.13.0
- **Testing**: pytest with markers (unit, integration, e2e, slow)
- **Linting**: black, flake8, isort, mypy

## Architecture
- **Hexagonal**: `src/orchestrator/` (core), `src/domain/` (business logic), `src/adapters/` (external)
- **Pipeline**: Deterministic DAG-based execution with fault tolerance (circuit breaker)
- **Config**: YAML-based execution mappings in `config/`

## Code Style
- **Line length**: 88 chars (Black default)
- **Imports**: stdlib → third-party → first-party (isort, Black profile)
- **Type hints**: Optional but encouraged
- **Docstrings**: Google style for public APIs
- **Comments**: Only when necessary for complex logic
