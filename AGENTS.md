# AGENTS.md - FARFAN 3.0 Development Guide

## Commands

### Setup
```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
# Choose ONE of the following:
# For PyTorch (RECOMMENDED):
pip install -r requirements-torch.txt
# OR for TensorFlow:
# pip install -r requirements-tensorflow.txt
# OR for both (NOT RECOMMENDED - see DEPENDENCY_CONFLICTS.md):
# pip install -r requirements-both.txt
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
- **Language**: Python 3.10-3.11 (3.12 has limited support for deep learning libraries)
- **NLP**: spaCy (Spanish models), transformers, sentence-transformers, NLTK, stanza
- **ML**: scikit-learn, PyTorch 2.0.1 OR TensorFlow 2.13.0 (see DEPENDENCY_CONFLICTS.md)
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
