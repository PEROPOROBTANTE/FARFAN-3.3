# AGENTS.md - FARFAN 3.0 Development Guide

## Commands

### Setup
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download es_core_news_sm
```

### Build
No build step required (Python project)

### Lint
```bash
flake8 . --max-line-length=120 --exclude=venv,__pycache__,.git
black --check . --exclude=venv
```

### Test
```bash
python test_architecture_compilation.py
python test_orchestrator_integration.py
pytest test_*.py -v
```

### Run
```bash
# Analyze single plan
python run_farfan.py --plan path/to/plan.pdf

# Analyze batch of plans
python run_farfan.py --batch plans/ --max-plans 170 --workers 8

# System health check
python run_farfan.py --health

# Run canary regression tests
python tests/canary_runner.py
```

## Tech Stack
- **Language**: Python 3.10+
- **NLP**: spaCy (Spanish), transformers, sentence-transformers, BETO (Spanish BERT)
- **ML/Analytics**: PyTorch, TensorFlow, scikit-learn, numpy, pandas, NetworkX (causal graphs)
- **Architecture**: Orchestrator pattern with 8+ specialized modules (dereck_beach, policy_processor, teoria_cambio, etc.)

## Repo Structure
- `orchestrator/` - Core orchestration engine with 9 module adapters
- `run_farfan.py` - Main entry point for policy analysis
- `tests/` - Canary regression detection system (413 adapter method tests)
- Individual analyzer modules: `Analyzer_one.py`, `causal_proccesor.py`, `contradiction_deteccion.py`, `dereck_beach.py`, etc.
- `requirements.txt` - Python dependencies (includes spaCy models via URLs)

## Code Style
- **Formatting**: Follow existing style, prefer Black defaults (line length: 120 chars)
- **Imports**: Group by stdlib, third-party, local; no unused imports
- **Docstrings**: Use for public methods, include types and examples
- **Naming**: snake_case for functions/variables, PascalCase for classes (Spanish variable names for domain logic, English for infrastructure)
- **Type hints**: Required; use dataclasses for config
- **Comments**: Minimal; code should be self-documenting
