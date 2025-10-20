# AGENTS.md - FARFAN 3.0 Policy Analysis System

## Commands

**Setup:**
```bash
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Run Tests:** `python test_architecture_compilation.py` or `python test_orchestrator_integration.py`  
**Run Lint:** `flake8 orchestrator/ *.py` or `black --check .`  
**Run Main:** `python run_farfan.py --plan path/to/plan.pdf` or `python run_farfan.py --batch plans_directory/`

## Tech Stack & Architecture

- **Language:** Python 3.10+ (tested on 3.11.9)
- **NLP:** spaCy, transformers, sentence-transformers (Spanish models: es-core-news-lg, BETO)
- **Data:** pandas, numpy, scikit-learn
- **Architecture:** Orchestrator pattern with modular adapters for policy analysis
  - `orchestrator/`: Core orchestration engine with circuit breaker, choreographer, question router
  - Root modules: 8+ specialized analyzers (causal, contradiction, financial viability, theory of change, etc.)
  - Multi-level reporting: MICRO/MESO/MACRO analysis of 300-question questionnaire

## Code Style

- **Conventions:** Type hints, dataclasses, docstrings with module-level architecture descriptions
- **No comments:** Code is self-documenting; docstrings explain "why", not "what"
- **Testing:** Custom test runners (no pytest framework); validate imports, class structure, and invocation chains
