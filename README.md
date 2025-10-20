# FARFAN 3.0 - Deterministic Policy Analysis Pipeline

A deterministic pipeline for comprehensive policy document analysis using NLP and ML techniques.

## Overview

FARFAN 3.0 is a production-ready policy analysis system that processes policy documents and questionnaires through a deterministic pipeline, ensuring reproducible and auditable results.

### Key Features

- **Deterministic Execution**: Same input always produces same output
- **Fault Tolerance**: Circuit breaker pattern prevents cascading failures
- **Dependency Management**: DAG-based module orchestration
- **Scalable Architecture**: Parallel execution of independent modules
- **Comprehensive Testing**: Unit, integration, and E2E test coverage
- **Audit Trail**: Complete execution logging and traceability

## Quick Start

### Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Download spaCy models
python -m spacy download es_core_news_sm
python -m spacy download es_core_news_lg
```

### Running the Pipeline

```bash
# Single plan analysis
python run_farfan.py --plan path/to/plan.json

# Batch processing
python run_farfan.py --batch path/to/plans/ --max-plans 10

# With custom workers
python run_farfan.py --plan path/to/plan.json --workers 4
```

## Project Structure

```
FARFAN-3.0/
├── src/                    # Source code
│   ├── orchestrator/      # Core orchestration
│   ├── domain/            # Business logic
│   ├── adapters/          # External interfaces
│   └── stages/            # Pipeline stages
├── config/                # Configuration files
├── tests/                 # Test suite
│   ├── unit/             # Unit tests
│   ├── integration/      # Integration tests
│   └── e2e/              # End-to-end tests
├── data/                  # Data artifacts
│   ├── raw/              # Input data
│   ├── processed/        # Intermediate data
│   └── output/           # Results
├── logs/                  # Execution traces
└── docs/                  # Documentation
```

See [Project Structure Guide](docs/guides/PROJECT_STRUCTURE.md) for detailed organization.

## Development

### Setup Development Environment

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run linters
black src/ tests/
flake8 src/ tests/
isort src/ tests/

# Run type checking
mypy src/
```

### Running Tests

```bash
# All tests
pytest

# Unit tests only
pytest -m unit

# Integration tests
pytest -m integration

# E2E tests
pytest -m e2e

# Skip slow tests
pytest -m "not slow"

# With coverage
pytest --cov=src --cov-report=html
```

### Code Style

- **Line length**: 88 characters (Black default)
- **Import order**: stdlib, third-party, first-party
- **Type hints**: Encouraged but not required
- **Docstrings**: Google style for public APIs

## Architecture

### Core Components

1. **QuestionRouter**: Routes questions to appropriate modules based on execution mapping
2. **Choreographer**: Orchestrates module execution in correct dependency order
3. **CircuitBreaker**: Prevents cascading failures with fault tolerance
4. **ReportAssembly**: Assembles final reports from module outputs
5. **MappingLoader**: Loads and validates execution mappings

### Pipeline Stages

1. **Document Preprocessing**: Policy segmentation and normalization
2. **Analysis**: Semantic analysis, embedding generation, theory of change
3. **Synthesis**: Financial viability, contradiction detection, final reporting

### Module Execution Order

Modules execute in dependency waves for optimal parallelization:
- **Wave 1**: Policy segmentation, text normalization
- **Wave 2**: Semantic chunking, embedding generation
- **Wave 3**: Municipal analysis, theory of change
- **Wave 4**: Causal analysis, contradiction detection
- **Wave 5**: Financial viability synthesis

## Configuration

Configuration files in `config/`:
- `execution_mapping.yaml`: Question routing rules
- `module_config.yaml`: Module-specific settings
- `pipeline_config.yaml`: Pipeline execution parameters

## Documentation

- [Architecture Guide](docs/architecture/)
- [API Documentation](docs/api/)
- [Development Guide](docs/guides/AGENTS.md)
- [Project Structure](docs/guides/PROJECT_STRUCTURE.md)

## Tech Stack

- **Python**: 3.10+
- **NLP**: spaCy, transformers, sentence-transformers, NLTK, stanza
- **ML**: scikit-learn, PyTorch, TensorFlow
- **Testing**: pytest with custom markers
- **Code Quality**: black, flake8, isort, mypy

## License

Copyright © 2024 FARFAN 3.0 Team. All rights reserved.

## Support

For issues, questions, or contributions, please refer to the project documentation or contact the development team.
