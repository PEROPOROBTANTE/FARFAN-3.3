# FARFAN 3.0 Project Structure

## Overview

FARFAN 3.0 follows deterministic pipeline best practices with clear separation of concerns.

## Directory Structure

```
FARFAN-3.0/
├── src/                        # Source code
│   ├── orchestrator/          # Core orchestration components
│   │   ├── question_router.py
│   │   ├── choreographer.py
│   │   ├── circuit_breaker.py
│   │   ├── report_assembly.py
│   │   ├── mapping_loader.py
│   │   └── module_adapters.py
│   ├── domain/                # Business logic modules
│   │   ├── policy_processor.py
│   │   ├── policy_segmenter.py
│   │   ├── teoria_cambio.py
│   │   └── ...
│   ├── adapters/              # External interface adapters
│   └── stages/                # Pipeline stage definitions
│       └── dependency_tracker.py
│
├── config/                     # Configuration files
│   ├── execution_mapping.yaml
│   ├── module_config.yaml
│   └── pipeline_config.yaml
│
├── tests/                      # Test suite
│   ├── unit/                  # Unit tests
│   │   ├── test_orchestrator/
│   │   ├── test_domain/
│   │   ├── test_adapters/
│   │   └── test_stages/
│   ├── integration/           # Integration tests
│   │   ├── test_pipeline/
│   │   ├── test_choreographer/
│   │   └── test_circuit_breaker/
│   ├── e2e/                   # End-to-end tests
│   │   └── test_full_pipeline/
│   └── conftest.py            # Shared fixtures
│
├── data/                       # Data artifacts
│   ├── raw/                   # Raw input data
│   ├── processed/             # Processed intermediate data
│   └── output/                # Final output data
│
├── logs/                       # Execution traces and logs
│
├── docs/                       # Documentation
│   ├── architecture/          # Architecture documentation
│   ├── api/                   # API documentation
│   └── guides/                # User guides
│
├── scripts/                    # Utility scripts
│
├── pyproject.toml             # Package configuration
├── pytest.ini                 # Test configuration
├── requirements.txt           # Dependencies
├── .gitignore                 # Git ignore rules
└── run_farfan.py             # Main entry point
```

## Source Code Organization (`src/`)

### Orchestrator Package
Core orchestration components that coordinate pipeline execution:
- **QuestionRouter**: Routes questions to appropriate modules
- **Choreographer**: Manages module execution order and dependencies
- **CircuitBreaker**: Handles fault tolerance and error recovery
- **ReportAssembly**: Assembles final reports from module outputs
- **MappingLoader**: Loads execution mapping configurations
- **ModuleAdapters**: Registry for module instances

### Domain Package
Business logic modules for policy analysis:
- **PolicyProcessor**: Policy document processing and normalization
- **PolicySegmenter**: Document segmentation
- **TeoriaCambio**: Theory of change analysis
- **EmbeddingPolicy**: Policy embedding generation
- **SemanticChunkingPolicy**: Semantic text chunking
- Additional domain modules for specialized analysis

### Adapters Package
External interface adapters that provide unified interfaces to domain modules.

### Stages Package
Pipeline stage definitions and dependency management.

## Configuration (`config/`)

Configuration files in YAML/JSON format:
- `execution_mapping.yaml`: Question-to-module routing rules
- `module_config.yaml`: Module-specific configuration
- `pipeline_config.yaml`: Pipeline execution settings

## Tests (`tests/`)

### Unit Tests
Test individual components in isolation with mocked dependencies.

### Integration Tests
Test component interactions and data flow between modules.

### End-to-End Tests
Test complete pipeline execution with real data.

### Test Fixtures
Shared test fixtures in `conftest.py` for consistent test setup.

## Data Management (`data/`)

### Raw Data
Unprocessed input files (policy documents, questionnaires).

### Processed Data
Intermediate processing results.

### Output Data
Final analysis results and reports.

## Logging (`logs/`)

Execution traces, error logs, and audit trails.

## Documentation (`docs/`)

### Architecture
System architecture, design decisions, and patterns.

### API
API documentation for modules and interfaces.

### Guides
User guides, development guides, and tutorials.

## Import Conventions

All imports use absolute paths from the project root:

```python
# Correct - absolute imports from src
from src.orchestrator.question_router import QuestionRouter
from src.domain.policy_processor import PolicyProcessor
from src.stages.dependency_tracker import DependencyTracker

# Incorrect - relative imports
from ..orchestrator.question_router import QuestionRouter  # Don't use
```

## Package Configuration

`pyproject.toml` defines the package structure and enables consistent imports:
- Package root: `src/`
- Main packages: `orchestrator`, `domain`, `adapters`, `stages`
- Test configuration in `[tool.pytest.ini_options]`
- Code style in `[tool.black]` and `[tool.isort]`

## Running the Pipeline

```bash
# Main entry point
python run_farfan.py --plan <path> [--workers N]

# Batch processing
python run_farfan.py --batch <dir> [--max-plans N]
```

## Development Workflow

1. **Code**: Develop in `src/` packages
2. **Test**: Write tests in appropriate `tests/` directory
3. **Lint**: Run `black`, `flake8`, `isort`
4. **Test**: Run `pytest` with appropriate markers
5. **Document**: Update relevant docs in `docs/`
