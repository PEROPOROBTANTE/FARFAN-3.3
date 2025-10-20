# Migration Guide: Old Structure → New Structure

## Overview

This guide explains the changes from the old project structure to the new deterministic pipeline architecture.

## Structural Changes

### Source Code Location

**OLD**: Code scattered in root directories
```
FARFAN-3.0/
├── orchestrator/
├── domain/
├── adapters/
└── stages/
```

**NEW**: Centralized source code
```
FARFAN-3.0/
└── src/
    ├── orchestrator/
    ├── domain/
    ├── adapters/
    └── stages/
```

### Import Statements

**OLD**: Relative imports
```python
from orchestrator.question_router import QuestionRouter
from domain.policy_processor import PolicyProcessor
```

**NEW**: Absolute imports from src
```python
from src.orchestrator.question_router import QuestionRouter
from src.domain.policy_processor import PolicyProcessor
```

### Test Organization

**OLD**: Flat test structure
```
tests/
├── test_adapter_old.py
├── test_canary_system.py
└── fault_injection/
```

**NEW**: Organized by test type
```
tests/
├── unit/
│   ├── test_orchestrator/
│   ├── test_domain/
│   ├── test_adapters/
│   └── test_stages/
├── integration/
│   ├── test_pipeline/
│   ├── test_choreographer/
│   └── test_circuit_breaker/
└── e2e/
    └── test_full_pipeline/
```

### Configuration

**OLD**: Mixed configuration
```
config/
├── various_configs.yaml
└── execution_mapping.json
```

**NEW**: Structured configuration
```
config/
├── execution_mapping.yaml
├── module_config.yaml
└── pipeline_config.yaml
```

### Data Management

**OLD**: Unstructured data directories
```
data/
└── mixed_files/
```

**NEW**: Structured data pipeline
```
data/
├── raw/          # Input data
├── processed/    # Intermediate results
└── output/       # Final outputs
```

## Removed Components

### Outdated Tests
- `tests/unit/test_contract_validator.py` - Pre-dates ModuleController
- `tests/unit/test_fault_injection_framework.py` - Legacy testing approach
- `tests/validation/test_interface_contracts.py` - Outdated adapter validation
- `tests/contracts/` - Legacy contract testing
- `tests/fault_injection/` - Replaced by integration tests
- `tests/e2e/test_canary_system.py` - Legacy canary system
- `tests/e2e/canary_*.py` - Legacy canary generators

### Legacy Directories
- `delivery_package/` - Archived refactoring work
- `unified_diffs/` - Archived diff files
- `artifact/` - Temporary build artifacts

### Temporary Files
- `orchestrator_integration_report.txt`
- `test_architecture.log`
- `validation_report.txt`

## Updated Configuration

### pyproject.toml

**Changes**:
- Package structure now points to `src/`
- Includes proper `known_first_party` for isort
- Updated excludes for new structure
- Added test markers for unit/integration/e2e

### .gitignore

**Added**:
- Data artifact patterns (`data/raw/*`, `data/processed/*`, `data/output/*`)
- Logs directory (`logs/`)
- Model files (`.model`, `.pkl`, `.h5`, `.pt`)
- Documentation builds (`docs/_build/`)

### pytest.ini

**Added**:
- `pythonpath = ["."]` for import resolution
- Test markers (unit, integration, e2e, slow)

## Migration Steps

### For Developers

1. **Update Imports**
   ```bash
   # Find all import statements
   grep -r "from orchestrator\|from domain\|from adapters\|from stages" . --include="*.py"
   
   # Update to use src prefix
   # OLD: from orchestrator.question_router import QuestionRouter
   # NEW: from src.orchestrator.question_router import QuestionRouter
   ```

2. **Update Test Paths**
   ```bash
   # Move unit tests to appropriate subdirectory
   # tests/test_foo.py → tests/unit/test_orchestrator/test_foo.py
   ```

3. **Update Data Paths**
   ```python
   # OLD: data_path = "data/input.json"
   # NEW: data_path = "data/raw/input.json"
   ```

4. **Run Tests**
   ```bash
   pytest tests/unit/       # Unit tests
   pytest tests/integration/ # Integration tests
   pytest tests/e2e/        # E2E tests
   ```

### For CI/CD

1. **Update Build Scripts**
   - Use `pytest -m unit` for unit test stage
   - Use `pytest -m integration` for integration test stage
   - Use `pytest -m e2e` for e2e test stage

2. **Update Linting**
   - Include `src/` in linting paths
   - Exclude old directories

3. **Update Coverage**
   - Coverage reports should target `src/` package

## Breaking Changes

### Import Resolution
All imports must use absolute paths from `src/`. Relative imports will fail.

### Test Discovery
Tests must be in appropriate subdirectories (`unit/`, `integration/`, `e2e/`) to be discovered.

### Data Paths
Data files must be organized into `raw/`, `processed/`, `output/` subdirectories.

## Benefits of New Structure

1. **Clear Separation**: Source, tests, config, data, and docs are clearly separated
2. **Import Consistency**: All imports use absolute paths from defined root
3. **Test Organization**: Tests organized by type for better CI/CD integration
4. **Pipeline Convention**: Data flow through raw → processed → output
5. **Maintainability**: Clear conventions reduce cognitive load

## Support

For questions about the migration:
1. Check [Project Structure Guide](PROJECT_STRUCTURE.md)
2. Review [Development Guide](AGENTS.md)
3. Contact development team
