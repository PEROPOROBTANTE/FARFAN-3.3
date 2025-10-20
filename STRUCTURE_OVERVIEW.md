# FARFAN 3.0 - Project Structure Overview

## Directory Structure

```
FARFAN-3.0/
│
├── src/                              # Source code (NEW)
│   ├── orchestrator/                # Core orchestration
│   │   ├── question_router.py
│   │   ├── choreographer.py
│   │   ├── circuit_breaker.py
│   │   ├── report_assembly.py
│   │   ├── mapping_loader.py
│   │   └── module_adapters.py
│   ├── domain/                      # Business logic modules
│   │   ├── policy_processor.py
│   │   ├── policy_segmenter.py
│   │   ├── teoria_cambio.py
│   │   └── ...
│   ├── adapters/                    # External interfaces
│   └── stages/                      # Pipeline stages
│       └── dependency_tracker.py
│
├── orchestrator/                     # Legacy (kept for backward compat)
├── domain/                           # Legacy (kept for backward compat)
├── adapters/                         # Legacy (kept for backward compat)
├── stages/                           # Legacy (kept for backward compat)
│
├── config/                           # Configuration
│   ├── execution_mapping.yaml
│   ├── responsibility_map.json
│   └── __init__.py
│
├── tests/                            # Test suite (REORGANIZED)
│   ├── unit/                        # Unit tests
│   │   ├── test_orchestrator/
│   │   ├── test_domain/
│   │   ├── test_adapters/
│   │   └── test_stages/
│   ├── integration/                 # Integration tests
│   │   ├── test_pipeline/
│   │   ├── test_choreographer/
│   │   └── test_circuit_breaker/
│   ├── e2e/                        # End-to-end tests
│   │   └── test_full_pipeline/
│   ├── conftest.py                 # Shared fixtures
│   └── __init__.py
│
├── data/                            # Data artifacts (STRUCTURED)
│   ├── raw/                        # Raw input data
│   │   ├── test_samples/
│   │   └── .gitkeep
│   ├── processed/                  # Intermediate data
│   │   ├── test_samples/
│   │   └── .gitkeep
│   └── output/                     # Final outputs
│       ├── test_results/
│       └── .gitkeep
│
├── logs/                            # Execution traces
│   └── .gitkeep
│
├── docs/                            # Documentation
│   ├── architecture/               # Architecture docs
│   │   ├── README.md
│   │   ├── DEPENDENCY_FRAMEWORK.md
│   │   └── EXECUTION_MAPPING_MASTER.md
│   ├── api/                        # API documentation
│   │   └── README.md
│   ├── guides/                     # User/dev guides
│   │   ├── AGENTS.md
│   │   ├── PROJECT_STRUCTURE.md
│   │   ├── MIGRATION_GUIDE.md
│   │   ├── IMPLEMENTATION_GUIDE.md
│   │   └── VALIDATION_EXECUTION_GUIDE.md
│   └── README.md
│
├── scripts/                         # Utility scripts
│   ├── validate_traceability.py
│   ├── validate_question_routing.py
│   └── ...
│
├── web_dashboard/                   # Web dashboard (separate)
├── cicd/                           # CI/CD configs
│
├── pyproject.toml                  # Package configuration
├── pytest.ini                      # Test configuration
├── requirements.txt                # Dependencies
├── .gitignore                      # Git ignore rules
├── README.md                       # Project README
└── run_farfan.py                  # Main entry point
```

## Key Changes

### 1. Source Code Organization
- **NEW**: `src/` directory for all source code
- **Legacy**: `orchestrator/`, `domain/`, `adapters/`, `stages/` kept for backward compatibility
- **Imports**: Use absolute imports from `src.` (e.g., `from src.orchestrator.question_router import QuestionRouter`)

### 2. Test Organization
- **Unit tests**: `tests/unit/test_<package>/test_<module>.py`
- **Integration tests**: `tests/integration/test_<component>/`
- **E2E tests**: `tests/e2e/test_full_pipeline/`
- **Shared fixtures**: `tests/conftest.py`

### 3. Data Management
- **Raw**: `data/raw/` - Input documents
- **Processed**: `data/processed/` - Intermediate results
- **Output**: `data/output/` - Final reports

### 4. Documentation
- **Architecture**: System design and patterns
- **API**: Component interfaces
- **Guides**: User and developer documentation

## Import Guidelines

### Correct (Absolute imports from src)
```python
from src.orchestrator.question_router import QuestionRouter
from src.domain.policy_processor import PolicyProcessor
from src.stages.dependency_tracker import DependencyTracker
```

### Incorrect (Relative or legacy imports)
```python
from orchestrator.question_router import QuestionRouter  # Legacy
from ..domain.policy_processor import PolicyProcessor    # Relative
```

## Test Guidelines

### Unit Tests
```bash
pytest tests/unit/                    # All unit tests
pytest tests/unit/test_orchestrator/  # Specific package
pytest -m unit                        # Using markers
```

### Integration Tests
```bash
pytest tests/integration/             # All integration tests
pytest -m integration                 # Using markers
```

### E2E Tests
```bash
pytest tests/e2e/                     # All e2e tests
pytest -m e2e                         # Using markers
```

## Configuration

### Package Configuration: `pyproject.toml`
- Package structure
- Dependencies
- Build configuration
- Tool settings (black, isort, mypy)

### Test Configuration: `pytest.ini`
- Test discovery paths
- Test markers
- Logging configuration

### Git Configuration: `.gitignore`
- Source code exclusions
- Data artifact patterns
- Log exclusions
- Build artifact patterns

## Deleted Components

### Test Files (Outdated)
- `tests/unit/test_contract_validator.py` - Pre-dates ModuleController
- `tests/unit/test_fault_injection_framework.py` - Legacy testing
- `tests/validation/test_interface_contracts.py` - Outdated validation
- `tests/contracts/` - Legacy contract testing
- `tests/fault_injection/` - Replaced by integration tests
- `tests/e2e/test_canary_system.py` - Legacy canary system
- `tests/e2e/canary_*.py` - Legacy canary generators

### Legacy Directories
- `delivery_package/` - Archived refactoring work
- `unified_diffs/` - Archived diffs
- `artifact/` - Temporary build artifacts

### Temporary Files
- `orchestrator_integration_report.txt`
- `test_architecture.log`
- `validation_report.txt`

## Migration Status

✅ **Completed**:
- Source code reorganization (`src/` package)
- Test structure reorganization
- Data directory structure
- Documentation reorganization
- Configuration updates (pyproject.toml, pytest.ini, .gitignore)
- Legacy file cleanup

⚠️ **Pending** (for backward compatibility):
- Update all imports from legacy to `src.` prefix
- Remove legacy `orchestrator/`, `domain/`, `adapters/`, `stages/` directories
- Update CI/CD pipelines to use new structure

## Next Steps

1. **Update imports**: Migrate all imports to use `src.` prefix
2. **Write new tests**: Add tests following new structure
3. **Remove legacy code**: After verifying all imports are migrated
4. **Update CI/CD**: Configure pipelines for new structure
5. **Generate API docs**: Use Sphinx or similar for API documentation

## Support

For questions or issues:
1. Check [Project Structure Guide](docs/guides/PROJECT_STRUCTURE.md)
2. Review [Migration Guide](docs/guides/MIGRATION_GUIDE.md)
3. Contact development team
