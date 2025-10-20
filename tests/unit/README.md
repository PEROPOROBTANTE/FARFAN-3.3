# Unit Tests

Unit tests for individual FARFAN components using the ModuleController architecture.

## Structure

```
tests/unit/
├── test_orchestrator/          # Orchestrator component tests
│   ├── test_question_router.py
│   ├── test_choreographer.py
│   ├── test_circuit_breaker.py
│   ├── test_report_assembly.py
│   └── test_mapping_loader.py
├── test_domain/                # Domain module tests
│   ├── test_policy_processor.py
│   ├── test_policy_segmenter.py
│   └── test_teoria_cambio.py
├── test_adapters/              # Adapter tests
│   └── test_module_adapters.py
└── test_stages/                # Pipeline stage tests
    └── test_dependency_tracker.py
```

## Testing Principles

1. **Dependency Injection**: All tests use dependency injection to provide mocks and stubs
2. **Isolation**: Each test module is independent and can run in isolation
3. **Determinism**: All tests produce deterministic results with fixed seeds
4. **ModuleController Pattern**: Tests validate the unified module controller architecture

## Running Tests

```bash
# Run all unit tests
pytest tests/unit

# Run specific test module
pytest tests/unit/test_orchestrator/test_question_router.py

# Run with markers
pytest -m unit
pytest -m "unit and not slow"
```

## Test Conventions

- Test files must start with `test_`
- Test classes must start with `Test`
- Test methods must start with `test_`
- Use descriptive names: `test_router_handles_invalid_question_gracefully`
- Use fixtures for common setup
- Mock external dependencies
- Assert both success and failure cases
