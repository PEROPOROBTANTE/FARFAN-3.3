# Integration Tests

Integration tests for FARFAN component interactions.

## Structure

```
tests/integration/
├── test_pipeline/          # Full pipeline integration tests
├── test_choreographer/     # Choreographer coordination tests
└── test_circuit_breaker/   # Circuit breaker integration tests
```

## Test Scope

Integration tests validate:
- Component interactions (e.g., router → choreographer → modules)
- Data flow between components
- Error propagation and handling
- Circuit breaker behavior with real adapters
- Dependency resolution in choreographer

## Running Tests

```bash
# Run all integration tests
pytest tests/integration

# Run specific integration test suite
pytest tests/integration/test_choreographer/

# Run with markers
pytest -m integration
```

## Test Conventions

- Use real components where possible, mock only external dependencies
- Test error paths and edge cases
- Validate data transformations between components
- Test concurrent execution scenarios
- Verify circuit breaker state transitions
