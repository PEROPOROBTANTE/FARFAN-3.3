# End-to-End Tests

End-to-end tests for complete FARFAN pipeline execution.

## Structure

```
tests/e2e/
└── test_full_pipeline/     # Complete pipeline execution tests
```

## Test Scope

E2E tests validate:
- Full pipeline execution from input to output
- All components working together
- Real data processing
- Performance benchmarks
- Deterministic output verification

## Running Tests

```bash
# Run all e2e tests
pytest tests/e2e

# Run with verbose output
pytest tests/e2e -v -s

# Run specific test
pytest tests/e2e/test_full_pipeline/test_complete_execution.py

# Run with markers
pytest -m e2e
```

## Test Data

E2E tests use data from:
- `data/raw/test_samples/` - Sample policy documents
- `data/processed/test_samples/` - Processed test data
- `data/output/test_results/` - Expected outputs for validation

## Test Conventions

- Use real policy documents for testing
- Validate output determinism (same input → same output)
- Measure execution time and resource usage
- Test complete error recovery scenarios
- Generate comprehensive execution reports
