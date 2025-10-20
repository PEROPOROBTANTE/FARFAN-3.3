"""
FARFAN Test Suite
=================

Test organization:
- tests/unit/           Unit tests for individual components
- tests/integration/    Integration tests for component interactions
- tests/e2e/           End-to-end tests for full pipeline execution

Test markers:
- @pytest.mark.unit        Unit tests
- @pytest.mark.integration Integration tests
- @pytest.mark.e2e         End-to-end tests
- @pytest.mark.slow        Tests that take significant time

Running tests:
    pytest                      # All tests
    pytest -m unit              # Unit tests only
    pytest -m integration       # Integration tests only
    pytest -m e2e              # E2E tests only
    pytest -m "not slow"       # Skip slow tests
"""
