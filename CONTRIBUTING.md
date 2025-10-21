# Contributing to FARFAN 3.0

Thank you for your interest in contributing to FARFAN 3.0! This document outlines the standards and practices for contributing to this deterministic policy analysis pipeline.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Guidelines](#development-guidelines)
- [Determinism Requirements](#determinism-requirements)
- [Contract Enforcement](#contract-enforcement)
- [Audit Trail Requirements](#audit-trail-requirements)
- [Testing Standards](#testing-standards)
- [Documentation Standards](#documentation-standards)
- [Submission Process](#submission-process)

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please be respectful, professional, and constructive in all interactions.

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git for version control
- Familiarity with the FARFAN architecture (see [docs/architecture.md](docs/architecture.md))

### Setting Up Your Environment

```bash
# Clone the repository
git clone https://github.com/PEROPOROBTANTE/FARFAN-3.3.git
cd FARFAN-3.3

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install development dependencies
pip install -e ".[dev]"

# Download required NLP models
python -m spacy download es_core_news_sm
python -m spacy download es_core_news_lg
```

## Development Guidelines

### Code Style

- **Line length**: 88 characters (Black default)
- **Import order**: stdlib, third-party, first-party (use `isort`)
- **Type hints**: Required for all public APIs and adapter methods
- **Docstrings**: Google style for all public classes and methods
- **Variable naming**: Descriptive names following PEP 8 conventions

### Linting and Formatting

Before submitting code, run:

```bash
# Format code
black src/ tests/
isort src/ tests/

# Check linting
flake8 src/ tests/

# Type checking
mypy src/
```

## Determinism Requirements

**CRITICAL**: FARFAN 3.0 is a deterministic system. All contributions MUST maintain determinism guarantees.

### Determinism Rules

1. **Fixed Random Seeds**
   - Always use seeded random number generators
   - Document seed values in code comments
   - Never use unseeded `random.random()` or `np.random.random()`
   
   ```python
   # CORRECT
   import random
   random.seed(42)  # Documented seed for reproducibility
   value = random.randint(1, 100)
   
   # INCORRECT
   import random
   value = random.randint(1, 100)  # No seed - non-deterministic!
   ```

2. **No System-Dependent Operations**
   - Avoid `datetime.now()` without explicit timezone
   - Don't use `os.listdir()` without sorting
   - No reliance on dictionary iteration order (use `OrderedDict` or sorted keys)
   
   ```python
   # CORRECT
   import os
   files = sorted(os.listdir(directory))
   
   # INCORRECT
   import os
   files = os.listdir(directory)  # Order not guaranteed!
   ```

3. **Thread-Safe Operations**
   - Use locks for shared state modification
   - Document thread safety guarantees
   - Prefer immutable data structures (see Contract Enforcement)

4. **Floating-Point Determinism**
   - Use `decimal.Decimal` for financial calculations
   - Document precision requirements
   - Be aware of platform-specific floating-point behavior

5. **External API Determinism**
   - Cache external API responses with deterministic keys
   - Use mock responses for testing
   - Document any non-deterministic external dependencies

### Determinism Verification

All code changes affecting determinism MUST pass the determinism gate:

```bash
# Run determinism verification (3 identical executions)
python -m pytest tests/validation/test_determinism.py -v

# CI/CD Gate 4: Determinism Verification
# - Executes pipeline 3 times with seed=42
# - Compares SHA-256 hashes of all outputs
# - Fails if any output differs
```

## Contract Enforcement

FARFAN uses immutable data contracts to ensure type safety and prevent accidental mutations.

### Contract Rules

1. **All Adapter Methods MUST Use Pydantic Models**
   - Input parameters: Frozen Pydantic models
   - Return values: Frozen Pydantic models
   - No mutable dictionaries or lists in signatures
   
   ```python
   from pydantic import BaseModel, Field
   from typing import Tuple
   
   class AnalysisInput(BaseModel, frozen=True):
       """Input contract for analysis adapter."""
       question_id: str = Field(pattern=r"^P\d+-D\d+-Q\d+$")
       policy_text: str
       evidence_sources: Tuple[str, ...]  # Use tuples, not lists
   
   class AnalysisOutput(BaseModel, frozen=True):
       """Output contract for analysis adapter."""
       question_id: str
       score: float = Field(ge=0.0, le=3.0)
       evidence: Tuple[str, ...]  # Immutable
       confidence: float = Field(ge=0.0, le=1.0)
   
   def analyze_policy(input_data: AnalysisInput) -> AnalysisOutput:
       """Analyze policy using immutable contracts."""
       # Implementation must maintain immutability
       return AnalysisOutput(
           question_id=input_data.question_id,
           score=2.5,
           evidence=("Evidence 1", "Evidence 2"),
           confidence=0.85
       )
   ```

2. **Contract Validation**
   - All contracts MUST have corresponding `contract.yaml` files
   - JSON Schema validation enforced in CI/CD Gate 1
   - Method count alignment tracked in `execution_mapping.yaml`

3. **SIN_CARRETA Clause**
   - "Without Load/Cart" - No carrying of mutable state
   - Adapters MUST be stateless or use immutable state
   - Document any state with clear rationale
   
   ```python
   # SIN_CARRETA Compliant
   class StatelessAdapter:
       """Adapter with no mutable state (SIN_CARRETA)."""
       
       def __init__(self, config: Config):
           # Immutable configuration only
           self._config = config  # Config is frozen
       
       def process(self, input_data: Input) -> Output:
           # Pure function - no side effects
           return compute_result(input_data, self._config)
   ```

4. **Contract Version Tracking**
   - All contract changes MUST be documented in CODE_FIX_REPORT.md
   - Breaking changes require migration plan
   - Semantic versioning for contract schemas

### Contract Testing

```bash
# Run contract validation tests
python -m pytest tests/test_immutable_data_contracts.py -v

# Validate all adapter contracts
python validate_contracts.py --strict
```

## Determinism & Contracts (SIN_CARRETA) - ModuleAdapterRegistry

**IMPORTANT**: When working with adapter orchestration, ALWAYS use the canonical `ModuleAdapterRegistry` implementation to maintain determinism and explicit contracts.

### ModuleAdapterRegistry Invocation Contract

All adapter method invocations MUST go through `ModuleAdapterRegistry.execute_module_method()`, NOT direct adapter instance access.

**Required Pattern:**
```python
from src.orchestrator.adapter_registry import ModuleAdapterRegistry

# CORRECT: Use registry's execute_module_method
registry = ModuleAdapterRegistry()
result = registry.execute_module_method(
    module_name="teoria_cambio",
    method_name="calculate_bayesian_confidence",
    args=[plan_text, evidence],
    kwargs={"confidence_threshold": 0.8}
)

# result is a ModuleMethodResult with:
# - status (success|error|unavailable|missing_method|missing_adapter)
# - trace_id (UUID4 for correlation)
# - execution_time (deterministic via injected clock in tests)
# - evidence (structured list from adapter)
# - error_type, error_message (on failure)

# INCORRECT: Direct adapter access bypasses contract
adapter = registry.adapters["teoria_cambio"]  # ❌ DON'T DO THIS
result = adapter.calculate_bayesian_confidence(...)  # ❌ NO CONTRACT
```

### Contract Enforcement Rules

1. **ContractViolation Exceptions**
   - Attempting to execute an unavailable adapter raises `ContractViolation`
   - Attempting to execute a non-existent adapter raises `ContractViolation`
   - Missing methods return `ModuleMethodResult` with `missing_method` status (no exception)
   
   ```python
   from src.orchestrator.adapter_registry import ContractViolation
   
   try:
       result = registry.execute_module_method(
           module_name="unavailable_adapter",
           method_name="some_method"
       )
   except ContractViolation as e:
       logger.error(f"Contract violated: {e}")
       # Handle error - do NOT silently continue
   ```

2. **Deterministic Testing Requirements**
   - ALWAYS inject clock and trace_id_generator in tests
   - NEVER rely on system time or UUID generation in tests
   
   ```python
   def test_adapter_execution():
       # Deterministic clock
       counter = [0.0]
       def clock():
           counter[0] += 0.001
           return counter[0]
       
       # Deterministic trace IDs
       trace_counter = [0]
       def trace_gen():
           trace_counter[0] += 1
           return f"trace-{trace_counter[0]:04d}"
       
       registry = ModuleAdapterRegistry(
           clock=clock,
           trace_id_generator=trace_gen
       )
       
       # Results are now reproducible
       result = registry.execute_module_method(...)
       assert result.trace_id == "trace-0001"  # Deterministic!
   ```

3. **Structured Logging**
   - Every `execute_module_method` call emits structured JSON log
   - Logs include trace_id for correlation across distributed systems
   - Parse logs programmatically for audit trails
   
   ```json
   {
     "module_name": "teoria_cambio",
     "adapter_class": "ModulosAdapter",
     "method_name": "calculate_bayesian_confidence",
     "status": "success",
     "execution_time": 0.123,
     "confidence": 1.0,
     "trace_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
   }
   ```

4. **Method Validation Before Execution**
   - Use `registry.list_adapter_methods(module_name)` for pre-flight checks
   - Validates method existence without executing
   
   ```python
   methods = registry.list_adapter_methods("teoria_cambio")
   if "calculate_bayesian_confidence" in methods:
       result = registry.execute_module_method(...)
   ```

### Migration from Legacy Registry

If you encounter code using the old `AdapterRegistry`:

```python
# OLD (deprecated)
from src.orchestrator.module_adapters import AdapterRegistry
registry = AdapterRegistry()
adapter = registry.get("adapter_name")
result = adapter.method()

# NEW (correct)
from src.orchestrator.adapter_registry import ModuleAdapterRegistry
registry = ModuleAdapterRegistry()
result = registry.execute_module_method(
    module_name="adapter_name",
    method_name="method"
)
```

See `CODE_FIX_REPORT.md` for complete migration guide.

## Audit Trail Requirements

Every code change affecting system behavior MUST have a complete audit trail.

### Audit Trail Components

1. **Change Documentation**
   - File-level change logs in CODE_FIX_REPORT.md
   - Rationale for each modification
   - Test references validating the change
   - SIN_CARRETA compliance notes

2. **Commit Standards**
   - Clear, descriptive commit messages
   - Reference issue/ticket numbers
   - Include "BREAKING CHANGE:" prefix if applicable
   
   ```
   feat(adapter): Add deterministic caching to embedding adapter
   
   - Implements SHA-256-based cache keys for determinism
   - Updates contract.yaml with new caching parameters
   - Tests: test_embedding_cache_determinism.py
   - SIN_CARRETA: Cache stored in immutable dictionary
   
   Refs: #123
   ```

3. **Telemetry Events**
   - All adapter executions MUST emit telemetry events
   - Events follow schema in docs/TELEMETRY_SCHEMA.md
   - Include execution metadata (duration, input hash, output hash)

4. **Execution Tracing**
   - Use structured logging with consistent format
   - Include correlation IDs for request tracing
   - Log all decision points with rationale
   
   ```python
   import logging
   from datetime import datetime, timezone
   
   logger = logging.getLogger(__name__)
   
   def process_question(question_id: str, correlation_id: str):
       """Process question with full audit trail."""
       logger.info(
           "Processing question",
           extra={
               "question_id": question_id,
               "correlation_id": correlation_id,
               "timestamp": datetime.now(timezone.utc).isoformat(),
               "adapter": "teoria_cambio",
               "method": "analyze_theory_of_change"
           }
       )
       # Process...
       logger.info(
           "Question processed",
           extra={
               "question_id": question_id,
               "correlation_id": correlation_id,
               "duration_ms": 1234,
               "result_hash": "sha256:abc123..."
           }
       )
   ```

### Audit Trail Validation

```bash
# Verify audit trail completeness
python validate_audit_trail.py --check-all

# Generate audit report
python generate_audit_report.py --output AUDIT_REPORT.md
```

## Testing Standards

### Test Categories

FARFAN uses pytest markers for test categorization:

- `@pytest.mark.unit` - Fast, isolated unit tests
- `@pytest.mark.integration` - Component integration tests
- `@pytest.mark.e2e` - End-to-end pipeline tests
- `@pytest.mark.slow` - Long-running tests (>5 seconds)
- `@pytest.mark.determinism` - Determinism verification tests

### Test Requirements

1. **All New Features**
   - Minimum 80% code coverage
   - Include positive and negative test cases
   - Test edge cases and boundary conditions

2. **Bug Fixes**
   - Include regression test reproducing the bug
   - Test must fail without the fix
   - Document bug in test docstring

3. **Determinism Tests**
   - Run tests multiple times with same seed
   - Verify outputs are identical
   - Use SHA-256 hashing for comparison

4. **Contract Tests**
   - Validate input/output contracts
   - Test schema violations
   - Verify immutability guarantees

### Running Tests

```bash
# Run all tests
pytest

# Run specific test category
pytest -m unit
pytest -m integration
pytest -m e2e

# Run determinism validation
pytest -m determinism --count=3

# Run with coverage
pytest --cov=src --cov-report=html --cov-report=term

# Skip slow tests
pytest -m "not slow"
```

## Documentation Standards

### Required Documentation

1. **Code Documentation**
   - Docstrings for all public APIs (Google style)
   - Inline comments for complex logic
   - Type hints for all function signatures
   - Rationale comments for non-obvious decisions

2. **Architecture Documentation**
   - Update docs/architecture.md for structural changes
   - Document design decisions in ADRs (Architecture Decision Records)
   - Maintain dependency diagrams

3. **Contract Documentation**
   - Update contract.yaml for all adapter changes
   - Document breaking changes in CHANGELOG.md
   - Maintain API documentation in docs/api/

4. **Change Documentation**
   - Update CODE_FIX_REPORT.md for every code change
   - Document determinism impact
   - Reference test files validating changes

### Documentation Updates

Documentation MUST be updated with every code change affecting:
- Determinism guarantees
- Data contracts
- External interfaces
- Configuration parameters
- System behavior

## Submission Process

### Pull Request Checklist

Before submitting a pull request, ensure:

- [ ] Code follows style guidelines (Black, isort, flake8)
- [ ] Type hints added for all new functions
- [ ] Docstrings added in Google style
- [ ] Tests added with appropriate markers
- [ ] All tests pass locally
- [ ] Coverage maintains or increases
- [ ] Determinism verification passes
- [ ] Contract validation passes
- [ ] CODE_FIX_REPORT.md updated
- [ ] CHANGELOG.md updated (if applicable)
- [ ] Documentation updated
- [ ] Commit messages follow standards
- [ ] No merge conflicts with main branch

### CI/CD Validation Gates

Your pull request will be automatically validated through six gates:

1. **Contract Validation**: All 413 adapter methods have valid contracts
2. **Canary Regression Tests**: SHA-256 hash comparison against baselines
3. **Binding Validation**: Execution mapping integrity
4. **Determinism Verification**: Three identical executions with seed=42
5. **Performance Regression**: P99 latency within 10% of baseline
6. **Schema Drift Detection**: Structural change tracking

All gates MUST pass before merge.

### Review Process

1. Submit pull request with clear description
2. Address automated feedback from CI/CD gates
3. Respond to reviewer comments
4. Update PR based on feedback
5. Request re-review after changes
6. Merge after approval and passing gates

## Getting Help

- Read the [Architecture Guide](docs/architecture/)
- Check the [Implementation Guide](docs/guides/IMPLEMENTATION_GUIDE.md)
- Review existing tests for examples
- Ask questions in GitHub issues
- Contact the development team

## License

By contributing to FARFAN 3.0, you agree that your contributions will be licensed under the project's license.

---

**Remember**: Determinism, immutability, and auditability are core principles. When in doubt, prefer explicit over implicit, and documented over undocumented.
