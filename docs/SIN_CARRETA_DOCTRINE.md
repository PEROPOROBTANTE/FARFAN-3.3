# SIN_CARRETA Doctrine: Comprehensive Guide
## Determinism & Contracts Enforcement for FARFAN 3.0

**Version**: 1.0.0  
**Date**: 2025-10-21  
**Status**: Active

---

## Table of Contents

1. [Introduction](#introduction)
2. [Core Principles](#core-principles)
3. [Determinism Requirements](#determinism-requirements)
4. [Contract Enforcement](#contract-enforcement)
5. [Cognitive Complexity Rationale](#cognitive-complexity-rationale)
6. [CI/CD Validation Gates](#cicd-validation-gates)
7. [Enforcement Scripts](#enforcement-scripts)
8. [Compliance Checklist](#compliance-checklist)

---

## Introduction

The **SIN_CARRETA** doctrine (Spanish: "Without Cart/Load") is the foundational principle ensuring FARFAN 3.0's deterministic, auditable, and maintainable behavior. The doctrine mandates:

- **No mutable state** carried between operations
- **No side effects** without explicit declaration
- **Full determinism** in all computations
- **Complete auditability** through structured contracts

### Etymology

"SIN_CARRETA" translates to "without cart" or "without load" - emphasizing stateless, pure functional operations that don't carry baggage (mutable state) between invocations.

---

## Core Principles

### 1. Statelessness (SIN_CARRETA)

**Principle**: Adapters must not carry mutable state between method invocations.

**Rationale**:
- Enables parallel execution without race conditions
- Guarantees reproducibility across runs
- Simplifies testing and verification
- Eliminates entire classes of bugs

**Implementation**:
```python
# ✅ CORRECT: Stateless adapter
class CompliantAdapter:
    def __init__(self, config: FrozenConfig):
        self._config = config  # Immutable configuration only
    
    def process(self, input_data: FrozenInput) -> FrozenOutput:
        # Pure function - no side effects
        result = compute(input_data, self._config)
        return result

# ❌ INCORRECT: Stateful adapter
class NonCompliantAdapter:
    def __init__(self):
        self._cache = {}  # Mutable state!
    
    def process(self, input_data):
        if input_data.key in self._cache:  # Non-deterministic!
            return self._cache[input_data.key]
        # ...
```

### 2. Immutability

**Principle**: All data contracts must use immutable data structures.

**Rationale**:
- Prevents accidental mutations
- Enables safe parallel processing
- Simplifies reasoning about code
- Documents intent clearly

**Implementation**:
```python
from pydantic import BaseModel, Field
from typing import Tuple

# ✅ CORRECT: Frozen Pydantic models
class Input(BaseModel, frozen=True):
    question_id: str
    evidence: Tuple[str, ...]  # Immutable tuple, not list
    
class Output(BaseModel, frozen=True):
    score: float = Field(ge=0.0, le=3.0)
    evidence: Tuple[str, ...]

# ❌ INCORRECT: Mutable structures
class BadInput:
    def __init__(self):
        self.evidence = []  # Mutable list!
```

### 3. Determinism

**Principle**: Same input must always produce same output.

**Rationale**:
- Enables reproducible research
- Facilitates debugging and testing
- Required for audit compliance
- Builds trust in system outputs

**Requirements**:
- Fixed random seeds
- Sorted iterations (no dict order dependence)
- Explicit timestamps (no `datetime.now()` without timezone)
- Deterministic external dependencies

### 4. Explicit Contracts

**Principle**: Every adapter method must have a formal contract.

**Rationale**:
- Documents expected behavior
- Enables automated validation
- Facilitates integration testing
- Provides API stability guarantees

---

## Determinism Requirements

### Random Number Generation

**Rule**: All randomness must be seeded and documented.

```python
import random
import numpy as np

# ✅ CORRECT: Seeded RNG
def deterministic_function(seed: int = 42):
    """Generate random data deterministically.
    
    Args:
        seed: Random seed for reproducibility (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    return random.randint(1, 100)

# ❌ INCORRECT: Unseeded RNG
def non_deterministic():
    return random.randint(1, 100)  # Different every run!
```

### Time Handling

**Rule**: Use explicit timestamps or injected clocks.

```python
from datetime import datetime, timezone

# ✅ CORRECT: Explicit timezone
def get_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()

# ✅ CORRECT: Injected clock for testing
def process(input_data, clock=datetime.now):
    timestamp = clock(timezone.utc).isoformat()
    return {"timestamp": timestamp, "data": input_data}

# ❌ INCORRECT: Timezone-naive datetime
def bad_timestamp():
    return datetime.now().isoformat()  # System-dependent!
```

### Collection Ordering

**Rule**: Sort collections before iteration.

```python
import os

# ✅ CORRECT: Sorted iteration
def process_files(directory: str):
    files = sorted(os.listdir(directory))
    for file in files:
        process(file)

# ❌ INCORRECT: Unsorted iteration
def non_deterministic_files(directory: str):
    for file in os.listdir(directory):  # Order not guaranteed!
        process(file)
```

### Floating-Point Precision

**Rule**: Use `decimal.Decimal` for financial calculations.

```python
from decimal import Decimal

# ✅ CORRECT: Decimal for money
def calculate_cost(quantity: int, price: Decimal) -> Decimal:
    return Decimal(quantity) * price

# ⚠️ ACCEPTABLE: Float with documented precision limits
def calculate_similarity(vec1: list, vec2: list) -> float:
    """Calculate cosine similarity.
    
    Note: Uses float64 arithmetic. Precision may vary
    across platforms by ±1e-15.
    """
    # ... implementation
```

---

## Contract Enforcement

### Contract Structure

Every adapter method must have a corresponding `contract.yaml`:

```yaml
version: "1.0.0"
method: "analyze_policy"
adapter: "PolicyAnalyzer"
description: "Analyze policy document for compliance"

deterministic: true
sin_carreta_compliant: true

input:
  type: object
  properties:
    question_id:
      type: string
      pattern: "^P\\d+-D\\d+-Q\\d+$"
    policy_text:
      type: string
      minLength: 1
  required:
    - question_id
    - policy_text

output:
  type: object
  properties:
    score:
      type: number
      minimum: 0.0
      maximum: 3.0
    evidence:
      type: array
      items:
        type: string
  required:
    - score
    - evidence

side_effects: []

exceptions:
  - type: ContractViolation
    condition: "Invalid input parameters"
  - type: AdapterUnavailable
    condition: "Adapter not initialized"

performance:
  expected_latency_ms: 100
  complexity: "O(n)"
  
audit_trail:
  telemetry_enabled: true
  log_level: INFO
```

### Contract Validation

Contracts are validated in CI/CD Gate 1:

```bash
python cicd/run_pipeline.py
# Gate 1: Contract Validation
# - Checks 413 expected methods
# - Validates JSON Schema compliance
# - Ensures all methods have contracts
```

### Generating Contracts

Use the contract generator for new methods:

```bash
# Generate contracts for all missing methods
python cicd/generate_contracts.py --missing-only

# Regenerate all contracts
python cicd/generate_contracts.py --all

# Generate for specific adapter
python cicd/generate_contracts.py --adapter teoria_cambio
```

---

## Cognitive Complexity Rationale

### Why Cognitive Complexity Matters

Cognitive complexity measures how difficult code is to understand. It directly impacts:

1. **Audit Trail Clarity**
   - Complex code is harder to trace through execution
   - Difficult to verify all execution paths
   - Obscures deterministic guarantees

2. **Determinism Verification**
   - More paths = more combinations to test
   - Harder to ensure all branches are deterministic
   - Increases risk of non-deterministic behavior

3. **Security Review**
   - Complex code hides security vulnerabilities
   - More places for bugs to lurk
   - Harder to perform thorough code review

4. **Maintenance Cost**
   - Exponential relationship: 2x complexity ≈ 4x cost
   - Difficult to modify without breaking
   - Higher bug introduction rate

5. **Test Coverage**
   - More paths require more test cases
   - Harder to achieve full coverage
   - Integration tests become brittle

### Complexity Thresholds

| Complexity | Status | Action Required |
|-----------|--------|-----------------|
| 0-5 | ✅ Excellent | None |
| 6-10 | ⚠️ Acceptable | Monitor |
| 11-15 | ⚠️ Complex | Plan refactoring |
| 16+ | ❌ Too Complex | Must refactor |

### Measuring Complexity

```bash
# Check entire codebase
python cicd/cognitive_complexity.py --path src/

# Check specific file
python cicd/cognitive_complexity.py --file src/orchestrator/choreographer.py

# Set custom threshold
python cicd/cognitive_complexity.py --threshold 10

# Generate JSON report
python cicd/cognitive_complexity.py --report complexity_report.json
```

### Refactoring Strategies

**Strategy 1: Extract Methods**
```python
# Before (complexity: 12)
def process_complex(data):
    if condition1:
        if condition2:
            for item in items:
                if item.valid:
                    result = transform(item)
                    # ... more logic
    # ... more conditions

# After (complexity: 4 + 3 + 2 = 9 distributed)
def process_complex(data):
    if should_process(data):
        return process_items(data.items)
    return None

def should_process(data):
    return condition1 and condition2

def process_items(items):
    return [transform(item) for item in items if item.valid]
```

**Strategy 2: Use Guard Clauses**
```python
# Before (complexity: 8)
def process(data):
    if data:
        if data.valid:
            if data.ready:
                return compute(data)
            else:
                return None
    return None

# After (complexity: 4)
def process(data):
    if not data:
        return None
    if not data.valid:
        return None
    if not data.ready:
        return None
    return compute(data)
```

**Strategy 3: Simplify Boolean Logic**
```python
# Before (complexity: 6)
if (a and b) or (c and d) or (e and (f or g)):
    process()

# After (complexity: 2)
def should_process():
    return (a and b) or (c and d) or (e and (f or g))

if should_process():
    process()
```

---

## CI/CD Validation Gates

FARFAN enforces SIN_CARRETA doctrine through six validation gates:

### Gate 1: Contract Validation

**Purpose**: Ensure all adapter methods have valid contracts

**Checks**:
- 413 expected methods present
- All methods have contract.yaml files
- JSON Schema validation passes
- Contract completeness

**Remediation**:
```bash
python cicd/generate_contracts.py --missing-only
```

### Gate 2: Canary Regression

**Purpose**: Detect unintended output changes

**Checks**:
- SHA-256 hash comparison against baselines
- Signed changelog entries for changes

**Remediation**:
```bash
# Add changelog entry first
echo "## Method: teoria_cambio - Reason: Updated algorithm" >> CHANGELOG_SIGNED.md

# Then rebaseline
python cicd/rebaseline.py --method teoria_cambio
```

### Gate 3: Binding Validation

**Purpose**: Verify execution mapping integrity

**Checks**:
- No missing source bindings
- No type mismatches
- No circular dependencies

**Remediation**:
```bash
python cicd/fix_bindings.py --auto-correct
```

### Gate 4: Determinism Verification

**Purpose**: Ensure reproducibility

**Checks**:
- Three runs with seed=42
- SHA-256 hash comparison
- All outputs identical

**Requirements**:
- Fixed random seeds
- Sorted iterations
- No system time dependencies

### Gate 5: Performance Regression

**Purpose**: Prevent performance degradation

**Checks**:
- P99 latency within 10% of baseline
- Memory usage reasonable
- No infinite loops

**Remediation**:
```bash
python cicd/profile_adapters.py --optimize
```

### Gate 6: Schema Drift Detection

**Purpose**: Track structural changes

**Checks**:
- file_manifest.json hash comparison
- Migration plan present for changes

**Remediation**:
```bash
python cicd/generate_migration.py
```

---

## Enforcement Scripts

### Available Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `cicd/run_pipeline.py` | Run all validation gates | `python cicd/run_pipeline.py` |
| `cicd/generate_contracts.py` | Generate contract.yaml files | `python cicd/generate_contracts.py --missing-only` |
| `cicd/fix_bindings.py` | Fix execution mapping issues | `python cicd/fix_bindings.py --auto-correct` |
| `cicd/rebaseline.py` | Update canary baselines | `python cicd/rebaseline.py --method <name>` |
| `cicd/cognitive_complexity.py` | Check code complexity | `python cicd/cognitive_complexity.py --path src/` |

### Running Validation Locally

```bash
# Full validation (recommended before PR)
python cicd/run_pipeline.py

# Individual gates
python -c "from cicd.validation_gates import ContractValidator; print(ContractValidator().validate())"
python -c "from cicd.validation_gates import DeterminismValidator; print(DeterminismValidator().validate())"

# Complexity check
python cicd/cognitive_complexity.py --threshold 15
```

### CI Integration

The validation pipeline runs automatically on:
- Pull requests to `main` or `develop`
- Pushes to `main` or `develop`

Results are:
- Posted as PR comments
- Saved as artifacts (30 days retention)
- Block merge if any gate fails

---

## Compliance Checklist

### Before Starting Development

- [ ] Read this SIN_CARRETA doctrine guide
- [ ] Review existing contracts for similar methods
- [ ] Understand determinism requirements
- [ ] Set up local validation scripts

### During Development

- [ ] Use frozen Pydantic models for all contracts
- [ ] Document random seeds explicitly
- [ ] Sort all collection iterations
- [ ] Keep cognitive complexity ≤ 15
- [ ] Use injected clocks for testing
- [ ] Emit structured telemetry events
- [ ] Add rationale comments for complex logic

### Before Committing

- [ ] Run `python cicd/cognitive_complexity.py --path src/`
- [ ] Generate contracts: `python cicd/generate_contracts.py --missing-only`
- [ ] Run full validation: `python cicd/run_pipeline.py`
- [ ] Update CODE_FIX_REPORT.md with changes
- [ ] Add tests for new functionality
- [ ] Document any determinism edge cases

### Before PR Submission

- [ ] All validation gates pass locally
- [ ] No cognitive complexity violations
- [ ] All methods have contracts
- [ ] Changelog updated (if applicable)
- [ ] Documentation updated
- [ ] Tests pass with coverage ≥ 80%

### PR Review Focus

- [ ] SIN_CARRETA compliance verified
- [ ] Contracts are complete and accurate
- [ ] Determinism guarantees maintained
- [ ] Cognitive complexity acceptable
- [ ] Audit trail complete
- [ ] Performance within SLA

---

## Conclusion

The SIN_CARRETA doctrine ensures FARFAN 3.0 remains:

- **Deterministic**: Reproducible results every time
- **Auditable**: Complete execution traces
- **Maintainable**: Low cognitive complexity
- **Reliable**: Contract-enforced behavior
- **Scalable**: Stateless parallel execution

By following these principles and using the enforcement tools, we maintain the highest standards of code quality and system reliability.

---

**Questions or Issues?**
- Review `CONTRIBUTING.md` for detailed guidelines
- Check `CODE_FIX_REPORT.md` for implementation examples
- Contact the development team for clarifications

**Last Updated**: 2025-10-21  
**Version**: 1.0.0  
**Status**: Active - Mandatory Compliance
