# Contributing to FARFAN 3.0

Thank you for your interest in contributing to FARFAN 3.0! This document provides guidelines for maintaining code quality, ensuring compliance with SIN_CARRETA doctrine, and preserving the deterministic, auditable nature of the system.

---

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Architectural Principles](#architectural-principles)
3. [Questionnaire Data Guidelines](#questionnaire-data-guidelines)
4. [Development Workflow](#development-workflow)
5. [Testing Requirements](#testing-requirements)
6. [SIN_CARRETA Compliance](#sin_carreta-compliance)
7. [Pull Request Process](#pull-request-process)

---

## Code of Conduct

### Core Values

- **Determinism First:** All code must produce reproducible results
- **Contract Integrity:** Maintain strict interfaces and type safety
- **No Silent Failures:** Explicit error handling required
- **Traceability:** Document all data flows and decisions
- **Auditability:** Log all significant operations

---

## Architectural Principles

### 1. Single Source of Truth

**CRITICAL RULE:**  
All questionnaire data MUST be accessed through `QuestionnaireParser`.

✅ **CORRECT:**
```python
from orchestrator.questionnaire_parser import get_questionnaire_parser

parser = get_questionnaire_parser()
dimension = parser.get_dimension("D1")
rubric = parser.get_rubric_for_question("D1-Q1")
```

❌ **PROHIBITED:**
```python
import json

# NEVER do this!
with open('cuestionario.json', 'r') as f:
    data = json.load(f)
```

**Rationale:** Direct JSON loading bypasses validation, breaks caching, and violates determinism guarantees.

### 2. Immutability

All data structures should be immutable where possible.

✅ **CORRECT:**
```python
from dataclasses import dataclass

@dataclass(frozen=True)
class QuestionData:
    id: str
    text: str
    rubric: Dict[str, float]
```

❌ **INCORRECT:**
```python
@dataclass
class QuestionData:  # Mutable - can lead to bugs
    id: str
    text: str
```

### 3. Explicit Error Handling

Never suppress exceptions or convert failures to warnings.

✅ **CORRECT:**
```python
def load_plan(path: Path) -> Plan:
    if not path.exists():
        raise FileNotFoundError(f"Plan not found: {path}")
    
    try:
        return Plan.from_file(path)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid plan format: {e}")
```

❌ **PROHIBITED:**
```python
def load_plan(path: Path) -> Optional[Plan]:
    try:
        return Plan.from_file(path)
    except Exception:
        logger.warning("Failed to load plan")  # SILENT FAILURE!
        return None
```

---

## Questionnaire Data Guidelines

### Modifying cuestionario.json

**IMPORTANT:** Changes to `cuestionario.json` must be:
1. **Syntax-only** for corrections
2. **Documented** in CODE_FIX_REPORT.md
3. **Validated** before commit
4. **Versioned** (update metadata.version)

### Validation Checklist

Before committing changes to `cuestionario.json`:

```bash
# 1. Validate JSON syntax
python3 -m json.tool cuestionario.json > /dev/null

# 2. Test parser loading
python3 << 'EOF'
from orchestrator.questionnaire_parser import QuestionnaireParser
parser = QuestionnaireParser()
assert parser.total_questions == 300
assert len(parser.get_all_dimensions()) == 6
assert len(parser.get_all_policy_points()) == 10
print("✓ Validation passed")
EOF

# 3. Check no data loss
git diff cuestionario.json | grep "^-" | grep -v "^---" | wc -l
# Should be minimal for syntax fixes
```

### Required Structure

Maintain these invariants:
- 6 dimensions (D1-D6)
- 10 policy points (P1-P10)
- 30 base questions (5 per dimension)
- Standard rubric levels: EXCELENTE, BUENO, ACEPTABLE, INSUFICIENTE

---

## Development Workflow

### 1. Setting Up

```bash
# Clone repository
git clone https://github.com/kkkkknhh/FARFAN-3.0.git
cd FARFAN-3.0

# Install dependencies
pip install -r requirements.txt

# Verify parser initialization
python3 -c "from orchestrator.questionnaire_parser import get_questionnaire_parser; get_questionnaire_parser()"
```

### 2. Creating a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 3. Making Changes

**For New Modules:**

1. Add module to `orchestrator/config.py`:
```python
modules: Dict[str, ModuleConfig] = field(default_factory=lambda: {
    "your_module": ModuleConfig(
        name="Your Module",
        file_path=Path("your_module.py"),
        entry_function="YourClass",
        required_dimensions=["D1", "D2"],
        priority=1
    ),
    # ... existing modules
})
```

2. Map to dimensions:
```python
dimension_module_map: Dict[str, List[str]] = field(default_factory=lambda: {
    "D1": ["existing_modules", "your_module"],
    # ...
})
```

3. **NEVER** load cuestionario.json directly - use parser:
```python
from orchestrator.questionnaire_parser import get_questionnaire_parser

class YourModule:
    def __init__(self):
        self.parser = get_questionnaire_parser()
        self.dimensions = self.parser.get_all_dimensions()
```

**For Orchestration Changes:**

1. Update relevant orchestrator component
2. Ensure QuestionnaireParser integration
3. Maintain deterministic behavior
4. Add logging for traceability

### 4. Testing Your Changes

```bash
# Run existing tests
python3 test_architecture_compilation.py
python3 test_orchestrator_integration.py

# Test parser integration
python3 << 'EOF'
from orchestrator.core_orchestrator import FARFANOrchestrator

orchestrator = FARFANOrchestrator()
print(f"Parser version: {orchestrator.parser.version}")
print(f"Total questions: {orchestrator.parser.total_questions}")
print("✓ Integration test passed")
EOF
```

---

## Testing Requirements

### Minimum Test Coverage

All contributions must include:

1. **Unit Tests:** Test individual components
2. **Integration Tests:** Test module interactions
3. **Parser Tests:** Verify QuestionnaireParser usage
4. **Determinism Tests:** Verify reproducible results

### Example Test

```python
def test_question_router_uses_parser():
    """Verify QuestionRouter uses QuestionnaireParser"""
    from orchestrator.question_router import QuestionRouter
    
    router = QuestionRouter()
    
    # Should have parser instance
    assert hasattr(router, 'parser')
    assert router.parser is not None
    
    # Should load 300 questions
    assert len(router.questions) == 300
    
    # Questions should be deterministic
    q_ids_1 = list(router.questions.keys())
    
    router2 = QuestionRouter()
    q_ids_2 = list(router2.questions.keys())
    
    assert q_ids_1 == q_ids_2, "Question order must be deterministic"
```

---

## SIN_CARRETA Compliance

### Hard Rules (Non-Negotiable)

1. **No Silent Failures**
   - All errors must be explicit exceptions
   - No `try-except-pass` blocks
   - No error-to-warning conversions

2. **Deterministic Behavior**
   - No randomness without explicit seeding
   - No non-deterministic operations
   - Fixed iteration order (sorted)

3. **Contract Integrity**
   - Type hints required
   - Immutable dataclasses preferred
   - Explicit interfaces

4. **Single Source of Truth**
   - QuestionnaireParser for all questionnaire data
   - No duplicate data sources
   - Traceability to cuestionario.json

5. **Auditability**
   - Log significant operations
   - Version tracking
   - Change documentation

### Compliance Checklist

Before submitting PR:

- [ ] No direct JSON loading of cuestionario.json
- [ ] All errors raise exceptions (not warnings)
- [ ] Deterministic behavior verified
- [ ] Immutable dataclasses used
- [ ] Type hints added
- [ ] Logging added for operations
- [ ] Tests pass
- [ ] Documentation updated

---

## Pull Request Process

### 1. Pre-Submission

```bash
# Ensure code quality
black orchestrator/*.py  # Format code
mypy orchestrator/      # Type checking (if available)

# Run tests
python3 test_architecture_compilation.py
python3 test_orchestrator_integration.py

# Validate questionnaire if modified
python3 -m json.tool cuestionario.json > /dev/null
```

### 2. Commit Message Format

```
<type>: <short description>

<detailed description>

SIN_CARRETA Compliance:
- Clause X.Y: <how this change complies>
- Determinism: <how determinism is preserved>
- Traceability: <data flow documentation>

Tests:
- <test descriptions>
```

**Types:** `feat`, `fix`, `refactor`, `docs`, `test`, `chore`

### 3. PR Description Template

```markdown
## Summary
Brief description of changes

## Motivation
Why this change is needed

## Changes Made
- File 1: Description
- File 2: Description

## SIN_CARRETA Compliance
- [ ] No silent failures
- [ ] Deterministic behavior
- [ ] Uses QuestionnaireParser
- [ ] Immutable structures
- [ ] Explicit errors

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Parser integration verified
- [ ] Determinism verified

## Documentation
- [ ] CODE_FIX_REPORT.md updated (if applicable)
- [ ] README.md updated (if applicable)
- [ ] Code comments added
```

### 4. Review Process

PRs will be reviewed for:

1. **Correctness:** Code does what it claims
2. **Compliance:** Adheres to SIN_CARRETA doctrine
3. **Determinism:** Reproducible results
4. **Traceability:** Clear data flows
5. **Testing:** Adequate coverage
6. **Documentation:** Changes documented

---

## Common Pitfalls

### ❌ Anti-Pattern: Direct JSON Loading

```python
# NEVER DO THIS
with open('cuestionario.json') as f:
    questions = json.load(f)
```

### ✅ Correct Pattern: Use Parser

```python
from orchestrator.questionnaire_parser import get_questionnaire_parser

parser = get_questionnaire_parser()
questions = parser.get_questions_for_dimension("D1")
```

---

### ❌ Anti-Pattern: Silent Failure

```python
# NEVER DO THIS
try:
    result = process()
except Exception as e:
    logger.warning(f"Process failed: {e}")
    return None
```

### ✅ Correct Pattern: Explicit Error

```python
try:
    result = process()
except ProcessError as e:
    logger.error(f"Process failed: {e}")
    raise  # Propagate the error
```

---

### ❌ Anti-Pattern: Mutable State

```python
# AVOID THIS
class Question:
    def __init__(self):
        self.text = ""
        self.rubric = {}
    
    def set_text(self, text):
        self.text = text  # Mutation!
```

### ✅ Correct Pattern: Immutable

```python
@dataclass(frozen=True)
class Question:
    text: str
    rubric: Dict[str, float]
```

---

## Documentation Standards

### Code Comments

Add comments for:
- Complex logic
- SIN_CARRETA compliance notes
- Traceability references
- Non-obvious design decisions

```python
def generate_questions(self) -> List[str]:
    """
    Generate all 300 question IDs in deterministic order.
    
    SIN_CARRETA Compliance:
    - Clause 3.1: Deterministic generation via sorted() iteration
    - Clause 5.1: Traceability to cuestionario.json via parser
    
    Returns:
        List of question IDs in format "P#-D#-Q#"
        Order: P1-P10 (sorted), D1-D6 (sorted), Q1-Q5
    """
    # Implementation
```

### Docstrings

Required for all public methods:

```python
def get_dimension_weight_for_point(
    self, 
    dimension_code: str, 
    point_code: str
) -> float:
    """
    Get weight of a dimension for a specific policy point.
    
    Args:
        dimension_code: Dimension identifier (D1-D6)
        point_code: Policy point identifier (P1-P10)
        
    Returns:
        Weight value (0.0-1.0), or 0.0 if not found
        
    Raises:
        ValueError: If codes are invalid format
        
    Example:
        >>> parser.get_dimension_weight_for_point("D1", "P1")
        0.2
    """
```

---

## Contact and Questions

For questions about:
- **Architecture:** Review `README.md` and `CODE_FIX_REPORT.md`
- **Parser Usage:** See `orchestrator/questionnaire_parser.py` docstrings
- **SIN_CARRETA:** Consult internal compliance documentation
- **Bugs:** Open an issue with reproduction steps

---

## Traceability Notes

### Data Flow

```
cuestionario.json (canonical source)
    ↓
QuestionnaireParser (validation & parsing)
    ↓
QuestionRouter (question-to-module mapping)
    ↓
ExecutionChoreographer (module execution)
    ↓
ReportAssembler (result aggregation)
    ↓
Final Reports (MICRO/MESO/MACRO)
```

### Change Impact Analysis

When modifying questionnaire data:

1. **cuestionario.json** → QuestionnaireParser validates structure
2. **QuestionnaireParser** → QuestionRouter loads questions
3. **QuestionRouter** → Choreographer maps to modules
4. **Choreographer** → Modules execute analysis
5. **ReportAssembler** → Uses rubric levels from parser

**Impact:** Changes propagate deterministically through pipeline

---

## Versioning

### Questionnaire Version

Update `cuestionario.json` metadata:

```json
{
  "metadata": {
    "version": "2.1.0",  // Increment for structural changes
    "created_date": "2025-10-17",
    // ...
  }
}
```

### Code Version

Document in `CODE_FIX_REPORT.md`:

```markdown
## Version X.Y.Z (YYYY-MM-DD)
- Changes made
- SIN_CARRETA compliance notes
- Testing verification
```

---

## License

[License information to be added]

---

**Last Updated:** 2025-10-17  
**Maintained By:** FARFAN Development Team  
**Compliance Status:** ✅ SIN_CARRETA Validated
