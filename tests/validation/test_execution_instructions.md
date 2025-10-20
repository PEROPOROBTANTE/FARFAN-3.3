# FARFAN 3.0 Interface Contract Validation - Test Execution Instructions

## Overview

This document provides comprehensive instructions for executing the five interface contract validation test suites in `tests/validation/test_interface_contracts.py`.

## Prerequisites

Ensure all dependencies are installed:

```bash
source venv/bin/activate  # Or: venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Test Suites Overview

### 1. test_questionnaire_parser_alignment
Validates that all 300 questions from `cuestionario.json` parse correctly with complete field mapping and canonical ID generation.

### 2. test_adapter_method_signatures
Uses Python's `inspect.signature` to validate orchestrator-to-adapter call compatibility across all 9 adapters.

### 3. test_staticmethod_invocations
Scans `module_adapters.py` source code for incorrect instance-based calls to `@staticmethod` decorated methods.

### 4. test_question_traceability
Traces each question through `execution_mapping.yaml` to verify that referenced adapter methods exist in source modules.

### 5. test_rubric_scoring_integration
Validates TYPE_A through TYPE_F scoring modality formulas and aggregation rules using deterministic fixtures.

## Execution Commands

### Run All Interface Contract Tests

```bash
pytest tests/validation/test_interface_contracts.py -v
```

**Expected Output:**
- 5 test suites should pass
- Execution time: ~10-30 seconds depending on system

### Run Individual Test Suites

```bash
# Test 1: Questionnaire Parser Alignment (300 questions)
pytest tests/validation/test_interface_contracts.py::test_questionnaire_parser_alignment -v

# Test 2: Adapter Method Signatures (9 adapters)
pytest tests/validation/test_interface_contracts.py::test_adapter_method_signatures -v

# Test 3: Static Method Invocations
pytest tests/validation/test_interface_contracts.py::test_staticmethod_invocations -v

# Test 4: Question Traceability
pytest tests/validation/test_interface_contracts.py::test_question_traceability -v

# Test 5: Rubric Scoring Integration
pytest tests/validation/test_interface_contracts.py::test_rubric_scoring_integration -v
```

### Run with Coverage Analysis

```bash
pytest tests/validation/test_interface_contracts.py -v --cov=orchestrator --cov-report=html --cov-report=term
```

**Coverage Targets:**
- `orchestrator/questionnaire_parser.py`: ≥70% coverage
- `orchestrator/module_adapters.py`: ≥60% coverage (partial coverage expected due to module availability)
- `orchestrator/report_assembly.py`: ≥65% coverage

**View HTML Coverage Report:**
```bash
open htmlcov/index.html  # macOS/Linux
start htmlcov/index.html  # Windows
```

### Run with Detailed Output

```bash
pytest tests/validation/test_interface_contracts.py -v -s
```

The `-s` flag shows all print statements, providing detailed validation messages.

### Run with Performance Profiling

```bash
pytest tests/validation/test_interface_contracts.py -v --durations=10
```

Shows the 10 slowest test execution times.

## Interpreting Test Results

### SUCCESS Indicators

#### test_questionnaire_parser_alignment ✓
```
✓ Successfully validated 300 questions with complete field mapping
```

**What this means:**
- All 300 questions parsed successfully
- All dimensions (D1-D6) represented with 50 questions each
- All policy areas (P1-P10) represented with 30 questions each
- All canonical IDs (P#-D#-Q# format) are unique and valid
- All required fields present: question_id, dimension, policy_area, text, scoring_modality, max_score

#### test_adapter_method_signatures ✓
```
✓ Validated method signatures for 9 adapter classes
✓ Found N unique adapter references in execution mapping
```

**What this means:**
- All 9 adapter classes found and inspected
- BaseAdapter has required helper methods
- ModuleResult dataclass has all required fields
- Adapter methods referenced in execution_mapping.yaml exist (or warnings issued)

#### test_staticmethod_invocations ✓
```
✓ Found N @staticmethod decorated methods
✓ Found N proper class-based static method calls
✓ No incorrect instance-based calls to @staticmethod methods detected
```

**What this means:**
- All `@staticmethod` decorators identified
- No incorrect `self.method()` calls to static methods
- All static methods called correctly as `ClassName.method()`

#### test_question_traceability ✓
```
✓ Traceability coverage: X/Y questions (Z%)
```

**What this means:**
- Minimum 50% of questions have execution mappings
- Referenced adapter classes exist
- Referenced methods exist in adapter classes (or warnings issued)

#### test_rubric_scoring_integration ✓
```
✓ Validated all 6 scoring modalities (TYPE_A through TYPE_F)
✓ Validated 4-level aggregation formulas
✓ Validated 5 score bands with correct thresholds
✓ Tested deterministic scoring with N fixtures
```

**What this means:**
- All scoring types (TYPE_A-F) have correct formulas
- Aggregation levels compute correctly (0-3 → 0-100%)
- Score bands (EXCELENTE, BUENO, etc.) have correct thresholds
- Deterministic test cases produce expected scores

### FAILURE Indicators and Remediation

#### Failure: "Expected 300 questions, got X"

**Cause:** `cuestionario.json` structure changed or incomplete.

**Remediation:**
1. Check `cuestionario.json` metadata field: `"total_questions": 300`
2. Verify dimensions D1-D6 all exist
3. Verify policy areas P1-P10 all exist
4. Check preguntas_base array has correct structure

#### Failure: "Invalid canonical ID format"

**Cause:** Question ID doesn't match P#-D#-Q# pattern.

**Remediation:**
1. Check `QuestionSpec.canonical_id` property implementation
2. Verify `policy_area`, `dimension`, `question_no` fields are correctly set
3. Expected format: `P1-D1-Q1`, `P10-D6-Q5`, etc.

#### Failure: "Missing scoring modality: TYPE_X"

**Cause:** `rubric_scoring.json` missing expected scoring type.

**Remediation:**
1. Open `rubric_scoring.json`
2. Verify `scoring_modalities` section has all 6 types: TYPE_A, TYPE_B, TYPE_C, TYPE_D, TYPE_E, TYPE_F
3. Each must have: `id`, `description`, `formula`, `max_score` (3.0)

#### Failure: "Adapter class X not found"

**Cause:** Adapter class referenced in `execution_mapping.yaml` doesn't exist in `module_adapters.py`.

**Remediation:**
1. Check `execution_mapping.yaml` adapter references
2. Verify corresponding class exists in `module_adapters.py`
3. Expected classes: PolicyProcessorAdapter, PolicySegmenterAdapter, ModulosAdapter, AnalyzerOneAdapter, DerekBeachAdapter, EmbeddingPolicyAdapter, SemanticChunkingPolicyAdapter, ContradictionDetectionAdapter, FinancialViabilityAdapter

#### Failure: "Method X not found in adapter Y"

**Cause:** Method referenced in execution chain doesn't exist.

**Remediation:**
1. Check `execution_mapping.yaml` execution_chain `method` fields
2. Verify method exists in corresponding adapter class
3. Check for typos in method names
4. This may be a warning if method is dynamically added

#### Warning: "⚠ Found N incorrect instance calls to @staticmethod methods"

**Cause:** Code calls static methods as `self.method()` instead of `ClassName.method()`.

**Remediation:**
1. Review `module_adapters.py` for patterns like `self.{method_name}()`
2. If method has `@staticmethod` decorator, change to `ClassName.{method_name}()`
3. Example: Change `self.parse_data()` to `PolicyProcessorAdapter.parse_data()`

#### Failure: "Aggregation formula failed"

**Cause:** Scoring aggregation doesn't match expected formula.

**Remediation:**
1. Check formula: `(sum_of_5_questions / 15) * 100` for dimension scores
2. Verify max_score per question is 3.0
3. Verify 5 questions per dimension
4. Check `report_assembly.py` aggregation logic

#### Failure: "Insufficient execution mapping coverage: X%"

**Cause:** Less than 50% of questions have execution mappings.

**Remediation:**
1. This indicates incomplete `execution_mapping.yaml`
2. Add execution_chain entries for more questions
3. Target: ≥50% coverage (150+ questions mapped)

## Advanced Usage

### Run Tests with Markers

```bash
# Run only fast tests (if markers configured)
pytest tests/validation/test_interface_contracts.py -v -m "not slow"
```

### Generate JUnit XML Report

```bash
pytest tests/validation/test_interface_contracts.py -v --junitxml=test-results.xml
```

### Run with Parallel Execution

```bash
pip install pytest-xdist
pytest tests/validation/test_interface_contracts.py -v -n auto
```

### Integration with CI/CD

Add to `.github/workflows/tests.yml` or similar:

```yaml
- name: Run Interface Contract Validation
  run: |
    pytest tests/validation/test_interface_contracts.py -v --junitxml=results.xml --cov=orchestrator
    
- name: Upload Coverage
  uses: codecov/codecov-action@v3
  with:
    files: ./coverage.xml
```

## Troubleshooting

### Import Errors

**Error:** `ModuleNotFoundError: No module named 'orchestrator'`

**Solution:**
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"  # Linux/macOS
set PYTHONPATH=%PYTHONPATH%;%CD%  # Windows CMD
```

### File Not Found Errors

**Error:** `FileNotFoundError: cuestionario.json`

**Solution:**
Ensure you're running from project root:
```bash
cd /path/to/farfan-project-root
pytest tests/validation/test_interface_contracts.py -v
```

### Encoding Issues

**Error:** `UnicodeDecodeError`

**Solution:**
Files must be UTF-8 encoded. Convert with:
```bash
iconv -f ISO-8859-1 -t UTF-8 file.json > file_utf8.json
```

## Expected Test Execution Time

| Test Suite                          | Expected Duration | Notes                      |
|-------------------------------------|-------------------|----------------------------|
| test_questionnaire_parser_alignment | 2-5 seconds       | Parses 300 questions       |
| test_adapter_method_signatures      | 1-2 seconds       | Inspects 9 adapters        |
| test_staticmethod_invocations       | 1-2 seconds       | Regex scan of source file  |
| test_question_traceability          | 2-4 seconds       | Maps questions to adapters |
| test_rubric_scoring_integration     | 1-3 seconds       | Validates formulas         |
| **TOTAL**                           | **7-16 seconds**  | On modern hardware         |

## Success Criteria

All tests pass when:
1. ✓ 300 questions parse with valid canonical IDs
2. ✓ 9 adapters have inspectable method signatures
3. ✓ No incorrect static method invocations (or only warnings)
4. ✓ ≥50% question traceability coverage
5. ✓ All 6 scoring modalities validate correctly with deterministic fixtures

## Contact and Support

For issues or questions:
- Review test output carefully - error messages are descriptive
- Check fixture files in `tests/fixtures/` are present
- Ensure all source files (`cuestionario.json`, `execution_mapping.yaml`, `rubric_scoring.json`) are present and valid
- Run individual tests to isolate failures

**Next Steps After Success:**
1. Run full test suite: `pytest tests/ -v`
2. Review coverage report to identify untested code paths
3. Add additional test cases for edge cases
4. Integrate into CI/CD pipeline
