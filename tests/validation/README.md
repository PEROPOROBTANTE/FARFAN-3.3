# FARFAN 3.0 Interface Contract Validation Test Suite

## Overview

This directory contains comprehensive interface contract validation tests for the FARFAN 3.0 system, validating critical integration points across the questionnaire parser, module adapters, execution mapping, and scoring rubrics.

## Files Created

### Test Implementation
- **`test_interface_contracts.py`** (630 lines)
  - 5 comprehensive test suites
  - 10 pytest fixtures
  - Validates 300-question questionnaire, 9 adapters, execution mapping, and scoring formulas

### Fixtures
- **`../fixtures/cuestionario_fixture.json`** (105 lines)
  - Representative sample of cuestionario.json structure
  - 2 dimensions (D1, D2), 2 policy areas (P1, P2), 3 sample questions
  
- **`../fixtures/execution_mapping_fixture.yaml`** (93 lines)
  - Sample execution mapping with adapter registry
  - 2 dimension mappings with execution chains
  
- **`../fixtures/mock_adapter_responses.json`** (256 lines)
  - Deterministic mock responses for 6 adapters
  - Includes scoring test cases for TYPE_A, TYPE_B, TYPE_C
  - Aggregation test cases for formula validation

### Documentation
- **`test_execution_instructions.md`** (340 lines)
  - Complete execution guide
  - Coverage targets and interpretation
  - Failure remediation procedures
  - CI/CD integration examples

## Test Suites

### 1. test_questionnaire_parser_alignment
**Purpose:** Validates all 300 questions parse correctly with complete field mapping

**Validates:**
- 300 total questions (6 dimensions × 10 policy areas × 5 questions)
- P#-D#-Q# canonical ID generation and uniqueness
- Required fields: question_id, dimension, policy_area, text, scoring_modality, max_score
- Dimension format (D1-D6) and policy area format (P1-P10)
- Question numbers (1-5) and max_score (3.0)
- Complete coverage: 50 questions per dimension, 30 per policy area

**Key Assertions:**
```python
assert len(all_questions) == 300
assert re.match(r'^P\d+-D\d+-Q\d+$', canonical_id)
assert question.max_score == 3.0
assert dimension_counts[dimension] == 50
```

### 2. test_adapter_method_signatures
**Purpose:** Validates orchestrator-to-adapter call compatibility using inspect.signature

**Validates:**
- All 9 adapter classes are inspectable
- Method signatures can be extracted via inspect.signature()
- BaseAdapter has required helper methods (_create_unavailable_result, _create_error_result)
- ModuleResult dataclass has all required fields
- Adapter methods referenced in execution_mapping.yaml exist (with warnings for missing)

**Adapters Validated:**
- PolicyProcessorAdapter
- PolicySegmenterAdapter
- ModulosAdapter
- AnalyzerOneAdapter
- DerekBeachAdapter
- EmbeddingPolicyAdapter
- SemanticChunkingPolicyAdapter
- ContradictionDetectionAdapter
- FinancialViabilityAdapter

### 3. test_staticmethod_invocations
**Purpose:** Scans module_adapters.py for incorrect instance-based calls to @staticmethod methods

**Validates:**
- Identifies all @staticmethod decorated methods via regex
- Detects instance calls (self.method()) that should be class calls
- Reports incorrect invocation patterns
- Counts proper class-based static method calls

**Note:** Current implementation found 0 @staticmethod decorators in module_adapters.py, so this test serves as future protection if static methods are added.

### 4. test_question_traceability
**Purpose:** Traces each question through execution_mapping.yaml to verify adapter methods exist

**Validates:**
- Questions in execution_mapping.yaml have valid execution chains
- Referenced adapter classes exist in adapter registry
- Referenced methods exist in adapter classes (with warnings for missing)
- Minimum 50% coverage threshold
- Traceability from question → execution chain → adapter → method

**Coverage Calculation:**
```python
coverage_pct = (questions_with_mapping / total_questions) * 100
assert coverage_pct >= 50  # Minimum coverage requirement
```

### 5. test_rubric_scoring_integration
**Purpose:** Validates TYPE_A-F scoring modality formulas and aggregation using deterministic fixtures

**Validates:**
- All 6 scoring modalities exist (TYPE_A through TYPE_F)
- Each has required fields: id, description, formula, max_score (3.0)
- TYPE_A: (elements_found / 4) * 3 with conversion table
- TYPE_B: min(elements_found, 3) with direct mapping
- TYPE_C: (elements_found / 2) * 3 with conversion table
- TYPE_D: ratio_quantitative with thresholds
- TYPE_E: logical_rule with custom logic
- TYPE_F: semantic_analysis with similarity threshold (0.6)
- 4-level aggregation:
  - Level 1: Question score (0-3 points)
  - Level 2: Dimension score (0-100%, formula: (sum/15)*100)
  - Level 3: Point score (0-100%, average of 6 dimensions)
  - Level 4: Global score (0-100%, average excluding N/A)
- 5 score bands: EXCELENTE (85-100), BUENO (70-84), SATISFACTORIO (55-69), INSUFICIENTE (40-54), DEFICIENTE (0-39)
- Deterministic test cases produce expected scores

## Quick Start

```bash
# Run all tests
pytest tests/validation/test_interface_contracts.py -v

# Run with coverage
pytest tests/validation/test_interface_contracts.py -v --cov=orchestrator --cov-report=html

# Run individual test
pytest tests/validation/test_interface_contracts.py::test_questionnaire_parser_alignment -v
```

## Expected Results

All 5 tests should pass with output similar to:

```
tests/validation/test_interface_contracts.py::test_questionnaire_parser_alignment PASSED
✓ Successfully validated 300 questions with complete field mapping

tests/validation/test_interface_contracts.py::test_adapter_method_signatures PASSED
✓ Validated method signatures for 9 adapter classes

tests/validation/test_interface_contracts.py::test_staticmethod_invocations PASSED
✓ Found 0 @staticmethod decorated methods

tests/validation/test_interface_contracts.py::test_question_traceability PASSED
✓ Traceability coverage: X/Y questions (Z%)

tests/validation/test_interface_contracts.py::test_rubric_scoring_integration PASSED
✓ Validated all 6 scoring modalities (TYPE_A through TYPE_F)
```

## Coverage Targets

| Module                           | Target Coverage | Purpose                    |
|----------------------------------|-----------------|----------------------------|
| orchestrator/questionnaire_parser.py | ≥70%            | Question parsing validation |
| orchestrator/module_adapters.py  | ≥60%            | Adapter interface validation |
| orchestrator/report_assembly.py  | ≥65%            | Scoring formula validation |

## Integration with Existing Tests

This validation suite complements existing test files:
- `test_architecture_compilation.py` - Architecture integrity
- `test_orchestrator_integration.py` - Integration smoke tests
- `test_choreographer_integration.py` - Choreographer unit tests
- `test_circuit_breaker_*.py` - Circuit breaker tests
- `test_report_assembler_scoring.py` - Report assembly tests

## Maintenance

When updating FARFAN 3.0 contracts:

1. **Adding questions**: Update expected count in test_questionnaire_parser_alignment
2. **Adding adapters**: Add new adapter class to all_adapter_classes fixture
3. **Changing scoring**: Update test_rubric_scoring_integration with new formulas
4. **Modifying execution mapping**: Update test_question_traceability expectations
5. **Adding @staticmethod**: test_staticmethod_invocations will automatically detect

## Troubleshooting

See `test_execution_instructions.md` for detailed troubleshooting, including:
- Import errors and PYTHONPATH configuration
- File not found errors
- Encoding issues
- Interpreting specific failure messages
- Remediation procedures

## Contact

For issues or questions about interface contract validation:
- Review test output for descriptive error messages
- Check fixture files are present and valid JSON/YAML
- Ensure source files (cuestionario.json, execution_mapping.yaml, rubric_scoring.json) exist
- Run individual tests to isolate failures
