# Choreographer Metadata Enrichment Extension

## Overview

Extended the `ExecutionChoreographer` class to implement a comprehensive metadata-driven orchestration system that:
- Extracts and injects `cuestionario.json` metadata into module invocations
- Validates module outputs against question-specific requirements
- Implements retry logic with circuit breaker integration on validation failures
- Tracks dependency satisfaction across question chains

## Architecture

### 1. **QuestionContext Dataclass**
```python
@dataclass
class QuestionContext:
    question_id: str
    question_text: str
    constraints: Dict[str, Any]          # Operational constraints
    expected_format: Dict[str, Any]      # Expected output format/schema
    validation_rules: Dict[str, Any]     # Validation criteria
    dependencies: List[str]              # Question dependencies
    error_strategy: str                  # Error handling strategy
    dimension: str                       # Dimension (D1-D6)
    punto_decalogo: str                  # Decalogo point (P1-P10)
    peso: float                          # Question weight
    is_critical: bool                    # Critical flag
    minimum_score: float                 # Minimum score threshold
```

### 2. **ValidationResult Dataclass**
```python
@dataclass
class ValidationResult:
    is_valid: bool
    violations: List[str]
    confidence_score: float
    validation_metadata: Dict[str, Any]
```

### 3. **Enhanced ExecutionResult**
Extended with:
- `question_context: Optional[QuestionContext]` - Attached question metadata
- `validation_result: Optional[ValidationResult]` - Validation outcome
- `retry_count: int` - Number of retry attempts

## Key Features

### Metadata Extraction and Enrichment

#### `_load_cuestionario_metadata()`
Loads `cuestionario.json` at initialization with error handling for missing files.

#### `extract_question_context(question_id: str) -> Optional[QuestionContext]`
Extracts and enriches metadata for a specific question:
- Parses question_id format: `D<dim>_P<punto>_Q<num>`
- Retrieves dimension metadata from `dimensiones` section
- Retrieves punto metadata from `puntos_decalogo` section
- Extracts specific question data from `preguntas` array
- Builds comprehensive `QuestionContext` with:
  - Constraints (sources, scope, indicators)
  - Expected format (tipo_respuesta, escala, formato_esperado)
  - Validation rules (thresholds, required evidence, patterns, ranges)
  - Dependencies for orchestration ordering
- Implements caching for performance optimization

### Context-Aware Module Invocation

#### Enhanced `execute_question_chain()`
1. Extracts `QuestionContext` from cuestionario.json
2. Validates dependencies are satisfied before execution
3. Injects `question_context` into module invocation kwargs
4. Executes with validation and retry logic
5. Tracks validated results for downstream dependency checks

### Post-Processing Validation Engine

#### `_validate_module_response(result, question_context) -> ValidationResult`
Comprehensive validation checks:
1. **Confidence Threshold**: `result.confidence >= minimum_confidence`
2. **Required Evidence Types**: All required evidence types present
3. **Expected Format**: Output structure matches expected format (cuantitativa/cualitativa)
4. **Range Constraints**: Numeric values within specified ranges
5. **Pattern Matching**: Text patterns match validation patterns (regex)

Returns `ValidationResult` with:
- Boolean validity flag
- List of violations
- Validation confidence score
- Detailed validation metadata

### Retry Logic with Circuit Breaker Integration

#### `_execute_step_with_validation()` with Retry
1. Executes module method via `_execute_single_step()`
2. Validates response against question metadata
3. On validation failure:
   - Logs violations with structured logging
   - Records failure in circuit breaker
   - Retries execution if `retry_count < max_retries`
   - Applies `error_strategy` from question context
4. Returns `ExecutionResult` with validation metadata

#### `_log_validation_violations()`
Structured logging format:
```
VALIDATION_VIOLATION | adapter=<name> | method=<name> | question=<id> | violations=<count>
  [1] <violation_detail>
  [2] <violation_detail>
  ...
```

### Dependency Management

#### `_check_dependencies_satisfied(question_context) -> bool`
Verifies all dependencies are satisfied before execution:
- Checks if dependency question_id exists in `dependency_results`
- Validates dependency execution status is `COMPLETED`
- Ensures dependency validation passed
- Returns `False` if any dependency unsatisfied or failed validation

### Validation Statistics

#### `get_validation_statistics(results) -> Dict[str, Any]`
Calculates comprehensive validation metrics:
- `total_steps`: Total execution steps
- `validated_steps`: Steps with validation performed
- `valid_steps`: Steps passing validation
- `failed_validation_steps`: Steps failing validation
- `retried_steps`: Steps requiring retry
- `avg_execution_confidence`: Average execution confidence
- `avg_validation_confidence`: Average validation confidence
- `total_violations`: Total validation violations
- `validation_rate`: Percentage of valid steps
- `unique_violations`: List of unique violation messages

## Integration with Existing Components

### ModuleController
Modules receive `question_context` in kwargs:
```python
def some_module_method(self, text: str, question_context: Dict = None):
    if question_context:
        # Use constraints, expected_format, validation_rules
        constraints = question_context['constraints']
        # Adapt processing based on question requirements
```

### Circuit Breaker
Extended error recording with validation failure severity:
```python
circuit_breaker.record_failure(
    adapter_name=adapter_name,
    error=f"Validation failed: {violations}",
    severity="DEGRADED"  # Validation failures are DEGRADED, not CRITICAL
)
```

### Execution Status
Added new statuses:
- `VALIDATION_FAILED`: Max retries exceeded on validation failure
- `RETRYING`: Currently retrying after validation failure

## Configuration

### Choreographer Initialization
```python
choreographer = ExecutionChoreographer(
    max_workers=4,                      # Parallel workers
    cuestionario_path="cuestionario.json",  # Metadata path
    max_retries=3                       # Max retry attempts
)
```

### Question Metadata Schema (cuestionario.json)
```json
{
  "preguntas": [
    {
      "id": "D1_P1_Q001",
      "texto": "Question text",
      "tipo_respuesta": "cualitativa",
      "formato_esperado": "texto_estructurado",
      "umbral_confianza": 0.6,
      "tipos_evidencia_requeridos": ["pattern_match", "semantic_similarity"],
      "patrones_validacion": ["regex_pattern"],
      "restricciones_rango": {"score": {"min": 0, "max": 100}},
      "fuentes_verificacion": ["plan_desarrollo"],
      "alcance": "municipal",
      "dependencias": ["D1_P1_Q002"],
      "estrategia_error": "retry_specific"
    }
  ]
}
```

## Testing

Comprehensive test suite in `test_choreographer_metadata.py`:

### Test 1: QuestionContext Extraction
- ✓ Extracts context from cuestionario.json
- ✓ Parses question_id format correctly
- ✓ Retrieves dimension and punto metadata
- ✓ Builds complete QuestionContext
- ✓ Implements caching

### Test 2: Validation Engine
- ✓ Validates correct responses
- ✓ Detects low confidence
- ✓ Detects missing evidence types
- ✓ Detects out-of-range values
- ✓ Validates against patterns

### Test 3: Dependency Checking
- ✓ Detects unsatisfied dependencies
- ✓ Tracks partial satisfaction
- ✓ Confirms full satisfaction
- ✓ Detects failed validation in dependencies

### Test 4: Validation Statistics
- ✓ Calculates validation metrics
- ✓ Tracks retry counts
- ✓ Computes validation rates
- ✓ Aggregates unique violations

**Test Results: 4/4 passed (100%)**

## Benefits

1. **Verifiable Workflow**: All module outputs validated against cuestionario requirements
2. **Adaptive Processing**: Modules can adapt behavior based on question metadata
3. **Resilient Execution**: Automatic retry on validation failures with circuit breaker protection
4. **Observability**: Structured logging of violations for debugging and monitoring
5. **Dependency Satisfaction**: Ensures data dependencies met before execution
6. **Quality Assurance**: Enforces minimum confidence and evidence requirements
7. **Traceability**: Full audit trail of validation results and retry attempts

## Usage Example

```python
from orchestrator.choreographer import ExecutionChoreographer
from orchestrator.circuit_breaker import CircuitBreaker
from orchestrator.module_adapters import ModuleAdapterRegistry

# Initialize components
choreographer = ExecutionChoreographer(
    cuestionario_path="cuestionario.json",
    max_retries=3
)
circuit_breaker = CircuitBreaker()
module_registry = ModuleAdapterRegistry()

# Execute question chain with metadata
results = choreographer.execute_question_chain(
    question_spec=question_spec,
    plan_text=plan_text,
    module_adapter_registry=module_registry,
    circuit_breaker=circuit_breaker
)

# Get validation statistics
stats = choreographer.get_validation_statistics(results)
print(f"Validation rate: {stats['validation_rate']:.1%}")
print(f"Total violations: {stats['total_violations']}")
```

## Files Modified

1. **orchestrator/choreographer.py**: Extended with metadata enrichment and validation
2. **test_choreographer_metadata.py**: Comprehensive test suite
3. **CHOREOGRAPHER_METADATA_EXTENSION.md**: This documentation

## Version

- **Version**: 3.0.0
- **Author**: FARFAN Integration Team
- **Python**: 3.10+
- **Status**: ✅ Tested and Validated
