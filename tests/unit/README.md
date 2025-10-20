# Module Adapter Unit Test Suite

Comprehensive unit tests for all 9 FARFAN 3.0 module adapters validating standardized interfaces, error handling, and dependency management.

## Overview

The test suite validates:
- **ModuleResult structure**: All 11 required fields with proper types
- **Execute method signatures**: Standardized `execute(method_name, args, kwargs)` interface
- **Availability checks**: Detection of missing dependencies
- **Error handling**: Consistent exception-to-ModuleResult conversion
- **Configuration propagation**: Parameter passing to wrapped modules
- **Execution time tracking**: Performance metrics in results

## Test Files

### `test_adapters.py`
Comprehensive pytest-based test suite with 19 test classes covering all 9 adapters.

**Requirements**: pytest, numpy, and full FARFAN dependencies

**Run with**:
```bash
python -m pytest tests/unit/test_adapters.py -v
```

### `test_adapters_simple.py`
Standalone test runner with no pytest dependency for basic validation.

**Requirements**: Standard library only (unittest.mock)

**Run with**:
```bash
python tests/unit/test_adapters_simple.py
```

## Adapter Dependency Graph

| Adapter | Module Name | Source Module |
|---------|-------------|---------------|
| PolicyProcessorAdapter | policy_processor | policy_processor |
| PolicySegmenterAdapter | policy_segmenter | policy_segmenter |
| AnalyzerOneAdapter | analyzer_one | Analyzer_one |
| EmbeddingPolicyAdapter | embedding_policy | emebedding_policy |
| SemanticChunkingPolicyAdapter | semantic_chunking_policy | semantic_chunking_policy |
| FinancialViabilityAdapter | financial_viability | financiero_viabilidad_tablas |
| DerekBeachAdapter | dereck_beach | dereck_beach |
| ContradictionDetectionAdapter | contradiction_detection | contradiction_deteccion |
| ModulosAdapter | teoria_cambio | teoria_cambio |

## Test Coverage

### ModuleResult Structure Tests
- ✓ All required fields present (module_name, class_name, method_name, status, data, evidence, confidence, execution_time)
- ✓ Optional fields present (errors, warnings, metadata)
- ✓ Success state representation
- ✓ Failure state representation
- ✓ Field type validation

### BaseAdapter Tests
- ✓ Initialization with module_name
- ✓ `_create_unavailable_result()` returns proper failed result
- ✓ `_create_error_result()` converts exceptions to ModuleResult
- ✓ All adapters inherit from BaseAdapter

### Availability Tests (Parametrized across all 9 adapters)
- ✓ Adapters detect missing dependencies
- ✓ Unavailable adapters return failed ModuleResult
- ✓ Available flag is boolean type
- ✓ Module name attribute present and correct

### Execute Method Tests (Parametrized across all 9 adapters)
- ✓ All adapters have execute() method
- ✓ Execute accepts (method_name, args, kwargs) signature
- ✓ Execute returns ModuleResult object
- ✓ Execution time tracking (>= 0.0)

### Error Handling Tests (Parametrized across all 9 adapters)
- ✓ Unknown method names return failed results
- ✓ Invalid arguments handled gracefully  
- ✓ Exceptions converted to error results (ValueError, TypeError, KeyError, AttributeError)
- ✓ Errors list populated in failed results

### Configuration Tests
- ✓ PolicyProcessorAdapter propagates config
- ✓ AnalyzerOneAdapter handles configuration parameters

### Registry Tests
- ✓ ModuleAdapterRegistry initializes all 9 adapters
- ✓ execute_module_method() routes to correct adapter
- ✓ Unknown modules return error results
- ✓ get_available_modules() returns list
- ✓ get_module_status() returns dict

### Data Validation Tests
- ✓ Adapters handle None returns
- ✓ Adapters handle empty data structures
- ✓ ModuleResult structure validated

## Key Classes

### ModuleResult
```python
@dataclass
class ModuleResult:
    module_name: str
    class_name: str
    method_name: str
    status: str  # "success" or "failed"
    data: Dict[str, Any]
    evidence: List[Dict[str, Any]]
    confidence: float
    execution_time: float
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### BaseAdapter
```python
class BaseAdapter:
    def __init__(self, module_name: str)
    def execute(self, method_name: str, args: List, kwargs: Dict) -> ModuleResult
    def _create_unavailable_result(self, method_name: str, start_time: float) -> ModuleResult
    def _create_error_result(self, method_name: str, start_time: float, error: Exception) -> ModuleResult
```

## Test Fixtures

### `adapter_dependency_graph`
Maps each adapter to its source modules and key classes for availability validation.

### `mock_module_result`
Sample ModuleResult for structure validation.

### `all_adapter_classes`
List of all 9 adapter classes for parametrized tests.

### Module Mocks
- `mock_policy_processor`: Mock policy_processor module
- `mock_analyzer_one`: Mock Analyzer_one module

## Usage Examples

### Run all tests
```bash
python -m pytest tests/unit/test_adapters.py -v
```

### Run specific test class
```bash
python -m pytest tests/unit/test_adapters.py::TestModuleResultStructure -v
```

### Run parametrized tests for single adapter
```bash
python -m pytest tests/unit/test_adapters.py -k "PolicyProcessorAdapter" -v
```

### Run with coverage
```bash
python -m pytest tests/unit/test_adapters.py --cov=orchestrator.module_adapters --cov-report=html
```

### Run simple tests (no pytest needed)
```bash
python tests/unit/test_adapters_simple.py
```

## Extending Tests

To add tests for new adapters:

1. Add adapter to `adapter_dependency_graph` fixture:
```python
"NewAdapter": {
    "adapter_class": NewAdapter,
    "module_name": "new_module",
    "source_modules": ["new_module"],
    "key_classes": ["MainClass", "HelperClass"]
}
```

2. Parametrized tests will automatically include the new adapter

3. Add adapter-specific tests if needed:
```python
def test_new_adapter_specific_behavior(self):
    adapter = NewAdapter()
    result = adapter.execute("special_method", [], {})
    assert result.data["special_field"] == expected_value
```

## CI/CD Integration

Add to CI pipeline:
```yaml
- name: Run adapter unit tests
  run: |
    python -m pytest tests/unit/test_adapters.py -v --junitxml=test-results.xml
```

## Notes

- Tests use `unittest.mock` for dependency isolation
- Parametrized tests run across all 9 adapters automatically
- Some tests may skip if adapter dependencies not installed
- Execution time tests verify timing >= 0.0 (may be very small)
- Registry tests verify attempted registration (success depends on dependencies)

## Known Issues

1. **orchestrator/module_adapters.py syntax errors**: File has indentation errors from incomplete method implementations around line 4600. Fix by adding proper return statements to all `_execute_*` methods.

2. **Circular import**: orchestrator/__init__.py imports question_router which requires numpy. Tests work around this by importing module_adapters directly.

3. **Dependency availability**: Full test suite requires all 9 module dependencies. Use `test_adapters_simple.py` for basic validation without dependencies.
