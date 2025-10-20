# Contract Specification System

This directory contains a YAML-based contract specification system for all 313 adapter methods in the FARFAN 3.0 architecture.

## Overview

The contract system provides:

1. **JSON Schema Validation** - Type-safe inputs and outputs
2. **Deterministic Hash Verification** - SHA-256 hashes for reproducibility
3. **RNG Seed Propagation** - Controlled randomness
4. **Binding Compatibility** - Producer/consumer type matching
5. **Fail-Fast Diagnostics** - Detailed violation reporting

## Structure

### Contract Files

Each adapter method has a YAML file: `{Adapter}_{method}.yaml`

Example: `PolicyProcessorAdapter_validate.yaml`

```yaml
adapter: PolicyProcessorAdapter
method: validate
input_schema:
  type: object
  properties:
    config:
      type: object
  required:
  - config
output_schema:
  type: boolean
deterministic: true
rng_seed_param: null
canonical_canary:
  config: {}
sample_hash: a406ca974c3a0e8d688a8927c4a4b63822baa73051db1703250775621b71a891
allowed_side_effects: []
max_latency_ms: 5000
retry_policy:
  max_retries: 0
  backoff_multiplier: 1.5
  initial_delay_ms: 100
  max_delay_ms: 5000
```

### Field Descriptions

- **adapter**: Adapter class name
- **method**: Method name (without `_execute_` prefix)
- **input_schema**: JSON Schema for input parameters
- **output_schema**: JSON Schema for output type
- **deterministic**: Boolean flag for reproducibility
- **rng_seed_param**: Name of RNG seed parameter (if applicable)
- **canonical_canary**: Sample input for hash verification
- **sample_hash**: SHA-256 hash of canonical output
- **allowed_side_effects**: List of allowed side effects
- **max_latency_ms**: Maximum execution time
- **retry_policy**: Retry configuration

## Coverage

### Adapters (9 total)

1. **PolicyProcessorAdapter** - 29 methods
2. **PolicySegmenterAdapter** - 30 methods
3. **AnalyzerOneAdapter** - 34 methods
4. **EmbeddingPolicyAdapter** - 33 methods
5. **SemanticChunkingPolicyAdapter** - 15 methods
6. **FinancialViabilityAdapter** - 20 methods
7. **DerekBeachAdapter** - 75 methods
8. **ContradictionDetectionAdapter** - 48 methods
9. **ModulosAdapter** - 29 methods

**Total: 313 methods**

## Usage

### Basic Validation

```python
from pathlib import Path
from tests.contracts.contract_validator import ContractValidator

# Initialize validator (loads all contracts)
validator = ContractValidator(Path('tests/contracts'))

# Validate input
input_data = {'config': {'min_confidence': 0.5}}
result = validator.validate_input('PolicyProcessorAdapter', 'validate', input_data)

if result.passed:
    print("✓ Input valid")
else:
    for violation in result.violations:
        print(f"✗ {violation.violation_type}: {violation.description}")
```

### Full Contract Validation

```python
# Validate all 313 contracts
result = validator.validate_all()

if result.passed:
    print("✓ All contracts valid")
else:
    validator.print_report(result)
```

### Integration with Adapters

```python
from tests.contracts.integration_example import ContractEnforcedAdapter

adapter = ContractEnforcedAdapter('PolicyProcessorAdapter', validator)

result = adapter.execute_with_validation(
    'validate',
    {'config': {'min_confidence': 0.5}}
)

if result['success']:
    output = result['output']
else:
    violations = result['violations']
```

## Validation Features

### 1. Schema Validation

- **Input Validation**: Verifies all required parameters are present with correct types
- **Output Validation**: Ensures return values match declared types
- **Type Safety**: Prevents type mismatches at method boundaries

### 2. Deterministic Hash Verification

For deterministic methods:
- Computes SHA-256 hash of output
- Compares with `sample_hash` in contract
- Detects non-deterministic behavior

### 3. RNG Seed Propagation

For non-deterministic methods:
- Verifies `rng_seed_param` is in input schema
- Ensures reproducibility when seed is provided
- Validates seed parameter type

### 4. Binding Compatibility

- Checks producer output types match consumer input types
- Identifies orphaned types (consumers without producers)
- Builds type compatibility graph

### 5. Fail-Fast Diagnostics

When validation fails, provides:
- Contract file path
- Adapter and method names
- Violation type
- Detailed description
- Relevant context (schema paths, failed values, etc.)

## Compliance Indicators

### Formal Compliance

✓ **Schema Validity** - All 313 schemas are valid JSON Schema Draft 7  
✓ **Contract Structure** - All required fields present  
✓ **Type System** - Consistent type definitions across contracts

### Material Compliance

✓ **Type Coherence** - Type system is internally consistent  
✓ **Determinism Classification** - Accurate deterministic/non-deterministic categorization  
✓ **Semantic Coherence** - Similar methods have compatible contracts  
✓ **Binding Compatibility** - Producer/consumer types match

## Running Tests

```bash
# Run all contract validation tests
pytest tests/test_contract_validator.py -v

# Run specific test classes
pytest tests/test_contract_validator.py::TestSchemaValidation -v
pytest tests/test_contract_validator.py::TestFormalCompliance -v
pytest tests/test_contract_validator.py::TestMaterialCompliance -v

# Run contract validator directly
python tests/contracts/contract_validator.py
```

## Test Results

```
28 tests passed:
- 4 tests: Contract Registry
- 5 tests: Schema Validation
- 2 tests: Deterministic Methods
- 2 tests: RNG Seed Propagation
- 2 tests: Binding Compatibility
- 2 tests: Latency Constraints
- 2 tests: Retry Policy
- 1 test: Side Effects
- 3 tests: Full Validation
- 2 tests: Formal Compliance
- 3 tests: Material Compliance
```

## Generation

Contracts are generated automatically:

```bash
python tests/contracts/contract_generator.py
```

This extracts all adapter methods from `orchestrator/module_adapters.py` and generates contract files.

## Examples

### Example 1: PolicyProcessorAdapter.validate

```yaml
adapter: PolicyProcessorAdapter
method: validate
deterministic: true
input_schema:
  type: object
  properties:
    config: {type: object}
  required: [config]
output_schema:
  type: boolean
```

**Usage:**
```python
result = validator.validate_input(
    'PolicyProcessorAdapter',
    'validate',
    {'config': {'min_confidence': 0.5}}
)
```

### Example 2: EmbeddingPolicyAdapter.semantic_search

```yaml
adapter: EmbeddingPolicyAdapter
method: semantic_search
deterministic: false
input_schema:
  type: object
  properties:
    query: {type: string}
    top_k: {type: integer}
    filters: {type: array}
  required: [query, top_k, filters]
output_schema:
  type: object
```

**Usage:**
```python
result = validator.validate_input(
    'EmbeddingPolicyAdapter',
    'semantic_search',
    {'query': 'sustainability', 'top_k': 10, 'filters': []}
)
```

## Architecture

```
tests/contracts/
├── README.md                          # This file
├── contract_generator.py              # Generates all 313 contracts
├── contract_validator.py              # Main validator class
├── integration_example.py             # Usage examples
├── {Adapter}_{method}.yaml           # 313 contract files
└── ...
```

## References

- **JSON Schema Draft 7**: https://json-schema.org/draft-07/json-schema-release-notes.html
- **Module Adapters**: `orchestrator/module_adapters.py`
- **Tests**: `tests/test_contract_validator.py`
