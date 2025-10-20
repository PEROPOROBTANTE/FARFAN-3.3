# Canary Regression Detection System

## Overview

The Canary Regression Detection System provides automated testing for all **413 adapter methods** across FARFAN 3.0's 9 module adapters. It detects three types of regressions:

1. **HASH_DELTA** - Output determinism violations (hash mismatch)
2. **CONTRACT_TYPE_ERROR** - JSON schema validation failures
3. **INVALID_EVIDENCE** - Missing required keys in output

## Architecture

```
tests/canaries/
├── <adapter>/
│   ├── <method>/
│   │   ├── input.json          # Deterministic test inputs
│   │   ├── expected.json       # Baseline expected output
│   │   └── expected_hash.txt   # SHA-256 hash of expected.json
│   └── ...
├── generation_report.json      # Canary generation summary
├── test_report.json           # Regression test results
└── fix_report.json            # Fix operations report
```

## Quick Start

### Run Complete Canary Pipeline

```bash
# Generate canaries, run tests, generate fixes (all in one)
./tests/run_canary_system.sh
```

### Individual Commands

```bash
# 1. Generate baseline canaries
python tests/canary_generator.py

# 2. Run regression tests
python tests/canary_runner.py

# 3. Generate fix operations
python tests/canary_fix_generator.py

# 4. Execute automatic fixes (rebaseline)
python tests/canary_fix_generator.py --execute-rebaseline
```

## Module Coverage

The system covers all **413 methods** across 9 adapters:

| Adapter | Methods | Status |
|---------|---------|--------|
| `policy_processor` | 34 | ✓ |
| `policy_segmenter` | 33 | ✓ |
| `analyzer_one` | 39 | ✓ |
| `embedding_policy` | 37 | ✓ |
| `semantic_chunking_policy` | 18 | ✓ |
| `financial_viability` | 60 | ✓ |
| `dereck_beach` | 89 | ✓ |
| `contradiction_detection` | 52 | ✓ |
| `teoria_cambio` | 51 | ✓ |
| **TOTAL** | **413** | ✓ |

## Violation Types

### HASH_DELTA

**Cause**: Method output changed between runs (non-deterministic or intentional change)

**Detection**: SHA-256 hash of output doesn't match `expected_hash.txt`

**Resolution**:
```bash
# If change is intentional, rebaseline:
python tests/canary_generator.py --adapter <adapter> --method <method> --force

# Or rebaseline entire adapter:
python tests/canary_generator.py --adapter <adapter>

# Or rebaseline all (use with caution!):
python tests/canary_fix_generator.py --execute-rebaseline
```

### CONTRACT_TYPE_ERROR

**Cause**: Output doesn't match expected ModuleResult contract schema

**Detection**: Missing required keys or wrong data types

**Expected Contract**:
```python
{
    "module_name": str,
    "class_name": str,
    "method_name": str,
    "status": str,
    "data": dict,
    "evidence": list,
    "confidence": float,
    "errors": list,         # optional
    "warnings": list,       # optional
    "metadata": dict        # optional
}
```

**Resolution**: Fix adapter implementation to match contract

### INVALID_EVIDENCE

**Cause**: Evidence structure missing required fields

**Detection**: Evidence items missing 'type' field or malformed

**Expected Evidence Structure**:
```python
{
    "evidence": [
        {
            "type": str,        # Required
            "confidence": float,  # Recommended
            # ... other fields
        }
    ]
}
```

**Resolution**: Fix adapter to include proper evidence structure

## Workflow

### 1. Initial Setup (First Time)

```bash
# Generate baseline canaries for all 413 methods
python tests/canary_generator.py
```

This creates:
- `tests/canaries/<adapter>/<method>/input.json` - Test inputs
- `tests/canaries/<adapter>/<method>/expected.json` - Expected outputs
- `tests/canaries/<adapter>/<method>/expected_hash.txt` - Output hash

### 2. Run Tests (Continuous Integration)

```bash
# Run regression detection
python tests/canary_runner.py
```

Output:
```
================================================================================
CANARY REGRESSION DETECTION - ALL 413 ADAPTER METHODS
================================================================================

================================================================================
Testing adapter: policy_processor
================================================================================
  ✓ process
  ✓ normalize_unicode
  ✓ segment_into_sentences
  ...

CANARY TEST SUMMARY
================================================================================
Total Methods Tested: 413
  ✓ Passed: 410
  ✗ Failed: 3

Pass Rate: 99.3%
```

### 3. Analyze Violations

```bash
# View detailed report
cat tests/canaries/test_report.json

# Generate fix operations
python tests/canary_fix_generator.py
```

### 4. Resolve Issues

**Automatic (for HASH_DELTA only)**:
```bash
python tests/canary_fix_generator.py --execute-rebaseline
```

**Manual (for CONTRACT_TYPE_ERROR and INVALID_EVIDENCE)**:
1. Review fix report: `tests/canaries/fix_report.json`
2. Fix adapter code in `orchestrator/module_adapters.py`
3. Rerun tests: `python tests/canary_runner.py`
4. If fixed, rebaseline: `python tests/canary_generator.py --adapter <adapter>`

## Rebaseline Commands

### Rebaseline Single Method
```bash
python tests/canary_generator.py --adapter policy_processor --method process
```

### Rebaseline Entire Adapter
```bash
python tests/canary_generator.py --adapter policy_processor
```

### Rebaseline All Adapters
```bash
python tests/canary_generator.py --all
# OR
python tests/canary_fix_generator.py --execute-rebaseline
```

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Canary Regression Tests

on: [push, pull_request]

jobs:
  canary-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.10
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      
      - name: Run canary tests
        run: |
          python tests/canary_runner.py
      
      - name: Upload reports
        if: failure()
        uses: actions/upload-artifact@v2
        with:
          name: canary-reports
          path: tests/canaries/*.json
```

## Best Practices

### When to Rebaseline

**✓ Rebaseline when**:
- Intentionally changed adapter logic
- Updated underlying module behavior
- Fixed a bug that changes output
- Updated dependencies that affect output

**✗ Don't rebaseline when**:
- Tests are failing due to bugs
- Output is non-deterministic (fix the code first)
- Contract violations exist (fix the contract first)

### Development Workflow

1. **Before making changes**: Run canary tests to ensure baseline
2. **After making changes**: Run canary tests to detect regressions
3. **If intentional**: Review changes, then rebaseline affected methods
4. **If unintentional**: Debug and fix the issue, rerun tests

### Code Review

Include canary test results in PRs:
```bash
python tests/canary_runner.py > canary_results.txt
```

## Troubleshooting

### "Module not available" errors

**Cause**: Adapter module not loaded

**Fix**:
```bash
# Check if module exists
python -c "import <module_name>"

# Check adapter registry
python -c "from orchestrator.module_adapters import ModuleAdapterRegistry; r = ModuleAdapterRegistry(); print(r.get_module_status())"
```

### "Method not found" errors

**Cause**: Method name doesn't exist in adapter

**Fix**: Check method definitions in `orchestrator/module_adapters.py`

### Non-deterministic outputs

**Cause**: Method includes timestamps, random values, or external dependencies

**Fix**:
1. Make method deterministic (use fixed seeds, mock external calls)
2. Exclude non-deterministic fields from hash computation
3. Update `_compute_hash()` in `canary_runner.py` to skip fields

### All tests failing after dependency update

**Cause**: Dependency change affected all outputs

**Fix**:
```bash
# Review changes carefully
git diff tests/canaries/

# If legitimate, rebaseline all
python tests/canary_fix_generator.py --execute-rebaseline
```

## Reports

### generation_report.json

Summary of canary generation:
```json
{
  "total_adapters": 9,
  "total_methods": 413,
  "generated": 400,
  "failed": 13,
  "adapters": {
    "policy_processor": {
      "generated": 34,
      "failed": 0
    }
  }
}
```

### test_report.json

Detailed test results:
```json
{
  "timestamp": "2024-01-15T10:30:00",
  "summary": {
    "total_methods": 413,
    "passed": 410,
    "failed": 3,
    "pass_rate": 99.3
  },
  "violations": [
    {
      "adapter": "policy_processor",
      "method": "process",
      "type": "HASH_DELTA",
      "details": "Output hash changed..."
    }
  ]
}
```

### fix_report.json

Fix operations to resolve violations:
```json
{
  "timestamp": "2024-01-15T10:35:00",
  "total_violations": 3,
  "total_fix_operations": 3,
  "fix_operations": [
    {
      "type": "REBASELINE",
      "adapter": "policy_processor",
      "method": "process",
      "command": "python tests/canary_generator.py --adapter policy_processor --method process --force",
      "priority": 2
    }
  ]
}
```

## Advanced Usage

### Custom Test Inputs

Edit method definitions in `tests/canary_generator.py`:

```python
def _get_method_definitions(self, adapter_name: str):
    method_defs = {
        "policy_processor": [
            {
                "name": "process",
                "inputs": {
                    "args": [my_custom_text],
                    "kwargs": {"option": "value"}
                }
            }
        ]
    }
    return method_defs.get(adapter_name, [])
```

### Skip Non-Deterministic Fields

Modify `_compute_hash()` in `canary_runner.py`:

```python
def _compute_hash(self, data: Dict[str, Any]) -> str:
    data_copy = data.copy()
    # Remove non-deterministic fields
    data_copy.pop("execution_time", None)
    data_copy.pop("timestamp", None)
    # ... add more fields to skip
    
    json_str = json.dumps(data_copy, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(json_str.encode('utf-8')).hexdigest()
```

### Parallel Execution

For faster testing:

```bash
# Run adapters in parallel (requires GNU parallel)
parallel python tests/canary_runner.py --adapter {} ::: \
    policy_processor policy_segmenter analyzer_one \
    embedding_policy semantic_chunking_policy \
    financial_viability dereck_beach \
    contradiction_detection teoria_cambio
```

## Metrics

Track regression detection metrics:

- **Coverage**: 413/413 methods (100%)
- **Pass Rate**: Target >95%
- **Detection Rate**: HASH_DELTA, CONTRACT_TYPE_ERROR, INVALID_EVIDENCE
- **False Positive Rate**: <5% (intentional changes)
- **Execution Time**: ~5-10 minutes for full suite

## Support

For issues or questions:

1. Check this README
2. Review generated reports in `tests/canaries/`
3. Check adapter implementation in `orchestrator/module_adapters.py`
4. Review execution mapping in `EXECUTION_MAPPING_MASTER.md`

---

**Version**: 1.0.0  
**Last Updated**: 2024-01-15  
**Maintainer**: Integration Team
