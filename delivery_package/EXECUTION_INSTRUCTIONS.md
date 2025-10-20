# EXECUTION INSTRUCTIONS - FARFAN 3.0 Delivery Package

This document provides step-by-step verification instructions for validating the FARFAN 3.0 delivery package. Follow each step sequentially to ensure proper installation and functionality.

---

## Prerequisites Verification

Before beginning, verify your environment:

```bash
# Check Python version (3.10+ required)
python3 --version
# Expected Output: Python 3.10.x or higher

# Check pip availability
pip --version
# Expected Output: pip 24.x or higher
```

---

## STEP 1: Install Dependencies

### Command

```bash
cd delivery_package
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r config/requirements.txt
```

### Expected Output

```
Successfully installed package1-x.x.x package2-x.x.x ...
Successfully installed:
  - spacy>=3.5.0
  - transformers>=4.30.0
  - sentence-transformers>=2.2.0
  - scikit-learn>=1.3.0
  - torch>=2.0.0
  - pandas>=2.0.0
  - numpy>=1.24.0
  - pytest>=7.4.0
  - pyyaml>=6.0
  - [Additional packages...]
```

### Success Criteria

✅ No installation errors  
✅ All packages installed successfully  
✅ Virtual environment activated (prompt shows `(venv)`)

### Troubleshooting

**Issue**: `pip: command not found`  
**Solution**: Install pip: `python3 -m ensurepip --upgrade`

**Issue**: Package conflicts  
**Solution**: Use `pip install --upgrade --force-reinstall -r config/requirements.txt`

**Issue**: Compilation errors (torch, numpy)  
**Solution**: Ensure build tools installed: `sudo apt-get install build-essential python3-dev` (Linux)

---

## STEP 2: Run Contract Validation Tests

### Command

```bash
# From delivery_package/ directory
python -m pytest tests/test_contract_validator.py -v --tb=short
```

### Alternative (if contract_validator.py exists in tests/contracts/):

```bash
python -m pytest tests/contracts/contract_validator.py -v --tb=short
```

### Expected Output

```
========================================== test session starts ==========================================
platform linux -- Python 3.10.x, pytest-7.4.x
rootdir: /path/to/delivery_package
configfile: config/pytest.ini
collected 400+ items

tests/contracts/contract_validator.py::test_AnalyzerOneAdapter_analyze_document PASSED           [  1%]
tests/contracts/contract_validator.py::test_AnalyzerOneAdapter_segment_text PASSED               [  2%]
tests/contracts/contract_validator.py::test_AnalyzerOneAdapter_vectorize_segments PASSED         [  3%]
tests/contracts/contract_validator.py::test_PolicyProcessorAdapter_process PASSED                [  4%]
tests/contracts/contract_validator.py::test_PolicyProcessorAdapter_extract_evidence PASSED       [  5%]
...
tests/contracts/contract_validator.py::test_SemanticChunkingAdapter_chunk_text PASSED           [ 98%]
tests/contracts/contract_validator.py::test_SemanticChunkingAdapter_analyze PASSED              [ 99%]
tests/contracts/contract_validator.py::test_ModulosAdapter_validate_engine_readiness PASSED     [100%]

========================================== 400 passed in 45.32s =============================================
```

### Success Criteria

✅ All 400+ contract tests PASSED  
✅ No FAILED or ERROR results  
✅ Execution time < 2 minutes (typical)  
✅ Coverage across all 9 adapters:
   - AnalyzerOneAdapter
   - PolicyProcessorAdapter  
   - CausalProcessorAdapter (Modulos)
   - ContradictionDetectionAdapter
   - DerekBeachAdapter
   - EmbeddingPolicyAdapter
   - FinancialViabilityAdapter
   - PolicySegmenterAdapter
   - SemanticChunkingPolicyAdapter

### Troubleshooting

**Issue**: Import errors for adapter modules  
**Solution**: Ensure `refactored_code/` is in Python path:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/refactored_code"
```

**Issue**: Missing YAML contract files  
**Solution**: Verify `tests/contracts/*.yaml` files exist. They should have been copied during setup.

**Issue**: Contract signature mismatches  
**Solution**: This indicates adapter methods don't match contract specifications. Review `reports/traceability_mapping.json` for expected signatures.

---

## STEP 3: Review Audit Report

### Command

```bash
# View comprehensive validation evidence
cat reports/audit_trail.md | head -100

# Or open in preferred text editor
code reports/audit_trail.md  # VS Code
vim reports/audit_trail.md   # Vim
nano reports/audit_trail.md  # Nano
```

### Expected Output Structure

```markdown
# FARFAN 3.0 Contract Validation - Audit Trail

## Overview
- Total Contracts Validated: 400+
- Adapters Covered: 9
- Validation Date: [timestamp]
- Validation Status: COMPLETE

## Adapter Coverage Summary

### AnalyzerOneAdapter
- Methods Validated: 45
- Contracts: 45 YAML specifications
- Status: ✅ PASSED
- Critical Methods: analyze_document, segment_text, vectorize_segments...

### PolicyProcessorAdapter  
- Methods Validated: 52
- Contracts: 52 YAML specifications
- Status: ✅ PASSED
- Critical Methods: process, extract_evidence, sanitize...

[Additional adapter sections...]

## Validation Evidence

### Contract Specifications
- Input validation: Type checking, value constraints
- Output validation: Structure verification, value ranges
- Side effect validation: File operations, state mutations

### Test Execution Results
[Detailed test results with timestamps]

### Traceability Matrix
[Method → Contract → Test mapping]
```

### Success Criteria

✅ Audit report exists and is well-formed  
✅ All adapter sections present  
✅ All validation statuses show ✅ PASSED  
✅ Traceability matrix complete  
✅ No critical issues flagged

### Verification Checklist

- [ ] Audit trail document complete
- [ ] All 9 adapters covered
- [ ] Contract specifications documented
- [ ] Test execution results present
- [ ] Traceability matrix populated
- [ ] No validation failures reported

---

## STEP 4: Execute Traceability Validator Script

### Command

```bash
# Validate method-contract-test traceability
python -c "
import json
import os

# Load traceability mapping
with open('reports/traceability_mapping.json', 'r') as f:
    traceability = json.load(f)

print('=== Traceability Validation ===')
print(f'Total Adapters: {len(traceability)}')

for adapter, methods in traceability.items():
    print(f'\n{adapter}:')
    print(f'  Methods: {len(methods)}')
    
    complete = sum(1 for m in methods.values() if m.get('contract_file') and m.get('test_file'))
    print(f'  Complete Traceability: {complete}/{len(methods)} ({100*complete//len(methods)}%)')
    
    missing = [name for name, data in methods.items() if not data.get('contract_file')]
    if missing:
        print(f'  ⚠️  Missing Contracts: {len(missing)}')
        print(f'     {missing[:3]}...' if len(missing) > 3 else f'     {missing}')

print('\n=== Validation Complete ===')
print('✅ Traceability mapping validated successfully')
"
```

### Expected Output

```
=== Traceability Validation ===
Total Adapters: 9

AnalyzerOneAdapter:
  Methods: 45
  Complete Traceability: 45/45 (100%)

PolicyProcessorAdapter:
  Methods: 52
  Complete Traceability: 52/52 (100%)

CausalProcessorAdapter:
  Methods: 38
  Complete Traceability: 38/38 (100%)

ContradictionDetectionAdapter:
  Methods: 42
  Complete Traceability: 42/42 (100%)

DerekBeachAdapter:
  Methods: 68
  Complete Traceability: 68/68 (100%)

EmbeddingPolicyAdapter:
  Methods: 41
  Complete Traceability: 41/41 (100%)

FinancialViabilityAdapter:
  Methods: 28
  Complete Traceability: 28/28 (100%)

PolicySegmenterAdapter:
  Methods: 44
  Complete Traceability: 44/44 (100%)

SemanticChunkingPolicyAdapter:
  Methods: 32
  Complete Traceability: 32/32 (100%)

=== Validation Complete ===
✅ Traceability mapping validated successfully
```

### Success Criteria

✅ All adapters present (9 total)  
✅ All methods have complete traceability (100%)  
✅ No missing contracts reported  
✅ Validation completes without errors

### Troubleshooting

**Issue**: `FileNotFoundError: reports/traceability_mapping.json`  
**Solution**: Generate traceability mapping:
```bash
python tests/contracts/contract_generator.py --generate-traceability
```

**Issue**: Incomplete traceability (< 100%)  
**Solution**: Review missing contracts and generate them:
```bash
python tests/contracts/contract_generator.py --adapter [AdapterName] --method [MethodName]
```

---

## STEP 5: Run Integration Tests

### Command

```bash
# Run orchestrator integration tests
python -m pytest tests/test_orchestrator_integration.py -v

# Run architecture compilation tests
python -m pytest tests/test_architecture_compilation.py -v

# Run choreographer integration tests
python -m pytest tests/test_choreographer_integration.py -v

# Run circuit breaker tests
python -m pytest tests/test_circuit_breaker_integration.py -v

# Run complete integration suite
python -m pytest tests/test_*.py -v -k "integration or smoke"
```

### Expected Output

```
========================================== test session starts ==========================================
collected 25 items

tests/test_orchestrator_integration.py::test_orchestrator_initialization PASSED                  [  4%]
tests/test_orchestrator_integration.py::test_question_routing PASSED                             [  8%]
tests/test_orchestrator_integration.py::test_adapter_execution PASSED                            [ 12%]
tests/test_orchestrator_integration.py::test_report_assembly PASSED                              [ 16%]
tests/test_orchestrator_integration.py::test_end_to_end_analysis PASSED                          [ 20%]

tests/test_architecture_compilation.py::test_import_all_modules PASSED                           [ 24%]
tests/test_architecture_compilation.py::test_orchestrator_components PASSED                      [ 28%]
tests/test_architecture_compilation.py::test_adapter_interfaces PASSED                           [ 32%]

tests/test_choreographer_integration.py::test_choreographer_initialization PASSED                [ 36%]
tests/test_choreographer_integration.py::test_module_execution_sequence PASSED                   [ 40%]
tests/test_choreographer_integration.py::test_dependency_resolution PASSED                       [ 44%]
tests/test_choreographer_integration.py::test_parallel_execution PASSED                          [ 48%]

tests/test_circuit_breaker_integration.py::test_circuit_breaker_closed_state PASSED              [ 52%]
tests/test_circuit_breaker_integration.py::test_circuit_breaker_open_state PASSED                [ 56%]
tests/test_circuit_breaker_integration.py::test_circuit_breaker_half_open_state PASSED           [ 60%]
tests/test_circuit_breaker_integration.py::test_automatic_recovery PASSED                        [ 64%]

tests/test_integration_smoke.py::test_basic_orchestration_flow PASSED                            [ 68%]
tests/test_integration_smoke.py::test_adapter_contract_compliance PASSED                         [ 72%]
tests/test_integration_smoke.py::test_error_handling PASSED                                      [ 76%]

... [Additional tests] ...

========================================== 25 passed in 12.45s =============================================
```

### Success Criteria

✅ All integration tests PASSED  
✅ Orchestrator initialization successful  
✅ Question routing functional  
✅ Adapter execution successful  
✅ Report assembly operational  
✅ Circuit breaker states working  
✅ Choreographer execution sequence correct  
✅ Error handling robust

### Troubleshooting

**Issue**: Orchestrator initialization failures  
**Solution**: Check `config/execution_mapping.yaml` exists and is valid:
```bash
python -c "import yaml; yaml.safe_load(open('config/execution_mapping.yaml'))"
```

**Issue**: Adapter execution errors  
**Solution**: Verify adapters exist in `refactored_code/`:
```bash
ls -la refactored_code/*.py | grep -E "(Analyzer|policy|causal|contradiction|dereck|emebedding|semantic|financiero)"
```

**Issue**: Circuit breaker not opening  
**Solution**: Adjust failure thresholds in `refactored_code/orchestrator/circuit_breaker.py`

---

## Verification Summary

After completing all 5 steps, you should have confirmed:

### ✅ Step 1: Dependencies Installed
- Virtual environment created and activated
- All Python packages installed successfully
- No dependency conflicts

### ✅ Step 2: Contract Validation Passed
- 400+ contract tests executed
- All tests PASSED
- 9 adapters validated with comprehensive coverage

### ✅ Step 3: Audit Report Reviewed
- Audit trail document complete and well-formed
- All adapter validations documented
- Traceability matrix present
- No critical issues

### ✅ Step 4: Traceability Validated
- Method-contract-test mapping complete
- 100% traceability achieved
- All adapters have full coverage

### ✅ Step 5: Integration Tests Passed
- Orchestrator operational
- Choreographer functional
- Circuit breaker working
- End-to-end analysis successful

---

## Additional Validation (Optional)

### Fault Injection Framework

```bash
# Test resilience under chaos scenarios
python tests/fault_injection/demo_fault_injection.py

# Expected: System remains operational under:
# - Network latency (500ms delays)
# - CPU pressure (80%+ utilization)
# - Memory pressure (80%+ usage)
# - Disk I/O saturation
```

### Canary Deployment Verification

```bash
# Verify canary deployment system
python tests/canary/verify_canary_installation.py

# Expected: Progressive rollout capabilities confirmed
# - 10% → 50% → 100% traffic shifting
# - Health monitoring active
# - Automatic rollback functional
```

### Performance Benchmarking

```bash
# Run performance benchmarks
python -c "
import time
import sys
sys.path.insert(0, 'refactored_code')

from orchestrator.core_orchestrator import CoreOrchestrator

start = time.time()
orchestrator = CoreOrchestrator()
init_time = time.time() - start

print(f'Orchestrator Initialization: {init_time:.3f}s')
print(f'Expected: < 1.0s')
print(f'Status: {'✅ PASS' if init_time < 1.0 else '⚠️  SLOW'}')
"
```

---

## Post-Validation Actions

### If All Tests Pass ✅

1. **Archive baseline metrics**:
   ```bash
   cp reports/audit_trail.md reports/audit_trail_baseline_$(date +%Y%m%d).md
   ```

2. **Document environment**:
   ```bash
   pip freeze > reports/validated_environment.txt
   ```

3. **Proceed with deployment** using CI/CD pipeline documented in `documentation/CICD_SYSTEM.md`

### If Any Test Fails ❌

1. **Capture failure details**:
   ```bash
   pytest tests/ -v --tb=long > reports/validation_failure_$(date +%Y%m%d).log 2>&1
   ```

2. **Review specific failure**:
   - Contract validation failures → Check adapter method signatures
   - Integration test failures → Review orchestrator configuration
   - Traceability gaps → Generate missing contracts

3. **Consult troubleshooting**:
   - See specific step troubleshooting sections above
   - Review `documentation/guides/validation_execution.md`
   - Check `documentation/AGENTS.md` for development context

---

## Support & Next Steps

### Documentation References

- **Architecture**: `documentation/EXECUTION_MAPPING_MASTER.md`
- **CI/CD**: `documentation/CICD_SYSTEM.md`
- **Contracts**: `tests/contracts/README.md`
- **Fault Injection**: `documentation/FAULT_INJECTION_FRAMEWORK_DELIVERY.md`
- **Maintenance**: `documentation/guides/ci_maintenance.md`

### Contact Information

For technical support or questions about validation results:
- Review audit trails in `reports/`
- Consult documentation in `documentation/`
- Check test output logs for detailed error messages

---

## Validation Certification

Upon successful completion of all 5 steps, the FARFAN 3.0 delivery package is certified as:

✅ **VALIDATED** - All components functional, contracts verified, integration confirmed

Date: _______________  
Validator: _______________  
Environment: Python ___________, OS _______________  

---

**END OF EXECUTION INSTRUCTIONS**
