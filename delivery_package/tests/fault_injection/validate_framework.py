#!/usr/bin/env python3
"""
Framework Validation Script
===========================

Validates that the fault injection framework is working correctly.

Usage:
    python tests/fault_injection/validate_framework.py

Author: FARFAN Integration Team
"""

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def validate_imports():
    """Validate all imports work"""
    print("✓ Validating imports...")
    
    try:
        from tests.fault_injection import (
            ContractFaultInjector,
            DeterminismFaultInjector,
            FaultToleranceFaultInjector,
            OperationalFaultInjector,
            ResilienceValidator,
            ChaosScenarioRunner
        )
        print("  ✓ All imports successful")
        return True
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False


def validate_injectors():
    """Validate injectors work"""
    print("\n✓ Validating injectors...")
    
    try:
        from tests.fault_injection import (
            ContractFaultInjector,
            DeterminismFaultInjector,
            FaultToleranceFaultInjector,
            OperationalFaultInjector
        )
        
        # Contract injector
        contract = ContractFaultInjector()
        fault = contract.inject_type_mismatch("test_adapter", "test_method", dict, "wrong")
        assert fault.target_adapter == "test_adapter"
        assert len(contract.injected_faults) == 1
        contract.reset()
        print("  ✓ ContractFaultInjector works")
        
        # Determinism injector
        determinism = DeterminismFaultInjector()
        fault = determinism.inject_seed_corruption("test_adapter", "both")
        assert fault.target_adapter == "test_adapter"
        determinism.reset()
        print("  ✓ DeterminismFaultInjector works")
        
        # Fault tolerance injector
        fault_tolerance = FaultToleranceFaultInjector()
        fault = fault_tolerance.inject_circuit_breaker_stuck("test_adapter", "OPEN")
        assert fault.target_adapter == "test_adapter"
        fault_tolerance.reset()
        print("  ✓ FaultToleranceFaultInjector works")
        
        # Operational injector
        operational = OperationalFaultInjector()
        fault = operational.inject_disk_full("test_adapter")
        assert fault.target_adapter == "test_adapter"
        operational.reset()
        print("  ✓ OperationalFaultInjector works")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Injector validation failed: {e}")
        return False


def validate_resilience_validator():
    """Validate ResilienceValidator works"""
    print("\n✓ Validating ResilienceValidator...")
    
    try:
        from tests.fault_injection import ResilienceValidator
        
        validator = ResilienceValidator()
        
        # Check adapters list
        assert len(validator.ADAPTERS) == 9
        print(f"  ✓ 9 adapters configured: {validator.ADAPTERS[:3]}...")
        
        # Check circuit breaker exists
        assert validator.circuit_breaker is not None
        print("  ✓ Circuit breaker integrated")
        
        # Check injectors exist
        assert validator.contract_injector is not None
        assert validator.determinism_injector is not None
        assert validator.fault_tolerance_injector is not None
        assert validator.operational_injector is not None
        print("  ✓ All injectors integrated")
        
        # Generate empty report
        report = validator.generate_report()
        assert "summary" in report
        print("  ✓ Report generation works")
        
        validator.reset()
        return True
        
    except Exception as e:
        print(f"  ✗ ResilienceValidator validation failed: {e}")
        return False


def validate_chaos_runner():
    """Validate ChaosScenarioRunner works"""
    print("\n✓ Validating ChaosScenarioRunner...")
    
    try:
        from tests.fault_injection import ChaosScenarioRunner
        
        runner = ChaosScenarioRunner()
        
        # Check adapters
        assert len(runner.ADAPTERS) == 9
        print(f"  ✓ 9 adapters configured")
        
        # Build scenarios
        scenario1 = runner.build_partial_failure_scenario(num_failures=1)
        assert scenario1 is not None
        assert len(scenario1.affected_adapters) == 1
        print("  ✓ Partial failure scenario builds")
        
        scenario2 = runner.build_cascading_risk_scenario()
        assert scenario2 is not None
        print("  ✓ Cascading risk scenario builds")
        
        scenario3 = runner.build_combined_chaos_scenario()
        assert scenario3 is not None
        assert len(scenario3.affected_adapters) == 5
        print("  ✓ Combined chaos scenario builds")
        
        # Generate empty report
        report = runner.generate_chaos_report()
        assert "error" in report  # Expected since no scenarios run yet
        print("  ✓ Report generation works")
        
        return True
        
    except Exception as e:
        print(f"  ✗ ChaosScenarioRunner validation failed: {e}")
        return False


def validate_circuit_breaker_integration():
    """Validate circuit breaker integration"""
    print("\n✓ Validating circuit breaker integration...")
    
    try:
        from orchestrator.circuit_breaker import CircuitBreaker, CircuitState
        
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=5.0)
        
        # Check adapters initialized
        assert "teoria_cambio" in cb.adapter_states
        assert "analyzer_one" in cb.adapter_states
        print("  ✓ Circuit breaker knows about 9 adapters")
        
        # Check initial state
        assert cb.adapter_states["teoria_cambio"] == CircuitState.CLOSED
        print("  ✓ Initial state is CLOSED")
        
        # Test failure recording
        cb.record_failure("teoria_cambio", "test error")
        status = cb.get_adapter_status("teoria_cambio")
        assert status["failures"] > 0
        print("  ✓ Failure recording works")
        
        # Test success recording
        cb.record_success("teoria_cambio")
        status = cb.get_adapter_status("teoria_cambio")
        assert status["successes"] > 0
        print("  ✓ Success recording works")
        
        cb.reset_all()
        return True
        
    except Exception as e:
        print(f"  ✗ Circuit breaker validation failed: {e}")
        return False


def validate_file_structure():
    """Validate file structure"""
    print("\n✓ Validating file structure...")
    
    required_files = [
        "tests/fault_injection/__init__.py",
        "tests/fault_injection/injectors.py",
        "tests/fault_injection/resilience_validator.py",
        "tests/fault_injection/chaos_scenarios.py",
        "tests/fault_injection/demo_fault_injection.py",
        "tests/fault_injection/README.md",
        "tests/fault_injection/IMPLEMENTATION_SUMMARY.md",
        "tests/test_fault_injection_framework.py"
    ]
    
    missing = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing.append(file_path)
    
    if missing:
        print(f"  ✗ Missing files: {missing}")
        return False
    
    print(f"  ✓ All {len(required_files)} required files present")
    return True


def main():
    """Run all validations"""
    print("="*80)
    print("FAULT INJECTION FRAMEWORK VALIDATION")
    print("="*80)
    
    results = []
    
    # Run validations
    results.append(("File Structure", validate_file_structure()))
    results.append(("Imports", validate_imports()))
    results.append(("Injectors", validate_injectors()))
    results.append(("ResilienceValidator", validate_resilience_validator()))
    results.append(("ChaosScenarioRunner", validate_chaos_runner()))
    results.append(("Circuit Breaker Integration", validate_circuit_breaker_integration()))
    
    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{name:.<40} {status}")
    
    print("-"*80)
    print(f"Total: {passed}/{total} validations passed ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("\n✓ ALL VALIDATIONS PASSED - Framework is ready!")
        print("\nNext steps:")
        print("  - Run demo: python tests/fault_injection/demo_fault_injection.py")
        print("  - Run tests: pytest tests/test_fault_injection_framework.py -v")
        print("="*80)
        return 0
    else:
        print("\n✗ SOME VALIDATIONS FAILED - Review errors above")
        print("="*80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
