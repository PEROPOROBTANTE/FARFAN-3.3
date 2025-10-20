#!/usr/bin/env python3
"""
Demo Script - Fault Injection Framework
========================================

Demonstrates fault injection testing framework capabilities.

Usage:
    python tests/fault_injection/demo_fault_injection.py

Author: FARFAN Integration Team
"""

import sys
import logging
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.fault_injection import (
    ContractFaultInjector,
    DeterminismFaultInjector,
    FaultToleranceFaultInjector,
    OperationalFaultInjector,
    ResilienceValidator,
    ChaosScenarioRunner
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demo_contract_faults():
    """Demo: Contract fault injection"""
    print("\n" + "="*80)
    print("DEMO 1: Contract Fault Injection")
    print("="*80)
    
    injector = ContractFaultInjector()
    
    # Type mismatch
    fault1 = injector.inject_type_mismatch(
        "teoria_cambio",
        "execute",
        dict,
        "wrong_type_string"
    )
    print(f"✓ Injected type mismatch: {fault1.description}")
    
    # Missing binding
    fault2 = injector.inject_missing_binding(
        "policy_processor",
        "semantic_chunking_policy",
        "normalized_text"
    )
    print(f"✓ Injected missing binding: {fault2.description}")
    
    # Schema break
    fault3 = injector.inject_schema_break(
        "analyzer_one",
        "malformed_module_result"
    )
    print(f"✓ Injected schema break: {fault3.description}")
    
    print(f"\nTotal faults injected: {len(injector.injected_faults)}")
    
    injector.reset()
    print("✓ Cleaned up faults")


def demo_determinism_faults():
    """Demo: Determinism fault injection"""
    print("\n" + "="*80)
    print("DEMO 2: Determinism Fault Injection")
    print("="*80)
    
    injector = DeterminismFaultInjector()
    
    # Seed corruption
    fault1 = injector.inject_seed_corruption(
        "dereck_beach",
        "both"
    )
    print(f"✓ Injected seed corruption: {fault1.description}")
    
    # Timestamp noise
    fault2 = injector.inject_timestamp_noise(
        "analyzer_one",
        "analyze"
    )
    print(f"✓ Injected timestamp noise: {fault2.description}")
    
    # Random noise
    fault3 = injector.inject_random_noise(
        "teoria_cambio",
        noise_level=0.15
    )
    print(f"✓ Injected random noise: {fault3.description}")
    
    print(f"\nTotal faults injected: {len(injector.injected_faults)}")
    
    injector.reset()
    print("✓ Restored determinism")


def demo_fault_tolerance_faults():
    """Demo: Fault tolerance fault injection"""
    print("\n" + "="*80)
    print("DEMO 3: Fault Tolerance Fault Injection")
    print("="*80)
    
    injector = FaultToleranceFaultInjector()
    
    # Circuit breaker stuck
    fault1 = injector.inject_circuit_breaker_stuck(
        "embedding_policy",
        "OPEN"
    )
    print(f"✓ Injected circuit breaker stuck: {fault1.description}")
    
    # Retry storm
    fault2 = injector.inject_retry_storm(
        "financial_viability",
        max_retries=100,
        no_backoff=True
    )
    print(f"✓ Injected retry storm: {fault2.description}")
    
    # Timeout misconfiguration
    fault3 = injector.inject_timeout_misconfiguration(
        "contradiction_detection",
        timeout_ms=50,
        timeout_type="premature"
    )
    print(f"✓ Injected timeout misconfiguration: {fault3.description}")
    
    print(f"\nTotal faults injected: {len(injector.injected_faults)}")
    
    injector.reset()
    print("✓ Cleaned up faults")


def demo_operational_faults():
    """Demo: Operational fault injection"""
    print("\n" + "="*80)
    print("DEMO 4: Operational Fault Injection")
    print("="*80)
    
    injector = OperationalFaultInjector()
    
    # Disk full
    fault1 = injector.inject_disk_full(
        "policy_segmenter",
        affected_paths=["/tmp/cache"]
    )
    print(f"✓ Injected disk full: {fault1.description}")
    
    # Clock skew
    fault2 = injector.inject_clock_skew(
        "teoria_cambio",
        skew_seconds=3600.0
    )
    print(f"✓ Injected clock skew: {fault2.description}")
    
    # Network partition
    fault3 = injector.inject_network_partition(
        "embedding_policy",
        partition_type="complete"
    )
    print(f"✓ Injected network partition: {fault3.description}")
    
    # Memory pressure
    fault4 = injector.inject_memory_pressure(
        "dereck_beach",
        pressure_level="high"
    )
    print(f"✓ Injected memory pressure: {fault4.description}")
    
    print(f"\nTotal faults injected: {len(injector.injected_faults)}")
    
    injector.reset()
    print("✓ Cleaned up faults")


def demo_resilience_validation():
    """Demo: Resilience validation"""
    print("\n" + "="*80)
    print("DEMO 5: Resilience Validation")
    print("="*80)
    
    validator = ResilienceValidator()
    
    adapter_name = "teoria_cambio"
    print(f"\nValidating adapter: {adapter_name}")
    
    # Circuit breaker transitions
    print("\n1. Validating circuit breaker state transitions...")
    result1 = validator.validate_circuit_breaker_transitions(adapter_name, failure_count=8)
    print(f"   Status: {result1.status.value}")
    print(f"   Observed sequence: {result1.metrics.get('observed_sequence', [])}")
    
    # Reset for next validation
    validator.circuit_breaker.reset_adapter(adapter_name)
    
    # Timeout enforcement
    print("\n2. Validating timeout enforcement...")
    result2 = validator.validate_timeout_enforcement(adapter_name, max_latency_ms=1000)
    print(f"   Status: {result2.status.value}")
    print(f"   Max latency: {result2.metrics.get('max_latency_ms')}ms")
    
    # Idempotency
    print("\n3. Validating idempotency detection...")
    result3 = validator.validate_idempotency_detection(
        adapter_name,
        "test_method",
        {"input": "test_data"}
    )
    print(f"   Status: {result3.status.value}")
    print(f"   Duplicate detected: {result3.metrics.get('duplicate_detected', False)}")
    
    # Generate report
    report = validator.generate_report()
    print(f"\n✓ Completed {report['summary']['total_validations']} validations")
    print(f"  Success rate: {report['summary']['success_rate']:.1%}")
    
    validator.reset()


def demo_chaos_scenarios():
    """Demo: Chaos testing scenarios"""
    print("\n" + "="*80)
    print("DEMO 6: Chaos Testing Scenarios")
    print("="*80)
    
    runner = ChaosScenarioRunner()
    
    # Build scenarios
    print("\nBuilding chaos scenarios...")
    
    scenario1 = runner.build_partial_failure_scenario(num_failures=2)
    print(f"✓ {scenario1.name}: {len(scenario1.affected_adapters)} adapters, "
          f"{len(scenario1.injected_faults)} faults")
    
    scenario2 = runner.build_cascading_risk_scenario()
    print(f"✓ {scenario2.name}: {len(scenario2.affected_adapters)} adapters, "
          f"{len(scenario2.injected_faults)} faults")
    
    scenario3 = runner.build_combined_chaos_scenario()
    print(f"✓ {scenario3.name}: {len(scenario3.affected_adapters)} adapters, "
          f"{len(scenario3.injected_faults)} faults")
    
    # Run one scenario
    print(f"\nRunning scenario: {scenario1.name}...")
    result = runner.run_scenario(scenario1)
    
    print(f"  Status: {result.status.value}")
    print(f"  Duration: {result.duration:.2f}s")
    print(f"  Cascading failures: {len(result.cascading_failures)}")
    print(f"  Graceful degradation: {result.graceful_degradation}")
    print(f"  Healthy adapters: {result.metrics.get('healthy_adapters', 0)}/{len(runner.ADAPTERS)}")
    
    # Generate report
    report = runner.generate_chaos_report()
    print(f"\n✓ Chaos testing completed")
    print(f"  Total scenarios: {report['summary']['total_scenarios']}")
    print(f"  Success rate: {report['summary']['success_rate']:.1%}")
    print(f"  Graceful degradation rate: {report['summary']['graceful_degradation_rate']:.1%}")


def main():
    """Run all demos"""
    print("\n" + "="*80)
    print("FAULT INJECTION TESTING FRAMEWORK - FARFAN 3.0")
    print("="*80)
    print("\nDemonstration of fault injection capabilities across 9 adapters:")
    print("  1. teoria_cambio")
    print("  2. analyzer_one")
    print("  3. dereck_beach")
    print("  4. embedding_policy")
    print("  5. semantic_chunking_policy")
    print("  6. contradiction_detection")
    print("  7. financial_viability")
    print("  8. policy_processor")
    print("  9. policy_segmenter")
    
    try:
        demo_contract_faults()
        demo_determinism_faults()
        demo_fault_tolerance_faults()
        demo_operational_faults()
        demo_resilience_validation()
        demo_chaos_scenarios()
        
        print("\n" + "="*80)
        print("✓ ALL DEMOS COMPLETED SUCCESSFULLY")
        print("="*80)
        print("\nFramework is ready for testing!")
        print("\nNext steps:")
        print("  - Run full test suite: pytest tests/test_fault_injection_framework.py -v")
        print("  - Run chaos scenarios: python tests/fault_injection/demo_fault_injection.py")
        print("  - See README: tests/fault_injection/README.md")
        
    except Exception as e:
        logger.exception("Error running demos")
        print(f"\n✗ ERROR: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
