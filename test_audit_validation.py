#!/usr/bin/env python3
"""
Test Audit Validation - Verify all audit requirements
======================================================
Tests the audit findings documented in AUDIT_REPORT.md
"""
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

def test_module_controller():
    """Test ModuleController architecture"""
    print("Testing ModuleController...")
    
    from orchestrator.module_controller import ModuleController
    from orchestrator.module_adapters import ModuleAdapterRegistry
    from orchestrator.circuit_breaker import CircuitBreaker
    
    # Test 1: Registry-based initialization
    registry = ModuleAdapterRegistry()
    breaker = CircuitBreaker()
    
    controller = ModuleController(
        module_adapter_registry=registry,
        circuit_breaker=breaker
    )
    
    assert len(controller.adapters) >= 9, "Should have at least 9 adapters"
    assert controller.circuit_breaker is not None, "Circuit breaker should be set"
    assert controller.responsibility_map is not None, "Responsibility map should be loaded"
    
    print(f"  ✅ ModuleController: {len(controller.adapters)} adapters registered")
    print(f"  ✅ Responsibility map: {len(controller.responsibility_map.get('dimensions', {}))} dimensions")
    return True


def test_circuit_breaker():
    """Test CircuitBreaker fault tolerance"""
    print("Testing CircuitBreaker...")
    
    from orchestrator.circuit_breaker import CircuitBreaker, CircuitState
    
    breaker = CircuitBreaker(
        failure_threshold=5,
        recovery_timeout=60.0,
        half_open_max_calls=3
    )
    
    # Test initial state
    assert breaker.can_execute("teoria_cambio"), "Should allow execution initially"
    
    # Test failure counting
    for i in range(5):
        breaker.record_failure("teoria_cambio", "test error")
    
    # Should be open after 5 failures
    assert breaker.adapter_states["teoria_cambio"] == CircuitState.OPEN, "Circuit should be open"
    assert not breaker.can_execute("teoria_cambio"), "Should block execution when open"
    
    # Test success recording
    breaker.record_success("analyzer_one", 0.5)
    assert breaker.adapter_metrics["analyzer_one"].success_count == 1, "Should track successes"
    
    print("  ✅ Circuit breaker: Failure threshold working")
    print("  ✅ Circuit breaker: State transitions working")
    return True


def test_orchestrator_integration():
    """Test FARFANOrchestrator auto-initialization"""
    print("Testing FARFANOrchestrator...")
    
    from orchestrator import FARFANOrchestrator
    
    # Test auto-initialization
    orch = FARFANOrchestrator()
    
    assert orch.module_registry is not None, "Registry should be created"
    assert orch.questionnaire_parser is not None, "Parser should be created"
    assert orch.module_controller is not None, "Controller should be created"
    assert orch.circuit_breaker is not None, "Circuit breaker should be created"
    assert orch.choreographer is not None, "Choreographer should be created"
    
    # Test adapter count
    adapter_count = len(orch.module_registry.adapters)
    assert adapter_count >= 9, f"Should have at least 9 adapters, got {adapter_count}"
    
    print(f"  ✅ FARFANOrchestrator: Auto-initialization working")
    print(f"  ✅ Adapters registered: {adapter_count}")
    return True


def test_responsibility_map():
    """Test responsibility_map.json structure"""
    print("Testing responsibility_map.json...")
    
    import json
    from pathlib import Path
    
    map_path = Path("orchestrator/responsibility_map.json")
    assert map_path.exists(), "responsibility_map.json should exist"
    
    with open(map_path, 'r') as f:
        mapping = json.load(f)
    
    # Validate structure
    assert "dimensions" in mapping, "Should have dimensions"
    assert "policy_areas" in mapping, "Should have policy_areas"
    assert "adapter_capabilities" in mapping, "Should have adapter_capabilities"
    assert "scoring_integration" in mapping, "Should have scoring_integration"
    
    # Validate dimensions
    dimensions = mapping["dimensions"]
    assert len(dimensions) == 6, f"Should have 6 dimensions, got {len(dimensions)}"
    
    for dim_id, dim_data in dimensions.items():
        assert "primary_adapters" in dim_data, f"{dim_id} missing primary_adapters"
        assert "secondary_adapters" in dim_data, f"{dim_id} missing secondary_adapters"
        assert "execution_strategy" in dim_data, f"{dim_id} missing execution_strategy"
    
    # Validate policy areas
    policy_areas = mapping["policy_areas"]
    assert len(policy_areas) == 10, f"Should have 10 policy areas, got {len(policy_areas)}"
    
    for policy_id, policy_data in policy_areas.items():
        assert "name" in policy_data, f"{policy_id} missing name"
        assert "scoring_weights" in policy_data, f"{policy_id} missing scoring_weights"
    
    print(f"  ✅ Responsibility map: {len(dimensions)} dimensions validated")
    print(f"  ✅ Responsibility map: {len(policy_areas)} policy areas validated")
    return True


def test_rubric_scoring():
    """Test rubric_scoring.json structure"""
    print("Testing rubric_scoring.json...")
    
    import json
    from pathlib import Path
    
    rubric_path = Path("rubric_scoring.json")
    assert rubric_path.exists(), "rubric_scoring.json should exist"
    
    with open(rubric_path, 'r') as f:
        rubric = json.load(f)
    
    # Validate structure
    assert "scoring_modalities" in rubric, "Should have scoring_modalities"
    assert "aggregation_levels" in rubric, "Should have aggregation_levels"
    assert "dimensions" in rubric, "Should have dimensions"
    
    # Validate modalities
    modalities = rubric["scoring_modalities"]
    expected_types = ["TYPE_A", "TYPE_B", "TYPE_C", "TYPE_D", "TYPE_E", "TYPE_F"]
    
    for mod_type in expected_types:
        assert mod_type in modalities, f"Missing modality {mod_type}"
        mod_data = modalities[mod_type]
        assert "max_score" in mod_data, f"{mod_type} missing max_score"
        assert mod_data["max_score"] == 3.0, f"{mod_type} max_score should be 3.0"
    
    # Validate dimensions
    dimensions = rubric["dimensions"]
    assert len(dimensions) == 6, f"Should have 6 dimensions, got {len(dimensions)}"
    
    for dim_id, dim_data in dimensions.items():
        assert "max_score" in dim_data, f"{dim_id} missing max_score"
        assert dim_data["max_score"] == 15, f"{dim_id} max_score should be 15"
        assert len(dim_data["questions"]) == 5, f"{dim_id} should have 5 questions"
    
    print(f"  ✅ Rubric scoring: {len(modalities)} modalities validated")
    print(f"  ✅ Rubric scoring: {len(dimensions)} dimensions validated")
    return True


def test_adapter_registry():
    """Test ModuleAdapterRegistry"""
    print("Testing ModuleAdapterRegistry...")
    
    from orchestrator.module_adapters import ModuleAdapterRegistry
    
    registry = ModuleAdapterRegistry()
    
    # Check adapter registration
    expected_adapters = [
        "teoria_cambio",
        "analyzer_one",
        "dereck_beach",
        "embedding_policy",
        "semantic_chunking_policy",
        "contradiction_detection",
        "financial_viability",
        "policy_processor",
        "policy_segmenter"
    ]
    
    for adapter_name in expected_adapters:
        assert adapter_name in registry.adapters, f"Missing adapter: {adapter_name}"
    
    # Test execute_module_method
    available_modules = registry.get_available_modules()
    assert len(available_modules) >= 9, f"Should have at least 9 available modules"
    
    print(f"  ✅ ModuleAdapterRegistry: {len(registry.adapters)} adapters registered")
    print(f"  ✅ Available modules: {len(available_modules)}")
    return True


def main():
    """Run all audit validation tests"""
    print("=" * 80)
    print("FARFAN 3.0 Audit Validation Tests")
    print("=" * 80)
    print()
    
    tests = [
        ("ModuleController", test_module_controller),
        ("CircuitBreaker", test_circuit_breaker),
        ("FARFANOrchestrator", test_orchestrator_integration),
        ("responsibility_map.json", test_responsibility_map),
        ("rubric_scoring.json", test_rubric_scoring),
        ("ModuleAdapterRegistry", test_adapter_registry),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"  ❌ {test_name} FAILED")
        except Exception as e:
            failed += 1
            print(f"  ❌ {test_name} ERROR: {e}")
        print()
    
    print("=" * 80)
    print(f"Results: {passed}/{len(tests)} tests passed")
    
    if failed == 0:
        print("✅ ALL AUDIT REQUIREMENTS VALIDATED")
        print("=" * 80)
        return 0
    else:
        print(f"❌ {failed} tests failed")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
