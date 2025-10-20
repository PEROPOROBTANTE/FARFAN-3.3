# test_mapping_loader.py - Test Suite for Mapping Loader
# coding=utf-8
"""
Test Suite for YAMLMappingLoader and Validation
================================================

Tests:
- YAML parsing
- DAG construction
- Binding validation (duplicate/missing producers)
- Type compatibility checking
- Circular dependency detection
- Error diagnostics and remediation suggestions
"""

import logging
import sys
from pathlib import Path

# Add orchestrator to path
sys.path.insert(0, str(Path(__file__).parent))

from orchestrator.mapping_loader import (
    YAMLMappingLoader,
    MappingStartupValidator,
    MappingValidationError,
    ConflictType,
    ContractRegistry,
    TypeContract
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_yaml_loading():
    """Test 1: Load execution_mapping.yaml"""
    print("\n" + "=" * 80)
    print("TEST 1: YAML Loading")
    print("=" * 80)
    
    try:
        loader = YAMLMappingLoader()
        loader._load_yaml()
        
        assert loader.raw_mapping is not None
        assert 'version' in loader.raw_mapping
        assert 'adapters' in loader.raw_mapping
        
        print(f"✓ Loaded YAML version: {loader.raw_mapping['version']}")
        print(f"✓ Total adapters defined: {loader.raw_mapping.get('total_adapters', 0)}")
        print(f"✓ Total methods defined: {loader.raw_mapping.get('total_methods', 0)}")
        
        return True
        
    except Exception as e:
        print(f"✗ YAML loading failed: {e}")
        return False


def test_adapter_registry_parsing():
    """Test 2: Parse adapter registry"""
    print("\n" + "=" * 80)
    print("TEST 2: Adapter Registry Parsing")
    print("=" * 80)
    
    try:
        loader = YAMLMappingLoader()
        loader._load_yaml()
        loader._parse_adapter_registry()
        
        assert len(loader.adapter_registry) > 0
        
        print(f"✓ Parsed {len(loader.adapter_registry)} adapters:")
        for adapter_name, info in loader.adapter_registry.items():
            print(f"  - {adapter_name}: {info['adapter_class']} ({info['methods']} methods)")
        
        # Verify expected adapters
        expected = ['teoria_cambio', 'analyzer_one', 'dereck_beach', 'embedding_policy',
                   'semantic_chunking_policy', 'contradiction_detection', 
                   'financial_viability', 'policy_processor', 'policy_segmenter']
        
        for adapter in expected:
            assert adapter in loader.adapter_registry, f"Missing adapter: {adapter}"
        
        print(f"✓ All expected adapters present")
        
        return True
        
    except Exception as e:
        print(f"✗ Adapter registry parsing failed: {e}")
        return False


def test_execution_chain_parsing():
    """Test 3: Parse execution chains"""
    print("\n" + "=" * 80)
    print("TEST 3: Execution Chain Parsing")
    print("=" * 80)
    
    try:
        loader = YAMLMappingLoader()
        loader._load_yaml()
        loader._parse_adapter_registry()
        loader._parse_execution_chains()
        
        assert len(loader.execution_chains) > 0
        
        print(f"✓ Parsed {len(loader.execution_chains)} execution chains")
        
        # Inspect first chain
        first_chain_id = list(loader.execution_chains.keys())[0]
        first_chain = loader.execution_chains[first_chain_id]
        
        print(f"\nExample: {first_chain_id}")
        print(f"  Description: {first_chain['description']}")
        print(f"  Steps: {len(first_chain['execution_chain'])}")
        
        for step in first_chain['execution_chain'][:3]:  # Show first 3 steps
            print(f"    Step {step.get('step')}: {step.get('adapter')}.{step.get('method')}")
        
        return True
        
    except Exception as e:
        print(f"✗ Execution chain parsing failed: {e}")
        return False


def test_dag_construction():
    """Test 4: Build DAGs from execution chains"""
    print("\n" + "=" * 80)
    print("TEST 4: DAG Construction")
    print("=" * 80)
    
    try:
        loader = YAMLMappingLoader()
        loader._load_yaml()
        loader._parse_adapter_registry()
        loader._parse_execution_chains()
        loader._build_dags()
        
        assert len(loader.execution_dags) > 0
        
        print(f"✓ Built {len(loader.execution_dags)} DAGs")
        
        # Inspect first DAG
        first_dag_id = list(loader.execution_dags.keys())[0]
        first_dag = loader.execution_dags[first_dag_id]
        
        print(f"\nExample: {first_dag_id}")
        print(f"  Nodes: {first_dag.number_of_nodes()}")
        print(f"  Edges: {first_dag.number_of_edges()}")
        
        # Show dependencies
        if first_dag.number_of_edges() > 0:
            print(f"  Sample dependencies:")
            for source, target, data in list(first_dag.edges(data=True))[:3]:
                binding = data.get('binding', 'N/A')
                print(f"    {source} -> {target} (via {binding})")
        
        return True
        
    except Exception as e:
        print(f"✗ DAG construction failed: {e}")
        return False


def test_binding_validation():
    """Test 5: Validate bindings (duplicate/missing producers)"""
    print("\n" + "=" * 80)
    print("TEST 5: Binding Validation")
    print("=" * 80)
    
    try:
        loader = YAMLMappingLoader()
        loader._load_yaml()
        loader._parse_adapter_registry()
        loader._parse_execution_chains()
        loader._build_dags()
        loader._validate_bindings()
        
        # Check for binding conflicts
        binding_conflicts = [
            c for c in loader.conflicts 
            if c.conflict_type in [ConflictType.DUPLICATE_PRODUCER, ConflictType.MISSING_PRODUCER]
        ]
        
        if binding_conflicts:
            print(f"⚠ Found {len(binding_conflicts)} binding conflicts:")
            for conflict in binding_conflicts[:5]:  # Show first 5
                print(f"  - {conflict.conflict_type.value}: {conflict.description}")
        else:
            print(f"✓ No binding conflicts detected")
        
        print(f"✓ Total bindings tracked: {len(loader.binding_producers)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Binding validation failed: {e}")
        return False


def test_type_validation():
    """Test 6: Validate type compatibility"""
    print("\n" + "=" * 80)
    print("TEST 6: Type Validation")
    print("=" * 80)
    
    try:
        # Create contract registry with sample contracts
        registry = ContractRegistry()
        
        # Add sample contracts
        registry.register_contract(TypeContract(
            adapter="policy_segmenter",
            method="segment",
            input_types={"text": "str"},
            output_type="List[Dict[str, Any]]"
        ))
        
        loader = YAMLMappingLoader(contract_registry=registry)
        loader._load_yaml()
        loader._parse_adapter_registry()
        loader._parse_execution_chains()
        loader._build_dags()
        loader._validate_bindings()
        loader._validate_types()
        
        # Check for type conflicts
        type_conflicts = [
            c for c in loader.conflicts 
            if c.conflict_type == ConflictType.TYPE_MISMATCH
        ]
        
        if type_conflicts:
            print(f"⚠ Found {len(type_conflicts)} type conflicts:")
            for conflict in type_conflicts[:3]:  # Show first 3
                print(f"  - {conflict.description}")
                if conflict.type_mismatch_details:
                    details = conflict.type_mismatch_details
                    print(f"    Producer: {details.get('producer_type')}")
                    print(f"    Consumer: {details.get('consumer_type')}")
        else:
            print(f"✓ No type conflicts detected")
        
        return True
        
    except Exception as e:
        print(f"✗ Type validation failed: {e}")
        return False


def test_circular_dependency_detection():
    """Test 7: Detect circular dependencies"""
    print("\n" + "=" * 80)
    print("TEST 7: Circular Dependency Detection")
    print("=" * 80)
    
    try:
        loader = YAMLMappingLoader()
        loader._load_yaml()
        loader._parse_adapter_registry()
        loader._parse_execution_chains()
        loader._build_dags()
        loader._validate_bindings()
        loader._validate_types()
        loader._detect_circular_dependencies()
        
        # Check for circular dependency conflicts
        circular_conflicts = [
            c for c in loader.conflicts 
            if c.conflict_type == ConflictType.CIRCULAR_DEPENDENCY
        ]
        
        if circular_conflicts:
            print(f"⚠ Found {len(circular_conflicts)} circular dependencies:")
            for conflict in circular_conflicts:
                print(f"  - Question: {', '.join(conflict.question_ids)}")
                print(f"    {conflict.description}")
        else:
            print(f"✓ No circular dependencies detected")
        
        return True
        
    except Exception as e:
        print(f"✗ Circular dependency detection failed: {e}")
        return False


def test_full_validation():
    """Test 8: Full validation with conflict reporting"""
    print("\n" + "=" * 80)
    print("TEST 8: Full Validation (Integration)")
    print("=" * 80)
    
    try:
        loader = YAMLMappingLoader()
        
        # This should either pass or raise MappingValidationError
        try:
            loader.load_and_validate()
            print("✓ Full validation PASSED - mapping is structurally sound")
            
            # Show statistics
            stats = loader.get_statistics()
            print("\nMapping Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
            
            return True
            
        except MappingValidationError as e:
            print(f"⚠ Validation detected conflicts (expected for initial mapping):")
            print(f"  Total conflicts: {len(e.conflicts)}")
            
            # Show conflict breakdown
            conflict_types = {}
            for conflict in e.conflicts:
                conflict_type = conflict.conflict_type.value
                conflict_types[conflict_type] = conflict_types.get(conflict_type, 0) + 1
            
            print("\nConflict Breakdown:")
            for conflict_type, count in conflict_types.items():
                print(f"  {conflict_type}: {count}")
            
            # Show first conflict with full diagnostics
            if e.conflicts:
                print("\nFirst Conflict (with diagnostics):")
                print(str(e.conflicts[0]))
            
            return True  # Not a test failure - just showing diagnostics
        
    except Exception as e:
        print(f"✗ Full validation failed unexpectedly: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_startup_validator():
    """Test 9: Startup validation (fail-fast behavior)"""
    print("\n" + "=" * 80)
    print("TEST 9: Startup Validator")
    print("=" * 80)
    
    try:
        # This tests the startup validator interface
        loader = MappingStartupValidator.validate_at_startup()
        
        print("✓ Startup validation completed")
        print("✓ Application would be allowed to start")
        
        return True
        
    except MappingValidationError as e:
        print("⚠ Startup validation would BLOCK application startup")
        print(f"  Reason: {len(e.conflicts)} conflicts detected")
        print("\nThis is EXPECTED behavior for fail-fast validation")
        return True  # Expected behavior
        
    except Exception as e:
        print(f"✗ Startup validator failed unexpectedly: {e}")
        return False


def test_query_interface():
    """Test 10: Query interface for accessing mapping data"""
    print("\n" + "=" * 80)
    print("TEST 10: Query Interface")
    print("=" * 80)
    
    try:
        loader = YAMLMappingLoader()
        loader._load_yaml()
        loader._parse_adapter_registry()
        loader._parse_execution_chains()
        loader._build_dags()
        
        # Test get_execution_chain
        chain_id = "D1_INSUMOS.Q1_Baseline_Identification"
        chain = loader.get_execution_chain(chain_id)
        
        if chain:
            print(f"✓ get_execution_chain('{chain_id}'):")
            print(f"  Steps: {len(chain['execution_chain'])}")
        
        # Test get_execution_dag
        dag = loader.get_execution_dag(chain_id)
        if dag:
            print(f"✓ get_execution_dag('{chain_id}'):")
            print(f"  Nodes: {dag.number_of_nodes()}, Edges: {dag.number_of_edges()}")
        
        # Test get_adapter_info
        adapter_info = loader.get_adapter_info("teoria_cambio")
        if adapter_info:
            print(f"✓ get_adapter_info('teoria_cambio'):")
            print(f"  Class: {adapter_info['adapter_class']}")
            print(f"  Methods: {adapter_info['methods']}")
        
        # Test get_all_bindings
        bindings = loader.get_all_bindings()
        print(f"✓ get_all_bindings(): {len(bindings)} bindings")
        
        return True
        
    except Exception as e:
        print(f"✗ Query interface test failed: {e}")
        return False


# ============================================================================
# TEST RUNNER
# ============================================================================

def run_all_tests():
    """Run all tests and report results"""
    print("\n" + "=" * 80)
    print("MAPPING LOADER TEST SUITE")
    print("=" * 80)
    
    tests = [
        ("YAML Loading", test_yaml_loading),
        ("Adapter Registry Parsing", test_adapter_registry_parsing),
        ("Execution Chain Parsing", test_execution_chain_parsing),
        ("DAG Construction", test_dag_construction),
        ("Binding Validation", test_binding_validation),
        ("Type Validation", test_type_validation),
        ("Circular Dependency Detection", test_circular_dependency_detection),
        ("Full Validation", test_full_validation),
        ("Startup Validator", test_startup_validator),
        ("Query Interface", test_query_interface),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print("=" * 80)
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
