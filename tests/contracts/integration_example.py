"""
Integration Example: Using Contract Validator with Module Adapters.

Demonstrates:
1. Runtime validation of inputs/outputs
2. Binding compatibility checks
3. Fail-fast with detailed diagnostics
"""
import sys
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.contracts.contract_validator import ContractValidator


class ContractEnforcedAdapter:
    """
    Wrapper that enforces contracts on adapter method calls.
    
    This demonstrates how to integrate contract validation into
    the adapter execution pipeline.
    """
    
    def __init__(self, adapter_name: str, validator: ContractValidator):
        self.adapter_name = adapter_name
        self.validator = validator
    
    def execute_with_validation(self, method_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute adapter method with full contract enforcement.
        
        Returns:
            {
                'success': bool,
                'output': Any,  # Only if success
                'violations': List[ContractViolation],  # Only if failed
                'execution_time_ms': float
            }
        """
        import time
        start = time.time()
        
        # 1. Validate input against contract
        input_result = self.validator.validate_input(self.adapter_name, method_name, input_data)
        
        if not input_result.passed:
            return {
                'success': False,
                'violations': input_result.violations,
                'execution_time_ms': (time.time() - start) * 1000
            }
        
        # 2. Execute the actual method (simulated here)
        try:
            output = self._execute_method(method_name, input_data)
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'execution_time_ms': (time.time() - start) * 1000
            }
        
        # 3. Validate output against contract
        output_result = self.validator.validate_output(self.adapter_name, method_name, output)
        
        if not output_result.passed:
            return {
                'success': False,
                'violations': output_result.violations,
                'execution_time_ms': (time.time() - start) * 1000
            }
        
        # 4. For deterministic methods, verify hash
        contract = self.validator.registry.get_contract(self.adapter_name, method_name)
        if contract and contract.get('deterministic'):
            canonical_input = contract.get('canonical_canary', {})
            hash_result = self.validator.verify_deterministic_hash(
                self.adapter_name, method_name, output, canonical_input
            )
            
            if not hash_result.passed:
                return {
                    'success': False,
                    'violations': hash_result.violations,
                    'execution_time_ms': (time.time() - start) * 1000
                }
        
        return {
            'success': True,
            'output': output,
            'execution_time_ms': (time.time() - start) * 1000
        }
    
    def _execute_method(self, method_name: str, input_data: Dict[str, Any]) -> Any:
        """Simulate method execution (replace with actual adapter call)."""
        # This is where you would call the actual adapter method
        # For demonstration, return mock data
        
        if method_name == 'validate':
            return True
        elif method_name == 'process':
            return {
                'status': 'success',
                'results': [],
                'confidence': 0.85
            }
        else:
            return {'status': 'success', 'data': {}}


def demonstrate_contract_enforcement():
    """Demonstrate contract enforcement in action."""
    
    print("="*80)
    print("CONTRACT ENFORCEMENT DEMONSTRATION")
    print("="*80)
    
    # Initialize validator
    contracts_dir = Path('tests/contracts')
    validator = ContractValidator(contracts_dir)
    
    # Create contract-enforced adapter
    adapter = ContractEnforcedAdapter('PolicyProcessorAdapter', validator)
    
    # Test 1: Valid input
    print("\nTest 1: Valid Input")
    print("-" * 80)
    result = adapter.execute_with_validation(
        'validate',
        {'config': {'min_confidence': 0.5}}
    )
    print(f"Success: {result['success']}")
    print(f"Output: {result.get('output')}")
    print(f"Execution Time: {result['execution_time_ms']:.2f}ms")
    
    # Test 2: Missing required field
    print("\nTest 2: Missing Required Field")
    print("-" * 80)
    result = adapter.execute_with_validation(
        'validate',
        {}  # Missing 'config'
    )
    print(f"Success: {result['success']}")
    if not result['success']:
        print(f"Violations: {len(result['violations'])}")
        for v in result['violations']:
            print(f"  - {v.violation_type}: {v.description}")
    
    # Test 3: Wrong type
    print("\nTest 3: Wrong Type")
    print("-" * 80)
    result = adapter.execute_with_validation(
        'validate',
        {'config': 'invalid_string'}  # Should be object
    )
    print(f"Success: {result['success']}")
    if not result['success']:
        print(f"Violations: {len(result['violations'])}")
        for v in result['violations']:
            print(f"  - {v.violation_type}: {v.description}")
    
    # Test 4: Check binding compatibility
    print("\nTest 4: Binding Compatibility Analysis")
    print("-" * 80)
    
    # Show which methods can consume outputs from 'process'
    process_contract = validator.registry.get_contract('PolicyProcessorAdapter', 'process')
    if process_contract:
        output_type = process_contract['output_schema']['type']
        print(f"PolicyProcessorAdapter.process returns: {output_type}")
        
        # Find methods that accept this type
        compatible_consumers = []
        for contract_key in validator.registry.contracts:
            contract_data = validator.registry.contracts[contract_key]
            contract = contract_data['contract']
            
            input_schema = contract.get('input_schema', {})
            for param, param_schema in input_schema.get('properties', {}).items():
                if param_schema.get('type') == output_type:
                    compatible_consumers.append(
                        f"{contract['adapter']}.{contract['method']} (param: {param})"
                    )
        
        print(f"\nFound {len(compatible_consumers)} compatible consumers")
        for consumer in compatible_consumers[:10]:
            print(f"  - {consumer}")
        if len(compatible_consumers) > 10:
            print(f"  ... and {len(compatible_consumers) - 10} more")
    
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80)


def demonstrate_fail_fast_diagnostics():
    """Demonstrate fail-fast with detailed diagnostics."""
    
    print("\n" + "="*80)
    print("FAIL-FAST DIAGNOSTICS DEMONSTRATION")
    print("="*80)
    
    contracts_dir = Path('tests/contracts')
    validator = ContractValidator(contracts_dir)
    
    # Simulate a complex validation scenario
    print("\nScenario: Validating a pipeline of method calls")
    print("-" * 80)
    
    pipeline = [
        ('PolicyProcessorAdapter', 'process', {'text': 'Sample policy document'}),
        ('PolicySegmenterAdapter', 'segment', {'text': 'Sample policy document'}),
        ('EmbeddingPolicyAdapter', 'semantic_search', {'query': 'sustainability', 'top_k': 10, 'filters': []})
    ]
    
    for adapter, method, inputs in pipeline:
        result = validator.validate_input(adapter, method, inputs)
        
        print(f"\n{adapter}.{method}")
        if result.passed:
            print("  ✓ Input validation passed")
        else:
            print(f"  ✗ Input validation failed with {len(result.violations)} violations")
            for v in result.violations:
                print(f"    • {v.violation_type}")
                print(f"      {v.description}")
                if v.details:
                    for key, value in list(v.details.items())[:2]:
                        print(f"      - {key}: {value}")
            print("  ⚠ FAIL-FAST: Stopping pipeline execution")
            break
    
    print("\n" + "="*80)


if __name__ == '__main__':
    demonstrate_contract_enforcement()
    demonstrate_fail_fast_diagnostics()
