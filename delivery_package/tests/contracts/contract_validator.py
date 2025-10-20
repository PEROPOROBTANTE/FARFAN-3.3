"""
Contract Validator - Enforces YAML-based contract specifications.

Validates:
1. JSON Schema validation against input/output types
2. SHA-256 hash verification for deterministic methods
3. RNG seed parameter propagation
4. Binding compatibility (producer output -> consumer input)
5. Fail-fast with detailed diagnostics
"""
import yaml
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import jsonschema
from jsonschema import validate, ValidationError, Draft7Validator


@dataclass
class ContractViolation:
    """Represents a contract violation."""
    contract_file: str
    adapter: str
    method: str
    violation_type: str
    description: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of contract validation."""
    passed: bool
    violations: List[ContractViolation] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def add_violation(self, violation: ContractViolation):
        """Add a violation and mark as failed."""
        self.violations.append(violation)
        self.passed = False
    
    def add_warning(self, message: str):
        """Add a warning."""
        self.warnings.append(message)


class ContractRegistry:
    """Registry of all contract specifications."""
    
    def __init__(self):
        self.contracts: Dict[str, Dict[str, Any]] = {}
        self.adapter_methods: Dict[str, Set[str]] = defaultdict(set)
        self.method_lookup: Dict[str, str] = {}  # method_name -> adapter_name
        
    def load_contracts(self, contracts_dir: Path) -> int:
        """Load all contract YAML files from directory."""
        count = 0
        
        for contract_file in sorted(contracts_dir.glob("*.yaml")):
            try:
                with open(contract_file, 'r', encoding='utf-8') as f:
                    contract = yaml.safe_load(f)
                
                adapter = contract['adapter']
                method = contract['method']
                key = f"{adapter}.{method}"
                
                self.contracts[key] = {
                    'contract': contract,
                    'file': str(contract_file)
                }
                
                self.adapter_methods[adapter].add(method)
                self.method_lookup[method] = adapter
                
                count += 1
                
            except Exception as e:
                print(f"Warning: Failed to load {contract_file}: {e}")
        
        return count
    
    def get_contract(self, adapter: str, method: str) -> Optional[Dict[str, Any]]:
        """Get contract for specific adapter method."""
        key = f"{adapter}.{method}"
        return self.contracts.get(key, {}).get('contract')
    
    def get_all_contracts(self) -> List[Dict[str, Any]]:
        """Get all contracts."""
        return [v['contract'] for v in self.contracts.values()]
    
    def get_adapters(self) -> List[str]:
        """Get all registered adapters."""
        return sorted(self.adapter_methods.keys())
    
    def get_methods(self, adapter: str) -> List[str]:
        """Get all methods for an adapter."""
        return sorted(self.adapter_methods.get(adapter, []))


class ContractValidator:
    """
    Validates contracts and enforces specifications.
    
    Features:
    - JSON Schema validation for input/output
    - SHA-256 hash verification for deterministic methods
    - RNG seed parameter propagation checks
    - Binding compatibility validation
    - Fail-fast with detailed diagnostics
    """
    
    def __init__(self, contracts_dir: Path):
        self.contracts_dir = Path(contracts_dir)
        self.registry = ContractRegistry()
        self.result = ValidationResult(passed=True)
        
        # Load all contracts at startup
        count = self.registry.load_contracts(self.contracts_dir)
        print(f"Loaded {count} contracts from {self.contracts_dir}")
    
    def validate_all(self) -> ValidationResult:
        """Validate all contracts."""
        self.result = ValidationResult(passed=True)
        
        # 1. Validate schema definitions
        self._validate_schema_definitions()
        
        # 2. Validate deterministic properties
        self._validate_deterministic_properties()
        
        # 3. Validate RNG seed propagation
        self._validate_rng_seed_propagation()
        
        # 4. Validate binding compatibility
        self._validate_binding_compatibility()
        
        # 5. Validate latency constraints
        self._validate_latency_constraints()
        
        return self.result
    
    def validate_input(self, adapter: str, method: str, input_data: Dict[str, Any]) -> ValidationResult:
        """Validate input data against contract schema."""
        result = ValidationResult(passed=True)
        contract = self.registry.get_contract(adapter, method)
        
        if not contract:
            result.add_violation(ContractViolation(
                contract_file="N/A",
                adapter=adapter,
                method=method,
                violation_type="CONTRACT_NOT_FOUND",
                description=f"No contract found for {adapter}.{method}"
            ))
            return result
        
        input_schema = contract.get('input_schema', {})
        
        try:
            validate(instance=input_data, schema=input_schema)
        except ValidationError as e:
            result.add_violation(ContractViolation(
                contract_file=self.registry.contracts[f"{adapter}.{method}"]['file'],
                adapter=adapter,
                method=method,
                violation_type="INPUT_SCHEMA_VIOLATION",
                description=f"Input validation failed: {e.message}",
                details={
                    "schema_path": list(e.schema_path),
                    "validator": e.validator,
                    "failed_value": str(e.instance)[:100]
                }
            ))
        
        # Check required parameters
        required = input_schema.get('required', [])
        missing = set(required) - set(input_data.keys())
        if missing:
            result.add_violation(ContractViolation(
                contract_file=self.registry.contracts[f"{adapter}.{method}"]['file'],
                adapter=adapter,
                method=method,
                violation_type="MISSING_REQUIRED_PARAMETERS",
                description=f"Missing required parameters: {missing}",
                details={"missing_keys": list(missing)}
            ))
        
        return result
    
    def validate_output(self, adapter: str, method: str, output_data: Any) -> ValidationResult:
        """Validate output data against contract schema."""
        result = ValidationResult(passed=True)
        contract = self.registry.get_contract(adapter, method)
        
        if not contract:
            result.add_violation(ContractViolation(
                contract_file="N/A",
                adapter=adapter,
                method=method,
                violation_type="CONTRACT_NOT_FOUND",
                description=f"No contract found for {adapter}.{method}"
            ))
            return result
        
        output_schema = contract.get('output_schema', {})
        
        try:
            validate(instance=output_data, schema=output_schema)
        except ValidationError as e:
            result.add_violation(ContractViolation(
                contract_file=self.registry.contracts[f"{adapter}.{method}"]['file'],
                adapter=adapter,
                method=method,
                violation_type="OUTPUT_SCHEMA_VIOLATION",
                description=f"Output validation failed: {e.message}",
                details={
                    "schema_path": list(e.schema_path),
                    "validator": e.validator,
                    "failed_value": str(e.instance)[:100]
                }
            ))
        
        return result
    
    def verify_deterministic_hash(self, adapter: str, method: str, output_data: Any, 
                                  canonical_input: Dict[str, Any]) -> ValidationResult:
        """Verify SHA-256 hash for deterministic method output."""
        result = ValidationResult(passed=True)
        contract = self.registry.get_contract(adapter, method)
        
        if not contract:
            return result
        
        if not contract.get('deterministic', False):
            return result  # Non-deterministic methods don't need hash verification
        
        # Compute hash of output
        output_json = json.dumps(output_data, sort_keys=True)
        computed_hash = hashlib.sha256(output_json.encode()).hexdigest()
        expected_hash = contract.get('sample_hash')
        
        if computed_hash != expected_hash:
            result.add_violation(ContractViolation(
                contract_file=self.registry.contracts[f"{adapter}.{method}"]['file'],
                adapter=adapter,
                method=method,
                violation_type="DETERMINISTIC_HASH_MISMATCH",
                description=f"Output hash mismatch for deterministic method",
                details={
                    "expected_hash": expected_hash,
                    "computed_hash": computed_hash,
                    "canonical_input": canonical_input
                }
            ))
        
        return result
    
    def _validate_schema_definitions(self):
        """Validate all schema definitions are valid JSON Schema."""
        for contract_data in self.registry.contracts.values():
            contract = contract_data['contract']
            adapter = contract['adapter']
            method = contract['method']
            file_path = contract_data['file']
            
            # Validate input schema
            input_schema = contract.get('input_schema', {})
            try:
                Draft7Validator.check_schema(input_schema)
            except Exception as e:
                self.result.add_violation(ContractViolation(
                    contract_file=file_path,
                    adapter=adapter,
                    method=method,
                    violation_type="INVALID_INPUT_SCHEMA",
                    description=f"Invalid input schema: {str(e)}",
                    details={"schema": input_schema}
                ))
            
            # Validate output schema
            output_schema = contract.get('output_schema', {})
            try:
                Draft7Validator.check_schema(output_schema)
            except Exception as e:
                self.result.add_violation(ContractViolation(
                    contract_file=file_path,
                    adapter=adapter,
                    method=method,
                    violation_type="INVALID_OUTPUT_SCHEMA",
                    description=f"Invalid output schema: {str(e)}",
                    details={"schema": output_schema}
                ))
    
    def _validate_deterministic_properties(self):
        """Validate deterministic properties and hash requirements."""
        for contract_data in self.registry.contracts.values():
            contract = contract_data['contract']
            adapter = contract['adapter']
            method = contract['method']
            file_path = contract_data['file']
            
            deterministic = contract.get('deterministic', False)
            sample_hash = contract.get('sample_hash')
            
            if deterministic and not sample_hash:
                self.result.add_violation(ContractViolation(
                    contract_file=file_path,
                    adapter=adapter,
                    method=method,
                    violation_type="MISSING_SAMPLE_HASH",
                    description="Deterministic method missing sample_hash",
                    details={"deterministic": True}
                ))
            
            if not deterministic and contract.get('rng_seed_param'):
                # Verify retry policy allows retries
                retry_policy = contract.get('retry_policy', {})
                if retry_policy.get('max_retries', 0) == 0:
                    self.result.add_warning(
                        f"{adapter}.{method}: Non-deterministic method with RNG seed "
                        f"should allow retries"
                    )
    
    def _validate_rng_seed_propagation(self):
        """Validate RNG seed parameters are correctly specified."""
        for contract_data in self.registry.contracts.values():
            contract = contract_data['contract']
            adapter = contract['adapter']
            method = contract['method']
            file_path = contract_data['file']
            
            rng_seed_param = contract.get('rng_seed_param')
            deterministic = contract.get('deterministic', False)
            
            # If deterministic, should not have RNG seed
            if deterministic and rng_seed_param:
                self.result.add_violation(ContractViolation(
                    contract_file=file_path,
                    adapter=adapter,
                    method=method,
                    violation_type="INVALID_RNG_SEED_DETERMINISTIC",
                    description="Deterministic method should not have rng_seed_param",
                    details={
                        "deterministic": True,
                        "rng_seed_param": rng_seed_param
                    }
                ))
            
            # If has RNG seed, verify it's in input schema
            if rng_seed_param:
                input_schema = contract.get('input_schema', {})
                properties = input_schema.get('properties', {})
                
                if rng_seed_param not in properties:
                    self.result.add_violation(ContractViolation(
                        contract_file=file_path,
                        adapter=adapter,
                        method=method,
                        violation_type="RNG_SEED_NOT_IN_SCHEMA",
                        description=f"rng_seed_param '{rng_seed_param}' not in input schema",
                        details={
                            "rng_seed_param": rng_seed_param,
                            "available_params": list(properties.keys())
                        }
                    ))
    
    def _validate_binding_compatibility(self):
        """Validate producer output schemas match consumer input schemas."""
        # Build a compatibility graph
        type_producers = defaultdict(list)  # type -> list of (adapter, method)
        type_consumers = defaultdict(list)  # type -> list of (adapter, method)
        
        for contract_data in self.registry.contracts.values():
            contract = contract_data['contract']
            adapter = contract['adapter']
            method = contract['method']
            
            output_type = contract.get('output_schema', {}).get('type', 'any')
            type_producers[output_type].append((adapter, method))
            
            input_schema = contract.get('input_schema', {})
            for param, param_schema in input_schema.get('properties', {}).items():
                param_type = param_schema.get('type', 'any')
                type_consumers[param_type].append((adapter, method, param))
        
        # Check for orphaned consumers (types with consumers but no producers)
        all_producer_types = set(type_producers.keys())
        all_consumer_types = set(type_consumers.keys())
        
        orphaned_types = all_consumer_types - all_producer_types - {'string', 'integer', 'number', 'boolean', 'any'}
        
        if orphaned_types:
            for orphaned_type in orphaned_types:
                consumers = type_consumers[orphaned_type]
                self.result.add_warning(
                    f"Type '{orphaned_type}' has {len(consumers)} consumer(s) but no producers. "
                    f"First consumer: {consumers[0][0]}.{consumers[0][1]} (param: {consumers[0][2]})"
                )
    
    def _validate_latency_constraints(self):
        """Validate latency constraints are reasonable."""
        for contract_data in self.registry.contracts.values():
            contract = contract_data['contract']
            adapter = contract['adapter']
            method = contract['method']
            file_path = contract_data['file']
            
            max_latency = contract.get('max_latency_ms', 0)
            
            if max_latency <= 0:
                self.result.add_violation(ContractViolation(
                    contract_file=file_path,
                    adapter=adapter,
                    method=method,
                    violation_type="INVALID_LATENCY_CONSTRAINT",
                    description=f"max_latency_ms must be positive, got {max_latency}",
                    details={"max_latency_ms": max_latency}
                ))
            
            if max_latency > 300000:  # 5 minutes
                self.result.add_warning(
                    f"{adapter}.{method}: max_latency_ms very high ({max_latency}ms)"
                )
    
    def print_report(self, result: Optional[ValidationResult] = None):
        """Print detailed validation report."""
        if result is None:
            result = self.result
        
        print("\n" + "="*80)
        print("CONTRACT VALIDATION REPORT")
        print("="*80)
        
        print(f"\nTotal Contracts: {len(self.registry.contracts)}")
        print(f"Total Adapters: {len(self.registry.get_adapters())}")
        
        if result.passed:
            print("\n✓ ALL CONTRACTS PASSED VALIDATION")
        else:
            print(f"\n✗ VALIDATION FAILED: {len(result.violations)} violations")
        
        if result.warnings:
            print(f"\n⚠ {len(result.warnings)} warnings")
            for warning in result.warnings[:10]:
                print(f"  • {warning}")
            if len(result.warnings) > 10:
                print(f"  ... and {len(result.warnings) - 10} more warnings")
        
        if result.violations:
            print("\n" + "-"*80)
            print("VIOLATIONS:")
            print("-"*80)
            
            # Group violations by type
            by_type = defaultdict(list)
            for v in result.violations:
                by_type[v.violation_type].append(v)
            
            for vtype, violations in sorted(by_type.items()):
                print(f"\n{vtype}: {len(violations)} violation(s)")
                for v in violations[:5]:
                    print(f"\n  Contract: {v.contract_file}")
                    print(f"  Adapter:  {v.adapter}")
                    print(f"  Method:   {v.method}")
                    print(f"  Details:  {v.description}")
                    if v.details:
                        for key, value in list(v.details.items())[:3]:
                            print(f"    - {key}: {value}")
                
                if len(violations) > 5:
                    print(f"\n  ... and {len(violations) - 5} more {vtype} violations")
        
        print("\n" + "="*80)
        print(f"RESULT: {'PASSED' if result.passed else 'FAILED'}")
        print("="*80 + "\n")


def main():
    """Run contract validation."""
    contracts_dir = Path('tests/contracts')
    
    print("Initializing Contract Validator...")
    validator = ContractValidator(contracts_dir)
    
    print("\nRunning full contract validation...")
    result = validator.validate_all()
    
    validator.print_report(result)
    
    return 0 if result.passed else 1


if __name__ == '__main__':
    exit(main())
