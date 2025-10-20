#!/usr/bin/env python3
"""
FARFAN 3.0 - Execution Mapping Validation Audit
================================================

Validates execution_mapping.yaml against module_adapters.py and questionnaire_parser.py:
- Checks all adapter registry entries are implemented
- Validates dimension sections and question execution chains
- Verifies all referenced adapters exist with specified methods
- Validates binding types between producer outputs and consumer inputs
- Detects circular dependencies
- Identifies orphaned execution chains

Outputs: execution_mapping_validation.json
"""

import json
import yaml
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Any, Optional
from collections import defaultdict
from dataclasses import dataclass, field, asdict
import ast

# =================================================================
# DATA STRUCTURES
# =================================================================

@dataclass
class AdapterMethodInfo:
    """Information about an adapter method"""
    adapter_name: str
    class_name: str
    method_name: str
    is_private: bool
    parameters: List[str]
    exists_in_code: bool = False
    line_number: Optional[int] = None

@dataclass
class BindingInfo:
    """Information about a binding"""
    name: str
    producer_type: str
    producer_adapter: str
    producer_method: str
    producer_step: int
    consumers: List[Dict] = field(default_factory=list)

@dataclass
class ValidationIssue:
    """Validation issue found"""
    severity: str  # 'error', 'warning', 'info'
    category: str
    question_id: str
    step: Optional[int]
    adapter: Optional[str]
    method: Optional[str]
    binding: Optional[str]
    description: str
    remediation: str

@dataclass
class ValidationReport:
    """Complete validation report"""
    summary: Dict[str, Any]
    adapters_found: Dict[str, Any]
    missing_adapters: List[Dict]
    missing_methods: List[Dict]
    type_mismatches: List[Dict]
    circular_dependencies: List[Dict]
    orphaned_chains: List[Dict]
    binding_issues: List[Dict]
    all_issues: List[ValidationIssue]

# =================================================================
# MAIN VALIDATOR
# =================================================================

class ExecutionMappingValidator:
    """Comprehensive validator for execution mapping integrity"""
    
    def __init__(self):
        self.base_path = Path.cwd()
        self.mapping_path = self.base_path / "orchestrator" / "execution_mapping.yaml"
        self.adapters_path = self.base_path / "orchestrator" / "module_adapters.py"
        
        # Parsed data
        self.mapping_data = None
        self.adapter_registry = {}
        self.execution_chains = {}
        self.adapters_in_code = {}
        
        # Validation results
        self.issues: List[ValidationIssue] = []
        self.bindings_map: Dict[str, List[BindingInfo]] = defaultdict(list)
        
    def validate_all(self) -> ValidationReport:
        """Run complete validation"""
        print("=" * 80)
        print("FARFAN 3.0 - Execution Mapping Validation Audit")
        print("=" * 80)
        
        # Step 1: Load and parse YAML
        print("\n[1/7] Loading execution_mapping.yaml...")
        self._load_yaml()
        
        # Step 2: Parse adapter registry from YAML
        print("[2/7] Parsing adapter registry...")
        self._parse_adapter_registry()
        
        # Step 3: Parse execution chains
        print("[3/7] Parsing execution chains...")
        self._parse_execution_chains()
        
        # Step 4: Load module_adapters.py
        print("[4/7] Analyzing module_adapters.py...")
        self._analyze_adapter_code()
        
        # Step 5: Validate adapters exist
        print("[5/7] Validating adapter existence...")
        self._validate_adapters_exist()
        
        # Step 6: Validate bindings and types
        print("[6/7] Validating bindings and types...")
        self._validate_bindings_and_types()
        
        # Step 7: Detect circular dependencies
        print("[7/7] Detecting circular dependencies...")
        self._detect_circular_dependencies()
        
        # Generate report
        print("\n" + "=" * 80)
        print("Generating validation report...")
        return self._generate_report()
    
    def _load_yaml(self):
        """Load YAML file"""
        with open(self.mapping_path, 'r', encoding='utf-8') as f:
            self.mapping_data = yaml.safe_load(f)
        print(f"  ✓ Loaded {self.mapping_path}")
        print(f"    Version: {self.mapping_data.get('version')}")
        print(f"    Adapters: {self.mapping_data.get('total_adapters')}")
        print(f"    Methods: {self.mapping_data.get('total_methods')}")
    
    def _parse_adapter_registry(self):
        """Parse adapter registry from YAML"""
        adapters = self.mapping_data.get('adapters', {})
        for adapter_name, info in adapters.items():
            self.adapter_registry[adapter_name] = {
                'adapter_class': info.get('adapter_class'),
                'methods': info.get('methods', 0),
                'specialization': info.get('specialization', ''),
                'sub_adapters': info.get('sub_adapters', []),
                'status': info.get('status', 'complete')
            }
        print(f"  ✓ Parsed {len(self.adapter_registry)} adapter registrations")
    
    def _parse_execution_chains(self):
        """Parse all execution chains from dimensions"""
        dimension_keys = [k for k in self.mapping_data.keys() if k.startswith('D') and '_' in k]
        
        chain_count = 0
        for dim_key in dimension_keys:
            dim_data = self.mapping_data[dim_key]
            if not isinstance(dim_data, dict):
                continue
            
            for q_key, q_data in dim_data.items():
                if not q_key.startswith('Q') or not isinstance(q_data, dict):
                    continue
                
                execution_chain = q_data.get('execution_chain', [])
                if execution_chain:
                    question_id = f"{dim_key}.{q_key}"
                    self.execution_chains[question_id] = {
                        'description': q_data.get('description', ''),
                        'steps': execution_chain,
                        'aggregation': q_data.get('aggregation', {})
                    }
                    chain_count += 1
        
        print(f"  ✓ Parsed {chain_count} execution chains")
    
    def _analyze_adapter_code(self):
        """Analyze module_adapters.py to extract actual implementations"""
        with open(self.adapters_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse AST
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            print(f"  ✗ Failed to parse module_adapters.py: {e}")
            return
        
        # Extract classes
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_name = node.name
                
                # Check if it's an adapter class
                if class_name.endswith('Adapter'):
                    adapter_methods = []
                    
                    # Extract methods
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            method_name = item.name
                            params = [arg.arg for arg in item.args.args if arg.arg != 'self']
                            
                            adapter_methods.append(AdapterMethodInfo(
                                adapter_name=self._class_to_adapter_name(class_name),
                                class_name=class_name,
                                method_name=method_name,
                                is_private=method_name.startswith('_'),
                                parameters=params,
                                exists_in_code=True,
                                line_number=item.lineno
                            ))
                    
                    adapter_name = self._class_to_adapter_name(class_name)
                    self.adapters_in_code[adapter_name] = {
                        'class_name': class_name,
                        'methods': adapter_methods,
                        'method_count': len([m for m in adapter_methods if not m.method_name.startswith('__')])
                    }
        
        print(f"  ✓ Found {len(self.adapters_in_code)} adapter classes in code")
        for name, info in self.adapters_in_code.items():
            print(f"    - {name}: {info['method_count']} methods")
    
    def _class_to_adapter_name(self, class_name: str) -> str:
        """Convert class name to adapter name"""
        mapping = {
            'ModulosAdapter': 'teoria_cambio',
            'AnalyzerOneAdapter': 'analyzer_one',
            'DerekBeachAdapter': 'dereck_beach',
            'EmbeddingPolicyAdapter': 'embedding_policy',
            'SemanticChunkingPolicyAdapter': 'semantic_chunking_policy',
            'ContradictionDetectionAdapter': 'contradiction_detection',
            'FinancialViabilityAdapter': 'financial_viability',
            'PolicyProcessorAdapter': 'policy_processor',
            'PolicySegmenterAdapter': 'policy_segmenter'
        }
        return mapping.get(class_name, class_name.lower().replace('adapter', ''))
    
    def _validate_adapters_exist(self):
        """Validate all referenced adapters exist in code"""
        # Check adapters from registry
        for adapter_name, info in self.adapter_registry.items():
            if adapter_name not in self.adapters_in_code:
                self.issues.append(ValidationIssue(
                    severity='error',
                    category='missing_adapter',
                    question_id='REGISTRY',
                    step=None,
                    adapter=adapter_name,
                    method=None,
                    binding=None,
                    description=f"Adapter '{adapter_name}' declared in registry but not found in module_adapters.py",
                    remediation=f"Implement {info['adapter_class']} in module_adapters.py"
                ))
        
        # Check adapters from execution chains
        for question_id, chain in self.execution_chains.items():
            for step in chain['steps']:
                adapter = step.get('adapter')
                method = step.get('method')
                step_num = step.get('step')
                
                if adapter not in self.adapters_in_code:
                    self.issues.append(ValidationIssue(
                        severity='error',
                        category='missing_adapter',
                        question_id=question_id,
                        step=step_num,
                        adapter=adapter,
                        method=method,
                        binding=None,
                        description=f"Adapter '{adapter}' used but not found in module_adapters.py",
                        remediation=f"Implement adapter '{adapter}' in module_adapters.py"
                    ))
                else:
                    # Check if method exists
                    adapter_info = self.adapters_in_code[adapter]
                    method_names = [m.method_name for m in adapter_info['methods']]
                    
                    if method not in method_names:
                        self.issues.append(ValidationIssue(
                            severity='error',
                            category='missing_method',
                            question_id=question_id,
                            step=step_num,
                            adapter=adapter,
                            method=method,
                            binding=None,
                            description=f"Method '{adapter}.{method}' not found in module_adapters.py",
                            remediation=f"Implement method '{method}' in {adapter_info['class_name']}"
                        ))
    
    def _validate_bindings_and_types(self):
        """Validate bindings and type compatibility"""
        for question_id, chain in self.execution_chains.items():
            steps = chain['steps']
            
            # Build binding map for this chain
            local_bindings: Dict[str, BindingInfo] = {}
            
            # First pass: collect producers
            for step in steps:
                returns = step.get('returns', {})
                binding_name = returns.get('binding')
                
                if binding_name:
                    if binding_name in local_bindings:
                        # Duplicate producer
                        self.issues.append(ValidationIssue(
                            severity='error',
                            category='duplicate_producer',
                            question_id=question_id,
                            step=step.get('step'),
                            adapter=step.get('adapter'),
                            method=step.get('method'),
                            binding=binding_name,
                            description=f"Binding '{binding_name}' has multiple producers",
                            remediation=f"Use unique binding names or merge steps"
                        ))
                    else:
                        local_bindings[binding_name] = BindingInfo(
                            name=binding_name,
                            producer_type=returns.get('type', 'Any'),
                            producer_adapter=step.get('adapter'),
                            producer_method=step.get('method'),
                            producer_step=step.get('step'),
                            consumers=[]
                        )
            
            # Second pass: check consumers
            for step in steps:
                args = step.get('args', [])
                step_num = step.get('step')
                
                for arg in args:
                    if isinstance(arg, dict):
                        source = arg.get('source')
                        expected_type = arg.get('type')
                        
                        # Skip special sources (inputs from outside the chain)
                        special_sources = ['plan_text', 'normalized_text', 'entity_name',
                                         'extracted_tables', 'extracted_indicators',
                                         'institutional_mechanism', 'prior_assessments',
                                         'capacity_evidence', 'temporal_statements',
                                         'process_indicators', 'process_sequence',
                                         'process_mechanism', 'financial_data',
                                         'process_pillar', 'document_segments',
                                         'semantic_chunks']
                        
                        if source and source not in special_sources:
                            if source not in local_bindings:
                                # Missing producer
                                self.issues.append(ValidationIssue(
                                    severity='error',
                                    category='missing_producer',
                                    question_id=question_id,
                                    step=step_num,
                                    adapter=step.get('adapter'),
                                    method=step.get('method'),
                                    binding=source,
                                    description=f"Step {step_num} references undefined binding '{source}'",
                                    remediation=f"Add producer step for '{source}' before step {step_num}"
                                ))
                            else:
                                # Type checking
                                binding_info = local_bindings[source]
                                binding_info.consumers.append({
                                    'step': step_num,
                                    'adapter': step.get('adapter'),
                                    'method': step.get('method'),
                                    'expected_type': expected_type
                                })
                                
                                if expected_type and binding_info.producer_type != 'Any':
                                    if not self._types_compatible(binding_info.producer_type, expected_type):
                                        self.issues.append(ValidationIssue(
                                            severity='warning',
                                            category='type_mismatch',
                                            question_id=question_id,
                                            step=step_num,
                                            adapter=step.get('adapter'),
                                            method=step.get('method'),
                                            binding=source,
                                            description=f"Type mismatch: binding '{source}' produces '{binding_info.producer_type}' but step {step_num} expects '{expected_type}'",
                                            remediation=f"Add type conversion or fix producer/consumer types"
                                        ))
            
            # Store bindings for this question
            self.bindings_map[question_id] = list(local_bindings.values())
    
    def _types_compatible(self, producer_type: str, consumer_type: str) -> bool:
        """Check if types are compatible"""
        # Exact match
        if producer_type == consumer_type:
            return True
        
        # Any matches everything
        if producer_type == 'Any' or consumer_type == 'Any':
            return True
        
        # List compatibility
        if producer_type.startswith('List') and consumer_type.startswith('List'):
            return True
        
        # Dict compatibility
        if producer_type.startswith('Dict') and consumer_type.startswith('Dict'):
            return True
        
        # str/string compatibility
        if producer_type in ['str', 'string'] and consumer_type in ['str', 'string']:
            return True
        
        # int/float compatibility (int can be used as float)
        if producer_type == 'int' and consumer_type == 'float':
            return True
        
        return False
    
    def _detect_circular_dependencies(self):
        """Detect circular dependencies in execution chains"""
        for question_id, chain in self.execution_chains.items():
            # Build dependency graph
            bindings = self.bindings_map.get(question_id, [])
            binding_map = {b.name: b for b in bindings}
            
            for step in chain['steps']:
                step_num = step.get('step')
                args = step.get('args', [])
                
                for arg in args:
                    if isinstance(arg, dict):
                        source = arg.get('source')
                        
                        if source and source in binding_map:
                            producer_step = binding_map[source].producer_step
                            if producer_step >= step_num:
                                # Circular or backward dependency
                                self.issues.append(ValidationIssue(
                                    severity='error',
                                    category='circular_dependency',
                                    question_id=question_id,
                                    step=step_num,
                                    adapter=step.get('adapter'),
                                    method=step.get('method'),
                                    binding=source,
                                    description=f"Step {step_num} depends on binding '{source}' from step {producer_step} (circular/backward dependency)",
                                    remediation=f"Reorder steps or remove circular dependency"
                                ))
    
    def _generate_report(self) -> ValidationReport:
        """Generate comprehensive validation report"""
        # Categorize issues
        missing_adapters = []
        missing_methods = []
        type_mismatches = []
        circular_deps = []
        orphaned_chains = []
        binding_issues = []
        
        for issue in self.issues:
            issue_dict = {
                'severity': issue.severity,
                'question_id': issue.question_id,
                'step': issue.step,
                'adapter': issue.adapter,
                'method': issue.method,
                'binding': issue.binding,
                'description': issue.description,
                'remediation': issue.remediation
            }
            
            if issue.category == 'missing_adapter':
                missing_adapters.append(issue_dict)
            elif issue.category == 'missing_method':
                missing_methods.append(issue_dict)
            elif issue.category == 'type_mismatch':
                type_mismatches.append(issue_dict)
            elif issue.category == 'circular_dependency':
                circular_deps.append(issue_dict)
            elif issue.category in ['missing_producer', 'duplicate_producer']:
                binding_issues.append(issue_dict)
        
        # Check for orphaned chains
        for question_id, chain in self.execution_chains.items():
            has_valid_adapters = True
            for step in chain['steps']:
                adapter = step.get('adapter')
                if adapter not in self.adapters_in_code:
                    has_valid_adapters = False
                    break
            
            if not has_valid_adapters:
                orphaned_chains.append({
                    'question_id': question_id,
                    'description': chain['description'],
                    'reason': 'Contains references to missing adapters'
                })
        
        # Summary statistics
        error_count = len([i for i in self.issues if i.severity == 'error'])
        warning_count = len([i for i in self.issues if i.severity == 'warning'])
        
        summary = {
            'validation_status': 'FAILED' if error_count > 0 else 'PASSED',
            'total_issues': len(self.issues),
            'errors': error_count,
            'warnings': warning_count,
            'adapters': {
                'registered': len(self.adapter_registry),
                'found_in_code': len(self.adapters_in_code),
                'missing': len(missing_adapters)
            },
            'execution_chains': {
                'total': len(self.execution_chains),
                'orphaned': len(orphaned_chains)
            },
            'methods': {
                'total_declared': sum(info['methods'] for info in self.adapter_registry.values()),
                'total_found': sum(info['method_count'] for info in self.adapters_in_code.values()),
                'missing': len(missing_methods)
            },
            'bindings': {
                'total_questions_with_bindings': len(self.bindings_map),
                'type_mismatches': len(type_mismatches),
                'binding_issues': len(binding_issues)
            },
            'circular_dependencies': len(circular_deps)
        }
        
        # Adapters found details
        adapters_found = {}
        for adapter_name, info in self.adapters_in_code.items():
            registry_info = self.adapter_registry.get(adapter_name, {})
            adapters_found[adapter_name] = {
                'class_name': info['class_name'],
                'methods_found': info['method_count'],
                'methods_declared': registry_info.get('methods', 0),
                'status': registry_info.get('status', 'unknown'),
                'specialization': registry_info.get('specialization', ''),
                'method_list': [m.method_name for m in info['methods'] if not m.method_name.startswith('__')][:20]  # First 20
            }
        
        return ValidationReport(
            summary=summary,
            adapters_found=adapters_found,
            missing_adapters=missing_adapters,
            missing_methods=missing_methods,
            type_mismatches=type_mismatches,
            circular_dependencies=circular_deps,
            orphaned_chains=orphaned_chains,
            binding_issues=binding_issues,
            all_issues=self.issues
        )

# =================================================================
# MAIN
# =================================================================

if __name__ == "__main__":
    validator = ExecutionMappingValidator()
    report = validator.validate_all()
    
    # Convert to dict for JSON serialization
    report_dict = {
        'summary': report.summary,
        'adapters_found': report.adapters_found,
        'missing_adapters': report.missing_adapters,
        'missing_methods': report.missing_methods,
        'type_mismatches': report.type_mismatches,
        'circular_dependencies': report.circular_dependencies,
        'orphaned_chains': report.orphaned_chains,
        'binding_issues': report.binding_issues,
        'all_issues': [asdict(issue) for issue in report.all_issues]
    }
    
    # Write to file
    output_path = Path.cwd() / "execution_mapping_validation.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report_dict, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 80)
    print("VALIDATION REPORT SUMMARY")
    print("=" * 80)
    print(f"Status: {report.summary['validation_status']}")
    print(f"Total Issues: {report.summary['total_issues']} ({report.summary['errors']} errors, {report.summary['warnings']} warnings)")
    print(f"\nAdapters: {report.summary['adapters']['found_in_code']}/{report.summary['adapters']['registered']} found")
    print(f"Missing Adapters: {report.summary['adapters']['missing']}")
    print(f"Missing Methods: {report.summary['methods']['missing']}")
    print(f"Type Mismatches: {report.summary['bindings']['type_mismatches']}")
    print(f"Binding Issues: {report.summary['bindings']['binding_issues']}")
    print(f"Circular Dependencies: {report.summary['circular_dependencies']}")
    print(f"Orphaned Chains: {report.summary['execution_chains']['orphaned']}")
    print(f"\n✓ Report written to: {output_path}")
    print("=" * 80)
