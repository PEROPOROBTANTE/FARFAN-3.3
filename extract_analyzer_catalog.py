#!/usr/bin/env python3
"""
Extract class and method catalog from Analyzer_one.py and cross-reference with AnalyzerOneAdapter
"""

import re
import ast
import json
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple
from collections import defaultdict


def extract_class_methods(content: str) -> Dict[str, Dict[str, Any]]:
    """Extract all classes and their methods with decorators"""
    
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        print(f"Syntax error parsing file: {e}")
        return {}
    
    class_info = {}
    
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            class_name = node.name
            methods = {
                'staticmethods': [],
                'classmethods': [],
                'instance_methods': [],
                'constructors': [],
                'private_methods': [],
                'line_numbers': {},
                'parameters': {}
            }
            
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    method_name = item.name
                    lineno = item.lineno
                    
                    # Extract parameters
                    args = [arg.arg for arg in item.args.args]
                    methods['parameters'][method_name] = args
                    
                    # Check decorators
                    decorators = []
                    for d in item.decorator_list:
                        if isinstance(d, ast.Name):
                            decorators.append(d.id)
                        elif isinstance(d, ast.Attribute):
                            decorators.append(d.attr)
                        else:
                            decorators.append(str(type(d).__name__))
                    
                    # Categorize method
                    if 'staticmethod' in decorators:
                        methods['staticmethods'].append(method_name)
                    elif 'classmethod' in decorators:
                        methods['classmethods'].append(method_name)
                    elif method_name == '__init__':
                        methods['constructors'].append(method_name)
                    elif method_name.startswith('_') and not method_name.startswith('__'):
                        methods['private_methods'].append(method_name)
                    else:
                        methods['instance_methods'].append(method_name)
                    
                    methods['line_numbers'][method_name] = lineno
            
            class_info[class_name] = methods
    
    return class_info


def find_method_invocations(content: str, line_prefix_context: int = 200) -> List[Dict[str, Any]]:
    """Find all method invocations in the adapter content"""
    
    invocations = []
    lines = content.split('\n')
    
    # Patterns to match different invocation styles
    patterns = [
        # self.ClassName.method() - static/class method called via self
        (r'self\.([A-Z][a-zA-Z_0-9]*)\.([a-z_][a-zA-Z_0-9]*)\s*\(', 'static_or_class'),
        # self.ClassName().method() - instance method via instantiation
        (r'self\.([A-Z][a-zA-Z_0-9]*)\(\)\.([a-z_][a-zA-Z_0-9]*)\s*\(', 'instance_via_init'),
        # ClassName.method() - direct static/class method call
        (r'^(?!.*self\.).*?([A-Z][a-zA-Z_0-9]*)\.([a-z_][a-zA-Z_0-9]*)\s*\(', 'static_or_class_direct'),
        # ClassName().method() - instance method via new instantiation
        (r'([A-Z][a-zA-Z_0-9]*)\(\)\.([a-z_][a-zA-Z_0-9]*)\s*\(', 'instance_new'),
        # variable.method() - instance method on variable
        (r'([a-z_][a-zA-Z_0-9]*)\.([a-z_][a-zA-Z_0-9]*)\s*\(', 'instance_var'),
    ]
    
    for line_no, line in enumerate(lines, 1):
        for pattern, call_type in patterns:
            matches = re.finditer(pattern, line)
            for match in matches:
                if call_type in ['instance_var']:
                    # Skip obvious built-in or common methods
                    method_name = match.group(2)
                    if method_name in ['append', 'extend', 'get', 'items', 'keys', 'values', 
                                       'split', 'join', 'strip', 'lower', 'upper', 'replace',
                                       'format', 'update', 'pop', 'insert', 'remove',
                                       'startswith', 'endswith', 'find', 'count']:
                        continue
                
                invocation = {
                    'line_number': line_no,
                    'line_content': line.strip(),
                    'call_type': call_type,
                    'groups': match.groups()
                }
                
                if len(match.groups()) >= 2:
                    invocation['class_name'] = match.group(1)
                    invocation['method_name'] = match.group(2)
                
                invocations.append(invocation)
    
    return invocations


def cross_reference_invocations(
    class_catalog: Dict[str, Dict[str, Any]], 
    invocations: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Cross-reference invocations against the class catalog to find violations"""
    
    violations = []
    
    for inv in invocations:
        class_name = inv.get('class_name', '')
        method_name = inv.get('method_name', '')
        
        # Skip if not referencing a cataloged class
        if class_name not in class_catalog:
            continue
        
        class_info = class_catalog[class_name]
        call_type = inv['call_type']
        
        # Check if method exists
        all_methods = (
            class_info['staticmethods'] + 
            class_info['classmethods'] + 
            class_info['instance_methods'] + 
            class_info['constructors'] + 
            class_info['private_methods']
        )
        
        if method_name not in all_methods:
            violations.append({
                'type': 'non_existent_method',
                'severity': 'high',
                'line_number': inv['line_number'],
                'line_content': inv['line_content'],
                'class_name': class_name,
                'method_name': method_name,
                'description': f"Method '{method_name}' does not exist in class '{class_name}'",
                'recommendation': f"Check method name spelling or class definition"
            })
            continue
        
        # Check for invocation pattern violations
        if method_name in class_info['staticmethods']:
            # Static method - should be called via ClassName.method() or self.ClassName.method()
            if call_type in ['instance_via_init', 'instance_new', 'instance_var']:
                violations.append({
                    'type': 'incorrect_static_invocation',
                    'severity': 'medium',
                    'line_number': inv['line_number'],
                    'line_content': inv['line_content'],
                    'class_name': class_name,
                    'method_name': method_name,
                    'description': f"Static method '{method_name}' called as instance method",
                    'recommendation': f"Use {class_name}.{method_name}() instead of instantiation"
                })
        
        elif method_name in class_info['classmethods']:
            # Class method - should be called via ClassName.method() or self.ClassName.method()
            if call_type in ['instance_via_init', 'instance_new', 'instance_var']:
                violations.append({
                    'type': 'incorrect_classmethod_invocation',
                    'severity': 'medium',
                    'line_number': inv['line_number'],
                    'line_content': inv['line_content'],
                    'class_name': class_name,
                    'method_name': method_name,
                    'description': f"Class method '{method_name}' called as instance method",
                    'recommendation': f"Use {class_name}.{method_name}() instead of instantiation"
                })
        
        elif method_name in class_info['instance_methods']:
            # Instance method - needs instantiation
            if call_type in ['static_or_class', 'static_or_class_direct']:
                violations.append({
                    'type': 'missing_instantiation',
                    'severity': 'high',
                    'line_number': inv['line_number'],
                    'line_content': inv['line_content'],
                    'class_name': class_name,
                    'method_name': method_name,
                    'description': f"Instance method '{method_name}' called without instantiation",
                    'recommendation': f"Use {class_name}().{method_name}() or store instance in variable"
                })
    
    return violations


def generate_audit_report(
    class_catalog: Dict[str, Dict[str, Any]], 
    violations: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Generate comprehensive audit report"""
    
    # Calculate statistics
    total_classes = len(class_catalog)
    total_methods = 0
    for class_info in class_catalog.values():
        total_methods += (
            len(class_info['staticmethods']) +
            len(class_info['classmethods']) +
            len(class_info['instance_methods']) +
            len(class_info['constructors']) +
            len(class_info['private_methods'])
        )
    
    # Prepare method inventory
    method_inventory = {}
    for class_name, class_info in class_catalog.items():
        methods_by_type = {}
        
        for method_type in ['staticmethods', 'classmethods', 'instance_methods', 
                           'constructors', 'private_methods']:
            methods_list = []
            for method_name in class_info[method_type]:
                line_no = class_info['line_numbers'].get(method_name, 0)
                params = class_info['parameters'].get(method_name, [])
                
                # Determine correct invocation pattern
                if method_type == 'staticmethods':
                    invocation_pattern = f"{class_name}.{method_name}({', '.join(params[1:] if params and params[0] == 'self' else params)})"
                elif method_type == 'classmethods':
                    invocation_pattern = f"{class_name}.{method_name}({', '.join(params[1:] if params and params[0] == 'cls' else params)})"
                elif method_type == 'constructors':
                    invocation_pattern = f"{class_name}({', '.join(params[1:] if params and params[0] == 'self' else params)})"
                else:  # instance methods and private methods
                    instance_var = class_name.lower().replace('analyzer', 'analyzer').replace('engine', 'engine')
                    invocation_pattern = f"{instance_var} = {class_name}(); {instance_var}.{method_name}({', '.join(params[1:] if params and params[0] == 'self' else params)})"
                
                methods_list.append({
                    'name': method_name,
                    'line_number': line_no,
                    'parameters': params,
                    'correct_invocation_pattern': invocation_pattern
                })
            
            if methods_list:
                methods_by_type[method_type] = methods_list
        
        method_inventory[class_name] = methods_by_type
    
    # Prepare correction matrix
    correction_matrix = []
    for violation in violations:
        correction = {
            'violation_type': violation['type'],
            'severity': violation['severity'],
            'location': {
                'file': 'orchestrator/module_adapters.py',
                'line_number': violation['line_number'],
                'line_content': violation['line_content']
            },
            'issue': {
                'class_name': violation['class_name'],
                'method_name': violation['method_name'],
                'description': violation['description']
            },
            'recommendation': violation['recommendation']
        }
        correction_matrix.append(correction)
    
    # Sort violations by severity and line number
    correction_matrix.sort(key=lambda x: (
        0 if x['severity'] == 'high' else 1,
        x['location']['line_number']
    ))
    
    # Build final report
    report = {
        'metadata': {
            'generated_at': '2024-01-01T00:00:00',
            'analyzer_file': 'Analyzer_one.py',
            'adapter_file': 'orchestrator/module_adapters.py',
            'analysis_version': '1.0.0'
        },
        'statistics': {
            'total_classes': total_classes,
            'total_methods': total_methods,
            'total_violations': len(violations),
            'high_severity_violations': sum(1 for v in violations if v['severity'] == 'high'),
            'medium_severity_violations': sum(1 for v in violations if v['severity'] == 'medium'),
            'low_severity_violations': sum(1 for v in violations if v['severity'] == 'low')
        },
        'method_inventory': method_inventory,
        'correction_matrix': correction_matrix
    }
    
    return report


def main():
    """Main execution"""
    
    print("="*80)
    print("ANALYZER_ONE INVOCATION AUDIT")
    print("="*80)
    
    # Load files
    print("\n1. Loading Analyzer_one.py...")
    with open('Analyzer_one.py', 'r', encoding='utf-8') as f:
        analyzer_content = f.read()
    
    print("2. Loading orchestrator/module_adapters.py...")
    with open('orchestrator/module_adapters.py', 'r', encoding='utf-8') as f:
        adapter_content = f.read()
    
    # Extract class catalog
    print("\n3. Extracting class and method catalog from Analyzer_one.py...")
    class_catalog = extract_class_methods(analyzer_content)
    
    print(f"   Found {len(class_catalog)} classes:")
    for class_name, class_info in class_catalog.items():
        total_methods = sum(len(class_info[k]) for k in ['staticmethods', 'classmethods', 
                                                          'instance_methods', 'constructors', 
                                                          'private_methods'])
        print(f"   - {class_name}: {total_methods} methods")
    
    # Extract AnalyzerOneAdapter section
    print("\n4. Extracting AnalyzerOneAdapter section...")
    adapter_match = re.search(
        r'class AnalyzerOneAdapter.*?(?=\nclass [A-Z]|\Z)', 
        adapter_content, 
        re.DOTALL
    )
    
    if not adapter_match:
        print("   WARNING: AnalyzerOneAdapter not found in module_adapters.py")
        adapter_section = ""
    else:
        adapter_section = adapter_match.group(0)
        print(f"   Found AnalyzerOneAdapter ({len(adapter_section)} characters)")
    
    # Find invocations
    print("\n5. Analyzing method invocations in AnalyzerOneAdapter...")
    invocations = find_method_invocations(adapter_section)
    print(f"   Found {len(invocations)} potential method invocations")
    
    # Cross-reference
    print("\n6. Cross-referencing invocations against class catalog...")
    violations = cross_reference_invocations(class_catalog, invocations)
    print(f"   Found {len(violations)} invocation pattern violations")
    
    # Generate report
    print("\n7. Generating audit report...")
    audit_report = generate_audit_report(class_catalog, violations)
    
    # Save report
    output_file = 'analyzer_one_invocation_audit.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(audit_report, f, indent=2, ensure_ascii=False)
    
    print(f"\n8. Audit report saved to: {output_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("AUDIT SUMMARY")
    print("="*80)
    print(f"Total Classes:                    {audit_report['statistics']['total_classes']}")
    print(f"Total Methods:                    {audit_report['statistics']['total_methods']}")
    print(f"Total Violations Found:           {audit_report['statistics']['total_violations']}")
    print(f"  - High Severity:                {audit_report['statistics']['high_severity_violations']}")
    print(f"  - Medium Severity:              {audit_report['statistics']['medium_severity_violations']}")
    print(f"  - Low Severity:                 {audit_report['statistics']['low_severity_violations']}")
    
    if violations:
        print("\nTop 5 Violations:")
        for i, violation in enumerate(audit_report['correction_matrix'][:5], 1):
            print(f"\n{i}. [{violation['severity'].upper()}] Line {violation['location']['line_number']}")
            print(f"   Type: {violation['violation_type']}")
            print(f"   Issue: {violation['issue']['description']}")
            print(f"   Fix: {violation['recommendation']}")
    
    print("\n" + "="*80)
    print(f"Complete audit report: {output_file}")
    print("="*80)


if __name__ == '__main__':
    main()
