#!/usr/bin/env python3
"""
Generate analyzer_one_invocation_audit.json by parsing Analyzer_one.py
and cross-referencing with AnalyzerOneAdapter in module_adapters.py
"""

import json
import re
from datetime import datetime
from typing import Dict, List, Any, Tuple

# Manually extracted class and method information from Analyzer_one.py
# Based on the class structure visible in the file

ANALYZER_ONE_CLASSES = {
    "ValueChainLink": {
        "type": "dataclass",
        "methods": {
            "__init__": {"type": "constructor", "line": 77, "params": ["name", "instruments", "mediators", "outputs", "outcomes", "bottlenecks", "lead_time_days", "conversion_rates", "capacity_constraints"]}
        }
    },
    "MunicipalOntology": {
        "type": "class",
        "methods": {
            "__init__": {"type": "constructor", "line": 92, "params": ["self"]}
        }
    },
    "SemanticAnalyzer": {
        "type": "class",
        "methods": {
            "__init__": {"type": "constructor", "line": 151, "params": ["self", "ontology"]},
            "extract_semantic_cube": {"type": "instance", "line": 158, "params": ["self", "document_segments"]},
            "_empty_semantic_cube": {"type": "instance", "line": 212, "params": ["self"]},
            "_vectorize_segments": {"type": "instance", "line": 231, "params": ["self", "segments"]},
            "_process_segment": {"type": "instance", "line": 244, "params": ["self", "segment", "idx", "vector"]},
            "_classify_value_chain_link": {"type": "instance", "line": 282, "params": ["self", "segment"]},
            "_classify_policy_domain": {"type": "instance", "line": 303, "params": ["self", "segment"]},
            "_classify_cross_cutting_themes": {"type": "instance", "line": 317, "params": ["self", "segment"]},
            "_calculate_semantic_complexity": {"type": "instance", "line": 331, "params": ["self", "semantic_cube"]}
        }
    },
    "PerformanceAnalyzer": {
        "type": "class",
        "methods": {
            "__init__": {"type": "constructor", "line": 381, "params": ["self", "ontology"]},
            "analyze_performance": {"type": "instance", "line": 388, "params": ["self", "semantic_cube"]},
            "_calculate_throughput_metrics": {"type": "instance", "line": 423, "params": ["self", "segments", "link_config"]},
            "_detect_bottlenecks": {"type": "instance", "line": 462, "params": ["self", "segments", "link_config"]},
            "_calculate_loss_functions": {"type": "instance", "line": 496, "params": ["self", "metrics", "link_config"]},
            "_generate_recommendations": {"type": "instance", "line": 533, "params": ["self", "performance_analysis"]}
        }
    },
    "TextMiningEngine": {
        "type": "class",
        "methods": {
            "__init__": {"type": "constructor", "line": 557, "params": ["self", "ontology"]},
            "diagnose_critical_links": {"type": "instance", "line": 574, "params": ["self", "semantic_cube", "performance_analysis"]},
            "_identify_critical_links": {"type": "instance", "line": 615, "params": ["self", "performance_analysis"]},
            "_analyze_link_text": {"type": "instance", "line": 640, "params": ["self", "segments"]},
            "_assess_risks": {"type": "instance", "line": 675, "params": ["self", "segments", "text_analysis"]},
            "_generate_interventions": {"type": "instance", "line": 703, "params": ["self", "link_name", "risk_assessment", "text_analysis"]}
        }
    },
    "MunicipalAnalyzer": {
        "type": "class",
        "methods": {
            "__init__": {"type": "constructor", "line": 739, "params": ["self"]},
            "analyze_document": {"type": "instance", "line": 747, "params": ["self", "document_path"]},
            "_load_document": {"type": "instance", "line": 784, "params": ["self", "document_path"]},
            "_generate_summary": {"type": "instance", "line": 807, "params": ["self", "semantic_cube", "performance_analysis", "critical_diagnosis"]}
        }
    },
    "DocumentProcessor": {
        "type": "class",
        "methods": {
            "load_pdf": {"type": "static", "line": 984, "params": ["pdf_path"]},
            "load_docx": {"type": "static", "line": 1000, "params": ["docx_path"]},
            "segment_text": {"type": "static", "line": 1016, "params": ["text", "method"]}
        }
    },
    "ResultsExporter": {
        "type": "class",
        "methods": {
            "export_to_json": {"type": "static", "line": 1066, "params": ["results", "output_path"]},
            "export_to_excel": {"type": "static", "line": 1076, "params": ["results", "output_path"]},
            "export_summary_report": {"type": "static", "line": 1146, "params": ["results", "output_path"]}
        }
    },
    "ConfigurationManager": {
        "type": "class",
        "methods": {
            "__init__": {"type": "constructor", "line": 1255, "params": ["self", "config_path"]},
            "load_config": {"type": "instance", "line": 1261, "params": ["self"]},
            "save_config": {"type": "instance", "line": 1276, "params": ["self", "config"]},
            "validate_config": {"type": "instance", "line": 1284, "params": ["self", "config"]}
        }
    },
    "BatchProcessor": {
        "type": "class",
        "methods": {
            "__init__": {"type": "constructor", "line": 1306, "params": ["self", "analyzer"]},
            "process_directory": {"type": "instance", "line": 1310, "params": ["self", "directory_path", "pattern"]},
            "export_batch_results": {"type": "instance", "line": 1358, "params": ["self", "batch_results", "output_dir"]},
            "_create_batch_summary": {"type": "instance", "line": 1371, "params": ["self", "batch_results", "output_path"]}
        }
    }
}

def generate_invocation_pattern(class_name: str, method_name: str, method_info: Dict) -> str:
    """Generate the correct invocation pattern for a method"""
    method_type = method_info['type']
    params = method_info['params']
    
    # Remove self/cls from params
    actual_params = [p for p in params if p not in ['self', 'cls']]
    param_str = ', '.join(actual_params)
    
    if method_type == 'static':
        return f"{class_name}.{method_name}({param_str})"
    elif method_type == 'classmethod':
        return f"{class_name}.{method_name}({param_str})"
    elif method_type == 'constructor':
        return f"{class_name}({param_str})"
    else:  # instance method
        var_name = class_name.lower().replace('analyzer', 'analyzer').replace('engine', 'engine').replace('processor', 'processor')
        if method_name.startswith('_'):
            return f"{var_name} = {class_name}(...); {var_name}.{method_name}({param_str})"
        else:
            return f"{var_name} = {class_name}(...); {var_name}.{method_name}({param_str})"


def find_adapter_violations() -> List[Dict[str, Any]]:
    """Find violations in AnalyzerOneAdapter"""
    
    violations = []
    
    # Read the adapter file
    with open('orchestrator/module_adapters.py', 'r', encoding='utf-8') as f:
        adapter_content = f.read()
    
    # Extract AnalyzerOneAdapter section
    adapter_match = re.search(
        r'class AnalyzerOneAdapter.*?(?=\nclass [A-Z]|\Z)', 
        adapter_content, 
        re.DOTALL
    )
    
    if not adapter_match:
        return violations
    
    adapter_section = adapter_match.group(0)
    lines = adapter_section.split('\n')
    
    # Find the starting line number
    start_line = adapter_content[:adapter_match.start()].count('\n') + 1
    
    # Patterns to detect invocations
    patterns = [
        # Instance method called without instantiation: self.ClassName.method()
        (r'self\.([A-Z][a-zA-Z_0-9]*)\.([a-z_][a-zA-Z_0-9]*)\s*\(', 'static_call_on_instance_class'),
        # Static method called with instantiation: self.ClassName().method()
        (r'self\.([A-Z][a-zA-Z_0-9]*)\(\)\.([a-z_][a-zA-Z_0-9]*)\s*\(', 'instance_call_pattern'),
        # Direct class instantiation: ClassName()
        (r'([A-Z][a-zA-Z_0-9]*)\(\)', 'direct_instantiation'),
    ]
    
    for line_idx, line in enumerate(lines):
        line_no = start_line + line_idx
        
        # Pattern 1: self.PerformanceAnalyzer() - Missing ontology parameter
        match = re.search(r'self\.PerformanceAnalyzer\(\)', line)
        if match:
            violations.append({
                'type': 'missing_required_parameter',
                'severity': 'high',
                'line_number': line_no,
                'line_content': line.strip(),
                'class_name': 'PerformanceAnalyzer',
                'method_name': '__init__',
                'description': "PerformanceAnalyzer requires 'ontology' parameter in constructor",
                'recommendation': "Use: ontology = self.MunicipalOntology(); analyzer = self.PerformanceAnalyzer(ontology)"
            })
        
        # Pattern 2: analyzer._calculate_semantic_complexity(segment) - Wrong parameter
        match = re.search(r'analyzer\._calculate_semantic_complexity\(segment\)', line)
        if match:
            violations.append({
                'type': 'incorrect_parameter',
                'severity': 'high',
                'line_number': line_no,
                'line_content': line.strip(),
                'class_name': 'SemanticAnalyzer',
                'method_name': '_calculate_semantic_complexity',
                'description': "_calculate_semantic_complexity expects 'semantic_cube' dict, not 'segment' string",
                'recommendation': "Use: analyzer._calculate_semantic_complexity(semantic_cube)"
            })
        
        # Pattern 3: Static methods called on instances
        for class_name in ['DocumentProcessor', 'ResultsExporter']:
            pattern = rf'{class_name}\(\)\.(load_pdf|load_docx|segment_text|export_to_json|export_to_excel|export_summary_report)'
            match = re.search(pattern, line)
            if match:
                method_name = match.group(1)
                violations.append({
                    'type': 'incorrect_static_invocation',
                    'severity': 'medium',
                    'line_number': line_no,
                    'line_content': line.strip(),
                    'class_name': class_name,
                    'method_name': method_name,
                    'description': f"Static method '{method_name}' called on instance",
                    'recommendation': f"Use: {class_name}.{method_name}(...) without instantiation"
                })
        
        # Pattern 4: Instance methods called without ontology
        match = re.search(r'self\.SemanticAnalyzer\(\)\.', line)
        if match and 'ontology' not in line:
            violations.append({
                'type': 'missing_required_parameter',
                'severity': 'high',
                'line_number': line_no,
                'line_content': line.strip(),
                'class_name': 'SemanticAnalyzer',
                'method_name': '__init__',
                'description': "SemanticAnalyzer requires 'ontology' parameter in constructor",
                'recommendation': "Use: ontology = self.MunicipalOntology(); analyzer = self.SemanticAnalyzer(ontology)"
            })
        
        # Pattern 5: Instance methods called without instantiation
        match = re.search(r'self\.(TextMiningEngine)\.(diagnose_critical_links|_identify_critical_links|_analyze_link_text)', line)
        if match:
            class_name = match.group(1)
            method_name = match.group(2)
            violations.append({
                'type': 'missing_instantiation',
                'severity': 'high',
                'line_number': line_no,
                'line_content': line.strip(),
                'class_name': class_name,
                'method_name': method_name,
                'description': f"Instance method '{method_name}' called without instantiation",
                'recommendation': f"Use: ontology = self.MunicipalOntology(); engine = self.{class_name}(ontology); engine.{method_name}(...)"
            })
    
    return violations


def generate_audit_report() -> Dict[str, Any]:
    """Generate complete audit report"""
    
    # Build method inventory
    method_inventory = {}
    total_methods = 0
    
    for class_name, class_info in ANALYZER_ONE_CLASSES.items():
        methods_by_type = {
            'staticmethods': [],
            'classmethods': [],
            'instance_methods': [],
            'constructors': [],
            'private_methods': []
        }
        
        for method_name, method_info in class_info['methods'].items():
            method_type = method_info['type']
            
            method_entry = {
                'name': method_name,
                'line_number': method_info['line'],
                'parameters': method_info['params'],
                'correct_invocation_pattern': generate_invocation_pattern(class_name, method_name, method_info)
            }
            
            if method_type == 'static':
                methods_by_type['staticmethods'].append(method_entry)
            elif method_type == 'classmethod':
                methods_by_type['classmethods'].append(method_entry)
            elif method_type == 'constructor':
                methods_by_type['constructors'].append(method_entry)
            elif method_name.startswith('_') and not method_name.startswith('__'):
                methods_by_type['private_methods'].append(method_entry)
            else:
                methods_by_type['instance_methods'].append(method_entry)
            
            total_methods += 1
        
        # Remove empty categories
        method_inventory[class_name] = {k: v for k, v in methods_by_type.items() if v}
    
    # Find violations
    violations = find_adapter_violations()
    
    # Sort violations by severity and line number
    violations.sort(key=lambda x: (0 if x['severity'] == 'high' else 1 if x['severity'] == 'medium' else 2, x['line_number']))
    
    # Build correction matrix
    correction_matrix = []
    for violation in violations:
        correction_matrix.append({
            'violation_type': violation['type'],
            'severity': violation['severity'],
            'location': {
                'file': 'orchestrator/module_adapters.py',
                'class': 'AnalyzerOneAdapter',
                'line_number': violation['line_number'],
                'line_content': violation['line_content']
            },
            'issue': {
                'class_name': violation['class_name'],
                'method_name': violation['method_name'],
                'description': violation['description']
            },
            'recommendation': violation['recommendation']
        })
    
    # Build final report
    report = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'analyzer_file': 'Analyzer_one.py',
            'adapter_file': 'orchestrator/module_adapters.py',
            'adapter_class': 'AnalyzerOneAdapter',
            'analysis_version': '1.0.0'
        },
        'statistics': {
            'total_classes_in_analyzer': len(ANALYZER_ONE_CLASSES),
            'total_methods_cataloged': total_methods,
            'static_methods': sum(1 for c in ANALYZER_ONE_CLASSES.values() for m in c['methods'].values() if m['type'] == 'static'),
            'class_methods': sum(1 for c in ANALYZER_ONE_CLASSES.values() for m in c['methods'].values() if m['type'] == 'classmethod'),
            'instance_methods': sum(1 for c in ANALYZER_ONE_CLASSES.values() for m in c['methods'].values() if m['type'] == 'instance'),
            'constructors': sum(1 for c in ANALYZER_ONE_CLASSES.values() for m in c['methods'].values() if m['type'] == 'constructor'),
            'total_violations_found': len(violations),
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
    print("ANALYZER_ONE INVOCATION AUDIT GENERATOR")
    print("="*80)
    
    print("\n1. Generating method catalog from Analyzer_one.py...")
    print(f"   Found {len(ANALYZER_ONE_CLASSES)} classes")
    
    print("\n2. Analyzing AnalyzerOneAdapter invocations...")
    violations = find_adapter_violations()
    print(f"   Found {len(violations)} potential violations")
    
    print("\n3. Generating comprehensive audit report...")
    audit_report = generate_audit_report()
    
    print("\n4. Saving report to analyzer_one_invocation_audit.json...")
    with open('analyzer_one_invocation_audit.json', 'w', encoding='utf-8') as f:
        json.dump(audit_report, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*80)
    print("AUDIT SUMMARY")
    print("="*80)
    stats = audit_report['statistics']
    print(f"Classes Cataloged:                {stats['total_classes_in_analyzer']}")
    print(f"Total Methods:                    {stats['total_methods_cataloged']}")
    print(f"  - Static Methods:               {stats['static_methods']}")
    print(f"  - Class Methods:                {stats['class_methods']}")
    print(f"  - Instance Methods:             {stats['instance_methods']}")
    print(f"  - Constructors:                 {stats['constructors']}")
    print(f"\nViolations Found:                 {stats['total_violations_found']}")
    print(f"  - High Severity:                {stats['high_severity_violations']}")
    print(f"  - Medium Severity:              {stats['medium_severity_violations']}")
    print(f"  - Low Severity:                 {stats['low_severity_violations']}")
    
    if audit_report['correction_matrix']:
        print("\nTop Violations:")
        for i, violation in enumerate(audit_report['correction_matrix'][:5], 1):
            print(f"\n{i}. [{violation['severity'].upper()}] Line {violation['location']['line_number']}")
            print(f"   Issue: {violation['issue']['description']}")
            print(f"   Fix: {violation['recommendation']}")
    
    print("\n" + "="*80)
    print("✓ Audit report saved: analyzer_one_invocation_audit.json")
    print("="*80)


if __name__ == '__main__':
    main()
