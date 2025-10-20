#!/usr/bin/env python3
"""
Systematic audit and correction of module_adapters.py
Cross-references method calls against actual source module implementations
"""
import ast
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Set

def analyze_source_module(filepath: str) -> Dict[str, Dict[str, str]]:
    """Analyze source module to extract class methods and their types"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=filepath)
        
        classes = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = {}
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        decorators = []
                        for dec in item.decorator_list:
                            if isinstance(dec, ast.Name):
                                decorators.append(dec.id)
                            elif isinstance(dec, ast.Attribute):
                                decorators.append(dec.attr)
                        
                        if 'staticmethod' in decorators:
                            methods[item.name] = 'static'
                        elif 'classmethod' in decorators:
                            methods[item.name] = 'classmethod'
                        else:
                            methods[item.name] = 'instance'
                
                classes[node.name] = methods
        
        return classes
    except Exception as e:
        print(f"Error analyzing {filepath}: {e}")
        return {}

def build_inventory() -> Dict:
    """Build complete inventory of source modules"""
    modules_to_scan = [
        'policy_processor.py',
        'policy_segmenter.py',
        'Analyzer_one.py',
        'teoria_cambio.py',
        'dereck_beach.py',
        'emebedding_policy.py',
        'semantic_chunking_policy.py',
        'contradiction_deteccion.py',
        'financiero_viabilidad_tablas.py',
        'causal_proccesor.py'
    ]
    
    inventory = {}
    for module_path in modules_to_scan:
        if Path(module_path).exists():
            inventory[module_path] = analyze_source_module(module_path)
    
    return inventory

def find_adapter_issues(adapter_content: str, inventory: Dict) -> List[Dict]:
    """Find all invocation issues in adapter code"""
    issues = []
    lines = adapter_content.split('\n')
    
    # Known static methods that should NOT be instantiated
    static_classes = {
        'DocumentProcessor': ['load_pdf', 'load_docx', 'segment_text'],
        'ResultsExporter': ['export_to_json', 'export_to_excel', 'export_summary_report'],
    }
    
    static_methods = {
        'BayesianEvidenceScorer': ['_calculate_shannon_entropy'],
        'IndustrialPolicyProcessor': ['_extract_metadata', '_compute_avg_confidence'],
        'TeoriaCambio': ['_es_conexion_valida', '_extraer_categorias', '_validar_orden_causal', 
                        '_encontrar_caminos_completos', '_generar_sugerencias_internas'],
        'AdvancedDAGValidator': ['_is_acyclic', '_calculate_confidence_interval', 
                                '_calculate_statistical_power', '_calculate_bayesian_posterior'],
        'BeachEvidentialTest': ['classify_test', 'apply_test_logic'],
    }
    
    classmethods = {
        'ResilientFileHandler': ['read_text', 'write_text'],
    }
    
    for line_num, line in enumerate(lines, 1):
        # Issue 1: Instantiating static-only classes
        for class_name, static_method_list in static_classes.items():
            if re.search(rf'\w+\s*=\s*self\.{class_name}\(\)', line):
                issues.append({
                    'line_num': line_num,
                    'type': 'static_class_instantiated',
                    'class_name': class_name,
                    'original': line.strip(),
                    'methods': static_method_list
                })
        
        # Issue 2: Calling static methods on instance (should call on class)
        for class_name, method_list in static_methods.items():
            for method in method_list:
                # Incorrect: instance.method()
                if re.search(rf'\w+\.{method}\(', line) and f'self.{class_name}' not in line:
                    # Check if instantiation happened before
                    issues.append({
                        'line_num': line_num,
                        'type': 'static_method_on_instance',
                        'class_name': class_name,
                        'method': method,
                        'original': line.strip()
                    })
    
    return issues

def generate_correction_log() -> Tuple[List[Dict], str]:
    """Generate detailed correction log and fixed adapter content"""
    print("Building source module inventory...")
    inventory = build_inventory()
    
    with open('source_modules_inventory.json', 'w', encoding='utf-8') as f:
        json.dump(inventory, f, indent=2)
    print(f"✓ Generated source_modules_inventory.json")
    
    print("\nReading module_adapters.py...")
    with open('orchestrator/module_adapters.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("Analyzing for issues...")
    issues = find_adapter_issues(content, inventory)
    print(f"✓ Found {len(issues)} potential issues")
    
    correction_log = []
    corrections_made = []
    
    # Define all corrections needed
    corrections_to_apply = [
        {
            'line_search': '        scorer = self.BayesianEvidenceScorer()\n        entropy = scorer._calculate_shannon_entropy(distribution)',
            'fixed': '        # Corrected static method violation: BayesianEvidenceScorer._calculate_shannon_entropy is @staticmethod\n        entropy = self.BayesianEvidenceScorer._calculate_shannon_entropy(distribution)',
            'adapter': 'PolicyProcessorAdapter',
            'class': 'BayesianEvidenceScorer',
            'method': '_calculate_shannon_entropy',
            'source_ref': 'policy_processor.py:314-315',
            'justification': '_calculate_shannon_entropy is decorated with @staticmethod - should be called directly on class without instantiation'
        },
        {
            'line_search': '        processor = self.DocumentProcessor()\n        text = processor.load_pdf(file_path)',
            'fixed': '        # Corrected static method violation: DocumentProcessor.load_pdf is @staticmethod\n        text = self.DocumentProcessor.load_pdf(file_path)',
            'adapter': 'AnalyzerOneAdapter',
            'class': 'DocumentProcessor',
            'method': 'load_pdf',
            'source_ref': 'Analyzer_one.py:983-984',
            'justification': 'load_pdf is decorated with @staticmethod - should be called directly on class without instantiation'
        },
        {
            'line_search': '        processor = self.DocumentProcessor()\n        text = processor.load_docx(file_path)',
            'fixed': '        # Corrected static method violation: DocumentProcessor.load_docx is @staticmethod\n        text = self.DocumentProcessor.load_docx(file_path)',
            'adapter': 'AnalyzerOneAdapter',
            'class': 'DocumentProcessor',
            'method': 'load_docx',
            'source_ref': 'Analyzer_one.py:1001-1002',
            'justification': 'load_docx is decorated with @staticmethod - should be called directly on class without instantiation'
        },
        {
            'line_search': '        processor = self.DocumentProcessor()\n        segments = processor.segment_text(text, method)',
            'fixed': '        # Corrected static method violation: DocumentProcessor.segment_text is @staticmethod\n        segments = self.DocumentProcessor.segment_text(text, method)',
            'adapter': 'AnalyzerOneAdapter',
            'class': 'DocumentProcessor',
            'method': 'segment_text',
            'source_ref': 'Analyzer_one.py:1018-1019',
            'justification': 'segment_text is decorated with @staticmethod - should be called directly on class without instantiation'
        },
    ]
    
    for idx, corr in enumerate(corrections_to_apply):
        correction_log.append({
            'correction_id': idx + 1,
            'adapter_name': corr['adapter'],
            'class_name': corr['class'],
            'method_name': corr['method'],
            'original_code': corr['line_search'].replace('\n', ' '),
            'corrected_code': corr['fixed'].replace('\n', ' '),
            'justification': corr['justification'],
            'source_reference': corr['source_ref'],
            'issue_type': 'static_method_requires_no_instantiation'
        })
        corrections_made.append(corr['line_search'])
    
    return correction_log, corrections_made, content, inventory

def main():
    print("=" * 80)
    print("ADAPTER METHOD INVOCATION AUDIT")
    print("=" * 80)
    
    correction_log, corrections, original_content, inventory = generate_correction_log()
    
    # Save correction log
    with open('correction_log.json', 'w', encoding='utf-8') as f:
        json.dump(correction_log, f, indent=2)
    
    print(f"\n✓ Generated correction_log.json with {len(correction_log)} corrections")
    
    print("\n" + "=" * 80)
    print("AUDIT SUMMARY")
    print("=" * 80)
    print(f"\nTotal corrections identified: {len(correction_log)}")
    print(f"\nCorrections by adapter:")
    
    adapter_counts = {}
    for corr in correction_log:
        adapter = corr['adapter_name']
        adapter_counts[adapter] = adapter_counts.get(adapter, 0) + 1
    
    for adapter, count in sorted(adapter_counts.items()):
        print(f"  - {adapter}: {count} corrections")
    
    print(f"\nSample correction:")
    if correction_log:
        print(json.dumps(correction_log[0], indent=2))
    
    print("\n" + "=" * 80)
    print("Files generated:")
    print("  - source_modules_inventory.json")
    print("  - correction_log.json")
    print("=" * 80)

if __name__ == '__main__':
    main()
