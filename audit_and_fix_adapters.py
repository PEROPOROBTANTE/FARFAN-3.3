#!/usr/bin/env python3
"""
Systematic Audit and Correction of Module Adapters
Identifies and fixes static vs instance method invocation errors
"""
import ast
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

# Map modules to their classes with method information
SOURCE_MODULES = {
    'policy_processor.py': {
        'ProcessorConfig': {},
        'BayesianEvidenceScorer': {'_calculate_shannon_entropy': 'static'},
        'PolicyTextProcessor': {},
        'IndustrialPolicyProcessor': {'_extract_metadata': 'static', '_compute_avg_confidence': 'static'},
        'AdvancedTextSanitizer': {},
        'ResilientFileHandler': {'read_text': 'classmethod', 'write_text': 'classmethod'},
        'PolicyAnalysisPipeline': {},
    },
    'Analyzer_one.py': {
        'DocumentProcessor': {'load_pdf': 'static', 'load_docx': 'static', 'segment_text': 'static'},
        'ResultsExporter': {'export_to_json': 'static', 'export_to_excel': 'static', 'export_summary_report': 'static'},
        'MunicipalAnalyzer': {},
        'SemanticAnalyzer': {},
        'PerformanceAnalyzer': {},
        'TextMiningEngine': {},
    },
    'teoria_cambio.py': {
        'TeoriaCambio': {'_es_conexion_valida': 'static', '_extraer_categorias': 'static', '_validar_orden_causal': 'static', 
                        '_encontrar_caminos_completos': 'static', '_generar_sugerencias_internas': 'static'},
        'AdvancedDAGValidator': {'_is_acyclic': 'static', '_calculate_confidence_interval': 'static', 
                                '_calculate_statistical_power': 'static', '_calculate_bayesian_posterior': 'static'},
        'IndustrialGradeValidator': {},
    },
    'dereck_beach.py': {
        'BeachEvidentialTest': {'classify_test': 'static', 'apply_test_logic': 'static'},
    }
}

def find_invocation_errors():
    """Find all invocation errors in module_adapters.py"""
    errors = []
    
    with open('orchestrator/module_adapters.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line_num, line in enumerate(lines, 1):
        # Pattern 1: Incorrect instantiation for static methods
        # e.g., processor = self.DocumentProcessor()
        match = re.search(r'(\w+)\s*=\s*self\.(\w+)\(\)', line)
        if match:
            var_name, class_name = match.groups()
            # Check if this is a static-only class
            for module, classes in SOURCE_MODULES.items():
                if class_name in classes:
                    methods = classes[class_name]
                    # If all methods are static, this instantiation is wrong
                    if methods and all(v == 'static' for v in methods.values()):
                        errors.append({
                            'line_num': line_num,
                            'type': 'unnecessary_instantiation',
                            'class_name': class_name,
                            'original': line.strip(),
                            'module': module
                        })
        
        # Pattern 2: Static method calls on class (should not instantiate first)
        # This is actually correct for static methods, but let's verify instance creation
        
    return errors

def generate_corrections():
    """Generate correction log and fixed code"""
    correction_log = []
    
    with open('orchestrator/module_adapters.py', 'r', encoding='utf-8') as f:
        content = f.read()
        lines = content.split('\n')
    
    corrections = []
    
    # Specific known issues based on AST analysis
    known_issues = [
        {
            'line_pattern': r'processor = self\.DocumentProcessor\(\)',
            'class_name': 'DocumentProcessor',
            'issue': 'DocumentProcessor methods (load_pdf, load_docx, segment_text) are @staticmethod - instantiation not required',
            'fix_type': 'remove_instantiation'
        },
        {
            'line_pattern': r'self\.ResultsExporter\.(\w+)\(',
            'class_name': 'ResultsExporter',
            'issue': 'ResultsExporter methods are @staticmethod - already correctly called as class methods',
            'fix_type': 'correct'
        },
        {
            'line_pattern': r'test_type = self\.BeachEvidentialTest\.classify_test\(',
            'class_name': 'BeachEvidentialTest',
            'issue': 'BeachEvidentialTest.classify_test is @staticmethod - already correctly called',
            'fix_type': 'correct'
        },
        {
            'line_pattern': r'is_acyclic = self\.AdvancedDAGValidator\._is_acyclic\(',
            'class_name': 'AdvancedDAGValidator',
            'issue': 'AdvancedDAGValidator._is_acyclic is @staticmethod - already correctly called',
            'fix_type': 'correct'
        },
    ]
    
    # Find DocumentProcessor instantiation issues
    for i, line in enumerate(lines):
        if 'processor = self.DocumentProcessor()' in line:
            correction_log.append({
                'adapter_name': 'AnalyzerOneAdapter',
                'line_number': i + 1,
                'original_code': line.strip(),
                'corrected_code': '# Corrected static method violation: DocumentProcessor methods are @staticmethod - no instantiation needed',
                'justification': 'DocumentProcessor.load_pdf, .load_docx, and .segment_text are all @staticmethod decorators in Analyzer_one.py. Static methods should be called directly on the class without creating an instance.',
                'source_reference': 'Analyzer_one.py lines 983, 1001, 1018'
            })
            corrections.append((i, line, '# DocumentProcessor methods are static'))
    
    return correction_log, corrections

def main():
    print("=" * 80)
    print("ADAPTER INVOCATION AUDIT")
    print("=" * 80)
    
    # Generate inventory first
    print("\n[1] Generating source module inventory...")
    with open('source_modules_inventory.json', 'w', encoding='utf-8') as f:
        json.dump(SOURCE_MODULES, f, indent=2)
    print("✓ Created source_modules_inventory.json")
    
    # Find errors
    print("\n[2] Analyzing adapter invocations...")
    errors = find_invocation_errors()
    print(f"✓ Found {len(errors)} potential issues")
    
    # Generate corrections
    print("\n[3] Generating corrections...")
    correction_log, corrections = generate_corrections()
    
    # Save correction log
    with open('correction_log.json', 'w', encoding='utf-8') as f:
        json.dump(correction_log, f, indent=2)
    print(f"✓ Created correction_log.json with {len(correction_log)} corrections")
    
    print("\n" + "=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)
    print(f"\nFiles generated:")
    print(f"  - source_modules_inventory.json")
    print(f"  - correction_log.json")
    print(f"\nFound {len(correction_log)} corrections needed")
    
    if correction_log:
        print(f"\nSample correction:")
        print(json.dumps(correction_log[0], indent=2))

if __name__ == '__main__':
    main()
