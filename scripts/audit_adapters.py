#!/usr/bin/env python3
"""Audit and correct adapter method invocations against source modules"""
import ast
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

# Source modules to analyze
SOURCE_MODULES = [
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

def analyze_source_module(filepath: str) -> Dict[str, Dict]:
    """Analyze a source module to extract class and method information"""
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
                        
                        method_type = 'instance'
                        if 'staticmethod' in decorators:
                            method_type = 'static'
                        elif 'classmethod' in decorators:
                            method_type = 'classmethod'
                        
                        methods[item.name] = {
                            'type': method_type,
                            'decorators': decorators,
                            'lineno': item.lineno
                        }
                classes[node.name] = methods
        
        return classes
    except Exception as e:
        print(f"Error analyzing {filepath}: {e}")
        return {}

def analyze_adapter_invocations(adapter_code: str) -> List[Dict]:
    """Find method invocations in adapter code"""
    issues = []
    
    # Pattern 1: self.ClassName() - incorrect instantiation for static methods
    pattern1 = r'(\w+)\s*=\s*self\.(\w+)\(\)'
    for match in re.finditer(pattern1, adapter_code):
        var_name, class_name = match.groups()
        issues.append({
            'type': 'instantiation',
            'pattern': match.group(0),
            'class_name': class_name,
            'position': match.start()
        })
    
    # Pattern 2: self.ClassName.method() - static method call pattern
    pattern2 = r'self\.(\w+)\.(\w+)\('
    for match in re.finditer(pattern2, adapter_code):
        class_name, method_name = match.groups()
        issues.append({
            'type': 'static_call',
            'pattern': match.group(0),
            'class_name': class_name,
            'method_name': method_name,
            'position': match.start()
        })
    
    return issues

def generate_inventory():
    """Generate inventory of all source modules"""
    inventory = {}
    for module_path in SOURCE_MODULES:
        if Path(module_path).exists():
            print(f"Analyzing {module_path}...")
            inventory[module_path] = analyze_source_module(module_path)
        else:
            print(f"Warning: {module_path} not found")
    
    with open('source_modules_inventory.json', 'w', encoding='utf-8') as f:
        json.dump(inventory, f, indent=2)
    
    print(f"\n✓ Generated source_modules_inventory.json with {len(inventory)} modules")
    return inventory

if __name__ == '__main__':
    generate_inventory()
