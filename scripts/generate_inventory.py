#!/usr/bin/env python3
"""Generate source modules inventory for adapter verification"""
import ast
import json

def analyze_class(node):
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
    return methods

def analyze_module(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=filename)
        
        classes = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes[node.name] = analyze_class(node)
        
        return classes
    except Exception as e:
        return {'error': str(e)}

modules_to_analyze = [
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
for module in modules_to_analyze:
    inventory[module] = analyze_module(module)

with open('source_modules_inventory.json', 'w', encoding='utf-8') as f:
    json.dump(inventory, f, indent=2)

print('✓ Inventory generated: source_modules_inventory.json')
