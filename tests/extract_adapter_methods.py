"""
Extract all adapter methods from module_adapters.py to generate contract specifications.
"""
import re
import ast
from pathlib import Path
from typing import Dict, List, Tuple

def extract_adapter_methods(file_path: str) -> Dict[str, List[Tuple[str, str]]]:
    """Extract all adapter classes and their methods."""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    tree = ast.parse(content)
    
    adapters = {}
    current_adapter = None
    
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            if node.name.endswith('Adapter') and node.name != 'BaseAdapter':
                current_adapter = node.name
                adapters[current_adapter] = []
                
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        if item.name.startswith('_execute_'):
                            method_name = item.name.replace('_execute_', '')
                            
                            # Extract parameters
                            params = []
                            for arg in item.args.args:
                                if arg.arg not in ['self', 'kwargs']:
                                    params.append(arg.arg)
                            
                            adapters[current_adapter].append((method_name, params))
    
    return adapters

def main():
    file_path = Path('orchestrator/module_adapters.py')
    adapters = extract_adapter_methods(str(file_path))
    
    total_methods = 0
    for adapter_name, methods in sorted(adapters.items()):
        print(f"\n{adapter_name}: {len(methods)} methods")
        total_methods += len(methods)
        for method_name, params in sorted(methods):
            print(f"  - {method_name}({', '.join(params)})")
    
    print(f"\nTotal: {total_methods} methods across {len(adapters)} adapters")
    return adapters

if __name__ == '__main__':
    main()
