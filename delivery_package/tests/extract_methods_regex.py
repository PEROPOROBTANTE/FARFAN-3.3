"""
Extract all adapter methods using regex (since the file has syntax errors).
"""
import re
from pathlib import Path
from typing import Dict, List

def extract_adapter_methods_regex(file_path: str) -> Dict[str, List[str]]:
    """Extract all adapter classes and their _execute_ methods using regex."""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    adapters = {}
    
    # Find all adapter class definitions
    class_pattern = r'class\s+(\w+Adapter)\(BaseAdapter\):'
    adapter_classes = re.findall(class_pattern, content)
    
    for adapter_class in adapter_classes:
        # Find the class definition section
        class_start = content.find(f'class {adapter_class}(BaseAdapter):')
        if class_start == -1:
            continue
            
        # Find the next class definition
        next_class = content.find('\nclass ', class_start + 1)
        if next_class == -1:
            class_section = content[class_start:]
        else:
            class_section = content[class_start:next_class]
        
        # Find all _execute_ methods in this class
        method_pattern = r'def\s+(_execute_\w+)\s*\('
        methods = re.findall(method_pattern, class_section)
        
        # Extract method names (remove _execute_ prefix)
        method_names = [m.replace('_execute_', '') for m in methods]
        adapters[adapter_class] = sorted(set(method_names))
    
    return adapters

def main():
    file_path = Path('orchestrator/module_adapters.py')
    adapters = extract_adapter_methods_regex(str(file_path))
    
    total_methods = 0
    for adapter_name, methods in sorted(adapters.items()):
        print(f"\n{adapter_name}: {len(methods)} methods")
        total_methods += len(methods)
        for method_name in methods:
            print(f"  - {method_name}")
    
    print(f"\n{'='*60}")
    print(f"Total: {total_methods} methods across {len(adapters)} adapters")
    return adapters

if __name__ == '__main__':
    adapters = main()
