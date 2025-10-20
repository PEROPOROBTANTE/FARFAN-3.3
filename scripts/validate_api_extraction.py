#!/usr/bin/env python3
"""
Validation script for API surface extraction
"""

import json
from pathlib import Path

def main():
    # Load generated files
    with open('source_modules_inventory.json', 'r') as f:
        inventory = json.load(f)
    
    with open('baseline_metrics.json', 'r') as f:
        metrics = json.load(f)
    
    print("=" * 70)
    print("API EXTRACTION VALIDATION REPORT")
    print("=" * 70)
    
    # Check module count
    print(f"\n✓ Modules in inventory: {len(inventory)}")
    print(f"  Expected: 10")
    assert len(inventory) == 10, "Should have 10 modules"
    
    # List all modules
    print("\n✓ Modules processed:")
    for module_name in sorted(inventory.keys()):
        print(f"  - {module_name}")
    
    # Validate structure of each module
    print("\n✓ Module structure validation:")
    for module_name, module_data in inventory.items():
        assert 'module_name' in module_data
        assert 'docstring' in module_data
        assert 'classes' in module_data
        assert 'functions' in module_data
        
        # Validate each class
        for cls in module_data['classes']:
            assert 'name' in cls
            assert 'methods' in cls
            assert 'base_classes' in cls
            assert 'is_public' in cls
            assert 'lineno' in cls
            
            # Validate each method
            for method in cls['methods']:
                assert 'name' in method
                assert 'decorator' in method
                assert 'signature' in method
                assert 'parameters' in method
                assert 'invocation_pattern' in method
                
                # Check decorator-based invocation patterns
                if method['decorator'] == 'staticmethod':
                    assert f"{cls['name']}.{method['name']}()" in method['invocation_pattern']
                elif method['decorator'] == 'classmethod':
                    assert f"{cls['name']}.{method['name']}()" in method['invocation_pattern']
                else:
                    assert '_instance.' in method['invocation_pattern']
        
        # Validate each function
        for func in module_data['functions']:
            assert 'name' in func
            assert 'signature' in func
            assert 'parameters' in func
    
    print(f"  All {len(inventory)} modules have valid structure")
    
    # Check metrics totals
    print("\n✓ Baseline metrics validation:")
    totals = metrics['totals']
    print(f"  - Total classes: {totals['total_classes']}")
    print(f"  - Total functions: {totals['total_functions']}")
    print(f"  - Total methods: {totals['total_methods']}")
    print(f"    • Public methods: {totals['public_methods']}")
    print(f"    • Private methods: {totals['private_methods']}")
    print(f"    • Static methods: {totals['static_methods']}")
    print(f"    • Class methods: {totals['class_methods']}")
    
    # Calculate total API surface
    total_api_surface = (
        totals['total_classes'] +
        totals['total_functions'] +
        totals['total_methods']
    )
    
    threshold_95 = int(total_api_surface * 0.95)
    
    print(f"\n✓ API preservation target:")
    print(f"  - Total API elements: {total_api_surface}")
    print(f"  - 95% threshold: {threshold_95}")
    
    # Check decorator distribution
    print("\n✓ Decorator distribution:")
    static_methods = []
    class_methods = []
    regular_methods = 0
    
    for module_name, module_data in inventory.items():
        for cls in module_data['classes']:
            for method in cls['methods']:
                if method['decorator'] == 'staticmethod':
                    static_methods.append(f"{module_name}.{cls['name']}.{method['name']}")
                elif method['decorator'] == 'classmethod':
                    class_methods.append(f"{module_name}.{cls['name']}.{method['name']}")
                else:
                    regular_methods += 1
    
    print(f"  - Static methods: {len(static_methods)}")
    print(f"  - Class methods: {len(class_methods)}")
    print(f"  - Regular methods: {regular_methods}")
    
    # Show sample static methods
    if static_methods:
        print(f"\n✓ Sample static methods (require ClassName.method() invocation):")
        for sm in static_methods[:5]:
            print(f"  - {sm}")
    
    # Show sample class methods
    if class_methods:
        print(f"\n✓ Sample class methods (require ClassName.method() invocation):")
        for cm in class_methods[:5]:
            print(f"  - {cm}")
    
    # Verify per-module metrics match aggregation
    print("\n✓ Verifying metrics aggregation:")
    computed_totals = {
        'total_classes': 0,
        'total_functions': 0,
        'total_methods': 0,
        'public_methods': 0,
        'private_methods': 0,
        'static_methods': 0,
        'class_methods': 0
    }
    
    for module_name, module_metrics in metrics['modules'].items():
        for key in computed_totals:
            computed_totals[key] += module_metrics.get(key, 0)
    
    for key, value in computed_totals.items():
        reported = metrics['totals'][key]
        assert value == reported, f"Mismatch in {key}: computed={value}, reported={reported}"
    
    print(f"  All metrics match ✓")
    
    # Check for docstring coverage
    print("\n✓ Docstring coverage:")
    modules_with_docstrings = sum(1 for m in inventory.values() if m['docstring'])
    classes_with_docstrings = sum(
        1 for m in inventory.values()
        for c in m['classes']
        if c['docstring']
    )
    methods_with_docstrings = sum(
        1 for m in inventory.values()
        for c in m['classes']
        for method in c['methods']
        if method['docstring']
    )
    
    print(f"  - Modules with docstrings: {modules_with_docstrings}/{len(inventory)}")
    print(f"  - Classes with docstrings: {classes_with_docstrings}/{totals['total_classes']}")
    print(f"  - Methods with docstrings: {methods_with_docstrings}/{totals['total_methods']}")
    
    # Check for type hints
    print("\n✓ Type hint coverage:")
    methods_with_return_types = sum(
        1 for m in inventory.values()
        for c in m['classes']
        for method in c['methods']
        if method['return_type']
    )
    
    params_with_annotations = sum(
        1 for m in inventory.values()
        for c in m['classes']
        for method in c['methods']
        for param in method['parameters']
        if param['annotation']
    )
    
    total_params = sum(
        1 for m in inventory.values()
        for c in m['classes']
        for method in c['methods']
        for param in method['parameters']
    )
    
    print(f"  - Methods with return types: {methods_with_return_types}/{totals['total_methods']}")
    print(f"  - Parameters with type hints: {params_with_annotations}/{total_params}")
    
    print("\n" + "=" * 70)
    print("✓ ALL VALIDATIONS PASSED")
    print("=" * 70)
    
    print("\n📊 SUMMARY:")
    print(f"  • {len(inventory)} modules fully cataloged")
    print(f"  • {total_api_surface} total API elements documented")
    print(f"  • {threshold_95} elements required for 95% preservation")
    print(f"  • Static/class methods properly distinguished for invocation patterns")
    print(f"  • Full signatures with type hints and defaults captured")
    print(f"  • Ready for refactoring validation")


if __name__ == '__main__':
    main()
