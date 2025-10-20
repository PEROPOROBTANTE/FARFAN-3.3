"""
Canary System Installation Verification
========================================

Verifies that all components are properly installed and configured.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def verify_files():
    """Verify all required files exist"""
    print("=" * 80)
    print("VERIFYING CANARY SYSTEM INSTALLATION")
    print("=" * 80)
    print()
    
    required_files = [
        "tests/canary_generator.py",
        "tests/canary_runner.py",
        "tests/canary_fix_generator.py",
        "tests/test_canary_system.py",
        "tests/demo_canary_system.py",
        "tests/run_canary_system.sh",
        "tests/README_CANARIES.md",
        "tests/CANARY_SYSTEM_SUMMARY.md",
        "tests/canaries/.gitkeep",
        ".gitignore"
    ]
    
    print("Checking required files:")
    all_exist = True
    for file_path in required_files:
        path = Path(file_path)
        exists = path.exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {file_path}")
        if not exists:
            all_exist = False
    
    print()
    return all_exist


def verify_modules():
    """Verify required modules can be imported"""
    print("Checking module imports:")
    
    modules_to_check = [
        ("canary_generator", "CanaryGenerator"),
        ("canary_runner", "CanaryRunner"),
        ("canary_fix_generator", "CanaryFixGenerator")
    ]
    
    all_imported = True
    for module_name, class_name in modules_to_check:
        try:
            module = __import__(f"tests.{module_name}", fromlist=[class_name])
            cls = getattr(module, class_name)
            print(f"  ✓ {module_name}.{class_name}")
        except Exception as e:
            print(f"  ✗ {module_name}.{class_name}: {e}")
            all_imported = False
    
    print()
    return all_imported


def verify_adapter_registry():
    """Verify adapter registry is accessible"""
    print("Checking adapter registry:")
    
    try:
        from orchestrator.module_adapters import ModuleAdapterRegistry
        registry = ModuleAdapterRegistry()
        
        print(f"  ✓ ModuleAdapterRegistry initialized")
        print(f"  ✓ Registered adapters: {len(registry.adapters)}")
        print(f"  ✓ Available adapters: {len(registry.get_available_modules())}")
        
        print(f"\n  Adapter status:")
        for adapter_name, available in registry.get_module_status().items():
            status = "✓" if available else "✗"
            print(f"    {status} {adapter_name}")
        
        print()
        return True
        
    except Exception as e:
        print(f"  ✗ Failed to initialize registry: {e}")
        print()
        return False


def verify_coverage():
    """Verify method coverage is correct"""
    print("Checking method coverage:")
    
    try:
        from tests.canary_generator import CanaryGenerator
        
        generator = CanaryGenerator()
        total_methods = sum(generator.ADAPTER_METHODS.values())
        total_adapters = len(generator.ADAPTER_METHODS)
        
        print(f"  ✓ Total adapters: {total_adapters}")
        print(f"  ✓ Total methods: {total_methods}")
        
        if total_methods == 413:
            print(f"  ✓ Coverage: 413/413 methods (100%)")
        else:
            print(f"  ✗ Coverage: {total_methods}/413 methods ({total_methods/413*100:.1f}%)")
        
        print(f"\n  Per-adapter breakdown:")
        for adapter, count in sorted(generator.ADAPTER_METHODS.items()):
            print(f"    • {adapter:30s} : {count:3d} methods")
        
        print()
        return total_methods == 413
        
    except Exception as e:
        print(f"  ✗ Failed to check coverage: {e}")
        print()
        return False


def verify_contract_schema():
    """Verify contract schema is defined"""
    print("Checking contract schema:")
    
    try:
        from tests.canary_runner import CanaryRunner
        
        runner = CanaryRunner()
        schema = runner.EXPECTED_CONTRACT
        
        print(f"  ✓ Required keys: {len(schema['required_keys'])}")
        print(f"  ✓ Optional keys: {len(schema['optional_keys'])}")
        print(f"  ✓ Type definitions: {len(schema['types'])}")
        
        print()
        return True
        
    except Exception as e:
        print(f"  ✗ Failed to check schema: {e}")
        print()
        return False


def print_usage():
    """Print usage instructions"""
    print("=" * 80)
    print("INSTALLATION VERIFIED - READY TO USE")
    print("=" * 80)
    print()
    print("Quick Start:")
    print("  # Run complete pipeline")
    print("  ./tests/run_canary_system.sh")
    print()
    print("Individual Commands:")
    print("  # Generate baseline canaries")
    print("  python tests/canary_generator.py")
    print()
    print("  # Run regression tests")
    print("  python tests/canary_runner.py")
    print()
    print("  # Generate fix operations")
    print("  python tests/canary_fix_generator.py")
    print()
    print("  # Execute automatic fixes")
    print("  python tests/canary_fix_generator.py --execute-rebaseline")
    print()
    print("Documentation:")
    print("  tests/README_CANARIES.md")
    print("  tests/CANARY_SYSTEM_SUMMARY.md")
    print()
    print("Demonstration:")
    print("  python tests/demo_canary_system.py")
    print()
    print("Tests:")
    print("  pytest tests/test_canary_system.py -v")
    print()


def main():
    """Run all verification checks"""
    checks = [
        ("Files", verify_files),
        ("Modules", verify_modules),
        ("Adapter Registry", verify_adapter_registry),
        ("Coverage", verify_coverage),
        ("Contract Schema", verify_contract_schema)
    ]
    
    results = {}
    for name, check_func in checks:
        results[name] = check_func()
    
    print("=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    print()
    
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print()
    
    if all(results.values()):
        print_usage()
        return 0
    else:
        print("Some checks failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
