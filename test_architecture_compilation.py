#!/usr/bin/env python3
"""
FARFAN 3.0 - Architecture & Compilation Test
==============================================
Tests code structure, imports, and invocation WITHOUT requiring heavy dependencies.
This validates that the orchestrator architecture is sound.

Author: AI Assistant
Date: 2024
"""
import sys
import ast
import importlib.util
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('test_architecture.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


class ArchitectureValidator:
    """Validates FARFAN 3.0 architecture without importing heavy dependencies"""

    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.orchestrator_path = base_path / "orchestrator"
        self.errors = []
        self.warnings = []
        self.success_count = 0
        self.total_tests = 0

    def test_all(self) -> bool:
        """Run all architecture tests"""
        logger.info("="*80)
        logger.info("FARFAN 3.0 - ARCHITECTURE & COMPILATION TEST SUITE")
        logger.info("="*80)

        tests = [
            ("File Structure", self.test_file_structure),
            ("Python Syntax", self.test_python_syntax),
            ("Import Statements", self.test_import_statements),
            ("Class Definitions", self.test_class_definitions),
            ("Method Signatures", self.test_method_signatures),
            ("Adapter Registry", self.test_adapter_registry),
            ("Question Router", self.test_question_router),
            ("Choreographer", self.test_choreographer),
            ("Config Consistency", self.test_config_consistency),
            ("Invocation Chain", self.test_invocation_chain),
        ]

        for test_name, test_func in tests:
            self.total_tests += 1
            logger.info(f"\n{'='*80}")
            logger.info(f"TEST {self.total_tests}: {test_name}")
            logger.info("="*80)

            try:
                result = test_func()
                if result:
                    self.success_count += 1
                    logger.info(f"✓ {test_name} PASSED")
                else:
                    logger.error(f"✗ {test_name} FAILED")
            except Exception as e:
                logger.error(f"✗ {test_name} CRASHED: {e}", exc_info=True)

        self._print_summary()
        return len(self.errors) == 0

    def test_file_structure(self) -> bool:
        """Test that all required files exist"""
        logger.info("\nChecking file structure...")

        required_files = {
            "orchestrator/__init__.py": "Orchestrator package init",
            "orchestrator/config.py": "Configuration",
            "orchestrator/module_adapters.py": "Module adapters (CRITICAL)",
            "orchestrator/question_router.py": "Question router",
            "orchestrator/choreographer.py": "Execution choreographer",
            "orchestrator/core_orchestrator.py": "Core orchestrator",
            "orchestrator/circuit_breaker.py": "Circuit breaker",
            "run_farfan.py": "Main entry point",
        }

        all_exist = True
        for file_path, description in required_files.items():
            full_path = self.base_path / file_path
            if full_path.exists():
                logger.info(f"  ✓ {file_path:40s} - {description}")
            else:
                logger.error(f"  ✗ {file_path:40s} - MISSING")
                self.errors.append(f"Missing file: {file_path}")
                all_exist = False

        return all_exist

    def test_python_syntax(self) -> bool:
        """Test Python syntax of all files"""
        logger.info("\nChecking Python syntax...")

        python_files = list(self.orchestrator_path.glob("*.py"))
        python_files.append(self.base_path / "run_farfan.py")

        all_valid = True
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    code = f.read()
                    ast.parse(code)
                logger.info(f"  ✓ {py_file.name:40s} - Valid syntax")
            except SyntaxError as e:
                logger.error(f"  ✗ {py_file.name:40s} - Syntax error: {e}")
                self.errors.append(f"Syntax error in {py_file.name}: {e}")
                all_valid = False

        return all_valid

    def test_import_statements(self) -> bool:
        """Test that import statements are correct (structure only)"""
        logger.info("\nChecking import statements...")

        # Parse module_adapters.py
        adapters_file = self.orchestrator_path / "module_adapters.py"

        with open(adapters_file, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())

        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)

        logger.info(f"  ✓ Found {len(imports)} import statements")

        # Check for expected imports
        expected = ['logging', 'time', 'typing', 'dataclasses', 'pathlib']
        missing = [imp for imp in expected if not any(imp in i for i in imports)]

        if missing:
            logger.warning(f"  ⚠ Missing expected imports: {missing}")
            self.warnings.append(f"Missing imports: {missing}")

        return True

    def test_class_definitions(self) -> bool:
        """Test that all adapter classes are defined"""
        logger.info("\nChecking class definitions...")

        adapters_file = self.orchestrator_path / "module_adapters.py"

        with open(adapters_file, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())

        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]

        expected_classes = [
            "ModuleResult",
            "PolicyProcessorAdapter",
            "AnalyzerOneAdapter",
            "ContradictionDetectorAdapter",
            "DerekBeachAdapter",
            "EmbeddingPolicyAdapter",
            "FinancialAnalyzerAdapter",
            "CausalProcessorAdapter",
            "PolicySegmenterAdapter",
            "ModuleAdapterRegistry"
        ]

        all_found = True
        for expected_class in expected_classes:
            if expected_class in classes:
                logger.info(f"  ✓ {expected_class:35s} - Defined")
            else:
                logger.error(f"  ✗ {expected_class:35s} - MISSING")
                self.errors.append(f"Missing class: {expected_class}")
                all_found = False

        logger.info(f"\n  Total classes found: {len(classes)}")
        logger.info(f"  Expected classes: {len(expected_classes)}")

        return all_found

    def test_method_signatures(self) -> bool:
        """Test that adapters have required methods"""
        logger.info("\nChecking adapter method signatures...")

        adapters_file = self.orchestrator_path / "module_adapters.py"

        with open(adapters_file, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())

        # Find adapter classes and their methods
        adapter_methods = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if "Adapter" in node.name and node.name != "ModuleAdapterRegistry":
                    methods = [m.name for m in node.body if isinstance(m, ast.FunctionDef)]
                    adapter_methods[node.name] = methods

        # Check required methods
        required_methods = ['__init__', '_load_module', 'execute']

        all_valid = True
        for adapter_name, methods in adapter_methods.items():
            logger.info(f"\n  {adapter_name}:")
            for req_method in required_methods:
                if req_method in methods:
                    logger.info(f"    ✓ {req_method}")
                else:
                    logger.error(f"    ✗ {req_method} - MISSING")
                    self.errors.append(f"{adapter_name} missing {req_method}")
                    all_valid = False

        return all_valid

    def test_adapter_registry(self) -> bool:
        """Test ModuleAdapterRegistry structure"""
        logger.info("\nChecking ModuleAdapterRegistry...")

        adapters_file = self.orchestrator_path / "module_adapters.py"

        with open(adapters_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check that all 8 adapters are registered
        expected_registrations = [
            'policy_processor',
            'analyzer_one',
            'contradiction_detector',
            'dereck_beach',
            'embedding_policy',
            'financial_viability',
            'causal_processor',
            'policy_segmenter'
        ]

        all_registered = True
        for adapter_name in expected_registrations:
            # Check if adapter is mentioned in __init__ method
            if f'"{adapter_name}"' in content or f"'{adapter_name}'" in content:
                logger.info(f"  ✓ {adapter_name:30s} - Referenced")
            else:
                logger.error(f"  ✗ {adapter_name:30s} - NOT REFERENCED")
                self.errors.append(f"Adapter not registered: {adapter_name}")
                all_registered = False

        return all_registered

    def test_question_router(self) -> bool:
        """Test QuestionRouter structure"""
        logger.info("\nChecking QuestionRouter...")

        router_file = self.orchestrator_path / "question_router.py"

        with open(router_file, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())

        # Find QuestionRouter class
        router_class = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "QuestionRouter":
                router_class = node
                break

        if not router_class:
            logger.error("  ✗ QuestionRouter class not found")
            self.errors.append("QuestionRouter class missing")
            return False

        # Check methods
        methods = [m.name for m in router_class.body if isinstance(m, ast.FunctionDef)]

        required_methods = [
            '__init__',
            'get_modules_for_question',
            '_determine_required_modules'
        ]

        all_found = True
        for method in required_methods:
            if method in methods:
                logger.info(f"  ✓ {method}")
            else:
                logger.error(f"  ✗ {method} - MISSING")
                self.errors.append(f"QuestionRouter missing {method}")
                all_found = False

        # Check dimension mapping exists
        with open(router_file, 'r', encoding='utf-8') as f:
            content = f.read()

        dimensions = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6']
        for dim in dimensions:
            if f'"{dim}"' in content or f"'{dim}'" in content:
                logger.info(f"  ✓ Dimension {dim} mapped")
            else:
                logger.warning(f"  ⚠ Dimension {dim} may not be mapped")
                self.warnings.append(f"Dimension {dim} mapping unclear")

        return all_found

    def test_choreographer(self) -> bool:
        """Test ExecutionChoreographer structure"""
        logger.info("\nChecking ExecutionChoreographer...")

        choreo_file = self.orchestrator_path / "choreographer.py"

        with open(choreo_file, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())

        # Find ExecutionChoreographer class
        choreo_class = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "ExecutionChoreographer":
                choreo_class = node
                break

        if not choreo_class:
            logger.error("  ✗ ExecutionChoreographer class not found")
            self.errors.append("ExecutionChoreographer class missing")
            return False

        # Check methods
        methods = [m.name for m in choreo_class.body if isinstance(m, ast.FunctionDef)]

        required_methods = [
            '__init__',
            'execute_for_question',
            '_execute_module',
            '_execute_wave'
        ]

        all_found = True
        for method in required_methods:
            if method in methods:
                logger.info(f"  ✓ {method}")
            else:
                logger.error(f"  ✗ {method} - MISSING")
                self.errors.append(f"ExecutionChoreographer missing {method}")
                all_found = False

        # Check if it imports ModuleAdapterRegistry
        with open(choreo_file, 'r', encoding='utf-8') as f:
            content = f.read()

        if "ModuleAdapterRegistry" in content:
            logger.info(f"  ✓ Imports ModuleAdapterRegistry")
        else:
            logger.error(f"  ✗ Does NOT import ModuleAdapterRegistry")
            self.errors.append("Choreographer doesn't import ModuleAdapterRegistry")
            all_found = False

        return all_found

    def test_config_consistency(self) -> bool:
        """Test that config module names match adapter names"""
        logger.info("\nChecking config consistency...")

        config_file = self.orchestrator_path / "config.py"
        adapters_file = self.orchestrator_path / "module_adapters.py"

        # Get module names from config
        with open(config_file, 'r', encoding='utf-8') as f:
            config_content = f.read()

        # Get registered adapters
        with open(adapters_file, 'r', encoding='utf-8') as f:
            adapters_content = f.read()

        expected_modules = [
            'policy_processor',
            'analyzer_one',
            'contradiction_detector',
            'dereck_beach',
            'embedding_policy',
            'financial_viability',
            'causal_processor',
            'policy_segmenter'
        ]

        all_consistent = True
        for module_name in expected_modules:
            in_config = f'"{module_name}"' in config_content or f"'{module_name}'" in config_content
            in_adapters = f'"{module_name}"' in adapters_content or f"'{module_name}'" in adapters_content

            if in_config and in_adapters:
                logger.info(f"  ✓ {module_name:30s} - Consistent")
            elif in_config and not in_adapters:
                logger.error(f"  ✗ {module_name:30s} - In config but NOT in adapters")
                self.errors.append(f"{module_name} in config but not in adapters")
                all_consistent = False
            elif not in_config and in_adapters:
                logger.error(f"  ✗ {module_name:30s} - In adapters but NOT in config")
                self.errors.append(f"{module_name} in adapters but not in config")
                all_consistent = False

        return all_consistent

    def test_invocation_chain(self) -> bool:
        """Test the complete invocation chain"""
        logger.info("\nChecking invocation chain...")

        # Trace the call path: run_farfan.py -> orchestrator -> choreographer -> registry -> adapters

        steps = [
            ("run_farfan.py imports orchestrator", self._check_run_farfan_imports),
            ("Orchestrator imports choreographer", self._check_orchestrator_imports),
            ("Choreographer imports registry", self._check_choreographer_imports),
            ("Registry initializes adapters", self._check_registry_initialization),
            ("Adapters have execute method", self._check_adapter_execution),
        ]

        all_valid = True
        for step_name, check_func in steps:
            try:
                result = check_func()
                if result:
                    logger.info(f"  ✓ {step_name}")
                else:
                    logger.error(f"  ✗ {step_name}")
                    all_valid = False
            except Exception as e:
                logger.error(f"  ✗ {step_name} - Error: {e}")
                all_valid = False

        return all_valid

    def _check_run_farfan_imports(self) -> bool:
        """Check that run_farfan.py imports orchestrator"""
        with open(self.base_path / "run_farfan.py", 'r') as f:
            content = f.read()
        return "from orchestrator import" in content or "import orchestrator" in content

    def _check_orchestrator_imports(self) -> bool:
        """Check that core_orchestrator imports choreographer"""
        file_path = self.orchestrator_path / "core_orchestrator.py"
        if not file_path.exists():
            return True  # Optional file
        with open(file_path, 'r') as f:
            content = f.read()
        return "choreographer" in content.lower() or "ExecutionChoreographer" in content

    def _check_choreographer_imports(self) -> bool:
        """Check that choreographer imports registry"""
        with open(self.orchestrator_path / "choreographer.py", 'r') as f:
            content = f.read()
        return "ModuleAdapterRegistry" in content

    def _check_registry_initialization(self) -> bool:
        """Check that registry __init__ creates all adapters"""
        with open(self.orchestrator_path / "module_adapters.py", 'r') as f:
            content = f.read()

        # Find the __init__ method of ModuleAdapterRegistry
        tree = ast.parse(content)

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "ModuleAdapterRegistry":
                for method in node.body:
                    if isinstance(method, ast.FunctionDef) and method.name == "__init__":
                        # Check if it creates adapter instances
                        init_code = ast.unparse(method)
                        return "Adapter()" in init_code

        return False

    def _check_adapter_execution(self) -> bool:
        """Check that adapters have execute method"""
        with open(self.orchestrator_path / "module_adapters.py", 'r') as f:
            tree = ast.parse(f.read())

        adapter_classes = [node for node in ast.walk(tree)
                          if isinstance(node, ast.ClassDef) and "Adapter" in node.name]

        for adapter_class in adapter_classes:
            if adapter_class.name == "ModuleAdapterRegistry":
                continue

            methods = [m.name for m in adapter_class.body if isinstance(m, ast.FunctionDef)]
            if "execute" not in methods:
                return False

        return True

    def _print_summary(self):
        """Print test summary"""
        logger.info("\n" + "="*80)
        logger.info("TEST SUMMARY")
        logger.info("="*80)

        logger.info(f"\nTests passed: {self.success_count}/{self.total_tests}")
        success_rate = (self.success_count / self.total_tests * 100) if self.total_tests > 0 else 0
        logger.info(f"Success rate: {success_rate:.1f}%")

        if self.errors:
            logger.error(f"\n❌ ERRORS ({len(self.errors)}):")
            for i, error in enumerate(self.errors, 1):
                logger.error(f"  {i}. {error}")

        if self.warnings:
            logger.warning(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for i, warning in enumerate(self.warnings, 1):
                logger.warning(f"  {i}. {warning}")

        if not self.errors and not self.warnings:
            logger.info("\n✅ ALL CHECKS PASSED - ARCHITECTURE IS SOUND!")
        elif not self.errors:
            logger.info("\n✓ ALL CRITICAL CHECKS PASSED (some warnings)")
        else:
            logger.error("\n❌ CRITICAL ERRORS FOUND - MUST FIX BEFORE DEPLOYMENT")

        logger.info("\nLog saved to: test_architecture.log")
        logger.info("="*80)


def main():
    """Main test runner"""
    base_path = Path(__file__).parent

    validator = ArchitectureValidator(base_path)
    success = validator.test_all()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
