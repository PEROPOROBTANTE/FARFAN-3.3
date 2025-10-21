#!/usr/bin/env python3
"""
Dependency Validation Script

This script validates that the installed dependencies are compatible
and don't have conflicts. Run this after installing dependencies.

Usage:
    python validate_dependencies.py [--strategy torch|tensorflow|both]
"""

import sys
import argparse
from typing import List, Tuple, Optional


def check_python_version() -> Tuple[bool, str]:
    """Check if Python version is compatible."""
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        return False, f"❌ Python {version_str} is too old. Requires Python 3.10+"
    
    if version.major == 3 and version.minor >= 12:
        return True, f"⚠️  Python {version_str} detected. PyTorch 2.0.1 and TensorFlow 2.13.0 have limited Python 3.12 support. Consider using Python 3.10 or 3.11."
    
    return True, f"✅ Python {version_str} is compatible"


def check_package_installed(package_name: str) -> Tuple[bool, Optional[str]]:
    """Check if a package is installed and return its version."""
    try:
        module = __import__(package_name)
        version = getattr(module, "__version__", "unknown")
        return True, version
    except ImportError:
        return False, None


def validate_core_dependencies() -> List[Tuple[str, bool, str]]:
    """Validate core dependencies are installed."""
    results = []
    
    core_packages = [
        "spacy",
        "transformers",
        "sentence_transformers",
        "nltk",
        "sklearn",
        "pandas",
        "numpy",
        "yaml",
        "pydantic",
    ]
    
    for package in core_packages:
        installed, version = check_package_installed(package)
        if installed:
            results.append((package, True, f"✅ {package} {version} installed"))
        else:
            results.append((package, False, f"❌ {package} not installed"))
    
    return results


def validate_torch_dependencies() -> List[Tuple[str, bool, str]]:
    """Validate PyTorch dependencies."""
    results = []
    
    # Check torch
    installed, version = check_package_installed("torch")
    if installed:
        results.append(("torch", True, f"✅ torch {version} installed"))
        
        # Check if CUDA is available
        try:
            import torch
            if torch.cuda.is_available():
                cuda_version = torch.version.cuda
                results.append(("CUDA", True, f"✅ CUDA {cuda_version} available"))
            else:
                results.append(("CUDA", True, f"ℹ️  CUDA not available (CPU only)"))
        except Exception as e:
            results.append(("CUDA", False, f"❌ Error checking CUDA: {e}"))
    else:
        results.append(("torch", False, f"❌ torch not installed"))
    
    # Check torchvision
    installed, version = check_package_installed("torchvision")
    if installed:
        results.append(("torchvision", True, f"✅ torchvision {version} installed"))
    else:
        results.append(("torchvision", False, f"⚠️  torchvision not installed (optional)"))
    
    return results


def validate_tensorflow_dependencies() -> List[Tuple[str, bool, str]]:
    """Validate TensorFlow dependencies."""
    results = []
    
    # Check tensorflow
    installed, version = check_package_installed("tensorflow")
    if installed:
        results.append(("tensorflow", True, f"✅ tensorflow {version} installed"))
        
        # Check if GPU is available
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                results.append(("TF GPU", True, f"✅ {len(gpus)} GPU(s) available"))
            else:
                results.append(("TF GPU", True, f"ℹ️  No GPUs available (CPU only)"))
        except Exception as e:
            results.append(("TF GPU", False, f"❌ Error checking GPU: {e}"))
    else:
        results.append(("tensorflow", False, f"❌ tensorflow not installed"))
    
    # Check keras
    installed, version = check_package_installed("keras")
    if installed:
        results.append(("keras", True, f"✅ keras {version} installed"))
    else:
        results.append(("keras", False, f"⚠️  keras not installed (may be built into TF)"))
    
    return results


def check_dependency_conflicts() -> List[Tuple[str, bool, str]]:
    """Check for common dependency conflicts."""
    results = []
    
    # Check if both torch and tensorflow are installed
    torch_installed, torch_version = check_package_installed("torch")
    tf_installed, tf_version = check_package_installed("tensorflow")
    
    if torch_installed and tf_installed:
        results.append((
            "torch+tensorflow",
            True,
            f"⚠️  Both PyTorch {torch_version} and TensorFlow {tf_version} are installed. "
            "This may cause conflicts. See DEPENDENCY_CONFLICTS.md"
        ))
    
    # Check numpy version compatibility
    numpy_installed, numpy_version = check_package_installed("numpy")
    if numpy_installed:
        try:
            import numpy as np
            major, minor = map(int, np.__version__.split('.')[:2])
            if major == 1 and minor == 24:
                results.append(("numpy", True, f"✅ numpy {np.__version__} is compatible"))
            else:
                results.append((
                    "numpy",
                    False,
                    f"⚠️  numpy {np.__version__} may cause conflicts. Recommended: 1.24.x"
                ))
        except Exception as e:
            results.append(("numpy", False, f"❌ Error checking numpy: {e}"))
    
    # Check protobuf version
    protobuf_installed, protobuf_version = check_package_installed("google.protobuf")
    if protobuf_installed:
        results.append(("protobuf", True, f"✅ protobuf {protobuf_version} installed"))
    
    # Check pydantic version
    pydantic_installed, pydantic_version = check_package_installed("pydantic")
    if pydantic_installed:
        try:
            import pydantic
            major = int(pydantic.__version__.split('.')[0])
            if major == 2:
                results.append(("pydantic", True, f"✅ pydantic {pydantic.__version__} (v2)"))
            else:
                results.append((
                    "pydantic",
                    True,
                    f"ℹ️  pydantic {pydantic.__version__} (v1). Some features may expect v2."
                ))
        except Exception as e:
            results.append(("pydantic", False, f"❌ Error checking pydantic: {e}"))
    
    return results


def run_import_test(strategy: str) -> List[Tuple[str, bool, str]]:
    """Test importing key modules."""
    results = []
    
    # Test core imports
    core_imports = [
        ("spacy", "import spacy"),
        ("transformers", "from transformers import pipeline"),
        ("sentence-transformers", "from sentence_transformers import SentenceTransformer"),
    ]
    
    for name, import_stmt in core_imports:
        try:
            exec(import_stmt)
            results.append((name, True, f"✅ {name} imports successfully"))
        except Exception as e:
            results.append((name, False, f"❌ {name} import failed: {e}"))
    
    # Test strategy-specific imports
    if strategy in ["torch", "both"]:
        try:
            import torch
            results.append(("torch", True, f"✅ torch imports successfully"))
        except Exception as e:
            results.append(("torch", False, f"❌ torch import failed: {e}"))
    
    if strategy in ["tensorflow", "both"]:
        try:
            import tensorflow as tf
            results.append(("tensorflow", True, f"✅ tensorflow imports successfully"))
        except Exception as e:
            results.append(("tensorflow", False, f"❌ tensorflow import failed: {e}"))
    
    return results


def print_results(title: str, results: List[Tuple[str, bool, str]]):
    """Print validation results."""
    print(f"\n{'=' * 70}")
    print(f"{title}")
    print('=' * 70)
    
    for _, _, message in results:
        print(message)
    
    # Count failures
    failures = sum(1 for _, success, _ in results if not success)
    if failures > 0:
        print(f"\n⚠️  {failures} issue(s) found")
    else:
        print(f"\n✅ All checks passed")


def main():
    parser = argparse.ArgumentParser(
        description="Validate FARFAN dependencies installation"
    )
    parser.add_argument(
        "--strategy",
        choices=["torch", "tensorflow", "both"],
        default="torch",
        help="Dependency strategy to validate (default: torch)"
    )
    parser.add_argument(
        "--skip-imports",
        action="store_true",
        help="Skip import tests (useful if not all dependencies are installed)"
    )
    
    args = parser.parse_args()
    
    print("FARFAN 3.3 Dependency Validation")
    print("=" * 70)
    
    # Check Python version
    py_ok, py_msg = check_python_version()
    print(py_msg)
    if not py_ok:
        print("\n❌ Python version incompatible. Exiting.")
        sys.exit(1)
    
    # Validate core dependencies
    core_results = validate_core_dependencies()
    print_results("Core Dependencies", core_results)
    
    # Validate strategy-specific dependencies
    if args.strategy in ["torch", "both"]:
        torch_results = validate_torch_dependencies()
        print_results("PyTorch Dependencies", torch_results)
    
    if args.strategy in ["tensorflow", "both"]:
        tf_results = validate_tensorflow_dependencies()
        print_results("TensorFlow Dependencies", tf_results)
    
    # Check for conflicts
    conflict_results = check_dependency_conflicts()
    print_results("Dependency Conflict Check", conflict_results)
    
    # Run import tests
    if not args.skip_imports:
        import_results = run_import_test(args.strategy)
        print_results("Import Tests", import_results)
    
    print("\n" + "=" * 70)
    print("Validation Complete")
    print("=" * 70)
    print("\nFor more information on resolving conflicts, see:")
    print("  - DEPENDENCY_CONFLICTS.md")
    print("  - requirements-torch.txt")
    print("  - requirements-tensorflow.txt")
    print("  - requirements-both.txt")


if __name__ == "__main__":
    main()
