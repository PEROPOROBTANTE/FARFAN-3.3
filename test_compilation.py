#!/usr/bin/env python3
"""
Compilation Test Script for FARFAN 3.0

This script iterates through all Python module files in the project's source directories,
attempts to compile each using Python's ast.parse(), and logs any syntax errors or
compilation failures with specific file paths and error messages.
"""

import ast
import os
import sys
from pathlib import Path
from typing import List, Tuple


def get_python_files(base_dir: str = ".") -> List[Path]:
    """
    Get all Python files in the project, excluding venv and hidden directories.
    
    Args:
        base_dir: Base directory to search from
        
    Returns:
        List of Path objects for Python files
    """
    base_path = Path(base_dir)
    python_files = []
    
    # Directories to exclude
    exclude_dirs = {
        'venv', '.git', '__pycache__', '.pytest_cache', 
        'node_modules', '.tox', 'dist', 'build', 'egg-info',
        '.mypy_cache', '.hypothesis'
    }
    
    for file_path in base_path.rglob("*.py"):
        # Skip files in excluded directories
        if any(excluded in file_path.parts for excluded in exclude_dirs):
            continue
        python_files.append(file_path)
    
    return sorted(python_files)


def compile_file(file_path: Path) -> Tuple[bool, str]:
    """
    Attempt to compile a Python file using ast.parse().
    
    Args:
        file_path: Path to the Python file
        
    Returns:
        Tuple of (success: bool, error_message: str)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
        
        # Try to parse the file
        ast.parse(source_code, filename=str(file_path))
        return True, ""
    
    except SyntaxError as e:
        error_msg = f"SyntaxError: {e.msg} at line {e.lineno}"
        if e.text:
            error_msg += f"\n  {e.text.strip()}"
        return False, error_msg
    
    except UnicodeDecodeError as e:
        return False, f"UnicodeDecodeError: {str(e)}"
    
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)}"


def main():
    """Main function to run compilation tests on all Python files."""
    print("=" * 80)
    print("FARFAN 3.0 - Python Compilation Test")
    print("=" * 80)
    print()
    
    # Get all Python files
    python_files = get_python_files()
    
    if not python_files:
        print("No Python files found!")
        return 1
    
    print(f"Found {len(python_files)} Python files to check")
    print()
    
    # Track results
    passed = []
    failed = []
    
    # Test each file
    for file_path in python_files:
        success, error_msg = compile_file(file_path)
        
        if success:
            passed.append(file_path)
            print(f"✓ PASS: {file_path}")
        else:
            failed.append((file_path, error_msg))
            print(f"✗ FAIL: {file_path}")
            print(f"  Error: {error_msg}")
            print()
    
    # Print summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total files checked: {len(python_files)}")
    print(f"Passed: {len(passed)} ({100 * len(passed) / len(python_files):.1f}%)")
    print(f"Failed: {len(failed)} ({100 * len(failed) / len(python_files):.1f}%)")
    print()
    
    if failed:
        print("Files with compilation errors:")
        for file_path, error_msg in failed:
            print(f"  - {file_path}")
        print()
        return 1
    else:
        print("All files compiled successfully! ✓")
        return 0


if __name__ == "__main__":
    sys.exit(main())
