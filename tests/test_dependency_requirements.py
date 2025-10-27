"""
Test module for validating dependency requirements files.

This ensures that the requirements files are correctly structured and
don't contain conflicting specifications.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pytest


@pytest.fixture
def repo_root() -> Path:
    """Get the repository root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def requirements_files(repo_root: Path) -> Dict[str, Path]:
    """Get all requirements files."""
    return {
        "main": repo_root / "requirements.txt",
        "torch": repo_root / "requirements-torch.txt",
        "tensorflow": repo_root / "requirements-tensorflow.txt",
        "both": repo_root / "requirements-both.txt",
    }


def parse_requirements(file_path: Path) -> Dict[str, str]:
    """
    Parse a requirements file and extract package specifications.
    
    Returns a dict mapping package names to their version specifications.
    """
    packages = {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
            
            # Skip URLs and special formats
            if line.startswith('http') or '@' in line:
                continue
            
            # Parse package==version or package>=version format
            match = re.match(r'^([a-zA-Z0-9\-_]+)\s*([>=<~!]+)\s*([0-9.]+)', line)
            if match:
                package, operator, version = match.groups()
                packages[package.lower()] = f"{operator}{version}"
    
    return packages


@pytest.mark.unit
def test_requirements_files_exist(requirements_files: Dict[str, Path]):
    """Test that all requirements files exist."""
    for name, path in requirements_files.items():
        assert path.exists(), f"{name} requirements file does not exist: {path}"


@pytest.mark.unit
def test_main_requirements_no_dl_libraries(requirements_files: Dict[str, Path]):
    """Test that main requirements.txt does not contain torch or tensorflow."""
    main_packages = parse_requirements(requirements_files["main"])
    
    # These should NOT be in main requirements
    dl_libraries = ["torch", "torchvision", "torchaudio", "tensorflow", "tensorflow-estimator", "keras"]
    
    for lib in dl_libraries:
        assert lib not in main_packages, (
            f"{lib} should not be in main requirements.txt. "
            "It should be in requirements-torch.txt or requirements-tensorflow.txt"
        )


@pytest.mark.unit
def test_torch_requirements_has_torch(requirements_files: Dict[str, Path]):
    """Test that torch requirements file contains torch."""
    torch_packages = parse_requirements(requirements_files["torch"])
    
    assert "torch" in torch_packages, "requirements-torch.txt must contain torch"
    
    # Verify torch version is 2.0.1
    assert torch_packages["torch"] == "==2.0.1", (
        f"torch version should be ==2.0.1, got {torch_packages['torch']}"
    )


@pytest.mark.unit
def test_tensorflow_requirements_has_tensorflow(requirements_files: Dict[str, Path]):
    """Test that tensorflow requirements file contains tensorflow."""
    tf_packages = parse_requirements(requirements_files["tensorflow"])
    
    assert "tensorflow" in tf_packages, "requirements-tensorflow.txt must contain tensorflow"
    
    # Verify tensorflow version is 2.13.0
    assert tf_packages["tensorflow"] == "==2.13.0", (
        f"tensorflow version should be ==2.13.0, got {tf_packages['tensorflow']}"
    )


@pytest.mark.unit
def test_both_requirements_has_both(requirements_files: Dict[str, Path]):
    """Test that both requirements file contains both torch and tensorflow."""
    both_packages = parse_requirements(requirements_files["both"])
    
    assert "torch" in both_packages, "requirements-both.txt must contain torch"
    assert "tensorflow" in both_packages, "requirements-both.txt must contain tensorflow"


@pytest.mark.unit
def test_no_duplicate_packages_in_strategy_files(requirements_files: Dict[str, Path]):
    """Test that strategy files (torch/tensorflow/both) don't duplicate main requirements."""
    main_packages = parse_requirements(requirements_files["main"])
    
    for strategy in ["torch", "tensorflow", "both"]:
        strategy_packages = parse_requirements(requirements_files[strategy])
        
        # Allow some exceptions (numpy, protobuf can be pinned more strictly)
        allowed_duplicates = {"numpy", "protobuf"}
        
        duplicates = set(strategy_packages.keys()) & set(main_packages.keys())
        unexpected_duplicates = duplicates - allowed_duplicates
        
        assert not unexpected_duplicates, (
            f"requirements-{strategy}.txt has duplicate packages from main requirements: "
            f"{unexpected_duplicates}"
        )


@pytest.mark.unit
def test_transformers_in_main_requirements(requirements_files: Dict[str, Path]):
    """Test that transformers is in main requirements, not strategy files."""
    main_packages = parse_requirements(requirements_files["main"])
    
    assert "transformers" in main_packages, (
        "transformers should be in main requirements.txt"
    )
    
    # Verify it's not in strategy files
    for strategy in ["torch", "tensorflow", "both"]:
        strategy_packages = parse_requirements(requirements_files[strategy])
        assert "transformers" not in strategy_packages, (
            f"transformers should not be in requirements-{strategy}.txt"
        )


@pytest.mark.unit
def test_sentence_transformers_in_main_requirements(requirements_files: Dict[str, Path]):
    """Test that sentence-transformers is in main requirements."""
    main_packages = parse_requirements(requirements_files["main"])
    
    assert "sentence-transformers" in main_packages, (
        "sentence-transformers should be in main requirements.txt"
    )


@pytest.mark.unit
def test_pydantic_version_pinned(requirements_files: Dict[str, Path]):
    """Test that pydantic is pinned to v2.x."""
    main_packages = parse_requirements(requirements_files["main"])
    
    assert "pydantic" in main_packages, "pydantic should be in main requirements.txt"
    
    # Check it's version 2.x
    version = main_packages["pydantic"]
    assert version.startswith("==2."), (
        f"pydantic should be version 2.x, got {version}"
    )


@pytest.mark.unit
def test_numpy_version_compatible(requirements_files: Dict[str, Path]):
    """Test that numpy version is 1.24.x for compatibility."""
    # Check main requirements
    main_packages = parse_requirements(requirements_files["main"])
    
    if "numpy" in main_packages:
        numpy_version = main_packages["numpy"]
        # Should be 1.24.x
        assert "1.24" in numpy_version, (
            f"numpy version should be 1.24.x for compatibility, got {numpy_version}"
        )


@pytest.mark.unit
def test_dependency_conflicts_doc_exists(repo_root: Path):
    """Test that DEPENDENCY_CONFLICTS.md documentation exists."""
    doc_path = repo_root / "DEPENDENCY_CONFLICTS.md"
    assert doc_path.exists(), "DEPENDENCY_CONFLICTS.md documentation should exist"
    
    # Check it has substantial content
    with open(doc_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    assert len(content) > 1000, "DEPENDENCY_CONFLICTS.md should have substantial content"
    
    # Check for key sections
    assert "PyTorch" in content, "Documentation should mention PyTorch"
    assert "TensorFlow" in content, "Documentation should mention TensorFlow"
    assert "conflict" in content.lower(), "Documentation should discuss conflicts"


@pytest.mark.unit
def test_validate_dependencies_script_exists(repo_root: Path):
    """Test that validate_dependencies.py script exists."""
    script_path = repo_root / "validate_dependencies.py"
    assert script_path.exists(), "validate_dependencies.py script should exist"
    
    # Check it's executable (has shebang)
    with open(script_path, 'r', encoding='utf-8') as f:
        first_line = f.readline()
    
    assert first_line.startswith('#!'), "validate_dependencies.py should have shebang"


@pytest.mark.unit
def test_requirements_have_warnings(requirements_files: Dict[str, Path]):
    """Test that requirements files have appropriate warnings."""
    # Main requirements should warn about deep learning libraries
    with open(requirements_files["main"], 'r', encoding='utf-8') as f:
        main_content = f.read()
    
    assert "PyTorch" in main_content or "TensorFlow" in main_content, (
        "Main requirements should mention PyTorch/TensorFlow"
    )
    
    # Both requirements should have strong warnings
    with open(requirements_files["both"], 'r', encoding='utf-8') as f:
        both_content = f.read()
    
    assert "ADVERTENCIA" in both_content or "WARNING" in both_content.upper(), (
        "requirements-both.txt should have warnings about conflicts"
    )
    
    assert "NOT RECOMMENDED" in both_content.upper() or "NO RECOMENDADO" in both_content.upper(), (
        "requirements-both.txt should explicitly state it's not recommended"
    )


@pytest.mark.unit
def test_pyproject_toml_has_optional_deps(repo_root: Path):
    """Test that pyproject.toml has optional dependencies defined."""
    pyproject_path = repo_root / "pyproject.toml"
    assert pyproject_path.exists(), "pyproject.toml should exist"
    
    with open(pyproject_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Should have optional dependencies sections
    assert "[project.optional-dependencies]" in content, (
        "pyproject.toml should have [project.optional-dependencies] section"
    )
    
    # Should have torch and tensorflow as optional
    assert "torch" in content.lower(), "pyproject.toml should mention torch"
    assert "tensorflow" in content.lower(), "pyproject.toml should mention tensorflow"
