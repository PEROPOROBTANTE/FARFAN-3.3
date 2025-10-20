"""
Pytest configuration and shared fixtures for FARFAN tests
"""
import sys
from pathlib import Path
import pytest

# Add src to Python path for absolute imports
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Add project root to path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


@pytest.fixture
def project_root_path():
    """Return project root path"""
    return project_root


@pytest.fixture
def test_data_path():
    """Return test data directory path"""
    return project_root / "data" / "raw" / "test_samples"


@pytest.fixture
def test_output_path():
    """Return test output directory path"""
    output_path = project_root / "data" / "output" / "test_results"
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


@pytest.fixture
def config_path():
    """Return config directory path"""
    return project_root / "config"


# Pytest markers
def pytest_configure(config):
    """Register custom markers"""
    config.addinivalue_line("markers", "unit: Unit tests for individual components")
    config.addinivalue_line("markers", "integration: Integration tests for component interactions")
    config.addinivalue_line("markers", "e2e: End-to-end tests for full pipeline execution")
    config.addinivalue_line("markers", "slow: Tests that take significant time to run")
