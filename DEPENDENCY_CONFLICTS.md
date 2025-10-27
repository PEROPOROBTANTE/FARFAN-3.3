# Dependency Conflict Resolution Guide

## Overview

This document explains the dependency conflicts in FARFAN 3.3 and how to resolve them.

## Direct Conflicts Identified

### 1. Deep Learning Libraries Conflict

**Problem:** PyTorch and TensorFlow have overlapping and conflicting dependencies:

- **torch==2.0.1** and **tensorflow==2.13.0** both depend on:
  - Different versions of `numpy`
  - Different versions of `protobuf`
  - Different versions of `typing-extensions`
  - Binary libraries that may conflict (CUDA, cuDNN)

**Impact:**
- Installation failures due to version resolution conflicts
- Runtime errors from incompatible binary dependencies
- High memory usage when both libraries are loaded
- Disk space waste from duplicate dependencies

**Resolution:** Use separate requirements files:
- `requirements-torch.txt` - For PyTorch-only projects
- `requirements-tensorflow.txt` - For TensorFlow-only projects
- `requirements-both.txt` - For combined use (NOT RECOMMENDED)

### 2. Pydantic v2.x Compatibility

**Problem:** Pydantic v2.5.0 has breaking changes from v1.x:

- Some libraries (e.g., older FastAPI versions, SQLModel) expect pydantic v1.x
- API changes in validation and model definitions
- Performance characteristics differ

**Impact:**
- Type validation errors
- Serialization/deserialization failures
- Plugin compatibility issues

**Resolution:**
- Pin `pydantic==2.5.0` and `pydantic-core==2.14.5`
- Ensure all pydantic-dependent libraries support v2.x
- If issues arise, downgrade to `pydantic<2.0`

### 3. NumPy Version Conflicts

**Problem:** Different numpy versions required by:
- torch==2.0.1 → numpy>=1.24.3,<1.25.0
- tensorflow==2.13.0 → numpy>=1.24.3,<1.25.0
- pandas==2.0.3 → numpy>=1.21.0

**Resolution:** Pin `numpy==1.24.3` which is compatible with all libraries

## Transitive Dependency Conflicts

### 1. Protobuf Version Conflicts

**Affected by:**
- transformers
- tensorflow
- torch (indirectly)

**Resolution:** Pin `protobuf>=3.20.2,<4.0.0`

### 2. Typing Extensions

**Affected by:**
- torch
- tensorflow
- pydantic

**Resolution:** Pin `typing-extensions>=4.5.0`

### 3. GRPC and Binary Dependencies

**Affected by:**
- tensorflow (requires grpcio)
- May conflict with other networking libraries

**Resolution:** Pin `grpcio>=1.48.0,<1.49.0` when using TensorFlow

## Installation Strategies

### Strategy 1: PyTorch Only (RECOMMENDED for FARFAN)

```bash
# Create clean environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-torch.txt

# Verify installation
pip check
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

**Benefits:**
- Smaller installation size
- Faster installation
- Better compatibility with sentence-transformers
- Recommended by transformers library

### Strategy 2: TensorFlow Only

```bash
# Create clean environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-tensorflow.txt

# Verify installation
pip check
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"
```

**Benefits:**
- Good for TensorFlow-specific models
- Better keras integration

### Strategy 3: Both Libraries (NOT RECOMMENDED)

```bash
# Create clean environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-both.txt

# Verify installation (may show conflicts)
pip check
pipdeptree | grep -E "(torch|tensorflow)"

# Test imports
python -c "import torch; import tensorflow as tf; print('Both libraries loaded')"
```

**Risks:**
- Dependency resolution failures
- Runtime conflicts
- High memory usage
- Difficult troubleshooting

### Strategy 4: Separate Virtual Environments (BEST PRACTICE)

Create two separate environments:

```bash
# PyTorch environment
python3 -m venv venv-torch
source venv-torch/bin/activate
pip install -r requirements.txt -r requirements-torch.txt

# TensorFlow environment  
python3 -m venv venv-tensorflow
source venv-tensorflow/bin/activate
pip install -r requirements.txt -r requirements-tensorflow.txt
```

## Version Compatibility Matrix

| Library | PyTorch Version | TensorFlow Version | Compatible? |
|---------|----------------|-------------------|-------------|
| transformers==4.34.0 | 2.0.1 | 2.13.0 | ✅ Yes |
| sentence-transformers==2.2.2 | 2.0.1 | 2.13.0 | ✅ Yes |
| numpy==1.24.3 | 2.0.1 | 2.13.0 | ✅ Yes |
| protobuf==3.20.3 | 2.0.1 | 2.13.0 | ✅ Yes |
| pydantic==2.5.0 | N/A | N/A | ✅ Yes |

## Python Version Requirements

| Library | Min Python | Max Python | Notes |
|---------|-----------|-----------|-------|
| torch==2.0.1 | 3.8 | 3.11 | 3.12 support in torch 2.1+ |
| tensorflow==2.13.0 | 3.8 | 3.11 | 3.12 support in tensorflow 2.15+ |
| transformers==4.34.0 | 3.8 | 3.11 | Generally follows PyTorch support |
| pydantic==2.5.0 | 3.7 | 3.12 | Full 3.12 support |

**IMPORTANT:** 
- **Python 3.12** may have compatibility issues with torch==2.0.1 and tensorflow==2.13.0
- **RECOMMENDED:** Use Python 3.10 or 3.11 for production
- If you must use Python 3.12:
  - Consider upgrading to torch>=2.1.0 (requires updating other dependencies)
  - Consider upgrading to tensorflow>=2.15.0 (requires updating other dependencies)
  - Test thoroughly before deploying to production

## Troubleshooting

### Problem: "pip install" fails with dependency conflicts

**Solution 1:** Use the appropriate requirements file
```bash
pip install -r requirements.txt -r requirements-torch.txt
```

**Solution 2:** Force reinstall with --force-reinstall
```bash
pip install --force-reinstall -r requirements-torch.txt
```

**Solution 3:** Use --no-deps for problematic packages
```bash
pip install --no-deps torch==2.0.1
```

### Problem: ImportError at runtime

**Symptom:** `ImportError: cannot import name 'xxx' from 'yyy'`

**Solution:** Check for conflicting versions
```bash
pip list | grep -E "(torch|tensorflow|numpy|protobuf)"
pip check
```

### Problem: Version conflicts with other packages

**Solution:** Use pip-tools to generate compatible versions
```bash
pip install pip-tools
pip-compile requirements.in requirements-torch.in -o requirements-compiled.txt
```

### Problem: Binary conflicts (CUDA, cuDNN)

**Solution:** Ensure only one version of CUDA libraries
```bash
# Check for multiple versions
find $VIRTUAL_ENV -name "libcudart*"
find $VIRTUAL_ENV -name "libcudnn*"

# Reinstall with correct CUDA version
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
```

## Dependency Management Tools

### 1. validate_dependencies.py (NEW)
Custom validation script for FARFAN dependencies:
```bash
# Validate PyTorch installation
python validate_dependencies.py --strategy torch

# Validate TensorFlow installation
python validate_dependencies.py --strategy tensorflow

# Validate both (if installed)
python validate_dependencies.py --strategy both

# Skip import tests
python validate_dependencies.py --skip-imports
```

### 2. pip check
Verifies installed packages have compatible dependencies:
```bash
pip check
```

### 3. pipdeptree
Visualizes dependency tree:
```bash
pipdeptree
pipdeptree -p torch
pipdeptree -p tensorflow
pipdeptree --reverse -p numpy  # Shows what depends on numpy
```

### 4. pip-tools
Manages requirements with constraints:
```bash
pip-compile requirements.in -o requirements.txt
pip-sync requirements.txt
```

## Best Practices

1. **Use Virtual Environments:** Always use venv or conda environments
2. **Pin Versions:** Explicitly pin all direct and critical transitive dependencies
3. **Document Conflicts:** Keep this document updated with new conflicts
4. **Test Before Deploy:** Run `pip check` and test imports before deployment
5. **Separate Environments:** Use different environments for PyTorch and TensorFlow
6. **Regular Updates:** Review and update dependencies quarterly
7. **Security Patches:** Monitor for security vulnerabilities in dependencies

## Version Update Strategy

When updating deep learning libraries:

1. **Test in isolated environment first**
2. **Check compatibility matrices** on official documentation
3. **Update requirements files** with new versions
4. **Run full test suite** to verify compatibility
5. **Update this documentation** with findings
6. **Create migration guide** if breaking changes

## Contact and Support

For dependency-related issues:
1. Check this document first
2. Run diagnostic commands (`pip check`, `pipdeptree`)
3. Consult package documentation
4. File an issue with full error output

## References

- PyTorch Compatibility: https://pytorch.org/get-started/locally/
- TensorFlow Compatibility: https://www.tensorflow.org/install/pip
- Transformers Compatibility: https://huggingface.co/docs/transformers/installation
- Sentence-Transformers: https://www.sbert.net/docs/installation.html
- Pydantic v2 Migration: https://docs.pydantic.dev/latest/migration/
