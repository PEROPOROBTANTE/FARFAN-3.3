# VALIDATION EXECUTION GUIDE

## 1. Prerequisites

### Python Version
- **Required:** Python 3.11 or higher
- Verify with: `python --version` or `python3 --version`

### Dependencies
All dependencies specified in `requirements.txt` with version constraints:

**Core NLP Processing:**
- `spacy==3.7.2` - Spanish language processing
- `spacy-lookups-data==1.0.3` - Lookup tables for spaCy
- `es-core-news-sm @ https://github.com/explosion/spacy-models/releases/download/es_core_news_sm-3.7.0/es_core_news_sm-3.7.0-py3-none-any.whl` - Small Spanish model
- `es-core-news-lg @ https://github.com/explosion/spacy-models/releases/download/es_core_news_lg-3.7.0/es_core_news_lg-3.7.0-py3-none-any.whl` - Large Spanish model
- `stanza==1.5.0` - Advanced NLP for Spanish
- `nltk==3.8.1` - Natural language toolkit
- `pystemmer==2.2.0` - Spanish stemming
- `pyfreeling==0.2` - Spanish lemmatization

**Machine Learning & Transformers:**
- `transformers==4.34.0` - Transformer models (compatible with torch 2.0.1 and tensorflow 2.13.0)
- **Deep Learning Backend** (choose ONE):
  - PyTorch (RECOMMENDED): Install via `pip install -r requirements-torch.txt`
    - `torch==2.0.1` - PyTorch for deep learning
    - `torchvision==0.15.2` - Computer vision utilities
    - `torchaudio==2.0.2` - Audio processing
  - TensorFlow: Install via `pip install -r requirements-tensorflow.txt`
    - `tensorflow==2.13.0` - TensorFlow for deep learning
    - `tensorflow-estimator==2.13.0` - TensorFlow Estimator API
    - `keras==2.13.1` - High-level neural networks API
  - See `DEPENDENCY_CONFLICTS.md` for detailed information
- `sentence-transformers==2.2.2` - Sentence embeddings (uses PyTorch or TensorFlow as backend)
- `beto-uncased @ https://huggingface.co/dccuchile/bert-base-spanish-wwm-uncased` - Spanish BERT model

**Data Processing:**
- `numpy==1.24.3` - Numerical computing
- `pandas==2.0.3` - Data analysis
- `scikit-learn==1.3.0` - Machine learning utilities
- `matplotlib==3.7.2` - Plotting
- `seaborn==0.12.2` - Statistical visualization

**Advanced Text Analysis:**
- `pysentimiento==0.6.6` - Spanish sentiment analysis
- `freeling==4.2` - Morphological analysis for Spanish

**Development & Testing:**
- `pytest` - Testing framework
- `pytest-watch==7.4.2` - Test watching
- `black==23.9.1` - Code formatting
- `flake8==6.1.0` - Linting
- `isort==5.12.0` - Import sorting
- `mypy==1.5.1` - Type checking

**Utilities:**
- `tqdm==4.66.1` - Progress bars
- `requests==2.31.0` - HTTP library
- `python-dotenv==1.0.0` - Environment variables
- `pyyaml==6.0.1` - YAML parsing
- `networkx==3.1` - Graph processing
- `pip-tools==7.3.0` - Dependency management
- `pipdeptree==2.13.0` - Dependency tree visualization

**Documentation:**
- `sphinx==7.2.6` - Documentation generator
- `sphinx-rtd-theme==1.3.0` - ReadTheDocs theme

### System Requirements
- **Minimum RAM:** 16GB (required for full NLP model loading and processing)
- **Disk Space:** 10GB free (for models, dependencies, and processing artifacts)
- **Recommended:** Multi-core CPU for parallel processing, GPU for accelerated inference

---

## 2. Installation

### Step 1: Install Dependencies
```bash
# Install core dependencies
pip install -r requirements.txt

# Choose ONE of the following for deep learning backend:

# Option 1: PyTorch (RECOMMENDED)
pip install -r requirements-torch.txt

# Option 2: TensorFlow
# pip install -r requirements-tensorflow.txt

# Option 3: Both (NOT RECOMMENDED - see DEPENDENCY_CONFLICTS.md)
# pip install -r requirements-both.txt

# Validate installation
python validate_dependencies.py --strategy torch
```

### Step 2: Download Spanish NLP Models
The spaCy Spanish models are included in `requirements.txt`, but if manual installation is needed:

```bash
python -m spacy download es_core_news_sm
python -m spacy download es_core_news_lg
```

### Step 3: CUDA Configuration (Conditional - GPU Acceleration)
If NVIDIA GPU is available and CUDA is installed:

**Verify CUDA availability:**
```bash
# For PyTorch
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA Version: {torch.version.cuda}')"

# For TensorFlow
python -c "import tensorflow as tf; print(f'GPUs Available: {len(tf.config.list_physical_devices(\"GPU\"))}')"
```

**Configure environment variables (if needed):**
```bash
export CUDA_VISIBLE_DEVICES=0  # Use first GPU
export CUDA_LAUNCH_BLOCKING=1  # For debugging
```

**Install GPU-accelerated PyTorch (if not already installed):**
```bash
# For CUDA 11.8
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

**Verify GPU utilization:**
```bash
python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```

---

## 3. Contract Validation

### Directory Navigation
```bash
cd tests/contracts
```

### Run Complete Test Suite
Execute all contract validation tests with verbose output and full traceback:

```bash
pytest -v --tb=long
```

**Flags:**
- `-v` / `--verbose` - Verbose output showing each test result
- `--tb=long` - Full traceback on failures for detailed diagnostics

### Run Individual Test Functions
To execute specific test functions (e.g., question traceability):

```bash
pytest contract_validator.py::test_question_traceability -v
```

**Other example test functions:**
```bash
pytest contract_validator.py::test_adapter_signature_compliance -v
pytest contract_validator.py::test_invocation_patterns -v
pytest contract_validator.py::test_return_type_consistency -v
```

### Coverage Analysis
Combine pytest execution with coverage reporting, requiring 90% threshold for critical paths:

```bash
pytest --cov=orchestrator --cov=orchestrator/module_adapters --cov-report=term-missing --cov-report=html --cov-fail-under=90 -v
```

**Coverage flags:**
- `--cov=orchestrator` - Measure coverage for orchestrator module
- `--cov=orchestrator/module_adapters` - Measure coverage for adapter module
- `--cov-report=term-missing` - Show missing lines in terminal
- `--cov-report=html` - Generate HTML coverage report in `htmlcov/`
- `--cov-fail-under=90` - Fail if coverage below 90%

**View HTML coverage report:**
```bash
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

---

## 4. Expected Outcomes

### Interface Contract Tests
- **Pass Rate Requirement:** 100% - All interface contract tests MUST pass
- **Failure Significance:** Any failure signals a contract violation that breaks the adapter-orchestrator interface contract
- **Remediation Priority:** IMMEDIATE - Do not proceed with any integration, deployment, or further development until all contract violations are resolved

### Signature Compliance
- All adapter methods must match signatures defined in their respective YAML contract files
- Parameter names, types, and order must be exact
- Return types must conform to contract specifications

### Coverage Requirements
**Critical Modules (90% minimum):**
- `orchestrator/` - Core orchestration logic
- `orchestrator/module_adapters.py` - All adapter implementations

**Acceptable Coverage Gaps:**
- Error handling branches for rare edge cases
- Defensive code paths that are difficult to trigger in testing
- Deprecated methods marked for removal

**Coverage Analysis:**
- Review `htmlcov/index.html` for line-by-line coverage details
- Prioritize covering untested critical paths in adapter invocations
- Ensure all adapter method entry points have test coverage

---

## 5. Troubleshooting

### Common Failure Patterns

#### ModuleNotFoundError
**Symptom:**
```
ModuleNotFoundError: No module named 'orchestrator.module_adapters'
```

**Remediation:**
1. Verify `orchestrator/module_adapters.py` exists and contains all adapter classes
2. Check Python path includes repository root: `export PYTHONPATH=$(pwd):$PYTHONPATH`
3. Ensure `__init__.py` exists in `orchestrator/` directory
4. Verify imports in test files use correct module paths

#### Signature Mismatch Errors
**Symptom:**
```
AssertionError: Method signature does not match contract
Expected: analyze_document(text: str, config: Dict) -> Dict
Found: analyze_document(text: str) -> Dict
```

**Remediation:**
1. Consult `invocation_compatibility_matrix.csv` (if available) for documented signature changes
2. Update adapter method signature to match YAML contract specification
3. Update all invocation sites in orchestrator to use correct parameters
4. Re-run contract tests to verify fix

#### Static Method Invocation Violations
**Symptom:**
```
TypeError: analyze_document() missing 1 required positional argument: 'self'
```

**Remediation:**
1. Reference `analyzer_one_invocation_audit.json` correction matrix for proper invocation patterns
2. Verify whether method should be:
   - Instance method: `adapter.method()`
   - Static method: `@staticmethod` + `Adapter.method()`
   - Class method: `@classmethod` + `Adapter.method()`
3. Update orchestrator invocation code accordingly
4. Ensure adapter method has correct decorator

#### Missing Execution Chain Errors
**Symptom:**
```
KeyError: 'analyze_causal_chain' not found in execution mapping
```

**Remediation:**
1. Check `orchestrator/execution_mapping.yaml` for inconsistencies
2. Verify all orchestrator workflow steps reference valid adapter methods
3. Ensure execution mapping includes all required method chains
4. Validate YAML syntax with: `python -c "import yaml; yaml.safe_load(open('orchestrator/execution_mapping.yaml'))"`

#### Import Errors for Spanish NLP Models
**Symptom:**
```
OSError: [E050] Can't find model 'es_core_news_lg'
```

**Remediation:**
1. Re-download spaCy models: `python -m spacy download es_core_news_lg`
2. Verify model installation: `python -m spacy validate`
3. Check model is in correct path: `python -c "import spacy; spacy.load('es_core_news_lg')"`

---

## 6. Regression Testing

### Full Test Suite Execution
Run the complete test suite in verbose mode to detect any breaks in existing functionality:

```bash
pytest tests/ -v --tb=short
```

**Recommended flags:**
- `-v` - Verbose mode showing all test names
- `--tb=short` - Shorter traceback for quicker scanning
- `-x` - Stop on first failure (for rapid debugging)
- `--maxfail=3` - Stop after 3 failures

### Interactive Debugging Sessions
For interactive debugging with output capture:

```bash
pytest tests/ -v -s
```

**Flags:**
- `-s` / `--capture=no` - Disable output capture, show print statements
- `--pdb` - Drop into debugger on failures
- `--trace` - Start debugger at beginning of each test

### When to Run Regression Tests
**ALWAYS run regression tests after:**
1. Any modifications to `orchestrator/` modules
2. Changes to adapter method signatures or implementations
3. Updates to `orchestrator/module_adapters.py`
4. Modifications to execution chains or workflow logic
5. Dependency version updates in `requirements.txt`
6. Configuration changes affecting adapter behavior

**Recommended workflow:**
```bash
# Make changes
git status

# Run full regression suite
pytest tests/ -v

# If failures occur, run with debugging
pytest tests/ -v -s --pdb

# After fixes, re-run to verify
pytest tests/ -v
```

---

## 7. Performance Benchmarks

### Timing Analysis (Conditional)
If performance tests exist in the test suite:

```bash
pytest tests/ -v --durations=10
```

**Flags:**
- `--durations=10` - Show 10 slowest test durations
- `--durations=0` - Show all test durations

### Benchmark Specific Tests
If benchmark tests are marked with `@pytest.mark.benchmark`:

```bash
pytest tests/ -v -m benchmark
```

### Expected Execution Ranges
**300-Question Analysis Workflow:**
- **Baseline:** 45-60 minutes on 16GB RAM, 8-core CPU without GPU
- **GPU-Accelerated:** 20-35 minutes with CUDA-enabled GPU
- **Per-question average:** 8-12 seconds (baseline) / 4-7 seconds (GPU)

**Individual Component Benchmarks:**
- `PolicyProcessorAdapter.process()` - 2-5 seconds per document
- `ContradictionDetectionAdapter.detect()` - 3-7 seconds per policy pair
- `DerekBeachAdapter.audit_causal_implications()` - 5-10 seconds per causal chain
- `EmbeddingPolicyAdapter.semantic_search()` - 0.5-2 seconds per query

### Performance Regression Detection
**Significant deviations from baseline times may indicate:**
1. Performance regressions introduced by code changes
2. Memory leaks or inefficient algorithms
3. Misconfigured NLP models or embedding caches
4. Resource contention or system load issues

**Investigation steps:**
```bash
# Profile specific test
pytest tests/test_adapters.py::test_policy_processing -v --profile

# Run with memory profiling (requires memory-profiler)
python -m memory_profiler tests/test_adapters.py

# Monitor system resources during test execution
htop  # or top on macOS/Linux
```

**Performance regression threshold:**
- **Warning:** >20% increase in execution time
- **Critical:** >50% increase in execution time or failure to complete within timeout

---

## Summary

This guide ensures comprehensive validation of the FARFAN 3.0 orchestrator-adapter architecture through contract testing, regression verification, and performance benchmarking. All tests must pass at 100% before integration or deployment proceeds.

For additional support, consult:
- `tests/contracts/README.md` - Contract system documentation
- `tests/contracts/COMPLIANCE_REPORT.md` - Latest compliance status
- `tests/contracts/VALIDATION_EVIDENCE.md` - Evidence artifacts
- `AGENTS.md` - Development commands and architecture overview
