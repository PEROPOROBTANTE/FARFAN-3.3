# FARFAN CI/CD Pre-Merge Validation Gates

Comprehensive validation pipeline ensuring code quality, determinism, and performance before merging changes.

## Architecture

### Validation Gates (Sequential)

1. **Contract Validation** - Ensures all 413 adapter methods have valid contract.yaml files with JSON Schema compliance
2. **Canary Regression** - Compares SHA-256 hashes of method outputs against baseline expected_hash.txt files
3. **Binding Validation** - Parses execution_mapping.yaml to detect MAPPING_CONFLICT errors
4. **Determinism Verification** - Executes pipeline 3 times with identical seeds, fails on output differences
5. **Performance Regression** - Compares P99 latency against SLA baselines with 10% tolerance
6. **Schema Drift Detection** - Computes SHA-256 of file_manifest, requires migration_plan.md if changed

## Usage

### Run Validation Pipeline

```bash
python3 cicd/run_pipeline.py
```

Exit code 0 if all gates pass, 1 if any gate fails.

### Launch Dashboard

```bash
python3 cicd/dashboard.py
```

Opens web interface at http://localhost:5000

### Dashboard Features

- **Dependency Graph**: Interactive visualization with nodes colored by test status
- **Contract Coverage**: Fraction of methods with valid contracts (N/413)
- **Canary Test Grid**: Pass/fail/rebaseline status indexed by adapter and method
- **Circuit Breaker Status**: Real-time adapter health with color-coded indicators
- **Performance Charts**: Time-series P50/P95/P99 latency and success rate
- **Remediation Suggestions**: Automated fix templates with command snippets

## Validation Gate Details

### 1. Contract Validation

**Purpose**: Ensure every adapter method has a valid contract specification

**Checks**:
- Total method count equals 413
- Each method has contract.yaml file
- JSON Schema validation passes
- Contract includes input/output types

**Failure Codes**:
- `METHOD_COUNT_MISMATCH`: Method count != 413
- `SCHEMA_VALIDATION_FAILED`: Invalid JSON Schema
- `MISSING_CONTRACTS`: Methods lack contracts

**Remediation**:
```bash
python cicd/generate_contracts.py --missing-only
```

### 2. Canary Regression

**Purpose**: Detect unintended changes to method outputs

**Checks**:
- Compute SHA-256 hash of method output
- Compare to baseline expected_hash.txt
- Verify signed changelog entry if mismatch

**Failure Codes**:
- `HASH_DELTA`: Output hash mismatch without changelog

**Remediation**:
```bash
python cicd/rebaseline.py --method <method_name>
```

### 3. Binding Validation

**Purpose**: Validate execution_mapping.yaml integrity

**Checks**:
- All source bindings exist
- Type compatibility between chained methods
- No circular dependencies
- No orphaned bindings

**Failure Codes**:
- `MAPPING_CONFLICT`: Missing source binding
- `TYPE_MISMATCH`: Incompatible types in chain

**Remediation**:
```bash
python cicd/fix_bindings.py --auto-correct
```

### 4. Determinism Verification

**Purpose**: Ensure pipeline produces identical outputs with same seed

**Checks**:
- Run pipeline 3 times with seed=42
- Compare SHA-256 hashes of outputs
- All outputs must be identical

**Failure Codes**:
- `DETERMINISM_FAILURE`: Output differs across runs

**Remediation**:
- Identify non-deterministic operations (timestamps, random without seed, dict iteration)
- Add explicit seeding to random operations
- Use OrderedDict or sorted keys

### 5. Performance Regression

**Purpose**: Prevent performance degradation

**Checks**:
- Measure P50/P95/P99 latency for each adapter
- Compare P99 against baseline SLA
- Fail if current > baseline * 1.10 (10% tolerance)

**Failure Codes**:
- `PERFORMANCE_REGRESSION`: P99 exceeds SLA by >10%

**Remediation**:
```bash
python cicd/profile_adapters.py --optimize
```

### 6. Schema Drift Detection

**Purpose**: Track structural changes to system

**Checks**:
- Compute SHA-256 of file_manifest.json
- Compare to baseline manifest_hash.txt
- Require migration_plan.md if changed

**Failure Codes**:
- `SCHEMA_DRIFT`: Manifest changed without migration plan

**Remediation**:
```bash
python cicd/generate_migration.py
```

## Homeostasis Dashboard

### Dependency Graph

Interactive network visualization:
- **Nodes**: Adapters with method counts
- **Node Colors**:
  - Green: Circuit CLOSED (healthy)
  - Yellow: Circuit HALF_OPEN (recovering)
  - Red: Circuit OPEN (failed)
  - Gray: Circuit ISOLATED
- **Edges**: Binding relationships
- **Edge Colors**:
  - Red: Change impact propagating (failures)
  - Gray: Stable

### Contract Coverage Widget

Displays:
- Total methods: 413
- Valid contracts: N
- Missing contracts: 413 - N
- Coverage percentage: (N/413) * 100%
- Fraction: N/413

### Canary Test Grid

Matrix view:
- **Rows**: Adapters
- **Columns**: Methods
- **Cell Colors**:
  - Green: Pass (hash matches)
  - Red: Fail (hash mismatch)
  - Yellow: Rebaseline needed
  - Gray: Pending/not run

Click failed tests to rebaseline interactively.

### Circuit Breaker Panel

9 adapter cards showing:
- Circuit state (CLOSED/OPEN/HALF_OPEN)
- Success rate percentage
- Recent failure count
- Click to manually reset

### Performance Charts

Bar chart with:
- X-axis: Adapters
- Y-axis: Latency (ms)
- Three series:
  - P50 (blue)
  - P95 (yellow)
  - P99 (red)

Time-series available via adapter detail view.

### Remediation Panel

Auto-generated suggestions:
- Error code classification
- Human-readable fix description
- Exact command to run
- Diff snippets for code changes
- Priority level (high/medium/low)

**Example**:
```
Error: HASH_DELTA
Type: hash_mismatch
Fix: Rebaseline canary test with new expected hash
Command: python cicd/rebaseline.py --method extract_pdm_structure
Priority: medium
```

## File Structure

```
cicd/
├── validation_gates.py       # Six validation gates
├── dashboard.py               # Flask web interface
├── run_pipeline.py            # CLI runner
├── templates/
│   └── dashboard.html         # Dashboard UI
├── rebaseline.py              # Canary rebaseline tool
├── fix_bindings.py            # Binding auto-correction
├── generate_migration.py      # Migration plan generator
├── generate_contracts.py      # Contract generator
├── profile_adapters.py        # Performance profiler
└── README.md                  # This file

baselines/
├── <adapter_name>/
│   └── <method_name>/
│       ├── expected_hash.txt  # SHA-256 baseline
│       └── output.json        # Reference output
└── manifest_hash.txt          # File manifest baseline

contracts/
└── <adapter_name>/
    └── <method_name>.yaml     # Method contract

sla_baselines.json             # Performance baselines
file_manifest.json             # System structure manifest
validation_results.json        # Latest pipeline results
performance_history.json       # Historical metrics
```

## Integration

### Pre-Commit Hook

```bash
#!/bin/bash
python3 cicd/run_pipeline.py
if [ $? -ne 0 ]; then
    echo "❌ Validation gates failed. Fix errors before committing."
    exit 1
fi
```

### GitHub Actions

```yaml
name: CI/CD Validation Gates
on: [pull_request]
jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run validation gates
        run: python3 cicd/run_pipeline.py
      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: validation-results
          path: validation_results.json
```

### GitLab CI

```yaml
stages:
  - validate

validation_gates:
  stage: validate
  script:
    - python3 cicd/run_pipeline.py
  artifacts:
    reports:
      junit: validation_results.json
  only:
    - merge_requests
```

## Configuration

### Tolerance Thresholds

Edit `validation_gates.py`:

```python
class PerformanceRegressionValidator:
    def __init__(self):
        self.tolerance = 0.10  # 10% tolerance for P99
```

### Determinism Runs

```python
class DeterminismValidator:
    def __init__(self):
        self.runs = 3  # Number of verification runs
```

### Circuit Breaker

Edit `orchestrator/circuit_breaker.py`:

```python
CircuitBreaker(
    failure_threshold=5,      # Failures before opening
    recovery_timeout=60.0,    # Seconds before retry
    half_open_max_calls=3     # Test calls in half-open
)
```

## Troubleshooting

### Gate Failures

1. Check `validation_results.json` for detailed errors
2. Review remediation suggestions in dashboard
3. Run suggested fix commands
4. Re-run pipeline: `python3 cicd/run_pipeline.py`

### Dashboard Issues

- **Port conflict**: Change port in `dashboard.py` (default 5000)
- **Missing data**: Run pipeline first to generate baselines
- **Graph not rendering**: Check browser console for vis-network errors

### Performance Regressions

1. Profile slow adapters: `python cicd/profile_adapters.py`
2. Identify bottlenecks in flamegraph
3. Optimize hot paths
4. Update SLA baselines if intentional

## Maintenance

### Update Baselines

After verified changes:

```bash
# Rebaseline all canary tests
python cicd/rebaseline.py --all

# Update performance baselines
python cicd/update_sla_baselines.py

# Update manifest hash
python cicd/update_manifest_hash.py
```

### Add New Adapter

1. Add adapter to `module_adapters.py`
2. Update method count in `validation_gates.py`
3. Generate contracts: `python cicd/generate_contracts.py`
4. Run full pipeline to establish baselines
5. Update circuit breaker config if needed

## Metrics

Dashboard tracks:
- **Contract coverage**: Valid contracts / 413 methods
- **Canary pass rate**: Passed tests / Total tests
- **Circuit health**: Adapters in CLOSED state / 9 total
- **Performance**: P99 latency per adapter
- **Success rate**: Successful calls / Total calls per adapter

## Support

For issues or questions:
1. Check remediation suggestions in dashboard
2. Review validation logs in `validation_results.json`
3. Consult FARFAN documentation
4. Contact integration team

## Cognitive Complexity Analysis

### Purpose

The cognitive complexity checker enforces the SIN_CARRETA doctrine by ensuring code remains understandable, auditable, and maintainable.

### Usage

```bash
# Check entire codebase
python cicd/cognitive_complexity.py --path src/

# Check specific file
python cicd/cognitive_complexity.py --file src/orchestrator/choreographer.py

# Custom threshold
python cicd/cognitive_complexity.py --path src/ --threshold 10

# Generate JSON report
python cicd/cognitive_complexity.py --report complexity_report.json
```

### Complexity Thresholds

| Range | Status | Action |
|-------|--------|--------|
| 0-5 | ✅ Excellent | None |
| 6-10 | ⚠️ Acceptable | Monitor |
| 11-15 | ⚠️ Complex | Plan refactoring |
| 16+ | ❌ Too Complex | Must refactor |

### What It Measures

Cognitive complexity increases with:
- **Control flow structures**: if, while, for, try/except (+1 each + nesting penalty)
- **Nesting depth**: Each level adds to complexity
- **Boolean operators**: Multiple conditions in one expression
- **Recursion**: Recursive calls add complexity

### Why It Matters (SIN_CARRETA Rationale)

High cognitive complexity hinders:

1. **Audit Trail Clarity**: Complex code is harder to trace through execution
2. **Determinism Verification**: More paths = harder to verify all are deterministic
3. **Security Review**: Complex code hides vulnerabilities
4. **Maintenance Cost**: Exponential relationship (2x complexity ≈ 4x cost)
5. **Test Coverage**: More paths require more comprehensive testing

### Refactoring Strategies

**Extract Helper Functions**:
```python
# Before (complexity: 12)
def process(data):
    if data and data.valid:
        if data.ready:
            for item in data.items:
                if item.active:
                    result = transform(item)
                    # ...

# After (complexity: 4 + 3 + 2 = 9 distributed)
def process(data):
    if not should_process(data):
        return None
    return process_items(data.items)
```

**Use Guard Clauses**:
```python
# Before (complexity: 8)
def process(data):
    if data:
        if data.valid:
            if data.ready:
                return compute(data)
    return None

# After (complexity: 4)
def process(data):
    if not data:
        return None
    if not data.valid:
        return None
    if not data.ready:
        return None
    return compute(data)
```

See [SIN_CARRETA Doctrine](../docs/SIN_CARRETA_DOCTRINE.md) for complete guidelines.

