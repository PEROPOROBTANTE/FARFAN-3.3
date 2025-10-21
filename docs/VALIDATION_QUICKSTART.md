# Quick Start Guide: CI/CD Validation & Enforcement Tools

This guide shows how to use the FARFAN 3.0 determinism and contract enforcement tools.

## Before Committing

Always run the full validation pipeline:

```bash
python cicd/run_pipeline.py
```

This runs all 6 validation gates and reports results.

## Common Workflows

### 1. Adding a New Adapter Method

```bash
# 1. Implement your method in src/orchestrator/module_adapters.py
# 2. Generate a contract for it
python cicd/generate_contracts.py --missing-only

# 3. Edit the generated contract.yaml to add proper specifications
# 4. Run validation
python cicd/run_pipeline.py
```

### 2. Checking Code Complexity

Before submitting a PR, check cognitive complexity:

```bash
# Check entire src/ directory
python cicd/cognitive_complexity.py --path src/

# Check a specific file
python cicd/cognitive_complexity.py --file src/orchestrator/choreographer.py

# Save report for review
python cicd/cognitive_complexity.py --path src/ --report complexity_report.json
```

If you see functions with complexity > 15, refactor them before committing.

### 3. Fixing Binding Issues

When you see `MAPPING_CONFLICT` errors:

```bash
# See what's wrong
python cicd/fix_bindings.py --validate-only

# Auto-detect and log issues (dry run)
python cicd/fix_bindings.py --dry-run

# Note: Auto-correct is conservative - review changes manually
```

### 4. Updating Canary Baselines

After intentionally changing adapter output:

```bash
# 1. Add changelog entry
echo "## Method: teoria_cambio - Reason: Updated algorithm for accuracy" >> CHANGELOG_SIGNED.md

# 2. Rebaseline
python cicd/rebaseline.py --method teoria_cambio

# 3. Verify
python cicd/rebaseline.py --verify
```

### 5. Performance Profiling

To profile adapter performance:

```bash
# Profile all adapters
python cicd/profile_adapters.py --all

# Get optimization suggestions
python cicd/profile_adapters.py --all --optimize

# Save as new baselines
python cicd/profile_adapters.py --all --save-baselines
```

### 6. Handling Schema Changes

When file_manifest.json changes:

```bash
# Generate migration plan
python cicd/generate_migration.py --output migration_plan.md

# Review and edit migration_plan.md
# Then run validation again - it will pass with migration plan present
python cicd/run_pipeline.py
```

## Understanding Validation Results

### Exit Codes

- `0`: All gates passed ✅
- `1`: At least one gate failed ❌

### Reading Results

Check `validation_results.json` for detailed information:

```bash
cat validation_results.json | python -m json.tool
```

Each gate reports:
- `status`: "PASSED", "FAILED", or "ERROR"
- `passed`: boolean
- `errors`: list of error messages
- `warnings`: list of warnings
- `metrics`: detailed metrics
- `execution_time`: time taken in seconds

### Common Errors

**METHOD_COUNT_MISMATCH**
```
Expected 413 methods, found 13
```
**Fix**: Update the expected count in `cicd/validation_gates.py` or add missing methods.

**MAPPING_CONFLICT**
```
66 bindings have missing source
```
**Fix**: Run `python cicd/fix_bindings.py --validate-only` to see details.

**HASH_DELTA**
```
3 methods have output hash mismatches
```
**Fix**: Add changelog entry and rebaseline with `python cicd/rebaseline.py`.

**DETERMINISM_FAILURE**
```
Found 2 differences across 3 runs
```
**Fix**: Check for:
- Unseeded random number generation
- System time dependencies
- Unsorted dictionary iteration
- Non-deterministic external APIs

**PERFORMANCE_REGRESSION**
```
5 adapters exceed P99 SLA by >10%
```
**Fix**: Profile slow adapters with `python cicd/profile_adapters.py --optimize`.

## CI/CD Integration

### GitHub Actions

The validation pipeline runs automatically on PRs. Results appear as:
- PR comment with summary
- Workflow run logs
- Artifacts (validation_results.json)

### Local Pre-Commit Hook

Add to `.git/hooks/pre-commit`:

```bash
#!/bin/bash
echo "Running FARFAN validation gates..."
python cicd/run_pipeline.py
if [ $? -ne 0 ]; then
    echo "❌ Validation failed. Fix errors before committing."
    exit 1
fi
echo "✅ All validation gates passed"
```

Make executable:
```bash
chmod +x .git/hooks/pre-commit
```

## Best Practices

### 1. Run Validation Early and Often

Don't wait until PR time - run validation during development:
```bash
# Quick check during development
python cicd/cognitive_complexity.py --file path/to/your/file.py

# Full check before committing
python cicd/run_pipeline.py
```

### 2. Keep Complexity Low

Aim for cognitive complexity < 10:
- Extract helper functions
- Use guard clauses
- Simplify boolean logic
- Avoid deep nesting

### 3. Document Rationale

When you must exceed thresholds, document why:
```python
def complex_function():
    """Process complex business logic.
    
    Cognitive Complexity: 18
    Rationale: Implements state machine with 8 states and 
    transition validation. Refactoring would obscure the
    state machine logic and make it harder to verify correctness.
    
    TODO: Consider extracting to separate StateValidator class.
    """
    # ... implementation
```

### 4. Review Generated Contracts

Always review and enhance auto-generated contracts:
```yaml
# Generated (minimal)
input:
  type: object
  properties:
    policy_text:
      type: string

# Enhanced (better)
input:
  type: object
  properties:
    policy_text:
      type: string
      minLength: 100
      maxLength: 1000000
      description: "Policy document text in Spanish"
  required:
    - policy_text
```

## Troubleshooting

### "No module named 'numpy'"

Install required dependencies:
```bash
pip install -r requirements.txt
```

### "File not found: orchestrator/module_adapters.py"

Ensure you're running from the project root:
```bash
cd /path/to/FARFAN-3.3
python cicd/run_pipeline.py
```

### Validation is slow

Skip slow tests during development:
```bash
# Most gates run in < 1 second
# Determinism gate may take longer with actual pipeline runs
```

### Dashboard won't start

Check if port 5000 is available:
```bash
# Use different port
DASHBOARD_PORT=8080 python cicd/dashboard.py
```

## Further Reading

- [SIN_CARRETA Doctrine](../docs/SIN_CARRETA_DOCTRINE.md) - Complete philosophy and guidelines
- [CONTRIBUTING.md](../CONTRIBUTING.md) - Development standards
- [cicd/README.md](cicd/README.md) - Detailed gate documentation

## Questions?

- Check existing documentation first
- Review validation logs in `validation_results.json`
- Look at remediation suggestions in the validation output
- Contact the development team
