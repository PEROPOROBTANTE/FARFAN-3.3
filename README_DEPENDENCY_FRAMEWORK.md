# Dependency Tracking Framework - Quick Start

## Files Created

1. **dependency_tracker.py** (563 lines) - AST-based dependency graph builder
2. **refactoring_validator.py** (398 lines) - Method signature change validator  
3. **test_integration_smoke.py** (496 lines) - Full pipeline integration tests
4. **.git/hooks/pre-commit** (101 lines) - Automated pre-commit validation
5. **watch_tests.sh** (65 lines) - Auto-test runner with file monitoring
6. **DEPENDENCY_FRAMEWORK.md** (496 lines) - Complete documentation
7. **pytest.ini** - pytest configuration
8. **.gitignore** - Updated with dependency tracking artifacts

## Quick Commands

```bash
# Build dependency graph
python dependency_tracker.py

# Run integration tests  
pytest test_integration_smoke.py -v

# Run fast tests only
pytest test_integration_smoke.py -v -m "not slow"

# Watch for changes and auto-test
./watch_tests.sh

# Test pre-commit hook manually
.git/hooks/pre-commit
```

## What It Does

### 1. Static Analysis (dependency_tracker.py)
- Parses all Python files using AST
- Extracts imports, method signatures, call sites
- Builds directed dependency graph
- Detects broken references and orphaned calls
- Stores baseline in JSON

### 2. Pre-commit Hook
- Runs automatically on `git commit`
- Regenerates dependency graph
- Compares against baseline
- Validates method signature changes
- Blocks commit if issues found

### 3. Refactoring Validator
- Detects signature changes in staged files
- Cross-references dependency graph
- Identifies all affected call sites
- Ensures dependent files are updated
- Prevents cascading breakage

### 4. Integration Tests
- Full pipeline execution tests
- 300-question coverage validation
- Report structure validation
- Performance benchmarks
- Pytest fixtures for deterministic testing

## Example Workflow

```bash
# 1. Make changes to method signature
vim orchestrator/module_adapters.py

# 2. Try to commit
git add orchestrator/module_adapters.py
git commit -m "Add parameter"

# Hook detects issue:
# ❌ COMMIT BLOCKED: Affects 3 unstaged files

# 3. Update all callers
vim orchestrator/question_router.py
vim orchestrator/choreographer.py  

# 4. Stage everything together
git add orchestrator/*.py
git commit -m "Add parameter and update callers"

# ✅ All checks pass!
```

## Status

✅ All components created and syntax-validated
✅ Pre-commit hook installed and executable
✅ Test framework configured with pytest
✅ File watcher script ready
✅ Comprehensive documentation provided

For full documentation, see DEPENDENCY_FRAMEWORK.md
