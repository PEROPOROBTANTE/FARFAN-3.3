# Dependency Tracking Framework

## Overview

A comprehensive dependency tracking and validation system for FARFAN 3.0 that ensures code quality and prevents breaking changes through automated static analysis, refactoring validation, and integration testing.

## Components

### 1. Static Analysis Tool (`dependency_tracker.py`)

Parses Python AST to build a directed graph of all imports and method references.

**Features:**
- Import statement tracking
- Method signature extraction
- Call site analysis with line numbers
- Broken reference detection
- Orphaned call detection
- JSON serialization for baselines

**Usage:**
```bash
# Build dependency graph
python dependency_tracker.py

# Outputs: dependency_graph.json with full project analysis
```

**Example Output:**
```
Dependency Graph Statistics:
  Nodes (files): 42
  Edges: 187
  Method signatures: 1,847
  Call sites: 3,214
```

### 2. Pre-commit Git Hook (`.git/hooks/pre-commit`)

Automatically validates every commit for dependency issues.

**Validations:**
1. Regenerates dependency graph
2. Compares against baseline
3. Blocks commits with broken references
4. Validates method signature changes
5. Ensures dependent files are updated

**Example Output:**
```
🔍 Running dependency graph validation...
✅ Dependency graph validation passed!

🔧 Validating method signature changes...
✅ Refactoring validation passed!

📊 Dependency changes:
   Edges: 187 → 189 (+2)
   Methods: 1847 → 1852 (+5)

✅ All pre-commit checks passed!
```

**Blocked Commit Example:**
```
❌ COMMIT BLOCKED: Signature changes affect unstaged files

Signature change in orchestrator/module_adapters.py:PolicyProcessorAdapter.execute affects 3 call site(s):
  - orchestrator/question_router.py:142 in route_question
  - orchestrator/choreographer.py:89 in execute_chain
  - run_farfan.py:67 in analyze_plan
  → Update these files or stage them with this commit

💡 To fix:
   1. Update the affected files to match new signatures
   2. Stage those files with: git add <files>
   3. Or revert signature changes
```

### 3. Refactoring Validator (`refactoring_validator.py`)

Detects method signature changes and validates all call sites are updated.

**Features:**
- Git diff-based signature comparison
- Cross-references dependency graph
- Identifies all affected callers
- Ensures staged changes are complete
- Prevents cascading breakage

**Usage:**
```bash
# Validate staged changes
python refactoring_validator.py

# Automatically runs in pre-commit hook
```

**Example Detected Issue:**
```python
# OLD: orchestrator/module_adapters.py
def execute(self, method_name: str, args: List[Any]) -> ModuleResult:
    pass

# NEW: orchestrator/module_adapters.py  
def execute(self, method_name: str, args: List[Any], kwargs: Dict[str, Any]) -> ModuleResult:
    pass

# AFFECTED CALLER: orchestrator/question_router.py (NOT STAGED)
result = adapter.execute("process", [text])  # Missing kwargs parameter!
```

### 4. Integration Smoke Tests (`test_integration_smoke.py`)

Comprehensive integration tests with pytest fixtures.

**Test Coverage:**
- ✅ Dependency graph builds successfully
- ✅ Broken reference detection works
- ✅ Full pipeline execution (PDF → Report)
- ✅ All 300 questions produce results
- ✅ Report structure validation
- ✅ Performance benchmarks

**Usage:**
```bash
# Run all tests
pytest test_integration_smoke.py -v

# Run only fast tests (exclude slow)
pytest test_integration_smoke.py -v -m "not slow"

# Run with detailed output
pytest test_integration_smoke.py -v -s
```

**Example Output:**
```
test_integration_smoke.py::TestDependencyGraph::test_dependency_graph_builds PASSED
test_integration_smoke.py::TestDependencyGraph::test_broken_references_detection PASSED
test_integration_smoke.py::TestPipelineComponents::test_config_loads PASSED
test_integration_smoke.py::TestPipelineComponents::test_report_assembler_initializes PASSED
test_integration_smoke.py::TestFullPipeline::test_full_pipeline_execution PASSED
  ✅ Pipeline executed successfully
     Questions answered: 287
     Clusters: 4
     Overall score: 72.45

test_integration_smoke.py::TestFullPipeline::test_300_questions_produce_results PASSED
  ✅ Question coverage:
     Total questions: 300
     Non-null results: 287
     Coverage: 95.7%

test_integration_smoke.py::TestFullPipeline::test_report_structure_validation PASSED
  ✅ Report structure validated

========================== 8 passed in 45.23s ===========================
```

### 5. Auto-test Runner (`watch_tests.sh`)

Monitors file changes and automatically runs tests.

**Features:**
- Uses pytest-watch (preferred)
- Fallback to inotifywait
- Ignores venv, __pycache__, .git
- Runs fast tests only by default
- Clear console between runs

**Usage:**
```bash
# Start watching
./watch_tests.sh

# Output:
🔍 Starting pytest-watch...
   Monitoring: *.py files
   Running: integration smoke tests

[pytest output...]

✅ Tests complete. Watching for changes...
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Verify pytest-watch installed
ptw --version

# Make scripts executable
chmod +x .git/hooks/pre-commit
chmod +x watch_tests.sh

# Generate initial baseline
python dependency_tracker.py
mv dependency_graph.json dependency_graph_baseline.json
```

## Workflow

### Development Workflow

1. **Start test watcher:**
   ```bash
   ./watch_tests.sh
   ```

2. **Make code changes:**
   - Edit Python files
   - Tests run automatically on save
   - Immediate feedback on breakage

3. **Commit changes:**
   ```bash
   git add <files>
   git commit -m "Your message"
   
   # Pre-commit hook runs automatically:
   # ✅ Dependency validation
   # ✅ Refactoring validation
   # ✅ Baseline update
   ```

### Refactoring Workflow

**Scenario:** You want to add a parameter to `PolicyProcessorAdapter.execute()`

1. **Change the signature:**
   ```python
   # OLD
   def execute(self, method_name: str, args: List[Any]) -> ModuleResult:
   
   # NEW
   def execute(self, method_name: str, args: List[Any], kwargs: Dict[str, Any] = None) -> ModuleResult:
   ```

2. **Stage the change:**
   ```bash
   git add orchestrator/module_adapters.py
   ```

3. **Try to commit:**
   ```bash
   git commit -m "Add kwargs parameter to execute"
   
   # Pre-commit hook detects affected files:
   ❌ COMMIT BLOCKED: Signature changes affect unstaged files
   
   orchestrator/question_router.py:142
   orchestrator/choreographer.py:89
   run_farfan.py:67
   ```

4. **Update all callers:**
   ```python
   # orchestrator/question_router.py
   result = adapter.execute("process", [text], kwargs={})
   
   # orchestrator/choreographer.py
   result = adapter.execute(method, args, kwargs=params)
   
   # run_farfan.py
   result = adapter.execute("analyze", [plan_text], kwargs=options)
   ```

5. **Stage updated files:**
   ```bash
   git add orchestrator/question_router.py orchestrator/choreographer.py run_farfan.py
   ```

6. **Commit successfully:**
   ```bash
   git commit -m "Add kwargs parameter to execute and update all callers"
   
   ✅ Dependency graph validation passed!
   ✅ Refactoring validation passed!
   ✅ All pre-commit checks passed!
   ```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Dependency Framework                      │
└─────────────────────────────────────────────────────────────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
            ▼                ▼                ▼
    ┌──────────────┐ ┌─────────────┐ ┌──────────────┐
    │   Static     │ │ Refactoring │ │ Integration  │
    │   Analysis   │ │  Validator  │ │    Tests     │
    └──────┬───────┘ └──────┬──────┘ └──────┬───────┘
           │                │                │
           ▼                ▼                ▼
    ┌──────────────────────────────────────────────┐
    │          Pre-commit Git Hook                 │
    │  1. Rebuild dependency graph                 │
    │  2. Detect broken references                 │
    │  3. Validate signature changes               │
    │  4. Block commit if issues found             │
    └──────────────────────────────────────────────┘
```

## Dependency Graph Structure

```json
{
  "nodes": ["file1.py", "file2.py", ...],
  "edges": [
    {
      "source_file": "orchestrator/question_router.py",
      "target_file": "orchestrator/module_adapters.py",
      "edge_type": "import",
      "metadata": {"imported_names": ["PolicyProcessorAdapter"], "line": 12}
    },
    {
      "source_file": "orchestrator/question_router.py",
      "target_file": "orchestrator/module_adapters.py",
      "edge_type": "method_call",
      "metadata": {
        "callee_class": "PolicyProcessorAdapter",
        "callee_method": "execute",
        "line": 142,
        "caller_method": "route_question"
      }
    }
  ],
  "method_signatures": {
    "orchestrator/module_adapters.py": [
      {
        "module": "orchestrator/module_adapters.py",
        "class_name": "PolicyProcessorAdapter",
        "method_name": "execute",
        "args": ["self", "method_name", "args", "kwargs"],
        "kwargs": [],
        "return_annotation": "ModuleResult",
        "decorators": [],
        "line_number": 89
      }
    ]
  },
  "call_sites": {
    "orchestrator/question_router.py": [
      {
        "caller_file": "orchestrator/question_router.py",
        "caller_class": "QuestionRouter",
        "caller_method": "route_question",
        "caller_line": 142,
        "callee_module": "adapter",
        "callee_class": "PolicyProcessorAdapter",
        "callee_method": "execute",
        "args_signature": ["method_name", "args"],
        "kwargs_signature": []
      }
    ]
  }
}
```

## Configuration

### pytest.ini

```ini
[pytest]
minversion = 6.0
testpaths = .
python_files = test_*.py

markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests

addopts = -v --tb=short --strict-markers
log_cli = true
log_cli_level = INFO
```

### .gitignore

```
# Dependency tracking
dependency_graph.json
dependency_graph_baseline.json
dependency_graph_current.json

# Testing
.pytest_cache/
test_data/
```

## Best Practices

1. **Always run tests before committing:**
   ```bash
   pytest test_integration_smoke.py -v
   ```

2. **Keep dependency graph baseline updated:**
   - Automatically updated by pre-commit hook
   - Manual update: `python dependency_tracker.py && mv dependency_graph.json dependency_graph_baseline.json`

3. **When refactoring methods:**
   - Change signature
   - Update ALL callers
   - Stage everything together
   - Let pre-commit hook validate

4. **Monitor test watcher during development:**
   - Run `./watch_tests.sh` in separate terminal
   - See immediate feedback on changes
   - Catch issues before commit

5. **Review dependency changes:**
   - Pre-commit hook shows +/- in edges and methods
   - Large changes may indicate architectural issues
   - Consider splitting large refactorings

## Troubleshooting

### Pre-commit hook not running

```bash
# Check hook is executable
ls -l .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit

# Test manually
.git/hooks/pre-commit
```

### False positives in broken references

Some dynamic imports or runtime calls may be flagged incorrectly:

```python
# Add to exclusion list in dependency_tracker.py
EXCLUDED_METHODS = ['__init__', '__str__', '__repr__', 'your_dynamic_method']
```

### Test watcher not detecting changes

```bash
# Verify pytest-watch installed
pip install pytest-watch

# Try fallback script
./watch_tests.sh
```

### Baseline out of sync

```bash
# Regenerate baseline
python dependency_tracker.py
mv dependency_graph.json dependency_graph_baseline.json
```

## Metrics and Monitoring

Track these metrics over time:

- **Graph size:** Number of nodes and edges
- **Method count:** Total methods in codebase
- **Call site density:** Calls per method (coupling metric)
- **Broken references:** Should always be 0
- **Test coverage:** Percentage of questions with results
- **Test duration:** Pipeline execution time

## Future Enhancements

- [ ] Visualize dependency graph with NetworkX/Graphviz
- [ ] Detect circular dependencies
- [ ] Calculate code complexity metrics
- [ ] Generate refactoring suggestions
- [ ] IDE integration (VS Code extension)
- [ ] CI/CD pipeline integration
- [ ] Automated documentation generation from call graphs
- [ ] Performance profiling integration

## Related Documentation

- [AGENTS.md](AGENTS.md) - Agent guide with commands
- [ANALISIS_REPO.md](ANALISIS_REPO.md) - Repository analysis
- [EXECUTION_MAPPING_MASTER.md](EXECUTION_MAPPING_MASTER.md) - Execution mappings

## License

Part of FARFAN 3.0 - Colombian Municipal Development Plan Analysis System
