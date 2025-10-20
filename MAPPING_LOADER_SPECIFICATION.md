# Mapping Loader Specification - FARFAN 3.0

## Overview

The **Mapping Loader** (`orchestrator/mapping_loader.py`) is the execution integrity layer that sits between contract definitions and canary-based regression detection. It ensures the 300-question routing specified in `execution_mapping.yaml` is structurally sound before any test execution occurs.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Application Startup                        │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│         MappingStartupValidator.validate_at_startup()       │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  YAMLMappingLoader                                   │   │
│  │  ├─ Load execution_mapping.yaml                      │   │
│  │  ├─ Parse adapter registry                           │   │
│  │  ├─ Parse execution chains (300 questions)           │   │
│  │  ├─ Build DAGs (binding dependencies)                │   │
│  │  └─ Validate:                                         │   │
│  │     ├─ Bindings (one producer per source)            │   │
│  │     ├─ Types (producer/consumer compatibility)       │   │
│  │     └─ No circular dependencies                      │   │
│  └──────────────────────────────────────────────────────┘   │
└───────────────────────┬─────────────────────────────────────┘
                        │
                ┌───────┴───────┐
                │  Valid?       │
                └───┬───────┬───┘
                    │       │
              YES   │       │   NO
                    │       │
                    ▼       ▼
            ┌────────┐   ┌──────────────────────────┐
            │  Run   │   │ FAIL FAST with           │
            │  App   │   │ MAPPING_CONFLICT         │
            └────────┘   │ diagnostics              │
                         └──────────────────────────┘
```

## Core Components

### 1. YAMLMappingLoader

**Purpose**: Parse and validate execution mapping configuration

**Responsibilities**:
- Load `execution_mapping.yaml`
- Extract adapter registry (9 adapters)
- Extract execution chains (300 questions across dimensions D1-D10)
- Build execution DAGs using binding names as edges
- Validate structural integrity

**Key Methods**:
```python
loader = YAMLMappingLoader(
    mapping_path="orchestrator/execution_mapping.yaml",
    contract_registry=None  # Optional ContractRegistry instance
)

# Load and validate (raises MappingValidationError on conflict)
loader.load_and_validate()

# Query interface
chain = loader.get_execution_chain("D1_INSUMOS.Q1_Baseline_Identification")
dag = loader.get_execution_dag("D1_INSUMOS.Q1_Baseline_Identification")
adapter_info = loader.get_adapter_info("teoria_cambio")
bindings = loader.get_all_bindings()
stats = loader.get_statistics()
```

### 2. DAG Construction

**Binding-Based Edges**: Each execution chain step produces bindings (outputs) and consumes bindings (inputs via `args.source`). The DAG builder creates edges from producer steps to consumer steps:

```yaml
# Example from execution_mapping.yaml
execution_chain:
  - step: 1
    adapter: policy_segmenter
    method: segment
    returns:
      type: List[Dict[str, Any]]
      binding: document_segments  # PRODUCES binding
      
  - step: 2
    adapter: analyzer_one
    method: analyze
    args:
      - name: segments
        type: List[Dict]
        source: document_segments  # CONSUMES binding
```

This creates an edge: `step_1 → step_2` with label `document_segments`.

### 3. Validation Rules

#### 3.1 Binding Validation

**Rule**: Each `args.source` reference must have **exactly one** producer within the same question's execution chain.

**Violations**:
- **DUPLICATE_PRODUCER**: Multiple steps produce the same binding
- **MISSING_PRODUCER**: A step references a binding that no prior step produces

**Example Conflict**:
```
MAPPING_CONFLICT: DUPLICATE_PRODUCER
Questions Affected: D1_INSUMOS.Q2_Gap_Analysis
Description: Binding 'evidence_scores' has 2 producers (steps: [1, 3])
Affected Bindings: evidence_scores

Remediation:
Remove duplicate bindings. Each binding must have exactly one producer.
  Affected steps: [1, 3]
  Solution: Use different binding names or merge steps.
```

#### 3.2 Type Validation

**Rule**: Producer output types must match consumer input types (checked via ContractRegistry).

**Type Compatibility**:
- Exact match: `str` == `str`
- Generic match: `List[Dict]` compatible with `List`
- Any wildcard: `Any` compatible with any type

**Example Conflict**:
```
MAPPING_CONFLICT: TYPE_MISMATCH
Questions Affected: D2_PROCESOS.Q3_Process_Quality
Description: Type mismatch for binding 'process_dag' at step 4

Type Mismatch Details:
  binding: process_dag
  producer_type: nx.DiGraph
  consumer_type: Dict
  consumer_step: 4
  consumer_adapter: financial_viability
  consumer_method: calculate_quality_score

Remediation:
Fix type mismatch:
  Producer returns: nx.DiGraph
  Consumer expects: Dict
  Solution: Convert type in producer or update consumer signature.
```

#### 3.3 Circular Dependency Detection

**Rule**: Execution DAGs must be acyclic (no step can transitively depend on itself).

**Example Conflict**:
```
MAPPING_CONFLICT: CIRCULAR_DEPENDENCY
Questions Affected: D3_PRODUCTOS.Q2_Output_Dependencies
Description: Circular dependency detected in execution chain
Affected Bindings: intermediate_result, final_result

Detected cycles:
  step_1_adapter_a.method_x -> step_2_adapter_b.method_y -> step_1_adapter_a.method_x

Remediation:
Break circular dependency:
  Solution: Reorder steps or remove circular binding references.
```

### 4. Contract Registry Integration

**Interface**: The loader integrates with an optional `ContractRegistry` for type checking:

```python
class ContractRegistry:
    def register_contract(self, contract: TypeContract):
        """Register adapter method type contract"""
        
    def get_contract(self, adapter: str, method: str) -> Optional[TypeContract]:
        """Get contract for adapter.method"""
        
    def validate_type_compatibility(
        self, 
        producer_type: str, 
        consumer_type: str
    ) -> bool:
        """Check if types are compatible"""
```

**Note**: Current implementation includes a stub. The full contract registry should be implemented per the contract specification work item.

### 5. MappingStartupValidator

**Purpose**: Fail-fast validation at application startup

**Usage**:
```python
from orchestrator.mapping_loader import MappingStartupValidator

# In application entry point (run_farfan.py or test suite)
try:
    loader = MappingStartupValidator.validate_at_startup()
    # Continue with application logic...
except MappingValidationError as e:
    logger.error("FATAL: Execution mapping has structural errors")
    logger.error(str(e))
    sys.exit(1)
```

## Error Reporting

### Conflict Types

| ConflictType | Description |
|--------------|-------------|
| `DUPLICATE_PRODUCER` | Multiple steps produce same binding |
| `MISSING_PRODUCER` | Step references non-existent binding |
| `TYPE_MISMATCH` | Producer/consumer type incompatibility |
| `CIRCULAR_DEPENDENCY` | Cyclic dependencies in execution chain |
| `INVALID_BINDING` | Malformed binding reference |
| `UNKNOWN_ADAPTER` | Reference to unregistered adapter |
| `MALFORMED_CHAIN` | Missing required fields in step |

### MappingConflict Structure

```python
@dataclass
class MappingConflict:
    conflict_type: ConflictType
    question_ids: List[str]           # Affected questions
    description: str                  # Human-readable description
    affected_bindings: List[str]      # Problematic bindings
    type_mismatch_details: Optional[Dict]  # For TYPE_MISMATCH
    remediation: str                  # Actionable fix suggestions
```

### Diagnostic Output Example

```
================================================================================
MAPPING_CONFLICT: MISSING_PRODUCER
================================================================================

Questions Affected: D1_INSUMOS.Q4_Institutional_Capacity

Description: Step 3 references binding 'capacity_metrics' but no producer exists

Affected Bindings: capacity_metrics

Remediation:
Add a step that produces binding 'capacity_metrics' before step 3,
  or change the source reference to an existing binding.

================================================================================
```

## Integration Points

### 1. With Choreographer

The `ExecutionChoreographer` can use the validated loader to:
- Get execution order for questions
- Access DAGs for dependency tracking
- Retrieve adapter information

```python
from orchestrator.choreographer import ExecutionChoreographer
from orchestrator.mapping_loader import MappingStartupValidator

loader = MappingStartupValidator.validate_at_startup()
choreographer = ExecutionChoreographer()

# Execute question with validated chain
question_id = "D1_INSUMOS.Q1_Baseline_Identification"
chain = loader.get_execution_chain(question_id)
results = choreographer.execute_question_chain(
    question_spec=chain,
    plan_text=plan_text,
    module_adapter_registry=registry
)
```

### 2. With Test Framework

Test infrastructure can validate mapping before running canary tests:

```python
# In test_orchestrator_integration.py or similar
from orchestrator.mapping_loader import MappingStartupValidator

def setUp(self):
    # Validate mapping before tests
    try:
        self.loader = MappingStartupValidator.validate_at_startup()
    except MappingValidationError as e:
        self.fail(f"Mapping validation failed:\n{e}")
```

## Testing

### Test Suite: `test_mapping_loader.py`

Comprehensive test coverage:
1. YAML loading
2. Adapter registry parsing
3. Execution chain parsing
4. DAG construction
5. Binding validation
6. Type validation
7. Circular dependency detection
8. Full integration validation
9. Startup validator
10. Query interface

**Run Tests**:
```bash
# With dependencies installed (pyyaml, networkx)
python test_mapping_loader.py

# Or syntax check only
python -m py_compile orchestrator/mapping_loader.py
```

## Dependencies

Added to `requirements.txt`:
```
pyyaml==6.0.1
networkx==3.1
```

## Usage Examples

### Example 1: Validate at Startup
```python
from orchestrator.mapping_loader import MappingStartupValidator

if __name__ == "__main__":
    try:
        loader = MappingStartupValidator.validate_at_startup()
        print("✓ Mapping validation passed")
        # Run application...
    except MappingValidationError as e:
        print(f"✗ FATAL: {e}")
        exit(1)
```

### Example 2: Query Execution Chain
```python
from orchestrator.mapping_loader import YAMLMappingLoader

loader = YAMLMappingLoader()
loader.load_and_validate()

# Get chain for specific question
chain = loader.get_execution_chain("D1_INSUMOS.Q1_Baseline_Identification")
print(f"Steps: {len(chain['execution_chain'])}")

# Get DAG
dag = loader.get_execution_dag("D1_INSUMOS.Q1_Baseline_Identification")
print(f"Dependencies: {dag.number_of_edges()}")
```

### Example 3: Custom Contract Registry
```python
from orchestrator.mapping_loader import (
    YAMLMappingLoader, 
    ContractRegistry, 
    TypeContract
)

# Create registry with contracts
registry = ContractRegistry()
registry.register_contract(TypeContract(
    adapter="policy_segmenter",
    method="segment",
    input_types={"text": "str"},
    output_type="List[Dict[str, Any]]"
))

# Use with loader
loader = YAMLMappingLoader(contract_registry=registry)
loader.load_and_validate()  # Uses custom type checking
```

## Statistics

From validation run on current `execution_mapping.yaml`:
- **Adapters**: 9
- **Execution Chains**: ~30+ (across 10 dimensions, 300 questions)
- **Bindings Tracked**: 100+
- **DAG Nodes**: 4-8 per question
- **Validation Time**: <1 second

## Future Enhancements

1. **Full Contract Registry**: Replace stub with complete type registry from contract specification work item
2. **Performance Profiling**: Add metrics for chain execution time estimates
3. **Visualization**: Generate DAG diagrams for documentation
4. **Binding Inference**: Auto-detect missing binding producers from adapter signatures
5. **YAML Schema Validation**: Add JSON Schema for execution_mapping.yaml structure

## References

- `orchestrator/execution_mapping.yaml` - Source configuration
- `orchestrator/choreographer.py` - Execution orchestration
- `orchestrator/module_adapters.py` - Adapter implementations
- `AGENTS.md` - Repository commands and architecture
