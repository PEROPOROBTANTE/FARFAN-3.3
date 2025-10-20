# FARFAN 3.0 - Architecture Documentation

## Table of Contents
1. [ModuleController Dependency Injection Pattern](#modulecontroller-dependency-injection-pattern)
2. [QuestionRouter Mapping System](#questionrouter-mapping-system)
3. [CircuitBreaker Failure Tracking](#circuitbreaker-failure-tracking)
4. [Adapter Layer Modules](#adapter-layer-modules)
5. [Infrastructure Modules](#infrastructure-modules)
6. [API Reference](#api-reference)

---

## ModuleController Dependency Injection Pattern

### Overview
The ModuleController (implemented as `ModuleAdapterRegistry` in `orchestrator/module_adapters.py`) manages 11 adapters using a dependency injection pattern that enables:
- **Centralized adapter lifecycle management**: Single registry for all adapters
- **Uniform invocation interface**: All adapter methods accessed through `execute_module_method()`
- **Runtime adapter availability checking**: Graceful handling of missing dependencies
- **Standardized result format**: All adapters return `ModuleResult` objects

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                ModuleAdapterRegistry                         │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Adapter Initialization & Dependency Injection         │ │
│  └────────────────────────────────────────────────────────┘ │
│                           │                                  │
│         ┌─────────────────┼─────────────────┐               │
│         ▼                 ▼                 ▼                │
│  ┌──────────┐      ┌──────────┐     ┌──────────┐           │
│  │ Adapter1 │      │ Adapter2 │ ... │ Adapter11│           │
│  │ (teoria_ │      │(analyzer_│     │(policy_  │           │
│  │  cambio) │      │   one)   │     │segmenter)│           │
│  └──────────┘      └──────────┘     └──────────┘           │
└─────────────────────────────────────────────────────────────┘
         │                   │                   │
         ▼                   ▼                   ▼
    ┌─────────┐         ┌─────────┐        ┌─────────┐
    │ Module  │         │ Module  │        │ Module  │
    │ Result  │         │ Result  │        │ Result  │
    └─────────┘         └─────────┘        └─────────┘
```

### 11 Managed Adapters

| Adapter Name | Module File | Methods | Purpose |
|--------------|-------------|---------|---------|
| `teoria_cambio` | teoria_cambio.py | 51 | Theory of change analysis, causal pathway modeling |
| `analyzer_one` | Analyzer_one.py | 39 | Municipal development plan analysis |
| `dereck_beach` | dereck_beach.py | 89 | Causal deconstruction (CDAF) framework |
| `embedding_policy` | emebedding_policy.py | 37 | Semantic embedding and similarity analysis |
| `semantic_chunking_policy` | semantic_chunking_policy.py | 18 | Document segmentation and chunking |
| `contradiction_detection` | contradiction_deteccion.py | 52 | Policy contradiction detection |
| `financial_viability` | financiero_viabilidad_tablas.py | 60 | Financial viability assessment |
| `policy_processor` | policy_processor.py | 34 | Industrial policy processing |
| `policy_segmenter` | policy_segmenter.py | 33 | Policy document segmentation |
| `causal_processor` | causal_proccesor.py | ~30 | Causal relationship processing |
| `info_extractor` | info_info.py | ~20 | Information extraction utilities |

**Total: 413+ methods across 11 adapters**

### Dependency Injection Pattern

#### Initialization Phase
```python
# orchestrator/core_orchestrator.py
class FARFANOrchestrator:
    def __init__(
        self,
        module_adapter_registry: ModuleAdapterRegistry,  # ← Injected dependency
        questionnaire_parser: QuestionnaireParser,
        config: Optional[Dict[str, Any]] = None
    ):
        self.module_registry = module_adapter_registry
        # Registry contains all 11 initialized adapters
```

#### Adapter Registration
```python
# orchestrator/module_adapters.py
class ModuleAdapterRegistry:
    def __init__(self):
        self.adapters: Dict[str, BaseAdapter] = {}
        
        # Initialize all 11 adapters
        self._register_adapter("teoria_cambio", ModulosAdapter())
        self._register_adapter("analyzer_one", AnalyzerOneAdapter())
        self._register_adapter("dereck_beach", DerekBeachAdapter())
        self._register_adapter("embedding_policy", EmbeddingPolicyAdapter())
        self._register_adapter("semantic_chunking_policy", SemanticChunkingPolicyAdapter())
        self._register_adapter("contradiction_detection", ContradictionDetectionAdapter())
        self._register_adapter("financial_viability", FinancialViabilityAdapter())
        self._register_adapter("policy_processor", PolicyProcessorAdapter())
        self._register_adapter("policy_segmenter", PolicySegmenterAdapter())
        self._register_adapter("causal_processor", CausalProcessorAdapter())
        self._register_adapter("info_extractor", InfoExtractorAdapter())
```

#### Uniform Invocation Interface
```python
# All adapters invoked through same interface
result: ModuleResult = module_registry.execute_module_method(
    module_name="teoria_cambio",           # Adapter identifier
    method_name="calculate_bayesian_confidence",  # Method name
    args=[text, prior_confidence],         # Positional arguments
    kwargs={"use_cache": True}            # Keyword arguments
)

# Returns standardized ModuleResult
assert result.module_name == "teoria_cambio"
assert result.status in ["success", "failed"]
assert result.confidence >= 0.0 and result.confidence <= 1.0
```

#### Standardized Result Format
```python
@dataclass
class ModuleResult:
    """Standardized output format for all 11 adapters"""
    module_name: str                      # e.g., "teoria_cambio"
    class_name: str                       # e.g., "TeoriaCambioAnalyzer"
    method_name: str                      # e.g., "calculate_bayesian_confidence"
    status: str                           # "success" or "failed"
    data: Dict[str, Any]                  # Method-specific output
    evidence: List[Dict[str, Any]]        # Supporting evidence
    confidence: float                     # 0.0-1.0 confidence score
    execution_time: float                 # Seconds
    errors: List[str]                     # Error messages (if any)
    warnings: List[str]                   # Warning messages
    metadata: Dict[str, Any]              # Additional metadata
```

### Benefits of Dependency Injection Pattern

1. **Loose Coupling**: Orchestrator depends on `ModuleAdapterRegistry` interface, not concrete adapter implementations
2. **Testability**: Easy to inject mock adapters for unit testing
3. **Runtime Flexibility**: Adapters can be added/removed without modifying orchestrator code
4. **Graceful Degradation**: Missing adapters detected at initialization, not at runtime
5. **Centralized Error Handling**: All adapter failures handled uniformly by registry

---

## QuestionRouter Mapping System

### Overview
The QuestionRouter (`orchestrator/question_router.py`) maps **300 cuestionario.json question IDs** to **module:Class.method handlers** using `execution_mapping.yaml`. This enables:
- **Dimension-based routing**: D1-D6 dimensions mapped to appropriate adapter chains
- **Multi-step execution chains**: Complex questions decomposed into adapter method sequences
- **Confidence calibration**: Historical performance used to adjust confidence scores
- **Caching**: LRU cache (1000 entries) for fast repeated lookups

### Mapping Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                   cuestionario.json                            │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  300 Questions: P1-D1-Q1 ... P10-D6-Q50                  │ │
│  │  - Canonical IDs                                          │ │
│  │  - Dimension tags (D1-D6)                                 │ │
│  │  - Question text & rubric                                 │ │
│  └──────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────┐
│              execution_mapping.yaml                            │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  Dimension Sections:                                      │ │
│  │  - D1_INSUMOS: 50 question mappings                      │ │
│  │  - D2_PROCESOS: 50 question mappings                     │ │
│  │  - D3_PRODUCTOS: 50 question mappings                    │ │
│  │  - D4_RESULTADOS: 50 question mappings                   │ │
│  │  - D5_IMPACTOS: 50 question mappings                     │ │
│  │  - D6_TEORIA_CAMBIO: 50 question mappings                │ │
│  └──────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────┐
│                    QuestionRouter                              │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  route_question(question_id, dimension)                   │ │
│  │    → ExecutionChain with ordered steps                    │ │
│  │                                                            │ │
│  │  ExecutionChain:                                          │ │
│  │    - Step 1: adapter_name.method_name(args)               │ │
│  │    - Step 2: adapter_name.method_name(args)               │ │
│  │    - Step N: adapter_name.method_name(args)               │ │
│  │    - Aggregation strategy                                 │ │
│  └──────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────┘
```

### Dimension-to-Module Mapping

| Dimension | Description | Primary Adapters |
|-----------|-------------|------------------|
| **D1** (Insumos/Resources) | Resource analysis, initial state | `policy_segmenter`, `policy_processor`, `semantic_chunking`, `financial_viability` |
| **D2** (Procesos/Processes) | Intervention design, processes | `embedding_policy`, `policy_processor`, `dereck_beach`, `teoria_cambio` |
| **D3** (Productos/Outputs) | Direct outputs, deliverables | `financial_viability`, `analyzer_one`, `policy_processor` |
| **D4** (Resultados/Outcomes) | Intermediate results | `analyzer_one`, `teoria_cambio`, `dereck_beach` |
| **D5** (Impactos/Impacts) | Long-term impacts | `teoria_cambio`, `dereck_beach`, `contradiction_detection` |
| **D6** (Teoría de Cambio) | Theory of change | `teoria_cambio`, `dereck_beach`, `analyzer_one`, `contradiction_detection` |

### execution_mapping.yaml Structure

```yaml
version: "3.0.0"
total_adapters: 9
total_methods: 413

# Dimension-level mapping
D1_INSUMOS:
  Q1_Resource_Identification:
    canonical_id: "P1-D1-Q1"
    description: "Identify available resources and inputs"
    execution_chain:
      - step: 1
        adapter: "policy_segmenter"
        adapter_class: "PolicySegmenter"
        method: "segment_by_sections"
        args: [{text: "$plan_text"}]
        returns: {segments: "List[str]"}
        purpose: "Segment document into sections"
        confidence_expected: 0.85
      
      - step: 2
        adapter: "policy_processor"
        adapter_class: "IndustrialPolicyProcessor"
        method: "process"
        args: [{text: "$plan_text"}]
        returns: {analysis: "Dict[str, Any]"}
        purpose: "Process policy text for resource patterns"
        confidence_expected: 0.80
    
    aggregation:
      strategy: "weighted_average"
      weights:
        policy_segmenter: 0.4
        policy_processor: 0.6
      confidence_threshold: 0.70
```

### Routing Flow

```python
# 1. Question arrives from cuestionario.json
question_id = "P1-D1-Q1"
dimension = "D1"

# 2. Router looks up execution chain
router = QuestionRouter(execution_mapping_path="orchestrator/execution_mapping.yaml")
execution_chain = router.route_question(question_id, dimension)

# 3. ExecutionChain contains ordered steps
assert len(execution_chain.steps) >= 1
assert execution_chain.steps[0].adapter_name == "policy_segmenter"
assert execution_chain.steps[0].method_name == "segment_by_sections"
assert execution_chain.steps[0].confidence_expected == 0.85

# 4. Choreographer executes steps in order
for step in execution_chain.steps:
    result = module_registry.execute_module_method(
        module_name=step.adapter_name,
        method_name=step.method_name,
        args=step.args,
        kwargs=step.kwargs
    )
```

### Confidence Calibration

```python
def _calibrate_confidence(
    self,
    adapter_name: str,
    raw_confidence: float,
    temperature: float = 1.5
) -> float:
    """
    Calibrate confidence using historical performance
    
    Formula: calibrated = (raw * historical_performance) / temperature
    
    - raw_confidence: From execution_mapping.yaml (0.70-0.95)
    - historical_performance: Success rate from circuit breaker (0.0-1.0)
    - temperature: Scaling factor (higher = more conservative)
    
    Result clamped to [0.65, 0.95]
    """
    historical_performance = self.adapter_performance.get(adapter_name, 1.0)
    calibrated = (raw_confidence * historical_performance) / temperature
    return max(0.65, min(0.95, calibrated))
```

### Caching Strategy

- **Cache Type**: `@lru_cache(maxsize=1000)`
- **Cache Key**: `(question_id, dimension)` tuple
- **Invalidation**: Automatic when `execution_mapping.yaml` modified
- **Hit Rate**: Logged and tracked in `cache_stats`

---

## CircuitBreaker Failure Tracking

### Overview
The CircuitBreaker (`orchestrator/circuit_breaker.py`) implements **threshold-based state transitions** to protect against cascading failures across the 11 adapters. Key features:
- **Per-adapter circuit states**: Independent failure tracking for each adapter
- **Three-state model**: CLOSED → OPEN → HALF_OPEN → CLOSED
- **Automatic recovery**: Self-healing after timeout period
- **Fallback strategies**: Graceful degradation when circuits open

### State Transition Diagram

```
                    ┌─────────────────┐
                    │     CLOSED      │ ← Normal operation
                    │  (All requests  │
                    │   allowed)      │
                    └────────┬────────┘
                             │
                   5 failures in 60s
                             │
                             ▼
                    ┌─────────────────┐
                    │      OPEN       │ ← Circuit tripped
                    │  (All requests  │
                    │    blocked)     │
                    └────────┬────────┘
                             │
                  60s timeout expires
                             │
                             ▼
                    ┌─────────────────┐
                    │   HALF_OPEN     │ ← Testing recovery
                    │  (3 test calls  │
                    │    allowed)     │
                    └────────┬────────┘
                             │
                    ┌────────┴────────┐
                    │                 │
            3 successes        Any failure
                    │                 │
                    ▼                 ▼
           ┌─────────────┐   ┌─────────────┐
           │   CLOSED    │   │    OPEN     │
           └─────────────┘   └─────────────┘
```

### Threshold Configuration

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `failure_threshold` | 5 | Consecutive failures to trip circuit |
| `recovery_timeout` | 60.0 seconds | Time before attempting recovery |
| `half_open_max_calls` | 3 | Test calls in HALF_OPEN state |
| `failure_window` | 60.0 seconds | Time window for failure counting |

### State Descriptions

#### CLOSED State
- **Meaning**: Normal operation, no issues detected
- **Behavior**: All requests allowed through
- **Transition**: → OPEN after `failure_threshold` failures in `failure_window`
- **Metrics**: Success rate tracked, response times logged

#### OPEN State
- **Meaning**: Circuit tripped due to excessive failures
- **Behavior**: All requests blocked immediately
- **Transition**: → HALF_OPEN after `recovery_timeout` seconds
- **Metrics**: Failure count, time since trip

#### HALF_OPEN State
- **Meaning**: Testing recovery with limited requests
- **Behavior**: Allow `half_open_max_calls` requests through
- **Transitions**:
  - → CLOSED if all test calls succeed
  - → OPEN if any test call fails
- **Metrics**: Test call success rate

### Failure Tracking

```python
@dataclass
class FailureEvent:
    """Individual failure event"""
    timestamp: float                    # Unix timestamp
    severity: FailureSeverity          # TRANSIENT, DEGRADED, CRITICAL, CATASTROPHIC
    error_type: str                    # Exception class name
    error_message: str                 # Error message
    execution_time: float              # Time taken before failure
    adapter_name: str                  # Which adapter failed
    method_name: str                   # Which method failed
    recovery_attempt: int              # Recovery attempt number
```

### Performance Metrics

```python
@dataclass
class PerformanceMetrics:
    """Performance tracking for an adapter"""
    response_times: deque[float]       # Last 100 response times (rolling)
    success_count: int                 # Total successes
    failure_count: int                 # Total failures
    last_success: Optional[float]      # Timestamp of last success
    last_failure: Optional[float]      # Timestamp of last failure
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 1.0
```

### Usage Example

```python
# Initialize circuit breaker
circuit_breaker = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=60.0,
    half_open_max_calls=3
)

# Before adapter execution
if circuit_breaker.can_execute("teoria_cambio"):
    try:
        result = module_registry.execute_module_method(
            module_name="teoria_cambio",
            method_name="analyze_causal_pathways",
            args=[plan_text]
        )
        # Record success
        circuit_breaker.record_success(
            adapter_name="teoria_cambio",
            execution_time=result.execution_time
        )
    except Exception as e:
        # Record failure
        circuit_breaker.record_failure(
            adapter_name="teoria_cambio",
            error=str(e),
            execution_time=0.0,
            severity=FailureSeverity.CRITICAL
        )
else:
    # Circuit is OPEN, use fallback
    fallback_strategy = circuit_breaker.get_fallback_strategy("teoria_cambio")
```

### Fallback Strategies

Each adapter has a defined fallback strategy when circuit opens:

```python
fallback_strategies = {
    "teoria_cambio": {
        "use_cached": True,
        "alternative_adapters": ["analyzer_one"],
        "degraded_mode": "basic_causal_analysis"
    },
    "analyzer_one": {
        "use_cached": True,
        "alternative_adapters": ["embedding_policy"],
        "degraded_mode": "simple_analysis"
    }
}
```

---

## Adapter Layer Modules

### Core Adapter Modules (11 Total)

1. **teoria_cambio.py** - 51 methods
   - Theory of change analysis
   - Causal pathway modeling
   - Bayesian confidence scoring

2. **Analyzer_one.py** - 39 methods
   - Municipal development plan analysis
   - Policy alignment assessment

3. **dereck_beach.py** - 89 methods
   - CDAF causal deconstruction framework
   - Beach evidential tests (Straw-in-the-Wind, Hoop, Smoking Gun, Doubly-Decisive)

4. **emebedding_policy.py** - 37 methods
   - Semantic embedding generation
   - P-D-Q canonical notation system
   - Colombian PDM specialization

5. **semantic_chunking_policy.py** - 18 methods
   - Document segmentation
   - Semantic chunking with Bayesian boundaries

6. **contradiction_deteccion.py** - 52 methods
   - Policy contradiction detection
   - Temporal logic analysis

7. **financiero_viabilidad_tablas.py** - 60 methods
   - Financial viability assessment
   - PDET budget analysis
   - Causal DAG for financial risks

8. **policy_processor.py** - 34 methods
   - Industrial policy processing
   - Pattern matching and extraction

9. **policy_segmenter.py** - 33 methods
   - Document segmentation
   - Bayesian boundary scoring

10. **causal_proccesor.py** - ~30 methods
    - Causal relationship processing

11. **info_info.py** - ~20 methods
    - Information extraction utilities

---

## Infrastructure Modules

### orchestrator/core_orchestrator.py
**Purpose**: Main orchestration engine coordinating complete FARFAN 3.0 pipeline

**Key Classes**:
- `FARFANOrchestrator`: Main orchestrator class

**Key Methods**:
- `analyze_single_plan(plan_path, plan_name, output_dir, questions_to_analyze)`: Execute complete analysis pipeline for single development plan
  - **Parameters**:
    - `plan_path` (Path): Path to plan document (PDF/TXT/DOCX)
    - `plan_name` (Optional[str]): Plan identifier
    - `output_dir` (Optional[Path]): Directory for report outputs
    - `questions_to_analyze` (Optional[List[str]]): Subset of question IDs
  - **Returns**: Dict with success status, micro_answers, meso_clusters, macro_convergence, report_path
  - **Raises**: FileNotFoundError, ValueError

- `analyze_batch(plan_paths, output_dir)`: Analyze multiple plans in batch mode
- `get_orchestrator_status()`: Get current orchestrator health status

### orchestrator/choreographer.py
**Purpose**: DAG-based module orchestration with dependency management

**Key Classes**:
- `ExecutionChoreographer`: Manages parallel execution with dependency resolution
- `ExecutionResult`: Result from single adapter method execution

**Key Methods**:
- `execute_question_chain(question_spec, plan_text, module_adapter_registry, circuit_breaker)`: Execute execution chain for question
  - **Parameters**:
    - `question_spec`: Question specification object
    - `plan_text` (str): Plan document text
    - `module_adapter_registry`: Registry instance
    - `circuit_breaker`: CircuitBreaker instance
  - **Returns**: Dict of ExecutionResult objects
  - **Raises**: ExecutionError

### orchestrator/question_router.py
**Purpose**: Routes 300 questions to validated execution chains

**Key Classes**:
- `QuestionRouter`: Routes questions to execution chains
  - **Question IDs Handled**: All 300 questions from cuestionario.json (P1-D1-Q1 through P10-D6-Q50)
- `ExecutionChain`: Complete execution chain for a question
- `ExecutionStep`: Single step in an execution chain

**Key Methods**:
- `route_question(question_id, dimension)`: Route question to execution chain
  - **Parameters**:
    - `question_id` (str): Canonical question ID (e.g., "P1-D1-Q1")
    - `dimension` (str): Dimension ID (e.g., "D1")
  - **Returns**: Optional[ExecutionChain]
  - **Raises**: None (returns None on lookup failure)

- `get_dimension_modules(dimension)`: Get list of adapters for dimension
- `validate_mapping_consistency()`: Validate mapping consistency

### orchestrator/circuit_breaker.py
**Purpose**: Fault tolerance with threshold-based state transitions

**Key Classes**:
- `CircuitBreaker`: Per-adapter circuit breaking
- `CircuitState` (Enum): CLOSED, OPEN, HALF_OPEN, ISOLATED, RECOVERING
- `FailureEvent`: Individual failure event
- `PerformanceMetrics`: Performance tracking

**Key Methods**:
- `can_execute(adapter_name)`: Check if adapter can execute
  - **Parameters**: `adapter_name` (str)
  - **Returns**: bool
  
- `record_success(adapter_name, execution_time)`: Record successful execution
- `record_failure(adapter_name, error, execution_time, severity)`: Record failed execution
- `get_adapter_status(adapter_name)`: Get status for single adapter
- `get_all_status()`: Get status for all adapters
- `get_fallback_strategy(adapter_name)`: Get fallback strategy for failed adapter

---

## API Reference

### Module Docstring Standards

All modules follow this docstring format:

```python
def method_name(self, param1: Type1, param2: Type2) -> ReturnType:
    """
    Brief description of method purpose.
    
    **Addresses Question IDs:**
    - P1-D1-Q1: Question description
    - P2-D3-Q5: Question description
    
    Args:
        param1: Description and valid range/values
        param2: Description and valid range/values
    
    Returns:
        Description of return value structure
    
    Raises:
        ExceptionType1: When this occurs
        ExceptionType2: When this occurs
    """
```

### Question ID to Handler Mapping Examples

#### D1 (Insumos) Questions
- **P1-D1-Q1**: `policy_segmenter.segment()` + `policy_processor.process()`
- **P1-D1-Q2**: `financial_viability.analyze_budget_allocation()`
- **P2-D1-Q1**: `analyzer_one.analyze_institutional_capacity()`

#### D2 (Procesos) Questions
- **P1-D2-Q1**: `embedding_policy.embed_text()` + `teoria_cambio.analyze_causal_pathways()`
- **P1-D2-Q3**: `dereck_beach.evaluate_mechanism()`

#### D6 (Teoría de Cambio) Questions
- **P1-D6-Q1**: `teoria_cambio.extract_causal_chains()`
- **P1-D6-Q2**: `dereck_beach.apply_evidential_tests()`
- **P1-D6-Q5**: `contradiction_detection.detect_logical_inconsistencies()`

For complete question-to-method mappings, see `orchestrator/execution_mapping.yaml`.

---

## Migration Notes

See main README.md for deprecated method warnings and migration guide.
