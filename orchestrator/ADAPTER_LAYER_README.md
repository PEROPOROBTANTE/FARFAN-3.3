# Orchestrator Adapter Layer

## Overview

The adapter layer provides backward-compatible translation between the legacy 11-adapter architecture and the new unified module controller. Each adapter wraps core domain module functionality while preserving existing method signatures that calling code expects.

## Architecture

```
Legacy Code → Adapter Layer → Core Domain Modules
              (Translation)   (policy_processor.py, 
                               emebedding_policy.py,
                               dereck_beach.py, etc.)
```

## Seven Adapter Modules

### 1. PolicyProcessorAdapter
**File:** `orchestrator/adapter_policy_processor.py`
**Wraps:** `policy_processor.py`

**Primary Interface:**
- `process_text(text: str) -> Dict[str, Any]`
- `analyze_policy_file(file_path: str) -> Dict[str, Any]`
- `extract_evidence(text: str, patterns: List[str]) -> List[Dict[str, Any]]`
- `score_evidence_confidence(matches: List, context: str) -> float`
- `segment_text(text: str) -> List[str]`

**Legacy Aliases (Deprecated):**
- `analyze_document()` → `process_text()`
- `extract_causal_evidence()` → `extract_evidence()`
- `compute_confidence()` → `score_evidence_confidence()`
- `split_sentences()` → `segment_text()`

### 2. EmbeddingPolicyAdapter
**File:** `orchestrator/adapter_embedding_policy.py`
**Wraps:** `emebedding_policy.py`

**Primary Interface:**
- `chunk_document(text: str, metadata: Dict) -> List[Dict[str, Any]]`
- `embed_chunks(chunks: List[str]) -> NDArray`
- `compute_similarity(query: str, documents: List[str]) -> NDArray`
- `rerank_results(query: str, candidates: List[str]) -> List[Tuple[str, float]]`
- `evaluate_policy_metric(values: List[float]) -> Dict[str, Any]`

**Legacy Aliases (Deprecated):**
- `semantic_chunk()` → `chunk_document()`
- `generate_embeddings()` → `embed_chunks()`
- `calculate_similarity()` → `compute_similarity()`
- `cross_encode_rerank()` → `rerank_results()`
- `bayesian_metric_eval()` → `evaluate_policy_metric()`

### 3. DerekBeachAdapter
**File:** `orchestrator/adapter_dereck_beach.py`
**Wraps:** `dereck_beach.py`

**Primary Interface:**
- `extract_causal_hierarchy(text: str) -> nx.DiGraph`
- `extract_entity_activities(text: str) -> List[Dict[str, Any]]`
- `audit_evidence_traceability(nodes: Dict) -> Dict[str, Any]`
- `process_pdf_document(pdf_path: str, policy_code: str) -> Dict[str, Any]`
- `validate_dag_structure(graph: nx.DiGraph) -> Dict[str, Any]`

**Legacy Aliases (Deprecated):**
- `extract_causal_graph()` → `extract_causal_hierarchy()`
- `get_entity_activity_pairs()` → `extract_entity_activities()`
- `audit_traceability()` → `audit_evidence_traceability()`
- `process_pdf()` → `process_pdf_document()`
- `validate_graph()` → `validate_dag_structure()`

### 4. ContradictionDetectionAdapter
**File:** `orchestrator/adapter_contradiction_detection.py`
**Wraps:** `contradiction_deteccion.py`

**Primary Interface:**
- `detect_contradictions(text: str) -> List[Dict[str, Any]]`
- `analyze_semantic_contradictions(statements: List[str]) -> List[Dict[str, Any]]`
- `validate_temporal_consistency(statements: List[Dict]) -> List[Dict[str, Any]]`
- `check_numerical_consistency(claims: List[Dict]) -> List[Dict[str, Any]]`
- `build_contradiction_graph(statements: List[str]) -> nx.Graph`

**Legacy Aliases (Deprecated):**
- `find_contradictions()` → `detect_contradictions()`
- `semantic_analysis()` → `analyze_semantic_contradictions()`
- `temporal_validation()` → `validate_temporal_consistency()`
- `numerical_check()` → `check_numerical_consistency()`

### 5. CausalProcessorAdapter
**File:** `orchestrator/adapter_causal_processor.py`
**Wraps:** `causal_proccesor.py`

**Primary Interface:**
- `extract_causal_dimensions(text: str) -> Dict[str, Any]`
- `infer_causal_structure(data: Dict) -> Any`
- `compute_causal_effects(treatment: str, outcome: str, data: Dict) -> Dict[str, float]`
- `analyze_counterfactual(intervention: Dict, context: Dict) -> Dict[str, Any]`
- `validate_causal_assumptions(dag: Any) -> Dict[str, bool]`

**Legacy Aliases (Deprecated):**
- `extract_dimensions()` → `extract_causal_dimensions()`
- `learn_structure()` → `infer_causal_structure()`
- `estimate_effects()` → `compute_causal_effects()`
- `counterfactual_analysis()` → `analyze_counterfactual()`

### 6. TeoriaCambioAdapter
**File:** `orchestrator/adapter_teoria_cambio.py`
**Wraps:** `teoria_cambio.py`

**Primary Interface:**
- `validate_theory_of_change(model: Dict) -> Dict[str, Any]`
- `validate_causal_dag(graph: nx.DiGraph) -> Dict[str, Any]`
- `check_hierarchical_consistency(model: Dict) -> Dict[str, bool]`
- `run_monte_carlo_validation(dag: nx.DiGraph, n_simulations: int) -> Dict[str, Any]`
- `audit_validator_performance() -> Dict[str, Any]`

**Legacy Aliases (Deprecated):**
- `validate_model()` → `validate_theory_of_change()`
- `validate_dag()` → `validate_causal_dag()`
- `check_hierarchy()` → `check_hierarchical_consistency()`
- `monte_carlo_test()` → `run_monte_carlo_validation()`

### 7. FinancialViabilityAdapter
**File:** `orchestrator/adapter_financial_viability.py`
**Wraps:** `financiero_viabilidad_tablas.py`

**Primary Interface:**
- `analyze_financial_tables(pdf_path: str) -> Dict[str, Any]`
- `extract_budget_allocation(pdf_path: str) -> pd.DataFrame`
- `validate_financial_consistency(tables: List[pd.DataFrame]) -> Dict[str, Any]`
- `compute_financial_viability_score(data: Dict) -> float`
- `infer_resource_causal_chains(financial_data: Dict) -> Dict[str, Any]`

**Legacy Aliases (Deprecated):**
- `analyze_pdf()` → `analyze_financial_tables()`
- `get_budget_data()` → `extract_budget_allocation()`
- `check_consistency()` → `validate_financial_consistency()`
- `viability_score()` → `compute_financial_viability_score()`
- `causal_inference()` → `infer_resource_causal_chains()`

## Usage Pattern

### Basic Usage
```python
from orchestrator.adapter_policy_processor import PolicyProcessorAdapter

# Initialize adapter
adapter = PolicyProcessorAdapter()

# Use primary interface
result = adapter.process_text("Plan de desarrollo municipal...")

# Check availability
if adapter.is_available():
    print("Core module loaded successfully")
```

### Legacy Code Support
```python
# Old code using deprecated method still works
result = adapter.analyze_document(text)  # Issues DeprecationWarning
```

### Configuration
```python
# Pass configuration to adapter
config = {
    'confidence_threshold': 0.7,
    'context_window_chars': 500
}
adapter = PolicyProcessorAdapter(config=config)
```

## Deprecation Warnings

All legacy method aliases emit `DeprecationWarning` when called, guiding developers to use the new primary interface:

```
DeprecationWarning: analyze_document() is deprecated, use process_text() instead
```

## Design Principles

1. **Backward Compatibility:** Existing code continues to function without modifications
2. **Deprecation Path:** Legacy methods warn but don't break, allowing gradual migration
3. **Type Safety:** All methods preserve original type signatures
4. **Lazy Loading:** Core modules loaded only when needed
5. **Error Handling:** Graceful fallback when core modules unavailable
6. **Documentation:** Each adapter documents both primary and legacy interfaces

## Error Handling

```python
adapter = PolicyProcessorAdapter()

# Check if core module available
if not adapter.is_available():
    print("Core module not available")
    # Handle gracefully

# Adapter raises RuntimeError if core module missing
try:
    result = adapter.process_text(text)
except RuntimeError as e:
    print(f"Core module error: {e}")
```

## Testing

Run simple import test:
```bash
python test_adapters_simple.py
```

## Migration Guide

To migrate from legacy to new interface:

1. Find deprecated method calls (look for DeprecationWarnings)
2. Replace with primary interface equivalent
3. Update tests to use new method names
4. Remove legacy method usage

Example:
```python
# Old (deprecated)
chunks = adapter.semantic_chunk(text, metadata)

# New (primary interface)
chunks = adapter.chunk_document(text, metadata)
```

## Integration with Unified Module Controller

The adapter layer serves as the translation bridge:

```
Orchestrator
    ↓
Question Router
    ↓
Choreographer
    ↓
Adapter Layer (7 adapters) ← YOU ARE HERE
    ↓
Core Domain Modules (policy_processor.py, etc.)
```

The unified module controller can invoke adapters knowing they provide consistent interfaces while internally delegating to diverse core implementations.
