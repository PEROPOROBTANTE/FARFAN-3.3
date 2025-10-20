# Function Inference Heuristics for FARFAN 3.0

## Purpose
Systematic rules for resolving adapter-to-module method mismatches during interface validation.

## Resolution Strategy Decision Tree

```
Mismatch Detected
    ↓
┌───────────────────────────────────────┐
│ Compatible method exists in source?   │
└───────────────────────────────────────┘
    ↓ YES                    ↓ NO
[TIER 1: Adapt Adapter]    Can delegate to existing?
  • Change adapter call       ↓ YES              ↓ NO
  • Add inline comment      [TIER 2: Wrapper]  [TIER 3: Stub]
  • No source changes        • Thin wrapper      • NotImplementedError
  ✓ Preferred                • Complete docs     • Roadmap in docs
                             ✓ Acceptable        ⚠ Last resort
```

## Priority Rules

1. **TIER 1 (Preferred):** Modify adapter to call existing compatible method
   - Zero source changes
   - Leverage tested code
   - Document with inline comment

2. **TIER 2 (Acceptable):** Add minimal delegation wrapper to source
   - Delegates to existing methods
   - Complete docstring with NOTE field
   - Minimal new code

3. **TIER 3 (Last Resort):** Add NotImplementedError stub
   - No suitable alternative exists
   - Documents contract
   - Includes implementation roadmap

## Examples

### Example 1: TIER 1 - Adapt Adapter

**Before:**
```python
def _execute_analyze_policy(self, text: str, **kwargs) -> ModuleResult:
    processor = self.PolicyProcessor()
    results = processor.analyze_policy_text(text)  # Method doesn't exist!
    return ModuleResult(...)
```

**After:**
```python
def _execute_analyze_policy(self, text: str, **kwargs) -> ModuleResult:
    processor = self.PolicyProcessor()
    # ADAPTER INFERENCE: analyze_policy_text() -> process_policy_document()
    # Source provides process_policy_document(text: str) with compatible output
    results = processor.process_policy_document(text)
    return ModuleResult(...)
```

### Example 2: TIER 2 - Delegation Wrapper

**Source Before:**
```python
class Processor:
    def extract_causes(self, text: str) -> List[Dict]:
        # implementation
        return causes
    
    def link_effects(self, causes: List[Dict]) -> Dict:
        # implementation
        return effects
```

**Source After:**
```python
class Processor:
    def extract_causes(self, text: str) -> List[Dict]:
        return causes
    
    def link_effects(self, causes: List[Dict]) -> Dict:
        return effects
    
    def extract_causal_chains(self, text: str) -> Dict[str, Any]:
        """
        Extract complete causal chains from text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dict with 'causes', 'chains', and 'metadata' keys
            
        NOTE: Added for adapter compatibility with ProcessorAdapter.
              Delegates to extract_causes() and link_effects().
        """
        causes = self.extract_causes(text)
        chains = self.link_effects(causes)
        return {'causes': causes, 'chains': chains}
```

### Example 3: TIER 3 - NotImplementedError Stub

```python
def predict_policy_impact(
    self,
    policy_text: str,
    baseline: Dict[str, float]
) -> Dict[str, Any]:
    """
    Predict quantitative policy impact using ML models.
    
    Args:
        policy_text: Policy description
        baseline: Current metric values
        
    Returns:
        Predicted outcomes with confidence intervals
        
    Raises:
        NotImplementedError: ML model not yet integrated
        
    NOTE: Added for adapter compatibility with AnalyzerAdapter.
          Implementation requires causal ML model integration.
          See GitHub Issue #127, planned for FARFAN 3.1 (Q2 2024).
          Use analyze_baseline() as interim alternative.
    """
    raise NotImplementedError(
        "predict_policy_impact() requires causal ML model. "
        "Planned for v3.1. Use analyze_baseline() for now."
    )
```

### Example 4: Type Incompatibility Resolution

**Before:**
```python
def _execute_batch(self, docs: List[str], **kwargs) -> ModuleResult:
    processor = self.Processor()
    results = processor.process(docs)  # Expects str, not List[str]!
    return ModuleResult(...)
```

**After (TIER 1):**
```python
def _execute_batch(self, docs: List[str], **kwargs) -> ModuleResult:
    processor = self.Processor()
    # ADAPTER INFERENCE: Batch via iterative single-doc calls
    # Source process() expects str, adapt by iteration
    results = [processor.process(doc) for doc in docs]
    aggregated = {'results': results, 'count': len(results)}
    return ModuleResult(data=aggregated, ...)
```

## API Stability Rules

### Rule 1: No Deletion Without Deprecation

```python
# ✓ CORRECT: Deprecate first
def old_method(self):
    """DEPRECATED: Use new_method(). Removed in v4.0."""
    warnings.warn(
        "old_method() deprecated, use new_method()",
        DeprecationWarning,
        stacklevel=2
    )
    return self.new_method()

# ❌ FORBIDDEN: Direct deletion breaks contracts
```

### Rule 2: Backward-Compatible Signatures

```python
# ✓ CORRECT: Default parameters
def analyze(text: str, model: str = "default") -> dict:
    """New parameter added with default value"""
    pass

# ❌ FORBIDDEN: New required parameter
def analyze(text: str, model: str) -> dict:
    pass
```

### Rule 3: Return Type Wrappers

```python
def process(text: str) -> ProcessingResult:
    """Modern structured return"""
    pass

def process_dict(text: str) -> dict:
    """
    Legacy dict interface for compatibility.
    
    NOTE: Maintained for adapter compatibility.
          New code should use process().
    """
    return process(text).to_dict()
```

## Type Hint Requirements

All new methods must include Python 3.10+ compatible type hints:

```python
from typing import Dict, List, Any, Optional, Union, Tuple

# ✓ CORRECT: Complete annotations
def extract_evidence(
    text: str,
    threshold: float = 0.7,
    max_items: Optional[int] = None
) -> Dict[str, Union[List[Dict[str, Any]], float]]:
    """
    Extract evidence from text.
    
    Args:
        text: Input text
        threshold: Min confidence (0.0-1.0)
        max_items: Max results (None=unlimited)
        
    Returns:
        Dict with 'items' and 'avg_confidence' keys
    """
    pass

# ❌ INCORRECT: No type hints
def extract_evidence(text, threshold=0.7):
    pass
```

## Docstring Template

```python
def method_name(
    param1: str,
    param2: int = 10
) -> Dict[str, Any]:
    """
    One-line summary in imperative mood.
    
    Extended description with context and usage guidance.
    
    Args:
        param1: Parameter description
        param2: Parameter with default
        
    Returns:
        Dict containing:
            - 'key1': Description (type: str)
            - 'key2': Description (type: List)
    
    Raises:
        ValueError: When param1 is empty
        NotImplementedError: If feature not ready
        
    NOTE: Added for adapter compatibility with [AdapterName].
          [Explanation: why needed, what delegates to, or roadmap]
    
    Example:
        >>> result = method_name("text")
        >>> print(result['key1'])
    """
    pass
```

## Validation Checklist

### TIER 1 (Adapter Changes)
- [ ] Inline comment explains substitution
- [ ] Type compatibility verified
- [ ] Return value mapping documented
- [ ] Error handling updated

### TIER 2 (Wrapper Addition)
- [ ] Delegates to existing methods only
- [ ] Complete docstring with NOTE
- [ ] Type hints on all params/return
- [ ] No business logic duplication

### TIER 3 (Stub Addition)
- [ ] NotImplementedError with clear message
- [ ] Docstring includes roadmap reference
- [ ] Alternative methods documented
- [ ] Type hints complete

### API Stability
- [ ] No deletions without deprecation
- [ ] Signature changes backward-compatible
- [ ] CHANGELOG updated

## Integration

Works with:
- `test_architecture_compilation.py` - Import validation
- `orchestrator/circuit_breaker.py` - Graceful degradation
- `orchestrator/choreographer.py` - Method dispatch
- `refactoring_validator.py` - Contract preservation

## References
- FARFAN 3.0 Architecture: `ANALISIS_REPO.md`
- Adapters: `orchestrator/module_adapters.py`
- Source: `policy_processor.py`, `Analyzer_one.py`
- PEP 257: Docstrings
- PEP 484: Type Hints
