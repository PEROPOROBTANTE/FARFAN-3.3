# coding=utf-8
"""
Immutable Adapter Wrapper - Enforces Immutability for Adapter Operations
=========================================================================

Wraps adapter methods to ensure they operate on immutable data structures
and return immutable results. Prevents mutation of input parameters.

Author: FARFAN 3.0 Team
Version: 1.0.0
Python: 3.10+
"""

import functools
import copy
import logging
from typing import Any, Callable, Dict, List, Tuple, TypeVar
from orchestrator.data_models import (
    QuestionMetadata,
    PolicyChunk,
    ModuleResult,
    Evidence,
    ExecutionStatusEnum,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


def freeze_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively convert mutable dict to frozen representation

    Args:
        d: Dictionary to freeze

    Returns:
        Frozen dictionary with tuples instead of lists
    """
    frozen = {}
    for k, v in d.items():
        if isinstance(v, list):
            frozen[k] = tuple(
                freeze_dict(item) if isinstance(item, dict) else item for item in v
            )
        elif isinstance(v, dict):
            frozen[k] = freeze_dict(v)
        else:
            frozen[k] = v
    return frozen


def convert_to_immutable(data: Any) -> Any:
    """
    Convert mutable data structures to immutable equivalents

    Args:
        data: Data to convert (dict, list, or Pydantic model)

    Returns:
        Immutable version of data
    """
    if isinstance(data, dict):
        return freeze_dict(data)
    elif isinstance(data, list):
        return tuple(convert_to_immutable(item) for item in data)
    elif hasattr(data, "model_dump"):
        # Pydantic v2 model - already immutable if frozen=True
        return data
    else:
        return data


def ensure_immutable_inputs(func: Callable) -> Callable:
    """
    Decorator to ensure adapter method inputs are immutable

    Converts mutable inputs to immutable before calling the method

    Usage:
        @ensure_immutable_inputs
        def process_question(self, question_spec, plan_text):
            # question_spec will be immutable
            ...
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Convert positional args to immutable
        immutable_args = tuple(convert_to_immutable(arg) for arg in args)

        # Convert keyword args to immutable
        immutable_kwargs = {k: convert_to_immutable(v) for k, v in kwargs.items()}

        # Call original function with immutable data
        result = func(*immutable_args, **immutable_kwargs)

        return result

    return wrapper


def ensure_immutable_output(func: Callable) -> Callable:
    """
    Decorator to ensure adapter method outputs are immutable

    Converts method outputs to immutable ModuleResult

    Usage:
        @ensure_immutable_output
        def analyze(self, text):
            return {"result": "analysis"}
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)

        # If already a Pydantic model, return as-is
        if hasattr(result, "model_dump"):
            return result

        # Convert dict result to ModuleResult
        if isinstance(result, dict):
            return convert_dict_to_module_result(result, func.__name__)

        return result

    return wrapper


def convert_dict_to_module_result(
    data: Dict[str, Any],
    method_name: str,
    module_name: str = "unknown",
    class_name: str = "unknown",
) -> ModuleResult:
    """
    Convert dict result to immutable ModuleResult

    Args:
        data: Result dictionary from adapter method
        method_name: Name of the method that produced the result
        module_name: Name of the module
        class_name: Name of the adapter class

    Returns:
        Immutable ModuleResult
    """
    # Extract standard fields
    status_str = data.get("status", "completed")
    try:
        status = ExecutionStatusEnum(status_str)
    except ValueError:
        status = ExecutionStatusEnum.COMPLETED

    errors = data.get("errors", [])
    if isinstance(errors, list):
        errors = tuple(errors)

    # Extract evidence
    evidence_data = data.get("evidence", [])
    if isinstance(evidence_data, list):
        evidence = tuple(
            Evidence(
                text=e.get("text", str(e)) if isinstance(e, dict) else str(e),
                confidence=e.get("confidence", 1.0) if isinstance(e, dict) else 1.0,
                metadata=e.get("metadata", {}) if isinstance(e, dict) else {},
            )
            for e in evidence_data
        )
    else:
        evidence = ()

    return ModuleResult(
        module_name=module_name,
        class_name=class_name,
        method_name=method_name,
        status=status,
        data=freeze_dict(data) if isinstance(data, dict) else {},
        errors=errors,
        execution_time=data.get("execution_time", 0.0),
        evidence=evidence,
        confidence=data.get("confidence", 0.0),
        metadata=freeze_dict(data.get("metadata", {})),
    )


def verify_no_mutation(original: Any, current: Any) -> bool:
    """
    Verify that data has not been mutated

    Args:
        original: Original data snapshot
        current: Current data after operation

    Returns:
        True if data unchanged, False if mutated
    """
    try:
        if hasattr(original, "model_dump") and hasattr(current, "model_dump"):
            return original.model_dump() == current.model_dump()
        elif isinstance(original, dict) and isinstance(current, dict):
            return original == current
        else:
            return original == current
    except Exception:
        return False


class ImmutableAdapterWrapper:
    """
    Wrapper for adapter instances that enforces immutability

    Usage:
        adapter = SomeAdapter()
        wrapped_adapter = ImmutableAdapterWrapper(adapter, "SomeAdapter")

        # All method calls will enforce immutability
        result = wrapped_adapter.process_question(question_spec, text)
    """

    def __init__(self, adapter_instance: Any, class_name: str):
        """
        Initialize wrapper

        Args:
            adapter_instance: Adapter instance to wrap
            class_name: Name of adapter class
        """
        self._adapter = adapter_instance
        self._class_name = class_name
        self._method_calls = 0
        self._mutation_detected = 0

    def __getattr__(self, name: str) -> Any:
        """
        Intercept method calls to enforce immutability

        Args:
            name: Method name

        Returns:
            Wrapped method that enforces immutability
        """
        attr = getattr(self._adapter, name)

        if not callable(attr):
            return attr

        @functools.wraps(attr)
        def wrapped_method(*args, **kwargs):
            self._method_calls += 1

            # Take snapshots of mutable inputs
            snapshots = []
            for arg in args:
                if isinstance(arg, (dict, list)):
                    snapshots.append((arg, copy.deepcopy(arg)))
            for v in kwargs.values():
                if isinstance(v, (dict, list)):
                    snapshots.append((v, copy.deepcopy(v)))

            # Convert to immutable
            immutable_args = tuple(convert_to_immutable(arg) for arg in args)
            immutable_kwargs = {k: convert_to_immutable(v) for k, v in kwargs.items()}

            # Call original method
            try:
                result = attr(*immutable_args, **immutable_kwargs)
            except Exception as e:
                logger.error(f"Error in {self._class_name}.{name}: {e}")
                raise

            # Verify no mutation occurred
            for original, snapshot in snapshots:
                if not verify_no_mutation(snapshot, original):
                    self._mutation_detected += 1
                    logger.warning(
                        f"Mutation detected in {self._class_name}.{name}! "
                        f"Input parameter was modified."
                    )

            # Convert result to immutable ModuleResult if needed
            if isinstance(result, dict):
                result = convert_dict_to_module_result(
                    result, name, self._adapter.__class__.__module__, self._class_name
                )

            return result

        return wrapped_method

    def get_stats(self) -> Dict[str, int]:
        """Get statistics about wrapper usage"""
        return {
            "method_calls": self._method_calls,
            "mutations_detected": self._mutation_detected,
            "adapter_class": self._class_name,
        }


def wrap_adapter_registry(registry: Any) -> Any:
    """
    Wrap all adapters in a registry with immutability enforcement

    Args:
        registry: AdapterRegistry instance

    Returns:
        Registry with wrapped adapters
    """
    if not hasattr(registry, "_adapters"):
        return registry

    for name, adapter_wrapper in registry._adapters.items():
        if hasattr(adapter_wrapper, "module_instance"):
            # Wrap the module instance
            original_instance = adapter_wrapper.module_instance
            wrapped_instance = ImmutableAdapterWrapper(
                original_instance, adapter_wrapper.name
            )
            adapter_wrapper.module_instance = wrapped_instance
            logger.info(f"Wrapped adapter: {name} with immutability enforcement")

    return registry


# ============================================================================
# CONVENIENCE FUNCTIONS FOR COMMON CONVERSIONS
# ============================================================================


def dict_to_question_metadata(data: Dict[str, Any]) -> QuestionMetadata:
    """Convert dict to immutable QuestionMetadata"""
    return QuestionMetadata(
        canonical_id=data["canonical_id"],
        policy_area=data["policy_area"],
        dimension=data["dimension"],
        question_number=data["question_number"],
        question_text=data["question_text"],
        scoring_modality=data.get("scoring_modality", "TYPE_A"),
        expected_elements=tuple(data.get("expected_elements", [])),
        element_weights=data.get("element_weights", {}),
        numerical_thresholds=data.get("numerical_thresholds", {}),
        verification_patterns=tuple(data.get("verification_patterns", [])),
        metadata=data.get("metadata", {}),
    )


def dict_to_policy_chunk(data: Dict[str, Any]) -> PolicyChunk:
    """Convert dict to immutable PolicyChunk"""
    return PolicyChunk(
        chunk_id=data["chunk_id"],
        text=data["text"],
        start_position=data["start_position"],
        end_position=data["end_position"],
        section_title=data.get("section_title"),
        page_number=data.get("page_number"),
        metadata=data.get("metadata", {}),
    )


def list_to_policy_chunks(chunks_list: List[Dict[str, Any]]) -> Tuple[PolicyChunk, ...]:
    """Convert list of chunk dicts to tuple of immutable PolicyChunks"""
    return tuple(dict_to_policy_chunk(c) for c in chunks_list)


if __name__ == "__main__":
    # Demo usage
    print("=" * 60)
    print("Immutable Adapter Wrapper Demo")
    print("=" * 60)

    # Create a mock adapter
    class MockAdapter:
        def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
            # This would normally mutate the input (BAD!)
            # data["processed"] = True
            return {"status": "completed", "confidence": 0.9}

    # Wrap it
    adapter = MockAdapter()
    wrapped = ImmutableAdapterWrapper(adapter, "MockAdapter")

    # Use it
    test_data = {"key": "value"}
    result = wrapped.process(test_data)

    print(f"Result: {result}")
    print(f"Stats: {wrapped.get_stats()}")
