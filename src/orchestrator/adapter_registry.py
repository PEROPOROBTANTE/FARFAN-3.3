# coding=utf-8
"""
Module Adapter Registry - Canonical Deterministic Implementation
=================================================================

Implements a formal execution contract for adapter invocation with:
- Deterministic adapter registration of 9 core adapters
- Robust error isolation with structured telemetry
- execute_module_method API returning ModuleMethodResult
- Contract assertions via ContractViolation exceptions
- Deterministic logging with injected clock for tests
- Method introspection via list_adapter_methods

SIN_CARRETA Compliance:
- Determinism: Clock injection, trace ID generation (stubbed in tests)
- Contract Clarity: Explicit exceptions, no silent failures
- Auditability: Structured JSON logging per invocation

Author: FARFAN 3.0 Team
Version: 3.3.0
Python: 3.10+
"""

import logging
import json
import time
import uuid
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    """Execution status for adapter method calls"""
    SUCCESS = "success"
    ERROR = "error"
    UNAVAILABLE = "unavailable"
    MISSING_METHOD = "missing_method"
    MISSING_ADAPTER = "missing_adapter"


class ContractViolation(Exception):
    """
    Raised when adapter registry contract is violated
    
    Examples:
    - Attempting to execute an unavailable adapter without allow_degraded=True
    - Attempting to execute a non-existent adapter
    - Attempting to execute a non-existent method
    """
    pass


@dataclass
class ModuleMethodResult:
    """
    Result from a single adapter method execution
    
    Captures execution metadata, status, evidence, and error information
    for audit trail and deterministic testing.
    
    Attributes:
        module_name: Name of the adapter module (e.g., "teoria_cambio")
        adapter_class: Class name of the adapter (e.g., "ModulosAdapter")
        method_name: Name of the method executed (e.g., "calculate_bayesian_confidence")
        status: Execution status (success|error|unavailable|missing_method|missing_adapter)
        start_time: Monotonic clock timestamp at execution start
        end_time: Monotonic clock timestamp at execution end
        execution_time: Duration in seconds (end_time - start_time)
        evidence: List of structured evidence dicts returned from adapter
        error_type: Type of exception if execution failed
        error_message: Error message if execution failed
        confidence: Confidence score (1.0 for success, 0.0 for failure, custom for partial)
        trace_id: UUID4 trace identifier (deterministic in tests via patching)
    """
    module_name: str
    adapter_class: str
    method_name: str
    status: ExecutionStatus
    start_time: float
    end_time: float
    execution_time: float
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    confidence: float = 1.0
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "module_name": self.module_name,
            "adapter_class": self.adapter_class,
            "method_name": self.method_name,
            "status": self.status.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "execution_time": self.execution_time,
            "evidence": self.evidence,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "confidence": self.confidence,
            "trace_id": self.trace_id
        }


@dataclass
class AdapterAvailabilitySnapshot:
    """
    Snapshot of adapter availability status
    
    Used by get_status() to return typed structure instead of raw dict.
    """
    adapter_name: str
    available: bool
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    description: str = ""


class ModuleAdapterRegistry:
    """
    Canonical adapter registry with formal execution contract
    
    Features:
    - Deterministic registration of 9 core adapters with error isolation
    - execute_module_method API returning ModuleMethodResult
    - Contract enforcement via ContractViolation exceptions
    - Deterministic logging with clock injection for tests
    - Method introspection via list_adapter_methods
    
    Core Adapters (9):
    1. ModulosAdapter (teoria_cambio)
    2. AnalyzerOneAdapter (analyzer_one)
    3. DerekBeachAdapter (dereck_beach)
    4. EmbeddingPolicyAdapter (embedding_policy)
    5. SemanticChunkingPolicyAdapter (semantic_chunking_policy)
    6. ContradictionDetectionAdapter (contradiction_detection)
    7. FinancialViabilityAdapter (financial_viability)
    8. PolicyProcessorAdapter (policy_processor)
    9. PolicySegmenterAdapter (policy_segmenter)
    """
    
    def __init__(
        self,
        clock: Optional[Callable[[], float]] = None,
        trace_id_generator: Optional[Callable[[], str]] = None
    ):
        """
        Initialize registry with optional clock and trace ID injection
        
        Args:
            clock: Optional monotonic clock function (default: time.monotonic)
            trace_id_generator: Optional trace ID generator (default: uuid.uuid4)
        """
        self._adapters: Dict[str, Any] = {}
        self._availability: Dict[str, AdapterAvailabilitySnapshot] = {}
        self._clock = clock or time.monotonic
        self._trace_id_generator = trace_id_generator or (lambda: str(uuid.uuid4()))
        
        logger.info("ModuleAdapterRegistry initialized with deterministic contract")
    
    def register_adapter(
        self,
        module_name: str,
        adapter_instance: Any,
        adapter_class_name: str,
        description: str = ""
    ):
        """
        Register an adapter with error isolation
        
        Wraps registration in try/except, capturing exceptions and marking
        adapter as unavailable while logging structured telemetry.
        
        Args:
            module_name: Unique name for the adapter (e.g., "teoria_cambio")
            adapter_instance: Instance of the adapter class
            adapter_class_name: Class name for logging (e.g., "ModulosAdapter")
            description: Optional description of the adapter
        """
        try:
            self._adapters[module_name] = adapter_instance
            self._availability[module_name] = AdapterAvailabilitySnapshot(
                adapter_name=module_name,
                available=True,
                description=description
            )
            logger.info(
                json.dumps({
                    "event": "adapter_registered",
                    "module_name": module_name,
                    "adapter_class": adapter_class_name,
                    "available": True,
                    "description": description
                })
            )
        except Exception as e:
            error_type = type(e).__name__
            error_message = str(e)
            self._availability[module_name] = AdapterAvailabilitySnapshot(
                adapter_name=module_name,
                available=False,
                error_type=error_type,
                error_message=error_message,
                description=description
            )
            logger.error(
                json.dumps({
                    "event": "adapter_registration_failed",
                    "module_name": module_name,
                    "adapter_class": adapter_class_name,
                    "error_type": error_type,
                    "error_message": error_message
                })
            )
    
    def execute_module_method(
        self,
        module_name: str,
        method_name: str,
        args: Optional[List[Any]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        allow_degraded: bool = False
    ) -> ModuleMethodResult:
        """
        Execute adapter method with contract enforcement
        
        CONTRACT ENFORCEMENT:
        - Raises ContractViolation if adapter is unavailable (unless allow_degraded=True)
        - Raises ContractViolation if adapter does not exist
        - Returns ModuleMethodResult with missing_method status if method doesn't exist
        
        DETERMINISTIC EXECUTION:
        - Uses injected clock for start_time, end_time, execution_time
        - Uses injected trace_id_generator for trace_id
        - Logs structured JSON per invocation (INFO success, ERROR failure)
        
        Args:
            module_name: Name of the adapter
            method_name: Name of the method to invoke
            args: Positional arguments (default: [])
            kwargs: Keyword arguments (default: {})
            allow_degraded: Allow execution of unavailable adapters (default: False)
        
        Returns:
            ModuleMethodResult with execution outcome and metadata
        
        Raises:
            ContractViolation: If contract is violated (unavailable adapter, missing adapter)
        """
        args = args or []
        kwargs = kwargs or {}
        
        trace_id = self._trace_id_generator()
        start_time = self._clock()
        
        # Check if adapter exists
        if module_name not in self._availability:
            end_time = self._clock()
            result = ModuleMethodResult(
                module_name=module_name,
                adapter_class="Unknown",
                method_name=method_name,
                status=ExecutionStatus.MISSING_ADAPTER,
                start_time=start_time,
                end_time=end_time,
                execution_time=end_time - start_time,
                error_type="ContractViolation",
                error_message=f"Adapter '{module_name}' not registered",
                confidence=0.0,
                trace_id=trace_id
            )
            
            logger.error(json.dumps(result.to_dict()))
            raise ContractViolation(f"Adapter '{module_name}' not registered")
        
        availability = self._availability[module_name]
        adapter_class_name = availability.adapter_name
        
        # Check if adapter is available
        if not availability.available and not allow_degraded:
            end_time = self._clock()
            result = ModuleMethodResult(
                module_name=module_name,
                adapter_class=adapter_class_name,
                method_name=method_name,
                status=ExecutionStatus.UNAVAILABLE,
                start_time=start_time,
                end_time=end_time,
                execution_time=end_time - start_time,
                error_type=availability.error_type or "ContractViolation",
                error_message=availability.error_message or f"Adapter '{module_name}' is unavailable",
                confidence=0.0,
                trace_id=trace_id
            )
            
            logger.error(json.dumps(result.to_dict()))
            raise ContractViolation(
                f"Adapter '{module_name}' is unavailable. "
                f"Error: {availability.error_message}"
            )
        
        # Get adapter instance
        adapter_instance = self._adapters.get(module_name)
        
        if adapter_instance is None:
            end_time = self._clock()
            result = ModuleMethodResult(
                module_name=module_name,
                adapter_class=adapter_class_name,
                method_name=method_name,
                status=ExecutionStatus.UNAVAILABLE,
                start_time=start_time,
                end_time=end_time,
                execution_time=end_time - start_time,
                error_type="ContractViolation",
                error_message=f"Adapter '{module_name}' instance not found",
                confidence=0.0,
                trace_id=trace_id
            )
            
            logger.error(json.dumps(result.to_dict()))
            raise ContractViolation(f"Adapter '{module_name}' instance not found")
        
        # Check if method exists
        if not hasattr(adapter_instance, method_name):
            end_time = self._clock()
            result = ModuleMethodResult(
                module_name=module_name,
                adapter_class=adapter_class_name,
                method_name=method_name,
                status=ExecutionStatus.MISSING_METHOD,
                start_time=start_time,
                end_time=end_time,
                execution_time=end_time - start_time,
                error_type="AttributeError",
                error_message=f"Method '{method_name}' not found on adapter '{module_name}'",
                confidence=0.0,
                trace_id=trace_id
            )
            
            logger.error(json.dumps(result.to_dict()))
            return result
        
        # Execute method
        try:
            method = getattr(adapter_instance, method_name)
            raw_result = method(*args, **kwargs)
            end_time = self._clock()
            
            # Extract evidence from result
            evidence = []
            if isinstance(raw_result, dict):
                evidence = raw_result.get("evidence", [])
                if not isinstance(evidence, list):
                    evidence = [evidence] if evidence else []
            elif isinstance(raw_result, list):
                evidence = raw_result
            
            result = ModuleMethodResult(
                module_name=module_name,
                adapter_class=adapter_class_name,
                method_name=method_name,
                status=ExecutionStatus.SUCCESS,
                start_time=start_time,
                end_time=end_time,
                execution_time=end_time - start_time,
                evidence=evidence,
                confidence=1.0,
                trace_id=trace_id
            )
            
            logger.info(json.dumps(result.to_dict()))
            return result
            
        except Exception as e:
            end_time = self._clock()
            error_type = type(e).__name__
            error_message = str(e)
            
            result = ModuleMethodResult(
                module_name=module_name,
                adapter_class=adapter_class_name,
                method_name=method_name,
                status=ExecutionStatus.ERROR,
                start_time=start_time,
                end_time=end_time,
                execution_time=end_time - start_time,
                error_type=error_type,
                error_message=error_message,
                confidence=0.0,
                trace_id=trace_id
            )
            
            logger.error(json.dumps(result.to_dict()))
            return result
    
    def list_adapter_methods(self, module_name: str) -> List[str]:
        """
        List public methods available on an adapter
        
        Uses dir() filtered to exclude private and dunder methods.
        Useful for pre-flight validation.
        
        Args:
            module_name: Name of the adapter
        
        Returns:
            List of public method names
        
        Raises:
            ContractViolation: If adapter does not exist
        """
        if module_name not in self._adapters:
            raise ContractViolation(f"Adapter '{module_name}' not registered")
        
        adapter_instance = self._adapters[module_name]
        if adapter_instance is None:
            return []
        
        # Filter out private and dunder methods
        all_attrs = dir(adapter_instance)
        public_methods = [
            attr for attr in all_attrs
            if not attr.startswith('_') and callable(getattr(adapter_instance, attr, None))
        ]
        
        return public_methods
    
    def get_status(self) -> Dict[str, AdapterAvailabilitySnapshot]:
        """
        Get availability status for all registered adapters
        
        Returns:
            Dictionary mapping adapter name to AdapterAvailabilitySnapshot
        """
        return dict(self._availability)
    
    def is_available(self, module_name: str) -> bool:
        """
        Check if an adapter is available
        
        Args:
            module_name: Name of the adapter
        
        Returns:
            True if adapter is registered and available, False otherwise
        """
        if module_name not in self._availability:
            return False
        return self._availability[module_name].available
    
    @property
    def adapters(self) -> Dict[str, Any]:
        """
        Get all registered adapters (for backward compatibility)
        
        Returns:
            Dictionary mapping adapter name to adapter instance
        """
        return self._adapters


if __name__ == "__main__":
    print("=" * 80)
    print("MODULE ADAPTER REGISTRY - Canonical Implementation")
    print("=" * 80)
    
    # Example usage with stub adapter
    class StubAdapter:
        def process(self, text: str) -> Dict[str, Any]:
            return {"evidence": [{"text": text, "confidence": 0.9}]}
        
        def analyze(self) -> List[str]:
            return ["result1", "result2"]
    
    registry = ModuleAdapterRegistry()
    
    # Register adapter with error isolation
    registry.register_adapter(
        module_name="stub_adapter",
        adapter_instance=StubAdapter(),
        adapter_class_name="StubAdapter",
        description="Test stub adapter"
    )
    
    print(f"\nRegistered adapters: {list(registry.get_status().keys())}")
    print(f"Available methods: {registry.list_adapter_methods('stub_adapter')}")
    
    # Execute method
    result = registry.execute_module_method(
        module_name="stub_adapter",
        method_name="process",
        args=["test text"]
    )
    
    print(f"\nExecution result:")
    print(f"  Status: {result.status.value}")
    print(f"  Execution time: {result.execution_time:.6f}s")
    print(f"  Evidence count: {len(result.evidence)}")
    print(f"  Trace ID: {result.trace_id}")
    
    print("=" * 80)
