import json
import logging
import uuid
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class ExecutionStatus(str, Enum):
    """Enumeration of execution statuses for ModuleMethodResult"""
    SUCCESS = "success"
    ERROR = "error"
    UNAVAILABLE = "unavailable"
    MISSING_METHOD = "missing_method"
    MISSING_ADAPTER = "missing_adapter"


@dataclass(frozen=True)
class AdapterAvailabilitySnapshot:
    """
    Snapshot of adapter availability status.
    SIN_CARRETA: Explicit availability tracking for contract enforcement.
    """
    adapter_name: str
    available: bool
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    description: str = ""


class ContractViolation(Exception):
    """
    Raised cuando se viola el contrato explícito (p.ej. ejecutar adaptador
    marcado como unavailable sin permitir modo degradado).
    SIN_CARRETA: Mantener semántica de fallo explícito (no warnings silenciosos).
    """
    pass

@dataclass(frozen=True)
class ModuleMethodResult:
    """
    Resultado canónico de una invocación de método de adaptador.

    status: success | error | unavailable | missing_method | missing_adapter
    """
    module_name: str
    adapter_class: str
    method_name: str
    status: str
    start_time: float
    end_time: float
    execution_time: float
    data: Dict[str, Any] = field(default_factory=dict)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    trace_id: str = field(default_factory=lambda: "UNSET")
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "module_name": self.module_name,
            "adapter_class": self.adapter_class,
            "method_name": self.method_name,
            "status": self.status,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "execution_time": self.execution_time,
            "data": self.data,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "trace_id": self.trace_id,
            "metadata": self.metadata,
            "warnings": self.warnings,
            "errors": self.errors
        }

@dataclass
class RegisteredAdapter:
    name: str
    instance: Any
    adapter_class_name: str = "Unknown"
    available: bool = True
    registration_error: Optional[str] = None
    description: str = ""

class ModuleAdapterRegistry:
    """
    Registro canónico con contrato determinista de ejecución.

    SIN_CARRETA-RATIONALE:
    - Inyección de clock & id_factory → pruebas deterministas.
    - Telemetría estructurada JSON por invocación → auditabilidad.
    - Excepciones explícitas → sin degradación silenciosa.
    """

    def __init__(self,
        clock: Callable[[], float] = time.monotonic,
        id_factory: Callable[[], str] = None,
        trace_id_generator: Callable[[], str] = None,
        auto_register: bool = True
    ):
        self._clock = clock
        # Support both id_factory and trace_id_generator for backward compatibility
        if trace_id_generator is not None:
            self._id_factory = trace_id_generator
        elif id_factory is not None:
            self._id_factory = id_factory
        else:
            self._id_factory = lambda: str(uuid.uuid4())
        
        self._adapters: Dict[str, RegisteredAdapter] = {}
        self._availability: Dict[str, AdapterAvailabilitySnapshot] = {}
        if auto_register:
            self._register_all_adapters()

    def _resolve_class(self, class_name: str):
        """
        Resolve adapter class by name from consolidated_adapters module.
        
        SIN_CARRETA-RATIONALE:
        - Direct import from real adapter implementations
        - Explicit error on missing adapter (no silent stub fallback)
        - Clear traceability for debugging and auditing
        """
        try:
            from src.orchestrator.consolidated_adapters import (
                PolicyProcessorAdapter,
                PolicySegmenterAdapter,
                AnalyzerOneAdapter,
                DerekBeachAdapter,
                EmbeddingPolicyAdapter,
                SemanticChunkingPolicyAdapter,
                ContradictionDetectionAdapter,
                FinancialViabilityAdapter,
                ModulosAdapter,
            )
            
            adapter_map = {
                "PolicyProcessorAdapter": PolicyProcessorAdapter,
                "PolicySegmenterAdapter": PolicySegmenterAdapter,
                "AnalyzerOneAdapter": AnalyzerOneAdapter,
                "DerekBeachAdapter": DerekBeachAdapter,
                "EmbeddingPolicyAdapter": EmbeddingPolicyAdapter,
                "SemanticChunkingPolicyAdapter": SemanticChunkingPolicyAdapter,
                "ContradictionDetectionAdapter": ContradictionDetectionAdapter,
                "FinancialViabilityAdapter": FinancialViabilityAdapter,
                "ModulosAdapter": ModulosAdapter,
            }
            
            if class_name not in adapter_map:
                logger.warning(f"[registry] Adapter class '{class_name}' not found in adapter_map. Creating explicit stub.")
                # Return explicit stub with warning (not silent degradation)
                class StubAdapter:
                    def __init__(self):
                        logger.warning(f"Using stub for missing adapter: {class_name}")
                    
                    def analyze(self, *a, **k):
                        return {"ok": True, "adapter": class_name, "confidence": 1.0, "stub": True}
                return StubAdapter
            
            return adapter_map[class_name]
            
        except ImportError as e:
            logger.error(f"[registry] Failed to import adapter '{class_name}': {e}")
            # Return explicit stub with error indication
            class ErrorStubAdapter:
                def __init__(self):
                    logger.error(f"Using error stub for failed import: {class_name}")
                
                def analyze(self, *a, **k):
                    return {"ok": False, "adapter": class_name, "confidence": 0.0, "error": str(e)}
            return ErrorStubAdapter

    def _register_all_adapters(self) -> None:
        adapter_specs = [
            ("teoria_cambio", "ModulosAdapter"),
            ("analyzer_one", "AnalyzerOneAdapter"),
            ("dereck_beach", "DerekBeachAdapter"),
            ("embedding_policy", "EmbeddingPolicyAdapter"),
            ("semantic_chunking_policy", "SemanticChunkingPolicyAdapter"),
            ("contradiction_detection", "ContradictionDetectionAdapter"),
            ("financial_viability", "FinancialViabilityAdapter"),
            ("policy_processor", "PolicyProcessorAdapter"),
            ("policy_segmenter", "PolicySegmenterAdapter"),
        ]
        for name, class_name in adapter_specs:
            try:
                adapter_cls = self._resolve_class(class_name)
                instance = adapter_cls()
                self._adapters[name] = RegisteredAdapter(
                    name=name,
                    instance=instance,
                    adapter_class_name=class_name,
                    available=True
                )
                self._availability[name] = AdapterAvailabilitySnapshot(
                    adapter_name=name,
                    available=True,
                    description=f"Adapter for {name}"
                )
                logger.info(f"[registry] registered adapter={name} class={class_name}")
            except Exception as e:
                self._adapters[name] = RegisteredAdapter(
                    name=name,
                    instance=None,
                    adapter_class_name=class_name,
                    available=False,
                    registration_error=repr(e)
                )
                self._availability[name] = AdapterAvailabilitySnapshot(
                    adapter_name=name,
                    available=False,
                    error_type=type(e).__name__,
                    error_message=str(e)
                )
                logger.error(
                    f"[registry] failed adapter={name} class={class_name} error={e}"
                )

    def list_adapter_methods(self, module_name: str) -> List[str]:
        """
        List available public methods on an adapter.
        
        SIN_CARRETA: Explicit contract violation on missing adapter.
        """
        if module_name not in self._adapters:
            raise ContractViolation(f"Adapter '{module_name}' not registered")
        reg = self._adapters[module_name]
        if not reg.instance:
            raise ContractViolation(f"Adapter '{module_name}' has no instance")
        inst = reg.instance
        return [
            m for m in dir(inst)
            if not m.startswith("_") and callable(getattr(inst, m, None))
        ]

    @property
    def adapters(self) -> Dict[str, Any]:
        """
        Backward compatibility property.
        Returns dict of module_name -> adapter instance (not RegisteredAdapter).
        
        SIN_CARRETA: Maintains backward compatibility while enforcing new contract.
        """
        return {
            name: reg.instance
            for name, reg in self._adapters.items()
            if reg.instance is not None
        }

    def get_available_modules(self) -> List[str]:
        return [n for n, r in self._adapters.items() if r.available and r.instance]

    def get_status_snapshot(self) -> Dict[str, Any]:
        """Get availability snapshot for all adapters"""
        return {
            name: {
                "available": reg.available,
                "registration_error": reg.registration_error
            }
            for name, reg in self._adapters.items()
        }
    
    def get_status(self) -> Dict[str, AdapterAvailabilitySnapshot]:
        """
        Get detailed availability snapshots for all adapters.
        
        SIN_CARRETA: Explicit availability tracking for contract enforcement.
        """
        return self._availability.copy()
    
    def is_available(self, module_name: str) -> bool:
        """Check if an adapter is available"""
        return (
            module_name in self._availability and
            self._availability[module_name].available
        )
    
    def register_adapter(
        self,
        module_name: str,
        adapter_instance: Any,
        adapter_class_name: str,
        description: str = ""
    ) -> None:
        """
        Register an adapter instance.
        
        SIN_CARRETA: Explicit registration for testing and custom adapters.
        """
        self._adapters[module_name] = RegisteredAdapter(
            name=module_name,
            instance=adapter_instance,
            adapter_class_name=adapter_class_name,
            available=True,
            description=description
        )
        self._availability[module_name] = AdapterAvailabilitySnapshot(
            adapter_name=module_name,
            available=True,
            description=description
        )
        logger.info(f"[registry] registered custom adapter={module_name} class={adapter_class_name}")
    
    @property
    def adapters_dict(self) -> Dict[str, Any]:
        """
        Backward compatibility property.
        Returns dict of module_name -> adapter instance.
        """
        return {
            name: reg.instance
            for name, reg in self._adapters.items()
            if reg.instance is not None
        }
    
    def set_adapter_availability(self, module_name: str, available: bool):
        """
        Set adapter availability (mutable for testing).
        
        SIN_CARRETA: Allows tests to simulate unavailable adapters.
        """
        if module_name in self._availability:
            # Create new snapshot with updated availability
            old = self._availability[module_name]
            self._availability[module_name] = AdapterAvailabilitySnapshot(
                adapter_name=old.adapter_name,
                available=available,
                error_type=old.error_type,
                error_message=old.error_message,
                description=old.description
            )
            # Also update the adapter itself
            if module_name in self._adapters:
                reg = self._adapters[module_name]
                self._adapters[module_name] = RegisteredAdapter(
                    name=reg.name,
                    instance=reg.instance,
                    adapter_class_name=reg.adapter_class_name,
                    available=available,
                    registration_error=reg.registration_error,
                    description=reg.description
                )

    def execute_module_method(
        self,
        module_name: str,
        method_name: str,
        args: Optional[List[Any]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        allow_degraded: bool = False
    ) -> ModuleMethodResult:
        start = self._clock()
        args = args or []
        kwargs = kwargs or {}
        trace_id = self._id_factory()

        if module_name not in self._adapters:
            raise ContractViolation(f"Adapter '{module_name}' not registered")

        reg = self._adapters[module_name]

        if not reg.available or reg.instance is None:
            if not allow_degraded:
                raise ContractViolation(
                    f"Adapter '{module_name}' unavailable (error={reg.registration_error})"
                )
            # With allow_degraded, if instance exists, try to execute anyway
            if reg.instance is None:
                return self._finalize(
                    start, module_name, "UnavailableAdapter", method_name,
                    status=ExecutionStatus.UNAVAILABLE.value,
                    error_type="AdapterUnavailable",
                    error_message=reg.registration_error or "Unavailable",
                    confidence=0.0,
                    trace_id=trace_id
                )
            # Continue to execution with the instance

        inst = reg.instance

        if not hasattr(inst, method_name):
            return self._finalize(
                start, module_name, module_name, method_name,  # adapter_class is module_name for consistency
                status=ExecutionStatus.MISSING_METHOD.value,
                error_type="AttributeError",
                error_message=f"Method '{method_name}' not found",
                confidence=0.0,
                trace_id=trace_id
            )

        try:
            raw = getattr(inst, method_name)(*args, **kwargs)
            data = raw if isinstance(raw, dict) else {"result": raw}
            confidence = data.get("confidence", 1.0)
            evidence = data.get("evidence", [])
            return self._finalize(
                start, module_name, module_name, method_name,  # adapter_class is module_name for consistency
                status=ExecutionStatus.SUCCESS.value,
                data=data,
                evidence=evidence if isinstance(evidence, list) else [],
                confidence=confidence,
                trace_id=trace_id
            )
        except Exception as e:
            return self._finalize(
                start, module_name, module_name, method_name,  # adapter_class is module_name for consistency
                status=ExecutionStatus.ERROR.value,
                error_type=type(e).__name__,
                error_message=str(e),
                confidence=0.0,
                trace_id=trace_id
            )

    def _finalize(
        self,
        start: float,
        module_name: str,
        adapter_class: str,
        method_name: str,
        status: str,
        data: Optional[Dict[str, Any]] = None,
        evidence: Optional[List[Dict[str, Any]]] = None,
        confidence: float = 0.0,
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
        trace_id: str = ""
    ) -> ModuleMethodResult:
        end = self._clock()
        execution_time = end - start

        mmr = ModuleMethodResult(
            module_name=module_name,
            adapter_class=adapter_class,
            method_name=method_name,
            status=status,
            start_time=start,
            end_time=end,
            execution_time=execution_time,
            data=data or {},
            evidence=evidence or [],
            confidence=confidence,
            error_type=error_type,
            error_message=error_message,
            trace_id=trace_id,
            errors=[error_message] if error_message and status in {
                "error", "missing_method", "missing_adapter", "unavailable"
            } else []
        )

        log_record = {
            "event": "adapter_method_execution",
            "trace_id": mmr.trace_id,
            "module_name": mmr.module_name,
            "adapter_class": mmr.adapter_class,
            "method_name": mmr.method_name,
            "status": mmr.status,
            "execution_time": round(mmr.execution_time, 6),
            "confidence": mmr.confidence,
            "error_type": mmr.error_type,
            "error_message": mmr.error_message
        }
        if mmr.status == "success":
            logger.info(json.dumps(log_record, ensure_ascii=False))
        else:
            logger.error(json.dumps(log_record, ensure_ascii=False))

        return mmr
