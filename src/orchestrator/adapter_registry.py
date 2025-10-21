import json
import logging
import uuid
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


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
    available: bool = True
    registration_error: Optional[str] = None

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
        id_factory: Callable[[], str] = lambda: str(uuid.uuid4())
    ):
        self._clock = clock
        self._id_factory = id_factory
        self.adapters: Dict[str, RegisteredAdapter] = {}
        self._register_all_adapters()

    def _resolve_class(self, class_name: str):
        class StubAdapter:
            def analyze(self, *a, **k):
                return {"ok": True, "adapter": class_name, "confidence": 1.0}
        return StubAdapter

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
                self.adapters[name] = RegisteredAdapter(name=name, instance=instance)
                logger.info(f"[registry] registered adapter={{name}} class={{class_name}}")
            except Exception as e:
                self.adapters[name] = RegisteredAdapter(
                    name=name,
                    instance=None,
                    available=False,
                    registration_error=repr(e)
                )
                logger.error(
                    f"[registry] failed adapter={{name}} class={{class_name}} error={{e}}"
                )

    def list_adapter_methods(self, module_name: str) -> List[str]:
        if module_name not in self.adapters:
            return []
        reg = self.adapters[module_name]
        if not reg.instance:
            return []
        inst = reg.instance
        return [
            m for m in dir(inst)
            if not m.startswith("_") and callable(getattr(inst, m, None))
        ]

    def get_available_modules(self) -> List[str]:
        return [n for n, r in self.adapters.items() if r.available and r.instance]

    def get_status_snapshot(self) -> Dict[str, Any]:
        return {
            name: {
                "available": reg.available,
                "registration_error": reg.registration_error
            }
            for name, reg in self.adapters.items()
        }

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

        if module_name not in self.adapters:
            return self._finalize(
                start, module_name, "UnknownAdapter", method_name,
                status="missing_adapter",
                error_type="AdapterNotFound",
                error_message=f"Adapter '{{module_name}}' not registered",
                confidence=0.0,
                trace_id=trace_id
            )

        reg = self.adapters[module_name]

        if not reg.available or reg.instance is None:
            if not allow_degraded:
                raise ContractViolation(
                    f"Adapter '{{module_name}}' unavailable (error={{reg.registration_error}})"
                )
            return self._finalize(
                start, module_name, "UnavailableAdapter", method_name,
                status="unavailable",
                error_type="AdapterUnavailable",
                error_message=reg.registration_error or "Unavailable",
                confidence=0.0,
                trace_id=trace_id
            )

        inst = reg.instance

        if not hasattr(inst, method_name):
            return self._finalize(
                start, module_name, inst.__class__.__name__, method_name,
                status="missing_method",
                error_type="MethodNotFound",
                error_message=f"Method '{{method_name}}' not found",
                confidence=0.0,
                trace_id=trace_id
            )

        try:
            raw = getattr(inst, method_name)(*args, **kwargs)
            data = raw if isinstance(raw, dict) else {"result": raw}
            confidence = data.get("confidence", 1.0)
            evidence = data.get("evidence", [])
            return self._finalize(
                start, module_name, inst.__class__.__name__, method_name,
                status="success",
                data=data,
                evidence=evidence if isinstance(evidence, list) else [],
                confidence=confidence,
                trace_id=trace_id
            )
        except Exception as e:
            return self._finalize(
                start, module_name, inst.__class__.__name__, method_name,
                status="error",
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
