"""
Dereck Beach Adapter - Causal Deconstruction & Analysis Framework (CDAF)
==========================================================================

Wraps dereck_beach.py functionality with standardized interfaces for the module controller.
Preserves all original class signatures while providing alias methods for controller integration.

Author: FARFAN 3.0 Integration Team
Version: 1.0.0
Python: 3.10+
"""

import logging
import time
import warnings
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class AdapterResult:
    """Standardized result format for adapter operations"""
    module_name: str
    class_name: str
    method_name: str
    status: str
    data: Dict[str, Any]
    evidence: List[Dict[str, Any]]
    confidence: float
    execution_time: float
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class DerekBeachAdapter:
    """
    Adapter for dereck_beach.py - Causal Deconstruction Analysis Framework.
    
    Responsibility Map (cuestionario.json):
    - D6 (Causalidad): Q26-Q30 (Causal mechanisms, sufficiency/necessity tests)
    - D1 (Insumos): Q3 (Resource traceability)
    - D3 (Productos): Q17 (Financial allocation tracing)
    - D4 (Resultados): Q22 (Causal evidence auditing)
    
    Original Classes:
    - ConfigLoader: Configuration management
    - PDFDocumentLoader: PDF document extraction
    - CausalHierarchyExtractor: Extract causal hierarchy
    - FinancialAllocTracer: Trace financial allocations
    - CausalMechanismInference: Infer causal mechanisms
    - PolicyAuditor: Audit evidence traceability
    - CDAFOrchestrator: Main CDAF orchestrator
    """

    def __init__(self, config_path: Optional[Path] = None, output_dir: Optional[Path] = None):
        """
        Initialize adapter with optional configuration.
        
        Args:
            config_path: Path to configuration file (injected dependency)
            output_dir: Output directory for reports
        """
        self.module_name = "dereck_beach"
        self.available = False
        self.logger = logging.getLogger(f"{__name__}.DerekBeach")
        self._config_path = config_path
        self._output_dir = output_dir or Path("./output")
        self._load_module()

    def _load_module(self):
        """Load dereck_beach module and its components"""
        try:
            from dereck_beach import (
                TestType,
                MetaNode,
                ConfigLoader,
                PDFDocumentLoader,
                CausalHierarchyExtractor,
                FinancialAllocTracer,
                CausalMechanismInference,
                PolicyAuditor,
                CDAFOrchestrator,
            )
            
            self.TestType = TestType
            self.MetaNode = MetaNode
            self.ConfigLoader = ConfigLoader
            self.PDFDocumentLoader = PDFDocumentLoader
            self.CausalHierarchyExtractor = CausalHierarchyExtractor
            self.FinancialAllocTracer = FinancialAllocTracer
            self.CausalMechanismInference = CausalMechanismInference
            self.PolicyAuditor = PolicyAuditor
            self.CDAFOrchestrator = CDAFOrchestrator
            
            self.available = True
            self.logger.info(f"✓ {self.module_name} loaded successfully")
            
        except ImportError as e:
            self.logger.warning(f"✗ {self.module_name} NOT available: {e}")
            self.available = False

    # ========================================================================
    # ORIGINAL METHOD SIGNATURES (Preserved)
    # ========================================================================

    def extract_causal_hierarchy(self, extractor: Any, text: str) -> Any:
        """
        Original method: CausalHierarchyExtractor.extract_causal_hierarchy()
        Maps to cuestionario.json: D6.Q26
        """
        if not self.available:
            raise RuntimeError("Module not available")
        
        return extractor.extract_causal_hierarchy(text)

    def infer_mechanisms(
        self,
        inferrer: Any,
        nodes: Dict[str, Any],
        text: str,
        graph: Any,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Original method: CausalMechanismInference.infer_mechanisms()
        Maps to cuestionario.json: D6.Q28-Q29
        """
        if not self.available:
            raise RuntimeError("Module not available")
        
        return inferrer.infer_mechanisms(nodes, text, graph)

    def audit_evidence_traceability(
        self,
        auditor: Any,
        nodes: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Original method: PolicyAuditor.audit_evidence_traceability()
        Maps to cuestionario.json: D4.Q22
        """
        if not self.available:
            raise RuntimeError("Module not available")
        
        return auditor.audit_evidence_traceability(nodes)

    def trace_financial_allocation(
        self,
        tracer: Any,
        tables: List[Any],
        nodes: Dict[str, Any],
        graph: Any,
    ) -> Dict[str, Any]:
        """
        Original method: FinancialAllocTracer.trace_financial_allocation()
        Maps to cuestionario.json: D3.Q17
        """
        if not self.available:
            raise RuntimeError("Module not available")
        
        return tracer.trace_financial_allocation(tables, nodes, graph)

    def process_document(self, orchestrator: Any, pdf_path: Path, policy_code: str) -> bool:
        """
        Original method: CDAFOrchestrator.process_document()
        Full pipeline - Maps to multiple D6 questions
        """
        if not self.available:
            raise RuntimeError("Module not available")
        
        return orchestrator.process_document(pdf_path, policy_code)

    # ========================================================================
    # STANDARDIZED CONTROLLER INTERFACE (Alias Methods)
    # ========================================================================

    def extract_causal_structure(self, text: str) -> AdapterResult:
        """
        Controller method for D6.Q26: Extract causal hierarchy structure
        Alias for: extract_causal_hierarchy
        """
        start_time = time.time()
        
        try:
            if self._config_path is None:
                self._config_path = Path("./config.json")
            
            config = self.ConfigLoader(self._config_path)
            
            import spacy
            nlp = spacy.load("es_core_news_sm")
            
            extractor = self.CausalHierarchyExtractor(config, nlp)
            graph = extractor.extract_causal_hierarchy(text)
            
            data = {
                "node_count": graph.number_of_nodes() if hasattr(graph, 'number_of_nodes') else 0,
                "edge_count": graph.number_of_edges() if hasattr(graph, 'number_of_edges') else 0,
                "goals": [n for n, d in graph.nodes(data=True) if d.get("type") == "goal"],
            }
            
            return AdapterResult(
                module_name=self.module_name,
                class_name="CausalHierarchyExtractor",
                method_name="extract_causal_structure",
                status="success",
                data=data,
                evidence=[{"type": "causal_hierarchy", "nodes": data["node_count"]}],
                confidence=0.85,
                execution_time=time.time() - start_time,
            )
        except Exception as e:
            return self._error_result("extract_causal_structure", start_time, e)

    def infer_causal_mechanisms(
        self,
        text: str,
        nodes: Dict[str, Any],
        graph: Any,
    ) -> AdapterResult:
        """
        Controller method for D6.Q28-Q29: Infer causal mechanisms
        Alias for: infer_mechanisms
        """
        start_time = time.time()
        
        try:
            if self._config_path is None:
                self._config_path = Path("./config.json")
            
            config = self.ConfigLoader(self._config_path)
            
            import spacy
            nlp = spacy.load("es_core_news_sm")
            
            inferrer = self.CausalMechanismInference(config, nlp)
            mechanisms = inferrer.infer_mechanisms(nodes, text, graph)
            
            data = {
                "mechanism_count": len(mechanisms),
                "mechanisms": {k: v.get("mechanism_type", {}) for k, v in mechanisms.items()},
                "avg_confidence": sum(v.get("confidence", 0) for v in mechanisms.values()) / len(mechanisms) if mechanisms else 0,
            }
            
            return AdapterResult(
                module_name=self.module_name,
                class_name="CausalMechanismInference",
                method_name="infer_causal_mechanisms",
                status="success",
                data=data,
                evidence=[{"type": "mechanism_inference", "count": len(mechanisms)}],
                confidence=data["avg_confidence"],
                execution_time=time.time() - start_time,
            )
        except Exception as e:
            return self._error_result("infer_causal_mechanisms", start_time, e)

    def audit_causal_evidence(self, nodes: Dict[str, Any]) -> AdapterResult:
        """
        Controller method for D4.Q22: Audit evidence traceability
        Alias for: audit_evidence_traceability
        """
        start_time = time.time()
        
        try:
            if self._config_path is None:
                self._config_path = Path("./config.json")
            
            config = self.ConfigLoader(self._config_path)
            auditor = self.PolicyAuditor(config)
            audit_results = auditor.audit_evidence_traceability(nodes)
            
            data = {
                "audited_nodes": len(audit_results),
                "passed": sum(1 for r in audit_results.values() if r.status == "pass"),
                "failed": sum(1 for r in audit_results.values() if r.status == "fail"),
                "audit_summary": {k: {"status": v.status, "score": v.score} for k, v in list(audit_results.items())[:5]},
            }
            
            confidence = data["passed"] / data["audited_nodes"] if data["audited_nodes"] > 0 else 0.5
            
            return AdapterResult(
                module_name=self.module_name,
                class_name="PolicyAuditor",
                method_name="audit_causal_evidence",
                status="success" if data["failed"] == 0 else "warnings",
                data=data,
                evidence=[{"type": "evidence_audit", "passed": data["passed"]}],
                confidence=confidence,
                execution_time=time.time() - start_time,
                warnings=[f"{data['failed']} nodes failed audit"] if data["failed"] > 0 else [],
            )
        except Exception as e:
            return self._error_result("audit_causal_evidence", start_time, e)

    def trace_budget_allocation(
        self,
        tables: List[Any],
        nodes: Dict[str, Any],
        graph: Any,
    ) -> AdapterResult:
        """
        Controller method for D3.Q17: Trace financial allocations
        Alias for: trace_financial_allocation
        """
        start_time = time.time()
        
        try:
            if self._config_path is None:
                self._config_path = Path("./config.json")
            
            config = self.ConfigLoader(self._config_path)
            tracer = self.FinancialAllocTracer(config)
            trace_result = tracer.trace_financial_allocation(tables, nodes, graph)
            
            data = {
                "traced_allocations": trace_result.get("allocations", []),
                "total_budget": trace_result.get("total_budget", 0),
                "traced_percentage": trace_result.get("traced_percentage", 0),
                "warnings": trace_result.get("warnings", []),
            }
            
            return AdapterResult(
                module_name=self.module_name,
                class_name="FinancialAllocTracer",
                method_name="trace_budget_allocation",
                status="success",
                data=data,
                evidence=[{"type": "financial_tracing", "allocations": len(data["traced_allocations"])}],
                confidence=data["traced_percentage"] / 100 if data["traced_percentage"] > 0 else 0.5,
                execution_time=time.time() - start_time,
                warnings=data["warnings"],
            )
        except Exception as e:
            return self._error_result("trace_budget_allocation", start_time, e)

    # ========================================================================
    # DEPRECATED SHIM METHODS (With Warnings)
    # ========================================================================

    def extract_hierarchy(self, document: str) -> Dict[str, Any]:
        """
        DEPRECATED: Use extract_causal_structure() instead.
        Legacy method from earlier refactoring iteration.
        """
        warnings.warn(
            "extract_hierarchy() is deprecated. Use extract_causal_structure() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        result = self.extract_causal_structure(document)
        return result.data

    def check_mechanisms(self, text: str, nodes: Dict) -> Dict[str, Any]:
        """
        DEPRECATED: Use infer_causal_mechanisms() instead.
        Returns only simplified mechanism info.
        """
        warnings.warn(
            "check_mechanisms() is deprecated. Use infer_causal_mechanisms() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        import networkx as nx
        graph = nx.DiGraph()
        result = self.infer_causal_mechanisms(text, nodes, graph)
        return result.data

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _error_result(self, method_name: str, start_time: float, error: Exception) -> AdapterResult:
        """Create error result"""
        return AdapterResult(
            module_name=self.module_name,
            class_name="Unknown",
            method_name=method_name,
            status="failed",
            data={},
            evidence=[],
            confidence=0.0,
            execution_time=time.time() - start_time,
            errors=[str(error)],
        )


def create_derek_beach_adapter(
    config_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> DerekBeachAdapter:
    """Factory function to create DerekBeachAdapter instance"""
    return DerekBeachAdapter(config_path=config_path, output_dir=output_dir)
