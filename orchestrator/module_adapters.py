# module_adapters.py - Complete Integration

import logging
import time
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import importlib.util
import numpy as np
import pandas as pd
import re
import hashlib
import json
from collections import defaultdict

# Add parent directory to path for module imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


# ============================================================================
# STANDARDIZED OUTPUT FORMAT
# ============================================================================

@dataclass
class ModuleResult:
    """
    Formato estandarizado de salida de TODOS los módulos
    """
    module_name: str
    class_name: str
    method_name: str
    status: str  # "success" | "partial" | "failed"
    data: Dict[str, Any]
    evidence: List[Dict[str, Any]]
    confidence: float
    execution_time: float
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# ADAPTER 1: POLICY PROCESSOR
# ============================================================================

class PolicyProcessorAdapter:
    """
    Adapter para policy_processor.py
    """

    def __init__(self):
        self.module_name = "policy_processor"
        self.available = False
        self._load_module()

    def _load_module(self):
        """Carga las clases REALES del módulo"""
        try:
            from policy_processor import (
                IndustrialPolicyProcessor,
                PolicyTextProcessor,
                BayesianEvidenceScorer,
                EvidenceBundle
            )
            self.IndustrialPolicyProcessor = IndustrialPolicyProcessor
            self.PolicyTextProcessor = PolicyTextProcessor
            self.BayesianEvidenceScorer = BayesianEvidenceScorer
            self.EvidenceBundle = EvidenceBundle
            self.available = True
            logger.info(f"✓ {self.module_name} loaded with 4 classes")
        except ImportError as e:
            logger.warning(f"✗ {self.module_name} NOT available: {e}")
            self.available = False

    def execute(self, method_name: str, args: List[Any], kwargs: Dict[str, Any]) -> ModuleResult:
        """
        MÉTODOS SOPORTADOS:
        - process(text: str) -> Dict
        - _extract_point_evidence(text: str, dimension: str) -> List[str]
        - extract_policy_sections(text: str) -> Dict[str, str]
        - score_evidence(bundle: EvidenceBundle) -> float
        """
        start_time = time.time()

        if not self.available:
            return self._create_unavailable_result(method_name, start_time)

        try:
            if method_name == "process":
                result = self._execute_process(*args, **kwargs)
            elif method_name == "_extract_point_evidence":
                result = self._execute_extract_point_evidence(*args, **kwargs)
            elif method_name == "extract_policy_sections":
                result = self._execute_extract_sections(*args, **kwargs)
            elif method_name == "score_evidence":
                result = self._execute_score_evidence(*args, **kwargs)
            else:
                raise ValueError(f"Unknown method: {method_name}")

            result.execution_time = time.time() - start_time
            return result

        except Exception as e:
            logger.error(f"{self.module_name}.{method_name} failed: {e}", exc_info=True)
            return ModuleResult(
                module_name=self.module_name,
                class_name="IndustrialPolicyProcessor",
                method_name=method_name,
                status="failed",
                data={},
                evidence=[],
                confidence=0.0,
                execution_time=time.time() - start_time,
                errors=[str(e)]
            )

    def _execute_process(self, text: str, **kwargs) -> ModuleResult:
        """Ejecuta IndustrialPolicyProcessor.process()"""
        config = kwargs.get('config', {
            "enable_causal_analysis": True,
            "enable_bayesian_scoring": True,
            "dimension_taxonomy": ["D1", "D2", "D3", "D4", "D5", "D6"]
        })

        processor = self.IndustrialPolicyProcessor(config=config)
        result = processor.process(text)

        evidence = []
        for dimension in ["D1", "D2", "D3", "D4", "D5", "D6"]:
            if dimension in result.get("dimensions", {}):
                dim_data = result["dimensions"][dimension]
                evidence.append({
                    "dimension": dimension,
                    "point_evidence": dim_data.get("point_evidence", []),
                    "bayesian_score": dim_data.get("bayesian_score", 0.0),
                    "causal_links": dim_data.get("causal_links", []),
                    "confidence": dim_data.get("confidence", 0.5)
                })

        confidence = result.get("overall_score", 0.5)

        return ModuleResult(
            module_name=self.module_name,
            class_name="IndustrialPolicyProcessor",
            method_name="process",
            status="success",
            data=result,
            evidence=evidence,
            confidence=confidence,
            execution_time=0.0,
            metadata={"dimensions_processed": len(evidence)}
        )

    def _execute_extract_point_evidence(self, text: str, dimension: str, **kwargs) -> ModuleResult:
        """Ejecuta IndustrialPolicyProcessor._extract_point_evidence()"""
        processor = self.IndustrialPolicyProcessor()
        point_evidence = processor._extract_point_evidence(text, dimension)

        evidence = [{
            "dimension": dimension,
            "evidence_items": point_evidence,
            "count": len(point_evidence)
        }]

        confidence = min(1.0, len(point_evidence) / 5.0)

        return ModuleResult(
            module_name=self.module_name,
            class_name="IndustrialPolicyProcessor",
            method_name="_extract_point_evidence",
            status="success",
            data={"point_evidence": point_evidence, "dimension": dimension},
            evidence=evidence,
            confidence=confidence,
            execution_time=0.0
        )

    def _execute_extract_sections(self, text: str, **kwargs) -> ModuleResult:
        """Ejecuta PolicyTextProcessor.extract_policy_sections()"""
        processor = self.PolicyTextProcessor()
        sections = processor.extract_policy_sections(text)

        evidence = [{
            "sections_extracted": list(sections.keys()),
            "section_count": len(sections)
        }]

        return ModuleResult(
            module_name=self.module_name,
            class_name="PolicyTextProcessor",
            method_name="extract_policy_sections",
            status="success",
            data=sections,
            evidence=evidence,
            confidence=0.8,
            execution_time=0.0
        )

    def _execute_score_evidence(self, bundle, **kwargs) -> ModuleResult:
        """Ejecuta BayesianEvidenceScorer.score_evidence()"""
        scorer = self.BayesianEvidenceScorer()
        score = scorer.score_evidence(bundle)

        return ModuleResult(
            module_name=self.module_name,
            class_name="BayesianEvidenceScorer",
            method_name="score_evidence",
            status="success",
            data={"score": score},
            evidence=[{"bayesian_score": score}],
            confidence=score,
            execution_time=0.0
        )

    def _create_unavailable_result(self, method_name: str, start_time: float) -> ModuleResult:
        """Resultado cuando el módulo no está disponible"""
        return ModuleResult(
            module_name=self.module_name,
            class_name="Unknown",
            method_name=method_name,
            status="failed",
            data={},
            evidence=[],
            confidence=0.0,
            execution_time=time.time() - start_time,
            errors=["Module not available"]
        )


# ============================================================================
# ADAPTER 2: ANALYZER ONE (MUNICIPAL ANALYZER)
# ============================================================================

class AnalyzerOneAdapter:
    """
    Adapter para Analyzer_one.py
    """

    def __init__(self):
        self.module_name = "analyzer_one"
        self.available = False
        self._load_module()

    def _load_module(self):
        try:
            from Analyzer_one import (
                MunicipalAnalyzer,
                SemanticAnalyzer,
                PerformanceAnalyzer,
                TextMiningEngine
            )
            self.MunicipalAnalyzer = MunicipalAnalyzer
            self.SemanticAnalyzer = SemanticAnalyzer
            self.PerformanceAnalyzer = PerformanceAnalyzer
            self.TextMiningEngine = TextMiningEngine
            self.available = True
            logger.info(f"✓ {self.module_name} loaded with 4 classes")
        except ImportError as e:
            logger.warning(f"✗ {self.module_name} NOT available: {e}")

    def execute(self, method_name: str, args: List[Any], kwargs: Dict[str, Any]) -> ModuleResult:
        """
        MÉTODOS SOPORTADOS:
        - analyze_document(text: str) -> Dict
        - extract_semantic_cube(text: str) -> Dict
        - diagnose_critical_links(value_chain: Dict) -> List[Dict]
        - extract_value_chain(text: str) -> Dict
        """
        start_time = time.time()

        if not self.available:
            return self._create_unavailable_result(method_name, start_time)

        try:
            if method_name == "analyze_document":
                result = self._execute_analyze_document(*args, **kwargs)
            elif method_name == "extract_semantic_cube":
                result = self._execute_extract_semantic_cube(*args, **kwargs)
            elif method_name == "diagnose_critical_links":
                result = self._execute_diagnose_critical_links(*args, **kwargs)
            elif method_name == "extract_value_chain":
                result = self._execute_extract_value_chain(*args, **kwargs)
            else:
                raise ValueError(f"Unknown method: {method_name}")

            result.execution_time = time.time() - start_time
            return result

        except Exception as e:
            logger.error(f"{self.module_name}.{method_name} failed: {e}", exc_info=True)
            return ModuleResult(
                module_name=self.module_name,
                class_name="MunicipalAnalyzer",
                method_name=method_name,
                status="failed",
                data={},
                evidence=[],
                confidence=0.0,
                execution_time=time.time() - start_time,
                errors=[str(e)]
            )

    def _execute_analyze_document(self, text: str, **kwargs) -> ModuleResult:
        """Ejecuta MunicipalAnalyzer.analyze_document()"""
        model_name = kwargs.get('model_name', 'bert-base-multilingual-cased')
        analyzer = self.MunicipalAnalyzer(model_name=model_name)
        result = analyzer.analyze_document(text)

        evidence = []

        if "semantic_analysis" in result:
            semantic = result["semantic_analysis"]
            evidence.append({
                "type": "semantic_cube",
                "data": semantic.get("semantic_cube", {}),
                "confidence": semantic.get("confidence", 0.6)
            })

        if "value_chain" in result:
            value_chain = result["value_chain"]
            evidence.append({
                "type": "value_chain",
                "insumos": value_chain.get("insumos", []),
                "actividades": value_chain.get("actividades", []),
                "productos": value_chain.get("productos", []),
                "resultados": value_chain.get("resultados", []),
                "impactos": value_chain.get("impactos", [])
            })

        if "critical_links" in result:
            evidence.append({
                "type": "critical_links",
                "links": result["critical_links"],
                "count": len(result["critical_links"])
            })

        confidence = result.get("overall_confidence", 0.6)

        return ModuleResult(
            module_name=self.module_name,
            class_name="MunicipalAnalyzer",
            method_name="analyze_document",
            status="success",
            data=result,
            evidence=evidence,
            confidence=confidence,
            execution_time=0.0,
            metadata={"evidence_types": len(evidence)}
        )

    def _execute_extract_semantic_cube(self, text: str, **kwargs) -> ModuleResult:
        """Ejecuta SemanticAnalyzer.extract_semantic_cube()"""
        analyzer = self.SemanticAnalyzer()
        semantic_cube = analyzer.extract_semantic_cube(text)

        evidence = [{
            "type": "semantic_cube",
            "dimensions": semantic_cube.keys(),
            "data": semantic_cube
        }]

        return ModuleResult(
            module_name=self.module_name,
            class_name="SemanticAnalyzer",
            method_name="extract_semantic_cube",
            status="success",
            data=semantic_cube,
            evidence=evidence,
            confidence=0.7,
            execution_time=0.0
        )

    def _execute_diagnose_critical_links(self, value_chain: Dict, **kwargs) -> ModuleResult:
        """Ejecuta PerformanceAnalyzer.diagnose_critical_links()"""
        analyzer = self.PerformanceAnalyzer()
        critical_links = analyzer.diagnose_critical_links(value_chain)

        evidence = [{
            "type": "bottleneck_diagnosis",
            "critical_links": critical_links,
            "bottleneck_count": len([l for l in critical_links if l.get("bottleneck_severity", 0) > 0.7])
        }]

        bottleneck_penalty = len([l for l in critical_links if l.get("bottleneck_severity", 0) > 0.7]) * 0.1
        confidence = max(0.3, 0.8 - bottleneck_penalty)

        return ModuleResult(
            module_name=self.module_name,
            class_name="PerformanceAnalyzer",
            method_name="diagnose_critical_links",
            status="success",
            data={"critical_links": critical_links},
            evidence=evidence,
            confidence=confidence,
            execution_time=0.0
        )

    def _execute_extract_value_chain(self, text: str, **kwargs) -> ModuleResult:
        """Ejecuta TextMiningEngine.extract_value_chain()"""
        engine = self.TextMiningEngine()
        value_chain = engine.extract_value_chain(text)

        evidence = [{
            "type": "value_chain_extraction",
            "chain_length": sum(len(v) for v in value_chain.values() if isinstance(v, list)),
            "dimensions": list(value_chain.keys())
        }]

        return ModuleResult(
            module_name=self.module_name,
            class_name="TextMiningEngine",
            method_name="extract_value_chain",
            status="success",
            data=value_chain,
            evidence=evidence,
            confidence=0.7,
            execution_time=0.0
        )

    def _create_unavailable_result(self, method_name: str, start_time: float) -> ModuleResult:
        return ModuleResult(
            module_name=self.module_name,
            class_name="Unknown",
            method_name=method_name,
            status="failed",
            data={},
            evidence=[],
            confidence=0.0,
            execution_time=time.time() - start_time,
            errors=["Module not available"]
        )


# ============================================================================
# ADAPTER 3: DEREK BEACH (CDAF FRAMEWORK)
# ============================================================================

class DerekBeachAdapter:
    """
    Adapter para dereck_beach (CDAFFramework)
    """

    def __init__(self):
        self.module_name = "dereck_beach"
        self.available = False
        self._load_module()

    def _load_module(self):
        try:
            # Cargar módulo sin extensión .py
            spec = importlib.util.spec_from_file_location(
                "dereck_beach",
                "/Users/recovered/PycharmProjects/FLUX/FARFAN-3.0/dereck_beach"
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Extraer TODAS las clases principales
                self.CDAFFramework = getattr(module, "CDAFFramework")
                self.BeachEvidentialTest = getattr(module, "BeachEvidentialTest")
                self.CausalExtractor = getattr(module, "CausalExtractor", None)
                self.MechanismPartExtractor = getattr(module, "MechanismPartExtractor", None)
                self.BayesianMechanismInference = getattr(module, "BayesianMechanismInference", None)
                self.FinancialAuditor = getattr(module, "FinancialAuditor", None)
                self.OperationalizationAuditor = getattr(module, "OperationalizationAuditor", None)
                self.ReportingEngine = getattr(module, "ReportingEngine", None)
                self.ConfigLoader = getattr(module, "ConfigLoader", None)

                self.available = True
                logger.info(f"✓ {self.module_name} loaded with CDAFFramework + 8 core classes")
        except Exception as e:
            logger.warning(f"✗ {self.module_name} NOT available: {e}")

    def execute(self, method_name: str, args: List[Any], kwargs: Dict[str, Any]) -> ModuleResult:
        """
        MÉTODOS SOPORTADOS:
        - process_document(pdf_path_or_text, plan_name) -> Dict
        - classify_test(necessity, sufficiency) -> TestType
        - apply_test_logic(test_type, evidence_found, prior, bayes_factor) -> Tuple[float, str]
        - extract_causal_hierarchy(text) -> Tuple[DiGraph, List[CausalLink]]
        - extract_entity_activity(text) -> List[EntityActivity]
        - infer_mechanisms(nodes, links, activities) -> List[Dict]
        - trace_financial_allocation(nodes, tables) -> Dict
        - audit_evidence_traceability(nodes, links) -> AuditResult
        - generate_confidence_report(mechanisms) -> Dict
        """
        start_time = time.time()

        if not self.available:
            return self._create_unavailable_result(method_name, start_time)

        try:
            if method_name == "process_document":
                result = self._execute_process_document(*args, **kwargs)
            elif method_name == "classify_test":
                result = self._execute_classify_test(*args, **kwargs)
            elif method_name == "apply_test_logic":
                result = self._execute_apply_test_logic(*args, **kwargs)
            elif method_name == "extract_causal_hierarchy":
                result = self._execute_extract_causal_hierarchy(*args, **kwargs)
            elif method_name == "extract_entity_activity":
                result = self._execute_extract_entity_activity(*args, **kwargs)
            elif method_name == "infer_mechanisms":
                result = self._execute_infer_mechanisms(*args, **kwargs)
            elif method_name == "trace_financial_allocation":
                result = self._execute_trace_financial(*args, **kwargs)
            elif method_name == "audit_evidence_traceability":
                result = self._execute_audit_evidence(*args, **kwargs)
            elif method_name == "generate_confidence_report":
                result = self._execute_generate_confidence_report(*args, **kwargs)
            else:
                raise ValueError(f"Unknown method: {method_name}")

            result.execution_time = time.time() - start_time
            return result

        except Exception as e:
            logger.error(f"{self.module_name}.{method_name} failed: {e}", exc_info=True)
            return ModuleResult(
                module_name=self.module_name,
                class_name="CDAFFramework",
                method_name=method_name,
                status="failed",
                data={},
                evidence=[],
                confidence=0.0,
                execution_time=time.time() - start_time,
                errors=[str(e)]
            )

    def _execute_process_document(self, pdf_path_or_text: str, plan_name: str, **kwargs) -> ModuleResult:
        """Ejecuta CDAFFramework.process_document()"""
        config_path = kwargs.get('config_path', Path("config/cdaf_config.yaml"))

        framework = self.CDAFFramework(config_path=config_path)
        result = framework.process_document(pdf_path_or_text, plan_name)

        evidence = []

        if "causal_hierarchy" in result:
            graph = result["causal_hierarchy"]
            evidence.append({
                "type": "causal_graph",
                "node_count": graph.number_of_nodes() if hasattr(graph, 'number_of_nodes') else 0,
                "edge_count": graph.number_of_edges() if hasattr(graph, 'number_of_edges') else 0
            })

        if "mechanism_inferences" in result:
            mechanisms = result["mechanism_inferences"]
            evidence.append({
                "type": "mechanisms",
                "mechanism_count": len(mechanisms),
                "high_confidence_mechanisms": len([m for m in mechanisms if m.get("posterior_confidence", 0) > 0.7])
            })

        if "bayesian_confidence_report" in result:
            report = result["bayesian_confidence_report"]
            evidence.append({
                "type": "bayesian_confidence",
                "mean_confidence": report.get("mean_confidence", 0.5),
                "high_confidence_count": report.get("high_confidence_count", 0)
            })

        confidence = result.get("bayesian_confidence_report", {}).get("mean_confidence", 0.5)

        return ModuleResult(
            module_name=self.module_name,
            class_name="CDAFFramework",
            method_name="process_document",
            status="success",
            data=result,
            evidence=evidence,
            confidence=confidence,
            execution_time=0.0,
            metadata={
                "plan_name": plan_name,
                "evidence_types": len(evidence)
            }
        )

    def _execute_classify_test(self, necessity: float, sufficiency: float, **kwargs) -> ModuleResult:
        """Ejecuta BeachEvidentialTest.classify_test()"""
        test_type = self.BeachEvidentialTest.classify_test(necessity, sufficiency)

        evidence = [{
            "type": "beach_test_classification",
            "test_type": test_type,
            "necessity": necessity,
            "sufficiency": sufficiency
        }]

        return ModuleResult(
            module_name=self.module_name,
            class_name="BeachEvidentialTest",
            method_name="classify_test",
            status="success",
            data={"test_type": test_type, "necessity": necessity, "sufficiency": sufficiency},
            evidence=evidence,
            confidence=0.9,  # Alta confianza en clasificación formal
            execution_time=0.0
        )

    def _execute_apply_test_logic(self, test_type: str, evidence_found: bool,
                                  prior: float, bayes_factor: float, **kwargs) -> ModuleResult:
        """Ejecuta BeachEvidentialTest.apply_test_logic()"""
        posterior_confidence, interpretation = self.BeachEvidentialTest.apply_test_logic(
            test_type, evidence_found, prior, bayes_factor
        )

        evidence = [{
            "type": "beach_test_result",
            "test_type": test_type,
            "evidence_found": evidence_found,
            "prior": prior,
            "bayes_factor": bayes_factor,
            "posterior_confidence": posterior_confidence,
            "interpretation": interpretation
        }]

        return ModuleResult(
            module_name=self.module_name,
            class_name="BeachEvidentialTest",
            method_name="apply_test_logic",
            status="success",
            data={
                "posterior_confidence": posterior_confidence,
                "interpretation": interpretation
            },
            evidence=evidence,
            confidence=posterior_confidence,
            execution_time=0.0,
            metadata={"test_type": test_type}
        )

    def _execute_extract_causal_hierarchy(self, text: str, **kwargs) -> ModuleResult:
        """Ejecuta CausalExtractor.extract_causal_hierarchy()"""
        if not self.CausalExtractor:
            raise RuntimeError("CausalExtractor not available")

        extractor = self.CausalExtractor(text, config={})
        graph, links = extractor.extract_causal_hierarchy(text)

        evidence = [{
            "type": "causal_hierarchy",
            "node_count": graph.number_of_nodes(),
            "link_count": len(links)
        }]

        return ModuleResult(
            module_name=self.module_name,
            class_name="CausalExtractor",
            method_name="extract_causal_hierarchy",
            status="success",
            data={"graph": graph, "links": links},
            evidence=evidence,
            confidence=0.7,
            execution_time=0.0
        )

    def _execute_extract_entity_activity(self, text: str, **kwargs) -> ModuleResult:
        """Ejecuta MechanismPartExtractor.extract_entity_activity()"""
        if not self.MechanismPartExtractor:
            raise RuntimeError("MechanismPartExtractor not available")

        extractor = self.MechanismPartExtractor(config={})
        entity_activities = extractor.extract_entity_activity(text)

        evidence = [{
            "type": "entity_activities",
            "activity_count": len(entity_activities),
            "sample": entity_activities[:5]
        }]

        return ModuleResult(
            module_name=self.module_name,
            class_name="MechanismPartExtractor",
            method_name="extract_entity_activity",
            status="success",
            data={"entity_activities": entity_activities},
            evidence=evidence,
            confidence=0.7,
            execution_time=0.0
        )

    def _execute_infer_mechanisms(self, nodes, links, entity_activities, **kwargs) -> ModuleResult:
        """Ejecuta BayesianMechanismInference.infer_mechanisms()"""
        if not self.BayesianMechanismInference:
            raise RuntimeError("BayesianMechanismInference not available")

        inference = self.BayesianMechanismInference(config={})
        mechanisms = inference.infer_mechanisms(nodes, links, entity_activities)

        evidence = [{
            "type": "inferred_mechanisms",
            "mechanism_count": len(mechanisms),
            "high_confidence": len([m for m in mechanisms if m.get("posterior_confidence", 0) > 0.7])
        }]

        avg_confidence = sum(m.get("posterior_confidence", 0) for m in mechanisms) / max(1, len(mechanisms))

        return ModuleResult(
            module_name=self.module_name,
            class_name="BayesianMechanismInference",
            method_name="infer_mechanisms",
            status="success",
            data={"mechanisms": mechanisms},
            evidence=evidence,
            confidence=avg_confidence,
            execution_time=0.0
        )

    def _execute_trace_financial(self, nodes, tables, **kwargs) -> ModuleResult:
        """Ejecuta FinancialAuditor.trace_financial_allocation()"""
        if not self.FinancialAuditor:
            raise RuntimeError("FinancialAuditor not available")

        auditor = self.FinancialAuditor(config={})
        audit_result = auditor.trace_financial_allocation(nodes, tables)

        evidence = [{
            "type": "financial_audit",
            "traceability_complete": audit_result.get("budget_traceability", False),
            "missing_count": len(audit_result.get("missing_allocations", [])),
            "discrepancy_count": len(audit_result.get("discrepancies", []))
        }]

        confidence = 1.0 if audit_result.get("budget_traceability") else 0.4

        return ModuleResult(
            module_name=self.module_name,
            class_name="FinancialAuditor",
            method_name="trace_financial_allocation",
            status="success",
            data=audit_result,
            evidence=evidence,
            confidence=confidence,
            execution_time=0.0
        )

    def _execute_audit_evidence(self, nodes, links, **kwargs) -> ModuleResult:
        """Ejecuta OperationalizationAuditor.audit_evidence_traceability()"""
        if not self.OperationalizationAuditor:
            raise RuntimeError("OperationalizationAuditor not available")

        auditor = self.OperationalizationAuditor(config={})
        audit_result = auditor.audit_evidence_traceability(nodes, links)

        evidence = [{
            "type": "operationalization_audit",
            "passed": audit_result.get("passed", False),
            "warning_count": len(audit_result.get("warnings", [])),
            "error_count": len(audit_result.get("errors", []))
        }]

        confidence = 1.0 if audit_result.get("passed") else 0.5

        return ModuleResult(
            module_name=self.module_name,
            class_name="OperationalizationAuditor",
            method_name="audit_evidence_traceability",
            status="success",
            data=audit_result,
            evidence=evidence,
            confidence=confidence,
            execution_time=0.0
        )

    def _execute_generate_confidence_report(self, mechanisms, **kwargs) -> ModuleResult:
        """Ejecuta ReportingEngine.generate_confidence_report()"""
        if not self.ReportingEngine:
            raise RuntimeError("ReportingEngine not available")

        reporter = self.ReportingEngine(config={})
        report = reporter.generate_confidence_report(mechanisms)

        evidence = [{
            "type": "confidence_report",
            "mean_confidence": report.get("mean_confidence", 0.5),
            "high_confidence_count": report.get("high_confidence_count", 0)
        }]

        return ModuleResult(
            module_name=self.module_name,
            class_name="ReportingEngine",
            method_name="generate_confidence_report",
            status="success",
            data=report,
            evidence=evidence,
            confidence=report.get("mean_confidence", 0.5),
            execution_time=0.0
        )

    def _create_unavailable_result(self, method_name: str, start_time: float) -> ModuleResult:
        return ModuleResult(
            module_name=self.module_name,
            class_name="CDAFFramework",
            method_name=method_name,
            status="failed",
            data={},
            evidence=[],
            confidence=0.0,
            execution_time=time.time() - start_time,
            errors=["Module not available"]
        )


# ============================================================================
# ADAPTER 4: CONTRADICTION DETECTOR
# ============================================================================

class ContradictionDetectorAdapter:
    """
    Adapter para contradiction_deteccion.py
    """

    def __init__(self):
        self.module_name = "contradiction_detector"
        self.available = False
        self._load_module()

    def _load_module(self):
        try:
            from contradiction_deteccion import (
                PolicyContradictionDetector,
                BayesianConfidenceCalculator,
                TemporalLogicVerifier
            )
            self.PolicyContradictionDetector = PolicyContradictionDetector
            self.BayesianConfidenceCalculator = BayesianConfidenceCalculator
            self.TemporalLogicVerifier = TemporalLogicVerifier
            self.available = True
            logger.info(f"✓ {self.module_name} loaded with 7 classes")
        except ImportError as e:
            logger.warning(f"✗ {self.module_name} NOT available: {e}")

    def execute(self, method_name: str, args: List[Any], kwargs: Dict[str, Any]) -> ModuleResult:
        """
        MÉTODOS SOPORTADOS:
        - detect(text: str, plan_name: str, dimension: PolicyDimension) -> Dict
        - calculate_confidence(contradictions: List[Dict]) -> float
        - verify_temporal_logic(statements: List) -> Tuple[bool, List[Dict]]
        """
        start_time = time.time()

        if not self.available:
            return self._create_unavailable_result(method_name, start_time)

        try:
            if method_name == "detect":
                result = self._execute_detect(*args, **kwargs)
            elif method_name == "calculate_confidence":
                result = self._execute_calculate_confidence(*args, **kwargs)
            elif method_name == "verify_temporal_logic":
                result = self._execute_verify_temporal(*args, **kwargs)
            else:
                raise ValueError(f"Unknown method: {method_name}")

            result.execution_time = time.time() - start_time
            return result

        except Exception as e:
            logger.error(f"{self.module_name}.{method_name} failed: {e}", exc_info=True)
            return ModuleResult(
                module_name=self.module_name,
                class_name="PolicyContradictionDetector",
                method_name=method_name,
                status="failed",
                data={},
                evidence=[],
                confidence=0.0,
                execution_time=time.time() - start_time,
                errors=[str(e)]
            )

    def _execute_detect(self, text: str, plan_name: str = "Unknown", **kwargs) -> ModuleResult:
        """Ejecuta PolicyContradictionDetector.detect()"""
        dimension = kwargs.get('dimension', PolicyDimension.ESTRATEGICO)
        detector = self.PolicyContradictionDetector()
        result = detector.detect(text, plan_name, dimension)

        contradictions = result.get("contradictions", [])
        evidence = [{
            "type": "contradictions",
            "total": len(contradictions),
            "high_severity": len([c for c in contradictions if c.get("severity", 0) > 0.7])
        }]

        confidence = result.get("coherence_metrics", {}).get("coherence_score", 0.5)

        return ModuleResult(
            module_name=self.module_name,
            class_name="PolicyContradictionDetector",
            method_name="detect",
            status="success",
            data=result,
            evidence=evidence,
            confidence=confidence,
            execution_time=0.0
        )

    def _execute_calculate_confidence(self, contradictions: List[Dict], **kwargs) -> ModuleResult:
        """Ejecuta BayesianConfidenceCalculator.calculate()"""
        calculator = self.BayesianConfidenceCalculator()
        confidence = calculator.calculate(contradictions)

        return ModuleResult(
            module_name=self.module_name,
            class_name="BayesianConfidenceCalculator",
            method_name="calculate_confidence",
            status="success",
            data={"confidence": confidence},
            evidence=[{"bayesian_confidence": confidence}],
            confidence=confidence,
            execution_time=0.0
        )

    def _execute_verify_temporal(self, statements: List, **kwargs) -> ModuleResult:
        """Ejecuta TemporalLogicVerifier.verify()"""
        verifier = self.TemporalLogicVerifier()
        result = verifier.verify_temporal_consistency(statements)

        return ModuleResult(
            module_name=self.module_name,
            class_name="TemporalLogicVerifier",
            method_name="verify_temporal_logic",
            status="success",
            data=result,
            evidence=[{"temporal_violations": len(result[1])}],
            confidence=0.8,
            execution_time=0.0
        )

    def _create_unavailable_result(self, method_name: str, start_time: float) -> ModuleResult:
        return ModuleResult(
            module_name=self.module_name,
            class_name="PolicyContradictionDetector",
            method_name=method_name,
            status="failed",
            data={},
            evidence=[],
            confidence=0.0,
            execution_time=time.time() - start_time,
            errors=["Module not available"]
        )


# ============================================================================
# ADAPTER 5: EMBEDDING POLICY
# ============================================================================

class EmbeddingPolicyAdapter:
    """
    Adapter para emebedding_policy.py
    """

    def __init__(self):
        self.module_name = "embedding_policy"
        self.available = False
        self._load_module()

    def _load_module(self):
        try:
            from emebedding_policy import (
                AdvancedSemanticChunker,
                BayesianNumericalAnalyzer,
                PolicyCrossEncoderReranker,
                PolicyAnalysisEmbedder,
                ChunkingConfig,
                PolicyEmbeddingConfig,
                PolicyDomain,
                AnalyticalDimension,
                PDQIdentifier,
                SemanticChunk
            )
            self.AdvancedSemanticChunker = AdvancedSemanticChunker
            self.BayesianNumericalAnalyzer = BayesianNumericalAnalyzer
            self.PolicyCrossEncoderReranker = PolicyCrossEncoderReranker
            self.PolicyAnalysisEmbedder = PolicyAnalysisEmbedder
            self.ChunkingConfig = ChunkingConfig
            self.PolicyEmbeddingConfig = PolicyEmbeddingConfig
            self.PolicyDomain = PolicyDomain
            self.AnalyticalDimension = AnalyticalDimension
            self.PDQIdentifier = PDQIdentifier
            self.SemanticChunk = SemanticChunk
            self.available = True
            logger.info(f"✓ {self.module_name} loaded with 4 classes + supporting types")
        except ImportError as e:
            logger.warning(f"✗ {self.module_name} NOT available: {e}")

    def execute(self, method_name: str, args: List[Any], kwargs: Dict[str, Any]) -> ModuleResult:
        """
        MÉTODOS SOPORTADOS:
        - chunk_document(text: str, document_metadata: dict) -> List[SemanticChunk]
        - evaluate_policy_metric(observed_values: List[float]) -> BayesianEvaluation
        - rerank(query: str, candidates: List[SemanticChunk]) -> List[Tuple[SemanticChunk, float]]
        - process_document(text: str, document_metadata: dict) -> List[SemanticChunk]
        - semantic_search(query: str, document_chunks: List[SemanticChunk]) -> List[Tuple[SemanticChunk, float]]
        - evaluate_policy_numerical_consistency(chunks: List[SemanticChunk], pdq_context: PDQIdentifier) -> BayesianEvaluation
        """
        start_time = time.time()

        if not self.available:
            return self._create_unavailable_result(method_name, start_time)

        try:
            if method_name == "chunk_document":
                result = self._execute_chunk_document(*args, **kwargs)
            elif method_name == "evaluate_policy_metric":
                result = self._execute_evaluate_policy_metric(*args, **kwargs)
            elif method_name == "rerank":
                result = self._execute_rerank(*args, **kwargs)
            elif method_name == "process_document":
                result = self._execute_process_document(*args, **kwargs)
            elif method_name == "semantic_search":
                result = self._execute_semantic_search(*args, **kwargs)
            elif method_name == "evaluate_policy_numerical_consistency":
                result = self._execute_evaluate_policy_numerical_consistency(*args, **kwargs)
            else:
                raise ValueError(f"Unknown method: {method_name}")

            result.execution_time = time.time() - start_time
            return result

        except Exception as e:
            logger.error(f"{self.module_name}.{method_name} failed: {e}", exc_info=True)
            return ModuleResult(
                module_name=self.module_name,
                class_name="PolicyAnalysisEmbedder",
                method_name=method_name,
                status="failed",
                data={},
                evidence=[],
                confidence=0.0,
                execution_time=time.time() - start_time,
                errors=[str(e)]
            )

    def _execute_chunk_document(self, text: str, document_metadata: Dict[str, Any], **kwargs) -> ModuleResult:
        """Ejecuta AdvancedSemanticChunker.chunk_document()"""
        config = kwargs.get('config', self.ChunkingConfig())
        chunker = self.AdvancedSemanticChunker(config)
        chunks = chunker.chunk_document(text, document_metadata)

        evidence = [{
            "type": "semantic_chunks",
            "chunk_count": len(chunks),
            "avg_token_count": sum(c['token_count'] for c in chunks) / max(1, len(chunks)),
            "pdq_contexts": len([c for c in chunks if c['pdq_context']])
        }]

        return ModuleResult(
            module_name=self.module_name,
            class_name="AdvancedSemanticChunker",
            method_name="chunk_document",
            status="success",
            data={"chunks": chunks},
            evidence=evidence,
            confidence=0.9,
            execution_time=0.0
        )

    def _execute_evaluate_policy_metric(self, observed_values: List[float], **kwargs) -> ModuleResult:
        """Ejecuta BayesianNumericalAnalyzer.evaluate_policy_metric()"""
        prior_strength = kwargs.get('prior_strength', 1.0)
        analyzer = self.BayesianNumericalAnalyzer(prior_strength=prior_strength)
        result = analyzer.evaluate_policy_metric(observed_values)

        evidence = [{
            "type": "bayesian_evaluation",
            "point_estimate": result['point_estimate'],
            "credible_interval": result['credible_interval_95'],
            "evidence_strength": result['evidence_strength'],
            "numerical_coherence": result['numerical_coherence']
        }]

        return ModuleResult(
            module_name=self.module_name,
            class_name="BayesianNumericalAnalyzer",
            method_name="evaluate_policy_metric",
            status="success",
            data=result,
            evidence=evidence,
            confidence=result['point_estimate'],
            execution_time=0.0
        )

    def _execute_rerank(self, query: str, candidates: List[SemanticChunk], **kwargs) -> ModuleResult:
        """Ejecuta PolicyCrossEncoderReranker.rerank()"""
        model_name = kwargs.get('model_name', "cross-encoder/ms-marco-MiniLM-L-6-v2")
        max_length = kwargs.get('max_length', 512)
        top_k = kwargs.get('top_k', 10)
        min_score = kwargs.get('min_score', 0.0)

        reranker = self.PolicyCrossEncoderReranker(model_name=model_name, max_length=max_length)
        reranked = reranker.rerank(query, candidates, top_k=top_k, min_score=min_score)

        evidence = [{
            "type": "reranking",
            "candidates_count": len(candidates),
            "reranked_count": len(reranked)
        }]

        return ModuleResult(
            module_name=self.module_name,
            class_name="PolicyCrossEncoderReranker",
            method_name="rerank",
            status="success",
            data={"reranked": reranked},
            evidence=evidence,
            confidence=0.85,
            execution_time=0.0
        )

    def _execute_process_document(self, text: str, document_metadata: Dict[str, Any], **kwargs) -> ModuleResult:
        """Ejecuta PolicyAnalysisEmbedder.process_document()"""
        config = kwargs.get('config', self.PolicyEmbeddingConfig())
        embedder = self.PolicyAnalysisEmbedder(config=config)
        chunks = embedder.process_document(text, document_metadata)

        evidence = [{
            "type": "processed_chunks",
            "chunk_count": len(chunks),
            "avg_embedding_dim": np.mean([c['embedding'].shape[0] for c in chunks if c['embedding'] is not None])
        }]

        return ModuleResult(
            module_name=self.module_name,
            class_name="PolicyAnalysisEmbedder",
            method_name="process_document",
            status="success",
            data={"chunks": chunks},
            evidence=evidence,
            confidence=0.9,
            execution_time=0.0
        )

    def _execute_semantic_search(self, query: str, document_chunks: List[SemanticChunk], **kwargs) -> ModuleResult:
        """Ejecuta PolicyAnalysisEmbedder.semantic_search()"""
        top_k_candidates = kwargs.get('top_k_candidates', 50)
        top_k_rerank = kwargs.get('top_k_rerank', 10)
        pdq_filter = kwargs.get('pdq_filter', None)
        use_reranking = kwargs.get('use_reranking', True)

        embedder = self.PolicyAnalysisEmbedder()
        results = embedder.semantic_search(
            query, document_chunks,
            top_k_candidates=top_k_candidates,
            top_k_rerank=top_k_rerank,
            pdq_filter=pdq_filter,
            use_reranking=use_reranking
        )

        evidence = [{
            "type": "semantic_search",
            "results_count": len(results),
            "top_score": results[0][1] if results else 0.0
        }]

        return ModuleResult(
            module_name=self.module_name,
            class_name="PolicyAnalysisEmbedder",
            method_name="semantic_search",
            status="success",
            data={"results": results},
            evidence=evidence,
            confidence=results[0][1] if results else 0.0,
            execution_time=0.0
        )

    def _execute_evaluate_policy_numerical_consistency(self, chunks: List[SemanticChunk], pdq_context: PDQIdentifier,
                                                       **kwargs) -> ModuleResult:
        """Ejecuta PolicyAnalysisEmbedder.evaluate_policy_numerical_consistency()"""
        embedder = self.PolicyAnalysisEmbedder()
        result = embedder.evaluate_policy_numerical_consistency(chunks, pdq_context)

        evidence = [{
            "type": "numerical_consistency",
            "point_estimate": result['point_estimate'],
            "credible_interval": result['credible_interval_95'],
            "evidence_strength": result['evidence_strength']
        }]

        return ModuleResult(
            module_name=self.module_name,
            class_name="PolicyAnalysisEmbedder",
            method_name="evaluate_policy_numerical_consistency",
            status="success",
            data=result,
            evidence=evidence,
            confidence=result['point_estimate'],
            execution_time=0.0
        )

    def _create_unavailable_result(self, method_name: str, start_time: float) -> ModuleResult:
        return ModuleResult(
            module_name=self.module_name,
            class_name="PolicyAnalysisEmbedder",
            method_name=method_name,
            status="failed",
            data={},
            evidence=[],
            confidence=0.0,
            execution_time=time.time() - start_time,
            errors=["Module not available"]
        )


# ============================================================================
# ADAPTER 6: FINANCIAL ANALYZER
# ============================================================================

class FinancialAnalyzerAdapter:
    """
    Adapter para financiero_viabilidad_tablas.py
    """

    def __init__(self):
        self.module_name = "financial_analyzer"
        self.available = False
        self._load_module()

    def _load_module(self):
        try:
            from financiero_viabilidad_tablas import (
                PDETMunicipalPlanAnalyzer,
                ColombianMunicipalContext
            )
            self.PDETMunicipalPlanAnalyzer = PDETMunicipalPlanAnalyzer
            self.ColombianMunicipalContext = ColombianMunicipalContext
            self.available = True
            logger.info(f"✓ {self.module_name} loaded with 11+ classes")
        except ImportError as e:
            logger.warning(f"✗ {self.module_name} NOT available: {e}")

    def execute(self, method_name: str, args: List[Any], kwargs: Dict[str, Any]) -> ModuleResult:
        """
        MÉTODOS SOPORTADOS:
        - extract_tables(pdf_path: str) -> List[ExtractedTable]
        - analyze_financial_feasibility(tables: List[ExtractedTable], text: str) -> Dict[str, Any]
        - identify_responsible_entities(text: str, tables: List[ExtractedTable]) -> List[ResponsibleEntity]
        - construct_causal_dag(text: str, tables: List[ExtractedTable], financial_analysis: Dict) -> CausalDAG
        """
        start_time = time.time()

        if not self.available:
            return self._create_unavailable_result(method_name, start_time)

        try:
            if method_name == "extract_tables":
                result = self._execute_extract_tables(*args, **kwargs)
            elif method_name == "analyze_financial_feasibility":
                result = self._execute_analyze_financial_feasibility(*args, **kwargs)
            elif method_name == "identify_responsible_entities":
                result = self._execute_identify_responsible_entities(*args, **kwargs)
            elif method_name == "construct_causal_dag":
                result = self._execute_construct_causal_dag(*args, **kwargs)
            else:
                raise ValueError(f"Unknown method: {method_name}")

            result.execution_time = time.time() - start_time
            return result

        except Exception as e:
            logger.error(f"{self.module_name}.{method_name} failed: {e}", exc_info=True)
            return ModuleResult(
                module_name=self.module_name,
                class_name="PDETMunicipalPlanAnalyzer",
                method_name=method_name,
                status="failed",
                data={},
                evidence=[],
                confidence=0.0,
                execution_time=time.time() - start_time,
                errors=[str(e)]
            )

    def _execute_extract_tables(self, pdf_path: str, **kwargs) -> ModuleResult:
        """Ejecuta PDETMunicipalPlanAnalyzer.extract_tables()"""
        use_gpu = kwargs.get('use_gpu', True)
        analyzer = self.PDETMunicipalPlanAnalyzer(use_gpu=use_gpu)

        # This is an async method, so we need to handle it differently
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                result = loop.run_until_complete(analyzer.extract_tables(pdf_path))
            else:
                result = asyncio.run(analyzer.extract_tables(pdf_path))
        except Exception as e:
            logger.error(f"Error in async table extraction: {e}")
            result = []

        evidence = [{
            "type": "extracted_tables",
            "table_count": len(result),
            "avg_confidence": np.mean([t.confidence_score for t in result]) if result else 0.0
        }]

        return ModuleResult(
            module_name=self.module_name,
            class_name="PDETMunicipalPlanAnalyzer",
            method_name="extract_tables",
            status="success",
            data={"tables": result},
            evidence=evidence,
            confidence=evidence[0]["avg_confidence"] if evidence else 0.0,
            execution_time=0.0
        )

    def _execute_analyze_financial_feasibility(self, tables: List[Any], text: str, **kwargs) -> ModuleResult:
        """Ejecuta PDETMunicipalPlanAnalyzer.analyze_financial_feasibility()"""
        use_gpu = kwargs.get('use_gpu', True)
        analyzer = self.PDETMunicipalPlanAnalyzer(use_gpu=use_gpu)
        result = analyzer.analyze_financial_feasibility(tables, text)

        evidence = [{
            "type": "financial_feasibility",
            "total_budget": result.get("total_budget", 0),
            "sustainability_score": result.get("sustainability_score", 0.0),
            "risk_assessment": result.get("risk_assessment", {})
        }]

        return ModuleResult(
            module_name=self.module_name,
            class_name="PDETMunicipalPlanAnalyzer",
            method_name="analyze_financial_feasibility",
            status="success",
            data=result,
            evidence=evidence,
            confidence=result.get("risk_assessment", {}).get("risk_score", 0.5),
            execution_time=0.0
        )

    def _execute_identify_responsible_entities(self, text: str, tables: List[Any], **kwargs) -> ModuleResult:
        """Ejecuta PDETMunicipalPlanAnalyzer.identify_responsible_entities()"""
        use_gpu = kwargs.get('use_gpu', True)
        analyzer = self.PDETMunicipalPlanAnalyzer(use_gpu=use_gpu)
        result = analyzer.identify_responsible_entities(text, tables)

        evidence = [{
            "type": "responsible_entities",
            "entity_count": len(result),
            "avg_specificity": np.mean([e.specificity_score for e in result]) if result else 0.0
        }]

        return ModuleResult(
            module_name=self.module_name,
            class_name="PDETMunicipalPlanAnalyzer",
            method_name="identify_responsible_entities",
            status="success",
            data={"entities": result},
            evidence=evidence,
            confidence=evidence[0]["avg_specificity"] if evidence else 0.0,
            execution_time=0.0
        )

    def _execute_construct_causal_dag(self, text: str, tables: List[Any], financial_analysis: Dict,
                                      **kwargs) -> ModuleResult:
        """Ejecuta PDETMunicipalPlanAnalyzer.construct_causal_dag()"""
        use_gpu = kwargs.get('use_gpu', True)
        analyzer = self.PDETMunicipalPlanAnalyzer(use_gpu=use_gpu)
        result = analyzer.construct_causal_dag(text, tables, financial_analysis)

        evidence = [{
            "type": "causal_dag",
            "node_count": result.graph.number_of_nodes(),
            "edge_count": result.graph.number_of_edges(),
            "acyclic": nx.is_directed_acyclic_graph(result.graph)
        }]

        return ModuleResult(
            module_name=self.module_name,
            class_name="PDETMunicipalPlanAnalyzer",
            method_name="construct_causal_dag",
            status="success",
            data=result,
            evidence=evidence,
            confidence=0.8,
            execution_time=0.0
        )

    def _create_unavailable_result(self, method_name: str, start_time: float) -> ModuleResult:
        return ModuleResult(
            module_name=self.module_name,
            class_name="PDETMunicipalPlanAnalyzer",
            method_name=method_name,
            status="failed",
            data={},
            evidence=[],
            confidence=0.0,
            execution_time=time.time() - start_time,
            errors=["Module not available"]
        )


# ============================================================================
# ADAPTER 7: CAUSAL PROCESSOR
# ============================================================================

class CausalProcessorAdapter:
    """
    Adapter para causal_proccesor.py
    """

    def __init__(self):
        self.module_name = "causal_processor"
        self.available = False
        self._load_module()

    def _load_module(self):
        try:
            from causal_proccesor import (
                SemanticProcessor,
                BayesianEvidenceIntegrator,
                PolicyDocumentAnalyzer,
                SemanticConfig
            )
            self.SemanticProcessor = SemanticProcessor
            self.BayesianEvidenceIntegrator = BayesianEvidenceIntegrator
            self.PolicyDocumentAnalyzer = PolicyDocumentAnalyzer
            self.SemanticConfig = SemanticConfig
            self.available = True
            logger.info(f"✓ {self.module_name} loaded with 4 classes")
        except ImportError as e:
            logger.warning(f"✗ {self.module_name} NOT available: {e}")

    def execute(self, method_name: str, args: List[Any], kwargs: Dict[str, Any]) -> ModuleResult:
        """
        MÉTODOS SOPORTADOS:
        - chunk_text(text: str, preserve_structure: bool) -> List[Dict]
        - analyze(text: str) -> Dict
        """
        start_time = time.time()

        if not self.available:
            return self._create_unavailable_result(method_name, start_time)

        try:
            if method_name == "chunk_text":
                result = self._execute_chunk_text(*args, **kwargs)
            elif method_name == "analyze":
                result = self._execute_analyze(*args, **kwargs)
            else:
                raise ValueError(f"Unknown method: {method_name}")

            result.execution_time = time.time() - start_time
            return result

        except Exception as e:
            logger.error(f"{self.module_name}.{method_name} failed: {e}", exc_info=True)
            return ModuleResult(
                module_name=self.module_name,
                class_name="PolicyDocumentAnalyzer",
                method_name=method_name,
                status="failed",
                data={},
                evidence=[],
                confidence=0.0,
                execution_time=time.time() - start_time,
                errors=[str(e)]
            )

    def _execute_chunk_text(self, text: str, **kwargs) -> ModuleResult:
        """Ejecuta SemanticProcessor.chunk_text()"""
        preserve_structure = kwargs.get('preserve_structure', True)
        config = kwargs.get('config', self.SemanticConfig())
        processor = self.SemanticProcessor(config)
        chunks = processor.chunk_text(text, preserve_structure=preserve_structure)

        evidence = [{
            "type": "semantic_chunks",
            "chunk_count": len(chunks),
            "avg_token_count": np.mean([c['token_count'] for c in chunks]) if chunks else 0
        }]

        return ModuleResult(
            module_name=self.module_name,
            class_name="SemanticProcessor",
            method_name="chunk_text",
            status="success",
            data={"chunks": chunks},
            evidence=evidence,
            confidence=0.9,
            execution_time=0.0
        )

    def _execute_analyze(self, text: str, **kwargs) -> ModuleResult:
        """Ejecuta PolicyDocumentAnalyzer.analyze()"""
        config = kwargs.get('config', self.SemanticConfig())
        analyzer = self.PolicyDocumentAnalyzer(config)
        result = analyzer.analyze(text)

        evidence = [{
            "type": "policy_analysis",
            "total_chunks": result["summary"]["total_chunks"],
            "sections_detected": result["summary"]["sections_detected"],
            "avg_confidence": result["document_statistics"]["avg_confidence"]
        }]

        return ModuleResult(
            module_name=self.module_name,
            class_name="PolicyDocumentAnalyzer",
            method_name="analyze",
            status="success",
            data=result,
            evidence=evidence,
            confidence=result["document_statistics"]["avg_confidence"],
            execution_time=0.0
        )

    def _create_unavailable_result(self, method_name: str, start_time: float) -> ModuleResult:
        return ModuleResult(
            module_name=self.module_name,
            class_name="PolicyDocumentAnalyzer",
            method_name=method_name,
            status="failed",
            data={},
            evidence=[],
            confidence=0.0,
            execution_time=time.time() - start_time,
            errors=["Module not available"]
        )


# ============================================================================
# ADAPTER 8: POLICY SEGMENTER
# ============================================================================

class PolicySegmenterAdapter:
    """
    Adapter para policy_segmenter.py
    """

    def __init__(self):
        self.module_name = "policy_segmenter"
        self.available = False
        self._load_module()

    def _load_module(self):
        try:
            from policy_segmenter import (
                DocumentSegmenter,
                SegmenterConfig
            )
            self.DocumentSegmenter = DocumentSegmenter
            self.SegmenterConfig = SegmenterConfig
            self.available = True
            logger.info(f"✓ {self.module_name} loaded with 2 classes")
        except ImportError as e:
            logger.warning(f"✗ {self.module_name} NOT available: {e}")

    def execute(self, method_name: str, args: List[Any], kwargs: Dict[str, Any]) -> ModuleResult:
        """
        MÉTODOS SOPORTADOS:
        - segment(text: str) -> List[Dict]
        - get_segmentation_report() -> Dict
        """
        start_time = time.time()

        if not self.available:
            return self._create_unavailable_result(method_name, start_time)

        try:
            if method_name == "segment":
                result = self._execute_segment(*args, **kwargs)
            elif method_name == "get_segmentation_report":
                result = self._execute_get_report(*args, **kwargs)
            else:
                raise ValueError(f"Unknown method: {method_name}")

            result.execution_time = time.time() - start_time
            return result

        except Exception as e:
            logger.error(f"{self.module_name}.{method_name} failed: {e}", exc_info=True)
            return ModuleResult(
                module_name=self.module_name,
                class_name="DocumentSegmenter",
                method_name=method_name,
                status="failed",
                data={},
                evidence=[],
                confidence=0.0,
                execution_time=time.time() - start_time,
                errors=[str(e)]
            )

    def _execute_segment(self, text: str, **kwargs) -> ModuleResult:
        """Ejecuta DocumentSegmenter.segment()"""
        config = kwargs.get('config', self.SegmenterConfig())
        segmenter = self.DocumentSegmenter(config)
        segments = segmenter.segment(text)

        evidence = [{
            "type": "document_segments",
            "segment_count": len(segments),
            "avg_char_length": np.mean([s['metrics'].char_count for s in segments]) if segments else 0
        }]

        return ModuleResult(
            module_name=self.module_name,
            class_name="DocumentSegmenter",
            method_name="segment",
            status="success",
            data={"segments": segments},
            evidence=evidence,
            confidence=0.9,
            execution_time=0.0
        )

    def _execute_get_report(self, **kwargs) -> ModuleResult:
        """Ejecuta DocumentSegmenter.get_segmentation_report()"""
        config = kwargs.get('config', self.SegmenterConfig())
        segmenter = self.DocumentSegmenter(config)
        report = segmenter.get_segmentation_report()

        evidence = [{
            "type": "segmentation_report",
            "total_segments": report["summary"]["total_segments"],
            "avg_char_length": report["summary"]["avg_char_length"],
            "overall_quality": report["quality_metrics"]["overall_quality"]
        }]

        return ModuleResult(
            module_name=self.module_name,
            class_name="DocumentSegmenter",
            method_name="get_segmentation_report",
            status="success",
            data=report,
            evidence=evidence,
            confidence=report["quality_metrics"]["overall_quality"],
            execution_time=0.0
        )

    def _create_unavailable_result(self, method_name: str, start_time: float) -> ModuleResult:
        return ModuleResult(
            module_name=self.module_name,
            class_name="DocumentSegmenter",
            method_name=method_name,
            status="failed",
            data={},
            evidence=[],
            confidence=0.0,
            execution_time=time.time() - start_time,
            errors=["Module not available"]
        )


# ============================================================================
# ADAPTER 9: TEORIA DE CAMBIO
# ============================================================================

class TeoriaCambioAdapter:
    """
    Adapter para teoria_cambio.py
    """

    def __init__(self):
        self.module_name = "teoria_cambio"
        self.available = False
        self._load_module()

    def _load_module(self):
        try:
            from teoria_cambio import (
                TeoriaCambio,
                AdvancedDAGValidator,
                IndustrialGradeValidator,
                MonteCarloAdvancedResult
            )
            self.TeoriaCambio = TeoriaCambio
            self.AdvancedDAGValidator = AdvancedDAGValidator
            self.IndustrialGradeValidator = IndustrialGradeValidator
            self.MonteCarloAdvancedResult = MonteCarloAdvancedResult
            self.available = True
            logger.info(f"✓ {self.module_name} loaded with 4 classes")
        except ImportError as e:
            logger.warning(f"✗ {self.module_name} NOT available: {e}")

    def execute(self, method_name: str, args: List[Any], kwargs: Dict[str, Any]) -> ModuleResult:
        """
        MÉTODOS SOPORTADOS:
        - construir_grafo_causal() -> nx.DiGraph
        - validacion_completa(grafo: nx.DiGraph) -> ValidacionResultado
        - calculate_acyclicity_pvalue(plan_name: str, iterations: int) -> MonteCarloAdvancedResult
        """
        start_time = time.time()

        if not self.available:
            return self._create_unavailable_result(method_name, start_time)

        try:
            if method_name == "construir_grafo_causal":
                result = self._execute_construir_grafo_causal(*args, **kwargs)
            elif method_name == "validacion_completa":
                result = self._execute_validacion_completa(*args, **kwargs)
            elif method_name == "calculate_acyclicity_pvalue":
                result = self._execute_calculate_acyclicity_pvalue(*args, **kwargs)
            else:
                raise ValueError(f"Unknown method: {method_name}")

            result.execution_time = time.time() - start_time
            return result

        except Exception as e:
            logger.error(f"{self.module_name}.{method_name} failed: {e}", exc_info=True)
            return ModuleResult(
                module_name=self.module_name,
                class_name="TeoriaCambio",
                method_name=method_name,
                status="failed",
                data={},
                evidence=[],
                confidence=0.0,
                execution_time=time.time() - start_time,
                errors=[str(e)]
            )

    def _execute_construir_grafo_causal(self, **kwargs) -> ModuleResult:
        """Ejecuta TeoriaCambio.construir_grafo_causal()"""
        tc = self.TeoriaCambio()
        graph = tc.construir_grafo_causal()

        evidence = [{
            "type": "causal_graph",
            "node_count": graph.number_of_nodes(),
            "edge_count": graph.number_of_edges()
        }]

        return ModuleResult(
            module_name=self.module_name,
            class_name="TeoriaCambio",
            method_name="construir_grafo_causal",
            status="success",
            data={"graph": graph},
            evidence=evidence,
            confidence=0.8,
            execution_time=0.0
        )

    def _execute_validacion_completa(self, grafo, **kwargs) -> ModuleResult:
        """Ejecuta TeoriaCambio.validacion_completa()"""
        tc = self.TeoriaCambio()
        result = tc.validacion_completa(grafo)

        evidence = [{
            "type": "validation_result",
            "is_valid": result.es_valida,
            "violations_count": len(result.violaciones_orden),
            "complete_paths_count": len(result.caminos_completos),
            "suggestions_count": len(result.sugerencias)
        }]

        return ModuleResult(
            module_name=self.module_name,
            class_name="TeoriaCambio",
            method_name="validacion_completa",
            status="success",
            data={"result": result},
            evidence=evidence,
            confidence=1.0 if result.es_valida else 0.3,
            execution_time=0.0
        )

    def _execute_calculate_acyclicity_pvalue(self, plan_name: str, iterations: int, **kwargs) -> ModuleResult:
        """Ejecuta AdvancedDAGValidator.calculate_acyclicity_pvalue()"""
        validator = self.AdvancedDAGValidator()
        result = validator.calculate_acyclicity_pvalue(plan_name, iterations)

        evidence = [{
            "type": "acyclicity_analysis",
            "p_value": result.p_value,
            "robustness_score": result.robustness_score,
            "statistical_power": result.statistical_power
        }]

        return ModuleResult(
            module_name=self.module_name,
            class_name="AdvancedDAGValidator",
            method_name="calculate_acyclicity_pvalue",
            status="success",
            data={"result": result},
            evidence=evidence,
            confidence=1.0 - result.p_value,
            execution_time=0.0
        )

    def _create_unavailable_result(self, method_name: str, start_time: float) -> ModuleResult:
        return ModuleResult(
            module_name=self.module_name,
            class_name="TeoriaCambio",
            method_name=method_name,
            status="failed",
            data={},
            evidence=[],
            confidence=0.0,
            execution_time=time.time() - start_time,
            errors=["Module not available"]
        )


# ============================================================================
# ADAPTER 10: SEMANTIC CHUNKING POLICY
# ============================================================================

class SemanticChunkingPolicyAdapter:
    """
    Adapter para semantic_chunking_policy.py
    """

    def __init__(self):
        self.module_name = "semantic_chunking_policy"
        self.available = False
        self._load_module()

    def _load_module(self):
        try:
            from semantic_chunking_policy import (
                SemanticProcessor,
                BayesianEvidenceIntegrator,
                PolicyDocumentAnalyzer,
                SemanticConfig
            )
            self.SemanticProcessor = SemanticProcessor
            self.BayesianEvidenceIntegrator = BayesianEvidenceIntegrator
            self.PolicyDocumentAnalyzer = PolicyDocumentAnalyzer
            self.SemanticConfig = SemanticConfig
            self.available = True
            logger.info(f"✓ {self.module_name} loaded with 4 classes")
        except ImportError as e:
            logger.warning(f"✗ {self.module_name} NOT available: {e}")

    def execute(self, method_name: str, args: List[Any], kwargs: Dict[str, Any]) -> ModuleResult:
        """
        MÉTODOS SOPORTADOS:
        - chunk_text(text: str, preserve_structure: bool) -> List[Dict]
        - analyze(text: str) -> Dict
        """
        start_time = time.time()

        if not self.available:
            return self._create_unavailable_result(method_name, start_time)

        try:
            if method_name == "chunk_text":
                result = self._execute_chunk_text(*args, **kwargs)
            elif method_name == "analyze":
                result = self._execute_analyze(*args, **kwargs)
            else:
                raise ValueError(f"Unknown method: {method_name}")

            result.execution_time = time.time() - start_time
            return result

        except Exception as e:
            logger.error(f"{self.module_name}.{method_name} failed: {e}", exc_info=True)
            return ModuleResult(
                module_name=self.module_name,
                class_name="PolicyDocumentAnalyzer",
                method_name=method_name,
                status="failed",
                data={},
                evidence=[],
                confidence=0.0,
                execution_time=time.time() - start_time,
                errors=[str(e)]
            )

    def _execute_chunk_text(self, text: str, **kwargs) -> ModuleResult:
        """Ejecuta SemanticProcessor.chunk_text()"""
        preserve_structure = kwargs.get('preserve_structure', True)
        config = kwargs.get('config', self.SemanticConfig())
        processor = self.SemanticProcessor(config)
        chunks = processor.chunk_text(text, preserve_structure=preserve_structure)

        evidence = [{
            "type": "semantic_chunks",
            "chunk_count": len(chunks),
            "avg_token_count": np.mean([c['token_count'] for c in chunks]) if chunks else 0
        }]

        return ModuleResult(
            module_name=self.module_name,
            class_name="SemanticProcessor",
            method_name="chunk_text",
            status="success",
            data={"chunks": chunks},
            evidence=evidence,
            confidence=0.9,
            execution_time=0.0
        )

    def _execute_analyze(self, text: str, **kwargs) -> ModuleResult:
        """Ejecuta PolicyDocumentAnalyzer.analyze()"""
        config = kwargs.get('config', self.SemanticConfig())
        analyzer = self.PolicyDocumentAnalyzer(config)
        result = analyzer.analyze(text)

        evidence = [{
            "type": "policy_analysis",
            "total_chunks": result["summary"]["total_chunks"],
            "sections_detected": result["summary"]["sections_detected"],
            "avg_confidence": np.mean([
                d.get("confidence", 0.5)
                for d in result["causal_dimensions"].values()
            ]) if result["causal_dimensions"] else 0.5
        }]

        return ModuleResult(
            module_name=self.module_name,
            class_name="PolicyDocumentAnalyzer",
            method_name="analyze",
            status="success",
            data=result,
            evidence=evidence,
            confidence=evidence[0]["avg_confidence"] if evidence else 0.5,
            execution_time=0.0
        )

    def _create_unavailable_result(self, method_name: str, start_time: float) -> ModuleResult:
        return ModuleResult(
            module_name=self.module_name,
            class_name="PolicyDocumentAnalyzer",
            method_name=method_name,
            status="failed",
            data={},
            evidence=[],
            confidence=0.0,
            execution_time=time.time() - start_time,
            errors=["Module not available"]
        )


# ============================================================================
# MAIN ADAPTER REGISTRY
# ============================================================================

class AdapterRegistry:
    """
    Registry for all adapters to provide a unified interface
    """

    def __init__(self):
        self.adapters = {
            "policy_processor": PolicyProcessorAdapter(),
            "analyzer_one": AnalyzerOneAdapter(),
            "derek_beach": DerekBeachAdapter(),
            "contradiction_detector": ContradictionDetectorAdapter(),
            "embedding_policy": EmbeddingPolicyAdapter(),
            "financial_analyzer": FinancialAnalyzerAdapter(),
            "causal_processor": CausalProcessorAdapter(),
            "policy_segmenter": PolicySegmenterAdapter(),
            "teoria_cambio": TeoriaCambioAdapter(),
            "semantic_chunking_policy": SemanticChunkingPolicyAdapter()
        }

        self.available_modules = {
            name: adapter.available
            for name, adapter in self.adapters.items()
        }

        logger.info(f"AdapterRegistry initialized with {len(self.available_modules)} available modules")

    def get_adapter(self, module_name: str):
        """Get a specific adapter by name"""
        if module_name not in self.adapters:
            raise ValueError(f"Unknown module: {module_name}")

        if not self.adapters[module_name].available:
            logger.warning(f"Module {module_name} is not available")

        return self.adapters[module_name]

    def execute(self, module_name: str, method_name: str, args: List[Any], kwargs: Dict[str, Any]) -> ModuleResult:
        """Execute a method on a specific module"""
        adapter = self.get_adapter(module_name)
        return adapter.execute(method_name, args, kwargs)


# ============================================================================
# POLICY ANALYSIS PIPELINE
# ============================================================================

class PolicyAnalysisPipeline:
    """
    Main pipeline for analyzing policy documents using all available modules
    """

    def __init__(self):
        self.registry = AdapterRegistry()
        self.logger = logging.getLogger(__name__)

    def analyze_policy_document(self, document_text: str, document_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a policy document using all available modules

        Returns a comprehensive analysis report
        """
        self.logger.info("Starting comprehensive policy document analysis")

        results = {
            "document_metadata": document_metadata,
            "analysis_results": {}
        }

        # Process document with embedding_policy
        if "embedding_policy" in self.registry.available_modules:
            self.logger.info("Processing with embedding_policy module")
            adapter = self.registry.get_adapter("embedding_policy")

            # Chunk the document
            chunk_result = adapter.execute(
                "process_document",
                [document_text, document_metadata]
            )
            results["analysis_results"]["embedding_policy"] = {
                "status": chunk_result.status,
                "data": chunk_result.data,
                "evidence": chunk_result.evidence,
                "confidence": chunk_result.confidence,
                "execution_time": chunk_result.execution_time
            }

            # Extract semantic chunks for further analysis
            chunks = chunk_result.data.get("chunks", [])

            # Analyze numerical consistency for P1-D1-Q1 (baseline data)
            pdq_context = {
                "question_unique_id": "P1-D1-Q1",
                "policy": "P1",
                "dimension": "D1",
                "question": 1,
                "rubric_key": "D1-Q1"
            }

            consistency_result = adapter.execute(
                "evaluate_policy_numerical_consistency",
                [chunks, pdq_context]
            )
            results["analysis_results"]["numerical_consistency"] = {
                "status": consistency_result.status,
                "data": consistency_result.data,
                "evidence": consistency_result.evidence,
                "confidence": consistency_result.confidence,
                "execution_time": consistency_result.execution_time
            }

        # Process with policy_processor for causal analysis
        if "policy_processor" in self.registry.available_modules:
            self.logger.info("Processing with policy_processor module")
            adapter = self.registry.get_adapter("policy_processor")

            process_result = adapter.execute(
                "process",
                [document_text]
            )
            results["analysis_results"]["policy_processor"] = {
                "status": process_result.status,
                "data": process_result.data,
                "evidence": process_result.evidence,
                "confidence": process_result.confidence,
                "execution_time": process_result.execution_time
            }

        # Process with analyzer_one for value chain analysis
        if "analyzer_one" in self.registry.available_modules:
            self.logger.info("Processing with analyzer_one module")
            adapter = self.registry.get_adapter("analyzer_one")

            analyze_result = adapter.execute(
                "analyze_document",
                [document_text]
            )
            results["analysis_results"]["analyzer_one"] = {
                "status": analyze_result.status,
                "data": analyze_result.data,
                "evidence": analyze_result.evidence,
                "confidence": analyze_result.confidence,
                "execution_time": analyze_result.execution_time
            }

        # Process with derek_beach for causal framework analysis
        if "derek_beach" in self.registry.available_modules:
            self.logger.info("Processing with derek_beach module")
            adapter = self.registry.get_adapter("derek_beach")

            process_result = adapter.execute(
                "process_document",
                [document_text, "PDM Analysis"]
            )
            results["analysis_results"]["derek_beach"] = {
                "status": process_result.status,
                "data": process_result.data,
                "evidence": process_result.evidence,
                "confidence": process_result.confidence,
                "execution_time": process_result.execution_time
            }

        # Process with contradiction_detector for consistency checks
        if "contradiction_detector" in self.registry.available_modules:
            self.logger.info("Processing with contradiction_detector module")
            adapter = self.registry.get_adapter("contradiction_detector")

            detect_result = adapter.execute(
                "detect",
                [document_text, "PDM Analysis", "PolicyDimension.ESTRATEGICO"]
            )
            results["analysis_results"]["contradiction_detector"] = {
                "status": detect_result.status,
                "data": detect_result.data,
                "evidence": detect_result.evidence,
                "confidence": detect_result.confidence,
                "execution_time": detect_result.execution_time
            }

        # Process with financial_analyzer for budget analysis
        if "financial_analyzer" in self.registry.available_modules:
            self.logger.info("Processing with financial_analyzer module")
            adapter = self.registry.get_adapter("financial_analyzer")

            # Create dummy tables for this example
            tables = []

            analyze_result = adapter.execute(
                "analyze_financial_feasibility",
                [tables, document_text]
            )
            results["analysis_results"]["financial_analyzer"] = {
                "status": analyze_result.status,
                "data": analyze_result.data,
                "evidence": analyze_result.evidence,
                "confidence": analyze_result.confidence,
                "execution_time": analyze_result.execution_time
            }

        # Process with causal_processor for causal analysis
        if "causal_processor" in self.registry.available_modules:
            self.logger.info("Processing with causal_processor module")
            adapter = self.registry.get_adapter("causal_processor")

            analyze_result = adapter.execute(
                "analyze",
                [document_text]
            )
            results["analysis_results"]["causal_processor"] = {
                "status": analyze_result.status,
                "data": analyze_result.data,
                "evidence": analyze_result.evidence,
                "confidence": analyze_result.confidence,
                "execution_time": analyze_result.execution_time
            }

        # Process with policy_segmenter for document segmentation
        if "policy_segmenter" in self.registry.available_modules:
            self.logger.info("Processing with policy_segmenter module")
            adapter = self.registry.get_adapter("policy_segmenter")

            segment_result = adapter.execute(
                "segment",
                [document_text]
            )
            results["analysis_results"]["policy_segmenter"] = {
                "status": segment_result.status,
                "data": segment_result.data,
                "evidence": segment_result.evidence,
                "confidence": segment_result.confidence,
                "execution_time": segment_result.execution_time
            }

        # Process with teoria_cambio for theory of change analysis
        if "teoria_cambio" in self.registry.available_modules:
            self.logger.info("Processing with teoria_cambio module")
            adapter = self.registry.get_adapter("teoria_cambio")

            # Create a simple graph for this example
            import networkx as nx
            graph = nx.DiGraph()
            graph.add_node("P1")
            graph.add_node("D1")
            graph.add_node("Q1")
            graph.add_edge("P1", "D1")
            graph.add_edge("D1", "Q1")

            validation_result = adapter.execute(
                "validacion_completa",
                [graph]
            )
            results["analysis_results"]["teoria_cambio"] = {
                "status": validation_result.status,
                "data": validation_result.data,
                "evidence": validation_result.evidence,
                "confidence": validation_result.confidence,
                "execution_time": validation_result.execution_time
            }

        # Process with semantic_chunking_policy for semantic analysis
        if "semantic_chunking_policy" in self.registry.available_modules:
            self.logger.info("Processing with semantic_chunking_policy module")
            adapter = self.registry.get_adapter("semantic_chunking_policy")

            analyze_result = adapter.execute(
                "analyze",
                [document_text]
            )
            results["analysis_results"]["semantic_chunking_policy"] = {
                "status": analyze_result.status,
                "data": analyze_result.data,
                "evidence": analyze_result.evidence,
                "confidence": analyze_result.confidence,
                "execution_time": analyze_result.execution_time
            }

        # Calculate overall confidence
        confidences = [
            result["confidence"]
            for result in results["analysis_results"].values()
            if result["confidence"] > 0
        ]
        overall_confidence = np.mean(confidences) if confidences else 0.0

        results["overall_confidence"] = overall_confidence
        results["analysis_timestamp"] = time.time()

        self.logger.info(f"Analysis completed with overall confidence: {overall_confidence:.2f}")

        return results


# ============================================================================
# POLICY QUESTION ANSWER
# ============================================================================

class PolicyQuestionAnswerer:
    """
    Answer specific policy questions based on analysis results
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def answer_question(self, question_id: str, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Answer a specific policy question based on analysis results

        Args:
            question_id: The question identifier (e.g., "P1-D1-Q1")
            analysis_results: Results from PolicyAnalysisPipeline

        Returns:
            Answer to the question with evidence and confidence
        """
        self.logger.info(f"Answering question: {question_id}")

        # Parse question components
        policy, dimension, question_num = self._parse_question_id(question_id)

        # Initialize answer
        answer = {
            "question_id": question_id,
            "policy": policy,
            "dimension": dimension,
            "question_num": question_num,
            "answer": "",
            "evidence": [],
            "confidence": 0.0,
            "status": "not_answered"
        }

        # Extract relevant analysis results
        embedding_results = analysis_results.get("analysis_results", {}).get("embedding_policy", {})
        policy_processor_results = analysis_results.get("analysis_results", {}).get("policy_processor", {})

        # Answer based on question type
        if question_id == "P1-D1-Q1":
            answer = self._answer_p1_d1_q1(embedding_results, policy_processor_results)
        elif question_id == "P1-D1-Q2":
            answer = self._answer_p1_d1_q2(embedding_results, policy_processor_results)
        elif question_id == "P1-D1-Q3":
            answer = self._answer_p1_d1_q3(embedding_results, policy_processor_results)
        elif question_id == "P1-D1-Q4":
            answer = self._answer_p1_d1_q4(embedding_results, policy_processor_results)
        elif question_id == "P1-D1-Q5":
            answer = self._answer_p1_d1_q5(embedding_results, policy_processor_results)
        elif question_id == "P1-D2-Q6":
            answer = self._answer_p1_d2_q6(embedding_results, policy_processor_results)
        elif question_id == "P1-D2-Q7":
            answer = self._answer_p1_d2_q7(embedding_results, policy_processor_results)
        elif question_id == "P1-D2-Q8":
            answer = self._answer_p1_d2_q8(embedding_results, policy_processor_results)
        elif question_id == "P1-D2-Q9":
            answer = self._answer_p1_d2_q9(embedding_results, policy_processor_results)
        elif question_id == "P1-D2-Q10":
            answer = self._answer_p1_d2_q10(embedding_results, policy_processor_results)
        # Add more question handlers as needed...
        else:
            answer["status"] = "question_not_supported"
            answer["answer"] = f"Question {question_id} is not yet supported"
            answer["confidence"] = 0.0

        return answer

    def _parse_question_id(self, question_id: str) -> Tuple[str, str, int]:
        """Parse question ID into components"""
        parts = question_id.split("-")
        if len(parts) != 3:
            return "Unknown", "Unknown", 0

        policy = parts[0]
        dimension = parts[1]
        question_num = int(parts[2].replace("Q", ""))

        return policy, dimension, question_num

    def _answer_p1_d1_q1(self, embedding_results: Dict, policy_processor_results: Dict) -> Dict[str, Any]:
        """Answer P1-D1-Q1: Does the diagnosis present numerical data for gender equality as baseline?"""
        answer = {
            "question_id": "P1-D1-Q1",
            "policy": "P1",
            "dimension": "D1",
            "question_num": 1,
            "answer": "",
            "evidence": [],
            "confidence": 0.0,
            "status": "not_answered"
        }

        # Check numerical consistency results
        numerical_results = embedding_results.get("numerical_consistency", {})
        if numerical_results.get("status") == "success":
            point_estimate = numerical_results.get("data", {}).get("point_estimate", 0.0)
            evidence_strength = numerical_results.get("data", {}).get("evidence_strength", 0.0)

            if point_estimate > 0.5 and evidence_strength > 0.5:
                answer[
                    "answer"] = "Sí, el diagnóstico presenta datos numéricos para la igualdad de género que sirven como línea base."
                answer["confidence"] = (point_estimate + evidence_strength) / 2
                answer["status"] = "answered"

                answer["evidence"] = numerical_results.get("evidence", [])
            else:
                answer[
                    "answer"] = "No se encontraron suficientes datos numéricos para la igualdad de género como línea base."
                answer["confidence"] = point_estimate
                answer["status"] = "answered_with_low_confidence"

        # Check policy processor results
        policy_data = policy_processor_results.get("data", {})
        dimensions = policy_data.get("dimensions", {})
        d1_data = dimensions.get("d1_insumos", {})

        if d1_data.get("point_evidence"):
            # Extract evidence about numerical data
            numerical_evidence = [
                evidence for evidence in d1_data.get("point_evidence", [])
                if any(keyword in evidence.lower() for keyword in
                       ["tasa", "porcentaje", "brecha", "cifra", "dato", "estadística"])
            ]

            if numerical_evidence:
                if answer["status"] == "not_answered":
                    answer["answer"] = "Se encontraron menciones de datos numéricos en el diagnóstico."
                    answer["confidence"] = min(0.7, len(numerical_evidence) / 5.0)
                    answer["status"] = "answered_with_moderate_confidence"
                    answer["evidence"] = numerical_evidence[:3]

        return answer

    def _answer_p1_d1_q2(self, embedding_results: Dict, policy_processor_results: Dict) -> Dict[str, Any]:
        """Answer P1-D1-Q2: Does the text quantify the gender inequality gap?"""
        answer = {
            "question_id": "P1-D1-Q1",
            "policy": "P1",
            "dimension": "D1",
            "question_num": 2,
            "answer": "",
            "evidence": [],
            "confidence": 0.0,
            "status": "not_answered"
        }

        # Check policy processor results
        policy_data = policy_processor_results.get("data", {})
        dimensions = policy_data.get("dimensions", {})
        d1_data = dimensions.get("d1_insumos", {})

        # Look for evidence of quantification
        quantification_keywords = [
            "brecha", "déficit", "porcentaje", "tasa", "proporción", "magnitud"
        ]

        evidence_items = []
        for evidence in d1_data.get("point_evidence", []):
            if any(keyword in evidence.lower() for keyword in quantification_keywords):
                evidence_items.append(evidence)

        if evidence_items:
            answer[
                "answer"] = "Sí, el texto dimensiona el problema de la desigualdad de género cuantificando la brecha o déficit."
            answer["confidence"] = min(0.8, len(evidence_items) / 3.0)
            answer["status"] = "answered"
            answer["evidence"] = evidence_items[:3]

        # Check for limitations in data
        limitation_keywords = [
            "subregistro", "información insuficiente", "vacíos de información"
        ]

        limitation_items = [
            evidence for evidence in d1_data.get("point_evidence", [])
            if any(keyword in evidence.lower() for keyword in limitation_keywords)
        ]

        if limitation_items:
            if answer["status"] == "not_answered":
                answer["answer"] = "Se identifican limitaciones en los datos sobre la situación de las mujeres."
                answer["confidence"] = min(0.7, len(limitation_items) / 3.0)
                answer["status"] = "answered_with_moderate_confidence"
                answer["evidence"] = limitation_items[:3]
            else:
                answer["answer"] = "El texto tanto cuantifica la brecha como reconoce limitaciones en los datos."
                answer["confidence"] = (answer["confidence"] + 0.7) / 2
                answer["status"] = "answered"
                answer["evidence"] = limitation_items[:2]

        return answer

    def _answer_p1_d1_q3(self, embedding_results: Dict, policy_processor_results: Dict) -> Dict[str, Any]:
        """Answer P1-D1-Q3: Are financial resources explicitly allocated to gender equality programs?"""
        answer = {
            "question_id": "P1-D1-Q3",
            "policy": "P1",
            "dimension": "D1",
            "question_num": 3,
            "answer": "",
            "evidence": [],
            "confidence": 0.0,
            "status": "not_answered"
        }

        # Check policy processor results
        policy_data = policy_processor_results.get("data", {})
        dimensions = policy_data.get("dimensions", {})
        d1_data = dimensions.get("d1_insumos", {})

        # Look for financial allocation evidence
        financial_keywords = [
            "presupuesto", "recursos", "millones", "cop", "asignado", "financiado"
        ]

        gender_keywords = [
            "género", "mujer", "igualdad", "equidad", "casa de la mujer", "atención a vbg"
        ]

        evidence_items = []
        for evidence in d1_data.get("point_evidence", []):
            if (any(keyword in evidence.lower() for keyword in financial_keywords) and
                    any(keyword in evidence.lower() for keyword in gender_keywords)):
                evidence_items.append(evidence)

        if evidence_items:
            answer[
                "answer"] = "Sí, se identifican recursos monetarios explícitamente asignados a programas para la equidad de género."
            answer["confidence"] = min(0.8, len(evidence_items) / 3.0)
            answer["status"] = "answered"
            answer["evidence"] = evidence_items[:3]

        # Check for PPI mention
        ppi_keywords = ["plan plurianual de inversiones", "ppi", "inversiones"]

        ppi_items = [
            evidence for evidence in d1_data.get("point_evidence", [])
            if any(keyword in evidence.lower() for keyword in ppi_keywords)
        ]

        if ppi_items:
            if answer["status"] == "not_answered":
                answer["answer"] = "Se menciona el Plan Plurianual de Inversiones en relación con programas de género."
                answer["confidence"] = min(0.7, len(ppi_items) / 3.0)
                answer["status"] = "answered_with_moderate_confidence"
                answer["evidence"] = ppi_items[:2]
            else:
                answer[
                    "answer"] = "Se identifican tanto asignaciones presupuestales como menciones al PPI para programas de género."
                answer["confidence"] = (answer["confidence"] + 0.7) / 2
                answer["status"] = "answered"
                answer["evidence"] = evidence_items[:2]

        return answer

    def _answer_p1_d1_q4(self, embedding_results: Dict, policy_processor_results: Dict) -> Dict[str, Any]:
        """Answer P1-D1-Q4: Does the PDM describe institutional capacities for gender policy management?"""
        answer = {
            "question_id": "P1-D1-Q4",
            "policy": "P1",
            "dimension": "D1",
            "question_num": 4,
            "answer": "",
            "evidence": [],
            "confidence": 0.0,
            "status": "not_answered"
        }

        # Check policy processor results
        policy_data = policy_processor_results.get("data", {})
        dimensions = policy_data.get("dimensions", {})
        d1_data = dimensions.get("d1_insumos", {})

        # Look for institutional capacity evidence
        institutional_keywords = [
            "secretaría", "comisaría", "entidad", "capacidad", "institucional"
        ]

        gender_keywords = [
            "género", "mujer", "igualdad", "equidad", "vbg", "violencia de género"
        ]

        evidence_items = []
        for evidence in d1_data.get("point_evidence", []):
            if (any(keyword in evidence.lower() for keyword in institutional_keywords) and
                    any(keyword in evidence.lower() for keyword in gender_keywords)):
                evidence_items.append(evidence)

        if evidence_items:
            answer["answer"] = "Sí, el PDM describe capacidades institucionales para gestionar la política de género."
            answer["confidence"] = min(0.8, len(evidence_items) / 3.0)
            answer["status"] = "answered"
            answer["evidence"] = evidence_items[:3]

        # Look for specific capacity elements
        capacity_keywords = [
            "equipo psicosocial", "protocolo de atención", "limitaciones", "barreras institucionales"
        ]

        capacity_items = [
            evidence for evidence in d1_data.get("point_evidence", [])
            if any(keyword in evidence.lower() for keyword in capacity_keywords)
        ]

        if capacity_items:
            if answer["status"] == "not_answered":
                answer["answer"] = "Se mencionan elementos de capacidad institucional para la política de género."
                answer["confidence"] = min(0.7, len(capacity_items) / 3.0)
                answer["status"] = "answered_with_moderate_confidence"
                answer["evidence"] = capacity_items[:2]
            else:
                answer[
                    "answer"] = "El PDM describe tanto capacidades institucionales como elementos específicos de capacidad."
                answer["confidence"] = (answer["confidence"] + 0.7) / 2
                answer["status"] = "answered"
                answer["evidence"] = capacity_items[:2]

        return answer

    def _answer_p1_d1_q5(self, embedding_results: Dict, policy_processor_results: Dict) -> Dict[str, Any]:
        """Answer P1-D1-Q5: Does the plan justify its scope mentioning legal framework or constraints?"""
        answer = {
            "framework_keywords": [
                "ley 1257", "marco legal", "límite fiscal", "cuatrienio", "restricciones"
            ],
            "policy": "P1",
            "dimension": "D1",
            "question_num": 5,
            "answer": "",
            "evidence": [],
            "confidence": 0.0,
            "status": "not_answered"
        }

        # Check policy processor results
        policy_data = policy_processor_results.get("data", {})
        dimensions = policy_data.get("dimensions", {})
        d1_data = dimensions.get("d1_insumos", {})

        # Look for legal framework evidence
        legal_items = [
            evidence for evidence in d1_data.get("point_evidence", [])
            if any(keyword in evidence.lower() for keyword in answer["framework_keywords"])
        ]

        if legal_items:
            answer["answer"] = "Sí, el plan justifica su alcance mencionando el marco legal o restricciones."
            answer["confidence"] = min(0.8, len(legal_items) / 3.0)
            answer["status"] = "answered"
            answer["evidence"] = legal_items[:3]

        # Look for constraint evidence
        constraint_items = [
            evidence for evidence in d1_data.get("point_evidence", [])
            if any(keyword in evidence.lower() for keyword in
                   ["límite fiscal", "restricción", "limitación", "presupuesto"])
        ]

        if constraint_items:
            if answer["status"] == "not_answered":
                answer["answer"] = "Se mencionan restricciones explícitas de tipo presupuestal o temporal."
                answer["confidence"] = min(0.7, len(constraint_items) / 3.0)
                answer["status"] = "answered_with_moderate_confidence"
                answer["evidence"] = constraint_items[:2]
            else:
                answer["answer"] = "El plan menciona tanto marco legal como restricciones explícitas."
                answer["confidence"] = (answer["confidence"] + 0.7) / 2
                answer["status"] = "answered"
                answer["evidence"] = legal_items[:2]

        return answer

    def _answer_p1_d2_q6(self, embedding_results: Dict, policy_processor_results: Dict) -> Dict[str, Any]:
        """Answer P1-D2-Q6: Are gender equality activities presented in structured format?"""
        answer = {
            "question_id": "P1-D2-Q6",
            "policy": "P1",
            "dimension": "D2",
            "question_num": 6,
            "answer": "",
            "evidence": [],
            "confidence": 0.0,
            "status": "not_answered"
        }

        # Check policy processor results
        policy_data = policy_processor_results.get("data", {})
        dimensions = policy_data.get("dimensions", {})
        d2_data = dimensions.get("d2_actividades", {})

        # Look for structured format evidence
        structure_keywords = [
            "tabla", "cuadro", "columna", "responsable", "producto", "cronograma", "costo", "presupuesto"
        ]

        structure_items = [
            evidence for evidence in d2_data.get("point_evidence", [])
            if any(keyword in evidence.lower() for keyword in structure_keywords)
        ]

        if structure_items:
            answer["answer"] = "Sí, las actividades para la equidad de género se presentan en formato estructurado."
            answer["confidence"] = min(0.8, len(structure_items) / 3.0)
            answer["status"] = "answered"
            answer["evidence"] = structure_items[:3]

        return answer

    def _answer_p1_d2_q7(self, embedding_results: Dict, policy_processor_results: Dict) -> Dict[str, Any]:
        """Answer P1-D2-Q7: Do activities detail instruments, target population, and causal logic?"""
        answer = {
            "question_id": "P1-D2-Q7",
            "policy": "P1",
            "dimension": "D2",
            "question_num": 7,
            "answer": "",
            "evidence": [],
            "confidence": 0.0,
            "status": "not_answered"
        }

        # Check policy processor results
        policy_data = policy_processor.get("data", {})
        dimensions = policy_data.get("dimensions", {})
        d2_data = dimensions.get("d2_actividades", {})

        # Look for instrument evidence
        instrument_keywords = [
            "mediante", "a través de", "instrumento", "herramienta"
        ]

        instrument_items = [
            evidence for evidence in d2_data.get("point_evidence", [])
            if any(keyword in evidence.lower() for keyword in instrument_keywords)
        ]

        # Look for target population evidence
        population_keywords = [
            "mujeres rurales", "madres cabeza de familia", "población objetivo"
        ]

        population_items = [
            evidence for evidence in d2_data.get("point_evidence", [])
            if any(keyword in evidence.lower() for keyword in population_keywords)
        ]

        # Look for causal logic evidence
        causal_keywords = [
            "para generar", "para reducir", "porque reduce", "para mejorar"
        ]

        causal_items = [
            evidence for evidence in d2_data.get("point_evidence", [])
            if any(keyword in evidence.lower() for keyword in causal_keywords)
        ]

        if instrument_items and population_items and causal_items:
            answer[
                "answer"] = "Sí, las actividades de género detallan el instrumento, la población objetivo y la lógica causal."
            answer["confidence"] = min(0.8, (
                    len(instrument_items) + len(population_items) + len(causal_items)) / 9.0
                                       )
            answer["status"] = "answered"
            answer["evidence"] = instrument_items[:2] + population_items[:1] + causal_items[:1]
        elif instrument_items and population_items:
            answer[
                "answer"] = "Las actividades de género mencionan el instrumento y la población objetivo, pero no la lógica causal."
            answer["confidence"] = min(0.6, (len(instrument_items) + len(population_items)) / 6.0)
            answer["status"] = "answered_with_moderate_confidence"
            answer["evidence"] = instrument_items[:2] + population_items[:1]
        elif instrument_items:
            answer[
                "answer"] = "Las actividades de género mencionan el instrumento, pero no la población objetivo ni la lógica causal."
            answer["confidence"] = min(0.5, len(instrument_items) / 3.0)
            answer["status"] = "answered_with_low_confidence"
            answer["evidence"] = instrument_items[:2]

        return answer

    def _answer_p1_d2_q8(self, embedding_results: Dict, policy_processor_results: Dict) -> Dict[str, Any]:
        """Answer P1-D2-Q8: Do activities link to root causes of gender inequality?"""
        answer = {
            "question_id": " cause_keywords": [
            "causa raíz", "causas raíz", "causa fundamental", "causa estructural"
        ],
        "policy": "P1",
        "dimension": "D2",
        "question_num": 8,
        "answer": "",
        "evidence": [],
        "confidence": 0.0,
        "status": "not_answered"
        }

        # Check policy processor results
        policy_data = policy_processor_results.get("data", {})
        dimensions = policy_data.get("dimensions", {})
        d2_data = dimensions.get("d2_actividades", {})

        # Look for cause-effect linking evidence
        cause_items = [
            evidence for evidence in d2_data.get("point_evidence", [])
            if any(keyword in evidence.lower() for keyword in answer["cause_keywords"])
        ]

        # Look for root cause specific evidence
        root_cause_keywords = [
            "dependencia económica", "patrones culturales", "desigualdad estructural"
        ]

        root_cause_items = [
            evidence for evidence in d2_data.get("point_evidence", [])
            if any(keyword in evidence.lower() for keyword in root_cause_keywords)
        ]

        if cause_items and root_cause_items:
            answer["answer"] = "Sí, las actividades se vinculan con las causas raíz de la desigualdad de género."
            answer["confidence"] = min(0.8, (len(cause_items) + len(root_cause_items)) / 6.0)
            answer["status"] = "answered"
            answer["evidence"] = cause_items[:2] + root_cause_items[:1]
        elif cause_items:
            answer["answer"] = "Las actividades mencionan causas, pero no necesariamente causas raíz de la desigualdad."
            answer["confidence"] = min(0.6, len(cause_items) / 3.0)
            answer["status"] = "answered_with_moderate_confidence"
            answer["evidence"] = cause_items[:2]

        return answer

    def _answer_p1_d2_q9(self, embedding_results: Dict, policy_processor_results: Dict) -> Dict[str, Any]:
        """Answer P1-D2-Q9: Does the plan identify risks and mitigation measures?"""
        answer = {
            "risk_keywords": [
                "riesgo", "obstáculo", "barrera", "mitigación", "medida de mitigación"
            ],
            "policy": "P1",
            "dimension": "D2",
            "question_num": 9,
            "answer": "",
            "evidence": [],
            "confidence": 0.0,
            "status": "not_answered"
        }

        # Check policy processor results
        policy_data = policy_data = policy_processor_results.get("data", {})
        dimensions = policy_data.get("dimensions", {})
        d2_data = dimensions.get("d2_actividades", {})

        # Look for risk evidence
        risk_items = [
            evidence for evidence in d2_data.get("point_evidence", [])
            if any(keyword in evidence.lower() for keyword in answer["risk_keywords"])
        ]

        # Look for mitigation evidence
        mitigation_items = [
            evidence for evidence in d2_data.get("point_evidence", [])
            if "mitigación" in evidence.lower() or "mitigar" in evidence.lower()
        ]

        if risk_items and mitigation_items:
            answer["answer"] = "Sí, el plan identifica riesgos y propone medidas de mitigación."
            answer["confidence"] = min(0.8, (len(risk_items) + len(mitigation_items)) / 6.0)
            answer["status"] = "answered"
            answer["evidence"] = risk_items[:2] + mitigation_items[:1]
        elif risk_items:
            answer["answer"] = "El plan identifica riesgos, pero no medidas de mitigación claras."
            answer["confidence"] = min(0.5, len(risk_items) / 3.0)
            answer["status"] = "answered_with_low_confidence"
            answer["evidence"] = risk_items[:2]

        return answer

    def _answer_p1_d2_q10(self, embedding_results: Dict, policy_processor_results: Dict) -> Dict[str, Any]:
        """Answer P1-D2-Q10: Do activities demonstrate a coherent strategy?"""
        answer = {
            "strategy_keywords": [
                "complementariedad", "sinergia", "secuencia lógica", "estrategia coherente"
            ],
            "policy": "P1",
            "dimension": "D2",
            "question_num": 10,
            "answer": "",
            "evidence": [],
            "confidence": 0.0,
            "status": "not_answered"
        }

        # Check policy processor results
        policy_data = policy_processor_results.get("data", {})
        dimensions = policy_data.get("dimensions", {})
        d2_data = dimensions.get("d2_actividades", {})

        # Look for strategy evidence
        strategy_items = [
            evidence for evidence in d2_data.get("point_evidence", [])
            if any(keyword in evidence.lower() for keyword in answer["strategy_keywords"])
        ]

        if strategy_items:
            answer["answer"] = "Sí, el conjunto de actividades demuestra una estrategia coherente."
            answer["confidence"] = min(0.8, len(strategy_items) / 3.0)
            answer["status"] = "answered"
            answer["evidence"] = strategy_items[:3]

        return answer

    # Add more question handlers as needed for other P1 questions...

    def _answer_p2_d1_q1(self, embedding_results: Dict, policy_processor_results: Dict) -> Dict[str, Any]:
        """Answer P2-D1-Q1: Does diagnosis present numerical data for violence prevention?"""
        answer = {
            "numerical_keywords": [
                "tasa de homicidio", "cifras de desplazamiento", "casos de reclutamiento"
            ],
            "policy": "P2",
            "dimension": "D1",
            "question_num": 1,
            "answer": "",
            "evidence": [],
            "confidence": 0.0,
            "status": "not_answered"
        }

        # Check numerical consistency results
        numerical_results = embedding_results.get("numerical_consistency", {})
        if numerical_results.get("status") == "success":
            point_estimate = numerical_results.get("data", {}).get("point_estimate", 0.0)
            evidence_strength = numerical_results.get("data", {}).get("evidence_strength", 0.0)

            if point_estimate > 0.5 and evidence_strength > 0.5:
                answer["answer"] = "Sí, el diagnóstico presenta datos numéricos para la prevención de la violencia."
                answer["confidence"] = (point_estimate + evidence_strength) / 2
                answer["status"] = "answered"

                answer["evidence"] = numerical_results.get("evidence", [])

        # Check policy processor results
        policy_data = policy_processor_results.get("data", {})
        dimensions = policy_data.get("dimensions", {})
        d1_data = dimensions.get("d1_insumos", {})

        # Look for numerical evidence
        numerical_evidence = [
            evidence for evidence in d1_data.get("point_evidence", [])
            if any(keyword in evidence.lower() for keyword in answer["numerical_keywords"])
        ]

        if numerical_evidence:
            if answer["status"] == "not_answered":
                answer["answer"] = "Se encontraron menciones de datos numéricos en el diagnóstico."
                answer["confidence"] = min(0.7, len(numerical_evidence) / 5.0)
                answer["status"] = "answered_with_moderate_confidence"
                answer["evidence"] = numerical_evidence[:3]

        return answer

    # Add more question handlers for P2, P3, etc. as needed...

    def answer_all_questions(self, analysis_results: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Answer all P1-P10 questions based on analysis results

        Args:
            analysis_results: Results from PolicyAnalysisPipeline

        Returns:
            Dictionary with answers for all questions
        """
        self.logger.info("Answering all policy questions")

        all_answers = {}

        # P1 questions
        for i in range(1, 31):
            question_id = f"P1-D{i // 5}-{(1 if i % 5 != 0 else 5)}"
            all_answers[question_id] = self.answer_question(question_id, analysis_results)

        # P2 questions
        for i in range(1, 31):
            question_id = f"P2-D{i // 5}-{1 if i % 5 != 0 else 5}"
            all_answers[question_id] = self.answer_question(question_id, analysis_results)

        # P3 questions
        for i in range(1, 31):
            question_id = f"P3-D{i // 5}-{1 if i % 5 != 0 else 5}"
            all_answers[question_id] = self.answer_question(question_id, analysis_results)

        # P4 questions
        for i in range(1, 31):
            question_id = f"P4-D{i // 5}-{1 if i % 5 != 0 else 5}"
            all_answers[question_id] = self.answer_question(question_id, analysis_results)

        # P5 questions
        for i in range(1, 31):
            question_id = f"P5-D{i // 5}-{1 if i % 5 != 0 else 5}"
            all_answers[question_id] = self.answer_question(question_id, analysis_results)

        # P6 questions
        for i in range(1, 31):
            question_id = f"P6-D{i // 5}-{1 if i % 5 != 0 else 5}"
            all_answers[question_id] = self.answer_question(question_id, analysis_results)

        # P7 questions
        for i in range(1, 31):
            question_id = f"P7-D{i // 5}-{1 if i % 5 != 0 else 5}"
            all_answers[question_id] = self.answer_question(question_id, analysis_results)

        # P8 questions
        for i in range(1, 31):
            question_id = f"P8-D{i // 5}-{1 if i % 5 != 0 else 5}"
            all_answers[question_id] = self.answer_question(question_id, analysis_results)

        # P9 questions
        for i in range(1, 31):
            question_id = f"P9-D{i // 5}-{1 if i % 5 != 0 else 5}"
            all_answers[question_id] = self.answer_question(question_id, analysis_results)

        # P10 questions
        for i in range(1, 31):
            question_id = f"P10-D{i // 5}-{1 if i % 5 != 0 else 5}"
            all_answers[question_id] = self.answer_question(question_id, analysis_results)

        return all_answers


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Example usage
    pipeline = PolicyAnalysisPipeline()

    # Sample document text
    sample_text = """
    PLAN DE DESARROLLO MUNICIPAL 2024-2027
    MUNICIPIO DE EJEMPLO, COLOMBIA

    1. DIAGNÓSTICO MUNICIPAL

    El municipio presenta una población de 75,320 habitantes según proyecciones DANE 2023. La caracterización socioeconómica evidencia que el 18.5% de la población se encuentra en situación de vulnerabilidad.

    2. OBJETIVO ESTRATÉGICO: DERECHOS DE LAS MUJERES E IGUALDAD DE GÉNERO

    Reducir las brechas de género en el municipio mediante la implementación de políticas públicas integrales que promuevan la autonomía económica, la participación política y la prevención de violencias basadas en género.

    2.1 LÍNEA BASE Y RECURSOS ASIGNADOS

    Indicador de línea base: Tasa de participación laboral femenina 42.3%
    Meta cuatrienio: Alcanzar 55.8% (incremento del 32%)
    Presupuesto asignado: $450 millones de pesos para el período 2024-2027
    Fuentes de financiación: 60% recursos propios, 40% transferencias SGP
    """

    document_metadata = {
        "doc_id": "sample_pdm",
        "municipality": "Ejemplo",
        "year": "2024-2027"
    }

    # Analyze the document
    results = pipeline.analyze_policy_document(sample_text, document_metadata)

    # Answer all questions
    answers = PolicyQuestionAnswerer().answer_all_questions(results)

    # Print results for P1-D1-Q1 as an example
    print("\nP1-D1-Q1 Answer:")
    print(f"Answer: {answers['P1-D1-Q1']['answer']}")
    print(f"Confidence: {answers['P1-D1-Q1']['confidence']:.2f}")
    print(f"Evidence: {answers['P1-D1-Q1']['evidence']}")

    # Print overall confidence
    print(f"\nOverall Confidence: {results['overall_confidence']:.2f}")