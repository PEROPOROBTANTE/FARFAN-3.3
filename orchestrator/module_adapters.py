# module_adapters.py - Complete Integration Implementation
# Production-ready version with all adapters properly implemented

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
# ADAPTER REGISTRY
# ============================================================================

class ModuleAdapterRegistry:
    """
    Central registry for all module adapters
    """

    def __init__(self):
        self.adapters = {}
        self._register_all_adapters()

    def _register_all_adapters(self):
        """Register all available adapters"""
        self.adapters["policy_processor"] = PolicyProcessorAdapter()
        self.adapters["analyzer_one"] = AnalyzerOneAdapter()
        self.adapters["contradiction_detector"] = ContradictionDetectorAdapter()
        self.adapters["dereck_beach"] = DerekBeachAdapter()
        self.adapters["embedding_policy"] = EmbeddingPolicyAdapter()
        self.adapters["financial_analyzer"] = FinancialAnalyzerAdapter()
        self.adapters["causal_processor"] = CausalProcessorAdapter()
        self.adapters["policy_segmenter"] = PolicySegmenterAdapter()
        self.adapters["semantic_processor"] = SemanticProcessorAdapter()
        self.adapters["bayesian_integrator"] = BayesianIntegratorAdapter()
        self.adapters["validation_framework"] = ValidationFrameworkAdapter()
        self.adapters["municipal_analyzer"] = MunicipalAnalyzerAdapter()
        self.adapters["pdet_analyzer"] = PDETAnalyzerAdapter()
        self.adapters["decologo_processor"] = DecologoProcessorAdapter()
        self.adapters["embedding_analyzer"] = EmbeddingAnalyzerAdapter()
        self.adapters["causal_validator"] = CausalValidatorAdapter()

        logger.info(f"Registered {len(self.adapters)} module adapters")

    def execute_module_method(self, module_name: str, method_name: str,
                              args: List[Any], kwargs: Dict[str, Any]) -> ModuleResult:
        """Execute a method on a registered module"""
        if module_name not in self.adapters:
            return ModuleResult(
                module_name=module_name,
                class_name="Unknown",
                method_name=method_name,
                status="failed",
                data={},
                evidence=[],
                confidence=0.0,
                execution_time=0.0,
                errors=[f"Module {module_name} not registered"]
            )

        adapter = self.adapters[module_name]
        return adapter.execute(method_name, args, kwargs)

    def get_available_modules(self) -> List[str]:
        """Get list of available modules"""
        return [name for name, adapter in self.adapters.items() if adapter.available]

    def get_module_status(self) -> Dict[str, bool]:
        """Get status of all modules"""
        return {name: adapter.available for name, adapter in self.adapters.items()}


# ============================================================================
# BASE ADAPTER CLASS
# ============================================================================

class BaseAdapter:
    """Base class for all module adapters with common functionality"""

    def __init__(self, module_name: str):
        self.module_name = module_name
        self.available = False
        self.logger = logging.getLogger(f"{__name__}.{module_name}")

    def _create_unavailable_result(self, method_name: str, start_time: float) -> ModuleResult:
        """Create a standard result when module is not available"""
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

    def _create_error_result(self, method_name: str, start_time: float, error: Exception) -> ModuleResult:
        """Create a standard error result"""
        return ModuleResult(
            module_name=self.module_name,
            class_name="Unknown",
            method_name=method_name,
            status="failed",
            data={},
            evidence=[],
            confidence=0.0,
            execution_time=time.time() - start_time,
            errors=[str(error)]
        )


# ============================================================================
# ADAPTER 1: POLICY PROCESSOR (DECALOGO FRAMEWORK)
# ============================================================================

class PolicyProcessorAdapter(BaseAdapter):
    """
    Adapter for IndustrialPolicyProcessor from DECALOGO framework
    """

    def __init__(self):
        super().__init__("policy_processor")
        self._load_module()

    def _load_module(self):
        """Load the IndustrialPolicyProcessor module"""
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
            self.logger.info(f"✓ {self.module_name} loaded with DECALOGO framework")
        except ImportError as e:
            self.logger.warning(f"✗ {self.module_name} NOT available: {e}")
            self.available = False

    def execute(self, method_name: str, args: List[Any], kwargs: Dict[str, Any]) -> ModuleResult:
        """
        Supported methods:
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
            self.logger.error(f"{self.module_name}.{method_name} failed: {e}", exc_info=True)
            return self._create_error_result(method_name, start_time, e)

    def _execute_process(self, text: str, **kwargs) -> ModuleResult:
        """Execute IndustrialPolicyProcessor.process()"""
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
        """Execute IndustrialPolicyProcessor._extract_point_evidence()"""
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
        """Execute PolicyTextProcessor.extract_policy_sections()"""
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
        """Execute BayesianEvidenceScorer.score_evidence()"""
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


# ============================================================================
# ADAPTER 2: ANALYZER ONE (MUNICIPAL ANALYZER)
# ============================================================================

class AnalyzerOneAdapter(BaseAdapter):
    """
    Adapter for MunicipalAnalyzer from Advanced Municipal Plan Analyzer
    """

    def __init__(self):
        super().__init__("analyzer_one")
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
            self.logger.info(f"✓ {self.module_name} loaded with Municipal Analyzer framework")
        except ImportError as e:
            self.logger.warning(f"✗ {self.module_name} NOT available: {e}")

    def execute(self, method_name: str, args: List[Any], kwargs: Dict[str, Any]) -> ModuleResult:
        """
        Supported methods:
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
            self.logger.error(f"{self.module_name}.{method_name} failed: {e}", exc_info=True)
            return self._create_error_result(method_name, start_time, e)

    def _execute_analyze_document(self, text: str, **kwargs) -> ModuleResult:
        """Execute MunicipalAnalyzer.analyze_document()"""
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
        """Execute SemanticAnalyzer.extract_semantic_cube()"""
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
        """Execute PerformanceAnalyzer.diagnose_critical_links()"""
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
        """Execute TextMiningEngine.extract_value_chain()"""
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


# ============================================================================
# ADAPTER 3: CONTRADICTION DETECTOR
# ============================================================================

class ContradictionDetectorAdapter(BaseAdapter):
    """
    Adapter for PolicyContradictionDetector from Advanced Policy Contradiction Detection System
    """

    def __init__(self):
        super().__init__("contradiction_detector")
        self._load_module()

    def _load_module(self):
        try:
            from contradiction_deteccion import (
                PolicyContradictionDetector,
                BayesianConfidenceCalculator,
                TemporalLogicVerifier,
                ContradictionType,
                PolicyDimension
            )
            self.PolicyContradictionDetector = PolicyContradictionDetector
            self.BayesianConfidenceCalculator = BayesianConfidenceCalculator
            self.TemporalLogicVerifier = TemporalLogicVerifier
            self.ContradictionType = ContradictionType
            self.PolicyDimension = PolicyDimension
            self.available = True
            self.logger.info(f"✓ {self.module_name} loaded with Contradiction Detection framework")
        except ImportError as e:
            self.logger.warning(f"✗ {self.module_name} NOT available: {e}")

    def execute(self, method_name: str, args: List[Any], kwargs: Dict[str, Any]) -> ModuleResult:
        """
        Supported methods:
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
            self.logger.error(f"{self.module_name}.{method_name} failed: {e}", exc_info=True)
            return self._create_error_result(method_name, start_time, e)

    def _execute_detect(self, text: str, plan_name: str = "Unknown", **kwargs) -> ModuleResult:
        """Execute PolicyContradictionDetector.detect()"""
        dimension = kwargs.get('dimension', self.PolicyDimension.ESTRATEGICO)
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
        """Execute BayesianConfidenceCalculator.calculate()"""
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
        """Execute TemporalLogicVerifier.verify()"""
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


# ============================================================================
# ADAPTER 4: DEREK BEACH (CDAF FRAMEWORK)
# ============================================================================

class DerekBeachAdapter(BaseAdapter):
    """
    Adapter for CDAFFramework from Causal Deconstruction and Audit Framework
    """

    def __init__(self):
        super().__init__("dereck_beach")
        self._load_module()

    def _load_module(self):
        try:
            # Load module without extension
            spec = importlib.util.spec_from_file_location(
                "dereck_beach",
                "/Users/recovered/PycharmProjects/FLUX/FARFAN-3.0/dereck_beach"
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Extract core classes
                self.CDAFFramework = getattr(module, "CDAFFramework", None)
                self.BeachEvidentialTest = getattr(module, "BeachEvidentialTest", None)
                self.CausalExtractor = getattr(module, "CausalExtractor", None)
                self.MechanismPartExtractor = getattr(module, "MechanismPartExtractor", None)
                self.BayesianMechanismInference = getattr(module, "BayesianMechanismInference", None)
                self.FinancialAuditor = getattr(module, "FinancialAuditor", None)
                self.OperationalizationAuditor = getattr(module, "OperationalizationAuditor", None)
                self.ReportingEngine = getattr(module, "ReportingEngine", None)
                self.ConfigLoader = getattr(module, "ConfigLoader", None)

                self.available = all([
                    self.CDAFFramework is not None,
                    self.BeachEvidentialTest is not None
                ])

                if self.available:
                    self.logger.info(f"✓ {self.module_name} loaded with CDAF framework")
                else:
                    self.logger.warning(f"✗ {self.module_name} missing core classes")
        except Exception as e:
            self.logger.warning(f"✗ {self.module_name} NOT available: {e}")

    def execute(self, method_name: str, args: List[Any], kwargs: Dict[str, Any]) -> ModuleResult:
        """
        Supported methods:
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
            self.logger.error(f"{self.module_name}.{method_name} failed: {e}", exc_info=True)
            return self._create_error_result(method_name, start_time, e)

    def _execute_process_document(self, pdf_path_or_text: str, plan_name: str, **kwargs) -> ModuleResult:
        """Execute CDAFFramework.process_document()"""
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
        """Execute BeachEvidentialTest.classify_test()"""
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
            confidence=0.9,  # High confidence in formal classification
            execution_time=0.0
        )

    def _execute_apply_test_logic(self, test_type: str, evidence_found: bool,
                                  prior: float, bayes_factor: float, **kwargs) -> ModuleResult:
        """Execute BeachEvidentialTest.apply_test_logic()"""
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
        """Execute CausalExtractor.extract_causal_hierarchy()"""
        if not self.CausalExtractor:
            raise RuntimeError("CausalExtractor not available")

        config = kwargs.get('config', {})
        extractor = self.CausalExtractor(config)
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
        """Execute MechanismPartExtractor.extract_entity_activity()"""
        if not self.MechanismPartExtractor:
            raise RuntimeError("MechanismPartExtractor not available")

        config = kwargs.get('config', {})
        extractor = self.MechanismPartExtractor(config)
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
        """Execute BayesianMechanismInference.infer_mechanisms()"""
        if not self.BayesianMechanismInference:
            raise RuntimeError("BayesianMechanismInference not available")

        config = kwargs.get('config', {})
        inference = self.BayesianMechanismInference(config)
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
        """Execute FinancialAuditor.trace_financial_allocation()"""
        if not self.FinancialAuditor:
            raise RuntimeError("FinancialAuditor not available")

        config = kwargs.get('config', {})
        auditor = self.FinancialAuditor(config)
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
        """Execute OperationalizationAuditor.audit_evidence_traceability()"""
        if not self.OperationalizationAuditor:
            raise RuntimeError("OperationalizationAuditor not available")

        config = kwargs.get('config', {})
        auditor = self.OperationalizationAuditor(config)
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
        """Execute ReportingEngine.generate_confidence_report()"""
        if not self.ReportingEngine:
            raise RuntimeError("ReportingEngine not available")

        config = kwargs.get('config', {})
        reporter = self.ReportingEngine(config)
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


# ============================================================================
# ADAPTER 5: EMBEDDING POLICY
# ============================================================================

class EmbeddingPolicyAdapter(BaseAdapter):
    """
    Adapter for PolicyAnalysisEmbedder from Sistema de Incrustación Semántica
    """

    def __init__(self):
        super().__init__("embedding_policy")
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
            self.logger.info(f"✓ {self.module_name} loaded with Semantic Embedding framework")
        except ImportError as e:
            self.logger.warning(f"✗ {self.module_name} NOT available: {e}")

    def execute(self, method_name: str, args: List[Any], kwargs: Dict[str, Any]) -> ModuleResult:
        """
        Supported methods:
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
            self.logger.error(f"{self.module_name}.{method_name} failed: {e}", exc_info=True)
            return self._create_error_result(method_name, start_time, e)

    def _execute_chunk_document(self, text: str, document_metadata: Dict[str, Any], **kwargs) -> ModuleResult:
        """Execute AdvancedSemanticChunker.chunk_document()"""
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
        """Execute BayesianNumericalAnalyzer.evaluate_policy_metric()"""
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
        """Execute PolicyCrossEncoderReranker.rerank()"""
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
        """Execute PolicyAnalysisEmbedder.process_document()"""
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
        """Execute PolicyAnalysisEmbedder.semantic_search()"""
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
        """Execute PolicyAnalysisEmbedder.evaluate_policy_numerical_consistency()"""
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


# ============================================================================
# ADAPTER 6: FINANCIAL ANALYZER
# ============================================================================

class FinancialAnalyzerAdapter(BaseAdapter):
    """
    Adapter for financial analysis components from PDET Causal Analysis
    """

    def __init__(self):
        super().__init__("financial_analyzer")
        self._load_module()

    def _load_module(self):
        try:
            from financiero_viabilidad_tablas import (
                FinancialAnalyzer,
                FinancialIndicator,
                BudgetAnalyzer
            )
            self.FinancialAnalyzer = FinancialAnalyzer
            self.FinancialIndicator = FinancialIndicator
            self.BudgetAnalyzer = BudgetAnalyzer
            self.available = True
            self.logger.info(f"✓ {self.module_name} loaded with Financial Analysis framework")
        except ImportError as e:
            self.logger.warning(f"✗ {self.module_name} NOT available: {e}")

    def execute(self, method_name: str, args: List[Any], kwargs: Dict[str, Any]) -> ModuleResult:
        """
        Supported methods:
        - analyze_financial_feasibility(text: str) -> Dict
        - extract_financial_indicators(text: str) -> List[FinancialIndicator]
        - analyze_budget_allocation(budget_data: Dict) -> Dict
        """
        start_time = time.time()

        if not self.available:
            return self._create_unavailable_result(method_name, start_time)

        try:
            if method_name == "analyze_financial_feasibility":
                result = self._execute_analyze_financial_feasibility(*args, **kwargs)
            elif method_name == "extract_financial_indicators":
                result = self._execute_extract_financial_indicators(*args, **kwargs)
            elif method_name == "analyze_budget_allocation":
                result = self._execute_analyze_budget_allocation(*args, **kwargs)
            else:
                raise ValueError(f"Unknown method: {method_name}")

            result.execution_time = time.time() - start_time
            return result

        except Exception as e:
            self.logger.error(f"{self.module_name}.{method_name} failed: {e}", exc_info=True)
            return self._create_error_result(method_name, start_time, e)

    def _execute_analyze_financial_feasibility(self, text: str, **kwargs) -> ModuleResult:
        """Execute FinancialAnalyzer.analyze_financial_feasibility()"""
        analyzer = self.FinancialAnalyzer()
        result = analyzer.analyze_financial_feasibility(text)

        evidence = [{
            "type": "financial_feasibility",
            "viability_score": result.get("viability_score", 0.5),
            "risk_factors": result.get("risk_factors", []),
            "funding_sources": result.get("funding_sources", [])
        }]

        confidence = result.get("viability_score", 0.5)

        return ModuleResult(
            module_name=self.module_name,
            class_name="FinancialAnalyzer",
            method_name="analyze_financial_feasibility",
            status="success",
            data=result,
            evidence=evidence,
            confidence=confidence,
            execution_time=0.0
        )

    def _execute_extract_financial_indicators(self, text: str, **kwargs) -> ModuleResult:
        """Execute FinancialAnalyzer.extract_financial_indicators()"""
        analyzer = self.FinancialAnalyzer()
        indicators = analyzer.extract_financial_indicators(text)

        evidence = [{
            "type": "financial_indicators",
            "indicator_count": len(indicators),
            "indicators": [ind.to_dict() if hasattr(ind, 'to_dict') else str(ind) for ind in indicators[:5]]
        }]

        return ModuleResult(
            module_name=self.module_name,
            class_name="FinancialAnalyzer",
            method_name="extract_financial_indicators",
            status="success",
            data={"indicators": indicators},
            evidence=evidence,
            confidence=0.8,
            execution_time=0.0
        )

    def _execute_analyze_budget_allocation(self, budget_data: Dict, **kwargs) -> ModuleResult:
        """Execute BudgetAnalyzer.analyze_budget_allocation()"""
        analyzer = self.BudgetAnalyzer()
        result = analyzer.analyze_budget_allocation(budget_data)

        evidence = [{
            "type": "budget_allocation",
            "total_budget": result.get("total_budget", 0),
            "allocation_efficiency": result.get("allocation_efficiency", 0.5),
            "discrepancies": result.get("discrepancies", [])
        }]

        confidence = result.get("allocation_efficiency", 0.5)

        return ModuleResult(
            module_name=self.module_name,
            class_name="BudgetAnalyzer",
            method_name="analyze_budget_allocation",
            status="success",
            data=result,
            evidence=evidence,
            confidence=confidence,
            execution_time=0.0
        )


# ============================================================================
# ADAPTER 7: CAUSAL PROCESSOR
# ============================================================================

class CausalProcessorAdapter(BaseAdapter):
    """
    Adapter for causal analysis components from PDET Causal Analysis
    """

    def __init__(self):
        super().__init__("causal_processor")
        self._load_module()

    def _load_module(self):
        try:
            from pdet_causal_analysis import (
                PDETMunicipalPlanAnalyzer,
                CausalNode,
                CausalEdge,
                CausalDAG,
                CausalEffect
            )
            self.PDETMunicipalPlanAnalyzer = PDETMunicipalPlanAnalyzer
            self.CausalNode = CausalNode
            self.CausalEdge = CausalEdge
            self.CausalDAG = CausalDAG
            self.CausalEffect = CausalEffect
            self.available = True
            self.logger.info(f"✓ {self.module_name} loaded with Causal Analysis framework")
        except ImportError as e:
            self.logger.warning(f"✗ {self.module_name} NOT available: {e}")

    def execute(self, method_name: str, args: List[Any], kwargs: Dict[str, Any]) -> ModuleResult:
        """
        Supported methods:
        - analyze(text: str) -> Dict
        - construct_causal_dag(text: str) -> CausalDAG
        - estimate_causal_effects(dag: CausalDAG, treatment: str, outcome: str) -> List[CausalEffect]
        - generate_counterfactuals(dag: CausalDAG, intervention: Dict) -> List[Dict]
        """
        start_time = time.time()

        if not self.available:
            return self._create_unavailable_result(method_name, start_time)

        try:
            if method_name == "analyze":
                result = self._execute_analyze(*args, **kwargs)
            elif method_name == "construct_causal_dag":
                result = self._execute_construct_causal_dag(*args, **kwargs)
            elif method_name == "estimate_causal_effects":
                result = self._execute_estimate_causal_effects(*args, **kwargs)
            elif method_name == "generate_counterfactuals":
                result = self._execute_generate_counterfactuals(*args, **kwargs)
            else:
                raise ValueError(f"Unknown method: {method_name}")

            result.execution_time = time.time() - start_time
            return result

        except Exception as e:
            self.logger.error(f"{self.module_name}.{method_name} failed: {e}", exc_info=True)
            return self._create_error_result(method_name, start_time, e)

    def _execute_analyze(self, text: str, **kwargs) -> ModuleResult:
        """Execute PDETMunicipalPlanAnalyzer.analyze_municipal_plan()"""
        analyzer = self.PDETMunicipalPlanAnalyzer()
        result = analyzer.analyze_municipal_plan(text)

        evidence = []

        if "causal_dag" in result:
            dag = result["causal_dag"]
            evidence.append({
                "type": "causal_dag",
                "node_count": len(dag.nodes) if hasattr(dag, 'nodes') else 0,
                "edge_count": len(dag.edges) if hasattr(dag, 'edges') else 0
            })

        if "financial_analysis" in result:
            financial = result["financial_analysis"]
            evidence.append({
                "type": "financial_analysis",
                "viability_score": financial.get("viability_score", 0.5),
                "sustainability_score": financial.get("sustainability_score", 0.5)
            })

        if "responsible_entities" in result:
            entities = result["responsible_entities"]
            evidence.append({
                "type": "responsible_entities",
                "entity_count": len(entities),
                "high_specificity_count": len([e for e in entities if e.get("specificity_score", 0) > 0.7])
            })

        confidence = result.get("quality_score", {}).get("overall_score", 0.5)

        return ModuleResult(
            module_name=self.module_name,
            class_name="PDETMunicipalPlanAnalyzer",
            method_name="analyze",
            status="success",
            data=result,
            evidence=evidence,
            confidence=confidence,
            execution_time=0.0,
            metadata={"evidence_types": len(evidence)}
        )

    def _execute_construct_causal_dag(self, text: str, **kwargs) -> ModuleResult:
        """Execute PDETMunicipalPlanAnalyzer.construct_causal_dag()"""
        analyzer = self.PDETMunicipalPlanAnalyzer()
        dag = analyzer.construct_causal_dag(text)

        evidence = [{
            "type": "causal_dag_construction",
            "node_count": len(dag.nodes) if hasattr(dag, 'nodes') else 0,
            "edge_count": len(dag.edges) if hasattr(dag, 'edges') else 0,
            "acyclic": dag.is_acyclic() if hasattr(dag, 'is_acyclic') else True
        }]

        return ModuleResult(
            module_name=self.module_name,
            class_name="PDETMunicipalPlanAnalyzer",
            method_name="construct_causal_dag",
            status="success",
            data={"dag": dag},
            evidence=evidence,
            confidence=0.7,
            execution_time=0.0
        )

    def _execute_estimate_causal_effects(self, dag: CausalDAG, treatment: str, outcome: str, **kwargs) -> ModuleResult:
        """Execute PDETMunicipalPlanAnalyzer.estimate_causal_effects()"""
        analyzer = self.PDETMunicipalPlanAnalyzer()
        effects = analyzer.estimate_causal_effects(dag, treatment, outcome)

        evidence = [{
            "type": "causal_effects",
            "treatment": treatment,
            "outcome": outcome,
            "effect_count": len(effects),
            "avg_effect_size": np.mean([e.effect_size for e in effects]) if effects else 0.0
        }]

        confidence = np.mean([e.confidence for e in effects]) if effects else 0.0

        return ModuleResult(
            module_name=self.module_name,
            class_name="PDETMunicipalPlanAnalyzer",
            method_name="estimate_causal_effects",
            status="success",
            data={"effects": effects},
            evidence=evidence,
            confidence=confidence,
            execution_time=0.0
        )

    def _execute_generate_counterfactuals(self, dag: CausalDAG, intervention: Dict, **kwargs) -> ModuleResult:
        """Execute PDETMunicipalPlanAnalyzer.generate_counterfactuals()"""
        analyzer = self.PDETMunicipalPlanAnalyzer()
        scenarios = analyzer.generate_counterfactuals(dag, intervention)

        evidence = [{
            "type": "counterfactual_scenarios",
            "intervention": intervention,
            "scenario_count": len(scenarios),
            "avg_outcome_change": np.mean([s.outcome_change for s in scenarios]) if scenarios else 0.0
        }]

        confidence = np.mean([s.confidence for s in scenarios]) if scenarios else 0.0

        return ModuleResult(
            module_name=self.module_name,
            class_name="PDETMunicipalPlanAnalyzer",
            method_name="generate_counterfactuals",
            status="success",
            data={"scenarios": scenarios},
            evidence=evidence,
            confidence=confidence,
            execution_time=0.0
        )


# ============================================================================
# ADAPTER 8: POLICY SEGMENTER
# ============================================================================

class PolicySegmenterAdapter(BaseAdapter):
    """
    Adapter for DocumentSegmenter from Segmentación Avanzada PDM Colombia
    """

    def __init__(self):
        super().__init__("policy_segmenter")
        self._load_module()

    def _load_module(self):
        try:
            from pdm_segmentation import (
                DocumentSegmenter,
                SegmenterConfig,
                SpanishSentenceSegmenter,
                BayesianBoundaryScorer,
                DPSegmentOptimizer
            )
            self.DocumentSegmenter = DocumentSegmenter
            self.SegmenterConfig = SegmenterConfig
            self.SpanishSentenceSegmenter = SpanishSentenceSegmenter
            self.BayesianBoundaryScorer = BayesianBoundaryScorer
            self.DPSegmentOptimizer = DPSegmentOptimizer
            self.available = True
            self.logger.info(f"✓ {self.module_name} loaded with PDM Segmentation framework")
        except ImportError as e:
            self.logger.warning(f"✗ {self.module_name} NOT available: {e}")

    def execute(self, method_name: str, args: List[Any], kwargs: Dict[str, Any]) -> ModuleResult:
        """
        Supported methods:
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
                result = self._execute_get_segmentation_report(*args, **kwargs)
            else:
                raise ValueError(f"Unknown method: {method_name}")

            result.execution_time = time.time() - start_time
            return result

        except Exception as e:
            self.logger.error(f"{self.module_name}.{method_name} failed: {e}", exc_info=True)
            return self._create_error_result(method_name, start_time, e)

    def _execute_segment(self, text: str, **kwargs) -> ModuleResult:
        """Execute DocumentSegmenter.segment()"""
        config = kwargs.get('config', self.SegmenterConfig())
        segmenter = self.DocumentSegmenter(config)
        segments = segmenter.segment(text)

        evidence = [{
            "type": "document_segments",
            "segment_count": len(segments),
            "avg_segment_length": np.mean([len(s.get("text", "")) for s in segments]) if segments else 0,
            "section_types": list(set(s.get("section_type", "unknown") for s in segments))
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

    def _execute_get_segmentation_report(self, **kwargs) -> ModuleResult:
        """Execute DocumentSegmenter.get_segmentation_report()"""
        segmenter = self.DocumentSegmenter()
        report = segmenter.get_segmentation_report()

        evidence = [{
            "type": "segmentation_report",
            "metrics": report.get("metrics", {}),
            "stats": report.get("stats", {})
        }]

        return ModuleResult(
            module_name=self.module_name,
            class_name="DocumentSegmenter",
            method_name="get_segmentation_report",
            status="success",
            data=report,
            evidence=evidence,
            confidence=0.8,
            execution_time=0.0
        )


# ============================================================================
# ADAPTER 9: SEMANTIC PROCESSOR
# ============================================================================

class SemanticProcessorAdapter(BaseAdapter):
    """
    Adapter for SemanticProcessor from Policy Document Analyzer
    """

    def __init__(self):
        super().__init__("semantic_processor")
        self._load_module()

    def _load_module(self):
        try:
            from policy_document_analyzer import (
                SemanticProcessor,
                SemanticConfig,
                BayesianEvidenceIntegrator,
                PolicyDocumentAnalyzer
            )
            self.SemanticProcessor = SemanticProcessor
            self.SemanticConfig = SemanticConfig
            self.BayesianEvidenceIntegrator = BayesianEvidenceIntegrator
            self.PolicyDocumentAnalyzer = PolicyDocumentAnalyzer
            self.available = True
            self.logger.info(f"✓ {self.module_name} loaded with Semantic Processing framework")
        except ImportError as e:
            self.logger.warning(f"✗ {self.module_name} NOT available: {e}")

    def execute(self, method_name: str, args: List[Any], kwargs: Dict[str, Any]) -> ModuleResult:
        """
        Supported methods:
        - chunk_text(text: str) -> List[Dict]
        - embed_single(text: str) -> np.ndarray
        - detect_pdm_structure(text: str) -> Dict
        """
        start_time = time.time()

        if not self.available:
            return self._create_unavailable_result(method_name, start_time)

        try:
            if method_name == "chunk_text":
                result = self._execute_chunk_text(*args, **kwargs)
            elif method_name == "embed_single":
                result = self._execute_embed_single(*args, **kwargs)
            elif method_name == "detect_pdm_structure":
                result = self._execute_detect_pdm_structure(*args, **kwargs)
            else:
                raise ValueError(f"Unknown method: {method_name}")

            result.execution_time = time.time() - start_time
            return result

        except Exception as e:
            self.logger.error(f"{self.module_name}.{method_name} failed: {e}", exc_info=True)
            return self._create_error_result(method_name, start_time, e)

    def _execute_chunk_text(self, text: str, **kwargs) -> ModuleResult:
        """Execute SemanticProcessor.chunk_text()"""
        config = kwargs.get('config', self.SemanticConfig())
        processor = self.SemanticProcessor(config)
        chunks = processor.chunk_text(text)

        evidence = [{
            "type": "semantic_chunks",
            "chunk_count": len(chunks),
            "avg_chunk_length": np.mean([len(c.get("text", "")) for c in chunks]) if chunks else 0
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

    def _execute_embed_single(self, text: str, **kwargs) -> ModuleResult:
        """Execute SemanticProcessor.embed_single()"""
        config = kwargs.get('config', self.SemanticConfig())
        processor = self.SemanticProcessor(config)
        embedding = processor.embed_single(text)

        evidence = [{
            "type": "text_embedding",
            "embedding_dim": len(embedding) if embedding is not None else 0,
            "text_preview": text[:100] + "..." if len(text) > 100 else text
        }]

        return ModuleResult(
            module_name=self.module_name,
            class_name="SemanticProcessor",
            method_name="embed_single",
            status="success",
            data={"embedding": embedding},
            evidence=evidence,
            confidence=0.9,
            execution_time=0.0
        )

    def _execute_detect_pdm_structure(self, text: str, **kwargs) -> ModuleResult:
        """Execute SemanticProcessor._detect_pdm_structure()"""
        config = kwargs.get('config', self.SemanticConfig())
        processor = self.SemanticProcessor(config)
        structure = processor._detect_pdm_structure(text)

        evidence = [{
            "type": "pdm_structure",
            "sections_found": list(structure.keys()),
            "section_count": len(structure)
        }]

        return ModuleResult(
            module_name=self.module_name,
            class_name="SemanticProcessor",
            method_name="_detect_pdm_structure",
            status="success",
            data={"structure": structure},
            evidence=evidence,
            confidence=0.8,
            execution_time=0.0
        )


# ============================================================================
# ADAPTER 10: BAYESIAN INTEGRATOR
# ============================================================================

class BayesianIntegratorAdapter(BaseAdapter):
    """
    Adapter for BayesianEvidenceIntegrator from Policy Document Analyzer
    """

    def __init__(self):
        super().__init__("bayesian_integrator")
        self._load_module()

    def _load_module(self):
        try:
            from policy_document_analyzer import (
                BayesianEvidenceIntegrator,
                SemanticConfig
            )
            self.BayesianEvidenceIntegrator = BayesianEvidenceIntegrator
            self.SemanticConfig = SemanticConfig
            self.available = True
            self.logger.info(f"✓ {self.module_name} loaded with Bayesian Integration framework")
        except ImportError as e:
            self.logger.warning(f"✗ {self.module_name} NOT available: {e}")

    def execute(self, method_name: str, args: List[Any], kwargs: Dict[str, Any]) -> ModuleResult:
        """
        Supported methods:
        - integrate_evidence(evidence_items: List[Dict]) -> Dict
        - causal_strength(evidence: Dict) -> float
        """
        start_time = time.time()

        if not self.available:
            return self._create_unavailable_result(method_name, start_time)

        try:
            if method_name == "integrate_evidence":
                result = self._execute_integrate_evidence(*args, **kwargs)
            elif method_name == "causal_strength":
                result = self._execute_causal_strength(*args, **kwargs)
            else:
                raise ValueError(f"Unknown method: {method_name}")

            result.execution_time = time.time() - start_time
            return result

        except Exception as e:
            self.logger.error(f"{self.module_name}.{method_name} failed: {e}", exc_info=True)
            return self._create_error_result(method_name, start_time, e)

    def _execute_integrate_evidence(self, evidence_items: List[Dict], **kwargs) -> ModuleResult:
        """Execute BayesianEvidenceIntegrator.integrate_evidence()"""
        config = kwargs.get('config', self.SemanticConfig())
        integrator = self.BayesianEvidenceIntegrator(config)
        result = integrator.integrate_evidence(evidence_items)

        evidence = [{
            "type": "bayesian_integration",
            "evidence_count": len(evidence_items),
            "posterior_confidence": result.get("posterior_confidence", 0.5),
            "information_gain": result.get("information_gain", 0.0)
        }]

        confidence = result.get("posterior_confidence", 0.5)

        return ModuleResult(
            module_name=self.module_name,
            class_name="BayesianEvidenceIntegrator",
            method_name="integrate_evidence",
            status="success",
            data=result,
            evidence=evidence,
            confidence=confidence,
            execution_time=0.0
        )

    def _execute_causal_strength(self, evidence: Dict, **kwargs) -> ModuleResult:
        """Execute BayesianEvidenceIntegrator.causal_strength()"""
        config = kwargs.get('config', self.SemanticConfig())
        integrator = self.BayesianEvidenceIntegrator(config)
        strength = integrator.causal_strength(evidence)

        evidence_result = [{
            "type": "causal_strength",
            "strength": strength,
            "evidence_summary": {
                "semantic_similarity": evidence.get("semantic_similarity", 0.0),
                "reliability_weight": evidence.get("reliability_weight", 0.0)
            }
        }]

        return ModuleResult(
            module_name=self.module_name,
            class_name="BayesianEvidenceIntegrator",
            method_name="causal_strength",
            status="success",
            data={"strength": strength},
            evidence=evidence_result,
            confidence=strength,
            execution_time=0.0
        )


# ============================================================================
# ADAPTER 11: VALIDATION FRAMEWORK
# ============================================================================

class ValidationFrameworkAdapter(BaseAdapter):
    """
    Adapter for IndustrialGradeValidator from Validación Causal y Estocástica
    """

    def __init__(self):
        super().__init__("validation_framework")
        self._load_module()

    def _load_module(self):
        try:
            from validacion_causal import (
                IndustrialGradeValidator,
                TeoriaCambio,
                AdvancedDAGValidator,
                CategoriaCausal,
                GraphType
            )
            self.IndustrialGradeValidator = IndustrialGradeValidator
            self.TeoriaCambio = TeoriaCambio
            self.AdvancedDAGValidator = AdvancedDAGValidator
            self.CategoriaCausal = CategoriaCausal
            self.GraphType = GraphType
            self.available = True
            self.logger.info(f"✓ {self.module_name} loaded with Validation framework")
        except ImportError as e:
            self.logger.warning(f"✗ {self.module_name} NOT available: {e}")

    def execute(self, method_name: str, args: List[Any], kwargs: Dict[str, Any]) -> ModuleResult:
        """
        Supported methods:
        - validate_causal_model(graph: Dict) -> Dict
        - check_acyclicity(graph: Dict) -> Dict
        - perform_sensitivity_analysis(graph: Dict, parameters: Dict) -> Dict
        """
        start_time = time.time()

        if not self.available:
            return self._create_unavailable_result(method_name, start_time)

        try:
            if method_name == "validate_causal_model":
                result = self._execute_validate_causal_model(*args, **kwargs)
            elif method_name == "check_acyclicity":
                result = self._execute_check_acyclicity(*args, **kwargs)
            elif method_name == "perform_sensitivity_analysis":
                result = self._execute_perform_sensitivity_analysis(*args, **kwargs)
            else:
                raise ValueError(f"Unknown method: {method_name}")

            result.execution_time = time.time() - start_time
            return result

        except Exception as e:
            self.logger.error(f"{self.module_name}.{method_name} failed: {e}", exc_info=True)
            return self._create_error_result(method_name, start_time, e)

    def _execute_validate_causal_model(self, graph: Dict, **kwargs) -> ModuleResult:
        """Execute IndustrialGradeValidator.execute_suite()"""
        validator = self.IndustrialGradeValidator()
        result = validator.execute_suite(graph)

        evidence = [{
            "type": "causal_model_validation",
            "validation_passed": result.get("validation_passed", False),
            "acyclicity_pvalue": result.get("acyclicity_pvalue", 0.0),
            "sensitivity_metrics": result.get("sensitivity_metrics", {})
        }]

        confidence = result.get("overall_confidence", 0.5)

        return ModuleResult(
            module_name=self.module_name,
            class_name="IndustrialGradeValidator",
            method_name="validate_causal_model",
            status="success",
            data=result,
            evidence=evidence,
            confidence=confidence,
            execution_time=0.0
        )

    def _execute_check_acyclicity(self, graph: Dict, **kwargs) -> ModuleResult:
        """Execute AdvancedDAGValidator.calculate_acyclicity_pvalue()"""
        validator = self.AdvancedDAGValidator()
        result = validator.calculate_acyclicity_pvalue(graph)

        evidence = [{
            "type": "acyclicity_check",
            "is_acyclic": result.get("is_acyclic", False),
            "pvalue": result.get("pvalue", 0.0),
            "confidence_interval": result.get("confidence_interval", [0.0, 0.0])
        }]

        confidence = 1.0 - result.get("pvalue", 0.5)

        return ModuleResult(
            module_name=self.module_name,
            class_name="AdvancedDAGValidator",
            method_name="check_acyclicity",
            status="success",
            data=result,
            evidence=evidence,
            confidence=confidence,
            execution_time=0.0
        )

    def _execute_perform_sensitivity_analysis(self, graph: Dict, parameters: Dict, **kwargs) -> ModuleResult:
        """Execute AdvancedDAGValidator._perform_sensitivity_analysis_internal()"""
        validator = self.AdvancedDAGValidator()
        result = validator._perform_sensitivity_analysis_internal(graph, parameters)

        evidence = [{
            "type": "sensitivity_analysis",
            "robustness_value": result.get("robustness_value", 0.0),
            "e_value": result.get("e_value", 0.0),
            "critical_parameters": result.get("critical_parameters", [])
        }]

        confidence = result.get("robustness_value", 0.5)

        return ModuleResult(
            module_name=self.module_name,
            class_name="AdvancedDAGValidator",
            method_name="perform_sensitivity_analysis",
            status="success",
            data=result,
            evidence=evidence,
            confidence=confidence,
            execution_time=0.0
        )


# ============================================================================
# ADAPTER 12: MUNICIPAL ANALYZER
# ============================================================================

class MunicipalAnalyzerAdapter(BaseAdapter):
    """
    Adapter for MunicipalAnalyzer from Advanced Municipal Plan Analyzer
    """

    def __init__(self):
        super().__init__("municipal_analyzer")
        self._load_module()

    def _load_module(self):
        try:
            from analyzer_one import (
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
            self.logger.info(f"✓ {self.module_name} loaded with Municipal Analyzer framework")
        except ImportError as e:
            self.logger.warning(f"✗ {self.module_name} NOT available: {e}")

    def execute(self, method_name: str, args: List[Any], kwargs: Dict[str, Any]) -> ModuleResult:
        """
        Supported methods:
        - analyze_document(text: str) -> Dict
        - extract_semantic_cube(text: str) -> Dict
        - diagnose_critical_links(value_chain: Dict) -> List[Dict]
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
            else:
                raise ValueError(f"Unknown method: {method_name}")

            result.execution_time = time.time() - start_time
            return result

        except Exception as e:
            self.logger.error(f"{self.module_name}.{method_name} failed: {e}", exc_info=True)
            return self._create_error_result(method_name, start_time, e)

    def _execute_analyze_document(self, text: str, **kwargs) -> ModuleResult:
        """Execute MunicipalAnalyzer.analyze_document()"""
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
        """Execute SemanticAnalyzer.extract_semantic_cube()"""
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
        """Execute PerformanceAnalyzer.diagnose_critical_links()"""
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


# ============================================================================
# ADAPTER 13: PDET ANALYZER
# ============================================================================

class PDETAnalyzerAdapter(BaseAdapter):
    """
    Adapter for PDETMunicipalPlanAnalyzer from PDET Causal Analysis
    """

    def __init__(self):
        super().__init__("pdet_analyzer")
        self._load_module()

    def _load_module(self):
        try:
            from pdet_causal_analysis import (
                PDETMunicipalPlanAnalyzer,
                CausalNode,
                CausalEdge,
                CausalDAG,
                CausalEffect
            )
            self.PDETMunicipalPlanAnalyzer = PDETMunicipalPlanAnalyzer
            self.CausalNode = CausalNode
            self.CausalEdge = CausalEdge
            self.CausalDAG = CausalDAG
            self.CausalEffect = CausalEffect
            self.available = True
            self.logger.info(f"✓ {self.module_name} loaded with PDET Analysis framework")
        except ImportError as e:
            self.logger.warning(f"✗ {self.module_name} NOT available: {e}")

    def execute(self, method_name: str, args: List[Any], kwargs: Dict[str, Any]) -> ModuleResult:
        """
        Supported methods:
        - analyze_municipal_plan(text: str) -> Dict
        - extract_tables(document_path: str) -> List[pd.DataFrame]
        - analyze_financial_feasibility(text: str) -> Dict
        """
        start_time = time.time()

        if not self.available:
            return self._create_unavailable_result(method_name, start_time)

        try:
            if method_name == "analyze_municipal_plan":
                result = self._execute_analyze_municipal_plan(*args, **kwargs)
            elif method_name == "extract_tables":
                result = self._execute_extract_tables(*args, **kwargs)
            elif method_name == "analyze_financial_feasibility":
                result = self._execute_analyze_financial_feasibility(*args, **kwargs)
            else:
                raise ValueError(f"Unknown method: {method_name}")

            result.execution_time = time.time() - start_time
            return result

        except Exception as e:
            self.logger.error(f"{self.module_name}.{method_name} failed: {e}", exc_info=True)
            return self._create_error_result(method_name, start_time, e)

    def _execute_analyze_municipal_plan(self, text: str, **kwargs) -> ModuleResult:
        """Execute PDETMunicipalPlanAnalyzer.analyze_municipal_plan()"""
        analyzer = self.PDETMunicipalPlanAnalyzer()
        result = analyzer.analyze_municipal_plan(text)

        evidence = []

        if "causal_dag" in result:
            dag = result["causal_dag"]
            evidence.append({
                "type": "causal_dag",
                "node_count": len(dag.nodes) if hasattr(dag, 'nodes') else 0,
                "edge_count": len(dag.edges) if hasattr(dag, 'edges') else 0
            })

        if "financial_analysis" in result:
            financial = result["financial_analysis"]
            evidence.append({
                "type": "financial_analysis",
                "viability_score": financial.get("viability_score", 0.5),
                "sustainability_score": financial.get("sustainability_score", 0.5)
            })

        if "responsible_entities" in result:
            entities = result["responsible_entities"]
            evidence.append({
                "type": "responsible_entities",
                "entity_count": len(entities),
                "high_specificity_count": len([e for e in entities if e.get("specificity_score", 0) > 0.7])
            })

        confidence = result.get("quality_score", {}).get("overall_score", 0.5)

        return ModuleResult(
            module_name=self.module_name,
            class_name="PDETMunicipalPlanAnalyzer",
            method_name="analyze_municipal_plan",
            status="success",
            data=result,
            evidence=evidence,
            confidence=confidence,
            execution_time=0.0,
            metadata={"evidence_types": len(evidence)}
        )

    def _execute_extract_tables(self, document_path: str, **kwargs) -> ModuleResult:
        """Execute PDETMunicipalPlanAnalyzer.extract_tables()"""
        analyzer = self.PDETMunicipalPlanAnalyzer()
        tables = analyzer.extract_tables(document_path)

        evidence = [{
            "type": "extracted_tables",
            "table_count": len(tables),
            "sample_shapes": [table.shape for table in tables[:3]] if tables else []
        }]

        return ModuleResult(
            module_name=self.module_name,
            class_name="PDETMunicipalPlanAnalyzer",
            method_name="extract_tables",
            status="success",
            data={"tables": tables},
            evidence=evidence,
            confidence=0.8,
            execution_time=0.0
        )

    def _execute_analyze_financial_feasibility(self, text: str, **kwargs) -> ModuleResult:
        """Execute PDETMunicipalPlanAnalyzer.analyze_financial_feasibility()"""
        analyzer = self.PDETMunicipalPlanAnalyzer()
        result = analyzer.analyze_financial_feasibility(text)

        evidence = [{
            "type": "financial_feasibility",
            "viability_score": result.get("viability_score", 0.5),
            "risk_level": result.get("risk_level", "unknown"),
            "funding_sources": result.get("funding_sources", [])
        }]

        confidence = result.get("viability_score", 0.5)

        return ModuleResult(
            module_name=self.module_name,
            class_name="PDETMunicipalPlanAnalyzer",
            method_name="analyze_financial_feasibility",
            status="success",
            data=result,
            evidence=evidence,
            confidence=confidence,
            execution_time=0.0
        )


# ============================================================================
# ADAPTER 14: DECOLOGO PROCESSOR
# ============================================================================

class DecologoProcessorAdapter(BaseAdapter):
    """
    Adapter for IndustrialPolicyProcessor from Procesador Causal Industrial DECALOGO
    """

    def __init__(self):
        super().__init__("decologo_processor")
        self._load_module()

    def _load_module(self):
        try:
            from decologo_processor import (
                IndustrialPolicyProcessor,
                ProcessorConfig,
                BayesianEvidenceScorer,
                PolicyTextProcessor,
                EvidenceBundle
            )
            self.IndustrialPolicyProcessor = IndustrialPolicyProcessor
            self.ProcessorConfig = ProcessorConfig
            self.BayesianEvidenceScorer = BayesianEvidenceScorer
            self.PolicyTextProcessor = PolicyTextProcessor
            self.EvidenceBundle = EvidenceBundle
            self.available = True
            self.logger.info(f"✓ {self.module_name} loaded with DECALOGO framework")
        except ImportError as e:
            self.logger.warning(f"✗ {self.module_name} NOT available: {e}")

    def execute(self, method_name: str, args: List[Any], kwargs: Dict[str, Any]) -> ModuleResult:
        """
        Supported methods:
        - process(text: str) -> Dict
        - _extract_point_evidence(text: str, dimension: str) -> List[str]
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
            elif method_name == "score_evidence":
                result = self._execute_score_evidence(*args, **kwargs)
            else:
                raise ValueError(f"Unknown method: {method_name}")

            result.execution_time = time.time() - start_time
            return result

        except Exception as e:
            self.logger.error(f"{self.module_name}.{method_name} failed: {e}", exc_info=True)
            return self._create_error_result(method_name, start_time, e)

    def _execute_process(self, text: str, **kwargs) -> ModuleResult:
        """Execute IndustrialPolicyProcessor.process()"""
        config = kwargs.get('config', self.ProcessorConfig())
        processor = self.IndustrialPolicyProcessor(config)
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
        """Execute IndustrialPolicyProcessor._extract_point_evidence()"""
        config = kwargs.get('config', self.ProcessorConfig())
        processor = self.IndustrialPolicyProcessor(config)
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

    def _execute_score_evidence(self, bundle, **kwargs) -> ModuleResult:
        """Execute BayesianEvidenceScorer.score_evidence()"""
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


# ============================================================================
# ADAPTER 15: EMBEDDING ANALYZER
# ============================================================================

class EmbeddingAnalyzerAdapter(BaseAdapter):
    """
    Adapter for PolicyAnalysisEmbedder from Sistema de Incrustación Semántica
    """

    def __init__(self):
        super().__init__("embedding_analyzer")
        self._load_module()

    def _load_module(self):
        try:
            from embedding_policy import (
                PolicyAnalysisEmbedder,
                AdvancedSemanticChunker,
                BayesianNumericalAnalyzer,
                PolicyCrossEncoderReranker,
                PolicyEmbeddingConfig,
                ChunkingConfig
            )
            self.PolicyAnalysisEmbedder = PolicyAnalysisEmbedder
            self.AdvancedSemanticChunker = AdvancedSemanticChunker
            self.BayesianNumericalAnalyzer = BayesianNumericalAnalyzer
            self.PolicyCrossEncoderReranker = PolicyCrossEncoderReranker
            self.PolicyEmbeddingConfig = PolicyEmbeddingConfig
            self.ChunkingConfig = ChunkingConfig
            self.available = True
            self.logger.info(f"✓ {self.module_name} loaded with Embedding Analysis framework")
        except ImportError as e:
            self.logger.warning(f"✗ {self.module_name} NOT available: {e}")

    def execute(self, method_name: str, args: List[Any], kwargs: Dict[str, Any]) -> ModuleResult:
        """
        Supported methods:
        - process_document(text: str, document_metadata: dict) -> List[SemanticChunk]
        - semantic_search(query: str, document_chunks: List[SemanticChunk]) -> List[Tuple[SemanticChunk, float]]
        - evaluate_policy_numerical_consistency(chunks: List[SemanticChunk], pdq_context: PDQIdentifier) -> BayesianEvaluation
        """
        start_time = time.time()

        if not self.available:
            return self._create_unavailable_result(method_name, start_time)

        try:
            if method_name == "process_document":
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
            self.logger.error(f"{self.module_name}.{method_name} failed: {e}", exc_info=True)
            return self._create_error_result(method_name, start_time, e)

    def _execute_process_document(self, text: str, document_metadata: Dict[str, Any], **kwargs) -> ModuleResult:
        """Execute PolicyAnalysisEmbedder.process_document()"""
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

    def _execute_semantic_search(self, query: str, document_chunks: List, **kwargs) -> ModuleResult:
        """Execute PolicyAnalysisEmbedder.semantic_search()"""
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

    def _execute_evaluate_policy_numerical_consistency(self, chunks: List, pdq_context, **kwargs) -> ModuleResult:
        """Execute PolicyAnalysisEmbedder.evaluate_policy_numerical_consistency()"""
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


# ============================================================================
# ADAPTER 16: CAUSAL VALIDATOR
# ============================================================================

class CausalValidatorAdapter(BaseAdapter):
    """
    Adapter for IndustrialGradeValidator from Validación Causal y Estocástica
    """

    def __init__(self):
        super().__init__("causal_validator")
        self._load_module()

    def _load_module(self):
        try:
            from validacion_causal import (
                IndustrialGradeValidator,
                TeoriaCambio,
                AdvancedDAGValidator,
                CategoriaCausal
            )
            self.IndustrialGradeValidator = IndustrialGradeValidator
            self.TeoriaCambio = TeoriaCambio
            self.AdvancedDAGValidator = AdvancedDAGValidator
            self.CategoriaCausal = CategoriaCausal
            self.available = True
            self.logger.info(f"✓ {self.module_name} loaded with Causal Validation framework")
        except ImportError as e:
            self.logger.warning(f"✗ {self.module_name} NOT available: {e}")

    def execute(self, method_name: str, args: List[Any], kwargs: Dict[str, Any]) -> ModuleResult:
        """
        Supported methods:
        - validate_causal_model(graph: Dict) -> Dict
        - check_acyclicity(graph: Dict) -> Dict
        - perform_sensitivity_analysis(graph: Dict, parameters: Dict) -> Dict
        """
        start_time = time.time()

        if not self.available:
            return self._create_unavailable_result(method_name, start_time)

        try:
            if method_name == "validate_causal_model":
                result = self._execute_validate_causal_model(*args, **kwargs)
            elif method_name == "check_acyclicity":
                result = self._execute_check_acyclicity(*args, **kwargs)
            elif method_name == "perform_sensitivity_analysis":
                result = self._execute_perform_sensitivity_analysis(*args, **kwargs)
            else:
                raise ValueError(f"Unknown method: {method_name}")

            result.execution_time = time.time() - start_time
            return result

        except Exception as e:
            self.logger.error(f"{self.module_name}.{method_name} failed: {e}", exc_info=True)
            return self._create_error_result(method_name, start_time, e)

    def _execute_validate_causal_model(self, graph: Dict, **kwargs) -> ModuleResult:
        """Execute IndustrialGradeValidator.execute_suite()"""
        validator = self.IndustrialGradeValidator()
        result = validator.execute_suite(graph)

        evidence = [{
            "type": "causal_model_validation",
            "validation_passed": result.get("validation_passed", False),
            "acyclicity_pvalue": result.get("acyclicity_pvalue", 0.0),
            "sensitivity_metrics": result.get("sensitivity_metrics", {})
        }]

        confidence = result.get("overall_confidence", 0.5)

        return ModuleResult(
            module_name=self.module_name,
            class_name="IndustrialGradeValidator",
            method_name="validate_causal_model",
            status="success",
            data=result,
            evidence=evidence,
            confidence=confidence,
            execution_time=0.0
        )

    def _execute_check_acyclicity(self, graph: Dict, **kwargs) -> ModuleResult:
        """Execute AdvancedDAGValidator.calculate_acyclicity_pvalue()"""
        validator = self.AdvancedDAGValidator()
        result = validator.calculate_acyclicity_pvalue(graph)

        evidence = [{
            "type": "acyclicity_check",
            "is_acyclic": result.get("is_acyclic", False),
            "pvalue": result.get("pvalue", 0.0),
            "confidence_interval": result.get("confidence_interval", [0.0, 0.0])
        }]

        confidence = 1.0 - result.get("pvalue", 0.5)

        return ModuleResult(
            module_name=self.module_name,
            class_name="AdvancedDAGValidator",
            method_name="check_acyclicity",
            status="success",
            data=result,
            evidence=evidence,
            confidence=confidence,
            execution_time=0.0
        )

    def _execute_perform_sensitivity_analysis(self, graph: Dict, parameters: Dict, **kwargs) -> ModuleResult:
        """Execute AdvancedDAGValidator._perform_sensitivity_analysis_internal()"""
        validator = self.AdvancedDAGValidator()
        result = validator._perform_sensitivity_analysis_internal(graph, parameters)

        evidence = [{
            "type": "sensitivity_analysis",
            "robustness_value": result.get("robustness_value", 0.0),
            "e_value": result.get("e_value", 0.0),
            "critical_parameters": result.get("critical_parameters", [])
        }]

        confidence = result.get("robustness_value", 0.5)

        return ModuleResult(
            module_name=self.module_name,
            class_name="AdvancedDAGValidator",
            method_name="perform_sensitivity_analysis",
            status="success",
            data=result,
            evidence=evidence,
            confidence=confidence,
            execution_time=0.0
        )