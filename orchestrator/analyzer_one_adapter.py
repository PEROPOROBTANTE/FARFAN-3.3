"""
Analyzer One Adapter - Municipal Development Plan Analysis
===========================================================

Wraps Analyzer_one.py functionality with standardized interfaces for the module controller.
Preserves all original class signatures while providing alias methods for controller integration.

Author: FARFAN 3.0 Integration Team
Version: 1.0.0
Python: 3.10+
"""

import logging
import time
import warnings
from typing import Dict, List, Any, Optional, Tuple
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


class AnalyzerOneAdapter:
    """
    Adapter for Analyzer_one.py - Municipal Development Plan Analysis System.
    
    Responsibility Map (cuestionario.json):
    - D1 (Insumos): Q1 (Baseline identification), Q3 (Resource assessment)
    - D2 (Actividades): Q11 (Activity structure), Q13 (Responsibility assignment)
    - D4 (Resultados): Q20 (Result metrics), Q21 (Indicator quality)
    - D6 (Causalidad): Q29 (Causal chains), Q30 (Theory of change validation)
    
    Original Classes:
    - MunicipalOntology: Ontological structure for municipal analysis
    - SemanticCubeExtractor: Extract semantic cube from documents
    - PerformanceAnalyzer: Analyze performance metrics
    - CriticalLinksAnalyzer: Diagnose critical value chain links
    - MunicipalAnalyzer: Main analysis orchestrator
    """

    def __init__(self, ontology: Optional[Any] = None):
        """
        Initialize adapter with optional ontology dependency.
        
        Args:
            ontology: MunicipalOntology instance (injected dependency)
        """
        self.module_name = "analyzer_one"
        self.available = False
        self.logger = logging.getLogger(f"{__name__}.AnalyzerOne")
        self._ontology = ontology
        self._load_module()

    def _load_module(self):
        """Load Analyzer_one module and its components"""
        try:
            from Analyzer_one import (
                MunicipalOntology,
                SemanticCubeExtractor,
                PerformanceAnalyzer,
                CriticalLinksAnalyzer,
                MunicipalAnalyzer,
            )
            
            self.MunicipalOntology = MunicipalOntology
            self.SemanticCubeExtractor = SemanticCubeExtractor
            self.PerformanceAnalyzer = PerformanceAnalyzer
            self.CriticalLinksAnalyzer = CriticalLinksAnalyzer
            self.MunicipalAnalyzer = MunicipalAnalyzer
            
            if self._ontology is None:
                self._ontology = MunicipalOntology()
            
            self.available = True
            self.logger.info(f"✓ {self.module_name} loaded successfully")
            
        except ImportError as e:
            self.logger.warning(f"✗ {self.module_name} NOT available: {e}")
            self.available = False

    # ========================================================================
    # ORIGINAL METHOD SIGNATURES (Preserved)
    # ========================================================================

    def extract_semantic_cube(self, document_segments: List[str]) -> Dict[str, Any]:
        """
        Original method: SemanticCubeExtractor.extract_semantic_cube()
        Maps to cuestionario.json: D1.Q1, D6.Q29
        """
        if not self.available:
            return {"error": "Module not available"}
        
        start_time = time.time()
        try:
            extractor = self.SemanticCubeExtractor(self._ontology)
            result = extractor.extract_semantic_cube(document_segments)
            result["_execution_time"] = time.time() - start_time
            return result
        except Exception as e:
            self.logger.error(f"extract_semantic_cube failed: {e}", exc_info=True)
            return {"error": str(e), "_execution_time": time.time() - start_time}

    def analyze_performance(self, semantic_cube: Dict[str, Any]) -> Dict[str, Any]:
        """
        Original method: PerformanceAnalyzer.analyze_performance()
        Maps to cuestionario.json: D2.Q11, D4.Q20
        """
        if not self.available:
            return {"error": "Module not available"}
        
        start_time = time.time()
        try:
            analyzer = self.PerformanceAnalyzer(self._ontology)
            result = analyzer.analyze_performance(semantic_cube)
            result["_execution_time"] = time.time() - start_time
            return result
        except Exception as e:
            self.logger.error(f"analyze_performance failed: {e}", exc_info=True)
            return {"error": str(e), "_execution_time": time.time() - start_time}

    def diagnose_critical_links(
        self,
        semantic_cube: Dict[str, Any],
        performance_analysis: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Original method: CriticalLinksAnalyzer.diagnose_critical_links()
        Maps to cuestionario.json: D6.Q30
        """
        if not self.available:
            return {"error": "Module not available"}
        
        start_time = time.time()
        try:
            analyzer = self.CriticalLinksAnalyzer(self._ontology)
            result = analyzer.diagnose_critical_links(semantic_cube, performance_analysis)
            result["_execution_time"] = time.time() - start_time
            return result
        except Exception as e:
            self.logger.error(f"diagnose_critical_links failed: {e}", exc_info=True)
            return {"error": str(e), "_execution_time": time.time() - start_time}

    def analyze_document(self, document_path: str) -> Dict[str, Any]:
        """
        Original method: MunicipalAnalyzer.analyze_document()
        Full pipeline analysis - Maps to multiple cuestionario.json dimensions
        """
        if not self.available:
            return {"error": "Module not available"}
        
        start_time = time.time()
        try:
            analyzer = self.MunicipalAnalyzer()
            result = analyzer.analyze_document(document_path)
            result["_execution_time"] = time.time() - start_time
            return result
        except Exception as e:
            self.logger.error(f"analyze_document failed: {e}", exc_info=True)
            return {"error": str(e), "_execution_time": time.time() - start_time}

    # ========================================================================
    # STANDARDIZED CONTROLLER INTERFACE (Alias Methods)
    # ========================================================================

    def analyze_baseline_data(self, text: str, segments: List[str]) -> AdapterResult:
        """
        Controller method for D1.Q1: Baseline identification
        Alias for: extract_semantic_cube + baseline analysis
        """
        start_time = time.time()
        
        try:
            semantic_cube = self.extract_semantic_cube(segments)
            
            baseline_data = {
                "value_chain_distribution": semantic_cube.get("value_chain_distribution", {}),
                "domain_coverage": semantic_cube.get("policy_domains", {}),
                "segment_count": semantic_cube.get("total_segments", 0),
                "baseline_indicators": self._extract_baseline_indicators(semantic_cube),
            }
            
            return AdapterResult(
                module_name=self.module_name,
                class_name="SemanticCubeExtractor",
                method_name="analyze_baseline_data",
                status="success",
                data=baseline_data,
                evidence=[{"type": "semantic_cube", "segments": len(segments)}],
                confidence=semantic_cube.get("semantic_complexity", 0.7),
                execution_time=time.time() - start_time,
            )
        except Exception as e:
            return self._error_result("analyze_baseline_data", start_time, e)

    def analyze_activity_structure(self, semantic_cube: Dict[str, Any]) -> AdapterResult:
        """
        Controller method for D2.Q11: Activity structure analysis
        Alias for: analyze_performance (activity focus)
        """
        start_time = time.time()
        
        try:
            performance = self.analyze_performance(semantic_cube)
            
            activity_data = {
                "activities": performance.get("value_chain_performance", {}).get("Actividades", {}),
                "bottlenecks": self._filter_by_link(performance.get("bottlenecks", []), "Actividades"),
                "throughput": self._filter_by_link(performance.get("throughput_metrics", {}), "Actividades"),
            }
            
            return AdapterResult(
                module_name=self.module_name,
                class_name="PerformanceAnalyzer",
                method_name="analyze_activity_structure",
                status="success",
                data=activity_data,
                evidence=[{"type": "performance_analysis", "link": "Actividades"}],
                confidence=0.85,
                execution_time=time.time() - start_time,
            )
        except Exception as e:
            return self._error_result("analyze_activity_structure", start_time, e)

    def analyze_result_metrics(self, semantic_cube: Dict[str, Any]) -> AdapterResult:
        """
        Controller method for D4.Q20: Result metrics analysis
        Alias for: analyze_performance (results focus)
        """
        start_time = time.time()
        
        try:
            performance = self.analyze_performance(semantic_cube)
            
            result_data = {
                "results": performance.get("value_chain_performance", {}).get("Resultados", {}),
                "indicators": self._extract_result_indicators(performance),
                "bottlenecks": self._filter_by_link(performance.get("bottlenecks", []), "Resultados"),
            }
            
            return AdapterResult(
                module_name=self.module_name,
                class_name="PerformanceAnalyzer",
                method_name="analyze_result_metrics",
                status="success",
                data=result_data,
                evidence=[{"type": "performance_analysis", "link": "Resultados"}],
                confidence=0.82,
                execution_time=time.time() - start_time,
            )
        except Exception as e:
            return self._error_result("analyze_result_metrics", start_time, e)

    def validate_causal_chains(
        self,
        semantic_cube: Dict[str, Any],
        performance_analysis: Dict[str, Any],
    ) -> AdapterResult:
        """
        Controller method for D6.Q29-Q30: Causal chain validation
        Alias for: diagnose_critical_links
        """
        start_time = time.time()
        
        try:
            diagnosis = self.diagnose_critical_links(semantic_cube, performance_analysis)
            
            causal_data = {
                "critical_links": diagnosis.get("critical_links", {}),
                "interventions": diagnosis.get("recommended_interventions", []),
                "causal_coherence": self._calculate_causal_coherence(diagnosis),
            }
            
            return AdapterResult(
                module_name=self.module_name,
                class_name="CriticalLinksAnalyzer",
                method_name="validate_causal_chains",
                status="success",
                data=causal_data,
                evidence=[{"type": "critical_links_diagnosis"}],
                confidence=0.80,
                execution_time=time.time() - start_time,
            )
        except Exception as e:
            return self._error_result("validate_causal_chains", start_time, e)

    # ========================================================================
    # DEPRECATED SHIM METHODS (With Warnings)
    # ========================================================================

    def get_baseline_analysis(self, text: str) -> Dict[str, Any]:
        """
        DEPRECATED: Use analyze_baseline_data() instead.
        Legacy method from earlier refactoring iteration.
        """
        warnings.warn(
            "get_baseline_analysis() is deprecated. Use analyze_baseline_data() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Route to new method
        result = self.analyze_baseline_data(text, text.split('\n'))
        return result.data

    def extract_activity_structure(self, document: str) -> Dict[str, Any]:
        """
        DEPRECATED: Use analyze_activity_structure() instead.
        Routes to standardized controller method.
        """
        warnings.warn(
            "extract_activity_structure() is deprecated. Use analyze_activity_structure() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        analyzer = self.MunicipalAnalyzer()
        full_result = analyzer.analyze_document(document)
        result = self.analyze_activity_structure(full_result.get("semantic_cube", {}))
        return result.data

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _extract_baseline_indicators(self, semantic_cube: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract baseline indicators from semantic cube"""
        indicators = []
        for segment in semantic_cube.get("segments", []):
            if segment.get("value_chain_link") == "Insumos":
                indicators.append({
                    "text": segment.get("text", "")[:200],
                    "complexity": segment.get("complexity", 0),
                    "domain": segment.get("policy_domain", "unknown"),
                })
        return indicators

    def _extract_result_indicators(self, performance: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract result indicators from performance analysis"""
        indicators = []
        results = performance.get("value_chain_performance", {}).get("Resultados", {})
        if results:
            indicators.append({
                "coverage": results.get("coverage", 0),
                "bottleneck_score": results.get("bottleneck_score", 0),
                "loss": results.get("loss", 0),
            })
        return indicators

    def _filter_by_link(self, items: Any, link_name: str) -> Any:
        """Filter items by value chain link"""
        if isinstance(items, dict):
            return items.get(link_name, {})
        elif isinstance(items, list):
            return [item for item in items if item.get("link") == link_name]
        return []

    def _calculate_causal_coherence(self, diagnosis: Dict[str, Any]) -> float:
        """Calculate causal coherence score"""
        links = diagnosis.get("critical_links", {})
        if not links:
            return 0.0
        scores = [v for v in links.values() if isinstance(v, (int, float))]
        return sum(scores) / len(scores) if scores else 0.0

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


def create_analyzer_one_adapter(ontology: Optional[Any] = None) -> AnalyzerOneAdapter:
    """Factory function to create AnalyzerOneAdapter instance"""
    return AnalyzerOneAdapter(ontology=ontology)
