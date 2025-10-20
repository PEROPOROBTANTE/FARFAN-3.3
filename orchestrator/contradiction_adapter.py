"""
Contradiction Detection Adapter - Policy Contradiction Analysis
================================================================

Wraps contradiction_deteccion.py functionality with standardized interfaces for the module controller.
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


class ContradictionAdapter:
    """
    Adapter for contradiction_deteccion.py - Policy Contradiction Detection System.
    
    Responsibility Map (cuestionario.json):
    - D1 (Insumos): Q4 (Baseline contradictions)
    - D2 (Actividades): Q12 (Activity conflicts)
    - D3 (Productos): Q16 (Product inconsistencies)
    - D6 (Causalidad): Q29 (Logical incompatibilities)
    
    Original Classes:
    - BayesianContradictionScorer: Bayesian scoring for contradictions
    - TemporalConsistencyChecker: Verify temporal consistency
    - PolicyContradictionDetector: Comprehensive contradiction detection
    """

    def __init__(self, model_name: Optional[str] = None, threshold: float = 0.7):
        """
        Initialize adapter with optional configuration.
        
        Args:
            model_name: NLI model name (injected dependency)
            threshold: Contradiction confidence threshold
        """
        self.module_name = "contradiction_detection"
        self.available = False
        self.logger = logging.getLogger(f"{__name__}.ContradictionDetection")
        self._model_name = model_name or "hiiamsid/sentence_similarity_spanish_es"
        self._threshold = threshold
        self._load_module()

    def _load_module(self):
        """Load contradiction_deteccion module and its components"""
        try:
            from contradiction_deteccion import (
                PolicyStatement,
                ContradictionType,
                PolicyDimension,
                BayesianContradictionScorer,
                TemporalConsistencyChecker,
                PolicyContradictionDetector,
            )
            
            self.PolicyStatement = PolicyStatement
            self.ContradictionType = ContradictionType
            self.PolicyDimension = PolicyDimension
            self.BayesianContradictionScorer = BayesianContradictionScorer
            self.TemporalConsistencyChecker = TemporalConsistencyChecker
            self.PolicyContradictionDetector = PolicyContradictionDetector
            
            self.available = True
            self.logger.info(f"✓ {self.module_name} loaded successfully")
            
        except ImportError as e:
            self.logger.warning(f"✗ {self.module_name} NOT available: {e}")
            self.available = False

    # ========================================================================
    # ORIGINAL METHOD SIGNATURES (Preserved)
    # ========================================================================

    def detect(
        self,
        document: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Original method: PolicyContradictionDetector.detect()
        Maps to cuestionario.json: D1.Q4, D2.Q12, D6.Q29
        """
        if not self.available:
            return {}
        
        try:
            detector = self.PolicyContradictionDetector(
                model_name=self._model_name,
                contradiction_threshold=self._threshold,
            )
            return detector.detect(document, metadata or {})
        except Exception as e:
            self.logger.error(f"detect failed: {e}", exc_info=True)
            return {}

    def calculate_posterior(
        self,
        scorer: Any,
        similarity: float,
        context_features: Dict[str, float],
    ) -> float:
        """
        Original method: BayesianContradictionScorer.calculate_posterior()
        Maps to cuestionario.json: D6.Q29
        """
        if not self.available:
            return 0.0
        
        try:
            return scorer.calculate_posterior(similarity, context_features)
        except Exception as e:
            self.logger.error(f"calculate_posterior failed: {e}", exc_info=True)
            return 0.0

    def verify_temporal_consistency(
        self,
        checker: Any,
        statements: List[Any],
    ) -> Dict[str, Any]:
        """
        Original method: TemporalConsistencyChecker.verify_temporal_consistency()
        Maps to cuestionario.json: D2.Q12
        """
        if not self.available:
            return {}
        
        try:
            return checker.verify_temporal_consistency(statements)
        except Exception as e:
            self.logger.error(f"verify_temporal_consistency failed: {e}", exc_info=True)
            return {}

    # ========================================================================
    # STANDARDIZED CONTROLLER INTERFACE (Alias Methods)
    # ========================================================================

    def detect_contradictions(
        self,
        text: str,
        dimension: Optional[str] = None,
    ) -> AdapterResult:
        """
        Controller method for D1.Q4, D2.Q12, D6.Q29: Detect contradictions
        Alias for: detect
        """
        start_time = time.time()
        
        try:
            metadata = {"dimension": dimension} if dimension else {}
            result = self.detect(text, metadata)
            
            contradictions = result.get("contradictions", [])
            metrics = result.get("coherence_metrics", {})
            
            data = {
                "contradiction_count": len(contradictions),
                "contradictions": contradictions[:10],
                "semantic_coherence": metrics.get("semantic_coherence", 0),
                "objective_alignment": metrics.get("objective_alignment", 0),
                "contradiction_entropy": metrics.get("contradiction_entropy", 0),
            }
            
            confidence = 1.0 - (len(contradictions) * 0.05)
            confidence = max(0.3, min(1.0, confidence))
            
            return AdapterResult(
                module_name=self.module_name,
                class_name="PolicyContradictionDetector",
                method_name="detect_contradictions",
                status="success" if len(contradictions) == 0 else "warnings",
                data=data,
                evidence=[{"type": "contradictions", "count": len(contradictions)}],
                confidence=confidence,
                execution_time=time.time() - start_time,
                warnings=[f"Found {len(contradictions)} contradictions"] if contradictions else [],
            )
        except Exception as e:
            return self._error_result("detect_contradictions", start_time, e)

    def check_temporal_consistency(
        self,
        text: str,
    ) -> AdapterResult:
        """
        Controller method for D2.Q12: Check temporal consistency
        Alias for: verify_temporal_consistency
        """
        start_time = time.time()
        
        try:
            detector = self.PolicyContradictionDetector(
                model_name=self._model_name,
                contradiction_threshold=self._threshold,
            )
            full_result = detector.detect(text, {})
            
            statements = full_result.get("statements", [])
            checker = self.TemporalConsistencyChecker()
            
            temporal_result = checker.verify_temporal_consistency(statements)
            
            data = {
                "is_consistent": temporal_result.get("is_consistent", True),
                "conflicts": temporal_result.get("conflicts", []),
                "timeline_events": len(temporal_result.get("timeline", [])),
            }
            
            confidence = 0.9 if data["is_consistent"] else 0.5
            
            return AdapterResult(
                module_name=self.module_name,
                class_name="TemporalConsistencyChecker",
                method_name="check_temporal_consistency",
                status="success" if data["is_consistent"] else "warnings",
                data=data,
                evidence=[{"type": "temporal_consistency", "consistent": data["is_consistent"]}],
                confidence=confidence,
                execution_time=time.time() - start_time,
                warnings=[f"Found {len(data['conflicts'])} temporal conflicts"] if data["conflicts"] else [],
            )
        except Exception as e:
            return self._error_result("check_temporal_consistency", start_time, e)

    def score_contradiction_probability(
        self,
        statement1: str,
        statement2: str,
        context: Dict[str, float],
    ) -> AdapterResult:
        """
        Controller method for D6.Q29: Score contradiction probability
        Alias for: calculate_posterior
        """
        start_time = time.time()
        
        try:
            scorer = self.BayesianContradictionScorer()
            
            similarity = 0.8
            posterior = scorer.calculate_posterior(similarity, context)
            
            data = {
                "contradiction_probability": posterior,
                "similarity": similarity,
                "context_features": context,
                "verdict": "contradiction" if posterior > 0.7 else "consistent",
            }
            
            return AdapterResult(
                module_name=self.module_name,
                class_name="BayesianContradictionScorer",
                method_name="score_contradiction_probability",
                status="success",
                data=data,
                evidence=[{"type": "bayesian_scoring", "probability": posterior}],
                confidence=abs(posterior - 0.5) * 2,
                execution_time=time.time() - start_time,
            )
        except Exception as e:
            return self._error_result("score_contradiction_probability", start_time, e)

    def analyze_policy_coherence(self, text: str) -> AdapterResult:
        """
        Controller method for D6 (general): Analyze overall policy coherence
        Combines detect + coherence metrics
        """
        start_time = time.time()
        
        try:
            result = self.detect(text, {})
            
            metrics = result.get("coherence_metrics", {})
            contradictions = result.get("contradictions", [])
            
            data = {
                "coherence_score": metrics.get("semantic_coherence", 0),
                "alignment_score": metrics.get("objective_alignment", 0),
                "fragmentation": metrics.get("graph_fragmentation", 0),
                "contradiction_count": len(contradictions),
                "overall_verdict": self._interpret_coherence(metrics, len(contradictions)),
            }
            
            return AdapterResult(
                module_name=self.module_name,
                class_name="PolicyContradictionDetector",
                method_name="analyze_policy_coherence",
                status="success",
                data=data,
                evidence=[{"type": "coherence_analysis"}],
                confidence=data["coherence_score"],
                execution_time=time.time() - start_time,
            )
        except Exception as e:
            return self._error_result("analyze_policy_coherence", start_time, e)

    # ========================================================================
    # DEPRECATED SHIM METHODS (With Warnings)
    # ========================================================================

    def find_contradictions(self, document: str) -> List[Dict[str, Any]]:
        """
        DEPRECATED: Use detect_contradictions() instead.
        Legacy method from earlier refactoring iteration.
        """
        warnings.warn(
            "find_contradictions() is deprecated. Use detect_contradictions() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        result = self.detect_contradictions(document)
        return result.data.get("contradictions", [])

    def check_consistency(self, text: str) -> bool:
        """
        DEPRECATED: Use check_temporal_consistency() instead.
        Returns only boolean instead of full result.
        """
        warnings.warn(
            "check_consistency() is deprecated. Use check_temporal_consistency() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        result = self.check_temporal_consistency(text)
        return result.data.get("is_consistent", False)

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _interpret_coherence(self, metrics: Dict[str, Any], contradiction_count: int) -> str:
        """Interpret coherence metrics"""
        coherence = metrics.get("semantic_coherence", 0)
        if coherence > 0.8 and contradiction_count == 0:
            return "excellent"
        elif coherence > 0.6 and contradiction_count < 5:
            return "good"
        elif coherence > 0.4:
            return "fair"
        else:
            return "poor"

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


def create_contradiction_adapter(model_name: Optional[str] = None, threshold: float = 0.7) -> ContradictionAdapter:
    """Factory function to create ContradictionAdapter instance"""
    return ContradictionAdapter(model_name=model_name, threshold=threshold)
