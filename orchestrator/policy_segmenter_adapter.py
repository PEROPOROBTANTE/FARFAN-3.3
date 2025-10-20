"""
Policy Segmenter Adapter - Document Segmentation with Bayesian Boundaries
===========================================================================

Wraps policy_segmenter.py functionality with standardized interfaces for the module controller.
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


class PolicySegmenterAdapter:
    """
    Adapter for policy_segmenter.py - Document Segmentation System.
    
    Responsibility Map (cuestionario.json):
    - D1 (Insumos): Q1 (Document structure analysis)
    - D2 (Actividades): Q11 (Activity section detection)
    - D3 (Productos): Q15 (Product section detection)
    - All dimensions: Document preprocessing
    
    Original Classes:
    - SentenceSegmenter: Sentence-level segmentation
    - SemanticBoundaryScorer: Score semantic boundaries
    - StructureDetector: Detect document structures
    - DynamicProgrammingSegmenter: Optimal segmentation
    - PolicySegmenter: Main segmentation orchestrator
    """

    def __init__(self, config: Optional[Any] = None):
        """
        Initialize adapter with optional configuration.
        
        Args:
            config: SegmenterConfig instance (injected dependency)
        """
        self.module_name = "policy_segmenter"
        self.available = False
        self.logger = logging.getLogger(f"{__name__}.PolicySegmenter")
        self._config = config
        self._load_module()

    def _load_module(self):
        """Load policy_segmenter module and its components"""
        try:
            from policy_segmenter import (
                SegmenterConfig,
                SentenceSegmenter,
                SemanticBoundaryScorer,
                StructureDetector,
                DynamicProgrammingSegmenter,
                PolicySegmenter,
            )
            
            self.SegmenterConfig = SegmenterConfig
            self.SentenceSegmenter = SentenceSegmenter
            self.SemanticBoundaryScorer = SemanticBoundaryScorer
            self.StructureDetector = StructureDetector
            self.DynamicProgrammingSegmenter = DynamicProgrammingSegmenter
            self.PolicySegmenter = PolicySegmenter
            
            if self._config is None:
                self._config = SegmenterConfig()
            
            self.available = True
            self.logger.info(f"✓ {self.module_name} loaded successfully")
            
        except ImportError as e:
            self.logger.warning(f"✗ {self.module_name} NOT available: {e}")
            self.available = False

    # ========================================================================
    # ORIGINAL METHOD SIGNATURES (Preserved)
    # ========================================================================

    def segment(self, text: str) -> List[Dict[str, Any]]:
        """
        Original method: PolicySegmenter.segment()
        Maps to cuestionario.json: All dimensions (preprocessing)
        """
        if not self.available:
            return []
        
        try:
            segmenter = self.PolicySegmenter(self._config)
            return segmenter.segment(text)
        except Exception as e:
            self.logger.error(f"segment failed: {e}", exc_info=True)
            return []

    def get_segmentation_report(self, segmenter: Any) -> Dict[str, Any]:
        """
        Original method: PolicySegmenter.get_segmentation_report()
        Maps to cuestionario.json: Quality metrics
        """
        if not self.available:
            return {}
        
        try:
            return segmenter.get_segmentation_report()
        except Exception as e:
            self.logger.error(f"get_segmentation_report failed: {e}", exc_info=True)
            return {}

    def detect_structures(self, text: str) -> Dict[str, Any]:
        """
        Original method: StructureDetector.detect_structures()
        Maps to cuestionario.json: D1.Q1
        """
        if not self.available:
            return {}
        
        try:
            return self.StructureDetector.detect_structures(text)
        except Exception as e:
            self.logger.error(f"detect_structures failed: {e}", exc_info=True)
            return {}

    def score_boundaries(
        self,
        scorer: Any,
        sentences: List[str],
        preserve_structure: bool = True,
    ) -> Any:
        """
        Original method: SemanticBoundaryScorer.score_boundaries()
        Maps to cuestionario.json: Segmentation quality
        """
        if not self.available:
            return None
        
        try:
            return scorer.score_boundaries(sentences, preserve_structure)
        except Exception as e:
            self.logger.error(f"score_boundaries failed: {e}", exc_info=True)
            return None

    # ========================================================================
    # STANDARDIZED CONTROLLER INTERFACE (Alias Methods)
    # ========================================================================

    def segment_document(self, text: str) -> AdapterResult:
        """
        Controller method for all dimensions: Segment document into analyzable chunks
        Alias for: segment
        """
        start_time = time.time()
        
        try:
            segments = self.segment(text)
            
            data = {
                "segment_count": len(segments),
                "segments": segments,
                "avg_length": sum(len(s.get("text", "")) for s in segments) / len(segments) if segments else 0,
                "section_types": list(set(s.get("section_type", "unknown") for s in segments)),
            }
            
            return AdapterResult(
                module_name=self.module_name,
                class_name="PolicySegmenter",
                method_name="segment_document",
                status="success",
                data=data,
                evidence=[{"type": "document_segmentation", "segments": len(segments)}],
                confidence=0.90,
                execution_time=time.time() - start_time,
            )
        except Exception as e:
            return self._error_result("segment_document", start_time, e)

    def analyze_document_structure(self, text: str) -> AdapterResult:
        """
        Controller method for D1.Q1: Analyze document structure
        Alias for: detect_structures
        """
        start_time = time.time()
        
        try:
            structures = self.detect_structures(text)
            
            data = {
                "has_tables": structures.get("has_tables", False),
                "has_lists": structures.get("has_lists", False),
                "table_regions": structures.get("table_regions", []),
                "list_regions": structures.get("list_regions", []),
                "structure_complexity": len(structures.get("table_regions", [])) + len(structures.get("list_regions", [])),
            }
            
            return AdapterResult(
                module_name=self.module_name,
                class_name="StructureDetector",
                method_name="analyze_document_structure",
                status="success",
                data=data,
                evidence=[{"type": "structure_analysis"}],
                confidence=0.85,
                execution_time=time.time() - start_time,
            )
        except Exception as e:
            return self._error_result("analyze_document_structure", start_time, e)

    def get_segmentation_quality(self, text: str) -> AdapterResult:
        """
        Controller method for quality assessment: Get segmentation quality metrics
        Alias for: segment + get_segmentation_report
        """
        start_time = time.time()
        
        try:
            segmenter = self.PolicySegmenter(self._config)
            segments = segmenter.segment(text)
            report = segmenter.get_segmentation_report()
            
            data = {
                "segment_count": len(segments),
                "quality_metrics": report.get("quality_metrics", {}),
                "statistics": report.get("statistics", {}),
                "consistency_score": report.get("quality_metrics", {}).get("consistency_score", 0),
            }
            
            return AdapterResult(
                module_name=self.module_name,
                class_name="PolicySegmenter",
                method_name="get_segmentation_quality",
                status="success",
                data=data,
                evidence=[{"type": "quality_assessment"}],
                confidence=data["consistency_score"],
                execution_time=time.time() - start_time,
            )
        except Exception as e:
            return self._error_result("get_segmentation_quality", start_time, e)

    def score_semantic_boundaries(self, text: str) -> AdapterResult:
        """
        Controller method for semantic analysis: Score semantic boundaries
        Alias for: score_boundaries
        """
        start_time = time.time()
        
        try:
            sentences = self.SentenceSegmenter.segment(text)
            scorer = self.SemanticBoundaryScorer(model_name="hiiamsid/sentence_similarity_spanish_es")
            scores = scorer.score_boundaries(sentences, preserve_structure=True)
            
            data = {
                "sentence_count": len(sentences),
                "boundary_scores": scores.tolist() if hasattr(scores, 'tolist') else scores,
                "avg_score": float(scores.mean()) if hasattr(scores, 'mean') else 0,
            }
            
            return AdapterResult(
                module_name=self.module_name,
                class_name="SemanticBoundaryScorer",
                method_name="score_semantic_boundaries",
                status="success",
                data=data,
                evidence=[{"type": "boundary_scoring"}],
                confidence=0.88,
                execution_time=time.time() - start_time,
            )
        except Exception as e:
            return self._error_result("score_semantic_boundaries", start_time, e)

    # ========================================================================
    # DEPRECATED SHIM METHODS (With Warnings)
    # ========================================================================

    def split_document(self, text: str) -> List[str]:
        """
        DEPRECATED: Use segment_document() instead.
        Legacy method from earlier refactoring iteration.
        """
        warnings.warn(
            "split_document() is deprecated. Use segment_document() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        result = self.segment_document(text)
        return [s.get("text", "") for s in result.data.get("segments", [])]

    def analyze_structure(self, document: str) -> Dict[str, bool]:
        """
        DEPRECATED: Use analyze_document_structure() instead.
        Returns only simplified structure info.
        """
        warnings.warn(
            "analyze_structure() is deprecated. Use analyze_document_structure() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        result = self.analyze_document_structure(document)
        return {
            "has_tables": result.data.get("has_tables", False),
            "has_lists": result.data.get("has_lists", False),
        }

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


def create_policy_segmenter_adapter(config: Optional[Any] = None) -> PolicySegmenterAdapter:
    """Factory function to create PolicySegmenterAdapter instance"""
    return PolicySegmenterAdapter(config=config)
