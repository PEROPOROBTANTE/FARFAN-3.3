"""
Causal Processor Adapter - Semantic Causal Dimension Analysis
================================================================

Wraps causal_proccesor.py functionality with standardized interfaces for the module controller.
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


class CausalProcessorAdapter:
    """
    Adapter for causal_proccesor.py - Semantic Causal Dimension Analysis System.
    
    Responsibility Map (cuestionario.json):
    - D6 (Causalidad): Q26-Q30 (Causal strength, dimension analysis)
    - D1 (Insumos): Q2 (Baseline semantic analysis)
    - D4 (Resultados): Q19 (Result causal linkage)
    
    Original Classes:
    - SemanticChunker: Chunk text with PDM structure awareness
    - DirichletBayesianIntegrator: Bayesian evidence integration
    - CausalDimensionAnalyzer: Analyze 6 causal dimensions
    """

    def __init__(self, config: Optional[Any] = None):
        """
        Initialize adapter with optional configuration.
        
        Args:
            config: SemanticConfig instance (injected dependency)
        """
        self.module_name = "causal_processor"
        self.available = False
        self.logger = logging.getLogger(f"{__name__}.CausalProcessor")
        self._config = config
        self._load_module()

    def _load_module(self):
        """Load causal_proccesor module and its components"""
        try:
            from causal_proccesor import (
                SemanticConfig,
                CausalDimension,
                SemanticChunker,
                DirichletBayesianIntegrator,
                CausalDimensionAnalyzer,
            )
            
            self.SemanticConfig = SemanticConfig
            self.CausalDimension = CausalDimension
            self.SemanticChunker = SemanticChunker
            self.DirichletBayesianIntegrator = DirichletBayesianIntegrator
            self.CausalDimensionAnalyzer = CausalDimensionAnalyzer
            
            if self._config is None:
                self._config = SemanticConfig()
            
            self.available = True
            self.logger.info(f"✓ {self.module_name} loaded successfully")
            
        except ImportError as e:
            self.logger.warning(f"✗ {self.module_name} NOT available: {e}")
            self.available = False

    # ========================================================================
    # ORIGINAL METHOD SIGNATURES (Preserved)
    # ========================================================================

    def chunk_text(self, text: str, preserve_structure: bool = True) -> List[Dict[str, Any]]:
        """
        Original method: SemanticChunker.chunk_text()
        Maps to cuestionario.json: D1.Q2
        """
        if not self.available:
            return []
        
        try:
            chunker = self.SemanticChunker(self._config)
            return chunker.chunk_text(text, preserve_structure)
        except Exception as e:
            self.logger.error(f"chunk_text failed: {e}", exc_info=True)
            return []

    def integrate_evidence(
        self,
        query_embedding: Any,
        chunk_embeddings: List[Any],
        chunk_metadata: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        Original method: DirichletBayesianIntegrator.integrate_evidence()
        Maps to cuestionario.json: D6.Q27
        """
        if not self.available:
            return {}
        
        try:
            integrator = self.DirichletBayesianIntegrator()
            return integrator.integrate_evidence(
                query_embedding,
                chunk_embeddings,
                chunk_metadata,
            )
        except Exception as e:
            self.logger.error(f"integrate_evidence failed: {e}", exc_info=True)
            return {}

    def causal_strength(
        self,
        cause_text: str,
        effect_text: str,
        context_chunks: List[Dict[str, Any]],
    ) -> float:
        """
        Original method: DirichletBayesianIntegrator.causal_strength()
        Maps to cuestionario.json: D6.Q28
        """
        if not self.available:
            return 0.0
        
        try:
            integrator = self.DirichletBayesianIntegrator()
            return integrator.causal_strength(cause_text, effect_text, context_chunks)
        except Exception as e:
            self.logger.error(f"causal_strength failed: {e}", exc_info=True)
            return 0.0

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Original method: CausalDimensionAnalyzer.analyze()
        Maps to cuestionario.json: D6.Q26-Q30
        """
        if not self.available:
            return {}
        
        try:
            analyzer = self.CausalDimensionAnalyzer(self._config)
            return analyzer.analyze(text)
        except Exception as e:
            self.logger.error(f"analyze failed: {e}", exc_info=True)
            return {}

    # ========================================================================
    # STANDARDIZED CONTROLLER INTERFACE (Alias Methods)
    # ========================================================================

    def analyze_causal_dimensions(self, text: str) -> AdapterResult:
        """
        Controller method for D6.Q26-Q30: Analyze all 6 causal dimensions
        Alias for: analyze
        """
        start_time = time.time()
        
        try:
            analysis = self.analyze(text)
            
            dimensions = analysis.get("dimensions", {})
            data = {
                "dimension_scores": dimensions,
                "top_dimension": max(dimensions.items(), key=lambda x: x[1])[0] if dimensions else None,
                "excerpts": analysis.get("key_excerpts", []),
                "chunk_count": len(analysis.get("chunks", [])),
            }
            
            avg_score = sum(dimensions.values()) / len(dimensions) if dimensions else 0.0
            
            return AdapterResult(
                module_name=self.module_name,
                class_name="CausalDimensionAnalyzer",
                method_name="analyze_causal_dimensions",
                status="success",
                data=data,
                evidence=[{"type": "causal_dimensions", "dimensions": len(dimensions)}],
                confidence=avg_score,
                execution_time=time.time() - start_time,
            )
        except Exception as e:
            return self._error_result("analyze_causal_dimensions", start_time, e)

    def chunk_document(self, text: str, preserve_pdm: bool = True) -> AdapterResult:
        """
        Controller method for D1.Q2: Semantic chunking with PDM awareness
        Alias for: chunk_text
        """
        start_time = time.time()
        
        try:
            chunks = self.chunk_text(text, preserve_structure=preserve_pdm)
            
            data = {
                "chunk_count": len(chunks),
                "chunks": chunks,
                "avg_chunk_size": sum(len(c.get("text", "")) for c in chunks) / len(chunks) if chunks else 0,
            }
            
            return AdapterResult(
                module_name=self.module_name,
                class_name="SemanticChunker",
                method_name="chunk_document",
                status="success",
                data=data,
                evidence=[{"type": "semantic_chunks", "count": len(chunks)}],
                confidence=0.90,
                execution_time=time.time() - start_time,
            )
        except Exception as e:
            return self._error_result("chunk_document", start_time, e)

    def compute_causal_strength(
        self,
        cause: str,
        effect: str,
        context: str,
    ) -> AdapterResult:
        """
        Controller method for D6.Q28: Compute causal strength
        Alias for: causal_strength
        """
        start_time = time.time()
        
        try:
            chunks = self.chunk_text(context)
            strength = self.causal_strength(cause, effect, chunks)
            
            data = {
                "causal_strength": strength,
                "cause": cause,
                "effect": effect,
                "confidence_level": "high" if strength > 0.7 else "medium" if strength > 0.4 else "low",
            }
            
            return AdapterResult(
                module_name=self.module_name,
                class_name="DirichletBayesianIntegrator",
                method_name="compute_causal_strength",
                status="success",
                data=data,
                evidence=[{"type": "causal_strength", "value": strength}],
                confidence=strength,
                execution_time=time.time() - start_time,
            )
        except Exception as e:
            return self._error_result("compute_causal_strength", start_time, e)

    def bayesian_evidence_integration(
        self,
        query: str,
        document: str,
    ) -> AdapterResult:
        """
        Controller method for D6.Q27: Bayesian evidence integration
        Alias for: integrate_evidence
        """
        start_time = time.time()
        
        try:
            chunker = self.SemanticChunker(self._config)
            chunks = chunker.chunk_text(document)
            
            if not chunks:
                return self._error_result("bayesian_evidence_integration", start_time, 
                                        Exception("No chunks generated"))
            
            query_emb = chunker.embed_single(query)
            chunk_embs = [chunker.embed_single(c["text"]) for c in chunks]
            chunk_meta = [{"position": i, "type": "semantic"} for i in range(len(chunks))]
            
            integrator = self.DirichletBayesianIntegrator()
            evidence = integrator.integrate_evidence(query_emb, chunk_embs, chunk_meta)
            
            data = {
                "evidence_scores": evidence,
                "max_evidence": max(evidence.values()) if evidence else 0.0,
                "chunk_count": len(chunks),
            }
            
            return AdapterResult(
                module_name=self.module_name,
                class_name="DirichletBayesianIntegrator",
                method_name="bayesian_evidence_integration",
                status="success",
                data=data,
                evidence=[{"type": "bayesian_evidence", "chunks": len(chunks)}],
                confidence=data["max_evidence"],
                execution_time=time.time() - start_time,
            )
        except Exception as e:
            return self._error_result("bayesian_evidence_integration", start_time, e)

    # ========================================================================
    # DEPRECATED SHIM METHODS (With Warnings)
    # ========================================================================

    def semantic_chunk(self, text: str) -> List[Dict[str, Any]]:
        """
        DEPRECATED: Use chunk_document() instead.
        Legacy method from earlier refactoring iteration.
        """
        warnings.warn(
            "semantic_chunk() is deprecated. Use chunk_document() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        result = self.chunk_document(text)
        return result.data.get("chunks", [])

    def calculate_causal_link(self, cause: str, effect: str, text: str) -> float:
        """
        DEPRECATED: Use compute_causal_strength() instead.
        Returns only float instead of full result.
        """
        warnings.warn(
            "calculate_causal_link() is deprecated. Use compute_causal_strength() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        result = self.compute_causal_strength(cause, effect, text)
        return result.data.get("causal_strength", 0.0)

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


def create_causal_processor_adapter(config: Optional[Any] = None) -> CausalProcessorAdapter:
    """Factory function to create CausalProcessorAdapter instance"""
    return CausalProcessorAdapter(config=config)
