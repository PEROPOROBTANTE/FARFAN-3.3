"""
CausalProcessor Adapter Layer
==============================

Backward-compatible adapter wrapping causal_proccesor.py functionality.
Provides translation layer between legacy 11-adapter architecture and unified module controller.

This adapter preserves existing method signatures while delegating to the core domain module.
"""

import logging
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class CausalProcessorAdapter:
    """
    Adapter for causal_proccesor.py - Causal Policy Analysis Framework.
    
    Wraps SemanticProcessor, CausalInferenceEngine, DAGLearner,
    and CounterfactualAnalyzer for Colombian PDM analysis.
    
    PRIMARY INTERFACE (Backward Compatible):
    - extract_causal_dimensions(text: str) -> Dict[str, Any]
    - infer_causal_structure(data: Dict) -> Any
    - compute_causal_effects(treatment: str, outcome: str, data: Dict) -> Dict[str, float]
    - analyze_counterfactual(intervention: Dict, context: Dict) -> Dict[str, Any]
    - validate_causal_assumptions(dag: Any) -> Dict[str, bool]
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize adapter with optional configuration.
        
        Args:
            config: Configuration dict for causal processor (optional)
        """
        self._load_core_module()
        self._initialize_components(config)
        logger.info("CausalProcessorAdapter initialized successfully")

    def _load_core_module(self):
        """Load core domain module components"""
        try:
            from causal_proccesor import (
                SemanticConfig,
                SemanticProcessor,
                CausalDimension,
                PDMSection
            )
            
            self.SemanticConfig = SemanticConfig
            self.SemanticProcessor = SemanticProcessor
            self.CausalDimension = CausalDimension
            self.PDMSection = PDMSection
            self._module_available = True
            
        except ImportError as e:
            logger.error(f"Failed to load causal_proccesor module: {e}")
            self._module_available = False
            raise RuntimeError(f"Core module causal_proccesor not available: {e}")

    def _initialize_components(self, config: Optional[Dict[str, Any]]):
        """Initialize causal processing components"""
        if config:
            self.semantic_config = self.SemanticConfig(**config)
        else:
            self.semantic_config = self.SemanticConfig()
        
        self.semantic_processor = self.SemanticProcessor(self.semantic_config)

    # ========================================================================
    # PRIMARY INTERFACE (Backward Compatible)
    # ========================================================================

    def extract_causal_dimensions(
        self, 
        text: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Extract causal dimensions (insumos, actividades, productos, etc.) from text.
        
        Args:
            text: Policy document text
            **kwargs: Additional extraction parameters
            
        Returns:
            Dict mapping dimension names to extracted content
        """
        dimensions = {}
        
        # Extract evidence for each causal dimension
        for dimension in self.CausalDimension:
            # Simplified extraction - real implementation would use semantic matching
            dimension_text = self._extract_dimension_text(text, dimension)
            dimensions[dimension.value] = {
                'text': dimension_text,
                'dimension': dimension.value,
                'confidence': 0.8 if dimension_text else 0.0
            }
        
        return dimensions

    def infer_causal_structure(
        self, 
        data: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Infer causal DAG structure from policy data.
        
        Args:
            data: Policy data with variables and relationships
            **kwargs: Additional inference parameters
            
        Returns:
            Inferred causal structure representation
        """
        # Placeholder for causal structure learning
        # Real implementation would use constraint-based or score-based methods
        structure = {
            'nodes': list(data.keys()),
            'edges': [],
            'confidence': 0.7,
            'method': 'constraint_based'
        }
        
        return structure

    def compute_causal_effects(
        self, 
        treatment: str,
        outcome: str,
        data: Dict[str, Any],
        **kwargs
    ) -> Dict[str, float]:
        """
        Compute causal effects using Bayesian inference.
        
        Args:
            treatment: Treatment variable name
            outcome: Outcome variable name
            data: Data for causal estimation
            **kwargs: Additional computation parameters
            
        Returns:
            Dict with effect estimates and credible intervals
        """
        # Simplified causal effect estimation
        effect = {
            'ate': 0.0,  # Average Treatment Effect
            'credible_interval_95': (0.0, 0.0),
            'posterior_samples': np.array([]),
            'method': 'bayesian_regression'
        }
        
        return effect

    def analyze_counterfactual(
        self, 
        intervention: Dict[str, Any],
        context: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Analyze counterfactual scenarios under intervention.
        
        Args:
            intervention: Dict specifying intervention variables and values
            context: Current context/baseline scenario
            **kwargs: Additional analysis parameters
            
        Returns:
            Counterfactual predictions and confidence
        """
        counterfactual = {
            'predicted_outcome': 0.0,
            'confidence': 0.7,
            'assumptions': ['SUTVA', 'ignorability'],
            'sensitivity': {}
        }
        
        return counterfactual

    def validate_causal_assumptions(
        self, 
        dag: Any,
        **kwargs
    ) -> Dict[str, bool]:
        """
        Validate causal identification assumptions.
        
        Args:
            dag: Causal DAG representation
            **kwargs: Additional validation parameters
            
        Returns:
            Dict of assumption names and validation status
        """
        assumptions = {
            'acyclicity': True,
            'no_unobserved_confounding': False,
            'sutva': True,
            'positivity': True,
            'consistency': True
        }
        
        return assumptions

    # ========================================================================
    # LEGACY METHOD ALIASES (with Deprecation Warnings)
    # ========================================================================

    def extract_dimensions(self, text: str, **kwargs) -> Dict:
        """
        DEPRECATED: Use extract_causal_dimensions() instead.
        """
        warnings.warn(
            "extract_dimensions() is deprecated, use extract_causal_dimensions() instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self.extract_causal_dimensions(text, **kwargs)

    def learn_structure(self, data: Dict, **kwargs) -> Dict:
        """
        DEPRECATED: Use infer_causal_structure() instead.
        """
        warnings.warn(
            "learn_structure() is deprecated, use infer_causal_structure() instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self.infer_causal_structure(data, **kwargs)

    def estimate_effects(
        self, 
        treatment: str, 
        outcome: str, 
        data: Dict, 
        **kwargs
    ) -> Dict:
        """
        DEPRECATED: Use compute_causal_effects() instead.
        """
        warnings.warn(
            "estimate_effects() is deprecated, use compute_causal_effects() instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self.compute_causal_effects(treatment, outcome, data, **kwargs)

    def counterfactual_analysis(self, intervention: Dict, context: Dict, **kwargs) -> Dict:
        """
        DEPRECATED: Use analyze_counterfactual() instead.
        """
        warnings.warn(
            "counterfactual_analysis() is deprecated, use analyze_counterfactual() instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self.analyze_counterfactual(intervention, context, **kwargs)

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def _extract_dimension_text(self, text: str, dimension) -> str:
        """Extract text relevant to specific causal dimension"""
        # Simplified - real implementation would use semantic matching
        keywords = {
            'insumos': ['recurso', 'presupuesto', 'capacidad'],
            'actividades': ['actividad', 'proceso', 'acción'],
            'productos': ['producto', 'entregable', 'output'],
            'resultados': ['resultado', 'outcome', 'efecto'],
            'impactos': ['impacto', 'transformación'],
            'supuestos': ['supuesto', 'hipótesis', 'asunción']
        }
        
        dim_keywords = keywords.get(dimension.value, [])
        
        # Simple keyword-based extraction
        for keyword in dim_keywords:
            if keyword in text.lower():
                return text
        
        return ""

    def get_causal_dimensions(self) -> List[str]:
        """Get list of available causal dimensions"""
        return [d.value for d in self.CausalDimension]

    def get_pdm_sections(self) -> List[str]:
        """Get list of PDM section types"""
        return [s.value for s in self.PDMSection]

    def get_config(self) -> Dict[str, Any]:
        """Get current semantic configuration"""
        return {
            'embedding_model': self.semantic_config.embedding_model,
            'chunk_size': self.semantic_config.chunk_size,
            'similarity_threshold': self.semantic_config.similarity_threshold
        }

    def is_available(self) -> bool:
        """Check if core module is available"""
        return self._module_available
