"""
ContradictionDetection Adapter Layer
=====================================

Backward-compatible adapter wrapping contradiction_deteccion.py functionality.
Provides translation layer between legacy 11-adapter architecture and unified module controller.

This adapter preserves existing method signatures while delegating to the core domain module.
"""

import logging
import warnings
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


class ContradictionDetectionAdapter:
    """
    Adapter for contradiction_deteccion.py - Policy Contradiction Detection System.
    
    Wraps AdvancedContradictionDetector, SemanticAnalyzer, TemporalValidator,
    NumericalConsistencyChecker, and GraphReasoningEngine.
    
    PRIMARY INTERFACE (Backward Compatible):
    - detect_contradictions(text: str) -> List[Dict[str, Any]]
    - analyze_semantic_contradictions(statements: List[str]) -> List[Dict[str, Any]]
    - validate_temporal_consistency(statements: List[Dict]) -> List[Dict[str, Any]]
    - check_numerical_consistency(claims: List[Dict]) -> List[Dict[str, Any]]
    - build_contradiction_graph(statements: List[str]) -> nx.Graph
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize adapter with optional configuration.
        
        Args:
            config: Configuration dict for contradiction detection (optional)
        """
        self._load_core_module()
        self._initialize_components(config)
        logger.info("ContradictionDetectionAdapter initialized successfully")

    def _load_core_module(self):
        """Load core domain module components"""
        try:
            from contradiction_deteccion import (
                AdvancedContradictionDetector,
                PolicyStatement,
                ContradictionEvidence,
                ContradictionType,
                PolicyDimension
            )
            
            self.AdvancedContradictionDetector = AdvancedContradictionDetector
            self.PolicyStatement = PolicyStatement
            self.ContradictionEvidence = ContradictionEvidence
            self.ContradictionType = ContradictionType
            self.PolicyDimension = PolicyDimension
            self._module_available = True
            
        except ImportError as e:
            logger.error(f"Failed to load contradiction_deteccion module: {e}")
            self._module_available = False
            raise RuntimeError(f"Core module contradiction_deteccion not available: {e}")

    def _initialize_components(self, config: Optional[Dict[str, Any]]):
        """Initialize contradiction detection components"""
        self.detector = self.AdvancedContradictionDetector()
        self.config = config or {}

    # ========================================================================
    # PRIMARY INTERFACE (Backward Compatible)
    # ========================================================================

    def detect_contradictions(
        self, 
        text: str,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Detect all types of contradictions in policy text.
        
        Args:
            text: Policy document text
            **kwargs: Additional detection parameters
            
        Returns:
            List of contradiction evidence dicts
        """
        # Parse text into statements
        statements = self._parse_statements(text)
        
        # Detect contradictions
        contradictions = self.detector.detect_contradictions(statements)
        
        # Convert to dict format for backward compatibility
        return [
            {
                'statement_a': c.statement_a.text,
                'statement_b': c.statement_b.text,
                'contradiction_type': c.contradiction_type.name,
                'confidence': c.confidence,
                'severity': c.severity,
                'semantic_similarity': c.semantic_similarity,
                'affected_dimensions': [d.value for d in c.affected_dimensions],
                'resolution_suggestions': c.resolution_suggestions
            }
            for c in contradictions
        ]

    def analyze_semantic_contradictions(
        self, 
        statements: List[str],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Analyze semantic contradictions between statements.
        
        Args:
            statements: List of policy statements
            **kwargs: Additional analysis parameters
            
        Returns:
            List of semantic contradiction findings
        """
        threshold = kwargs.get('threshold', 0.85)
        
        # Create statement objects
        stmt_objects = [
            self.PolicyStatement(
                text=stmt,
                dimension=self.PolicyDimension.ESTRATEGICO,
                position=(0, len(stmt))
            )
            for stmt in statements
        ]
        
        # Detect contradictions
        all_contradictions = self.detector.detect_contradictions(stmt_objects)
        
        # Filter for semantic type
        semantic_contradictions = [
            c for c in all_contradictions
            if c.contradiction_type == self.ContradictionType.SEMANTIC_OPPOSITION
            and c.confidence >= threshold
        ]
        
        return [
            {
                'statement_a': c.statement_a.text,
                'statement_b': c.statement_b.text,
                'confidence': c.confidence,
                'semantic_similarity': c.semantic_similarity
            }
            for c in semantic_contradictions
        ]

    def validate_temporal_consistency(
        self, 
        statements: List[Dict[str, Any]],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Validate temporal consistency of policy statements.
        
        Args:
            statements: List of statement dicts with temporal markers
            **kwargs: Additional validation parameters
            
        Returns:
            List of temporal inconsistency findings
        """
        # Create statement objects with temporal markers
        stmt_objects = [
            self.PolicyStatement(
                text=stmt['text'],
                dimension=self.PolicyDimension.PROGRAMATICO,
                position=(0, len(stmt['text'])),
                temporal_markers=stmt.get('temporal_markers', [])
            )
            for stmt in statements
        ]
        
        # Detect contradictions
        all_contradictions = self.detector.detect_contradictions(stmt_objects)
        
        # Filter for temporal type
        temporal_contradictions = [
            c for c in all_contradictions
            if c.contradiction_type == self.ContradictionType.TEMPORAL_CONFLICT
        ]
        
        return [
            {
                'statement_a': c.statement_a.text,
                'statement_b': c.statement_b.text,
                'confidence': c.confidence,
                'temporal_consistent': c.temporal_consistency
            }
            for c in temporal_contradictions
        ]

    def check_numerical_consistency(
        self, 
        claims: List[Dict[str, Any]],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Check numerical consistency across policy claims.
        
        Args:
            claims: List of claim dicts with quantitative data
            **kwargs: Additional checking parameters
            
        Returns:
            List of numerical inconsistency findings
        """
        # Create statement objects with quantitative claims
        stmt_objects = [
            self.PolicyStatement(
                text=claim['text'],
                dimension=self.PolicyDimension.FINANCIERO,
                position=(0, len(claim['text'])),
                quantitative_claims=claim.get('quantitative_claims', [])
            )
            for claim in claims
        ]
        
        # Detect contradictions
        all_contradictions = self.detector.detect_contradictions(stmt_objects)
        
        # Filter for numerical type
        numerical_contradictions = [
            c for c in all_contradictions
            if c.contradiction_type == self.ContradictionType.NUMERICAL_INCONSISTENCY
        ]
        
        return [
            {
                'statement_a': c.statement_a.text,
                'statement_b': c.statement_b.text,
                'confidence': c.confidence,
                'numerical_divergence': c.numerical_divergence
            }
            for c in numerical_contradictions
        ]

    def build_contradiction_graph(
        self, 
        statements: List[str],
        **kwargs
    ) -> nx.Graph:
        """
        Build graph representation of contradiction relationships.
        
        Args:
            statements: List of policy statements
            **kwargs: Additional graph parameters
            
        Returns:
            NetworkX graph with statements as nodes and contradictions as edges
        """
        # Create statement objects
        stmt_objects = [
            self.PolicyStatement(
                text=stmt,
                dimension=self.PolicyDimension.ESTRATEGICO,
                position=(0, len(stmt))
            )
            for stmt in statements
        ]
        
        # Detect contradictions
        contradictions = self.detector.detect_contradictions(stmt_objects)
        
        # Build graph
        G = nx.Graph()
        
        # Add nodes
        for i, stmt in enumerate(statements):
            G.add_node(i, text=stmt)
        
        # Add edges for contradictions
        for c in contradictions:
            idx_a = statements.index(c.statement_a.text)
            idx_b = statements.index(c.statement_b.text)
            G.add_edge(
                idx_a, 
                idx_b,
                contradiction_type=c.contradiction_type.name,
                confidence=c.confidence,
                severity=c.severity
            )
        
        return G

    # ========================================================================
    # LEGACY METHOD ALIASES (with Deprecation Warnings)
    # ========================================================================

    def find_contradictions(self, text: str, **kwargs) -> List[Dict]:
        """
        DEPRECATED: Use detect_contradictions() instead.
        """
        warnings.warn(
            "find_contradictions() is deprecated, use detect_contradictions() instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self.detect_contradictions(text, **kwargs)

    def semantic_analysis(self, statements: List[str], **kwargs) -> List[Dict]:
        """
        DEPRECATED: Use analyze_semantic_contradictions() instead.
        """
        warnings.warn(
            "semantic_analysis() is deprecated, use analyze_semantic_contradictions() instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self.analyze_semantic_contradictions(statements, **kwargs)

    def temporal_validation(self, statements: List[Dict], **kwargs) -> List[Dict]:
        """
        DEPRECATED: Use validate_temporal_consistency() instead.
        """
        warnings.warn(
            "temporal_validation() is deprecated, use validate_temporal_consistency() instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self.validate_temporal_consistency(statements, **kwargs)

    def numerical_check(self, claims: List[Dict], **kwargs) -> List[Dict]:
        """
        DEPRECATED: Use check_numerical_consistency() instead.
        """
        warnings.warn(
            "numerical_check() is deprecated, use check_numerical_consistency() instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self.check_numerical_consistency(claims, **kwargs)

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def _parse_statements(self, text: str) -> List:
        """Parse text into PolicyStatement objects"""
        # Simple sentence splitting for demonstration
        import re
        sentences = re.split(r'[.!?]+', text)
        
        return [
            self.PolicyStatement(
                text=sent.strip(),
                dimension=self.PolicyDimension.ESTRATEGICO,
                position=(0, len(sent))
            )
            for sent in sentences if sent.strip()
        ]

    def get_contradiction_types(self) -> List[str]:
        """Get list of available contradiction types"""
        return [t.name for t in self.ContradictionType]

    def get_policy_dimensions(self) -> List[str]:
        """Get list of policy dimensions"""
        return [d.value for d in self.PolicyDimension]

    def is_available(self) -> bool:
        """Check if core module is available"""
        return self._module_available
