"""
DerekBeach Adapter Layer
========================

Backward-compatible adapter wrapping dereck_beach.py functionality.
Provides translation layer between legacy 11-adapter architecture and unified module controller.

This adapter preserves existing method signatures while delegating to the core domain module.
"""

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)


class DerekBeachAdapter:
    """
    Adapter for dereck_beach.py - Causal Deconstruction and Audit Framework (CDAF).
    
    Wraps PDFExtractor, CausalGraphExtractor, BayesianInferenceEngine,
    DAGValidator, and CDAFProcessor.
    
    PRIMARY INTERFACE (Backward Compatible):
    - extract_causal_hierarchy(text: str) -> nx.DiGraph
    - extract_entity_activities(text: str) -> List[Dict[str, Any]]
    - audit_evidence_traceability(nodes: Dict) -> Dict[str, Any]
    - process_pdf_document(pdf_path: str, policy_code: str) -> Dict[str, Any]
    - validate_dag_structure(graph: nx.DiGraph) -> Dict[str, Any]
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize adapter with optional configuration file.
        
        Args:
            config_path: Path to YAML configuration file (optional)
        """
        self._load_core_module()
        self._initialize_components(config_path)
        logger.info("DerekBeachAdapter initialized successfully")

    def _load_core_module(self):
        """Load core domain module components"""
        try:
            from dereck_beach import (
                ConfigLoader,
                PDFExtractor,
                CausalGraphExtractor,
                BayesianInferenceEngine,
                DAGValidator,
                CDAFProcessor,
                MetaNode,
                EntityActivity,
                BeachEvidentialTest
            )
            
            self.ConfigLoader = ConfigLoader
            self.PDFExtractor = PDFExtractor
            self.CausalGraphExtractor = CausalGraphExtractor
            self.BayesianInferenceEngine = BayesianInferenceEngine
            self.DAGValidator = DAGValidator
            self.CDAFProcessor = CDAFProcessor
            self.MetaNode = MetaNode
            self.EntityActivity = EntityActivity
            self.BeachEvidentialTest = BeachEvidentialTest
            self._module_available = True
            
        except ImportError as e:
            logger.error(f"Failed to load dereck_beach module: {e}")
            self._module_available = False
            raise RuntimeError(f"Core module dereck_beach not available: {e}")

    def _initialize_components(self, config_path: Optional[str]):
        """Initialize CDAF components"""
        if config_path:
            config_loader = self.ConfigLoader(Path(config_path))
        else:
            config_loader = self.ConfigLoader(Path("config.yaml"))
        
        self.config = config_loader.config
        self.graph_extractor = self.CausalGraphExtractor(self.config)
        self.bayesian_engine = self.BayesianInferenceEngine(self.config)
        self.dag_validator = self.DAGValidator()
        self.processor = None  # Lazy loaded when needed

    def _ensure_processor(self):
        """Lazy load CDAF processor"""
        if self.processor is None:
            self.processor = self.CDAFProcessor(config=self.config)

    # ========================================================================
    # PRIMARY INTERFACE (Backward Compatible)
    # ========================================================================

    def extract_causal_hierarchy(
        self, 
        text: str,
        **kwargs
    ) -> nx.DiGraph:
        """
        Extract causal hierarchy from policy text.
        
        Args:
            text: Policy document text
            **kwargs: Additional extraction parameters
            
        Returns:
            NetworkX DiGraph representing causal structure
        """
        return self.graph_extractor.extract_causal_hierarchy(text)

    def extract_entity_activities(
        self, 
        text: str,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Extract entity-activity tuples (Beach mechanistic evidence).
        
        Args:
            text: Text to analyze for entities and activities
            **kwargs: Additional extraction parameters
            
        Returns:
            List of dicts with entity, activity, verb_lemma, confidence
        """
        entity_activity = self.graph_extractor.extract_entity_activity(text)
        
        if entity_activity:
            return [{
                'entity': entity_activity.entity,
                'activity': entity_activity.activity,
                'verb_lemma': entity_activity.verb_lemma,
                'confidence': entity_activity.confidence
            }]
        else:
            return []

    def audit_evidence_traceability(
        self, 
        nodes: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Audit evidence traceability for policy nodes.
        
        Args:
            nodes: Dict of MetaNode objects keyed by node ID
            **kwargs: Additional audit parameters
            
        Returns:
            Audit results with passed status, warnings, errors, recommendations
        """
        self._ensure_processor()
        return self.processor.audit_evidence_traceability(nodes)

    def process_pdf_document(
        self, 
        pdf_path: str, 
        policy_code: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process PDF document and extract causal structure.
        
        Args:
            pdf_path: Path to PDF file
            policy_code: Policy code identifier
            **kwargs: Additional processing parameters
            
        Returns:
            Dict with extracted text, tables, sections, causal graph
        """
        pdf_extractor = self.PDFExtractor(Path(pdf_path))
        
        return {
            'text': pdf_extractor.extract_text(),
            'tables': pdf_extractor.extract_tables(),
            'sections': pdf_extractor.extract_sections(),
            'causal_graph': self.graph_extractor.extract_causal_hierarchy(
                pdf_extractor.extract_text()
            )
        }

    def validate_dag_structure(
        self, 
        graph: nx.DiGraph,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Validate DAG structure and logical consistency.
        
        Args:
            graph: NetworkX DiGraph to validate
            **kwargs: Additional validation parameters
            
        Returns:
            Validation results with is_valid, cycles, warnings
        """
        is_acyclic = nx.is_directed_acyclic_graph(graph)
        cycles = [] if is_acyclic else list(nx.simple_cycles(graph))
        
        warnings_list = []
        if not is_acyclic:
            warnings_list.append(f"Graph contains {len(cycles)} cycles")
        
        return {
            'is_valid': is_acyclic,
            'is_acyclic': is_acyclic,
            'cycles': cycles,
            'warnings': warnings_list,
            'node_count': graph.number_of_nodes(),
            'edge_count': graph.number_of_edges()
        }

    # ========================================================================
    # LEGACY METHOD ALIASES (with Deprecation Warnings)
    # ========================================================================

    def extract_causal_graph(self, text: str, **kwargs) -> nx.DiGraph:
        """
        DEPRECATED: Use extract_causal_hierarchy() instead.
        """
        warnings.warn(
            "extract_causal_graph() is deprecated, use extract_causal_hierarchy() instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self.extract_causal_hierarchy(text, **kwargs)

    def get_entity_activity_pairs(self, text: str, **kwargs) -> List[Dict]:
        """
        DEPRECATED: Use extract_entity_activities() instead.
        """
        warnings.warn(
            "get_entity_activity_pairs() is deprecated, use extract_entity_activities() instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self.extract_entity_activities(text, **kwargs)

    def audit_traceability(self, nodes: Dict, **kwargs) -> Dict:
        """
        DEPRECATED: Use audit_evidence_traceability() instead.
        """
        warnings.warn(
            "audit_traceability() is deprecated, use audit_evidence_traceability() instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self.audit_evidence_traceability(nodes, **kwargs)

    def process_pdf(self, pdf_path: str, policy_code: str, **kwargs) -> Dict:
        """
        DEPRECATED: Use process_pdf_document() instead.
        """
        warnings.warn(
            "process_pdf() is deprecated, use process_pdf_document() instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self.process_pdf_document(pdf_path, policy_code, **kwargs)

    def validate_graph(self, graph: nx.DiGraph, **kwargs) -> Dict:
        """
        DEPRECATED: Use validate_dag_structure() instead.
        """
        warnings.warn(
            "validate_graph() is deprecated, use validate_dag_structure() instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self.validate_dag_structure(graph, **kwargs)

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def apply_beach_test(
        self, 
        test_type: str, 
        evidence_found: bool,
        prior: float,
        bayes_factor: float
    ) -> Tuple[float, str]:
        """Apply Beach evidential test logic"""
        return self.BeachEvidentialTest.apply_test_logic(
            test_type, evidence_found, prior, bayes_factor
        )

    def classify_evidential_test(
        self, 
        necessity: float, 
        sufficiency: float
    ) -> str:
        """Classify evidential test type based on necessity and sufficiency"""
        return self.BeachEvidentialTest.classify_test(necessity, sufficiency)

    def audit_sequence_logic(self, graph: nx.DiGraph) -> List[str]:
        """Audit temporal/logical sequence in causal graph"""
        self._ensure_processor()
        return self.processor.audit_sequence_logic(graph)

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration"""
        return self.config

    def is_available(self) -> bool:
        """Check if core module is available"""
        return self._module_available
