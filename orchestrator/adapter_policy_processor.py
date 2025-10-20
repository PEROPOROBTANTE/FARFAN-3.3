"""
PolicyProcessor Adapter Layer
==============================

Backward-compatible adapter wrapping policy_processor.py functionality.
Provides translation layer between legacy 11-adapter architecture and unified module controller.

This adapter preserves existing method signatures while delegating to the core domain module.
"""

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PolicyProcessorAdapter:
    """
    Adapter for policy_processor.py - Industrial Policy Processing System.
    
    Wraps ProcessorConfig, BayesianEvidenceScorer, PolicyTextProcessor,
    EvidenceBundle, IndustrialPolicyProcessor, and PolicyAnalysisPipeline.
    
    PRIMARY INTERFACE (Backward Compatible):
    - process_text(text: str) -> Dict[str, Any]
    - analyze_policy_file(file_path: str) -> Dict[str, Any]
    - extract_evidence(text: str, patterns: List[str]) -> List[Dict[str, Any]]
    - score_evidence_confidence(matches: List, context: str) -> float
    - segment_text(text: str) -> List[str]
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize adapter with optional configuration.
        
        Args:
            config: Configuration dict for processor (optional)
        """
        self._load_core_module()
        self._initialize_components(config)
        logger.info("PolicyProcessorAdapter initialized successfully")

    def _load_core_module(self):
        """Load core domain module components"""
        try:
            from policy_processor import (
                ProcessorConfig,
                BayesianEvidenceScorer,
                PolicyTextProcessor,
                EvidenceBundle,
                IndustrialPolicyProcessor,
                PolicyAnalysisPipeline
            )
            
            self.ProcessorConfig = ProcessorConfig
            self.BayesianEvidenceScorer = BayesianEvidenceScorer
            self.PolicyTextProcessor = PolicyTextProcessor
            self.EvidenceBundle = EvidenceBundle
            self.IndustrialPolicyProcessor = IndustrialPolicyProcessor
            self.PolicyAnalysisPipeline = PolicyAnalysisPipeline
            self._module_available = True
            
        except ImportError as e:
            logger.error(f"Failed to load policy_processor module: {e}")
            self._module_available = False
            raise RuntimeError(f"Core module policy_processor not available: {e}")

    def _initialize_components(self, config: Optional[Dict[str, Any]]):
        """Initialize processor components"""
        if config:
            self.config = self.ProcessorConfig(**config)
        else:
            self.config = self.ProcessorConfig()
        
        self.processor = self.IndustrialPolicyProcessor(config=self.config)
        self.pipeline = self.PolicyAnalysisPipeline(config=self.config)
        self.scorer = self.BayesianEvidenceScorer()
        self.text_processor = self.PolicyTextProcessor(self.config)

    # ========================================================================
    # PRIMARY INTERFACE (Backward Compatible)
    # ========================================================================

    def process_text(self, text: str, **kwargs) -> Dict[str, Any]:
        """
        Process policy text and extract causal evidence.
        
        Args:
            text: Raw policy document text
            **kwargs: Additional processing parameters
            
        Returns:
            Dict containing extracted evidence, confidence scores, and metadata
        """
        return self.processor.process(text)

    def analyze_policy_file(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """
        Analyze policy file and extract structured evidence.
        
        Args:
            file_path: Path to policy document file
            **kwargs: Additional analysis parameters
            
        Returns:
            Complete analysis results with evidence bundles
        """
        return self.pipeline.analyze_file(file_path)

    def extract_evidence(
        self, 
        text: str, 
        patterns: Optional[List[str]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Extract evidence matching specified patterns.
        
        Args:
            text: Text to search for evidence
            patterns: Optional list of regex patterns
            **kwargs: Additional extraction parameters
            
        Returns:
            List of evidence bundles with matches and confidence
        """
        result = self.processor.process(text)
        evidence_list = []
        
        for point_result in result.get('points', []):
            evidence_list.append({
                'dimension': point_result.get('dimension'),
                'matches': point_result.get('matches', []),
                'confidence': point_result.get('confidence', 0.0),
                'evidence': point_result.get('evidence', [])
            })
        
        return evidence_list

    def score_evidence_confidence(
        self, 
        matches: List[str], 
        context: str,
        **kwargs
    ) -> float:
        """
        Score evidence confidence using Bayesian methods.
        
        Args:
            matches: List of text matches
            context: Context for evidence evaluation
            **kwargs: Additional scoring parameters
            
        Returns:
            Confidence score [0.0, 1.0]
        """
        total_corpus_size = len(context)
        return self.scorer.compute_evidence_score(
            matches=matches,
            total_corpus_size=total_corpus_size
        )

    def segment_text(self, text: str, **kwargs) -> List[str]:
        """
        Segment text into sentences with boundary detection.
        
        Args:
            text: Text to segment
            **kwargs: Additional segmentation parameters
            
        Returns:
            List of sentence strings
        """
        return self.text_processor.segment_into_sentences(text)

    # ========================================================================
    # LEGACY METHOD ALIASES (with Deprecation Warnings)
    # ========================================================================

    def analyze_document(self, text: str, **kwargs) -> Dict[str, Any]:
        """
        DEPRECATED: Use process_text() instead.
        
        Legacy alias for backward compatibility.
        """
        warnings.warn(
            "analyze_document() is deprecated, use process_text() instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self.process_text(text, **kwargs)

    def extract_causal_evidence(self, text: str, **kwargs) -> List[Dict[str, Any]]:
        """
        DEPRECATED: Use extract_evidence() instead.
        
        Legacy alias for backward compatibility.
        """
        warnings.warn(
            "extract_causal_evidence() is deprecated, use extract_evidence() instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self.extract_evidence(text, **kwargs)

    def compute_confidence(self, matches: List, context: str, **kwargs) -> float:
        """
        DEPRECATED: Use score_evidence_confidence() instead.
        
        Legacy alias for backward compatibility.
        """
        warnings.warn(
            "compute_confidence() is deprecated, use score_evidence_confidence() instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self.score_evidence_confidence(matches, context, **kwargs)

    def split_sentences(self, text: str, **kwargs) -> List[str]:
        """
        DEPRECATED: Use segment_text() instead.
        
        Legacy alias for backward compatibility.
        """
        warnings.warn(
            "split_sentences() is deprecated, use segment_text() instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self.segment_text(text, **kwargs)

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def normalize_text(self, text: str) -> str:
        """Normalize Unicode text"""
        return self.text_processor.normalize_unicode(text)

    def extract_context_window(
        self, 
        text: str, 
        position: int, 
        window_size: int = 400
    ) -> str:
        """Extract contextual window around position"""
        return self.text_processor.extract_contextual_window(
            text, position, window_size
        )

    def validate_config(self) -> bool:
        """Validate processor configuration"""
        try:
            self.config.validate()
            return True
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return dict(self.processor.statistics)

    def is_available(self) -> bool:
        """Check if core module is available"""
        return self._module_available
