"""
EmbeddingPolicy Adapter Layer
==============================

Backward-compatible adapter wrapping emebedding_policy.py functionality.
Provides translation layer between legacy 11-adapter architecture and unified module controller.

This adapter preserves existing method signatures while delegating to the core domain module.
"""

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class EmbeddingPolicyAdapter:
    """
    Adapter for emebedding_policy.py - Semantic Embedding System.
    
    Wraps AdvancedSemanticChunker, BayesianNumericalAnalyzer, 
    CrossEncoderReranker, and GraphHopReasoner.
    
    PRIMARY INTERFACE (Backward Compatible):
    - chunk_document(text: str, metadata: Dict) -> List[Dict[str, Any]]
    - embed_chunks(chunks: List[str]) -> NDArray
    - compute_similarity(query: str, documents: List[str]) -> NDArray
    - rerank_results(query: str, candidates: List[str]) -> List[Tuple[str, float]]
    - evaluate_policy_metric(values: List[float]) -> Dict[str, Any]
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize adapter with optional configuration.
        
        Args:
            config: Configuration dict for embedding system (optional)
        """
        self._load_core_module()
        self._initialize_components(config)
        logger.info("EmbeddingPolicyAdapter initialized successfully")

    def _load_core_module(self):
        """Load core domain module components"""
        try:
            from emebedding_policy import (
                ChunkingConfig,
                AdvancedSemanticChunker,
                BayesianNumericalAnalyzer,
                CrossEncoderReranker,
                GraphHopReasoner,
                SemanticChunk
            )
            
            self.ChunkingConfig = ChunkingConfig
            self.AdvancedSemanticChunker = AdvancedSemanticChunker
            self.BayesianNumericalAnalyzer = BayesianNumericalAnalyzer
            self.CrossEncoderReranker = CrossEncoderReranker
            self.GraphHopReasoner = GraphHopReasoner
            self.SemanticChunk = SemanticChunk
            self._module_available = True
            
        except ImportError as e:
            logger.error(f"Failed to load emebedding_policy module: {e}")
            self._module_available = False
            raise RuntimeError(f"Core module emebedding_policy not available: {e}")

    def _initialize_components(self, config: Optional[Dict[str, Any]]):
        """Initialize embedding components"""
        if config:
            self.chunking_config = self.ChunkingConfig(**config)
        else:
            self.chunking_config = self.ChunkingConfig()
        
        self.chunker = self.AdvancedSemanticChunker(self.chunking_config)
        self.bayesian_analyzer = self.BayesianNumericalAnalyzer()
        self.reranker = None  # Lazy loaded
        self.graph_reasoner = None  # Lazy loaded
        self._embedding_model = None  # Lazy loaded

    def _ensure_reranker(self):
        """Lazy load reranker"""
        if self.reranker is None:
            self.reranker = self.CrossEncoderReranker()

    def _ensure_graph_reasoner(self):
        """Lazy load graph reasoner"""
        if self.graph_reasoner is None:
            self.graph_reasoner = self.GraphHopReasoner()

    def _ensure_embedding_model(self):
        """Lazy load embedding model"""
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer(
                "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
            )

    # ========================================================================
    # PRIMARY INTERFACE (Backward Compatible)
    # ========================================================================

    def chunk_document(
        self, 
        text: str, 
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Chunk document with semantic awareness.
        
        Args:
            text: Document text to chunk
            metadata: Document metadata
            **kwargs: Additional chunking parameters
            
        Returns:
            List of semantic chunks with metadata
        """
        if metadata is None:
            metadata = {'doc_id': 'unknown'}
        
        chunks = self.chunker.chunk_document(text, metadata)
        
        # Convert to dict format for backward compatibility
        return [
            {
                'chunk_id': chunk['chunk_id'],
                'content': chunk['content'],
                'metadata': chunk['metadata'],
                'token_count': chunk['token_count'],
                'position': chunk['position'],
                'pdq_context': chunk['pdq_context']
            }
            for chunk in chunks
        ]

    def embed_chunks(
        self, 
        chunks: List[str],
        **kwargs
    ) -> NDArray[np.float32]:
        """
        Generate embeddings for text chunks.
        
        Args:
            chunks: List of text chunks to embed
            **kwargs: Additional embedding parameters
            
        Returns:
            Numpy array of embeddings [n_chunks, embedding_dim]
        """
        self._ensure_embedding_model()
        return self._embedding_model.encode(
            chunks,
            batch_size=kwargs.get('batch_size', 32),
            normalize_embeddings=True
        )

    def compute_similarity(
        self, 
        query: str, 
        documents: List[str],
        **kwargs
    ) -> NDArray[np.float32]:
        """
        Compute semantic similarity between query and documents.
        
        Args:
            query: Query text
            documents: List of document texts
            **kwargs: Additional similarity parameters
            
        Returns:
            Array of similarity scores [0, 1]
        """
        self._ensure_embedding_model()
        
        query_embedding = self._embedding_model.encode([query])
        doc_embeddings = self._embedding_model.encode(documents)
        
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(query_embedding, doc_embeddings)
        
        return similarities[0]

    def rerank_results(
        self, 
        query: str, 
        candidates: List[str],
        top_k: int = 10,
        **kwargs
    ) -> List[Tuple[str, float]]:
        """
        Rerank candidate documents using cross-encoder.
        
        Args:
            query: Query text
            candidates: List of candidate documents
            top_k: Number of top results to return
            **kwargs: Additional reranking parameters
            
        Returns:
            List of (document, score) tuples sorted by relevance
        """
        self._ensure_reranker()
        
        reranked = self.reranker.rerank(
            query=query,
            candidates=candidates,
            top_k=top_k
        )
        
        return [(item['text'], item['score']) for item in reranked]

    def evaluate_policy_metric(
        self, 
        values: List[float],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Bayesian evaluation of policy metrics with uncertainty quantification.
        
        Args:
            values: Observed metric values
            **kwargs: Additional evaluation parameters
            
        Returns:
            Dict with point_estimate, credible_interval, evidence_strength, etc.
        """
        evaluation = self.bayesian_analyzer.evaluate_policy_metric(values)
        
        return {
            'point_estimate': evaluation['point_estimate'],
            'credible_interval_95': evaluation['credible_interval_95'],
            'evidence_strength': evaluation['evidence_strength'],
            'numerical_coherence': evaluation['numerical_coherence']
        }

    # ========================================================================
    # LEGACY METHOD ALIASES (with Deprecation Warnings)
    # ========================================================================

    def semantic_chunk(self, text: str, metadata: Dict, **kwargs) -> List[Dict]:
        """
        DEPRECATED: Use chunk_document() instead.
        """
        warnings.warn(
            "semantic_chunk() is deprecated, use chunk_document() instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self.chunk_document(text, metadata, **kwargs)

    def generate_embeddings(self, chunks: List[str], **kwargs) -> NDArray:
        """
        DEPRECATED: Use embed_chunks() instead.
        """
        warnings.warn(
            "generate_embeddings() is deprecated, use embed_chunks() instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self.embed_chunks(chunks, **kwargs)

    def calculate_similarity(self, query: str, docs: List[str], **kwargs) -> NDArray:
        """
        DEPRECATED: Use compute_similarity() instead.
        """
        warnings.warn(
            "calculate_similarity() is deprecated, use compute_similarity() instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self.compute_similarity(query, docs, **kwargs)

    def cross_encode_rerank(
        self, 
        query: str, 
        candidates: List[str], 
        **kwargs
    ) -> List[Tuple[str, float]]:
        """
        DEPRECATED: Use rerank_results() instead.
        """
        warnings.warn(
            "cross_encode_rerank() is deprecated, use rerank_results() instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self.rerank_results(query, candidates, **kwargs)

    def bayesian_metric_eval(self, values: List[float], **kwargs) -> Dict:
        """
        DEPRECATED: Use evaluate_policy_metric() instead.
        """
        warnings.warn(
            "bayesian_metric_eval() is deprecated, use evaluate_policy_metric() instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self.evaluate_policy_metric(values, **kwargs)

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def infer_pdq_context(self, chunk_text: str) -> Optional[Dict[str, Any]]:
        """Infer P-D-Q context from chunk text"""
        return self.chunker._infer_pdq_context(chunk_text)

    def get_chunking_config(self) -> Dict[str, Any]:
        """Get current chunking configuration"""
        return {
            'chunk_size': self.chunking_config.chunk_size,
            'chunk_overlap': self.chunking_config.chunk_overlap,
            'min_chunk_size': self.chunking_config.min_chunk_size,
            'respect_boundaries': self.chunking_config.respect_boundaries
        }

    def is_available(self) -> bool:
        """Check if core module is available"""
        return self._module_available
