# coding=utf-8
"""
Immutable Data Models - Frozen Data Structures for FARFAN 3.0
==============================================================

Defines immutable Pydantic models with frozen=True for all data contracts
between modules, adapters, and the orchestrator pipeline.

Version History:
- v1.0.0: Initial frozen models for question metadata, execution results,
          policy chunks, embeddings, and analysis results

Author: FARFAN 3.0 Team
Version: 1.0.0
Python: 3.10+
"""

from typing import Dict, Optional, Tuple, Any
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, ConfigDict
from enum import Enum


# ============================================================================
# SCHEMA VERSION CONSTANTS
# ============================================================================

QUESTION_SCHEMA_VERSION = "1.0.0"
EXECUTION_SCHEMA_VERSION = "1.0.0"
POLICY_CHUNK_SCHEMA_VERSION = "1.0.0"
EMBEDDING_SCHEMA_VERSION = "1.0.0"
ANALYSIS_SCHEMA_VERSION = "1.0.0"


# ============================================================================
# ENUMS
# ============================================================================


class ExecutionStatusEnum(str, Enum):
    """Execution status for adapter method calls"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    DEGRADED = "degraded"


class QualitativeLevelEnum(str, Enum):
    """Qualitative scoring levels"""

    EXCELENTE = "EXCELENTE"
    BUENO = "BUENO"
    SATISFACTORIO = "SATISFACTORIO"
    ACEPTABLE = "ACEPTABLE"
    INSUFICIENTE = "INSUFICIENTE"
    DEFICIENTE = "DEFICIENTE"


# ============================================================================
# QUESTION METADATA MODELS (from cuestionario.json)
# ============================================================================


class QuestionMetadata(BaseModel):
    """
    Immutable question metadata from cuestionario.json

    Schema Version: 1.0.0
    """

    model_config = ConfigDict(frozen=True)

    schema_version: str = Field(default=QUESTION_SCHEMA_VERSION)
    canonical_id: str = Field(..., description="P#-D#-Q# format (e.g., P1-D1-Q1)")
    policy_area: str = Field(..., pattern=r"^P([1-9]|10)$")
    dimension: str = Field(..., pattern=r"^D[1-6]$")
    question_number: int = Field(..., ge=1, le=5)
    question_text: str = Field(..., min_length=10)
    scoring_modality: str = Field(..., description="TYPE_A, TYPE_B, TYPE_C, TYPE_D")
    expected_elements: Tuple[str, ...] = Field(default=())
    element_weights: Dict[str, float] = Field(default_factory=dict)
    numerical_thresholds: Dict[str, float] = Field(default_factory=dict)
    verification_patterns: Tuple[str, ...] = Field(default=())
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("expected_elements", mode="before")
    @classmethod
    def convert_to_tuple(cls, v):
        """Convert list to tuple for immutability"""
        if isinstance(v, list):
            return tuple(v)
        return v

    @field_validator("verification_patterns", mode="before")
    @classmethod
    def convert_patterns_to_tuple(cls, v):
        """Convert list to tuple for immutability"""
        if isinstance(v, list):
            return tuple(v)
        return v


class ExecutionStep(BaseModel):
    """
    Immutable execution step specification

    Schema Version: 1.0.0
    """

    model_config = ConfigDict(frozen=True)

    schema_version: str = Field(default=EXECUTION_SCHEMA_VERSION)
    adapter: str = Field(..., description="Adapter name (e.g., 'teoria_cambio')")
    method: str = Field(..., description="Method name to invoke")
    args: Tuple[Any, ...] = Field(default=())
    kwargs: Dict[str, Any] = Field(default_factory=dict)
    depends_on: Tuple[str, ...] = Field(
        default=(), description="Dependencies on previous steps"
    )

    @field_validator("args", mode="before")
    @classmethod
    def convert_args_to_tuple(cls, v):
        """Convert list to tuple for immutability"""
        if isinstance(v, list):
            return tuple(v)
        return v

    @field_validator("depends_on", mode="before")
    @classmethod
    def convert_depends_to_tuple(cls, v):
        """Convert list to tuple for immutability"""
        if isinstance(v, list):
            return tuple(v)
        return v


class QuestionSpec(BaseModel):
    """
    Complete immutable question specification

    Schema Version: 1.0.0
    Combines metadata with execution chain
    """

    model_config = ConfigDict(frozen=True)

    schema_version: str = Field(default=QUESTION_SCHEMA_VERSION)
    metadata: QuestionMetadata
    execution_chain: Tuple[ExecutionStep, ...] = Field(default=())

    @property
    def canonical_id(self) -> str:
        return self.metadata.canonical_id

    @property
    def policy_area(self) -> str:
        return self.metadata.policy_area

    @property
    def dimension(self) -> str:
        return self.metadata.dimension

    @field_validator("execution_chain", mode="before")
    @classmethod
    def convert_chain_to_tuple(cls, v):
        """Convert list to tuple for immutability"""
        if isinstance(v, list):
            return tuple(v)
        return v


# ============================================================================
# POLICY PROCESSING MODELS
# ============================================================================


class PolicyChunk(BaseModel):
    """
    Immutable policy document chunk

    Schema Version: 1.0.0
    """

    model_config = ConfigDict(frozen=True)

    schema_version: str = Field(default=POLICY_CHUNK_SCHEMA_VERSION)
    chunk_id: str = Field(..., description="Unique chunk identifier")
    text: str = Field(..., min_length=1)
    start_position: int = Field(..., ge=0)
    end_position: int = Field(..., ge=0)
    section_title: Optional[str] = None
    page_number: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("end_position")
    @classmethod
    def validate_positions(cls, v, info):
        """Ensure end_position >= start_position"""
        if "start_position" in info.data and v < info.data["start_position"]:
            raise ValueError("end_position must be >= start_position")
        return v


class PolicySegment(BaseModel):
    """
    Immutable policy segment with semantic boundaries

    Schema Version: 1.0.0
    """

    model_config = ConfigDict(frozen=True)

    schema_version: str = Field(default=POLICY_CHUNK_SCHEMA_VERSION)
    segment_id: str = Field(..., description="Unique segment identifier")
    chunks: Tuple[PolicyChunk, ...] = Field(..., min_length=1)
    segment_type: str = Field(
        ..., description="Type of segment (e.g., 'diagnostic', 'objective')"
    )
    confidence: float = Field(..., ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("chunks", mode="before")
    @classmethod
    def convert_chunks_to_tuple(cls, v):
        """Convert list to tuple for immutability"""
        if isinstance(v, list):
            return tuple(v)
        return v


# ============================================================================
# EMBEDDING MODELS
# ============================================================================


class EmbeddingVector(BaseModel):
    """
    Immutable embedding vector

    Schema Version: 1.0.0
    """

    model_config = ConfigDict(frozen=True)

    schema_version: str = Field(default=EMBEDDING_SCHEMA_VERSION)
    vector_id: str = Field(..., description="Unique vector identifier")
    values: Tuple[float, ...] = Field(..., min_length=1)
    model_name: str = Field(..., description="Embedding model used")
    dimension: int = Field(..., ge=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("values", mode="before")
    @classmethod
    def convert_values_to_tuple(cls, v):
        """Convert list/array to tuple for immutability"""
        if isinstance(v, (list, tuple)):
            return tuple(float(x) for x in v)
        # Handle numpy arrays
        try:
            import numpy as np

            if isinstance(v, np.ndarray):
                return tuple(float(x) for x in v.flatten())
        except ImportError:
            pass
        return v

    @field_validator("dimension")
    @classmethod
    def validate_dimension(cls, v, info):
        """Ensure dimension matches values length"""
        if "values" in info.data and v != len(info.data["values"]):
            raise ValueError(
                f'dimension {v} does not match values length {len(info.data["values"])}'
            )
        return v


class ChunkEmbedding(BaseModel):
    """
    Immutable chunk with associated embedding

    Schema Version: 1.0.0
    """

    model_config = ConfigDict(frozen=True)

    schema_version: str = Field(default=EMBEDDING_SCHEMA_VERSION)
    chunk: PolicyChunk
    embedding: EmbeddingVector
    similarity_scores: Dict[str, float] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# EXECUTION RESULT MODELS
# ============================================================================


class Evidence(BaseModel):
    """
    Immutable evidence item

    Schema Version: 1.0.0
    """

    model_config = ConfigDict(frozen=True)

    schema_version: str = Field(default=EXECUTION_SCHEMA_VERSION)
    text: str = Field(..., min_length=1)
    source_chunk_id: Optional[str] = None
    confidence: float = Field(..., ge=0.0, le=1.0)
    position: Optional[Tuple[int, int]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ModuleResult(BaseModel):
    """
    Immutable result from module adapter method execution

    Schema Version: 1.0.0
    """

    model_config = ConfigDict(frozen=True)

    schema_version: str = Field(default=EXECUTION_SCHEMA_VERSION)
    module_name: str = Field(..., description="Name of the module adapter")
    class_name: str = Field(..., description="Adapter class name")
    method_name: str = Field(..., description="Method invoked")
    status: ExecutionStatusEnum
    data: Dict[str, Any] = Field(default_factory=dict)
    errors: Tuple[str, ...] = Field(default=())
    execution_time: float = Field(..., ge=0.0)
    evidence: Tuple[Evidence, ...] = Field(default=())
    confidence: float = Field(..., ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)

    @field_validator("errors", mode="before")
    @classmethod
    def convert_errors_to_tuple(cls, v):
        """Convert list to tuple for immutability"""
        if isinstance(v, list):
            return tuple(v)
        return v

    @field_validator("evidence", mode="before")
    @classmethod
    def convert_evidence_to_tuple(cls, v):
        """Convert list to tuple for immutability"""
        if isinstance(v, list):
            return tuple(v)
        return v


class ExecutionResult(BaseModel):
    """
    Immutable execution result from choreographer

    Schema Version: 1.0.0
    """

    model_config = ConfigDict(frozen=True)

    schema_version: str = Field(default=EXECUTION_SCHEMA_VERSION)
    module_name: str
    adapter_class: str
    method_name: str
    status: ExecutionStatusEnum
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: float = Field(..., ge=0.0)
    evidence_extracted: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# ANALYSIS RESULT MODELS
# ============================================================================


class AnalysisResult(BaseModel):
    """
    Immutable analysis result for a question

    Schema Version: 1.0.0
    """

    model_config = ConfigDict(frozen=True)

    schema_version: str = Field(default=ANALYSIS_SCHEMA_VERSION)
    question_id: str = Field(..., pattern=r"^P([1-9]|10)-D[1-6]-Q[1-5]$")
    qualitative_level: QualitativeLevelEnum
    quantitative_score: float = Field(..., ge=0.0, le=3.0)
    evidence: Tuple[Evidence, ...] = Field(default=())
    explanation: str = Field(..., min_length=100)
    confidence: float = Field(..., ge=0.0, le=1.0)
    scoring_modality: str
    elements_found: Dict[str, bool] = Field(default_factory=dict)
    modules_executed: Tuple[str, ...] = Field(default=())
    execution_time: float = Field(..., ge=0.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)

    @field_validator("evidence", mode="before")
    @classmethod
    def convert_evidence_to_tuple(cls, v):
        """Convert list to tuple for immutability"""
        if isinstance(v, list):
            return tuple(v)
        return v

    @field_validator("modules_executed", mode="before")
    @classmethod
    def convert_modules_to_tuple(cls, v):
        """Convert list to tuple for immutability"""
        if isinstance(v, list):
            return tuple(v)
        return v


class DimensionAnalysis(BaseModel):
    """
    Immutable dimension-level analysis (D1-D6)

    Schema Version: 1.0.0
    """

    model_config = ConfigDict(frozen=True)

    schema_version: str = Field(default=ANALYSIS_SCHEMA_VERSION)
    dimension_id: str = Field(..., pattern=r"^D[1-6]$")
    dimension_name: str
    avg_score: float = Field(..., ge=0.0, le=100.0)
    question_results: Tuple[AnalysisResult, ...] = Field(default=())
    strengths: Tuple[str, ...] = Field(default=())
    weaknesses: Tuple[str, ...] = Field(default=())
    recommendations: Tuple[str, ...] = Field(default=())
    confidence: float = Field(..., ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "question_results", "strengths", "weaknesses", "recommendations", mode="before"
    )
    @classmethod
    def convert_to_tuple(cls, v):
        """Convert list to tuple for immutability"""
        if isinstance(v, list):
            return tuple(v)
        return v


class PolicyAreaAnalysis(BaseModel):
    """
    Immutable policy area analysis (P1-P10)

    Schema Version: 1.0.0
    """

    model_config = ConfigDict(frozen=True)

    schema_version: str = Field(default=ANALYSIS_SCHEMA_VERSION)
    policy_area_id: str = Field(..., pattern=r"^P([1-9]|10)$")
    policy_area_name: str
    avg_score: float = Field(..., ge=0.0, le=100.0)
    dimension_scores: Dict[str, float] = Field(default_factory=dict)
    critical_gaps: Tuple[str, ...] = Field(default=())
    recommendations: Tuple[str, ...] = Field(default=())
    confidence: float = Field(..., ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("critical_gaps", "recommendations", mode="before")
    @classmethod
    def convert_to_tuple(cls, v):
        """Convert list to tuple for immutability"""
        if isinstance(v, list):
            return tuple(v)
        return v


# ============================================================================
# DOCUMENT PROCESSING MODELS
# ============================================================================


class DocumentMetadata(BaseModel):
    """
    Immutable document metadata

    Schema Version: 1.0.0
    """

    model_config = ConfigDict(frozen=True)

    document_id: str
    filename: str
    file_size: int = Field(..., ge=0)
    page_count: Optional[int] = Field(None, ge=0)
    processed_date: datetime = Field(default_factory=datetime.now)
    encoding: str = "utf-8"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ProcessedDocument(BaseModel):
    """
    Immutable processed document

    Schema Version: 1.0.0
    """

    model_config = ConfigDict(frozen=True)

    schema_version: str = Field(default=POLICY_CHUNK_SCHEMA_VERSION)
    metadata: DocumentMetadata
    raw_text: str = Field(..., min_length=1)
    chunks: Tuple[PolicyChunk, ...] = Field(default=())
    segments: Tuple[PolicySegment, ...] = Field(default=())
    embeddings: Tuple[ChunkEmbedding, ...] = Field(default=())

    @field_validator("chunks", "segments", "embeddings", mode="before")
    @classmethod
    def convert_to_tuple(cls, v):
        """Convert list to tuple for immutability"""
        if isinstance(v, list):
            return tuple(v)
        return v


# ============================================================================
# ROUTE INFORMATION MODEL
# ============================================================================


class RouteInfo(BaseModel):
    """
    Immutable route information for question routing

    Schema Version: 1.0.0
    """

    model_config = ConfigDict(frozen=True)

    question_id: str
    module_name: str
    class_name: str
    method_name: str
    dimension: str = Field(..., pattern=r"^D[1-6]$")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def dict_to_frozen_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively convert mutable dict to frozen dict

    Note: Pydantic models with frozen=True handle this automatically,
    but this utility is provided for manual conversions.
    """
    return {k: tuple(v) if isinstance(v, list) else v for k, v in d.items()}


def validate_schema_version(model_version: str, expected_version: str) -> bool:
    """
    Validate schema version compatibility

    Args:
        model_version: Version from model instance
        expected_version: Expected version constant

    Returns:
        True if compatible (major.minor match)
    """
    model_parts = model_version.split(".")
    expected_parts = expected_version.split(".")

    # Major and minor versions must match
    return model_parts[:2] == expected_parts[:2]
