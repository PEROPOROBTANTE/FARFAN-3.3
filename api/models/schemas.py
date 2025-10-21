# coding=utf-8
"""
API Data Schemas for AtroZ Dashboard
====================================

SIN_CARRETA: Strict Pydantic v2 schemas with contract validation.
All endpoints must validate input/output against these schemas.
No silent fallbacks - explicit 400/403 errors for violations.

Author: FARFAN 3.0 Team
Version: 1.0.0
Python: 3.10+
"""

from typing import List, Dict, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, ConfigDict
from enum import Enum


# ============================================================================
# ENUMS
# ============================================================================


class QualitativeLevelEnum(str, Enum):
    """SIN_CARRETA: Qualitative assessment levels"""
    EXCELENTE = "EXCELENTE"
    BUENO = "BUENO"
    SATISFACTORIO = "SATISFACTORIO"
    ACEPTABLE = "ACEPTABLE"
    INSUFICIENTE = "INSUFICIENTE"
    DEFICIENTE = "DEFICIENTE"


class DimensionEnum(str, Enum):
    """SIN_CARRETA: Six evaluation dimensions"""
    D1 = "D1"
    D2 = "D2"
    D3 = "D3"
    D4 = "D4"
    D5 = "D5"
    D6 = "D6"


class PolicyAreaEnum(str, Enum):
    """SIN_CARRETA: Ten policy areas"""
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"
    P4 = "P4"
    P5 = "P5"
    P6 = "P6"
    P7 = "P7"
    P8 = "P8"
    P9 = "P9"
    P10 = "P10"


# ============================================================================
# PDET REGION SCHEMAS
# ============================================================================


class RegionCoordinates(BaseModel):
    """SIN_CARRETA: Geographic coordinates"""
    model_config = ConfigDict(frozen=True)
    
    latitude: float = Field(..., ge=-4.3, le=12.6, description="Latitude within Colombia bounds")
    longitude: float = Field(..., ge=-81.8, le=-66.8, description="Longitude within Colombia bounds")


class RegionMetadata(BaseModel):
    """SIN_CARRETA: Additional region metadata"""
    model_config = ConfigDict(frozen=True)
    
    population: int = Field(..., gt=0, description="Total population")
    area_km2: float = Field(..., gt=0, description="Area in square kilometers")
    municipalities_count: int = Field(..., gt=0, description="Number of municipalities")
    creation_date: str = Field(..., description="ISO date when region was created")


class RegionSummary(BaseModel):
    """
    SIN_CARRETA: Summary of PDET region
    
    Used in list endpoints
    """
    model_config = ConfigDict(frozen=True)
    
    id: str = Field(..., pattern=r"^REGION_\d{3}$", description="Region ID (REGION_001 format)")
    name: str = Field(..., min_length=3, max_length=100, description="Region name")
    coordinates: RegionCoordinates = Field(..., description="Geographic center")
    overall_score: float = Field(..., ge=0.0, le=100.0, description="Overall assessment score")
    
    @field_validator("overall_score")
    @classmethod
    def round_score(cls, v):
        """SIN_CARRETA: Enforce 2 decimal places"""
        return round(v, 2)


class RegionDetail(BaseModel):
    """
    SIN_CARRETA: Detailed PDET region data
    
    Used in single region endpoint
    """
    model_config = ConfigDict(frozen=True)
    
    id: str = Field(..., pattern=r"^REGION_\d{3}$", description="Region ID")
    name: str = Field(..., min_length=3, max_length=100, description="Region name")
    coordinates: RegionCoordinates = Field(..., description="Geographic center")
    overall_score: float = Field(..., ge=0.0, le=100.0, description="Overall score")
    dimension_scores: Dict[DimensionEnum, float] = Field(..., description="Scores by dimension")
    policy_area_scores: Dict[PolicyAreaEnum, float] = Field(..., description="Scores by policy area")
    metadata: RegionMetadata = Field(..., description="Additional metadata")
    last_updated: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    
    @field_validator("dimension_scores")
    @classmethod
    def validate_dimension_scores(cls, v):
        """SIN_CARRETA: All dimensions must be present with valid scores"""
        required_dims = {d for d in DimensionEnum}
        provided_dims = {DimensionEnum(k) if isinstance(k, str) else k for k in v.keys()}
        
        if provided_dims != required_dims:
            missing = required_dims - provided_dims
            raise ValueError(f"Missing dimensions: {missing}")
        
        for score in v.values():
            if not (0.0 <= score <= 100.0):
                raise ValueError(f"Dimension score must be in [0, 100]: {score}")
        
        return v
    
    @field_validator("policy_area_scores")
    @classmethod
    def validate_policy_scores(cls, v):
        """SIN_CARRETA: All policy areas must be present with valid scores"""
        required_policies = {p for p in PolicyAreaEnum}
        provided_policies = {PolicyAreaEnum(k) if isinstance(k, str) else k for k in v.keys()}
        
        if provided_policies != required_policies:
            missing = required_policies - provided_policies
            raise ValueError(f"Missing policy areas: {missing}")
        
        for score in v.values():
            if not (0.0 <= score <= 100.0):
                raise ValueError(f"Policy area score must be in [0, 100]: {score}")
        
        return v


class RegionListResponse(BaseModel):
    """SIN_CARRETA: Response for GET /api/v1/pdet/regions"""
    regions: List[RegionSummary] = Field(..., min_length=1, description="List of regions")
    total: int = Field(..., ge=1, description="Total count")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")


class RegionDetailResponse(BaseModel):
    """SIN_CARRETA: Response for GET /api/v1/pdet/regions/{id}"""
    region: RegionDetail = Field(..., description="Region details")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")


# ============================================================================
# MUNICIPALITY SCHEMAS
# ============================================================================


class MunicipalityMetadata(BaseModel):
    """SIN_CARRETA: Municipality metadata"""
    model_config = ConfigDict(frozen=True)
    
    population: int = Field(..., gt=0, description="Population")
    area_km2: float = Field(..., gt=0, description="Area in km2")
    altitude_m: int = Field(..., ge=0, description="Altitude in meters")


class MunicipalitySummary(BaseModel):
    """SIN_CARRETA: Municipality summary for list endpoints"""
    model_config = ConfigDict(frozen=True)
    
    id: str = Field(..., pattern=r"^MUN_\d{5}$", description="Municipality ID")
    name: str = Field(..., min_length=3, max_length=100, description="Municipality name")
    region_id: str = Field(..., pattern=r"^REGION_\d{3}$", description="Parent region ID")
    coordinates: RegionCoordinates = Field(..., description="Geographic coordinates")
    overall_score: float = Field(..., ge=0.0, le=100.0, description="Overall score")


class MunicipalityDetail(BaseModel):
    """SIN_CARRETA: Detailed municipality data"""
    model_config = ConfigDict(frozen=True)
    
    id: str = Field(..., pattern=r"^MUN_\d{5}$", description="Municipality ID")
    name: str = Field(..., min_length=3, max_length=100, description="Municipality name")
    region_id: str = Field(..., pattern=r"^REGION_\d{3}$", description="Parent region ID")
    coordinates: RegionCoordinates = Field(..., description="Geographic coordinates")
    overall_score: float = Field(..., ge=0.0, le=100.0, description="Overall score")
    dimension_scores: Dict[DimensionEnum, float] = Field(..., description="Scores by dimension")
    policy_area_scores: Dict[PolicyAreaEnum, float] = Field(..., description="Scores by policy area")
    metadata: MunicipalityMetadata = Field(..., description="Additional metadata")
    last_updated: datetime = Field(default_factory=datetime.now, description="Last update")


class MunicipalityListResponse(BaseModel):
    """SIN_CARRETA: Response for GET /api/v1/pdet/regions/{id}/municipalities"""
    municipalities: List[MunicipalitySummary] = Field(..., description="List of municipalities")
    region_id: str = Field(..., pattern=r"^REGION_\d{3}$", description="Parent region")
    total: int = Field(..., ge=0, description="Total count")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")


class MunicipalityDetailResponse(BaseModel):
    """SIN_CARRETA: Response for GET /api/v1/municipalities/{id}"""
    municipality: MunicipalityDetail = Field(..., description="Municipality details")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")


# ============================================================================
# ANALYSIS SCHEMAS
# ============================================================================


class Evidence(BaseModel):
    """SIN_CARRETA: Evidence supporting an analysis"""
    model_config = ConfigDict(frozen=True)
    
    text: str = Field(..., min_length=10, max_length=500, description="Evidence text")
    source: str = Field(..., min_length=5, description="Evidence source")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    position: Optional[str] = Field(None, description="Position in source document")


class QuestionAnalysis(BaseModel):
    """SIN_CARRETA: Analysis for a single question"""
    model_config = ConfigDict(frozen=True)
    
    question_id: str = Field(..., pattern=r"^P([1-9]|10)-D[1-6]-Q[1-5]$", description="Question ID")
    question_text: str = Field(..., min_length=10, description="Question text")
    qualitative_level: QualitativeLevelEnum = Field(..., description="Qualitative assessment")
    quantitative_score: float = Field(..., ge=0.0, le=3.0, description="Numeric score 0-3")
    explanation: str = Field(..., min_length=50, description="Analysis explanation")
    evidence: List[Evidence] = Field(default_factory=list, description="Supporting evidence")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Analysis confidence")
    
    @field_validator("quantitative_score")
    @classmethod
    def round_score(cls, v):
        """SIN_CARRETA: Enforce 2 decimal places"""
        return round(v, 2)


class DimensionAnalysis(BaseModel):
    """SIN_CARRETA: Analysis for a dimension"""
    model_config = ConfigDict(frozen=True)
    
    dimension_id: DimensionEnum = Field(..., description="Dimension ID")
    dimension_name: str = Field(..., min_length=5, description="Dimension name")
    score: float = Field(..., ge=0.0, le=100.0, description="Dimension score")
    questions: List[QuestionAnalysis] = Field(..., min_length=5, max_length=5, description="5 questions")
    strengths: List[str] = Field(default_factory=list, description="Key strengths")
    weaknesses: List[str] = Field(default_factory=list, description="Key weaknesses")
    
    @field_validator("questions")
    @classmethod
    def validate_five_questions(cls, v):
        """SIN_CARRETA: Must have exactly 5 questions per dimension"""
        if len(v) != 5:
            raise ValueError(f"Dimension must have exactly 5 questions, got {len(v)}")
        return v


class MunicipalityAnalysisResponse(BaseModel):
    """SIN_CARRETA: Response for GET /api/v1/municipalities/{id}/analysis"""
    municipality_id: str = Field(..., pattern=r"^MUN_\d{5}$", description="Municipality ID")
    overall_score: float = Field(..., ge=0.0, le=100.0, description="Overall score")
    dimensions: List[DimensionAnalysis] = Field(..., min_length=6, max_length=6, description="6 dimensions")
    summary: str = Field(..., min_length=100, description="Analysis summary")
    timestamp: datetime = Field(default_factory=datetime.now, description="Analysis timestamp")
    
    @field_validator("dimensions")
    @classmethod
    def validate_six_dimensions(cls, v):
        """SIN_CARRETA: Must have exactly 6 dimensions"""
        if len(v) != 6:
            raise ValueError(f"Analysis must have exactly 6 dimensions, got {len(v)}")
        return v


# ============================================================================
# CLUSTER ANALYSIS SCHEMAS
# ============================================================================


class ClusterMember(BaseModel):
    """SIN_CARRETA: Municipality in a cluster"""
    model_config = ConfigDict(frozen=True)
    
    municipality_id: str = Field(..., pattern=r"^MUN_\d{5}$", description="Municipality ID")
    municipality_name: str = Field(..., min_length=3, description="Municipality name")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Similarity to cluster centroid")


class Cluster(BaseModel):
    """SIN_CARRETA: A cluster of similar municipalities"""
    model_config = ConfigDict(frozen=True)
    
    cluster_id: str = Field(..., pattern=r"^CLUSTER_\d{2}$", description="Cluster ID")
    cluster_name: str = Field(..., min_length=5, description="Descriptive cluster name")
    centroid_scores: Dict[DimensionEnum, float] = Field(..., description="Cluster centroid")
    members: List[ClusterMember] = Field(..., min_length=1, description="Cluster members")
    characteristics: List[str] = Field(..., description="Key cluster characteristics")
    
    @field_validator("centroid_scores")
    @classmethod
    def validate_centroid(cls, v):
        """SIN_CARRETA: Centroid must have all 6 dimensions"""
        required_dims = {d for d in DimensionEnum}
        provided_dims = {DimensionEnum(k) if isinstance(k, str) else k for k in v.keys()}
        
        if provided_dims != required_dims:
            raise ValueError(f"Centroid must have all 6 dimensions")
        
        return v


class ClusterAnalysisResponse(BaseModel):
    """SIN_CARRETA: Response for GET /api/v1/analysis/clusters/{regionId}"""
    region_id: str = Field(..., pattern=r"^REGION_\d{3}$", description="Region ID")
    clusters: List[Cluster] = Field(..., min_length=1, description="Identified clusters")
    summary: str = Field(..., min_length=100, description="Cluster analysis summary")
    timestamp: datetime = Field(default_factory=datetime.now, description="Analysis timestamp")


class QuestionAnalysisResponse(BaseModel):
    """SIN_CARRETA: Response for GET /api/v1/analysis/questions/{municipalityId}"""
    municipality_id: str = Field(..., pattern=r"^MUN_\d{5}$", description="Municipality ID")
    total_questions: int = Field(..., ge=300, le=300, description="Must be exactly 300 questions")
    questions: List[QuestionAnalysis] = Field(..., min_length=300, max_length=300, description="All 300 questions")
    by_dimension: Dict[DimensionEnum, List[QuestionAnalysis]] = Field(..., description="Questions grouped by dimension")
    by_policy_area: Dict[PolicyAreaEnum, List[QuestionAnalysis]] = Field(..., description="Questions grouped by policy")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    
    @field_validator("questions")
    @classmethod
    def validate_300_questions(cls, v):
        """SIN_CARRETA: Must have exactly 300 questions (10 policies × 6 dimensions × 5 questions)"""
        if len(v) != 300:
            raise ValueError(f"Must have exactly 300 questions, got {len(v)}")
        return v


# ============================================================================
# ERROR SCHEMAS
# ============================================================================


class ErrorDetail(BaseModel):
    """SIN_CARRETA: Detailed error information"""
    field: Optional[str] = Field(None, description="Field that caused the error")
    message: str = Field(..., description="Error message")
    code: str = Field(..., description="Error code")


class ErrorResponse(BaseModel):
    """SIN_CARRETA: Standard error response"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable message")
    details: List[ErrorDetail] = Field(default_factory=list, description="Detailed errors")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")


# ============================================================================
# VISUALIZATION SCHEMAS
# ============================================================================


class ConstellationNode(BaseModel):
    """SIN_CARRETA: Node in constellation visualization"""
    model_config = ConfigDict(frozen=True)
    
    region_id: str = Field(..., pattern=r"^REGION_\d{3}$", description="Region ID")
    region_name: str = Field(..., min_length=3, description="Region name")
    x: float = Field(..., ge=0.0, le=100.0, description="X coordinate (0-100)")
    y: float = Field(..., ge=0.0, le=100.0, description="Y coordinate (0-100)")
    score: float = Field(..., ge=0.0, le=100.0, description="Overall score")
    size: float = Field(..., ge=1.0, le=10.0, description="Node size indicator")


class ConstellationEdge(BaseModel):
    """SIN_CARRETA: Edge in constellation visualization"""
    model_config = ConfigDict(frozen=True)
    
    source: str = Field(..., pattern=r"^REGION_\d{3}$", description="Source region ID")
    target: str = Field(..., pattern=r"^REGION_\d{3}$", description="Target region ID")
    strength: float = Field(..., ge=0.0, le=1.0, description="Connection strength (0-1)")


class ConstellationVisualizationResponse(BaseModel):
    """SIN_CARRETA: Response for GET /api/v1/visualization/constellation"""
    nodes: List[ConstellationNode] = Field(..., min_length=1, description="Constellation nodes")
    edges: List[ConstellationEdge] = Field(..., description="Connections between regions")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")


class PhylogramNode(BaseModel):
    """SIN_CARRETA: Node in phylogram (tree) visualization"""
    model_config = ConfigDict(frozen=True)
    
    id: str = Field(..., description="Node ID")
    name: str = Field(..., min_length=3, description="Node name")
    parent_id: Optional[str] = Field(None, description="Parent node ID")
    depth: int = Field(..., ge=0, description="Depth in tree")
    score: float = Field(..., ge=0.0, le=100.0, description="Node score")


class PhylogramVisualizationResponse(BaseModel):
    """SIN_CARRETA: Response for GET /api/v1/visualization/phylogram/{regionId}"""
    region_id: str = Field(..., pattern=r"^REGION_\d{3}$", description="Region ID")
    nodes: List[PhylogramNode] = Field(..., min_length=1, description="Tree nodes")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")


class MeshNode(BaseModel):
    """SIN_CARRETA: Node in mesh visualization"""
    model_config = ConfigDict(frozen=True)
    
    municipality_id: str = Field(..., pattern=r"^MUN_\d{5}$", description="Municipality ID")
    municipality_name: str = Field(..., min_length=3, description="Municipality name")
    x: float = Field(..., ge=0.0, le=100.0, description="X coordinate")
    y: float = Field(..., ge=0.0, le=100.0, description="Y coordinate")
    z: float = Field(..., ge=0.0, le=100.0, description="Z coordinate")
    dimension_scores: Dict[DimensionEnum, float] = Field(..., description="Scores by dimension")


class MeshVisualizationResponse(BaseModel):
    """SIN_CARRETA: Response for GET /api/v1/visualization/mesh/{regionId}"""
    region_id: str = Field(..., pattern=r"^REGION_\d{3}$", description="Region ID")
    nodes: List[MeshNode] = Field(..., min_length=1, description="Mesh nodes")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")


class HelixPoint(BaseModel):
    """SIN_CARRETA: Point in helix visualization"""
    model_config = ConfigDict(frozen=True)
    
    dimension: DimensionEnum = Field(..., description="Dimension")
    dimension_name: str = Field(..., min_length=3, description="Dimension name")
    angle: float = Field(..., ge=0.0, le=360.0, description="Angle in degrees")
    height: float = Field(..., ge=0.0, le=100.0, description="Height (score)")


class HelixVisualizationResponse(BaseModel):
    """SIN_CARRETA: Response for GET /api/v1/visualization/helix/{municipalityId}"""
    municipality_id: str = Field(..., pattern=r"^MUN_\d{5}$", description="Municipality ID")
    municipality_name: str = Field(..., min_length=3, description="Municipality name")
    points: List[HelixPoint] = Field(..., min_length=6, max_length=6, description="6 dimension points")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")


class RadarAxis(BaseModel):
    """SIN_CARRETA: Axis in radar chart"""
    model_config = ConfigDict(frozen=True)
    
    policy_area: PolicyAreaEnum = Field(..., description="Policy area")
    policy_name: str = Field(..., min_length=5, description="Policy area name")
    score: float = Field(..., ge=0.0, le=100.0, description="Score")


class RadarVisualizationResponse(BaseModel):
    """SIN_CARRETA: Response for GET /api/v1/visualization/radar/{municipalityId}"""
    municipality_id: str = Field(..., pattern=r"^MUN_\d{5}$", description="Municipality ID")
    municipality_name: str = Field(..., min_length=3, description="Municipality name")
    axes: List[RadarAxis] = Field(..., min_length=10, max_length=10, description="10 policy axes")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")


# ============================================================================
# TEMPORAL SCHEMAS
# ============================================================================


class TimelineEntry(BaseModel):
    """SIN_CARRETA: Entry in timeline"""
    model_config = ConfigDict(frozen=True)
    
    timestamp: str = Field(..., description="ISO8601 timestamp")
    event_type: str = Field(..., min_length=3, description="Event type")
    description: str = Field(..., min_length=10, description="Event description")
    score: Optional[float] = Field(None, ge=0.0, le=100.0, description="Associated score")


class TimelineRegionsResponse(BaseModel):
    """SIN_CARRETA: Response for GET /api/v1/timeline/regions/{regionId}"""
    region_id: str = Field(..., pattern=r"^REGION_\d{3}$", description="Region ID")
    events: List[TimelineEntry] = Field(..., min_length=1, description="Timeline events")
    start_date: str = Field(..., description="Start date ISO8601")
    end_date: str = Field(..., description="End date ISO8601")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")


class TimelineMunicipalitiesResponse(BaseModel):
    """SIN_CARRETA: Response for GET /api/v1/timeline/municipalities/{municipalityId}"""
    municipality_id: str = Field(..., pattern=r"^MUN_\d{5}$", description="Municipality ID")
    events: List[TimelineEntry] = Field(..., min_length=1, description="Timeline events")
    start_date: str = Field(..., description="Start date ISO8601")
    end_date: str = Field(..., description="End date ISO8601")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")


class ComparisonItem(BaseModel):
    """SIN_CARRETA: Item for comparison"""
    model_config = ConfigDict(frozen=True)
    
    entity_id: str = Field(..., description="Entity ID (region or municipality)")
    entity_name: str = Field(..., min_length=3, description="Entity name")
    dimension_scores: Dict[DimensionEnum, float] = Field(..., description="Scores by dimension")
    overall_score: float = Field(..., ge=0.0, le=100.0, description="Overall score")


class ComparisonRegionsResponse(BaseModel):
    """SIN_CARRETA: Response for GET /api/v1/comparison/regions"""
    items: List[ComparisonItem] = Field(..., min_length=2, description="Items to compare")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")


class ComparisonMatrixRequest(BaseModel):
    """SIN_CARRETA: Request for POST /api/v1/comparison/matrix"""
    entity_ids: List[str] = Field(..., min_length=2, max_length=20, description="Entity IDs to compare")
    dimensions: Optional[List[DimensionEnum]] = Field(None, description="Specific dimensions to compare")


class ComparisonMatrixCell(BaseModel):
    """SIN_CARRETA: Cell in comparison matrix"""
    model_config = ConfigDict(frozen=True)
    
    row_entity: str = Field(..., description="Row entity ID")
    col_entity: str = Field(..., description="Column entity ID")
    similarity: float = Field(..., ge=0.0, le=1.0, description="Similarity score (0-1)")


class ComparisonMatrixResponse(BaseModel):
    """SIN_CARRETA: Response for POST /api/v1/comparison/matrix"""
    entity_ids: List[str] = Field(..., description="Entities in comparison")
    matrix: List[ComparisonMatrixCell] = Field(..., description="Similarity matrix")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")


class HistoricalDataPoint(BaseModel):
    """SIN_CARRETA: Historical data point"""
    model_config = ConfigDict(frozen=True)
    
    year: int = Field(..., ge=2016, le=2030, description="Year")
    dimension_scores: Dict[DimensionEnum, float] = Field(..., description="Scores by dimension")
    overall_score: float = Field(..., ge=0.0, le=100.0, description="Overall score")


class HistoricalDataResponse(BaseModel):
    """SIN_CARRETA: Response for GET /api/v1/historical/{entityType}/{id}/years/{start}/{end}"""
    entity_type: str = Field(..., pattern=r"^(region|municipality)$", description="Entity type")
    entity_id: str = Field(..., description="Entity ID")
    data_points: List[HistoricalDataPoint] = Field(..., min_length=1, description="Historical data")
    start_year: int = Field(..., ge=2016, description="Start year")
    end_year: int = Field(..., le=2030, description="End year")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")


# ============================================================================
# EVIDENCE SCHEMAS
# ============================================================================


class EvidenceStreamItem(BaseModel):
    """SIN_CARRETA: Item in evidence stream"""
    model_config = ConfigDict(frozen=True)
    
    evidence_id: str = Field(..., pattern=r"^EV_\d{6}$", description="Evidence ID")
    text: str = Field(..., min_length=10, max_length=1000, description="Evidence text")
    source: str = Field(..., min_length=5, description="Source document")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    timestamp: str = Field(..., description="ISO8601 timestamp")
    entity_id: Optional[str] = Field(None, description="Associated entity")


class EvidenceStreamResponse(BaseModel):
    """SIN_CARRETA: Response for GET /api/v1/evidence/stream"""
    items: List[EvidenceStreamItem] = Field(..., description="Evidence items")
    total: int = Field(..., ge=0, description="Total count")
    page: int = Field(..., ge=1, description="Current page")
    per_page: int = Field(..., ge=1, le=100, description="Items per page")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")


class DocumentReference(BaseModel):
    """SIN_CARRETA: Document reference"""
    model_config = ConfigDict(frozen=True)
    
    document_id: str = Field(..., pattern=r"^DOC_\d{6}$", description="Document ID")
    title: str = Field(..., min_length=5, description="Document title")
    author: str = Field(..., min_length=3, description="Document author")
    date: str = Field(..., description="Publication date ISO8601")
    url: Optional[str] = Field(None, description="Document URL")


class DocumentReferencesResponse(BaseModel):
    """SIN_CARRETA: Response for GET /api/v1/documents/references/{regionId}"""
    region_id: str = Field(..., pattern=r"^REGION_\d{3}$", description="Region ID")
    references: List[DocumentReference] = Field(..., description="Document references")
    total: int = Field(..., ge=0, description="Total count")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")


class DocumentSource(BaseModel):
    """SIN_CARRETA: Document source for question"""
    model_config = ConfigDict(frozen=True)
    
    source_id: str = Field(..., pattern=r"^SRC_\d{6}$", description="Source ID")
    document_title: str = Field(..., min_length=5, description="Document title")
    excerpt: str = Field(..., min_length=10, max_length=500, description="Relevant excerpt")
    page_number: Optional[int] = Field(None, ge=1, description="Page number")
    relevance: float = Field(..., ge=0.0, le=1.0, description="Relevance score")


class DocumentSourcesResponse(BaseModel):
    """SIN_CARRETA: Response for GET /api/v1/documents/sources/{questionId}"""
    question_id: str = Field(..., pattern=r"^P([1-9]|10)-D[1-6]-Q[1-5]$", description="Question ID")
    sources: List[DocumentSource] = Field(..., description="Document sources")
    total: int = Field(..., ge=0, description="Total count")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")


class Citation(BaseModel):
    """SIN_CARRETA: Citation for indicator"""
    model_config = ConfigDict(frozen=True)
    
    citation_id: str = Field(..., pattern=r"^CIT_\d{6}$", description="Citation ID")
    text: str = Field(..., min_length=10, description="Citation text")
    source: str = Field(..., min_length=5, description="Source")
    year: int = Field(..., ge=2000, le=2030, description="Publication year")
    citation_format: str = Field(..., description="Formatted citation (APA style)")


class CitationsResponse(BaseModel):
    """SIN_CARRETA: Response for GET /api/v1/citations/{indicatorId}"""
    indicator_id: str = Field(..., description="Indicator ID")
    citations: List[Citation] = Field(..., description="Citations")
    total: int = Field(..., ge=0, description="Total count")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")


# ============================================================================
# EXPORT/REPORTING SCHEMAS
# ============================================================================


class ExportFormat(str, Enum):
    """SIN_CARRETA: Export format options"""
    PDF = "pdf"
    XLSX = "xlsx"
    CSV = "csv"
    JSON = "json"


class ExportDashboardRequest(BaseModel):
    """SIN_CARRETA: Request for POST /api/v1/export/dashboard"""
    format: ExportFormat = Field(..., description="Export format")
    include_visualizations: bool = Field(True, description="Include visualizations")
    include_raw_data: bool = Field(False, description="Include raw data")


class ExportRegionRequest(BaseModel):
    """SIN_CARRETA: Request for POST /api/v1/export/region/{id}"""
    format: ExportFormat = Field(..., description="Export format")
    include_municipalities: bool = Field(True, description="Include municipalities")
    include_analysis: bool = Field(True, description="Include analysis")


class ExportComparisonRequest(BaseModel):
    """SIN_CARRETA: Request for POST /api/v1/export/comparison"""
    entity_ids: List[str] = Field(..., min_length=2, description="Entity IDs to compare")
    format: ExportFormat = Field(..., description="Export format")
    dimensions: Optional[List[DimensionEnum]] = Field(None, description="Specific dimensions")


class ExportResponse(BaseModel):
    """SIN_CARRETA: Response for export endpoints"""
    export_id: str = Field(..., pattern=r"^EXP_\d{8}$", description="Export ID")
    format: ExportFormat = Field(..., description="Export format")
    download_url: str = Field(..., description="Download URL")
    expires_at: str = Field(..., description="Expiration timestamp ISO8601")
    size_bytes: int = Field(..., ge=0, description="File size in bytes")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")


class ReportType(str, Enum):
    """SIN_CARRETA: Report type options"""
    EXECUTIVE_SUMMARY = "executive_summary"
    DETAILED_ANALYSIS = "detailed_analysis"
    COMPARISON = "comparison"
    TRENDS = "trends"


class CustomReportRequest(BaseModel):
    """SIN_CARRETA: Request for POST /api/v1/reports/custom"""
    title: str = Field(..., min_length=5, max_length=200, description="Report title")
    entity_ids: List[str] = Field(..., min_length=1, description="Entity IDs to include")
    sections: List[str] = Field(..., min_length=1, description="Report sections")
    format: ExportFormat = Field(ExportFormat.PDF, description="Report format")


class ReportResponse(BaseModel):
    """SIN_CARRETA: Response for report generation endpoints"""
    report_id: str = Field(..., pattern=r"^RPT_\d{8}$", description="Report ID")
    report_type: str = Field(..., description="Report type")
    download_url: str = Field(..., description="Download URL")
    expires_at: str = Field(..., description="Expiration timestamp ISO8601")
    size_bytes: int = Field(..., ge=0, description="File size in bytes")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
