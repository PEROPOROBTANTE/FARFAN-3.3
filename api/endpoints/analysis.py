# coding=utf-8
"""
Analysis Endpoints
==================

SIN_CARRETA: API endpoints for cluster and question analysis with strict contract validation.

Endpoints:
- GET /api/v1/analysis/clusters/{regionId} - Get cluster analysis for region
- GET /api/v1/analysis/questions/{municipalityId} - Get all 300 questions for municipality

Author: FARFAN 3.0 Team
Version: 1.0.0
Python: 3.10+
"""

import time
from fastapi import APIRouter, HTTPException, Path
from api.models.schemas import (
    ClusterAnalysisResponse,
    QuestionAnalysisResponse,
    DimensionEnum,
    PolicyAreaEnum,
    ErrorResponse
)
from api.utils.data_generator import get_data_generator
from api.utils.telemetry import StructuredLogger

router = APIRouter(prefix="/api/v1/analysis", tags=["Analysis"])


@router.get(
    "/clusters/{region_id}",
    response_model=ClusterAnalysisResponse,
    responses={
        200: {"description": "Cluster analysis for region"},
        400: {"model": ErrorResponse, "description": "Invalid region ID format"},
        404: {"model": ErrorResponse, "description": "Region not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    summary="Get cluster analysis for region",
    description="SIN_CARRETA: Returns cluster analysis grouping similar municipalities "
                "within a region based on their performance profiles across dimensions."
)
async def get_cluster_analysis(
    region_id: str = Path(
        ...,
        pattern=r"^REGION_\d{3}$",
        description="Region ID in format REGION_001",
        example="REGION_001"
    )
) -> ClusterAnalysisResponse:
    """
    SIN_CARRETA: Get cluster analysis for region
    
    Args:
        region_id: Region ID (REGION_001 format)
        
    Returns:
        ClusterAnalysisResponse with clusters of similar municipalities
        
    Raises:
        HTTPException: On validation errors or not found
    """
    start_time = time.time()
    
    try:
        # Validate region exists (check ID range)
        region_num = int(region_id.split("_")[1])
        if region_num < 1 or region_num > 10:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "NotFound",
                    "message": f"Region {region_id} not found. Valid range: REGION_001 to REGION_010"
                }
            )
        
        # Generate deterministic cluster analysis
        generator = get_data_generator(base_seed=42)
        clusters = generator.generate_clusters(region_id)
        
        # Generate summary
        summary = (
            f"Análisis de clustering para {region_id} identifica {len(clusters)} grupos "
            f"distintos de municipios con patrones de desarrollo similares. "
            f"Los clusters revelan heterogeneidad significativa en los perfiles de desempeño, "
            f"con variaciones notables en las dimensiones de evaluación. "
            f"Esta segmentación permite diseñar intervenciones diferenciadas según las "
            f"características específicas de cada grupo."
        )
        
        response = ClusterAnalysisResponse(
            region_id=region_id,
            clusters=clusters,
            summary=summary
        )
        
        # Emit telemetry
        duration_ms = (time.time() - start_time) * 1000
        StructuredLogger.log_api_call(
            endpoint=f"/api/v1/analysis/clusters/{region_id}",
            method="GET",
            params={"region_id": region_id},
            duration_ms=duration_ms,
            success=True
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        StructuredLogger.log_api_call(
            endpoint=f"/api/v1/analysis/clusters/{region_id}",
            method="GET",
            params={"region_id": region_id},
            duration_ms=duration_ms,
            success=False
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": f"Failed to generate cluster analysis: {str(e)}"
            }
        )


@router.get(
    "/questions/{municipality_id}",
    response_model=QuestionAnalysisResponse,
    responses={
        200: {"description": "All 300 questions for municipality"},
        400: {"model": ErrorResponse, "description": "Invalid municipality ID format"},
        404: {"model": ErrorResponse, "description": "Municipality not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    summary="Get all 300 questions for municipality",
    description="SIN_CARRETA: Returns comprehensive analysis for all 300 questions "
                "(10 policy areas × 6 dimensions × 5 questions) for a municipality."
)
async def get_question_analysis(
    municipality_id: str = Path(
        ...,
        pattern=r"^MUN_\d{5}$",
        description="Municipality ID in format MUN_00001",
        example="MUN_00101"
    )
) -> QuestionAnalysisResponse:
    """
    SIN_CARRETA: Get analysis for all 300 questions
    
    Args:
        municipality_id: Municipality ID (MUN_00001 format)
        
    Returns:
        QuestionAnalysisResponse with all 300 questions
        
    Raises:
        HTTPException: On validation errors or not found
    """
    start_time = time.time()
    
    try:
        # Validate municipality ID range
        mun_num = int(municipality_id.split("_")[1])
        region_num = mun_num // 100
        
        if region_num < 1 or region_num > 10:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "NotFound",
                    "message": f"Municipality {municipality_id} not found. Invalid region number."
                }
            )
        
        mun_idx = mun_num % 100
        if mun_idx >= 10:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "NotFound",
                    "message": f"Municipality {municipality_id} not found. Valid range: 0-9 within region."
                }
            )
        
        # Generate all 300 questions (10 policies × 6 dimensions × 5 questions)
        generator = get_data_generator(base_seed=42)
        
        all_questions = []
        by_dimension = {dim: [] for dim in DimensionEnum}
        by_policy_area = {policy: [] for policy in PolicyAreaEnum}
        
        # Generate questions for each combination
        for policy in PolicyAreaEnum:
            for dimension in DimensionEnum:
                for q_num in range(1, 6):  # 5 questions per dimension
                    # Create deterministic seed for this question
                    q_seed = generator._get_seed_for_entity(
                        f"{municipality_id}_{policy.value}_{dimension.value}_Q{q_num}"
                    )
                    
                    question = generator.generate_question_analysis(
                        policy, dimension, q_num, q_seed
                    )
                    
                    all_questions.append(question)
                    by_dimension[dimension].append(question)
                    by_policy_area[policy].append(question)
        
        response = QuestionAnalysisResponse(
            municipality_id=municipality_id,
            total_questions=len(all_questions),
            questions=all_questions,
            by_dimension=by_dimension,
            by_policy_area=by_policy_area
        )
        
        # Emit telemetry
        duration_ms = (time.time() - start_time) * 1000
        StructuredLogger.log_api_call(
            endpoint=f"/api/v1/analysis/questions/{municipality_id}",
            method="GET",
            params={"municipality_id": municipality_id},
            duration_ms=duration_ms,
            success=True
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        StructuredLogger.log_api_call(
            endpoint=f"/api/v1/analysis/questions/{municipality_id}",
            method="GET",
            params={"municipality_id": municipality_id},
            duration_ms=duration_ms,
            success=False
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": f"Failed to generate question analysis: {str(e)}"
            }
        )
