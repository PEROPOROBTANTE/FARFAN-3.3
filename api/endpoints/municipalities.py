# coding=utf-8
"""
Municipalities Endpoints
========================

SIN_CARRETA: API endpoints for municipality data with strict contract validation.

Endpoints:
- GET /api/v1/municipalities/{id} - Get municipality details
- GET /api/v1/municipalities/{id}/analysis - Get municipality analysis

Author: FARFAN 3.0 Team
Version: 1.0.0
Python: 3.10+
"""

import time
from fastapi import APIRouter, HTTPException, Path
from api.models.schemas import (
    MunicipalityDetailResponse,
    MunicipalityAnalysisResponse,
    DimensionEnum,
    ErrorResponse
)
from api.utils.data_generator import get_data_generator
from api.utils.telemetry import StructuredLogger

router = APIRouter(prefix="/api/v1/municipalities", tags=["Municipalities"])


@router.get(
    "/{municipality_id}",
    response_model=MunicipalityDetailResponse,
    responses={
        200: {"description": "Municipality details"},
        400: {"model": ErrorResponse, "description": "Invalid municipality ID format"},
        404: {"model": ErrorResponse, "description": "Municipality not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    summary="Get municipality details",
    description="SIN_CARRETA: Returns detailed data for a specific municipality. "
                "Municipality ID must match pattern MUN_\\d{5}."
)
async def get_municipality(
    municipality_id: str = Path(
        ...,
        pattern=r"^MUN_\d{5}$",
        description="Municipality ID in format MUN_00001",
        example="MUN_00101"
    )
) -> MunicipalityDetailResponse:
    """
    SIN_CARRETA: Get detailed municipality data
    
    Args:
        municipality_id: Municipality ID (MUN_00001 format)
        
    Returns:
        MunicipalityDetailResponse with municipality details
        
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
        
        # Check municipality index within region
        mun_idx = mun_num % 100
        if mun_idx >= 10:  # 10 municipalities per region (0-9)
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "NotFound",
                    "message": f"Municipality {municipality_id} not found. Valid range: 0-9 within region."
                }
            )
        
        # Generate deterministic data
        generator = get_data_generator(base_seed=42)
        municipality = generator.generate_municipality_detail(municipality_id)
        
        response = MunicipalityDetailResponse(municipality=municipality)
        
        # Emit telemetry
        duration_ms = (time.time() - start_time) * 1000
        StructuredLogger.log_api_call(
            endpoint=f"/api/v1/municipalities/{municipality_id}",
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
            endpoint=f"/api/v1/municipalities/{municipality_id}",
            method="GET",
            params={"municipality_id": municipality_id},
            duration_ms=duration_ms,
            success=False
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": f"Failed to generate municipality details: {str(e)}"
            }
        )


@router.get(
    "/{municipality_id}/analysis",
    response_model=MunicipalityAnalysisResponse,
    responses={
        200: {"description": "Municipality analysis with 6 dimensions"},
        400: {"model": ErrorResponse, "description": "Invalid municipality ID format"},
        404: {"model": ErrorResponse, "description": "Municipality not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    summary="Get municipality analysis",
    description="SIN_CARRETA: Returns comprehensive analysis for a municipality including "
                "all 6 dimensions with 5 questions each (30 total questions per dimension)."
)
async def get_municipality_analysis(
    municipality_id: str = Path(
        ...,
        pattern=r"^MUN_\d{5}$",
        description="Municipality ID in format MUN_00001",
        example="MUN_00101"
    )
) -> MunicipalityAnalysisResponse:
    """
    SIN_CARRETA: Get comprehensive municipality analysis
    
    Args:
        municipality_id: Municipality ID (MUN_00001 format)
        
    Returns:
        MunicipalityAnalysisResponse with 6 dimensions of analysis
        
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
        
        # Generate deterministic analysis data
        generator = get_data_generator(base_seed=42)
        
        # Generate all 6 dimensions
        dimensions = []
        for dim in DimensionEnum:
            dim_analysis = generator.generate_dimension_analysis(dim, municipality_id)
            dimensions.append(dim_analysis)
        
        # Calculate overall score from dimension scores
        overall_score = sum(d.score for d in dimensions) / len(dimensions)
        
        # Generate summary
        summary = (
            f"Análisis integral del municipio {municipality_id} revela un desempeño "
            f"{'destacado' if overall_score >= 70 else 'que requiere mejoras'} "
            f"con una puntuación global de {overall_score:.2f}/100. "
            f"Las dimensiones muestran variabilidad en su nivel de desarrollo, "
            f"destacándose {dimensions[0].dimension_name} como área de fortaleza, "
            f"mientras que se identifican oportunidades de mejora en "
            f"{dimensions[-1].dimension_name}."
        )
        
        response = MunicipalityAnalysisResponse(
            municipality_id=municipality_id,
            overall_score=round(overall_score, 2),
            dimensions=dimensions,
            summary=summary
        )
        
        # Emit telemetry
        duration_ms = (time.time() - start_time) * 1000
        StructuredLogger.log_api_call(
            endpoint=f"/api/v1/municipalities/{municipality_id}/analysis",
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
            endpoint=f"/api/v1/municipalities/{municipality_id}/analysis",
            method="GET",
            params={"municipality_id": municipality_id},
            duration_ms=duration_ms,
            success=False
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": f"Failed to generate municipality analysis: {str(e)}"
            }
        )
