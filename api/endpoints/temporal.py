# coding=utf-8
"""
Temporal Endpoints
==================

SIN_CARRETA: API endpoints for temporal data (timeline, comparison, historical).

Endpoints:
- GET /api/v1/timeline/regions/{regionId}
- GET /api/v1/timeline/municipalities/{municipalityId}
- GET /api/v1/comparison/regions
- POST /api/v1/comparison/matrix
- GET /api/v1/historical/{entityType}/{id}/years/{start}/{end}

Author: FARFAN 3.0 Team
Version: 1.0.0
Python: 3.10+
"""

import time
from fastapi import APIRouter, HTTPException, Path, Query
from api.models.schemas import (
    TimelineRegionsResponse,
    TimelineMunicipalitiesResponse,
    ComparisonRegionsResponse,
    ComparisonMatrixRequest,
    ComparisonMatrixResponse,
    HistoricalDataResponse,
    ErrorResponse
)
from api.utils.data_generator import get_data_generator
from api.utils.telemetry import StructuredLogger

router = APIRouter(prefix="/api/v1", tags=["Temporal"])


@router.get(
    "/timeline/regions/{region_id}",
    response_model=TimelineRegionsResponse,
    responses={
        200: {"description": "Timeline for region"},
        400: {"model": ErrorResponse, "description": "Invalid region ID format"},
        404: {"model": ErrorResponse, "description": "Region not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    summary="Get timeline for region",
    description="SIN_CARRETA: Returns chronological timeline of events and milestones "
                "for a region. All timestamps in ISO8601 format."
)
async def get_timeline_regions(
    region_id: str = Path(
        ...,
        pattern=r"^REGION_\d{3}$",
        description="Region ID in format REGION_001",
        example="REGION_001"
    )
) -> TimelineRegionsResponse:
    """
    SIN_CARRETA: Get timeline for region
    
    Args:
        region_id: Region ID (REGION_001 format)
        
    Returns:
        TimelineRegionsResponse with timeline events
        
    Raises:
        HTTPException: On validation errors or not found
    """
    start_time = time.time()
    
    try:
        # Validate region exists
        region_num = int(region_id.split("_")[1])
        if region_num < 1 or region_num > 10:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "NotFound",
                    "message": f"Region {region_id} not found. Valid range: REGION_001 to REGION_010"
                }
            )
        
        generator = get_data_generator(base_seed=42)
        timeline_data = generator.generate_timeline_region(region_id)
        
        # Emit telemetry
        duration_ms = (time.time() - start_time) * 1000
        StructuredLogger.log_api_call(
            endpoint=f"/api/v1/timeline/regions/{region_id}",
            method="GET",
            params={"region_id": region_id},
            duration_ms=duration_ms,
            success=True
        )
        
        return timeline_data
        
    except HTTPException:
        raise
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        StructuredLogger.log_api_call(
            endpoint=f"/api/v1/timeline/regions/{region_id}",
            method="GET",
            params={"region_id": region_id},
            duration_ms=duration_ms,
            success=False
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": f"Failed to generate timeline: {str(e)}"
            }
        )


@router.get(
    "/timeline/municipalities/{municipality_id}",
    response_model=TimelineMunicipalitiesResponse,
    responses={
        200: {"description": "Timeline for municipality"},
        400: {"model": ErrorResponse, "description": "Invalid municipality ID format"},
        404: {"model": ErrorResponse, "description": "Municipality not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    summary="Get timeline for municipality",
    description="SIN_CARRETA: Returns chronological timeline of events and milestones "
                "for a municipality. All timestamps in ISO8601 format."
)
async def get_timeline_municipalities(
    municipality_id: str = Path(
        ...,
        pattern=r"^MUN_\d{5}$",
        description="Municipality ID in format MUN_00001",
        example="MUN_00101"
    )
) -> TimelineMunicipalitiesResponse:
    """
    SIN_CARRETA: Get timeline for municipality
    
    Args:
        municipality_id: Municipality ID (MUN_00001 format)
        
    Returns:
        TimelineMunicipalitiesResponse with timeline events
        
    Raises:
        HTTPException: On validation errors or not found
    """
    start_time = time.time()
    
    try:
        # Validate municipality ID
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
        
        generator = get_data_generator(base_seed=42)
        timeline_data = generator.generate_timeline_municipality(municipality_id)
        
        # Emit telemetry
        duration_ms = (time.time() - start_time) * 1000
        StructuredLogger.log_api_call(
            endpoint=f"/api/v1/timeline/municipalities/{municipality_id}",
            method="GET",
            params={"municipality_id": municipality_id},
            duration_ms=duration_ms,
            success=True
        )
        
        return timeline_data
        
    except HTTPException:
        raise
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        StructuredLogger.log_api_call(
            endpoint=f"/api/v1/timeline/municipalities/{municipality_id}",
            method="GET",
            params={"municipality_id": municipality_id},
            duration_ms=duration_ms,
            success=False
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": f"Failed to generate timeline: {str(e)}"
            }
        )


@router.get(
    "/comparison/regions",
    response_model=ComparisonRegionsResponse,
    responses={
        200: {"description": "Comparison of all regions"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    summary="Get comparison of all regions",
    description="SIN_CARRETA: Returns comparison data for all regions with scores "
                "across all dimensions for side-by-side analysis."
)
async def get_comparison_regions() -> ComparisonRegionsResponse:
    """
    SIN_CARRETA: Get comparison of all regions
    
    Returns:
        ComparisonRegionsResponse with comparison items
        
    Raises:
        HTTPException: On validation or generation errors
    """
    start_time = time.time()
    
    try:
        generator = get_data_generator(base_seed=42)
        comparison_data = generator.generate_comparison_regions()
        
        # Emit telemetry
        duration_ms = (time.time() - start_time) * 1000
        StructuredLogger.log_api_call(
            endpoint="/api/v1/comparison/regions",
            method="GET",
            params={},
            duration_ms=duration_ms,
            success=True
        )
        
        return comparison_data
        
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        StructuredLogger.log_api_call(
            endpoint="/api/v1/comparison/regions",
            method="GET",
            params={},
            duration_ms=duration_ms,
            success=False
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": f"Failed to generate comparison: {str(e)}"
            }
        )


@router.post(
    "/comparison/matrix",
    response_model=ComparisonMatrixResponse,
    responses={
        200: {"description": "Comparison matrix"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    summary="Generate comparison matrix",
    description="SIN_CARRETA: Generates pairwise similarity matrix for specified entities. "
                "Similarity scores range from 0 (completely different) to 1 (identical)."
)
async def post_comparison_matrix(request: ComparisonMatrixRequest) -> ComparisonMatrixResponse:
    """
    SIN_CARRETA: Generate comparison matrix
    
    Args:
        request: Comparison matrix request with entity IDs
        
    Returns:
        ComparisonMatrixResponse with similarity matrix
        
    Raises:
        HTTPException: On validation or generation errors
    """
    start_time = time.time()
    
    try:
        generator = get_data_generator(base_seed=42)
        matrix_data = generator.generate_comparison_matrix(
            request.entity_ids,
            request.dimensions
        )
        
        # Emit telemetry
        duration_ms = (time.time() - start_time) * 1000
        StructuredLogger.log_api_call(
            endpoint="/api/v1/comparison/matrix",
            method="POST",
            params={"entity_count": len(request.entity_ids)},
            duration_ms=duration_ms,
            success=True
        )
        
        return matrix_data
        
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        StructuredLogger.log_api_call(
            endpoint="/api/v1/comparison/matrix",
            method="POST",
            params={"entity_count": len(request.entity_ids)},
            duration_ms=duration_ms,
            success=False
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": f"Failed to generate matrix: {str(e)}"
            }
        )


@router.get(
    "/historical/{entity_type}/{entity_id}/years/{start_year}/{end_year}",
    response_model=HistoricalDataResponse,
    responses={
        200: {"description": "Historical data"},
        400: {"model": ErrorResponse, "description": "Invalid parameters"},
        404: {"model": ErrorResponse, "description": "Entity not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    summary="Get historical data for entity",
    description="SIN_CARRETA: Returns historical score data for a region or municipality "
                "over a specified year range. Years must be in range [2016, 2030]."
)
async def get_historical_data(
    entity_type: str = Path(
        ...,
        pattern=r"^(region|municipality)$",
        description="Entity type: 'region' or 'municipality'",
        example="region"
    ),
    entity_id: str = Path(
        ...,
        description="Entity ID (REGION_001 or MUN_00001 format)",
        example="REGION_001"
    ),
    start_year: int = Path(
        ...,
        ge=2016,
        le=2030,
        description="Start year (inclusive)",
        example=2018
    ),
    end_year: int = Path(
        ...,
        ge=2016,
        le=2030,
        description="End year (inclusive)",
        example=2023
    )
) -> HistoricalDataResponse:
    """
    SIN_CARRETA: Get historical data for entity
    
    Args:
        entity_type: Type of entity (region or municipality)
        entity_id: Entity ID
        start_year: Start year
        end_year: End year
        
    Returns:
        HistoricalDataResponse with historical data points
        
    Raises:
        HTTPException: On validation errors or not found
    """
    start_time = time.time()
    
    try:
        # Validate year range
        if start_year > end_year:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "ValidationError",
                    "message": "start_year must be less than or equal to end_year"
                }
            )
        
        # Validate entity ID format
        if entity_type == "region":
            if not entity_id.startswith("REGION_"):
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "ValidationError",
                        "message": "Region ID must match pattern REGION_XXX"
                    }
                )
            region_num = int(entity_id.split("_")[1])
            if region_num < 1 or region_num > 10:
                raise HTTPException(
                    status_code=404,
                    detail={
                        "error": "NotFound",
                        "message": f"Region {entity_id} not found"
                    }
                )
        elif entity_type == "municipality":
            if not entity_id.startswith("MUN_"):
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "ValidationError",
                        "message": "Municipality ID must match pattern MUN_XXXXX"
                    }
                )
            mun_num = int(entity_id.split("_")[1])
            region_num = mun_num // 100
            if region_num < 1 or region_num > 10 or (mun_num % 100) >= 10:
                raise HTTPException(
                    status_code=404,
                    detail={
                        "error": "NotFound",
                        "message": f"Municipality {entity_id} not found"
                    }
                )
        
        generator = get_data_generator(base_seed=42)
        historical_data = generator.generate_historical_data(
            entity_type, entity_id, start_year, end_year
        )
        
        # Emit telemetry
        duration_ms = (time.time() - start_time) * 1000
        StructuredLogger.log_api_call(
            endpoint=f"/api/v1/historical/{entity_type}/{entity_id}/years/{start_year}/{end_year}",
            method="GET",
            params={
                "entity_type": entity_type,
                "entity_id": entity_id,
                "start_year": start_year,
                "end_year": end_year
            },
            duration_ms=duration_ms,
            success=True
        )
        
        return historical_data
        
    except HTTPException:
        raise
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        StructuredLogger.log_api_call(
            endpoint=f"/api/v1/historical/{entity_type}/{entity_id}/years/{start_year}/{end_year}",
            method="GET",
            params={
                "entity_type": entity_type,
                "entity_id": entity_id,
                "start_year": start_year,
                "end_year": end_year
            },
            duration_ms=duration_ms,
            success=False
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": f"Failed to generate historical data: {str(e)}"
            }
        )
