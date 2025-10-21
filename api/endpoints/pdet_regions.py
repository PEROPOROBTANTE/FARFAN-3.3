# coding=utf-8
"""
PDET Regions Endpoints
======================

SIN_CARRETA: API endpoints for PDET region data with strict contract validation.

Endpoints:
- GET /api/v1/pdet/regions - List all regions
- GET /api/v1/pdet/regions/{id} - Get region details
- GET /api/v1/pdet/regions/{id}/municipalities - List region municipalities

Author: FARFAN 3.0 Team
Version: 1.0.0
Python: 3.10+
"""

import time
from typing import List
from fastapi import APIRouter, HTTPException, Path
from api.models.schemas import (
    RegionListResponse,
    RegionDetailResponse,
    MunicipalityListResponse,
    ErrorResponse
)
from api.utils.data_generator import get_data_generator
from api.utils.telemetry import StructuredLogger

router = APIRouter(prefix="/api/v1/pdet", tags=["PDET Regions"])


@router.get(
    "/regions",
    response_model=RegionListResponse,
    responses={
        200: {"description": "List of PDET regions"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    summary="List all PDET regions",
    description="SIN_CARRETA: Returns list of all PDET regions with summary data. "
                "Data is deterministically generated from base seed 42."
)
async def list_regions() -> RegionListResponse:
    """
    SIN_CARRETA: List all PDET regions
    
    Returns:
        RegionListResponse with list of regions
        
    Raises:
        HTTPException: On validation or generation errors
    """
    start_time = time.time()
    
    try:
        # Generate deterministic data
        generator = get_data_generator(base_seed=42)
        regions = generator.generate_regions(count=10)
        
        response = RegionListResponse(
            regions=regions,
            total=len(regions)
        )
        
        # Emit telemetry
        duration_ms = (time.time() - start_time) * 1000
        StructuredLogger.log_api_call(
            endpoint="/api/v1/pdet/regions",
            method="GET",
            params={},
            duration_ms=duration_ms,
            success=True
        )
        
        return response
        
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        StructuredLogger.log_api_call(
            endpoint="/api/v1/pdet/regions",
            method="GET",
            params={},
            duration_ms=duration_ms,
            success=False
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": f"Failed to generate regions: {str(e)}"
            }
        )


@router.get(
    "/regions/{region_id}",
    response_model=RegionDetailResponse,
    responses={
        200: {"description": "Region details"},
        400: {"model": ErrorResponse, "description": "Invalid region ID format"},
        404: {"model": ErrorResponse, "description": "Region not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    summary="Get region details",
    description="SIN_CARRETA: Returns detailed data for a specific PDET region. "
                "Region ID must match pattern REGION_\\d{3}."
)
async def get_region(
    region_id: str = Path(
        ...,
        pattern=r"^REGION_\d{3}$",
        description="Region ID in format REGION_001",
        example="REGION_001"
    )
) -> RegionDetailResponse:
    """
    SIN_CARRETA: Get detailed region data
    
    Args:
        region_id: Region ID (REGION_001 format)
        
    Returns:
        RegionDetailResponse with region details
        
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
        
        # Generate deterministic data
        generator = get_data_generator(base_seed=42)
        region = generator.generate_region_detail(region_id)
        
        response = RegionDetailResponse(region=region)
        
        # Emit telemetry
        duration_ms = (time.time() - start_time) * 1000
        StructuredLogger.log_api_call(
            endpoint=f"/api/v1/pdet/regions/{region_id}",
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
            endpoint=f"/api/v1/pdet/regions/{region_id}",
            method="GET",
            params={"region_id": region_id},
            duration_ms=duration_ms,
            success=False
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": f"Failed to generate region details: {str(e)}"
            }
        )


@router.get(
    "/regions/{region_id}/municipalities",
    response_model=MunicipalityListResponse,
    responses={
        200: {"description": "List of municipalities in region"},
        400: {"model": ErrorResponse, "description": "Invalid region ID format"},
        404: {"model": ErrorResponse, "description": "Region not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    summary="List municipalities in region",
    description="SIN_CARRETA: Returns list of municipalities for a specific PDET region. "
                "Data is deterministically generated based on region ID."
)
async def list_municipalities(
    region_id: str = Path(
        ...,
        pattern=r"^REGION_\d{3}$",
        description="Region ID in format REGION_001",
        example="REGION_001"
    )
) -> MunicipalityListResponse:
    """
    SIN_CARRETA: List municipalities in a region
    
    Args:
        region_id: Region ID (REGION_001 format)
        
    Returns:
        MunicipalityListResponse with list of municipalities
        
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
        
        # Generate deterministic data
        generator = get_data_generator(base_seed=42)
        municipalities = generator.generate_municipalities(region_id, count=10)
        
        response = MunicipalityListResponse(
            municipalities=municipalities,
            region_id=region_id,
            total=len(municipalities)
        )
        
        # Emit telemetry
        duration_ms = (time.time() - start_time) * 1000
        StructuredLogger.log_api_call(
            endpoint=f"/api/v1/pdet/regions/{region_id}/municipalities",
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
            endpoint=f"/api/v1/pdet/regions/{region_id}/municipalities",
            method="GET",
            params={"region_id": region_id},
            duration_ms=duration_ms,
            success=False
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": f"Failed to generate municipalities: {str(e)}"
            }
        )
