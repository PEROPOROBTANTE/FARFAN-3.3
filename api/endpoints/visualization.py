# coding=utf-8
"""
Visualization Endpoints
=======================

SIN_CARRETA: API endpoints for visualization data (constellation, phylogram, mesh, helix, radar).

Endpoints:
- GET /api/v1/visualization/constellation
- GET /api/v1/visualization/phylogram/{regionId}
- GET /api/v1/visualization/mesh/{regionId}
- GET /api/v1/visualization/helix/{municipalityId}
- GET /api/v1/visualization/radar/{municipalityId}

Author: FARFAN 3.0 Team
Version: 1.0.0
Python: 3.10+
"""

import time
from fastapi import APIRouter, HTTPException, Path
from api.models.schemas import (
    ConstellationVisualizationResponse,
    PhylogramVisualizationResponse,
    MeshVisualizationResponse,
    HelixVisualizationResponse,
    RadarVisualizationResponse,
    ErrorResponse
)
from api.utils.data_generator import get_data_generator
from api.utils.telemetry import StructuredLogger

router = APIRouter(prefix="/api/v1/visualization", tags=["Visualization"])


@router.get(
    "/constellation",
    response_model=ConstellationVisualizationResponse,
    responses={
        200: {"description": "Constellation visualization data"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    summary="Get constellation visualization",
    description="SIN_CARRETA: Returns constellation network visualization showing all regions "
                "as nodes with edges representing relationships. Coordinates are normalized to 0-100 range."
)
async def get_constellation() -> ConstellationVisualizationResponse:
    """
    SIN_CARRETA: Get constellation visualization of all regions
    
    Returns:
        ConstellationVisualizationResponse with nodes and edges
        
    Raises:
        HTTPException: On validation or generation errors
    """
    start_time = time.time()
    
    try:
        generator = get_data_generator(base_seed=42)
        constellation_data = generator.generate_constellation()
        
        # Emit telemetry
        duration_ms = (time.time() - start_time) * 1000
        StructuredLogger.log_api_call(
            endpoint="/api/v1/visualization/constellation",
            method="GET",
            params={},
            duration_ms=duration_ms,
            success=True
        )
        
        return constellation_data
        
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        StructuredLogger.log_api_call(
            endpoint="/api/v1/visualization/constellation",
            method="GET",
            params={},
            duration_ms=duration_ms,
            success=False
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": f"Failed to generate constellation: {str(e)}"
            }
        )


@router.get(
    "/phylogram/{region_id}",
    response_model=PhylogramVisualizationResponse,
    responses={
        200: {"description": "Phylogram visualization data"},
        400: {"model": ErrorResponse, "description": "Invalid region ID format"},
        404: {"model": ErrorResponse, "description": "Region not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    summary="Get phylogram visualization for region",
    description="SIN_CARRETA: Returns tree-based (phylogram) visualization showing hierarchical "
                "structure of municipalities within a region."
)
async def get_phylogram(
    region_id: str = Path(
        ...,
        pattern=r"^REGION_\d{3}$",
        description="Region ID in format REGION_001",
        example="REGION_001"
    )
) -> PhylogramVisualizationResponse:
    """
    SIN_CARRETA: Get phylogram visualization for region
    
    Args:
        region_id: Region ID (REGION_001 format)
        
    Returns:
        PhylogramVisualizationResponse with tree nodes
        
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
        phylogram_data = generator.generate_phylogram(region_id)
        
        # Emit telemetry
        duration_ms = (time.time() - start_time) * 1000
        StructuredLogger.log_api_call(
            endpoint=f"/api/v1/visualization/phylogram/{region_id}",
            method="GET",
            params={"region_id": region_id},
            duration_ms=duration_ms,
            success=True
        )
        
        return phylogram_data
        
    except HTTPException:
        raise
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        StructuredLogger.log_api_call(
            endpoint=f"/api/v1/visualization/phylogram/{region_id}",
            method="GET",
            params={"region_id": region_id},
            duration_ms=duration_ms,
            success=False
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": f"Failed to generate phylogram: {str(e)}"
            }
        )


@router.get(
    "/mesh/{region_id}",
    response_model=MeshVisualizationResponse,
    responses={
        200: {"description": "3D mesh visualization data"},
        400: {"model": ErrorResponse, "description": "Invalid region ID format"},
        404: {"model": ErrorResponse, "description": "Region not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    summary="Get 3D mesh visualization for region",
    description="SIN_CARRETA: Returns 3D mesh visualization with municipalities positioned "
                "in 3D space based on their performance profiles. Coordinates normalized to 0-100."
)
async def get_mesh(
    region_id: str = Path(
        ...,
        pattern=r"^REGION_\d{3}$",
        description="Region ID in format REGION_001",
        example="REGION_001"
    )
) -> MeshVisualizationResponse:
    """
    SIN_CARRETA: Get 3D mesh visualization for region
    
    Args:
        region_id: Region ID (REGION_001 format)
        
    Returns:
        MeshVisualizationResponse with 3D nodes
        
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
        mesh_data = generator.generate_mesh(region_id)
        
        # Emit telemetry
        duration_ms = (time.time() - start_time) * 1000
        StructuredLogger.log_api_call(
            endpoint=f"/api/v1/visualization/mesh/{region_id}",
            method="GET",
            params={"region_id": region_id},
            duration_ms=duration_ms,
            success=True
        )
        
        return mesh_data
        
    except HTTPException:
        raise
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        StructuredLogger.log_api_call(
            endpoint=f"/api/v1/visualization/mesh/{region_id}",
            method="GET",
            params={"region_id": region_id},
            duration_ms=duration_ms,
            success=False
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": f"Failed to generate mesh: {str(e)}"
            }
        )


@router.get(
    "/helix/{municipality_id}",
    response_model=HelixVisualizationResponse,
    responses={
        200: {"description": "Helix visualization data"},
        400: {"model": ErrorResponse, "description": "Invalid municipality ID format"},
        404: {"model": ErrorResponse, "description": "Municipality not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    summary="Get helix visualization for municipality",
    description="SIN_CARRETA: Returns helix visualization showing 6 dimensions as points "
                "along a helical path. Each dimension positioned at specific angle and height."
)
async def get_helix(
    municipality_id: str = Path(
        ...,
        pattern=r"^MUN_\d{5}$",
        description="Municipality ID in format MUN_00001",
        example="MUN_00101"
    )
) -> HelixVisualizationResponse:
    """
    SIN_CARRETA: Get helix visualization for municipality
    
    Args:
        municipality_id: Municipality ID (MUN_00001 format)
        
    Returns:
        HelixVisualizationResponse with helix points
        
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
        helix_data = generator.generate_helix(municipality_id)
        
        # Emit telemetry
        duration_ms = (time.time() - start_time) * 1000
        StructuredLogger.log_api_call(
            endpoint=f"/api/v1/visualization/helix/{municipality_id}",
            method="GET",
            params={"municipality_id": municipality_id},
            duration_ms=duration_ms,
            success=True
        )
        
        return helix_data
        
    except HTTPException:
        raise
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        StructuredLogger.log_api_call(
            endpoint=f"/api/v1/visualization/helix/{municipality_id}",
            method="GET",
            params={"municipality_id": municipality_id},
            duration_ms=duration_ms,
            success=False
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": f"Failed to generate helix: {str(e)}"
            }
        )


@router.get(
    "/radar/{municipality_id}",
    response_model=RadarVisualizationResponse,
    responses={
        200: {"description": "Radar chart visualization data"},
        400: {"model": ErrorResponse, "description": "Invalid municipality ID format"},
        404: {"model": ErrorResponse, "description": "Municipality not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    summary="Get radar chart visualization for municipality",
    description="SIN_CARRETA: Returns radar chart data showing all 10 policy areas "
                "as axes with scores plotted for the municipality."
)
async def get_radar(
    municipality_id: str = Path(
        ...,
        pattern=r"^MUN_\d{5}$",
        description="Municipality ID in format MUN_00001",
        example="MUN_00101"
    )
) -> RadarVisualizationResponse:
    """
    SIN_CARRETA: Get radar chart visualization for municipality
    
    Args:
        municipality_id: Municipality ID (MUN_00001 format)
        
    Returns:
        RadarVisualizationResponse with radar axes
        
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
        radar_data = generator.generate_radar(municipality_id)
        
        # Emit telemetry
        duration_ms = (time.time() - start_time) * 1000
        StructuredLogger.log_api_call(
            endpoint=f"/api/v1/visualization/radar/{municipality_id}",
            method="GET",
            params={"municipality_id": municipality_id},
            duration_ms=duration_ms,
            success=True
        )
        
        return radar_data
        
    except HTTPException:
        raise
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        StructuredLogger.log_api_call(
            endpoint=f"/api/v1/visualization/radar/{municipality_id}",
            method="GET",
            params={"municipality_id": municipality_id},
            duration_ms=duration_ms,
            success=False
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": f"Failed to generate radar: {str(e)}"
            }
        )
