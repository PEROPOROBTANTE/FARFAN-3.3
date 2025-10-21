# coding=utf-8
"""
Export and Reporting Endpoints
===============================

SIN_CARRETA: API endpoints for data export and report generation.

Endpoints:
- POST /api/v1/export/dashboard
- POST /api/v1/export/region/{id}
- POST /api/v1/export/comparison
- GET /api/v1/reports/generate/{type}
- POST /api/v1/reports/custom

Author: FARFAN 3.0 Team
Version: 1.0.0
Python: 3.10+
"""

import time
from fastapi import APIRouter, HTTPException, Path
from api.models.schemas import (
    ExportDashboardRequest,
    ExportRegionRequest,
    ExportComparisonRequest,
    ExportResponse,
    ReportType,
    CustomReportRequest,
    ReportResponse,
    ErrorResponse
)
from api.utils.data_generator import get_data_generator
from api.utils.telemetry import StructuredLogger

router = APIRouter(prefix="/api/v1", tags=["Export"])


@router.post(
    "/export/dashboard",
    response_model=ExportResponse,
    responses={
        200: {"description": "Dashboard export initiated"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    summary="Export dashboard data",
    description="SIN_CARRETA: Initiates export of complete dashboard data in specified format. "
                "Returns export ID and download URL. Files expire after 24 hours."
)
async def post_export_dashboard(request: ExportDashboardRequest) -> ExportResponse:
    """
    SIN_CARRETA: Export dashboard data
    
    Args:
        request: Export request with format options
        
    Returns:
        ExportResponse with download information
        
    Raises:
        HTTPException: On validation or generation errors
    """
    start_time = time.time()
    
    try:
        generator = get_data_generator(base_seed=42)
        export_data = generator.generate_export_dashboard(
            request.format,
            request.include_visualizations,
            request.include_raw_data
        )
        
        # Emit telemetry
        duration_ms = (time.time() - start_time) * 1000
        StructuredLogger.log_api_call(
            endpoint="/api/v1/export/dashboard",
            method="POST",
            params={"format": request.format.value},
            duration_ms=duration_ms,
            success=True
        )
        
        return export_data
        
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        StructuredLogger.log_api_call(
            endpoint="/api/v1/export/dashboard",
            method="POST",
            params={"format": request.format.value},
            duration_ms=duration_ms,
            success=False
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": f"Failed to generate export: {str(e)}"
            }
        )


@router.post(
    "/export/region/{region_id}",
    response_model=ExportResponse,
    responses={
        200: {"description": "Region export initiated"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        404: {"model": ErrorResponse, "description": "Region not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    summary="Export region data",
    description="SIN_CARRETA: Initiates export of region data including municipalities and analysis. "
                "Returns export ID and download URL. Files expire after 24 hours."
)
async def post_export_region(
    region_id: str = Path(
        ...,
        pattern=r"^REGION_\d{3}$",
        description="Region ID in format REGION_001",
        example="REGION_001"
    ),
    request: ExportRegionRequest = None
) -> ExportResponse:
    """
    SIN_CARRETA: Export region data
    
    Args:
        region_id: Region ID (REGION_001 format)
        request: Export request with format options
        
    Returns:
        ExportResponse with download information
        
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
        export_data = generator.generate_export_region(
            region_id,
            request.format,
            request.include_municipalities,
            request.include_analysis
        )
        
        # Emit telemetry
        duration_ms = (time.time() - start_time) * 1000
        StructuredLogger.log_api_call(
            endpoint=f"/api/v1/export/region/{region_id}",
            method="POST",
            params={"region_id": region_id, "format": request.format.value},
            duration_ms=duration_ms,
            success=True
        )
        
        return export_data
        
    except HTTPException:
        raise
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        StructuredLogger.log_api_call(
            endpoint=f"/api/v1/export/region/{region_id}",
            method="POST",
            params={"region_id": region_id},
            duration_ms=duration_ms,
            success=False
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": f"Failed to generate export: {str(e)}"
            }
        )


@router.post(
    "/export/comparison",
    response_model=ExportResponse,
    responses={
        200: {"description": "Comparison export initiated"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    summary="Export comparison data",
    description="SIN_CARRETA: Initiates export of comparison data for multiple entities. "
                "Returns export ID and download URL. Files expire after 24 hours."
)
async def post_export_comparison(request: ExportComparisonRequest) -> ExportResponse:
    """
    SIN_CARRETA: Export comparison data
    
    Args:
        request: Export request with entity IDs and format
        
    Returns:
        ExportResponse with download information
        
    Raises:
        HTTPException: On validation or generation errors
    """
    start_time = time.time()
    
    try:
        generator = get_data_generator(base_seed=42)
        export_data = generator.generate_export_comparison(
            request.entity_ids,
            request.format,
            request.dimensions
        )
        
        # Emit telemetry
        duration_ms = (time.time() - start_time) * 1000
        StructuredLogger.log_api_call(
            endpoint="/api/v1/export/comparison",
            method="POST",
            params={"entity_count": len(request.entity_ids), "format": request.format.value},
            duration_ms=duration_ms,
            success=True
        )
        
        return export_data
        
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        StructuredLogger.log_api_call(
            endpoint="/api/v1/export/comparison",
            method="POST",
            params={"entity_count": len(request.entity_ids)},
            duration_ms=duration_ms,
            success=False
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": f"Failed to generate export: {str(e)}"
            }
        )


@router.get(
    "/reports/generate/{report_type}",
    response_model=ReportResponse,
    responses={
        200: {"description": "Report generation initiated"},
        400: {"model": ErrorResponse, "description": "Invalid report type"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    summary="Generate standard report",
    description="SIN_CARRETA: Initiates generation of standard report type. "
                "Returns report ID and download URL. Reports expire after 48 hours."
)
async def get_report_generate(
    report_type: ReportType = Path(
        ...,
        description="Report type",
        example="executive_summary"
    )
) -> ReportResponse:
    """
    SIN_CARRETA: Generate standard report
    
    Args:
        report_type: Type of report to generate
        
    Returns:
        ReportResponse with download information
        
    Raises:
        HTTPException: On validation or generation errors
    """
    start_time = time.time()
    
    try:
        generator = get_data_generator(base_seed=42)
        report_data = generator.generate_standard_report(report_type)
        
        # Emit telemetry
        duration_ms = (time.time() - start_time) * 1000
        StructuredLogger.log_api_call(
            endpoint=f"/api/v1/reports/generate/{report_type.value}",
            method="GET",
            params={"report_type": report_type.value},
            duration_ms=duration_ms,
            success=True
        )
        
        return report_data
        
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        StructuredLogger.log_api_call(
            endpoint=f"/api/v1/reports/generate/{report_type.value}",
            method="GET",
            params={"report_type": report_type.value},
            duration_ms=duration_ms,
            success=False
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": f"Failed to generate report: {str(e)}"
            }
        )


@router.post(
    "/reports/custom",
    response_model=ReportResponse,
    responses={
        200: {"description": "Custom report generation initiated"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    summary="Generate custom report",
    description="SIN_CARRETA: Initiates generation of custom report with specified sections. "
                "Returns report ID and download URL. Reports expire after 48 hours."
)
async def post_report_custom(request: CustomReportRequest) -> ReportResponse:
    """
    SIN_CARRETA: Generate custom report
    
    Args:
        request: Custom report request with sections and entities
        
    Returns:
        ReportResponse with download information
        
    Raises:
        HTTPException: On validation or generation errors
    """
    start_time = time.time()
    
    try:
        generator = get_data_generator(base_seed=42)
        report_data = generator.generate_custom_report(
            request.title,
            request.entity_ids,
            request.sections,
            request.format
        )
        
        # Emit telemetry
        duration_ms = (time.time() - start_time) * 1000
        StructuredLogger.log_api_call(
            endpoint="/api/v1/reports/custom",
            method="POST",
            params={
                "entity_count": len(request.entity_ids),
                "section_count": len(request.sections),
                "format": request.format.value
            },
            duration_ms=duration_ms,
            success=True
        )
        
        return report_data
        
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        StructuredLogger.log_api_call(
            endpoint="/api/v1/reports/custom",
            method="POST",
            params={
                "entity_count": len(request.entity_ids),
                "section_count": len(request.sections)
            },
            duration_ms=duration_ms,
            success=False
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": f"Failed to generate custom report: {str(e)}"
            }
        )
