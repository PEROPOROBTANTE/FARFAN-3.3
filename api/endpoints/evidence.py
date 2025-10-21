# coding=utf-8
"""
Evidence and Documents Endpoints
=================================

SIN_CARRETA: API endpoints for evidence, documents, and citations.

Endpoints:
- GET /api/v1/evidence/stream
- GET /api/v1/documents/references/{regionId}
- GET /api/v1/documents/sources/{questionId}
- GET /api/v1/citations/{indicatorId}

Author: FARFAN 3.0 Team
Version: 1.0.0
Python: 3.10+
"""

import time
from fastapi import APIRouter, HTTPException, Path, Query
from api.models.schemas import (
    EvidenceStreamResponse,
    DocumentReferencesResponse,
    DocumentSourcesResponse,
    CitationsResponse,
    ErrorResponse
)
from api.utils.data_generator import get_data_generator
from api.utils.telemetry import StructuredLogger

router = APIRouter(prefix="/api/v1", tags=["Evidence"])


@router.get(
    "/evidence/stream",
    response_model=EvidenceStreamResponse,
    responses={
        200: {"description": "Evidence stream data"},
        400: {"model": ErrorResponse, "description": "Invalid parameters"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    summary="Get evidence stream",
    description="SIN_CARRETA: Returns paginated stream of evidence items with timestamps "
                "in ISO8601 format. Supports pagination via page and per_page parameters."
)
async def get_evidence_stream(
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    per_page: int = Query(20, ge=1, le=100, description="Items per page")
) -> EvidenceStreamResponse:
    """
    SIN_CARRETA: Get evidence stream
    
    Args:
        page: Page number (1-indexed)
        per_page: Items per page (1-100)
        
    Returns:
        EvidenceStreamResponse with paginated evidence items
        
    Raises:
        HTTPException: On validation or generation errors
    """
    start_time = time.time()
    
    try:
        generator = get_data_generator(base_seed=42)
        stream_data = generator.generate_evidence_stream(page, per_page)
        
        # Emit telemetry
        duration_ms = (time.time() - start_time) * 1000
        StructuredLogger.log_api_call(
            endpoint="/api/v1/evidence/stream",
            method="GET",
            params={"page": page, "per_page": per_page},
            duration_ms=duration_ms,
            success=True
        )
        
        return stream_data
        
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        StructuredLogger.log_api_call(
            endpoint="/api/v1/evidence/stream",
            method="GET",
            params={"page": page, "per_page": per_page},
            duration_ms=duration_ms,
            success=False
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": f"Failed to generate evidence stream: {str(e)}"
            }
        )


@router.get(
    "/documents/references/{region_id}",
    response_model=DocumentReferencesResponse,
    responses={
        200: {"description": "Document references for region"},
        400: {"model": ErrorResponse, "description": "Invalid region ID format"},
        404: {"model": ErrorResponse, "description": "Region not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    summary="Get document references for region",
    description="SIN_CARRETA: Returns list of document references cited in region analysis. "
                "All dates in ISO8601 format."
)
async def get_document_references(
    region_id: str = Path(
        ...,
        pattern=r"^REGION_\d{3}$",
        description="Region ID in format REGION_001",
        example="REGION_001"
    )
) -> DocumentReferencesResponse:
    """
    SIN_CARRETA: Get document references for region
    
    Args:
        region_id: Region ID (REGION_001 format)
        
    Returns:
        DocumentReferencesResponse with document references
        
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
        references_data = generator.generate_document_references(region_id)
        
        # Emit telemetry
        duration_ms = (time.time() - start_time) * 1000
        StructuredLogger.log_api_call(
            endpoint=f"/api/v1/documents/references/{region_id}",
            method="GET",
            params={"region_id": region_id},
            duration_ms=duration_ms,
            success=True
        )
        
        return references_data
        
    except HTTPException:
        raise
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        StructuredLogger.log_api_call(
            endpoint=f"/api/v1/documents/references/{region_id}",
            method="GET",
            params={"region_id": region_id},
            duration_ms=duration_ms,
            success=False
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": f"Failed to generate references: {str(e)}"
            }
        )


@router.get(
    "/documents/sources/{question_id}",
    response_model=DocumentSourcesResponse,
    responses={
        200: {"description": "Document sources for question"},
        400: {"model": ErrorResponse, "description": "Invalid question ID format"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    summary="Get document sources for question",
    description="SIN_CARRETA: Returns list of document sources supporting a specific question. "
                "Includes excerpts and relevance scores."
)
async def get_document_sources(
    question_id: str = Path(
        ...,
        pattern=r"^P([1-9]|10)-D[1-6]-Q[1-5]$",
        description="Question ID in format P1-D1-Q1",
        example="P1-D1-Q1"
    )
) -> DocumentSourcesResponse:
    """
    SIN_CARRETA: Get document sources for question
    
    Args:
        question_id: Question ID (P1-D1-Q1 format)
        
    Returns:
        DocumentSourcesResponse with document sources
        
    Raises:
        HTTPException: On validation or generation errors
    """
    start_time = time.time()
    
    try:
        generator = get_data_generator(base_seed=42)
        sources_data = generator.generate_document_sources(question_id)
        
        # Emit telemetry
        duration_ms = (time.time() - start_time) * 1000
        StructuredLogger.log_api_call(
            endpoint=f"/api/v1/documents/sources/{question_id}",
            method="GET",
            params={"question_id": question_id},
            duration_ms=duration_ms,
            success=True
        )
        
        return sources_data
        
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        StructuredLogger.log_api_call(
            endpoint=f"/api/v1/documents/sources/{question_id}",
            method="GET",
            params={"question_id": question_id},
            duration_ms=duration_ms,
            success=False
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": f"Failed to generate sources: {str(e)}"
            }
        )


@router.get(
    "/citations/{indicator_id}",
    response_model=CitationsResponse,
    responses={
        200: {"description": "Citations for indicator"},
        400: {"model": ErrorResponse, "description": "Invalid indicator ID"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    summary="Get citations for indicator",
    description="SIN_CARRETA: Returns list of citations supporting a specific indicator. "
                "Citations formatted in APA style."
)
async def get_citations(
    indicator_id: str = Path(
        ...,
        description="Indicator ID",
        example="IND_001"
    )
) -> CitationsResponse:
    """
    SIN_CARRETA: Get citations for indicator
    
    Args:
        indicator_id: Indicator ID
        
    Returns:
        CitationsResponse with citations
        
    Raises:
        HTTPException: On validation or generation errors
    """
    start_time = time.time()
    
    try:
        generator = get_data_generator(base_seed=42)
        citations_data = generator.generate_citations(indicator_id)
        
        # Emit telemetry
        duration_ms = (time.time() - start_time) * 1000
        StructuredLogger.log_api_call(
            endpoint=f"/api/v1/citations/{indicator_id}",
            method="GET",
            params={"indicator_id": indicator_id},
            duration_ms=duration_ms,
            success=True
        )
        
        return citations_data
        
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        StructuredLogger.log_api_call(
            endpoint=f"/api/v1/citations/{indicator_id}",
            method="GET",
            params={"indicator_id": indicator_id},
            duration_ms=duration_ms,
            success=False
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "InternalServerError",
                "message": f"Failed to generate citations: {str(e)}"
            }
        )
