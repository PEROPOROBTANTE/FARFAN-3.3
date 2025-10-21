# coding=utf-8
"""
AtroZ Dashboard API - Main Application
=======================================

SIN_CARRETA: FastAPI application for AtroZ dashboard backend.
Implements all core data endpoints with strict contract validation,
deterministic sample data, and structured telemetry.

Author: FARFAN 3.0 Team
Version: 1.0.0
Python: 3.10+
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from datetime import datetime
import logging

from api.endpoints import pdet_regions, municipalities, analysis
from api.utils.telemetry import TelemetryMiddleware, setup_logging
from api.models.schemas import ErrorResponse, ErrorDetail

# Setup logging
setup_logging(log_level="INFO")
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="AtroZ Dashboard API",
    description="SIN_CARRETA: Core data API for AtroZ dashboard with deterministic sample data",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add telemetry middleware
app.add_middleware(TelemetryMiddleware, service_name="atroz-api")

# Include routers
app.include_router(pdet_regions.router)
app.include_router(municipalities.router)
app.include_router(analysis.router)


# ============================================================================
# EXCEPTION HANDLERS
# ============================================================================


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    SIN_CARRETA: Handle Pydantic validation errors with explicit 400 response.
    No silent fallbacks - contract violations result in clear errors.
    
    Args:
        request: HTTP request
        exc: Validation error
        
    Returns:
        JSON error response with 400 status
    """
    errors = []
    for error in exc.errors():
        errors.append(ErrorDetail(
            field=".".join(str(loc) for loc in error["loc"]),
            message=error["msg"],
            code=error["type"]
        ))
    
    logger.warning(
        f"Validation error on {request.method} {request.url.path}: {len(errors)} errors",
        extra={"validation_errors": [e.dict() for e in errors]}
    )
    
    error_response = ErrorResponse(
        error="ValidationError",
        message="Request validation failed. Check 'details' for specific errors.",
        details=errors,
        timestamp=datetime.now()
    )
    
    return JSONResponse(
        status_code=400,
        content=error_response.model_dump(mode='json')
    )


@app.exception_handler(ValidationError)
async def pydantic_validation_handler(request: Request, exc: ValidationError):
    """
    SIN_CARRETA: Handle Pydantic model validation errors
    
    Args:
        request: HTTP request
        exc: Pydantic validation error
        
    Returns:
        JSON error response with 400 status
    """
    errors = []
    for error in exc.errors():
        errors.append(ErrorDetail(
            field=".".join(str(loc) for loc in error["loc"]),
            message=error["msg"],
            code=error["type"]
        ))
    
    logger.warning(
        f"Model validation error on {request.method} {request.url.path}",
        extra={"validation_errors": [e.dict() for e in errors]}
    )
    
    error_response = ErrorResponse(
        error="ValidationError",
        message="Data validation failed. Check 'details' for specific errors.",
        details=errors,
        timestamp=datetime.now()
    )
    
    return JSONResponse(
        status_code=400,
        content=error_response.model_dump(mode='json')
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """
    SIN_CARRETA: Handle unexpected exceptions
    
    Args:
        request: HTTP request
        exc: Exception
        
    Returns:
        JSON error response with 500 status
    """
    logger.exception(
        f"Unhandled exception on {request.method} {request.url.path}",
        exc_info=exc
    )
    
    error_response = ErrorResponse(
        error="InternalServerError",
        message="An internal error occurred. Please try again later.",
        details=[ErrorDetail(
            field=None,
            message=str(exc),
            code="internal_error"
        )],
        timestamp=datetime.now()
    )
    
    return JSONResponse(
        status_code=500,
        content=error_response.model_dump(mode='json')
    )


# ============================================================================
# ROOT ENDPOINTS
# ============================================================================


@app.get("/")
async def root():
    """
    Root endpoint with API information
    
    Returns:
        API metadata
    """
    return {
        "service": "AtroZ Dashboard API",
        "version": "1.0.0",
        "description": "SIN_CARRETA: Core data API with deterministic sample data",
        "endpoints": {
            "regions": "/api/v1/pdet/regions",
            "region_detail": "/api/v1/pdet/regions/{id}",
            "municipalities": "/api/v1/pdet/regions/{id}/municipalities",
            "municipality_detail": "/api/v1/municipalities/{id}",
            "municipality_analysis": "/api/v1/municipalities/{id}/analysis",
            "cluster_analysis": "/api/v1/analysis/clusters/{regionId}",
            "question_analysis": "/api/v1/analysis/questions/{municipalityId}"
        },
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc"
        },
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health")
async def health():
    """
    Health check endpoint
    
    Returns:
        Health status
    """
    return {
        "status": "healthy",
        "service": "atroz-api",
        "timestamp": datetime.now().isoformat()
    }


# ============================================================================
# STARTUP/SHUTDOWN EVENTS
# ============================================================================


@app.on_event("startup")
async def startup_event():
    """SIN_CARRETA: Log startup"""
    logger.info("AtroZ Dashboard API starting up")
    logger.info("Deterministic data generation enabled with base seed: 42")


@app.on_event("shutdown")
async def shutdown_event():
    """SIN_CARRETA: Log shutdown"""
    logger.info("AtroZ Dashboard API shutting down")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
