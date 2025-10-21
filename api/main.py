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

import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError
from datetime import datetime
import logging

from api.endpoints import pdet_regions, municipalities, analysis, visualization, temporal, evidence, export
from api.utils.telemetry import TelemetryMiddleware, setup_logging
from api.models.schemas import ErrorResponse, ErrorDetail

# Import security and monitoring (with graceful fallback)
try:
    from api.utils.security import (
        HTTPSRedirectMiddleware,
        SecurityHeadersMiddleware,
        get_cors_config,
        get_rate_limiter,
        ComplianceHeaders
    )
    from api.utils.monitoring import get_metrics_collector
    SECURITY_ENABLED = True
except ImportError as e:
    SECURITY_ENABLED = False
    logging.warning(f"Security/monitoring modules not available: {e}")

# Setup logging
setup_logging(log_level="INFO")
logger = logging.getLogger(__name__)

# Environment detection
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
IS_PRODUCTION = ENVIRONMENT == "production"

# Create FastAPI application
app = FastAPI(
    title="AtroZ Dashboard API",
    description="SIN_CARRETA: Core data API for AtroZ dashboard with deterministic sample data, performance monitoring, and security hardening",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ============================================================================
# SECURITY MIDDLEWARE (if available)
# ============================================================================

if SECURITY_ENABLED:
    # HTTPS enforcement (production only)
    app.add_middleware(HTTPSRedirectMiddleware, enabled=IS_PRODUCTION)
    
    # Security headers
    app.add_middleware(SecurityHeadersMiddleware)
    
    # CORS configuration
    cors_config = get_cors_config()
    app.add_middleware(CORSMiddleware, **cors_config)
    
    # Rate limiting
    limiter = get_rate_limiter()
    
    logger.info(
        "Security hardening enabled",
        extra={
            "https_enforcement": IS_PRODUCTION,
            "environment": ENVIRONMENT
        }
    )
else:
    logger.warning("Security modules not loaded - running in degraded mode")

# Add telemetry middleware (includes performance monitoring)
app.add_middleware(TelemetryMiddleware, service_name="atroz-api")

# Include routers
app.include_router(pdet_regions.router)
app.include_router(municipalities.router)
app.include_router(analysis.router)
app.include_router(visualization.router)
app.include_router(temporal.router)
app.include_router(evidence.router)
app.include_router(export.router)


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
            "question_analysis": "/api/v1/analysis/questions/{municipalityId}",
            "visualization": {
                "constellation": "/api/v1/visualization/constellation",
                "phylogram": "/api/v1/visualization/phylogram/{regionId}",
                "mesh": "/api/v1/visualization/mesh/{regionId}",
                "helix": "/api/v1/visualization/helix/{municipalityId}",
                "radar": "/api/v1/visualization/radar/{municipalityId}"
            },
            "temporal": {
                "timeline_regions": "/api/v1/timeline/regions/{regionId}",
                "timeline_municipalities": "/api/v1/timeline/municipalities/{municipalityId}",
                "comparison_regions": "/api/v1/comparison/regions",
                "comparison_matrix": "/api/v1/comparison/matrix",
                "historical": "/api/v1/historical/{entityType}/{id}/years/{start}/{end}"
            },
            "evidence": {
                "stream": "/api/v1/evidence/stream",
                "references": "/api/v1/documents/references/{regionId}",
                "sources": "/api/v1/documents/sources/{questionId}",
                "citations": "/api/v1/citations/{indicatorId}"
            },
            "export": {
                "dashboard": "/api/v1/export/dashboard",
                "region": "/api/v1/export/region/{id}",
                "comparison": "/api/v1/export/comparison",
                "report_generate": "/api/v1/reports/generate/{type}",
                "report_custom": "/api/v1/reports/custom"
            }
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
    SIN_CARRETA: Enhanced health check with performance metrics
    
    Rationale: Provide comprehensive health status including
    system metrics, security status, and performance indicators.
    
    Returns:
        Health status with metrics
    """
    health_data = {
        "status": "healthy",
        "service": "atroz-api",
        "version": "1.0.0",
        "environment": ENVIRONMENT,
        "timestamp": datetime.now().isoformat()
    }
    
    # Add metrics if available
    if SECURITY_ENABLED:
        try:
            metrics_collector = get_metrics_collector()
            metrics_summary = metrics_collector.get_metrics_summary()
            health_data["metrics"] = metrics_summary
            health_data["security"] = {
                "https_enforced": IS_PRODUCTION,
                "rate_limiting": True,
                "cors_enabled": True
            }
        except Exception as e:
            logger.warning(f"Failed to collect metrics for health check: {e}")
            health_data["metrics_error"] = str(e)
    
    return health_data


# ============================================================================
# STARTUP/SHUTDOWN EVENTS
# ============================================================================


@app.get("/metrics")
async def metrics():
    """
    SIN_CARRETA: Prometheus metrics endpoint
    
    Rationale: Expose performance metrics in Prometheus format
    for external monitoring and alerting systems.
    
    Returns:
        Prometheus-formatted metrics
    """
    if not SECURITY_ENABLED:
        return JSONResponse(
            status_code=503,
            content={"error": "Metrics not available - monitoring module not loaded"}
        )
    
    try:
        metrics_collector = get_metrics_collector()
        prometheus_metrics = metrics_collector.get_prometheus_metrics()
        return Response(content=prometheus_metrics, media_type="text/plain")
    except Exception as e:
        logger.exception("Failed to generate metrics")
        # Don't expose stack trace details to external users (security)
        return JSONResponse(
            status_code=500, content={"error": "Failed to generate metrics"}
        )


@app.get("/security/status")
async def security_status():
    """
    SIN_CARRETA: Security configuration status
    
    Rationale: Provide visibility into security controls for
    compliance and audit purposes.
    
    Returns:
        Security status and configuration
    """
    status = {
        "security_enabled": SECURITY_ENABLED,
        "environment": ENVIRONMENT,
        "timestamp": datetime.now().isoformat()
    }
    
    if SECURITY_ENABLED:
        status.update({
            "https_enforcement": IS_PRODUCTION,
            "rate_limiting": {
                "enabled": True,
                "default_limit": os.getenv("RATE_LIMIT_PER_MINUTE", "100") + "/minute",
                "auth_limit": os.getenv("RATE_LIMIT_AUTH_PER_MINUTE", "20") + "/minute"
            },
            "cors": {
                "enabled": True,
                "allowed_origins": os.getenv("ALLOWED_ORIGINS", "*").split(",")
            },
            "jwt": {
                "enabled": True,
                "algorithm": os.getenv("JWT_ALGORITHM", "HS256"),
                "expiration_minutes": int(os.getenv("JWT_EXPIRATION_MINUTES", "30"))
            },
            "security_headers": {
                "csp": True,
                "x_frame_options": True,
                "x_content_type_options": True,
                "referrer_policy": True,
                "permissions_policy": True
            },
            "compliance": {
                "gdpr": True,
                "colombian_law_1581": True
            }
        })
        
        # Add compliance headers
        compliance_headers = ComplianceHeaders.get_all_compliance_headers()
        status["compliance_headers"] = compliance_headers
    
    return status


@app.on_event("startup")
async def startup_event():
    """
    SIN_CARRETA: Enhanced startup logging
    
    Rationale: Log startup with full configuration visibility
    for audit and troubleshooting.
    """
    logger.info("=" * 80)
    logger.info("AtroZ Dashboard API starting up")
    logger.info(f"Environment: {ENVIRONMENT}")
    logger.info("Deterministic data generation enabled with base seed: 42")
    
    if SECURITY_ENABLED:
        logger.info("Security hardening: ENABLED")
        logger.info(f"HTTPS enforcement: {IS_PRODUCTION}")
        logger.info("Performance monitoring: ENABLED")
        logger.info("Rate limiting: ENABLED")
        logger.info("CORS protection: ENABLED")
        logger.info("JWT authentication: CONFIGURED")
    else:
        logger.warning("Security hardening: DISABLED (modules not loaded)")
    
    logger.info("=" * 80)


@app.on_event("shutdown")
async def shutdown_event():
    """
    SIN_CARRETA: Enhanced shutdown logging
    
    Rationale: Log final metrics and status before shutdown.
    """
    logger.info("=" * 80)
    logger.info("AtroZ Dashboard API shutting down")
    
    if SECURITY_ENABLED:
        try:
            metrics_collector = get_metrics_collector()
            metrics_summary = metrics_collector.get_metrics_summary()
            logger.info(
                "Final metrics",
                extra={"metrics": metrics_summary}
            )
        except Exception as e:
            logger.warning(f"Failed to collect final metrics: {e}")
    
    logger.info("=" * 80)


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
