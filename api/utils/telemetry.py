# coding=utf-8
"""
Structured Telemetry Middleware
================================

SIN_CARRETA: Emit structured telemetry for every API request.
Captures timing, status, errors, and context for observability.

Author: FARFAN 3.0 Team
Version: 1.0.0
Python: 3.10+
"""

import time
import logging
import json
from typing import Callable, Dict, Any, Optional
from datetime import datetime
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)


class TelemetryMiddleware(BaseHTTPMiddleware):
    """
    SIN_CARRETA: Middleware to emit structured telemetry on every request
    
    Captures:
    - Request path, method, headers
    - Response status, timing
    - Errors and exceptions
    - Custom context fields
    """
    
    def __init__(self, app: ASGIApp, service_name: str = "atroz-api"):
        super().__init__(app)
        self.service_name = service_name
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request and emit telemetry
        
        Args:
            request: Incoming HTTP request
            call_next: Next middleware/handler
            
        Returns:
            HTTP response
        """
        # Start timing
        start_time = time.time()
        request_id = self._generate_request_id()
        
        # Capture request context
        request_context = self._capture_request_context(request, request_id)
        
        # Process request
        response = None
        error = None
        
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as e:
            error = e
            status_code = 500
            logger.exception(f"Request {request_id} failed with exception")
            response = JSONResponse(
                status_code=500,
                content={
                    "error": "InternalServerError",
                    "message": "An internal error occurred",
                    "request_id": request_id,
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        # Calculate timing
        duration_ms = (time.time() - start_time) * 1000
        
        # Capture response context
        response_context = self._capture_response_context(
            response, status_code, duration_ms, error
        )
        
        # Emit structured telemetry
        self._emit_telemetry(request_context, response_context)
        
        # Add telemetry headers
        if response:
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time-Ms"] = f"{duration_ms:.2f}"
        
        return response
    
    def _generate_request_id(self) -> str:
        """
        Generate unique request ID
        
        Returns:
            Request ID string
        """
        import uuid
        return str(uuid.uuid4())
    
    def _capture_request_context(self, request: Request, request_id: str) -> Dict[str, Any]:
        """
        Capture request context for telemetry
        
        Args:
            request: HTTP request
            request_id: Unique request ID
            
        Returns:
            Request context dictionary
        """
        return {
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "method": request.method,
            "path": str(request.url.path),
            "query_params": dict(request.query_params),
            "headers": {
                "user-agent": request.headers.get("user-agent", "unknown"),
                "accept": request.headers.get("accept", "unknown"),
                "content-type": request.headers.get("content-type", "unknown"),
            },
            "client": {
                "host": request.client.host if request.client else "unknown",
                "port": request.client.port if request.client else 0,
            }
        }
    
    def _capture_response_context(
        self,
        response: Optional[Response],
        status_code: int,
        duration_ms: float,
        error: Optional[Exception]
    ) -> Dict[str, Any]:
        """
        Capture response context for telemetry
        
        Args:
            response: HTTP response
            status_code: HTTP status code
            duration_ms: Request duration in milliseconds
            error: Exception if any
            
        Returns:
            Response context dictionary
        """
        context = {
            "status_code": status_code,
            "duration_ms": round(duration_ms, 2),
            "success": 200 <= status_code < 300,
        }
        
        if error:
            context["error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
        
        if response:
            context["headers"] = {
                "content-type": response.headers.get("content-type", "unknown"),
                "content-length": response.headers.get("content-length", "0"),
            }
        
        return context
    
    def _emit_telemetry(
        self,
        request_context: Dict[str, Any],
        response_context: Dict[str, Any]
    ) -> None:
        """
        Emit structured telemetry event
        
        Args:
            request_context: Request context
            response_context: Response context
        """
        telemetry = {
            "service": self.service_name,
            "event_type": "http_request",
            "request": request_context,
            "response": response_context,
        }
        
        # Log as structured JSON
        level = logging.INFO if response_context["success"] else logging.ERROR
        
        logger.log(
            level,
            f"HTTP {request_context['method']} {request_context['path']} "
            f"-> {response_context['status_code']} ({response_context['duration_ms']}ms)",
            extra={"telemetry": telemetry}
        )


class StructuredLogger:
    """
    SIN_CARRETA: Helper for emitting structured log events
    """
    
    @staticmethod
    def log_event(
        event_type: str,
        message: str,
        level: int = logging.INFO,
        **context: Any
    ) -> None:
        """
        Log structured event
        
        Args:
            event_type: Type of event
            message: Human-readable message
            level: Log level
            **context: Additional context fields
        """
        event = {
            "event_type": event_type,
            "timestamp": datetime.now().isoformat(),
            "message": message,
            **context
        }
        
        logger.log(level, message, extra={"event": event})
    
    @staticmethod
    def log_api_call(
        endpoint: str,
        method: str,
        params: Dict[str, Any],
        duration_ms: float,
        success: bool
    ) -> None:
        """
        Log API call telemetry
        
        Args:
            endpoint: API endpoint path
            method: HTTP method
            params: Request parameters
            duration_ms: Call duration
            success: Whether call succeeded
        """
        StructuredLogger.log_event(
            event_type="api_call",
            message=f"{method} {endpoint}",
            level=logging.INFO if success else logging.ERROR,
            endpoint=endpoint,
            method=method,
            params=params,
            duration_ms=round(duration_ms, 2),
            success=success
        )


def setup_logging(log_level: str = "INFO") -> None:
    """
    SIN_CARRETA: Configure structured logging
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configure JSON output for production
    # In production, you'd use a JSON formatter here
    logger.info(f"Logging configured at {log_level} level")
