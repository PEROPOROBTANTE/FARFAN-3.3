# coding=utf-8
"""
Security Hardening Components
==============================

SIN_CARRETA: Comprehensive security controls for AtroZ Dashboard API.
Implements HTTPS enforcement, JWT authentication, CORS, XSS/CSRF protection,
rate limiting, and GDPR/Colombian law compliance.

Author: FARFAN 3.3 Team
Version: 1.0.0
Python: 3.10+
"""

import os
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

from fastapi import Request, HTTPException, status
from fastapi.responses import RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from jose import JWTError, jwt
from passlib.context import CryptContext
from slowapi import Limiter
from slowapi.util import get_remote_address

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Environment detection
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
IS_PRODUCTION = ENVIRONMENT == "production"

# JWT Configuration
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRATION_MINUTES = int(os.getenv("JWT_EXPIRATION_MINUTES", "30"))

# CORS Configuration
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

# Rate Limiting Configuration
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "100"))
RATE_LIMIT_AUTH_PER_MINUTE = int(os.getenv("RATE_LIMIT_AUTH_PER_MINUTE", "20"))

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# ============================================================================
# HTTPS ENFORCEMENT
# ============================================================================

class HTTPSRedirectMiddleware(BaseHTTPMiddleware):
    """
    SIN_CARRETA: Enforce HTTPS in production
    
    Rationale: Prevent man-in-the-middle attacks by ensuring all traffic
    uses TLS encryption. Only active in production to avoid dev friction.
    """
    
    def __init__(self, app: ASGIApp, enabled: bool = True):
        super().__init__(app)
        self.enabled = enabled and IS_PRODUCTION
        
        if self.enabled:
            logger.info("HTTPS enforcement enabled (production mode)")
        else:
            logger.info("HTTPS enforcement disabled (development mode)")
    
    async def dispatch(self, request: Request, call_next):
        """
        Redirect HTTP to HTTPS in production
        
        Args:
            request: Incoming request
            call_next: Next middleware
            
        Returns:
            Response or redirect
        """
        if self.enabled:
            # Check if request is HTTP
            if request.url.scheme == "http":
                # Build HTTPS URL
                https_url = request.url.replace(scheme="https")
                
                logger.warning(
                    f"HTTP request redirected to HTTPS: {request.url}",
                    extra={
                        "event_type": "security",
                        "action": "https_redirect",
                        "client_ip": request.client.host if request.client else "unknown"
                    }
                )
                
                return RedirectResponse(url=str(https_url), status_code=301)
        
        return await call_next(request)


# ============================================================================
# JWT AUTHENTICATION
# ============================================================================

class JWTAuth:
    """
    SIN_CARRETA: JWT token management
    
    Rationale: Implement secure authentication with configurable expiration
    to prevent unauthorized access and session hijacking.
    """
    
    @staticmethod
    def create_access_token(
        data: Dict[str, Any],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Create JWT access token
        
        Args:
            data: Token payload data
            expires_delta: Optional custom expiration
            
        Returns:
            Encoded JWT token
        """
        to_encode = data.copy()
        
        # Set expiration
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=JWT_EXPIRATION_MINUTES)
        
        to_encode.update({"exp": expire, "iat": datetime.utcnow()})
        
        # Encode token
        encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
        
        logger.info(
            "JWT token created",
            extra={
                "event_type": "security",
                "action": "jwt_create",
                "expires_at": expire.isoformat(),
                "subject": data.get("sub", "unknown")
            }
        )
        
        return encoded_jwt
    
    @staticmethod
    def verify_token(token: str) -> Dict[str, Any]:
        """
        Verify and decode JWT token
        
        Args:
            token: JWT token to verify
            
        Returns:
            Decoded token payload
            
        Raises:
            HTTPException: If token invalid or expired
        """
        try:
            payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            
            # Check expiration
            exp = payload.get("exp")
            if exp and datetime.utcfromtimestamp(exp) < datetime.utcnow():
                logger.warning(
                    "Expired JWT token rejected",
                    extra={
                        "event_type": "security",
                        "action": "jwt_expired",
                        "expired_at": datetime.utcfromtimestamp(exp).isoformat()
                    }
                )
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has expired"
                )
            
            logger.debug(
                "JWT token verified",
                extra={
                    "event_type": "security",
                    "action": "jwt_verify",
                    "subject": payload.get("sub", "unknown")
                }
            )
            
            return payload
            
        except JWTError as e:
            logger.warning(
                f"Invalid JWT token rejected: {str(e)}",
                extra={
                    "event_type": "security",
                    "action": "jwt_invalid",
                    "error": str(e)
                }
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication token"
            )
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password for storage"""
        return pwd_context.hash(password)
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return pwd_context.verify(plain_password, hashed_password)


# ============================================================================
# CORS CONFIGURATION
# ============================================================================

def get_cors_config() -> Dict[str, Any]:
    """
    SIN_CARRETA: Get CORS configuration
    
    Rationale: Configure CORS to prevent unauthorized cross-origin requests
    while allowing legitimate frontend access.
    
    Returns:
        CORS configuration dictionary
    """
    config = {
        "allow_origins": ALLOWED_ORIGINS,
        "allow_credentials": True,
        "allow_methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
        "allow_headers": [
            "Content-Type",
            "Authorization",
            "X-Request-ID",
            "X-API-Key",
            "Accept",
            "Origin",
            "User-Agent"
        ],
        "expose_headers": [
            "X-Request-ID",
            "X-Response-Time-Ms",
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
            "X-RateLimit-Reset"
        ],
        "max_age": 600  # 10 minutes
    }
    
    logger.info(
        "CORS configured",
        extra={
            "environment": ENVIRONMENT,
            "allowed_origins": ALLOWED_ORIGINS
        }
    )
    
    return config


# ============================================================================
# RATE LIMITING
# ============================================================================

def get_rate_limiter() -> Limiter:
    """
    SIN_CARRETA: Create rate limiter
    
    Rationale: Prevent abuse and DDoS attacks by limiting requests per IP.
    Uses in-memory storage for simplicity; Redis for production scale.
    
    Returns:
        Configured rate limiter
    """
    limiter = Limiter(
        key_func=get_remote_address,
        default_limits=[f"{RATE_LIMIT_PER_MINUTE}/minute"],
        storage_uri="memory://"  # Use Redis in production: "redis://localhost:6379"
    )
    
    logger.info(
        "Rate limiter configured",
        extra={
            "default_limit": f"{RATE_LIMIT_PER_MINUTE}/minute",
            "auth_limit": f"{RATE_LIMIT_AUTH_PER_MINUTE}/minute"
        }
    )
    
    return limiter


# ============================================================================
# SECURITY HEADERS
# ============================================================================

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    SIN_CARRETA: Add security headers to all responses
    
    Rationale: Implement defense-in-depth with multiple security headers
    to prevent XSS, clickjacking, MIME sniffing, and other attacks.
    """
    
    async def dispatch(self, request: Request, call_next):
        """
        Add security headers to response
        
        Args:
            request: Incoming request
            call_next: Next middleware
            
        Returns:
            Response with security headers
        """
        response = await call_next(request)
        
        # Content Security Policy (XSS protection)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self' data:; "
            "connect-src 'self' https:; "
            "frame-ancestors 'none';"
        )
        
        # Prevent clickjacking
        response.headers["X-Frame-Options"] = "DENY"
        
        # Prevent MIME type sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"
        
        # XSS protection (legacy, but still useful)
        response.headers["X-XSS-Protection"] = "1; mode=block"
        
        # Referrer policy
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # Permissions policy (formerly Feature-Policy)
        response.headers["Permissions-Policy"] = (
            "geolocation=(), "
            "microphone=(), "
            "camera=(), "
            "payment=(), "
            "usb=(), "
            "magnetometer=()"
        )
        
        # HSTS (only in production with HTTPS)
        if IS_PRODUCTION:
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains; preload"
            )
        
        # GDPR/Privacy headers
        response.headers["X-Privacy-Policy"] = "https://example.com/privacy"
        response.headers["X-Data-Protection"] = "GDPR-compliant"
        
        # Cookie security (SameSite for CSRF protection)
        if "Set-Cookie" in response.headers:
            cookie = response.headers["Set-Cookie"]
            if "SameSite" not in cookie:
                response.headers["Set-Cookie"] = f"{cookie}; SameSite=Strict; Secure; HttpOnly"
        
        return response


# ============================================================================
# WEBSOCKET SECURITY
# ============================================================================

class WebSocketAuthenticator:
    """
    SIN_CARRETA: WebSocket authentication
    
    Rationale: Secure WebSocket connections with token-based auth
    to prevent unauthorized real-time data access.
    """
    
    @staticmethod
    async def authenticate_websocket(token: Optional[str]) -> Dict[str, Any]:
        """
        Authenticate WebSocket connection
        
        Args:
            token: JWT token from query or header
            
        Returns:
            User data from token
            
        Raises:
            HTTPException: If authentication fails
        """
        if not token:
            logger.warning(
                "WebSocket connection rejected: No token provided",
                extra={
                    "event_type": "security",
                    "action": "ws_auth_failed",
                    "reason": "missing_token"
                }
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required for WebSocket connection"
            )
        
        try:
            # Verify token
            payload = JWTAuth.verify_token(token)
            
            logger.info(
                "WebSocket authenticated",
                extra={
                    "event_type": "security",
                    "action": "ws_auth_success",
                    "subject": payload.get("sub", "unknown")
                }
            )
            
            return payload
            
        except HTTPException:
            logger.warning(
                "WebSocket connection rejected: Invalid token",
                extra={
                    "event_type": "security",
                    "action": "ws_auth_failed",
                    "reason": "invalid_token"
                }
            )
            raise


# ============================================================================
# COMPLIANCE UTILITIES
# ============================================================================

class ComplianceHeaders:
    """
    SIN_CARRETA: GDPR and Colombian law compliance headers
    
    Rationale: Ensure compliance with data protection regulations
    through proper headers and documentation.
    """
    
    @staticmethod
    def get_gdpr_headers() -> Dict[str, str]:
        """
        Get GDPR compliance headers
        
        Returns:
            Dictionary of GDPR headers
        """
        return {
            "X-Privacy-Policy": "https://example.com/privacy",
            "X-Data-Subject-Rights": "access,rectification,erasure,restriction,portability,objection",
            "X-Data-Controller": "AtroZ Dashboard Team",
            "X-Data-Retention": "30 days",
            "X-Data-Processing-Lawful-Basis": "consent,legitimate-interest"
        }
    
    @staticmethod
    def get_colombian_law_headers() -> Dict[str, str]:
        """
        Get Colombian law compliance headers (Law 1581/2012)
        
        Returns:
            Dictionary of Colombian law headers
        """
        return {
            "X-Data-Protection-Law": "Ley 1581 de 2012",
            "X-Personal-Data-Policy": "https://example.com/politica-datos-personales",
            "X-Data-Authorization": "required",
            "X-Habeas-Data": "rights-protected"
        }
    
    @staticmethod
    def get_all_compliance_headers() -> Dict[str, str]:
        """
        Get all compliance headers
        
        Returns:
            Combined GDPR and Colombian law headers
        """
        headers = ComplianceHeaders.get_gdpr_headers()
        headers.update(ComplianceHeaders.get_colombian_law_headers())
        return headers


# ============================================================================
# SECURITY AUDIT LOGGER
# ============================================================================

class SecurityAuditLogger:
    """
    SIN_CARRETA: Security event audit logger
    
    Rationale: Maintain audit trail of all security events for
    compliance and forensics.
    """
    
    @staticmethod
    def log_auth_attempt(
        username: str,
        success: bool,
        ip_address: str,
        reason: Optional[str] = None
    ) -> None:
        """Log authentication attempt"""
        logger.log(
            logging.INFO if success else logging.WARNING,
            f"Authentication {'succeeded' if success else 'failed'} for {username}",
            extra={
                "event_type": "security_audit",
                "action": "auth_attempt",
                "username": username,
                "success": success,
                "ip_address": ip_address,
                "reason": reason,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    @staticmethod
    def log_rate_limit(ip_address: str, endpoint: str, limit: str) -> None:
        """Log rate limit hit"""
        logger.warning(
            f"Rate limit exceeded for {ip_address} on {endpoint}",
            extra={
                "event_type": "security_audit",
                "action": "rate_limit_exceeded",
                "ip_address": ip_address,
                "endpoint": endpoint,
                "limit": limit,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    @staticmethod
    def log_suspicious_activity(
        ip_address: str,
        activity_type: str,
        details: Dict[str, Any]
    ) -> None:
        """Log suspicious activity"""
        logger.warning(
            f"Suspicious activity detected from {ip_address}: {activity_type}",
            extra={
                "event_type": "security_audit",
                "action": "suspicious_activity",
                "ip_address": ip_address,
                "activity_type": activity_type,
                "details": details,
                "timestamp": datetime.now().isoformat()
            }
        )
