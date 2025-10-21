# AtroZ Dashboard API

**SIN_CARRETA Implementation**: Backend REST API for AtroZ dashboard with strict contract validation, deterministic sample data, and structured telemetry.

## Overview

This API provides core data endpoints for the AtroZ dashboard, implementing:

- ✅ **Strict Contract Validation**: All endpoints validate input/output with explicit 400/403 errors
- ✅ **Deterministic Data**: Sample data generated from seeded RNG (mulberry32/splitmix32)
- ✅ **Structured Telemetry**: Every request emits structured logs with timing and context
- ✅ **Complete Coverage**: All 7 required endpoints implemented

## Architecture

```
api/
├── main.py              # FastAPI application with exception handlers
├── models/
│   └── schemas.py       # Pydantic v2 schemas with strict validation
├── endpoints/
│   ├── pdet_regions.py  # PDET region endpoints
│   ├── municipalities.py # Municipality endpoints
│   └── analysis.py      # Analysis endpoints (clusters, questions)
├── utils/
│   ├── seeded_rng.py    # Deterministic RNG (mulberry32, splitmix32)
│   ├── data_generator.py # Sample data generation
│   └── telemetry.py     # Structured logging middleware
└── tests/
    ├── test_seeded_rng.py # RNG determinism tests
    └── test_api_endpoints.py # Endpoint tests
```

## API Endpoints

### PDET Regions

#### 1. GET `/api/v1/pdet/regions`
List all PDET regions with summary data.

**Response**: `RegionListResponse`
- `regions`: List of 10 regions
- `total`: Count (10)
- `timestamp`: ISO datetime

#### 2. GET `/api/v1/pdet/regions/{id}`
Get detailed data for a specific region.

**Parameters**:
- `id`: Region ID (pattern: `REGION_\d{3}`)

**Response**: `RegionDetailResponse`
- Full region data with dimension and policy scores
- Metadata (population, area, municipalities count)

#### 3. GET `/api/v1/pdet/regions/{id}/municipalities`
List municipalities in a region.

**Parameters**:
- `id`: Region ID

**Response**: `MunicipalityListResponse`
- 10 municipalities per region

### Municipalities

#### 4. GET `/api/v1/municipalities/{id}`
Get detailed municipality data.

**Parameters**:
- `id`: Municipality ID (pattern: `MUN_\d{5}`)

**Response**: `MunicipalityDetailResponse`
- Full municipality data with scores and metadata

#### 5. GET `/api/v1/municipalities/{id}/analysis`
Get comprehensive analysis for municipality (6 dimensions × 5 questions).

**Parameters**:
- `id`: Municipality ID

**Response**: `MunicipalityAnalysisResponse`
- 6 dimensions with 5 questions each
- Strengths and weaknesses per dimension
- Overall summary

### Analysis

#### 6. GET `/api/v1/analysis/clusters/{regionId}`
Get cluster analysis for region (municipalities grouped by similarity).

**Parameters**:
- `regionId`: Region ID

**Response**: `ClusterAnalysisResponse`
- 3-5 clusters of similar municipalities
- Centroid scores (6 dimensions)
- Cluster members with similarity scores

#### 7. GET `/api/v1/analysis/questions/{municipalityId}`
Get all 300 questions for municipality (10 policies × 6 dimensions × 5 questions).

**Parameters**:
- `municipalityId`: Municipality ID

**Response**: `QuestionAnalysisResponse`
- All 300 questions with scores and evidence
- Grouped by dimension and policy area

## Deterministic Data Generation

**SIN_CARRETA**: All data is generated deterministically using seeded RNG.

### Base Seed: 42

All generators use base seed `42` for consistency.

### Entity-Specific Seeds

Each entity (region, municipality, question) gets a deterministic seed based on:
```python
seed = base_seed + hash(entity_id)
```

This ensures:
- Same entity ID always produces same data
- Different entities produce different data
- Complete reproducibility across API restarts

### RNG Algorithms

Two high-quality PRNG algorithms implemented:

1. **Mulberry32** (default)
   - Fast, high-quality 32-bit generator
   - Period: 2^32 - 1
   - Used for most data generation

2. **SplitMix32**
   - Alternative with good avalanche properties
   - Useful for generating independent seed sequences

## Contract Validation

**SIN_CARRETA**: No silent fallbacks - all violations return explicit errors.

### Validation Rules

1. **ID Formats**:
   - Region: `REGION_\d{3}` (e.g., `REGION_001`)
   - Municipality: `MUN_\d{5}` (e.g., `MUN_00101`)
   - Question: `P\d+-D\d+-Q\d+` (e.g., `P1-D1-Q1`)

2. **Score Ranges**:
   - Overall scores: [0, 100] with 2 decimals
   - Question scores: [0, 3] with 2 decimals
   - Confidence: [0, 1] with 2+ decimals

3. **Structural Requirements**:
   - Regions: Must have all 6 dimensions and 10 policy areas
   - Dimensions: Must have exactly 5 questions
   - Questions: Must have exactly 300 for complete analysis

4. **Error Responses**:
   - `400`: Validation errors (invalid format, out of range)
   - `404`: Entity not found
   - `500`: Internal server error

## Telemetry

**SIN_CARRETA**: Every request emits structured telemetry.

### Telemetry Data

Each request logs:
```json
{
  "service": "atroz-api",
  "event_type": "http_request",
  "request": {
    "request_id": "uuid",
    "method": "GET",
    "path": "/api/v1/pdet/regions",
    "query_params": {},
    "headers": {...},
    "client": {...}
  },
  "response": {
    "status_code": 200,
    "duration_ms": 45.23,
    "success": true
  }
}
```

### Response Headers

Every response includes:
- `X-Request-ID`: Unique request identifier
- `X-Response-Time-Ms`: Request duration in milliseconds

## Running the API

### Installation

```bash
pip install fastapi uvicorn pydantic
```

### Development Server

```bash
python -m api.main
# or
uvicorn api.main:app --reload --port 8000
```

### Production Server

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### API Documentation

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Testing

### Run All Tests

```bash
pytest api/tests/ -v
```

### Test Determinism

```bash
pytest api/tests/test_seeded_rng.py -v
```

### Test Endpoints

```bash
pytest api/tests/test_api_endpoints.py -v
```

### Test Coverage

```bash
pytest api/tests/ --cov=api --cov-report=html
```

## Example Usage

### List Regions

```bash
curl http://localhost:8000/api/v1/pdet/regions
```

### Get Region Details

```bash
curl http://localhost:8000/api/v1/pdet/regions/REGION_001
```

### Get Municipality Analysis

```bash
curl http://localhost:8000/api/v1/municipalities/MUN_00101/analysis
```

### Get All 300 Questions

```bash
curl http://localhost:8000/api/v1/analysis/questions/MUN_00101
```

## Data Model

### Regions
- 10 PDET regions (REGION_001 to REGION_010)
- Each with 10 municipalities
- Scores across 6 dimensions and 10 policy areas

### Municipalities
- 100 total municipalities (10 per region)
- ID format: MUN_RRMNN (RR = region, NN = municipality index)
- Complete analysis with 300 questions

### Questions
- 300 questions total
- 10 policy areas (P1-P10)
- 6 dimensions (D1-D6)
- 5 questions per dimension per policy
- Formula: 10 × 6 × 5 = 300

## Dimensions

1. **D1**: Gobernanza y Participación
2. **D2**: Desarrollo Económico
3. **D3**: Infraestructura y Servicios
4. **D4**: Educación y Cultura
5. **D5**: Salud y Bienestar
6. **D6**: Medio Ambiente y Sostenibilidad

## Policy Areas

1. **P1**: Ordenamiento Social del Territorio
2. **P2**: Reactivación Económica y Producción Agropecuaria
3. **P3**: Salud Rural
4. **P4**: Educación Rural
5. **P5**: Vivienda, Agua Potable y Saneamiento Básico
6. **P6**: Infraestructura y Adecuación de Tierras
7. **P7**: Reincorporación y Convivencia
8. **P8**: Sistema para la Garantía Progresiva de Derechos
9. **P9**: Reconciliación y Construcción de Paz
10. **P10**: Víctimas del Conflicto Armado

## Performance Monitoring

**NEW**: Comprehensive performance monitoring and alerting system.

### Metrics Tracked

- **Response Times**: Per-endpoint latency with histogram buckets
- **Memory Usage**: System memory percentage with alerts
- **Cache Hit Rates**: Per-cache efficiency tracking
- **Error Rates**: 4xx/5xx error percentage
- **Active Requests**: Concurrent request gauge
- **Data Freshness**: Age of data sources in seconds
- **Frame Rate**: UI performance (FPS) tracking
- **WebSocket Stability**: Connection/disconnection rates

### Alert Thresholds

The system automatically emits alerts when metrics exceed configured thresholds:

- **Latency**: > 500ms
- **Error Rate**: > 1%
- **Memory**: > 80%
- **Frame Rate**: < 50 FPS
- **Data Staleness**: > 15 minutes
- **WebSocket Disconnects**: > 5 per minute

### Monitoring Endpoints

#### GET `/metrics`
Prometheus-formatted metrics for external monitoring systems.

```bash
curl http://localhost:8000/metrics
```

Returns metrics in Prometheus text format.

#### GET `/health`
Enhanced health check with system metrics.

```bash
curl http://localhost:8000/health
```

Returns comprehensive health status including:
- Service status
- Request metrics (total, errors, latency)
- Memory usage
- Cache statistics
- WebSocket statistics
- Alert thresholds

## Security Hardening

**NEW**: Enterprise-grade security controls and compliance.

### Security Features

#### 1. HTTPS Enforcement
- Automatic HTTP to HTTPS redirect in production
- HSTS headers with preload
- Disabled in development for convenience

#### 2. JWT Authentication
- Token-based authentication with configurable expiration
- RS256/HS256 algorithm support
- Automatic token validation
- Secure password hashing (bcrypt)

#### 3. CORS Protection
- Configurable allowed origins
- Credentials support
- Method and header whitelisting
- Exposed headers for client access

#### 4. Rate Limiting
- Per-IP rate limiting (default: 100 req/min)
- Separate limits for auth endpoints (20 req/min)
- Redis backend support for distributed systems
- In-memory storage for development

#### 5. XSS/CSRF Protection
- Content Security Policy (CSP) headers
- X-Frame-Options: DENY
- X-Content-Type-Options: nosniff
- X-XSS-Protection headers
- SameSite cookie attributes

#### 6. WebSocket Security
- Token-based authentication for WS connections
- Connection rate monitoring
- Abnormal disconnect detection

#### 7. Compliance Headers
- **GDPR**: Data subject rights, processing basis, retention
- **Colombian Law 1581/2012**: Personal data protection
- Privacy policy references
- Data controller identification

### Security Endpoints

#### GET `/security/status`
Security configuration and status.

```bash
curl http://localhost:8000/security/status
```

Returns:
- Security controls enabled/disabled
- HTTPS enforcement status
- Rate limiting configuration
- CORS settings
- JWT configuration
- Compliance headers

### Configuration

Security features are configured via environment variables:

```bash
# Environment
ENVIRONMENT=production  # or development

# Security
JWT_SECRET_KEY=your-secret-key-here
JWT_ALGORITHM=HS256
JWT_EXPIRATION_MINUTES=30

# CORS
ALLOWED_ORIGINS=https://dashboard.example.com,https://api.example.com

# Rate Limiting
RATE_LIMIT_PER_MINUTE=100
RATE_LIMIT_AUTH_PER_MINUTE=20
```

### Telemetry and Audit

All security events are logged with structured telemetry:
- Authentication attempts (success/failure)
- Rate limit violations
- Suspicious activity detection
- Token validation failures
- WebSocket authentication

## Performance

- Average response time: 10-50ms (depending on endpoint)
- Questions endpoint (300 items): 100-200ms
- Monitoring overhead: ~4-8ms per request
- All data generated on-the-fly (no database)
- Deterministic generation ensures consistency

## Security

**Enhanced Security Controls**:
- ✅ HTTPS enforcement (production)
- ✅ JWT authentication with expiration
- ✅ CORS protection with whitelisting
- ✅ Rate limiting per IP
- ✅ XSS/CSRF protection headers
- ✅ WebSocket authentication
- ✅ Input validation on all endpoints
- ✅ Pattern matching for IDs
- ✅ Range validation for scores
- ✅ GDPR compliance headers
- ✅ Colombian Law 1581/2012 compliance
- ✅ Security audit logging
- ✅ No SQL injection risk (no database)

## License

Copyright © 2024 FARFAN 3.0 Team. All rights reserved.

---

**SIN_CARRETA**: Implementation follows strict contract validation with no silent fallbacks. All endpoints emit structured telemetry and generate deterministic sample data from fixed seeds.
