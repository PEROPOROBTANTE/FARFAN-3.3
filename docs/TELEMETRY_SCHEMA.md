# Telemetry Schema Documentation

## Overview

This document defines the telemetry event schema for FARFAN 3.0. All system components emit structured telemetry events that enable monitoring, debugging, auditing, and performance analysis.

**Schema Version**: 1.0.0  
**Last Updated**: 2025-01-21

## Table of Contents

- [Event Format](#event-format)
- [Event Types](#event-types)
- [Common Fields](#common-fields)
- [Adapter Events](#adapter-events)
- [Orchestrator Events](#orchestrator-events)
- [Pipeline Events](#pipeline-events)
- [Error Events](#error-events)
- [Performance Events](#performance-events)
- [Audit Events](#audit-events)
- [Event Emission](#event-emission)
- [Event Storage](#event-storage)
- [Query Examples](#query-examples)

## Event Format

All telemetry events follow this base structure:

```json
{
  "event_id": "uuid-v4",
  "event_type": "adapter.execution.started",
  "timestamp": "2025-01-21T10:30:00.123456Z",
  "correlation_id": "req-uuid-v4",
  "source": {
    "component": "adapter",
    "module": "teoria_cambio",
    "method": "analyze_theory_of_change",
    "version": "3.0.0"
  },
  "context": {
    "question_id": "P1-D2-Q015",
    "policy_id": "PDM-2024-001",
    "execution_wave": 3,
    "session_id": "session-uuid-v4"
  },
  "data": {
    "event_specific_fields": "..."
  },
  "metadata": {
    "environment": "production",
    "host": "farfan-node-01",
    "process_id": 12345
  }
}
```

### Field Descriptions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `event_id` | UUID v4 | Yes | Unique identifier for this event |
| `event_type` | String | Yes | Event type (see Event Types) |
| `timestamp` | ISO 8601 | Yes | Event timestamp in UTC |
| `correlation_id` | UUID v4 | Yes | Request correlation ID for tracing |
| `source` | Object | Yes | Event source information |
| `context` | Object | No | Execution context (varies by event) |
| `data` | Object | Yes | Event-specific data payload |
| `metadata` | Object | No | Runtime metadata |

## Event Types

Event types follow a hierarchical naming convention: `category.subcategory.action`

### Categories

| Category | Description | Examples |
|----------|-------------|----------|
| `adapter` | Adapter method execution | `adapter.execution.started` |
| `orchestrator` | Orchestration activities | `orchestrator.wave.started` |
| `pipeline` | Pipeline lifecycle | `pipeline.execution.started` |
| `circuit_breaker` | Circuit breaker state | `circuit_breaker.opened` |
| `error` | Error and exception events | `error.adapter.timeout` |
| `performance` | Performance metrics | `performance.method.completed` |
| `audit` | Audit trail events | `audit.contract.validated` |
| `telemetry` | Telemetry system events | `telemetry.buffer.flushed` |

## Common Fields

### Source Object

```json
{
  "component": "adapter | orchestrator | pipeline | circuit_breaker",
  "module": "module_name",
  "method": "method_name",
  "version": "semver_version"
}
```

### Context Object

```json
{
  "question_id": "P#-D#-Q###",
  "policy_id": "policy_identifier",
  "execution_wave": 1-5,
  "session_id": "uuid-v4",
  "parent_span_id": "span-id",
  "trace_id": "trace-id"
}
```

### Metadata Object

```json
{
  "environment": "development | staging | production",
  "host": "hostname",
  "process_id": 12345,
  "thread_id": "thread-name",
  "user_agent": "client-identifier"
}
```

## Adapter Events

### adapter.execution.started

Emitted when an adapter method begins execution.

```json
{
  "event_type": "adapter.execution.started",
  "data": {
    "adapter_name": "teoria_cambio_adapter",
    "method_name": "analyze_theory_of_change",
    "input_hash": "sha256:abc123...",
    "input_size_bytes": 4096,
    "parameters": {
      "seed": 42,
      "confidence_threshold": 0.7
    }
  }
}
```

### adapter.execution.completed

Emitted when an adapter method completes successfully.

```json
{
  "event_type": "adapter.execution.completed",
  "data": {
    "adapter_name": "teoria_cambio_adapter",
    "method_name": "analyze_theory_of_change",
    "duration_ms": 1234,
    "input_hash": "sha256:abc123...",
    "output_hash": "sha256:def456...",
    "output_size_bytes": 8192,
    "result_status": "success",
    "evidence_count": 5,
    "confidence_score": 0.85
  }
}
```

### adapter.execution.failed

Emitted when an adapter method fails.

```json
{
  "event_type": "adapter.execution.failed",
  "data": {
    "adapter_name": "teoria_cambio_adapter",
    "method_name": "analyze_theory_of_change",
    "duration_ms": 567,
    "error_type": "ValidationError",
    "error_message": "Invalid input format",
    "error_stack": "...",
    "input_hash": "sha256:abc123...",
    "retry_count": 0,
    "will_retry": false
  }
}
```

### adapter.cache.hit

Emitted when a cached result is returned.

```json
{
  "event_type": "adapter.cache.hit",
  "data": {
    "adapter_name": "embedding_policy_adapter",
    "method_name": "generate_embeddings",
    "cache_key": "sha256:key123...",
    "cache_age_seconds": 3600,
    "saved_duration_ms": 5000
  }
}
```

### adapter.cache.miss

Emitted when no cached result is found.

```json
{
  "event_type": "adapter.cache.miss",
  "data": {
    "adapter_name": "embedding_policy_adapter",
    "method_name": "generate_embeddings",
    "cache_key": "sha256:key123..."
  }
}
```

## Orchestrator Events

### orchestrator.wave.started

Emitted when an execution wave begins.

```json
{
  "event_type": "orchestrator.wave.started",
  "data": {
    "wave_number": 3,
    "module_count": 5,
    "modules": [
      "teoria_cambio",
      "analyzer_one",
      "dereck_beach",
      "embedding_policy",
      "semantic_chunking"
    ],
    "expected_duration_ms": 10000
  }
}
```

### orchestrator.wave.completed

Emitted when an execution wave completes.

```json
{
  "event_type": "orchestrator.wave.completed",
  "data": {
    "wave_number": 3,
    "duration_ms": 9876,
    "modules_succeeded": 4,
    "modules_failed": 1,
    "modules_skipped": 0,
    "results_hash": "sha256:wave3..."
  }
}
```

### orchestrator.module.routed

Emitted when a question is routed to a module.

```json
{
  "event_type": "orchestrator.module.routed",
  "data": {
    "question_id": "P1-D2-Q015",
    "module_name": "teoria_cambio",
    "routing_rule": "question_type_theory_of_change",
    "confidence": 1.0
  }
}
```

## Pipeline Events

### pipeline.execution.started

Emitted when pipeline execution begins.

```json
{
  "event_type": "pipeline.execution.started",
  "data": {
    "pipeline_id": "pipe-uuid",
    "policy_count": 10,
    "question_count": 300,
    "execution_mode": "batch",
    "configuration": {
      "workers": 4,
      "timeout_seconds": 3600,
      "retry_enabled": true
    }
  }
}
```

### pipeline.execution.completed

Emitted when pipeline execution completes.

```json
{
  "event_type": "pipeline.execution.completed",
  "data": {
    "pipeline_id": "pipe-uuid",
    "duration_ms": 45000,
    "questions_processed": 300,
    "questions_succeeded": 295,
    "questions_failed": 5,
    "total_adapters_executed": 1250,
    "output_hash": "sha256:pipeline..."
  }
}
```

### pipeline.checkpoint.created

Emitted when a pipeline checkpoint is created.

```json
{
  "event_type": "pipeline.checkpoint.created",
  "data": {
    "pipeline_id": "pipe-uuid",
    "checkpoint_id": "checkpoint-uuid",
    "progress_percentage": 65.5,
    "questions_completed": 197,
    "checkpoint_hash": "sha256:checkpoint..."
  }
}
```

## Error Events

### error.adapter.timeout

Emitted when an adapter method times out.

```json
{
  "event_type": "error.adapter.timeout",
  "data": {
    "adapter_name": "dereck_beach_adapter",
    "method_name": "perform_causal_analysis",
    "timeout_ms": 30000,
    "elapsed_ms": 30001,
    "input_hash": "sha256:abc123...",
    "will_retry": true,
    "retry_delay_ms": 5000
  }
}
```

### error.contract.validation_failed

Emitted when contract validation fails.

```json
{
  "event_type": "error.contract.validation_failed",
  "data": {
    "adapter_name": "financial_viability_adapter",
    "method_name": "assess_viability",
    "validation_errors": [
      {
        "field": "score",
        "error": "Field required",
        "location": ["body", "score"]
      }
    ],
    "input_data_hash": "sha256:invalid..."
  }
}
```

### error.circuit_breaker.opened

Emitted when circuit breaker opens.

```json
{
  "event_type": "error.circuit_breaker.opened",
  "data": {
    "circuit_name": "teoria_cambio_circuit",
    "failure_count": 5,
    "failure_threshold": 5,
    "recovery_timeout_ms": 60000,
    "last_error": "Connection timeout"
  }
}
```

## Performance Events

### performance.method.completed

Emitted for all adapter method completions with performance data.

```json
{
  "event_type": "performance.method.completed",
  "data": {
    "adapter_name": "embedding_policy_adapter",
    "method_name": "generate_embeddings",
    "duration_ms": 2345,
    "cpu_ms": 2100,
    "memory_mb": 512,
    "input_size_bytes": 10240,
    "output_size_bytes": 4096,
    "throughput_mb_per_sec": 4.37
  }
}
```

### performance.wave.statistics

Emitted with statistics for each execution wave.

```json
{
  "event_type": "performance.wave.statistics",
  "data": {
    "wave_number": 3,
    "duration_ms": 9876,
    "module_count": 5,
    "statistics": {
      "min_duration_ms": 1234,
      "max_duration_ms": 5678,
      "mean_duration_ms": 1975,
      "p50_duration_ms": 1800,
      "p95_duration_ms": 4500,
      "p99_duration_ms": 5200
    }
  }
}
```

## Audit Events

### audit.contract.validated

Emitted when a contract is successfully validated.

```json
{
  "event_type": "audit.contract.validated",
  "data": {
    "adapter_name": "teoria_cambio_adapter",
    "method_name": "analyze_theory_of_change",
    "contract_version": "1.0.0",
    "schema_hash": "sha256:contract...",
    "validation_duration_ms": 12
  }
}
```

### audit.determinism.verified

Emitted when determinism is verified.

```json
{
  "event_type": "audit.determinism.verified",
  "data": {
    "adapter_name": "analyzer_one_adapter",
    "method_name": "analyze_municipal_plan",
    "seed": 42,
    "run_count": 3,
    "output_hashes": [
      "sha256:run1...",
      "sha256:run1...",
      "sha256:run1..."
    ],
    "is_deterministic": true
  }
}
```

### audit.execution.traced

Emitted for complete execution trace.

```json
{
  "event_type": "audit.execution.traced",
  "data": {
    "question_id": "P1-D2-Q015",
    "execution_chain": [
      {
        "step": 1,
        "adapter": "teoria_cambio",
        "method": "analyze_theory_of_change",
        "duration_ms": 1234,
        "output_hash": "sha256:step1..."
      },
      {
        "step": 2,
        "adapter": "dereck_beach",
        "method": "perform_evidential_test",
        "duration_ms": 2345,
        "output_hash": "sha256:step2..."
      }
    ],
    "total_duration_ms": 3579,
    "final_output_hash": "sha256:final..."
  }
}
```

## Event Emission

### Python API

```python
from orchestrator.telemetry import emit_event, TelemetryContext

# Emit adapter execution start
with TelemetryContext(correlation_id="req-123") as ctx:
    emit_event(
        event_type="adapter.execution.started",
        source={
            "component": "adapter",
            "module": "teoria_cambio",
            "method": "analyze_theory_of_change",
            "version": "3.0.0"
        },
        context={
            "question_id": "P1-D2-Q015",
            "execution_wave": 3
        },
        data={
            "input_hash": compute_hash(input_data),
            "input_size_bytes": len(input_data)
        }
    )
```

### Decorator-Based Emission

```python
from orchestrator.telemetry import telemetry_tracked

@telemetry_tracked(event_prefix="adapter.execution")
def analyze_theory_of_change(input_data: AnalysisInput) -> AnalysisOutput:
    """Method automatically emits start/complete/failed events."""
    # Implementation
    return result
```

### Batched Emission

```python
from orchestrator.telemetry import TelemetryBatcher

with TelemetryBatcher(flush_interval_ms=1000) as batcher:
    for item in batch:
        batcher.emit({
            "event_type": "performance.method.completed",
            "data": {"duration_ms": process_item(item)}
        })
    # Auto-flushes on context exit
```

## Event Storage

### Storage Backend

Events are stored in:
1. **Time-series database**: For performance and monitoring queries
2. **Object storage**: For long-term audit retention
3. **Elasticsearch**: For full-text search and analysis

### Retention Policy

| Event Category | Hot Storage | Warm Storage | Cold Storage |
|----------------|-------------|--------------|--------------|
| Performance | 7 days | 30 days | 1 year |
| Audit | 30 days | 90 days | 7 years |
| Error | 30 days | 90 days | 2 years |
| Telemetry | 1 day | 7 days | 30 days |

### Compression

Events older than 24 hours are compressed using zstd before moving to warm storage.

## Query Examples

### Query by Correlation ID

```python
# Trace all events for a request
events = query_events(
    correlation_id="req-123",
    time_range=("2025-01-21T10:00:00Z", "2025-01-21T11:00:00Z")
)
```

### Query by Question ID

```python
# Get all events for a specific question
events = query_events(
    context__question_id="P1-D2-Q015",
    sort_by="timestamp"
)
```

### Query Performance Data

```python
# Get P99 latency for adapter method
metrics = query_aggregates(
    event_type="performance.method.completed",
    filters={
        "data.adapter_name": "teoria_cambio_adapter",
        "data.method_name": "analyze_theory_of_change"
    },
    aggregations={
        "p99": ("data.duration_ms", "percentile", 99),
        "mean": ("data.duration_ms", "avg"),
        "count": ("event_id", "count")
    },
    time_range="last_7_days"
)
```

### Query Error Patterns

```python
# Find all timeout errors
errors = query_events(
    event_type="error.adapter.timeout",
    time_range="last_24_hours",
    group_by="data.adapter_name",
    count=True
)
```

## Event Schema Validation

All events are validated against JSON Schema before emission:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["event_id", "event_type", "timestamp", "correlation_id", "source", "data"],
  "properties": {
    "event_id": {
      "type": "string",
      "format": "uuid"
    },
    "event_type": {
      "type": "string",
      "pattern": "^[a-z_]+\\.[a-z_]+\\.[a-z_]+$"
    },
    "timestamp": {
      "type": "string",
      "format": "date-time"
    },
    "correlation_id": {
      "type": "string",
      "format": "uuid"
    },
    "source": {
      "type": "object",
      "required": ["component", "version"],
      "properties": {
        "component": {
          "type": "string",
          "enum": ["adapter", "orchestrator", "pipeline", "circuit_breaker"]
        },
        "module": {"type": "string"},
        "method": {"type": "string"},
        "version": {"type": "string", "pattern": "^\\d+\\.\\d+\\.\\d+$"}
      }
    },
    "data": {
      "type": "object"
    }
  }
}
```

## Best Practices

1. **Always Include Correlation ID**: Enable request tracing across components
2. **Emit Start and Complete Events**: Track both success and duration
3. **Include Input/Output Hashes**: Enable determinism verification
4. **Use Structured Data**: Avoid string formatting in data fields
5. **Batch When Possible**: Reduce overhead for high-frequency events
6. **Validate Before Emit**: Ensure schema compliance
7. **Include Context**: Add execution context for better filtering
8. **Handle Failures Gracefully**: Don't let telemetry failures break execution

## Monitoring Dashboard

Telemetry events power the FARFAN monitoring dashboard at `http://localhost:5000`:

- Real-time event stream
- Performance metrics (P50/P95/P99)
- Error rate tracking
- Circuit breaker status
- Determinism verification results
- Audit trail visualization

---

**Version History**:
- 1.0.0 (2025-01-21): Initial schema definition
