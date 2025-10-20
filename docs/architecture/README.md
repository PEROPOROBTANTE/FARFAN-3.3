# FARFAN 3.0 Architecture

Architecture documentation for the FARFAN 3.0 pipeline.

## Overview

FARFAN 3.0 follows a deterministic pipeline architecture with clear separation of concerns:
- **Orchestration Layer**: Coordinates execution
- **Domain Layer**: Business logic
- **Adapter Layer**: External interfaces
- **Pipeline Layer**: Stage definitions

## Core Principles

1. **Determinism**: Same input → same output
2. **Fault Tolerance**: Graceful degradation with circuit breakers
3. **Dependency Management**: DAG-based execution order
4. **Parallelization**: Independent modules execute concurrently
5. **Auditability**: Complete execution traces

## Architecture Diagrams

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Entry Point                           │
│                   (run_farfan.py)                        │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│              Orchestration Layer                         │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Router    │→ │ Choreographer│→ │    Report    │  │
│  └─────────────┘  └──────┬───────┘  └──────────────┘  │
│                           │                              │
│                    ┌──────▼───────┐                     │
│                    │Circuit Breaker│                     │
│                    └──────┬───────┘                     │
└───────────────────────────┼─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                  Domain Layer                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐             │
│  │ Policy   │  │ Teoria   │  │Financial │             │
│  │Processor │  │ Cambio   │  │Viability │  ...        │
│  └──────────┘  └──────────┘  └──────────┘             │
└─────────────────────────────────────────────────────────┘
```

### Execution Flow

```
Input (Policy + Questionnaire)
        │
        ▼
   Question Router ──────┐
        │                 │
        ▼                 ▼
  Execution Plan    Mapping Loader
        │
        ▼
   Choreographer
        │
   ┌────┴────┬────────┬────────┐
   ▼         ▼        ▼        ▼
 Wave 1   Wave 2   Wave 3   Wave 4
   │         │        │        │
   └─────────┴────────┴────────┘
                │
                ▼
         Report Assembly
                │
                ▼
        Final Output (JSON/HTML)
```

### Dependency Graph (DAG)

```
┌─────────────┐  ┌─────────────┐
│   Policy    │  │   Policy    │
│ Segmenter   │  │  Processor  │
└──────┬──────┘  └──────┬──────┘
       │                │
       │   Wave 1       │
       ├────────────────┤
       │                │
       ▼                ▼
┌─────────────┐  ┌─────────────┐
│  Semantic   │  │ Embedding   │
│  Chunking   │  │   Policy    │
└──────┬──────┘  └──────┬──────┘
       │                │
       │   Wave 2       │
       ├────────────────┤
       │                │
       ▼                ▼
┌─────────────┐  ┌─────────────┐
│ Analyzer    │  │   Teoria    │
│    One      │  │   Cambio    │
└──────┬──────┘  └──────┬──────┘
       │                │
       │   Wave 3       │
       ├────────────────┤
       │                │
       ▼                ▼
┌─────────────┐  ┌─────────────┐
│   Derek     │  │Contradiction│
│   Beach     │  │  Detection  │
└──────┬──────┘  └──────┬──────┘
       │                │
       │   Wave 4       │
       └────────┬───────┘
                │
                ▼
        ┌─────────────┐
        │  Financial  │
        │ Viability   │
        └─────────────┘
              Wave 5
```

## Component Details

### Orchestrator Components

#### QuestionRouter
- Routes questions to modules
- Uses execution mapping configuration
- Handles question variations

#### Choreographer
- Manages execution order
- Resolves dependencies
- Coordinates parallel execution
- Integrates with circuit breaker

#### CircuitBreaker
- Monitors module health
- Prevents cascading failures
- Provides graceful degradation
- Tracks failure patterns

#### ReportAssembly
- Collects module outputs
- Generates final reports
- Validates completeness
- Formats results

### Domain Modules

Each domain module:
- Implements specific analysis logic
- Produces deterministic outputs
- Handles errors gracefully
- Logs execution traces

### Pipeline Stages

Stages coordinate:
- Module dependencies
- Data flow
- Parallel execution
- Resource management

## Design Patterns

### Circuit Breaker Pattern
Prevents cascading failures by monitoring module health and opening circuits when failure thresholds are exceeded.

### Dependency Injection
All components use dependency injection for testability and flexibility.

### Adapter Pattern
Unified interface to domain modules through adapter layer.

### Pipeline Pattern
Data flows through defined stages with clear transformations.

## Configuration

### Execution Mapping
Defines question-to-module routing rules.

### Module Configuration
Module-specific settings and parameters.

### Pipeline Configuration
Pipeline execution parameters (workers, timeouts, etc.).

## Data Flow

1. **Input**: Policy documents + questionnaire
2. **Routing**: Questions mapped to modules
3. **Execution**: Modules execute in waves
4. **Assembly**: Results collected and formatted
5. **Output**: Final report generated

## Error Handling

### Levels
1. **Module Level**: Try/catch in module code
2. **Adapter Level**: Error standardization
3. **Circuit Breaker Level**: Failure pattern detection
4. **Orchestrator Level**: Graceful degradation

### Recovery Strategies
- Retry with exponential backoff
- Circuit breaker opening/closing
- Fallback to cached results
- Partial result assembly

## Performance

### Parallelization
Independent modules in same wave execute concurrently.

### Resource Management
- Thread pool for parallel execution
- Memory-efficient data structures
- Streaming for large documents

### Optimization
- Lazy loading of models
- Caching of intermediate results
- Batch processing support

## Security

- No secrets in configuration
- Input validation at entry points
- Output sanitization
- Audit logging

## Testing Strategy

### Unit Tests
Test individual components in isolation.

### Integration Tests
Test component interactions.

### E2E Tests
Test complete pipeline with real data.

### Performance Tests
Measure execution time and resource usage.

## Further Reading

- [Dependency Framework](DEPENDENCY_FRAMEWORK.md)
- [Execution Mapping](EXECUTION_MAPPING_MASTER.md)
- [Project Structure](../guides/PROJECT_STRUCTURE.md)
- [Development Guide](../guides/AGENTS.md)
