# FARFAN 3.0 API Documentation

API documentation for FARFAN 3.0 components.

## Orchestrator API

### QuestionRouter
Routes questions to appropriate modules based on execution mapping.

### Choreographer
Orchestrates module execution in correct dependency order.

### CircuitBreaker
Provides fault tolerance with circuit breaker pattern.

### ReportAssembly
Assembles final reports from module outputs.

### MappingLoader
Loads and validates execution mapping configurations.

### AdapterRegistry
Registry for module adapter instances.

## Domain API

### PolicyProcessor
Processes and normalizes policy documents.

### PolicySegmenter
Segments policy documents into logical units.

### QuestionnaireParser
Parses questionnaire inputs.

### TeoriaCambio
Analyzes theory of change in policies.

### EmbeddingPolicy
Generates policy embeddings for semantic analysis.

### SemanticChunkingPolicy
Performs semantic chunking of policy text.

## Detailed Documentation

Detailed API documentation will be generated using Sphinx or similar tools.

For now, refer to docstrings in source files:
- `src/orchestrator/`
- `src/domain/`
- `src/adapters/`
- `src/stages/`
