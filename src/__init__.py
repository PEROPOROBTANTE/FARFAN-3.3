"""
FARFAN 3.0 - Deterministic Policy Analysis Pipeline
====================================================

A deterministic pipeline for policy document analysis using NLP and ML techniques.

Package structure:
- orchestrator: Core orchestration components (router, choreographer, circuit breaker)
- domain: Business logic modules for policy analysis
- adapters: External interface adapters
- stages: Pipeline stage coordination
- config: Configuration management

Version: 3.0.0
Python: 3.10+
"""

__version__ = "3.0.0"
__author__ = "FARFAN 3.0 Team"

# Define public API
__all__ = [
    "orchestrator",
    "domain",
    "adapters",
    "stages",
    "config",
]
