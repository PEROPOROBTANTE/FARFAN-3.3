#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Module Controller for Derek Beach CDAF Classes
=====================================================

This module consolidates core classes from dereck_beach.py as a unified
controller for the orchestrator adapter pattern. It provides clean interfaces for:

- BayesianMechanismInference: Hierarchical Bayesian mechanism inference
- CausalExtractor: Causal hierarchy extraction from policy documents
- ConfigLoader: Configuration management with Pydantic validation
- FinancialAuditor: Financial traceability and auditing
- OperationalizationAuditor: Operationalization quality auditing
- PDFProcessor: PDF document processing and extraction
- ReportingEngine: Visualization and report generation

All classes preserve their original method signatures, parameter types,
return types, and docstring formats for backward compatibility during
migration to the unified module controller architecture.

Author: Orchestrator Integration Team
Version: 3.0.0 - Unified Controller
Python: 3.10+
"""

# Import all required classes and types from dereck_beach
from dereck_beach import (
    # Type definitions (Literal types)
    NodeType,
    RigorStatus,
    TestType,
    DynamicsType,
    
    # Core Beach evidential test class
    BeachEvidentialTest,
    
    # Exception hierarchy
    CDAFException,
    CDAFValidationError,
    CDAFProcessingError,
    CDAFBayesianError,
    CDAFConfigError,
    
    # Pydantic configuration models
    BayesianThresholdsConfig,
    MechanismTypeConfig,
    PerformanceConfig,
    SelfReflectionConfig,
    CDAFConfigSchema,
    
    # TypedDict definitions
    CausalLink,
    AuditResult,
    
    # NamedTuple definitions
    GoalClassification,
    EntityActivity,
    
    # Dataclass
    MetaNode,
    
    # Main controller classes (7 required classes)
    ConfigLoader,
    PDFProcessor,
    CausalExtractor,
    FinancialAuditor,
    OperationalizationAuditor,
    BayesianMechanismInference,
    ReportingEngine,
)

__all__ = [
    # Type definitions
    'NodeType',
    'RigorStatus',
    'TestType',
    'DynamicsType',
    
    # Core classes
    'BeachEvidentialTest',
    
    # Exception classes
    'CDAFException',
    'CDAFValidationError',
    'CDAFProcessingError',
    'CDAFBayesianError',
    'CDAFConfigError',
    
    # Pydantic configuration models
    'BayesianThresholdsConfig',
    'MechanismTypeConfig',
    'PerformanceConfig',
    'SelfReflectionConfig',
    'CDAFConfigSchema',
    
    # TypedDict and NamedTuple definitions
    'GoalClassification',
    'EntityActivity',
    'CausalLink',
    'AuditResult',
    'MetaNode',
    
    # Main controller classes
    'ConfigLoader',
    'PDFProcessor',
    'CausalExtractor',
    'FinancialAuditor',
    'OperationalizationAuditor',
    'BayesianMechanismInference',
    'ReportingEngine',
]
