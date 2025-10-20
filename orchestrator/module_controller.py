"""
ModuleController - Unified Interface for Adapter Orchestration
==============================================================

CORE RESPONSIBILITY: Centralized adapter invocation with responsibility mapping
-------------------------------------------------------------------------------
Accepts all 11 adapter instances via constructor dependency injection and provides
a unified interface for processing questions by delegating to the appropriate 
adapter based on responsibility mapping loaded from responsibility_map.json.

KEY FEATURES:
- Constructor dependency injection for all adapters
- Responsibility-based routing via JSON configuration
- Circuit breaker integration for fault tolerance
- Standardized result format across all adapters
- Performance tracking and monitoring

ARCHITECTURE:
-------------
1. Adapters injected at construction (dependency injection pattern)
2. Responsibility map loaded from JSON configuration
3. Question routing based on dimension/policy area mapping
4. Circuit breaker wraps each adapter method call
5. Results aggregated and normalized

Author: FARFAN Integration Team
Version: 3.0.0
Python: 3.10+
"""

import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class RoutingDecision:
    """
    Encapsulates routing decision for a question
    """
    question_id: str
    dimension: str
    policy_area: str
    primary_adapter: str
    secondary_adapters: List[str]
    execution_strategy: str
    confidence_threshold: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "dimension": self.dimension,
            "policy_area": self.policy_area,
            "primary_adapter": self.primary_adapter,
            "secondary_adapters": self.secondary_adapters,
            "execution_strategy": self.execution_strategy,
            "confidence_threshold": self.confidence_threshold
        }


class ModuleController:
    """
    Unified controller for all 9+ adapter instances
    
    Manages adapter lifecycle, routing decisions, and result aggregation
    through dependency injection and responsibility mapping.
    """

    def __init__(
            self,
            # Core adapters (9 primary)
            teoria_cambio_adapter=None,
            analyzer_one_adapter=None,
            dereck_beach_adapter=None,
            embedding_policy_adapter=None,
            semantic_chunking_policy_adapter=None,
            contradiction_detection_adapter=None,
            financial_viability_adapter=None,
            policy_processor_adapter=None,
            policy_segmenter_adapter=None,
            # Additional adapters (2 for future expansion)
            causal_processor_adapter=None,
            impact_assessment_adapter=None,
            # Configuration
            responsibility_map_path: Optional[Path] = None,
            circuit_breaker=None,
            # Alternative: Pass ModuleAdapterRegistry directly
            module_adapter_registry=None
    ):
        """
        Initialize ModuleController with all adapter instances via dependency injection
        
        Args:
            teoria_cambio_adapter: ModulosAdapter instance
            analyzer_one_adapter: AnalyzerOneAdapter instance
            dereck_beach_adapter: DerekBeachAdapter instance
            embedding_policy_adapter: EmbeddingPolicyAdapter instance
            semantic_chunking_policy_adapter: SemanticChunkingPolicyAdapter instance
            contradiction_detection_adapter: ContradictionDetectionAdapter instance
            financial_viability_adapter: FinancialViabilityAdapter instance
            policy_processor_adapter: PolicyProcessorAdapter instance
            policy_segmenter_adapter: PolicySegmenterAdapter instance
            causal_processor_adapter: Optional CausalProcessorAdapter instance
            impact_assessment_adapter: Optional ImpactAssessmentAdapter instance
            responsibility_map_path: Path to responsibility_map.json
            circuit_breaker: Optional CircuitBreaker instance for fault tolerance
        """
        logger.info("Initializing ModuleController with dependency injection")
        
        # If module_adapter_registry is provided, use it to get adapters
        if module_adapter_registry is not None:
            logger.info("Using ModuleAdapterRegistry to populate adapters")
            self.adapters = module_adapter_registry.adapters.copy()
            self.module_adapter_registry = module_adapter_registry
        else:
            # Store all adapter instances (11 total)
            self.adapters = {
                "teoria_cambio": teoria_cambio_adapter,
                "analyzer_one": analyzer_one_adapter,
                "dereck_beach": dereck_beach_adapter,
                "embedding_policy": embedding_policy_adapter,
                "semantic_chunking_policy": semantic_chunking_policy_adapter,
                "contradiction_detection": contradiction_detection_adapter,
                "financial_viability": financial_viability_adapter,
                "policy_processor": policy_processor_adapter,
                "policy_segmenter": policy_segmenter_adapter,
                "causal_processor": causal_processor_adapter,
                "impact_assessment": impact_assessment_adapter
            }
            
            # Filter out None values (optional adapters)
            self.adapters = {k: v for k, v in self.adapters.items() if v is not None}
            self.module_adapter_registry = None
        
        self.circuit_breaker = circuit_breaker
        self.responsibility_map_path = responsibility_map_path or Path(__file__).parent / "responsibility_map.json"
        
        # Load responsibility mapping
        self.responsibility_map = self._load_responsibility_map()
        
        # Performance tracking
        self.performance_metrics = defaultdict(lambda: {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "avg_execution_time": 0.0
        })
        
        logger.info(
            f"ModuleController initialized: {len(self.adapters)} adapters registered, "
            f"responsibility map loaded from {self.responsibility_map_path}"
        )

    def _load_responsibility_map(self) -> Dict[str, Any]:
        """
        Load responsibility mapping from JSON configuration
        
        Returns:
            Dictionary with dimension/policy area to adapter mappings
            
        Raises:
            FileNotFoundError: If responsibility_map.json not found
            ValueError: If JSON is invalid
        """
        if not self.responsibility_map_path.exists():
            logger.warning(f"Responsibility map not found at {self.responsibility_map_path}, creating default")
            return self._create_default_responsibility_map()
        
        logger.info(f"Loading responsibility map from {self.responsibility_map_path}")
        
        try:
            with open(self.responsibility_map_path, 'r', encoding='utf-8') as f:
                mapping = json.load(f)
            
            logger.info(
                f"Loaded responsibility map: "
                f"{len(mapping.get('dimensions', {}))} dimensions, "
                f"{len(mapping.get('policy_areas', {}))} policy areas"
            )
            
            return mapping
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in responsibility_map.json: {e}")
            raise ValueError(f"Invalid responsibility map JSON: {e}")

    def _create_default_responsibility_map(self) -> Dict[str, Any]:
        """
        Create default responsibility mapping based on FARFAN architecture
        
        Returns:
            Default responsibility map structure
        """
        default_map = {
            "metadata": {
                "version": "1.0",
                "created": "2025-01-15",
                "description": "Default responsibility mapping for FARFAN 3.0"
            },
            "dimensions": {
                "D1": {
                    "name": "Diagnóstico y Recursos",
                    "primary_adapters": ["policy_segmenter", "policy_processor"],
                    "secondary_adapters": ["semantic_chunking_policy", "financial_viability"],
                    "execution_strategy": "parallel"
                },
                "D2": {
                    "name": "Diseño de Intervención",
                    "primary_adapters": ["embedding_policy", "policy_processor"],
                    "secondary_adapters": ["dereck_beach", "teoria_cambio"],
                    "execution_strategy": "parallel"
                },
                "D3": {
                    "name": "Productos y Outputs",
                    "primary_adapters": ["financial_viability", "analyzer_one"],
                    "secondary_adapters": ["policy_processor"],
                    "execution_strategy": "sequential"
                },
                "D4": {
                    "name": "Resultados y Outcomes",
                    "primary_adapters": ["analyzer_one", "teoria_cambio"],
                    "secondary_adapters": ["dereck_beach"],
                    "execution_strategy": "parallel"
                },
                "D5": {
                    "name": "Impactos de Largo Plazo",
                    "primary_adapters": ["teoria_cambio", "dereck_beach"],
                    "secondary_adapters": ["contradiction_detection"],
                    "execution_strategy": "sequential"
                },
                "D6": {
                    "name": "Teoría de Cambio",
                    "primary_adapters": ["teoria_cambio", "dereck_beach"],
                    "secondary_adapters": ["analyzer_one", "contradiction_detection"],
                    "execution_strategy": "parallel"
                }
            },
            "policy_areas": {
                "P1": {"name": "Salud", "specialized_adapters": ["financial_viability"]},
                "P2": {"name": "Educación", "specialized_adapters": ["financial_viability"]},
                "P3": {"name": "Infraestructura", "specialized_adapters": ["financial_viability"]},
                "P4": {"name": "Economía", "specialized_adapters": ["financial_viability"]},
                "P5": {"name": "Medio Ambiente", "specialized_adapters": ["contradiction_detection"]},
                "P6": {"name": "Seguridad", "specialized_adapters": []},
                "P7": {"name": "Cultura", "specialized_adapters": []},
                "P8": {"name": "Deporte", "specialized_adapters": []},
                "P9": {"name": "Tecnología", "specialized_adapters": []},
                "P10": {"name": "Desarrollo Social", "specialized_adapters": ["analyzer_one"]}
            },
            "method_routing": {
                "semantic_analysis": ["embedding_policy", "semantic_chunking_policy"],
                "causal_inference": ["teoria_cambio", "dereck_beach"],
                "financial_analysis": ["financial_viability"],
                "contradiction_detection": ["contradiction_detection"],
                "text_processing": ["policy_processor", "policy_segmenter"]
            }
        }
        
        return default_map

    def route_question(
            self,
            question_id: str,
            dimension: str,
            policy_area: str
    ) -> RoutingDecision:
        """
        Route question to appropriate adapters based on responsibility map
        
        Args:
            question_id: Canonical question ID (e.g., "P1-D1-Q1")
            dimension: Dimension ID (e.g., "D1")
            policy_area: Policy area ID (e.g., "P1")
            
        Returns:
            RoutingDecision with primary and secondary adapters
        """
        dimension_map = self.responsibility_map.get("dimensions", {}).get(dimension, {})
        policy_map = self.responsibility_map.get("policy_areas", {}).get(policy_area, {})
        
        primary_adapters = dimension_map.get("primary_adapters", [])
        secondary_adapters = dimension_map.get("secondary_adapters", [])
        specialized_adapters = policy_map.get("specialized_adapters", [])
        
        # Merge and deduplicate adapters
        all_secondary = list(set(secondary_adapters + specialized_adapters))
        
        execution_strategy = dimension_map.get("execution_strategy", "sequential")
        
        decision = RoutingDecision(
            question_id=question_id,
            dimension=dimension,
            policy_area=policy_area,
            primary_adapter=primary_adapters[0] if primary_adapters else "policy_processor",
            secondary_adapters=all_secondary,
            execution_strategy=execution_strategy,
            confidence_threshold=0.70
        )
        
        logger.debug(f"Routed {question_id} → primary={decision.primary_adapter}, secondary={all_secondary}")
        
        return decision

    def execute_adapter_method(
            self,
            adapter_name: str,
            method_name: str,
            args: List[Any],
            kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute method on specified adapter with circuit breaker protection
        
        This is the unified interface that replaces direct adapter calls throughout
        the orchestrator modules.
        
        Args:
            adapter_name: Name of adapter (e.g., "teoria_cambio")
            method_name: Name of method to call
            args: Positional arguments
            kwargs: Keyword arguments
            
        Returns:
            Dictionary with standardized result format
        """
        start_time = time.time()
        
        # Validate adapter exists
        if adapter_name not in self.adapters:
            logger.error(f"Adapter '{adapter_name}' not found in controller")
            return self._create_error_result(
                adapter_name,
                method_name,
                f"Adapter not found. Available: {list(self.adapters.keys())}",
                start_time
            )
        
        adapter = self.adapters[adapter_name]
        
        # Check if adapter is available
        if hasattr(adapter, 'available') and not adapter.available:
            logger.warning(f"Adapter '{adapter_name}' is not available")
            return self._create_error_result(
                adapter_name,
                method_name,
                "Adapter not available",
                start_time
            )
        
        # Check circuit breaker
        if self.circuit_breaker and not self.circuit_breaker.can_execute(adapter_name):
            logger.warning(f"Circuit breaker OPEN for '{adapter_name}'")
            return self._create_error_result(
                adapter_name,
                method_name,
                "Circuit breaker open",
                start_time
            )
        
        # Execute adapter method
        try:
            logger.debug(f"Executing {adapter_name}.{method_name}")
            
            # Use adapter's execute method if available (standardized in module_adapters.py)
            if hasattr(adapter, 'execute'):
                result = adapter.execute(method_name, args, kwargs)
            elif self.module_adapter_registry:
                # Use registry's execute_module_method for standardized execution
                module_result = self.module_adapter_registry.execute_module_method(
                    module_name=adapter_name,
                    method_name=method_name,
                    args=args,
                    kwargs=kwargs
                )
                # Convert ModuleResult to dict
                result = module_result.to_dict() if hasattr(module_result, 'to_dict') else {
                    'status': module_result.status,
                    'data': module_result.data,
                    'evidence': module_result.evidence,
                    'confidence': module_result.confidence
                }
            else:
                # Direct method call (fallback)
                method = getattr(adapter, method_name)
                result = method(*args, **kwargs)
            
            execution_time = time.time() - start_time
            
            # Record success
            self._record_success(adapter_name, execution_time)
            
            if self.circuit_breaker:
                self.circuit_breaker.record_success(adapter_name, execution_time)
            
            # Normalize result format
            normalized_result = self._normalize_result(result, adapter_name, method_name, execution_time)
            
            return normalized_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Error executing {adapter_name}.{method_name}: {e}", exc_info=True)
            
            # Record failure
            self._record_failure(adapter_name, execution_time)
            
            if self.circuit_breaker:
                self.circuit_breaker.record_failure(adapter_name, str(e), execution_time)
            
            return self._create_error_result(adapter_name, method_name, str(e), start_time)

    def process_question(
            self,
            question_spec: Any,
            plan_text: str
    ) -> Dict[str, Any]:
        """
        Process complete question using routed adapters
        
        High-level method that:
        1. Routes question to appropriate adapters
        2. Executes adapters in order (parallel or sequential)
        3. Aggregates results
        4. Returns unified response
        
        Args:
            question_spec: Question specification from questionnaire parser
            plan_text: Plan document text
            
        Returns:
            Dictionary with aggregated results from all adapters
        """
        question_id = getattr(question_spec, 'canonical_id', 'unknown')
        dimension = self._extract_dimension(question_id)
        policy_area = self._extract_policy_area(question_id)
        
        routing_decision = self.route_question(question_id, dimension, policy_area)
        
        results = {
            "question_id": question_id,
            "routing_decision": routing_decision.to_dict(),
            "adapter_results": {},
            "aggregated_confidence": 0.0,
            "execution_strategy": routing_decision.execution_strategy
        }
        
        # Execute primary adapter
        primary_result = self._execute_adapter_for_question(
            routing_decision.primary_adapter,
            question_spec,
            plan_text
        )
        results["adapter_results"][routing_decision.primary_adapter] = primary_result
        
        # Execute secondary adapters
        for adapter_name in routing_decision.secondary_adapters:
            if adapter_name in self.adapters:
                secondary_result = self._execute_adapter_for_question(
                    adapter_name,
                    question_spec,
                    plan_text
                )
                results["adapter_results"][adapter_name] = secondary_result
        
        # Aggregate confidence scores
        all_confidences = [
            r.get("confidence", 0.0)
            for r in results["adapter_results"].values()
            if r.get("status") == "success"
        ]
        
        if all_confidences:
            results["aggregated_confidence"] = sum(all_confidences) / len(all_confidences)
        
        return results

    def _execute_adapter_for_question(
            self,
            adapter_name: str,
            question_spec: Any,
            plan_text: str
    ) -> Dict[str, Any]:
        """
        Execute adapter for a specific question
        
        Determines appropriate method based on question type and adapter capabilities
        
        Args:
            adapter_name: Name of adapter
            question_spec: Question specification
            plan_text: Plan document text
            
        Returns:
            Result dictionary from adapter execution
        """
        # Get execution chain from question spec if available
        execution_chain = getattr(question_spec, 'execution_chain', [])
        
        # Find matching step for this adapter
        matching_step = None
        for step in execution_chain:
            if isinstance(step, dict) and step.get('adapter') == adapter_name:
                matching_step = step
                break
        
        if matching_step:
            return self.execute_adapter_method(
                adapter_name=adapter_name,
                method_name=matching_step.get('method', 'analyze'),
                args=[plan_text],
                kwargs={}
            )
        else:
            # Default to generic analysis method
            return self.execute_adapter_method(
                adapter_name=adapter_name,
                method_name="analyze",
                args=[plan_text],
                kwargs={}
            )

    def _normalize_result(
            self,
            result: Any,
            adapter_name: str,
            method_name: str,
            execution_time: float
    ) -> Dict[str, Any]:
        """
        Normalize adapter result to standardized format
        
        Args:
            result: Raw result from adapter
            adapter_name: Name of adapter
            method_name: Name of method
            execution_time: Execution time in seconds
            
        Returns:
            Normalized result dictionary
        """
        # Handle ModuleResult dataclass
        if hasattr(result, 'module_name'):
            return {
                "status": "success",
                "adapter_name": adapter_name,
                "method_name": method_name,
                "data": result.data,
                "evidence": result.evidence,
                "confidence": result.confidence,
                "execution_time": execution_time,
                "errors": result.errors,
                "warnings": result.warnings
            }
        
        # Handle dictionary result
        elif isinstance(result, dict):
            return {
                "status": "success",
                "adapter_name": adapter_name,
                "method_name": method_name,
                "data": result,
                "evidence": [],
                "confidence": result.get("confidence", 0.75),
                "execution_time": execution_time,
                "errors": [],
                "warnings": []
            }
        
        # Handle other types
        else:
            return {
                "status": "success",
                "adapter_name": adapter_name,
                "method_name": method_name,
                "data": {"result": str(result)},
                "evidence": [],
                "confidence": 0.50,
                "execution_time": execution_time,
                "errors": [],
                "warnings": []
            }

    def _create_error_result(
            self,
            adapter_name: str,
            method_name: str,
            error_message: str,
            start_time: float
    ) -> Dict[str, Any]:
        """Create standardized error result"""
        return {
            "status": "failed",
            "adapter_name": adapter_name,
            "method_name": method_name,
            "data": {},
            "evidence": [],
            "confidence": 0.0,
            "execution_time": time.time() - start_time,
            "errors": [error_message],
            "warnings": []
        }

    def _record_success(self, adapter_name: str, execution_time: float):
        """Record successful execution for metrics"""
        metrics = self.performance_metrics[adapter_name]
        metrics["total_calls"] += 1
        metrics["successful_calls"] += 1
        
        # Update average execution time
        prev_avg = metrics["avg_execution_time"]
        total = metrics["total_calls"]
        metrics["avg_execution_time"] = (prev_avg * (total - 1) + execution_time) / total

    def _record_failure(self, adapter_name: str, execution_time: float):
        """Record failed execution for metrics"""
        metrics = self.performance_metrics[adapter_name]
        metrics["total_calls"] += 1
        metrics["failed_calls"] += 1
        
        # Update average execution time
        prev_avg = metrics["avg_execution_time"]
        total = metrics["total_calls"]
        metrics["avg_execution_time"] = (prev_avg * (total - 1) + execution_time) / total

    def _extract_dimension(self, question_id: str) -> str:
        """Extract dimension from question ID (e.g., P1-D1-Q1 → D1)"""
        import re
        match = re.search(r'D(\d+)', question_id)
        return f"D{match.group(1)}" if match else "D1"

    def _extract_policy_area(self, question_id: str) -> str:
        """Extract policy area from question ID (e.g., P1-D1-Q1 → P1)"""
        import re
        match = re.search(r'P(\d+)', question_id)
        return f"P{match.group(1)}" if match else "P1"

    def get_controller_status(self) -> Dict[str, Any]:
        """
        Get current controller status and health metrics
        
        Returns:
            Dictionary with adapter availability and performance metrics
        """
        return {
            "total_adapters": len(self.adapters),
            "available_adapters": [
                name for name, adapter in self.adapters.items()
                if not hasattr(adapter, 'available') or adapter.available
            ],
            "performance_metrics": dict(self.performance_metrics),
            "circuit_breaker_status": (
                self.circuit_breaker.get_all_status()
                if self.circuit_breaker else None
            ),
            "responsibility_map_loaded": bool(self.responsibility_map)
        }

    def get_adapter_methods(self, adapter_name: str) -> List[str]:
        """
        Get list of available methods for an adapter
        
        Args:
            adapter_name: Name of adapter
            
        Returns:
            List of method names
        """
        if adapter_name not in self.adapters:
            return []
        
        adapter = self.adapters[adapter_name]
        
        # Get all non-private methods
        methods = [
            method for method in dir(adapter)
            if not method.startswith('_') and callable(getattr(adapter, method))
        ]
        
        return methods


if __name__ == "__main__":
    # Example usage (would need real adapter instances)
    print("=" * 80)
    print("MODULE CONTROLLER - Unified Adapter Interface")
    print("=" * 80)
    print("\nThis module provides centralized adapter orchestration with:")
    print("  - Constructor dependency injection for all adapters")
    print("  - Responsibility-based routing via JSON configuration")
    print("  - Circuit breaker integration for fault tolerance")
    print("  - Standardized result format")
    print("=" * 80)
