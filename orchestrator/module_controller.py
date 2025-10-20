"""
Module Controller - Unified Adapter Execution Interface
========================================================

Provides centralized control over all module adapter executions with:
- Unified data structure for analysis results
- Module trace collection with operation results and metadata
- Simplified interface for question processing
- Integration with circuit breaker for fault tolerance

Author: FARFAN Integration Team
Version: 3.0.0
Python: 3.10+
"""

import logging
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ModuleTrace:
    """Trace of a single module method execution"""
    module_name: str
    method_name: str
    status: str  # 'success', 'failed', 'skipped', 'degraded'
    execution_time: float
    result_data: Dict[str, Any]
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UnifiedAnalysisData:
    """
    Unified data structure containing all analysis results
    
    This structure is returned by ModuleController and consumed by ReportAssembly
    """
    question_id: str
    module_traces: List[ModuleTrace]
    operation_results: Dict[str, Any]  # Aggregated results by operation
    execution_metadata: Dict[str, Any]
    total_execution_time: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class ModuleController:
    """
    Centralized controller for module adapter execution
    
    Provides high-level interface for executing module operations through
    ModuleAdapterRegistry while tracking execution traces and aggregating results
    """

    def __init__(
            self,
            module_adapter_registry: Any,
            circuit_breaker: Optional[Any] = None
    ):
        """
        Initialize module controller
        
        Args:
            module_adapter_registry: ModuleAdapterRegistry instance
            circuit_breaker: Optional CircuitBreaker for fault tolerance
        """
        self.registry = module_adapter_registry
        self.circuit_breaker = circuit_breaker
        
        self.execution_stats: Dict[str, Any] = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "skipped_executions": 0
        }
        
        logger.info("ModuleController initialized")

    def execute_module_method(
            self,
            module_name: str,
            method_name: str,
            args: Optional[List[Any]] = None,
            kwargs: Optional[Dict[str, Any]] = None
    ) -> ModuleTrace:
        """
        Execute a single module method with circuit breaker protection
        
        Args:
            module_name: Name of the module adapter
            method_name: Name of the method to execute
            args: Positional arguments for the method
            kwargs: Keyword arguments for the method
            
        Returns:
            ModuleTrace with execution results
        """
        start_time = time.time()
        args = args or []
        kwargs = kwargs or {}
        
        # Check circuit breaker
        if self.circuit_breaker and not self.circuit_breaker.can_execute(module_name):
            logger.warning(f"Circuit breaker open for {module_name}, skipping execution")
            self.execution_stats["skipped_executions"] += 1
            
            return ModuleTrace(
                module_name=module_name,
                method_name=method_name,
                status='skipped',
                execution_time=time.time() - start_time,
                result_data={},
                error="Circuit breaker open"
            )
        
        # Execute through registry
        try:
            module_result = self.registry.execute_module_method(
                module_name=module_name,
                method_name=method_name,
                args=args,
                kwargs=kwargs
            )
            
            execution_time = time.time() - start_time
            
            # Convert ModuleResult to ModuleTrace
            trace = ModuleTrace(
                module_name=module_name,
                method_name=method_name,
                status=module_result.status,
                execution_time=execution_time,
                result_data=module_result.data,
                evidence=module_result.evidence,
                confidence=module_result.confidence,
                error=module_result.errors[0] if module_result.errors else None,
                metadata=module_result.metadata
            )
            
            # Update circuit breaker
            if self.circuit_breaker:
                if module_result.status == 'success':
                    self.circuit_breaker.record_success(module_name, execution_time)
                    self.execution_stats["successful_executions"] += 1
                else:
                    self.circuit_breaker.record_failure(
                        module_name,
                        module_result.errors[0] if module_result.errors else "Unknown error",
                        execution_time
                    )
                    self.execution_stats["failed_executions"] += 1
            
            self.execution_stats["total_executions"] += 1
            
            return trace
            
        except Exception as e:
            logger.error(f"Error executing {module_name}.{method_name}: {e}", exc_info=True)
            
            execution_time = time.time() - start_time
            
            if self.circuit_breaker:
                self.circuit_breaker.record_failure(module_name, str(e), execution_time)
            
            self.execution_stats["failed_executions"] += 1
            self.execution_stats["total_executions"] += 1
            
            return ModuleTrace(
                module_name=module_name,
                method_name=method_name,
                status='failed',
                execution_time=execution_time,
                result_data={},
                error=str(e)
            )

    def execute_question_analysis(
            self,
            question_spec: Any,
            plan_text: str,
            execution_chain: Optional[List[Dict[str, Any]]] = None
    ) -> UnifiedAnalysisData:
        """
        Execute complete analysis for a question
        
        Args:
            question_spec: Question specification
            plan_text: Plan document text
            execution_chain: Optional execution chain (uses question_spec.execution_chain if not provided)
            
        Returns:
            UnifiedAnalysisData with all module traces and aggregated results
        """
        start_time = time.time()
        
        question_id = getattr(question_spec, 'canonical_id', 'unknown')
        execution_chain = execution_chain or getattr(question_spec, 'execution_chain', [])
        
        logger.info(f"Executing question analysis for {question_id} ({len(execution_chain)} steps)")
        
        module_traces: List[ModuleTrace] = []
        operation_results: Dict[str, Any] = {}
        
        # Execute each step in the chain
        for step in execution_chain:
            module_name = step.get('adapter')
            method_name = step.get('method')
            args = step.get('args', [])
            kwargs = step.get('kwargs', {})
            
            if not module_name or not method_name:
                logger.warning(f"Incomplete step in chain: {step}")
                continue
            
            # Prepare arguments (resolve references to plan_text and previous results)
            prepared_args = self._prepare_arguments(args, operation_results, plan_text)
            prepared_kwargs = self._prepare_arguments(kwargs, operation_results, plan_text)
            
            # Execute module method
            trace = self.execute_module_method(
                module_name=module_name,
                method_name=method_name,
                args=prepared_args,
                kwargs=prepared_kwargs
            )
            
            module_traces.append(trace)
            
            # Store result for potential use by later steps
            operation_key = f"{module_name}.{method_name}"
            operation_results[operation_key] = trace.result_data
        
        total_time = time.time() - start_time
        
        # Compile execution metadata
        execution_metadata = {
            "question_id": question_id,
            "total_steps": len(execution_chain),
            "executed_steps": len(module_traces),
            "successful_steps": sum(1 for t in module_traces if t.status == 'success'),
            "failed_steps": sum(1 for t in module_traces if t.status == 'failed'),
            "skipped_steps": sum(1 for t in module_traces if t.status == 'skipped'),
            "modules_used": list(set(t.module_name for t in module_traces)),
            "avg_confidence": sum(t.confidence for t in module_traces) / len(module_traces) if module_traces else 0.0
        }
        
        return UnifiedAnalysisData(
            question_id=question_id,
            module_traces=module_traces,
            operation_results=operation_results,
            execution_metadata=execution_metadata,
            total_execution_time=total_time
        )

    def _prepare_arguments(
            self,
            args_spec: Any,
            previous_results: Dict[str, Any],
            plan_text: str
    ) -> Any:
        """
        Prepare arguments by resolving references
        
        Args:
            args_spec: Argument specification
            previous_results: Results from previous executions
            plan_text: Plan document text
            
        Returns:
            Resolved arguments
        """
        if isinstance(args_spec, list):
            prepared = []
            for arg in args_spec:
                if isinstance(arg, dict):
                    source = arg.get('source')
                    if source == 'plan_text':
                        prepared.append(plan_text)
                    elif source in previous_results:
                        prepared.append(previous_results[source])
                    elif 'value' in arg:
                        prepared.append(arg['value'])
                    else:
                        prepared.append(arg)
                else:
                    prepared.append(arg)
            return prepared
        
        elif isinstance(args_spec, dict):
            prepared = {}
            for key, value in args_spec.items():
                if isinstance(value, dict):
                    source = value.get('source')
                    if source == 'plan_text':
                        prepared[key] = plan_text
                    elif source in previous_results:
                        prepared[key] = previous_results[source]
                    elif 'value' in value:
                        prepared[key] = value['value']
                    else:
                        prepared[key] = value
                else:
                    prepared[key] = value
            return prepared
        
        return args_spec

    def get_available_modules(self) -> List[str]:
        """Get list of available modules"""
        return self.registry.get_available_modules()

    def get_module_status(self, module_name: str) -> Dict[str, Any]:
        """
        Get status for a specific module
        
        Args:
            module_name: Name of the module
            
        Returns:
            Status dictionary including circuit breaker state
        """
        status = {
            "module_name": module_name,
            "available": module_name in self.registry.adapters,
            "circuit_breaker_status": None
        }
        
        if self.circuit_breaker:
            status["circuit_breaker_status"] = self.circuit_breaker.get_adapter_status(module_name)
        
        return status

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        return dict(self.execution_stats)

    def reset_stats(self):
        """Reset execution statistics"""
        self.execution_stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "skipped_executions": 0
        }
        logger.info("Execution statistics reset")
