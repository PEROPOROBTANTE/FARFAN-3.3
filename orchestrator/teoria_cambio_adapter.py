"""
Teoria Cambio Adapter - Theory of Change Causal Graph Analysis
================================================================

Wraps teoria_cambio.py functionality with standardized interfaces for the module controller.
Preserves all original class signatures while providing alias methods for controller integration.

Author: FARFAN 3.0 Integration Team
Version: 1.0.0
Python: 3.10+
"""

import logging
import time
import warnings
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class AdapterResult:
    """Standardized result format for adapter operations"""
    module_name: str
    class_name: str
    method_name: str
    status: str
    data: Dict[str, Any]
    evidence: List[Dict[str, Any]]
    confidence: float
    execution_time: float
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class TeoriaCambioAdapter:
    """
    Adapter for teoria_cambio.py - Theory of Change Causal Graph System.
    
    Responsibility Map (cuestionario.json):
    - D6 (Causalidad): Q26-Q30 (Causal theory, DAG validation, statistical tests)
    - D5 (Impactos): Q24-Q25 (Long-term impact projections)
    - D4 (Resultados): Q19-Q21 (Result chain validation)
    
    Original Classes:
    - ValidadorTopologiaCausal: Validate causal topology
    - AdvancedDAGValidator: DAG validation with statistical tests
    - ValidationSuite: Comprehensive validation suite
    """

    def __init__(self):
        """Initialize adapter and load teoria_cambio module"""
        self.module_name = "teoria_cambio"
        self.available = False
        self.logger = logging.getLogger(f"{__name__}.TeoriaCambio")
        self._load_module()

    def _load_module(self):
        """Load teoria_cambio module and its components"""
        try:
            from teoria_cambio import (
                CategoriaCausal,
                NodoCausal,
                ValidacionResultado,
                ValidadorTopologiaCausal,
                AdvancedGraphNode,
                AdvancedDAGValidator,
                ValidationSuite,
                GraphType,
            )
            
            self.CategoriaCausal = CategoriaCausal
            self.NodoCausal = NodoCausal
            self.ValidacionResultado = ValidacionResultado
            self.ValidadorTopologiaCausal = ValidadorTopologiaCausal
            self.AdvancedGraphNode = AdvancedGraphNode
            self.AdvancedDAGValidator = AdvancedDAGValidator
            self.ValidationSuite = ValidationSuite
            self.GraphType = GraphType
            
            self.available = True
            self.logger.info(f"✓ {self.module_name} loaded successfully")
            
        except ImportError as e:
            self.logger.warning(f"✗ {self.module_name} NOT available: {e}")
            self.available = False

    # ========================================================================
    # ORIGINAL METHOD SIGNATURES (Preserved)
    # ========================================================================

    def construir_grafo_causal(self) -> Any:
        """
        Original method: ValidadorTopologiaCausal.construir_grafo_causal()
        Maps to cuestionario.json: D6.Q26
        """
        if not self.available:
            raise RuntimeError("Module not available")
        
        validator = self.ValidadorTopologiaCausal()
        return validator.construir_grafo_causal()

    def validacion_completa(self, grafo: Any) -> Any:
        """
        Original method: ValidadorTopologiaCausal.validacion_completa()
        Maps to cuestionario.json: D6.Q27
        """
        if not self.available:
            raise RuntimeError("Module not available")
        
        validator = self.ValidadorTopologiaCausal()
        return validator.validacion_completa(grafo)

    def calculate_acyclicity_pvalue(
        self,
        validator: Any,
        n_simulations: int = 1000,
        alpha: float = 0.05,
        plan_name: str = "default_plan",
    ) -> Dict[str, Any]:
        """
        Original method: AdvancedDAGValidator.calculate_acyclicity_pvalue()
        Maps to cuestionario.json: D6.Q28
        """
        if not self.available:
            raise RuntimeError("Module not available")
        
        return validator.calculate_acyclicity_pvalue(n_simulations, alpha, plan_name)

    def get_graph_stats(self, validator: Any) -> Dict[str, Any]:
        """
        Original method: AdvancedDAGValidator.get_graph_stats()
        Maps to cuestionario.json: D6.Q30
        """
        if not self.available:
            raise RuntimeError("Module not available")
        
        return validator.get_graph_stats()

    def execute_suite(self, suite: Any) -> bool:
        """
        Original method: ValidationSuite.execute_suite()
        Comprehensive validation - Maps to multiple D6 questions
        """
        if not self.available:
            raise RuntimeError("Module not available")
        
        return suite.execute_suite()

    # ========================================================================
    # STANDARDIZED CONTROLLER INTERFACE (Alias Methods)
    # ========================================================================

    def build_causal_graph(self) -> AdapterResult:
        """
        Controller method for D6.Q26: Build causal graph
        Alias for: construir_grafo_causal
        """
        start_time = time.time()
        
        try:
            grafo = self.construir_grafo_causal()
            
            data = {
                "node_count": grafo.number_of_nodes(),
                "edge_count": grafo.number_of_edges(),
                "is_dag": not bool(list(self._find_cycles(grafo))),
            }
            
            return AdapterResult(
                module_name=self.module_name,
                class_name="ValidadorTopologiaCausal",
                method_name="build_causal_graph",
                status="success",
                data=data,
                evidence=[{"type": "causal_graph", "nodes": data["node_count"]}],
                confidence=0.90,
                execution_time=time.time() - start_time,
            )
        except Exception as e:
            return self._error_result("build_causal_graph", start_time, e)

    def validate_causal_topology(self, grafo: Any) -> AdapterResult:
        """
        Controller method for D6.Q27: Validate causal topology
        Alias for: validacion_completa
        """
        start_time = time.time()
        
        try:
            validacion = self.validacion_completa(grafo)
            
            data = {
                "is_valid": validacion.es_valido,
                "error_count": len(validacion.errores),
                "warning_count": len(validacion.advertencias),
                "errors": validacion.errores,
                "warnings": validacion.advertencias,
                "suggestions": validacion.sugerencias,
            }
            
            confidence = 1.0 if validacion.es_valido else 0.4
            
            return AdapterResult(
                module_name=self.module_name,
                class_name="ValidadorTopologiaCausal",
                method_name="validate_causal_topology",
                status="success" if validacion.es_valido else "warnings",
                data=data,
                evidence=[{"type": "topology_validation", "valid": validacion.es_valido}],
                confidence=confidence,
                execution_time=time.time() - start_time,
                warnings=validacion.advertencias if not validacion.es_valido else [],
            )
        except Exception as e:
            return self._error_result("validate_causal_topology", start_time, e)

    def test_dag_acyclicity(
        self,
        graph_type: str = "CAUSAL_DAG",
        n_simulations: int = 1000,
        plan_name: str = "default_plan",
    ) -> AdapterResult:
        """
        Controller method for D6.Q28: Statistical test for DAG acyclicity
        Alias for: calculate_acyclicity_pvalue
        """
        start_time = time.time()
        
        try:
            validator = self.AdvancedDAGValidator(graph_type=self.GraphType[graph_type])
            result = self.calculate_acyclicity_pvalue(
                validator,
                n_simulations=n_simulations,
                plan_name=plan_name,
            )
            
            data = {
                "p_value": result.get("p_value", 0),
                "is_acyclic": result.get("is_acyclic", False),
                "confidence_interval": result.get("confidence_interval", [0, 0]),
                "statistical_power": result.get("statistical_power", 0),
            }
            
            return AdapterResult(
                module_name=self.module_name,
                class_name="AdvancedDAGValidator",
                method_name="test_dag_acyclicity",
                status="success",
                data=data,
                evidence=[{"type": "acyclicity_test", "p_value": data["p_value"]}],
                confidence=data["statistical_power"],
                execution_time=time.time() - start_time,
            )
        except Exception as e:
            return self._error_result("test_dag_acyclicity", start_time, e)

    def get_causal_statistics(self, graph_type: str = "CAUSAL_DAG") -> AdapterResult:
        """
        Controller method for D6.Q30: Get causal graph statistics
        Alias for: get_graph_stats
        """
        start_time = time.time()
        
        try:
            validator = self.AdvancedDAGValidator(graph_type=self.GraphType[graph_type])
            stats = self.get_graph_stats(validator)
            
            return AdapterResult(
                module_name=self.module_name,
                class_name="AdvancedDAGValidator",
                method_name="get_causal_statistics",
                status="success",
                data=stats,
                evidence=[{"type": "graph_statistics"}],
                confidence=0.95,
                execution_time=time.time() - start_time,
            )
        except Exception as e:
            return self._error_result("get_causal_statistics", start_time, e)

    def run_validation_suite(self) -> AdapterResult:
        """
        Controller method for D6 (all questions): Comprehensive validation
        Alias for: execute_suite
        """
        start_time = time.time()
        
        try:
            suite = self.ValidationSuite()
            success = self.execute_suite(suite)
            
            data = {
                "all_tests_passed": success,
                "suite_executed": True,
            }
            
            return AdapterResult(
                module_name=self.module_name,
                class_name="ValidationSuite",
                method_name="run_validation_suite",
                status="success" if success else "warnings",
                data=data,
                evidence=[{"type": "validation_suite", "passed": success}],
                confidence=1.0 if success else 0.5,
                execution_time=time.time() - start_time,
            )
        except Exception as e:
            return self._error_result("run_validation_suite", start_time, e)

    # ========================================================================
    # DEPRECATED SHIM METHODS (With Warnings)
    # ========================================================================

    def validate_theory_of_change(self, plan_text: str) -> Dict[str, Any]:
        """
        DEPRECATED: Use build_causal_graph() + validate_causal_topology() instead.
        Legacy method from earlier refactoring iteration.
        """
        warnings.warn(
            "validate_theory_of_change() is deprecated. "
            "Use build_causal_graph() + validate_causal_topology() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        
        grafo_result = self.build_causal_graph()
        if grafo_result.status != "success":
            return grafo_result.data
        
        grafo = self.construir_grafo_causal()
        validation_result = self.validate_causal_topology(grafo)
        return validation_result.data

    def check_dag_validity(self, graph: Any) -> bool:
        """
        DEPRECATED: Use validate_causal_topology() instead.
        Returns only boolean instead of full validation result.
        """
        warnings.warn(
            "check_dag_validity() is deprecated. Use validate_causal_topology() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        
        result = self.validate_causal_topology(graph)
        return result.data.get("is_valid", False)

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _find_cycles(self, grafo: Any) -> List[List[str]]:
        """Find cycles in graph (if networkx available)"""
        try:
            import networkx as nx
            cycles = list(nx.simple_cycles(grafo))
            return cycles
        except:
            return []

    def _error_result(self, method_name: str, start_time: float, error: Exception) -> AdapterResult:
        """Create error result"""
        return AdapterResult(
            module_name=self.module_name,
            class_name="Unknown",
            method_name=method_name,
            status="failed",
            data={},
            evidence=[],
            confidence=0.0,
            execution_time=time.time() - start_time,
            errors=[str(error)],
        )


def create_teoria_cambio_adapter() -> TeoriaCambioAdapter:
    """Factory function to create TeoriaCambioAdapter instance"""
    return TeoriaCambioAdapter()
