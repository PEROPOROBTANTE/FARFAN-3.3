"""
TeoriaCambio Adapter Layer
===========================

Backward-compatible adapter wrapping teoria_cambio.py functionality.
Provides translation layer between legacy 11-adapter architecture and unified module controller.

This adapter preserves existing method signatures while delegating to the core domain module.
"""

import logging
import warnings
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


class TeoriaCambioAdapter:
    """
    Adapter for teoria_cambio.py - Theory of Change Validation Framework.
    
    Wraps TeoriaCambio validator, AdvancedDAGValidator, and 
    IndustrialGradeValidator for causal model validation.
    
    PRIMARY INTERFACE (Backward Compatible):
    - validate_theory_of_change(model: Dict) -> Dict[str, Any]
    - validate_causal_dag(graph: nx.DiGraph) -> Dict[str, Any]
    - check_hierarchical_consistency(model: Dict) -> Dict[str, bool]
    - run_monte_carlo_validation(dag: nx.DiGraph, n_simulations: int) -> Dict[str, Any]
    - audit_validator_performance() -> Dict[str, Any]
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize adapter with optional configuration.
        
        Args:
            config: Configuration dict for validator (optional)
        """
        self._load_core_module()
        self._initialize_components(config)
        logger.info("TeoriaCambioAdapter initialized successfully")

    def _load_core_module(self):
        """Load core domain module components"""
        try:
            from teoria_cambio import (
                TeoriaCambio,
                AdvancedDAGValidator,
                IndustrialGradeValidator,
                CategoriaCausal
            )
            
            self.TeoriaCambio = TeoriaCambio
            self.AdvancedDAGValidator = AdvancedDAGValidator
            self.IndustrialGradeValidator = IndustrialGradeValidator
            self.CategoriaCausal = CategoriaCausal
            self._module_available = True
            
        except ImportError as e:
            logger.error(f"Failed to load teoria_cambio module: {e}")
            self._module_available = False
            raise RuntimeError(f"Core module teoria_cambio not available: {e}")

    def _initialize_components(self, config: Optional[Dict[str, Any]]):
        """Initialize teoria cambio components"""
        self.validator = self.TeoriaCambio()
        self.dag_validator = self.AdvancedDAGValidator()
        self.industrial_validator = self.IndustrialGradeValidator()
        self.config = config or {}

    # ========================================================================
    # PRIMARY INTERFACE (Backward Compatible)
    # ========================================================================

    def validate_theory_of_change(
        self, 
        model: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Validate theory of change model against axioms.
        
        Args:
            model: Theory of change model with nodes and edges
            **kwargs: Additional validation parameters
            
        Returns:
            Validation results with passed tests, errors, warnings
        """
        # Convert dict model to graph if needed
        if 'graph' in model:
            graph = model['graph']
        else:
            graph = self._dict_to_graph(model)
        
        # Run validation
        validation_result = self.validator.validar_modelo_completo(graph)
        
        return {
            'valid': validation_result.get('es_valido', False),
            'passed_tests': validation_result.get('pruebas_pasadas', []),
            'failed_tests': validation_result.get('pruebas_fallidas', []),
            'warnings': validation_result.get('advertencias', []),
            'recommendations': validation_result.get('recomendaciones', [])
        }

    def validate_causal_dag(
        self, 
        graph: nx.DiGraph,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Validate causal DAG structure and properties.
        
        Args:
            graph: NetworkX DiGraph representing causal model
            **kwargs: Additional validation parameters
            
        Returns:
            Validation results with acyclicity, connectivity, robustness
        """
        n_simulations = kwargs.get('n_simulations', 1000)
        
        validation_result = self.dag_validator.validar_con_monte_carlo(
            graph, 
            n_simulations=n_simulations
        )
        
        return {
            'is_acyclic': validation_result.get('es_aciclico', False),
            'is_connected': validation_result.get('esta_conectado', False),
            'robustness_score': validation_result.get('puntaje_robustez', 0.0),
            'structural_power': validation_result.get('poder_estadistico', 0.0),
            'monte_carlo_passed': validation_result.get('monte_carlo_paso', False)
        }

    def check_hierarchical_consistency(
        self, 
        model: Dict[str, Any],
        **kwargs
    ) -> Dict[str, bool]:
        """
        Check hierarchical consistency of causal categories.
        
        Args:
            model: Model with categorized nodes
            **kwargs: Additional checking parameters
            
        Returns:
            Dict of consistency checks and their status
        """
        graph = self._dict_to_graph(model)
        
        # Validate hierarchy
        hierarchy_result = self.validator.validar_jerarquia_causal(graph)
        
        return {
            'respects_hierarchy': hierarchy_result.get('jerarquia_valida', False),
            'no_backward_edges': hierarchy_result.get('sin_aristas_inversas', False),
            'proper_sequencing': hierarchy_result.get('secuencia_correcta', True),
            'category_consistency': hierarchy_result.get('categorias_consistentes', True)
        }

    def run_monte_carlo_validation(
        self, 
        dag: nx.DiGraph,
        n_simulations: int = 1000,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation-based validation.
        
        Args:
            dag: Causal DAG to validate
            n_simulations: Number of Monte Carlo simulations
            **kwargs: Additional simulation parameters
            
        Returns:
            Simulation results with convergence and stability metrics
        """
        result = self.dag_validator.validar_con_monte_carlo(dag, n_simulations)
        
        return {
            'simulations_run': n_simulations,
            'convergence_rate': result.get('tasa_convergencia', 0.0),
            'stability_score': result.get('puntaje_estabilidad', 0.0),
            'robustness': result.get('puntaje_robustez', 0.0),
            'passed': result.get('monte_carlo_paso', False)
        }

    def audit_validator_performance(self, **kwargs) -> Dict[str, Any]:
        """
        Audit performance and correctness of validator itself.
        
        Args:
            **kwargs: Additional audit parameters
            
        Returns:
            Audit results with performance metrics and test results
        """
        audit_result = self.industrial_validator.ejecutar_auditoria()
        
        return {
            'tests_passed': audit_result.get('pruebas_pasadas', 0),
            'tests_failed': audit_result.get('pruebas_fallidas', 0),
            'average_execution_time': audit_result.get('tiempo_promedio_ms', 0.0),
            'correctness_score': audit_result.get('puntaje_correctitud', 0.0),
            'recommendations': audit_result.get('recomendaciones', [])
        }

    # ========================================================================
    # LEGACY METHOD ALIASES (with Deprecation Warnings)
    # ========================================================================

    def validate_model(self, model: Dict, **kwargs) -> Dict:
        """
        DEPRECATED: Use validate_theory_of_change() instead.
        """
        warnings.warn(
            "validate_model() is deprecated, use validate_theory_of_change() instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self.validate_theory_of_change(model, **kwargs)

    def validate_dag(self, graph: nx.DiGraph, **kwargs) -> Dict:
        """
        DEPRECATED: Use validate_causal_dag() instead.
        """
        warnings.warn(
            "validate_dag() is deprecated, use validate_causal_dag() instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self.validate_causal_dag(graph, **kwargs)

    def check_hierarchy(self, model: Dict, **kwargs) -> Dict:
        """
        DEPRECATED: Use check_hierarchical_consistency() instead.
        """
        warnings.warn(
            "check_hierarchy() is deprecated, use check_hierarchical_consistency() instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self.check_hierarchical_consistency(model, **kwargs)

    def monte_carlo_test(self, dag: nx.DiGraph, n: int, **kwargs) -> Dict:
        """
        DEPRECATED: Use run_monte_carlo_validation() instead.
        """
        warnings.warn(
            "monte_carlo_test() is deprecated, use run_monte_carlo_validation() instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self.run_monte_carlo_validation(dag, n, **kwargs)

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def _dict_to_graph(self, model: Dict) -> nx.DiGraph:
        """Convert dict model representation to NetworkX DiGraph"""
        G = nx.DiGraph()
        
        # Add nodes with categories
        for node_id, node_data in model.get('nodes', {}).items():
            category = node_data.get('category', 'INSUMOS')
            G.add_node(node_id, category=category, **node_data)
        
        # Add edges
        for edge in model.get('edges', []):
            source = edge.get('source')
            target = edge.get('target')
            if source and target:
                G.add_edge(source, target, **edge)
        
        return G

    def get_causal_categories(self) -> List[str]:
        """Get list of causal category types"""
        return [c.name for c in self.CategoriaCausal]

    def create_empty_model(self) -> Dict[str, Any]:
        """Create empty theory of change model template"""
        return {
            'nodes': {},
            'edges': [],
            'metadata': {
                'version': '1.0',
                'created_by': 'adapter'
            }
        }

    def is_available(self) -> bool:
        """Check if core module is available"""
        return self._module_available
