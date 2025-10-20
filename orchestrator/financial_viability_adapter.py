"""
Financial Viability Adapter - PDET Financial Analysis with Causal DAG
=======================================================================

Wraps financiero_viabilidad_tablas.py functionality with standardized interfaces for the module controller.
Preserves all original class signatures while providing alias methods for controller integration.

Author: FARFAN 3.0 Integration Team
Version: 1.0.0
Python: 3.10+
"""

import logging
import time
import warnings
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path

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


class FinancialViabilityAdapter:
    """
    Adapter for financiero_viabilidad_tablas.py - PDET Financial Analysis System.
    
    Responsibility Map (cuestionario.json):
    - D1 (Insumos): Q3 (Resource sufficiency), Q5 (Institutional capacity)
    - D3 (Productos): Q17 (Financial indicators), Q18 (Budget coherence)
    - D6 (Causalidad): Q28 (Causal effects), Q30 (Counterfactuals)
    
    Original Classes:
    - PDETFinancialAnalyzer: Comprehensive financial analysis
      - analyze_financial_feasibility: Analyze financial feasibility
      - identify_responsible_entities: Identify responsible entities
      - construct_causal_dag: Construct causal DAG
      - estimate_causal_effects: Estimate causal effects
      - generate_counterfactuals: Generate counterfactual scenarios
      - calculate_quality_score: Calculate quality score
    """

    def __init__(self, use_gpu: bool = False, language: str = 'es'):
        """
        Initialize adapter with optional configuration.
        
        Args:
            use_gpu: Use GPU for processing (injected dependency)
            language: Language for NLP models
        """
        self.module_name = "financial_viability"
        self.available = False
        self.logger = logging.getLogger(f"{__name__}.FinancialViability")
        self._use_gpu = use_gpu
        self._language = language
        self._load_module()

    def _load_module(self):
        """Load financiero_viabilidad_tablas module and its components"""
        try:
            from financiero_viabilidad_tablas import (
                ExtractedTable,
                FinancialIndicator,
                ResponsibleEntity,
                CausalNode,
                CausalEdge,
                CausalDAG,
                CausalEffect,
                CounterfactualScenario,
                QualityScore,
                PDETFinancialAnalyzer,
            )
            
            self.ExtractedTable = ExtractedTable
            self.FinancialIndicator = FinancialIndicator
            self.ResponsibleEntity = ResponsibleEntity
            self.CausalNode = CausalNode
            self.CausalEdge = CausalEdge
            self.CausalDAG = CausalDAG
            self.CausalEffect = CausalEffect
            self.CounterfactualScenario = CounterfactualScenario
            self.QualityScore = QualityScore
            self.PDETFinancialAnalyzer = PDETFinancialAnalyzer
            
            self.available = True
            self.logger.info(f"✓ {self.module_name} loaded successfully")
            
        except ImportError as e:
            self.logger.warning(f"✗ {self.module_name} NOT available: {e}")
            self.available = False

    # ========================================================================
    # ORIGINAL METHOD SIGNATURES (Preserved)
    # ========================================================================

    def analyze_financial_feasibility(
        self,
        analyzer: Any,
        tables: List[Any],
        text: str,
    ) -> Dict[str, Any]:
        """
        Original method: PDETFinancialAnalyzer.analyze_financial_feasibility()
        Maps to cuestionario.json: D1.Q3, D3.Q17
        """
        if not self.available:
            raise RuntimeError("Module not available")
        
        return analyzer.analyze_financial_feasibility(tables, text)

    def identify_responsible_entities(
        self,
        analyzer: Any,
        text: str,
        tables: List[Any],
    ) -> List[Any]:
        """
        Original method: PDETFinancialAnalyzer.identify_responsible_entities()
        Maps to cuestionario.json: D1.Q5
        """
        if not self.available:
            raise RuntimeError("Module not available")
        
        return analyzer.identify_responsible_entities(text, tables)

    def construct_causal_dag(
        self,
        analyzer: Any,
        text: str,
        tables: List[Any],
        financial_analysis: Dict[str, Any],
    ) -> Any:
        """
        Original method: PDETFinancialAnalyzer.construct_causal_dag()
        Maps to cuestionario.json: D6.Q28
        """
        if not self.available:
            raise RuntimeError("Module not available")
        
        return analyzer.construct_causal_dag(text, tables, financial_analysis)

    def estimate_causal_effects(
        self,
        analyzer: Any,
        dag: Any,
        text: str,
        financial_analysis: Dict[str, Any],
    ) -> List[Any]:
        """
        Original method: PDETFinancialAnalyzer.estimate_causal_effects()
        Maps to cuestionario.json: D6.Q28
        """
        if not self.available:
            raise RuntimeError("Module not available")
        
        return analyzer.estimate_causal_effects(dag, text, financial_analysis)

    def generate_counterfactuals(
        self,
        analyzer: Any,
        dag: Any,
        causal_effects: List[Any],
        financial_analysis: Dict[str, Any],
    ) -> List[Any]:
        """
        Original method: PDETFinancialAnalyzer.generate_counterfactuals()
        Maps to cuestionario.json: D6.Q30
        """
        if not self.available:
            raise RuntimeError("Module not available")
        
        return analyzer.generate_counterfactuals(dag, causal_effects, financial_analysis)

    def calculate_quality_score(
        self,
        analyzer: Any,
        text: str,
        tables: List[Any],
        financial_analysis: Dict[str, Any],
        entities: List[Any],
        dag: Any,
        effects: List[Any],
    ) -> Any:
        """
        Original method: PDETFinancialAnalyzer.calculate_quality_score()
        Maps to cuestionario.json: D3.Q18
        """
        if not self.available:
            raise RuntimeError("Module not available")
        
        return analyzer.calculate_quality_score(text, tables, financial_analysis, entities, dag, effects)

    # ========================================================================
    # STANDARDIZED CONTROLLER INTERFACE (Alias Methods)
    # ========================================================================

    def analyze_financial_resources(
        self,
        text: str,
        tables: List[Dict[str, Any]],
    ) -> AdapterResult:
        """
        Controller method for D1.Q3, D3.Q17: Analyze financial resources
        Alias for: analyze_financial_feasibility
        """
        start_time = time.time()
        
        try:
            analyzer = self.PDETFinancialAnalyzer(use_gpu=self._use_gpu, language=self._language)
            
            extracted_tables = []
            for table_data in tables:
                extracted_tables.append(
                    self.ExtractedTable(
                        dataframe=table_data.get("df"),
                        category="budget",
                        confidence=0.8,
                    )
                )
            
            analysis = analyzer.analyze_financial_feasibility(extracted_tables, text)
            
            data = {
                "total_budget": sum(ind.amount for ind in analysis.get("indicators", [])),
                "funding_sources": analysis.get("funding_sources", {}),
                "sustainability": analysis.get("sustainability", {}),
                "risk_level": analysis.get("risk_interpretation", "unknown"),
                "indicator_count": len(analysis.get("indicators", [])),
            }
            
            return AdapterResult(
                module_name=self.module_name,
                class_name="PDETFinancialAnalyzer",
                method_name="analyze_financial_resources",
                status="success",
                data=data,
                evidence=[{"type": "financial_analysis", "indicators": data["indicator_count"]}],
                confidence=0.80,
                execution_time=time.time() - start_time,
            )
        except Exception as e:
            return self._error_result("analyze_financial_resources", start_time, e)

    def identify_institutional_actors(
        self,
        text: str,
        tables: List[Dict[str, Any]],
    ) -> AdapterResult:
        """
        Controller method for D1.Q5: Identify responsible entities
        Alias for: identify_responsible_entities
        """
        start_time = time.time()
        
        try:
            analyzer = self.PDETFinancialAnalyzer(use_gpu=self._use_gpu, language=self._language)
            
            extracted_tables = [
                self.ExtractedTable(
                    dataframe=table_data.get("df"),
                    category="responsibility",
                    confidence=0.8,
                )
                for table_data in tables
            ]
            
            entities = analyzer.identify_responsible_entities(text, extracted_tables)
            
            data = {
                "entity_count": len(entities),
                "entities": [{"name": e.name, "type": e.type, "confidence": e.confidence} for e in entities[:10]],
                "entity_types": list(set(e.type for e in entities)),
            }
            
            return AdapterResult(
                module_name=self.module_name,
                class_name="PDETFinancialAnalyzer",
                method_name="identify_institutional_actors",
                status="success",
                data=data,
                evidence=[{"type": "entity_identification", "count": len(entities)}],
                confidence=0.78,
                execution_time=time.time() - start_time,
            )
        except Exception as e:
            return self._error_result("identify_institutional_actors", start_time, e)

    def build_causal_model(
        self,
        text: str,
        tables: List[Dict[str, Any]],
        financial_analysis: Dict[str, Any],
    ) -> AdapterResult:
        """
        Controller method for D6.Q28: Build causal DAG model
        Alias for: construct_causal_dag
        """
        start_time = time.time()
        
        try:
            analyzer = self.PDETFinancialAnalyzer(use_gpu=self._use_gpu, language=self._language)
            
            extracted_tables = [
                self.ExtractedTable(
                    dataframe=table_data.get("df"),
                    category="budget",
                    confidence=0.8,
                )
                for table_data in tables
            ]
            
            dag = analyzer.construct_causal_dag(text, extracted_tables, financial_analysis)
            
            data = {
                "node_count": len(dag.nodes),
                "edge_count": len(dag.edges),
                "is_acyclic": dag.is_acyclic,
                "nodes": list(dag.nodes.keys())[:10],
            }
            
            return AdapterResult(
                module_name=self.module_name,
                class_name="PDETFinancialAnalyzer",
                method_name="build_causal_model",
                status="success",
                data=data,
                evidence=[{"type": "causal_dag", "nodes": len(dag.nodes)}],
                confidence=0.82,
                execution_time=time.time() - start_time,
            )
        except Exception as e:
            return self._error_result("build_causal_model", start_time, e)

    def estimate_causal_impacts(
        self,
        dag: Any,
        text: str,
        financial_analysis: Dict[str, Any],
    ) -> AdapterResult:
        """
        Controller method for D6.Q28: Estimate causal effects
        Alias for: estimate_causal_effects
        """
        start_time = time.time()
        
        try:
            analyzer = self.PDETFinancialAnalyzer(use_gpu=self._use_gpu, language=self._language)
            effects = analyzer.estimate_causal_effects(dag, text, financial_analysis)
            
            data = {
                "effect_count": len(effects),
                "effects": [
                    {
                        "treatment": e.treatment,
                        "outcome": e.outcome,
                        "effect_size": e.effect_size,
                        "confidence": e.confidence,
                    }
                    for e in effects[:10]
                ],
                "avg_effect_size": sum(e.effect_size for e in effects) / len(effects) if effects else 0,
            }
            
            return AdapterResult(
                module_name=self.module_name,
                class_name="PDETFinancialAnalyzer",
                method_name="estimate_causal_impacts",
                status="success",
                data=data,
                evidence=[{"type": "causal_effects", "count": len(effects)}],
                confidence=0.75,
                execution_time=time.time() - start_time,
            )
        except Exception as e:
            return self._error_result("estimate_causal_impacts", start_time, e)

    def generate_policy_scenarios(
        self,
        dag: Any,
        causal_effects: List[Any],
        financial_analysis: Dict[str, Any],
    ) -> AdapterResult:
        """
        Controller method for D6.Q30: Generate counterfactual scenarios
        Alias for: generate_counterfactuals
        """
        start_time = time.time()
        
        try:
            analyzer = self.PDETFinancialAnalyzer(use_gpu=self._use_gpu, language=self._language)
            scenarios = analyzer.generate_counterfactuals(dag, causal_effects, financial_analysis)
            
            data = {
                "scenario_count": len(scenarios),
                "scenarios": [
                    {
                        "description": s.description,
                        "expected_outcomes": s.expected_outcomes,
                        "confidence": s.confidence,
                    }
                    for s in scenarios[:5]
                ],
            }
            
            return AdapterResult(
                module_name=self.module_name,
                class_name="PDETFinancialAnalyzer",
                method_name="generate_policy_scenarios",
                status="success",
                data=data,
                evidence=[{"type": "counterfactuals", "count": len(scenarios)}],
                confidence=0.72,
                execution_time=time.time() - start_time,
            )
        except Exception as e:
            return self._error_result("generate_policy_scenarios", start_time, e)

    # ========================================================================
    # DEPRECATED SHIM METHODS (With Warnings)
    # ========================================================================

    def analyze_budget(self, text: str, tables: List) -> Dict[str, Any]:
        """
        DEPRECATED: Use analyze_financial_resources() instead.
        Legacy method from earlier refactoring iteration.
        """
        warnings.warn(
            "analyze_budget() is deprecated. Use analyze_financial_resources() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        result = self.analyze_financial_resources(text, tables)
        return result.data

    def get_responsible_entities(self, document: str) -> List[str]:
        """
        DEPRECATED: Use identify_institutional_actors() instead.
        Returns only entity names instead of full result.
        """
        warnings.warn(
            "get_responsible_entities() is deprecated. Use identify_institutional_actors() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        result = self.identify_institutional_actors(document, [])
        return [e["name"] for e in result.data.get("entities", [])]

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

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


def create_financial_viability_adapter(
    use_gpu: bool = False,
    language: str = 'es',
) -> FinancialViabilityAdapter:
    """Factory function to create FinancialViabilityAdapter instance"""
    return FinancialViabilityAdapter(use_gpu=use_gpu, language=language)
