# report_assembly.py - Updated for full integration with new architecture
# coding=utf-8
"""
Report Assembly - MICRO/MESO/MACRO multi-level reporting
Fully integrated with:
- questionnaire_parser.py (question specs and scoring)
- module_adapters.py (standardized ModuleResult)
- rubric_scoring.json (scoring modalities)
- execution_mapping.yaml (execution chains)
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import statistics
import re
from datetime import datetime

from .config import CONFIG
from .choreographer import ExecutionResult, ExecutionStatus

logger = logging.getLogger(__name__)


@dataclass
class MicroLevelAnswer:
    """MICRO level: Individual question answer with full traceability"""
    question_id: str  # P#-D#-Q#
    qualitative_note: str  # EXCELENTE/BUENO/ACEPTABLE/INSUFICIENTE
    quantitative_score: float  # 0.0-3.0 (question level)
    evidence: List[str]  # Extracts from plan
    explanation: str  # 150-300 words, doctoral level
    confidence: float  # 0.0-1.0
    
    # Detailed scoring breakdown
    scoring_modality: str  # TYPE_A, TYPE_B, etc.
    elements_found: Dict[str, bool]  # Which expected elements were found
    search_pattern_matches: Dict[str, Any]  # Pattern matching results
    
    # Module execution details
    modules_executed: List[str]  # Modules that ran
    module_results: Dict[str, Any]  # Module outputs
    execution_time: float  # Total execution time
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MesoLevelCluster:
    """MESO level: Cluster aggregation"""
    cluster_name: str  # CLUSTER_1, CLUSTER_2, etc
    cluster_description: str
    policy_areas: List[str]  # [P1, P2, etc]
    avg_score: float  # 0-100 percentage
    dimension_scores: Dict[str, float]  # D1: 75.0, D2: 65.0, etc (percentages)
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]
    question_coverage: float  # % of questions answered
    total_questions: int
    answered_questions: int
    evidence_quality: Dict[str, float] = field(default_factory=dict)


@dataclass
class MacroLevelConvergence:
    """MACRO level: Overall convergence with Decálogo"""
    overall_score: float  # 0-100 percentage
    convergence_by_dimension: Dict[str, float]  # D1-D6 percentages
    convergence_by_policy_area: Dict[str, float]  # P1-P10 percentages
    gap_analysis: Dict[str, Any]
    agenda_alignment: float  # 0.0-1.0
    critical_gaps: List[str]
    strategic_recommendations: List[str]
    plan_classification: str  # "EXCELENTE"/"BUENO"/"SATISFACTORIO"/"INSUFICIENTE"/"DEFICIENTE"
    evidence_synthesis: Dict[str, Any] = field(default_factory=dict)
    implementation_roadmap: List[Dict[str, Any]] = field(default_factory=list)


class ReportAssembler:
    """
    Assembles comprehensive reports at three levels with full scoring integration
    """

    def __init__(self, dimension_descriptions: Optional[Dict[str, str]] = None):
        # Rubric levels (percentage thresholds)
        self.rubric_levels = {
            "EXCELENTE": (85, 100),
            "BUENO": (70, 84),
            "SATISFACTORIO": (55, 69),
            "INSUFICIENTE": (40, 54),
            "DEFICIENTE": (0, 39)
        }

        # Question-level rubric (0-3 scale)
        self.question_rubric = {
            "EXCELENTE": (2.55, 3.00),  # 85% of 3
            "BUENO": (2.10, 2.54),      # 70% of 3
            "ACEPTABLE": (1.65, 2.09),  # 55% of 3
            "INSUFICIENTE": (0.00, 1.64)  # Below 55%
        }

        self.dimension_descriptions = dimension_descriptions or {
            "D1": "Diagnóstico y Recursos - Líneas base, magnitud del problema, recursos y capacidades",
            "D2": "Diseño de Intervención - Actividades, mecanismos causales, secuencias temporales",
            "D3": "Productos y Outputs - Entregables, verificación, presupuesto",
            "D4": "Resultados y Outcomes - Indicadores de resultado, causalidad mediano plazo",
            "D5": "Impactos y Efectos de Largo Plazo - Transformación estructural, sostenibilidad",
            "D6": "Teoría de Cambio y Coherencia Causal - Coherencia causal global, auditoría completa"
        }

    # ============================================================================
    # MICRO LEVEL - Question-by-Question Analysis
    # ============================================================================

    def generate_micro_answer(
            self,
            question_spec,  # QuestionSpec from questionnaire_parser
            execution_results: Dict[str, ExecutionResult],
            plan_text: str
    ) -> MicroLevelAnswer:
        """
        Generate MICRO-level answer for a single question using rubric scoring
        """
        logger.info(f"Generating MICRO answer for {question_spec.canonical_id}")

        start_time = datetime.now()

        # Step 1: Apply scoring modality
        score, elements_found, pattern_matches = self._apply_scoring_modality(
            question_spec,
            execution_results,
            plan_text
        )

        # Step 2: Map to qualitative level (question scale 0-3)
        qualitative = self._score_to_qualitative_question(score)

        # Step 3: Aggregate evidence from execution results
        evidence_excerpts = self._extract_evidence_excerpts(
            question_spec,
            execution_results,
            elements_found,
            plan_text
        )

        # Step 4: Calculate confidence
        confidence = self._calculate_confidence(
            execution_results,
            elements_found,
            pattern_matches
        )

        # Step 5: Generate doctoral-level explanation