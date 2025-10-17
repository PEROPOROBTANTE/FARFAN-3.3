# question_router.py - Enhanced with comprehensive module mapping
# Production-ready version with complete 300-question mapping
# Refactored to use QuestionnaireParser as canonical source

import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from .config import CONFIG
from .questionnaire_parser import get_questionnaire_parser, QuestionnaireParser

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Question:
    """Immutable question representation"""
    policy_area: str  # P1-P10
    dimension: str  # D1-D6
    question_num: int  # Q1-Q5
    text: str
    rubric_levels: Dict[str, float]  # "EXCELENTE": 0.85, etc
    verification_patterns: List[str]
    required_modules: List[str]
    primary_module: str  # Main module responsible for answering
    supporting_modules: List[str]  # Additional modules that contribute evidence

    @property
    def canonical_id(self) -> str:
        """Returns P#-D#-Q# notation"""
        return f"{self.policy_area}-{self.dimension}-Q{self.question_num}"


class QuestionRouter:
    """
    Routes each of the 300 questions to appropriate processing modules.

    Enhanced with comprehensive module mapping based on the full inventory
    of available classes and methods.
    
    Uses QuestionnaireParser as the canonical source for question data.
    """

    def __init__(self, cuestionario_path: Optional[Path] = None):
        # Use QuestionnaireParser for all questionnaire data
        self.parser = get_questionnaire_parser(cuestionario_path)
        self.questions: Dict[str, Question] = {}
        self.routing_table: Dict[str, List[str]] = {}
        self._load_questionnaire()
        self._build_routing_table()

    def _load_questionnaire(self):
        """Load the 300-question configuration from QuestionnaireParser"""
        logger.info(f"Loading questionnaire via QuestionnaireParser from {self.parser.questionnaire_path}")

        # Get all policy points and dimensions from parser
        policy_points = self.parser.get_all_policy_points()
        dimensions = self.parser.get_all_dimensions()

        # Generate 300 questions (10 policy areas × 6 dimensions × 5 questions)
        for point_code in sorted(policy_points.keys()):
            for dim_code in sorted(dimensions.keys()):
                dimension = dimensions[dim_code]
                
                # Get base questions for this dimension
                base_questions = self.parser.get_questions_for_dimension(dim_code)

                for q_num in range(1, dimension.num_questions + 1):
                    # Determine required modules based on dimension and question
                    required_modules, primary_module, supporting_modules = self._determine_modules(dim_code, q_num)

                    # Get base question data if available
                    base_q_id = f"{dim_code}-Q{q_num}"
                    base_q = self.parser.get_question(base_q_id)
                    
                    # Get text and patterns
                    if base_q:
                        text = base_q.text_template
                        verification_patterns = base_q.verification_patterns
                    else:
                        text = f"Question {q_num} for {dim_code}"
                        verification_patterns = []
                    
                    # Get rubric from parser
                    rubric_levels = self.parser.get_rubric_for_question(base_q_id)

                    question = Question(
                        policy_area=point_code,
                        dimension=dim_code,
                        question_num=q_num,
                        text=text,
                        rubric_levels=rubric_levels,
                        verification_patterns=verification_patterns,
                        required_modules=required_modules,
                        primary_module=primary_module,
                        supporting_modules=supporting_modules
                    )

                    self.questions[question.canonical_id] = question

        logger.info(f"Loaded {len(self.questions)} questions from QuestionnaireParser")

    def _extract_base_questions(self, dim_data: Dict, dim_code: str) -> List[Dict]:
        """
        Extract base questions from dimension data
        
        NOTE: This method is deprecated. Use parser.get_questions_for_dimension() instead.
        Kept for backward compatibility.
        """
        # Delegate to parser
        questions = self.parser.get_questions_for_dimension(dim_code)
        dimension = self.parser.get_dimension(dim_code)
        num_questions = dimension.num_questions if dimension else 5
        
        # Convert to legacy format for backward compatibility
        base_questions = []
        for q in questions:
            base_questions.append({
                "text": q.text_template,
                "rubric": q.rubric_levels,
                "patterns": q.verification_patterns
            })
        
        # Fill in missing questions with defaults if needed
        while len(base_questions) < num_questions:
            base_questions.append({
                "text": f"Question {len(base_questions) + 1} for {dim_code}",
                "rubric": {
                    "EXCELENTE": 0.85,
                    "BUENO": 0.70,
                    "ACEPTABLE": 0.55,
                    "INSUFICIENTE": 0.0
                },
                "patterns": []
            })

        return base_questions

    def _determine_modules(self, dimension: str, question_num: int) -> Tuple[List[str], str, List[str]]:
        """
        Determine which modules are required for a specific D#-Q# combination.

        Returns:
            Tuple of (all_modules, primary_module, supporting_modules)
        """
        # Comprehensive module mapping based on the full inventory
        module_mapping = {
            "D1": {
                1: (  # Baseline Identification
                    ["semantic_processor", "embedding_policy", "analyzer_one", "policy_segmenter"],
                    "semantic_processor",
                    ["embedding_policy", "analyzer_one", "policy_segmenter"]
                ),
                2: (  # Gap Analysis
                    ["bayesian_integrator", "semantic_processor", "municipal_analyzer", "embedding_analyzer"],
                    "bayesian_integrator",
                    ["semantic_processor", "municipal_analyzer", "embedding_analyzer"]
                ),
                3: (  # Budget Allocation
                    ["financial_analyzer", "dereck_beach", "pdet_analyzer", "causal_processor"],
                    "financial_analyzer",
                    ["dereck_beach", "pdet_analyzer", "causal_processor"]
                ),
                4: (  # Capacity Assessment
                    ["analyzer_one", "municipal_analyzer", "causal_processor", "decologo_processor"],
                    "analyzer_one",
                    ["municipal_analyzer", "causal_processor", "decologo_processor"]
                ),
                5: (  # Restriction Identification
                    ["contradiction_detector", "dereck_beach", "causal_validator", "policy_processor"],
                    "contradiction_detector",
                    ["dereck_beach", "causal_validator", "policy_processor"]
                )
            },
            "D2": {
                1: (  # Activity Format
                    ["policy_segmenter", "semantic_processor", "analyzer_one", "policy_processor"],
                    "policy_segmenter",
                    ["semantic_processor", "analyzer_one", "policy_processor"]
                ),
                2: (  # Mechanism Specification
                    ["dereck_beach", "causal_processor", "pdet_analyzer", "causal_validator"],
                    "dereck_beach",
                    ["causal_processor", "pdet_analyzer", "causal_validator"]
                ),
                3: (  # Causal Links
                    ["causal_processor", "dereck_beach", "pdet_analyzer", "validation_framework"],
                    "causal_processor",
                    ["dereck_beach", "pdet_analyzer", "validation_framework"]
                ),
                4: (  # Risk Assessment
                    ["contradiction_detector", "analyzer_one", "municipal_analyzer", "causal_processor"],
                    "contradiction_detector",
                    ["analyzer_one", "municipal_analyzer", "causal_processor"]
                ),
                5: (  # Sequencing Logic
                    ["contradiction_detector", "causal_processor", "causal_validator", "semantic_processor"],
                    "contradiction_detector",
                    ["causal_processor", "causal_validator", "semantic_processor"]
                )
            },
            "D3": {
                1: (  # DNP Ficha Completeness
                    ["dereck_beach", "policy_processor", "semantic_processor", "pdet_analyzer"],
                    "dereck_beach",
                    ["policy_processor", "semantic_processor", "pdet_analyzer"]
                ),
                2: (  # Indicator Specification
                    ["embedding_policy", "semantic_processor", "bayesian_integrator", "embedding_analyzer"],
                    "embedding_policy",
                    ["semantic_processor", "bayesian_integrator", "embedding_analyzer"]
                ),
                3: (  # Budget Alignment
                    ["financial_analyzer", "dereck_beach", "causal_processor", "pdet_analyzer"],
                    "financial_analyzer",
                    ["dereck_beach", "causal_processor", "pdet_analyzer"]
                ),
                4: (  # Feasibility Assessment
                    ["analyzer_one", "municipal_analyzer", "pdet_analyzer", "causal_processor"],
                    "analyzer_one",
                    ["municipal_analyzer", "pdet_analyzer", "causal_processor"]
                ),
                5: (  # Mechanism Clarity
                    ["dereck_beach", "semantic_processor", "causal_processor", "decologo_processor"],
                    "dereck_beach",
                    ["semantic_processor", "causal_processor", "decologo_processor"]
                )
            },
            "D4": {
                1: (  # Measurability
                    ["embedding_policy", "semantic_processor", "bayesian_integrator", "embedding_analyzer"],
                    "embedding_policy",
                    ["semantic_processor", "bayesian_integrator", "embedding_analyzer"]
                ),
                2: (  # Causal Chain Completeness
                    ["causal_processor", "dereck_beach", "pdet_analyzer", "validation_framework"],
                    "causal_processor",
                    ["dereck_beach", "pdet_analyzer", "validation_framework"]
                ),
                3: (  # Timeframe Specification
                    ["contradiction_detector", "causal_processor", "semantic_processor", "causal_validator"],
                    "contradiction_detector",
                    ["causal_processor", "semantic_processor", "causal_validator"]
                ),
                4: (  # Monitoring Mechanism
                    ["dereck_beach", "analyzer_one", "municipal_analyzer", "policy_processor"],
                    "dereck_beach",
                    ["analyzer_one", "municipal_analyzer", "policy_processor"]
                ),
                5: (  # Strategic Alignment
                    ["semantic_processor", "policy_processor", "decologo_processor", "embedding_analyzer"],
                    "semantic_processor",
                    ["policy_processor", "decologo_processor", "embedding_analyzer"]
                )
            },
            "D5": {
                1: (  # Projection Methodology
                    ["embedding_policy", "pdet_analyzer", "bayesian_integrator", "embedding_analyzer"],
                    "embedding_policy",
                    ["pdet_analyzer", "bayesian_integrator", "embedding_analyzer"]
                ),
                2: (  # Proxy Indicators
                    ["semantic_processor", "embedding_policy", "embedding_analyzer", "analyzer_one"],
                    "semantic_processor",
                    ["embedding_policy", "embedding_analyzer", "analyzer_one"]
                ),
                3: (  # Validity Assessment
                    ["dereck_beach", "causal_processor", "causal_validator", "validation_framework"],
                    "dereck_beach",
                    ["causal_processor", "causal_validator", "validation_framework"]
                ),
                4: (  # Risk Analysis
                    ["contradiction_detector", "pdet_analyzer", "causal_processor", "municipal_analyzer"],
                    "contradiction_detector",
                    ["pdet_analyzer", "causal_processor", "municipal_analyzer"]
                ),
                5: (  # Unwanted Effects
                    ["contradiction_detector", "pdet_analyzer", "causal_processor", "causal_validator"],
                    "contradiction_detector",
                    ["pdet_analyzer", "causal_processor", "causal_validator"]
                )
            },
            "D6": {
                1: (  # Theory of Change
                    ["causal_processor", "dereck_beach", "validation_framework", "decologo_processor"],
                    "causal_processor",
                    ["dereck_beach", "validation_framework", "decologo_processor"]
                ),
                2: (  # Causal Logic
                    ["dereck_beach", "causal_processor", "causal_validator", "bayesian_integrator"],
                    "dereck_beach",
                    ["causal_processor", "causal_validator", "bayesian_integrator"]
                ),
                3: (  # Inconsistency Detection
                    ["contradiction_detector", "causal_processor", "causal_validator", "validation_framework"],
                    "contradiction_detector",
                    ["causal_processor", "causal_validator", "validation_framework"]
                ),
                4: (  # Adaptive Monitoring
                    ["dereck_beach", "analyzer_one", "municipal_analyzer", "policy_processor"],
                    "dereck_beach",
                    ["analyzer_one", "municipal_analyzer", "policy_processor"]
                ),
                5: (  # Differential Approach
                    ["embedding_policy", "semantic_processor", "analyzer_one", "embedding_analyzer"],
                    "embedding_policy",
                    ["semantic_processor", "analyzer_one", "embedding_analyzer"]
                )
            }
        }

        return module_mapping.get(dimension, {}).get(
            question_num,
            (["policy_processor"], "policy_processor", [])
        )

    def _build_routing_table(self):
        """Build the complete routing table for all 300 questions"""
        for q_id, question in self.questions.items():
            self.routing_table[q_id] = question.required_modules

        logger.info(f"Built routing table with {len(self.routing_table)} entries")

    def get_modules_for_question(self, question_id: str) -> List[str]:
        """
        Get the list of modules required to answer a specific question.

        Args:
            question_id: Canonical ID in format "P#-D#-Q#"

        Returns:
            List of module names required for processing
        """
        return self.routing_table.get(question_id, [])

    def get_primary_module_for_question(self, question_id: str) -> str:
        """
        Get the primary module responsible for answering a specific question.

        Args:
            question_id: Canonical ID in format "P#-D#-Q#"

        Returns:
            Name of the primary module
        """
        question = self.questions.get(question_id)
        return question.primary_module if question else "policy_processor"

    def get_supporting_modules_for_question(self, question_id: str) -> List[str]:
        """
        Get the list of supporting modules that contribute evidence for a specific question.

        Args:
            question_id: Canonical ID in format "P#-D#-Q#"

        Returns:
            List of supporting module names
        """
        question = self.questions.get(question_id)
        return question.supporting_modules if question else []

    def get_question(self, question_id: str) -> Optional[Question]:
        """Retrieve a question by its canonical ID"""
        return self.questions.get(question_id)

    def get_all_questions_for_dimension(self, dimension: str) -> List[Question]:
        """Get all 50 questions for a specific dimension (5 questions * 10 policy areas)"""
        return [
            q for q in self.questions.values()
            if q.dimension == dimension
        ]

    def get_all_questions_for_policy_area(self, policy_area: str) -> List[Question]:
        """Get all 30 questions for a specific policy area (5 questions * 6 dimensions)"""
        return [
            q for q in self.questions.values()
            if q.policy_area == policy_area
        ]

    def get_execution_order(self, question_id: str) -> List[Tuple[int, str]]:
        """
        Get the priority-ordered execution sequence for a question's modules.

        Returns:
            List of (priority, module_name) tuples, sorted by priority
        """
        modules = self.get_modules_for_question(question_id)

        # Get priorities from config
        execution_order = [
            (CONFIG.modules[mod].priority, mod)
            for mod in modules
            if mod in CONFIG.modules
        ]

        # Sort by priority (lower number = higher priority)
        execution_order.sort(key=lambda x: x[0])

        return execution_order

    def get_statistics(self) -> Dict[str, any]:
        """Get routing statistics for debugging/validation"""
        module_usage = {}
        primary_module_usage = {}
        supporting_module_usage = {}

        for modules in self.routing_table.values():
            for mod in modules:
                module_usage[mod] = module_usage.get(mod, 0) + 1

        for question in self.questions.values():
            primary = question.primary_module
            primary_module_usage[primary] = primary_module_usage.get(primary, 0) + 1

            for supporting in question.supporting_modules:
                supporting_module_usage[supporting] = supporting_module_usage.get(supporting, 0) + 1

        return {
            "total_questions": len(self.questions),
            "total_routes": len(self.routing_table),
            "module_usage_frequency": module_usage,
            "primary_module_usage": primary_module_usage,
            "supporting_module_usage": supporting_module_usage,
            "avg_modules_per_question": sum(len(v) for v in self.routing_table.values()) / len(self.routing_table),
            "dimensions": list(set(q.dimension for q in self.questions.values())),
            "policy_areas": list(set(q.policy_area for q in self.questions.values()))
        }