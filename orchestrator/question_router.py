"""
Question Router - Deterministic mapping of 300 questions to processing modules
Implements the canonical P#-D#-Q# notation system
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from .config import CONFIG

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class Question:
    """Immutable question representation"""
    policy_area: str  # P1-P10
    dimension: str    # D1-D6
    question_num: int # Q1-Q5
    text: str
    rubric_levels: Dict[str, float]  # "EXCELENTE": 0.85, etc
    verification_patterns: List[str]
    required_modules: List[str]

    @property
    def canonical_id(self) -> str:
        """Returns P#-D#-Q# notation"""
        return f"{self.policy_area}-{self.dimension}-Q{self.question_num}"


class QuestionRouter:
    """
    Routes each of the 300 questions to appropriate processing modules.

    Strategy:
    - D1-D4: Heavy use of all extractors (policy_processor, analyzer_one, etc)
    - D5: Focus on embeddings and long-term projection
    - D6: Contradiction detection + Derek Beach causal framework
    """

    def __init__(self, cuestionario_path: Optional[Path] = None):
        self.cuestionario_path = cuestionario_path or CONFIG.cuestionario_path
        self.questions: Dict[str, Question] = {}
        self.routing_table: Dict[str, List[str]] = {}
        self._load_questionnaire()
        self._build_routing_table()

    def _load_questionnaire(self):
        """Load the 300-question configuration"""
        logger.info(f"Loading questionnaire from {self.cuestionario_path}")

        with open(self.cuestionario_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # The questionnaire has 30 base questions (5 per dimension * 6 dimensions)
        # replicated across 10 policy areas = 300 total questions

        policy_areas = [f"P{i}" for i in range(1, 11)]
        dimensions_data = {
            f"D{i}": data["dimensiones"][f"D{i}"]
            for i in range(1, 7)
        }

        for policy_area in policy_areas:
            for dim_code, dim_data in dimensions_data.items():
                # Get base questions for this dimension
                base_questions = self._extract_base_questions(dim_data, dim_code)

                for q_num, q_data in enumerate(base_questions, start=1):
                    question = Question(
                        policy_area=policy_area,
                        dimension=dim_code,
                        question_num=q_num,
                        text=q_data["text"],
                        rubric_levels=q_data.get("rubric", {}),
                        verification_patterns=q_data.get("patterns", []),
                        required_modules=self._determine_required_modules(dim_code, q_num)
                    )

                    self.questions[question.canonical_id] = question

        logger.info(f"Loaded {len(self.questions)} questions")

    def _extract_base_questions(self, dim_data: Dict, dim_code: str) -> List[Dict]:
        """Extract base questions from dimension data"""
        # This is a simplified extraction - in production,
        # you'd parse the full question structure from cuestionario.json

        num_questions = dim_data.get("preguntas", 5)
        base_questions = []

        for i in range(1, num_questions + 1):
            # Placeholder - replace with actual question text from JSON
            base_questions.append({
                "text": f"{dim_data['nombre']} - Question {i}",
                "rubric": {
                    "EXCELENTE": 0.85,
                    "BUENO": 0.70,
                    "ACEPTABLE": 0.55,
                    "INSUFICIENTE": 0.0
                },
                "patterns": dim_data.get("causal_verification_template", {}).get("validation_patterns", [])
            })

        return base_questions

    def _determine_required_modules(self, dimension: str, question_num: int) -> List[str]:
        """
        Determine which modules are required for a specific D#-Q# combination.

        Mapping logic (based on your system architecture):

        D1 (Insumos):
        - Q1/Q2 (Baselines/Gaps): policy_processor + embedding_policy + analyzer_one
        - Q3 (Budget): financial_viability + policy_processor
        - Q4/Q5 (Capacity/Restrictions): policy_processor + causal_processor

        D2 (Actividades):
        - Q1 (Format): analyzer_one + policy_segmenter
        - Q2/Q3 (Mechanisms/Links): causal_processor + policy_processor
        - Q4 (Risks): embedding_policy + analyzer_one
        - Q5 (Sequencing): causal_processor + policy_processor

        D3 (Productos):
        - Q1 (DNP Ficha): policy_processor + financial_viability
        - Q2/Q3 (Indicators/Budget): embedding_policy + financial_viability
        - Q4/Q5 (Feasibility/Mechanism): causal_processor + analyzer_one

        D4 (Resultados):
        - Q1/Q2 (Measurable/Chain): causal_processor + policy_processor
        - Q3/Q4 (Timeframe/Monitoring): analyzer_one + embedding_policy
        - Q5 (Alignment): policy_processor + causal_processor

        D5 (Impactos):
        - Q1/Q2 (Projection/Proxies): causal_processor + embedding_policy
        - Q3/Q4 (Validity/Risks): analyzer_one + policy_processor
        - Q5 (Unwanted Effects): contradiction_detector

        D6 (Causalidad):
        - Q1/Q2 (Theory/Logic): dereck_beach + causal_processor
        - Q3 (Inconsistencies): contradiction_detector
        - Q4 (Adaptive Monitoring): policy_processor
        - Q5 (Differential Approach): causal_processor + embedding_policy
        """

        mapping = {
            "D1": {
                1: ["policy_processor", "embedding_policy", "analyzer_one", "policy_segmenter"],
                2: ["policy_processor", "embedding_policy", "analyzer_one", "contradiction_detector"],
                3: ["financial_viability", "policy_processor", "analyzer_one"],
                4: ["policy_processor", "causal_processor", "analyzer_one"],
                5: ["policy_processor", "causal_processor", "embedding_policy"]
            },
            "D2": {
                1: ["analyzer_one", "policy_segmenter", "policy_processor"],
                2: ["causal_processor", "policy_processor", "embedding_policy"],
                3: ["causal_processor", "policy_processor", "dereck_beach"],
                4: ["embedding_policy", "analyzer_one", "policy_processor"],
                5: ["causal_processor", "policy_processor", "policy_segmenter"]
            },
            "D3": {
                1: ["policy_processor", "financial_viability", "analyzer_one"],
                2: ["embedding_policy", "policy_processor", "analyzer_one"],
                3: ["embedding_policy", "financial_viability", "policy_processor"],
                4: ["causal_processor", "analyzer_one", "policy_processor"],
                5: ["causal_processor", "policy_processor", "embedding_policy"]
            },
            "D4": {
                1: ["causal_processor", "policy_processor", "embedding_policy"],
                2: ["causal_processor", "policy_processor", "dereck_beach"],
                3: ["analyzer_one", "embedding_policy", "policy_processor"],
                4: ["analyzer_one", "policy_processor", "causal_processor"],
                5: ["policy_processor", "causal_processor", "embedding_policy"]
            },
            "D5": {
                1: ["causal_processor", "embedding_policy", "policy_processor"],
                2: ["causal_processor", "embedding_policy", "analyzer_one"],
                3: ["analyzer_one", "policy_processor", "embedding_policy"],
                4: ["analyzer_one", "policy_processor", "causal_processor"],
                5: ["contradiction_detector", "policy_processor", "embedding_policy"]
            },
            "D6": {
                1: ["dereck_beach", "causal_processor", "policy_segmenter"],
                2: ["dereck_beach", "causal_processor", "contradiction_detector"],
                3: ["contradiction_detector", "causal_processor", "policy_processor"],
                4: ["policy_processor", "causal_processor", "analyzer_one"],
                5: ["causal_processor", "embedding_policy", "policy_processor"]
            }
        }

        return mapping.get(dimension, {}).get(question_num, ["policy_processor"])

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
        for modules in self.routing_table.values():
            for mod in modules:
                module_usage[mod] = module_usage.get(mod, 0) + 1

        return {
            "total_questions": len(self.questions),
            "total_routes": len(self.routing_table),
            "module_usage_frequency": module_usage,
            "avg_modules_per_question": sum(len(v) for v in self.routing_table.values()) / len(self.routing_table),
            "dimensions": list(set(q.dimension for q in self.questions.values())),
            "policy_areas": list(set(q.policy_area for q in self.questions.values()))
        }
