# coding=utf-8
# questionnaire_parser.py
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class QuestionSpec:
    """Immutable specification for a single question"""
    policy_area: str  # P1-P10
    dimension: str  # D1-D6
    question_num: int  # Q1-Q5
    text: str
    rubric_levels: Dict[str, float]  # "EXCELENTE": 0.85, etc.
    verification_patterns: List[str]
    required_modules: List[str]
    primary_module: str
    supporting_modules: List[str]
    evidence_requirements: Dict[str, Any]
    weight: float = 1.0

    @property
    def canonical_id(self) -> str:
        """Returns P#-D#-Q# notation"""
        return f"{self.policy_area}-{self.dimension}-Q{self.question_num}"


class QuestionnaireParser:
    """
    Parses the TXT questionnaire file and converts it to canonical form.
    """

    def __init__(self, txt_path: Path):
        self.txt_path = txt_path
        self._questionnaire_text = None
        self._questions = {}
        self._policy_areas = {}
        self._dimensions = {}
        self._rubric_mapping = {
            "EXCELENTE": 0.85,
            "BUENO": 0.70,
            "ACEPTABLE": 0.55,
            "INSUFICIENTE": 0.0
        }
        self._load_questionnaire()

    def _load_questionnaire(self):
        """Load and parse the questionnaire TXT file"""
        logger.info(f"Loading questionnaire from {self.txt_path}")

        try:
            with open(self.txt_path, 'r', encoding='utf-8') as f:
                self._questionnaire_text = f.read()
        except Exception as e:
            logger.error(f"Failed to load questionnaire: {e}")
            raise

        # Parse the content
        self._parse_content()

    def _parse_content(self):
        """Parse the questionnaire content"""
        # Extract policy areas
        self._extract_policy_areas()

        # Extract dimensions
        self._extract_dimensions()

        # Extract questions
        self._extract_questions()

        logger.info(f"Parsed {len(self._questions)} questions from questionnaire")

    def _extract_policy_areas(self):
        """Extract policy areas from the questionnaire"""
        # Pattern to match policy area headers like "### **P1: Derechos de las mujeres e igualdad de género**"
        policy_pattern = r'### \*\*P(\d+): ([^\*]+)\*\*'

        for match in re.finditer(policy_pattern, self._questionnaire_text):
            policy_id = f"P{match.group(1)}"
            policy_name = match.group(2).strip()

            self._policy_areas[policy_id] = {
                "id": policy_id,
                "name": policy_name
            }

    def _extract_dimensions(self):
        """Extract dimensions from the questionnaire"""
        # Pattern to match dimension headers like "#### **Dimensión 1: INSUMOS (D1)**"
        dimension_pattern = r'#### \*\*Dimensión (\d+): ([^\(]+) \((D\d+)\)\*\*'

        for match in re.finditer(dimension_pattern, self._questionnaire_text):
            dim_num = match.group(1)
            dim_name = match.group(2).strip()
            dim_code = match.group(3)

            self._dimensions[dim_code] = {
                "number": dim_num,
                "name": dim_name,
                "code": dim_code
            }

    def _extract_questions(self):
        """Extract questions from the questionnaire"""
        # Pattern to match questions like "* **P1-D1-Q1:** ¿El diagnóstico presenta datos numéricos...?"
        question_pattern = r'\* \*\*(P\d+-D\d+-Q\d+):\*\* ([^\n]+)'

        for match in re.finditer(question_pattern, self._questionnaire_text):
            question_id = match.group(1)
            question_text = match.group(2).strip()

            # Parse the question ID
            parts = question_id.split("-")
            policy_area = parts[0]
            dimension = parts[1]
            question_num = int(parts[2][1:])  # Remove 'Q' prefix

            # Extract verification patterns from the question text
            verification_patterns = self._extract_verification_patterns(question_text)

            # Get module mapping based on dimension and question number
            module_info = self._get_module_mapping(dimension, question_num)

            # Create question spec
            question = QuestionSpec(
                policy_area=policy_area,
                dimension=dimension,
                question_num=question_num,
                text=question_text,
                rubric_levels=self._rubric_mapping,
                verification_patterns=verification_patterns,
                required_modules=module_info["required"],
                primary_module=module_info["primary"],
                supporting_modules=module_info["supporting"],
                evidence_requirements=self._get_evidence_requirements(dimension, question_num),
                weight=self._get_question_weight(dimension, question_num)
            )

            self._questions[question_id] = question

    def _extract_verification_patterns(self, question_text: str) -> List[str]:
        """Extract verification patterns from question text"""
        patterns = []

        # Look for specific verification terms in the question
        if "línea base" in question_text:
            patterns.append("baseline_present")
        if "fuente" in question_text:
            patterns.append("source_specified")
        if "cifras" in question_text or "números" in question_text:
            patterns.append("quantitative_data")
        if "meta" in question_text:
            patterns.append("target_specified")
        if "responsable" in question_text:
            patterns.append("responsibility_assigned")
        if "presupuesto" in question_text or "recursos" in question_text:
            patterns.append("budget_allocated")
        if "cronograma" in question_text or "plazo" in question_text:
            patterns.append("timeline_specified")

        return patterns

    def _get_module_mapping(self, dimension: str, question_num: int) -> Dict[str, Any]:
        """Get module mapping based on dimension and question number"""
        # This is the same mapping as in the original question_router.py
        module_mapping = {
            "D1": {
                1: (["semantic_processor", "embedding_policy", "analyzer_one", "policy_segmenter"],
                    "semantic_processor", ["embedding_policy", "analyzer_one", "policy_segmenter"]),
                2: (["bayesian_integrator", "semantic_processor", "municipal_analyzer", "embedding_analyzer"],
                    "bayesian_integrator", ["semantic_processor", "municipal_analyzer", "embedding_analyzer"]),
                3: (["financial_analyzer", "dereck_beach", "pdet_analyzer", "causal_processor"], "financial_analyzer",
                    ["dereck_beach", "pdet_analyzer", "causal_processor"]),
                4: (["analyzer_one", "municipal_analyzer", "causal_processor", "decologo_processor"], "analyzer_one",
                    ["municipal_analyzer", "causal_processor", "decologo_processor"]),
                5: (["contradiction_detector", "dereck_beach", "causal_validator", "policy_processor"],
                    "contradiction_detector", ["dereck_beach", "causal_validator", "policy_processor"])
            },
            "D2": {
                1: (["policy_segmenter", "semantic_processor", "analyzer_one", "policy_processor"], "policy_segmenter",
                    ["semantic_processor", "analyzer_one", "policy_processor"]),
                2: (["dereck_beach", "causal_processor", "pdet_analyzer", "causal_validator"], "dereck_beach",
                    ["causal_processor", "pdet_analyzer", "causal_validator"]),
                3: (["causal_processor", "dereck_beach", "pdet_analyzer", "validation_framework"], "causal_processor",
                    ["dereck_beach", "pdet_analyzer", "validation_framework"]),
                4: (["contradiction_detector", "analyzer_one", "municipal_analyzer", "causal_processor"],
                    "contradiction_detector", ["analyzer_one", "municipal_analyzer", "causal_processor"]),
                5: (["contradiction_detector", "causal_processor", "causal_validator", "semantic_processor"],
                    "contradiction_detector", ["causal_processor", "causal_validator", "semantic_processor"])
            },
            "D3": {
                1: (["dereck_beach", "policy_processor", "semantic_processor", "pdet_analyzer"], "dereck_beach",
                    ["policy_processor", "semantic_processor", "pdet_analyzer"]),
                2: (["embedding_policy", "semantic_processor", "bayesian_integrator", "embedding_analyzer"],
                    "embedding_policy", ["semantic_processor", "bayesian_integrator", "embedding_analyzer"]),
                3: (["financial_analyzer", "dereck_beach", "causal_processor", "pdet_analyzer"], "financial_analyzer",
                    ["dereck_beach", "causal_processor", "pdet_analyzer"]),
                4: (["analyzer_one", "municipal_analyzer", "pdet_analyzer", "causal_processor"], "analyzer_one",
                    ["municipal_analyzer", "pdet_analyzer", "causal_processor"]),
                5: (["dereck_beach", "semantic_processor", "causal_processor", "decologo_processor"], "dereck_beach",
                    ["semantic_processor", "causal_processor", "decologo_processor"])
            },
            "D4": {
                1: (["embedding_policy", "semantic_processor", "bayesian_integrator", "embedding_analyzer"],
                    "embedding_policy", ["semantic_processor", "bayesian_integrator", "embedding_analyzer"]),
                2: (["causal_processor", "dereck_beach", "pdet_analyzer", "validation_framework"], "causal_processor",
                    ["dereck_beach", "pdet_analyzer", "validation_framework"]),
                3: (["contradiction_detector", "causal_processor", "semantic_processor", "causal_validator"],
                    "contradiction_detector", ["causal_processor", "semantic_processor", "causal_validator"]),
                4: (["dereck_beach", "analyzer_one", "municipal_analyzer", "policy_processor"], "dereck_beach",
                    ["analyzer_one", "municipal_analyzer", "policy_processor"]),
                5: (["semantic_processor", "policy_processor", "decologo_processor", "embedding_analyzer"],
                    "semantic_processor", ["policy_processor", "decologo_processor", "embedding_analyzer"])
            },
            "D5": {
                1: (["embedding_policy", "pdet_analyzer", "bayesian_integrator", "embedding_analyzer"],
                    "embedding_policy", ["pdet_analyzer", "bayesian_integrator", "embedding_analyzer"]),
                2: (["semantic_processor", "embedding_policy", "embedding_analyzer", "analyzer_one"],
                    "semantic_processor", ["embedding_policy", "embedding_analyzer", "analyzer_one"]),
                3: (["dereck_beach", "causal_processor", "causal_validator", "validation_framework"], "dereck_beach",
                    ["causal_processor", "causal_validator", "validation_framework"]),
                4: (["contradiction_detector", "pdet_analyzer", "causal_processor", "municipal_analyzer"],
                    "contradiction_detector", ["pdet_analyzer", "causal_processor", "municipal_analyzer"]),
                5: (["contradiction_detector", "pdet_analyzer", "causal_processor", "causal_validator"],
                    "contradiction_detector", ["pdet_analyzer", "causal_processor", "causal_validator"])
            },
            "D6": {
                1: (["causal_processor", "dereck_beach", "validation_framework", "decologo_processor"],
                    "causal_processor", ["dereck_beach", "validation_framework", "decologo_processor"]),
                2: (["dereck_beach", "causal_processor", "causal_validator", "bayesian_integrator"], "dereck_beach",
                    ["causal_processor", "causal_validator", "bayesian_integrator"]),
                3: (["contradiction_detector", "causal_processor", "causal_validator", "validation_framework"],
                    "contradiction_detector", ["causal_processor", "causal_validator", "validation_framework"]),
                4: (["dereck_beach", "analyzer_one", "municipal_analyzer", "policy_processor"], "dereck_beach",
                    ["analyzer_one", "municipal_analyzer", "policy_processor"]),
                5: (["embedding_policy", "semantic_processor", "analyzer_one", "embedding_analyzer"],
                    "embedding_policy", ["semantic_processor", "analyzer_one", "embedding_analyzer"])
            }
        }

        return {
            "required":
                module_mapping.get(dimension, {}).get(question_num, (["policy_processor"], "policy_processor", []))[0],
            "primary":
                module_mapping.get(dimension, {}).get(question_num, (["policy_processor"], "policy_processor", []))[1],
            "supporting":
                module_mapping.get(dimension, {}).get(question_num, (["policy_processor"], "policy_processor", []))[2]
        }

    def _get_evidence_requirements(self, dimension: str, question_num: int) -> Dict[str, Any]:
        """Get evidence requirements based on dimension and question number"""
        # Define evidence requirements for each dimension and question
        return {
            "min_evidence_count": 2,
            "required_evidence_types": ["quantitative", "qualitative"],
            "confidence_threshold": 0.6
        }

    def _get_question_weight(self, dimension: str, question_num: int) -> float:
        """Get question weight based on dimension and question number"""
        # All questions have equal weight for now
        return 1.0

    def get_question(self, question_id: str) -> Optional[QuestionSpec]:
        """Get a question by ID"""
        return self._questions.get(question_id)

    def get_all_questions(self) -> Dict[str, QuestionSpec]:
        """Get all questions"""
        return self._questions.copy()

    def get_questions_by_dimension(self, dimension: str) -> List[QuestionSpec]:
        """Get all questions for a dimension"""
        return [q for q in self._questions.values() if q.dimension == dimension]

    def get_questions_by_policy_area(self, policy_area: str) -> List[QuestionSpec]:
        """Get all questions for a policy area"""
        return [q for q in self._questions.values() if q.policy_area == policy_area]

    def get_policy_areas(self) -> Dict[str, Dict[str, str]]:
        """Get all policy areas"""
        return self._policy_areas.copy()

    def get_dimensions(self) -> Dict[str, Dict[str, str]]:
        """Get all dimensions"""
        return self._dimensions.copy()

    def get_rubric_mapping(self) -> Dict[str, float]:
        """Get the rubric mapping"""
        return self._rubric_mapping.copy()