# coding=utf-8
"""
Questionnaire Parser - Loads and Validates Questionnaire
=========================================================

Loads and validates cuestionario.json containing 300 questions.

Author: FARFAN 3.0 Team
Version: 3.0.0
Python: 3.10+
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class QuestionSpec:
    """Specification for a single question"""
    question_id: str
    dimension: str
    question_no: int
    policy_area: str
    template: str
    text: str
    scoring_modality: str
    max_score: float
    expected_elements: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def canonical_id(self) -> str:
        """Returns P#-D#-Q# notation"""
        return f"{self.policy_area}-{self.dimension}-Q{self.question_no}"


class QuestionnaireParser:
    """
    Loads and validates cuestionario.json containing 300 questions
    
    Features:
    - Loads question definitions from cuestionario.json
    - Validates question structure and completeness
    - Provides access to questions by various criteria
    - Generates 300 questions from templates and policy areas
    """

    def __init__(self, cuestionario_path: Optional[Path] = None):
        """
        Initialize questionnaire parser
        
        Args:
            cuestionario_path: Path to cuestionario.json (defaults to repo root)
        """
        self.cuestionario_path = cuestionario_path or Path("cuestionario.json")
        self._questions: Dict[str, QuestionSpec] = {}
        self._dimensions: Dict[str, Any] = {}
        self._policy_areas: Dict[str, Any] = {}
        
        self._load_questionnaire()
        
        logger.info(
            f"QuestionnaireParser initialized: {len(self._questions)} questions, "
            f"{len(self._dimensions)} dimensions, {len(self._policy_areas)} policy areas"
        )

    def _load_questionnaire(self):
        """Load and validate cuestionario.json"""
        if not self.cuestionario_path.exists():
            raise FileNotFoundError(
                f"cuestionario.json not found at {self.cuestionario_path}"
            )
        
        try:
            with open(self.cuestionario_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self._validate_structure(data)
            
            self._load_dimensions(data)
            self._load_policy_areas(data)
            self._load_questions(data)
            
            logger.info(f"Loaded questionnaire from {self.cuestionario_path}")
            
        except Exception as e:
            logger.error(f"Failed to load cuestionario.json: {e}")
            raise

    def _validate_structure(self, data: Dict[str, Any]):
        """Validate JSON structure"""
        required_keys = ["metadata", "dimensiones", "puntos_tematicos"]
        missing = [k for k in required_keys if k not in data]
        
        if missing:
            logger.warning(f"Missing keys in cuestionario.json: {missing}")

    def _load_dimensions(self, data: Dict[str, Any]):
        """Load dimension definitions"""
        dimensions_data = data.get("dimensiones", {})
        
        for dim_id, dim_info in dimensions_data.items():
            self._dimensions[dim_id] = {
                "id": dim_id,
                "nombre": dim_info.get("nombre", dim_id),
                "descripcion": dim_info.get("descripcion", ""),
                "preguntas": dim_info.get("preguntas", 5)
            }

    def _load_policy_areas(self, data: Dict[str, Any]):
        """Load policy area definitions"""
        policy_data = data.get("puntos_tematicos", {})
        
        for policy_id, policy_info in policy_data.items():
            self._policy_areas[policy_id] = {
                "id": policy_id,
                "titulo": policy_info.get("titulo", policy_id),
                "palabras_clave": policy_info.get("palabras_clave", [])
            }

    def _load_questions(self, data: Dict[str, Any]):
        """Load and generate 300 questions"""
        questions_data = data.get("preguntas", [])
        
        if not questions_data:
            questions_data = self._extract_questions_from_dimensions(data)
        
        for policy_num in range(1, 11):
            policy_id = f"P{policy_num}"
            policy_title = self._policy_areas.get(
                policy_id, {}
            ).get("titulo", policy_id)
            
            for question_data in questions_data:
                base_id = question_data.get("id", "")
                dimension = question_data.get("dimension", "")
                question_no = question_data.get("question_no", 1)
                
                template = question_data.get("template", "")
                question_text = template.replace("{PUNTO_TEMATICO}", policy_title)
                
                question_id = f"{policy_id}-{base_id}"
                
                question = QuestionSpec(
                    question_id=base_id,
                    dimension=dimension,
                    question_no=question_no,
                    policy_area=policy_id,
                    template=template,
                    text=question_text,
                    scoring_modality=question_data.get("scoring_modality", "TYPE_A"),
                    max_score=question_data.get("max_score", 3.0),
                    expected_elements=question_data.get("expected_elements", []),
                    metadata={
                        "policy_title": policy_title,
                        "dimension_name": self._dimensions.get(dimension, {}).get(
                            "nombre", dimension
                        )
                    }
                )
                
                self._questions[question_id] = question
        
        logger.info(f"Generated {len(self._questions)} questions")

    def _extract_questions_from_dimensions(self, data: Dict[str, Any]) -> List[Dict]:
        """Extract question templates from dimension definitions"""
        questions = []
        dimensions = data.get("dimensiones", {})
        
        for dim_id, dim_data in dimensions.items():
            dim_questions = dim_data.get("preguntas_base", [])
            
            if not dim_questions and isinstance(dim_data.get("preguntas"), int):
                num_questions = dim_data["preguntas"]
                for q_num in range(1, num_questions + 1):
                    questions.append({
                        "id": f"{dim_id}-Q{q_num}",
                        "dimension": dim_id,
                        "question_no": q_num,
                        "template": f"Pregunta {q_num} sobre {{PUNTO_TEMATICO}}",
                        "scoring_modality": "TYPE_A",
                        "max_score": 3.0
                    })
            else:
                questions.extend(dim_questions)
        
        return questions

    def get_question(self, question_id: str) -> Optional[QuestionSpec]:
        """Get a question by ID"""
        return self._questions.get(question_id)

    def get_all_questions(self) -> Dict[str, QuestionSpec]:
        """Get all questions"""
        return self._questions.copy()

    def get_questions_by_dimension(self, dimension: str) -> List[QuestionSpec]:
        """Get all questions for a dimension"""
        return [
            q for q in self._questions.values()
            if q.dimension == dimension
        ]

    def get_questions_by_policy_area(self, policy_area: str) -> List[QuestionSpec]:
        """Get all questions for a policy area"""
        return [
            q for q in self._questions.values()
            if q.policy_area == policy_area
        ]

    def validate_questionnaire(self) -> Dict[str, Any]:
        """
        Validate questionnaire completeness
        
        Returns:
            Validation report with any issues found
        """
        issues = []
        
        expected_count = 10 * 30
        if len(self._questions) != expected_count:
            issues.append(
                f"Expected {expected_count} questions, found {len(self._questions)}"
            )
        
        for dim_id in ["D1", "D2", "D3", "D4", "D5", "D6"]:
            dim_questions = self.get_questions_by_dimension(dim_id)
            expected_per_dim = 50
            if len(dim_questions) != expected_per_dim:
                issues.append(
                    f"Dimension {dim_id}: expected {expected_per_dim} questions, "
                    f"found {len(dim_questions)}"
                )
        
        return {
            "valid": len(issues) == 0,
            "total_questions": len(self._questions),
            "issues": issues
        }


if __name__ == "__main__":
    parser = QuestionnaireParser()
    
    print("=" * 60)
    print("Questionnaire Parser")
    print("=" * 60)
    print(f"\nTotal questions: {len(parser.get_all_questions())}")
    print(f"Dimensions: {len(parser._dimensions)}")
    print(f"Policy areas: {len(parser._policy_areas)}")
    
    validation = parser.validate_questionnaire()
    print(f"\nValidation: {'✓ PASS' if validation['valid'] else '✗ FAIL'}")
    if validation['issues']:
        print("Issues:")
        for issue in validation['issues']:
            print(f"  - {issue}")
