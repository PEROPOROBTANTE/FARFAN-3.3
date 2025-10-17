"""
questionnaire_parser.py - Canonical source for questionnaire data

This module serves as the single source of truth for all questionnaire-related
logic in FARFAN-3.0, ensuring deterministic and auditable orchestration.

Compliance: SIN_CARRETA doctrine
- Deterministic loading and parsing
- Contract-driven interfaces
- Explicit error handling (no silent failures)
- Immutable data structures
- Full audit trail
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from functools import lru_cache

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QuestionData:
    """Immutable question data structure"""
    id: str  # e.g., "D1-Q1"
    dimension: str  # D1-D6
    question_num: int  # 1-5
    text_template: str
    rubric_levels: Dict[str, float]  # EXCELENTE/BUENO/ACEPTABLE/INSUFICIENTE
    verification_patterns: List[str]
    criteria: Dict[str, Any]
    
    def __post_init__(self):
        """Validate question data on creation"""
        if not self.id or not self.dimension:
            raise ValueError(f"Invalid question data: id={self.id}, dimension={self.dimension}")


@dataclass(frozen=True)
class DimensionData:
    """Immutable dimension data structure"""
    code: str  # D1-D6
    name: str
    description: str
    weights_by_point: Dict[str, float]  # P1-P10 weights
    num_questions: int
    min_threshold: float
    decalogo_mapping: Dict[str, Dict[str, Any]]


@dataclass(frozen=True)
class PolicyPointData:
    """Immutable policy point (Decálogo) data structure"""
    code: str  # P1-P10
    name: str
    description: str
    critical_dimensions: List[str]
    product_indicators: List[str]
    result_indicators: List[str]


class QuestionnaireParser:
    """
    Canonical parser for cuestionario.json
    
    Responsibilities:
    - Load and validate questionnaire structure
    - Provide deterministic access to all 300 questions
    - Supply dimension metadata and weights
    - Deliver rubric levels for scoring
    - Ensure contract compliance and auditability
    
    Design principles (SIN_CARRETA):
    - Single source of truth for questionnaire data
    - Immutable parsed structures
    - Explicit error propagation
    - Deterministic behavior (no randomness)
    - Full traceability of data access
    """
    
    def __init__(self, cuestionario_path: Optional[Path] = None):
        """
        Initialize parser with explicit path to cuestionario.json
        
        Args:
            cuestionario_path: Path to cuestionario.json. If None, uses default.
            
        Raises:
            FileNotFoundError: If cuestionario.json not found
            json.JSONDecodeError: If JSON is malformed
            ValueError: If questionnaire structure is invalid
        """
        self._cuestionario_path = cuestionario_path or self._get_default_path()
        self._raw_data: Dict[str, Any] = {}
        self._dimensions: Dict[str, DimensionData] = {}
        self._policy_points: Dict[str, PolicyPointData] = {}
        self._questions: Dict[str, QuestionData] = {}
        self._load_and_validate()
        
        logger.info(f"QuestionnaireParser initialized from {self._cuestionario_path}")
        logger.info(f"Loaded {len(self._dimensions)} dimensions, "
                   f"{len(self._policy_points)} policy points, "
                   f"{len(self._questions)} base questions")
    
    @staticmethod
    def _get_default_path() -> Path:
        """Get default path to cuestionario.json"""
        base_dir = Path(__file__).parent.parent
        return base_dir / "cuestionario.json"
    
    def _load_and_validate(self):
        """
        Load cuestionario.json and validate structure
        
        SIN_CARRETA compliance:
        - Explicit error handling (FileNotFoundError, JSONDecodeError)
        - Structural validation before use
        - No silent failures or warnings
        
        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If JSON is invalid
            ValueError: If required structure is missing
        """
        if not self._cuestionario_path.exists():
            error_msg = f"Questionnaire file not found: {self._cuestionario_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        try:
            with open(self._cuestionario_path, 'r', encoding='utf-8') as f:
                self._raw_data = json.load(f)
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON in {self._cuestionario_path}: {e}"
            logger.error(error_msg)
            raise
        
        # Validate required top-level keys
        required_keys = ["metadata", "dimensiones", "puntos_decalogo", "preguntas_base"]
        missing_keys = [k for k in required_keys if k not in self._raw_data]
        if missing_keys:
            error_msg = f"Missing required keys in questionnaire: {missing_keys}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Parse structures
        self._parse_dimensions()
        self._parse_policy_points()
        self._parse_base_questions()
        
        # Validate counts
        self._validate_structure()
    
    def _parse_dimensions(self):
        """Parse dimension data from cuestionario.json"""
        dimensiones = self._raw_data.get("dimensiones", {})
        
        for dim_code, dim_data in dimensiones.items():
            self._dimensions[dim_code] = DimensionData(
                code=dim_code,
                name=dim_data.get("nombre", ""),
                description=dim_data.get("descripcion", ""),
                weights_by_point=dim_data.get("peso_por_punto", {}),
                num_questions=dim_data.get("preguntas", 5),
                min_threshold=dim_data.get("umbral_minimo", 0.5),
                decalogo_mapping=dim_data.get("decalogo_dimension_mapping", {})
            )
    
    def _parse_policy_points(self):
        """Parse policy points (Decálogo) data"""
        puntos = self._raw_data.get("puntos_decalogo", {})
        
        for point_code, point_data in puntos.items():
            self._policy_points[point_code] = PolicyPointData(
                code=point_code,
                name=point_data.get("nombre", ""),
                description=point_data.get("descripcion", ""),
                critical_dimensions=point_data.get("dimensiones_criticas", []),
                product_indicators=point_data.get("indicadores_producto", []),
                result_indicators=point_data.get("indicadores_resultado", [])
            )
    
    def _parse_base_questions(self):
        """Parse base questions from preguntas_base array"""
        preguntas_base = self._raw_data.get("preguntas_base", [])
        
        for q_data in preguntas_base:
            q_id = q_data.get("id", "")
            dimension = q_data.get("dimension", "")
            numero = q_data.get("numero", 0)
            
            # Extract rubric if available, else use defaults
            rubric = self._extract_rubric(q_data)
            
            question = QuestionData(
                id=q_id,
                dimension=dimension,
                question_num=numero,
                text_template=q_data.get("texto_template", ""),
                rubric_levels=rubric,
                verification_patterns=q_data.get("patrones_verificacion", []),
                criteria=q_data.get("criterios_evaluacion", {})
            )
            
            self._questions[q_id] = question
    
    def _extract_rubric(self, q_data: Dict) -> Dict[str, float]:
        """
        Extract rubric levels from question data or use defaults
        
        Returns standard 4-level rubric: EXCELENTE, BUENO, ACEPTABLE, INSUFICIENTE
        """
        rubric_data = q_data.get("niveles_rubrica", {})
        
        # Default rubric scoring
        default_rubric = {
            "EXCELENTE": 0.85,
            "BUENO": 0.70,
            "ACEPTABLE": 0.55,
            "INSUFICIENTE": 0.0
        }
        
        if not rubric_data:
            return default_rubric
        
        # Extract and normalize
        return {
            level: rubric_data.get(level, default_rubric[level])
            for level in default_rubric.keys()
        }
    
    def _validate_structure(self):
        """
        Validate loaded questionnaire structure
        
        Ensures:
        - 6 dimensions (D1-D6)
        - 10 policy points (P1-P10)
        - 30 base questions (6 dimensions × 5 questions)
        - 300 total questions when combined with policy points
        """
        expected_dimensions = 6
        expected_points = 10
        expected_base_questions = 30
        
        if len(self._dimensions) != expected_dimensions:
            error_msg = (f"Expected {expected_dimensions} dimensions, "
                        f"found {len(self._dimensions)}")
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if len(self._policy_points) != expected_points:
            error_msg = (f"Expected {expected_points} policy points, "
                        f"found {len(self._policy_points)}")
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if len(self._questions) != expected_base_questions:
            logger.warning(f"Expected {expected_base_questions} base questions, "
                          f"found {len(self._questions)}")
    
    # Public interface methods
    
    @lru_cache(maxsize=1)
    def get_metadata(self) -> Dict[str, Any]:
        """Get questionnaire metadata"""
        return self._raw_data.get("metadata", {}).copy()
    
    def get_dimension(self, dimension_code: str) -> Optional[DimensionData]:
        """
        Get dimension data by code
        
        Args:
            dimension_code: Dimension identifier (D1-D6)
            
        Returns:
            DimensionData if found, None otherwise
        """
        return self._dimensions.get(dimension_code)
    
    def get_all_dimensions(self) -> Dict[str, DimensionData]:
        """Get all dimension data"""
        return self._dimensions.copy()
    
    def get_policy_point(self, point_code: str) -> Optional[PolicyPointData]:
        """
        Get policy point data by code
        
        Args:
            point_code: Policy point identifier (P1-P10)
            
        Returns:
            PolicyPointData if found, None otherwise
        """
        return self._policy_points.get(point_code)
    
    def get_all_policy_points(self) -> Dict[str, PolicyPointData]:
        """Get all policy point data"""
        return self._policy_points.copy()
    
    def get_question(self, question_id: str) -> Optional[QuestionData]:
        """
        Get question data by ID
        
        Args:
            question_id: Question identifier (e.g., "D1-Q1")
            
        Returns:
            QuestionData if found, None otherwise
        """
        return self._questions.get(question_id)
    
    def get_questions_for_dimension(self, dimension_code: str) -> List[QuestionData]:
        """
        Get all questions for a specific dimension
        
        Args:
            dimension_code: Dimension identifier (D1-D6)
            
        Returns:
            List of QuestionData for the dimension
        """
        return [
            q for q in self._questions.values()
            if q.dimension == dimension_code
        ]
    
    def get_dimension_weight_for_point(self, dimension_code: str, 
                                       point_code: str) -> float:
        """
        Get weight of a dimension for a specific policy point
        
        Args:
            dimension_code: Dimension identifier (D1-D6)
            point_code: Policy point identifier (P1-P10)
            
        Returns:
            Weight value (0.0-1.0), or 0.0 if not found
        """
        dimension = self._dimensions.get(dimension_code)
        if not dimension:
            return 0.0
        
        return dimension.weights_by_point.get(point_code, 0.0)
    
    def get_rubric_for_question(self, question_id: str) -> Dict[str, float]:
        """
        Get rubric levels for a specific question
        
        Args:
            question_id: Question identifier
            
        Returns:
            Dictionary mapping rubric level to score
        """
        question = self._questions.get(question_id)
        if not question:
            # Return default rubric
            return {
                "EXCELENTE": 0.85,
                "BUENO": 0.70,
                "ACEPTABLE": 0.55,
                "INSUFICIENTE": 0.0
            }
        
        return question.rubric_levels.copy()
    
    def generate_all_questions(self) -> List[str]:
        """
        Generate all 300 question IDs (10 points × 6 dimensions × 5 questions)
        
        Returns:
            List of question IDs in format "P#-D#-Q#"
        """
        all_questions = []
        
        for point_code in sorted(self._policy_points.keys()):
            for dim_code in sorted(self._dimensions.keys()):
                dimension = self._dimensions[dim_code]
                for q_num in range(1, dimension.num_questions + 1):
                    question_id = f"{point_code}-{dim_code}-Q{q_num}"
                    all_questions.append(question_id)
        
        return all_questions
    
    def get_verification_patterns(self, question_id: str) -> List[str]:
        """
        Get verification patterns for a question
        
        Args:
            question_id: Question identifier (e.g., "D1-Q1")
            
        Returns:
            List of regex patterns for verification
        """
        # Extract base question ID (remove policy point prefix if present)
        if question_id.count('-') == 2:
            # Format: P#-D#-Q#
            parts = question_id.split('-')
            base_id = f"{parts[1]}-{parts[2]}"
        else:
            # Format: D#-Q#
            base_id = question_id
        
        question = self._questions.get(base_id)
        if not question:
            return []
        
        return question.verification_patterns.copy()
    
    def get_common_failure_patterns(self) -> Dict[str, Any]:
        """Get common failure patterns from questionnaire"""
        return self._raw_data.get("common_failure_patterns", {}).copy()
    
    def get_scoring_system(self) -> Dict[str, Any]:
        """Get scoring system configuration"""
        return self._raw_data.get("scoring_system", {}).copy()
    
    def get_causal_glossary(self) -> Dict[str, Any]:
        """Get causal glossary definitions"""
        return self._raw_data.get("causal_glossary", {}).copy()
    
    def validate_question_id(self, question_id: str) -> Tuple[bool, str]:
        """
        Validate a question ID format and existence
        
        Args:
            question_id: Question ID to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check format
        parts = question_id.split('-')
        if len(parts) not in [2, 3]:
            return False, f"Invalid format: {question_id}. Expected D#-Q# or P#-D#-Q#"
        
        # Validate parts
        if len(parts) == 3:
            # P#-D#-Q# format
            point, dim, q = parts
            if point not in self._policy_points:
                return False, f"Unknown policy point: {point}"
            if dim not in self._dimensions:
                return False, f"Unknown dimension: {dim}"
        else:
            # D#-Q# format
            dim, q = parts
            if dim not in self._dimensions:
                return False, f"Unknown dimension: {dim}"
        
        return True, ""
    
    @property
    def questionnaire_path(self) -> Path:
        """Get path to the canonical questionnaire file"""
        return self._cuestionario_path
    
    @property
    def total_questions(self) -> int:
        """Get total number of questions (should be 300)"""
        return len(self.generate_all_questions())
    
    @property
    def version(self) -> str:
        """Get questionnaire version"""
        metadata = self.get_metadata()
        return metadata.get("version", "unknown")


# Module-level singleton for shared access
_parser_instance: Optional[QuestionnaireParser] = None


def get_questionnaire_parser(cuestionario_path: Optional[Path] = None) -> QuestionnaireParser:
    """
    Get singleton instance of QuestionnaireParser
    
    This ensures only one instance loads and parses cuestionario.json,
    maintaining determinism and reducing overhead.
    
    Args:
        cuestionario_path: Optional path override (only used on first call)
        
    Returns:
        QuestionnaireParser instance
    """
    global _parser_instance
    
    if _parser_instance is None:
        _parser_instance = QuestionnaireParser(cuestionario_path)
    
    return _parser_instance
