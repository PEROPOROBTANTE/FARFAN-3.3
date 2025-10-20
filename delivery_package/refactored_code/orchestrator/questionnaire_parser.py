# coding=utf-8
# questionnaire_parser.py - Updated for full integration with module_adapters.py
"""
Questionnaire Parser - Loads questions from cuestionario.json and maps to execution chains
Integrates with:
- execution_mapping.yaml (execution chains)
- rubric_scoring.json (scoring modalities)
- module_adapters.py (actual module execution)
"""
import re
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ScoringModality:
    """Scoring modality from rubric_scoring.json"""
    id: str
    description: str
    formula: str
    max_score: float
    expected_elements: Optional[int] = None
    conversion_table: Optional[Dict[str, float]] = None
    uses_thresholds: bool = False
    uses_quantitative_data: bool = False
    uses_custom_logic: bool = False
    uses_semantic_matching: bool = False
    similarity_threshold: float = 0.6


@dataclass
class ExecutionChain:
    """Execution chain from execution_mapping.yaml"""
    description: str
    steps: List[Dict[str, Any]]
    aggregation: Dict[str, Any]


@dataclass
class QuestionSpec:
    """Complete specification for a single question"""
    # Basic identifiers
    question_id: str  # D1-Q1, D2-Q2, etc.
    dimension: str  # D1, D2, etc.
    question_no: int  # 1-5
    policy_area: str  # P1-P10
    
    # Question content
    template: str  # Question template
    text: str  # Fully interpolated question text
    
    # Scoring
    scoring_modality: str  # TYPE_A, TYPE_B, etc.
    max_score: float  # 0-3
    expected_elements: List[str]
    search_patterns: Dict[str, Any]
    
    # Execution mapping
    execution_chain: Optional[ExecutionChain] = None
    required_modules: List[str] = field(default_factory=list)
    primary_module: str = "policy_processor"
    supporting_modules: List[str] = field(default_factory=list)
    
    # Evidence and verification
    evidence_sources: Dict[str, Any] = field(default_factory=dict)
    verification_patterns: List[str] = field(default_factory=list)
    rubric_levels: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    weight: float = 1.0
    evidence_requirements: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def canonical_id(self) -> str:
        """Returns P#-D#-Q# notation"""
        return f"{self.policy_area}-{self.dimension}-Q{self.question_no}"


class QuestionnaireParser:
    """
    Comprehensive parser that integrates:
    1. cuestionario.json - Question templates and scoring
    2. execution_mapping.yaml - Execution chains
    3. rubric_scoring.json - Scoring modalities
    """

    def __init__(
            self,
            cuestionario_path: Path,
            execution_mapping_path: Optional[Path] = None,
            rubric_scoring_path: Optional[Path] = None
    ):
        self.cuestionario_path = cuestionario_path
        self.execution_mapping_path = execution_mapping_path or Path("config/execution_mapping.yaml")
        self.rubric_scoring_path = rubric_scoring_path or Path("config/rubric_scoring.json")
        
        # Storage
        self._questions = {}
        self._dimensions = {}
        self._policy_areas = {}
        self._scoring_modalities = {}
        self._execution_chains = {}
        
        # Default rubric mapping
        self._rubric_mapping = {
            "EXCELENTE": 0.85,
            "BUENO": 0.70,
            "ACEPTABLE": 0.55,
            "INSUFICIENTE": 0.0
        }
        
        # Load all configurations
        self._load_all()

    def _load_all(self):
        """Load all configuration files"""
        logger.info("Loading questionnaire configurations...")
        
        # Load rubric scoring modalities first
        self._load_rubric_scoring()
        
        # Load execution chains
        self._load_execution_mapping()
        
        # Load questions from cuestionario.json
        self._load_cuestionario()
        
        logger.info(f"Loaded {len(self._questions)} questions across {len(self._dimensions)} dimensions")

    def _load_rubric_scoring(self):
        """Load scoring modalities from rubric_scoring.json"""
        try:
            with open(self.rubric_scoring_path, 'r', encoding='utf-8') as f:
                rubric_data = json.load(f)
            
            # Load scoring modalities
            for modality_id, modality_data in rubric_data.get("scoring_modalities", {}).items():
                self._scoring_modalities[modality_id] = ScoringModality(
                    id=modality_id,
                    description=modality_data["description"],
                    formula=modality_data["formula"],
                    max_score=modality_data["max_score"],
                    expected_elements=modality_data.get("expected_elements"),
                    conversion_table=modality_data.get("conversion_table"),
                    uses_thresholds=modality_data.get("uses_thresholds", False),
                    uses_quantitative_data=modality_data.get("uses_quantitative_data", False),
                    uses_custom_logic=modality_data.get("uses_custom_logic", False),
                    uses_semantic_matching=modality_data.get("uses_semantic_matching", False),
                    similarity_threshold=modality_data.get("similarity_threshold", 0.6)
                )
            
            # Load dimensions
            for dim_id, dim_data in rubric_data.get("dimensions", {}).items():
                self._dimensions[dim_id] = {
                    "id": dim_id,
                    "name": dim_data["name"],
                    "description": dim_data["description"],
                    "questions": dim_data["questions"],
                    "max_score": dim_data["max_score"]
                }
            
            # Load policy areas
            for point_id, point_data in rubric_data.get("thematic_points", {}).items():
                self._policy_areas[point_id] = {
                    "id": point_id,
                    "title": point_data["title"],
                    "keywords": point_data["keywords"],
                    "applies_all_questions": point_data.get("applies_all_questions", True),
                    "can_be_na": point_data.get("can_be_na", False),
                    "na_condition": point_data.get("na_condition")
                }
            
            logger.info(f"Loaded {len(self._scoring_modalities)} scoring modalities")
            logger.info(f"Loaded {len(self._dimensions)} dimensions")
            logger.info(f"Loaded {len(self._policy_areas)} policy areas")
            
        except Exception as e:
            logger.error(f"Failed to load rubric_scoring.json: {e}")
            raise

    def _load_execution_mapping(self):
        """Load execution chains from execution_mapping.yaml"""
        try:
            with open(self.execution_mapping_path, 'r', encoding='utf-8') as f:
                execution_data = yaml.safe_load(f)
            
            # Load execution chains for each dimension
            for dimension_key in ["D1_INSUMOS", "D2_ACTIVIDADES", "D3_PRODUCTOS", 
                                 "D4_RESULTADOS", "D5_IMPACTOS", "D6_CAUSALIDAD"]:
                if dimension_key in execution_data:
                    dim_data = execution_data[dimension_key]
                    dimension = dimension_key.split("_")[0]  # Extract D1, D2, etc.
                    
                    # Load chains for each question (Q1-Q5)
                    for q_num in range(1, 6):
                        q_key = f"Q{q_num}_{self._get_question_key_suffix(dimension, q_num)}"
                        
                        if q_key in dim_data:
                            chain_data = dim_data[q_key]
                            chain_id = f"{dimension}-Q{q_num}"
                            
                            self._execution_chains[chain_id] = ExecutionChain(
                                description=chain_data.get("description", ""),
                                steps=chain_data.get("execution_chain", []),
                                aggregation=chain_data.get("aggregation", {})
                            )
            
            logger.info(f"Loaded {len(self._execution_chains)} execution chains")
            
        except Exception as e:
            logger.warning(f"Failed to load execution_mapping.yaml: {e}")
            # Continue without execution chains - will use fallback

    def _get_question_key_suffix(self, dimension: str, q_num: int) -> str:
        """Get the question key suffix based on dimension and question number"""
        # This maps to the YAML structure
        suffixes = {
            "D1": ["Baseline_Identification", "Gap_Analysis", "Budget_Allocation", 
                   "Capacity_Assessment", "Restriction_Identification"],
            "D2": ["Activity_Format", "Mechanism_Specification", "Causal_Links", 
                   "Risk_Assessment", "Sequencing_Logic"],
            "D3": ["DNP_Ficha_Completeness", "Indicator_Specification", "Budget_Alignment", 
                   "Feasibility_Assessment", "Mechanism_Clarity"],
            "D4": ["Measurability", "Causal_Chain_Completeness", "Timeframe_Specification", 
                   "Monitoring_Mechanism", "Strategic_Alignment"],
            "D5": ["Projection_Methodology", "Proxy_Indicators", "Validity_Assessment", 
                   "Risk_Analysis", "Unwanted_Effects"],
            "D6": ["Theory_of_Change", "Causal_Logic", "Inconsistency_Detection", 
                   "Adaptive_Monitoring", "Differential_Approach"]
        }
        return suffixes.get(dimension, ["Unknown"] * 5)[q_num - 1]

    def _load_cuestionario(self):
        """Load questions from cuestionario.json"""
        try:
            with open(self.cuestionario_path, 'r', encoding='utf-8') as f:
                cuestionario_data = json.load(f)
            
            # Load base questions from rubric_scoring.json
            questions_data = cuestionario_data.get("questions", [])
            
            # Generate all 300 questions (10 policy areas × 30 base questions)
            for policy_id in range(1, 11):
                policy_area = f"P{policy_id}"
                
                for question_data in questions_data:
                    # Extract dimension and question number
                    base_id = question_data["id"]  # e.g., "D1-Q1"
                    dimension = question_data["dimension"]
                    question_no = question_data["question_no"]
                    
                    # Create full question ID
                    question_id = f"{policy_area}-{base_id}"
                    
                    # Interpolate question template with policy area
                    template = question_data["template"]
                    policy_name = self._policy_areas.get(policy_area, {}).get("title", policy_area)
                    question_text = template.replace("{PUNTO_TEMATICO}", policy_name)
                    
                    # Get execution chain
                    chain_id = f"{dimension}-Q{question_no}"
                    execution_chain = self._execution_chains.get(chain_id)
                    
                    # Extract modules from execution chain
                    required_modules = []
                    if execution_chain:
                        required_modules = list(set(
                            step["module"] for step in execution_chain.steps
                        ))
                    
                    # Fallback module mapping if no execution chain
                    if not required_modules:
                        required_modules = self._get_fallback_modules(dimension, question_no)
                    
                    primary_module = required_modules[0] if required_modules else "policy_processor"
                    supporting_modules = required_modules[1:] if len(required_modules) > 1 else []
                    
                    # Create question spec
                    question = QuestionSpec(
                        question_id=base_id,
                        dimension=dimension,
                        question_no=question_no,
                        policy_area=policy_area,
                        template=template,
                        text=question_text,
                        scoring_modality=question_data.get("scoring_modality", "TYPE_A"),
                        max_score=question_data.get("max_score", 3.0),
                        expected_elements=question_data.get("expected_elements", []),
                        search_patterns=question_data.get("search_patterns", {}),
                        execution_chain=execution_chain,
                        required_modules=required_modules,
                        primary_module=primary_module,
                        supporting_modules=supporting_modules,
                        evidence_sources=question_data.get("evidence_sources", {}),
                        verification_patterns=self._extract_verification_patterns(question_text),
                        rubric_levels=self._rubric_mapping,
                        weight=1.0,
                        evidence_requirements={
                            "min_evidence_count": 2,
                            "required_evidence_types": ["quantitative", "qualitative"],
                            "confidence_threshold": 0.6
                        },
                        metadata={
                            "policy_area": policy_area,
                            "policy_name": policy_name,
                            "dimension_name": self._dimensions.get(dimension, {}).get("name", dimension),
                            "can_be_na": self._policy_areas.get(policy_area, {}).get("can_be_na", False),
                            "na_condition": self._policy_areas.get(policy_area, {}).get("na_condition")
                        }
                    )
                    
                    self._questions[question_id] = question
            
            logger.info(f"Generated {len(self._questions)} questions")
            
        except Exception as e:
            logger.error(f"Failed to load cuestionario.json: {e}")
            raise

    def _get_fallback_modules(self, dimension: str, question_num: int) -> List[str]:
        """Fallback module mapping if execution chain not found"""
        # This is the original module mapping from choreographer
        module_mapping = {
            "D1": {
                1: ["semantic_processor", "embedding_policy", "analyzer_one", "policy_segmenter"],
                2: ["bayesian_integrator", "semantic_processor", "municipal_analyzer", "embedding_analyzer"],
                3: ["financial_analyzer", "dereck_beach", "pdet_analyzer", "causal_processor"],
                4: ["analyzer_one", "municipal_analyzer", "causal_processor", "decologo_processor"],
                5: ["contradiction_detector", "dereck_beach", "causal_validator", "policy_processor"]
            },
            "D2": {
                1: ["policy_segmenter", "semantic_processor", "analyzer_one", "policy_processor"],
                2: ["dereck_beach", "causal_processor", "pdet_analyzer", "causal_validator"],
                3: ["causal_processor", "dereck_beach", "pdet_analyzer", "validation_framework"],
                4: ["contradiction_detector", "analyzer_one", "municipal_analyzer", "causal_processor"],
                5: ["contradiction_detector", "causal_processor", "causal_validator", "semantic_processor"]
            },
            "D3": {
                1: ["dereck_beach", "policy_processor", "semantic_processor", "pdet_analyzer"],
                2: ["embedding_policy", "semantic_processor", "bayesian_integrator", "embedding_analyzer"],
                3: ["financial_analyzer", "dereck_beach", "causal_processor", "pdet_analyzer"],
                4: ["analyzer_one", "municipal_analyzer", "pdet_analyzer", "causal_processor"],
                5: ["dereck_beach", "semantic_processor", "causal_processor", "decologo_processor"]
            },
            "D4": {
                1: ["embedding_policy", "semantic_processor", "bayesian_integrator", "embedding_analyzer"],
                2: ["causal_processor", "dereck_beach", "pdet_analyzer", "validation_framework"],
                3: ["contradiction_detector", "causal_processor", "semantic_processor", "causal_validator"],
                4: ["dereck_beach", "analyzer_one", "municipal_analyzer", "policy_processor"],
                5: ["semantic_processor", "policy_processor", "decologo_processor", "embedding_analyzer"]
            },
            "D5": {
                1: ["embedding_policy", "pdet_analyzer", "bayesian_integrator", "embedding_analyzer"],
                2: ["semantic_processor", "embedding_policy", "embedding_analyzer", "analyzer_one"],
                3: ["dereck_beach", "causal_processor", "causal_validator", "validation_framework"],
                4: ["contradiction_detector", "pdet_analyzer", "causal_processor", "municipal_analyzer"],
                5: ["contradiction_detector", "pdet_analyzer", "causal_processor", "causal_validator"]
            },
            "D6": {
                1: ["causal_processor", "dereck_beach", "validation_framework", "decologo_processor"],
                2: ["dereck_beach", "causal_processor", "causal_validator", "bayesian_integrator"],
                3: ["contradiction_detector", "causal_processor", "causal_validator", "validation_framework"],
                4: ["dereck_beach", "analyzer_one", "municipal_analyzer", "policy_processor"],
                5: ["embedding_policy", "semantic_processor", "analyzer_one", "embedding_analyzer"]
            }
        }
        
        return module_mapping.get(dimension, {}).get(question_num, ["policy_processor"])

    def _extract_verification_patterns(self, question_text: str) -> List[str]:
        """Extract verification patterns from question text"""
        patterns = []

        # Common patterns
        if "línea base" in question_text or "baseline" in question_text:
            patterns.append("baseline_present")
        if "fuente" in question_text or "source" in question_text:
            patterns.append("source_specified")
        if "cifras" in question_text or "números" in question_text or "cuantif" in question_text:
            patterns.append("quantitative_data")
        if "meta" in question_text or "target" in question_text:
            patterns.append("target_specified")
        if "responsable" in question_text or "responsible" in question_text:
            patterns.append("responsibility_assigned")
        if "presupuesto" in question_text or "recursos" in question_text or "budget" in question_text:
            patterns.append("budget_allocated")
        if "cronograma" in question_text or "plazo" in question_text or "timeline" in question_text:
            patterns.append("timeline_specified")
        if "causal" in question_text or "mecanismo" in question_text:
            patterns.append("causal_mechanism")
        if "indicador" in question_text or "indicator" in question_text:
            patterns.append("indicator_present")

        return patterns

    # Public API methods

    def get_question(self, question_id: str) -> Optional[QuestionSpec]:
        """Get a question by full ID (P#-D#-Q#)"""
        return self._questions.get(question_id)

    def get_all_questions(self) -> Dict[str, QuestionSpec]:
        """Get all 300 questions"""
        return self._questions.copy()

    def get_questions_by_dimension(self, dimension: str) -> List[QuestionSpec]:
        """Get all questions for a dimension across all policy areas"""
        return [q for q in self._questions.values() if q.dimension == dimension]

    def get_questions_by_policy_area(self, policy_area: str) -> List[QuestionSpec]:
        """Get all 30 questions for a policy area"""
        return [q for q in self._questions.values() if q.policy_area == policy_area]

    def get_questions_by_policy_and_dimension(self, policy_area: str, dimension: str) -> List[QuestionSpec]:
        """Get questions for a specific policy area and dimension (5 questions)"""
        return [
            q for q in self._questions.values()
            if q.policy_area == policy_area and q.dimension == dimension
        ]

    def get_policy_areas(self) -> Dict[str, Dict[str, Any]]:
        """Get all policy areas"""
        return self._policy_areas.copy()

    def get_dimensions(self) -> Dict[str, Dict[str, str]]:
        """Get all dimensions"""
        return self._dimensions.copy()

    def get_dimension_descriptions(self) -> Dict[str, str]:
        """Get dimension descriptions for report assembly"""
        return {
            dim_id: dim_data["description"]
            for dim_id, dim_data in self._dimensions.items()
        }

    def get_rubric_mapping(self) -> Dict[str, float]:
        """Get the rubric mapping"""
        return self._rubric_mapping.copy()

    def get_scoring_modality(self, modality_id: str) -> Optional[ScoringModality]:
        """Get a scoring modality by ID"""
        return self._scoring_modalities.get(modality_id)

    def get_execution_chain(self, dimension: str, question_no: int) -> Optional[ExecutionChain]:
        """Get execution chain for a dimension and question"""
        chain_id = f"{dimension}-Q{question_no}"
        return self._execution_chains.get(chain_id)

    def validate_questionnaire(self) -> Dict[str, Any]:
        """Validate the loaded questionnaire"""
        validation_result = {
            "valid": True,
            "total_questions": len(self._questions),
            "expected_questions": 300,
            "dimensions": len(self._dimensions),
            "policy_areas": len(self._policy_areas),
            "scoring_modalities": len(self._scoring_modalities),
            "execution_chains": len(self._execution_chains),
            "issues": []
        }

        # Check question count
        if len(self._questions) != 300:
            validation_result["valid"] = False
            validation_result["issues"].append(
                f"Expected 300 questions, found {len(self._questions)}"
            )

        # Check dimension coverage
        for dim in ["D1", "D2", "D3", "D4", "D5", "D6"]:
            dim_questions = self.get_questions_by_dimension(dim)
            expected = 50  # 10 policy areas × 5 questions
            if len(dim_questions) != expected:
                validation_result["valid"] = False
                validation_result["issues"].append(
                    f"Dimension {dim}: expected {expected} questions, found {len(dim_questions)}"
                )

        # Check policy area coverage
        for policy in [f"P{i}" for i in range(1, 11)]:
            policy_questions = self.get_questions_by_policy_area(policy)
            expected = 30  # 6 dimensions × 5 questions
            if len(policy_questions) != expected:
                validation_result["valid"] = False
                validation_result["issues"].append(
                    f"Policy area {policy}: expected {expected} questions, found {len(policy_questions)}"
                )

        # Check module mappings
        questions_without_modules = [
            q.canonical_id for q in self._questions.values()
            if not q.required_modules
        ]
        if questions_without_modules:
            validation_result["issues"].append(
                f"{len(questions_without_modules)} questions without module mappings"
            )

        return validation_result