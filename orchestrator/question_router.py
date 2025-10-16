# question_router.py - Enhanced with comprehensive module mapping
# Production-ready version with complete 300-question mapping

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
    
    Enforces the use of cuestionario.json to ensure homogeneous evaluation
    across all development plans.
    """

    def __init__(self, cuestionario_path: Optional[Path] = None, validate: bool = True):
        self.cuestionario_path = cuestionario_path or CONFIG.cuestionario_path
        self.questions: Dict[str, Question] = {}
        self.routing_table: Dict[str, List[str]] = {}
        self.validation_passed = False
        
        self._load_questionnaire()
        self._build_routing_table()
        
        # Validate that cuestionario.json is properly loaded and enforced
        if validate:
            self._validate_cuestionario_usage()

    def _load_questionnaire(self):
        """Load the 300-question configuration from cuestionario.json"""
        logger.info(f"Loading questionnaire from {self.cuestionario_path}")

        try:
            with open(self.cuestionario_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            logger.error(f"Questionnaire file not found: {self.cuestionario_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in questionnaire file: {e}")
            raise

        # Extract policy points (Decálogo), dimensions, and base questions
        policy_points_dict = data.get("puntos_decalogo", {})
        dimensions_data = data.get("dimensiones", {})
        base_questions_list = data.get("preguntas_base", [])
        
        # The 300 questions are organized as 10 blocks of 30 (one per policy point)
        # Each block has 6 dimensions × 5 questions = 30 questions
        # Order: P1(D1-D6, Q1-Q5), P2(D1-D6, Q1-Q5), ..., P10(D1-D6, Q1-Q5)
        policy_points = list(policy_points_dict.keys())  # ['P1', 'P2', ..., 'P10']
        questions_per_policy = 30  # 6 dimensions × 5 questions
        
        logger.info(f"Loading {len(base_questions_list)} questions for {len(policy_points)} policy points")

        # Process each of the 300 questions, matching it to its policy point
        for idx, q_data in enumerate(base_questions_list):
            # Determine which policy point this question belongs to
            policy_idx = idx // questions_per_policy
            if policy_idx >= len(policy_points):
                logger.warning(f"Question index {idx} exceeds expected range, skipping")
                continue
                
            policy_id = policy_points[policy_idx]
            
            # Extract question metadata
            dim_code = q_data.get("dimension", "")
            q_num = q_data.get("numero", 0)
            
            # Determine required modules based on dimension and question
            required_modules, primary_module, supporting_modules = self._determine_modules(dim_code, q_num)
            
            # Extract verification patterns from the question
            verification_patterns = q_data.get("patrones_verificacion", [])
            
            # Extract scoring criteria
            scoring = q_data.get("scoring", {})
            rubric_levels = {
                "EXCELENTE": scoring.get("excelente", {}).get("min_score", 0.85),
                "BUENO": scoring.get("bueno", {}).get("min_score", 0.70),
                "ACEPTABLE": scoring.get("aceptable", {}).get("min_score", 0.55),
                "INSUFICIENTE": scoring.get("insuficiente", {}).get("min_score", 0.0)
            }

            question = Question(
                policy_area=policy_id,
                dimension=dim_code,
                question_num=q_num,
                text=q_data.get("texto_template", f"Question {q_num} for {dim_code}"),
                rubric_levels=rubric_levels,
                verification_patterns=verification_patterns,
                required_modules=required_modules,
                primary_module=primary_module,
                supporting_modules=supporting_modules
            )

            self.questions[question.canonical_id] = question

        logger.info(f"Loaded {len(self.questions)} questions")

    def _extract_base_questions(self, dim_data: Dict, dim_code: str) -> List[Dict]:
        """Extract base questions from dimension data"""
        num_questions = dim_data.get("preguntas", 5)
        base_questions = []

        # Question templates for each dimension
        question_templates = {
            "D1": [
                "¿El plan identifica adecuadamente las líneas base para los indicadores propuestos?",
                "¿El análisis de brechas es metodológicamente riguroso y basado en evidencia?",
                "¿La asignación presupuestal es coherente con las prioridades del plan?",
                "¿El plan evalúa la capacidad institucional requerida para la implementación?",
                "¿El plan identifica las restricciones y limitaciones clave?"
            ],
            "D2": [
                "¿Las actividades están formuladas con el formato adecuado según estándares DNP?",
                "¿Los mecanismos de implementación están claramente especificados?",
                "¿Los enlaces causales entre actividades y productos son explícitos?",
                "¿El plan identifica y evalúa los riesgos asociados a las actividades?",
                "¿La secuencia de actividades es lógica y temporalmente coherente?"
            ],
            "D3": [
                "¿Los productos cuentan con ficha técnica DNP completa?",
                "¿Los indicadores de producto son específicos, medibles y alcanzables?",
                "¿La asignación presupuestal a productos es proporcional a su importancia?",
                "¿La viabilidad de los productos está adecuadamente evaluada?",
                "¿Los mecanismos para lograr los productos están claramente definidos?"
            ],
            "D4": [
                "¿Los resultados propuestos son medibles con indicadores claros?",
                "¿La cadena causal completa desde productos hasta resultados está explicitada?",
                "¿Los plazos para lograr los resultados son realistas?",
                "¿El plan define mecanismos de monitoreo para los resultados?",
                "¿Los resultados están alineados con los objetivos estratégicos del plan?"
            ],
            "D5": [
                "¿La metodología de proyección de impactos es técnicamente sólida?",
                "¿Se utilizan indicadores proxy adecuados cuando los impactos directos no son medibles?",
                "¿La validez de las hipótesis de impacto está adecuadamente fundamentada?",
                "¿El plan analiza los riesgos que podrían afectar los impactos esperados?",
                "¿El plan considera posibles efectos no deseados de las intervenciones?"
            ],
            "D6": [
                "¿El plan presenta una teoría del cambio explícita y coherente?",
                "¿La lógica causal entre insumos, actividades, productos, resultados e impactos es consistente?",
                "¿El plan detecta y gestiona posibles inconsistencias en la lógica causal?",
                "¿El plan define mecanismos de monitoreo adaptativo basados en evidencia?",
                "¿El plan adopta un enfoque diferencial para grupos poblacionales específicos?"
            ]
        }

        templates = question_templates.get(dim_code, ["Question text not available"] * num_questions)

        for i in range(min(num_questions, len(templates))):
            base_questions.append({
                "text": templates[i],
                "rubric": {
                    "EXCELENTE": 0.85,
                    "BUENO": 0.70,
                    "ACEPTABLE": 0.55,
                    "INSUFICIENTE": 0.0
                },
                "patterns": dim_data.get("causal_verification_template", {}).get("validation_patterns", [])
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
            "policy_areas": list(set(q.policy_area for q in self.questions.values())),
            "validation_status": "PASSED" if self.validation_passed else "NOT VALIDATED"
        }
    
    def _validate_cuestionario_usage(self):
        """
        Validate that cuestionario.json is properly loaded and enforced.
        
        This ensures homogeneous evaluation across all 170 development plans.
        """
        try:
            from .cuestionario_validator import CuestionarioValidator
            
            logger.info("Validating cuestionario.json usage...")
            
            validator = CuestionarioValidator(self.cuestionario_path)
            is_valid, results = validator.run_full_validation(self.questions)
            
            if not is_valid:
                error_msg = "Cuestionario validation FAILED - evaluation will not be homogeneous!"
                logger.error(error_msg)
                # Don't raise exception to allow system to continue, but log prominently
                logger.error("="*80)
                logger.error("CRITICAL: cuestionario.json validation failed!")
                logger.error("This may result in inconsistent evaluation across plans.")
                logger.error("="*80)
            else:
                logger.info("✓ Cuestionario validation PASSED - evaluation will be homogeneous")
                self.validation_passed = True
            
            return is_valid
            
        except ImportError as e:
            logger.warning(f"Could not import CuestionarioValidator: {e}")
            return False
        except Exception as e:
            logger.error(f"Error during cuestionario validation: {e}")
            return False