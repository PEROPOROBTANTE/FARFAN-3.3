"""
Cuestionario Validator - Ensures homogeneous evaluation across all development plans
Enforces the use of cuestionario.json standards to maintain consistency
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation check"""
    is_valid: bool
    message: str
    severity: str  # "ERROR", "WARNING", "INFO"
    details: Dict[str, Any] = None


class CuestionarioValidator:
    """
    Validates that the evaluation process strictly follows cuestionario.json standards.
    
    This ensures homogeneous evaluation across all 170 development plans by:
    1. Verifying all 300 questions are used
    2. Ensuring verification patterns are applied
    3. Validating scoring rubrics are followed
    4. Checking that policy point mapping is correct
    """
    
    def __init__(self, cuestionario_path: Path):
        """Initialize validator with cuestionario.json"""
        self.cuestionario_path = cuestionario_path
        self.cuestionario_data = None
        self.validation_results: List[ValidationResult] = []
        
        self._load_cuestionario()
    
    def _load_cuestionario(self):
        """Load and validate cuestionario.json structure"""
        try:
            with open(self.cuestionario_path, 'r', encoding='utf-8') as f:
                self.cuestionario_data = json.load(f)
            
            required_keys = ['metadata', 'dimensiones', 'puntos_decalogo', 
                           'preguntas_base', 'scoring_system']
            
            missing = [k for k in required_keys if k not in self.cuestionario_data]
            if missing:
                raise ValueError(f"cuestionario.json missing required keys: {missing}")
            
            logger.info(f"✓ Loaded cuestionario.json v{self.cuestionario_data['metadata']['version']}")
            
        except Exception as e:
            logger.error(f"✗ Failed to load cuestionario.json: {e}")
            raise
    
    def validate_question_coverage(
            self, 
            loaded_questions: Dict[str, Any]
    ) -> ValidationResult:
        """
        Validate that all 300 questions from cuestionario.json are loaded.
        
        Args:
            loaded_questions: Dictionary of loaded questions keyed by canonical_id
            
        Returns:
            ValidationResult indicating if all questions are present
        """
        expected_count = self.cuestionario_data['metadata']['total_questions']
        actual_count = len(loaded_questions)
        
        if actual_count == expected_count:
            return ValidationResult(
                is_valid=True,
                message=f"✓ All {expected_count} questions loaded correctly",
                severity="INFO",
                details={"expected": expected_count, "actual": actual_count}
            )
        else:
            return ValidationResult(
                is_valid=False,
                message=f"✗ Question count mismatch: expected {expected_count}, got {actual_count}",
                severity="ERROR",
                details={"expected": expected_count, "actual": actual_count}
            )
    
    def validate_policy_point_mapping(
            self,
            loaded_questions: Dict[str, Any]
    ) -> ValidationResult:
        """
        Validate that questions are correctly mapped to policy points.
        
        Each policy point (P1-P10) should have exactly 30 questions (6 dimensions × 5 questions).
        """
        policy_points = list(self.cuestionario_data['puntos_decalogo'].keys())
        
        # Count questions per policy point
        questions_by_policy = {}
        for qid, question in loaded_questions.items():
            policy_area = question.policy_area
            if policy_area not in questions_by_policy:
                questions_by_policy[policy_area] = []
            questions_by_policy[policy_area].append(qid)
        
        # Validate each policy point
        errors = []
        warnings = []
        
        for policy_id in policy_points:
            expected_count = 30  # 6 dimensions × 5 questions
            actual_count = len(questions_by_policy.get(policy_id, []))
            
            if actual_count != expected_count:
                errors.append(
                    f"{policy_id}: expected {expected_count} questions, got {actual_count}"
                )
        
        # Check for unexpected policy points
        for policy_id in questions_by_policy.keys():
            if policy_id not in policy_points:
                warnings.append(f"Unexpected policy point: {policy_id}")
        
        if errors:
            return ValidationResult(
                is_valid=False,
                message=f"✗ Policy point mapping errors: {'; '.join(errors)}",
                severity="ERROR",
                details={"errors": errors, "warnings": warnings}
            )
        elif warnings:
            return ValidationResult(
                is_valid=True,
                message=f"⚠ Policy point warnings: {'; '.join(warnings)}",
                severity="WARNING",
                details={"warnings": warnings}
            )
        else:
            return ValidationResult(
                is_valid=True,
                message=f"✓ All {len(policy_points)} policy points correctly mapped",
                severity="INFO",
                details={"policy_points": policy_points}
            )
    
    def validate_verification_patterns(
            self,
            loaded_questions: Dict[str, Any]
    ) -> ValidationResult:
        """
        Validate that verification patterns from cuestionario.json are loaded.
        
        Ensures that the rich verification patterns are not ignored.
        """
        questions_with_patterns = 0
        questions_without_patterns = []
        total_patterns = 0
        
        for qid, question in loaded_questions.items():
            patterns = question.verification_patterns
            if patterns and len(patterns) > 0:
                questions_with_patterns += 1
                total_patterns += len(patterns)
            else:
                questions_without_patterns.append(qid)
        
        avg_patterns = total_patterns / len(loaded_questions) if loaded_questions else 0
        
        if questions_without_patterns:
            return ValidationResult(
                is_valid=False,
                message=f"✗ {len(questions_without_patterns)} questions missing verification patterns",
                severity="WARNING",
                details={
                    "questions_with_patterns": questions_with_patterns,
                    "questions_without_patterns": len(questions_without_patterns),
                    "avg_patterns_per_question": avg_patterns,
                    "sample_without_patterns": questions_without_patterns[:5]
                }
            )
        else:
            return ValidationResult(
                is_valid=True,
                message=f"✓ All questions have verification patterns (avg: {avg_patterns:.1f} per question)",
                severity="INFO",
                details={
                    "total_patterns": total_patterns,
                    "avg_patterns_per_question": avg_patterns
                }
            )
    
    def validate_scoring_rubrics(
            self,
            loaded_questions: Dict[str, Any]
    ) -> ValidationResult:
        """
        Validate that scoring rubrics from cuestionario.json are loaded.
        
        Ensures consistent scoring across all evaluations.
        """
        required_levels = ["EXCELENTE", "BUENO", "ACEPTABLE", "INSUFICIENTE"]
        questions_with_complete_rubrics = 0
        questions_with_incomplete_rubrics = []
        
        for qid, question in loaded_questions.items():
            rubric_levels = question.rubric_levels
            
            if all(level in rubric_levels for level in required_levels):
                questions_with_complete_rubrics += 1
            else:
                missing_levels = [l for l in required_levels if l not in rubric_levels]
                questions_with_incomplete_rubrics.append((qid, missing_levels))
        
        if questions_with_incomplete_rubrics:
            return ValidationResult(
                is_valid=False,
                message=f"✗ {len(questions_with_incomplete_rubrics)} questions have incomplete rubrics",
                severity="ERROR",
                details={
                    "required_levels": required_levels,
                    "incomplete_questions": questions_with_incomplete_rubrics[:5]
                }
            )
        else:
            return ValidationResult(
                is_valid=True,
                message=f"✓ All questions have complete scoring rubrics",
                severity="INFO",
                details={"required_levels": required_levels}
            )
    
    def validate_dimension_coverage(
            self,
            loaded_questions: Dict[str, Any]
    ) -> ValidationResult:
        """
        Validate that all 6 dimensions are properly covered.
        
        Each dimension should have 50 questions (10 policy points × 5 questions).
        """
        expected_dimensions = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6']
        
        # Count questions per dimension
        questions_by_dimension = {}
        for qid, question in loaded_questions.items():
            dim = question.dimension
            if dim not in questions_by_dimension:
                questions_by_dimension[dim] = []
            questions_by_dimension[dim].append(qid)
        
        errors = []
        for dim in expected_dimensions:
            expected_count = 50  # 10 policy points × 5 questions
            actual_count = len(questions_by_dimension.get(dim, []))
            
            if actual_count != expected_count:
                errors.append(f"{dim}: expected {expected_count}, got {actual_count}")
        
        if errors:
            return ValidationResult(
                is_valid=False,
                message=f"✗ Dimension coverage errors: {'; '.join(errors)}",
                severity="ERROR",
                details={"errors": errors}
            )
        else:
            return ValidationResult(
                is_valid=True,
                message=f"✓ All 6 dimensions properly covered (50 questions each)",
                severity="INFO",
                details={"dimensions": expected_dimensions}
            )
    
    def run_full_validation(
            self,
            loaded_questions: Dict[str, Any]
    ) -> Tuple[bool, List[ValidationResult]]:
        """
        Run complete validation suite.
        
        Args:
            loaded_questions: Dictionary of loaded questions
            
        Returns:
            Tuple of (is_valid, validation_results)
        """
        logger.info("Running comprehensive cuestionario.json validation...")
        
        results = [
            self.validate_question_coverage(loaded_questions),
            self.validate_policy_point_mapping(loaded_questions),
            self.validate_verification_patterns(loaded_questions),
            self.validate_scoring_rubrics(loaded_questions),
            self.validate_dimension_coverage(loaded_questions)
        ]
        
        # Overall validation passes if no ERROR-level results
        is_valid = all(r.is_valid or r.severity != "ERROR" for r in results)
        
        # Log results
        for result in results:
            if result.severity == "ERROR":
                logger.error(result.message)
            elif result.severity == "WARNING":
                logger.warning(result.message)
            else:
                logger.info(result.message)
        
        if is_valid:
            logger.info("✓ Cuestionario validation PASSED")
        else:
            logger.error("✗ Cuestionario validation FAILED")
        
        return is_valid, results
    
    def generate_validation_report(
            self,
            validation_results: List[ValidationResult],
            output_path: Path
    ):
        """Generate a detailed validation report"""
        report = {
            "timestamp": Path(__file__).stat().st_mtime,
            "cuestionario_version": self.cuestionario_data['metadata']['version'],
            "validation_results": [
                {
                    "is_valid": r.is_valid,
                    "message": r.message,
                    "severity": r.severity,
                    "details": r.details
                }
                for r in validation_results
            ],
            "overall_status": "PASSED" if all(
                r.is_valid or r.severity != "ERROR" for r in validation_results
            ) else "FAILED"
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Validation report saved to {output_path}")
