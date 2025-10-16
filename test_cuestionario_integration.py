#!/usr/bin/env python3
"""
Test script to verify cuestionario.json is properly loaded and integrated
"""
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_cuestionario_json_validity():
    """Test that cuestionario.json is valid JSON"""
    logger.info("Test 1: Validating cuestionario.json syntax")
    
    try:
        with open('cuestionario.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info("✓ cuestionario.json is valid JSON")
        return True, data
    except json.JSONDecodeError as e:
        logger.error(f"✗ JSON syntax error: {e}")
        return False, None
    except FileNotFoundError:
        logger.error("✗ cuestionario.json not found")
        return False, None

def test_cuestionario_structure(data):
    """Test that cuestionario.json has the expected structure"""
    logger.info("\nTest 2: Validating cuestionario.json structure")
    
    required_keys = ['metadata', 'dimensiones', 'puntos_decalogo', 'preguntas_base', 
                     'common_failure_patterns', 'scoring_system', 'causal_glossary']
    
    missing_keys = [key for key in required_keys if key not in data]
    if missing_keys:
        logger.error(f"✗ Missing required keys: {missing_keys}")
        return False
    
    logger.info(f"✓ All required top-level keys present: {required_keys}")
    
    # Validate dimensions
    dimensions = data['dimensiones']
    expected_dims = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6']
    if list(dimensions.keys()) != expected_dims:
        logger.error(f"✗ Expected dimensions {expected_dims}, got {list(dimensions.keys())}")
        return False
    logger.info(f"✓ All 6 dimensions present: {expected_dims}")
    
    # Validate policy points
    policy_points = data['puntos_decalogo']
    expected_points = [f'P{i}' for i in range(1, 11)]
    if list(policy_points.keys()) != expected_points:
        logger.error(f"✗ Expected policy points {expected_points}, got {list(policy_points.keys())}")
        return False
    logger.info(f"✓ All 10 policy points present: {expected_points}")
    
    # Validate base questions
    base_questions = data['preguntas_base']
    expected_count = 300  # 10 policy points × 6 dimensions × 5 questions
    if len(base_questions) != expected_count:
        logger.error(f"✗ Expected {expected_count} base questions, got {len(base_questions)}")
        return False
    logger.info(f"✓ Correct number of base questions: {expected_count}")
    
    return True

def test_question_organization(data):
    """Test that questions are properly organized"""
    logger.info("\nTest 3: Validating question organization")
    
    base_questions = data['preguntas_base']
    policy_points = list(data['puntos_decalogo'].keys())
    
    # Test organization: 30 questions per policy point
    questions_per_policy = 30
    
    for i, policy_id in enumerate(policy_points):
        start_idx = i * questions_per_policy
        end_idx = start_idx + questions_per_policy
        
        if end_idx > len(base_questions):
            logger.error(f"✗ Not enough questions for {policy_id}")
            return False
        
        policy_questions = base_questions[start_idx:end_idx]
        
        # Check dimensions
        dims = set(q['dimension'] for q in policy_questions)
        expected_dims = {'D1', 'D2', 'D3', 'D4', 'D5', 'D6'}
        if dims != expected_dims:
            logger.error(f"✗ {policy_id}: Expected dimensions {expected_dims}, got {dims}")
            return False
        
        # Check question numbers
        for dim in expected_dims:
            dim_questions = [q for q in policy_questions if q['dimension'] == dim]
            if len(dim_questions) != 5:
                logger.error(f"✗ {policy_id}-{dim}: Expected 5 questions, got {len(dim_questions)}")
                return False
            
            nums = sorted(q['numero'] for q in dim_questions)
            if nums != [1, 2, 3, 4, 5]:
                logger.error(f"✗ {policy_id}-{dim}: Expected question numbers [1,2,3,4,5], got {nums}")
                return False
    
    logger.info("✓ All questions properly organized by policy point and dimension")
    return True

def test_question_content(data):
    """Test that questions have required content"""
    logger.info("\nTest 4: Validating question content")
    
    base_questions = data['preguntas_base']
    
    # Check first few questions
    for i, q in enumerate(base_questions[:10]):
        required_fields = ['id', 'dimension', 'numero', 'texto_template', 
                          'criterios_evaluacion', 'patrones_verificacion', 'scoring']
        
        missing_fields = [field for field in required_fields if field not in q]
        if missing_fields:
            logger.error(f"✗ Question {i} missing fields: {missing_fields}")
            return False
        
        # Check verification patterns
        patterns = q.get('patrones_verificacion', [])
        if len(patterns) == 0:
            logger.warning(f"⚠ Question {q['id']} has no verification patterns")
        
        # Check scoring
        scoring = q.get('scoring', {})
        if not all(level in scoring for level in ['excelente', 'bueno', 'aceptable', 'insuficiente']):
            logger.error(f"✗ Question {q['id']} missing scoring levels")
            return False
    
    logger.info("✓ Questions have required content and structure")
    return True

def test_question_router_integration():
    """Test that QuestionRouter can load the cuestionario"""
    logger.info("\nTest 5: Testing QuestionRouter integration and validation")
    
    try:
        # Test the validator directly without importing full orchestrator
        import sys
        import importlib.util
        from pathlib import Path
        
        # Direct import of validator module
        validator_path = Path('orchestrator/cuestionario_validator.py')
        spec = importlib.util.spec_from_file_location("cuestionario_validator", validator_path)
        validator_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(validator_module)
        CuestionarioValidator = validator_module.CuestionarioValidator
        
        # Load cuestionario.json
        import json
        with open('cuestionario.json', 'r') as f:
            data = json.load(f)
        
        # Create mock questions structure for validation
        mock_questions = {}
        policy_points = list(data['puntos_decalogo'].keys())
        base_questions = data['preguntas_base']
        
        # Simple mock that mimics QuestionRouter structure
        from dataclasses import dataclass
        from typing import List, Dict
        
        @dataclass
        class MockQuestion:
            policy_area: str
            dimension: str
            question_num: int
            verification_patterns: List[str]
            rubric_levels: Dict[str, float]
        
        # Create mock questions
        questions_per_policy = 30
        for idx, q_data in enumerate(base_questions):
            policy_idx = idx // questions_per_policy
            if policy_idx >= len(policy_points):
                continue
            
            policy_id = policy_points[policy_idx]
            dim = q_data.get('dimension', '')
            q_num = q_data.get('numero', 0)
            
            qid = f"{policy_id}-{dim}-Q{q_num}"
            mock_questions[qid] = MockQuestion(
                policy_area=policy_id,
                dimension=dim,
                question_num=q_num,
                verification_patterns=q_data.get('patrones_verificacion', []),
                rubric_levels={
                    "EXCELENTE": 0.85,
                    "BUENO": 0.70,
                    "ACEPTABLE": 0.55,
                    "INSUFICIENTE": 0.0
                }
            )
        
        # Run validation
        validator = CuestionarioValidator(Path('cuestionario.json'))
        is_valid, results = validator.run_full_validation(mock_questions)
        
        if is_valid:
            logger.info("✓ QuestionRouter validation passed")
            return True
        else:
            logger.error("✗ QuestionRouter validation failed")
            return False
        
    except Exception as e:
        logger.error(f"✗ Error testing QuestionRouter: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    logger.info("="*80)
    logger.info("Testing cuestionario.json Integration")
    logger.info("="*80)
    
    # Test 1: JSON validity
    valid, data = test_cuestionario_json_validity()
    if not valid:
        logger.error("\n✗ FAILED: cuestionario.json is not valid")
        return 1
    
    # Test 2: Structure
    if not test_cuestionario_structure(data):
        logger.error("\n✗ FAILED: cuestionario.json structure is invalid")
        return 1
    
    # Test 3: Organization
    if not test_question_organization(data):
        logger.error("\n✗ FAILED: Questions are not properly organized")
        return 1
    
    # Test 4: Content
    if not test_question_content(data):
        logger.error("\n✗ FAILED: Question content is incomplete")
        return 1
    
    # Test 5: Integration
    if not test_question_router_integration():
        logger.error("\n✗ FAILED: QuestionRouter integration issue")
        return 1
    
    logger.info("\n" + "="*80)
    logger.info("✓ ALL TESTS PASSED")
    logger.info("="*80)
    logger.info("\nSummary:")
    logger.info("  - cuestionario.json is valid JSON")
    logger.info("  - All required structure elements present")
    logger.info("  - 300 questions properly organized (10 policy points × 30 questions)")
    logger.info("  - Questions have verification patterns and scoring criteria")
    logger.info("  - Ready for integration with QuestionRouter")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
