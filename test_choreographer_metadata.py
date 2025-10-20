"""
Test Suite for Choreographer Metadata Enrichment and Validation
================================================================

Tests the enhanced ExecutionChoreographer with:
- QuestionContext extraction from cuestionario.json
- Metadata injection into module invocations
- Post-processing validation against question requirements
- Retry logic on validation failures
- Dependency satisfaction checking

Author: FARFAN Integration Team
Version: 3.0.0
Python: 3.10+
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any

# Add orchestrator to path
sys.path.insert(0, str(Path(__file__).parent / "orchestrator"))

from choreographer import (
    ExecutionChoreographer,
    QuestionContext,
    ValidationResult,
    ExecutionResult,
    ExecutionStatus
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_cuestionario():
    """Create test cuestionario.json with preguntas array"""
    test_data = {
        "metadata": {
            "version": "2.0.0",
            "total_questions": 2
        },
        "dimensiones": {
            "D1": {
                "nombre": "Insumos (Diagnóstico y Líneas Base)",
                "peso_por_punto": {"P1": 0.2},
                "umbral_minimo": 0.5,
                "decalogo_dimension_mapping": {
                    "P1": {
                        "weight": 0.2,
                        "is_critical": True,
                        "minimum_score": 0.5
                    }
                }
            }
        },
        "puntos_decalogo": {
            "P1": {
                "nombre": "Derechos de las mujeres e igualdad de género",
                "dimensiones_criticas": ["D1"],
                "indicadores_producto": ["Mujeres formadas"],
                "indicadores_resultado": ["Tasa de violencia"]
            }
        },
        "preguntas": [
            {
                "id": "D1_P1_Q001",
                "texto": "¿El plan incluye diagnóstico con líneas base?",
                "tipo_respuesta": "cualitativa",
                "escala": {"min": 0, "max": 100},
                "formato_esperado": "texto_estructurado",
                "umbral_confianza": 0.6,
                "tipos_evidencia_requeridos": ["pattern_match", "semantic_similarity"],
                "patrones_validacion": ["diagnóstico|línea base"],
                "restricciones_rango": {},
                "fuentes_verificacion": ["plan_desarrollo"],
                "alcance": "municipal",
                "dependencias": [],
                "estrategia_error": "retry_specific"
            }
        ]
    }
    
    test_path = Path("test_cuestionario.json")
    with open(test_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    
    return str(test_path)


def test_question_context_extraction():
    """Test extraction of QuestionContext from cuestionario.json"""
    print("\n" + "=" * 80)
    print("TEST 1: QuestionContext Extraction")
    print("=" * 80)
    
    # Create test cuestionario
    test_path = create_test_cuestionario()
    
    # Initialize choreographer
    choreographer = ExecutionChoreographer(
        cuestionario_path=test_path,
        max_retries=3
    )
    
    # Extract context
    context = choreographer.extract_question_context("D1_P1_Q001")
    
    assert context is not None, "Context should not be None"
    assert context.question_id == "D1_P1_Q001"
    assert context.dimension == "D1"
    assert context.punto_decalogo == "P1"
    assert context.is_critical is True
    assert context.minimum_score == 0.5
    assert "diagnóstico" in context.question_text.lower() or "línea base" in context.question_text.lower()
    
    print(f"✓ Context extracted successfully")
    print(f"  Question ID: {context.question_id}")
    print(f"  Dimension: {context.dimension}")
    print(f"  Critical: {context.is_critical}")
    print(f"  Min Score: {context.minimum_score}")
    print(f"  Constraints: {list(context.constraints.keys())}")
    print(f"  Validation Rules: {list(context.validation_rules.keys())}")
    
    # Test caching
    context2 = choreographer.extract_question_context("D1_P1_Q001")
    assert context is context2, "Should return cached context"
    print(f"✓ Context caching works")
    
    # Cleanup
    Path(test_path).unlink()
    
    return True


def test_validation_engine():
    """Test post-processing validation engine"""
    print("\n" + "=" * 80)
    print("TEST 2: Validation Engine")
    print("=" * 80)
    
    test_path = create_test_cuestionario()
    choreographer = ExecutionChoreographer(cuestionario_path=test_path)
    
    # Create test context
    context = QuestionContext(
        question_id="D1_P1_Q001",
        question_text="Test question",
        validation_rules={
            "minimum_confidence": 0.7,
            "required_evidence_types": ["pattern_match"],
            "range_constraints": {"score": {"min": 0, "max": 100}}
        }
    )
    
    # Test Case 1: Valid result
    result_valid = ExecutionResult(
        module_name="test_adapter",
        adapter_class="TestAdapter",
        method_name="test_method",
        status=ExecutionStatus.COMPLETED,
        output={"score": 85, "text": "Valid output"},
        confidence=0.8,
        evidence_extracted={
            "evidence": [
                {"type": "pattern_match", "value": "found"}
            ]
        }
    )
    
    validation = choreographer._validate_module_response(result_valid, context)
    
    assert validation.is_valid, "Should be valid"
    assert len(validation.violations) == 0
    assert validation.confidence_score > 0.8
    print(f"✓ Valid result passes validation")
    print(f"  Confidence: {validation.confidence_score:.2f}")
    print(f"  Violations: {len(validation.violations)}")
    
    # Test Case 2: Low confidence
    result_low_conf = ExecutionResult(
        module_name="test_adapter",
        adapter_class="TestAdapter",
        method_name="test_method",
        status=ExecutionStatus.COMPLETED,
        output={"score": 85},
        confidence=0.5,
        evidence_extracted={"evidence": [{"type": "pattern_match"}]}
    )
    
    validation_low = choreographer._validate_module_response(result_low_conf, context)
    
    assert not validation_low.is_valid, "Should be invalid due to low confidence"
    assert any("Confidence" in v for v in validation_low.violations)
    print(f"✓ Low confidence detected")
    print(f"  Violations: {validation_low.violations}")
    
    # Test Case 3: Missing evidence
    result_no_evidence = ExecutionResult(
        module_name="test_adapter",
        adapter_class="TestAdapter",
        method_name="test_method",
        status=ExecutionStatus.COMPLETED,
        output={"score": 85},
        confidence=0.8,
        evidence_extracted={"evidence": []}
    )
    
    validation_no_ev = choreographer._validate_module_response(result_no_evidence, context)
    
    assert not validation_no_ev.is_valid, "Should be invalid due to missing evidence"
    assert any("evidence" in v.lower() for v in validation_no_ev.violations)
    print(f"✓ Missing evidence detected")
    print(f"  Violations: {validation_no_ev.violations}")
    
    # Test Case 4: Out of range
    context_with_range = QuestionContext(
        question_id="D1_P1_Q001",
        question_text="Test",
        validation_rules={
            "minimum_confidence": 0.5,
            "range_constraints": {"score": {"min": 0, "max": 100}}
        }
    )
    
    result_out_of_range = ExecutionResult(
        module_name="test_adapter",
        adapter_class="TestAdapter",
        method_name="test_method",
        status=ExecutionStatus.COMPLETED,
        output={"score": 150},
        confidence=0.8,
        evidence_extracted={}
    )
    
    validation_range = choreographer._validate_module_response(result_out_of_range, context_with_range)
    
    assert not validation_range.is_valid, "Should be invalid due to out of range"
    assert any("above maximum" in v for v in validation_range.violations)
    print(f"✓ Range violation detected")
    print(f"  Violations: {validation_range.violations}")
    
    Path(test_path).unlink()
    
    return True


def test_dependency_checking():
    """Test dependency satisfaction checking"""
    print("\n" + "=" * 80)
    print("TEST 3: Dependency Checking")
    print("=" * 80)
    
    test_path = create_test_cuestionario()
    choreographer = ExecutionChoreographer(cuestionario_path=test_path)
    
    # Create context with dependencies
    context = QuestionContext(
        question_id="D2_P1_Q010",
        question_text="Dependent question",
        dependencies=["D1_P1_Q001", "D1_P1_Q002"]
    )
    
    # No dependencies satisfied yet
    satisfied = choreographer._check_dependencies_satisfied(context)
    assert not satisfied, "Should not be satisfied"
    print(f"✓ Unsatisfied dependencies detected")
    
    # Add one dependency
    choreographer.dependency_results["D1_P1_Q001"] = ExecutionResult(
        module_name="test",
        adapter_class="Test",
        method_name="test",
        status=ExecutionStatus.COMPLETED,
        validation_result=ValidationResult(is_valid=True, violations=[])
    )
    
    satisfied = choreographer._check_dependencies_satisfied(context)
    assert not satisfied, "Should still not be satisfied (1/2 deps)"
    print(f"✓ Partial dependencies detected (1/2)")
    
    # Add second dependency
    choreographer.dependency_results["D1_P1_Q002"] = ExecutionResult(
        module_name="test",
        adapter_class="Test",
        method_name="test",
        status=ExecutionStatus.COMPLETED,
        validation_result=ValidationResult(is_valid=True, violations=[])
    )
    
    satisfied = choreographer._check_dependencies_satisfied(context)
    assert satisfied, "Should be satisfied"
    print(f"✓ All dependencies satisfied (2/2)")
    
    # Test with failed validation in dependency
    choreographer.dependency_results["D1_P1_Q002"] = ExecutionResult(
        module_name="test",
        adapter_class="Test",
        method_name="test",
        status=ExecutionStatus.COMPLETED,
        validation_result=ValidationResult(is_valid=False, violations=["test violation"])
    )
    
    satisfied = choreographer._check_dependencies_satisfied(context)
    assert not satisfied, "Should not be satisfied (failed validation)"
    print(f"✓ Failed validation in dependency detected")
    
    Path(test_path).unlink()
    
    return True


def test_validation_statistics():
    """Test validation statistics calculation"""
    print("\n" + "=" * 80)
    print("TEST 4: Validation Statistics")
    print("=" * 80)
    
    test_path = create_test_cuestionario()
    choreographer = ExecutionChoreographer(cuestionario_path=test_path)
    
    # Create sample results
    results = {
        "step1": ExecutionResult(
            module_name="test1",
            adapter_class="Test",
            method_name="method1",
            status=ExecutionStatus.COMPLETED,
            confidence=0.9,
            validation_result=ValidationResult(
                is_valid=True,
                violations=[],
                confidence_score=0.95
            )
        ),
        "step2": ExecutionResult(
            module_name="test2",
            adapter_class="Test",
            method_name="method2",
            status=ExecutionStatus.COMPLETED,
            confidence=0.7,
            validation_result=ValidationResult(
                is_valid=False,
                violations=["Low confidence", "Missing evidence"],
                confidence_score=0.4
            ),
            retry_count=2
        ),
        "step3": ExecutionResult(
            module_name="test3",
            adapter_class="Test",
            method_name="method3",
            status=ExecutionStatus.FAILED,
            confidence=0.0
        )
    }
    
    stats = choreographer.get_validation_statistics(results)
    
    assert stats["total_steps"] == 3
    assert stats["validated_steps"] == 2
    assert stats["valid_steps"] == 1
    assert stats["failed_validation_steps"] == 1
    assert stats["retried_steps"] == 1
    assert stats["validation_rate"] == 0.5
    assert stats["total_violations"] == 2
    
    print(f"✓ Statistics calculated correctly")
    print(f"  Total steps: {stats['total_steps']}")
    print(f"  Valid steps: {stats['valid_steps']}")
    print(f"  Failed validation: {stats['failed_validation_steps']}")
    print(f"  Retried steps: {stats['retried_steps']}")
    print(f"  Validation rate: {stats['validation_rate']:.1%}")
    print(f"  Avg execution confidence: {stats['avg_execution_confidence']:.2f}")
    print(f"  Avg validation confidence: {stats['avg_validation_confidence']:.2f}")
    print(f"  Total violations: {stats['total_violations']}")
    
    Path(test_path).unlink()
    
    return True


def run_all_tests():
    """Run all choreographer metadata tests"""
    print("\n" + "=" * 80)
    print("CHOREOGRAPHER METADATA ENRICHMENT TEST SUITE")
    print("=" * 80)
    
    tests = [
        ("QuestionContext Extraction", test_question_context_extraction),
        ("Validation Engine", test_validation_engine),
        ("Dependency Checking", test_dependency_checking),
        ("Validation Statistics", test_validation_statistics)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"\n✓ {test_name} PASSED")
        except Exception as e:
            failed += 1
            print(f"\n✗ {test_name} FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print(f"TEST SUMMARY: {passed}/{len(tests)} passed, {failed}/{len(tests)} failed")
    print("=" * 80)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
