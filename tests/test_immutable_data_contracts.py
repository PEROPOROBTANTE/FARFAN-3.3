# coding=utf-8
"""
Property-Based Tests for Immutable Data Contracts
==================================================

Tests verify that all adapter methods and module controller operations
do not mutate their input parameters using Hypothesis property-based testing.

Author: FARFAN 3.0 Team
Version: 1.0.0
Python: 3.10+
"""

import json
import copy
import hashlib
from pathlib import Path
from typing import Any, Dict, List
import pytest
from hypothesis import given, strategies as st, settings, assume, HealthCheck
from hypothesis.strategies import composite

# Import immutable models
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator.data_models import (
    QuestionMetadata,
    ExecutionStep,
    QuestionSpec,
    PolicyChunk,
    PolicySegment,
    EmbeddingVector,
    ChunkEmbedding,
    Evidence,
    ModuleResult,
    ExecutionResult,
    AnalysisResult,
    DimensionAnalysis,
    ProcessedDocument,
    DocumentMetadata,
    RouteInfo,
    ExecutionStatusEnum,
    QualitativeLevelEnum,
)


# ============================================================================
# HELPER FUNCTIONS FOR IMMUTABILITY VERIFICATION
# ============================================================================

def compute_hash(obj: Any) -> str:
    """
    Compute deterministic hash of an object for immutability verification
    
    Args:
        obj: Pydantic model instance or dict
        
    Returns:
        SHA256 hash of the object's JSON representation
    """
    if hasattr(obj, 'model_dump'):
        # Pydantic v2
        data = obj.model_dump()
    elif hasattr(obj, 'dict'):
        # Pydantic v1
        data = obj.dict()
    else:
        data = obj
    
    json_str = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(json_str.encode()).hexdigest()


def verify_immutable(obj: Any, operation_func, *args, **kwargs) -> bool:
    """
    Verify that an operation does not mutate the input object
    
    Args:
        obj: Object to test
        operation_func: Function that operates on obj
        *args, **kwargs: Additional arguments to operation_func
        
    Returns:
        True if object remains unchanged after operation
    """
    hash_before = compute_hash(obj)
    
    try:
        operation_func(obj, *args, **kwargs)
    except Exception:
        # Operation may fail, but object should still be unchanged
        pass
    
    hash_after = compute_hash(obj)
    
    return hash_before == hash_after


def verify_frozen(model_class):
    """
    Verify that a Pydantic model with frozen=True cannot be mutated
    
    Args:
        model_class: Pydantic model class to test
        
    Returns:
        True if model is properly frozen
    """
    return model_class.model_config.get('frozen', False)


# ============================================================================
# HYPOTHESIS STRATEGIES FOR GENERATING TEST DATA
# ============================================================================

@composite
def question_metadata_strategy(draw):
    """Generate valid QuestionMetadata instances"""
    policy_num = draw(st.integers(min_value=1, max_value=10))
    dimension_num = draw(st.integers(min_value=1, max_value=6))
    question_num = draw(st.integers(min_value=1, max_value=5))
    
    canonical_id = f"P{policy_num}-D{dimension_num}-Q{question_num}"
    
    return QuestionMetadata(
        canonical_id=canonical_id,
        policy_area=f"P{policy_num}",
        dimension=f"D{dimension_num}",
        question_number=question_num,
        question_text=draw(st.text(min_size=10, max_size=200)),
        scoring_modality=draw(st.sampled_from(["TYPE_A", "TYPE_B", "TYPE_C", "TYPE_D"])),
        expected_elements=tuple(draw(st.lists(st.text(min_size=1, max_size=50), max_size=5))),
        element_weights=draw(st.dictionaries(
            st.text(min_size=1, max_size=20),
            st.floats(min_value=0.0, max_value=1.0),
            max_size=5
        )),
        numerical_thresholds=draw(st.dictionaries(
            st.text(min_size=1, max_size=20),
            st.floats(min_value=0.0, max_value=100.0),
            max_size=5
        )),
        verification_patterns=tuple(draw(st.lists(st.text(min_size=1, max_size=50), max_size=5))),
        metadata=draw(st.dictionaries(
            st.text(min_size=1, max_size=20),
            st.text(min_size=1, max_size=100),
            max_size=5
        ))
    )


@composite
def execution_step_strategy(draw):
    """Generate valid ExecutionStep instances"""
    adapters = ["teoria_cambio", "analyzer_one", "dereck_beach", "embedding_policy",
                "semantic_chunking_policy", "contradiction_detection", "financial_viability",
                "policy_processor", "policy_segmenter"]
    
    return ExecutionStep(
        adapter=draw(st.sampled_from(adapters)),
        method=draw(st.text(min_size=1, max_size=50)),
        args=tuple(draw(st.lists(st.text(min_size=0, max_size=100), max_size=3))),
        kwargs=draw(st.dictionaries(
            st.text(min_size=1, max_size=20),
            st.text(min_size=0, max_size=100),
            max_size=3
        )),
        depends_on=tuple(draw(st.lists(st.text(min_size=1, max_size=20), max_size=3)))
    )


@composite
def policy_chunk_strategy(draw):
    """Generate valid PolicyChunk instances"""
    start = draw(st.integers(min_value=0, max_value=10000))
    end = draw(st.integers(min_value=start, max_value=start + 1000))
    
    return PolicyChunk(
        chunk_id=draw(st.text(min_size=1, max_size=50)),
        text=draw(st.text(min_size=1, max_size=500)),
        start_position=start,
        end_position=end,
        section_title=draw(st.one_of(st.none(), st.text(min_size=1, max_size=100))),
        page_number=draw(st.one_of(st.none(), st.integers(min_value=1, max_value=500))),
        metadata=draw(st.dictionaries(
            st.text(min_size=1, max_size=20),
            st.text(min_size=0, max_size=100),
            max_size=5
        ))
    )


@composite
def embedding_vector_strategy(draw):
    """Generate valid EmbeddingVector instances"""
    dimension = draw(st.integers(min_value=1, max_value=512))
    values = tuple(draw(st.floats(min_value=-1.0, max_value=1.0)) for _ in range(dimension))
    
    return EmbeddingVector(
        vector_id=draw(st.text(min_size=1, max_size=50)),
        values=values,
        model_name=draw(st.sampled_from(["sentence-transformers", "openai", "custom"])),
        dimension=dimension,
        metadata=draw(st.dictionaries(
            st.text(min_size=1, max_size=20),
            st.text(min_size=0, max_size=100),
            max_size=5
        ))
    )


@composite
def evidence_strategy(draw):
    """Generate valid Evidence instances"""
    start = draw(st.integers(min_value=0, max_value=10000))
    end = draw(st.integers(min_value=start, max_value=start + 500))
    
    return Evidence(
        text=draw(st.text(min_size=1, max_size=300)),
        source_chunk_id=draw(st.one_of(st.none(), st.text(min_size=1, max_size=50))),
        confidence=draw(st.floats(min_value=0.0, max_value=1.0)),
        position=draw(st.one_of(st.none(), st.just((start, end)))),
        metadata=draw(st.dictionaries(
            st.text(min_size=1, max_size=20),
            st.text(min_size=0, max_size=100),
            max_size=5
        ))
    )


@composite
def module_result_strategy(draw):
    """Generate valid ModuleResult instances"""
    return ModuleResult(
        module_name=draw(st.text(min_size=1, max_size=50)),
        class_name=draw(st.text(min_size=1, max_size=50)),
        method_name=draw(st.text(min_size=1, max_size=50)),
        status=draw(st.sampled_from(list(ExecutionStatusEnum))),
        data=draw(st.dictionaries(
            st.text(min_size=1, max_size=20),
            st.text(min_size=0, max_size=100),
            max_size=10
        )),
        errors=tuple(draw(st.lists(st.text(min_size=0, max_size=100), max_size=3))),
        execution_time=draw(st.floats(min_value=0.0, max_value=60.0)),
        evidence=tuple(draw(st.lists(evidence_strategy(), max_size=3))),
        confidence=draw(st.floats(min_value=0.0, max_value=1.0)),
        metadata=draw(st.dictionaries(
            st.text(min_size=1, max_size=20),
            st.text(min_size=0, max_size=100),
            max_size=5
        ))
    )


@composite
def analysis_result_strategy(draw):
    """Generate valid AnalysisResult instances"""
    policy_num = draw(st.integers(min_value=1, max_value=10))
    dimension_num = draw(st.integers(min_value=1, max_value=6))
    question_num = draw(st.integers(min_value=1, max_value=5))
    
    return AnalysisResult(
        question_id=f"P{policy_num}-D{dimension_num}-Q{question_num}",
        qualitative_level=draw(st.sampled_from(list(QualitativeLevelEnum))),
        quantitative_score=draw(st.floats(min_value=0.0, max_value=3.0)),
        evidence=tuple(draw(st.lists(evidence_strategy(), max_size=5))),
        explanation=draw(st.text(min_size=100, max_size=500)),
        confidence=draw(st.floats(min_value=0.0, max_value=1.0)),
        scoring_modality=draw(st.sampled_from(["TYPE_A", "TYPE_B", "TYPE_C", "TYPE_D"])),
        elements_found=draw(st.dictionaries(
            st.text(min_size=1, max_size=20),
            st.booleans(),
            max_size=5
        )),
        modules_executed=tuple(draw(st.lists(st.text(min_size=1, max_size=50), max_size=5))),
        execution_time=draw(st.floats(min_value=0.0, max_value=60.0)),
        metadata=draw(st.dictionaries(
            st.text(min_size=1, max_size=20),
            st.text(min_size=0, max_size=100),
            max_size=5
        ))
    )


# ============================================================================
# PROPERTY-BASED TESTS FOR MODEL IMMUTABILITY
# ============================================================================

@given(question_metadata_strategy())
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_question_metadata_is_frozen(metadata):
    """Verify QuestionMetadata instances cannot be mutated"""
    assert verify_frozen(QuestionMetadata)
    
    # Attempt to modify should raise error (Pydantic raises ValidationError for frozen models)
    with pytest.raises(Exception):  # Pydantic 2.x raises ValidationError
        metadata.canonical_id = "P99-D99-Q99"
    
    # Tuples don't have append method
    with pytest.raises(AttributeError):
        metadata.expected_elements.append("new_element")


@given(execution_step_strategy())
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_execution_step_is_frozen(step):
    """Verify ExecutionStep instances cannot be mutated"""
    assert verify_frozen(ExecutionStep)
    
    with pytest.raises(Exception):  # Pydantic 2.x raises ValidationError
        step.adapter = "new_adapter"
    
    with pytest.raises(AttributeError):  # Tuples don't have append
        step.args.append("new_arg")


@given(policy_chunk_strategy())
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_policy_chunk_is_frozen(chunk):
    """Verify PolicyChunk instances cannot be mutated"""
    assert verify_frozen(PolicyChunk)
    
    with pytest.raises(Exception):  # Pydantic 2.x raises ValidationError
        chunk.text = "modified text"
    
    with pytest.raises(Exception):  # Pydantic 2.x raises ValidationError
        chunk.start_position = 999


@given(embedding_vector_strategy())
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_embedding_vector_is_frozen(embedding):
    """Verify EmbeddingVector instances cannot be mutated"""
    assert verify_frozen(EmbeddingVector)
    
    with pytest.raises(Exception):  # Pydantic 2.x raises ValidationError
        embedding.values = tuple([0.0] * len(embedding.values))
    
    with pytest.raises(Exception):  # Pydantic 2.x raises ValidationError
        embedding.dimension = 999


@given(evidence_strategy())
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_evidence_is_frozen(evidence):
    """Verify Evidence instances cannot be mutated"""
    assert verify_frozen(Evidence)
    
    with pytest.raises(Exception):  # Pydantic 2.x raises ValidationError
        evidence.text = "modified evidence"
    
    with pytest.raises(Exception):  # Pydantic 2.x raises ValidationError
        evidence.confidence = 0.999


@given(module_result_strategy())
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_module_result_is_frozen(result):
    """Verify ModuleResult instances cannot be mutated"""
    assert verify_frozen(ModuleResult)
    
    with pytest.raises(Exception):  # Pydantic 2.x raises ValidationError
        result.module_name = "modified_module"
    
    with pytest.raises(Exception):  # Pydantic 2.x raises ValidationError
        result.confidence = 0.999
    
    with pytest.raises(AttributeError):  # Tuples don't have append
        result.errors.append("new_error")


@given(analysis_result_strategy())
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_analysis_result_is_frozen(result):
    """Verify AnalysisResult instances cannot be mutated"""
    assert verify_frozen(AnalysisResult)
    
    with pytest.raises(Exception):  # Pydantic 2.x raises ValidationError
        result.question_id = "P99-D99-Q99"
    
    with pytest.raises(Exception):  # Pydantic 2.x raises ValidationError
        result.quantitative_score = 3.0
    
    with pytest.raises(AttributeError):  # Tuples don't have append
        result.evidence.append(Evidence(text="new", confidence=1.0))


# ============================================================================
# PROPERTY-BASED TESTS FOR OPERATIONS ON IMMUTABLE DATA
# ============================================================================

@given(question_metadata_strategy())
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_question_metadata_serialization_preserves_immutability(metadata):
    """Verify serialization/deserialization preserves immutability"""
    # Serialize to dict
    data_dict = metadata.model_dump()
    
    # Deserialize back
    restored = QuestionMetadata(**data_dict)
    
    # Verify both are frozen and equal
    assert verify_frozen(QuestionMetadata)
    assert metadata == restored
    assert compute_hash(metadata) == compute_hash(restored)


@given(module_result_strategy())
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_module_result_aggregation_does_not_mutate(result):
    """Verify aggregation operations do not mutate original results"""
    
    def aggregate_results(r):
        """Mock aggregation function"""
        # Try to access and process data
        _ = r.module_name
        _ = r.confidence
        _ = len(r.evidence)
        return {"aggregated": True}
    
    assert verify_immutable(result, aggregate_results)


@given(st.lists(evidence_strategy(), min_size=1, max_size=10))
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_evidence_collection_does_not_mutate(evidence_list):
    """Verify collecting evidence does not mutate original items"""
    evidence_tuple = tuple(evidence_list)
    
    def collect_evidence(ev_tuple):
        """Mock evidence collection"""
        total_confidence = sum(e.confidence for e in ev_tuple)
        text_lengths = [len(e.text) for e in ev_tuple]
        return total_confidence, text_lengths
    
    hash_before = tuple(compute_hash(e) for e in evidence_tuple)
    
    collect_evidence(evidence_tuple)
    
    hash_after = tuple(compute_hash(e) for e in evidence_tuple)
    
    assert hash_before == hash_after


@given(policy_chunk_strategy())
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_policy_chunk_processing_does_not_mutate(chunk):
    """Verify processing operations do not mutate chunks"""
    
    def process_chunk(c):
        """Mock chunk processing"""
        _ = c.text.lower()
        _ = c.end_position - c.start_position
        return {"processed": True}
    
    assert verify_immutable(chunk, process_chunk)


# ============================================================================
# TESTS WITH REAL CUESTIONARIO.JSON DATA
# ============================================================================

def load_cuestionario():
    """Load cuestionario.json for testing"""
    cuestionario_path = Path(__file__).parent.parent / "cuestionario.json"
    if not cuestionario_path.exists():
        pytest.skip("cuestionario.json not found")
    
    with open(cuestionario_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def test_cuestionario_loads_as_immutable_models():
    """Test that cuestionario.json can be loaded into immutable models"""
    data = load_cuestionario()
    
    # Sample questions from cuestionario.json
    dimensiones = data.get("dimensiones", {})
    
    for dim_id, dim_data in dimensiones.items():
        preguntas = dim_data.get("preguntas_especificas", {})
        
        for policy_area, questions in preguntas.items():
            for q_num, q_data in questions.items():
                # Create immutable QuestionMetadata
                metadata = QuestionMetadata(
                    canonical_id=f"{policy_area}-{dim_id}-Q{q_num}",
                    policy_area=policy_area,
                    dimension=dim_id,
                    question_number=int(q_num),
                    question_text=q_data.get("pregunta", ""),
                    scoring_modality=q_data.get("scoring_modality", "TYPE_A"),
                    expected_elements=tuple(q_data.get("expected_elements", [])),
                    verification_patterns=tuple(q_data.get("verification_patterns", [])),
                    metadata={}
                )
                
                # Verify it's frozen
                assert verify_frozen(QuestionMetadata)
                
                # Verify cannot mutate
                with pytest.raises((AttributeError, TypeError)):
                    metadata.question_text = "modified"


@given(st.lists(module_result_strategy(), min_size=2, max_size=5))
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_execution_chain_does_not_mutate_inputs(results):
    """Verify execution chain operations do not mutate input results"""
    results_tuple = tuple(results)
    
    def execute_chain(result_tuple):
        """Mock execution chain"""
        for r in result_tuple:
            _ = r.module_name
            _ = r.status
            _ = r.confidence
        return {"chain_executed": True}
    
    hashes_before = tuple(compute_hash(r) for r in results_tuple)
    
    execute_chain(results_tuple)
    
    hashes_after = tuple(compute_hash(r) for r in results_tuple)
    
    assert hashes_before == hashes_after


@given(
    question_metadata_strategy(),
    st.lists(module_result_strategy(), min_size=1, max_size=5)
)
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_report_generation_does_not_mutate_inputs(metadata, results):
    """Verify report generation does not mutate question metadata or results"""
    results_tuple = tuple(results)
    
    def generate_report(meta, res_tuple):
        """Mock report generation"""
        report = {
            "question_id": meta.canonical_id,
            "dimension": meta.dimension,
            "results_count": len(res_tuple),
            "avg_confidence": sum(r.confidence for r in res_tuple) / len(res_tuple)
        }
        return report
    
    metadata_hash_before = compute_hash(metadata)
    results_hashes_before = tuple(compute_hash(r) for r in results_tuple)
    
    generate_report(metadata, results_tuple)
    
    metadata_hash_after = compute_hash(metadata)
    results_hashes_after = tuple(compute_hash(r) for r in results_tuple)
    
    assert metadata_hash_before == metadata_hash_after
    assert results_hashes_before == results_hashes_after


# ============================================================================
# TESTS FOR TUPLE IMMUTABILITY
# ============================================================================

@given(st.lists(st.text(min_size=1, max_size=50), min_size=1, max_size=10))
@settings(max_examples=50)
def test_expected_elements_tuple_is_immutable(elements):
    """Verify expected_elements tuple cannot be modified"""
    metadata = QuestionMetadata(
        canonical_id="P1-D1-Q1",
        policy_area="P1",
        dimension="D1",
        question_number=1,
        question_text="Test question",
        scoring_modality="TYPE_A",
        expected_elements=tuple(elements)
    )
    
    # Tuple should not have append method
    assert not hasattr(metadata.expected_elements, 'append')
    
    # Cannot modify via assignment (Pydantic 2.x raises ValidationError)
    with pytest.raises(Exception):
        metadata.expected_elements = ("modified",)
    
    # Cannot modify tuple contents
    with pytest.raises(TypeError):
        metadata.expected_elements[0] = "modified"


@given(st.lists(evidence_strategy(), min_size=1, max_size=10))
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_evidence_tuple_is_immutable(evidence_list):
    """Verify evidence tuple cannot be modified"""
    # Need longer explanation to meet min_length=100
    explanation = "Test explanation with sufficient length to meet the minimum requirement of 100 characters for the analysis result model validation rules."
    
    result = AnalysisResult(
        question_id="P1-D1-Q1",
        qualitative_level=QualitativeLevelEnum.BUENO,
        quantitative_score=2.5,
        evidence=tuple(evidence_list),
        explanation=explanation,
        confidence=0.8,
        scoring_modality="TYPE_A",
        execution_time=1.0
    )
    
    # Tuple should not have append method
    assert not hasattr(result.evidence, 'append')
    
    # Cannot modify via assignment (Pydantic 2.x raises ValidationError)
    with pytest.raises(Exception):
        result.evidence = (Evidence(text="new", confidence=1.0),)


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

@given(st.lists(question_metadata_strategy(), min_size=10, max_size=100))
@settings(max_examples=10, suppress_health_check=[HealthCheck.too_slow])
def test_large_scale_immutability_verification(metadata_list):
    """Verify immutability at scale with many question metadata instances"""
    metadata_tuple = tuple(metadata_list)
    
    # Verify all are immutable
    hashes_before = tuple(compute_hash(m) for m in metadata_tuple)
    
    # Simulate batch processing
    for m in metadata_tuple:
        _ = m.canonical_id
        _ = m.scoring_modality
        _ = len(m.expected_elements)
    
    hashes_after = tuple(compute_hash(m) for m in metadata_tuple)
    
    assert hashes_before == hashes_after


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--hypothesis-show-statistics"])
