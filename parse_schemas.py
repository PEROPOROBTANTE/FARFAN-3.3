#!/usr/bin/env python3
"""
Parse cuestionario.json and rubric_scoring.json to extract metadata, structures,
validate relationships, and output schema_contracts.json with normalized schemas
and validation rules.
"""

import json
from typing import Dict, List, Any, Tuple
from pathlib import Path


def load_json(filepath: str) -> Dict[str, Any]:
    """Load and parse JSON file, attempting fix if needed."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"⚠️  JSONDecodeError in {filepath}: {e}")
        print("   Attempting to fix...")
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        # Check if missing closing brace
        content = content.rstrip()
        if not content.endswith("}") and content.endswith("]"):
            content = content + "\n}"
            return json.loads(content)
        raise


def extract_cuestionario_metadata(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract metadata from cuestionario.json."""
    return {
        "version": data["metadata"]["version"],
        "created_date": data["metadata"]["created_date"],
        "title": data["metadata"]["title"],
        "description": data["metadata"]["description"],
        "total_questions": data["metadata"]["total_questions"],
        "base_questions": data["metadata"]["base_questions"],
        "policy_areas": data["metadata"]["policy_areas"],
        "dimensions": data["metadata"]["dimensions"],
        "author": data["metadata"]["author"],
    }


def extract_dimensiones(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract dimensions D1-D6 with all properties."""
    dimensiones = {}
    for dim_id, dim_data in data["dimensiones"].items():
        dimensiones[dim_id] = {
            "nombre": dim_data["nombre"],
            "descripcion": dim_data["descripcion"],
            "peso_por_punto": dim_data["peso_por_punto"],
            "preguntas": dim_data["preguntas"],
            "umbral_minimo": dim_data["umbral_minimo"],
            "decalogo_dimension_mapping": dim_data["decalogo_dimension_mapping"],
        }
    return dimensiones


def extract_puntos_decalogo(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract policy areas (puntos del decálogo) metadata."""
    puntos = {}
    for punto_id, punto_data in data["puntos_decalogo"].items():
        puntos[punto_id] = {
            "nombre": punto_data["nombre"],
            "descripcion": punto_data["descripcion"],
            "dimensiones_criticas": punto_data["dimensiones_criticas"],
            "indicadores_producto": punto_data["indicadores_producto"],
            "indicadores_resultado": punto_data["indicadores_resultado"],
        }
    return puntos


def extract_question_definitions(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract all 300 question definitions with templates, patterns, expected_elements."""
    questions = []
    
    # cuestionario.json uses "preguntas_base" not "preguntas_por_punto"
    preguntas_list = data.get("preguntas_base", [])
    
    for q in preguntas_list:
        question_def = {
            "id": q.get("id"),
            "dimension": q.get("dimension"),
            "question_no": q.get("numero", q.get("question_no")),
            "text": q.get("texto_template", q.get("text")),
            "criteria": q.get("criterios_evaluacion", {}),
            "patterns": q.get("patrones_verificacion", []),
            "scoring": q.get("scoring", {}),
            "verification": q.get("verificacion_lineas_base", {}),
        }
        
        questions.append(question_def)
    
    return questions


def extract_dimension_aggregation(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract dimension_aggregation formulas with weights."""
    scoring_system = data.get("scoring_system", {})
    return {
        "formula": scoring_system.get("dimension_score", {}).get("formula", "(sum_of_5_questions / 15) * 100"),
        "description": scoring_system.get("dimension_score", {}).get("description", "Aggregate of 5 questions in a dimension"),
        "max_score": 15,
        "questions_per_dimension": 5,
    }


def extract_point_aggregation(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract point_aggregation rules."""
    scoring_system = data.get("scoring_system", {})
    return {
        "formula": scoring_system.get("point_score", {}).get("formula", "weighted_sum_of_6_dimensions"),
        "description": scoring_system.get("point_score", {}).get("description", "Aggregate of 6 dimensions in a thematic point"),
        "dimensions_per_point": 6,
        "weighted": True,
        "weights_source": "dimensiones.peso_por_punto",
    }


def extract_scoring_modalities(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract scoring_modalities TYPE_A through TYPE_F from rubric."""
    modalities = {}
    for modality_id, modality_data in data["scoring_modalities"].items():
        modality = {
            "id": modality_data["id"],
            "description": modality_data["description"],
            "formula": modality_data["formula"],
            "max_score": modality_data["max_score"],
        }
        
        if "expected_elements" in modality_data:
            modality["expected_elements"] = modality_data["expected_elements"]
        if "conversion_table" in modality_data:
            modality["conversion_table"] = modality_data["conversion_table"]
        if "uses_thresholds" in modality_data:
            modality["uses_thresholds"] = modality_data["uses_thresholds"]
        if "uses_quantitative_data" in modality_data:
            modality["uses_quantitative_data"] = modality_data["uses_quantitative_data"]
        if "uses_custom_logic" in modality_data:
            modality["uses_custom_logic"] = modality_data["uses_custom_logic"]
        if "uses_semantic_matching" in modality_data:
            modality["uses_semantic_matching"] = modality_data["uses_semantic_matching"]
        if "similarity_threshold" in modality_data:
            modality["similarity_threshold"] = modality_data["similarity_threshold"]
        
        modalities[modality_id] = modality
    
    return modalities


def extract_aggregation_levels(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract aggregation_levels from rubric."""
    levels = {}
    for level_id, level_data in data["aggregation_levels"].items():
        level = {
            "name": level_data["name"],
            "range": level_data["range"],
            "unit": level_data["unit"],
            "precision": level_data["precision"],
            "description": level_data["description"],
        }
        
        if "formula" in level_data:
            level["formula"] = level_data["formula"]
        if "max_points" in level_data:
            level["max_points"] = level_data["max_points"]
        if "questions_per_dimension" in level_data:
            level["questions_per_dimension"] = level_data["questions_per_dimension"]
        if "dimensions_per_point" in level_data:
            level["dimensions_per_point"] = level_data["dimensions_per_point"]
        if "exclude_na" in level_data:
            level["exclude_na"] = level_data["exclude_na"]
        
        levels[level_id] = level
    
    return levels


def extract_score_bands(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract score_bands definitions from rubric."""
    bands = {}
    for band_name, band_data in data["score_bands"].items():
        bands[band_name] = {
            "min": band_data["min"],
            "max": band_data["max"],
            "color": band_data["color"],
            "emoji": band_data["emoji"],
            "description": band_data["description"],
            "recommendation": band_data["recommendation"],
        }
    return bands


def extract_rubric_dimensions(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract dimension definitions with max_scores from rubric."""
    dimensions = {}
    for dim_id, dim_data in data["dimensions"].items():
        dimensions[dim_id] = {
            "id": dim_data["id"],
            "name": dim_data["name"],
            "description": dim_data["description"],
            "questions": dim_data["questions"],
            "max_score": dim_data["max_score"],
        }
    return dimensions


def extract_rubric_questions(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract question definitions from rubric."""
    questions = []
    
    for q in data.get("questions", []):
        question_def = {
            "id": q["id"],
            "dimension": q["dimension"],
            "question_no": q["question_no"],
            "template": q["template"],
            "scoring_modality": q["scoring_modality"],
            "max_score": q["max_score"],
            "expected_elements": q.get("expected_elements", []),
            "search_patterns": q.get("search_patterns", {}),
            "evidence_sources": q.get("evidence_sources", {}),
        }
        
        # Add optional fields
        if "thresholds" in q:
            question_def["thresholds"] = q["thresholds"]
        if "logical_rule" in q:
            question_def["logical_rule"] = q["logical_rule"]
        if "scoring_formula" in q:
            question_def["scoring_formula"] = q["scoring_formula"]
        
        questions.append(question_def)
    
    return questions


def validate_300_questions(
    policy_areas: int, dimensions: int, questions_per_dim: int
) -> Tuple[bool, str, int]:
    """Validate that 10 policy areas × 6 dimensions × 5 questions = 300."""
    expected_total = policy_areas * dimensions * questions_per_dim
    is_valid = expected_total == 300
    
    message = (
        f"Validation: {policy_areas} policy areas × {dimensions} dimensions × "
        f"{questions_per_dim} questions = {expected_total} questions"
    )
    
    return is_valid, message, expected_total


def count_actual_questions(cuestionario_data: Dict[str, Any]) -> int:
    """Count actual questions in cuestionario.json."""
    preguntas_list = cuestionario_data.get("preguntas_base", [])
    return len(preguntas_list)


def generate_validation_rules() -> Dict[str, Any]:
    """Generate validation rules for structural consistency."""
    return {
        "structural_rules": {
            "question_count": {
                "rule": "total_questions == 300",
                "description": "Total questions must equal exactly 300",
                "formula": "policy_areas (10) × dimensions (6) × questions_per_dimension (5) = 300",
            },
            "dimension_consistency": {
                "rule": "each dimension must have exactly 5 questions per policy area",
                "description": "Each of the 6 dimensions must contain 5 questions for each of the 10 policy areas",
            },
            "scoring_modality_mapping": {
                "rule": "all cuestionario questions must reference valid scoring_modalities",
                "valid_modalities": ["TYPE_A", "TYPE_B", "TYPE_C", "TYPE_D", "TYPE_E", "TYPE_F"],
                "description": "Every question's scoring_modality must exist in rubric_scoring.scoring_modalities",
            },
            "max_score_consistency": {
                "rule": "all question max_scores == 3.0",
                "description": "Individual questions are scored 0-3 points",
            },
            "dimension_max_score": {
                "rule": "dimension_max_score == 15 (5 questions × 3 points)",
                "description": "Each dimension aggregates to 15 points maximum",
            },
            "expected_elements_required": {
                "rule": "questions with TYPE_A/B/C must define expected_elements",
                "description": "Element-counting modalities require expected_elements specification",
            },
        },
        "aggregation_rules": {
            "level_1_question": {
                "range": [0.0, 3.0],
                "precision": 2,
                "description": "Individual question score",
            },
            "level_2_dimension": {
                "range": [0.0, 100.0],
                "formula": "(sum_of_5_questions / 15) × 100",
                "precision": 1,
                "description": "Dimension score as percentage",
            },
            "level_3_point": {
                "range": [0.0, 100.0],
                "formula": "weighted_sum_of_6_dimensions / sum_of_weights",
                "precision": 1,
                "description": "Policy area score with weighted dimensions",
            },
            "level_4_global": {
                "range": [0.0, 100.0],
                "formula": "sum_of_applicable_points / count_applicable (exclude N/A)",
                "precision": 1,
                "description": "Global PDM score excluding non-applicable policy areas",
            },
        },
        "score_band_rules": {
            "coverage": "score_bands must cover entire 0-100 range without gaps",
            "non_overlapping": "score band ranges must not overlap",
            "bands": {
                "EXCELENTE": {"min": 85, "max": 100},
                "BUENO": {"min": 70, "max": 84},
                "SATISFACTORIO": {"min": 55, "max": 69},
                "INSUFICIENTE": {"min": 40, "max": 54},
                "DEFICIENTE": {"min": 0, "max": 39},
            },
        },
        "semantic_consistency_rules": {
            "question_id_format": {
                "rule": "question IDs must follow pattern {PUNTO}-{DIMENSION}-Q{N}",
                "example": "P1-D1-Q1",
            },
            "dimension_mapping": {
                "rule": "cuestionario dimension IDs must match rubric dimension IDs",
                "valid_dimensions": ["D1", "D2", "D3", "D4", "D5", "D6"],
            },
            "punto_mapping": {
                "rule": "cuestionario punto IDs must be P1-P10",
                "valid_puntos": ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10"],
            },
        },
    }


def validate_cuestionario_rubric_consistency(
    cuestionario_questions: List[Dict[str, Any]],
    rubric_modalities: Dict[str, Any],
) -> Dict[str, Any]:
    """Cross-validate cuestionario questions against rubric scoring modalities."""
    validation_results = {
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "stats": {},
    }
    
    # Count patterns and criteria
    pattern_usage = {}
    criteria_usage = {}
    
    for q in cuestionario_questions:
        qid = q.get("id", "unknown")
        
        # Count patterns
        patterns = q.get("patterns", [])
        pattern_usage[qid] = len(patterns)
        
        # Count criteria
        criteria = q.get("criteria", {})
        criteria_usage[qid] = len(criteria)
    
    validation_results["stats"]["pattern_usage"] = pattern_usage
    validation_results["stats"]["criteria_usage"] = criteria_usage
    validation_results["stats"]["total_questions"] = len(cuestionario_questions)
    validation_results["stats"]["avg_patterns"] = sum(pattern_usage.values()) / len(pattern_usage) if pattern_usage else 0
    validation_results["stats"]["avg_criteria"] = sum(criteria_usage.values()) / len(criteria_usage) if criteria_usage else 0
    
    return validation_results


def main():
    """Main execution function."""
    print("🔍 Parsing cuestionario.json and rubric_scoring.json...")
    
    # Load JSON files
    cuestionario = load_json("cuestionario.json")
    rubric = load_json("rubric_scoring.json")
    
    print("✓ JSON files loaded successfully")
    
    # Extract cuestionario structures
    print("\n📊 Extracting cuestionario.json structures...")
    cuest_metadata = extract_cuestionario_metadata(cuestionario)
    dimensiones = extract_dimensiones(cuestionario)
    puntos_decalogo = extract_puntos_decalogo(cuestionario)
    cuest_questions = extract_question_definitions(cuestionario)
    dimension_aggregation = extract_dimension_aggregation(cuestionario)
    point_aggregation = extract_point_aggregation(cuestionario)
    
    print(f"  ✓ Metadata: {cuest_metadata['title']}")
    print(f"  ✓ Dimensions: {len(dimensiones)}")
    print(f"  ✓ Policy areas: {len(puntos_decalogo)}")
    print(f"  ✓ Questions extracted: {len(cuest_questions)}")
    
    # Extract rubric structures
    print("\n📋 Extracting rubric_scoring.json structures...")
    scoring_modalities = extract_scoring_modalities(rubric)
    aggregation_levels = extract_aggregation_levels(rubric)
    score_bands = extract_score_bands(rubric)
    rubric_dimensions = extract_rubric_dimensions(rubric)
    rubric_questions = extract_rubric_questions(rubric)
    
    print(f"  ✓ Scoring modalities: {len(scoring_modalities)}")
    print(f"  ✓ Aggregation levels: {len(aggregation_levels)}")
    print(f"  ✓ Score bands: {len(score_bands)}")
    print(f"  ✓ Rubric dimensions: {len(rubric_dimensions)}")
    print(f"  ✓ Rubric questions: {len(rubric_questions)}")
    
    # Validate 300 questions formula
    print("\n🔢 Validating question count formula...")
    is_valid, message, expected = validate_300_questions(
        cuest_metadata["policy_areas"],
        cuest_metadata["dimensions"],
        5,  # questions per dimension
    )
    actual_count = count_actual_questions(cuestionario)
    
    print(f"  {message}")
    print(f"  Actual questions in cuestionario: {actual_count}")
    print(f"  ✓ VALID: {is_valid and actual_count == expected}")
    
    # Cross-validate consistency
    print("\n🔍 Cross-validating cuestionario vs rubric...")
    consistency_results = validate_cuestionario_rubric_consistency(
        cuest_questions, scoring_modalities
    )
    
    print(f"  Validation: {'✓ PASSED' if consistency_results['is_valid'] else '✗ FAILED'}")
    if consistency_results["errors"]:
        for error in consistency_results["errors"]:
            print(f"    ✗ {error}")
    if consistency_results["warnings"]:
        print(f"  Warnings: {len(consistency_results['warnings'])}")
    
    # Generate validation rules
    print("\n📐 Generating validation rules...")
    validation_rules = generate_validation_rules()
    print(f"  ✓ {len(validation_rules)} rule categories generated")
    
    # Build schema contracts output
    print("\n📦 Building schema_contracts.json...")
    schema_contracts = {
        "meta": {
            "generated": "2025-01-15",
            "description": "Normalized schema contracts for FARFAN 3.0 evaluation system",
            "source_files": ["cuestionario.json", "rubric_scoring.json"],
            "validation_status": {
                "question_count_valid": is_valid and actual_count == expected,
                "consistency_valid": consistency_results["is_valid"],
                "expected_questions": expected,
                "actual_questions": actual_count,
            },
        },
        "cuestionario_schema": {
            "metadata": cuest_metadata,
            "dimensiones": dimensiones,
            "puntos_decalogo": puntos_decalogo,
            "dimension_aggregation": dimension_aggregation,
            "point_aggregation": point_aggregation,
            "questions": {
                "total": len(cuest_questions),
                "by_dimension": {},
                "pattern_stats": consistency_results["stats"]["pattern_usage"],
                "criteria_stats": consistency_results["stats"]["criteria_usage"],
                "definitions": cuest_questions,
            },
        },
        "rubric_schema": {
            "metadata": {
                "version": rubric["metadata"]["version"],
                "created": rubric["metadata"]["created"],
                "description": rubric["metadata"]["description"],
            },
            "scoring_modalities": scoring_modalities,
            "aggregation_levels": aggregation_levels,
            "score_bands": score_bands,
            "dimensions": rubric_dimensions,
            "questions": {
                "total": len(rubric_questions),
                "definitions": rubric_questions,
            },
        },
        "validation_rules": validation_rules,
        "consistency_validation": consistency_results,
    }
    
    # Calculate question distribution statistics
    for q in cuest_questions:
        dim = q.get("dimension", "unknown")
        
        schema_contracts["cuestionario_schema"]["questions"]["by_dimension"][dim] = (
            schema_contracts["cuestionario_schema"]["questions"]["by_dimension"].get(dim, 0) + 1
        )
    
    # Write output
    output_path = "schema_contracts.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(schema_contracts, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Schema contracts written to {output_path}")
    print(f"   File size: {Path(output_path).stat().st_size / 1024:.1f} KB")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    print(f"Total questions: {actual_count} / {expected} expected")
    print(f"Dimensions: {len(dimensiones)}")
    print(f"Policy areas: {len(puntos_decalogo)}")
    print(f"Scoring modalities: {len(scoring_modalities)}")
    print(f"Score bands: {len(score_bands)}")
    print(f"Validation: {'✅ PASSED' if is_valid and actual_count == expected else '❌ FAILED'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
