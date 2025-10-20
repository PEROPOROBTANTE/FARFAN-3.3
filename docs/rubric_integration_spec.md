# Rubric Integration Specification

## 1. Verification Matrix

### TYPE_A: Element Counting
- JSON Formula: (elements_found / 4) * 3
- Implementation: (found_count / len(required_elements)) * 3.0
- Status: ✅ CORRECT

### TYPE_B: Weighted Sum
- JSON Formula: weighted sum approach
- Implementation: (weighted_score / total_weight) * 3.0
- Status: ✅ CORRECT

### TYPE_C: Quality Assessment
- JSON Formula: (elements_found / 2) * 3
- Implementation: avg_confidence * 3.0
- Status: ⚠️ DIVERGES (uses confidence vs element count)

### TYPE_D: Numerical Thresholds
- JSON Formula: f(ratio) with thresholds
- Implementation: (met_count / total_thresholds) * 3.0
- Status: ✅ CORRECT

### TYPE_E: Logical Rules
- Status: ⚠️ NOT IMPLEMENTED

### TYPE_F: Semantic Matching
- Status: ⚠️ NOT IMPLEMENTED

## 2. Three-Tier Architecture

### MICRO (line 168): generate_micro_answer()
- Validates preconditions → Applies scoring modality → Extracts evidence
- Output: Score 0-3, qualitative note, evidence, confidence

### MESO (line 716): generate_meso_cluster()
- Aggregates dimension scores via (score/3.0)*100
- Identifies strengths (>=70%) and weaknesses (<55%)
- Output: Dimension scores (%), avg_score, recommendations

### MACRO (line 977): generate_macro_convergence()
- Computes overall score, dimension/policy convergence
- Applies band classification:
  - EXCELENTE: 85-100
  - BUENO: 70-84
  - SATISFACTORIO: 55-69
  - INSUFICIENTE: 40-54
  - DEFICIENTE: 0-39

## 3. Timing Diagram

ExecutionChoreographer
  -> For each question (Q1-Q300)
     -> TIER 1: MICRO (generate_micro_answer)
        1. Validate preconditions
        2. Apply scoring modality
        3. Extract evidence
        -> Output: MicroLevelAnswer
     -> TIER 2: MESO (generate_meso_cluster)
        1. Validate micro answers
        2. Calculate dimension scores
        3. Identify strengths/weaknesses
        -> Output: MesoLevelCluster
     -> TIER 3: MACRO (generate_macro_convergence)
        1. Validate all inputs
        2. Calculate overall score
        3. Apply band classification
        -> Output: MacroLevelConvergence

## 4. Precondition Checklist

### MICRO Level Preconditions:
- expected_elements defined
- execution_results populated with adapter data
- plan_text available for context
- evidence_excerpts extractable
- pattern_matches extractable

### MESO Level Preconditions:
- micro_answers non-empty list
- scores in valid range [0, 3]
- dimension metadata present

### MACRO Level Preconditions:
- all_micro_answers complete
- all_meso_clusters computed
- dimension coverage D1-D6 present
- rubric_levels defined for classification

## 5. Summary

### Verified Components:
- TYPE_A, TYPE_B, TYPE_D scoring: Correct
- MICRO->MESO->MACRO pipeline: Properly implemented
- Score band classification: Matches JSON exactly
- Dimension aggregation formulas: Match JSON

### Identified Gaps:
- TYPE_C: Diverges from spec (confidence vs element count)
- TYPE_E: Not implemented
- TYPE_F: Not implemented

### Recommendations:
1. Implement TYPE_E with conditional logic evaluation
2. Implement TYPE_F with semantic matching
3. Clarify TYPE_C approach or update JSON spec
4. Consider dynamic JSON loading vs hardcoded values
5. Add unit tests for conversion table verification
