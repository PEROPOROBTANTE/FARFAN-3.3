# Cuestionario.json Integration Guide

## Overview

The `cuestionario.json` file is the **core evaluation framework** for the FARFAN 3.0 system. It contains a comprehensive set of 300 questions that translate municipal development plan obligations into operational assessments. This document ensures **homogeneous evaluation** across all 170 development plans.

## Structure

### Top-Level Organization

```
cuestionario.json
├── metadata                    # Version, title, author information
├── dimensiones (D1-D6)        # 6 causal dimensions
├── puntos_decalogo (P1-P10)   # 10 policy areas from the Decálogo
├── preguntas_base (300)       # The 300 evaluation questions
├── common_failure_patterns    # Common issues to detect
├── scoring_system             # Rubric and scoring methodology
└── causal_glossary            # Definitions and verification patterns
```

### The 300 Questions

Questions are organized as:
- **10 Policy Points (P1-P10)** - Decálogo areas (gender rights, violence prevention, environment, etc.)
- **6 Dimensions (D1-D6)** - Causal chain levels (inputs, activities, products, results, impacts, causality)
- **5 Questions per dimension** - Specific evaluation criteria

**Total**: 10 policy points × 6 dimensions × 5 questions = **300 questions**

### Question Organization

Questions are sequentially ordered in `preguntas_base`:
```
Positions 0-29:    P1 (Gender Rights) - D1-D6, Q1-Q5
Positions 30-59:   P2 (Violence Prevention) - D1-D6, Q1-Q5  
Positions 60-89:   P3 (Environment) - D1-D6, Q1-Q5
...
Positions 270-299: P10 (Migration) - D1-D6, Q1-Q5
```

## Key Components

### 1. Question Structure

Each question in `preguntas_base` contains:

```json
{
  "id": "D1-Q1",
  "dimension": "D1",
  "numero": 1,
  "texto_template": "¿El diagnóstico presenta datos numéricos...",
  "criterios_evaluacion": {
    "indicadores_cuantitativos_minimos": 3,
    "fuentes_oficiales_minimas": 2
  },
  "patrones_verificacion": [
    "línea base|año base|situación inicial",
    "fuente:|según|reportado por",
    "DANE|Medicina Legal|Fiscalía"
  ],
  "scoring": {
    "excelente": { "min_score": 0.85 },
    "bueno": { "min_score": 0.70 },
    "aceptable": { "min_score": 0.55 },
    "insuficiente": { "min_score": 0.0 }
  }
}
```

### 2. Verification Patterns

Verification patterns are **regex patterns** or keyword groups used to scan development plans for evidence. They ensure:
- Objective, repeatable evaluations
- Evidence-based scoring
- Consistency across all 170 plans

Example patterns for dimension D1-Q1 (Baseline identification):
- `línea base|año base|situación inicial|diagnóstico de género`
- `fuente:|según|reportado por|con datos de`
- `DANE|Medicina Legal|Fiscalía|Policía Nacional`
- `\d+(\.\d+)?\s*%` (percentage detection)

### 3. Scoring Rubrics

Each question has a standardized 4-level rubric:
- **EXCELENTE** (0.85-1.00): Exceeds expectations
- **BUENO** (0.70-0.84): Meets expectations
- **ACEPTABLE** (0.55-0.69): Minimally acceptable
- **INSUFICIENTE** (0.00-0.54): Does not meet standards

## Integration Points

### 1. Question Router (`orchestrator/question_router.py`)

**Purpose**: Loads questions from cuestionario.json and routes them to appropriate processing modules.

**Key Methods**:
```python
QuestionRouter(cuestionario_path, validate=True)
- Loads all 300 questions from cuestionario.json
- Maps questions to policy points based on position
- Extracts verification patterns and rubrics
- Validates the loaded structure
```

**Usage**:
```python
from orchestrator import QuestionRouter

router = QuestionRouter()  # Automatically validates
question = router.get_question("P1-D1-Q1")
patterns = question.verification_patterns  # From cuestionario.json
rubric = question.rubric_levels  # Scoring thresholds
```

### 2. Report Assembler (`orchestrator/report_assembly.py`)

**Purpose**: Uses verification patterns to score plans and generate evidence-based reports.

**Key Methods**:
```python
def _match_verification_patterns(patterns, text):
    """Match regex patterns from cuestionario.json against plan text"""
    # Returns count of matched patterns and list of matches

def _calculate_question_score(question, evidence, plan_text):
    """Score using patterns from cuestionario.json"""
    # 1. Match verification patterns against plan
    # 2. Calculate base score from pattern matches
    # 3. Adjust by module confidence
    # 4. Apply rubric thresholds
```

**Pattern Matching**:
- Each pattern from `patrones_verificacion` is tested against the plan text
- Pattern match rate (matched/total) forms the base score
- Combined with module confidence for final score
- Results stored in evidence for transparency

### 3. Validator (`orchestrator/cuestionario_validator.py`)

**Purpose**: Ensures cuestionario.json is properly loaded and enforced across all evaluations.

**Validation Checks**:
1. **Question Coverage**: All 300 questions loaded
2. **Policy Point Mapping**: Each P1-P10 has exactly 30 questions
3. **Verification Patterns**: All questions have patterns loaded
4. **Scoring Rubrics**: All questions have complete 4-level rubrics
5. **Dimension Coverage**: Each D1-D6 has exactly 50 questions

**Usage**:
```python
from orchestrator.cuestionario_validator import CuestionarioValidator

validator = CuestionarioValidator('cuestionario.json')
is_valid, results = validator.run_full_validation(questions)

if not is_valid:
    # Evaluation will not be homogeneous!
    logger.error("Cuestionario validation FAILED")
```

## Ensuring Homogeneous Evaluation

The cuestionario.json ensures consistent evaluation through:

### 1. Standardized Questions
- Same 300 questions applied to all 170 plans
- No ad-hoc or improvised questions
- Policy-specific but structurally identical

### 2. Objective Verification
- Regex-based pattern matching
- No subjective interpretation
- Repeatable and auditable

### 3. Consistent Scoring
- Same rubric thresholds for all plans
- Evidence-based scoring (pattern matches)
- Transparent scoring methodology

### 4. Validation Enforcement
- Automatic validation at startup
- Fails loudly if cuestionario.json is incomplete
- Logs validation status for audit trail

### 5. Version Control
- cuestionario.json has version metadata
- Changes are tracked
- Version compatibility can be verified

## Testing

Run the comprehensive test suite:

```bash
cd /home/runner/work/FARFAN-3.0/FARFAN-3.0
python3 test_cuestionario_integration.py
```

**Tests Include**:
1. JSON syntax validation
2. Structure validation (dimensions, policy points, questions)
3. Question organization validation
4. Question content validation (patterns, rubrics)
5. QuestionRouter integration validation

**Expected Output**:
```
✓ cuestionario.json is valid JSON
✓ All required structure elements present
✓ 300 questions properly organized (10 policy points × 30 questions)
✓ Questions have verification patterns and scoring criteria
✓ All 6 dimensions properly covered (50 questions each)
✓ ALL TESTS PASSED
```

## Troubleshooting

### Issue: Questions not loading correctly

**Symptoms**: QuestionRouter loads fewer than 300 questions

**Solution**:
1. Verify cuestionario.json syntax: `python3 -m json.tool cuestionario.json`
2. Check validation results: Run `test_cuestionario_integration.py`
3. Review logs for parsing errors

### Issue: Verification patterns not matching

**Symptoms**: All questions score 0.0 or very low scores

**Solution**:
1. Verify patterns are loaded: `question.verification_patterns`
2. Check pattern syntax (should be valid regex)
3. Test individual patterns against sample text
4. Review `evidence["pattern_matching"]` in results

### Issue: Inconsistent scoring across plans

**Symptoms**: Same content gets different scores for different plans

**Solution**:
1. Check cuestionario.json hasn't changed between evaluations
2. Verify validation passes for both evaluations
3. Compare module confidence scores
4. Review pattern matching results in evidence

## Maintenance

### Updating Questions

1. Edit `cuestionario.json` directly
2. Increment version in `metadata.version`
3. Run test suite: `python3 test_cuestionario_integration.py`
4. Commit changes with clear description
5. Re-run validation on all systems

### Adding New Policy Points

1. Add new entry to `puntos_decalogo` (e.g., P11)
2. Create 30 new questions (6 dimensions × 5 questions)
3. Add to `preguntas_base` array
4. Update `metadata.total_questions`
5. Update validator expected counts

### Modifying Verification Patterns

1. Locate question in `preguntas_base`
2. Update `patrones_verificacion` array
3. Test new patterns against sample plans
4. Document pattern purpose in comments
5. Run validation suite

## Best Practices

1. **Never bypass cuestionario.json** - Always use the structured questions
2. **Always run validation** - Ensure homogeneous evaluation
3. **Log pattern matches** - Include in evidence for transparency
4. **Version control** - Track all changes to cuestionario.json
5. **Test thoroughly** - Verify patterns work before production use
6. **Document changes** - Explain why patterns were modified
7. **Review regularly** - Update patterns based on evaluation experience

## References

- `cuestionario.json` - The master evaluation framework
- `orchestrator/question_router.py` - Question loading and routing
- `orchestrator/report_assembly.py` - Scoring and evidence generation
- `orchestrator/cuestionario_validator.py` - Validation framework
- `test_cuestionario_integration.py` - Comprehensive test suite

---

**Version**: 2.0.0  
**Last Updated**: 2025-10-16  
**Author**: JCRR / FARFAN 3.0 Team
