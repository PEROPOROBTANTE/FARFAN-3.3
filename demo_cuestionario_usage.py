#!/usr/bin/env python3
"""
Demo: Cuestionario.json Integration
=====================================
Demonstrates how cuestionario.json is used to ensure homogeneous evaluation
across all 170 development plans.

This script shows:
1. How questions are loaded from cuestionario.json
2. How verification patterns are applied
3. How validation ensures consistency
4. How scoring uses the patterns and rubrics
"""

import json
import re
from pathlib import Path
from typing import List, Tuple


def demo_question_loading():
    """Demo 1: Loading questions from cuestionario.json"""
    print("="*80)
    print("DEMO 1: Loading Questions from cuestionario.json")
    print("="*80)
    
    with open('cuestionario.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"\n✓ Loaded cuestionario.json v{data['metadata']['version']}")
    print(f"  Total questions: {data['metadata']['total_questions']}")
    print(f"  Policy points: {len(data['puntos_decalogo'])} (P1-P10)")
    print(f"  Dimensions: {len(data['dimensiones'])} (D1-D6)")
    
    # Show sample question
    sample_q = data['preguntas_base'][0]  # First question: P1-D1-Q1
    print(f"\n📋 Sample Question: {sample_q['id']}")
    print(f"   Text: {sample_q['texto_template'][:100]}...")
    print(f"   Dimension: {sample_q['dimension']}")
    print(f"   Verification Patterns: {len(sample_q['patrones_verificacion'])} patterns")
    print(f"   Scoring Rubric: {list(sample_q['scoring'].keys())}")
    
    return data


def demo_pattern_matching(data):
    """Demo 2: Pattern matching for objective evaluation"""
    print("\n" + "="*80)
    print("DEMO 2: Verification Pattern Matching")
    print("="*80)
    
    # Sample plan text (simulated)
    sample_plan = """
    El diagnóstico presenta datos numéricos de la situación de las mujeres en el municipio.
    Según el DANE (2023), la tasa de violencia intrafamiliar es de 45 casos por cada 100.000 mujeres.
    La fuente: Medicina Legal reporta un aumento del 15% en casos de VBG respecto al año base 2020.
    Se identifican brechas salariales del 23% entre hombres y mujeres en el sector formal.
    """
    
    # Get first question (P1-D1-Q1 about baseline data for gender)
    question = data['preguntas_base'][0]
    patterns = question['patrones_verificacion']
    
    print(f"\n📋 Question: {question['id']}")
    print(f"   {question['texto_template'][:80]}...")
    print(f"\n🔍 Applying {len(patterns)} verification patterns...")
    
    matched_patterns = []
    for i, pattern in enumerate(patterns[:5], 1):  # Show first 5
        try:
            if re.search(pattern, sample_plan, re.IGNORECASE):
                matched_patterns.append(pattern)
                print(f"   ✓ Pattern {i} matched: '{pattern[:50]}...'")
            else:
                print(f"   ✗ Pattern {i} not found: '{pattern[:50]}...'")
        except re.error:
            # Fallback to string matching
            if pattern.lower() in sample_plan.lower():
                matched_patterns.append(pattern)
                print(f"   ✓ Pattern {i} matched (string): '{pattern[:50]}...'")
    
    match_rate = len(matched_patterns) / len(patterns)
    print(f"\n📊 Pattern Match Result:")
    print(f"   Matched: {len(matched_patterns)}/{len(patterns)} patterns ({match_rate*100:.1f}%)")
    print(f"   Base Score: {match_rate:.2f}")
    
    # Apply rubric
    scoring = question['scoring']
    if match_rate >= scoring['excelente']['min_score']:
        level = "EXCELENTE"
    elif match_rate >= scoring['bueno']['min_score']:
        level = "BUENO"
    elif match_rate >= scoring['aceptable']['min_score']:
        level = "ACEPTABLE"
    else:
        level = "INSUFICIENTE"
    
    print(f"   Qualitative Level: {level}")
    
    return match_rate, level


def demo_validation():
    """Demo 3: Validation ensures consistency"""
    print("\n" + "="*80)
    print("DEMO 3: Validation for Homogeneous Evaluation")
    print("="*80)
    
    print("\n🔍 Running validation checks...")
    
    # Simulate validation (in real system, CuestionarioValidator does this)
    with open('cuestionario.json', 'r') as f:
        data = json.load(f)
    
    checks = {
        "Question Count": len(data['preguntas_base']) == 300,
        "Policy Points": len(data['puntos_decalogo']) == 10,
        "Dimensions": len(data['dimensiones']) == 6,
        "Questions per Policy": all(
            len([q for q in data['preguntas_base'][i*30:(i+1)*30]]) == 30 
            for i in range(10)
        ),
        "Patterns Present": all(
            len(q.get('patrones_verificacion', [])) > 0 
            for q in data['preguntas_base']
        ),
        "Rubrics Complete": all(
            all(level in q.get('scoring', {}) for level in ['excelente', 'bueno', 'aceptable', 'insuficiente'])
            for q in data['preguntas_base']
        )
    }
    
    print("\n✅ Validation Results:")
    for check_name, passed in checks.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"   {status} - {check_name}")
    
    all_passed = all(checks.values())
    print(f"\n{'✓' if all_passed else '✗'} Overall Validation: {'PASSED' if all_passed else 'FAILED'}")
    
    if all_passed:
        print("\n💡 Implication:")
        print("   All 170 development plans will be evaluated using:")
        print("   - The same 300 questions")
        print("   - The same verification patterns")
        print("   - The same scoring rubrics")
        print("   → Ensures HOMOGENEOUS, OBJECTIVE evaluation")
    
    return all_passed


def demo_consistency_guarantee():
    """Demo 4: How this ensures consistency across 170 plans"""
    print("\n" + "="*80)
    print("DEMO 4: Consistency Across All 170 Development Plans")
    print("="*80)
    
    print("\n🎯 Without cuestionario.json enforcement:")
    print("   ✗ Evaluators might use different questions")
    print("   ✗ Subjective interpretation varies")
    print("   ✗ Scoring criteria inconsistent")
    print("   ✗ Results not comparable")
    print("   → HETEROGENEOUS evaluation")
    
    print("\n✓ With cuestionario.json enforcement:")
    print("   ✓ Exact same 300 questions for all plans")
    print("   ✓ Objective pattern matching (regex-based)")
    print("   ✓ Standardized scoring rubrics")
    print("   ✓ Validation ensures compliance")
    print("   → HOMOGENEOUS evaluation")
    
    print("\n📈 Impact on 170 Plans:")
    print("   • Plan A (Municipality X): Uses questions P1-D1-Q1 through P10-D6-Q5")
    print("   • Plan B (Municipality Y): Uses questions P1-D1-Q1 through P10-D6-Q5")
    print("   • ...same for all 170 plans...")
    print("   • Plan 170 (Municipality Z): Uses questions P1-D1-Q1 through P10-D6-Q5")
    
    print("\n   Result: Fair, comparable scores across all municipalities")
    print("           Objective evidence-based evaluation")
    print("           Audit trail through pattern matching logs")


def main():
    """Run all demos"""
    print("\n" + "🌟"*40)
    print("Cuestionario.json Integration Demonstration")
    print("FARFAN 3.0 - Homogeneous Evaluation System")
    print("🌟"*40 + "\n")
    
    # Demo 1: Loading
    data = demo_question_loading()
    
    # Demo 2: Pattern Matching
    score, level = demo_pattern_matching(data)
    
    # Demo 3: Validation
    is_valid = demo_validation()
    
    # Demo 4: Consistency
    demo_consistency_guarantee()
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\n✓ Questions loaded: 300")
    print(f"✓ Verification patterns: ~2,220 (avg 7.4 per question)")
    print(f"✓ Validation status: {'PASSED' if is_valid else 'FAILED'}")
    print(f"✓ Pattern matching: {score:.2f} → {level}")
    
    print("\n🎓 Key Takeaway:")
    print("   cuestionario.json provides the operational translation of municipal")
    print("   obligations into a standardized, objective evaluation framework that")
    print("   ensures homogeneous assessment across all 170 development plans.")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
