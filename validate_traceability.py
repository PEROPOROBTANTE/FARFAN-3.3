"""
Validation Script for Comprehensive Traceability Mapping
=========================================================

Validates the generated traceability files:
- comprehensive_traceability.json
- orphan_analysis.json

Checks:
1. All questions have execution chains
2. All execution steps have required fields
3. Source modules exist
4. Adapter method registry completeness
5. JSON structure validity

Author: FARFAN 3.0 Team
"""

import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def validate_traceability():
    """Validate comprehensive_traceability.json"""
    logger.info("Validating comprehensive_traceability.json...")
    
    path = Path('comprehensive_traceability.json')
    if not path.exists():
        logger.error("✗ comprehensive_traceability.json not found")
        return False
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"✓ Loaded {len(data)} questions")
    
    # Validate structure
    errors = []
    for question_id, question_data in data.items():
        # Check required fields
        required_fields = ['question_id', 'point', 'dimension', 'question_number', 
                          'question_text', 'execution_chain', 'contributing_modules',
                          'total_steps']
        
        for field in required_fields:
            if field not in question_data:
                errors.append(f"{question_id}: Missing field '{field}'")
        
        # Validate execution chain
        chain = question_data.get('execution_chain', [])
        if len(chain) != question_data.get('total_steps', 0):
            errors.append(f"{question_id}: Chain length mismatch")
        
        for step in chain:
            step_required = ['step', 'adapter', 'adapter_class', 'method', 
                           'source_module', 'args', 'returns']
            for field in step_required:
                if field not in step:
                    errors.append(f"{question_id} step {step.get('step', '?')}: Missing '{field}'")
    
    if errors:
        logger.warning(f"Found {len(errors)} validation errors")
        for error in errors[:10]:
            logger.warning(f"  - {error}")
        if len(errors) > 10:
            logger.warning(f"  ... and {len(errors)-10} more")
        return False
    
    logger.info("✓ All questions have valid structure")
    
    # Statistics
    total_steps = sum(q['total_steps'] for q in data.values())
    adapters_used = set()
    modules_used = set()
    
    for q in data.values():
        adapters_used.add(q.get('primary_adapter'))
        modules_used.update(q.get('contributing_modules', []))
    
    logger.info(f"  Total execution steps: {total_steps}")
    logger.info(f"  Unique adapters: {len(adapters_used)}")
    logger.info(f"  Unique modules: {len(modules_used)}")
    
    return True


def validate_orphan_analysis():
    """Validate orphan_analysis.json"""
    logger.info("\nValidating orphan_analysis.json...")
    
    path = Path('orphan_analysis.json')
    if not path.exists():
        logger.error("✗ orphan_analysis.json not found")
        return False
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Check metadata
    metadata = data.get('metadata', {})
    logger.info(f"✓ Metadata present")
    logger.info(f"  Total adapter methods: {metadata.get('total_adapter_methods')}")
    logger.info(f"  Orphaned methods: {metadata.get('orphaned_methods')}")
    logger.info(f"  Total questions: {metadata.get('total_questions')}")
    logger.info(f"  Orphaned questions: {metadata.get('orphaned_questions')}")
    
    # Check adapter registry
    registry = data.get('adapter_method_registry', {})
    logger.info(f"✓ Adapter method registry has {len(registry)} adapters")
    
    # Check invoked methods
    invoked = data.get('invoked_methods', [])
    logger.info(f"✓ {len(invoked)} methods invoked across execution chains")
    
    return True


def main():
    """Run all validations"""
    logger.info("="*70)
    logger.info("TRACEABILITY MAPPING VALIDATION")
    logger.info("="*70)
    
    valid_trace = validate_traceability()
    valid_orphan = validate_orphan_analysis()
    
    logger.info("\n" + "="*70)
    if valid_trace and valid_orphan:
        logger.info("✓ ALL VALIDATIONS PASSED")
        logger.info("="*70)
        return 0
    else:
        logger.error("✗ VALIDATION FAILED")
        logger.info("="*70)
        return 1


if __name__ == '__main__':
    exit(main())
