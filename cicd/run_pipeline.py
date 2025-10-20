#!/usr/bin/env python3
"""
CI/CD Pipeline Runner
=====================

Execute validation gate pipeline and report results.
"""

import sys
import json
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cicd.validation_gates import ValidationGatePipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 80)
    logger.info("FARFAN CI/CD Validation Pipeline")
    logger.info("=" * 80)
    
    pipeline = ValidationGatePipeline()
    results = pipeline.run_all()
    
    print("\n" + "=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)
    print(f"Status: {'✓ PASSED' if results['success'] else '✗ FAILED'}")
    print(f"Gates Passed: {results['passed_gates']}/{results['total_gates']}")
    print(f"Execution Time: {results['execution_time']:.2f}s")
    print()
    
    for result in results['results']:
        status_icon = "✓" if result['passed'] else "✗"
        print(f"{status_icon} {result['gate_name']}: {result['status']}")
        
        if result['errors']:
            for error in result['errors']:
                print(f"  ERROR: {error}")
        
        if result['warnings']:
            for warning in result['warnings']:
                print(f"  WARNING: {warning}")
        
        if result.get('metrics', {}).get('remediation_suggestions'):
            print("  REMEDIATION:")
            for suggestion in result['metrics']['remediation_suggestions']:
                print(f"    - {suggestion['suggested_fix']}")
                if suggestion.get('command'):
                    print(f"      Command: {suggestion['command']}")
        print()
    
    output_path = Path("validation_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_path}")
    print("=" * 80)
    
    sys.exit(0 if results['success'] else 1)


if __name__ == "__main__":
    main()
