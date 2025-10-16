#!/usr/bin/env python3
"""
FARFAN 3.0 - Main Execution Script
====================================
Runs the complete policy analysis orchestrator

Usage:
    # Analyze single plan
    python run_farfan.py --plan path/to/plan.pdf

    # Analyze batch of plans
    python run_farfan.py --batch path/to/plans_directory/

    # Analyze 170 plans with 8 workers
    python run_farfan.py --batch plans/ --max-plans 170 --workers 8

    # Get system health
    python run_farfan.py --health
"""
import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from orchestrator import FARFANOrchestrator
from orchestrator.config import CONFIG

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler(CONFIG.logs_dir / f"farfan_{datetime.now():%Y%m%d_%H%M%S}.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="FARFAN 3.0 - Policy Analysis Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single plan analysis
  python run_farfan.py --plan plan.pdf

  # Batch analysis
  python run_farfan.py --batch plans_directory/

  # Analyze 170 plans
  python run_farfan.py --batch plans/ --max-plans 170

  # System health check
  python run_farfan.py --health
        """
    )

    parser.add_argument(
        '--plan',
        type=Path,
        help='Path to single plan document (PDF/TXT)'
    )

    parser.add_argument(
        '--batch',
        type=Path,
        help='Directory containing multiple plan documents'
    )

    parser.add_argument(
        '--output',
        type=Path,
        default=CONFIG.output_dir,
        help='Output directory for reports (default: ./output)'
    )

    parser.add_argument(
        '--max-plans',
        type=int,
        help='Maximum number of plans to process in batch mode'
    )

    parser.add_argument(
        '--workers',
        type=int,
        default=CONFIG.max_parallel_workers,
        help='Number of parallel workers (default: 4)'
    )

    parser.add_argument(
        '--health',
        action='store_true',
        help='Display system health status and exit'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )

    args = parser.parse_args()

    # Set debug level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

    # Initialize orchestrator
    logger.info("="*80)
    logger.info("FARFAN 3.0 - Policy Analysis Orchestrator")
    logger.info("World's First Causal Mechanism Analysis System")
    logger.info("="*80)

    orchestrator = FARFANOrchestrator()

    # Health check mode
    if args.health:
        logger.info("System Health Check")
        health = orchestrator.get_system_health()

        print("\n" + "="*80)
        print("SYSTEM HEALTH STATUS")
        print("="*80)

        cb_health = health["circuit_breaker"]
        print(f"\nCircuit Breaker: {cb_health['status'].upper()}")
        print(f"  Available Modules: {cb_health['available_modules']}/{cb_health['total_modules']}")
        print(f"  Overall Success Rate: {cb_health['overall_success_rate']:.1%}")

        print(f"\nExecution Statistics:")
        stats = health["execution_stats"]
        print(f"  Plans Processed: {stats['total_plans_processed']}")
        print(f"  Questions Answered: {stats['total_questions_answered']}")
        print(f"  Total Execution Time: {stats['total_execution_time']:.2f}s")

        print("\nModule Performance:")
        for module, perf in stats['module_performance'].items():
            success_rate = perf['successes'] / perf['calls'] if perf['calls'] > 0 else 0.0
            print(f"  {module:30s}: {perf['calls']:4d} calls, {success_rate:5.1%} success")

        print("\n" + "="*80)
        return 0

    # Single plan mode
    if args.plan:
        if not args.plan.exists():
            logger.error(f"Plan file not found: {args.plan}")
            return 1

        logger.info(f"Mode: Single Plan Analysis")
        logger.info(f"Plan: {args.plan}")

        try:
            result = orchestrator.analyze_single_plan(
                args.plan,
                output_dir=args.output
            )

            print("\n" + "="*80)
            print("ANALYSIS COMPLETE")
            print("="*80)
            print(f"Plan: {result['plan_name']}")
            print(f"Classification: {result['macro_convergence'].plan_classification}")
            print(f"Overall Score: {result['macro_convergence'].overall_score:.2f}")
            print(f"Agenda Alignment: {result['macro_convergence'].agenda_alignment:.2f}")
            print(f"Execution Time: {result['execution_time']:.2f}s")
            print(f"Output Directory: {result['output_dir']}")
            print("="*80)

            return 0

        except Exception as e:
            logger.exception(f"Analysis failed: {e}")
            return 1

    # Batch mode
    if args.batch:
        if not args.batch.exists() or not args.batch.is_dir():
            logger.error(f"Batch directory not found: {args.batch}")
            return 1

        logger.info(f"Mode: Batch Analysis")
        logger.info(f"Directory: {args.batch}")
        logger.info(f"Max Plans: {args.max_plans or 'unlimited'}")
        logger.info(f"Workers: {args.workers}")

        try:
            results = orchestrator.analyze_batch(
                args.batch,
                output_dir=args.output,
                max_plans=args.max_plans
            )

            # Display summary
            successful = sum(1 for r in results if r.get('status') != 'failed')
            failed = len(results) - successful

            print("\n" + "="*80)
            print("BATCH ANALYSIS COMPLETE")
            print("="*80)
            print(f"Total Plans: {len(results)}")
            print(f"Successful: {successful}")
            print(f"Failed: {failed}")
            print(f"Success Rate: {successful/len(results)*100:.1f}%")
            print(f"Output Directory: {args.output}")
            print("="*80)

            # Display top 10 plans
            scored_results = [
                r for r in results
                if r.get('status') != 'failed' and r.get('macro_convergence')
            ]

            if scored_results:
                print("\nTOP 10 PLANS BY SCORE:")
                print("-"*80)

                top_10 = sorted(
                    scored_results,
                    key=lambda r: r['macro_convergence'].overall_score,
                    reverse=True
                )[:10]

                for i, result in enumerate(top_10, 1):
                    macro = result['macro_convergence']
                    print(f"{i:2d}. {result['plan_name']:40s} "
                          f"{macro.overall_score:5.2f}  {macro.plan_classification}")

                print("="*80)

            return 0

        except Exception as e:
            logger.exception(f"Batch analysis failed: {e}")
            return 1

    # No arguments provided
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
