#!/usr/bin/env python3
"""
FARFAN 3.0 - Main Execution Script
===================================
Runs the complete policy analysis orchestrator

Usage:
    python run_farfan.py --plan path/to/plan.pdf --output output/directory
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

# Setup absolute imports from project root
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))


def setup_logging(output_dir: Path) -> None:
    """Configure logging to file and console"""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.FileHandler(
                log_dir / f"farfan_{datetime.now():%Y%m%d_%H%M%S}.log"
            ),
            logging.StreamHandler(sys.stdout)
        ]
    )


def initialize_dependencies():
    """Initialize all required dependencies for FARFANOrchestrator"""
    from src.orchestrator.module_adapters import AdapterRegistry
    from src.domain.questionnaire_parser import QuestionnaireParser
    
    logger = logging.getLogger(__name__)
    
    logger.info("Initializing dependencies...")
    
    # Initialize adapter registry
    module_adapter_registry = AdapterRegistry()
    logger.info("✓ AdapterRegistry initialized")
    
    # Initialize questionnaire parser
    questionnaire_parser = QuestionnaireParser()
    logger.info("✓ QuestionnaireParser initialized")
    
    return module_adapter_registry, questionnaire_parser


def main():
    """Main execution entry point"""
    parser = argparse.ArgumentParser(
        description="FARFAN 3.0 - Policy Analysis Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_farfan.py --plan plan.pdf --output ./reports
  python run_farfan.py --plan /path/to/plan.txt --output /path/to/output
        """
    )

    parser.add_argument(
        '--plan',
        type=Path,
        required=True,
        help='Path to plan document (PDF/TXT/DOCX)'
    )

    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output directory for generated reports'
    )

    args = parser.parse_args()

    # Validate plan file exists
    if not args.plan.exists():
        print(f"ERROR: Plan file not found: {args.plan}", file=sys.stderr)
        return 1
    
    # Create output directory if it doesn't exist
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    setup_logging(args.output)
    logger = logging.getLogger(__name__)
    
    print("=" * 80)
    print("FARFAN 3.0 - Policy Analysis Orchestrator")
    print("=" * 80)
    print(f"Plan: {args.plan}")
    print(f"Output Directory: {args.output}")
    print("=" * 80)
    print()
    
    try:
        # Initialize dependencies
        module_adapter_registry, questionnaire_parser = initialize_dependencies()
        
        # Import and instantiate FARFANOrchestrator
        from src.orchestrator.core_orchestrator import FARFANOrchestrator
        
        logger.info("Instantiating FARFANOrchestrator...")
        orchestrator = FARFANOrchestrator(
            module_adapter_registry=module_adapter_registry,
            questionnaire_parser=questionnaire_parser,
            config={}
        )
        logger.info("✓ FARFANOrchestrator instantiated")
        
        # Execute analysis
        logger.info(f"Starting analysis of plan: {args.plan.name}")
        print(f"Analyzing plan: {args.plan.name}...")
        print()
        
        result = orchestrator.analyze_single_plan(
            plan_path=args.plan,
            output_dir=args.output
        )
        
        # Check if analysis was successful
        if not result.get('success', False):
            error_msg = result.get('error', 'Unknown error occurred')
            logger.error(f"Analysis failed: {error_msg}")
            print()
            print("=" * 80)
            print("ANALYSIS FAILED")
            print("=" * 80)
            print(f"Error: {error_msg}")
            print(f"Execution Time: {result.get('execution_time', 0):.2f}s")
            print("=" * 80)
            return 1
        
        # Display success summary
        logger.info("Analysis completed successfully")
        
        print()
        print("=" * 80)
        print("ANALYSIS COMPLETE - SUCCESS")
        print("=" * 80)
        print(f"Plan: {result['plan_name']}")
        
        # Extract execution summary details
        exec_summary = result.get('execution_summary', {})
        macro = result.get('macro_convergence')
        
        print(f"Questions Analyzed: {exec_summary.get('questions_analyzed', 0)}")
        print(f"Clusters Generated: {exec_summary.get('clusters_generated', 0)}")
        
        if macro:
            print(f"Overall Score: {macro.overall_score:.2f}%")
            print(f"Classification: {macro.plan_classification}")
        
        print(f"Execution Time: {exec_summary.get('total_execution_time', 0):.2f}s")
        print()
        print("Generated Reports:")
        print(f"  - Complete Report: {result.get('report_path', 'N/A')}")
        print(f"  - Execution Summary: {result.get('summary_path', 'N/A')}")
        print()
        print(f"All reports saved to: {args.output}")
        print("=" * 80)
        
        return 0
        
    except ImportError as e:
        logger.error(f"Import error: {e}", exc_info=True)
        print()
        print("=" * 80)
        print("ANALYSIS FAILED - IMPORT ERROR")
        print("=" * 80)
        print(f"Error: Failed to import required module: {e}")
        print("Please ensure all dependencies are installed:")
        print("  pip install -r requirements.txt")
        print("=" * 80)
        return 1
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}", exc_info=True)
        print()
        print("=" * 80)
        print("ANALYSIS FAILED - FILE NOT FOUND")
        print("=" * 80)
        print(f"Error: {e}")
        print("Please ensure all required configuration files exist.")
        print("=" * 80)
        return 1
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print()
        print("=" * 80)
        print("ANALYSIS FAILED - UNEXPECTED ERROR")
        print("=" * 80)
        print(f"Error: {e}")
        print("Check the log file for detailed error information.")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
