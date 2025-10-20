#!/usr/bin/env python3
"""
Test script for refactored orchestrator components
"""

import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all modules can be imported"""
    try:
        from orchestrator.module_controller import ModuleController, ModuleTrace, UnifiedAnalysisData
        logger.info("✓ ModuleController imports successful")
        
        from orchestrator.choreographer import Choreographer, Job, JobStatus
        logger.info("✓ Choreographer imports successful")
        
        from orchestrator.report_assembly import ReportAssembler
        logger.info("✓ ReportAssembler imports successful")
        
        from orchestrator.core_orchestrator import FARFANOrchestrator
        logger.info("✓ FARFANOrchestrator imports successful")
        
        return True
    except Exception as e:
        logger.error(f"✗ Import failed: {e}", exc_info=True)
        return False

def test_module_controller_creation():
    """Test ModuleController instantiation"""
    try:
        from orchestrator.module_controller import ModuleController
        from orchestrator.module_adapters import ModuleAdapterRegistry
        from orchestrator.circuit_breaker import CircuitBreaker
        
        registry = ModuleAdapterRegistry()
        breaker = CircuitBreaker()
        
        controller = ModuleController(
            module_adapter_registry=registry,
            circuit_breaker=breaker
        )
        
        logger.info(f"✓ ModuleController created with {len(controller.get_available_modules())} available modules")
        return True
    except Exception as e:
        logger.error(f"✗ ModuleController creation failed: {e}", exc_info=True)
        return False

def test_choreographer_creation():
    """Test Choreographer instantiation"""
    try:
        from orchestrator.choreographer import Choreographer
        from orchestrator.module_controller import ModuleController
        from orchestrator.module_adapters import ModuleAdapterRegistry
        from orchestrator.circuit_breaker import CircuitBreaker
        
        registry = ModuleAdapterRegistry()
        breaker = CircuitBreaker()
        controller = ModuleController(
            module_adapter_registry=registry,
            circuit_breaker=breaker
        )
        
        choreographer = Choreographer(module_controller=controller)
        
        logger.info("✓ Choreographer created successfully")
        logger.info(f"  Queue status: {choreographer.get_queue_status()}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Choreographer creation failed: {e}", exc_info=True)
        return False

def test_orchestrator_creation():
    """Test FARFANOrchestrator with new constructor"""
    try:
        from orchestrator.core_orchestrator import FARFANOrchestrator
        from orchestrator.module_controller import ModuleController
        from orchestrator.module_adapters import ModuleAdapterRegistry
        from orchestrator.circuit_breaker import CircuitBreaker
        from orchestrator.questionnaire_parser import QuestionnaireParser
        
        # Create dependencies
        registry = ModuleAdapterRegistry()
        breaker = CircuitBreaker()
        controller = ModuleController(
            module_adapter_registry=registry,
            circuit_breaker=breaker
        )
        
        parser = QuestionnaireParser()
        
        # Create orchestrator with new signature
        orchestrator = FARFANOrchestrator(
            module_controller=controller,
            questionnaire_parser=parser
        )
        
        logger.info("✓ FARFANOrchestrator created with new constructor")
        
        status = orchestrator.get_orchestrator_status()
        logger.info(f"  Adapters available: {status['adapters_available']}")
        logger.info(f"  Questions available: {status['questions_available']}")
        
        return True
    except Exception as e:
        logger.error(f"✗ FARFANOrchestrator creation failed: {e}", exc_info=True)
        return False

def main():
    """Run all tests"""
    print("=" * 80)
    print("REFACTORING VALIDATION TEST")
    print("=" * 80)
    print()
    
    tests = [
        ("Imports", test_imports),
        ("ModuleController Creation", test_module_controller_creation),
        ("Choreographer Creation", test_choreographer_creation),
        ("FARFANOrchestrator Creation", test_orchestrator_creation)
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\nRunning: {name}")
        print("-" * 80)
        result = test_func()
        results.append((name, result))
        print()
    
    print("=" * 80)
    print("TEST RESULTS SUMMARY")
    print("=" * 80)
    
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {name}: {status}")
    
    print("=" * 80)
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
