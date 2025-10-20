#!/usr/bin/env python3
"""
Test script to validate refactoring of core_orchestrator.py, choreographer.py, and report_assembly.py
"""
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all refactored modules can be imported"""
    logger.info("Testing imports...")
    
    try:
        from orchestrator.module_adapters import ModuleAdapterRegistry, ModuleController
        logger.info("✓ ModuleAdapterRegistry imported")
        logger.info("✓ ModuleController imported")
    except Exception as e:
        logger.error(f"✗ Failed to import module_adapters: {e}")
        return False
    
    try:
        from orchestrator.question_router import QuestionRouter
        logger.info("✓ QuestionRouter imported")
    except Exception as e:
        logger.error(f"✗ Failed to import QuestionRouter: {e}")
        return False
    
    try:
        from orchestrator.choreographer import Choreographer, ExecutionResult, JobSummary
        logger.info("✓ Choreographer imported")
        logger.info("✓ ExecutionResult imported")
        logger.info("✓ JobSummary imported")
    except Exception as e:
        logger.error(f"✗ Failed to import choreographer: {e}")
        return False
    
    try:
        from orchestrator.circuit_breaker import CircuitBreaker
        logger.info("✓ CircuitBreaker imported")
    except Exception as e:
        logger.error(f"✗ Failed to import CircuitBreaker: {e}")
        return False
    
    try:
        from orchestrator.report_assembly import ReportAssembler
        logger.info("✓ ReportAssembler imported")
    except Exception as e:
        logger.error(f"✗ Failed to import ReportAssembler: {e}")
        return False
    
    try:
        from orchestrator.core_orchestrator import FARFANOrchestrator
        logger.info("✓ FARFANOrchestrator imported")
    except Exception as e:
        logger.error(f"✗ Failed to import FARFANOrchestrator: {e}")
        return False
    
    return True

def test_initialization():
    """Test that core components can be initialized"""
    logger.info("\nTesting initialization...")
    
    try:
        from orchestrator.module_adapters import ModuleAdapterRegistry, ModuleController
        from orchestrator.question_router import QuestionRouter
        from orchestrator.choreographer import Choreographer
        from orchestrator.circuit_breaker import CircuitBreaker
        
        # Initialize registry
        registry = ModuleAdapterRegistry()
        logger.info(f"✓ ModuleAdapterRegistry initialized with {len(registry.adapters)} adapters")
        
        # Initialize router
        router = QuestionRouter()
        logger.info(f"✓ QuestionRouter initialized")
        
        # Initialize controller
        controller = ModuleController(registry, router)
        logger.info(f"✓ ModuleController initialized with responsibility map")
        
        # Initialize choreographer
        choreographer = Choreographer(module_controller=controller)
        logger.info(f"✓ Choreographer initialized")
        
        # Initialize circuit breaker
        circuit_breaker = CircuitBreaker()
        logger.info(f"✓ CircuitBreaker initialized for {len(circuit_breaker.adapters)} adapters")
        
        return True
    except Exception as e:
        logger.error(f"✗ Initialization failed: {e}", exc_info=True)
        return False

def test_module_controller_integration():
    """Test ModuleController integration with responsibility map"""
    logger.info("\nTesting ModuleController integration...")
    
    try:
        from orchestrator.module_adapters import ModuleAdapterRegistry, ModuleController
        from orchestrator.question_router import QuestionRouter
        
        registry = ModuleAdapterRegistry()
        router = QuestionRouter()
        controller = ModuleController(registry, router)
        
        # Test responsibility map
        d1_adapters = controller.get_responsible_adapters("D1")
        logger.info(f"✓ D1 responsible adapters: {d1_adapters}")
        
        d6_adapters = controller.get_responsible_adapters("D6")
        logger.info(f"✓ D6 responsible adapters: {d6_adapters}")
        
        # Test adapter capabilities
        teoria_caps = controller.get_adapter_capabilities("teoria_cambio")
        logger.info(f"✓ teoria_cambio capabilities: {teoria_caps}")
        
        return True
    except Exception as e:
        logger.error(f"✗ ModuleController integration test failed: {e}", exc_info=True)
        return False

def test_report_assembly_normalization():
    """Test ReportAssembler's ability to normalize different result formats"""
    logger.info("\nTesting ReportAssembler normalization...")
    
    try:
        from orchestrator.report_assembly import ReportAssembler
        from orchestrator.choreographer import ExecutionStatus
        from orchestrator.module_adapters import ModuleResult
        
        assembler = ReportAssembler()
        
        # Test with ModuleResult object
        module_result = ModuleResult(
            module_name="test_adapter",
            class_name="TestAdapter",
            method_name="test_method",
            status="success",
            data={"test": "data"},
            evidence=[{"text": "test evidence"}],
            confidence=0.85,
            execution_time=1.5
        )
        
        test_results = {"test_adapter.test_method": module_result}
        normalized = assembler._normalize_execution_results(test_results)
        
        logger.info(f"✓ Normalized ModuleResult: status={normalized['test_adapter.test_method']['status']}")
        logger.info(f"✓ Normalized confidence: {normalized['test_adapter.test_method']['confidence']}")
        
        return True
    except Exception as e:
        logger.error(f"✗ ReportAssembler normalization test failed: {e}", exc_info=True)
        return False

def main():
    """Run all tests"""
    logger.info("=" * 80)
    logger.info("REFACTORING VALIDATION TEST SUITE")
    logger.info("=" * 80)
    
    tests = [
        ("Import Test", test_imports),
        ("Initialization Test", test_initialization),
        ("ModuleController Integration Test", test_module_controller_integration),
        ("ReportAssembler Normalization Test", test_report_assembly_normalization)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"✗ {test_name} crashed: {e}", exc_info=True)
            results.append((test_name, False))
    
    logger.info("\n" + "=" * 80)
    logger.info("TEST RESULTS")
    logger.info("=" * 80)
    
    passed = 0
    failed = 0
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {test_name}")
        if result:
            passed += 1
        else:
            failed += 1
    
    logger.info("=" * 80)
    logger.info(f"Total: {passed} passed, {failed} failed")
    logger.info("=" * 80)
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
