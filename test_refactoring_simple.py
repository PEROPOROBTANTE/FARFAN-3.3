#!/usr/bin/env python3
"""
Simple test for new refactored components only
"""

import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_new_files_syntax():
    """Test that new files have valid syntax"""
    try:
        import py_compile
        
        files = [
            'orchestrator/module_controller.py',
            'orchestrator/choreographer.py',
            'orchestrator/core_orchestrator.py',
            'orchestrator/report_assembly.py'
        ]
        
        for file in files:
            try:
                py_compile.compile(file, doraise=True)
                logger.info(f"✓ {file} syntax valid")
            except py_compile.PyCompileError as e:
                logger.error(f"✗ {file} syntax error: {e}")
                return False
        
        return True
    except Exception as e:
        logger.error(f"✗ Syntax test failed: {e}", exc_info=True)
        return False

def test_imports_only():
    """Test imports of new components without dependencies"""
    try:
        logger.info("Testing ModuleController import...")
        from orchestrator.module_controller import ModuleController, ModuleTrace, UnifiedAnalysisData
        logger.info("✓ ModuleController imports successful")
        
        logger.info("Testing Choreographer import...")
        from orchestrator.choreographer import Choreographer, Job, JobStatus
        logger.info("✓ Choreographer imports successful")
        
        logger.info("Testing ReportAssembler changes...")
        from orchestrator.report_assembly import ReportAssembler
        logger.info("✓ ReportAssembler imports successful")
        
        return True
    except Exception as e:
        logger.error(f"✗ Import test failed: {e}", exc_info=True)
        return False

def test_dataclass_creation():
    """Test that new dataclasses can be instantiated"""
    try:
        from orchestrator.module_controller import ModuleTrace, UnifiedAnalysisData
        from orchestrator.choreographer import Job, JobStatus
        
        # Test ModuleTrace
        trace = ModuleTrace(
            module_name="test_module",
            method_name="test_method",
            status="success",
            execution_time=0.1,
            result_data={"test": "data"}
        )
        logger.info(f"✓ ModuleTrace created: {trace.module_name}.{trace.method_name}")
        
        # Test UnifiedAnalysisData
        unified = UnifiedAnalysisData(
            question_id="P1-D1-Q1",
            module_traces=[trace],
            operation_results={"test": "result"},
            execution_metadata={"steps": 1},
            total_execution_time=0.1
        )
        logger.info(f"✓ UnifiedAnalysisData created: {unified.question_id}")
        
        # Test Job
        from dataclasses import dataclass
        @dataclass
        class MockQuestion:
            canonical_id: str = "P1-D1-Q1"
        
        job = Job(
            job_id="test-job-123",
            question_spec=MockQuestion(),
            plan_text="test plan"
        )
        logger.info(f"✓ Job created: {job.job_id}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Dataclass test failed: {e}", exc_info=True)
        return False

def main():
    """Run all simple tests"""
    print("=" * 80)
    print("SIMPLE REFACTORING VALIDATION")
    print("=" * 80)
    print()
    
    tests = [
        ("Syntax Validation", test_new_files_syntax),
        ("Import Test", test_imports_only),
        ("Dataclass Creation", test_dataclass_creation)
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
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {name}: {status}")
    
    print("=" * 80)
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n✓ All simple tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
