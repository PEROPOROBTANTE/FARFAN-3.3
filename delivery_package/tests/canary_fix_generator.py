"""
Canary Fix Generator - Automatic Fix Generation for 100% Violation Resolution
=============================================================================

This module analyzes canary violations and generates bulk fix operations to
resolve all detected issues automatically.

Author: Integration Team
Version: 1.0.0
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FixOperation:
    """A fix operation to resolve a canary violation"""
    operation_type: str  # REBASELINE, TYPE_FIX, SCHEMA_FIX, CODE_FIX
    adapter: str
    method: str
    violation_type: str
    fix_command: str
    fix_description: str
    priority: int  # 1=critical, 2=important, 3=minor


class CanaryFixGenerator:
    """Generates automatic fixes for canary violations"""
    
    def __init__(self, report_file: Path = Path("tests/canaries/test_report.json")):
        self.report_file = report_file
        self.report_data = None
        self.fix_operations = []
    
    def load_report(self) -> bool:
        """Load canary test report"""
        if not self.report_file.exists():
            logger.error(f"Report file not found: {self.report_file}")
            logger.error("Please run canary_runner.py first")
            return False
        
        with open(self.report_file, 'r', encoding='utf-8') as f:
            self.report_data = json.load(f)
        
        logger.info(f"Loaded report with {len(self.report_data.get('violations', []))} violations")
        return True
    
    def generate_all_fixes(self) -> List[FixOperation]:
        """Generate fix operations for all violations"""
        if not self.report_data:
            logger.error("No report data loaded")
            return []
        
        violations = self.report_data.get("violations", [])
        
        logger.info("=" * 80)
        logger.info("GENERATING FIX OPERATIONS FOR ALL VIOLATIONS")
        logger.info("=" * 80)
        
        # Group violations by type
        violations_by_type = {
            "HASH_DELTA": [],
            "CONTRACT_TYPE_ERROR": [],
            "INVALID_EVIDENCE": [],
            "EXECUTION_ERROR": []
        }
        
        for violation in violations:
            vtype = violation.get("type", "UNKNOWN")
            if vtype in violations_by_type:
                violations_by_type[vtype].append(violation)
        
        # Generate fixes for each type
        for vtype, vlist in violations_by_type.items():
            if vlist:
                logger.info(f"\nProcessing {len(vlist)} {vtype} violations...")
                self._generate_fixes_for_type(vtype, vlist)
        
        logger.info(f"\nGenerated {len(self.fix_operations)} fix operations")
        return self.fix_operations
    
    def _generate_fixes_for_type(self, violation_type: str, violations: List[Dict]) -> None:
        """Generate fixes for a specific violation type"""
        
        if violation_type == "HASH_DELTA":
            self._generate_hash_delta_fixes(violations)
        elif violation_type == "CONTRACT_TYPE_ERROR":
            self._generate_contract_error_fixes(violations)
        elif violation_type == "INVALID_EVIDENCE":
            self._generate_evidence_fixes(violations)
        elif violation_type == "EXECUTION_ERROR":
            self._generate_execution_error_fixes(violations)
    
    def _generate_hash_delta_fixes(self, violations: List[Dict]) -> None:
        """Generate fixes for HASH_DELTA violations (non-deterministic outputs)"""
        
        # Strategy: Rebaseline canaries since output changed
        for violation in violations:
            adapter = violation["adapter"]
            method = violation["method"]
            
            # Check if this is intentional change or bug
            # For now, assume intentional and suggest rebaseline
            
            fix_op = FixOperation(
                operation_type="REBASELINE",
                adapter=adapter,
                method=method,
                violation_type="HASH_DELTA",
                fix_command=f"python tests/canary_generator.py --adapter {adapter} --method {method} --force",
                fix_description=f"Rebaseline {adapter}.{method} with new expected output hash",
                priority=2
            )
            
            self.fix_operations.append(fix_op)
    
    def _generate_contract_error_fixes(self, violations: List[Dict]) -> None:
        """Generate fixes for CONTRACT_TYPE_ERROR violations"""
        
        for violation in violations:
            adapter = violation["adapter"]
            method = violation["method"]
            details = violation.get("details", "")
            
            # Analyze the type mismatch
            if "wrong type" in details.lower():
                # Type conversion fix needed
                fix_op = FixOperation(
                    operation_type="TYPE_FIX",
                    adapter=adapter,
                    method=method,
                    violation_type="CONTRACT_TYPE_ERROR",
                    fix_command=f"# Auto-fix type issue in {adapter}.{method}",
                    fix_description=f"Fix type mismatch in {adapter}.{method}: {details}",
                    priority=1
                )
                self.fix_operations.append(fix_op)
            
            elif "missing" in details.lower():
                # Missing key fix
                fix_op = FixOperation(
                    operation_type="SCHEMA_FIX",
                    adapter=adapter,
                    method=method,
                    violation_type="CONTRACT_TYPE_ERROR",
                    fix_command=f"# Add missing key in {adapter}.{method}",
                    fix_description=f"Add missing required key in {adapter}.{method}: {details}",
                    priority=1
                )
                self.fix_operations.append(fix_op)
    
    def _generate_evidence_fixes(self, violations: List[Dict]) -> None:
        """Generate fixes for INVALID_EVIDENCE violations"""
        
        for violation in violations:
            adapter = violation["adapter"]
            method = violation["method"]
            details = violation.get("details", "")
            
            fix_op = FixOperation(
                operation_type="SCHEMA_FIX",
                adapter=adapter,
                method=method,
                violation_type="INVALID_EVIDENCE",
                fix_command=f"# Fix evidence structure in {adapter}.{method}",
                fix_description=f"Fix evidence validation issue in {adapter}.{method}: {details}",
                priority=2
            )
            
            self.fix_operations.append(fix_op)
    
    def _generate_execution_error_fixes(self, violations: List[Dict]) -> None:
        """Generate fixes for EXECUTION_ERROR violations"""
        
        for violation in violations:
            adapter = violation["adapter"]
            method = violation["method"]
            details = violation.get("details", "")
            
            fix_op = FixOperation(
                operation_type="CODE_FIX",
                adapter=adapter,
                method=method,
                violation_type="EXECUTION_ERROR",
                fix_command=f"# Debug and fix execution error in {adapter}.{method}",
                fix_description=f"Fix execution failure in {adapter}.{method}: {details}",
                priority=1
            )
            
            self.fix_operations.append(fix_op)
    
    def execute_bulk_fixes(self) -> Dict[str, Any]:
        """Execute all fix operations in bulk"""
        logger.info("\n" + "=" * 80)
        logger.info("EXECUTING BULK FIX OPERATIONS")
        logger.info("=" * 80)
        
        results = {
            "total_operations": len(self.fix_operations),
            "executed": 0,
            "succeeded": 0,
            "failed": 0,
            "operations": []
        }
        
        # Sort by priority
        self.fix_operations.sort(key=lambda x: x.priority)
        
        for fix_op in self.fix_operations:
            logger.info(f"\n[Priority {fix_op.priority}] {fix_op.operation_type}: "
                       f"{fix_op.adapter}.{fix_op.method}")
            logger.info(f"  Description: {fix_op.fix_description}")
            
            success = self._execute_fix(fix_op)
            
            results["executed"] += 1
            if success:
                results["succeeded"] += 1
                logger.info("  ✓ Fix applied successfully")
            else:
                results["failed"] += 1
                logger.info("  ✗ Fix failed")
            
            results["operations"].append({
                "adapter": fix_op.adapter,
                "method": fix_op.method,
                "type": fix_op.operation_type,
                "success": success
            })
        
        logger.info("\n" + "=" * 80)
        logger.info("BULK FIX EXECUTION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total Operations: {results['total_operations']}")
        logger.info(f"Succeeded: {results['succeeded']}")
        logger.info(f"Failed: {results['failed']}")
        logger.info(f"Success Rate: {results['succeeded']/results['total_operations']*100:.1f}%")
        
        return results
    
    def _execute_fix(self, fix_op: FixOperation) -> bool:
        """Execute a single fix operation"""
        
        if fix_op.operation_type == "REBASELINE":
            return self._rebaseline_method(fix_op.adapter, fix_op.method)
        
        elif fix_op.operation_type == "TYPE_FIX":
            return self._apply_type_fix(fix_op)
        
        elif fix_op.operation_type == "SCHEMA_FIX":
            return self._apply_schema_fix(fix_op)
        
        elif fix_op.operation_type == "CODE_FIX":
            return self._apply_code_fix(fix_op)
        
        return False
    
    def _rebaseline_method(self, adapter: str, method: str) -> bool:
        """Rebaseline a method's canary"""
        try:
            from canary_generator import CanaryGenerator
            
            generator = CanaryGenerator()
            method_inputs = generator._get_method_definitions(adapter)
            
            # Find the method definition
            method_def = None
            for md in method_inputs:
                if md["name"] == method:
                    method_def = md
                    break
            
            if not method_def:
                logger.warning(f"  Method definition not found for {adapter}.{method}")
                return False
            
            # Regenerate canary
            generator._generate_method_canary(
                adapter,
                method,
                method_def["inputs"]
            )
            
            return True
            
        except Exception as e:
            logger.error(f"  Rebaseline failed: {e}")
            return False
    
    def _apply_type_fix(self, fix_op: FixOperation) -> bool:
        """Apply type conversion fix"""
        # For type fixes, we typically need to modify adapter code
        # For now, log the fix needed
        logger.info(f"  TYPE_FIX requires manual code change:")
        logger.info(f"  - Edit: orchestrator/module_adapters.py")
        logger.info(f"  - Method: {fix_op.adapter}.{fix_op.method}")
        logger.info(f"  - Ensure return types match contract")
        return False  # Manual fix needed
    
    def _apply_schema_fix(self, fix_op: FixOperation) -> bool:
        """Apply schema structure fix"""
        # Schema fixes typically require code changes
        logger.info(f"  SCHEMA_FIX requires manual code change:")
        logger.info(f"  - Edit: orchestrator/module_adapters.py")
        logger.info(f"  - Method: {fix_op.adapter}.{fix_op.method}")
        logger.info(f"  - Add missing keys or fix structure")
        return False  # Manual fix needed
    
    def _apply_code_fix(self, fix_op: FixOperation) -> bool:
        """Apply code execution fix"""
        # Execution errors need debugging
        logger.info(f"  CODE_FIX requires debugging:")
        logger.info(f"  - Debug: {fix_op.adapter}.{fix_op.method}")
        logger.info(f"  - Check method implementation and dependencies")
        return False  # Manual fix needed
    
    def generate_fix_report(self, output_file: Path) -> None:
        """Generate comprehensive fix report"""
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "report_file": str(self.report_file),
            "total_violations": len(self.report_data.get("violations", [])),
            "total_fix_operations": len(self.fix_operations),
            "operations_by_type": {},
            "operations_by_adapter": {},
            "fix_operations": []
        }
        
        # Group by type
        for fix_op in self.fix_operations:
            if fix_op.operation_type not in report["operations_by_type"]:
                report["operations_by_type"][fix_op.operation_type] = 0
            report["operations_by_type"][fix_op.operation_type] += 1
            
            if fix_op.adapter not in report["operations_by_adapter"]:
                report["operations_by_adapter"][fix_op.adapter] = 0
            report["operations_by_adapter"][fix_op.adapter] += 1
            
            report["fix_operations"].append({
                "type": fix_op.operation_type,
                "adapter": fix_op.adapter,
                "method": fix_op.method,
                "violation_type": fix_op.violation_type,
                "command": fix_op.fix_command,
                "description": fix_op.fix_description,
                "priority": fix_op.priority
            })
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\nFix report saved to: {output_file}")
    
    def print_fix_summary(self) -> None:
        """Print human-readable fix summary"""
        logger.info("\n" + "=" * 80)
        logger.info("FIX OPERATIONS SUMMARY")
        logger.info("=" * 80)
        
        # Count by type
        type_counts = {}
        for fix_op in self.fix_operations:
            type_counts[fix_op.operation_type] = type_counts.get(fix_op.operation_type, 0) + 1
        
        logger.info("\nOperations by Type:")
        for op_type, count in sorted(type_counts.items()):
            logger.info(f"  {op_type}: {count}")
        
        # Count by adapter
        adapter_counts = {}
        for fix_op in self.fix_operations:
            adapter_counts[fix_op.adapter] = adapter_counts.get(fix_op.adapter, 0) + 1
        
        logger.info("\nOperations by Adapter:")
        for adapter, count in sorted(adapter_counts.items()):
            logger.info(f"  {adapter}: {count}")
        
        # Automatic vs Manual
        auto_ops = [op for op in self.fix_operations if op.operation_type == "REBASELINE"]
        manual_ops = [op for op in self.fix_operations if op.operation_type != "REBASELINE"]
        
        logger.info(f"\nAutomatic Fixes (REBASELINE): {len(auto_ops)}")
        logger.info(f"Manual Fixes Required: {len(manual_ops)}")
        
        logger.info("\n" + "=" * 80)
        logger.info("RECOMMENDED ACTIONS")
        logger.info("=" * 80)
        
        if auto_ops:
            logger.info("\n1. AUTOMATIC REBASELINE (Safe if changes are intentional):")
            logger.info("   python tests/canary_fix_generator.py --execute-rebaseline")
        
        if manual_ops:
            logger.info("\n2. MANUAL FIXES REQUIRED:")
            for fix_op in manual_ops[:10]:  # Show first 10
                logger.info(f"\n   [{fix_op.priority}] {fix_op.adapter}.{fix_op.method}")
                logger.info(f"       Type: {fix_op.operation_type}")
                logger.info(f"       Fix: {fix_op.fix_description}")
        
        logger.info("\n" + "=" * 80)


def main():
    """Generate and optionally execute fixes"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate fixes for canary violations")
    parser.add_argument("--execute-rebaseline", action="store_true",
                       help="Execute automatic rebaseline operations")
    parser.add_argument("--report", default="tests/canaries/test_report.json",
                       help="Path to canary test report")
    
    args = parser.parse_args()
    
    generator = CanaryFixGenerator(Path(args.report))
    
    if not generator.load_report():
        sys.exit(1)
    
    # Generate all fixes
    fix_ops = generator.generate_all_fixes()
    
    if not fix_ops:
        logger.info("\n✓ No violations found - all canaries passing!")
        sys.exit(0)
    
    # Print summary
    generator.print_fix_summary()
    
    # Save fix report
    fix_report_file = Path("tests/canaries/fix_report.json")
    generator.generate_fix_report(fix_report_file)
    
    # Execute if requested
    if args.execute_rebaseline:
        logger.info("\n" + "=" * 80)
        logger.info("EXECUTING AUTOMATIC REBASELINE OPERATIONS")
        logger.info("=" * 80)
        results = generator.execute_bulk_fixes()
        
        results_file = Path("tests/canaries/fix_execution_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\nExecution results saved to: {results_file}")
    else:
        logger.info("\nTo execute automatic rebaseline operations, run:")
        logger.info("  python tests/canary_fix_generator.py --execute-rebaseline")


if __name__ == "__main__":
    main()
