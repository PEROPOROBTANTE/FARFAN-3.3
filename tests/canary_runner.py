"""
Canary Runner - Regression Detection System for 413 Adapter Methods
====================================================================

This module runs canary tests to detect three types of regressions:
1. HASH_DELTA - Output determinism violations (hash mismatch)
2. CONTRACT_TYPE_ERROR - JSON schema validation failures
3. INVALID_EVIDENCE - Missing required keys in output

Author: Integration Team
Version: 1.0.0
"""

import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator.module_adapters import ModuleAdapterRegistry, ModuleResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CanaryViolation:
    """A detected regression violation"""
    adapter: str
    method: str
    violation_type: str  # HASH_DELTA, CONTRACT_TYPE_ERROR, INVALID_EVIDENCE
    expected: Any
    actual: Any
    details: str


@dataclass
class CanaryReport:
    """Complete canary test report"""
    timestamp: str
    total_methods: int
    passed: int
    failed: int
    violations: List[CanaryViolation] = field(default_factory=list)
    execution_times: Dict[str, float] = field(default_factory=dict)
    adapter_summary: Dict[str, Dict[str, int]] = field(default_factory=dict)


class CanaryRunner:
    """Runs canary regression tests across all adapter methods"""
    
    # Expected contract schema for ModuleResult
    EXPECTED_CONTRACT = {
        "required_keys": [
            "module_name", "class_name", "method_name", "status",
            "data", "evidence", "confidence"
        ],
        "optional_keys": ["errors", "warnings", "metadata", "execution_time"],
        "types": {
            "module_name": str,
            "class_name": str,
            "method_name": str,
            "status": str,
            "data": dict,
            "evidence": list,
            "confidence": (int, float),
            "errors": list,
            "warnings": list,
            "metadata": dict
        }
    }
    
    def __init__(self, canary_dir: Path = Path("tests/canaries")):
        self.canary_dir = canary_dir
        self.registry = ModuleAdapterRegistry()
        self.report = CanaryReport(
            timestamp=datetime.now().isoformat(),
            total_methods=0,
            passed=0,
            failed=0
        )
    
    def run_all_canaries(self) -> CanaryReport:
        """Run all canary tests and generate report"""
        logger.info("=" * 80)
        logger.info("CANARY REGRESSION DETECTION - ALL 413 ADAPTER METHODS")
        logger.info("=" * 80)
        
        if not self.canary_dir.exists():
            logger.error(f"Canary directory not found: {self.canary_dir}")
            logger.error("Please run canary_generator.py first to create baseline canaries")
            return self.report
        
        # Iterate through all adapter directories
        for adapter_dir in sorted(self.canary_dir.iterdir()):
            if not adapter_dir.is_dir() or adapter_dir.name.startswith('.'):
                continue
            
            adapter_name = adapter_dir.name
            logger.info(f"\n{'=' * 80}")
            logger.info(f"Testing adapter: {adapter_name}")
            logger.info(f"{'=' * 80}")
            
            self._run_adapter_canaries(adapter_name)
        
        # Generate summary
        self._generate_summary()
        
        return self.report
    
    def _run_adapter_canaries(self, adapter_name: str) -> None:
        """Run all canary tests for a single adapter"""
        adapter_dir = self.canary_dir / adapter_name
        
        if adapter_name not in self.report.adapter_summary:
            self.report.adapter_summary[adapter_name] = {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "hash_delta": 0,
                "contract_error": 0,
                "invalid_evidence": 0
            }
        
        # Check if adapter is available
        if adapter_name not in self.registry.adapters:
            logger.warning(f"  ✗ Adapter {adapter_name} not in registry - skipping")
            return
        
        adapter = self.registry.adapters[adapter_name]
        if not adapter.available:
            logger.warning(f"  ✗ Adapter {adapter_name} not available - skipping")
            return
        
        # Iterate through all method directories
        for method_dir in sorted(adapter_dir.iterdir()):
            if not method_dir.is_dir():
                continue
            
            method_name = method_dir.name
            self.report.total_methods += 1
            self.report.adapter_summary[adapter_name]["total"] += 1
            
            violations = self._test_method_canary(adapter_name, method_name, method_dir)
            
            if violations:
                self.report.failed += 1
                self.report.adapter_summary[adapter_name]["failed"] += 1
                self.report.violations.extend(violations)
                
                for violation in violations:
                    self.report.adapter_summary[adapter_name][
                        violation.violation_type.lower()
                    ] += 1
                
                logger.info(f"  ✗ {method_name}: {len(violations)} violation(s)")
            else:
                self.report.passed += 1
                self.report.adapter_summary[adapter_name]["passed"] += 1
                logger.info(f"  ✓ {method_name}")
    
    def _test_method_canary(self, adapter_name: str, method_name: str, 
                           method_dir: Path) -> List[CanaryViolation]:
        """Test a single method's canary and detect violations"""
        violations = []
        
        # Load canary files
        input_file = method_dir / "input.json"
        expected_file = method_dir / "expected.json"
        hash_file = method_dir / "expected_hash.txt"
        
        if not all([input_file.exists(), expected_file.exists(), hash_file.exists()]):
            logger.warning(f"    Missing canary files for {method_name}")
            return violations
        
        try:
            # Load inputs
            with open(input_file, 'r', encoding='utf-8') as f:
                inputs = json.load(f)
            
            # Load expected output and hash
            with open(expected_file, 'r', encoding='utf-8') as f:
                expected = json.load(f)
            
            with open(hash_file, 'r') as f:
                expected_hash = f.read().strip()
            
            # Execute method
            start_time = datetime.now()
            result = self.registry.execute_module_method(
                adapter_name,
                method_name,
                inputs.get("args", []),
                inputs.get("kwargs", {})
            )
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Convert result to dict
            actual = {
                "module_name": result.module_name,
                "class_name": result.class_name,
                "method_name": result.method_name,
                "status": result.status,
                "data": result.data,
                "evidence": result.evidence,
                "confidence": result.confidence,
                "errors": result.errors,
                "warnings": result.warnings,
                "metadata": result.metadata
            }
            
            self.report.execution_times[f"{adapter_name}.{method_name}"] = execution_time
            
            # CHECK 1: HASH_DELTA - Determinism violation
            actual_hash = self._compute_hash(actual)
            if actual_hash != expected_hash:
                violations.append(CanaryViolation(
                    adapter=adapter_name,
                    method=method_name,
                    violation_type="HASH_DELTA",
                    expected=expected_hash,
                    actual=actual_hash,
                    details=f"Output hash changed. Expected: {expected_hash[:16]}..., "
                           f"Got: {actual_hash[:16]}..."
                ))
            
            # CHECK 2: CONTRACT_TYPE_ERROR - Schema validation
            contract_violations = self._validate_contract(actual)
            for cv in contract_violations:
                violations.append(CanaryViolation(
                    adapter=adapter_name,
                    method=method_name,
                    violation_type="CONTRACT_TYPE_ERROR",
                    expected=cv["expected"],
                    actual=cv["actual"],
                    details=cv["details"]
                ))
            
            # CHECK 3: INVALID_EVIDENCE - Missing required keys
            evidence_violations = self._validate_evidence(actual)
            for ev in evidence_violations:
                violations.append(CanaryViolation(
                    adapter=adapter_name,
                    method=method_name,
                    violation_type="INVALID_EVIDENCE",
                    expected=ev["expected"],
                    actual=ev["actual"],
                    details=ev["details"]
                ))
            
        except Exception as e:
            logger.error(f"    Error testing {method_name}: {e}")
            violations.append(CanaryViolation(
                adapter=adapter_name,
                method=method_name,
                violation_type="EXECUTION_ERROR",
                expected="successful execution",
                actual=str(e),
                details=f"Method execution failed: {str(e)}"
            ))
        
        return violations
    
    def _compute_hash(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash of data"""
        # Remove non-deterministic fields
        data_copy = data.copy()
        data_copy.pop("execution_time", None)
        
        json_str = json.dumps(data_copy, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(json_str.encode('utf-8')).hexdigest()
    
    def _validate_contract(self, actual: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate output against expected contract schema"""
        violations = []
        
        # Check required keys
        for required_key in self.EXPECTED_CONTRACT["required_keys"]:
            if required_key not in actual:
                violations.append({
                    "expected": f"key '{required_key}' present",
                    "actual": "missing",
                    "details": f"Required key '{required_key}' missing from output"
                })
        
        # Check types
        for key, expected_type in self.EXPECTED_CONTRACT["types"].items():
            if key in actual and actual[key] is not None:
                if not isinstance(actual[key], expected_type):
                    violations.append({
                        "expected": f"type {expected_type}",
                        "actual": f"type {type(actual[key])}",
                        "details": f"Key '{key}' has wrong type. "
                                  f"Expected {expected_type}, got {type(actual[key])}"
                    })
        
        return violations
    
    def _validate_evidence(self, actual: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate evidence structure"""
        violations = []
        
        if "evidence" not in actual:
            return violations
        
        evidence = actual["evidence"]
        
        if not isinstance(evidence, list):
            violations.append({
                "expected": "evidence as list",
                "actual": f"type {type(evidence)}",
                "details": "Evidence must be a list"
            })
            return violations
        
        # Check each evidence item has required structure
        for idx, item in enumerate(evidence):
            if not isinstance(item, dict):
                violations.append({
                    "expected": "evidence item as dict",
                    "actual": f"type {type(item)}",
                    "details": f"Evidence item {idx} must be a dictionary"
                })
                continue
            
            # Check for at least 'type' key
            if "type" not in item:
                violations.append({
                    "expected": "'type' key in evidence",
                    "actual": "missing",
                    "details": f"Evidence item {idx} missing 'type' field"
                })
        
        return violations
    
    def _generate_summary(self) -> None:
        """Generate and print test summary"""
        logger.info("\n" + "=" * 80)
        logger.info("CANARY TEST SUMMARY")
        logger.info("=" * 80)
        
        logger.info(f"\nTotal Methods Tested: {self.report.total_methods}")
        logger.info(f"  ✓ Passed: {self.report.passed}")
        logger.info(f"  ✗ Failed: {self.report.failed}")
        
        if self.report.total_methods > 0:
            pass_rate = (self.report.passed / self.report.total_methods) * 100
            logger.info(f"\nPass Rate: {pass_rate:.1f}%")
        
        # Violation breakdown
        violation_counts = {
            "HASH_DELTA": 0,
            "CONTRACT_TYPE_ERROR": 0,
            "INVALID_EVIDENCE": 0,
            "EXECUTION_ERROR": 0
        }
        
        for violation in self.report.violations:
            violation_counts[violation.violation_type] = \
                violation_counts.get(violation.violation_type, 0) + 1
        
        logger.info("\nViolation Types:")
        for vtype, count in violation_counts.items():
            if count > 0:
                logger.info(f"  {vtype}: {count}")
        
        # Per-adapter summary
        logger.info("\n" + "=" * 80)
        logger.info("PER-ADAPTER SUMMARY")
        logger.info("=" * 80)
        
        for adapter_name in sorted(self.report.adapter_summary.keys()):
            summary = self.report.adapter_summary[adapter_name]
            logger.info(f"\n{adapter_name}:")
            logger.info(f"  Total: {summary['total']}")
            logger.info(f"  Passed: {summary['passed']}")
            logger.info(f"  Failed: {summary['failed']}")
            if summary['hash_delta'] > 0:
                logger.info(f"  HASH_DELTA: {summary['hash_delta']}")
            if summary['contract_error'] > 0:
                logger.info(f"  CONTRACT_TYPE_ERROR: {summary['contract_error']}")
            if summary['invalid_evidence'] > 0:
                logger.info(f"  INVALID_EVIDENCE: {summary['invalid_evidence']}")
        
        # Top violations detail
        if self.report.violations:
            logger.info("\n" + "=" * 80)
            logger.info("DETAILED VIOLATIONS (First 20)")
            logger.info("=" * 80)
            
            for idx, violation in enumerate(self.report.violations[:20]):
                logger.info(f"\n[{idx + 1}] {violation.adapter}.{violation.method}")
                logger.info(f"    Type: {violation.violation_type}")
                logger.info(f"    Details: {violation.details}")
    
    def save_report(self, output_file: Path) -> None:
        """Save report to JSON file"""
        report_data = {
            "timestamp": self.report.timestamp,
            "summary": {
                "total_methods": self.report.total_methods,
                "passed": self.report.passed,
                "failed": self.report.failed,
                "pass_rate": (self.report.passed / self.report.total_methods * 100)
                            if self.report.total_methods > 0 else 0
            },
            "adapter_summary": self.report.adapter_summary,
            "violations": [
                {
                    "adapter": v.adapter,
                    "method": v.method,
                    "type": v.violation_type,
                    "expected": str(v.expected),
                    "actual": str(v.actual),
                    "details": v.details
                }
                for v in self.report.violations
            ],
            "execution_times": self.report.execution_times
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\nDetailed report saved to: {output_file}")
    
    def generate_rebaseline_commands(self) -> List[str]:
        """Generate commands to rebaseline canaries"""
        commands = []
        
        if not self.report.violations:
            return commands
        
        logger.info("\n" + "=" * 80)
        logger.info("REBASELINE RECOMMENDATIONS")
        logger.info("=" * 80)
        
        logger.info("\nIf these changes are intentional, rebaseline canaries with:")
        logger.info("\n# Rebaseline specific method:")
        
        affected_methods = set()
        for violation in self.report.violations:
            method_key = f"{violation.adapter}/{violation.method}"
            if method_key not in affected_methods:
                affected_methods.add(method_key)
                cmd = f"python tests/canary_generator.py --adapter {violation.adapter} --method {violation.method}"
                commands.append(cmd)
                logger.info(f"  {cmd}")
        
        logger.info("\n# Or rebaseline entire adapter:")
        affected_adapters = set(v.adapter for v in self.report.violations)
        for adapter in sorted(affected_adapters):
            cmd = f"python tests/canary_generator.py --adapter {adapter}"
            logger.info(f"  {cmd}")
        
        logger.info("\n# Or rebaseline ALL canaries (use with caution!):")
        logger.info("  python tests/canary_generator.py --all")
        
        return commands


def main():
    """Run all canary tests"""
    runner = CanaryRunner()
    report = runner.run_all_canaries()
    
    # Save detailed report
    report_file = Path("tests/canaries/test_report.json")
    runner.save_report(report_file)
    
    # Generate rebaseline commands
    runner.generate_rebaseline_commands()
    
    # Exit with appropriate code
    if report.failed > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
