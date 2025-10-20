#!/usr/bin/env python3
"""
Orchestrator Invocation Audit Tool
===================================

Systematically audits method invocations from orchestrator files to module_adapters.py
by comparing signatures, verifying method existence, and detecting interface drift.

Focuses exclusively on the orchestrator-to-adapter boundary.
"""

import ast
import inspect
import csv
import sys
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import importlib.util

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class InvocationRecord:
    """Record of a method invocation from orchestrator to adapter"""
    orchestrator_file: str
    line_number: int
    caller_function: str
    target_module: str
    target_class: str
    target_method: str
    expected_args: List[str]
    expected_kwargs: List[str]
    actual_signature: Optional[str] = None
    status: str = "UNKNOWN"
    remediation: str = ""
    
    def to_csv_row(self) -> List[str]:
        """Convert to CSV row"""
        return [
            self.orchestrator_file,
            str(self.line_number),
            self.caller_function,
            f"{self.target_module}.{self.target_class}.{self.target_method}" if self.target_class else f"{self.target_module}.{self.target_method}",
            f"args={self.expected_args}, kwargs={self.expected_kwargs}",
            self.actual_signature or "N/A",
            self.status,
            self.remediation
        ]


@dataclass
class AuditResults:
    """Aggregated audit results"""
    invocations: List[InvocationRecord] = field(default_factory=list)
    total_invocations: int = 0
    ok_count: int = 0
    mismatch_count: int = 0
    missing_count: int = 0
    violation_count: int = 0
    
    def add_invocation(self, record: InvocationRecord):
        """Add invocation and update counts"""
        self.invocations.append(record)
        self.total_invocations += 1
        
        if record.status == "OK":
            self.ok_count += 1
        elif record.status == "MISMATCH":
            self.mismatch_count += 1
        elif record.status == "MISSING":
            self.missing_count += 1
        elif record.status == "STATIC_VIOLATION":
            self.violation_count += 1


# ============================================================================
# AST VISITOR FOR INVOCATION EXTRACTION
# ============================================================================

class InvocationExtractor(ast.NodeVisitor):
    """Extract method invocations from orchestrator files"""
    
    def __init__(self, filename: str):
        self.filename = filename
        self.invocations: List[InvocationRecord] = []
        self.current_function: str = "<module>"
        
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Track current function context"""
        old_function = self.current_function
        self.current_function = node.name
        self.generic_visit(node)
        self.current_function = old_function
    
    def visit_Call(self, node: ast.Call):
        """Extract method calls"""
        # Pattern 1: module_adapter_registry.execute_module_method(...)
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == "execute_module_method":
                self._extract_execute_module_method_call(node)
            # Pattern 2: self.module_registry.execute_module_method(...)
            elif isinstance(node.func.value, ast.Attribute):
                if node.func.value.attr in ["module_registry", "module_adapter_registry"]:
                    if node.func.attr == "execute_module_method":
                        self._extract_execute_module_method_call(node)
        
        self.generic_visit(node)
    
    def _extract_execute_module_method_call(self, node: ast.Call):
        """Extract details from execute_module_method call"""
        module_name = None
        method_name = None
        args = []
        kwargs = []
        
        # Extract keyword arguments
        for keyword in node.keywords:
            if keyword.arg == "module_name":
                module_name = self._extract_string_value(keyword.value)
            elif keyword.arg == "method_name":
                method_name = self._extract_string_value(keyword.value)
            elif keyword.arg == "args":
                args = self._extract_list_items(keyword.value)
            elif keyword.arg == "kwargs":
                kwargs = self._extract_dict_keys(keyword.value)
        
        if module_name and method_name:
            record = InvocationRecord(
                orchestrator_file=self.filename,
                line_number=node.lineno,
                caller_function=self.current_function,
                target_module=module_name,
                target_class="",  # Will be resolved later
                target_method=method_name,
                expected_args=args,
                expected_kwargs=kwargs
            )
            self.invocations.append(record)
    
    def _extract_string_value(self, node: ast.AST) -> Optional[str]:
        """Extract string value from AST node"""
        if isinstance(node, ast.Constant):
            return str(node.value)
        elif isinstance(node, ast.Str):
            return node.s
        return None
    
    def _extract_list_items(self, node: ast.AST) -> List[str]:
        """Extract items from list node"""
        items = []
        if isinstance(node, ast.List):
            for elt in node.elts:
                if isinstance(elt, (ast.Constant, ast.Str)):
                    items.append(str(elt.value if isinstance(elt, ast.Constant) else elt.s))
                else:
                    items.append("<complex>")
        return items
    
    def _extract_dict_keys(self, node: ast.AST) -> List[str]:
        """Extract keys from dict node"""
        keys = []
        if isinstance(node, ast.Dict):
            for key in node.keys:
                if isinstance(key, (ast.Constant, ast.Str)):
                    keys.append(str(key.value if isinstance(key, ast.Constant) else key.s))
                else:
                    keys.append("<complex>")
        return keys


# ============================================================================
# SIGNATURE INSPECTOR
# ============================================================================

class AdapterSignatureInspector:
    """Inspect actual signatures from module_adapters.py"""
    
    def __init__(self, adapters_path: Path):
        self.adapters_path = adapters_path
        self.adapter_map: Dict[str, Any] = {}
        self.method_signatures: Dict[str, Dict[str, inspect.Signature]] = {}
        self._load_adapters()
    
    def _load_adapters(self):
        """Load module_adapters.py dynamically"""
        try:
            spec = importlib.util.spec_from_file_location("module_adapters", self.adapters_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules["module_adapters"] = module
            spec.loader.exec_module(module)
            
            # Map module names to adapter classes
            self.adapter_map = {
                "teoria_cambio": getattr(module, "ModulosAdapter", None),
                "analyzer_one": getattr(module, "AnalyzerOneAdapter", None),
                "dereck_beach": getattr(module, "DerekBeachAdapter", None),
                "embedding_policy": getattr(module, "EmbeddingPolicyAdapter", None),
                "semantic_chunking_policy": getattr(module, "SemanticChunkingPolicyAdapter", None),
                "contradiction_detection": getattr(module, "ContradictionDetectionAdapter", None),
                "financial_viability": getattr(module, "FinancialViabilityAdapter", None),
                "policy_processor": getattr(module, "PolicyProcessorAdapter", None),
                "policy_segmenter": getattr(module, "PolicySegmenterAdapter", None),
            }
            
            # Extract method signatures from each adapter
            for module_name, adapter_class in self.adapter_map.items():
                if adapter_class:
                    self.method_signatures[module_name] = self._extract_methods(adapter_class)
            
            print(f"✓ Loaded {len(self.adapter_map)} adapters from module_adapters.py")
            
        except Exception as e:
            print(f"✗ Failed to load module_adapters.py: {e}")
            import traceback
            traceback.print_exc()
    
    def _extract_methods(self, adapter_class: type) -> Dict[str, inspect.Signature]:
        """Extract method signatures from adapter class"""
        methods = {}
        
        try:
            # Get all methods from the class
            for name, method in inspect.getmembers(adapter_class, predicate=inspect.ismethod):
                if not name.startswith('_'):
                    try:
                        sig = inspect.signature(method)
                        methods[name] = sig
                    except (ValueError, TypeError):
                        pass
            
            # Also check for instance methods via __init__
            try:
                instance = adapter_class()
                for name in dir(instance):
                    if not name.startswith('_') and callable(getattr(instance, name, None)):
                        method = getattr(instance, name)
                        try:
                            sig = inspect.signature(method)
                            methods[name] = sig
                        except (ValueError, TypeError):
                            pass
            except Exception:
                pass
                
        except Exception as e:
            print(f"Warning: Could not extract methods from {adapter_class.__name__}: {e}")
        
        return methods
    
    def get_signature(self, module_name: str, method_name: str) -> Optional[inspect.Signature]:
        """Get signature for a specific method"""
        if module_name in self.method_signatures:
            return self.method_signatures[module_name].get(method_name)
        return None
    
    def method_exists(self, module_name: str, method_name: str) -> bool:
        """Check if method exists in adapter"""
        return module_name in self.method_signatures and method_name in self.method_signatures[module_name]


# ============================================================================
# SIGNATURE COMPARATOR
# ============================================================================

class SignatureComparator:
    """Compare expected invocations with actual signatures"""
    
    def __init__(self, inspector: AdapterSignatureInspector):
        self.inspector = inspector
    
    def compare_invocation(self, record: InvocationRecord) -> InvocationRecord:
        """Compare invocation against actual signature"""
        # Check if method exists
        if not self.inspector.method_exists(record.target_module, record.target_method):
            record.status = "MISSING"
            record.actual_signature = "N/A - Method not found"
            record.remediation = f"Method '{record.target_method}' does not exist in {record.target_module} adapter. Verify method name or implement missing method."
            return record
        
        # Get actual signature
        signature = self.inspector.get_signature(record.target_module, record.target_method)
        if signature is None:
            record.status = "MISSING"
            record.actual_signature = "N/A - Signature unavailable"
            record.remediation = "Cannot retrieve method signature for comparison."
            return record
        
        record.actual_signature = str(signature)
        
        # Compare arguments
        params = list(signature.parameters.values())
        
        # Filter out 'self'
        params = [p for p in params if p.name != 'self']
        
        # Count expected vs actual
        expected_arg_count = len(record.expected_args)
        expected_kwarg_count = len(record.expected_kwargs)
        
        # Analyze parameters
        required_params = [p for p in params if p.default == inspect.Parameter.empty and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)]
        optional_params = [p for p in params if p.default != inspect.Parameter.empty]
        var_positional = [p for p in params if p.kind == inspect.Parameter.VAR_POSITIONAL]
        var_keyword = [p for p in params if p.kind == inspect.Parameter.VAR_KEYWORD]
        
        # Check compatibility
        issues = []
        
        # Check if we have enough required args
        if expected_arg_count < len(required_params):
            issues.append(f"Missing {len(required_params) - expected_arg_count} required positional argument(s)")
        
        # Check if we have too many args (and no *args)
        if not var_positional and expected_arg_count > len(params):
            issues.append(f"Too many positional arguments: expected max {len(params)}, got {expected_arg_count}")
        
        # Check for unknown kwargs (if no **kwargs)
        if not var_keyword:
            param_names = {p.name for p in params}
            unknown_kwargs = [k for k in record.expected_kwargs if k not in param_names]
            if unknown_kwargs:
                issues.append(f"Unknown keyword arguments: {unknown_kwargs}")
        
        if issues:
            record.status = "MISMATCH"
            record.remediation = "; ".join(issues)
        else:
            record.status = "OK"
            record.remediation = "Signature compatible"
        
        return record


# ============================================================================
# ORCHESTRATOR ANALYZER
# ============================================================================

class OrchestratorAnalyzer:
    """Main analyzer for orchestrator files"""
    
    def __init__(self, orchestrator_dir: Path, adapters_path: Path):
        self.orchestrator_dir = orchestrator_dir
        self.inspector = AdapterSignatureInspector(adapters_path)
        self.comparator = SignatureComparator(self.inspector)
        self.results = AuditResults()
    
    def analyze_orchestrator_file(self, filepath: Path) -> List[InvocationRecord]:
        """Analyze a single orchestrator file"""
        print(f"\nAnalyzing {filepath.name}...")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                source = f.read()
            
            tree = ast.parse(source, filename=str(filepath))
            extractor = InvocationExtractor(filepath.name)
            extractor.visit(tree)
            
            print(f"  Found {len(extractor.invocations)} invocations")
            
            # Compare each invocation
            for record in extractor.invocations:
                compared = self.comparator.compare_invocation(record)
                self.results.add_invocation(compared)
            
            return extractor.invocations
            
        except Exception as e:
            print(f"  Error analyzing {filepath.name}: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def analyze_all(self) -> AuditResults:
        """Analyze all orchestrator files"""
        orchestrator_files = [
            "core_orchestrator.py",
            "choreographer.py",
            "circuit_breaker.py",
            "report_assembly.py",
            "question_router.py"
        ]
        
        print("=" * 80)
        print("ORCHESTRATOR INVOCATION AUDIT")
        print("=" * 80)
        
        for filename in orchestrator_files:
            filepath = self.orchestrator_dir / filename
            if filepath.exists():
                self.analyze_orchestrator_file(filepath)
            else:
                print(f"\n✗ File not found: {filename}")
        
        return self.results


# ============================================================================
# REPORT GENERATORS
# ============================================================================

def generate_csv_report(results: AuditResults, output_path: Path):
    """Generate invocation_compatibility_matrix.csv"""
    print(f"\nGenerating CSV report: {output_path}")
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            "Orchestrator File",
            "Line Number",
            "Caller Function",
            "Target Module/Class/Method",
            "Expected Signature",
            "Actual Signature",
            "Status",
            "Recommended Remediation"
        ])
        
        # Data rows
        for record in results.invocations:
            writer.writerow(record.to_csv_row())
    
    print(f"✓ CSV report generated: {results.total_invocations} invocations")


def generate_markdown_report(results: AuditResults, output_path: Path):
    """Generate orchestrator_fixes.md"""
    print(f"\nGenerating Markdown report: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Orchestrator Fixes - Invocation Compatibility Analysis\n\n")
        f.write("**Generated:** " + str(Path.cwd()) + "\n\n")
        
        # Executive Summary
        f.write("## Executive Summary\n\n")
        f.write(f"- **Total Invocations Analyzed:** {results.total_invocations}\n")
        f.write(f"- **✓ OK:** {results.ok_count} ({results.ok_count/results.total_invocations*100:.1f}%)\n" if results.total_invocations > 0 else "- **✓ OK:** 0 (0.0%)\n")
        f.write(f"- **⚠ MISMATCH:** {results.mismatch_count} ({results.mismatch_count/results.total_invocations*100:.1f}%)\n" if results.total_invocations > 0 else "- **⚠ MISMATCH:** 0 (0.0%)\n")
        f.write(f"- **✗ MISSING:** {results.missing_count} ({results.missing_count/results.total_invocations*100:.1f}%)\n" if results.total_invocations > 0 else "- **✗ MISSING:** 0 (0.0%)\n")
        f.write(f"- **⊗ STATIC_VIOLATION:** {results.violation_count}\n\n")
        
        # Issues by File
        f.write("## Issues by Orchestrator File\n\n")
        
        by_file = defaultdict(list)
        for record in results.invocations:
            if record.status != "OK":
                by_file[record.orchestrator_file].append(record)
        
        if not by_file:
            f.write("**No issues found! All invocations are compatible.** ✓\n\n")
        else:
            for filename, records in sorted(by_file.items()):
                f.write(f"### {filename}\n\n")
                f.write(f"**Issues found:** {len(records)}\n\n")
                
                for record in sorted(records, key=lambda r: r.line_number):
                    f.write(f"#### Line {record.line_number}: `{record.caller_function}()`\n\n")
                    f.write(f"- **Target:** `{record.target_module}.{record.target_method}()`\n")
                    f.write(f"- **Status:** `{record.status}`\n")
                    f.write(f"- **Expected:** `args={record.expected_args}, kwargs={record.expected_kwargs}`\n")
                    f.write(f"- **Actual:** `{record.actual_signature}`\n")
                    f.write(f"- **Remediation:** {record.remediation}\n\n")
        
        # Detailed Recommendations
        f.write("## Detailed Recommendations\n\n")
        
        if results.missing_count > 0:
            f.write("### Missing Methods\n\n")
            f.write("The following methods are invoked but do not exist in their target adapters:\n\n")
            for record in results.invocations:
                if record.status == "MISSING":
                    f.write(f"- **{record.target_module}.{record.target_method}()** ")
                    f.write(f"(called from {record.orchestrator_file}:{record.line_number})\n")
                    f.write(f"  - Action: {record.remediation}\n\n")
        
        if results.mismatch_count > 0:
            f.write("### Signature Mismatches\n\n")
            f.write("The following invocations have signature compatibility issues:\n\n")
            for record in results.invocations:
                if record.status == "MISMATCH":
                    f.write(f"- **{record.target_module}.{record.target_method}()** ")
                    f.write(f"(called from {record.orchestrator_file}:{record.line_number})\n")
                    f.write(f"  - Issue: {record.remediation}\n")
                    f.write(f"  - Current signature: `{record.actual_signature}`\n\n")
        
        # Summary
        f.write("## Next Steps\n\n")
        if results.missing_count > 0 or results.mismatch_count > 0:
            f.write("1. Review each issue listed above\n")
            f.write("2. Update orchestrator call sites to match actual adapter signatures\n")
            f.write("3. Implement missing adapter methods where necessary\n")
            f.write("4. Re-run this audit to verify fixes\n")
        else:
            f.write("All orchestrator invocations are compatible with adapter signatures. No action required.\n")
    
    print(f"✓ Markdown report generated")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point"""
    # Paths
    base_dir = Path(__file__).parent
    orchestrator_dir = base_dir / "orchestrator"
    adapters_path = orchestrator_dir / "module_adapters.py"
    
    # Check paths
    if not orchestrator_dir.exists():
        print(f"✗ Orchestrator directory not found: {orchestrator_dir}")
        return 1
    
    if not adapters_path.exists():
        print(f"✗ Module adapters file not found: {adapters_path}")
        return 1
    
    # Run analysis
    analyzer = OrchestratorAnalyzer(orchestrator_dir, adapters_path)
    results = analyzer.analyze_all()
    
    # Generate reports
    print("\n" + "=" * 80)
    print("GENERATING REPORTS")
    print("=" * 80)
    
    csv_path = base_dir / "invocation_compatibility_matrix.csv"
    md_path = base_dir / "orchestrator_fixes.md"
    
    generate_csv_report(results, csv_path)
    generate_markdown_report(results, md_path)
    
    # Summary
    print("\n" + "=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)
    print(f"\n✓ Analyzed {results.total_invocations} invocations")
    print(f"✓ OK: {results.ok_count}")
    print(f"⚠ MISMATCH: {results.mismatch_count}")
    print(f"✗ MISSING: {results.missing_count}")
    print(f"⊗ STATIC_VIOLATION: {results.violation_count}")
    print(f"\n📄 Reports generated:")
    print(f"  - {csv_path}")
    print(f"  - {md_path}")
    
    return 0 if (results.mismatch_count == 0 and results.missing_count == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
