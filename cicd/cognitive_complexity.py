#!/usr/bin/env python3
"""
Cognitive Complexity Checker for FARFAN 3.0
============================================

Analyzes cognitive complexity of functions and enforces thresholds per SIN_CARRETA doctrine.

The cognitive complexity metric measures how difficult code is to understand, based on:
- Nesting depth
- Control flow complexity (if, for, while, etc.)
- Boolean operators
- Recursive calls

Rationale:
---------
High cognitive complexity indicates code that is difficult to:
1. Understand and review
2. Test thoroughly  
3. Maintain over time
4. Verify for determinism
5. Audit for security

SIN_CARRETA doctrine requires low cognitive complexity to ensure:
- Clear audit trails
- Predictable behavior
- Easy verification of deterministic properties
- Reduced bug surface area

Thresholds:
----------
- Simple functions: <= 5 (ideal)
- Moderate functions: 6-10 (acceptable)
- Complex functions: 11-15 (needs refactoring)
- Very complex: > 15 (must refactor)

Usage:
    python cicd/cognitive_complexity.py --path src/
    python cicd/cognitive_complexity.py --file src/orchestrator/choreographer.py
    python cicd/cognitive_complexity.py --threshold 10
    python cicd/cognitive_complexity.py --report complexity_report.json

Author: CI/CD Team
Version: 1.0.0
"""

import ast
import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ComplexityResult:
    """Result of cognitive complexity analysis."""
    file: str
    function: str
    line_number: int
    complexity: int
    threshold: int
    status: str  # ok, warning, error
    suggestions: List[str]


class CognitiveComplexityChecker(ast.NodeVisitor):
    """Calculate cognitive complexity of Python functions."""
    
    def __init__(self, threshold: int = 15):
        self.threshold = threshold
        self.results: List[ComplexityResult] = []
        self.current_file: Optional[str] = None
        self.nesting_level = 0
    
    def check_file(self, file_path: Path):
        """Check cognitive complexity of all functions in a file."""
        self.current_file = str(file_path)
        logger.info(f"Analyzing {file_path}")
        
        try:
            with open(file_path) as f:
                tree = ast.parse(f.read(), filename=str(file_path))
            self.visit(tree)
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
    
    def check_directory(self, dir_path: Path, recursive: bool = True):
        """Check all Python files in a directory."""
        pattern = "**/*.py" if recursive else "*.py"
        
        for file_path in dir_path.glob(pattern):
            if "test" in str(file_path) or "__pycache__" in str(file_path):
                continue
            self.check_file(file_path)
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Visit function definition and calculate complexity."""
        complexity = self._calculate_complexity(node)
        
        status = "ok"
        suggestions = []
        
        if complexity > self.threshold:
            status = "error"
            suggestions.append(f"Refactor to reduce complexity from {complexity} to <={self.threshold}")
            suggestions.append("Consider extracting helper functions")
            suggestions.append("Simplify conditional logic")
        elif complexity > 10:
            status = "warning"
            suggestions.append("Consider refactoring for better maintainability")
        
        result = ComplexityResult(
            file=self.current_file or "unknown",
            function=node.name,
            line_number=node.lineno,
            complexity=complexity,
            threshold=self.threshold,
            status=status,
            suggestions=suggestions
        )
        
        self.results.append(result)
        
        # Continue visiting child nodes
        self.generic_visit(node)
    
    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cognitive complexity of a function."""
        complexity = 0
        nesting = 0
        
        def visit_node(n, depth):
            nonlocal complexity
            
            # Control flow structures increase complexity
            if isinstance(n, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1 + depth  # Base + nesting penalty
                
                # Recursively visit children with increased depth
                for child in ast.iter_child_nodes(n):
                    if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                        visit_node(child, depth + 1)
                    else:
                        for subchild in ast.walk(child):
                            if isinstance(subchild, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                                visit_node(subchild, depth + 1)
            
            # Boolean operators increase complexity
            elif isinstance(n, ast.BoolOp):
                complexity += len(n.values) - 1
            
            # Ternary expressions increase complexity
            elif isinstance(n, ast.IfExp):
                complexity += 1
            
            # Recursion increases complexity
            elif isinstance(n, ast.Call):
                if isinstance(n.func, ast.Name) and n.func.id == node.name:
                    complexity += 1
        
        for child in ast.walk(node):
            visit_node(child, nesting)
        
        return complexity
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of complexity analysis."""
        if not self.results:
            return {
                "total_functions": 0,
                "ok": 0,
                "warnings": 0,
                "errors": 0,
                "avg_complexity": 0,
                "max_complexity": 0
            }
        
        complexities = [r.complexity for r in self.results]
        
        return {
            "total_functions": len(self.results),
            "ok": sum(1 for r in self.results if r.status == "ok"),
            "warnings": sum(1 for r in self.results if r.status == "warning"),
            "errors": sum(1 for r in self.results if r.status == "error"),
            "avg_complexity": sum(complexities) / len(complexities),
            "max_complexity": max(complexities),
            "threshold": self.threshold
        }
    
    def print_report(self):
        """Print complexity report to console."""
        summary = self.get_summary()
        
        print("\n" + "=" * 80)
        print("COGNITIVE COMPLEXITY REPORT")
        print("=" * 80)
        print(f"Total Functions: {summary['total_functions']}")
        print(f"OK: {summary['ok']} | Warnings: {summary['warnings']} | Errors: {summary['errors']}")
        print(f"Average Complexity: {summary['avg_complexity']:.1f}")
        print(f"Max Complexity: {summary['max_complexity']}")
        print(f"Threshold: {summary['threshold']}")
        print()
        
        # Show errors first
        errors = [r for r in self.results if r.status == "error"]
        if errors:
            print(f"❌ ERRORS ({len(errors)} functions exceed threshold):")
            for result in sorted(errors, key=lambda r: r.complexity, reverse=True)[:10]:
                print(f"  {result.file}:{result.line_number} - {result.function}()")
                print(f"    Complexity: {result.complexity} (threshold: {result.threshold})")
                for suggestion in result.suggestions:
                    print(f"    → {suggestion}")
            print()
        
        # Show warnings
        warnings = [r for r in self.results if r.status == "warning"]
        if warnings:
            print(f"⚠️  WARNINGS ({len(warnings)} functions):")
            for result in sorted(warnings, key=lambda r: r.complexity, reverse=True)[:5]:
                print(f"  {result.file}:{result.line_number} - {result.function}()")
                print(f"    Complexity: {result.complexity}")
            print()
        
        print("=" * 80)
    
    def save_report(self, output_path: Path):
        """Save report to JSON file."""
        report = {
            "summary": self.get_summary(),
            "results": [asdict(r) for r in self.results]
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Check cognitive complexity of Python code",
        epilog="""
SIN_CARRETA Doctrine Rationale:
-------------------------------
Cognitive complexity directly impacts:
1. Audit Trail Clarity - Complex code is harder to trace
2. Determinism Verification - Difficult to verify all paths
3. Security Review - More places for bugs to hide
4. Maintenance Cost - Higher complexity = higher cost
5. Test Coverage - More paths to test thoroughly

Keep functions simple for better system reliability.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("src"),
        help="Directory to analyze (default: src)"
    )
    parser.add_argument(
        "--file",
        type=Path,
        help="Single file to analyze"
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=15,
        help="Maximum allowed complexity (default: 15)"
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Save JSON report to file"
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Don't recurse into subdirectories"
    )
    
    args = parser.parse_args()
    
    checker = CognitiveComplexityChecker(threshold=args.threshold)
    
    if args.file:
        checker.check_file(args.file)
    else:
        checker.check_directory(args.path, recursive=not args.no_recursive)
    
    checker.print_report()
    
    if args.report:
        checker.save_report(args.report)
    
    summary = checker.get_summary()
    return 1 if summary["errors"] > 0 else 0


if __name__ == "__main__":
    exit(main())
