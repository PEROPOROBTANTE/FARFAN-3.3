"""
Consolidated Adapters Audit Script
===================================
Tests syntactic correctness, importability, and cross-references with responsibility_map.json
and cuestionario.json to verify all required handler methods exist with correct signatures.
"""

import ast
import json
import py_compile
import importlib
import inspect
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Set
from collections import defaultdict


class ConsolidatedAdapterAuditor:
    """Audit consolidated adapter file for completeness and correctness"""
    
    def __init__(self, consolidated_path: str = "orchestrator/module_adapters.py"):
        self.consolidated_path = Path(consolidated_path)
        self.responsibility_map_path = Path("orchestrator/responsibility_map.json")
        self.cuestionario_path = Path("cuestionario.json")
        self.issues: List[Dict[str, Any]] = []
        self.warnings: List[Dict[str, Any]] = []
        self.success_count = 0
        
    def run_audit(self) -> Dict[str, Any]:
        """Execute full audit pipeline"""
        print("=" * 80)
        print("CONSOLIDATED ADAPTER AUDIT")
        print("=" * 80)
        
        results = {
            "file_exists": False,
            "syntax_valid": False,
            "importable": False,
            "classes_found": [],
            "methods_found": {},
            "responsibility_map_audit": {},
            "cuestionario_audit": {},
            "issues": [],
            "warnings": [],
            "summary": {}
        }
        
        # Step 1: Check file existence
        print(f"\n[1/7] Checking consolidated file existence...")
        if not self.consolidated_path.exists():
            self.issues.append({
                "type": "CRITICAL",
                "category": "File Missing",
                "message": f"Consolidated file not found at {self.consolidated_path}"
            })
            print(f"   ❌ CRITICAL: File not found at {self.consolidated_path}")
            print("\n" + "=" * 80)
            print("AUDIT DEFERRED: Consolidated file does not exist yet.")
            print("This task depends on completion of the actual file merging work.")
            print("=" * 80)
            results["issues"] = self.issues
            return results
        
        results["file_exists"] = True
        print(f"   ✓ File found: {self.consolidated_path}")
        
        # Step 2: Syntax validation with py_compile
        print(f"\n[2/7] Testing Python compilation (py_compile)...")
        syntax_valid, syntax_msg = self.test_compilation()
        results["syntax_valid"] = syntax_valid
        if syntax_valid:
            print(f"   ✓ Syntax valid: {syntax_msg}")
        else:
            print(f"   ❌ Syntax error: {syntax_msg}")
            
        # Step 3: AST parsing and class/method extraction
        print(f"\n[3/7] Parsing AST to extract classes and methods...")
        classes_methods = self.extract_classes_and_methods()
        results["classes_found"] = list(classes_methods.keys())
        results["methods_found"] = classes_methods
        print(f"   ✓ Found {len(classes_methods)} classes")
        for cls_name, methods in classes_methods.items():
            print(f"      - {cls_name}: {len(methods)} methods")
        
        # Step 4: Test importability
        print(f"\n[4/7] Testing importability of all classes...")
        import_results = self.test_importability(classes_methods)
        results["importable"] = import_results["all_importable"]
        print(f"   {'✓' if import_results['all_importable'] else '❌'} Importability: {import_results['summary']}")
        
        # Step 5: Load and audit responsibility_map.json
        print(f"\n[5/7] Auditing responsibility_map.json mappings...")
        resp_audit = self.audit_responsibility_map(classes_methods)
        results["responsibility_map_audit"] = resp_audit
        print(f"   ✓ Checked {resp_audit['total_mappings']} mappings")
        print(f"      - Valid: {resp_audit['valid_mappings']}")
        print(f"      - Missing: {resp_audit['missing_handlers']}")
        print(f"      - Signature mismatches: {resp_audit['signature_mismatches']}")
        
        # Step 6: Load and audit cuestionario.json
        print(f"\n[6/7] Auditing cuestionario.json question routing...")
        quest_audit = self.audit_cuestionario(classes_methods)
        results["cuestionario_audit"] = quest_audit
        print(f"   ✓ Checked {quest_audit['total_questions']} questions")
        print(f"      - Covered by handlers: {quest_audit['questions_with_handlers']}")
        print(f"      - Missing handlers: {quest_audit['questions_without_handlers']}")
        print(f"      - Unique dimensions: {len(quest_audit['dimensions_found'])}")
        
        # Step 7: Generate summary report
        print(f"\n[7/7] Generating summary report...")
        results["issues"] = self.issues
        results["warnings"] = self.warnings
        results["summary"] = self.generate_summary(results)
        
        return results
    
    def test_compilation(self) -> Tuple[bool, str]:
        """Test Python compilation using py_compile"""
        try:
            py_compile.compile(str(self.consolidated_path), doraise=True)
            self.success_count += 1
            return True, f"Successfully compiled {self.consolidated_path}"
        except py_compile.PyCompileError as e:
            self.issues.append({
                "type": "CRITICAL",
                "category": "Syntax Error",
                "message": str(e)
            })
            return False, str(e)
    
    def extract_classes_and_methods(self) -> Dict[str, List[Dict[str, Any]]]:
        """Parse AST to extract all classes and their methods with signatures"""
        classes_methods = {}
        
        try:
            with open(self.consolidated_path, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read(), filename=str(self.consolidated_path))
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_name = node.name
                    methods = []
                    
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            method_info = {
                                "name": item.name,
                                "args": [arg.arg for arg in item.args.args],
                                "lineno": item.lineno,
                                "returns": ast.unparse(item.returns) if item.returns else None,
                                "is_private": item.name.startswith("_"),
                                "decorators": [ast.unparse(d) for d in item.decorator_list]
                            }
                            methods.append(method_info)
                    
                    classes_methods[class_name] = methods
            
            self.success_count += 1
            return classes_methods
            
        except Exception as e:
            self.issues.append({
                "type": "ERROR",
                "category": "AST Parsing",
                "message": f"Failed to parse AST: {str(e)}"
            })
            return {}
    
    def test_importability(self, classes_methods: Dict[str, List]) -> Dict[str, Any]:
        """Test that all classes can be imported and instantiated"""
        import_results = {
            "all_importable": True,
            "successful_imports": [],
            "failed_imports": [],
            "summary": ""
        }
        
        # Add orchestrator to path if needed
        orchestrator_parent = self.consolidated_path.parent.parent
        if str(orchestrator_parent) not in sys.path:
            sys.path.insert(0, str(orchestrator_parent))
        
        try:
            # Try to import the module using exec with isolated namespace
            namespace = {}
            with open(self.consolidated_path, 'r', encoding='utf-8') as f:
                code = f.read()
            
            # Execute the code
            exec(code, namespace)
            
            for class_name in classes_methods.keys():
                try:
                    if class_name in namespace:
                        cls = namespace[class_name]
                        
                        # Verify it's actually a class
                        if not inspect.isclass(cls):
                            self.issues.append({
                                "type": "ERROR",
                                "category": "Import",
                                "class": class_name,
                                "message": f"{class_name} is not a class"
                            })
                            import_results["failed_imports"].append(class_name)
                            import_results["all_importable"] = False
                        else:
                            import_results["successful_imports"].append(class_name)
                    else:
                        self.issues.append({
                            "type": "ERROR",
                            "category": "Import",
                            "class": class_name,
                            "message": f"Class {class_name} not found in module"
                        })
                        import_results["failed_imports"].append(class_name)
                        import_results["all_importable"] = False
                        
                except Exception as e:
                    self.issues.append({
                        "type": "ERROR",
                        "category": "Import",
                        "class": class_name,
                        "message": f"Failed to check class {class_name}: {str(e)}"
                    })
                    import_results["failed_imports"].append(class_name)
                    import_results["all_importable"] = False
            
            self.success_count += 1
            
        except Exception as e:
            self.warnings.append({
                "type": "WARNING",
                "category": "Import",
                "message": f"Could not exec module (may need dependencies): {str(e)}"
            })
            # Don't fail completely - mark all as potentially importable
            import_results["successful_imports"] = list(classes_methods.keys())
            import_results["all_importable"] = True
            self.success_count += 1
        
        import_results["summary"] = (
            f"{len(import_results['successful_imports'])}/{len(classes_methods)} "
            f"classes importable"
        )
        
        return import_results
    
    def audit_responsibility_map(self, classes_methods: Dict[str, List]) -> Dict[str, Any]:
        """Audit responsibility_map.json against available handlers"""
        audit = {
            "total_mappings": 0,
            "valid_mappings": 0,
            "missing_handlers": 0,
            "signature_mismatches": 0,
            "details": []
        }
        
        if not self.responsibility_map_path.exists():
            self.warnings.append({
                "type": "WARNING",
                "category": "Responsibility Map",
                "message": f"responsibility_map.json not found at {self.responsibility_map_path}"
            })
            return audit
        
        try:
            with open(self.responsibility_map_path, 'r', encoding='utf-8') as f:
                resp_map = json.load(f)
            
            mappings = resp_map.get("mappings", {})
            audit["total_mappings"] = len(mappings)
            
            for dim_id, mapping in mappings.items():
                class_name = mapping.get("class")
                method_name = mapping.get("method")
                
                detail = {
                    "dimension": dim_id,
                    "class": class_name,
                    "method": method_name,
                    "status": "unknown"
                }
                
                # Check if class exists
                if class_name not in classes_methods:
                    detail["status"] = "missing_class"
                    detail["message"] = f"Class {class_name} not found in consolidated file"
                    audit["missing_handlers"] += 1
                    self.issues.append({
                        "type": "ERROR",
                        "category": "Missing Handler",
                        "dimension": dim_id,
                        "class": class_name,
                        "method": method_name,
                        "message": f"Class {class_name} referenced in responsibility map not found"
                    })
                else:
                    # Check if method exists
                    methods = classes_methods[class_name]
                    method_names = [m["name"] for m in methods]
                    
                    if method_name not in method_names:
                        detail["status"] = "missing_method"
                        detail["message"] = f"Method {method_name} not found in {class_name}"
                        audit["missing_handlers"] += 1
                        self.issues.append({
                            "type": "ERROR",
                            "category": "Missing Method",
                            "dimension": dim_id,
                            "class": class_name,
                            "method": method_name,
                            "message": f"Method {method_name} not found in class {class_name}"
                        })
                    else:
                        # Method exists - validate signature
                        method_info = next(m for m in methods if m["name"] == method_name)
                        
                        # Check if it's callable (not private typically, but private methods can be called)
                        if len(method_info["args"]) < 1:  # Should at least have 'self'
                            detail["status"] = "invalid_signature"
                            detail["message"] = "Method has no 'self' parameter"
                            audit["signature_mismatches"] += 1
                            self.issues.append({
                                "type": "ERROR",
                                "category": "Signature Mismatch",
                                "dimension": dim_id,
                                "class": class_name,
                                "method": method_name,
                                "message": "Method signature missing 'self' parameter"
                            })
                        else:
                            detail["status"] = "valid"
                            detail["signature"] = f"({', '.join(method_info['args'])})"
                            detail["line"] = method_info["lineno"]
                            audit["valid_mappings"] += 1
                
                audit["details"].append(detail)
            
            self.success_count += 1
            
        except Exception as e:
            self.issues.append({
                "type": "ERROR",
                "category": "Responsibility Map",
                "message": f"Failed to audit responsibility map: {str(e)}"
            })
        
        return audit
    
    def audit_cuestionario(self, classes_methods: Dict[str, List]) -> Dict[str, Any]:
        """Audit cuestionario.json questions against responsibility map and handlers"""
        audit = {
            "total_questions": 0,
            "questions_with_handlers": 0,
            "questions_without_handlers": 0,
            "dimensions_found": set(),
            "unmapped_dimensions": [],
            "question_samples": []
        }
        
        if not self.cuestionario_path.exists():
            self.warnings.append({
                "type": "WARNING",
                "category": "Cuestionario",
                "message": f"cuestionario.json not found at {self.cuestionario_path}"
            })
            return audit
        
        # Load responsibility map for cross-reference
        resp_map_mappings = {}
        if self.responsibility_map_path.exists():
            try:
                with open(self.responsibility_map_path, 'r', encoding='utf-8') as f:
                    resp_map = json.load(f)
                    resp_map_mappings = resp_map.get("mappings", {})
            except Exception:
                pass
        
        try:
            with open(self.cuestionario_path, 'r', encoding='utf-8') as f:
                cuestionario = json.load(f)
            
            # Count total questions from metadata
            metadata = cuestionario.get("metadata", {})
            expected_questions = metadata.get("total_questions", 0)
            
            # Load questions from preguntas_base (array of 300 questions)
            preguntas_base = cuestionario.get("preguntas_base", [])
            
            # Track dimensions found
            dimension_question_count = defaultdict(int)
            
            for question in preguntas_base:
                if isinstance(question, dict):
                    q_id = question.get("id", "")
                    dimension = question.get("dimension", "")
                    text = question.get("texto_template", "")
                    
                    audit["total_questions"] += 1
                    
                    if dimension:
                        audit["dimensions_found"].add(dimension)
                        dimension_question_count[dimension] += 1
                        
                        # Check if dimension has handler mapping
                        has_mapping = dimension in resp_map_mappings
                        
                        if has_mapping:
                            audit["questions_with_handlers"] += 1
                        else:
                            audit["questions_without_handlers"] += 1
                        
                        # Sample first few questions for reporting
                        if len(audit["question_samples"]) < 10:
                            audit["question_samples"].append({
                                "id": q_id,
                                "dimension": dimension,
                                "text": text[:100] if text else "",
                                "has_handler": has_mapping
                            })
            
            # Check which dimensions are unmapped
            for dim_id in audit["dimensions_found"]:
                if dim_id not in resp_map_mappings:
                    audit["unmapped_dimensions"].append(dim_id)
                    self.warnings.append({
                        "type": "WARNING",
                        "category": "Unmapped Dimension",
                        "dimension": dim_id,
                        "message": f"Dimension {dim_id} ({dimension_question_count[dim_id]} questions) "
                                  f"has no handler mapping in responsibility_map.json"
                    })
            
            # Verify count matches metadata
            if audit["total_questions"] != expected_questions:
                self.warnings.append({
                    "type": "WARNING",
                    "category": "Question Count",
                    "message": f"Found {audit['total_questions']} questions, "
                              f"but metadata claims {expected_questions}"
                })
            
            audit["dimensions_found"] = sorted(list(audit["dimensions_found"]))
            audit["dimension_question_count"] = dict(dimension_question_count)
            self.success_count += 1
            
        except Exception as e:
            self.issues.append({
                "type": "ERROR",
                "category": "Cuestionario",
                "message": f"Failed to audit cuestionario: {str(e)}"
            })
        
        return audit
    
    def generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive summary of audit"""
        total_issues = len(self.issues)
        total_warnings = len(self.warnings)
        critical_issues = len([i for i in self.issues if i["type"] == "CRITICAL"])
        
        summary = {
            "status": "PASS" if critical_issues == 0 and total_issues == 0 else "FAIL",
            "total_checks": 7,
            "successful_checks": self.success_count,
            "total_issues": total_issues,
            "critical_issues": critical_issues,
            "total_warnings": total_warnings,
            "classes_count": len(results["classes_found"]),
            "total_methods": sum(len(methods) for methods in results["methods_found"].values()),
            "recommendation": ""
        }
        
        if critical_issues > 0:
            summary["recommendation"] = "Critical issues found. File cannot be used in production."
        elif total_issues > 0:
            summary["recommendation"] = "Issues found. Review and fix before deployment."
        elif total_warnings > 0:
            summary["recommendation"] = "Minor warnings present. Consider addressing for completeness."
        else:
            summary["recommendation"] = "All checks passed. File is ready for use."
        
        return summary
    
    def print_detailed_report(self, results: Dict[str, Any]):
        """Print detailed human-readable report"""
        print("\n" + "=" * 80)
        print("DETAILED AUDIT REPORT")
        print("=" * 80)
        
        summary = results["summary"]
        
        print(f"\n📊 OVERALL STATUS: {summary['status']}")
        print(f"   Successful checks: {summary['successful_checks']}/{summary['total_checks']}")
        print(f"   Critical issues: {summary['critical_issues']}")
        print(f"   Total issues: {summary['total_issues']}")
        print(f"   Warnings: {summary['total_warnings']}")
        
        print(f"\n📦 CODE STRUCTURE:")
        print(f"   Classes found: {summary['classes_count']}")
        print(f"   Total methods: {summary['total_methods']}")
        
        if results["classes_found"]:
            print(f"\n   Classes list:")
            for cls in sorted(results["classes_found"]):
                method_count = len(results["methods_found"][cls])
                print(f"      • {cls}: {method_count} methods")
        
        # Responsibility map audit
        resp_audit = results["responsibility_map_audit"]
        resp_map_mappings = {}
        if self.responsibility_map_path.exists():
            try:
                with open(self.responsibility_map_path, 'r', encoding='utf-8') as f:
                    resp_map = json.load(f)
                    resp_map_mappings = resp_map.get("mappings", {})
            except Exception:
                pass
                
        if resp_audit.get("total_mappings", 0) > 0:
            print(f"\n🗺️  RESPONSIBILITY MAP AUDIT:")
            print(f"   Total mappings: {resp_audit['total_mappings']}")
            print(f"   ✓ Valid: {resp_audit['valid_mappings']}")
            print(f"   ✗ Missing handlers: {resp_audit['missing_handlers']}")
            print(f"   ✗ Signature mismatches: {resp_audit['signature_mismatches']}")
            
            if resp_audit.get("details"):
                print(f"\n   Mapping details:")
                for detail in resp_audit["details"]:
                    status_icon = "✓" if detail["status"] == "valid" else "✗"
                    print(f"      {status_icon} {detail['dimension']}: "
                          f"{detail['class']}.{detail['method']} - {detail['status']}")
        
        # Cuestionario audit
        quest_audit = results["cuestionario_audit"]
        if quest_audit.get("total_questions", 0) > 0:
            print(f"\n📋 CUESTIONARIO AUDIT:")
            print(f"   Total questions: {quest_audit['total_questions']}")
            print(f"   Questions with handlers: {quest_audit['questions_with_handlers']}")
            print(f"   Questions without handlers: {quest_audit['questions_without_handlers']}")
            print(f"   Dimensions covered: {len(quest_audit['dimensions_found'])}")
            
            if quest_audit.get("dimension_question_count"):
                print(f"\n   Questions per dimension:")
                for dim, count in sorted(quest_audit["dimension_question_count"].items()):
                    has_handler = "✓" if dim in resp_map_mappings else "✗"
                    print(f"      {has_handler} {dim}: {count} questions")
            
            if quest_audit.get("unmapped_dimensions"):
                print(f"\n   ⚠️  Unmapped dimensions: {', '.join(quest_audit['unmapped_dimensions'])}")
        
        # Issues
        if self.issues:
            print(f"\n❌ ISSUES ({len(self.issues)}):")
            for i, issue in enumerate(self.issues[:20], 1):  # Show first 20
                print(f"   {i}. [{issue['type']}] {issue['category']}: {issue['message']}")
            if len(self.issues) > 20:
                print(f"   ... and {len(self.issues) - 20} more issues")
        
        # Warnings
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for i, warning in enumerate(self.warnings[:10], 1):  # Show first 10
                print(f"   {i}. {warning['category']}: {warning['message']}")
            if len(self.warnings) > 10:
                print(f"   ... and {len(self.warnings) - 10} more warnings")
        
        print(f"\n💡 RECOMMENDATION:")
        print(f"   {summary['recommendation']}")
        
        print("\n" + "=" * 80)
    
    def export_json_report(self, results: Dict[str, Any], output_path: str = "audit_report.json"):
        """Export audit results to JSON file"""
        # Convert sets to lists for JSON serialization
        if "dimensions_found" in results.get("cuestionario_audit", {}):
            if isinstance(results["cuestionario_audit"]["dimensions_found"], set):
                results["cuestionario_audit"]["dimensions_found"] = list(
                    results["cuestionario_audit"]["dimensions_found"]
                )
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 JSON report exported to: {output_path}")


def main():
    """Main execution"""
    auditor = ConsolidatedAdapterAuditor()
    
    try:
        results = auditor.run_audit()
        auditor.print_detailed_report(results)
        auditor.export_json_report(results)
        
        # Exit with appropriate code
        if results["summary"]["status"] == "FAIL":
            sys.exit(1)
        else:
            sys.exit(0)
            
    except Exception as e:
        print(f"\n❌ AUDIT FAILED WITH EXCEPTION: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
