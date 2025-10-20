"""
Refactoring Validator - Method Signature Change Detection
===========================================================

Intercepts method signature changes in git staged changes by diffing
function signatures, then cross-references the dependency graph to identify
all adapters that call those methods. Fails the commit if adapters have not
been updated to match new signatures.

Author: FARFAN 3.0 Dev Team
Version: 1.0.0
Python: 3.10+
"""

import ast
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass
import difflib

from dependency_tracker import MethodSignature, DependencyGraph

logger = logging.getLogger(__name__)


@dataclass
class SignatureChange:
    """Represents a change in method signature"""
    file_path: str
    class_name: Optional[str]
    method_name: str
    old_signature: Optional[MethodSignature]
    new_signature: Optional[MethodSignature]
    change_type: str
    
    def __str__(self) -> str:
        class_prefix = f"{self.class_name}." if self.class_name else ""
        return f"{self.file_path}: {class_prefix}{self.method_name} ({self.change_type})"


class RefactoringValidator:
    """
    Validates that method signature changes don't break dependencies
    """
    
    def __init__(self, project_root: Path, dependency_graph: DependencyGraph):
        self.project_root = project_root
        self.graph = dependency_graph
        
    def get_staged_files(self) -> List[Path]:
        """Get list of staged Python files"""
        try:
            result = subprocess.run(
                ['git', 'diff', '--cached', '--name-only', '--diff-filter=ACMR'],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True
            )
            
            files = []
            for line in result.stdout.strip().split('\n'):
                if line.endswith('.py'):
                    file_path = self.project_root / line
                    if file_path.exists():
                        files.append(file_path)
            
            return files
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get staged files: {e}")
            return []
    
    def get_file_content(self, file_path: Path, staged: bool = False) -> Optional[str]:
        """Get file content from git (HEAD or staged)"""
        try:
            relative_path = file_path.relative_to(self.project_root)
            
            if staged:
                result = subprocess.run(
                    ['git', 'show', f':{relative_path}'],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    check=True
                )
            else:
                result = subprocess.run(
                    ['git', 'show', f'HEAD:{relative_path}'],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    check=True
                )
            
            return result.stdout
            
        except subprocess.CalledProcessError:
            if not staged:
                return None
            
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
    
    def extract_signatures(self, content: str, file_path: str) -> List[MethodSignature]:
        """Extract method signatures from file content"""
        try:
            tree = ast.parse(content, filename=file_path)
        except SyntaxError:
            logger.warning(f"Syntax error parsing {file_path}")
            return []
        
        signatures = []
        
        class SignatureExtractor(ast.NodeVisitor):
            def __init__(self):
                self.current_class = None
                self.signatures = []
            
            def visit_ClassDef(self, node):
                prev_class = self.current_class
                self.current_class = node.name
                self.generic_visit(node)
                self.current_class = prev_class
            
            def visit_FunctionDef(self, node):
                args = [arg.arg for arg in node.args.args]
                kwargs = [arg.arg for arg in node.args.kwonlyargs]
                
                return_annotation = None
                if node.returns:
                    return_annotation = ast.unparse(node.returns) if hasattr(ast, 'unparse') else None
                
                decorators = []
                for dec in node.decorator_list:
                    if isinstance(dec, ast.Name):
                        decorators.append(dec.id)
                
                sig = MethodSignature(
                    module=file_path,
                    class_name=self.current_class,
                    method_name=node.name,
                    args=args,
                    kwargs=kwargs,
                    return_annotation=return_annotation,
                    decorators=decorators,
                    line_number=node.lineno
                )
                
                self.signatures.append(sig)
        
        extractor = SignatureExtractor()
        extractor.visit(tree)
        
        return extractor.signatures
    
    def detect_signature_changes(self, file_path: Path) -> List[SignatureChange]:
        """Detect signature changes between HEAD and staged version"""
        relative_path = str(file_path.relative_to(self.project_root))
        
        old_content = self.get_file_content(file_path, staged=False)
        new_content = self.get_file_content(file_path, staged=True)
        
        if old_content is None:
            old_signatures = []
        else:
            old_signatures = self.extract_signatures(old_content, relative_path)
        
        if new_content is None:
            new_signatures = []
        else:
            new_signatures = self.extract_signatures(new_content, relative_path)
        
        old_sig_dict = {
            (sig.class_name, sig.method_name): sig
            for sig in old_signatures
        }
        
        new_sig_dict = {
            (sig.class_name, sig.method_name): sig
            for sig in new_signatures
        }
        
        changes = []
        
        for key, new_sig in new_sig_dict.items():
            old_sig = old_sig_dict.get(key)
            
            if old_sig is None:
                changes.append(SignatureChange(
                    file_path=relative_path,
                    class_name=new_sig.class_name,
                    method_name=new_sig.method_name,
                    old_signature=None,
                    new_signature=new_sig,
                    change_type='added'
                ))
            
            elif not self._signatures_match(old_sig, new_sig):
                changes.append(SignatureChange(
                    file_path=relative_path,
                    class_name=new_sig.class_name,
                    method_name=new_sig.method_name,
                    old_signature=old_sig,
                    new_signature=new_sig,
                    change_type='modified'
                ))
        
        for key, old_sig in old_sig_dict.items():
            if key not in new_sig_dict:
                changes.append(SignatureChange(
                    file_path=relative_path,
                    class_name=old_sig.class_name,
                    method_name=old_sig.method_name,
                    old_signature=old_sig,
                    new_signature=None,
                    change_type='removed'
                ))
        
        return changes
    
    def _signatures_match(self, sig1: MethodSignature, sig2: MethodSignature) -> bool:
        """Check if two signatures are equivalent"""
        return (
            sig1.args == sig2.args and
            sig1.kwargs == sig2.kwargs and
            sig1.return_annotation == sig2.return_annotation
        )
    
    def find_affected_callers(self, change: SignatureChange) -> List[Tuple[str, int, str]]:
        """
        Find all call sites affected by a signature change
        
        Returns: List of (file_path, line_number, method_name) tuples
        """
        affected = []
        
        method_key = f"{change.class_name or ''}.{change.method_name}"
        
        for file_path, call_sites in self.graph.call_sites.items():
            for call in call_sites:
                call_key = f"{call.callee_class or ''}.{call.callee_method}"
                
                if call_key == method_key or call.callee_method == change.method_name:
                    target_file = None
                    if call.callee_module:
                        for imp in self.graph.imports.get(file_path, []):
                            if call.callee_module in imp.imported_names or \
                               call.callee_module == imp.imported_module:
                                if change.file_path in imp.imported_module or \
                                   change.file_path.replace('/', '.').replace('.py', '') in imp.imported_module:
                                    target_file = change.file_path
                                    break
                    
                    if target_file or self._could_reference(file_path, change.file_path):
                        affected.append((
                            file_path,
                            call.caller_line,
                            call.caller_method
                        ))
        
        return affected
    
    def _could_reference(self, caller_file: str, callee_file: str) -> bool:
        """Check if caller_file could reference callee_file"""
        if caller_file == callee_file:
            return True
        
        if callee_file in self.graph.get_dependencies(caller_file):
            return True
        
        return False
    
    def validate_staged_changes(self) -> Tuple[bool, List[str]]:
        """
        Validate all staged changes for signature compatibility
        
        Returns: (is_valid, error_messages)
        """
        staged_files = self.get_staged_files()
        
        if not staged_files:
            logger.info("No staged Python files to validate")
            return True, []
        
        logger.info(f"Validating {len(staged_files)} staged files...")
        
        all_changes = []
        for file_path in staged_files:
            changes = self.detect_signature_changes(file_path)
            all_changes.extend(changes)
        
        if not all_changes:
            logger.info("No signature changes detected")
            return True, []
        
        logger.info(f"Detected {len(all_changes)} signature changes")
        
        errors = []
        staged_file_strs = {str(f.relative_to(self.project_root)) for f in staged_files}
        
        for change in all_changes:
            if change.change_type in ('modified', 'removed'):
                affected_callers = self.find_affected_callers(change)
                
                if affected_callers:
                    unstaged_affected = [
                        (f, line, method)
                        for f, line, method in affected_callers
                        if f not in staged_file_strs
                    ]
                    
                    if unstaged_affected:
                        class_prefix = f"{change.class_name}." if change.class_name else ""
                        error_msg = (
                            f"Signature change in {change.file_path}:{class_prefix}{change.method_name} "
                            f"affects {len(unstaged_affected)} call site(s) in unstaged files:\n"
                        )
                        
                        for caller_file, line, caller_method in unstaged_affected[:5]:
                            error_msg += f"  - {caller_file}:{line} in {caller_method}\n"
                        
                        if len(unstaged_affected) > 5:
                            error_msg += f"  ... and {len(unstaged_affected) - 5} more\n"
                        
                        error_msg += "  → Update these files or stage them with this commit"
                        
                        errors.append(error_msg)
        
        is_valid = len(errors) == 0
        
        return is_valid, errors
    
    def print_signature_diff(self, change: SignatureChange) -> str:
        """Generate human-readable diff of signature change"""
        if change.change_type == 'added':
            return f"+ {change.new_signature.args}"
        
        elif change.change_type == 'removed':
            return f"- {change.old_signature.args}"
        
        else:
            old_sig_str = f"{change.old_signature.method_name}({', '.join(change.old_signature.args)})"
            new_sig_str = f"{change.new_signature.method_name}({', '.join(change.new_signature.args)})"
            
            if old_sig_str == new_sig_str:
                old_sig_str += f" -> {change.old_signature.return_annotation}"
                new_sig_str += f" -> {change.new_signature.return_annotation}"
            
            return f"- {old_sig_str}\n+ {new_sig_str}"


def validate_refactoring(project_root: Path, graph: DependencyGraph) -> int:
    """
    Main entry point for pre-commit validation
    
    Returns: 0 if valid, 1 if invalid
    """
    validator = RefactoringValidator(project_root, graph)
    
    is_valid, errors = validator.validate_staged_changes()
    
    if is_valid:
        logger.info("✅ Refactoring validation passed")
        return 0
    
    else:
        print("\n❌ COMMIT BLOCKED: Signature changes affect unstaged files\n")
        
        for error in errors:
            print(error)
            print()
        
        print("💡 To fix:")
        print("   1. Update the affected files to match new signatures")
        print("   2. Stage those files with: git add <files>")
        print("   3. Or revert signature changes")
        print()
        
        return 1


if __name__ == '__main__':
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    project_root = Path(__file__).parent
    
    try:
        graph = DependencyGraph.load(project_root / 'dependency_graph_baseline.json')
    except FileNotFoundError:
        print("⚠️  Dependency graph baseline not found. Run dependency_tracker.py first.")
        sys.exit(0)
    
    sys.exit(validate_refactoring(project_root, graph))
