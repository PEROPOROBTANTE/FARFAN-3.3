#!/usr/bin/env python3
"""
AST-based API Surface Extractor for FARFAN 3.0 Modules
=======================================================

Extracts comprehensive API information from source modules including:
- All classes (public and private)
- All methods with decorators (staticmethod, classmethod, regular methods)
- Full signatures with type hints and default values
- Docstrings
- Standalone functions

Generates:
1. source_modules_inventory.json - Complete API catalog
2. baseline_metrics.json - Aggregated metrics for 95% preservation validation
"""

import ast
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class MethodInfo:
    """Detailed method information"""
    name: str
    decorator: Optional[str] = None  # staticmethod, classmethod, or None
    signature: str = ""
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    return_type: Optional[str] = None
    docstring: Optional[str] = None
    is_public: bool = True
    lineno: int = 0


@dataclass
class ClassInfo:
    """Detailed class information"""
    name: str
    docstring: Optional[str] = None
    methods: List[MethodInfo] = field(default_factory=list)
    base_classes: List[str] = field(default_factory=list)
    is_public: bool = True
    lineno: int = 0


@dataclass
class FunctionInfo:
    """Detailed standalone function information"""
    name: str
    signature: str = ""
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    return_type: Optional[str] = None
    docstring: Optional[str] = None
    is_public: bool = True
    lineno: int = 0


@dataclass
class ModuleInfo:
    """Complete module API surface"""
    module_name: str
    classes: List[ClassInfo] = field(default_factory=list)
    functions: List[FunctionInfo] = field(default_factory=list)
    docstring: Optional[str] = None


class APIExtractor(ast.NodeVisitor):
    """AST visitor to extract complete API surface"""
    
    def __init__(self):
        self.module_info = ModuleInfo(module_name="")
        self.current_class: Optional[ClassInfo] = None
    
    def visit_Module(self, node: ast.Module) -> None:
        """Extract module docstring"""
        self.module_info.docstring = ast.get_docstring(node)
        self.generic_visit(node)
    
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Extract class information"""
        class_info = ClassInfo(
            name=node.name,
            docstring=ast.get_docstring(node),
            base_classes=[self._format_expr(base) for base in node.bases],
            is_public=not node.name.startswith('_'),
            lineno=node.lineno
        )
        
        # Temporarily set current class for method extraction
        previous_class = self.current_class
        self.current_class = class_info
        
        # Visit class body to extract methods
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                self.visit_FunctionDef(item)
        
        self.module_info.classes.append(class_info)
        self.current_class = previous_class
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Extract function or method information"""
        # Determine if this is a method or standalone function
        is_method = self.current_class is not None
        
        # Extract decorator information
        decorator = None
        for dec in node.decorator_list:
            dec_name = self._format_expr(dec)
            if dec_name in ('staticmethod', 'classmethod'):
                decorator = dec_name
                break
        
        # Extract parameters
        parameters = self._extract_parameters(node.args)
        
        # Extract return type
        return_type = self._format_expr(node.returns) if node.returns else None
        
        # Build signature
        signature = self._build_signature(node.name, parameters, return_type)
        
        # Extract docstring
        docstring = ast.get_docstring(node)
        
        # Determine if public
        is_public = not node.name.startswith('_')
        
        if is_method and self.current_class:
            # Add as method to current class
            method_info = MethodInfo(
                name=node.name,
                decorator=decorator,
                signature=signature,
                parameters=parameters,
                return_type=return_type,
                docstring=docstring,
                is_public=is_public,
                lineno=node.lineno
            )
            self.current_class.methods.append(method_info)
        else:
            # Add as standalone function
            func_info = FunctionInfo(
                name=node.name,
                signature=signature,
                parameters=parameters,
                return_type=return_type,
                docstring=docstring,
                is_public=is_public,
                lineno=node.lineno
            )
            self.module_info.functions.append(func_info)
    
    def _extract_parameters(self, args: ast.arguments) -> List[Dict[str, Any]]:
        """Extract parameter information with type hints and defaults"""
        parameters = []
        
        # Regular arguments
        all_args = args.args
        defaults = [None] * (len(all_args) - len(args.defaults)) + args.defaults
        
        for arg, default in zip(all_args, defaults):
            param_info = {
                'name': arg.arg,
                'annotation': self._format_expr(arg.annotation) if arg.annotation else None,
                'default': self._format_expr(default) if default else None
            }
            parameters.append(param_info)
        
        # *args
        if args.vararg:
            parameters.append({
                'name': f'*{args.vararg.arg}',
                'annotation': self._format_expr(args.vararg.annotation) if args.vararg.annotation else None,
                'default': None
            })
        
        # Keyword-only arguments
        kw_defaults = args.kw_defaults
        for arg, default in zip(args.kwonlyargs, kw_defaults):
            param_info = {
                'name': arg.arg,
                'annotation': self._format_expr(arg.annotation) if arg.annotation else None,
                'default': self._format_expr(default) if default else None
            }
            parameters.append(param_info)
        
        # **kwargs
        if args.kwarg:
            parameters.append({
                'name': f'**{args.kwarg.arg}',
                'annotation': self._format_expr(args.kwarg.annotation) if args.kwarg.annotation else None,
                'default': None
            })
        
        return parameters
    
    def _build_signature(self, name: str, parameters: List[Dict[str, Any]], 
                        return_type: Optional[str]) -> str:
        """Build human-readable signature string"""
        param_strs = []
        for p in parameters:
            s = p['name']
            if p['annotation']:
                s += f": {p['annotation']}"
            if p['default']:
                s += f" = {p['default']}"
            param_strs.append(s)
        
        sig = f"{name}({', '.join(param_strs)})"
        if return_type:
            sig += f" -> {return_type}"
        return sig
    
    def _format_expr(self, node: Optional[ast.expr]) -> str:
        """Format an expression node as string"""
        if node is None:
            return ""
        
        try:
            return ast.unparse(node)
        except Exception:
            # Fallback for complex expressions
            if isinstance(node, ast.Name):
                return node.id
            elif isinstance(node, ast.Constant):
                return repr(node.value)
            elif isinstance(node, ast.Attribute):
                return f"{self._format_expr(node.value)}.{node.attr}"
            elif isinstance(node, ast.Subscript):
                return f"{self._format_expr(node.value)}[{self._format_expr(node.slice)}]"
            return ast.dump(node)


def extract_module_api(file_path: Path) -> ModuleInfo:
    """Extract API surface from a Python module"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        
        tree = ast.parse(source, filename=str(file_path))
        extractor = APIExtractor()
        extractor.module_info.module_name = file_path.stem
        extractor.visit(tree)
        
        return extractor.module_info
    
    except SyntaxError as e:
        print(f"Syntax error in {file_path}: {e}", file=sys.stderr)
        return ModuleInfo(module_name=file_path.stem)
    except Exception as e:
        print(f"Error processing {file_path}: {e}", file=sys.stderr)
        return ModuleInfo(module_name=file_path.stem)


def serialize_module_info(module_info: ModuleInfo) -> Dict[str, Any]:
    """Convert ModuleInfo to JSON-serializable dict"""
    return {
        'module_name': module_info.module_name,
        'docstring': module_info.docstring,
        'classes': [
            {
                'name': cls.name,
                'docstring': cls.docstring,
                'base_classes': cls.base_classes,
                'is_public': cls.is_public,
                'lineno': cls.lineno,
                'methods': [
                    {
                        'name': m.name,
                        'decorator': m.decorator,
                        'signature': m.signature,
                        'parameters': m.parameters,
                        'return_type': m.return_type,
                        'docstring': m.docstring,
                        'is_public': m.is_public,
                        'lineno': m.lineno,
                        'invocation_pattern': (
                            f"{cls.name}.{m.name}()" if m.decorator == 'staticmethod' 
                            else f"{cls.name}.{m.name}()" if m.decorator == 'classmethod'
                            else f"{cls.name.lower()}_instance.{m.name}()"
                        )
                    }
                    for m in cls.methods
                ]
            }
            for cls in module_info.classes
        ],
        'functions': [
            {
                'name': f.name,
                'signature': f.signature,
                'parameters': f.parameters,
                'return_type': f.return_type,
                'docstring': f.docstring,
                'is_public': f.is_public,
                'lineno': f.lineno
            }
            for f in module_info.functions
        ]
    }


def calculate_metrics(module_info: ModuleInfo) -> Dict[str, int]:
    """Calculate baseline metrics for a module"""
    total_methods = sum(len(cls.methods) for cls in module_info.classes)
    public_methods = sum(
        sum(1 for m in cls.methods if m.is_public) 
        for cls in module_info.classes
    )
    private_methods = total_methods - public_methods
    
    staticmethods = sum(
        sum(1 for m in cls.methods if m.decorator == 'staticmethod')
        for cls in module_info.classes
    )
    
    classmethods = sum(
        sum(1 for m in cls.methods if m.decorator == 'classmethod')
        for cls in module_info.classes
    )
    
    return {
        'total_classes': len(module_info.classes),
        'total_functions': len(module_info.functions),
        'total_methods': total_methods,
        'public_methods': public_methods,
        'private_methods': private_methods,
        'static_methods': staticmethods,
        'class_methods': classmethods,
        'public_functions': sum(1 for f in module_info.functions if f.is_public),
        'private_functions': sum(1 for f in module_info.functions if not f.is_public)
    }


def main():
    """Main extraction workflow"""
    
    # Module files to process
    module_files = [
        'causal_proccesor.py',
        'contradiction_deteccion.py',
        'dereck_beach.py',
        'emebedding_policy.py',
        'policy_processor.py',
        'policy_segmenter.py',
        'semantic_chunking_policy.py',
        'teoria_cambio.py',
        'financiero_viabilidad_tablas.py',
        'Analyzer_one.py'
    ]
    
    inventory = {}
    baseline_metrics = {
        'modules': {},
        'totals': {
            'total_classes': 0,
            'total_functions': 0,
            'total_methods': 0,
            'public_methods': 0,
            'private_methods': 0,
            'static_methods': 0,
            'class_methods': 0,
            'public_functions': 0,
            'private_functions': 0
        }
    }
    
    print("Extracting API surface from modules...")
    
    for module_file in module_files:
        module_path = Path(module_file)
        if not module_path.exists():
            print(f"Warning: {module_file} not found, skipping", file=sys.stderr)
            continue
        
        print(f"Processing {module_file}...")
        module_info = extract_module_api(module_path)
        
        # Add to inventory
        inventory[module_info.module_name] = serialize_module_info(module_info)
        
        # Calculate metrics
        metrics = calculate_metrics(module_info)
        baseline_metrics['modules'][module_info.module_name] = metrics
        
        # Update totals
        for key in baseline_metrics['totals']:
            baseline_metrics['totals'][key] += metrics.get(key, 0)
        
        print(f"  - {metrics['total_classes']} classes, "
              f"{metrics['total_functions']} functions, "
              f"{metrics['total_methods']} methods")
    
    # Write inventory
    with open('source_modules_inventory.json', 'w', encoding='utf-8') as f:
        json.dump(inventory, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Generated source_modules_inventory.json")
    
    # Write baseline metrics
    with open('baseline_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(baseline_metrics, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Generated baseline_metrics.json")
    
    # Print summary
    print("\n" + "=" * 60)
    print("BASELINE METRICS SUMMARY")
    print("=" * 60)
    print(f"Total modules processed: {len(baseline_metrics['modules'])}")
    print(f"Total classes: {baseline_metrics['totals']['total_classes']}")
    print(f"Total functions: {baseline_metrics['totals']['total_functions']}")
    print(f"Total methods: {baseline_metrics['totals']['total_methods']}")
    print(f"  - Public methods: {baseline_metrics['totals']['public_methods']}")
    print(f"  - Private methods: {baseline_metrics['totals']['private_methods']}")
    print(f"  - Static methods: {baseline_metrics['totals']['static_methods']}")
    print(f"  - Class methods: {baseline_metrics['totals']['class_methods']}")
    print(f"Public functions: {baseline_metrics['totals']['public_functions']}")
    print(f"Private functions: {baseline_metrics['totals']['private_functions']}")
    print("=" * 60)
    
    # Calculate 95% threshold
    total_api_surface = (
        baseline_metrics['totals']['total_classes'] +
        baseline_metrics['totals']['total_functions'] +
        baseline_metrics['totals']['total_methods']
    )
    threshold_95 = int(total_api_surface * 0.95)
    print(f"\n95% Preservation Target: {threshold_95} API elements")
    print(f"(out of {total_api_surface} total elements)")
    print("=" * 60)


if __name__ == '__main__':
    main()
