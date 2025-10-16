#!/usr/bin/env python3
"""
Test de Verificación de Integración del Orquestador
Verifica que el orquestador tenga adaptadores para TODAS las clases y métodos
de los scripts principales de FARFAN-3.0
"""

import ast
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict


class ClassMethodExtractor(ast.NodeVisitor):
    """Extrae todas las clases y métodos públicos de un archivo Python"""

    def __init__(self):
        self.classes: Dict[str, List[str]] = {}
        self.current_class = None
        self.functions: List[str] = []

    def visit_ClassDef(self, node):
        """Visita definiciones de clases"""
        self.current_class = node.name
        self.classes[node.name] = []
        self.generic_visit(node)
        self.current_class = None

    def visit_FunctionDef(self, node):
        """Visita definiciones de funciones/métodos"""
        # Ignorar métodos privados (empiezan con _) excepto __init__
        if node.name.startswith('_') and node.name != '__init__':
            return

        if self.current_class:
            self.classes[self.current_class].append(node.name)
        else:
            self.functions.append(node.name)

    def visit_AsyncFunctionDef(self, node):
        """Visita definiciones de funciones asíncronas"""
        if node.name.startswith('_') and node.name != '__init__':
            return

        if self.current_class:
            self.classes[self.current_class].append(node.name)
        else:
            self.functions.append(node.name)


def extract_classes_methods(file_path: Path) -> Tuple[Dict[str, List[str]], List[str]]:
    """
    Extrae todas las clases y métodos públicos de un archivo Python

    Returns:
        (classes_dict, standalone_functions)
        classes_dict: {class_name: [method1, method2, ...]}
        standalone_functions: [func1, func2, ...]
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=str(file_path))

        extractor = ClassMethodExtractor()
        extractor.visit(tree)

        return extractor.classes, extractor.functions

    except SyntaxError as e:
        print(f"❌ ERROR DE SINTAXIS en {file_path}: {e}")
        return {}, []
    except Exception as e:
        print(f"❌ ERROR al analizar {file_path}: {e}")
        return {}, []


def analyze_main_scripts():
    """Analiza TODOS los scripts principales de FARFAN-3.0"""

    main_scripts = [
        "Analyzer_one.py",
        "causal_proccesor.py",
        "contradiction_deteccion.py",
        "dereck_beach",
        "emebedding_policy.py",
        "financiero_viabilidad_tablas.py",
        "policy_processor.py",
        "policy_segmenter.py",
        "semantic_chunking_policy.py",
        "teoria_cambio.py"
    ]

    print("=" * 80)
    print("ANÁLISIS DE SCRIPTS PRINCIPALES")
    print("=" * 80)

    all_main_classes = {}
    all_main_functions = {}

    for script in main_scripts:
        script_path = Path(script)
        if not script_path.exists():
            print(f"⚠️  ADVERTENCIA: {script} NO EXISTE")
            continue

        classes, functions = extract_classes_methods(script_path)

        if classes or functions:
            all_main_classes[script] = classes
            all_main_functions[script] = functions

            print(f"\n📄 {script}")
            print(f"   Clases: {len(classes)}")
            for class_name, methods in classes.items():
                print(f"      • {class_name}: {len(methods)} métodos")
            print(f"   Funciones standalone: {len(functions)}")

    return all_main_classes, all_main_functions


def analyze_orchestrator():
    """Analiza TODOS los archivos del orquestador"""

    orchestrator_files = [
        "orchestrator/__init__.py",
        "orchestrator/choreographer.py",
        "orchestrator/circuit_breaker.py",
        "orchestrator/config.py",
        "orchestrator/core_orchestrator.py",
        "orchestrator/dashboard_generator.py",
        "orchestrator/module_adapters.py",
        "orchestrator/question_router.py",
        "orchestrator/report_assembly.py"
    ]

    print("\n" + "=" * 80)
    print("ANÁLISIS DEL ORQUESTADOR")
    print("=" * 80)

    all_orch_classes = {}
    all_orch_functions = {}

    for orch_file in orchestrator_files:
        orch_path = Path(orch_file)
        if not orch_path.exists():
            print(f"⚠️  ADVERTENCIA: {orch_file} NO EXISTE")
            continue

        classes, functions = extract_classes_methods(orch_path)

        if classes or functions:
            all_orch_classes[orch_file] = classes
            all_orch_functions[orch_file] = functions

            print(f"\n📄 {orch_file}")
            print(f"   Clases: {len(classes)}")
            for class_name, methods in classes.items():
                print(f"      • {class_name}: {len(methods)} métodos")
            print(f"   Funciones standalone: {len(functions)}")

    return all_orch_classes, all_orch_functions


def check_integration(main_classes, orch_classes):
    """
    Verifica si el orquestador tiene adaptadores para las clases principales
    """

    print("\n" + "=" * 80)
    print("VERIFICACIÓN DE INTEGRACIÓN")
    print("=" * 80)

    # Extraer todos los nombres de clases principales
    all_main_class_names = set()
    for script, classes in main_classes.items():
        all_main_class_names.update(classes.keys())

    # Extraer todos los nombres de clases del orquestador
    all_orch_class_names = set()
    for orch_file, classes in orch_classes.items():
        all_orch_class_names.update(classes.keys())

    print(f"\n📊 RESUMEN:")
    print(f"   Total clases en scripts principales: {len(all_main_class_names)}")
    print(f"   Total clases en orquestador: {len(all_orch_class_names)}")

    # Buscar adaptadores (clases que contengan "Adapter" en el nombre)
    adapters = {name for name in all_orch_class_names if 'Adapter' in name or 'adapter' in name.lower()}
    print(f"   Clases adaptadoras encontradas: {len(adapters)}")

    if adapters:
        print(f"\n   Adaptadores:")
        for adapter in sorted(adapters):
            print(f"      • {adapter}")

    # Verificar qué clases principales NO tienen adaptadores evidentes
    print(f"\n⚠️  CLASES PRINCIPALES (primeras 20):")
    for i, class_name in enumerate(sorted(all_main_class_names)[:20], 1):
        # Buscar si hay un adaptador que mencione esta clase
        has_adapter = any(class_name.lower() in adapter.lower() for adapter in adapters)
        status = "✅" if has_adapter else "❌"
        print(f"      {status} {class_name}")

    if len(all_main_class_names) > 20:
        print(f"      ... y {len(all_main_class_names) - 20} clases más")

    return len(adapters), len(all_main_class_names)


def generate_detailed_report(main_classes, main_functions, orch_classes, orch_functions):
    """Genera reporte detallado en archivo"""

    report_path = Path("orchestrator_integration_report.txt")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("REPORTE DETALLADO DE INTEGRACIÓN ORQUESTADOR - FARFAN 3.0\n")
        f.write("=" * 80 + "\n\n")

        # Scripts principales
        f.write("SCRIPTS PRINCIPALES\n")
        f.write("-" * 80 + "\n\n")

        for script, classes in sorted(main_classes.items()):
            f.write(f"\n{script}\n")
            f.write(f"{'=' * len(script)}\n")

            for class_name, methods in sorted(classes.items()):
                f.write(f"\n  Clase: {class_name}\n")
                f.write(f"  Métodos ({len(methods)}):\n")
                for method in sorted(methods):
                    f.write(f"    • {method}\n")

            if script in main_functions and main_functions[script]:
                f.write(f"\n  Funciones standalone ({len(main_functions[script])}):\n")
                for func in sorted(main_functions[script]):
                    f.write(f"    • {func}\n")

        # Orquestador
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("ORQUESTADOR\n")
        f.write("-" * 80 + "\n\n")

        for orch_file, classes in sorted(orch_classes.items()):
            f.write(f"\n{orch_file}\n")
            f.write(f"{'=' * len(orch_file)}\n")

            for class_name, methods in sorted(classes.items()):
                f.write(f"\n  Clase: {class_name}\n")
                f.write(f"  Métodos ({len(methods)}):\n")
                for method in sorted(methods):
                    f.write(f"    • {method}\n")

            if orch_file in orch_functions and orch_functions[orch_file]:
                f.write(f"\n  Funciones standalone ({len(orch_functions[orch_file])}):\n")
                for func in sorted(orch_functions[orch_file]):
                    f.write(f"    • {func}\n")

    print(f"\n✅ Reporte detallado guardado en: {report_path.absolute()}")


def main():
    """Función principal"""

    print("\n🔍 INICIANDO ANÁLISIS DE INTEGRACIÓN ORQUESTADOR\n")

    # Analizar scripts principales
    main_classes, main_functions = analyze_main_scripts()

    # Analizar orquestador
    orch_classes, orch_functions = analyze_orchestrator()

    # Verificar integración
    num_adapters, num_main_classes = check_integration(main_classes, orch_classes)

    # Generar reporte detallado
    generate_detailed_report(main_classes, main_functions, orch_classes, orch_functions)

    # Resumen final
    print("\n" + "=" * 80)
    print("CONCLUSIÓN")
    print("=" * 80)

    coverage = (num_adapters / num_main_classes * 100) if num_main_classes > 0 else 0

    print(f"\n📊 Cobertura de adaptadores: {coverage:.1f}%")
    print(f"   ({num_adapters} adaptadores para {num_main_classes} clases principales)")

    if coverage < 50:
        print("\n❌ CRÍTICO: Cobertura insuficiente - El orquestador NO está integrando adecuadamente los módulos")
        return 1
    elif coverage < 80:
        print("\n⚠️  ADVERTENCIA: Cobertura parcial - Faltan adaptadores")
        return 1
    else:
        print("\n✅ Cobertura aceptable")
        return 0


if __name__ == "__main__":
    sys.exit(main())
