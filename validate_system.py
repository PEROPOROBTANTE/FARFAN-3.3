#!/usr/bin/env python3
"""
FARFAN 3.0 - Script de Validación del Sistema
==============================================
Verifica el estado del sistema y proporciona diagnóstico detallado.

Uso:
    python validate_system.py
    python validate_system.py --verbose
"""

import sys
import importlib
from pathlib import Path
from typing import Dict, List, Tuple

# Colores ANSI para terminal
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'


def print_header(text: str):
    """Imprime encabezado con estilo."""
    print(f"\n{BOLD}{BLUE}{'=' * 80}{RESET}")
    print(f"{BOLD}{BLUE}{text:^80}{RESET}")
    print(f"{BOLD}{BLUE}{'=' * 80}{RESET}\n")


def print_success(text: str):
    """Imprime mensaje de éxito."""
    print(f"{GREEN}✓ {text}{RESET}")


def print_warning(text: str):
    """Imprime mensaje de advertencia."""
    print(f"{YELLOW}⚠ {text}{RESET}")


def print_error(text: str):
    """Imprime mensaje de error."""
    print(f"{RED}✗ {text}{RESET}")


def check_python_version() -> bool:
    """Verifica versión de Python."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print_success(f"Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print_error(f"Python {version.major}.{version.minor}.{version.micro} - Se requiere Python 3.8+")
        return False


def check_dependencies() -> Tuple[int, int, List[str]]:
    """Verifica dependencias críticas."""
    critical_deps = [
        'numpy',
        'pandas',
        'scipy',
        'sklearn',
        'nltk',
        'networkx',
        'sentence_transformers',
        'transformers',
        'torch',
        'spacy',
        'PyPDF2',
    ]
    
    optional_deps = [
        'camelot',
        'pymongo',
        'motor',
        'statsmodels',
    ]
    
    print("\nDependencias Críticas:")
    critical_ok = 0
    critical_total = len(critical_deps)
    
    for dep in critical_deps:
        try:
            importlib.import_module(dep)
            print_success(f"{dep:30s} instalado")
            critical_ok += 1
        except ImportError:
            print_error(f"{dep:30s} NO INSTALADO")
    
    print("\nDependencias Opcionales:")
    optional_ok = 0
    
    for dep in optional_deps:
        try:
            importlib.import_module(dep)
            print_success(f"{dep:30s} instalado")
            optional_ok += 1
        except ImportError:
            print_warning(f"{dep:30s} no instalado (opcional)")
    
    missing = [dep for dep in critical_deps 
               if importlib.util.find_spec(dep) is None]
    
    return critical_ok, critical_total, missing


def check_spacy_model() -> bool:
    """Verifica modelo spaCy español."""
    try:
        import spacy
        nlp = spacy.load("es_core_news_lg")
        print_success("Modelo spaCy 'es_core_news_lg' disponible")
        return True
    except ImportError:
        print_error("spaCy no instalado")
        return False
    except OSError:
        print_error("Modelo spaCy 'es_core_news_lg' NO encontrado")
        print(f"  {YELLOW}→ Instalar con: python -m spacy download es_core_news_lg{RESET}")
        return False


def check_modules() -> Tuple[int, int, List[str]]:
    """Verifica módulos del sistema FARFAN."""
    modules = [
        'policy_processor',
        'causal_proccesor',
        'Analyzer_one',
        'contradiction_deteccion',
        'emebedding_policy',
        'financiero_viabilidad_tablas',
        'policy_segmenter',
        'semantic_chunking_policy'
    ]
    
    ok = 0
    total = len(modules)
    failed = []
    
    print("\nMódulos FARFAN:")
    for module in modules:
        try:
            importlib.import_module(module)
            print_success(f"{module:35s} OK")
            ok += 1
        except SyntaxError as e:
            print_error(f"{module:35s} ERROR DE SINTAXIS")
            print(f"  {YELLOW}→ {str(e)[:80]}{RESET}")
            failed.append(module)
        except Exception as e:
            print_error(f"{module:35s} ERROR")
            print(f"  {YELLOW}→ {str(e)[:80]}{RESET}")
            failed.append(module)
    
    return ok, total, failed


def check_derek_beach() -> bool:
    """Verifica si Derek Beach está disponible."""
    try:
        import dereck_beach
        print_success("Módulo 'dereck_beach' encontrado")
        
        # Verificar clases críticas
        critical_classes = [
            'CDAFFramework',
            'BeachEvidentialTest',
            'CausalExtractor',
            'MechanismPartExtractor'
        ]
        
        missing_classes = []
        for cls_name in critical_classes:
            if not hasattr(dereck_beach, cls_name):
                missing_classes.append(cls_name)
        
        if missing_classes:
            print_warning(f"Clases faltantes en dereck_beach: {', '.join(missing_classes)}")
            return False
        else:
            print_success("Todas las clases críticas presentes")
            return True
            
    except ImportError:
        print_error("Módulo 'dereck_beach' NO ENCONTRADO (CRÍTICO)")
        print(f"  {YELLOW}→ Este es el bloqueante principal del sistema{RESET}")
        return False


def check_orchestrator() -> bool:
    """Verifica que el orquestador funcione."""
    try:
        from orchestrator import FARFANOrchestrator
        orch = FARFANOrchestrator()
        print_success("Orquestador inicializado correctamente")
        
        # Verificar componentes
        if orch.router and orch.choreographer and orch.circuit_breaker:
            print_success("Todos los componentes del orquestador presentes")
            return True
        else:
            print_warning("Algunos componentes del orquestador faltan")
            return False
            
    except Exception as e:
        print_error(f"Error al inicializar orquestador: {str(e)[:80]}")
        return False


def check_files() -> Dict[str, bool]:
    """Verifica archivos importantes."""
    important_files = {
        'cuestionario.json': 'Cuestionario de 300 preguntas',
        'requirements_complete.txt': 'Dependencias completas',
        'ANALISIS_CRITICO_IMPLEMENTACION.md': 'Análisis del repositorio',
        'ROADMAP_IMPLEMENTACION.md': 'Plan de implementación',
        'GUIA_RAPIDA_PRIMEROS_PASOS.md': 'Guía de inicio',
        'orchestrator/execution_mapping.yaml': 'Configuración de ejecución',
    }
    
    print("\nArchivos Importantes:")
    results = {}
    
    for file_path, description in important_files.items():
        path = Path(file_path)
        if path.exists():
            print_success(f"{file_path:45s} ({description})")
            results[file_path] = True
        else:
            print_error(f"{file_path:45s} NO ENCONTRADO")
            results[file_path] = False
    
    return results


def check_syntax_errors() -> List[str]:
    """Verifica errores de sintaxis en archivos Python."""
    import py_compile
    import tempfile
    
    python_files = list(Path('.').glob('*.py'))
    errors = []
    
    print("\nVerificación de Sintaxis:")
    
    for py_file in python_files:
        if py_file.name == 'validate_system.py':
            continue
            
        try:
            with tempfile.NamedTemporaryFile() as tmp:
                py_compile.compile(str(py_file), doraise=True)
            print_success(f"{py_file.name:40s} sintaxis válida")
        except py_compile.PyCompileError as e:
            print_error(f"{py_file.name:40s} ERROR DE SINTAXIS")
            errors.append(str(py_file))
    
    return errors


def calculate_readiness_score(results: Dict) -> Tuple[int, str]:
    """Calcula score de preparación del sistema."""
    total_points = 100
    current_points = 0
    
    # Python version (5 puntos)
    if results['python_version']:
        current_points += 5
    
    # Dependencias críticas (30 puntos)
    if results['critical_deps_ok'] == results['critical_deps_total']:
        current_points += 30
    else:
        current_points += int(30 * results['critical_deps_ok'] / results['critical_deps_total'])
    
    # Módulos FARFAN (25 puntos)
    if results['modules_ok'] == results['modules_total']:
        current_points += 25
    else:
        current_points += int(25 * results['modules_ok'] / results['modules_total'])
    
    # Derek Beach (20 puntos - crítico)
    if results['derek_beach']:
        current_points += 20
    
    # Orquestador (10 puntos)
    if results['orchestrator']:
        current_points += 10
    
    # SpaCy model (5 puntos)
    if results['spacy_model']:
        current_points += 5
    
    # Sin errores de sintaxis (5 puntos)
    if len(results['syntax_errors']) == 0:
        current_points += 5
    
    # Determinar estado
    if current_points >= 90:
        status = "LISTO PARA PRODUCCIÓN"
    elif current_points >= 70:
        status = "CASI LISTO - Correcciones menores"
    elif current_points >= 50:
        status = "EN DESARROLLO - Trabajo significativo requerido"
    else:
        status = "BLOQUEADO - Problemas críticos"
    
    return current_points, status


def print_recommendations(results: Dict):
    """Imprime recomendaciones basadas en resultados."""
    print_header("RECOMENDACIONES")
    
    recommendations = []
    
    if not results['derek_beach']:
        recommendations.append(
            "🔴 CRÍTICO: Localizar o implementar módulo 'dereck_beach'\n"
            "   → Este es el bloqueante principal del sistema"
        )
    
    if results['missing_deps']:
        recommendations.append(
            f"🔴 CRÍTICO: Instalar dependencias faltantes:\n"
            f"   → pip install {' '.join(results['missing_deps'])}"
        )
    
    if not results['spacy_model']:
        recommendations.append(
            "🟡 IMPORTANTE: Instalar modelo spaCy:\n"
            "   → python -m spacy download es_core_news_lg"
        )
    
    if results['failed_modules']:
        recommendations.append(
            f"🟡 Corregir módulos con errores: {', '.join(results['failed_modules'])}"
        )
    
    if results['syntax_errors']:
        recommendations.append(
            f"🟡 Corregir errores de sintaxis en: {', '.join(results['syntax_errors'])}"
        )
    
    if not recommendations:
        print_success("¡Todo está en orden! Sistema listo para pruebas.")
    else:
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec}")


def main():
    """Función principal."""
    print_header("FARFAN 3.0 - VALIDACIÓN DEL SISTEMA")
    
    results = {}
    
    # 1. Python version
    print_header("1. VERSIÓN DE PYTHON")
    results['python_version'] = check_python_version()
    
    # 2. Dependencies
    print_header("2. DEPENDENCIAS")
    critical_ok, critical_total, missing = check_dependencies()
    results['critical_deps_ok'] = critical_ok
    results['critical_deps_total'] = critical_total
    results['missing_deps'] = missing
    
    # 3. SpaCy model
    print_header("3. MODELO SPACY")
    results['spacy_model'] = check_spacy_model()
    
    # 4. FARFAN modules
    print_header("4. MÓDULOS FARFAN")
    modules_ok, modules_total, failed = check_modules()
    results['modules_ok'] = modules_ok
    results['modules_total'] = modules_total
    results['failed_modules'] = failed
    
    # 5. Derek Beach
    print_header("5. MÓDULO DEREK BEACH")
    results['derek_beach'] = check_derek_beach()
    
    # 6. Orchestrator
    print_header("6. ORQUESTADOR")
    results['orchestrator'] = check_orchestrator()
    
    # 7. Important files
    print_header("7. ARCHIVOS IMPORTANTES")
    file_results = check_files()
    results.update(file_results)
    
    # 8. Syntax errors
    print_header("8. ERRORES DE SINTAXIS")
    syntax_errors = check_syntax_errors()
    results['syntax_errors'] = syntax_errors
    
    # Calculate readiness score
    print_header("EVALUACIÓN GENERAL")
    score, status = calculate_readiness_score(results)
    
    # Print progress bar
    bar_length = 50
    filled = int(bar_length * score / 100)
    bar = '█' * filled + '░' * (bar_length - filled)
    
    print(f"\n{BOLD}Score de Preparación:{RESET}")
    print(f"  [{bar}] {score}/100")
    print(f"\n{BOLD}Estado:{RESET} {status}")
    
    print(f"\n{BOLD}Detalles:{RESET}")
    print(f"  • Dependencias críticas: {results['critical_deps_ok']}/{results['critical_deps_total']}")
    print(f"  • Módulos FARFAN: {results['modules_ok']}/{results['modules_total']}")
    print(f"  • Derek Beach: {'✓' if results['derek_beach'] else '✗ (CRÍTICO)'}")
    print(f"  • Orquestador: {'✓' if results['orchestrator'] else '✗'}")
    print(f"  • Errores de sintaxis: {len(results['syntax_errors'])}")
    
    # Recommendations
    print_recommendations(results)
    
    # Exit code
    if score >= 70:
        print(f"\n{GREEN}{BOLD}✓ Sistema en buen estado{RESET}\n")
        return 0
    else:
        print(f"\n{YELLOW}{BOLD}⚠ Sistema requiere atención{RESET}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
