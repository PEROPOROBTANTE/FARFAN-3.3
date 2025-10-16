#!/usr/bin/env python3
"""
Script de Análisis de Repositorio Python
Genera un reporte completo del código para construir un orquestador
"""

import os
import ast
import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Any
import re


class RepositoryAnalyzer:
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.analysis = {
            "estructura": {},
            "archivos_python": [],
            "dependencias": {
                "imports_stdlib": set(),
                "imports_terceros": set(),
                "imports_locales": set()
            },
            "funciones": [],
            "clases": [],
            "puntos_entrada": [],
            "configuraciones": [],
            "patrones_detectados": [],
            "frameworks": set(),
            "apis_endpoints": [],
            "tareas_async": [],
            "conexiones_db": [],
            "arquitectura": {}
        }

    def analizar_todo(self):
        """Ejecuta todos los análisis"""
        print("🔍 Iniciando análisis del repositorio...")

        self.analizar_estructura()
        self.analizar_dependencias_requirements()
        self.analizar_archivos_python()
        self.detectar_frameworks()
        self.detectar_patrones_arquitectura()
        self.analizar_configuraciones()
        self.generar_reporte()

        print("✅ Análisis completado!")

    def analizar_estructura(self):
        """Analiza la estructura de directorios"""
        print("📁 Analizando estructura de directorios...")

        estructura = {
            "raiz": str(self.repo_path),
            "directorios": [],
            "archivos_config": [],
            "archivos_python": 0,
            "total_archivos": 0
        }

        for root, dirs, files in os.walk(self.repo_path):
            # Ignorar directorios comunes
            dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'node_modules', '.venv', 'venv', '.env']]

            rel_path = os.path.relpath(root, self.repo_path)
            if rel_path != '.':
                estructura["directorios"].append(rel_path)

            for file in files:
                estructura["total_archivos"] += 1

                if file.endswith('.py'):
                    estructura["archivos_python"] += 1

                if file in ['requirements.txt', 'setup.py', 'pyproject.toml',
                            'Dockerfile', 'docker-compose.yml', '.env.example',
                            'config.py', 'settings.py', 'config.yaml', 'config.json']:
                    estructura["archivos_config"].append(os.path.join(rel_path, file))

        self.analysis["estructura"] = estructura

    def analizar_dependencias_requirements(self):
        """Analiza requirements.txt y pyproject.toml"""
        print("📦 Analizando dependencias...")

        # Buscar requirements.txt
        req_files = list(self.repo_path.glob('**/requirements*.txt'))
        for req_file in req_files:
            try:
                with open(req_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            package = line.split('==')[0].split('>=')[0].split('<=')[0].strip()
                            self.analysis["dependencias"]["imports_terceros"].add(package)
            except Exception as e:
                print(f"⚠️  Error leyendo {req_file}: {e}")

        # Buscar pyproject.toml
        pyproject = self.repo_path / 'pyproject.toml'
        if pyproject.exists():
            try:
                with open(pyproject, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Extraer dependencias básicas
                    deps = re.findall(r'"([a-zA-Z0-9\-_]+)[>=<]', content)
                    for dep in deps:
                        self.analysis["dependencias"]["imports_terceros"].add(dep)
            except Exception as e:
                print(f"⚠️  Error leyendo pyproject.toml: {e}")

    def analizar_archivos_python(self):
        """Analiza todos los archivos Python del repositorio"""
        print("🐍 Analizando archivos Python...")

        python_files = list(self.repo_path.rglob('*.py'))

        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                rel_path = py_file.relative_to(self.repo_path)

                # Analizar AST
                try:
                    tree = ast.parse(content)
                    self.analizar_ast(tree, str(rel_path), content)
                except SyntaxError:
                    print(f"⚠️  Error de sintaxis en {rel_path}")

                # Detectar puntos de entrada
                if 'if __name__ == "__main__"' in content:
                    self.analysis["puntos_entrada"].append(str(rel_path))

                self.analysis["archivos_python"].append({
                    "path": str(rel_path),
                    "lineas": len(content.split('\n')),
                    "es_punto_entrada": 'if __name__ == "__main__"' in content
                })

            except Exception as e:
                print(f"⚠️  Error procesando {py_file}: {e}")

    def analizar_ast(self, tree: ast.AST, filepath: str, content: str):
        """Analiza el AST de un archivo Python"""
        for node in ast.walk(tree):
            # Analizar imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.clasificar_import(alias.name)

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    self.clasificar_import(node.module)

            # Analizar funciones
            elif isinstance(node, ast.FunctionDef):
                func_info = {
                    "nombre": node.name,
                    "archivo": filepath,
                    "linea": node.lineno,
                    "es_async": isinstance(node, ast.AsyncFunctionDef),
                    "decoradores": [self.get_decorator_name(d) for d in node.decorator_list],
                    "argumentos": [arg.arg for arg in node.args.args]
                }
                self.analysis["funciones"].append(func_info)

                # Detectar tareas async
                if isinstance(node, ast.AsyncFunctionDef):
                    self.analysis["tareas_async"].append(func_info)

                # Detectar endpoints API
                for decorator in node.decorator_list:
                    dec_name = self.get_decorator_name(decorator)
                    if any(x in dec_name.lower() for x in ['route', 'get', 'post', 'put', 'delete', 'patch', 'api']):
                        self.analysis["apis_endpoints"].append({
                            **func_info,
                            "decorador": dec_name
                        })

            # Analizar clases
            elif isinstance(node, ast.ClassDef):
                class_info = {
                    "nombre": node.name,
                    "archivo": filepath,
                    "linea": node.lineno,
                    "herencia": [self.get_base_name(base) for base in node.bases],
                    "metodos": [m.name for m in node.body if isinstance(m, ast.FunctionDef)]
                }
                self.analysis["clases"].append(class_info)

        # Buscar conexiones a base de datos
        db_patterns = [
            r'psycopg2\.connect',
            r'pymongo\.MongoClient',
            r'mysql\.connector',
            r'sqlite3\.connect',
            r'create_engine',
            r'AsyncIOMotorClient',
            r'redis\.Redis'
        ]

        for pattern in db_patterns:
            if re.search(pattern, content):
                self.analysis["conexiones_db"].append({
                    "tipo": pattern.split('.')[0],
                    "archivo": filepath
                })

    def get_decorator_name(self, decorator) -> str:
        """Obtiene el nombre de un decorador"""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Call):
            if isinstance(decorator.func, ast.Name):
                return decorator.func.id
            elif isinstance(decorator.func, ast.Attribute):
                return decorator.func.attr
        elif isinstance(decorator, ast.Attribute):
            return decorator.attr
        return "unknown"

    def get_base_name(self, base) -> str:
        """Obtiene el nombre de una clase base"""
        if isinstance(base, ast.Name):
            return base.id
        elif isinstance(base, ast.Attribute):
            return base.attr
        return "unknown"

    def clasificar_import(self, module_name: str):
        """Clasifica un import como stdlib, tercero o local"""
        stdlib_modules = sys.stdlib_module_names if hasattr(sys, 'stdlib_module_names') else set()

        base_module = module_name.split('.')[0]

        if base_module in stdlib_modules or base_module in ['os', 'sys', 'json', 'datetime', 'asyncio']:
            self.analysis["dependencias"]["imports_stdlib"].add(base_module)
        elif base_module.startswith('.'):
            self.analysis["dependencias"]["imports_locales"].add(module_name)
        else:
            # Verificar si es un módulo local del proyecto
            local_path = self.repo_path / base_module
            if local_path.exists() or (self.repo_path / f"{base_module}.py").exists():
                self.analysis["dependencias"]["imports_locales"].add(base_module)
            else:
                self.analysis["dependencias"]["imports_terceros"].add(base_module)

    def detectar_frameworks(self):
        """Detecta frameworks y librerías utilizadas"""
        print("🔧 Detectando frameworks...")

        frameworks_map = {
            'fastapi': 'FastAPI',
            'flask': 'Flask',
            'django': 'Django',
            'celery': 'Celery',
            'airflow': 'Apache Airflow',
            'prefect': 'Prefect',
            'dagster': 'Dagster',
            'luigi': 'Luigi',
            'asyncio': 'AsyncIO',
            'aiohttp': 'aiohttp',
            'sqlalchemy': 'SQLAlchemy',
            'pydantic': 'Pydantic',
            'pandas': 'Pandas',
            'numpy': 'NumPy',
            'redis': 'Redis',
            'kafka': 'Kafka',
            'rabbitmq': 'RabbitMQ',
            'pika': 'RabbitMQ (pika)',
            'dramatiq': 'Dramatiq',
            'rq': 'RQ (Redis Queue)',
            'apscheduler': 'APScheduler'
        }

        all_imports = (self.analysis["dependencias"]["imports_terceros"] |
                       self.analysis["dependencias"]["imports_locales"])

        for imp in all_imports:
            imp_lower = imp.lower()
            for key, name in frameworks_map.items():
                if key in imp_lower:
                    self.analysis["frameworks"].add(name)

    def detectar_patrones_arquitectura(self):
        """Detecta patrones de arquitectura"""
        print("🏗️  Detectando patrones de arquitectura...")

        patrones = []

        # Detectar patrones por estructura de directorios
        dirs = set(self.analysis["estructura"]["directorios"])

        if any('models' in d for d in dirs):
            patrones.append("MVC/MVT - Capa de modelos")
        if any('views' in d or 'controllers' in d for d in dirs):
            patrones.append("MVC - Capa de controladores/vistas")
        if any('services' in d or 'service' in d for d in dirs):
            patrones.append("Service Layer Pattern")
        if any('repository' in d or 'repositories' in d for d in dirs):
            patrones.append("Repository Pattern")
        if any('tasks' in d or 'jobs' in d for d in dirs):
            patrones.append("Task/Job Pattern")
        if any('api' in d for d in dirs):
            patrones.append("API Layer")
        if any('workers' in d or 'worker' in d for d in dirs):
            patrones.append("Worker Pattern")
        if any('dag' in d.lower() for d in dirs):
            patrones.append("DAG (Directed Acyclic Graph)")
        if any('pipeline' in d for d in dirs):
            patrones.append("Pipeline Pattern")
        if any('queue' in d for d in dirs):
            patrones.append("Queue Pattern")

        # Detectar por código
        if len(self.analysis["tareas_async"]) > 5:
            patrones.append("Async/Await Pattern")
        if len(self.analysis["apis_endpoints"]) > 0:
            patrones.append("REST API")
        if any('celery' in str(f.get('decoradores', [])).lower() for f in self.analysis["funciones"]):
            patrones.append("Celery Tasks")

        self.analysis["patrones_detectados"] = list(set(patrones))

    def analizar_configuraciones(self):
        """Analiza archivos de configuración"""
        print("⚙️  Analizando configuraciones...")

        config_files = [
            'config.py', 'settings.py', 'config.yaml', 'config.json',
            '.env.example', 'docker-compose.yml', 'Dockerfile'
        ]

        for config_file in config_files:
            file_path = self.repo_path / config_file
            if file_path.exists():
                self.analysis["configuraciones"].append({
                    "archivo": config_file,
                    "existe": True,
                    "tipo": file_path.suffix or 'sin_extension'
                })

    def generar_reporte(self):
        """Genera el reporte final"""
        print("\n📊 Generando reporte...")

        # Convertir sets a listas para JSON
        self.analysis["dependencias"]["imports_stdlib"] = sorted(list(self.analysis["dependencias"]["imports_stdlib"]))
        self.analysis["dependencias"]["imports_terceros"] = sorted(
            list(self.analysis["dependencias"]["imports_terceros"]))
        self.analysis["dependencias"]["imports_locales"] = sorted(
            list(self.analysis["dependencias"]["imports_locales"]))
        self.analysis["frameworks"] = sorted(list(self.analysis["frameworks"]))

        # Generar estadísticas
        stats = {
            "total_archivos_python": len(self.analysis["archivos_python"]),
            "total_funciones": len(self.analysis["funciones"]),
            "total_clases": len(self.analysis["clases"]),
            "total_funciones_async": len(self.analysis["tareas_async"]),
            "total_endpoints_api": len(self.analysis["apis_endpoints"]),
            "frameworks_detectados": len(self.analysis["frameworks"]),
            "patrones_detectados": len(self.analysis["patrones_detectados"])
        }

        self.analysis["estadisticas"] = stats

        # Guardar JSON
        output_json = self.repo_path / 'analisis_repo.json'
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(self.analysis, f, indent=2, ensure_ascii=False)
        print(f"✅ Reporte JSON guardado en: {output_json}")

        # Generar reporte Markdown
        self.generar_reporte_markdown()

    def generar_reporte_markdown(self):
        """Genera un reporte en formato Markdown"""
        output_md = self.repo_path / 'ANALISIS_REPO.md'

        with open(output_md, 'w', encoding='utf-8') as f:
            f.write("# 📋 Análisis del Repositorio Python\n\n")
            f.write(f"**Repositorio:** `{self.repo_path}`\n\n")

            # Estadísticas generales
            f.write("## 📊 Estadísticas Generales\n\n")
            stats = self.analysis["estadisticas"]
            f.write(f"- **Total de archivos Python:** {stats['total_archivos_python']}\n")
            f.write(f"- **Total de funciones:** {stats['total_funciones']}\n")
            f.write(f"- **Total de clases:** {stats['total_clases']}\n")
            f.write(f"- **Funciones asíncronas:** {stats['total_funciones_async']}\n")
            f.write(f"- **Endpoints API:** {stats['total_endpoints_api']}\n\n")

            # Estructura
            f.write("## 📁 Estructura del Proyecto\n\n")
            f.write(f"**Total de directorios:** {len(self.analysis['estructura']['directorios'])}\n\n")
            if self.analysis['estructura']['directorios']:
                f.write("**Directorios principales:**\n")
                for dir in sorted(self.analysis['estructura']['directorios'])[:20]:
                    f.write(f"- `{dir}`\n")
                f.write("\n")

            # Frameworks
            f.write("## 🔧 Frameworks y Librerías Detectadas\n\n")
            if self.analysis["frameworks"]:
                for fw in sorted(self.analysis["frameworks"]):
                    f.write(f"- {fw}\n")
            else:
                f.write("*No se detectaron frameworks conocidos*\n")
            f.write("\n")

            # Patrones
            f.write("## 🏗️ Patrones de Arquitectura Detectados\n\n")
            if self.analysis["patrones_detectados"]:
                for patron in self.analysis["patrones_detectados"]:
                    f.write(f"- {patron}\n")
            else:
                f.write("*No se detectaron patrones específicos*\n")
            f.write("\n")

            # Puntos de entrada
            f.write("## 🚀 Puntos de Entrada (Scripts Ejecutables)\n\n")
            if self.analysis["puntos_entrada"]:
                for entry in self.analysis["puntos_entrada"]:
                    f.write(f"- `{entry}`\n")
            else:
                f.write("*No se detectaron puntos de entrada*\n")
            f.write("\n")

            # APIs
            if self.analysis["apis_endpoints"]:
                f.write("## 🌐 Endpoints API Detectados\n\n")
                for endpoint in self.analysis["apis_endpoints"][:20]:
                    f.write(f"- **{endpoint['nombre']}** ")
                    f.write(f"(`{endpoint['archivo']}:{endpoint['linea']}`) ")
                    f.write(f"- Decorador: `{endpoint['decorador']}`\n")
                f.write("\n")

            # Tareas async
            if self.analysis["tareas_async"]:
                f.write("## ⚡ Funciones Asíncronas\n\n")
                f.write(f"Se detectaron {len(self.analysis['tareas_async'])} funciones asíncronas.\n\n")
                for task in self.analysis["tareas_async"][:10]:
                    f.write(f"- `{task['nombre']}` en `{task['archivo']}`\n")
                f.write("\n")

            # Conexiones DB
            if self.analysis["conexiones_db"]:
                f.write("## 💾 Conexiones a Bases de Datos\n\n")
                db_types = {}
                for conn in self.analysis["conexiones_db"]:
                    db_type = conn['tipo']
                    if db_type not in db_types:
                        db_types[db_type] = []
                    db_types[db_type].append(conn['archivo'])

                for db_type, files in db_types.items():
                    f.write(f"- **{db_type}:** {len(files)} archivo(s)\n")
                f.write("\n")

            # Dependencias
            f.write("## 📦 Dependencias Principales\n\n")
            f.write(
                f"**Total de dependencias de terceros:** {len(self.analysis['dependencias']['imports_terceros'])}\n\n")
            if self.analysis['dependencias']['imports_terceros']:
                f.write("**Paquetes principales:**\n")
                for dep in self.analysis['dependencias']['imports_terceros'][:30]:
                    f.write(f"- {dep}\n")
            f.write("\n")

            # Archivos de configuración
            f.write("## ⚙️ Archivos de Configuración\n\n")
            if self.analysis["configuraciones"]:
                for config in self.analysis["configuraciones"]:
                    f.write(f"- {config['archivo']}\n")
            else:
                f.write("*No se detectaron archivos de configuración estándar*\n")
            f.write("\n")

            # Recomendaciones para orquestador
            f.write("## 🎯 Recomendaciones para el Orquestador\n\n")
            f.write(self.generar_recomendaciones())

        print(f"✅ Reporte Markdown guardado en: {output_md}")

    def generar_recomendaciones(self) -> str:
        """Genera recomendaciones basadas en el análisis"""
        recomendaciones = []

        if 'Celery' in self.analysis["frameworks"]:
            recomendaciones.append(
                "- Ya usas **Celery**. Considera mantenerlo o migrarlo a una solución moderna como Prefect o Temporal.")
        elif 'Apache Airflow' in self.analysis["frameworks"]:
            recomendaciones.append("- Ya usas **Airflow**. Tienes un orquestador robusto para workflows tipo DAG.")
        else:
            recomendaciones.append(
                "- No se detectó un orquestador existente. Considera usar: **Celery** (tareas async), **Prefect** (workflows modernos), o **Temporal** (workflows complejos).")

        if len(self.analysis["tareas_async"]) > 10:
            recomendaciones.append(
                f"- Detectaste {len(self.analysis['tareas_async'])} funciones async. Aprovecha esto con un orquestador que soporte async nativo.")

        if len(self.analysis["apis_endpoints"]) > 0:
            recomendaciones.append(
                "- Tienes endpoints API. El orquestador debe poder ser invocado desde tu API (webhooks, REST calls).")

        if any(db in str(self.analysis["conexiones_db"]) for db in ['redis', 'Redis']):
            recomendaciones.append("- Ya usas Redis. Es ideal como broker para Celery o backend de resultados.")

        if 'Pipeline Pattern' in self.analysis["patrones_detectados"]:
            recomendaciones.append(
                "- Detectado patrón Pipeline. Considera orquestadores como Prefect o Dagster para pipelines de datos.")

        if not recomendaciones:
            recomendaciones.append(
                "- Analiza los requisitos específicos de tu orquestador: ¿tareas programadas, workflows, DAGs, o procesamiento en tiempo real?")

        return "\n".join(recomendaciones) + "\n"


def main():
    if len(sys.argv) < 2:
        print("Uso: python analizar_repo.py <ruta_del_repositorio>")
        print("Ejemplo: python analizar_repo.py /ruta/a/tu/proyecto")
        sys.exit(1)

    repo_path = sys.argv[1]

    if not os.path.exists(repo_path):
        print(f"❌ Error: La ruta '{repo_path}' no existe")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print("🔍 ANALIZADOR DE REPOSITORIO PYTHON")
    print(f"{'=' * 60}\n")

    analyzer = RepositoryAnalyzer(repo_path)
    analyzer.analizar_todo()

    print(f"\n{'=' * 60}")
    print("✅ ¡Análisis completado!")
    print(f"{'=' * 60}")
    print("\n📄 Archivos generados:")
    print(f"  - analisis_repo.json")
    print(f"  - ANALISIS_REPO.md")
    print("\n💡 Próximos pasos:")
    print("  1. Revisa el archivo ANALISIS_REPO.md")
    print("  2. Comparte el JSON conmigo para ayudarte con el orquestador")
    print("  3. Identifica las tareas que quieres orquestar\n")


if __name__ == "__main__":
    main()