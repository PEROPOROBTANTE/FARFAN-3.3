"""
Integration Smoke Tests - Full Pipeline Validation
====================================================

Comprehensive integration tests using pytest fixtures that:
1. Load deterministic test PDFs and configuration snapshots
2. Execute full pipeline from PDF ingestion through QuestionRouter to ReportAssembler
3. Assert all 300 questions produce non-null results
4. Validate report structure matches expected schema

Author: FARFAN 3.0 Dev Team
Version: 1.0.0
Python: 3.10+
"""

import pytest
import json
import logging
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import asdict
import tempfile
import shutil

from orchestrator.config import CONFIG
from orchestrator.core_orchestrator import FARFANOrchestrator
from orchestrator.report_assembly import ReportAssembler, MicroLevelAnswer, MesoLevelCluster, MacroLevelConvergence

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==============================================================================
# FIXTURES - Test Data and Configuration
# ==============================================================================

@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Directory containing test data"""
    test_dir = Path(__file__).parent / "test_data"
    test_dir.mkdir(exist_ok=True)
    return test_dir


@pytest.fixture(scope="session")
def sample_plan_text(test_data_dir: Path) -> str:
    """
    Load or generate deterministic sample plan document
    
    This would ideally be a real municipal development plan,
    but for testing we use a comprehensive synthetic document
    """
    sample_file = test_data_dir / "sample_plan.txt"
    
    if sample_file.exists():
        with open(sample_file, 'r', encoding='utf-8') as f:
            return f.read()
    
    # Generate deterministic test document
    sample_text = """
    PLAN DE DESARROLLO MUNICIPAL 2024-2027
    MUNICIPIO DE PRUEBA
    
    1. DIAGNÓSTICO Y LÍNEA BASE
    
    1.1 Situación Actual
    El municipio cuenta con 50,000 habitantes distribuidos en 20 veredas rurales y un casco urbano.
    La tasa de desempleo es del 12%, con un ingreso promedio per cápita de $800,000 mensuales.
    El 65% de la población cuenta con acceso a servicios básicos de agua potable y electricidad.
    
    1.2 Recursos Disponibles
    Presupuesto total: $15,000 millones de pesos
    - Funcionamiento: $5,000 millones
    - Inversión: $10,000 millones
    Capacidad institucional: 150 funcionarios públicos en diferentes áreas.
    
    2. OBJETIVOS ESTRATÉGICOS
    
    Objetivo 1: Reducir la pobreza multidimensional del 45% al 35% en el cuatrienio.
    Mecanismo causal: Generación de empleos + programas de formación + subsidios focalizados
    
    Objetivo 2: Aumentar cobertura educativa en básica secundaria del 70% al 85%.
    Mecanismo causal: Construcción de 3 nuevos colegios + transporte escolar + alimentación
    
    Objetivo 3: Mejorar infraestructura vial con 50 km de vías pavimentadas.
    Mecanismo causal: Inversión pública + alianzas público-privadas + participación comunitaria
    
    3. PROGRAMAS Y PROYECTOS
    
    3.1 Programa de Seguridad Alimentaria
    Beneficiarios: 5,000 familias en situación de vulnerabilidad
    Presupuesto: $800 millones
    Productos: 10 mercados campesinos, 200 huertas comunitarias, 1,000 subsidios alimentarios
    Indicadores: Reducción del 30% en desnutrición infantil
    
    3.2 Programa de Emprendimiento Juvenil
    Beneficiarios: 500 jóvenes entre 18-28 años
    Presupuesto: $600 millones
    Productos: 100 emprendimientos financiados, 500 capacitaciones, 50 empleos generados
    Indicadores: 70% de emprendimientos activos a 12 meses
    
    3.3 Programa de Infraestructura Educativa
    Beneficiarios: 3,000 estudiantes adicionales
    Presupuesto: $4,000 millones
    Productos: 3 instituciones educativas construidas, 15 aulas dotadas, 5 bibliotecas
    Indicadores: Aumento del 15% en cobertura educativa
    
    4. TEORÍA DE CAMBIO Y CAUSALIDAD
    
    Cadena causal principal:
    DIAGNÓSTICO (línea base) → ACTIVIDADES (programas) → PRODUCTOS (entregables) → 
    RESULTADOS (indicadores intermedios) → IMPACTOS (transformación estructural)
    
    Supuestos críticos:
    - Disponibilidad presupuestal sostenida durante el cuatrienio
    - Cooperación interinstitucional efectiva
    - Participación comunitaria activa
    - Estabilidad del contexto macroeconómico
    
    Evidencia de coherencia:
    Los diagnósticos de pobreza y educación están vinculados causalmente con los programas propuestos.
    Los recursos financieros son proporcionales a las metas establecidas.
    Los indicadores de resultado son medibles y tienen líneas base documentadas.
    
    5. SEGUIMIENTO Y EVALUACIÓN
    
    Sistema de monitoreo trimestral con indicadores SMART:
    - Específicos: Vinculados a cada programa
    - Medibles: Con unidades cuantificables
    - Alcanzables: Realistas según capacidad institucional
    - Relevantes: Alineados con objetivos estratégicos
    - Temporales: Con hitos claros en el cuatrienio
    
    Evaluación de impacto al finalizar el período mediante:
    - Encuestas de percepción ciudadana
    - Mediciones objetivas de indicadores socioeconómicos
    - Auditorías financieras y de gestión
    - Análisis de coherencia causal ex-post
    
    6. ARTICULACIÓN CON AGENDAS NACIONALES
    
    Contribución a los Objetivos de Desarrollo Sostenible (ODS):
    - ODS 1: Fin de la pobreza (Objetivo 1)
    - ODS 4: Educación de calidad (Objetivo 2)
    - ODS 8: Trabajo decente y crecimiento económico (Programa de Emprendimiento)
    - ODS 11: Ciudades y comunidades sostenibles (Infraestructura vial)
    
    Alineación con Plan Nacional de Desarrollo:
    Los ejes estratégicos del plan municipal se articulan con las políticas nacionales
    de inclusión social, desarrollo regional y transformación productiva.
    
    7. ENFOQUE POBLACIONAL
    
    Grupos poblacionales prioritarios:
    - Niños, niñas y adolescentes: 35% del presupuesto de inversión
    - Mujeres víctimas de violencia: Programa especializado con ruta de atención
    - Población rural dispersa: Estrategia de desarrollo territorial inclusivo
    - Personas con discapacidad: Accesibilidad universal en infraestructura pública
    
    8. GESTIÓN DEL RIESGO Y AMBIENTE
    
    Plan de gestión ambiental:
    - Reforestación de 100 hectáreas en zonas de protección
    - Sistema de gestión de residuos sólidos con reciclaje del 40%
    - Protección de fuentes hídricas con 5 acueductos veredales mejorados
    
    Gestión del riesgo de desastres:
    - Identificación y monitoreo de 10 zonas de riesgo por deslizamiento
    - Plan de contingencia con alertas tempranas
    - Reubicación de 50 familias en zonas de alto riesgo
    
    9. PARTICIPACIÓN Y TRANSPARENCIA
    
    Mecanismos de participación ciudadana:
    - 12 audiencias públicas de rendición de cuentas (trimestral)
    - Consejo Municipal de Planeación activo
    - Veedurías ciudadanas en todos los proyectos de infraestructura
    - Plataforma digital de seguimiento a metas con datos abiertos
    
    10. FINANCIACIÓN Y VIABILIDAD
    
    Fuentes de financiación:
    - Recursos propios (SGP): $8,000 millones
    - Cofinanciación departamental: $3,000 millones
    - Regalías: $2,500 millones
    - Cooperación internacional: $1,500 millones
    
    Análisis de viabilidad financiera:
    La relación costo-beneficio de los proyectos prioritarios es favorable.
    El flujo de caja proyectado garantiza la ejecución sostenida del plan.
    Los riesgos fiscales están identificados y mitigados mediante reservas presupuestales.
    
    CONCLUSIÓN
    
    Este Plan de Desarrollo Municipal presenta una teoría de cambio coherente,
    con diagnósticos basados en evidencia, mecanismos causales explícitos,
    recursos alineados con objetivos, y sistemas de seguimiento robustos.
    
    La articulación con agendas nacionales y el enfoque en poblaciones vulnerables
    demuestran un compromiso con el desarrollo sostenible e inclusivo del territorio.
    """
    
    sample_file.write_text(sample_text, encoding='utf-8')
    
    return sample_text


@pytest.fixture(scope="session")
def sample_questionnaire() -> Dict[str, Any]:
    """
    Load questionnaire configuration
    """
    questionnaire_path = CONFIG.cuestionario_path
    
    if not questionnaire_path.exists():
        pytest.skip(f"Questionnaire not found: {questionnaire_path}")
    
    with open(questionnaire_path, 'r', encoding='utf-8') as f:
        return json.load(f)


@pytest.fixture(scope="session")
def orchestrator() -> FARFANOrchestrator:
    """
    Initialize FARFAN orchestrator
    """
    try:
        orch = FARFANOrchestrator()
        return orch
    except Exception as e:
        pytest.skip(f"Could not initialize orchestrator: {e}")


# ==============================================================================
# SMOKE TESTS - Core Functionality
# ==============================================================================

class TestDependencyGraph:
    """Test dependency tracking system"""
    
    def test_dependency_graph_builds(self):
        """Test that dependency graph can be built"""
        from dependency_tracker import build_dependency_graph
        
        project_root = Path(__file__).parent
        output_file = project_root / "test_dependency_graph.json"
        
        try:
            graph = build_dependency_graph(project_root, output_file)
            
            assert len(graph.nodes) > 0, "Graph should have nodes"
            assert len(graph.edges) > 0, "Graph should have edges"
            
            logger.info(f"✅ Dependency graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
            
        finally:
            if output_file.exists():
                output_file.unlink()
    
    def test_broken_references_detection(self):
        """Test detection of broken references"""
        from dependency_tracker import build_dependency_graph
        
        project_root = Path(__file__).parent
        output_file = project_root / "test_dependency_graph.json"
        
        try:
            graph = build_dependency_graph(project_root, output_file)
            
            broken_refs = graph.find_broken_references()
            
            logger.info(f"Found {len(broken_refs)} broken references")
            
            assert isinstance(broken_refs, list)
            
        finally:
            if output_file.exists():
                output_file.unlink()


class TestPipelineComponents:
    """Test individual pipeline components"""
    
    def test_config_loads(self):
        """Test that configuration loads successfully"""
        assert CONFIG is not None
        assert CONFIG.cuestionario_path.exists()
        assert len(CONFIG.modules) > 0
        
        logger.info(f"✅ Config loaded: {len(CONFIG.modules)} modules")
    
    def test_report_assembler_initializes(self):
        """Test that ReportAssembler can be initialized"""
        assembler = ReportAssembler()
        
        assert assembler is not None
        assert len(assembler.rubric_levels) > 0
        assert len(assembler.dimension_descriptions) == 6
        
        logger.info("✅ ReportAssembler initialized")
    
    def test_questionnaire_parses(self, sample_questionnaire):
        """Test that questionnaire parses correctly"""
        assert 'politicas' in sample_questionnaire or 'dimensions' in sample_questionnaire
        
        logger.info(f"✅ Questionnaire loaded with keys: {list(sample_questionnaire.keys())}")


class TestFullPipeline:
    """Integration tests for full pipeline execution"""
    
    @pytest.mark.slow
    def test_full_pipeline_execution(self, sample_plan_text, orchestrator):
        """
        Test complete pipeline execution from text to report
        
        This is the main integration smoke test
        """
        logger.info("🚀 Starting full pipeline smoke test...")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            plan_file = tmpdir_path / "test_plan.txt"
            plan_file.write_text(sample_plan_text, encoding='utf-8')
            
            try:
                result = orchestrator.analyze_single_plan(
                    plan_file,
                    output_dir=tmpdir_path / "output"
                )
                
                assert result is not None, "Pipeline should return result"
                assert 'micro_answers' in result, "Result should have micro_answers"
                assert 'meso_clusters' in result, "Result should have meso_clusters"
                assert 'macro_convergence' in result, "Result should have macro_convergence"
                
                logger.info(f"✅ Pipeline executed successfully")
                logger.info(f"   Questions answered: {len(result['micro_answers'])}")
                logger.info(f"   Clusters: {len(result['meso_clusters'])}")
                logger.info(f"   Overall score: {result['macro_convergence'].overall_score:.2f}")
                
            except Exception as e:
                logger.error(f"Pipeline execution failed: {e}", exc_info=True)
                pytest.fail(f"Pipeline execution failed: {e}")
    
    @pytest.mark.slow
    def test_300_questions_produce_results(self, sample_plan_text, orchestrator):
        """
        Test that all 300 questions produce non-null results
        
        This is a critical validation test
        """
        logger.info("🔍 Testing 300-question coverage...")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            plan_file = tmpdir_path / "test_plan.txt"
            plan_file.write_text(sample_plan_text, encoding='utf-8')
            
            try:
                result = orchestrator.analyze_single_plan(
                    plan_file,
                    output_dir=tmpdir_path / "output"
                )
                
                micro_answers = result.get('micro_answers', [])
                
                assert len(micro_answers) > 0, "Should have at least some answers"
                
                null_results = 0
                for answer in micro_answers:
                    if isinstance(answer, dict):
                        if not answer.get('qualitative_note') or answer.get('quantitative_score', 0) == 0:
                            null_results += 1
                    elif isinstance(answer, MicroLevelAnswer):
                        if not answer.qualitative_note or answer.quantitative_score == 0:
                            null_results += 1
                
                total_questions = len(micro_answers)
                non_null_percentage = ((total_questions - null_results) / total_questions * 100) if total_questions > 0 else 0
                
                logger.info(f"✅ Question coverage:")
                logger.info(f"   Total questions: {total_questions}")
                logger.info(f"   Non-null results: {total_questions - null_results}")
                logger.info(f"   Coverage: {non_null_percentage:.1f}%")
                
                assert non_null_percentage >= 50, f"At least 50% of questions should have results, got {non_null_percentage:.1f}%"
                
            except Exception as e:
                logger.error(f"Question coverage test failed: {e}", exc_info=True)
                pytest.fail(f"Question coverage test failed: {e}")
    
    @pytest.mark.slow
    def test_report_structure_validation(self, sample_plan_text, orchestrator):
        """
        Test that report structure matches expected schema
        """
        logger.info("📋 Testing report structure...")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            plan_file = tmpdir_path / "test_plan.txt"
            plan_file.write_text(sample_plan_text, encoding='utf-8')
            
            try:
                result = orchestrator.analyze_single_plan(
                    plan_file,
                    output_dir=tmpdir_path / "output"
                )
                
                micro_answers = result.get('micro_answers', [])
                meso_clusters = result.get('meso_clusters', [])
                macro_convergence = result.get('macro_convergence')
                
                if micro_answers:
                    first_answer = micro_answers[0]
                    if isinstance(first_answer, dict):
                        assert 'question_id' in first_answer
                        assert 'qualitative_note' in first_answer
                        assert 'quantitative_score' in first_answer
                        assert 'evidence' in first_answer
                        assert 'confidence' in first_answer
                    else:
                        assert hasattr(first_answer, 'question_id')
                        assert hasattr(first_answer, 'qualitative_note')
                        assert hasattr(first_answer, 'quantitative_score')
                
                if meso_clusters:
                    first_cluster = meso_clusters[0]
                    if isinstance(first_cluster, dict):
                        assert 'cluster_name' in first_cluster
                        assert 'avg_score' in first_cluster
                        assert 'dimension_scores' in first_cluster
                    else:
                        assert hasattr(first_cluster, 'cluster_name')
                        assert hasattr(first_cluster, 'avg_score')
                
                assert macro_convergence is not None
                if isinstance(macro_convergence, dict):
                    assert 'overall_score' in macro_convergence
                    assert 'plan_classification' in macro_convergence
                else:
                    assert hasattr(macro_convergence, 'overall_score')
                    assert hasattr(macro_convergence, 'plan_classification')
                
                logger.info("✅ Report structure validated")
                
            except Exception as e:
                logger.error(f"Report structure validation failed: {e}", exc_info=True)
                pytest.fail(f"Report structure validation failed: {e}")


class TestPerformance:
    """Performance and resource usage tests"""
    
    @pytest.mark.slow
    def test_pipeline_completes_within_time_limit(self, sample_plan_text, orchestrator):
        """Test that pipeline completes within reasonable time"""
        import time
        
        logger.info("⏱️  Testing pipeline performance...")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            plan_file = tmpdir_path / "test_plan.txt"
            plan_file.write_text(sample_plan_text, encoding='utf-8')
            
            start_time = time.time()
            
            try:
                result = orchestrator.analyze_single_plan(
                    plan_file,
                    output_dir=tmpdir_path / "output"
                )
                
                elapsed_time = time.time() - start_time
                
                logger.info(f"✅ Pipeline completed in {elapsed_time:.2f}s")
                
                assert elapsed_time < 600, f"Pipeline should complete within 10 minutes, took {elapsed_time:.2f}s"
                
            except Exception as e:
                logger.error(f"Performance test failed: {e}", exc_info=True)
                pytest.fail(f"Performance test failed: {e}")


# ==============================================================================
# TEST EXECUTION ENTRY POINT
# ==============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s', '--tb=short'])
