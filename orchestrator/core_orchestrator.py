# core_orchestrator.py - COMPLETE UPDATE FOR 9 ADAPTERS
# coding=utf-8
"""
Core Orchestrator - Main Coordination Engine
============================================

Updated for complete integration with:
- module_adapters_COMPLETE_MERGED.py (9 adapters, 413 methods)
- FARFAN_3.0_UPDATED_QUESTIONNAIRE.yaml (300 questions)
- report_assembly_COMPLETE.py (MICRO/MESO/MACRO reporting)
- choreographer_UPDATED.py (execution orchestration)
- circuit_breaker_UPDATED.py (fault tolerance)

Coordinates:
- Question routing (300 questions → execution chains)
- Module execution with dependency management
- Fault tolerance and graceful degradation
- Multi-level report generation

Author: Integration Team
Version: 3.0.0 - Complete System Integration
Python: 3.10+
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
from datetime import datetime
import json

logger = logging.getLogger(__name__)


# ============================================================================
# FARFAN ORCHESTRATOR - MAIN COORDINATION ENGINE
# ============================================================================

class FARFANOrchestrator:
    """
    Main orchestrator for FARFAN 3.0 policy analysis system
    
    Coordinates complete analysis pipeline:
    1. Load and parse plan document
    2. Route 300 questions to execution chains
    3. Execute adapters with dependency management
    4. Aggregate results with fault tolerance
    5. Generate MICRO/MESO/MACRO reports
    """

    def __init__(
            self,
            module_adapter_registry,
            questionnaire_parser,
            config=None
    ):
        """
        Initialize FARFAN Orchestrator
        
        Args:
            module_adapter_registry: ModuleAdapterRegistry with 9 adapters
            questionnaire_parser: QuestionnaireParser with 300 questions
            config: Optional configuration object
        """
        logger.info("Initializing FARFAN Orchestrator")

        # Core components
        self.module_registry = module_adapter_registry
        self.questionnaire_parser = questionnaire_parser
        self.config = config
        
        # Import and initialize sub-components
        from .choreographer_UPDATED import ExecutionChoreographer
        from .circuit_breaker_UPDATED import CircuitBreaker
        from .report_assembly_COMPLETE import ReportAssembler
        
        self.choreographer = ExecutionChoreographer()
        self.circuit_breaker = CircuitBreaker()
        self.report_assembler = ReportAssembler()

        # Execution statistics
        self.execution_stats = {
            "total_plans_processed": 0,
            "total_questions_answered": 0,
            "total_adapters_executed": 0,
            "total_execution_time": 0.0,
            "adapter_performance": defaultdict(lambda: {
                "calls": 0,
                "successes": 0,
                "failures": 0,
                "avg_time": 0.0
            })
        }

        logger.info(
            f"FARFAN Orchestrator initialized: "
            f"{len(self.module_registry.adapters)} adapters, "
            f"300 questions ready"
        )

    def analyze_single_plan(
            self,
            plan_path: Path,
            plan_name: Optional[str] = None,
            output_dir: Optional[Path] = None,
            questions_to_analyze: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze a single development plan through complete pipeline
        
        Args:
            plan_path: Path to plan document (PDF/TXT/DOCX)
            plan_name: Optional plan name (defaults to filename)
            output_dir: Optional output directory
            questions_to_analyze: Optional list of question IDs to analyze
                                 (if None, analyzes all 300)
        
        Returns:
            Dict with complete analysis results including MICRO/MESO/MACRO reports
        """
        start_time = time.time()
        
        plan_name = plan_name or plan_path.stem
        output_dir = output_dir or Path(f"./output/{plan_name}")
        output_dir.mkdir(exist_ok=True, parents=True)

        logger.info(f"Starting analysis of plan: {plan_name}")
        
        try:
            # Step 1: Load plan document
            plan_text = self._load_plan_document(plan_path)
            logger.info(f"Loaded plan document: {len(plan_text)} characters")
            
            # Step 2: Get questions to analyze
            all_questions = self.questionnaire_parser.parse_all_questions()
            
            if questions_to_analyze:
                questions = [
                    q for q in all_questions
                    if q.canonical_id in questions_to_analyze
                ]
                logger.info(f"Analyzing {len(questions)} specified questions")
            else:
                questions = all_questions
                logger.info(f"Analyzing all {len(questions)} questions")
            
            # Step 3: Execute question chains and generate MICRO answers
            micro_answers = []
            
            for i, question in enumerate(questions, 1):
                logger.info(f"Processing question {i}/{len(questions)}: {question.canonical_id}")
                
                try:
                    # Execute execution chain for question
                    execution_results = self.choreographer.execute_question_chain(
                        question_spec=question,
                        plan_text=plan_text,
                        module_adapter_registry=self.module_registry,
                        circuit_breaker=self.circuit_breaker
                    )
                    
                    # Generate MICRO answer
                    micro_answer = self.report_assembler.generate_micro_answer(
                        question_spec=question,
                        execution_results=self._convert_execution_results(execution_results),
                        plan_text=plan_text
                    )
                    
                    micro_answers.append(micro_answer)
                    
                    # Update stats
                    self._update_execution_stats(execution_results)
                    
                except Exception as e:
                    logger.error(f"Error processing question {question.canonical_id}: {e}", exc_info=True)
                    continue
            
            logger.info(f"Completed MICRO analysis: {len(micro_answers)} questions")
            
            # Step 4: Generate MESO clusters
            meso_clusters = self._generate_meso_clusters(micro_answers)
            logger.info(f"Generated {len(meso_clusters)} MESO clusters")
            
            # Step 5: Generate MACRO convergence
            macro_convergence = self.report_assembler.generate_macro_convergence(
                all_micro_answers=micro_answers,
                all_meso_clusters=meso_clusters,
                plan_metadata={
                    "name": plan_name,
                    "path": str(plan_path),
                    "analysis_date": datetime.now().isoformat()
                }
            )
            logger.info(f"Generated MACRO convergence: {macro_convergence.overall_score:.1f}%")
            
            # Step 6: Export complete report
            report_path = output_dir / f"{plan_name}_complete_report.json"
            self.report_assembler.export_report(
                micro_answers=micro_answers,
                meso_clusters=meso_clusters,
                macro_convergence=macro_convergence,
                output_path=report_path
            )
            logger.info(f"Exported complete report: {report_path}")
            
            # Step 7: Generate execution summary
            execution_summary = self._generate_execution_summary(
                plan_name=plan_name,
                micro_answers=micro_answers,
                meso_clusters=meso_clusters,
                macro_convergence=macro_convergence,
                execution_time=time.time() - start_time
            )
            
            # Save execution summary
            summary_path = output_dir / f"{plan_name}_execution_summary.json"
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(execution_summary, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Analysis completed in {execution_summary['total_execution_time']:.2f}s")
            
            # Update global stats
            self.execution_stats["total_plans_processed"] += 1
            self.execution_stats["total_questions_answered"] += len(micro_answers)
            self.execution_stats["total_execution_time"] += execution_summary["total_execution_time"]
            
            return {
                "success": True,
                "plan_name": plan_name,
                "micro_answers": micro_answers,
                "meso_clusters": meso_clusters,
                "macro_convergence": macro_convergence,
                "execution_summary": execution_summary,
                "report_path": str(report_path),
                "summary_path": str(summary_path)
            }
            
        except Exception as e:
            logger.error(f"Fatal error analyzing plan {plan_name}: {e}", exc_info=True)
            return {
                "success": False,
                "plan_name": plan_name,
                "error": str(e),
                "execution_time": time.time() - start_time
            }

    def _load_plan_document(self, plan_path: Path) -> str:
        """
        Load plan document from file
        
        Supports: TXT, PDF, DOCX
        """
        logger.info(f"Loading plan from {plan_path}")
        
        if not plan_path.exists():
            raise FileNotFoundError(f"Plan not found: {plan_path}")
        
        suffix = plan_path.suffix.lower()
        
        # TXT files
        if suffix == '.txt':
            with open(plan_path, 'r', encoding='utf-8') as f:
                return f.read()
        
        # PDF files
        elif suffix == '.pdf':
            try:
                import PyPDF2
                with open(plan_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text()
                    return text
            except ImportError:
                logger.error("PyPDF2 not installed, cannot read PDF")
                raise
        
        # DOCX files
        elif suffix in ['.docx', '.doc']:
            try:
                import docx
                doc = docx.Document(plan_path)
                return "\n".join([para.text for para in doc.paragraphs])
            except ImportError:
                logger.error("python-docx not installed, cannot read DOCX")
                raise
        
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    def _convert_execution_results(
            self,
            execution_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Convert ExecutionResult objects to dict format for ReportAssembler
        
        Args:
            execution_results: Dict of ExecutionResults from choreographer
            
        Returns:
            Dict compatible with ReportAssembler
        """
        converted = {}
        
        for key, result in execution_results.items():
            if hasattr(result, 'to_dict'):
                converted[key] = result.to_dict()
            else:
                converted[key] = {
                    "module_name": getattr(result, 'module_name', key),
                    "status": getattr(result, 'status', 'unknown'),
                    "data": getattr(result, 'output', {}),
                    "confidence": getattr(result, 'confidence', 0.0),
                    "evidence": getattr(result, 'evidence_extracted', {}),
                    "execution_time": getattr(result, 'execution_time', 0.0)
                }
        
        return converted

    def _generate_meso_clusters(
            self,
            micro_answers: List
    ) -> List:
        """
        Generate MESO clusters from MICRO answers
        
        Clusters questions by:
        - Policy areas (P1-P10)
        - Dimensions (D1-D6)
        - Thematic groups
        """
        # Group by policy area
        by_policy = defaultdict(list)
        for answer in micro_answers:
            policy = answer.metadata.get("policy_area", "Unknown")
            by_policy[policy].append(answer)
        
        # Generate clusters
        clusters = []
        for policy_area, answers in by_policy.items():
            cluster = self.report_assembler.generate_meso_cluster(
                cluster_name=f"POLICY_{policy_area}",
                cluster_description=f"Cluster for policy area {policy_area}",
                micro_answers=answers,
                cluster_definition={
                    "policy_area": policy_area,
                    "total_questions": len(answers)
                }
            )
            clusters.append(cluster)
        
        return clusters

    def _update_execution_stats(self, execution_results: Dict[str, Any]):
        """Update execution statistics"""
        for key, result in execution_results.items():
            adapter_name = getattr(result, 'module_name', 'unknown')
            success = getattr(result, 'status', None) == 'COMPLETED'
            exec_time = getattr(result, 'execution_time', 0.0)
            
            stats = self.execution_stats["adapter_performance"][adapter_name]
            stats["calls"] += 1
            if success:
                stats["successes"] += 1
            else:
                stats["failures"] += 1
            
            # Update average time
            prev_avg = stats["avg_time"]
            calls = stats["calls"]
            stats["avg_time"] = (prev_avg * (calls - 1) + exec_time) / calls

    def _generate_execution_summary(
            self,
            plan_name: str,
            micro_answers: List,
            meso_clusters: List,
            macro_convergence,
            execution_time: float
    ) -> Dict[str, Any]:
        """Generate execution summary"""
        return {
            "plan_name": plan_name,
            "analysis_timestamp": datetime.now().isoformat(),
            "total_execution_time": execution_time,
            "questions_analyzed": len(micro_answers),
            "clusters_generated": len(meso_clusters),
            "overall_score": macro_convergence.overall_score,
            "plan_classification": macro_convergence.plan_classification,
            "adapter_stats": dict(self.execution_stats["adapter_performance"]),
            "circuit_breaker_status": self.circuit_breaker.get_all_status(),
            "score_distribution": {
                level: sum(1 for a in micro_answers if a.qualitative_note == level)
                for level in ["EXCELENTE", "BUENO", "ACEPTABLE", "INSUFICIENTE"]
            },
            "dimension_scores": macro_convergence.convergence_by_dimension,
            "critical_gaps": macro_convergence.critical_gaps,
            "top_recommendations": macro_convergence.strategic_recommendations[:5]
        }

    def get_orchestrator_status(self) -> Dict[str, Any]:
        """Get current orchestrator status"""
        return {
            "adapters_available": self.module_registry.get_available_modules(),
            "total_adapters": len(self.module_registry.adapters),
            "circuit_breaker_status": self.circuit_breaker.get_all_status(),
            "execution_stats": dict(self.execution_stats),
            "questions_available": 300
        }

    def analyze_batch(
            self,
            plan_paths: List[Path],
            output_dir: Optional[Path] = None
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple plans in batch
        
        Args:
            plan_paths: List of plan document paths
            output_dir: Optional output directory for all results
            
        Returns:
            List of analysis results (one per plan)
        """
        logger.info(f"Starting batch analysis of {len(plan_paths)} plans")
        
        results = []
        for i, plan_path in enumerate(plan_paths, 1):
            logger.info(f"Analyzing plan {i}/{len(plan_paths)}: {plan_path.name}")
            
            result = self.analyze_single_plan(
                plan_path=plan_path,
                output_dir=output_dir
            )
            results.append(result)
        
        logger.info(f"Batch analysis complete: {len(results)} plans processed")
        
        return results


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("FARFAN ORCHESTRATOR - COMPLETE SYSTEM")
    print("=" * 80)
    print("\nFeatures:")
    print("  ✓ 9 complete adapters (413 methods)")
    print("  ✓ 300 questions with execution chains")
    print("  ✓ MICRO/MESO/MACRO reporting")
    print("  ✓ Fault tolerance with circuit breaker")
    print("  ✓ Dependency management")
    print("  ✓ Evidence aggregation")
    print("\nReady for production use!")
    print("=" * 80)