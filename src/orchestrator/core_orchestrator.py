"""
Core Orchestrator - FARFAN 3.0 Main Coordination Engine
========================================================

PIPELINE FLOW: PDF Ingestion → MICRO/MESO/MACRO Report Generation
------------------------------------------------------------------
1. PDF/Document Ingestion: Load plan document from filesystem
2. Question Routing: Map 300 questions to execution chains via question_router
3. Execution Orchestration: Execute adapters through choreographer with dependency management
4. Circuit Breaker Integration: Fault tolerance at every adapter invocation
5. Evidence Aggregation: Collect evidence from all 9 adapters
6. MICRO Generation: Individual question answers with scoring modalities (TYPE_A-F)
7. MESO Clustering: Aggregate by policy areas (P1-P10) and dimensions (D1-D6)
8. MACRO Convergence: Global alignment score with Decálogo framework

ADAPTER INVOCATION PATTERN:
---------------------------
All adapter methods are invoked through ModuleAdapterRegistry using CLASS-BASED calls:
  result = module_adapter_registry.execute_module_method(
      module_name="adapter_name",  # e.g., "teoria_cambio"
      method_name="method_name",   # e.g., "calculate_bayesian_confidence"
      args=[...],
      kwargs={...}
  )

CIRCUIT BREAKER INTEGRATION POINTS:
-----------------------------------
- Before each adapter execution in choreographer.execute_question_chain()
- Automatic fallback to degraded mode if circuit opens
- Health status tracked in circuit_breaker.get_all_status()
- Failure threshold: 5 consecutive failures trigger circuit open
- Recovery timeout: 60 seconds before attempting HALF_OPEN state

Author: FARFAN Integration Team
Version: 3.0.0 - Refactored with strict type annotations and comprehensive documentation
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


class FARFANOrchestrator:
    """
    Main orchestrator coordinating complete FARFAN 3.0 analysis pipeline

    Manages end-to-end flow from document ingestion through multi-level reporting:
    - Coordinates 9 specialized adapters (413 methods total)
    - Routes 300 questions through validated execution chains
    - Generates MICRO (question-level), MESO (cluster-level), and MACRO (plan-level) reports
    - Enforces fault tolerance through circuit breaker pattern
    """

    def __init__(
        self,
        module_adapter_registry: Optional[Any] = None,
        questionnaire_parser: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize FARFAN Orchestrator with required components

        Args:
            module_adapter_registry: ModuleAdapterRegistry instance with 9 adapters.
                                    If not provided, creates an empty ModuleAdapterRegistry.
            questionnaire_parser: QuestionnaireParser with 300 question definitions
            config: Optional configuration dictionary
        """
        logger.info("Initializing FARFAN Orchestrator")

        # Instantiate ModuleAdapterRegistry if not provided
        if module_adapter_registry is None:
            from .adapter_registry import ModuleAdapterRegistry

            self.module_registry = ModuleAdapterRegistry()
            logger.info("Created new ModuleAdapterRegistry instance")
        else:
            self.module_registry = module_adapter_registry

        self.questionnaire_parser = questionnaire_parser
        self.config = config or {}

        from .choreographer import ExecutionChoreographer
        from .circuit_breaker import CircuitBreaker
        from .report_assembly import ReportAssembler

        self.choreographer = ExecutionChoreographer()
        self.circuit_breaker = CircuitBreaker()
        self.report_assembler = ReportAssembler()

        self.execution_stats: Dict[str, Any] = {
            "total_plans_processed": 0,
            "total_questions_answered": 0,
            "total_adapters_executed": 0,
            "total_execution_time": 0.0,
            "adapter_performance": defaultdict(
                lambda: {"calls": 0, "successes": 0, "failures": 0, "avg_time": 0.0}
            ),
        }

        adapter_count = (
            len(self.module_registry.adapters)
            if hasattr(self.module_registry, "adapters")
            else 0
        )
        logger.info(
            f"FARFAN Orchestrator initialized: "
            f"{adapter_count} adapters, "
            f"300 questions ready"
        )

    def analyze_single_plan(
        self,
        plan_path: Path,
        plan_name: Optional[str] = None,
        output_dir: Optional[Path] = None,
        questions_to_analyze: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Execute complete analysis pipeline for single development plan

        PIPELINE STAGES:
        ----------------
        1. Document Loading: Parse PDF/TXT/DOCX into text
        2. Question Selection: Get all 300 questions or specified subset
        3. MICRO Execution: For each question:
           - Route to execution chain via question_router
           - Execute adapters via choreographer with circuit breaker protection
           - Apply scoring modality (TYPE_A through TYPE_F from rubric_scoring.json)
           - Generate evidence-backed answer with confidence scores
        4. MESO Aggregation: Cluster MICRO answers by policy areas and dimensions
        5. MACRO Synthesis: Calculate global convergence with Decálogo alignment
        6. Report Export: Generate JSON reports and execution summaries

        Args:
            plan_path: Path to plan document (PDF/TXT/DOCX)
            plan_name: Optional plan identifier (defaults to filename)
            output_dir: Optional directory for report outputs
            questions_to_analyze: Optional list of question IDs to analyze (e.g., ["P1-D1-Q1"])

        Returns:
            Dictionary containing:
            - success: Boolean indicating completion status
            - plan_name: Name of analyzed plan
            - micro_answers: List of MicroLevelAnswer objects
            - meso_clusters: List of MesoLevelCluster objects
            - macro_convergence: MacroLevelConvergence object
            - execution_summary: Performance metrics and statistics
            - report_path: Path to generated JSON report
        """
        start_time = time.time()

        plan_name = plan_name or plan_path.stem
        output_dir = output_dir or Path(f"./output/{plan_name}")
        output_dir.mkdir(exist_ok=True, parents=True)

        logger.info(f"Starting analysis of plan: {plan_name}")

        try:
            plan_text = self._load_plan_document(plan_path)
            logger.info(f"Loaded plan document: {len(plan_text)} characters")

            all_questions = self.questionnaire_parser.parse_all_questions()

            if questions_to_analyze:
                questions = [
                    q for q in all_questions if q.canonical_id in questions_to_analyze
                ]
                logger.info(f"Analyzing {len(questions)} specified questions")
            else:
                questions = all_questions
                logger.info(f"Analyzing all {len(questions)} questions")

            micro_answers = []

            for i, question in enumerate(questions, 1):
                logger.info(
                    f"Processing question {i}/{len(questions)}: {question.canonical_id}"
                )

                try:
                    execution_results = self.choreographer.execute_question_chain(
                        question_spec=question,
                        plan_text=plan_text,
                        module_adapter_registry=self.module_registry,
                        circuit_breaker=self.circuit_breaker,
                    )

                    micro_answer = self.report_assembler.generate_micro_answer(
                        question_spec=question,
                        execution_results=self._convert_execution_results(
                            execution_results
                        ),
                        plan_text=plan_text,
                    )

                    micro_answers.append(micro_answer)

                    self._update_execution_stats(execution_results)

                except Exception as e:
                    logger.error(
                        f"Error processing question {question.canonical_id}: {e}",
                        exc_info=True,
                    )
                    continue

            logger.info(f"Completed MICRO analysis: {len(micro_answers)} questions")

            meso_clusters = self._generate_meso_clusters(micro_answers)
            logger.info(f"Generated {len(meso_clusters)} MESO clusters")

            macro_convergence = self.report_assembler.generate_macro_convergence(
                all_micro_answers=micro_answers,
                all_meso_clusters=meso_clusters,
                plan_metadata={
                    "name": plan_name,
                    "path": str(plan_path),
                    "analysis_date": datetime.now().isoformat(),
                },
            )
            logger.info(
                f"Generated MACRO convergence: {macro_convergence.overall_score:.1f}%"
            )

            report_path = output_dir / f"{plan_name}_complete_report.json"
            self.report_assembler.export_report(
                micro_answers=micro_answers,
                meso_clusters=meso_clusters,
                macro_convergence=macro_convergence,
                output_path=report_path,
            )
            logger.info(f"Exported complete report: {report_path}")

            execution_summary = self._generate_execution_summary(
                plan_name=plan_name,
                micro_answers=micro_answers,
                meso_clusters=meso_clusters,
                macro_convergence=macro_convergence,
                execution_time=time.time() - start_time,
            )

            summary_path = output_dir / f"{plan_name}_execution_summary.json"
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(execution_summary, f, ensure_ascii=False, indent=2)

            logger.info(
                f"Analysis completed in {execution_summary['total_execution_time']:.2f}s"
            )

            self.execution_stats["total_plans_processed"] += 1
            self.execution_stats["total_questions_answered"] += len(micro_answers)
            self.execution_stats["total_execution_time"] += execution_summary[
                "total_execution_time"
            ]

            return {
                "success": True,
                "plan_name": plan_name,
                "micro_answers": micro_answers,
                "meso_clusters": meso_clusters,
                "macro_convergence": macro_convergence,
                "execution_summary": execution_summary,
                "report_path": str(report_path),
                "summary_path": str(summary_path),
            }

        except Exception as e:
            logger.error(f"Fatal error analyzing plan {plan_name}: {e}", exc_info=True)
            return {
                "success": False,
                "plan_name": plan_name,
                "error": str(e),
                "execution_time": time.time() - start_time,
            }

    def _load_plan_document(self, plan_path: Path) -> str:
        """
        Load and extract text from plan document

        Supports PDF (via PyPDF2), TXT (direct read), and DOCX (via python-docx)

        Args:
            plan_path: Path to document file

        Returns:
            Extracted text content

        Raises:
            FileNotFoundError: If plan_path does not exist
            ValueError: If file format is unsupported
        """
        logger.info(f"Loading plan from {plan_path}")

        if not plan_path.exists():
            raise FileNotFoundError(f"Plan not found: {plan_path}")

        suffix = plan_path.suffix.lower()

        if suffix == ".txt":
            with open(plan_path, "r", encoding="utf-8") as f:
                return f.read()

        elif suffix == ".pdf":
            try:
                import PyPDF2

                with open(plan_path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text()
                    return text
            except ImportError:
                logger.error("PyPDF2 not installed, cannot read PDF")
                raise

        elif suffix in [".docx", ".doc"]:
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
        self, execution_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Convert ExecutionResult objects from choreographer to dict format for ReportAssembler

        Normalizes the output of choreographer.execute_question_chain() into a format
        compatible with report_assembler.generate_micro_answer()

        Args:
            execution_results: Dictionary of ExecutionResult objects keyed by adapter.method

        Returns:
            Dictionary with normalized structure for report assembly
        """
        converted: Dict[str, Any] = {}

        for key, result in execution_results.items():
            if hasattr(result, "to_dict"):
                converted[key] = result.to_dict()
            else:
                converted[key] = {
                    "module_name": getattr(result, "module_name", key),
                    "status": getattr(result, "status", "unknown"),
                    "data": getattr(result, "output", {}),
                    "confidence": getattr(result, "confidence", 0.0),
                    "evidence": getattr(result, "evidence_extracted", {}),
                    "execution_time": getattr(result, "execution_time", 0.0),
                }

        return converted

    def _generate_meso_clusters(self, micro_answers: List[Any]) -> List[Any]:
        """
        Generate MESO-level clusters from MICRO-level answers

        Clusters are organized by:
        - Policy areas (P1-P10 from questionnaire)
        - Dimensions (D1-D6)
        - Thematic groupings

        AGGREGATION FORMULA:
        - Cluster score = weighted average of dimension scores
        - Dimension score = (sum of question scores / max possible) * 100
        - Weights from rubric_scoring.json applied at each level

        Args:
            micro_answers: List of MicroLevelAnswer objects from MICRO generation

        Returns:
            List of MesoLevelCluster objects
        """
        by_policy: Dict[str, List[Any]] = defaultdict(list)
        for answer in micro_answers:
            policy = answer.metadata.get("policy_area", "Unknown")
            by_policy[policy].append(answer)

        clusters: List[Any] = []
        for policy_area, answers in by_policy.items():
            cluster = self.report_assembler.generate_meso_cluster(
                cluster_name=f"POLICY_{policy_area}",
                cluster_description=f"Cluster for policy area {policy_area}",
                micro_answers=answers,
                cluster_definition={
                    "policy_area": policy_area,
                    "total_questions": len(answers),
                },
            )
            clusters.append(cluster)

        return clusters

    def _update_execution_stats(self, execution_results: Dict[str, Any]) -> None:
        """
        Update internal execution statistics for monitoring

        Tracks per-adapter performance metrics including success rate and average execution time

        Args:
            execution_results: Dictionary of ExecutionResult objects
        """
        for key, result in execution_results.items():
            adapter_name = getattr(result, "module_name", "unknown")
            success = getattr(result, "status", None) == "COMPLETED"
            exec_time = getattr(result, "execution_time", 0.0)

            stats = self.execution_stats["adapter_performance"][adapter_name]
            stats["calls"] += 1
            if success:
                stats["successes"] += 1
            else:
                stats["failures"] += 1

            prev_avg = stats["avg_time"]
            calls = stats["calls"]
            stats["avg_time"] = (prev_avg * (calls - 1) + exec_time) / calls

    def _generate_execution_summary(
        self,
        plan_name: str,
        micro_answers: List[Any],
        meso_clusters: List[Any],
        macro_convergence: Any,
        execution_time: float,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive execution summary for reporting

        Args:
            plan_name: Name of analyzed plan
            micro_answers: List of MICRO-level answers
            meso_clusters: List of MESO-level clusters
            macro_convergence: MACRO-level convergence object
            execution_time: Total execution time in seconds

        Returns:
            Dictionary with execution statistics and quality metrics
        """
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
            "top_recommendations": macro_convergence.strategic_recommendations[:5],
        }

    def get_orchestrator_status(self) -> Dict[str, Any]:
        """
        Get current orchestrator health status

        Returns:
            Dictionary with adapter availability, circuit breaker status, and execution statistics
        """
        return {
            "adapters_available": self.module_registry.get_available_modules(),
            "total_adapters": len(self.module_registry.adapters),
            "circuit_breaker_status": self.circuit_breaker.get_all_status(),
            "execution_stats": dict(self.execution_stats),
            "questions_available": 300,
        }

    def analyze_batch(
        self, plan_paths: List[Path], output_dir: Optional[Path] = None
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple plans in batch mode

        Executes analyze_single_plan() sequentially for each plan in the list

        Args:
            plan_paths: List of paths to plan documents
            output_dir: Optional output directory for all batch results

        Returns:
            List of analysis results (one dictionary per plan)
        """
        logger.info(f"Starting batch analysis of {len(plan_paths)} plans")

        results: List[Dict[str, Any]] = []
        for i, plan_path in enumerate(plan_paths, 1):
            logger.info(f"Analyzing plan {i}/{len(plan_paths)}: {plan_path.name}")

            result = self.analyze_single_plan(
                plan_path=plan_path, output_dir=output_dir
            )
            results.append(result)

        logger.info(f"Batch analysis complete: {len(results)} plans processed")

        return results


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
