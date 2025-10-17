"""
Core Orchestrator - The main coordination engine
Integrates Router, Choreographer, Circuit Breaker, and Report Assembly

Ensures QuestionnaireParser is initialized and propagates dimension metadata.
"""
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
from datetime import datetime

from .config import CONFIG
from .question_router import QuestionRouter, Question
from .choreographer import ExecutionChoreographer, ExecutionResult
from .circuit_breaker import CircuitBreaker, create_module_specific_fallback
from .report_assembly import (
    ReportAssembler,
    MicroLevelAnswer,
    MesoLevelCluster,
    MacroLevelConvergence
)
from .questionnaire_parser import get_questionnaire_parser

logger = logging.getLogger(__name__)


class FARFANOrchestrator:
    """
    Main orchestrator for FARFAN 3.0 policy analysis system.

    Coordinates:
    - Question routing (300 questions → components)
    - Module execution with dependency management
    - Fault tolerance and graceful degradation
    - Multi-level report generation (MICRO/MESO/MACRO)
    
    Ensures QuestionnaireParser initialization for deterministic behavior.
    """

    def __init__(self):
        logger.info("Initializing FARFAN Orchestrator")

        # Initialize questionnaire parser (validates cuestionario.json)
        self.parser = get_questionnaire_parser()
        logger.info(f"Questionnaire parser initialized - Version {self.parser.version}, "
                   f"{self.parser.total_questions} total questions")

        self.router = QuestionRouter()
        self.choreographer = ExecutionChoreographer()
        self.circuit_breaker = CircuitBreaker()
        self.report_assembler = ReportAssembler()

        self.execution_stats = {
            "total_plans_processed": 0,
            "total_questions_answered": 0,
            "total_components_executed": 0,
            "total_execution_time": 0.0,
            "module_performance": defaultdict(lambda: {"calls": 0, "successes": 0, "failures": 0}),
            "component_performance": defaultdict(lambda: {"calls": 0, "successes": 0, "failures": 0})
        }

        logger.info("FARFAN Orchestrator initialized successfully")

    def analyze_single_plan(
            self,
            plan_path: Path,
            plan_name: Optional[str] = None,
            output_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Analyze a single development plan through the complete pipeline.

        Args:
            plan_path: Path to the plan document (PDF/TXT)
            plan_name: Optional name for the plan (defaults to filename)
            output_dir: Optional output directory (defaults to CONFIG.output_dir)

        Returns:
            Dict with full analysis results
        """
        start_time = time.time()

        plan_name = plan_name or plan_path.stem
        output_dir = output_dir or CONFIG.output_dir / plan_name

        output_dir.mkdir(exist_ok=True, parents=True)

        logger.info(f"=" * 80)
        logger.info(f"Starting analysis for plan: {plan_name}")
        logger.info(f"=" * 80)

        # Step 1: Load and preprocess plan
        logger.info("Step 1: Loading plan document")
        plan_text, plan_metadata = self._load_plan(plan_path)

        # Step 2: Process all 300 questions
        logger.info("Step 2: Processing 300 questions")
        micro_answers = self._process_all_questions(
            plan_text,
            plan_metadata,
            plan_name
        )

        # Step 3: Generate MESO-level cluster analysis
        logger.info("Step 3: Generating MESO-level cluster analysis")
        meso_clusters = self._generate_meso_level(micro_answers)

        # Step 4: Generate MACRO-level convergence analysis
        logger.info("Step 4: Generating MACRO-level convergence analysis")
        macro_convergence = self._generate_macro_level(micro_answers, meso_clusters)

        # Step 5: Export comprehensive report
        logger.info("Step 5: Exporting comprehensive report")
        self.report_assembler.export_full_report(
            plan_name,
            micro_answers,
            meso_clusters,
            macro_convergence,
            output_dir
        )

        execution_time = time.time() - start_time

        # Update statistics
        self.execution_stats["total_plans_processed"] += 1
        self.execution_stats["total_questions_answered"] += len(micro_answers)
        self.execution_stats["total_execution_time"] += execution_time

        logger.info(f"Analysis complete for {plan_name} in {execution_time:.2f}s")
        logger.info(f"Overall score: {macro_convergence.overall_score:.2f}")
        logger.info(f"Classification: {macro_convergence.plan_classification}")

        return {
            "plan_name": plan_name,
            "execution_time": execution_time,
            "micro_answers": micro_answers,
            "meso_clusters": meso_clusters,
            "macro_convergence": macro_convergence,
            "output_dir": str(output_dir)
        }

    def analyze_batch(
            self,
            plan_directory: Path,
            output_dir: Optional[Path] = None,
            max_plans: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Analyze a batch of development plans.

        Args:
            plan_directory: Directory containing plan documents
            output_dir: Optional output directory
            max_plans: Optional limit on number of plans to process

        Returns:
            List of analysis results for each plan
        """
        logger.info(f"Starting batch analysis from {plan_directory}")

        # Find all plan documents
        plan_files = list(plan_directory.glob("*.pdf")) + list(plan_directory.glob("*.txt"))

        if max_plans:
            plan_files = plan_files[:max_plans]

        logger.info(f"Found {len(plan_files)} plans to analyze")

        results = []
        for i, plan_path in enumerate(plan_files, 1):
            logger.info(f"\n{'=' * 80}")
            logger.info(f"Processing plan {i}/{len(plan_files)}: {plan_path.name}")
            logger.info(f"{'=' * 80}\n")

            try:
                result = self.analyze_single_plan(
                    plan_path,
                    output_dir=output_dir
                )
                results.append(result)

            except Exception as e:
                logger.exception(f"Failed to process {plan_path.name}: {e}")
                results.append({
                    "plan_name": plan_path.stem,
                    "status": "failed",
                    "error": str(e)
                })

            # Log progress
            logger.info(f"\nBatch progress: {i}/{len(plan_files)} plans completed")

        # Generate batch summary
        self._generate_batch_summary(results, output_dir or CONFIG.output_dir)

        logger.info(f"\nBatch analysis complete!")
        logger.info(f"Total plans processed: {len(results)}")
        logger.info(f"Successful: {sum(1 for r in results if r.get('status') != 'failed')}")

        return results

    def _load_plan(self, plan_path: Path) -> Tuple[str, Dict[str, Any]]:
        """
        Load and preprocess a plan document.

        Uses PyMuPDF for PDF extraction

        Args:
            plan_path: Path to plan document (PDF or TXT)

        Returns:
            Tuple of (plan_text, metadata)
        """
        logger.info(f"Loading plan from {plan_path}")

        plan_text = ""
        metadata = {
            "file_name": plan_path.name,
            "file_size": plan_path.stat().st_size,
            "extraction_date": datetime.now().isoformat(),
            "file_type": plan_path.suffix
        }

        try:
            if plan_path.suffix.lower() == ".pdf":
                # PDF extraction using PyMuPDF (fitz)
                try:
                    import fitz  # PyMuPDF

                    doc = fitz.open(str(plan_path))
                    text_parts = []

                    for page_num, page in enumerate(doc, start=1):
                        page_text = page.get_text()
                        text_parts.append(page_text)

                    plan_text = "\n".join(text_parts)

                    metadata.update({
                        "num_pages": len(doc),
                        "pdf_version": doc.metadata.get("format", "unknown"),
                        "title": doc.metadata.get("title", ""),
                        "author": doc.metadata.get("author", ""),
                        "extraction_method": "PyMuPDF"
                    })

                    doc.close()
                    logger.info(f"Extracted {len(doc)} pages from PDF")

                except ImportError:
                    logger.warning("PyMuPDF not installed, falling back to pypdf")
                    # Fallback to pypdf
                    try:
                        import pypdf

                        reader = pypdf.PdfReader(str(plan_path))
                        text_parts = []

                        for page_num, page in enumerate(reader.pages, start=1):
                            page_text = page.extract_text()
                            text_parts.append(page_text)

                        plan_text = "\n".join(text_parts)

                        metadata.update({
                            "num_pages": len(reader.pages),
                            "extraction_method": "pypdf"
                        })

                        logger.info(f"Extracted {len(reader.pages)} pages from PDF (pypdf)")

                    except ImportError:
                        logger.error("Neither PyMuPDF nor pypdf available. Install with: pip install PyMuPDF")
                        raise RuntimeError("PDF extraction libraries not available. Install PyMuPDF or pypdf.")

            else:
                # Text file - direct read
                with open(plan_path, 'r', encoding='utf-8', errors='ignore') as f:
                    plan_text = f.read()

                metadata.update({
                    "num_lines": plan_text.count('\n') + 1,
                    "extraction_method": "direct_read"
                })

                logger.info(f"Loaded text file with {metadata['num_lines']} lines")

        except Exception as e:
            logger.error(f"Failed to load plan from {plan_path}: {e}")
            raise

        # Basic validation
        if not plan_text or len(plan_text.strip()) < 100:
            logger.warning(f"Extracted text is suspiciously short ({len(plan_text)} chars)")

        metadata["text_length"] = len(plan_text)
        metadata["word_count"] = len(plan_text.split())

        logger.info(f"Loaded plan: {len(plan_text)} characters, {metadata['word_count']} words")
        return plan_text, metadata

    def _process_all_questions(
            self,
            plan_text: str,
            plan_metadata: Dict[str, Any],
            plan_name: str
    ) -> List[MicroLevelAnswer]:
        """
        Process all 300 questions for a plan.

        Strategy:
        - Questions are grouped by component requirements
        - Components run once and results are cached
        - Circuit breaker protects against repeated failures
        """
        logger.info("Processing 300 questions...")

        all_answers = []

        # Get all questions
        all_questions = list(self.router.questions.values())

        # Group questions by their required component set (optimization)
        question_groups = self._group_questions_by_components(all_questions)

        logger.info(f"Optimized into {len(question_groups)} component execution groups")

        for group_id, (components, questions) in enumerate(question_groups.items(), 1):
            logger.info(f"Processing group {group_id}/{len(question_groups)}: "
                        f"{len(questions)} questions using components {components}")

            # Execute components once for this group
            execution_results = self._execute_components_with_fallback(
                list(components),
                plan_text,
                plan_metadata
            )

            # Update component execution count
            self.execution_stats["total_components_executed"] += len(execution_results)

            # Generate answers for all questions in this group
            for question in questions:
                try:
                    answer = self.report_assembler.generate_micro_answer(
                        question,
                        execution_results,
                        plan_text
                    )
                    all_answers.append(answer)

                except Exception as e:
                    logger.error(f"Failed to generate answer for {question.canonical_id}: {e}")
                    # Create degraded answer
                    all_answers.append(self._create_degraded_answer(question))

        logger.info(f"Completed processing {len(all_answers)} questions")

        return all_answers

    def _group_questions_by_components(
            self,
            questions: List[Question]
    ) -> Dict[frozenset, List[Question]]:
        """
        Group questions that require the same set of components.

        This optimization allows us to run each component combination only once
        instead of 300 times.
        """
        groups = defaultdict(list)

        for question in questions:
            # Extract dimension and question number
            parts = question.canonical_id.split("-")
            if len(parts) >= 3:
                dimension = parts[1]
                question_num = parts[2]
                dimension_question = f"{dimension}-{question_num}"

                # Get component mapping for this question
                if hasattr(self.choreographer,
                           'component_mapping') and dimension_question in self.choreographer.component_mapping:
                    component_info = self.choreographer.component_mapping[dimension_question]
                    components = component_info["components"]

                    # Convert to frozenset for grouping
                    component_set = frozenset(f"{module}.{method}" for module, method in components)
                    groups[component_set].append(question)
                else:
                    # Fallback to module-based grouping
                    modules = frozenset(question.required_modules)
                    groups[modules].append(question)

        return groups

    def _execute_components_with_fallback(
            self,
            components: List[str],
            plan_text: str,
            plan_metadata: Dict[str, Any]
    ) -> Dict[str, ExecutionResult]:
        """
        Execute components with circuit breaker protection and fallbacks.

        Uses ModuleAdapterRegistry to execute actual component methods

        Args:
            components: List of component names (module.method) to execute
            plan_text: Full plan text
            plan_metadata: Plan metadata

        Returns:
            Dict mapping component_name to ExecutionResult
        """
        results = {}

        # Group components by module
        module_components = defaultdict(list)
        for component in components:
            if "." in component:
                module_name, method_name = component.split(".", 1)
                module_components[module_name].append(method_name)

        # Import module adapter registry
        from .module_adapters import ModuleAdapterRegistry
        registry = ModuleAdapterRegistry()

        for module_name, method_names in module_components.items():
            # Check circuit breaker
            if not self.circuit_breaker.is_available(module_name):
                logger.warning(f"Circuit breaker OPEN for {module_name}, using fallback")

                # Use fallback for all methods
                fallback_func = create_module_specific_fallback(module_name)
                fallback_result = fallback_func()

                for method_name in method_names:
                    component_key = f"{module_name}.{method_name}"
                    results[component_key] = ExecutionResult(
                        module_name=module_name,
                        component_name=component_key,
                        method_name=method_name,
                        status="completed",
                        output=fallback_result,
                        evidence_extracted={"status": "degraded"}
                    )
                continue

            # Execute with circuit breaker
            try:
                def execute():
                    """Execute module through adapter registry"""
                    module_results = {}

                    for method_name in method_names:
                        # Prepare arguments based on module
                        args = [plan_text]
                        kwargs = {"plan_metadata": plan_metadata}

                        # Execute module through adapter registry
                        module_result = registry.execute_module_method(
                            module_name=module_name,
                            method_name=method_name,
                            args=args,
                            kwargs=kwargs
                        )

                        # Convert ModuleResult to dict format
                        component_key = f"{module_name}.{method_name}"
                        module_results[component_key] = {
                            "module": module_name,
                            "component": component_key,
                            "method": method_name,
                            "class": module_result.class_name,
                            "status": module_result.status,
                            "data": module_result.data,
                            "evidence": module_result.evidence,
                            "confidence": module_result.confidence,
                            "execution_time": module_result.execution_time,
                            "errors": module_result.errors
                        }

                    return module_results

                module_results = self.circuit_breaker.call(
                    module_name,
                    execute,
                    fallback=create_module_specific_fallback(module_name)
                )

                # Convert to ExecutionResult
                for component_key, result in module_results.items():
                    results[component_key] = ExecutionResult(
                        module_name=result.get("module", module_name),
                        component_name=result.get("component", component_key),
                        method_name=result.get("method", "unknown"),
                        status="completed" if result.get("status") == "success" else "failed",
                        output=result.get("data", {}),
                        execution_time=result.get("execution_time", 0.0),
                        evidence_extracted=self._extract_evidence(result),
                        confidence=result.get("confidence", 0.0),
                        error=result.get("errors", [None])[0] if result.get("errors") else None
                    )

                    # Update stats
                    self.execution_stats["component_performance"][component_key]["calls"] += 1
                    if result.get("status") == "success":
                        self.execution_stats["component_performance"][component_key]["successes"] += 1
                    else:
                        self.execution_stats["component_performance"][component_key]["failures"] += 1

                # Update module stats
                self.execution_stats["module_performance"][module_name]["calls"] += 1
                if any(r.get("status") == "success" for r in module_results.values()):
                    self.execution_stats["module_performance"][module_name]["successes"] += 1
                else:
                    self.execution_stats["module_performance"][module_name]["failures"] += 1

            except Exception as e:
                logger.error(f"Module {module_name} failed: {e}")

                self.execution_stats["module_performance"][module_name]["calls"] += 1
                self.execution_stats["module_performance"][module_name]["failures"] += 1

                # Use fallback for all methods
                fallback_func = create_module_specific_fallback(module_name)
                fallback_result = fallback_func()

                for method_name in method_names:
                    component_key = f"{module_name}.{method_name}"
                    results[component_key] = ExecutionResult(
                        module_name=module_name,
                        component_name=component_key,
                        method_name=method_name,
                        status="failed",
                        output=fallback_result,
                        error=str(e)
                    )

                    self.execution_stats["component_performance"][component_key]["calls"] += 1
                    self.execution_stats["component_performance"][component_key]["failures"] += 1

        return results

    def _extract_evidence(self, component_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract structured evidence from component output.

        Maps component-specific output to canonical evidence format

        Args:
            component_output: Dict with keys: module, component, method, status, data, evidence, confidence

        Returns:
            Structured evidence dict with quantitative claims, causal links, contradictions
        """
        evidence = {
            "quantitative_claims": [],
            "causal_links": [],
            "contradictions": [],
            "confidence_scores": {},
            "raw_output": component_output.get("data", {})
        }

        module_name = component_output.get("module", "")
        component_name = component_output.get("component", "")
        method_name = component_output.get("method", "")
        module_data = component_output.get("data", {})
        module_evidence = component_output.get("evidence", [])
        confidence = component_output.get("confidence", 0.0)

        # Extract evidence from ModuleResult.evidence field (standardized across adapters)
        if isinstance(module_evidence, list):
            for ev_item in module_evidence:
                if isinstance(ev_item, dict):
                    # Policy processor evidence (dimensions D1-D6)
                    if "dimension" in ev_item:
                        evidence["quantitative_claims"].append({
                            "dimension": ev_item.get("dimension"),
                            "point_evidence": ev_item.get("point_evidence", []),
                            "bayesian_score": ev_item.get("bayesian_score", 0.0)
                        })

                    # Causal processor evidence
                    if "causal_dimensions" in ev_item:
                        evidence["causal_links"].append(ev_item)

                    # Contradiction detector evidence
                    if "contradictions" in ev_item:
                        evidence["contradictions"].extend(ev_item["contradictions"])

        # Module-specific evidence extraction from data field
        if module_name == "contradiction_detector":
            evidence["contradictions"] = module_data.get("contradictions", [])
            evidence["confidence_scores"]["coherence"] = module_data.get("coherence_metrics", {}).get("coherence_score",
                                                                                                      confidence)

        elif module_name == "causal_processor":
            evidence["causal_links"] = module_data.get("causal_dimensions", {})
            evidence["confidence_scores"]["causal_strength"] = module_data.get("information_gain", confidence)

        elif module_name == "dereck_beach":
            # Derek Beach process tracing evidence
            evidence["causal_links"] = module_data.get("mechanism_parts", [])
            evidence["confidence_scores"]["mechanism_confidence"] = module_data.get("rigor_status", confidence)
            evidence["quantitative_claims"].append({
                "causal_hierarchy": module_data.get("causal_hierarchy", {}),
                "mechanism_inferences": module_data.get("mechanism_inferences", [])
            })

        elif module_name == "policy_processor":
            # IndustrialPolicyProcessor dimension analysis
            dimensions_data = module_data.get("dimensions", {})
            for dim, dim_data in dimensions_data.items():
                evidence["quantitative_claims"].append({
                    "dimension": dim,
                    "point_evidence": dim_data.get("point_evidence", []),
                    "bayesian_score": dim_data.get("bayesian_score", 0.0)
                })
            evidence["confidence_scores"]["overall"] = module_data.get("overall_score", confidence)

        elif module_name == "financial_analyzer":
            evidence["quantitative_claims"] = module_data.get("budget_analysis", {})
            evidence["confidence_scores"]["financial_coherence"] = module_data.get("viability_score", confidence)

        elif module_name == "analyzer_one":
            # MunicipalAnalyzer evidence
            evidence["quantitative_claims"] = module_data.get("analysis_results", {})
            evidence["confidence_scores"]["semantic_quality"] = module_data.get("quality_score", confidence)

        elif module_name == "embedding_policy":
            # PolicyAnalysisEmbedder evidence
            evidence["quantitative_claims"].append({
                "chunks_processed": module_data.get("chunks_processed", 0),
                "embeddings_generated": module_data.get("embeddings_generated", False)
            })
            evidence["confidence_scores"]["embedding_quality"] = confidence

        elif module_name == "policy_segmenter":
            # DocumentSegmenter evidence
            segments = module_data.get("segments", [])
            evidence["quantitative_claims"].append({
                "num_segments": len(segments),
                "avg_segment_length": sum(len(s.get("text", "")) for s in segments) / len(segments) if segments else 0
            })
            evidence["confidence_scores"]["segmentation_quality"] = confidence

        # Add method-specific evidence if available
        if method_name:
            evidence["confidence_scores"][f"{method_name}_confidence"] = confidence

        # Add overall confidence from ModuleResult
        evidence["confidence_scores"]["module_confidence"] = confidence

        return evidence

    def _create_degraded_answer(self, question: Question) -> MicroLevelAnswer:
        """Create a degraded answer when processing fails"""
        return MicroLevelAnswer(
            question_id=question.canonical_id,
            qualitative_note="INSUFICIENTE",
            quantitative_score=0.0,
            evidence=[],
            explanation="Unable to process this question due to system limitations.",
            confidence=0.0,
            metadata={
                "dimension": question.dimension,
                "policy_area": question.policy_area,
                "status": "degraded"
            }
        )

    def _generate_meso_level(
            self,
            micro_answers: List[MicroLevelAnswer]
    ) -> List[MesoLevelCluster]:
        """Generate MESO-level cluster analysis"""
        logger.info("Generating MESO-level cluster analysis")

        meso_clusters = []

        for cluster_name, policy_areas in CONFIG.clusters.items():
            # Filter micro answers for this cluster
            cluster_answers = [
                answer for answer in micro_answers
                if answer.metadata["policy_area"] in policy_areas
            ]

            if cluster_answers:
                cluster = self.report_assembler.generate_meso_cluster(
                    cluster_name,
                    cluster_answers
                )
                meso_clusters.append(cluster)

        logger.info(f"Generated {len(meso_clusters)} MESO-level clusters")
        return meso_clusters

    def _generate_macro_level(
            self,
            micro_answers: List[MicroLevelAnswer],
            meso_clusters: List[MesoLevelCluster]
    ) -> MacroLevelConvergence:
        """Generate MACRO-level convergence analysis"""
        logger.info("Generating MACRO-level convergence analysis")

        macro_convergence = self.report_assembler.generate_macro_convergence(
            micro_answers,
            meso_clusters
        )

        logger.info(f"MACRO analysis complete: Overall score = {macro_convergence.overall_score:.2f}")
        return macro_convergence

    def _generate_batch_summary(
            self,
            results: List[Dict[str, Any]],
            output_dir: Path
    ):
        """Generate summary report for batch processing"""
        logger.info("Generating batch summary")

        summary = {
            "total_plans": len(results),
            "successful": sum(1 for r in results if r.get("status") != "failed"),
            "failed": sum(1 for r in results if r.get("status") == "failed"),
            "avg_execution_time": sum(r.get("execution_time", 0) for r in results) / len(results) if results else 0,
            "plans": []
        }

        for result in results:
            if result.get("status") != "failed":
                macro = result.get("macro_convergence")
                summary["plans"].append({
                    "name": result["plan_name"],
                    "score": macro.overall_score if macro else 0.0,
                    "classification": macro.plan_classification if macro else "UNKNOWN",
                    "execution_time": result.get("execution_time", 0)
                })
            else:
                summary["plans"].append({
                    "name": result["plan_name"],
                    "status": "failed",
                    "error": result.get("error", "Unknown error")
                })

        # Sort by score
        summary["plans"] = sorted(
            summary["plans"],
            key=lambda x: x.get("score", 0),
            reverse=True
        )

        # Save summary
        import json
        summary_path = output_dir / "batch_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"Batch summary saved to {summary_path}")

    def get_system_health(self) -> Dict[str, Any]:
        """Get current system health status"""
        return {
            "circuit_breaker": self.circuit_breaker.get_health_summary(),
            "execution_stats": self.execution_stats,
            "module_metrics": self.circuit_breaker.get_all_metrics()
        }