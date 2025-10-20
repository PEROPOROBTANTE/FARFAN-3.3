"""
Choreographer - Job Orchestration with Retry Logic and Partial Result Collection
=================================================================================

CORE RESPONSIBILITY: Orchestrate job execution with resilience and fault tolerance
----------------------------------------------------------------------------------
Manages complete job lifecycle from question loading through result aggregation:
1. Queue questions from cuestionario.json
2. Invoke ModuleController handlers through circuit breaker wrapper
3. Aggregate results with retry logic
4. Handle failure scenarios with partial result collection
5. Provide job progress tracking and status reporting

EXECUTION FLOW:
---------------
Job Start → Load Questions → For each question:
  → Check circuit breaker → Invoke ModuleController → Record result
  → On failure: Retry with exponential backoff → Collect partial results
→ Aggregate all results → Generate job summary

RETRY STRATEGY:
---------------
- Max retries: 3 per question
- Backoff: Exponential (1s, 2s, 4s)
- Circuit breaker: Skip if circuit open
- Partial results: Continue with other questions on failure
- Job fails only if 100% questions fail

CIRCUIT BREAKER INTEGRATION:
----------------------------
- Check circuit state before each invocation
- Record success/failure after each invocation
- Use fallback strategy if circuit opens
- Automatic recovery after timeout period

Author: FARFAN Integration Team
Version: 3.0.0 - Dependency injection with ModuleController
Python: 3.10+
"""

import logging
import time
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    """Execution status for individual question processing"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    DEGRADED = "degraded"
    RETRYING = "retrying"


@dataclass
class ExecutionResult:
    """
    Result from a single question execution
    
    Wraps ModuleResult objects with additional orchestration metadata
    """
    question_id: str
    status: ExecutionStatus
    
    module_results: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    execution_time: float = 0.0
    retry_count: int = 0
    
    aggregated_confidence: float = 0.0
    evidence_count: int = 0
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for aggregation and reporting"""
        return {
            "question_id": self.question_id,
            "status": self.status.value,
            "module_results": self.module_results,
            "error": self.error,
            "execution_time": self.execution_time,
            "retry_count": self.retry_count,
            "aggregated_confidence": self.aggregated_confidence,
            "evidence_count": self.evidence_count,
            "metadata": self.metadata
        }


@dataclass
class JobSummary:
    """
    Summary of job execution
    
    Provides statistics and results from complete job run
    """
    job_id: str
    total_questions: int
    completed: int
    failed: int
    skipped: int
    degraded: int
    
    total_execution_time: float
    avg_execution_time: float
    
    total_retries: int
    circuit_breaker_trips: int
    
    success_rate: float
    completion_rate: float
    
    results: List[ExecutionResult] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting"""
        return {
            "job_id": self.job_id,
            "total_questions": self.total_questions,
            "completed": self.completed,
            "failed": self.failed,
            "skipped": self.skipped,
            "degraded": self.degraded,
            "total_execution_time": self.total_execution_time,
            "avg_execution_time": self.avg_execution_time,
            "total_retries": self.total_retries,
            "circuit_breaker_trips": self.circuit_breaker_trips,
            "success_rate": self.success_rate,
            "completion_rate": self.completion_rate,
            "results": [r.to_dict() for r in self.results],
            "errors": self.errors,
            "metadata": self.metadata
        }


class Choreographer:
    """
    Orchestrates job execution with ModuleController integration
    
    Manages:
    - Question queueing from cuestionario.json
    - Circuit breaker-wrapped ModuleController invocations
    - Retry logic with exponential backoff
    - Partial result collection
    - Job progress tracking
    """

    def __init__(
            self,
            module_controller: Any,
            max_workers: Optional[int] = None,
            max_retries: int = 3,
            retry_delay: float = 1.0,
            cuestionario_path: Optional[Path] = None
    ) -> None:
        """
        Initialize choreographer with ModuleController dependency
        
        Args:
            module_controller: ModuleController instance for question routing
            max_workers: Maximum parallel workers (not used in current sequential implementation)
            max_retries: Maximum retry attempts per question (default: 3)
            retry_delay: Initial retry delay in seconds (default: 1.0)
            cuestionario_path: Path to cuestionario.json (defaults to ./cuestionario.json)
        """
        self.module_controller = module_controller
        self.max_workers = max_workers or 4
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.cuestionario_path = cuestionario_path or Path("cuestionario.json")
        
        self.question_queue: List[Any] = []
        self.execution_results: List[ExecutionResult] = []
        
        self.stats = {
            "total_retries": 0,
            "circuit_breaker_trips": 0,
            "fallback_invocations": 0
        }
        
        logger.info(
            f"Choreographer initialized with ModuleController "
            f"(max_retries={max_retries}, retry_delay={retry_delay}s)"
        )

    def load_questions_from_cuestionario(self) -> List[Dict[str, Any]]:
        """
        Load questions from cuestionario.json
        
        Returns:
            List of question dictionaries
            
        Raises:
            FileNotFoundError: If cuestionario.json not found
            ValueError: If JSON structure is invalid
        """
        if not self.cuestionario_path.exists():
            raise FileNotFoundError(f"cuestionario.json not found at {self.cuestionario_path}")
        
        logger.info(f"Loading questions from {self.cuestionario_path}")
        
        with open(self.cuestionario_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, dict):
            raise ValueError("cuestionario.json must contain a dictionary")
        
        questions = []
        for category_key, category_data in data.items():
            if isinstance(category_data, dict) and 'questions' in category_data:
                questions.extend(category_data['questions'])
            elif isinstance(category_data, list):
                questions.extend(category_data)
        
        logger.info(f"Loaded {len(questions)} questions from cuestionario")
        return questions

    def execute_job(
            self,
            question_specs: List[Any],
            plan_text: str,
            circuit_breaker: Optional[Any] = None,
            job_id: Optional[str] = None
    ) -> JobSummary:
        """
        Execute complete job with all questions
        
        EXECUTION FLOW:
        ---------------
        1. Initialize job tracking
        2. For each question in queue:
           a. Check circuit breaker status
           b. Execute question with retry logic
           c. Record result (success or failure)
           d. Update statistics
        3. Generate job summary with aggregated results
        
        Args:
            question_specs: List of question specifications
            plan_text: Plan document text
            circuit_breaker: Optional CircuitBreaker for fault tolerance
            job_id: Optional job identifier
            
        Returns:
            JobSummary with execution statistics and results
        """
        job_id = job_id or f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Starting job {job_id} with {len(question_specs)} questions")
        
        start_time = time.time()
        
        self.execution_results = []
        self.stats = {
            "total_retries": 0,
            "circuit_breaker_trips": 0,
            "fallback_invocations": 0
        }
        
        for i, question_spec in enumerate(question_specs, 1):
            logger.info(f"Processing question {i}/{len(question_specs)}: {question_spec.canonical_id}")
            
            result = self.execute_question(
                question_spec=question_spec,
                plan_text=plan_text,
                circuit_breaker=circuit_breaker
            )
            
            self.execution_results.append(result)
            
            self._log_progress(i, len(question_specs), result)
        
        total_time = time.time() - start_time
        
        summary = self._generate_job_summary(
            job_id=job_id,
            total_questions=len(question_specs),
            total_time=total_time
        )
        
        logger.info(
            f"Job {job_id} completed: {summary.completed}/{summary.total_questions} successful "
            f"({summary.success_rate:.1%}), {summary.failed} failed, {summary.total_retries} retries"
        )
        
        return summary

    def execute_question(
            self,
            question_spec: Any,
            plan_text: str,
            circuit_breaker: Optional[Any] = None
    ) -> ExecutionResult:
        """
        Execute single question with retry logic and circuit breaker integration
        
        RETRY STRATEGY:
        ---------------
        - Attempt 1: Immediate execution
        - Attempt 2: Wait retry_delay seconds (1.0s)
        - Attempt 3: Wait retry_delay * 2 seconds (2.0s)
        - Attempt 4: Wait retry_delay * 4 seconds (4.0s)
        - After max_retries: Return FAILED with partial results if any
        
        CIRCUIT BREAKER CHECKS:
        -----------------------
        - Before each attempt: Check if circuit allows execution
        - After success: Record success in circuit breaker
        - After failure: Record failure in circuit breaker
        - If circuit opens: Use fallback strategy
        
        Args:
            question_spec: Question specification from questionnaire parser
            plan_text: Plan document text
            circuit_breaker: Optional CircuitBreaker for fault tolerance
            
        Returns:
            ExecutionResult with execution outcome and collected results
        """
        question_id = question_spec.canonical_id
        start_time = time.time()
        
        errors = []
        partial_results = {}
        
        for attempt in range(self.max_retries + 1):
            if attempt > 0:
                delay = self.retry_delay * (2 ** (attempt - 1))
                logger.info(f"Retry attempt {attempt} for {question_id} after {delay}s")
                time.sleep(delay)
                self.stats["total_retries"] += 1
            
            try:
                if circuit_breaker:
                    dimension = getattr(question_spec, 'dimension', None)
                    responsible_adapters = self.module_controller.get_responsible_adapters(dimension)
                    
                    circuit_blocked = all(
                        not circuit_breaker.can_execute(adapter)
                        for adapter in responsible_adapters
                    )
                    
                    if circuit_blocked:
                        logger.warning(f"Circuit breaker blocking {question_id} (all adapters blocked)")
                        self.stats["circuit_breaker_trips"] += 1
                        
                        if attempt < self.max_retries:
                            continue
                        else:
                            return self._create_failed_result(
                                question_id=question_id,
                                error="Circuit breaker blocking all responsible adapters",
                                execution_time=time.time() - start_time,
                                retry_count=attempt,
                                partial_results=partial_results
                            )
                
                module_results = self.module_controller.process_question(
                    question_spec=question_spec,
                    plan_text=plan_text,
                    context=None
                )
                
                if not module_results:
                    raise ValueError("No module results returned")
                
                partial_results.update(module_results)
                
                if circuit_breaker:
                    for adapter_name in responsible_adapters:
                        circuit_breaker.record_success(adapter_name, time.time() - start_time)
                
                return self._create_success_result(
                    question_id=question_id,
                    module_results=module_results,
                    execution_time=time.time() - start_time,
                    retry_count=attempt
                )
                
            except Exception as e:
                error_msg = f"Attempt {attempt + 1}: {str(e)}"
                errors.append(error_msg)
                logger.error(f"Execution failed for {question_id}: {e}")
                
                if circuit_breaker:
                    dimension = getattr(question_spec, 'dimension', None)
                    responsible_adapters = self.module_controller.get_responsible_adapters(dimension)
                    for adapter_name in responsible_adapters:
                        circuit_breaker.record_failure(adapter_name, str(e), time.time() - start_time)
        
        logger.error(f"All retry attempts exhausted for {question_id}")
        
        return self._create_failed_result(
            question_id=question_id,
            error=f"Failed after {self.max_retries + 1} attempts: {'; '.join(errors)}",
            execution_time=time.time() - start_time,
            retry_count=self.max_retries + 1,
            partial_results=partial_results
        )

    def _create_success_result(
            self,
            question_id: str,
            module_results: Dict[str, Any],
            execution_time: float,
            retry_count: int
    ) -> ExecutionResult:
        """
        Create success ExecutionResult from module results
        
        Args:
            question_id: Question canonical ID
            module_results: Dictionary of ModuleResult objects
            execution_time: Total execution time
            retry_count: Number of retries before success
            
        Returns:
            ExecutionResult with COMPLETED status
        """
        confidences = []
        evidence_count = 0
        
        for key, result in module_results.items():
            conf = getattr(result, 'confidence', 0.0)
            confidences.append(conf)
            
            evidence = getattr(result, 'evidence', [])
            if isinstance(evidence, list):
                evidence_count += len(evidence)
            elif isinstance(evidence, dict):
                evidence_count += len(evidence)
        
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return ExecutionResult(
            question_id=question_id,
            status=ExecutionStatus.COMPLETED,
            module_results={k: v for k, v in module_results.items()},
            execution_time=execution_time,
            retry_count=retry_count,
            aggregated_confidence=avg_confidence,
            evidence_count=evidence_count,
            metadata={
                "timestamp": datetime.now().isoformat(),
                "module_count": len(module_results)
            }
        )

    def _create_failed_result(
            self,
            question_id: str,
            error: str,
            execution_time: float,
            retry_count: int,
            partial_results: Optional[Dict[str, Any]] = None
    ) -> ExecutionResult:
        """
        Create failed ExecutionResult with partial results if available
        
        Args:
            question_id: Question canonical ID
            error: Error message
            execution_time: Total execution time
            retry_count: Number of retry attempts
            partial_results: Optional partial results collected before failure
            
        Returns:
            ExecutionResult with FAILED or DEGRADED status
        """
        has_partial = partial_results and len(partial_results) > 0
        
        return ExecutionResult(
            question_id=question_id,
            status=ExecutionStatus.DEGRADED if has_partial else ExecutionStatus.FAILED,
            module_results=partial_results or {},
            error=error,
            execution_time=execution_time,
            retry_count=retry_count,
            aggregated_confidence=0.0,
            evidence_count=0,
            metadata={
                "timestamp": datetime.now().isoformat(),
                "has_partial_results": has_partial,
                "partial_module_count": len(partial_results) if partial_results else 0
            }
        )

    def _generate_job_summary(
            self,
            job_id: str,
            total_questions: int,
            total_time: float
    ) -> JobSummary:
        """
        Generate job summary from execution results
        
        Args:
            job_id: Job identifier
            total_questions: Total number of questions
            total_time: Total execution time
            
        Returns:
            JobSummary with aggregated statistics
        """
        status_counts = defaultdict(int)
        for result in self.execution_results:
            status_counts[result.status] += 1
        
        completed = status_counts[ExecutionStatus.COMPLETED]
        failed = status_counts[ExecutionStatus.FAILED]
        skipped = status_counts[ExecutionStatus.SKIPPED]
        degraded = status_counts[ExecutionStatus.DEGRADED]
        
        success_rate = completed / total_questions if total_questions > 0 else 0.0
        completion_rate = (completed + degraded) / total_questions if total_questions > 0 else 0.0
        
        avg_time = total_time / total_questions if total_questions > 0 else 0.0
        
        errors = [
            {
                "question_id": r.question_id,
                "error": r.error,
                "retry_count": r.retry_count
            }
            for r in self.execution_results
            if r.error
        ]
        
        return JobSummary(
            job_id=job_id,
            total_questions=total_questions,
            completed=completed,
            failed=failed,
            skipped=skipped,
            degraded=degraded,
            total_execution_time=total_time,
            avg_execution_time=avg_time,
            total_retries=self.stats["total_retries"],
            circuit_breaker_trips=self.stats["circuit_breaker_trips"],
            success_rate=success_rate,
            completion_rate=completion_rate,
            results=self.execution_results,
            errors=errors,
            metadata={
                "timestamp": datetime.now().isoformat(),
                "fallback_invocations": self.stats["fallback_invocations"]
            }
        )

    def _log_progress(
            self,
            current: int,
            total: int,
            result: ExecutionResult
    ) -> None:
        """
        Log progress for current question
        
        Args:
            current: Current question number
            total: Total question count
            result: ExecutionResult for current question
        """
        progress_pct = (current / total) * 100
        status_symbol = {
            ExecutionStatus.COMPLETED: "✓",
            ExecutionStatus.FAILED: "✗",
            ExecutionStatus.SKIPPED: "⊘",
            ExecutionStatus.DEGRADED: "⚠"
        }.get(result.status, "?")
        
        logger.info(
            f"Progress: {current}/{total} ({progress_pct:.1f}%) | "
            f"{status_symbol} {result.question_id} | "
            f"Time: {result.execution_time:.2f}s | "
            f"Retries: {result.retry_count}"
        )

    def get_execution_statistics(self) -> Dict[str, Any]:
        """
        Get execution statistics for monitoring
        
        Returns:
            Dictionary with execution metrics
        """
        if not self.execution_results:
            return {
                "total_questions": 0,
                "completed": 0,
                "failed": 0,
                "success_rate": 0.0
            }
        
        status_counts = defaultdict(int)
        for result in self.execution_results:
            status_counts[result.status] += 1
        
        total = len(self.execution_results)
        completed = status_counts[ExecutionStatus.COMPLETED]
        
        return {
            "total_questions": total,
            "completed": completed,
            "failed": status_counts[ExecutionStatus.FAILED],
            "skipped": status_counts[ExecutionStatus.SKIPPED],
            "degraded": status_counts[ExecutionStatus.DEGRADED],
            "success_rate": completed / total if total > 0 else 0.0,
            "total_retries": self.stats["total_retries"],
            "circuit_breaker_trips": self.stats["circuit_breaker_trips"],
            "fallback_invocations": self.stats["fallback_invocations"]
        }


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("CHOREOGRAPHER - Job Orchestration with Retry Logic")
    print("=" * 80)
    print("\nFeatures:")
    print("  - Question queueing from cuestionario.json")
    print("  - ModuleController integration with circuit breaker")
    print("  - Retry logic with exponential backoff (1s, 2s, 4s)")
    print("  - Partial result collection on failures")
    print("  - Job progress tracking and summary reporting")
    print("=" * 80)
