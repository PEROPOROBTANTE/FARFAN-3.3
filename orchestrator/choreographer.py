"""
Choreographer - Job Queuing and Sequential Execution with Circuit Breaker Integration
=====================================================================================

CORE RESPONSIBILITY: Queue jobs, execute question processing sequentially through
ModuleController, integrate circuit breaker for fault tolerance, and aggregate results

EXECUTION FLOW:
--------------
1. Job Queuing: Accept question specifications and plan text for processing
2. Sequential Execution: Process each question through ModuleController
3. Circuit Breaker Integration: Skip failed modules and log degraded operation
4. Result Aggregation: Compile results for ReportAssembly consumption

CIRCUIT BREAKER INTEGRATION:
----------------------------
- Automatically handled by ModuleController
- Failed modules are skipped with logged warnings
- System continues in degraded mode when modules fail
- Recovery attempts after timeout period

Author: FARFAN Integration Team
Version: 3.0.0 - Refactored for unified data structure
Python: 3.10+
"""

import logging
import time
import uuid
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Job execution status"""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    """Represents a question processing job"""
    job_id: str
    question_spec: Any
    plan_text: str
    status: JobStatus = JobStatus.QUEUED
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    execution_time: float = 0.0


class Choreographer:
    """
    Orchestrates job queuing and sequential question processing
    
    Features:
    - Job queue management
    - Sequential execution through ModuleController
    - Circuit breaker integration (handled by ModuleController)
    - Result aggregation for ReportAssembly
    - Degraded operation logging
    """

    def __init__(self, module_controller: Any):
        """
        Initialize choreographer with module controller
        
        Args:
            module_controller: ModuleController instance for adapter execution
        """
        self.module_controller = module_controller
        
        # Job management
        self.jobs: Dict[str, Job] = {}
        self.job_queue: List[str] = []
        
        # Statistics
        self.stats = {
            "total_jobs_queued": 0,
            "total_jobs_completed": 0,
            "total_jobs_failed": 0,
            "total_execution_time": 0.0,
            "degraded_executions": 0,
            "skipped_modules": 0
        }
        
        logger.info("Choreographer initialized with ModuleController")

    def queue_job(
            self,
            question_spec: Any,
            plan_text: str
    ) -> str:
        """
        Queue a job for question processing
        
        Args:
            question_spec: Question specification from questionnaire parser
            plan_text: Plan document text
            
        Returns:
            Job ID for tracking
        """
        job_id = str(uuid.uuid4())
        
        job = Job(
            job_id=job_id,
            question_spec=question_spec,
            plan_text=plan_text
        )
        
        self.jobs[job_id] = job
        self.job_queue.append(job_id)
        
        self.stats["total_jobs_queued"] += 1
        
        question_id = getattr(question_spec, 'canonical_id', 'unknown')
        logger.info(f"Queued job {job_id} for question {question_id}")
        
        return job_id

    def execute_job(self, job_id: str) -> Any:
        """
        Execute a queued job
        
        Args:
            job_id: Job identifier
            
        Returns:
            UnifiedAnalysisData from ModuleController
            
        Raises:
            ValueError: If job_id not found
        """
        if job_id not in self.jobs:
            raise ValueError(f"Job {job_id} not found")
        
        job = self.jobs[job_id]
        
        if job.status != JobStatus.QUEUED:
            logger.warning(f"Job {job_id} already processed (status: {job.status.value})")
            return job.result
        
        # Update job status
        job.status = JobStatus.RUNNING
        job.started_at = datetime.now().isoformat()
        
        question_id = getattr(job.question_spec, 'canonical_id', 'unknown')
        logger.info(f"Executing job {job_id} for question {question_id}")
        
        start_time = time.time()
        
        try:
            # Execute through ModuleController
            unified_data = self.module_controller.execute_question_analysis(
                question_spec=job.question_spec,
                plan_text=job.plan_text
            )
            
            # Check for degraded execution (failed or skipped modules)
            self._check_degraded_execution(unified_data, question_id)
            
            # Update job with result
            job.status = JobStatus.COMPLETED
            job.result = unified_data
            job.completed_at = datetime.now().isoformat()
            job.execution_time = time.time() - start_time
            
            # Update statistics
            self.stats["total_jobs_completed"] += 1
            self.stats["total_execution_time"] += job.execution_time
            
            logger.info(
                f"Completed job {job_id} for {question_id} in {job.execution_time:.2f}s "
                f"({unified_data.execution_metadata['successful_steps']}/{unified_data.execution_metadata['total_steps']} steps succeeded)"
            )
            
            return unified_data
            
        except Exception as e:
            logger.error(f"Error executing job {job_id} for {question_id}: {e}", exc_info=True)
            
            job.status = JobStatus.FAILED
            job.error = str(e)
            job.completed_at = datetime.now().isoformat()
            job.execution_time = time.time() - start_time
            
            self.stats["total_jobs_failed"] += 1
            
            raise

    def _check_degraded_execution(
            self,
            unified_data: Any,
            question_id: str
    ) -> None:
        """
        Check for degraded execution and log appropriately
        
        Args:
            unified_data: UnifiedAnalysisData from execution
            question_id: Question identifier for logging
        """
        metadata = unified_data.execution_metadata
        
        failed_steps = metadata.get('failed_steps', 0)
        skipped_steps = metadata.get('skipped_steps', 0)
        total_steps = metadata.get('total_steps', 0)
        
        if failed_steps > 0 or skipped_steps > 0:
            self.stats["degraded_executions"] += 1
            self.stats["skipped_modules"] += skipped_steps
            
            logger.warning(
                f"Degraded execution for {question_id}: "
                f"{failed_steps} failed, {skipped_steps} skipped out of {total_steps} total steps"
            )
            
            # Log failed modules
            for trace in unified_data.module_traces:
                if trace.status == 'failed':
                    logger.warning(
                        f"  - Module {trace.module_name}.{trace.method_name} failed: {trace.error}"
                    )
                elif trace.status == 'skipped':
                    logger.warning(
                        f"  - Module {trace.module_name}.{trace.method_name} skipped: {trace.error}"
                    )

    def execute_all_queued(self) -> List[Any]:
        """
        Execute all queued jobs sequentially
        
        Returns:
            List of UnifiedAnalysisData results
        """
        logger.info(f"Executing {len(self.job_queue)} queued jobs")
        
        results = []
        
        for job_id in list(self.job_queue):
            try:
                result = self.execute_job(job_id)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to execute job {job_id}: {e}")
                continue
        
        # Clear queue
        self.job_queue = []
        
        logger.info(f"Completed execution of {len(results)} jobs")
        
        return results

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get status of a specific job
        
        Args:
            job_id: Job identifier
            
        Returns:
            Job status dictionary
        """
        if job_id not in self.jobs:
            return {"error": f"Job {job_id} not found"}
        
        job = self.jobs[job_id]
        
        return {
            "job_id": job.job_id,
            "question_id": getattr(job.question_spec, 'canonical_id', 'unknown'),
            "status": job.status.value,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "completed_at": job.completed_at,
            "execution_time": job.execution_time,
            "error": job.error
        }

    def get_queue_status(self) -> Dict[str, Any]:
        """
        Get overall queue status
        
        Returns:
            Queue status dictionary
        """
        return {
            "queued_jobs": len([j for j in self.jobs.values() if j.status == JobStatus.QUEUED]),
            "running_jobs": len([j for j in self.jobs.values() if j.status == JobStatus.RUNNING]),
            "completed_jobs": len([j for j in self.jobs.values() if j.status == JobStatus.COMPLETED]),
            "failed_jobs": len([j for j in self.jobs.values() if j.status == JobStatus.FAILED]),
            "total_jobs": len(self.jobs)
        }

    def get_stats(self) -> Dict[str, Any]:
        """
        Get choreographer statistics
        
        Returns:
            Statistics dictionary
        """
        stats = dict(self.stats)
        
        # Add module controller stats
        stats["module_controller_stats"] = self.module_controller.get_execution_stats()
        
        return stats

    def clear_completed_jobs(self) -> int:
        """
        Clear completed jobs from memory
        
        Returns:
            Number of jobs cleared
        """
        completed_job_ids = [
            job_id for job_id, job in self.jobs.items()
            if job.status == JobStatus.COMPLETED
        ]
        
        for job_id in completed_job_ids:
            del self.jobs[job_id]
        
        logger.info(f"Cleared {len(completed_job_ids)} completed jobs")
        
        return len(completed_job_ids)

    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a queued job
        
        Args:
            job_id: Job identifier
            
        Returns:
            True if job was cancelled, False otherwise
        """
        if job_id not in self.jobs:
            return False
        
        job = self.jobs[job_id]
        
        if job.status != JobStatus.QUEUED:
            logger.warning(f"Cannot cancel job {job_id} with status {job.status.value}")
            return False
        
        job.status = JobStatus.CANCELLED
        
        if job_id in self.job_queue:
            self.job_queue.remove(job_id)
        
        logger.info(f"Cancelled job {job_id}")
        
        return True

    def aggregate_results(
            self,
            unified_data_list: List[Any]
    ) -> Dict[str, Any]:
        """
        Aggregate results from multiple executions for reporting
        
        Args:
            unified_data_list: List of UnifiedAnalysisData objects
            
        Returns:
            Aggregated results dictionary
        """
        if not unified_data_list:
            return {}
        
        aggregated = {
            "total_questions": len(unified_data_list),
            "total_execution_time": sum(d.total_execution_time for d in unified_data_list),
            "avg_execution_time": sum(d.total_execution_time for d in unified_data_list) / len(unified_data_list),
            "total_modules_executed": sum(len(d.module_traces) for d in unified_data_list),
            "successful_executions": sum(
                1 for d in unified_data_list
                if d.execution_metadata.get('successful_steps', 0) == d.execution_metadata.get('total_steps', 1)
            ),
            "degraded_executions": sum(
                1 for d in unified_data_list
                if d.execution_metadata.get('failed_steps', 0) > 0 or d.execution_metadata.get('skipped_steps', 0) > 0
            ),
            "avg_confidence": sum(
                d.execution_metadata.get('avg_confidence', 0.0) for d in unified_data_list
            ) / len(unified_data_list),
            "modules_used": list(set(
                trace.module_name
                for d in unified_data_list
                for trace in d.module_traces
            )),
            "failure_summary": self._aggregate_failures(unified_data_list)
        }
        
        return aggregated

    def _aggregate_failures(self, unified_data_list: List[Any]) -> Dict[str, int]:
        """
        Aggregate failure counts by module
        
        Args:
            unified_data_list: List of UnifiedAnalysisData objects
            
        Returns:
            Dictionary mapping module names to failure counts
        """
        failure_counts: Dict[str, int] = {}
        
        for data in unified_data_list:
            for trace in data.module_traces:
                if trace.status in ['failed', 'skipped']:
                    module_name = trace.module_name
                    failure_counts[module_name] = failure_counts.get(module_name, 0) + 1
        
        return failure_counts


# ============================================================================
# BACKWARD COMPATIBILITY - ExecutionChoreographer
# ============================================================================

class ExecutionChoreographer:
    """
    DEPRECATED: Legacy interface for backward compatibility
    
    Use Choreographer class instead
    """

    def __init__(self, max_workers: Optional[int] = None):
        logger.warning(
            "ExecutionChoreographer is deprecated. "
            "Use Choreographer class with ModuleController instead."
        )
        self.max_workers = max_workers or 4

    def execute_question_chain(
            self,
            question_spec: Any,
            plan_text: str,
            module_adapter_registry: Any,
            circuit_breaker: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        DEPRECATED: Legacy method for backward compatibility
        
        This method exists only to prevent breaking existing code.
        New code should use Choreographer.queue_job() and Choreographer.execute_job()
        """
        logger.warning(
            "execute_question_chain is deprecated. "
            "Use Choreographer.queue_job() and Choreographer.execute_job() instead."
        )
        
        # Create temporary ModuleController and Choreographer
        from .module_controller import ModuleController
        
        module_controller = ModuleController(
            module_adapter_registry=module_adapter_registry,
            circuit_breaker=circuit_breaker
        )
        
        choreographer = Choreographer(module_controller=module_controller)
        
        # Queue and execute
        job_id = choreographer.queue_job(question_spec, plan_text)
        unified_data = choreographer.execute_job(job_id)
        
        # Convert to legacy format
        legacy_results = {}
        for trace in unified_data.module_traces:
            key = f"{trace.module_name}.{trace.method_name}"
            legacy_results[key] = {
                "module_name": trace.module_name,
                "status": trace.status,
                "output": trace.result_data,
                "execution_time": trace.execution_time,
                "evidence_extracted": {"evidence": trace.evidence},
                "confidence": trace.confidence
            }
        
        return legacy_results


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 80)
    print("CHOREOGRAPHER - Job Queuing and Sequential Execution")
    print("=" * 80)
    print("\nThis module provides:")
    print("  - Job queue management")
    print("  - Sequential question processing through ModuleController")
    print("  - Circuit breaker integration for fault tolerance")
    print("  - Result aggregation for reporting")
    print("  - Degraded operation logging")
    print("=" * 80)
