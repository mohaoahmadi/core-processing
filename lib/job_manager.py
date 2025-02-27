from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Callable, Optional, List
from enum import Enum
from datetime import datetime
import asyncio
from loguru import logger

class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class Job:
    def __init__(self, job_id: str, processor: Callable, params: Dict[str, Any]):
        self.job_id = job_id
        self.processor = processor
        self.params = params
        self.status = JobStatus.PENDING
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

class JobManager:
    _instance = None
    _executor: Optional[ThreadPoolExecutor] = None
    _jobs: Dict[str, Job] = {}

    @classmethod
    def initialize(cls, max_workers: int = 4) -> None:
        """Initialize the job manager with a thread pool"""
        if cls._instance is None:
            cls._instance = cls()
            cls._executor = ThreadPoolExecutor(max_workers=max_workers)
            logger.info(f"JobManager initialized with {max_workers} workers")

    @classmethod
    async def shutdown(cls) -> None:
        """Shutdown the job manager and wait for all jobs to complete"""
        if cls._executor:
            cls._executor.shutdown(wait=True)
            cls._executor = None
            logger.info("JobManager shutdown completed")

    @classmethod
    async def submit_job(
        cls,
        job_id: str,
        processor: Callable,
        params: Dict[str, Any]
    ) -> str:
        """Submit a new processing job"""
        if cls._executor is None:
            raise RuntimeError("JobManager not initialized")

        job = Job(job_id, processor, params)
        cls._jobs[job_id] = job

        async def run_job():
            try:
                job.status = JobStatus.RUNNING
                job.start_time = datetime.utcnow()
                
                # Execute the processor
                job.result = await processor(**params)
                
                job.status = JobStatus.COMPLETED
            except Exception as e:
                job.status = JobStatus.FAILED
                job.error = str(e)
                logger.error(f"Job {job_id} failed: {e}")
            finally:
                job.end_time = datetime.utcnow()

        asyncio.create_task(run_job())
        return job_id

    @classmethod
    def get_job_status(cls, job_id: str) -> Dict[str, Any]:
        """Get the status of a specific job"""
        if job_id not in cls._jobs:
            raise KeyError(f"Job {job_id} not found")
        
        job = cls._jobs[job_id]
        return {
            "job_id": job.job_id,
            "status": job.status.value,
            "result": job.result,
            "error": job.error,
            "start_time": job.start_time.isoformat() if job.start_time else None,
            "end_time": job.end_time.isoformat() if job.end_time else None
        }

    @classmethod
    def list_jobs(cls) -> List[Dict[str, Any]]:
        """List all jobs and their statuses"""
        return [cls.get_job_status(job_id) for job_id in cls._jobs]