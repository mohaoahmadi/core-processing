"""Job management and processing coordination module.

This module provides a thread-based job management system for handling
long-running processing tasks asynchronously. It maintains job state
and provides status tracking capabilities.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Callable, Optional, List
from enum import Enum
from datetime import datetime
import asyncio
from loguru import logger
from dataclasses import dataclass, field
import httpx
from config import Settings as settings
import json

@dataclass
class ProcessingResult:
    """Model for processing result data."""
    status: str
    message: str
    output_path: Optional[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Ensure metadata is JSON serializable."""
        # If metadata contains non-serializable objects, convert them to strings
        if self.metadata:
            for key, value in list(self.metadata.items()):
                try:
                    json.dumps({key: value})
                except (TypeError, OverflowError):
                    self.metadata[key] = str(value)

class JobStatus(Enum):
    """Enumeration of possible job states.
    
    Attributes:
        PENDING: Job has been created but not yet started
        RUNNING: Job is currently being processed
        COMPLETED: Job has finished successfully
        FAILED: Job encountered an error during processing
    """
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class Job:
    """Represents a processing job.
    
    This class encapsulates all information about a specific processing job,
    including its current status, results, and timing information.
    
    Attributes:
        job_id (str): Unique identifier for the job
        processor (Callable): Function that performs the actual processing
        params (Dict[str, Any]): Parameters to pass to the processor
        status (JobStatus): Current status of the job
        result (Optional[ProcessingResult]): Results from the processor if completed
        error (Optional[str]): Error message if job failed
        start_time (Optional[datetime]): When the job started processing
        end_time (Optional[datetime]): When the job finished (success or failure)
    """
    def __init__(self, job_id: str, processor: Callable, params: Dict[str, Any]):
        self.job_id = job_id
        self.processor = processor
        self.params = params
        self.status = JobStatus.PENDING
        self.result: Optional[ProcessingResult] = None
        self.error: Optional[str] = None
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

class JobManager:
    """Manages processing jobs and their lifecycle.
    
    This class provides a singleton interface for managing processing jobs,
    including job submission, status tracking, and result retrieval.
    It uses a thread pool to execute jobs concurrently.
    
    Class Attributes:
        _instance: Singleton instance of the manager
        _executor: ThreadPoolExecutor for running jobs
        _jobs: Dictionary mapping job IDs to Job instances
    """
    _instance = None
    _executor: Optional[ThreadPoolExecutor] = None
    _jobs: Dict[str, Job] = {}

    @classmethod
    def initialize(cls, max_workers: int = 4) -> None:
        """Initialize the job manager.
        
        Args:
            max_workers (int, optional): Maximum number of concurrent jobs.
                Defaults to 4.
                
        Note:
            This method should be called during application startup.
            It creates the thread pool and prepares the manager for job processing.
        """
        if cls._instance is None:
            cls._instance = cls()
            cls._executor = ThreadPoolExecutor(max_workers=max_workers)
            logger.info(f"JobManager initialized with {max_workers} workers")

    @classmethod
    async def shutdown(cls) -> None:
        """Shutdown the job manager.
        
        Waits for all running jobs to complete and shuts down the thread pool.
        This method should be called during application shutdown.
        """
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
        """Submit a new job for processing.
        
        Args:
            job_id (str): Unique identifier for the job
            processor (Callable): Function that performs the processing
            params (Dict[str, Any]): Parameters to pass to the processor
            
        Returns:
            str: The job ID for tracking the job's status
            
        Raises:
            RuntimeError: If the job manager hasn't been initialized
            
        Note:
            The job is queued for processing and will be executed when
            a worker thread becomes available.
        """
        if cls._executor is None:
            logger.error("JobManager not initialized")
            raise RuntimeError("JobManager not initialized")

        logger.info(f"Submitting new job {job_id}")
        job = Job(job_id, processor, params)
        cls._jobs[job_id] = job

        async def run_job():
            try:
                logger.info(f"Starting job {job_id}")
                job.status = JobStatus.RUNNING
                job.start_time = datetime.utcnow()
                
                logger.info(f"Processing job {job_id} with parameters: {params}")
                result = await processor(**params)
                
                # Check if the result indicates an error
                if isinstance(result, dict) and result.get('status') == 'error':
                    raise Exception(result.get('message', 'Unknown error occurred'))
                elif isinstance(result, ProcessingResult) and result.status == 'error':
                    raise Exception(result.message)
                
                # If no error, format successful result
                if isinstance(result, dict):
                    job.result = ProcessingResult(
                        status="success",
                        message="Processing completed successfully",
                        output_path=result.get('output_path'),
                        metadata=result.get('metadata', {})
                    )
                elif isinstance(result, ProcessingResult):
                    job.result = result
                else:
                    job.result = ProcessingResult(
                        status="success",
                        message="Processing completed successfully",
                        output_path=None,
                        metadata={'raw_result': result}
                    )
                
                job.status = JobStatus.COMPLETED
                logger.info(f"Job {job_id} completed successfully")
                
                # Update the job result in the database first
                try:
                    from lib.supabase_client import get_supabase
                    supabase = get_supabase()
                    
                    # Format the result for database storage
                    result_data = {}
                    if isinstance(job.result, ProcessingResult):
                        # Ensure metadata is JSON serializable
                        metadata = {}
                        try:
                            # Test if metadata is JSON serializable
                            if job.result.metadata:
                                json.dumps(job.result.metadata)
                                metadata = job.result.metadata
                        except TypeError:
                            # If not serializable, convert to string representation
                            logger.warning(f"Metadata is not JSON serializable, converting to string")
                            metadata = str(job.result.metadata)
                        
                        result_data = {
                            "status": job.result.status,
                            "message": job.result.message,
                            "output_path": job.result.output_path,
                            "metadata": metadata
                        }
                    
                    # Update the job record with the result
                    update_result = supabase.table("processing_jobs").update({
                        "status": "completed",
                        "result": result_data,
                        "completed_at": datetime.utcnow().isoformat()
                    }).eq("id", job_id).execute()
                    
                    if update_result.data:
                        logger.info(f"Successfully updated job result in database for job {job_id}")
                    else:
                        logger.warning(f"Failed to update job result in database for job {job_id}")
                except Exception as e:
                    logger.error(f"Error updating job result in database: {str(e)}")
                
                # Call the complete_job endpoint to update the database
                try:
                    async with httpx.AsyncClient() as client:
                        complete_url = f"http://localhost:8000/api/v1/jobs/{job_id}/complete"
                        logger.info(f"Calling complete_job endpoint: {complete_url}")
                        response = await client.post(complete_url)
                        if response.status_code == 200:
                            logger.info(f"Successfully called complete_job endpoint for job {job_id}")
                        else:
                            logger.error(f"Failed to call complete_job endpoint for job {job_id}: {response.status_code} {response.text}")
                except Exception as e:
                    logger.error(f"Error calling complete_job endpoint: {str(e)}")
                
            except Exception as e:
                job.status = JobStatus.FAILED
                job.error = str(e)
                logger.error(f"Job {job_id} failed: {str(e)}")
                job.result = ProcessingResult(
                    status="error",
                    message=str(e),
                    output_path=None,
                    metadata={}
                )
            finally:
                job.end_time = datetime.utcnow()
                duration = job.end_time - job.start_time
                logger.info(f"Job {job_id} finished in {duration.total_seconds():.2f} seconds")

        asyncio.create_task(run_job())
        return job_id

    @classmethod
    def get_job_status(cls, job_id: str) -> Dict[str, Any]:
        """Get the current status of a job."""
        if job_id not in cls._jobs:
            logger.warning(f"Attempted to get status for non-existent job {job_id}")
            raise KeyError(f"Job {job_id} not found")
        
        job = cls._jobs[job_id]
        logger.debug(f"Retrieved status for job {job_id}: {job.status.value}")
        
        # Format the result properly
        result_dict = None
        if job.result and isinstance(job.result, ProcessingResult):
            result_dict = {
                'status': job.result.status,
                'message': job.result.message,
                'output_path': job.result.output_path,
                'metadata': job.result.metadata
            }
        
        # Ensure error is a string for failed jobs, None otherwise
        error_message = str(job.error) if job.status == JobStatus.FAILED and job.error else None
        
        return {
            "job_id": job.job_id,
            "status": job.status.value,
            "result": result_dict,
            "error": error_message,
            "start_time": job.start_time.isoformat() if job.start_time else None,
            "end_time": job.end_time.isoformat() if job.end_time else None
        }

    @classmethod
    def list_jobs(cls) -> List[Dict[str, Any]]:
        """List all jobs and their current status.
        
        Returns:
            List[Dict[str, Any]]: List of status information for all jobs
        """
        job_count = len(cls._jobs)
        logger.debug(f"Listing all jobs. Total count: {job_count}")
        return [cls.get_job_status(job_id) for job_id in cls._jobs]