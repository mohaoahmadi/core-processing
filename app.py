"""FastAPI application module for geospatial image processing.

This module implements the main FastAPI application for the Core Processing API.
It provides RESTful endpoints for:
- Geospatial image processing (NDVI, land cover, orthomosaic)
- Job management and status tracking
- GeoServer integration for processed results

The application uses:
- FastAPI for the REST API framework
- Pydantic for request/response validation
- Background tasks for asynchronous processing
- Supabase for data persistence
- AWS S3 for file storage
- GeoServer for geospatial data publishing

Architecture:
- Modular processor design for different processing types
- Asynchronous job processing with status tracking
- Centralized configuration management
- Comprehensive logging and error handling
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import uuid
from typing import List, Dict, Any
from pydantic import BaseModel

from config import get_settings
from lib.supabase_client import init_supabase, get_supabase
from lib.s3_manager import init_s3_client, get_presigned_url
from lib.geoserver_api import init_geoserver_client, publish_geotiff
from lib.job_manager import JobManager
from processors.landcover import LandCoverProcessor
from processors.ndvi import NDVIProcessor
from processors.orthomosaic import OrthomosaicProcessor
from utils.logging import setup_logging

settings = get_settings()

# Initialize logging
setup_logging()

# Define request/response models
class ProcessingJobRequest(BaseModel):
    """Request model for creating a new processing job.
    
    Attributes:
        process_type (str): Type of processing to perform:
            - "landcover": Land cover classification
            - "ndvi": Normalized Difference Vegetation Index
            - "orthomosaic": Orthomosaic generation
        input_file (str): Path or identifier of the input file
        org_id (str): Organization identifier
        project_id (str): Project identifier
        parameters (Dict[str, Any]): Additional processing parameters:
            - For landcover: classification thresholds
            - For NDVI: band indices
            - For orthomosaic: blending method
    """
    process_type: str
    input_file: str
    org_id: str
    project_id: str
    parameters: Dict[str, Any] = {}

class JobStatusResponse(BaseModel):
    """Response model for job status information.
    
    Attributes:
        job_id (str): Unique identifier of the job
        status (str): Current status of the job:
            - "pending": Job is queued
            - "running": Job is being processed
            - "completed": Job finished successfully
            - "failed": Job encountered an error
        result (Dict[str, Any], optional): Processing results if completed
        error (str, optional): Error message if failed
        start_time (str, optional): ISO formatted job start time
        end_time (str, optional): ISO formatted job completion time
    """
    job_id: str
    status: str
    result: Dict[str, Any] = None
    error: str = None
    start_time: str = None
    end_time: str = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events.
    
    Handles:
    - Service initialization during startup
    - Resource cleanup during shutdown
    - Dependency injection setup
    """
    # Startup
    logger.info("Initializing services...")
    init_supabase()
    init_s3_client()
    init_geoserver_client()
    JobManager.initialize(max_workers=settings.MAX_WORKERS)
    yield
    # Shutdown
    logger.info("Shutting down services...")
    await JobManager.shutdown()

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_PREFIX}/openapi.json",
    docs_url=f"{settings.API_V1_PREFIX}/docs",
    lifespan=lifespan
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint.
    
    Returns:
        dict: Health status and API version information
        
    Note:
        Used for monitoring and load balancer health checks.
        Returns 200 OK if the service is healthy.
    """
    return {"status": "healthy", "version": settings.PROJECT_NAME}

# Processing endpoints
@app.post(f"{settings.API_V1_PREFIX}/jobs", response_model=Dict[str, str])
async def create_job(
    job_request: ProcessingJobRequest,
    background_tasks: BackgroundTasks
):
    """Create a new processing job.
    
    Args:
        job_request (ProcessingJobRequest): Job creation request
        background_tasks (BackgroundTasks): FastAPI background tasks handler
        
    Returns:
        Dict[str, str]: Dictionary containing the job ID
        
    Raises:
        HTTPException: If process type is unsupported
        
    Note:
        The job is processed asynchronously in the background.
        The client should poll the job status endpoint for results.
    """
    job_id = str(uuid.uuid4())
    
    # Select the appropriate processor based on process_type
    processor_map = {
        "landcover": LandCoverProcessor(),
        "ndvi": NDVIProcessor(),
        "orthomosaic": OrthomosaicProcessor()
    }
    
    if job_request.process_type not in processor_map:
        raise HTTPException(status_code=400, detail=f"Unsupported process type: {job_request.process_type}")
    
    processor = processor_map[job_request.process_type]
    
    # Prepare path parameters
    s3_key = f"raw-imagery/{job_request.org_id}/{job_request.project_id}/{job_request.input_file}"
    output_name = f"{job_request.project_id}_{job_request.process_type}_{job_id[:8]}"
    
    # Prepare parameters based on process type
    params = {
        "input_path": s3_key,
        "output_name": output_name,
        **job_request.parameters
    }
    
    # Submit the job
    await JobManager.submit_job(job_id, processor, params)
    
    return {"job_id": job_id}

@app.get(f"{settings.API_V1_PREFIX}/jobs/{{job_id}}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get the status of a specific job.
    
    Args:
        job_id (str): ID of the job to check
        
    Returns:
        JobStatusResponse: Current job status and results
        
    Raises:
        HTTPException: If job ID is not found
        
    Note:
        Returns detailed status information including:
        - Current status
        - Processing results if completed
        - Error information if failed
        - Timing information
    """
    try:
        return JobManager.get_job_status(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

@app.get(f"{settings.API_V1_PREFIX}/jobs", response_model=List[JobStatusResponse])
async def list_jobs():
    """List all processing jobs.
    
    Returns:
        List[JobStatusResponse]: List of all jobs and their status
        
    Note:
        Returns status information for all jobs in the system,
        ordered by creation time (newest first).
    """
    return JobManager.list_jobs()

# GeoServer integration endpoints
@app.post(f"{settings.API_V1_PREFIX}/geoserver/publish")
async def publish_to_geoserver(
    layer_name: str,
    s3_key: str,
    workspace: str = None
):
    """Publish a processed result to GeoServer.
    
    Args:
        layer_name (str): Name for the new GeoServer layer
        s3_key (str): S3 key of the file to publish
        workspace (str, optional): Target GeoServer workspace
        
    Returns:
        dict: Publication status and layer information
        
    Raises:
        HTTPException: If publication fails
        
    Note:
        The process:
        1. Downloads file from S3 to temporary location
        2. Publishes to GeoServer as a new layer
        3. Cleans up temporary files
        4. Returns layer access information
    """
    try:
        # First, we need to download the file from S3 to a temporary location
        import tempfile
        from pathlib import Path
        
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            tmp_path = tmp.name
        
        # Download the file
        from lib.s3_manager import download_file
        await download_file(s3_key, tmp_path)
        
        # Publish to GeoServer
        result = await publish_geotiff(layer_name, tmp_path, workspace)
        
        # Clean up
        Path(tmp_path).unlink(missing_ok=True)
        
        return result
    except Exception as e:
        logger.error(f"Error publishing to GeoServer: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=settings.DEBUG)