"""FastAPI application module for geospatial image processing.

This module implements the main FastAPI application for the Core Processing API.
It provides RESTful endpoints for:
- Geospatial image processing (health indices, land cover, orthomosaic, terrain analysis)
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
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import uuid
import os
import sys
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import datetime
from enum import Enum

# Initialize GDAL before importing other modules
try:
    # Set GDAL environment variables
    os.environ['GDAL_DATA'] = '/usr/share/gdal'
    
    # Try to import and initialize GDAL
    from osgeo import gdal
    gdal.AllRegister()  # Register all drivers
    gdal.UseExceptions()  # Enable exceptions for better error messages
    
    # Log GDAL configuration
    logger.info(f"GDAL version: {gdal.VersionInfo('RELEASE_NAME')}")
    driver_count = gdal.GetDriverCount()
    logger.info(f"GDAL driver count: {driver_count}")
    if driver_count > 0:
        logger.info(f"Available GDAL drivers: {[gdal.GetDriver(i).ShortName for i in range(min(10, driver_count))]}")
    else:
        logger.warning("No GDAL drivers available!")
    
    # Check if GTiff driver is available
    gtiff_driver = gdal.GetDriverByName('GTiff')
    if gtiff_driver:
        logger.info("GTiff driver is available")
    else:
        logger.warning("GTiff driver is NOT available!")
except Exception as e:
    logger.error(f"Error initializing GDAL: {str(e)}")
    # Don't exit, as the app might still be useful for other endpoints

from config import get_settings
from lib.supabase_client import init_supabase, get_supabase
from lib.s3_manager import init_s3_client, get_presigned_url
from lib.geoserver_api import init_geoserver_client, publish_geotiff
from lib.job_manager import JobManager
from processors.landcover import LandCoverProcessor
from processors.orthomosaic import OrthomosaicProcessor
from processors.health_indices import HealthIndicesProcessor
from processors.terrain import TerrainAnalysisProcessor
from utils.logging import setup_logging
from utils.geo_utils import analyze_geotiff
from utils.supabase_raster_manager import get_project_raster_files

settings = get_settings()

# Initialize logging
setup_logging()

class ProcessType(str, Enum):
    """Enumeration of available processing types."""
    ORTHOMOSAIC = "orthomosaic"
    HEALTH_INDICES = "health_indices"
    CLASSIFICATION = "classification"
    LAND_COVER = "land_cover"
    TERRAIN = "terrain_analysis"

# Define request/response models
class ProcessingJobRequest(BaseModel):
    """Request model for creating a new processing job.
    
    Attributes:
        process_type (ProcessType): Type of processing to perform:
            - "orthomosaic": Orthomosaic generation
            - "health_indices": Multiple vegetation and health indices
            - "classification": Image classification
            - "land_cover": Land cover analysis
            - "terrain_analysis": Terrain analysis and DEM processing
        input_file (str): Path or identifier of the input file
        org_id (uuid.UUID): Organization identifier
        project_id (uuid.UUID): Project identifier
        parameters (Dict[str, Any]): Additional processing parameters:
            - For orthomosaic: blending method
            - For health_indices: indices list, sensor type, band mapping
            - For classification: model parameters
            - For land_cover: classification thresholds
            - For terrain_analysis: analysis types, parameters
    """
    process_type: ProcessType
    input_file: str
    org_id: uuid.UUID
    project_id: uuid.UUID
    parameters: Dict[str, Any] = {}
    
    class Config:
        json_encoders = {
            uuid.UUID: lambda v: str(v),
            ProcessType: lambda v: v.value  # Ensure ProcessType is serialized as its string value
        }
        use_enum_values = True  # This ensures the enum is handled as its value in JSON

class ProcessTypeInfo(BaseModel):
    """Information about a processing type."""
    name: str
    description: str
    parameters: Dict[str, Any]

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
    result: Dict[str, Any] | None = None
    error: str | None = None
    start_time: str | None = None
    end_time: str | None = None

    class Config:
        # Allow null values for optional fields
        json_encoders = {
            type(None): lambda _: None
        }

class GeoTiffAnalysisRequest(BaseModel):
    """Request model for GeoTIFF analysis.
    
    Attributes:
        file_path (str): Path to the GeoTIFF file (local or S3 path)
        org_id (str, optional): Organization identifier
        project_id (str, optional): Project identifier
    """
    file_path: str
    org_id: str | None = None
    project_id: str | None = None

class PresignedUrlRequest(BaseModel):
    """Request model for generating a presigned URL for S3 upload.
    
    Attributes:
        filename (str): Name of the file to be uploaded
        org_id (str): Organization identifier
        project_id (str): Project identifier
        content_type (str, optional): MIME type of the file
    """
    filename: str
    org_id: str
    project_id: str
    content_type: str = "image/tiff"

class CreateJobResponse(BaseModel):
    """Response model for job creation.
    
    Attributes:
        job_id (str): Unique identifier for the created job
        processed_raster_id (Optional[str]): ID of the processed raster record, if created
    """
    job_id: str
    processed_raster_id: Optional[str] = None

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
@app.post(f"{settings.API_V1_PREFIX}/jobs", response_model=CreateJobResponse)
async def create_job(
    job_request: ProcessingJobRequest,
    background_tasks: BackgroundTasks
):
    """Create a new processing job."""
    logger.info(f"Received job request - Process Type: {job_request.process_type}")
    logger.debug(f"Full job request details: {job_request.dict()}")
    
    try:
        job_id = str(uuid.uuid4())
        logger.debug(f"Generated job ID: {job_id}")
        
        # Select processor based on process type
        processor_map = {
            "orthomosaic": OrthomosaicProcessor(),
            "health_indices": HealthIndicesProcessor(),
            "land_cover": LandCoverProcessor(),
            "classification": LandCoverProcessor(),  # Using LandCover for now
            "terrain_analysis": TerrainAnalysisProcessor()
        }
        
        process_type_str = job_request.process_type if isinstance(job_request.process_type, str) else job_request.process_type.value
        
        if process_type_str not in processor_map:
            logger.error(f"Invalid process type requested: {process_type_str}")
            raise HTTPException(status_code=400, detail=f"Unsupported process type: {process_type_str}")
        
        processor = processor_map[process_type_str]
        logger.debug(f"Selected processor: {processor.__class__.__name__}")
        
        # Get input file information from raster_files table
        supabase = get_supabase()
        input_file_result = supabase.table("raster_files").select("*").eq(
            "project_id", str(job_request.project_id)  # Convert UUID to string
        ).eq("file_name", job_request.input_file).eq("type", "raw").execute()
        
        if not input_file_result.data:
            raise HTTPException(status_code=404, detail=f"Input file {job_request.input_file} not found")
        
        input_file = input_file_result.data[0]
        
        # Prepare S3 paths
        input_s3_key = input_file["s3_url"].replace("s3://", "") if input_file["s3_url"].startswith("s3://") else input_file["s3_url"]
        
        # For health indices, we'll create a base path since each index gets its own file
        if process_type_str == "health_indices":
            output_base_path = f"{str(job_request.org_id)}/{str(job_request.project_id)}/processed/health_indices"
            # We don't need output_name or output_s3_key for health indices as each index creates its own file
        else:
            output_name = f"{job_request.project_id}_{process_type_str}_{job_id[:8]}.tif"
            output_s3_key = f"{str(job_request.org_id)}/{str(job_request.project_id)}/processed/{process_type_str}/{output_name}"
        
        logger.debug(f"Input S3 key: {input_s3_key}")
        
        try:
            # Create initial processed_rasters record only for non-health-indices jobs
            if process_type_str != "health_indices":
                processed_raster_id = str(uuid.uuid4())
                processed_raster_record = {
                    "id": processed_raster_id,
                    "processing_job_id": None,  # Will update this after job creation
                    "raster_file_id": str(input_file["id"]),  # Convert UUID to string
                    "output_type": process_type_str,  # Use string version consistently
                    "s3_url": f"s3://mirzamspectrum/{output_s3_key}",
                    "width": 0,  # Will be updated after processing
                    "height": 0,
                    "band_count": 0,
                    "driver": "GTiff",
                    "bounds": {
                        "minx": 0,
                        "miny": 0,
                        "maxx": 0,
                        "maxy": 0
                    },
                    "metadata": {}
                }
                
                # Insert the processed raster record
                raster_result = supabase.table("processed_rasters").insert(processed_raster_record).execute()
                if not raster_result.data:
                    raise Exception("Failed to create processed raster record")
                logger.info(f"Created processed_rasters record: {processed_raster_id}")
            else:
                processed_raster_id = None  # For health indices, we'll create records when the job completes
            
            # Then create the job record
            job_record = {
                "id": job_id,
                "project_id": str(job_request.project_id),  # Convert UUID to string
                "organization_id": str(job_request.org_id),  # Convert UUID to string
                "input_file": job_request.input_file,
                "input_raster_id": str(input_file["id"]),  # Convert UUID to string
                "process_type": process_type_str,  # Use string version consistently
                "parameters": job_request.parameters,
                "status": "pending",
                "processed_raster_id": processed_raster_id,  # Will be None for health indices
                "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
            }
            
            # Insert job record
            job_result = supabase.table("processing_jobs").insert(job_record).execute()
            if not job_result.data:
                raise Exception("Failed to create processing job record")
            logger.info(f"Created processing job record: {job_id}")
            
            # Update the processed raster with the job ID (only for non-health-indices)
            if processed_raster_id:
                update_result = supabase.table("processed_rasters").update({
                    "processing_job_id": job_id
                }).eq("id", processed_raster_id).execute()
                if not update_result.data:
                    logger.warning(f"Failed to update processed raster with job ID")
            
        except Exception as db_error:
            # Try to clean up any created records in case of error
            try:
                if processed_raster_id and 'raster_result' in locals():
                    supabase.table("processed_rasters").delete().eq("id", processed_raster_id).execute()
            except Exception as cleanup_error:
                logger.error(f"Error during cleanup: {str(cleanup_error)}")
            
            logger.error(f"Database error: {str(db_error)}")
            raise HTTPException(status_code=500, detail=f"Failed to create database records: {str(db_error)}")
        
        # Prepare parameters for the processor
        params = {
            "input_path": input_s3_key,
            "output_base_path": output_base_path if process_type_str == "health_indices" else None,
            "output_name": output_name if process_type_str != "health_indices" else None,
            "output_path": output_s3_key if process_type_str != "health_indices" else None,
            "job_id": job_id,
            **job_request.parameters
        }
        
        logger.debug(f"Final processing parameters: {params}")
        
        # Submit job
        logger.info(f"Submitting job {job_id} to JobManager")
        await JobManager.submit_job(job_id, processor, params)
        
        logger.info(f"Job {job_id} submitted successfully")
        return CreateJobResponse(job_id=job_id, processed_raster_id=processed_raster_id)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating job: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

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
        # First, try to get job from JobManager
        try:
            return JobManager.get_job_status(job_id)
        except KeyError:
            # If not found in JobManager, try from Supabase
            supabase = get_supabase()
            job_result = supabase.table("processing_jobs").select("*").eq("id", job_id).execute()
            
            if not job_result.data:
                raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
            
            job = job_result.data[0]
            
            # Convert to JobStatusResponse format
            return JobStatusResponse(
                job_id=job["id"],
                status=job["status"],
                result=job.get("result"),
                error=job.get("error"),
                start_time=job.get("created_at"),
                end_time=job.get("completed_at")
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get(f"{settings.API_V1_PREFIX}/jobs", response_model=List[JobStatusResponse])
async def list_jobs(project_id: str = None):
    """List all processing jobs.
    
    Args:
        project_id (str, optional): Filter jobs by project ID
        
    Returns:
        List[JobStatusResponse]: List of all jobs and their status
        
    Note:
        Returns status information for all jobs in the system,
        ordered by creation time (newest first).
    """
    try:
        # Combine jobs from JobManager and Supabase
        active_jobs = JobManager.list_jobs()
        
        # Get historical jobs from Supabase
        supabase = get_supabase()
        query = supabase.table("processing_jobs").select("*")
        
        if project_id:
            query = query.eq("project_id", project_id)
        
        db_jobs_result = query.order("created_at", desc=True).execute()
        
        # Convert Supabase jobs to JobStatusResponse format
        db_jobs = []
        for job in db_jobs_result.data:
            # Skip jobs that are already in active_jobs
            if any(active_job["job_id"] == job["id"] for active_job in active_jobs):
                continue
                
            db_jobs.append(JobStatusResponse(
                job_id=job["id"],
                status=job["status"],
                result=job.get("result"),
                error=job.get("error"),
                start_time=job.get("created_at"),
                end_time=job.get("completed_at")
            ))
        
        # Convert active jobs to JobStatusResponse format
        formatted_active_jobs = []
        for job in active_jobs:
            formatted_active_jobs.append(JobStatusResponse(
                job_id=job["job_id"],
                status=job["status"],
                result=job["result"],
                error=job["error"] if job["error"] is not None else None,
                start_time=job["start_time"],
                end_time=job["end_time"]
            ))
        
        # Combine and return
        return formatted_active_jobs + db_jobs
    except Exception as e:
        logger.error(f"Error listing jobs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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
        
        # Extract org_id and project_id from s3_key if possible
        parts = s3_key.split('/')
        org_id = parts[0] if len(parts) >= 1 else None
        project_id = parts[1] if len(parts) >= 2 else None
        
        # Download the file
        from lib.s3_manager import download_file
        await download_file(s3_key, tmp_path)
        
        # Publish to GeoServer
        result = await publish_geotiff(layer_name, tmp_path, workspace)
        
        # Clean up
        Path(tmp_path).unlink(missing_ok=True)
        
        # Record layer in Supabase
        supabase = get_supabase()
        supabase.table("geoserver_layers").insert({
            "layer_name": layer_name,
            "workspace": workspace or settings.GEOSERVER_WORKSPACE,
            "s3_key": s3_key,
            "org_id": org_id,
            "project_id": project_id
        }).execute()
        
        return result
    except Exception as e:
        logger.error(f"Error publishing to GeoServer: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get(f"{settings.API_V1_PREFIX}/health-indices", response_model=Dict[str, Dict[str, Any]])
async def list_health_indices():
    """List all available health indices with descriptions and metadata.
    
    Returns:
        Dict[str, Dict[str, Any]]: Dictionary of health indices with their metadata
    """
    logger.info("Listing available health indices")
    
    try:
        # Query health indices from database
        supabase = get_supabase()
        result = supabase.table("health_indices").select("*").execute()
        
        if not result.data:
            logger.warning("No health indices found in database")
            return {}
            
        # Convert to the expected format
        indices = {}
        for row in result.data:
            indices[row['name']] = {
                'expr': row['formula'],
                'help': row['description'],
                'range': (row['min_value'], row['max_value']),
                'bands': row['required_bands']
            }
            
        return indices
    except Exception as e:
        logger.error(f"Error listing health indices: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing health indices: {str(e)}")

@app.get(f"{settings.API_V1_PREFIX}/sensors", response_model=Dict[str, Dict[str, int]])
async def list_sensor_band_mappings():
    """List all supported sensors and their default band mappings.
    
    Returns:
        Dict[str, Dict[str, int]]: Dictionary of sensors with their band mappings
    """
    logger.info("Listing supported sensors and band mappings")
    
    try:
        from processors.health_indices import DEFAULT_BAND_MAPPINGS
        return DEFAULT_BAND_MAPPINGS
    except Exception as e:
        logger.error(f"Error listing sensor band mappings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing sensor band mappings: {str(e)}")

@app.post(f"{settings.API_V1_PREFIX}/analyze-geotiff", response_model=Dict[str, Any])
async def analyze_geotiff_endpoint(request: GeoTiffAnalysisRequest):
    """Analyze a GeoTIFF file and return its metadata and band information.
    
    Args:
        request (GeoTiffAnalysisRequest): Request containing the file path
        
    Returns:
        Dict[str, Any]: Dictionary containing GeoTIFF metadata and band information
        
    Raises:
        HTTPException: If the file cannot be found or analyzed
    """
    logger.info(f"Received GeoTIFF analysis request for file: {request.file_path}")
    
    try:
        # Prepare the file path
        file_path = request.file_path
        
        # If org_id and project_id are provided, construct the S3 path
        if request.org_id and request.project_id:
            file_path = f"{request.org_id}/{request.project_id}/{request.file_path}"
            logger.debug(f"Constructed S3 path: {file_path}")
        
        # Use a synchronous version of analyze_geotiff for simplicity
        import numpy as np
        import os
        import tempfile
        
        local_file_path = file_path
        temp_file = None
        
        try:
            # Handle S3 paths
            if file_path.startswith("s3://") or not os.path.isfile(file_path):
                logger.info(f"Input is from S3 or not a local file: {file_path}")
                # Extract the key from s3://bucket/key or use as is
                s3_key = file_path.split('/', 3)[3] if file_path.startswith("s3://") else file_path
                logger.debug(f"Extracted S3 key: {s3_key}")
                
                from lib.s3_manager import download_file_sync
                
                temp_file = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
                local_file_path = temp_file.name
                temp_file.close()
                logger.debug(f"Created temporary file for download: {local_file_path}")
                
                try:
                    logger.info(f"Downloading from S3: {s3_key}")
                    download_file_sync(s3_key, local_file_path)
                    logger.info("S3 file downloaded successfully")
                except Exception as e:
                    logger.error(f"Failed to download from S3: {str(e)}")
                    raise Exception(f"Failed to download file from S3: {str(e)}")
            
            # Open the dataset with GDAL
            logger.info(f"Opening GeoTIFF file: {local_file_path}")
            
            # Check if file exists and has content
            if not os.path.exists(local_file_path):
                raise Exception(f"File does not exist: {local_file_path}")
                
            file_size = os.path.getsize(local_file_path)
            logger.info(f"File size: {file_size} bytes")
            if file_size == 0:
                raise Exception(f"File is empty: {local_file_path}")
                
            # Try to get file info
            try:
                import subprocess
                file_info = subprocess.check_output(['file', local_file_path]).decode('utf-8')
                logger.info(f"File info: {file_info}")
            except Exception as e:
                logger.warning(f"Could not get file info: {str(e)}")
            
            # Try to open with GDAL
            try:
                gdal.UseExceptions()  # Enable exceptions for better error messages
                ds = gdal.Open(local_file_path, gdal.GA_ReadOnly)
                if ds is None:
                    error_msg = gdal.GetLastErrorMsg() or "Unknown error"
                    logger.error(f"Failed to open GeoTIFF file: {local_file_path}. Error: {error_msg}")
                    raise Exception(f"Failed to open GeoTIFF file: {local_file_path}. Error: {error_msg}")
            except Exception as e:
                logger.error(f"GDAL exception: {str(e)}")
                
                # Try with rasterio as a fallback
                try:
                    import rasterio
                    logger.info("Trying to open with rasterio as fallback")
                    with rasterio.open(local_file_path) as src:
                        # Create a GDAL-like dataset from rasterio
                        ds = gdal.Open(local_file_path)
                        if ds is None:
                            raise Exception("Still failed with GDAL after rasterio check")
                except ImportError:
                    logger.error("Rasterio not available for fallback")
                    raise Exception(f"GDAL error opening file: {str(e)}")
                except Exception as rio_error:
                    logger.error(f"Rasterio fallback also failed: {str(rio_error)}")
                    raise Exception(f"Failed to open file with both GDAL and rasterio: {str(e)}, {str(rio_error)}")
            
            # Get basic metadata
            width = ds.RasterXSize
            height = ds.RasterYSize
            band_count = ds.RasterCount
            projection = ds.GetProjection()
            geotransform = ds.GetGeoTransform()
            
            # Calculate bounding box
            minx = geotransform[0]
            maxy = geotransform[3]
            maxx = minx + width * geotransform[1]
            miny = maxy + height * geotransform[5]  # Note: geotransform[5] is negative
            
            # Get driver metadata
            driver = ds.GetDriver().GetDescription()
            
            # Get band information
            bands_info = []
            for i in range(1, band_count + 1):
                band = ds.GetRasterBand(i)
                
                # Get band statistics
                stats = band.GetStatistics(True, True)
                
                # Get band data type
                dtype = gdal.GetDataTypeName(band.DataType)
                
                # Get nodata value
                nodata = band.GetNoDataValue()
                
                # Get color interpretation
                color_interp = gdal.GetColorInterpretationName(band.GetColorInterpretation())
                
                # Sample a small portion of the band to get a histogram
                # (avoid reading the entire band which could be large)
                xoff = width // 4
                yoff = height // 4
                win_width = min(width // 2, 1000)  # Limit to 1000 pixels
                win_height = min(height // 2, 1000)  # Limit to 1000 pixels
                
                data = band.ReadAsArray(xoff, yoff, win_width, win_height)
                
                # Calculate histogram for the sample
                hist_min = float(np.nanmin(data)) if data.size > 0 else 0
                hist_max = float(np.nanmax(data)) if data.size > 0 else 0
                
                # Add band info to the list
                bands_info.append({
                    "band_number": i,
                    "data_type": dtype,
                    "min": float(stats[0]),
                    "max": float(stats[1]),
                    "mean": float(stats[2]),
                    "stddev": float(stats[3]),
                    "nodata_value": float(nodata) if nodata is not None else None,
                    "color_interpretation": color_interp,
                    "histogram": {
                        "min": hist_min,
                        "max": hist_max
                    }
                })
            
            # Get metadata items
            metadata = {}
            domains = ds.GetMetadataDomainList() or []
            
            for domain in domains:
                domain_metadata = ds.GetMetadata(domain)
                if domain_metadata:
                    metadata[domain] = domain_metadata
            
            # Close the dataset
            ds = None
            
            # Prepare the result
            result = {
                "file_path": file_path,
                "width": width,
                "height": height,
                "band_count": band_count,
                "driver": driver,
                "projection": projection,
                "geotransform": geotransform,
                "bounds": {
                    "minx": minx,
                    "miny": miny,
                    "maxx": maxx,
                    "maxy": maxy
                },
                "bands": bands_info,
                "metadata": metadata
            }
            
            logger.info(f"Successfully analyzed GeoTIFF: {request.file_path}")
            return result
            
        finally:
            # Clean up temporary file if it exists
            if temp_file is not None and os.path.exists(local_file_path):
                try:
                    os.unlink(local_file_path)
                    logger.debug(f"Deleted temporary file: {local_file_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete temporary file {local_file_path}: {str(e)}")
        
    except Exception as e:
        logger.error(f"Error analyzing GeoTIFF: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error analyzing GeoTIFF: {str(e)}")

@app.post(f"{settings.API_V1_PREFIX}/uploads/request-url", response_model=Dict[str, str])
async def get_upload_url(request: PresignedUrlRequest):
    """Get a presigned URL for direct upload to S3.
    
    Args:
        request (PresignedUrlRequest): Request with file details
        
    Returns:
        dict: Presigned URL and S3 key
        
    Raises:
        HTTPException: If URL generation fails
    """
    try:
        # Generate S3 key
        s3_key = f"{request.org_id}/{request.project_id}/raw/{request.filename}"
        logger.info(f"Generated S3 key: {s3_key}")
        
        # Generate presigned URL for PUT operation
        presigned_url = await get_presigned_url(s3_key, http_method="PUT", expires_in=3600)
        
        # Try to record the upload request in Supabase
        try:
            supabase = get_supabase()
            
            # Check if the file_uploads table exists
            try:
                supabase.table("file_uploads").select("id").limit(1).execute()
                
                # If we get here, the table exists, so insert the record
                supabase.table("file_uploads").insert({
                    "id": str(uuid.uuid4()),
                    "filename": request.filename,
                    "s3_key": s3_key,
                    "org_id": request.org_id,
                    "project_id": request.project_id,
                    "content_type": request.content_type,
                    "status": "pending"
                }).execute()
                
            except Exception as table_error:
                # If the table doesn't exist, log a warning but continue
                if "relation \"public.file_uploads\" does not exist" in str(table_error):
                    logger.warning("The file_uploads table does not exist. Skipping record insertion.")
                    logger.warning("Please create the file_uploads table with the following schema:")
                    logger.warning("""
                    CREATE TABLE public.file_uploads (
                        id UUID PRIMARY KEY,
                        filename TEXT NOT NULL,
                        s3_key TEXT NOT NULL,
                        org_id UUID NOT NULL,
                        project_id UUID NOT NULL,
                        content_type TEXT NOT NULL,
                        status TEXT NOT NULL,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        FOREIGN KEY (org_id) REFERENCES organizations(id),
                        FOREIGN KEY (project_id) REFERENCES projects(id)
                    );
                    """)
                else:
                    # If it's a different error, re-raise it
                    raise table_error
                
        except Exception as db_error:
            # Log the database error but continue with URL generation
            logger.error(f"Error recording upload in database: {str(db_error)}")
            logger.warning("Continuing with URL generation despite database error")
        
        return {
            "upload_url": presigned_url,
            "s3_key": s3_key,
            "expires_in": str(3600)  # Convert to string to match response_model
        }
            
    except Exception as e:
        logger.error(f"Error generating presigned URL: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post(f"{settings.API_V1_PREFIX}/uploads/complete")
async def complete_upload(request: Request):
    """Mark an upload as complete.
    
    Args:
        request (Request): Request object containing s3_key and file_size
        
    Returns:
        dict: Success status
        
    Raises:
        HTTPException: If the upload record is not found
    """
    try:
        # Parse request body
        body = await request.json()
        s3_key = body.get("s3_key")
        file_size = body.get("file_size", 0)
        
        if not s3_key:
            raise HTTPException(status_code=422, detail="s3_key is required")
        
        logger.info(f"Completing upload for S3 key: {s3_key}, size: {file_size}")
        
        # Extract file information from s3_key
        parts = s3_key.split('/')
        if len(parts) >= 4:  # Expecting format: org_id/project_id/raw/filename
            org_id = parts[0]
            project_id = parts[1]
            filename = parts[-1]
        else:
            logger.error(f"Invalid S3 key format: {s3_key}")
            raise HTTPException(status_code=400, detail="Invalid S3 key format")
        
        # Update the upload status in Supabase
        supabase = get_supabase()
        
        try:
            # First, check if there's an existing record (including deleted ones)
            existing_result = supabase.table("raster_files").select("*").eq(
                "project_id", project_id
            ).eq("file_name", filename).execute()
            
            if existing_result.data:
                # Found existing record(s), update the most recent one
                existing_file = sorted(
                    existing_result.data,
                    key=lambda x: x.get("created_at", ""),
                    reverse=True
                )[0]
                
                # Update the existing record
                update_data = {
                    "s3_url": f"s3://{s3_key}",
                    "file_size": file_size,
                    "deleted": False,
                    "deleted_at": None
                }
                
                result = supabase.table("raster_files").update(
                    update_data
                ).eq("id", existing_file["id"]).execute()
                
                logger.info(f"Updated existing raster file record: {existing_file['id']}")
            else:
                # No existing record, create new one
                new_record = {
                    "id": str(uuid.uuid4()),
                    "project_id": project_id,
                    "file_name": filename,
                    "s3_url": f"s3://{s3_key}",
                    "file_size": file_size,
                    "deleted": False,
                    "width": 0,
                    "height": 0,
                    "band_count": 0,
                    "driver": "GTiff",
                    "bounds": {
                        "minx": 0,
                        "miny": 0,
                        "maxx": 0,
                        "maxy": 0
                    },
                    "metadata": {}
                }
                
                result = supabase.table("raster_files").insert(new_record).execute()
                logger.info(f"Created new raster file record")
            
        except Exception as db_error:
            logger.error(f"Database error: {str(db_error)}")
            if "relation \"public.raster_files\" does not exist" in str(db_error):
                logger.warning("The raster_files table does not exist. Skipping record update.")
            else:
                raise db_error
        
        return {"status": "success"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error completing upload: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get(f"{settings.API_V1_PREFIX}/raster-files/{{project_id}}", response_model=List[Dict[str, Any]])
async def list_project_raster_files(project_id: str, folder: str = None):
    """List raster files for a project, optionally filtered by folder.
    
    Args:
        project_id: UUID of the project
        folder: Optional folder filter ('raw' or 'processed')
        
    Returns:
        List[Dict[str, Any]]: List of raster files
    """
    try:
        supabase = get_supabase()
        query = supabase.table("raster_files").select("*").eq("project_id", project_id).eq("deleted", False)
        
        if folder == 'raw':
            query = query.eq("type", "raw")
        elif folder == 'processed':
            query = query.eq("type", "processed")
            
        result = query.order("created_at", desc=True).execute()
        
        return result.data
    except Exception as e:
        logger.error(f"Error listing project raster files: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get(f"{settings.API_V1_PREFIX}/raster-files/{{project_id}}/processed/{{raw_file_id}}", response_model=List[Dict[str, Any]])
async def list_processed_files_for_raw(project_id: str, raw_file_id: str):
    """List processed files derived from a specific raw file.
    
    Args:
        project_id: UUID of the project
        raw_file_id: UUID of the parent raw file
        
    Returns:
        List[Dict[str, Any]]: List of processed raster files
    """
    try:
        supabase = get_supabase()
        result = supabase.table("raster_files") \
            .select("*") \
            .eq("project_id", project_id) \
            .eq("parent_id", raw_file_id) \
            .eq("type", "processed") \
            .eq("deleted", False) \
            .order("created_at", desc=True) \
            .execute()
            
        return result.data
    except Exception as e:
        logger.error(f"Error listing processed files: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get(f"{settings.API_V1_PREFIX}/raster-files/{{raster_file_id}}/metadata", response_model=Dict[str, Any])
async def get_raster_file_metadata(raster_file_id: str):
    """Get metadata for a raster file.
    
    Args:
        raster_file_id: UUID of the raster file
        
    Returns:
        Dict[str, Any]: Raster file metadata
    """
    try:
        # Import the utility function from supabase_raster_manager
        from utils.supabase_raster_manager import get_raster_file_metadata
        
        # Use the dedicated function to get raster file metadata
        raster_file = await get_raster_file_metadata(raster_file_id)
        
        if not raster_file:
            logger.warning(f"Raster file not found: {raster_file_id}")
            raise HTTPException(status_code=404, detail="Raster file not found")
        
        logger.info(f"Retrieved metadata for raster file: {raster_file_id}")
        return raster_file
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting raster file metadata: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get(f"{settings.API_V1_PREFIX}/processed-rasters/{{project_id}}", response_model=List[Dict[str, Any]])
async def list_project_processed_rasters(project_id: str):
    """List all processed rasters for a project.
    
    Args:
        project_id: UUID of the project
        
    Returns:
        List[Dict[str, Any]]: List of processed rasters
    """
    try:
        # Query Supabase for completed processing jobs for this project
        supabase = get_supabase()
        result = supabase.table("processing_jobs").select("*").eq(
            "project_id", project_id
        ).eq("status", "completed").order("completed_at", desc=True).execute()
        
        return result.data
    except Exception as e:
        logger.error(f"Error listing processed rasters: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get(f"{settings.API_V1_PREFIX}/processed-rasters/{{processed_raster_id}}/metadata", response_model=Dict[str, Any])
async def get_processed_raster_metadata(processed_raster_id: str):
    """Get detailed metadata for a processed raster.
    
    Args:
        processed_raster_id: UUID of the processed raster
        
    Returns:
        Dict[str, Any]: Detailed metadata including:
            - Basic file information
            - Processing job details
            - Output-specific metadata
            - GeoTIFF metadata if available
    """
    try:
        supabase = get_supabase()
        
        # Get processed raster information with job details
        result = supabase.table("processed_rasters").select(
            "*",
            "processing_jobs(*)",
            "raster_file:raster_files(*)"
        ).eq("id", processed_raster_id).execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="Processed raster not found")
        
        processed_raster = result.data[0]
        
        # Get GeoTIFF metadata if the file exists
        geotiff_metadata = None
        s3_url = processed_raster.get("s3_url")
        
        if s3_url:
            try:
                # Extract S3 key from URL
                s3_key = s3_url[5:] if s3_url.startswith("s3://") else s3_url
                
                # Create analysis request
                analysis_request = GeoTiffAnalysisRequest(
                    file_path=s3_key
                )
                
                # Get GeoTIFF metadata
                geotiff_metadata = await analyze_geotiff_endpoint(analysis_request)
            except Exception as e:
                logger.warning(f"Could not analyze GeoTIFF: {str(e)}")
                geotiff_metadata = {"error": str(e)}
        
        # Construct response with all metadata
        response = {
            "id": processed_raster["id"],
            "file_info": {
                "s3_url": processed_raster["s3_url"],
                "output_type": processed_raster["output_type"],
                "width": processed_raster["width"],
                "height": processed_raster["height"],
                "band_count": processed_raster["band_count"],
                "driver": processed_raster["driver"],
                "bounds": processed_raster["bounds"],
                "created_at": processed_raster.get("created_at"),
                "updated_at": processed_raster.get("updated_at")
            },
            "processing_job": processed_raster.get("processing_jobs"),
            "input_raster": processed_raster.get("raster_file"),
            "metadata": processed_raster.get("metadata", {}),
            "geotiff_metadata": geotiff_metadata
        }
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting processed raster metadata: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete(f"{settings.API_V1_PREFIX}/processed-rasters/{{processed_raster_id}}")
async def delete_processed_raster(processed_raster_id: str):
    """Delete a processed raster and its associated resources.
    
    Args:
        processed_raster_id: UUID of the processed raster
        
    Returns:
        Dict[str, str]: Success message
        
    Raises:
        HTTPException: If the raster is not found or cannot be deleted
    """
    try:
        supabase = get_supabase()
        
        # Get processed raster information
        result = supabase.table("processed_rasters").select("*").eq("id", processed_raster_id).execute()
        
        if not result.data:
            logger.warning(f"Processed raster not found: {processed_raster_id}")
            raise HTTPException(status_code=404, detail="Processed raster not found")
        
        processed_raster = result.data[0]
        s3_url = processed_raster.get("s3_url")
        processing_job_id = processed_raster.get("processing_job_id")
        
        deletion_status = {
            "s3_deletion": False,
            "job_update": False,
            "record_deletion": False
        }
        
        # Delete from S3 if URL exists
        if s3_url:
            from lib.s3_manager import delete_file
            try:
                logger.info(f"Attempting to delete S3 file: {s3_url}")
                bucket_name = "mirzamspectrum"
                
                # Extract just the path part after the bucket name
                # Handle both s3://mirzamspectrum/path and just path formats
                if s3_url.startswith("s3://"):
                    # Remove s3:// and bucket name from the path
                    s3_key = "/".join(s3_url.split("/")[3:])
                else:
                    # If no s3:// prefix, just remove bucket name if present
                    s3_key = s3_url.replace(f"{bucket_name}/", "")
                
                logger.debug(f"Extracted S3 key for deletion: {s3_key}")
                
                deletion_success = await delete_file(s3_key, bucket_name)
                if deletion_success:
                    logger.info(f"Successfully deleted S3 file: {s3_key}")
                    deletion_status["s3_deletion"] = True
                else:
                    logger.warning(f"Failed to delete S3 file: {s3_key}, but continuing with database updates")
            except Exception as s3_error:
                logger.error(f"Error in S3 deletion attempt: {str(s3_error)}", exc_info=True)
        else:
            logger.info("No S3 URL found for processed raster")
            deletion_status["s3_deletion"] = True  # Mark as true since there's nothing to delete
        
        # Update the processing job to remove the reference
        if processing_job_id:
            try:
                logger.info(f"Updating processing job {processing_job_id} to remove processed raster reference")
                job_update = supabase.table("processing_jobs").update({
                    "processed_raster_id": None,
                    "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
                }).eq("id", processing_job_id).execute()
                
                if job_update.data:
                    logger.info(f"Successfully updated processing job {processing_job_id}")
                    deletion_status["job_update"] = True
                else:
                    logger.warning(f"Failed to update processing job {processing_job_id}")
            except Exception as job_error:
                logger.error(f"Error updating processing job: {str(job_error)}", exc_info=True)
        else:
            logger.info("No processing job ID found to update")
            deletion_status["job_update"] = True  # Mark as true since there's nothing to update
        
        # Delete the processed raster record
        try:
            logger.info(f"Deleting processed raster record: {processed_raster_id}")
            delete_result = supabase.table("processed_rasters").delete().eq("id", processed_raster_id).execute()
            
            if delete_result.data:
                logger.info(f"Successfully deleted processed raster record: {processed_raster_id}")
                deletion_status["record_deletion"] = True
            else:
                logger.error(f"Failed to delete processed raster record: {processed_raster_id}")
                raise Exception("Failed to delete processed raster record")
        except Exception as delete_error:
            logger.error(f"Error deleting processed raster record: {str(delete_error)}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to delete processed raster record")
        
        # Prepare detailed response
        success = all(deletion_status.values())
        message = "Processed raster deletion completed with "
        if success:
            message += "all operations successful"
        else:
            failed_ops = [k for k, v in deletion_status.items() if not v]
            message += f"some operations failed: {', '.join(failed_ops)}"
        
        return {
            "status": "success" if success else "partial_success",
            "message": message,
            "details": deletion_status
        }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting processed raster: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get(f"{settings.API_V1_PREFIX}/band-mappings", response_model=List[Dict[str, Any]])
async def list_band_mappings(project_id: str = None):
    """List band mappings.
    
    Args:
        project_id: Optional UUID of the project
        
    Returns:
        List[Dict[str, Any]]: List of band mappings
    """
    try:
        # Query Supabase for band mappings
        supabase = get_supabase()
        query = supabase.table("band_mappings").select("*")
        
        if project_id:
            # Get mappings for this project or global mappings
            query = query.or_(f"project_id.eq.{project_id},project_id.is.null")
        else:
            # Only get global mappings
            query = query.is_("project_id", "null")
        
        result = query.order("created_at", desc=True).execute()
        
        return result.data
    except Exception as e:
        logger.error(f"Error listing band mappings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post(f"{settings.API_V1_PREFIX}/band-mappings", response_model=Dict[str, str])
async def create_band_mapping(
    name: str,
    project_id: str,
    mapping: Dict[str, int],
    description: str = None
):
    """Create a custom band mapping.
    
    Args:
        name: Name of the band mapping
        mapping: Dictionary mapping band names to band numbers
        project_id: UUID of the project
        description: Optional description
        
    Returns:
        Dict[str, str]: Dictionary with the created band mapping ID
    """
    try:
        # Create band mapping in Supabase
        mapping_id = str(uuid.uuid4())
        
        supabase = get_supabase()
        supabase.table("band_mappings").insert({
            "id": mapping_id,
            "name": name,
            "mapping": mapping,
            "project_id": project_id,
            "description": description
        }).execute()
        
        return {"id": mapping_id}
    except Exception as e:
        logger.error(f"Error creating band mapping: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def construct_s3_path(org_id: uuid.UUID, project_id: uuid.UUID, filename: str) -> str:
    """Construct an S3 path using organization and project IDs."""
    return f"{org_id}/{project_id}/{filename}"

@app.get(f"{settings.API_V1_PREFIX}/organizations", response_model=List[Dict[str, Any]])
async def list_organizations():
    """List all organizations for the current user.
    
    Returns:
        List[Dict[str, Any]]: List of organizations
        
    Note:
        This endpoint retrieves all organizations from the database.
        In a production environment, this should be filtered by the authenticated user.
    """
    try:
        # Query Supabase for organizations
        supabase = get_supabase()
        result = supabase.table("organizations").select("*").execute()
        
        logger.info(f"Retrieved {len(result.data)} organizations")
        return result.data
    except Exception as e:
        logger.error(f"Error listing organizations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get(f"{settings.API_V1_PREFIX}/organizations/{{org_id}}/projects", response_model=List[Dict[str, Any]])
async def list_organization_projects(org_id: str):
    """List all projects for a specific organization.
    
    Args:
        org_id (str): Organization ID
        
    Returns:
        List[Dict[str, Any]]: List of projects
        
    Raises:
        HTTPException: If the organization is not found
    """
    try:
        # Verify organization exists
        supabase = get_supabase()
        org_result = supabase.table("organizations").select("*").eq("id", org_id).execute()
        
        if not org_result.data:
            logger.warning(f"Organization not found: {org_id}")
            raise HTTPException(status_code=404, detail=f"Organization with ID {org_id} not found")
        
        # Try to use service role client
        from lib.supabase_client import get_supabase_service_client
        service_client = get_supabase_service_client()
        
        if service_client:
            # Query projects using service role client
            logger.info(f"Querying projects for organization {org_id} using service role client")
            project_result = service_client.table("projects").select("*").eq("organization_id", org_id).execute()
            
            if project_result.data:
                logger.info(f"Found {len(project_result.data)} projects for organization {org_id}")
                return project_result.data
            else:
                logger.info(f"No projects found for organization {org_id}")
                return []
        else:
            # Fallback to regular client if service client is not available
            logger.warning("Service role client not available, using regular client")
            project_result = supabase.table("projects").select("*").eq("organization_id", org_id).execute()
            
            if project_result.data:
                logger.info(f"Found {len(project_result.data)} projects for organization {org_id}")
                return project_result.data
            else:
                logger.info(f"No projects found for organization {org_id}")
                return []
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing organization projects: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete(f"{settings.API_V1_PREFIX}/raster-files/{{raster_file_id}}")
async def delete_raster_file(raster_file_id: str):
    """Delete a raster file.
    
    Args:
        raster_file_id: UUID of the raster file
        
    Returns:
        Dict[str, str]: Success message
        
    Raises:
        HTTPException: If the file is not found or cannot be deleted
    """
    try:
        supabase = get_supabase()
        
        # Get file information
        result = supabase.table("raster_files").select("*").eq("id", raster_file_id).execute()
        
        if not result.data:
            logger.warning(f"Raster file not found: {raster_file_id}")
            raise HTTPException(status_code=404, detail="Raster file not found")
        
        file_data = result.data[0]
        s3_url = file_data.get("s3_url")
        
        deletion_status = {
            "s3_deletion": False,
            "record_update": False
        }
        
        # Delete from S3 if URL exists
        if s3_url:
            from lib.s3_manager import delete_file
            try:
                logger.info(f"Attempting to delete S3 file: {s3_url}")
                bucket_name = "mirzamspectrum"
                s3_key = s3_url[5:] if s3_url.startswith("s3://") else s3_url
                
                deletion_success = await delete_file(s3_key, bucket_name)
                if deletion_success:
                    logger.info(f"Successfully deleted S3 file: {s3_key}")
                    deletion_status["s3_deletion"] = True
                else:
                    logger.warning(f"Failed to delete S3 file: {s3_key}, but continuing with database update")
            except Exception as s3_error:
                logger.error(f"Error in S3 deletion attempt: {str(s3_error)}", exc_info=True)
        else:
            logger.info("No S3 URL found for raster file")
            deletion_status["s3_deletion"] = True  # Mark as true since there's nothing to delete
        
        # Update database record
        try:
            logger.info(f"Marking raster file as deleted: {raster_file_id}")
            update_result = supabase.table("raster_files").update({
                "deleted": True,
                "deleted_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
            }).eq("id", raster_file_id).execute()
            
            if update_result.data:
                logger.info(f"Successfully marked raster file as deleted: {raster_file_id}")
                deletion_status["record_update"] = True
            else:
                logger.error(f"Failed to update raster file record: {raster_file_id}")
                raise Exception("Failed to update raster file record")
        except Exception as update_error:
            logger.error(f"Error updating raster file record: {str(update_error)}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to update raster file record")
        
        # Prepare detailed response
        success = all(deletion_status.values())
        message = "Raster file deletion completed with "
        if success:
            message += "all operations successful"
        else:
            failed_ops = [k for k, v in deletion_status.items() if not v]
            message += f"some operations failed: {', '.join(failed_ops)}"
        
        return {
            "status": "success" if success else "partial_success",
            "message": message,
            "details": deletion_status
        }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting raster file: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post(f"{settings.API_V1_PREFIX}/jobs/{{job_id}}/complete", response_model=Dict[str, Any])
async def complete_job(job_id: str):
    """Mark a job as completed and record processed files."""
    try:
        # Get job information from Supabase
        supabase = get_supabase()
        job_result = supabase.table("processing_jobs").select("*").eq("id", job_id).execute()
        
        if not job_result.data:
            logger.error(f"Job not found: {job_id}")
            raise HTTPException(status_code=404, detail="Job not found")
        
        job_data = job_result.data[0]
        process_type = job_data.get("process_type")
        input_raster_id = job_data.get("input_raster_id")
        
        logger.info(f"Completing job {job_id} of type {process_type}")
        logger.debug(f"Job data: {job_data}")
        
        # Extract result data
        if not job_data.get("result"):
            logger.warning(f"No result data found for job {job_id}")
            return {"status": "completed", "message": "Job marked as completed, but no results found"}
        
        # For health indices, handle multiple output files
        if process_type == "health_indices":
            # Find indices results in different possible locations
            indices_results = None
            result_data = job_data["result"]
            
            # Check different possible paths for indices results
            if isinstance(result_data, dict):
                if "metadata" in result_data and "indices" in result_data["metadata"]:
                    indices_results = result_data["metadata"]["indices"]
                elif "indices" in result_data:
                    indices_results = result_data["indices"]
                elif "results" in result_data:
                    indices_results = result_data["results"]
                # Try to parse from raw_result if it exists
                elif "metadata" in result_data and "raw_result" in result_data["metadata"]:
                    try:
                        import ast
                        raw_result = result_data["metadata"]["raw_result"]
                        # Convert string representation to dict
                        if isinstance(raw_result, str):
                            metadata_str = raw_result.split("metadata=")[1]
                            metadata_dict = ast.literal_eval(metadata_str)
                            if "indices" in metadata_dict:
                                indices_results = metadata_dict["indices"]
                    except Exception as e:
                        logger.error(f"Error parsing raw_result: {str(e)}")
            
            if not indices_results:
                logger.warning("No indices results found in job output")
                return {"status": "completed", "message": "No indices results found"}
            
            logger.info(f"Found indices results: {list(indices_results.keys())}")
            
            # Track the first processed raster ID to update the job
            first_processed_raster_id = None
            
            # Process each index
            for index_name, index_data in indices_results.items():
                try:
                    # Extract or construct the output path
                    output_path = None
                    if isinstance(index_data, dict):
                        output_path = index_data.get("output_path")
                    
                    # If no output path found, construct it
                    if not output_path:
                        org_id = job_data.get("organization_id")
                        project_id = job_data.get("project_id")
                        output_path = f"{org_id}/{project_id}/processed/health_indices/{index_name}.tif"
                    
                    # Ensure the path starts with s3://mirzamspectrum/
                    if not output_path.startswith("s3://"):
                        output_path = f"s3://mirzamspectrum/{output_path}"
                    
                    # Create processed raster record
                    processed_raster_id = str(uuid.uuid4())
                    
                    # Store the first processed raster ID
                    if first_processed_raster_id is None:
                        first_processed_raster_id = processed_raster_id
                    
                    processed_raster = {
                        "id": processed_raster_id,
                        "processing_job_id": job_id,
                        "raster_file_id": input_raster_id,
                        "output_type": index_name,  # Use specific index name
                        "s3_url": output_path,
                        "width": index_data.get("width", 0),
                        "height": index_data.get("height", 0),
                        "band_count": 1,  # Health indices are single-band
                        "driver": "GTiff",
                        "bounds": {
                            "minx": index_data.get("bounds", {}).get("minx", 0),
                            "miny": index_data.get("bounds", {}).get("miny", 0),
                            "maxx": index_data.get("bounds", {}).get("maxx", 0),
                            "maxy": index_data.get("bounds", {}).get("maxy", 0)
                        },
                        "metadata": {
                            "index_name": index_name,
                            "formula": index_data.get("formula"),
                            "description": index_data.get("description"),
                            "value_range": index_data.get("value_range"),
                            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                            "status": "completed"
                        }
                    }
                    
                    # Insert the record
                    insert_result = supabase.table("processed_rasters").insert(processed_raster).execute()
                    if not insert_result.data:
                        logger.error(f"Failed to create processed raster record for {index_name}")
                    else:
                        logger.info(f"Created processed raster record for {index_name}: {processed_raster_id}")
                        
                except Exception as e:
                    logger.error(f"Error processing index {index_name}: {str(e)}")
                    continue
            
            # Update the job with the first processed raster ID
            if first_processed_raster_id:
                logger.info(f"Updating job with processed_raster_id: {first_processed_raster_id}")
                update_data = {
                    "status": "completed",
                    "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "processed_raster_id": first_processed_raster_id
                }
            else:
                update_data = {
                    "status": "completed",
                    "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
                }
        else:
            # For non-health-indices jobs
            update_data = {
                "status": "completed",
                "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
            }
        
        # Update job status
        job_update = supabase.table("processing_jobs").update(update_data).eq("id", job_id).execute()
        if not job_update.data:
            logger.warning(f"Failed to update job status")
        else:
            logger.info(f"Successfully updated job with status and processed_raster_id")
        
        return {"status": "completed", "message": "Job completed successfully"}
        
    except Exception as e:
        logger.error(f"Error completing job: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get(f"{settings.API_V1_PREFIX}/process-types", response_model=Dict[str, ProcessTypeInfo])
async def list_process_types():
    """List all available processing types with descriptions and required parameters.
    
    Returns:
        Dict[str, ProcessTypeInfo]: Dictionary of process types with their metadata
    """
    logger.info("Listing available process types")
    
    return {
        ProcessType.ORTHOMOSAIC: ProcessTypeInfo(
            name="Orthomosaic Generation",
            description="Generate orthomosaic from overlapping images",
            parameters={
                "blending_method": {
                    "type": "string",
                    "description": "Method used for blending overlapping images",
                    "options": ["average", "mosaic", "max", "min"]
                }
            }
        ),
        ProcessType.HEALTH_INDICES: ProcessTypeInfo(
            name="Health Indices",
            description="Calculate various vegetation and health indices",
            parameters={
                "indices": {
                    "type": "array",
                    "description": "List of indices to calculate"
                },
                "sensor_type": {
                    "type": "string",
                    "description": "Type of sensor/satellite"
                },
                "band_mapping": {
                    "type": "object",
                    "description": "Custom band number mapping"
                }
            }
        ),
        ProcessType.CLASSIFICATION: ProcessTypeInfo(
            name="Image Classification",
            description="Perform image classification using machine learning",
            parameters={
                "model": {
                    "type": "string",
                    "description": "Classification model to use"
                },
                "classes": {
                    "type": "array",
                    "description": "List of classes to identify"
                }
            }
        ),
        ProcessType.LAND_COVER: ProcessTypeInfo(
            name="Land Cover Analysis",
            description="Analyze and classify land cover types",
            parameters={
                "classification_type": {
                    "type": "string",
                    "description": "Type of land cover classification"
                },
                "thresholds": {
                    "type": "object",
                    "description": "Classification thresholds"
                }
            }
        ),
        ProcessType.TERRAIN: ProcessTypeInfo(
            name="Terrain Analysis",
            description="Analyze terrain features and generate DEM products",
            parameters={
                "analysis_types": {
                    "type": "array",
                    "description": "Types of terrain analysis to perform"
                },
                "resolution": {
                    "type": "number",
                    "description": "Output resolution in meters"
                }
            }
        )
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=settings.DEBUG)