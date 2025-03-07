"""FastAPI application module for geospatial image processing.

This module implements the main FastAPI application for the Core Processing API.
It provides RESTful endpoints for:
- Geospatial image processing (NDVI, land cover, orthomosaic, health indices)
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
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import uuid
import os
import sys
from typing import List, Dict, Any
from pydantic import BaseModel

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
from processors.ndvi import NDVIProcessor
from processors.orthomosaic import OrthomosaicProcessor
from processors.health_indices import HealthIndicesProcessor, HEALTH_INDICES
from utils.logging import setup_logging
from utils.geo_utils import analyze_geotiff

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
            - "health_indices": Multiple vegetation and health indices
        input_file (str): Path or identifier of the input file
        org_id (uuid.UUID): Organization identifier
        project_id (uuid.UUID): Project identifier
        parameters (Dict[str, Any]): Additional processing parameters:
            - For landcover: classification thresholds
            - For NDVI: band indices
            - For orthomosaic: blending method
            - For health_indices: indices list, sensor type, band mapping
    """
    process_type: str
    input_file: str
    org_id: uuid.UUID
    project_id: uuid.UUID
    parameters: Dict[str, Any] = {}
    
    class Config:
        json_encoders = {
            uuid.UUID: lambda v: str(v)
        }

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
    """Create a new processing job."""
    logger.info(f"Received job request - Process Type: {job_request.process_type}")
    logger.debug(f"Full job request details: {job_request.dict()}")
    
    try:
        job_id = str(uuid.uuid4())
        logger.debug(f"Generated job ID: {job_id}")
        
        # Select processor
        processor_map = {
            "landcover": LandCoverProcessor(),
            "ndvi": NDVIProcessor(),
            "orthomosaic": OrthomosaicProcessor(),
            "health_indices": HealthIndicesProcessor()
        }
        
        if job_request.process_type not in processor_map:
            logger.error(f"Invalid process type requested: {job_request.process_type}")
            raise HTTPException(status_code=400, detail=f"Unsupported process type: {job_request.process_type}")
        
        processor = processor_map[job_request.process_type]
        logger.debug(f"Selected processor: {processor.__class__.__name__}")
        
        # Prepare parameters
        s3_key = f"{job_request.org_id}/{job_request.project_id}/{job_request.input_file}"
        output_name = f"{job_request.project_id}_{job_request.process_type}_{job_id[:8]}"
        
        logger.debug(f"Prepared S3 key: {s3_key}")
        logger.debug(f"Prepared output name: {output_name}")
        
        params = {
            "input_path": s3_key,
            "output_name": output_name,
            **job_request.parameters
        }
        
        logger.debug(f"Final processing parameters: {params}")
        
        # Record the job in Supabase - using the string fields directly, not as UUIDs
        try:
            supabase = get_supabase()
            
            # Get organization name
            org_result = supabase.table("organizations").select("name").eq("id", job_request.org_id).execute()
            if not org_result.data:
                raise HTTPException(status_code=404, detail=f"Organization with ID {job_request.org_id} not found")
            org_name = org_result.data[0]["name"]

            # Get project name
            project_result = supabase.table("projects").select("name").eq("id", job_request.project_id).execute()
            if not project_result.data:
                raise HTTPException(status_code=404, detail=f"Project with ID {job_request.project_id} not found")
            project_name = project_result.data[0]["name"]
            
            # Create a proper insert that maps to your table structure
            # Store project_id as string instead of trying to convert to UUID
            supabase.table("processing_jobs").insert({
                "id": job_id,
                "project_id": job_request.project_id,
                "organization_id": job_request.org_id,
                "input_file": job_request.input_file,
                "process_type": job_request.process_type,
                "parameters": job_request.parameters,
                "status": "pending"
            }).execute()
            
            logger.info(f"Job record created in Supabase")
        except Exception as db_error:
            # Log the database error but continue with job processing
            logger.error(f"Error recording job in database: {str(db_error)}")
            # Continue processing anyway since the job manager doesn't require db record
        
        # Submit job
        logger.info(f"Submitting job {job_id} to JobManager")
        await JobManager.submit_job(job_id, processor, params)
        
        logger.info(f"Job {job_id} submitted successfully")
        return {"job_id": job_id}
        
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
        # Return the health indices dictionary with metadata
        return HEALTH_INDICES
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
        
        # Generate presigned URL for PUT operation
        presigned_url = await get_presigned_url(s3_key, http_method="PUT", expires_in=3600)
        
        # Record the upload request in Supabase
        supabase = get_supabase()
        supabase.table("file_uploads").insert({
            "id": str(uuid.uuid4()),
            "filename": request.filename,
            "s3_key": s3_key,
            "org_id": request.org_id,
            "project_id": request.project_id,
            "content_type": request.content_type,
            "status": "pending"
        }).execute()
        
        return {
            "upload_url": presigned_url,
            "s3_key": s3_key,
            "expires_in": 3600
        }
    except Exception as e:
        logger.error(f"Error generating presigned URL: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post(f"{settings.API_V1_PREFIX}/uploads/complete")
async def complete_upload(s3_key: str, file_size: int = 0):
    """Mark an upload as complete.
    
    Args:
        s3_key (str): S3 key of the uploaded file
        file_size (int): Size of the uploaded file in bytes
        
    Returns:
        dict: Success status
        
    Raises:
        HTTPException: If the upload record is not found
    """
    try:
        # Update the upload status in Supabase
        supabase = get_supabase()
        
        result = supabase.table("file_uploads").update({
            "status": "completed",
            "file_size": file_size,
            "uploaded_at": "now()"
        }).eq("s3_key", s3_key).execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="Upload record not found")
        
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error completing upload: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get(f"{settings.API_V1_PREFIX}/raster-files/{{project_id}}", response_model=List[Dict[str, Any]])
async def list_project_raster_files(project_id: str):
    """List all raster files for a project.
    
    Args:
        project_id: UUID of the project
        
    Returns:
        List[Dict[str, Any]]: List of raster files
    """
    try:
        # Query Supabase for completed file uploads for this project
        supabase = get_supabase()
        result = supabase.table("file_uploads").select("*").eq(
            "project_id", project_id
        ).eq("status", "completed").order("uploaded_at", desc=True).execute()
        
        return result.data
    except Exception as e:
        logger.error(f"Error listing project raster files: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get(f"{settings.API_V1_PREFIX}/raster-files/{{raster_file_id}}/metadata", response_model=Dict[str, Any])
async def get_raster_file_metadata_endpoint(raster_file_id: str):
    """Get metadata for a raster file.
    
    Args:
        raster_file_id: UUID of the raster file
        
    Returns:
        Dict[str, Any]: Raster file metadata
    """
    try:
        # Get file information from Supabase
        supabase = get_supabase()
        result = supabase.table("file_uploads").select("*").eq(
            "id", raster_file_id
        ).execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="Raster file not found")
        
        file_data = result.data[0]
        
        # Analyze the GeoTIFF to get metadata
        request = GeoTiffAnalysisRequest(
            file_path=file_data["s3_key"],
            org_id=file_data["org_id"],
            project_id=file_data["project_id"]
        )
        
        metadata = await analyze_geotiff_endpoint(request)
        
        # Combine file information with metadata
        return {
            "file_info": file_data,
            "geotiff_metadata": metadata
        }
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
async def get_processed_raster_metadata_endpoint(processed_raster_id: str):
    """Get metadata for a processed raster.
    
    Args:
        processed_raster_id: UUID of the processed raster
        
    Returns:
        Dict[str, Any]: Processed raster metadata
    """
    try:
        # Get job information from Supabase
        supabase = get_supabase()
        result = supabase.table("processing_jobs").select("*").eq(
            "id", processed_raster_id
        ).execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="Processed raster not found")
        
        job_data = result.data[0]
        
        # Extract output file path from job result
        if not job_data.get("result") or not job_data["result"].get("output_path"):
            raise HTTPException(status_code=404, detail="No output path found in job result")
        
        output_path = job_data["result"]["output_path"]
        
        # If it's an S3 path, extract the key
        if output_path.startswith("s3://"):
            output_path = output_path.split('/', 3)[3]
        
        # Analyze the GeoTIFF to get metadata
        request = GeoTiffAnalysisRequest(
            file_path=output_path
        )
        
        try:
            metadata = await analyze_geotiff_endpoint(request)
        except Exception as e:
            logger.warning(f"Error analyzing processed raster: {str(e)}")
            metadata = {"error": str(e)}
        
        # Combine job information with metadata
        return {
            "job_info": job_data,
            "geotiff_metadata": metadata
        }
    except Exception as e:
        logger.error(f"Error getting processed raster metadata: {str(e)}")
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=settings.DEBUG)