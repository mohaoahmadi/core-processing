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
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
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
        org_id (str): Organization identifier
        project_id (str): Project identifier
        parameters (Dict[str, Any]): Additional processing parameters:
            - For landcover: classification thresholds
            - For NDVI: band indices
            - For orthomosaic: blending method
            - For health_indices: indices list, sensor type, band mapping
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
    jobs = JobManager.list_jobs()
    # Ensure each job matches the JobStatusResponse model
    formatted_jobs = []
    for job in jobs:
        formatted_job = JobStatusResponse(
            job_id=job["job_id"],
            status=job["status"],
            result=job["result"],
            error=job["error"] if job["error"] is not None else None,
            start_time=job["start_time"],
            end_time=job["end_time"]
        )
        formatted_jobs.append(formatted_job)
    return formatted_jobs

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=settings.DEBUG)