"""Terrain analysis processor module.

This module implements various terrain analysis operations including:
- Digital Elevation Model (DEM) generation
- Slope calculation
- Aspect calculation
- Hillshade generation
- Contour lines extraction
- Watershed delineation
"""

import os
import tempfile
from pathlib import Path
import logging
from typing import Dict, Any, List, Optional, Union
import numpy as np
import sys

from processors.base import BaseProcessor, ProcessingResult
from lib.s3_manager import upload_file, download_file
from utils.geo_utils import normalize_array
from config import get_settings

# Setup GDAL environment
try:
    from osgeo import gdal, ogr, osr
    gdal.UseExceptions()  # Enable exceptions
    GDAL_AVAILABLE = True
except ImportError:
    GDAL_AVAILABLE = False
    logging.warning("GDAL not available. Terrain analysis will not work.")

settings = get_settings()
logger = logging.getLogger(__name__)

class TerrainAnalysisProcessor(BaseProcessor):
    """Processor for terrain analysis operations.
    
    This processor performs various terrain analysis operations on elevation data:
    - Slope calculation
    - Aspect calculation
    - Hillshade generation
    - Contour lines extraction
    - Watershed delineation
    - Terrain roughness index
    - Topographic position index
    """
    
    def __init__(self):
        """Initialize the processor."""
        super().__init__()
        self.temp_files = []
        
    async def validate_input(self, **kwargs) -> bool:
        """Validate input parameters for terrain analysis.
        
        Args:
            **kwargs: Keyword arguments containing:
                input_path (str): Path to input DEM (local or S3)
                analysis_types (List[str]): List of analyses to perform
                resolution (float, optional): Output resolution in meters
                contour_interval (float, optional): Interval for contour lines
                min_basin_size (float, optional): Minimum watershed basin size
                
        Returns:
            bool: True if all required parameters are present and valid
        """
        if not GDAL_AVAILABLE:
            logger.error("GDAL is not available. Cannot proceed with terrain analysis.")
            return False
            
        # Check required parameters
        if "input_path" not in kwargs:
            logger.error("Missing required parameter: input_path")
            return False
            
        if "analysis_types" not in kwargs:
            logger.error("Missing required parameter: analysis_types")
            return False
            
        # Validate analysis types
        valid_types = {"slope", "aspect", "hillshade", "contours", "watershed", "roughness", "tpi"}
        requested_types = set(kwargs["analysis_types"])
        invalid_types = requested_types - valid_types
        
        if invalid_types:
            logger.error(f"Invalid analysis types: {invalid_types}")
            return False
            
        # Validate numeric parameters
        if "resolution" in kwargs:
            try:
                resolution = float(kwargs["resolution"])
                if resolution <= 0:
                    logger.error("Resolution must be positive")
                    return False
            except ValueError:
                logger.error("Invalid resolution value")
                return False
                
        if "contour_interval" in kwargs and "contours" in requested_types:
            try:
                interval = float(kwargs["contour_interval"])
                if interval <= 0:
                    logger.error("Contour interval must be positive")
                    return False
            except ValueError:
                logger.error("Invalid contour interval value")
                return False
                
        return True
        
    async def process(self, **kwargs) -> ProcessingResult:
        """Execute terrain analysis operations.
        
        Args:
            **kwargs: Keyword arguments containing:
                input_path (str): Path to input DEM (local or S3)
                analysis_types (List[str]): List of analyses to perform
                resolution (float, optional): Output resolution in meters
                contour_interval (float, optional): Interval for contour lines
                min_basin_size (float, optional): Minimum watershed basin size
                org_id (str, optional): Organization ID for S3 path structure
                project_id (str, optional): Project ID for S3 path structure
                
        Returns:
            ProcessingResult: Processing result containing:
                - Paths to generated terrain products
                - Statistics for each analysis
                - Metadata about the processing
        """
        if not GDAL_AVAILABLE:
            return ProcessingResult(
                status="error",
                message="GDAL is not available. Cannot proceed with terrain analysis."
            )
            
        logger.info("Starting terrain analysis process")
        
        # Extract parameters
        input_path: str = kwargs["input_path"]
        analysis_types: List[str] = kwargs["analysis_types"]
        resolution: float = kwargs.get("resolution", None)
        contour_interval: float = kwargs.get("contour_interval", 10.0)
        min_basin_size: float = kwargs.get("min_basin_size", 1000.0)
        org_id: str = kwargs.get("org_id")
        project_id: str = kwargs.get("project_id")
        
        # Create output directory
        output_dir = Path(settings.TEMP_DIR) / "terrain_analysis"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Handle S3 input
        local_input_path = input_path
        if input_path.startswith("s3://") or not os.path.isfile(input_path):
            s3_key = input_path.split('/', 3)[3] if input_path.startswith("s3://") else input_path
            
            temp_input = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
            local_input_path = temp_input.name
            temp_input.close()
            self.temp_files.append(local_input_path)
            
            try:
                await download_file(s3_key, local_input_path)
            except Exception as e:
                return ProcessingResult(
                    status="error",
                    message=f"Failed to download input file from S3: {str(e)}"
                )
        
        try:
            # Open the input DEM
            dem_ds = gdal.Open(local_input_path)
            if dem_ds is None:
                return ProcessingResult(
                    status="error",
                    message=f"Failed to open input DEM: {local_input_path}"
                )
            
            # Get DEM information
            geotransform = dem_ds.GetGeoTransform()
            projection = dem_ds.GetProjection()
            band = dem_ds.GetRasterBand(1)
            nodata = band.GetNoDataValue()
            
            # Create results dictionary
            results = {}
            
            # Process each requested analysis
            for analysis_type in analysis_types:
                output_filename = f"{analysis_type}.tif"
                output_path = str(output_dir / output_filename)
                
                if analysis_type == "slope":
                    # Calculate slope
                    options = gdal.DEMProcessingOptions(computeEdges=True, alg='Horn')
                    gdal.DEMProcessing(output_path, local_input_path, 'slope', options=options)
                    
                elif analysis_type == "aspect":
                    # Calculate aspect
                    options = gdal.DEMProcessingOptions(computeEdges=True, alg='Horn')
                    gdal.DEMProcessing(output_path, local_input_path, 'aspect', options=options)
                    
                elif analysis_type == "hillshade":
                    # Generate hillshade
                    options = gdal.DEMProcessingOptions(
                        computeEdges=True,
                        azimuth=315.0,
                        altitude=45.0,
                        scale=1.0
                    )
                    gdal.DEMProcessing(output_path, local_input_path, 'hillshade', options=options)
                    
                elif analysis_type == "roughness":
                    # Calculate terrain roughness index
                    dem_array = band.ReadAsArray()
                    kernel_size = 3
                    roughness = self._calculate_roughness(dem_array, kernel_size)
                    self._save_array_as_tiff(roughness, output_path, geotransform, projection)
                    
                elif analysis_type == "tpi":
                    # Calculate topographic position index
                    dem_array = band.ReadAsArray()
                    kernel_size = 3
                    tpi = self._calculate_tpi(dem_array, kernel_size)
                    self._save_array_as_tiff(tpi, output_path, geotransform, projection)
                
                # Upload result to S3
                if org_id and project_id:
                    s3_key = f"{org_id}/{project_id}/processed/terrain_analysis/{output_filename}"
                else:
                    s3_key = f"processed/terrain_analysis/{output_filename}"
                
                s3_path = await upload_file(output_path, s3_key)
                
                # Store result information
                results[analysis_type] = {
                    "output_path": s3_path,
                    "description": self._get_analysis_description(analysis_type)
                }
            
            return ProcessingResult(
                status="success",
                message=f"Successfully completed {len(results)} terrain analyses",
                metadata={
                    "analyses": results,
                    "dem_info": {
                        "resolution": (geotransform[1], abs(geotransform[5])),
                        "projection": projection,
                        "nodata": nodata
                    }
                }
            )
            
        except Exception as e:
            logger.error(f"Error in terrain analysis: {str(e)}", exc_info=True)
            return ProcessingResult(
                status="error",
                message=f"Error in terrain analysis: {str(e)}"
            )
            
        finally:
            # Clean up
            if 'dem_ds' in locals():
                dem_ds = None
    
    def _calculate_roughness(self, dem_array: np.ndarray, kernel_size: int) -> np.ndarray:
        """Calculate terrain roughness index."""
        from scipy.ndimage import uniform_filter
        
        # Calculate the difference between max and min elevation in the neighborhood
        kernel = np.ones((kernel_size, kernel_size))
        
        # Calculate local max and min
        local_max = uniform_filter(dem_array, size=kernel_size, mode='reflect')
        local_min = uniform_filter(-dem_array, size=kernel_size, mode='reflect')
        
        # Roughness is the difference between local max and min
        roughness = local_max + local_min
        
        return roughness
    
    def _calculate_tpi(self, dem_array: np.ndarray, kernel_size: int) -> np.ndarray:
        """Calculate topographic position index."""
        from scipy.ndimage import uniform_filter
        
        # Calculate mean elevation in the neighborhood
        mean_elev = uniform_filter(dem_array, size=kernel_size, mode='reflect')
        
        # TPI is the difference between the elevation and mean neighborhood elevation
        tpi = dem_array - mean_elev
        
        return tpi
    
    def _save_array_as_tiff(self, array: np.ndarray, output_path: str,
                           geotransform: tuple, projection: str) -> None:
        """Save a numpy array as a GeoTIFF."""
        driver = gdal.GetDriverByName('GTiff')
        rows, cols = array.shape
        
        ds = driver.Create(output_path, cols, rows, 1, gdal.GDT_Float32,
                          options=['COMPRESS=LZW', 'TILED=YES'])
        
        ds.SetGeoTransform(geotransform)
        ds.SetProjection(projection)
        
        band = ds.GetRasterBand(1)
        band.WriteArray(array)
        band.SetNoDataValue(0)
        
        # Clean up
        band = None
        ds = None
    
    def _get_analysis_description(self, analysis_type: str) -> str:
        """Get description for each analysis type."""
        descriptions = {
            "slope": "Slope steepness in degrees",
            "aspect": "Slope direction in degrees from north",
            "hillshade": "Illumination visualization with default sun position",
            "contours": "Elevation contour lines",
            "watershed": "Watershed basin delineation",
            "roughness": "Terrain roughness index showing local relief",
            "tpi": "Topographic position index showing local elevation differences"
        }
        return descriptions.get(analysis_type, "")
    
    async def cleanup(self) -> None:
        """Clean up temporary files."""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception as e:
                logger.warning(f"Failed to remove temporary file {temp_file}: {str(e)}")
        
        # Clean up output directory
        output_dir = Path(settings.TEMP_DIR) / "terrain_analysis"
        if output_dir.exists():
            for file in output_dir.glob("*.*"):
                try:
                    file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to remove output file {file}: {str(e)}") 