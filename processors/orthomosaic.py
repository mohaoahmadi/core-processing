"""Orthomosaic generation processor module.

This module implements functionality for generating orthomosaic images
from multiple overlapping aerial or satellite images. It provides various
blending methods to create seamless mosaics while handling different
projections and resolutions.
"""

import numpy as np
import rasterio
from pathlib import Path
from typing import List, Dict, Any, Optional
import os
import tempfile
import logging
import uuid

from processors.base import BaseProcessor, ProcessingResult
from lib.s3_manager import upload_file, download_file
from utils.geo_utils import align_rasters, blend_overlapping_areas, analyze_geotiff
from utils.supabase_raster_manager import register_processed_raster
from config import get_settings

settings = get_settings()

class OrthomosaicProcessor(BaseProcessor):
    """Processor for generating orthomosaic from multiple images.
    
    This processor combines multiple overlapping images into a single
    seamless mosaic. It supports different blending methods and handles
    alignment of images with different projections and resolutions.
    
    The processor provides four blending methods:
    - average: Simple averaging of overlapping pixels
    - maximum: Takes the maximum value in overlapping areas
    - minimum: Takes the minimum value in overlapping areas
    - feather: Gradual blending based on distance from edges
    """

    def __init__(self):
        """Initialize the OrthomosaicProcessor."""
        self.logger = logging.getLogger(__name__)

    def validate_parameters(self, params):
        """Validate input parameters."""
        required_params = ["input_path", "job_id"]
        for param in required_params:
            if param not in params:
                raise ValueError(f"Missing required parameter: {param}")
        
        # Check if output_type is valid
        output_type = params.get("output_type", "rgb")
        if output_type not in ["rgb", "false_color"]:
            raise ValueError(f"Invalid output_type: {output_type}. Must be 'rgb' or 'false_color'")
        
        return True
        
    async def validate_input(self, **kwargs) -> bool:
        """Validate input parameters for orthomosaic generation.
        
        Args:
            **kwargs: Keyword arguments containing:
                input_path (str): Path to input image (local or S3)
                job_id (str): ID of the processing job
                output_type (str, optional): Type of output (rgb or false_color)
                
        Returns:
            bool: True if all required parameters are present and valid
        """
        # Check required parameters
        if "input_path" not in kwargs:
            self.logger.error("Missing required parameter: input_path")
            return False
            
        if "job_id" not in kwargs:
            self.logger.error("Missing required parameter: job_id")
            return False
            
        # Check if output_type is valid
        output_type = kwargs.get("output_type", "rgb")
        if output_type not in ["rgb", "false_color"]:
            self.logger.error(f"Invalid output_type: {output_type}. Must be 'rgb' or 'false_color'")
            return False
            
        return True
        
    async def process(self, **kwargs) -> ProcessingResult:
        """Process orthomosaic generation.
        
        Args:
            **kwargs: Keyword arguments containing:
                input_path (str): Path to input image (local or S3)
                job_id (str): ID of the processing job
                output_type (str, optional): Type of output (rgb or false_color)
                output_path (str, optional): Path for output file
                organization_id (str, optional): Organization ID
                project_id (str, optional): Project ID
                input_analysis (dict, optional): Pre-analyzed input metadata
        
        Returns:
            ProcessingResult: Result data including output path and metadata
        """
        try:
            input_path = kwargs["input_path"]
            job_id = kwargs["job_id"]
            output_type = kwargs.get("output_type", "rgb")
            
            # Get output path from params
            output_path = kwargs.get("output_path")
            
            # If output path wasn't provided, construct one
            if not output_path:
                org_id = kwargs.get("organization_id")
                project_id = kwargs.get("project_id")
                output_name = f"{project_id}_orthomosaic_{output_type}_{job_id[:8]}.tif"
                output_path = f"{org_id}/{project_id}/processed/orthomosaic/{output_name}"
            
            # Log processing details
            self.logger.info(f"Processing orthomosaic with type {output_type}")
            self.logger.info(f"Input path: {input_path}")
            self.logger.info(f"Output path: {output_path}")
            
            # Check if we already have input analysis data from the job creation
            input_analysis = kwargs.get("input_analysis")
            if input_analysis and isinstance(input_analysis, dict) and "file_path" in input_analysis:
                # Use the pre-downloaded file if available and exists
                if os.path.isfile(input_analysis["file_path"]):
                    self.logger.info(f"Using already downloaded file: {input_analysis['file_path']}")
                    local_input_path = input_analysis["file_path"]
                else:
                    self.logger.warning(f"Pre-downloaded file not found: {input_analysis['file_path']}")
                    local_input_path = None
            else:
                local_input_path = None
            
            # Download input file if needed
            if not local_input_path or not os.path.isfile(local_input_path):
                local_input_path = input_path
                if input_path.startswith("s3://") or not os.path.isfile(input_path):
                    self.logger.info(f"Input is from S3 or not a local file: {input_path}")
                    # Extract the key from s3://bucket/key or use as is
                    s3_key = input_path.split('/', 3)[3] if input_path.startswith("s3://") else input_path
                    self.logger.debug(f"Extracted S3 key: {s3_key}")
                    
                    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as temp_input:
                        local_input_path = temp_input.name
                    
                    try:
                        self.logger.info(f"Downloading from S3: {s3_key}")
                        await download_file(s3_key, local_input_path)
                        self.logger.info("S3 file downloaded successfully")
                    except Exception as e:
                        self.logger.error(f"Failed to download from S3: {str(e)}")
                        return ProcessingResult(
                            status="error",
                            message=f"Failed to download input file from S3: {str(e)}"
                        )
            
            # Create temporary output file
            with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as temp_output:
                local_output_path = temp_output.name
            
            # Import GDAL for processing
            try:
                from osgeo import gdal
                gdal.UseExceptions()  # Enable exceptions
                gdal.AllRegister()
            except ImportError:
                return ProcessingResult(
                    status="error",
                    message="GDAL is not available. Cannot proceed with orthomosaic processing."
                )
            
            # Open the input raster
            try:
                ds = gdal.Open(local_input_path)
                if ds is None:
                    return ProcessingResult(
                        status="error", 
                        message=f"Failed to open raster: {local_input_path}"
                    )
                
                # Get band count
                band_count = ds.RasterCount
                self.logger.info(f"Input raster has {band_count} bands")
                
                # Band mapping for different output types
                # For Landsat, RGB = 4,3,2 (Red, Green, Blue)
                # For Landsat, False Color = 5,4,3 (NIR, Red, Green)
                band_mapping = {}
                if output_type == "rgb":
                    # RGB composite - Red, Green, Blue bands
                    band_mapping = {
                        'r': 4,  # Red
                        'g': 3,  # Green
                        'b': 2   # Blue
                    }
                    self.logger.info(f"Creating RGB composite using bands {band_mapping}")
                else:  # false_color
                    # False Color (NIR, Red, Green)
                    band_mapping = {
                        'r': 5,  # NIR
                        'g': 4,  # Red
                        'b': 3   # Green
                    }
                    self.logger.info(f"Creating False Color composite using bands {band_mapping}")
                
                # Validate band mapping against available bands
                for band_name, band_number in band_mapping.items():
                    if band_number < 1 or band_number > band_count:
                        return ProcessingResult(
                            status="error",
                            message=f"Invalid band mapping: {band_name}={band_number}. Raster has {band_count} bands."
                        )
                
                # Create output raster
                driver = gdal.GetDriverByName('GTiff')
                options = ['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES', 'PHOTOMETRIC=RGB']
                out_ds = driver.Create(
                    local_output_path, 
                    ds.RasterXSize, 
                    ds.RasterYSize, 
                    3,  # Always 3 bands for RGB output
                    gdal.GDT_UInt16,
                    options=options
                )
                
                if out_ds is None:
                    return ProcessingResult(
                        status="error",
                        message=f"Failed to create output raster: {local_output_path}"
                    )
                
                # Set projection and geotransform
                out_ds.SetProjection(ds.GetProjection())
                out_ds.SetGeoTransform(ds.GetGeoTransform())
                
                # Set color interpretation before writing data
                for idx, color in enumerate(['r', 'g', 'b'], start=1):
                    out_band = out_ds.GetRasterBand(idx)
                    if idx == 1:
                        out_band.SetColorInterpretation(gdal.GCI_RedBand)
                    elif idx == 2:
                        out_band.SetColorInterpretation(gdal.GCI_GreenBand)
                    elif idx == 3:
                        out_band.SetColorInterpretation(gdal.GCI_BlueBand)
                
                # Copy the specified bands to the output
                import numpy as np
                
                # Track statistics for reporting
                stats = {}
                
                for idx, (color, band_num) in enumerate(band_mapping.items(), start=1):
                    in_band = ds.GetRasterBand(band_num)
                    out_band = out_ds.GetRasterBand(idx)
                    
                    # Read the band data
                    data = in_band.ReadAsArray()
                    
                    # Copy data to output band
                    out_band.WriteArray(data)
                    
                    # Copy metadata (except color interpretation which was set earlier)
                    out_band.SetNoDataValue(in_band.GetNoDataValue() or 0)
                    
                    # Compute statistics
                    if data is not None:
                        min_val = np.min(data) if data.size > 0 else 0
                        max_val = np.max(data) if data.size > 0 else 0
                        mean_val = np.mean(data) if data.size > 0 else 0
                        stats[color] = {
                            'min': float(min_val),
                            'max': float(max_val),
                            'mean': float(mean_val),
                            'source_band': band_num
                        }
                
                # Build overviews for faster rendering
                out_ds.BuildOverviews("NEAREST", [2, 4, 8, 16, 32])
                
                # Close datasets
                out_ds = None
                ds = None
                
                # Analyze the output file to get detailed metadata
                self.logger.info(f"Analyzing output file: {local_output_path}")
                try:
                    analysis_result = await analyze_geotiff(local_output_path)
                    self.logger.debug(f"Analysis result: {analysis_result}")
                except Exception as e:
                    self.logger.error(f"Error analyzing output file: {str(e)}")
                    analysis_result = None
                
                # Upload to S3
                try:
                    s3_path = await upload_file(local_output_path, output_path)
                    self.logger.info(f"Uploaded composite to S3: {s3_path}")
                    
                    # Register the processed raster in the database
                    if analysis_result:
                        try:
                            # Extract overall min/max/mean from the bands
                            min_value = min([band['min'] for band in analysis_result['bands']])
                            max_value = max([band['max'] for band in analysis_result['bands']])
                            mean_value = sum([band['mean'] for band in analysis_result['bands']]) / len(analysis_result['bands'])
                            
                            # Prepare metadata for registration
                            metadata = {
                                'width': analysis_result['width'],
                                'height': analysis_result['height'],
                                'band_count': analysis_result['band_count'],
                                'driver': analysis_result['driver'],
                                'projection': analysis_result['projection'],
                                'geotransform': analysis_result['geotransform'],
                                'bounds': analysis_result['bounds'],
                                'min_value': min_value,
                                'max_value': max_value,
                                'mean_value': mean_value,
                                'metadata': {
                                    'output_type': output_type,
                                    'band_mapping': band_mapping,
                                    'stats': stats
                                }
                            }
                            
                            # Get the source raster file ID
                            # In a real implementation, this would be retrieved from the database
                            # using the input_path. For now, we'll use None as we don't have this info.
                            source_raster_file_id = None
                            
                            # Register in database
                            processed_raster_id = await register_processed_raster(
                                job_id,
                                source_raster_file_id,
                                output_type,
                                s3_path,
                                metadata
                            )
                            
                            self.logger.info(f"Registered processed raster with ID: {processed_raster_id}")
                        except Exception as reg_error:
                            self.logger.error(f"Failed to register processed raster: {str(reg_error)}")
                except Exception as e:
                    self.logger.error(f"Failed to upload to S3: {str(e)}")
                    return ProcessingResult(
                        status="error",
                        message=f"Failed to upload output file to S3: {str(e)}"
                    )
                
                # Clean up temporary files, but don't delete input file if it was pre-downloaded
                try:
                    # Only delete the input file if we downloaded it in this method
                    if local_input_path != input_analysis.get("file_path", None):
                        os.unlink(local_input_path)
                    os.unlink(local_output_path)
                except Exception as e:
                    self.logger.warning(f"Failed to remove temporary file: {str(e)}")
                
                # Return success
                return ProcessingResult(
                    status="success",
                    message=f"Successfully generated {output_type} orthomosaic",
                    metadata={
                        "output_path": s3_path,
                        "output_type": output_type,
                        "band_mapping": band_mapping,
                        "statistics": stats,
                        "processing_time": "completed",
                        "analysis": analysis_result
                    }
                )
                
            except Exception as e:
                self.logger.error(f"Error processing with GDAL: {str(e)}", exc_info=True)
                return ProcessingResult(
                    status="error",
                    message=f"Error processing with GDAL: {str(e)}"
                )
            
        except Exception as e:
            self.logger.error(f"Error in orthomosaic processing: {str(e)}", exc_info=True)
            return ProcessingResult(
                status="error",
                message=f"Error in orthomosaic processing: {str(e)}"
            )

    async def cleanup(self) -> None:
        """Clean up temporary files.
        
        Removes all temporary files created during processing from the
        orthomosaic output directory.
        """
        self.logger.info("Cleaning up after orthomosaic processing")
        output_dir = Path(settings.TEMP_DIR) / "orthomosaic"
        if output_dir.exists():
            for file in output_dir.glob("*.tif"):
                try:
                    file.unlink()
                except Exception:
                    pass