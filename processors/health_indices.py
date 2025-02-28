"""Health indices calculation processor module.

This module implements various vegetation and health indices calculations
for satellite or aerial imagery using PyQGIS. It supports multiple indices
including NDVI, NDRE, EVI, SAVI, and others to assess vegetation health,
water content, and other environmental parameters.
"""

import os
import tempfile
from pathlib import Path
import logging
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np

from processors.base import BaseProcessor, ProcessingResult
from lib.s3_manager import upload_file, download_file
from utils.geo_utils import normalize_array
from config import get_settings

# Import PyQGIS modules
try:
    from qgis.core import (
        QgsApplication,
        QgsRasterLayer,
        QgsRasterCalculator,
        QgsRasterCalculatorEntry,
        QgsCoordinateReferenceSystem,
        QgsProject
    )
    QGIS_AVAILABLE = True
except ImportError:
    QGIS_AVAILABLE = False
    logging.warning("PyQGIS not available. Health indices processor will not work.")

settings = get_settings()
logger = logging.getLogger(__name__)

# Define the dictionary of index algorithms with formulas and metadata
HEALTH_INDICES = {
    'NDVI': {
        'expr': '(N - R) / (N + R)',
        'help': 'Normalized Difference Vegetation Index shows the amount of green vegetation.',
        'range': (-1, 1),
        'bands': ['N', 'R']
    },
    'NDYI': {
        'expr': '(G - B) / (G + B)',
        'help': 'Normalized difference yellowness index (NDYI), best model variability in relative yield potential in Canola.',
        'range': (-1, 1),
        'bands': ['G', 'B']
    },
    'NDRE': {
        'expr': '(N - Re) / (N + Re)',
        'help': 'Normalized Difference Red Edge Index shows the amount of green vegetation of permanent or later stage crops.',
        'range': (-1, 1),
        'bands': ['N', 'Re']
    },
    'NDWI': {
        'expr': '(G - N) / (G + N)',
        'help': 'Normalized Difference Water Index shows the amount of water content in water bodies.',
        'range': (-1, 1),
        'bands': ['G', 'N']
    },
    'NDVI_Blue': {
        'expr': '(N - B) / (N + B)',
        'help': 'Normalized Difference Vegetation Index using blue band shows the amount of green vegetation.',
        'range': (-1, 1),
        'bands': ['N', 'B']
    },
    'ENDVI': {
        'expr': '((N + G) - (2 * B)) / ((N + G) + (2 * B))',
        'help': 'Enhanced Normalized Difference Vegetation Index is like NDVI, but uses Blue and Green bands instead of only Red to isolate plant health.',
        'range': (-1, 1),
        'bands': ['N', 'G', 'B']
    },
    'vNDVI': {
        'expr': '0.5268 * ((R ^ -0.1294) * (G ^ 0.3389) * (B ^ -0.3118))',
        'help': 'Visible NDVI is an un-normalized index for RGB sensors using constants derived from citrus, grape, and sugarcane crop data.',
        'range': (0, 1),
        'bands': ['R', 'G', 'B']
    },
    'VARI': {
        'expr': '(G - R) / (G + R - B)',
        'help': 'Visual Atmospheric Resistance Index shows the areas of vegetation.',
        'range': (-1, 1),
        'bands': ['G', 'R', 'B']
    },
    'MPRI': {
        'expr': '(G - R) / (G + R)',
        'help': 'Modified Photochemical Reflectance Index',
        'range': (-1, 1),
        'bands': ['G', 'R']
    },
    'EXG': {
        'expr': '(2 * G) - (R + B)',
        'help': 'Excess Green Index (derived from only the RGB bands) emphasizes the greenness of leafy crops such as potatoes.',
        'range': (-2, 2),
        'bands': ['G', 'R', 'B']
    },
    'BAI': {
        'expr': '1.0 / (((0.1 - R) ^ 2) + ((0.06 - N) ^ 2))',
        'help': 'Burn Area Index highlights burned land in the red to near-infrared spectrum.',
        'range': (0, 100),
        'bands': ['R', 'N']
    },
    'GLI': {
        'expr': '((2 * G) - R - B) / ((2 * G) + R + B)',
        'help': 'Green Leaf Index shows green leaves and stems.',
        'range': (-1, 1),
        'bands': ['G', 'R', 'B']
    },
    'GNDVI': {
        'expr': '(N - G) / (N + G)',
        'help': 'Green Normalized Difference Vegetation Index is similar to NDVI, but measures the green spectrum instead of red.',
        'range': (-1, 1),
        'bands': ['N', 'G']
    },
    'GRVI': {
        'expr': 'N / G',
        'help': 'Green Ratio Vegetation Index is sensitive to photosynthetic rates in forests.',
        'range': (0, 10),
        'bands': ['N', 'G']
    },
    'SAVI': {
        'expr': '(1.5 * (N - R)) / (N + R + 0.5)',
        'help': 'Soil Adjusted Vegetation Index is similar to NDVI but attempts to remove the effects of soil areas using an adjustment factor (0.5).',
        'range': (-1, 1.5),
        'bands': ['N', 'R']
    },
    'MNLI': {
        'expr': '((N ^ 2 - R) * 1.5) / (N ^ 2 + R + 0.5)',
        'help': 'Modified Non-Linear Index improves the Non-Linear Index algorithm to account for soil areas.',
        'range': (-1, 1.5),
        'bands': ['N', 'R']
    },
    'MSR': {
        'expr': '((N / R) - 1) / (sqrt(N / R) + 1)',
        'help': 'Modified Simple Ratio is an improvement of the Simple Ratio (SR) index to be more sensitive to vegetation.',
        'range': (-1, 1),
        'bands': ['N', 'R']
    },
    'RDVI': {
        'expr': '(N - R) / sqrt(N + R)',
        'help': 'Renormalized Difference Vegetation Index uses the difference between near-IR and red, plus NDVI to show areas of healthy vegetation.',
        'range': (-1, 1),
        'bands': ['N', 'R']
    },
    'TDVI': {
        'expr': '1.5 * ((N - R) / sqrt((N ^ 2) + R + 0.5))',
        'help': 'Transformed Difference Vegetation Index highlights vegetation cover in urban environments.',
        'range': (-1, 1.5),
        'bands': ['N', 'R']
    },
    'OSAVI': {
        'expr': '(N - R) / (N + R + 0.16)',
        'help': 'Optimized Soil Adjusted Vegetation Index is based on SAVI, but tends to work better in areas with little vegetation where soil is visible.',
        'range': (-1, 1),
        'bands': ['N', 'R']
    },
    'LAI': {
        'expr': '3.618 * (2.5 * (N - R) / (N + 6 * R - 7.5 * B + 1)) * 0.118',
        'help': 'Leaf Area Index estimates foliage areas and predicts crop yields.',
        'range': (0, 10),
        'bands': ['N', 'R', 'B']
    },
    'EVI': {
        'expr': '2.5 * (N - R) / (N + 6 * R - 7.5 * B + 1)',
        'help': 'Enhanced Vegetation Index is useful in areas where NDVI might saturate, by using blue wavelengths to correct soil signals.',
        'range': (-1, 1),
        'bands': ['N', 'R', 'B']
    },
    'ARVI': {
        'expr': '(N - (2 * R) + B) / (N + (2 * R) + B)',
        'help': 'Atmospherically Resistant Vegetation Index. Useful when working with imagery for regions with high atmospheric aerosol content.',
        'range': (-1, 1),
        'bands': ['N', 'R', 'B']
    }
}

# Default band mapping for common satellite sensors
DEFAULT_BAND_MAPPINGS = {
    'WV3': {  # WorldView-3
        'B': 2,  # Blue
        'G': 3,  # Green
        'R': 5,  # Red
        'Re': 6,  # Red Edge
        'N': 7,  # NIR
    },
    'S2': {  # Sentinel-2
        'B': 2,  # Blue
        'G': 3,  # Green
        'R': 4,  # Red
        'Re': 6,  # Red Edge
        'N': 8,  # NIR
    },
    'L8': {  # Landsat 8
        'B': 2,  # Blue
        'G': 3,  # Green
        'R': 4,  # Red
        'N': 5,  # NIR
    }
}

class HealthIndicesProcessor(BaseProcessor):
    """Processor for calculating various vegetation and health indices using PyQGIS.
    
    This processor calculates multiple vegetation and health indices from 
    multispectral imagery using PyQGIS for raster calculations. It supports
    a wide range of indices including NDVI, NDRE, EVI, SAVI, and others.
    
    The processor can handle local files or S3 paths and supports different
    satellite sensors with appropriate band mappings.
    """
    
    def __init__(self):
        """Initialize the processor.
        
        Sets up the initial state and initializes QGIS if available.
        """
        super().__init__()
        self.qgis_app = None
        self.temp_files = []
        
        if QGIS_AVAILABLE:
            # Initialize QGIS in headless mode
            self.qgis_app = QgsApplication([], False)
            self.qgis_app.initQgis()
            logger.info("QGIS initialized successfully in headless mode")
        else:
            logger.warning("PyQGIS not available. Health indices processor will not work.")

    async def validate_input(self, **kwargs) -> bool:
        """Validate input parameters for health indices calculation.
        
        Args:
            **kwargs: Keyword arguments containing:
                input_path (str): Path to input image (local or S3)
                output_dir (str, optional): Directory for output files (default: "health_indices")
                indices (List[str], optional): List of indices to calculate (default: all)
                sensor_type (str, optional): Sensor type for default band mapping (default: "WV3")
                band_mapping (Dict[str, int], optional): Custom band mapping
                
        Returns:
            bool: True if all required parameters are present and valid
        """
        if not QGIS_AVAILABLE:
            logger.error("PyQGIS is not available. Cannot proceed with health indices calculation.")
            return False
            
        # Check required parameters
        if "input_path" not in kwargs:
            logger.error("Missing required parameter: input_path")
            return False
            
        # Validate indices if provided
        if "indices" in kwargs:
            indices = kwargs["indices"]
            if not isinstance(indices, list):
                logger.error("Parameter 'indices' must be a list")
                return False
                
            # Check if all requested indices are supported
            for index in indices:
                if index not in HEALTH_INDICES:
                    logger.error(f"Unsupported index: {index}")
                    return False
        
        # Validate sensor_type if provided
        if "sensor_type" in kwargs:
            sensor_type = kwargs["sensor_type"]
            if sensor_type not in DEFAULT_BAND_MAPPINGS:
                logger.error(f"Unsupported sensor type: {sensor_type}")
                return False
        
        # Validate custom band_mapping if provided
        if "band_mapping" in kwargs:
            band_mapping = kwargs["band_mapping"]
            if not isinstance(band_mapping, dict):
                logger.error("Parameter 'band_mapping' must be a dictionary")
                return False
                
            # Check if all required bands are in the mapping
            required_bands = set()
            indices_to_process = kwargs.get("indices", list(HEALTH_INDICES.keys()))
            
            for index in indices_to_process:
                if index in HEALTH_INDICES:
                    required_bands.update(HEALTH_INDICES[index]["bands"])
            
            for band in required_bands:
                if band not in band_mapping:
                    logger.error(f"Missing band in custom mapping: {band}")
                    return False
        
        return True

    async def process(self, **kwargs) -> ProcessingResult:
        """Execute health indices calculation using PyQGIS.
        
        Args:
            **kwargs: Keyword arguments containing:
                input_path (str): Path to input image (local or S3)
                output_dir (str, optional): Directory for output files (default: "health_indices")
                indices (List[str], optional): List of indices to calculate (default: all)
                sensor_type (str, optional): Sensor type for default band mapping (default: "WV3")
                band_mapping (Dict[str, int], optional): Custom band mapping
                scale_output (bool, optional): Whether to scale output to 8-bit (default: True)
                
        Returns:
            ProcessingResult: Processing result containing:
                - Paths to generated index images in S3
                - Statistics for each index
                - Band information used for calculation
        """
        if not QGIS_AVAILABLE:
            return ProcessingResult(
                status="error",
                message="PyQGIS is not available. Cannot proceed with health indices calculation."
            )
            
        logger.info("Starting health indices calculation process")
        
        # Extract parameters
        input_path: str = kwargs["input_path"]
        output_dir_name: str = kwargs.get("output_dir", "health_indices")
        indices_to_process: List[str] = kwargs.get("indices", list(HEALTH_INDICES.keys()))
        sensor_type: str = kwargs.get("sensor_type", "WV3")
        custom_band_mapping: Dict[str, int] = kwargs.get("band_mapping", {})
        scale_output: bool = kwargs.get("scale_output", True)
        
        # Use custom band mapping if provided, otherwise use default for the sensor type
        band_mapping = custom_band_mapping if custom_band_mapping else DEFAULT_BAND_MAPPINGS.get(sensor_type, {})
        
        logger.debug(f"Processing parameters - Input: {input_path}, Output dir: {output_dir_name}")
        logger.debug(f"Indices to process: {indices_to_process}")
        logger.debug(f"Band mapping: {band_mapping}")
        
        # Create output directory
        output_dir = Path(settings.TEMP_DIR) / output_dir_name
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created output directory: {output_dir}")
        
        # Handle S3 input
        local_input_path = input_path
        if input_path.startswith("s3://") or not os.path.isfile(input_path):
            logger.info(f"Input is from S3 or not a local file: {input_path}")
            # Extract the key from s3://bucket/key or use as is
            s3_key = input_path.split('/', 3)[3] if input_path.startswith("s3://") else input_path
            logger.debug(f"Extracted S3 key: {s3_key}")
            
            temp_input = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
            local_input_path = temp_input.name
            temp_input.close()
            self.temp_files.append(local_input_path)
            logger.debug(f"Created temporary file for download: {local_input_path}")
            
            try:
                logger.info(f"Attempting to download from S3: {s3_key}")
                await download_file(s3_key, local_input_path)
                logger.info("S3 file downloaded successfully")
            except Exception as e:
                logger.error(f"Failed to download from S3: {str(e)}")
                return ProcessingResult(
                    status="error",
                    message=f"Failed to download input file from S3: {str(e)}"
                )
        
        try:
            # Load the input raster
            raster_layer = QgsRasterLayer(local_input_path, "input_raster")
            if not raster_layer.isValid():
                logger.error("Failed to load the input raster")
                return ProcessingResult(
                    status="error",
                    message="Failed to load the input raster"
                )
            
            # Retrieve raster parameters
            extent = raster_layer.extent()
            width = raster_layer.width()
            height = raster_layer.height()
            
            # Create raster calculator entries
            entries = {}
            for band_name, band_number in band_mapping.items():
                entry = QgsRasterCalculatorEntry()
                entry.ref = band_name
                entry.raster = raster_layer
                entry.bandNumber = band_number
                entries[band_name] = entry
            
            # Process each requested index
            results = {}
            for index_name in indices_to_process:
                if index_name not in HEALTH_INDICES:
                    logger.warning(f"Skipping unsupported index: {index_name}")
                    continue
                
                index_info = HEALTH_INDICES[index_name]
                expr = index_info["expr"]
                
                # Check if all required bands are available
                required_bands = index_info["bands"]
                missing_bands = [band for band in required_bands if band not in entries]
                
                if missing_bands:
                    logger.warning(f"Skipping {index_name} due to missing bands: {missing_bands}")
                    continue
                
                # Create output filename
                output_filename = f"{index_name.replace(' ', '_')}.tif"
                output_path = str(output_dir / output_filename)
                
                logger.info(f"Calculating {index_name} -> {output_filename}")
                
                # Create list of entries for this calculation
                calc_entries = [entries[band] for band in required_bands]
                
                # Run the calculation
                calc = QgsRasterCalculator(
                    expr,
                    output_path,
                    "GTiff",
                    extent,
                    width,
                    height,
                    calc_entries
                )
                
                result = calc.processCalculation()
                if result != 0:
                    logger.error(f"Calculation for {index_name} failed with error code {result}")
                    continue
                
                # Upload to S3
                s3_key = f"{output_dir_name}/{output_filename}"
                s3_path = await upload_file(output_path, s3_key)
                
                # Store result information
                results[index_name] = {
                    "output_path": s3_path,
                    "formula": expr,
                    "description": index_info["help"],
                    "value_range": index_info.get("range", None)
                }
            
            if not results:
                return ProcessingResult(
                    status="error",
                    message="No indices were successfully calculated"
                )
            
            return ProcessingResult(
                status="success",
                message=f"Successfully calculated {len(results)} health indices",
                metadata={
                    "indices": results,
                    "band_mapping": band_mapping,
                    "sensor_type": sensor_type
                }
            )
            
        except Exception as e:
            logger.error(f"Error in health indices calculation: {str(e)}", exc_info=True)
            return ProcessingResult(
                status="error",
                message=f"Error in health indices calculation: {str(e)}"
            )

    async def cleanup(self) -> None:
        """Clean up temporary files and QGIS resources.
        
        Removes all temporary files created during processing and
        properly shuts down QGIS if it was initialized.
        """
        # Clean up temporary files
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                    logger.debug(f"Removed temporary file: {temp_file}")
            except Exception as e:
                logger.warning(f"Failed to remove temporary file {temp_file}: {str(e)}")
        
        # Clean up output directory if needed
        output_dir = Path(settings.TEMP_DIR) / "health_indices"
        if output_dir.exists():
            for file in output_dir.glob("*.tif"):
                try:
                    file.unlink()
                    logger.debug(f"Removed output file: {file}")
                except Exception as e:
                    logger.warning(f"Failed to remove output file {file}: {str(e)}")
        
        # Shutdown QGIS if it was initialized
        if self.qgis_app is not None:
            self.qgis_app.exitQgis()
            logger.info("QGIS shutdown completed") 