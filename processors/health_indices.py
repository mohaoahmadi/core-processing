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
import sys

from processors.base import BaseProcessor, ProcessingResult
from lib.s3_manager import upload_file, download_file
from utils.geo_utils import normalize_array
from config import get_settings

# Setup QGIS environment variables and paths
QGIS_AVAILABLE = False
try:
    # Try to detect QGIS installation path
    qgis_paths = [
        # Linux paths
        '/usr/share/qgis',
        '/usr/local/share/qgis',
        # Windows paths
        'C:\\Program Files\\QGIS 3.22',
        'C:\\Program Files\\QGIS 3.28',
        # macOS paths
        '/Applications/QGIS.app/Contents/MacOS',
        '/Applications/QGIS-LTR.app/Contents/MacOS'
    ]
    
    qgis_path = None
    for path in qgis_paths:
        if os.path.exists(path):
            qgis_path = path
            break
    
    if qgis_path:
        # Set environment variables
        os.environ['QGIS_PREFIX_PATH'] = qgis_path
        
        # Add QGIS Python bindings to path
        if sys.platform == 'win32':  # Windows
            pyqgis_path = os.path.join(qgis_path, 'python')
            if os.path.exists(pyqgis_path):
                sys.path.insert(0, pyqgis_path)
        elif sys.platform == 'darwin':  # macOS
            pyqgis_path = os.path.join(qgis_path, 'Python3')
            if os.path.exists(pyqgis_path):
                sys.path.insert(0, pyqgis_path)
        else:  # Linux
            # Try common Linux paths for QGIS Python
            for py_path in ['/usr/share/qgis/python', '/usr/local/share/qgis/python']:
                if os.path.exists(py_path):
                    sys.path.insert(0, py_path)
                    break
    
    # Now try to import QGIS modules
    from qgis.core import (
        QgsApplication,
        QgsRasterLayer,
        QgsCoordinateReferenceSystem,
        QgsProject
    )
    
    # Try to import QgsRasterCalculator - it might be in a different location in some QGIS versions
    try:
        from qgis.core import QgsRasterCalculator, QgsRasterCalculatorEntry
    except ImportError:
        # Try alternative import locations for different QGIS versions
        try:
            from qgis.analysis import QgsRasterCalculator, QgsRasterCalculatorEntry
            logging.info("QgsRasterCalculator imported from qgis.analysis")
        except ImportError:
            # Create a fallback implementation if needed
            class QgsRasterCalculatorEntry:
                def __init__(self):
                    self.ref = ""
                    self.raster = None
                    self.bandNumber = 0
            
            class QgsRasterCalculator:
                def __init__(self, expression, output_file, output_format, extent, width, height, entries):
                    self.expression = expression
                    self.output_file = output_file
                    self.output_format = output_format
                    self.extent = extent
                    self.width = width
                    self.height = height
                    self.entries = entries
                
                def processCalculation(self):
                    logging.error("QgsRasterCalculator not available in this QGIS version")
                    return 1  # Error code
            
            logging.warning("Using fallback implementation for QgsRasterCalculator")
    
    QGIS_AVAILABLE = True
    logging.info("PyQGIS modules imported successfully")
except ImportError as e:
    QGIS_AVAILABLE = False
    logging.warning(f"PyQGIS not available. Health indices processor will not work. Error: {str(e)}")
    logging.warning("Make sure QGIS is installed and properly configured.")

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
    'CNDVI': {
        'expr': '(N - C) / (N + C)',
        'help': 'Coastal Normalized Difference Vegetation Index uses the coastal band instead of red, useful for shallow water vegetation mapping.',
        'range': (-1, 1),
        'bands': ['N', 'C']
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
        'C': 1,  # Coastal
        'B': 2,  # Blue
        'G': 3,  # Green
        'Y': 4,  # Yellow
        'R': 5,  # Red
        'Re': 6,  # Red Edge
        'N': 7,  # NIR1
        'N2': 8,  # NIR2
    },
    'S2': {  # Sentinel-2
        'C': 1,  # Coastal/Aerosol
        'B': 2,  # Blue
        'G': 3,  # Green
        'R': 4,  # Red
        'Re': 6,  # Red Edge
        'N': 8,  # NIR
    },
    'L8': {  # Landsat 8
        'C': 1,  # Coastal/Aerosol
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
            try:
                # Initialize QGIS in headless mode
                os.environ["QT_QPA_PLATFORM"] = "offscreen"  # Force offscreen rendering
                self.qgis_app = QgsApplication([], False)
                self.qgis_app.setPrefixPath(os.environ.get('QGIS_PREFIX_PATH', '/usr'), True)
                self.qgis_app.initQgis()
                
                # Create a project instance to avoid segmentation faults
                self.project = QgsProject.instance()
                
                logger.info("QGIS initialized successfully in headless mode")
            except Exception as e:
                logger.error(f"Failed to initialize QGIS: {str(e)}", exc_info=True)
                self.qgis_app = None
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
                org_id (str, optional): Organization ID for S3 path structure
                project_id (str, optional): Project ID for S3 path structure
                
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
                org_id (str, optional): Organization ID for S3 path structure
                project_id (str, optional): Project ID for S3 path structure
                
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
        org_id: str = kwargs.get("org_id")
        project_id: str = kwargs.get("project_id")
        
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
        
        # Check the input raster and verify band mapping
        try:
            # Try using GDAL first to check the raster
            try:
                from osgeo import gdal
                ds = gdal.Open(local_input_path)
                if ds is None:
                    logger.error(f"Failed to open raster with GDAL: {local_input_path}")
                    return ProcessingResult(
                        status="error",
                        message=f"Failed to open raster with GDAL: {local_input_path}"
                    )
                
                # Get band count
                band_count = ds.RasterCount
                logger.info(f"Input raster has {band_count} bands")
                
                # Check if band mapping is valid
                invalid_bands = []
                for band_name, band_number in band_mapping.items():
                    if band_number < 1 or band_number > band_count:
                        invalid_bands.append((band_name, band_number))
                
                if invalid_bands:
                    error_msg = f"Invalid band mapping: {invalid_bands}. Raster has {band_count} bands."
                    logger.error(error_msg)
                    return ProcessingResult(
                        status="error",
                        message=error_msg
                    )
                
                # Close the dataset
                ds = None
            except ImportError:
                logger.warning("GDAL not available for raster checking")
        
        except Exception as e:
            logger.error(f"Error checking input raster: {str(e)}", exc_info=True)
            return ProcessingResult(
                status="error",
                message=f"Error checking input raster: {str(e)}"
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
            
            # Check if QgsRasterCalculator is properly working
            calculator_working = True
            try:
                # Create a simple test calculation to verify QgsRasterCalculator works
                test_entry = QgsRasterCalculatorEntry()
                test_entry.ref = "test"
                test_entry.raster = raster_layer
                test_entry.bandNumber = 1
                
                test_output = str(output_dir / "test_calc.tif")
                test_calc = QgsRasterCalculator(
                    "test",
                    test_output,
                    "GTiff",
                    extent,
                    width,
                    height,
                    [test_entry]
                )
                
                # Just check if the object is properly initialized
                if not hasattr(test_calc, 'processCalculation') or not callable(getattr(test_calc, 'processCalculation')):
                    calculator_working = False
                    logger.warning("QgsRasterCalculator is not properly working")
            except Exception as e:
                calculator_working = False
                logger.warning(f"QgsRasterCalculator test failed: {str(e)}")
            
            # If QgsRasterCalculator is not working, try to use GDAL directly
            if not calculator_working:
                logger.info("Attempting to use GDAL directly for calculations")
                try:
                    # Import numpy here to ensure it's available in this scope
                    import numpy as np
                    from osgeo import gdal
                except ImportError:
                    try:
                        from osgeo import gdal
                        import numpy as np
                    except ImportError:
                        return ProcessingResult(
                            status="error",
                            message="Neither QgsRasterCalculator nor GDAL is available for calculations"
                        )
            
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
                
                calculation_success = False
                
                # Try using QgsRasterCalculator if it's working
                if calculator_working:
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
                    if result == 0:
                        calculation_success = True
                    else:
                        logger.warning(f"QgsRasterCalculator failed for {index_name} with error code {result}, trying GDAL")
                
                # If QgsRasterCalculator failed or is not available, try GDAL
                if not calculation_success:
                    try:
                        logger.info(f"Attempting GDAL calculation for {index_name}")
                        # Import numpy here to ensure it's available in this scope
                        import numpy as np
                        from osgeo import gdal
                        
                        # Open the dataset
                        ds = gdal.Open(local_input_path)
                        if ds is None:
                            logger.error(f"Failed to open raster with GDAL: {local_input_path}")
                            continue
                        
                        # Read band data into numpy arrays
                        band_arrays = {}
                        for band_name, band_number in band_mapping.items():
                            if band_name in required_bands:
                                try:
                                    # GDAL bands are 1-indexed
                                    band = ds.GetRasterBand(band_number)
                                    if band is None:
                                        logger.error(f"Band {band_number} not found in raster")
                                        continue
                                    
                                    # Read data and handle no data values
                                    data = band.ReadAsArray().astype(np.float32)
                                    logger.debug(f"Band {band_name} data shape: {data.shape}")
                                    
                                    nodata = band.GetNoDataValue()
                                    if nodata is not None:
                                        # Replace nodata with nan
                                        data = np.where(data == nodata, np.nan, data)
                                    
                                    band_arrays[band_name] = data
                                    logger.info(f"Successfully read band {band_name} (#{band_number})")
                                except Exception as e:
                                    logger.error(f"Error reading band {band_number}: {str(e)}")
                                    raise
                        
                        # Check if all required bands were read successfully
                        missing_bands = [band for band in required_bands if band not in band_arrays]
                        if missing_bands:
                            logger.error(f"Missing bands for GDAL calculation: {missing_bands}")
                            continue
                        
                        # Log band statistics for debugging
                        for band_name, array in band_arrays.items():
                            valid_data = array[~np.isnan(array)]
                            if len(valid_data) > 0:
                                logger.info(f"Band {band_name} stats: min={np.min(valid_data):.4f}, max={np.max(valid_data):.4f}, mean={np.mean(valid_data):.4f}")
                            else:
                                logger.warning(f"Band {band_name} has no valid data")
                        
                        # Create a safe environment for eval
                        safe_dict = {
                            'np': np,
                            'sqrt': np.sqrt,
                            'abs': np.abs,
                            'min': np.minimum,
                            'max': np.maximum,
                            'exp': np.exp,
                            'log': np.log,
                            'sin': np.sin,
                            'cos': np.cos,
                            'tan': np.tan
                        }
                        
                        # Add band arrays to the safe dict
                        for band_name, array in band_arrays.items():
                            safe_dict[band_name] = array
                        
                        # Convert QGIS expression to numpy expression
                        numpy_expr = expr.replace('^', '**')  # Replace power operator
                        logger.info(f"Evaluating expression: {numpy_expr}")
                        
                        try:
                            # Calculate the index
                            result_array = eval(numpy_expr, {"__builtins__": {}}, safe_dict)
                            
                            # Handle NaN and Inf values
                            result_array = np.nan_to_num(result_array, nan=0.0, posinf=1.0, neginf=-1.0)
                            
                            # Log result statistics
                            logger.info(f"Result stats: min={np.min(result_array):.4f}, max={np.max(result_array):.4f}, mean={np.mean(result_array):.4f}")
                            
                            # Create output raster
                            driver = gdal.GetDriverByName('GTiff')
                            out_ds = driver.Create(output_path, ds.RasterXSize, ds.RasterYSize, 1, gdal.GDT_Float32,
                                                  options=['COMPRESS=LZW', 'TILED=YES'])
                            
                            if out_ds is None:
                                logger.error(f"Failed to create output raster: {output_path}")
                                continue
                            
                            # Set projection and geotransform
                            out_ds.SetProjection(ds.GetProjection())
                            out_ds.SetGeoTransform(ds.GetGeoTransform())
                            
                            # Write data
                            out_band = out_ds.GetRasterBand(1)
                            out_band.WriteArray(result_array)
                            
                            # Set nodata value
                            out_band.SetNoDataValue(0.0)
                            
                            # Clean up
                            out_band = None
                            out_ds = None
                            
                            calculation_success = True
                            logger.info(f"Successfully calculated {index_name} using GDAL")
                        except Exception as e:
                            logger.error(f"Error evaluating expression: {str(e)}", exc_info=True)
                    except Exception as e:
                        logger.error(f"GDAL calculation failed for {index_name}: {str(e)}", exc_info=True)
                    finally:
                        # Make sure to close the dataset
                        if 'ds' in locals() and ds is not None:
                            ds = None
                
                if not calculation_success:
                    logger.error(f"All calculation methods failed for {index_name}")
                    continue
                
                # Use explicitly provided org_id and project_id if available
                # Otherwise, try to extract them from the input path
                extracted_org_id = None
                extracted_project_id = None
                
                # Parse the input path to extract org_id and project_id if not explicitly provided
                if input_path.startswith("s3://"):
                    # Format: s3://bucket/org-id/project-id/file.tif
                    parts = input_path.split('/')
                    if len(parts) >= 5:  # s3://bucket/org-id/project-id/file.tif
                        extracted_org_id = parts[3]
                        extracted_project_id = parts[4]
                elif '/' in input_path:
                    # Format: org-id/project-id/file.tif
                    parts = input_path.split('/')
                    if len(parts) >= 3:
                        extracted_org_id = parts[0]
                        extracted_project_id = parts[1]
                
                # Use explicitly provided values if available, otherwise use extracted values
                effective_org_id = org_id if org_id else extracted_org_id
                effective_project_id = project_id if project_id else extracted_project_id
                
                # Construct the S3 key with the health_indices subfolder
                if effective_org_id and effective_project_id:
                    s3_key = f"{effective_org_id}/{effective_project_id}/health_indices/{output_filename}"
                    logger.info(f"Saving to organization/project structure: {s3_key}")
                else:
                    s3_key = f"{output_dir_name}/{output_filename}"
                    logger.info(f"Saving to default location: {s3_key}")
                
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
            try:
                # Clear the project first
                if hasattr(self, 'project') and self.project is not None:
                    self.project.clear()
                
                # Exit QGIS
                self.qgis_app.exitQgis()
                self.qgis_app = None
                logger.info("QGIS shutdown completed")
            except Exception as e:
                logger.error(f"Error during QGIS shutdown: {str(e)}", exc_info=True) 