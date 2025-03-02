"""Geospatial processing utility module.

This module provides utility functions for processing geospatial raster data,
including array normalization, color mapping, raster alignment, and various
index calculations (NDVI, NDWI, slope).
"""

import numpy as np
from typing import List, Tuple, Dict, Any
import rasterio
from rasterio.warp import calculate_default_transform, reproject
from rasterio.merge import merge
from rasterio.enums import Resampling
from rasterio.windows import Window
import pyproj
from shapely.geometry import box, Polygon
from shapely.ops import transform

def normalize_array(array: np.ndarray) -> np.ndarray:
    """Normalize array values to the range [0, 1].
    
    Args:
        array (np.ndarray): Input array to normalize
        
    Returns:
        np.ndarray: Normalized array
        
    Note:
        Uses min-max normalization: (x - min) / (max - min)
        A small epsilon (1e-10) is added to the denominator to avoid
        division by zero in case of constant arrays.
    """
    min_val = np.min(array)
    max_val = np.max(array)
    return (array - min_val) / (max_val - min_val + 1e-10)

def create_color_map(
    array: np.ndarray,
    class_map: Dict[int, str]
) -> Dict[str, List[int]]:
    """Create a color mapping for classified raster visualization.
    
    Args:
        array (np.ndarray): Classified raster array
        class_map (Dict[int, str]): Mapping of class IDs to names
        
    Returns:
        Dict[str, List[int]]: Mapping of class names to RGB colors
        
    Note:
        Predefined colors are used for common classes:
        - Unknown: [0, 0, 0] (black)
        - Water: [0, 0, 255] (blue)
        - Vegetation: [0, 255, 0] (green)
        - Built-up: [255, 0, 0] (red)
        - Bare Soil: [139, 69, 19] (brown)
        Other classes default to [128, 128, 128] (gray)
    """
    colors = {
        "Unknown": [0, 0, 0],
        "Water": [0, 0, 255],
        "Vegetation": [0, 255, 0],
        "Built-up": [255, 0, 0],
        "Bare Soil": [139, 69, 19]
    }
    return {class_map[k]: colors.get(class_map[k], [128, 128, 128]) for k in class_map}

def align_rasters(
    rasters: List[np.ndarray],
    profiles: List[Dict[str, Any]]
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Align multiple rasters to a common coordinate system and resolution.
    
    Args:
        rasters (List[np.ndarray]): List of raster arrays to align
        profiles (List[Dict[str, Any]]): List of raster profiles with metadata
        
    Returns:
        Tuple[np.ndarray, Dict[str, Any]]: Tuple containing:
            - Stack of aligned raster arrays
            - Updated profile for the aligned rasters
            
    Raises:
        ValueError: If rasters or profiles lists are empty
        
    Note:
        The alignment process:
        1. Uses the first raster's CRS as reference
        2. Finds the common extent of all rasters
        3. Calculates optimal resolution
        4. Reprojects all rasters to match
    """
    if not rasters or not profiles:
        raise ValueError("Empty rasters or profiles list")
    
    # Use the first raster's profile as reference
    ref_profile = profiles[0].copy()
    
    # Find the common extent
    bounds = None
    for profile in profiles:
        if bounds is None:
            bounds = box(*profile['transform'].bounds)
        else:
            bounds = bounds.intersection(box(*profile['transform'].bounds))
    
    # Calculate the optimal resolution
    resolutions = [profile['transform'].a for profile in profiles]
    target_res = np.mean(resolutions)
    
    # Update the reference profile
    ref_profile.update({
        'height': int(bounds.bounds[3] - bounds.bounds[1]) / target_res,
        'width': int(bounds.bounds[2] - bounds.bounds[0]) / target_res,
        'transform': rasterio.transform.from_bounds(
            *bounds.bounds,
            width=ref_profile['width'],
            height=ref_profile['height']
        )
    })
    
    # Reproject and align all rasters
    aligned = []
    for raster, profile in zip(rasters, profiles):
        if profile == ref_profile:
            aligned.append(raster)
            continue
            
        # Reproject the raster
        dst_shape = (ref_profile['height'], ref_profile['width'])
        dst_transform = ref_profile['transform']
        
        reproject_kwargs = {
            'src_transform': profile['transform'],
            'src_crs': profile['crs'],
            'dst_transform': dst_transform,
            'dst_crs': ref_profile['crs'],
            'resampling': Resampling.bilinear
        }
        
        dst_array = np.zeros(dst_shape)
        reproject(
            source=raster,
            destination=dst_array,
            **reproject_kwargs
        )
        aligned.append(dst_array)
    
    return np.stack(aligned), ref_profile

def blend_overlapping_areas(rasters: np.ndarray) -> np.ndarray:
    """Blend overlapping areas in multiple rasters using distance-based weights.
    
    Args:
        rasters (np.ndarray): 3D array of rasters (n_rasters, height, width)
        
    Returns:
        np.ndarray: Blended raster array
        
    Raises:
        ValueError: If input is not a 3D array
        
    Note:
        The blending process:
        1. Calculates distance from edges for each raster
        2. Uses these distances as weights for blending
        3. Normalizes weights to sum to 1
        4. Applies weighted average blending
    """
    if len(rasters.shape) != 3:
        raise ValueError("Expected 3D array (n_rasters, height, width)")
    
    n_rasters, height, width = rasters.shape
    if n_rasters < 2:
        return rasters[0]
    
    # Create distance-based weights for each raster
    weights = np.ones((n_rasters, height, width))
    
    # Calculate distance from edges for each raster
    for i in range(n_rasters):
        mask = rasters[i] > 0
        dist = np.zeros((height, width))
        for y in range(height):
            for x in range(width):
                if mask[y, x]:
                    # Simple distance from edges
                    edge_dist = min(x, width-x, y, height-y)
                    dist[y, x] = edge_dist
        weights[i] = dist
    
    # Normalize weights
    weights_sum = np.sum(weights, axis=0)
    weights_sum[weights_sum == 0] = 1
    weights /= weights_sum
    
    # Apply weighted blend
    result = np.sum(rasters * weights, axis=0)
    
    return result

def calculate_ndwi(
    green_band: np.ndarray,
    nir_band: np.ndarray
) -> np.ndarray:
    """Calculate Normalized Difference Water Index (NDWI).
    
    Args:
        green_band (np.ndarray): Green band array
        nir_band (np.ndarray): Near-infrared band array
        
    Returns:
        np.ndarray: NDWI array with values in [-1, 1]
        
    Note:
        NDWI = (Green - NIR) / (Green + NIR)
        Used for water body mapping and moisture content assessment.
        Positive values typically indicate water bodies.
    """
    return (green_band - nir_band) / (green_band + nir_band + 1e-10)

def calculate_slope(
    dem: np.ndarray,
    resolution: float
) -> np.ndarray:
    """Calculate slope from a Digital Elevation Model (DEM).
    
    Args:
        dem (np.ndarray): Digital Elevation Model array
        resolution (float): Spatial resolution of the DEM
        
    Returns:
        np.ndarray: Slope array in degrees
        
    Note:
        Uses the gradient method to calculate slope:
        1. Calculates elevation gradients in x and y directions
        2. Converts gradients to slope in degrees
        3. Returns values in range [0, 90]
    """
    dy, dx = np.gradient(dem, resolution)
    slope = np.degrees(np.arctan(np.sqrt(dx*dx + dy*dy)))
    return slope

async def analyze_geotiff(file_path: str) -> dict:
    """Analyze a GeoTIFF file and return its metadata and band information.
    
    Args:
        file_path (str): Path to the GeoTIFF file (local or S3 path)
        
    Returns:
        dict: Dictionary containing GeoTIFF metadata and band information
        
    Raises:
        Exception: If the file cannot be opened or analyzed
    """
    from osgeo import gdal
    import numpy as np
    import os
    from pathlib import Path
    import tempfile
    from loguru import logger
    
    # Import here to avoid circular imports
    from lib.s3_manager import download_file
    from config import get_settings
    
    settings = get_settings()
    local_file_path = file_path
    temp_file = None
    
    try:
        # Handle S3 paths
        if file_path.startswith("s3://") or not os.path.isfile(file_path):
            logger.info(f"Input is from S3 or not a local file: {file_path}")
            # Extract the key from s3://bucket/key or use as is
            s3_key = file_path.split('/', 3)[3] if file_path.startswith("s3://") else file_path
            logger.debug(f"Extracted S3 key: {s3_key}")
            
            temp_file = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
            local_file_path = temp_file.name
            temp_file.close()
            logger.debug(f"Created temporary file for download: {local_file_path}")
            
            try:
                logger.info(f"Downloading from S3: {s3_key}")
                await download_file(s3_key, local_file_path)
                logger.info("S3 file downloaded successfully")
            except Exception as e:
                logger.error(f"Failed to download from S3: {str(e)}")
                raise Exception(f"Failed to download file from S3: {str(e)}")
        
        # Open the dataset with GDAL
        logger.info(f"Opening GeoTIFF file: {local_file_path}")
        ds = gdal.Open(local_file_path)
        if ds is None:
            raise Exception(f"Failed to open GeoTIFF file: {local_file_path}")
        
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
        
        return result
    
    finally:
        # Clean up temporary file if it exists
        if temp_file is not None and os.path.exists(local_file_path):
            try:
                os.unlink(local_file_path)
                logger.debug(f"Deleted temporary file: {local_file_path}")
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {local_file_path}: {str(e)}") 