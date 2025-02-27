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
    """Normalize array values to 0-1 range"""
    min_val = np.min(array)
    max_val = np.max(array)
    return (array - min_val) / (max_val - min_val + 1e-10)

def create_color_map(
    array: np.ndarray,
    class_map: Dict[int, str]
) -> Dict[str, List[int]]:
    """Create a color map for classified raster"""
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
    """Align multiple rasters to the same coordinate system and resolution"""
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
    """Blend overlapping areas using a feathering technique"""
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
    """Calculate Normalized Difference Water Index"""
    return (green_band - nir_band) / (green_band + nir_band + 1e-10)

def calculate_slope(
    dem: np.ndarray,
    resolution: float
) -> np.ndarray:
    """Calculate slope from DEM"""
    dy, dx = np.gradient(dem, resolution)
    slope = np.degrees(np.arctan(np.sqrt(dx*dx + dy*dy)))
    return slope 