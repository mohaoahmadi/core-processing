#!/usr/bin/env python
"""
Analyze GeoTIFF file and suggest band mapping.

This script analyzes a GeoTIFF file and suggests a band mapping based on
band metadata. It can be used to determine the correct band mapping for
the health indices processor.

Usage:
    python analyze_geotiff.py path/to/sample.tif
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from osgeo import gdal
    gdal.UseExceptions()
except ImportError:
    logger.error("GDAL is required for this script. Please install it with 'pip install gdal' or 'pip install pygdal'.")
    sys.exit(1)

# Common band names and their wavelength ranges (in nanometers)
COMMON_BANDS = {
    'C': (400, 450),   # Coastal
    'B': (450, 520),   # Blue
    'G': (520, 600),   # Green
    'Y': (585, 625),   # Yellow
    'R': (630, 690),   # Red
    'Re': (690, 730),  # Red Edge
    'N': (760, 900),   # Near Infrared (NIR)
    'SWIR1': (1550, 1750),  # Short-wave Infrared 1
    'SWIR2': (2080, 2350)   # Short-wave Infrared 2
}

def analyze_geotiff(file_path: str) -> Dict:
    """Analyze a GeoTIFF file and extract band information.
    
    Args:
        file_path: Path to the GeoTIFF file
        
    Returns:
        Dictionary with raster information
    """
    try:
        # Open the dataset
        ds = gdal.Open(file_path)
        if ds is None:
            logger.error(f"Failed to open {file_path}")
            return {}
        
        # Get basic information
        width = ds.RasterXSize
        height = ds.RasterYSize
        band_count = ds.RasterCount
        projection = ds.GetProjection()
        geotransform = ds.GetGeoTransform()
        
        # Get driver information
        driver = ds.GetDriver().ShortName
        
        # Get band information
        bands = []
        for i in range(1, band_count + 1):
            band = ds.GetRasterBand(i)
            
            # Get band metadata
            metadata = band.GetMetadata_Dict()
            
            # Get band statistics
            stats = band.GetStatistics(True, True)
            
            # Get data type
            data_type = gdal.GetDataTypeName(band.DataType)
            
            # Get no data value
            nodata = band.GetNoDataValue()
            
            # Get color interpretation
            color_interp = gdal.GetColorInterpretationName(band.GetColorInterpretation())
            
            # Get wavelength if available
            wavelength = None
            if 'WAVELENGTH' in metadata:
                wavelength = metadata['WAVELENGTH']
            elif 'wavelength' in metadata:
                wavelength = metadata['wavelength']
            
            # Add band information
            bands.append({
                'band_number': i,
                'min': stats[0],
                'max': stats[1],
                'mean': stats[2],
                'stddev': stats[3],
                'data_type': data_type,
                'nodata': nodata,
                'color_interp': color_interp,
                'metadata': metadata,
                'wavelength': wavelength
            })
        
        # Close the dataset
        ds = None
        
        return {
            'width': width,
            'height': height,
            'band_count': band_count,
            'projection': projection,
            'geotransform': geotransform,
            'driver': driver,
            'bands': bands
        }
    
    except Exception as e:
        logger.error(f"Error analyzing GeoTIFF: {str(e)}")
        return {}

def suggest_band_mapping(raster_info: Dict) -> Dict[str, int]:
    """Suggest a band mapping based on raster information.
    
    Args:
        raster_info: Dictionary with raster information
        
    Returns:
        Dictionary mapping band names to band numbers
    """
    if not raster_info or 'bands' not in raster_info:
        return {}
    
    band_mapping = {}
    bands = raster_info['bands']
    
    # Try to determine band mapping from color interpretation
    for band in bands:
        color_interp = band['color_interp']
        band_number = band['band_number']
        
        if color_interp == 'Red':
            band_mapping['R'] = band_number
        elif color_interp == 'Green':
            band_mapping['G'] = band_number
        elif color_interp == 'Blue':
            band_mapping['B'] = band_number
        elif color_interp == 'NIR':
            band_mapping['N'] = band_number
        elif color_interp == 'Gray' and band_number == 1:
            # Often the first band is coastal
            band_mapping['C'] = band_number
    
    # If we couldn't determine all bands from color interpretation,
    # try to use wavelength information
    if len(band_mapping) < min(len(COMMON_BANDS), len(bands)):
        for band in bands:
            wavelength = band['wavelength']
            band_number = band['band_number']
            
            if wavelength is not None:
                try:
                    # Try to parse wavelength
                    if '-' in str(wavelength):
                        # Range format (e.g., "450-520")
                        wl_min, wl_max = map(float, wavelength.split('-'))
                    else:
                        # Single value format (e.g., "485")
                        wl_min = wl_max = float(wavelength)
                    
                    # Check which band it corresponds to
                    for band_name, (min_wl, max_wl) in COMMON_BANDS.items():
                        if min_wl <= wl_min <= max_wl or min_wl <= wl_max <= max_wl:
                            if band_name not in band_mapping:
                                band_mapping[band_name] = band_number
                                break
                except (ValueError, TypeError):
                    # Couldn't parse wavelength
                    pass
    
    # If we still don't have a complete mapping, make a guess based on band order
    # This is a common ordering for many satellite sensors including WorldView-3
    if len(band_mapping) < min(len(COMMON_BANDS), len(bands)):
        # WorldView-3 band order: Coastal, Blue, Green, Yellow, Red, Red Edge, NIR1, NIR2
        common_order = ['C', 'B', 'G', 'Y', 'R', 'Re', 'N', 'SWIR1', 'SWIR2']
        for i, band_name in enumerate(common_order):
            band_number = i + 1
            if band_number <= len(bands) and band_name not in band_mapping:
                band_mapping[band_name] = band_number
    
    return band_mapping

def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Analyze GeoTIFF file and suggest band mapping")
    parser.add_argument("input_path", help="Path to input GeoTIFF file")
    
    args = parser.parse_args()
    
    if not os.path.isfile(args.input_path):
        logger.error(f"Error: File not found: {args.input_path}")
        sys.exit(1)
    
    # Analyze the GeoTIFF file
    logger.info(f"Analyzing {args.input_path}...")
    raster_info = analyze_geotiff(args.input_path)
    
    if not raster_info:
        logger.error("Failed to analyze GeoTIFF file")
        sys.exit(1)
    
    # Print basic information
    logger.info(f"Raster size: {raster_info['width']} x {raster_info['height']}")
    logger.info(f"Number of bands: {raster_info['band_count']}")
    logger.info(f"Driver: {raster_info['driver']}")
    
    # Print band information
    logger.info("\nBand information:")
    for band in raster_info['bands']:
        logger.info(f"Band {band['band_number']}:")
        logger.info(f"  Data type: {band['data_type']}")
        logger.info(f"  Range: {band['min']} to {band['max']}")
        logger.info(f"  Mean: {band['mean']}, StdDev: {band['stddev']}")
        logger.info(f"  Color interpretation: {band['color_interp']}")
        if band['wavelength']:
            logger.info(f"  Wavelength: {band['wavelength']}")
    
    # Suggest band mapping
    band_mapping = suggest_band_mapping(raster_info)
    
    if band_mapping:
        logger.info("\nSuggested band mapping:")
        for band_name, band_number in band_mapping.items():
            logger.info(f"  {band_name} = {band_number}")
        
        # Generate command line for test_health_indices.py
        band_str = ",".join([f"{name}={num}" for name, num in band_mapping.items()])
        logger.info("\nTo use this mapping with test_health_indices.py, run:")
        logger.info(f"python test_health_indices.py {args.input_path} --bands \"{band_str}\"")
    else:
        logger.warning("Could not determine band mapping")

if __name__ == "__main__":
    main() 