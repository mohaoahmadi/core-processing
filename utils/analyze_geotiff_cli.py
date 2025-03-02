#!/usr/bin/env python
"""
Command-line utility for analyzing GeoTIFF files.

This script provides a simple command-line interface to analyze GeoTIFF files
using GDAL, returning metadata and band information.

Example usage:
    python analyze_geotiff_cli.py --input input.tif --output analysis.json
"""

import os
import sys
import json
import argparse
from pathlib import Path
from osgeo import gdal
import numpy as np

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="Analyze GeoTIFF files using GDAL"
    )
    
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to input GeoTIFF file (local file or S3 path)"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Path to output JSON file (default: print to stdout)"
    )
    
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the JSON output"
    )
    
    return parser.parse_args()

def analyze_geotiff(file_path: str) -> dict:
    """Analyze a GeoTIFF file and return its metadata and band information.
    
    Args:
        file_path (str): Path to the GeoTIFF file
        
    Returns:
        dict: Dictionary containing GeoTIFF metadata and band information
        
    Raises:
        Exception: If the file cannot be opened or analyzed
    """
    try:
        # Open the dataset with GDAL
        print(f"Opening GeoTIFF file: {file_path}")
        ds = gdal.Open(file_path)
        if ds is None:
            raise Exception(f"Failed to open GeoTIFF file: {file_path}")
        
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
    
    except Exception as e:
        print(f"Error analyzing GeoTIFF: {str(e)}", file=sys.stderr)
        raise

def main() -> None:
    """Main entry point for the script."""
    args = parse_args()
    
    try:
        # Analyze the GeoTIFF
        result = analyze_geotiff(args.input)
        
        # Format the result as JSON
        indent = 2 if args.pretty else None
        json_result = json.dumps(result, indent=indent)
        
        # Output the result
        if args.output:
            with open(args.output, 'w') as f:
                f.write(json_result)
            print(f"Analysis saved to: {args.output}")
        else:
            print(json_result)
            
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 