#!/usr/bin/env python
"""
Standalone script for generating health indices from multispectral imagery using PyQGIS.

This script provides a simple way to generate various vegetation and health indices
from multispectral imagery using PyQGIS without requiring the full FastAPI application.

Example usage:
    python generate_health_indices.py input.tif --indices NDVI,NDRE,EVI --output-dir indices
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

# Import PyQGIS modules
try:
    from qgis.core import (
        QgsApplication,
        QgsRasterLayer,
        QgsRasterCalculator,
        QgsRasterCalculatorEntry
    )
    QGIS_AVAILABLE = True
except ImportError:
    QGIS_AVAILABLE = False
    print("Warning: PyQGIS not available. Please install QGIS and ensure it's in your PATH.")
    sys.exit(1)

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

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="Generate health indices from multispectral imagery using PyQGIS"
    )
    
    parser.add_argument(
        "input_file",
        help="Path to input multispectral image"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        default="health_indices",
        help="Directory for output files (default: health_indices)"
    )
    
    parser.add_argument(
        "--indices", "-i",
        help="Comma-separated list of indices to calculate (default: all available)"
    )
    
    parser.add_argument(
        "--sensor", "-s",
        default="WV3",
        choices=["WV3", "S2", "L8"],
        help="Sensor type for default band mapping (default: WV3)"
    )
    
    parser.add_argument(
        "--band-mapping", "-b",
        help="Custom band mapping in format 'B:1,G:2,R:3,N:4,Re:5'"
    )
    
    parser.add_argument(
        "--list-indices", "-l",
        action="store_true",
        help="List all available indices and exit"
    )
    
    return parser.parse_args()

def parse_band_mapping(mapping_str: str) -> Dict[str, int]:
    """Parse band mapping string into a dictionary.
    
    Args:
        mapping_str: String in format 'B:1,G:2,R:3,N:4,Re:5'
        
    Returns:
        Dict[str, int]: Band mapping dictionary
    """
    if not mapping_str:
        return {}
        
    mapping = {}
    for pair in mapping_str.split(","):
        if ":" not in pair:
            continue
        band, number = pair.split(":")
        try:
            mapping[band.strip()] = int(number.strip())
        except ValueError:
            print(f"Warning: Invalid band number in '{pair}', skipping")
    
    return mapping

def list_available_indices() -> None:
    """Print a list of all available indices with descriptions."""
    print("\nAvailable Health Indices:")
    print("-" * 80)
    print(f"{'Index':<10} {'Range':<15} {'Description'}")
    print("-" * 80)
    
    for name, info in HEALTH_INDICES.items():
        range_str = f"{info.get('range', 'N/A')}"
        print(f"{name:<10} {range_str:<15} {info['help']}")

def run_calculation(
    input_path: str,
    output_dir: str,
    indices: Optional[List[str]] = None,
    sensor_type: str = "WV3",
    custom_band_mapping: Optional[Dict[str, int]] = None
) -> Dict[str, str]:
    """Run health indices calculations using PyQGIS.
    
    Args:
        input_path: Path to input multispectral image
        output_dir: Directory for output files
        indices: List of indices to calculate (default: all available)
        sensor_type: Sensor type for default band mapping
        custom_band_mapping: Custom band mapping
        
    Returns:
        Dict[str, str]: Dictionary mapping index names to output file paths
    """
    # Initialize QGIS in headless mode
    qgs = QgsApplication([], False)
    qgs.initQgis()
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Use custom band mapping if provided, otherwise use default for the sensor type
        band_mapping = custom_band_mapping if custom_band_mapping else DEFAULT_BAND_MAPPINGS.get(sensor_type, {})
        
        # Determine which indices to process
        indices_to_process = indices if indices else list(HEALTH_INDICES.keys())
        
        # Load the input raster
        raster_layer = QgsRasterLayer(input_path, "input_raster")
        if not raster_layer.isValid():
            raise ValueError("Failed to load the input raster")
        
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
                print(f"Warning: Skipping unsupported index: {index_name}")
                continue
            
            index_info = HEALTH_INDICES[index_name]
            expr = index_info["expr"]
            
            # Check if all required bands are available
            required_bands = index_info["bands"]
            missing_bands = [band for band in required_bands if band not in entries]
            
            if missing_bands:
                print(f"Warning: Skipping {index_name} due to missing bands: {missing_bands}")
                continue
            
            # Create output filename
            output_filename = f"{index_name.replace(' ', '_')}.tif"
            output_path = os.path.join(output_dir, output_filename)
            
            print(f"Calculating {index_name} -> {output_filename}")
            
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
                print(f"Error: Calculation for {index_name} failed with error code {result}")
                continue
            
            # Store result information
            results[index_name] = output_path
        
        return results
    
    finally:
        # Shutdown QGIS
        qgs.exitQgis()

def main() -> None:
    """Main entry point for the script."""
    args = parse_args()
    
    # List indices and exit if requested
    if args.list_indices:
        list_available_indices()
        return
    
    # Parse indices
    indices = None
    if args.indices:
        indices = [idx.strip() for idx in args.indices.split(",")]
    
    # Parse band mapping
    band_mapping = parse_band_mapping(args.band_mapping)
    
    print(f"Processing {args.input_file} with sensor type {args.sensor}")
    if indices:
        print(f"Calculating indices: {', '.join(indices)}")
    else:
        print("Calculating all available indices")
    
    try:
        # Run processing
        results = run_calculation(
            input_path=args.input_file,
            output_dir=args.output_dir,
            indices=indices,
            sensor_type=args.sensor,
            custom_band_mapping=band_mapping
        )
        
        # Print results
        if results:
            print(f"\nSuccess! Generated {len(results)} indices:")
            for index_name, output_path in results.items():
                print(f"  - {index_name}: {output_path}")
        else:
            print("\nError: No indices were successfully calculated")
    
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 