#!/usr/bin/env python
"""
Command-line utility for generating health indices from multispectral imagery.

This script provides a simple command-line interface to the HealthIndicesProcessor,
allowing users to generate various vegetation and health indices from multispectral
imagery using PyQGIS.

Example usage:
    python health_indices_cli.py --input input.tif --output-dir indices --indices NDVI,NDRE,EVI
"""

import os
import sys
import asyncio
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from processors.health_indices import HealthIndicesProcessor, HEALTH_INDICES
from config import get_settings

settings = get_settings()

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="Generate health indices from multispectral imagery using PyQGIS"
    )
    
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to input multispectral image (local file or S3 path)"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        default="health_indices",
        help="Directory name for output files (default: health_indices)"
    )
    
    parser.add_argument(
        "--indices", "-idx",
        help="Comma-separated list of indices to calculate (default: all available)"
    )
    
    parser.add_argument(
        "--sensor", "-s",
        default="WV3",
        choices=["WV3", "S2", "L8"],
        help="Sensor type for default band mapping (default: WV3)"
    )
    
    parser.add_argument(
        "--band-mapping", "-bm",
        help="Custom band mapping in format 'B:1,G:2,R:3,N:4,Re:5'"
    )
    
    parser.add_argument(
        "--list-indices",
        action="store_true",
        help="List all available indices and exit"
    )
    
    parser.add_argument(
        "--no-scale",
        action="store_true",
        help="Do not scale output to 8-bit (0-255)"
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

async def main() -> None:
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
    
    # Create processor
    processor = HealthIndicesProcessor()
    
    # Prepare parameters
    params = {
        "input_path": args.input,
        "output_dir": args.output_dir,
        "sensor_type": args.sensor,
        "scale_output": not args.no_scale
    }
    
    if indices:
        params["indices"] = indices
    
    if band_mapping:
        params["band_mapping"] = band_mapping
    
    # Run processing
    print(f"Processing {args.input} with sensor type {args.sensor}")
    if indices:
        print(f"Calculating indices: {', '.join(indices)}")
    else:
        print("Calculating all available indices")
    
    result = await processor(
        **params
    )
    
    # Print results
    if result.status == "success":
        print(f"\nSuccess! Generated {len(result.metadata['indices'])} indices:")
        for index_name, index_info in result.metadata["indices"].items():
            print(f"  - {index_name}: {index_info['output_path']}")
    else:
        print(f"\nError: {result.message}")

if __name__ == "__main__":
    asyncio.run(main()) 