#!/usr/bin/env python
"""
Test script for the health indices processor.

This script tests the health indices processor with a sample GeoTIFF file.
It calculates a few selected indices and prints the results.

Usage:
    python test_health_indices.py path/to/sample.tif
"""

import os
import sys
import asyncio
from pathlib import Path
from typing import Dict, Any, List

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from processors.health_indices import HealthIndicesProcessor, HEALTH_INDICES

async def test_health_indices(input_path: str) -> None:
    """Test the health indices processor with a sample file.
    
    Args:
        input_path: Path to a sample GeoTIFF file
    """
    print(f"Testing health indices processor with {input_path}")
    
    # Create processor
    processor = HealthIndicesProcessor()
    
    # Select a few indices to test
    test_indices = ["NDVI", "NDRE", "EVI", "SAVI", "NDWI"]
    print(f"Testing indices: {', '.join(test_indices)}")
    
    # Process the image
    result = await processor(
        input_path=input_path,
        output_dir="test_indices",
        indices=test_indices,
        sensor_type="WV3"  # Assuming WorldView-3 imagery
    )
    
    # Print results
    if result.status == "success":
        print("\nSuccess! Generated indices:")
        for index_name, index_info in result.metadata["indices"].items():
            print(f"  - {index_name}: {index_info['output_path']}")
            print(f"    Formula: {index_info['formula']}")
            print(f"    Description: {index_info['description']}")
            if index_info.get('value_range'):
                print(f"    Value Range: {index_info['value_range']}")
            print()
    else:
        print(f"\nError: {result.message}")

def main() -> None:
    """Main entry point for the script."""
    if len(sys.argv) < 2:
        print("Usage: python test_health_indices.py path/to/sample.tif")
        sys.exit(1)
    
    input_path = sys.argv[1]
    if not os.path.isfile(input_path):
        print(f"Error: File not found: {input_path}")
        sys.exit(1)
    
    asyncio.run(test_health_indices(input_path))

if __name__ == "__main__":
    main() 