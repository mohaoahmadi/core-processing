#!/usr/bin/env python
"""
Test script for the health indices processor.

This script tests the health indices processor with a sample GeoTIFF file.
It calculates a few selected indices and prints the results.

Usage:
    python test_health_indices.py path/to/sample.tif [--bands N=1,R=2,G=3,B=4]
"""

import os
import sys
import asyncio
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Check for GDAL availability
try:
    try:
        import gdal
        logger.info("GDAL is available (direct import)")
    except ImportError:
        try:
            from osgeo import gdal
            logger.info("GDAL is available (from osgeo)")
        except ImportError:
            logger.warning("GDAL is not available. This may limit functionality.")
except Exception as e:
    logger.warning(f"Error checking GDAL: {str(e)}")

# Check for QGIS availability
try:
    import qgis
    logger.info(f"QGIS is available: {qgis.__file__}")
    
    # Check QGIS version
    try:
        from qgis.core import Qgis
        logger.info(f"QGIS version: {Qgis.QGIS_VERSION}")
    except ImportError:
        logger.warning("Could not determine QGIS version")
    
    # Check for QgsRasterCalculator
    try:
        from qgis.core import QgsRasterCalculator
        logger.info("QgsRasterCalculator is available in qgis.core")
    except ImportError:
        try:
            from qgis.analysis import QgsRasterCalculator
            logger.info("QgsRasterCalculator is available in qgis.analysis")
        except ImportError:
            logger.warning("QgsRasterCalculator is not available")
except ImportError:
    logger.warning("QGIS is not available")
except Exception as e:
    logger.warning(f"Error checking QGIS: {str(e)}")

from processors.health_indices import HealthIndicesProcessor, HEALTH_INDICES

def parse_band_mapping(band_str: str) -> Dict[str, int]:
    """Parse band mapping string into a dictionary.
    
    Args:
        band_str: String in format "N=1,R=2,G=3,B=4"
        
    Returns:
        Dictionary mapping band names to band numbers
    """
    if not band_str:
        return {}
        
    band_mapping = {}
    for pair in band_str.split(','):
        if '=' in pair:
            name, number = pair.split('=')
            try:
                band_mapping[name.strip()] = int(number.strip())
            except ValueError:
                logger.warning(f"Invalid band number: {number}")
    
    return band_mapping

async def test_health_indices(input_path: str, band_mapping: Dict[str, int] = None) -> None:
    """Test the health indices processor with a sample file.
    
    Args:
        input_path: Path to a sample GeoTIFF file
        band_mapping: Optional custom band mapping
    """
    logger.info(f"Testing health indices processor with {input_path}")
    
    # Create processor
    processor = HealthIndicesProcessor()
    
    # Select a few indices to test
    test_indices = ["NDVI", "NDRE", "EVI", "SAVI", "NDWI", "CNDVI"]
    logger.info(f"Testing indices: {', '.join(test_indices)}")
    
    if band_mapping:
        logger.info(f"Using custom band mapping: {band_mapping}")
    
    try:
        # Process the image
        result = await processor(
            input_path=input_path,
            output_dir="test_indices",
            indices=test_indices,
            sensor_type="WV3",  # Assuming WorldView-3 imagery
            band_mapping=band_mapping
        )
        
        # Print results
        if result.status == "success":
            logger.info("\nSuccess! Generated indices:")
            for index_name, index_info in result.metadata["indices"].items():
                logger.info(f"  - {index_name}: {index_info['output_path']}")
                logger.info(f"    Formula: {index_info['formula']}")
                logger.info(f"    Description: {index_info['description']}")
                if index_info.get('value_range'):
                    logger.info(f"    Value Range: {index_info['value_range']}")
        else:
            logger.error(f"\nError: {result.message}")
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}", exc_info=True)
    finally:
        # Clean up
        if hasattr(processor, 'cleanup'):
            await processor.cleanup()

def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Test health indices processor")
    parser.add_argument("input_path", help="Path to input GeoTIFF file")
    parser.add_argument("--bands", help="Custom band mapping (e.g., 'N=1,R=2,G=3,B=4')")
    
    args = parser.parse_args()
    
    if not os.path.isfile(args.input_path):
        logger.error(f"Error: File not found: {args.input_path}")
        sys.exit(1)
    
    # Parse band mapping if provided
    band_mapping = parse_band_mapping(args.bands) if args.bands else None
    
    asyncio.run(test_health_indices(args.input_path, band_mapping))

if __name__ == "__main__":
    main() 