"""NDVI calculation processor module.

This module implements the Normalized Difference Vegetation Index (NDVI)
calculation for satellite or aerial imagery. NDVI is a standardized index
that allows generation of an image showing relative biomass or vegetation
health.
"""

import numpy as np
import rasterio
from pathlib import Path
import tempfile
import os
from typing import Dict, Any

from processors.base import BaseProcessor, ProcessingResult
from lib.s3_manager import upload_file, download_file
from utils.geo_utils import normalize_array
from config import get_settings

settings = get_settings()

class NDVIProcessor(BaseProcessor):
    """Processor for calculating Normalized Difference Vegetation Index.
    
    This processor calculates NDVI using the formula:
    NDVI = (NIR - Red) / (NIR + Red)
    
    The result is scaled to 0-255 for visualization, where:
    - Higher values (brighter) indicate healthy vegetation
    - Lower values (darker) indicate non-vegetated areas
    - Values around 0 typically represent water or bare soil
    """

    async def validate_input(self, **kwargs) -> bool:
        """Validate input parameters for NDVI calculation.
        
        Args:
            **kwargs: Keyword arguments containing:
                input_path (str): Path to input image (local or S3)
                output_name (str): Base name for output files
                red_band (int): Band number for red channel
                nir_band (int): Band number for near-infrared channel
                
        Returns:
            bool: True if all required parameters are present and valid
        """
        required = {
            "input_path",
            "output_name",
            "red_band",
            "nir_band"
        }
        return all(key in kwargs for key in required)

    async def process(self, **kwargs) -> ProcessingResult:
        """Execute NDVI calculation.
        
        Args:
            **kwargs: Keyword arguments containing:
                input_path (str): Path to input image (local or S3)
                output_name (str): Base name for output files
                red_band (int): Band number for red channel
                nir_band (int): Band number for near-infrared channel
                
        Returns:
            ProcessingResult: Processing result containing:
                - Path to NDVI image in S3
                - Statistics (min, max, mean, std)
                - Band information used for calculation
                
        Note:
            The output is scaled to 8-bit (0-255) for visualization:
            - NDVI values of -1 to 1 are mapped to 0-255
            - The formula used is: output = (NDVI + 1) * 127.5
        """
        input_path: str = kwargs["input_path"]
        output_name: str = kwargs["output_name"]
        red_band: int = kwargs["red_band"]
        nir_band: int = kwargs["nir_band"]
        
        # Create output directory if it doesn't exist
        output_dir = Path(settings.TEMP_DIR) / "ndvi"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"{output_name}.tif"
        
        # If input is S3 path, download it first
        temp_input = None
        if input_path.startswith("s3://"):
            # Extract the key from s3://bucket/key
            s3_key = input_path.split('/', 3)[3] if input_path.startswith("s3://") else input_path
            temp_input = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
            temp_input_path = temp_input.name
            temp_input.close()
            await download_file(s3_key, temp_input_path)
            input_path = temp_input_path
        
        try:
            with rasterio.open(input_path) as src:
                # Read the required bands
                red = src.read(red_band).astype(float)
                nir = src.read(nir_band).astype(float)
                
                # Avoid division by zero
                denominator = nir + red
                ndvi = np.where(
                    denominator > 0,
                    (nir - red) / denominator,
                    0
                )
                
                # Scale NDVI to 0-255 for visualization
                ndvi_scaled = ((ndvi + 1) * 127.5).astype(np.uint8)
                
                # Update the profile for the output
                profile = src.profile.copy()
                profile.update(
                    dtype=rasterio.uint8,
                    count=1,
                    nodata=0
                )
                
                # Write the NDVI image
                with rasterio.open(output_path, 'w', **profile) as dst:
                    dst.write(ndvi_scaled, 1)
                
                # Calculate statistics
                valid_pixels = ndvi[denominator > 0]
                stats = {
                    "min": float(np.min(valid_pixels)) if len(valid_pixels) > 0 else 0,
                    "max": float(np.max(valid_pixels)) if len(valid_pixels) > 0 else 0,
                    "mean": float(np.mean(valid_pixels)) if len(valid_pixels) > 0 else 0,
                    "std": float(np.std(valid_pixels)) if len(valid_pixels) > 0 else 0
                }
                
                # Upload to S3
                s3_key = f"ndvi/{output_name}.tif"
                s3_path = await upload_file(str(output_path), s3_key)
                
                return ProcessingResult(
                    status="success",
                    message="NDVI calculation completed",
                    output_path=s3_path,
                    metadata={
                        "statistics": stats,
                        "bands_used": {
                            "red": red_band,
                            "nir": nir_band
                        }
                    }
                )
        finally:
            # Clean up temp file if created
            if temp_input and os.path.exists(input_path):
                os.unlink(input_path)

    async def cleanup(self) -> None:
        """Clean up temporary files.
        
        Removes all temporary files created during processing from the
        NDVI output directory.
        """
        output_dir = Path(settings.TEMP_DIR) / "ndvi"
        if output_dir.exists():
            for file in output_dir.glob("*.tif"):
                try:
                    file.unlink()
                except Exception:
                    pass