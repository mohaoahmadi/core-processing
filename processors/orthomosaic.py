import numpy as np
import rasterio
from pathlib import Path
from typing import List, Dict, Any
import os
import tempfile

from processors.base import BaseProcessor, ProcessingResult
from lib.s3_manager import upload_file, download_file
from utils.geo_utils import align_rasters, blend_overlapping_areas
from config import get_settings

settings = get_settings()

class OrthomosaicProcessor(BaseProcessor):
    """Processor for generating orthomosaic from multiple images"""

    async def validate_input(self, **kwargs) -> bool:
        required = {
            "input_paths",
            "output_name",
            "method"
        }
        if not all(key in kwargs for key in required):
            return False
            
        valid_methods = {"average", "maximum", "minimum", "feather"}
        return kwargs["method"] in valid_methods

    async def process(self, **kwargs) -> ProcessingResult:
        input_paths: List[str] = kwargs["input_paths"]
        output_name: str = kwargs["output_name"]
        method: str = kwargs["method"]
        
        # Create output directory if it doesn't exist
        output_dir = Path(settings.TEMP_DIR) / "orthomosaic"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"{output_name}.tif"
        
        # Download input files if they are S3 paths
        local_input_paths = []
        temp_files = []
        
        for path in input_paths:
            if path.startswith("s3://"):
                # Extract the key from s3://bucket/key
                s3_key = path.split('/', 3)[3] if path.startswith("s3://") else path
                temp_file = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
                temp_path = temp_file.name
                temp_file.close()
                temp_files.append(temp_path)
                await download_file(s3_key, temp_path)
                local_input_paths.append(temp_path)
            else:
                local_input_paths.append(path)
        
        try:
            # Read and align all input rasters
            rasters = []
            profiles = []
            for path in local_input_paths:
                with rasterio.open(path) as src:
                    rasters.append(src.read())
                    profiles.append(src.profile.copy())
            
            # Align rasters to the same coordinate system and resolution
            aligned_rasters, aligned_profile = align_rasters(rasters, profiles)
            
            # Blend the aligned rasters
            if method == "average":
                mosaic = np.mean(aligned_rasters, axis=0)
            elif method == "maximum":
                mosaic = np.max(aligned_rasters, axis=0)
            elif method == "minimum":
                mosaic = np.min(aligned_rasters, axis=0)
            else:  # feather
                mosaic = blend_overlapping_areas(aligned_rasters)
            
            # Write the orthomosaic
            with rasterio.open(output_path, 'w', **aligned_profile) as dst:
                dst.write(mosaic.astype(aligned_profile['dtype']))
            
            # Upload to S3
            s3_key = f"orthomosaic/{output_name}.tif"
            s3_path = await upload_file(str(output_path), s3_key)
            
            # Calculate basic statistics
            stats = {
                "min": float(np.min(mosaic)),
                "max": float(np.max(mosaic)),
                "mean": float(np.mean(mosaic)),
                "std": float(np.std(mosaic))
            }
            
            return ProcessingResult(
                status="success",
                message="Orthomosaic generation completed",
                output_path=s3_path,
                metadata={
                    "input_count": len(input_paths),
                    "blend_method": method,
                    "statistics": stats,
                    "spatial_reference": aligned_profile.get("crs").to_string()
                    if aligned_profile.get("crs") else None
                }
            )
        finally:
            # Clean up temporary files
            for temp_path in temp_files:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

    async def cleanup(self) -> None:
        """Clean up temporary files"""
        output_dir = Path(settings.TEMP_DIR) / "orthomosaic"
        if output_dir.exists():
            for file in output_dir.glob("*.tif"):
                try:
                    file.unlink()
                except Exception:
                    pass