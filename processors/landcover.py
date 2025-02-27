import numpy as np
import rasterio
from typing import List, Dict, Any
import os
from pathlib import Path
import tempfile

from processors.base import BaseProcessor, ProcessingResult
from lib.s3_manager import upload_file, download_file
from utils.geo_utils import normalize_array, create_color_map
from config import get_settings

settings = get_settings()

class LandCoverProcessor(BaseProcessor):
    """Processor for land cover classification"""
    
    CLASSES = {
        0: "Unknown",
        1: "Water",
        2: "Vegetation",
        3: "Built-up",
        4: "Bare Soil"
    }

    async def validate_input(self, **kwargs) -> bool:
        required = {"input_path", "output_name"}
        return all(key in kwargs for key in required)

    async def process(self, **kwargs) -> ProcessingResult:
        input_path: str = kwargs["input_path"]
        output_name: str = kwargs["output_name"]
        
        # Create output directory if it doesn't exist
        output_dir = Path(settings.TEMP_DIR) / "landcover"
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
                # Read the input image
                image = src.read()
                profile = src.profile.copy()
                
                # Normalize the input image
                normalized = normalize_array(image)
                
                # Simple classification logic (example)
                # In practice, you would use a more sophisticated model
                classified = np.zeros_like(image[0])
                classified[(normalized[0] > 0.6) & (normalized[1] > 0.6)] = 2  # Vegetation
                classified[(normalized[0] < 0.2) & (normalized[1] < 0.2)] = 1  # Water
                classified[(normalized[0] > 0.5) & (normalized[1] < 0.3)] = 3  # Built-up
                classified[(normalized[0] > 0.4) & (normalized[1] > 0.4) & (normalized[2] > 0.4)] = 4  # Bare Soil
                
                # Update the profile for the output
                profile.update(
                    dtype=rasterio.uint8,
                    count=1,
                    nodata=0
                )
                
                # Write the classified image
                with rasterio.open(output_path, 'w', **profile) as dst:
                    dst.write(classified.astype(rasterio.uint8), 1)
                    
                # Create color map
                color_map = create_color_map(classified, self.CLASSES)
                
                # Upload to S3
                s3_key = f"landcover/{output_name}.tif"
                s3_path = await upload_file(str(output_path), s3_key)
                
                return ProcessingResult(
                    status="success",
                    message="Land cover classification completed",
                    output_path=s3_path,
                    metadata={
                        "classes": self.CLASSES,
                        "color_map": color_map,
                        "statistics": {
                            "pixel_counts": {
                                name: int(np.sum(classified == class_id))
                                for class_id, name in self.CLASSES.items()
                            }
                        }
                    }
                )
        finally:
            # Clean up temp file if created
            if temp_input and os.path.exists(input_path):
                os.unlink(input_path)

    async def cleanup(self) -> None:
        """Clean up temporary files"""
        output_dir = Path(settings.TEMP_DIR) / "landcover"
        if output_dir.exists():
            for file in output_dir.glob("*.tif"):
                try:
                    file.unlink()
                except Exception:
                    pass