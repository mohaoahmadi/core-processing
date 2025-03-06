#!/usr/bin/env python
"""
Check if the functions exist in Supabase.
"""

from config import get_settings
from supabase import create_client
import uuid
import json

def main():
    """Main function."""
    # Get settings
    settings = get_settings()
    
    # Initialize Supabase client
    supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
    
    # Check if functions exist
    print("Checking if functions exist...")
    
    # Generate UUIDs for testing
    project_id = str(uuid.uuid4())
    job_id = str(uuid.uuid4())
    
    # Test register_geotiff_metadata function
    try:
        # Minimal metadata for testing
        metadata = {
            "file_size": 1000,
            "width": 100,
            "height": 100,
            "band_count": 3,
            "driver": "GTiff",
            "projection": "EPSG:4326",
            "geotransform": [0, 1, 0, 0, 0, 1],
            "bounds": {"minx": 0, "miny": 0, "maxx": 100, "maxy": 100},
            "bands": []
        }
        
        print("\nTesting register_geotiff_metadata function...")
        response = supabase.rpc(
            'register_geotiff_metadata',
            {
                'p_project_id': project_id,
                'p_file_name': 'test.tif',
                'p_s3_url': 's3://test/test.tif',
                'p_metadata': metadata
            }
        ).execute()
        
        print(f"register_geotiff_metadata function exists: {response}")
    except Exception as e:
        print(f"Error testing register_geotiff_metadata function: {str(e)}")
    
    # Test register_processed_raster function
    try:
        # Minimal metadata for testing
        metadata = {
            "width": 100,
            "height": 100,
            "band_count": 1,
            "driver": "GTiff",
            "projection": "EPSG:4326",
            "geotransform": [0, 1, 0, 0, 0, 1],
            "bounds": {"minx": 0, "miny": 0, "maxx": 100, "maxy": 100},
            "min_value": 0,
            "max_value": 1,
            "mean_value": 0.5,
            "metadata": {"index_name": "NDVI"}
        }
        
        print("\nTesting register_processed_raster function...")
        response = supabase.rpc(
            'register_processed_raster',
            {
                'p_processing_job_id': job_id,
                'p_raster_file_id': str(uuid.uuid4()),
                'p_output_type': 'NDVI',
                'p_s3_url': 's3://test/ndvi.tif',
                'p_metadata': metadata
            }
        ).execute()
        
        print(f"register_processed_raster function exists: {response}")
    except Exception as e:
        print(f"Error testing register_processed_raster function: {str(e)}")

if __name__ == "__main__":
    main() 