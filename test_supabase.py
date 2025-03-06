#!/usr/bin/env python
"""
Test script for Supabase functions.
"""

from config import get_settings
from supabase import create_client
import json
import uuid

def main():
    """Main function."""
    # Get settings
    settings = get_settings()
    
    # Initialize Supabase client
    supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
    
    # Check if tables exist
    print("Checking if tables exist...")
    
    try:
        response = supabase.table('raster_files').select('*').limit(1).execute()
        print(f"raster_files table exists: {response}")
    except Exception as e:
        print(f"Error checking raster_files table: {str(e)}")
    
    try:
        response = supabase.table('raster_bands').select('*').limit(1).execute()
        print(f"raster_bands table exists: {response}")
    except Exception as e:
        print(f"Error checking raster_bands table: {str(e)}")
    
    try:
        response = supabase.table('band_mappings').select('*').limit(1).execute()
        print(f"band_mappings table exists: {response}")
    except Exception as e:
        print(f"Error checking band_mappings table: {str(e)}")
    
    try:
        response = supabase.table('processed_rasters').select('*').limit(1).execute()
        print(f"processed_rasters table exists: {response}")
    except Exception as e:
        print(f"Error checking processed_rasters table: {str(e)}")
    
    try:
        response = supabase.table('health_indices').select('*').limit(1).execute()
        print(f"health_indices table exists: {response}")
    except Exception as e:
        print(f"Error checking health_indices table: {str(e)}")
    
    # Check if functions exist
    print("\nChecking if functions exist...")
    
    try:
        # Generate UUIDs for testing
        project_id = str(uuid.uuid4())
        
        # Test with minimal metadata
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
        
        # Save the raster_file_id for the next test
        raster_file_id = response.data
        print(f"Created raster_file with ID: {raster_file_id}")
        
    except Exception as e:
        print(f"Error checking register_geotiff_metadata function: {str(e)}")
        raster_file_id = str(uuid.uuid4())  # Fallback
    
    try:
        # Generate UUIDs for testing
        processing_job_id = str(uuid.uuid4())
        
        # Test with minimal metadata
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
        
        response = supabase.rpc(
            'register_processed_raster',
            {
                'p_processing_job_id': processing_job_id,
                'p_raster_file_id': raster_file_id,
                'p_output_type': 'NDVI',
                'p_s3_url': 's3://test/ndvi.tif',
                'p_metadata': metadata
            }
        ).execute()
        
        print(f"register_processed_raster function exists: {response}")
        print(f"Created processed_raster with ID: {response.data}")
        
    except Exception as e:
        print(f"Error checking register_processed_raster function: {str(e)}")

if __name__ == "__main__":
    main() 