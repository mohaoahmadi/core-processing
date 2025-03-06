#!/usr/bin/env python
"""
Set up test data in Supabase for testing the raster metadata storage.
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
    
    # Generate UUIDs for test data
    org_id = str(uuid.uuid4())
    project_id = str(uuid.uuid4())
    
    # Create test organization
    try:
        response = supabase.table('organizations').insert({
            'id': org_id,
            'name': 'Test Organization',
            'description': 'Test organization for raster metadata storage'
        }).execute()
        
        print(f"Created organization: {response}")
    except Exception as e:
        print(f"Error creating organization: {str(e)}")
        return
    
    # Create test project
    try:
        response = supabase.table('projects').insert({
            'id': project_id,
            'name': 'Test Project',
            'description': 'Test project for raster metadata storage',
            'organization_id': org_id
        }).execute()
        
        print(f"Created project: {response}")
    except Exception as e:
        print(f"Error creating project: {str(e)}")
        return
    
    # Create test processing job
    job_id = str(uuid.uuid4())
    
    try:
        response = supabase.table('processing_jobs').insert({
            'id': job_id,
            'project_id': project_id,
            'input_file': 'LZW_cog.tif',
            'process_type': 'health_indices',
            'status': 'COMPLETED',
            'parameters': {
                'indices': ['NDVI', 'NDRE', 'EVI', 'SAVI', 'NDWI', 'CNDVI'],
                'sensor_type': 'WV3',
                'band_mapping': {
                    'C': 1,
                    'B': 2,
                    'G': 3,
                    'Y': 4,
                    'R': 5,
                    'Re': 6,
                    'N': 7,
                    'SWIR1': 8
                }
            }
        }).execute()
        
        print(f"Created processing job: {response}")
    except Exception as e:
        print(f"Error creating processing job: {str(e)}")
        return
    
    print("\nTest data setup complete.")
    print(f"Organization ID: {org_id}")
    print(f"Project ID: {project_id}")
    print(f"Job ID: {job_id}")
    
    # Save IDs to a file for later use
    with open('test_data_ids.txt', 'w') as f:
        f.write(f"ORGANIZATION_ID={org_id}\n")
        f.write(f"PROJECT_ID={project_id}\n")
        f.write(f"JOB_ID={job_id}\n")
    
    print("\nIDs saved to test_data_ids.txt")
    
    # Now let's try to create a test raster file
    try:
        # Minimal metadata for testing
        metadata = {
            "file_size": 1000000000,  # 1GB
            "width": 10000,
            "height": 10000,
            "band_count": 8,
            "driver": "GTiff",
            "projection": "EPSG:4326",
            "geotransform": [0, 1, 0, 0, 0, 1],
            "bounds": {"minx": 0, "miny": 0, "maxx": 10000, "maxy": 10000},
            "bands": []
        }
        
        # Try to insert directly into the raster_files table
        print("\nTrying to insert directly into raster_files table...")
        raster_file_id = str(uuid.uuid4())
        
        response = supabase.table('raster_files').insert({
            'id': raster_file_id,
            'project_id': project_id,
            'file_name': 'LZW_cog.tif',
            's3_url': f's3://{settings.S3_BUCKET_NAME}/{org_id}/{project_id}/LZW_cog.tif',
            'file_size': metadata['file_size'],
            'width': metadata['width'],
            'height': metadata['height'],
            'band_count': metadata['band_count'],
            'driver': metadata['driver'],
            'projection': metadata['projection'],
            'geotransform': json.dumps(metadata['geotransform']),
            'bounds': json.dumps(metadata['bounds']),
            'metadata': json.dumps({})
        }).execute()
        
        print(f"Inserted raster file: {response}")
        
        # Now try to insert a processed raster
        print("\nTrying to insert directly into processed_rasters table...")
        processed_raster_id = str(uuid.uuid4())
        
        response = supabase.table('processed_rasters').insert({
            'id': processed_raster_id,
            'processing_job_id': job_id,
            'raster_file_id': raster_file_id,
            'output_type': 'NDVI',
            's3_url': f's3://{settings.S3_BUCKET_NAME}/{org_id}/{project_id}/processed/{job_id}/NDVI.tif',
            'width': metadata['width'],
            'height': metadata['height'],
            'band_count': 1,
            'driver': metadata['driver'],
            'projection': metadata['projection'],
            'geotransform': json.dumps(metadata['geotransform']),
            'bounds': json.dumps(metadata['bounds']),
            'min_value': -1,
            'max_value': 1,
            'mean_value': 0.5,
            'metadata': json.dumps({
                'index_name': 'NDVI',
                'sensor_type': 'WV3'
            })
        }).execute()
        
        print(f"Inserted processed raster: {response}")
        
        # Now let's try to use the register_geotiff_metadata function
        print("\nTrying to use the register_geotiff_metadata function...")
        
        response = supabase.rpc(
            'register_geotiff_metadata',
            {
                'p_project_id': project_id,
                'p_file_name': 'LZW_cog_2.tif',
                'p_s3_url': f's3://{settings.S3_BUCKET_NAME}/{org_id}/{project_id}/LZW_cog_2.tif',
                'p_metadata': metadata
            }
        ).execute()
        
        print(f"register_geotiff_metadata result: {response}")
        
    except Exception as e:
        print(f"Error inserting test data: {str(e)}")
    
    # Print instructions for running the curl command with the correct IDs
    print("\n\nTo run the curl command with the correct IDs, use:")
    print(f"""
curl -X POST http://localhost:8000/api/v1/jobs \\
  -H "Content-Type: application/json" \\
  -d '{{
    "process_type": "health_indices",
    "input_file": "LZW_cog.tif",
    "org_id": "{org_id}",
    "project_id": "{project_id}",
    "parameters": {{
      "indices": ["NDVI", "NDRE", "EVI", "SAVI", "NDWI", "CNDVI"],
      "sensor_type": "WV3",
      "band_mapping": {{
        "C": 1,
        "B": 2,
        "G": 3,
        "Y": 4,
        "R": 5,
        "Re": 6,
        "N": 7,
        "SWIR1": 8
      }}
    }}
  }}'
""")

if __name__ == "__main__":
    main() 