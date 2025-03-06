#!/usr/bin/env python
"""
Check if the tables exist in Supabase.
"""

from config import get_settings
from supabase import create_client

def main():
    """Main function."""
    # Get settings
    settings = get_settings()
    
    # Initialize Supabase client
    supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
    
    # Check if tables exist
    tables = [
        'raster_files',
        'raster_bands',
        'band_mappings',
        'processed_rasters',
        'health_indices'
    ]
    
    print("Checking if tables exist...")
    
    for table in tables:
        try:
            response = supabase.table(table).select('*').limit(1).execute()
            print(f"{table}: {response}")
        except Exception as e:
            print(f"{table}: Error - {str(e)}")

if __name__ == "__main__":
    main() 