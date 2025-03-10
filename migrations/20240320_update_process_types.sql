-- Update process types in the database
-- Add check constraint for valid process types

-- First, update any existing NDVI process types to health_indices
UPDATE processing_jobs
SET process_type = 'health_indices'
WHERE process_type = 'ndvi';

UPDATE raster_files
SET process_type = 'health_indices'
WHERE process_type = 'ndvi';

-- Create an enum type for process types if it doesn't exist
DO $$ BEGIN
    CREATE TYPE process_type_enum AS ENUM (
        'orthomosaic',
        'health_indices',
        'classification',
        'land_cover',
        'terrain_analysis'
    );
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- Add check constraints to ensure valid process types
ALTER TABLE processing_jobs
    DROP CONSTRAINT IF EXISTS valid_process_type;

ALTER TABLE processing_jobs
    ADD CONSTRAINT valid_process_type
    CHECK (process_type IN (
        'orthomosaic',
        'health_indices',
        'classification',
        'land_cover',
        'terrain_analysis'
    ));

ALTER TABLE raster_files
    DROP CONSTRAINT IF EXISTS valid_process_type;

ALTER TABLE raster_files
    ADD CONSTRAINT valid_process_type
    CHECK (process_type IN (
        'orthomosaic',
        'health_indices',
        'classification',
        'land_cover',
        'terrain_analysis'
    )); 