# Supabase Raster Storage

This document describes the Supabase schema and implementation for storing GeoTIFF and raster processing metadata in the remote sensing platform.

## Schema Overview

The schema extends the existing Supabase database with tables for storing:

1. **Raster Files** - Metadata about uploaded GeoTIFF files
2. **Raster Bands** - Detailed information about each band in a raster file
3. **Band Mappings** - Band mappings for different sensors and custom mappings
4. **Processed Rasters** - Metadata about processed raster outputs (e.g., NDVI, EVI)
5. **Health Indices** - Definitions of available health indices

## Tables

### Raster Files

Stores metadata about uploaded GeoTIFF files.

```sql
CREATE TABLE raster_files (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
    file_name VARCHAR(255) NOT NULL,
    s3_url VARCHAR(255) NOT NULL,
    file_size BIGINT NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    band_count INTEGER NOT NULL,
    driver VARCHAR(50) NOT NULL,
    projection TEXT,
    geotransform JSONB,
    bounds JSONB NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_by UUID REFERENCES auth.users(id)
);
```

### Raster Bands

Stores detailed information about each band in a raster file.

```sql
CREATE TABLE raster_bands (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    raster_file_id UUID REFERENCES raster_files(id) ON DELETE CASCADE,
    band_number INTEGER NOT NULL,
    data_type VARCHAR(50) NOT NULL,
    min_value DOUBLE PRECISION,
    max_value DOUBLE PRECISION,
    mean_value DOUBLE PRECISION,
    stddev_value DOUBLE PRECISION,
    nodata_value DOUBLE PRECISION,
    color_interpretation VARCHAR(50),
    wavelength DOUBLE PRECISION,
    wavelength_unit VARCHAR(20),
    band_name VARCHAR(50),
    metadata JSONB DEFAULT '{}'::jsonb,
    UNIQUE (raster_file_id, band_number)
);
```

### Band Mappings

Stores band mappings for different sensors and custom mappings.

```sql
CREATE TABLE band_mappings (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    mapping JSONB NOT NULL, -- e.g., {"R": 3, "G": 2, "B": 1, "N": 4}
    is_system BOOLEAN DEFAULT FALSE, -- True for system-defined mappings (e.g., for known sensors)
    project_id UUID REFERENCES projects(id) ON DELETE CASCADE, -- NULL for system-defined mappings
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_by UUID REFERENCES auth.users(id),
    UNIQUE (name, project_id) -- Allow same name across different projects
);
```

### Processed Rasters

Stores metadata about processed raster outputs.

```sql
CREATE TABLE processed_rasters (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    processing_job_id UUID REFERENCES processing_jobs(id) ON DELETE CASCADE,
    raster_file_id UUID REFERENCES raster_files(id),
    output_type VARCHAR(50) NOT NULL, -- 'ndvi', 'evi', 'landcover', etc.
    s3_url VARCHAR(255) NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    band_count INTEGER NOT NULL,
    driver VARCHAR(50) NOT NULL,
    projection TEXT,
    geotransform JSONB,
    bounds JSONB NOT NULL,
    min_value DOUBLE PRECISION,
    max_value DOUBLE PRECISION,
    mean_value DOUBLE PRECISION,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

### Health Indices

Stores information about available health indices.

```sql
CREATE TABLE health_indices (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    name VARCHAR(50) NOT NULL UNIQUE,
    formula VARCHAR(255) NOT NULL,
    description TEXT,
    min_value DOUBLE PRECISION,
    max_value DOUBLE PRECISION,
    required_bands VARCHAR[] NOT NULL,
    is_system BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_by UUID REFERENCES auth.users(id)
);
```

## Row Level Security (RLS)

All tables have Row Level Security (RLS) policies to ensure that users can only access data from their organizations and projects.

## Database Functions

### register_geotiff_metadata

Registers GeoTIFF metadata in the database.

```sql
CREATE OR REPLACE FUNCTION register_geotiff_metadata(
    p_project_id UUID,
    p_file_name TEXT,
    p_s3_url TEXT,
    p_metadata JSONB
)
RETURNS UUID
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
-- Function body
$$;
```

### register_processed_raster

Registers processed raster metadata in the database.

```sql
CREATE OR REPLACE FUNCTION register_processed_raster(
    p_processing_job_id UUID,
    p_raster_file_id UUID,
    p_output_type TEXT,
    p_s3_url TEXT,
    p_metadata JSONB
)
RETURNS UUID
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
-- Function body
$$;
```

## API Integration

The FastAPI application has been updated to integrate with the Supabase database for storing and retrieving raster metadata.

### Utility Module

The `utils/supabase_raster_manager.py` module provides functions for interacting with the Supabase database:

- `register_geotiff_metadata` - Register GeoTIFF metadata
- `register_processed_raster` - Register processed raster metadata
- `get_raster_file_metadata` - Get metadata for a raster file
- `get_processed_raster_metadata` - Get metadata for a processed raster
- `get_band_mappings` - Get band mappings
- `get_health_indices` - Get available health indices
- `create_custom_band_mapping` - Create a custom band mapping
- `get_project_raster_files` - Get all raster files for a project
- `get_project_processed_rasters` - Get all processed rasters for a project

### API Endpoints

The following API endpoints have been added:

- `GET /api/v1/raster-files/{project_id}` - List all raster files for a project
- `GET /api/v1/raster-files/{raster_file_id}/metadata` - Get metadata for a raster file
- `GET /api/v1/processed-rasters/{project_id}` - List all processed rasters for a project
- `GET /api/v1/processed-rasters/{processed_raster_id}/metadata` - Get metadata for a processed raster
- `GET /api/v1/band-mappings` - List band mappings
- `POST /api/v1/band-mappings` - Create a custom band mapping

## Usage

### Analyzing a GeoTIFF File

When a GeoTIFF file is analyzed, its metadata is stored in the Supabase database:

```python
# API request
response = requests.post(
    "https://api.example.com/api/v1/analyze-geotiff",
    json={
        "file_path": "sample.tif",
        "org_id": "org-123",
        "project_id": "proj-456"
    }
)

# Response includes the raster_file_id
raster_file_id = response.json()["raster_file_id"]
```

### Processing Health Indices

When health indices are processed, the results are stored in S3 and their metadata is stored in the Supabase database:

```python
# API request
response = requests.post(
    "https://api.example.com/api/v1/jobs",
    json={
        "process_type": "health_indices",
        "input_file": "sample.tif",
        "org_id": "org-123",
        "project_id": "proj-456",
        "parameters": {
            "indices": ["NDVI", "EVI"],
            "sensor_type": "Sentinel-2",
            "raster_file_id": "raster-789"  # Optional, can be omitted
        }
    }
)

# Get job status
job_id = response.json()["job_id"]
job_status = requests.get(f"https://api.example.com/api/v1/jobs/{job_id}")

# Job result includes S3 URLs and metadata IDs
result = job_status.json()["result"]
ndvi_url = result["files"]["NDVI"]
ndvi_metadata_id = result["files"]["NDVI_metadata_id"]
```

### Retrieving Raster Metadata

Raster metadata can be retrieved using the API:

```python
# Get all raster files for a project
raster_files = requests.get(
    f"https://api.example.com/api/v1/raster-files/proj-456"
)

# Get metadata for a specific raster file
raster_metadata = requests.get(
    f"https://api.example.com/api/v1/raster-files/raster-789/metadata"
)

# Get all processed rasters for a project
processed_rasters = requests.get(
    f"https://api.example.com/api/v1/processed-rasters/proj-456"
)

# Get metadata for a specific processed raster
processed_metadata = requests.get(
    f"https://api.example.com/api/v1/processed-rasters/processed-123/metadata"
)
```

## Benefits

1. **Centralized Metadata Storage** - All raster metadata is stored in a central database
2. **Efficient Querying** - Metadata can be queried efficiently without accessing the actual raster files
3. **Access Control** - Row Level Security ensures that users can only access data they have permission to
4. **Integration with Processing Jobs** - Processed rasters are linked to their processing jobs
5. **Band Mapping Management** - Band mappings can be stored and reused across projects 