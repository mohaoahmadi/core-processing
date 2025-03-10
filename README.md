# Core Processing API

A FastAPI-based service for geospatial image processing, providing various analysis capabilities for remote sensing and GIS applications.

## Available Processing Types

The API supports the following processing types:

### 1. Orthomosaic Generation
- Creates seamless orthomosaic images from overlapping aerial/satellite imagery
- Supports various blending methods for optimal results
- Outputs georeferenced GeoTIFF files

### 2. Health Indices
- Calculates multiple vegetation and health indices
- Supports common indices like NDVI, EVI, SAVI
- Customizable band mapping for different sensors
- Outputs index maps as GeoTIFF files

### 3. Classification
- Performs image classification using machine learning
- Supports multiple classification models
- Outputs classified maps with confidence scores

### 4. Land Cover Analysis
- Analyzes and maps different land cover types
- Supports various classification schemes
- Outputs land cover maps with statistics

### 5. Terrain Analysis
- Processes Digital Elevation Models (DEM)
- Generates various terrain products:
  - Slope maps
  - Aspect maps
  - Hillshade visualizations
  - Contour lines
  - Watershed delineation
  - Terrain roughness index
  - Topographic position index

## API Endpoints

### Process Types
```
GET /api/v1/process-types
```
Returns available processing types with descriptions and required parameters.

### Job Creation
```
POST /api/v1/jobs
```
Creates a new processing job with the following parameters:
- `process_type`: Type of processing to perform
- `input_file`: Path to input file
- `parameters`: Processing-specific parameters
- `org_id`: Organization ID
- `project_id`: Project ID

### Job Status
```
GET /api/v1/jobs/{job_id}
```
Returns the status and results of a processing job.

## Dependencies

- FastAPI
- GDAL
- NumPy
- Supabase
- AWS S3
- GeoServer

## Installation

1. Install system dependencies:
```bash
apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    python3-gdal
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

The service requires the following environment variables:
- `SUPABASE_URL`: Supabase project URL
- `SUPABASE_KEY`: Supabase service role key
- `AWS_ACCESS_KEY_ID`: AWS access key
- `AWS_SECRET_ACCESS_KEY`: AWS secret key
- `S3_BUCKET`: S3 bucket name
- `GEOSERVER_URL`: GeoServer URL
- `GEOSERVER_USER`: GeoServer username
- `GEOSERVER_PASSWORD`: GeoServer password

## Development

1. Set up environment variables:
```bash
cp .env.example .env
```

2. Run the development server:
```bash
uvicorn app:app --reload
```

## Testing

Run tests with:
```bash
pytest tests/
```

## License

MIT License 