# Integrating PyQGIS with Your Current Codebase Environment

Looking at your current codebase structure, I can see you have a FastAPI application with processors for various geospatial operations. Let me explain how to integrate PyQGIS with your existing environment.

## Understanding Your Current Environment

Your codebase appears to be a Python-based geospatial processing service with:
- FastAPI for the web API
- Processors for different operations (NDVI, land cover, orthomosaic)
- S3 integration for file storage
- A structured project layout with utilities and libraries

## Integration Options for PyQGIS

### Option 1: Use a Docker Container (Recommended)

This is the most reliable approach for production environments:

1. **Create a Dockerfile that includes QGIS**:

```dockerfile
FROM ubuntu:20.04

# Avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install QGIS and dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    qgis \
    python3-qgis \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set up QGIS environment variables
ENV PYTHONPATH=/usr/share/qgis/python
ENV LD_LIBRARY_PATH=/usr/lib/qgis

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Copy application code
COPY . .

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

2. **Build and run the Docker container**:

```bash
docker build -t core-processing .
docker run -p 8000:8000 core-processing
```

### Option 2: System-wide QGIS Installation

If you prefer not to use Docker:

1. **Install QGIS on your server**:
   ```bash
   # For Ubuntu/Debian
   sudo apt update
   sudo apt install qgis python3-qgis
   ```

2. **Create a wrapper script** to set up the environment before starting your FastAPI application:

```bash
#!/bin/bash
# start_app.sh

# Set up QGIS environment
export PYTHONPATH=/usr/share/qgis/python:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/lib/qgis:$LD_LIBRARY_PATH
export QT_QPA_PLATFORM=offscreen  # Important for headless servers

# Start the FastAPI application
uvicorn app:app --host 0.0.0.0 --port 8000
```

Make it executable:
```bash
chmod +x start_app.sh
```

3. **Use this script to start your application**:
```bash
./start_app.sh
```

### Option 3: Python Virtual Environment with QGIS

For development environments:

1. **Create a virtual environment**:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. **Create an activation hook** to set up QGIS environment variables:

For Linux/macOS, create `.venv/bin/activate.d/qgis.sh`:
```bash
#!/bin/bash
export PYTHONPATH=/usr/share/qgis/python:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/lib/qgis:$LD_LIBRARY_PATH
export QT_QPA_PLATFORM=offscreen
```

For Windows, create `.venv\Scripts\activate.d\qgis.ps1`:
```powershell
$env:PYTHONPATH = "C:\Program Files\QGIS 3.xx\python;" + $env:PYTHONPATH
$env:PATH = "C:\Program Files\QGIS 3.xx\bin;" + $env:PATH
$env:QT_QPA_PLATFORM = "offscreen"
```

3. **Install your dependencies**:
```bash
pip install -r requirements.txt
```

## Modifying Your Code for PyQGIS Compatibility

Your `health_indices.py` processor already has good error handling for when PyQGIS is not available. Here are some additional considerations:

1. **Add a health check** to verify PyQGIS availability at startup:

```python
# Add to app.py startup
try:
    from qgis.core import QgsApplication
    logger.info(f"PyQGIS is available (QGIS version: {QgsApplication.QGIS_VERSION})")
    QGIS_AVAILABLE = True
except ImportError:
    logger.warning("PyQGIS is not available. Health indices processing will be disabled.")
    QGIS_AVAILABLE = False
```

2. **Conditionally register the health indices processor**:

```python
# In app.py where you define processor_map
processor_map = {
    "landcover": LandCoverProcessor(),
    "ndvi": NDVIProcessor(),
    "orthomosaic": OrthomosaicProcessor(),
}

# Only add health indices processor if QGIS is available
if QGIS_AVAILABLE:
    processor_map["health_indices"] = HealthIndicesProcessor()
```

3. **Add a PyQGIS status endpoint**:

```python
@app.get(f"{settings.API_V1_PREFIX}/qgis-status")
async def qgis_status():
    """Check if PyQGIS is available and return its status."""
    try:
        from qgis.core import QgsApplication
        return {
            "available": True,
            "version": QgsApplication.QGIS_VERSION,
            "health_indices_enabled": True
        }
    except ImportError:
        return {
            "available": False,
            "version": None,
            "health_indices_enabled": False
        }
```

## Running Health Indices in Development

For local development:

1. **Install QGIS** on your development machine
2. **Set up environment variables** before running your FastAPI app:

```bash
# Linux/macOS
export PYTHONPATH=/usr/share/qgis/python:$PYTHONPATH
export QT_QPA_PLATFORM=offscreen
uvicorn app:app --reload

# Windows (PowerShell)
$env:PYTHONPATH = "C:\Program Files\QGIS 3.xx\python;" + $env:PYTHONPATH
$env:PATH = "C:\Program Files\QGIS 3.xx\bin;" + $env:PATH
$env:QT_QPA_PLATFORM = "offscreen"
uvicorn app:app --reload
```

## Testing the Integration

1. **Create a simple test script** to verify PyQGIS works with your codebase:

```python
# test_qgis_integration.py
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from qgis.core import QgsApplication
    print(f"PyQGIS is available (QGIS version: {QgsApplication.QGIS_VERSION})")
    
    # Initialize QGIS
    qgs = QgsApplication([], False)
    qgs.initQgis()
    print("QGIS initialized successfully")
    
    # Test importing your processor
    from processors.health_indices import HealthIndicesProcessor
    processor = HealthIndicesProcessor()
    print("HealthIndicesProcessor imported and instantiated successfully")
    
    # Clean up
    qgs.exitQgis()
    print("QGIS shutdown completed")
    
except ImportError as e:
    print(f"Error importing PyQGIS: {e}")
```

2. **Run the test script** with the appropriate environment variables:

```bash
# Set QGIS environment variables first
python test_qgis_integration.py
```

By following these steps, you should be able to integrate PyQGIS with your existing codebase and use the health indices processor in your FastAPI application.
