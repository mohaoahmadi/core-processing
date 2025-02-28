# Health Indices Processor

This module provides functionality for calculating various vegetation and health indices from multispectral imagery using PyQGIS. It supports a wide range of indices including NDVI, NDRE, EVI, SAVI, and others to assess vegetation health, water content, and other environmental parameters.

## Features

- Calculate over 20 different vegetation and health indices
- Support for different satellite sensors (WorldView-3, Sentinel-2, Landsat 8)
- Custom band mapping for any multispectral imagery
- Batch processing of multiple indices
- Integration with FastAPI application or standalone usage
- S3 storage support for input and output files

## Installation

### Prerequisites

- QGIS 3.x installed and available in your PATH
- Python 3.8+

### Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### As a FastAPI Processor

The `HealthIndicesProcessor` class can be used within the FastAPI application:

```python
from processors.health_indices import HealthIndicesProcessor

# Create processor instance
processor = HealthIndicesProcessor()

# Process an image
result = await processor(
    input_path="path/to/image.tif",
    output_dir="health_indices",
    indices=["NDVI", "NDRE", "EVI"],
    sensor_type="WV3"
)

# Access results
if result.status == "success":
    for index_name, index_info in result.metadata["indices"].items():
        print(f"{index_name}: {index_info['output_path']}")
```

### Command-line Usage

For standalone usage, you can use the provided command-line scripts:

```bash
# Using the integrated CLI tool
python utils/health_indices_cli.py --input path/to/image.tif --indices NDVI,NDRE,EVI --sensor WV3

# Using the standalone script
python generate_health_indices.py path/to/image.tif --indices NDVI,NDRE,EVI --output-dir indices
```

## Supported Indices

The processor supports the following indices:

| Index | Description | Range |
|-------|-------------|-------|
| NDVI | Normalized Difference Vegetation Index | -1 to 1 |
| NDRE | Normalized Difference Red Edge Index | -1 to 1 |
| NDWI | Normalized Difference Water Index | -1 to 1 |
| EVI | Enhanced Vegetation Index | -1 to 1 |
| SAVI | Soil Adjusted Vegetation Index | -1 to 1.5 |
| GNDVI | Green Normalized Difference Vegetation Index | -1 to 1 |
| NDYI | Normalized Difference Yellowness Index | -1 to 1 |
| VARI | Visual Atmospheric Resistance Index | -1 to 1 |
| ENDVI | Enhanced Normalized Difference Vegetation Index | -1 to 1 |
| GLI | Green Leaf Index | -1 to 1 |
| ARVI | Atmospherically Resistant Vegetation Index | -1 to 1 |
| OSAVI | Optimized Soil Adjusted Vegetation Index | -1 to 1 |
| LAI | Leaf Area Index | 0 to 10 |
| MNLI | Modified Non-Linear Index | -1 to 1.5 |
| MSR | Modified Simple Ratio | -1 to 1 |
| RDVI | Renormalized Difference Vegetation Index | -1 to 1 |
| TDVI | Transformed Difference Vegetation Index | -1 to 1.5 |
| BAI | Burn Area Index | 0 to 100 |
| EXG | Excess Green Index | -2 to 2 |
| GRVI | Green Ratio Vegetation Index | 0 to 10 |
| MPRI | Modified Photochemical Reflectance Index | -1 to 1 |
| vNDVI | Visible NDVI | 0 to 1 |

## Default Band Mappings

The processor includes default band mappings for common satellite sensors:

### WorldView-3 (WV3)
- B: 2 (Blue)
- G: 3 (Green)
- R: 5 (Red)
- Re: 6 (Red Edge)
- N: 7 (NIR)

### Sentinel-2 (S2)
- B: 2 (Blue)
- G: 3 (Green)
- R: 4 (Red)
- Re: 6 (Red Edge)
- N: 8 (NIR)

### Landsat 8 (L8)
- B: 2 (Blue)
- G: 3 (Green)
- R: 4 (Red)
- N: 5 (NIR)

## Custom Band Mapping

You can provide a custom band mapping for any multispectral imagery:

```python
# Using the processor
custom_mapping = {
    'B': 1,  # Blue band is band 1
    'G': 2,  # Green band is band 2
    'R': 3,  # Red band is band 3
    'N': 4,  # NIR band is band 4
    'Re': 5  # Red Edge band is band 5
}

result = await processor(
    input_path="path/to/image.tif",
    band_mapping=custom_mapping
)

# Using the command-line tool
python utils/health_indices_cli.py --input path/to/image.tif --band-mapping "B:1,G:2,R:3,N:4,Re:5"
```

## Error Handling

The processor includes comprehensive error handling:

- Validation of input parameters
- Checking for missing bands required by each index
- Handling of S3 paths and local files
- Proper cleanup of temporary files and QGIS resources

## Performance Considerations

- The processor initializes QGIS in headless mode, which requires some overhead
- For batch processing of multiple images, it's more efficient to create a single processor instance
- Processing large images may require significant memory and CPU resources

## License

This project is licensed under the MIT License - see the LICENSE file for details. 