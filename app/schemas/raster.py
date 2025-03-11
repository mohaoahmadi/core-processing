from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

class BandInfo(BaseModel):
    band_number: int
    data_type: str
    min_value: float
    max_value: float
    mean_value: float
    stddev_value: float
    nodata_value: Optional[float] = None
    color_interpretation: str
    wavelength: Optional[float] = None
    wavelength_unit: Optional[str] = None
    band_name: Optional[str] = None

class SpatialInfo(BaseModel):
    width: int
    height: int
    band_count: int
    driver: str
    projection: str
    bounds: Dict[str, float]
    geotransform: List[float]

class BasicInfo(BaseModel):
    id: str
    file_name: str
    file_size: int
    created_at: datetime
    updated_at: datetime
    type: str = Field(..., description="Either 'raw' or 'processed'")
    process_type: Optional[str] = None

class RasterDetails(BaseModel):
    basic_info: BasicInfo
    spatial_info: SpatialInfo
    bands: List[BandInfo]
    metadata: Dict[str, Any] 