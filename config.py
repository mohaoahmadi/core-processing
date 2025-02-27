from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional

class Settings(BaseSettings):
    # API Configuration
    API_V1_PREFIX: str = "/api/v1"
    PROJECT_NAME: str = "Core Processing API"
    DEBUG: bool = False

    # Supabase Configuration
    SUPABASE_URL: str
    SUPABASE_KEY: str

    # AWS S3 Configuration
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_DEFAULT_REGION: str = "us-east-1"
    S3_BUCKET_NAME: str

    # GeoServer Configuration
    GEOSERVER_URL: str
    GEOSERVER_USERNAME: str
    GEOSERVER_PASSWORD: str
    GEOSERVER_WORKSPACE: str = "core-processing"

    # Processing Configuration
    MAX_WORKERS: int = 4
    TEMP_DIR: str = "/tmp/core-processing"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    return Settings() 