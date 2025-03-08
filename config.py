"""Configuration module for the application.

This module provides configuration settings for the application using Pydantic.
"""

import os
from typing import Dict, Any, Optional
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    """Application settings and configuration."""
    
    # API Configuration
    API_V1_PREFIX: str = "/api/v1"
    PROJECT_NAME: str = "Core Processing API"
    DEBUG: bool = False
    
    # Supabase Configuration
    SUPABASE_URL: Optional[str] = None
    SUPABASE_KEY: Optional[str] = None
    SUPABASE_SERVICE_KEY: Optional[str] = None
    # AWS S3 Configuration
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_DEFAULT_REGION: str = "eu-west-1"
    S3_BUCKET_NAME: Optional[str] = None
    
    # GeoServer Configuration
    GEOSERVER_URL: Optional[str] = None
    GEOSERVER_USERNAME: Optional[str] = None
    GEOSERVER_PASSWORD: Optional[str] = None
    GEOSERVER_WORKSPACE: str = "core-processing"
    
    # Processing Configuration
    MAX_WORKERS: int = 4
    TEMP_DIR: str = "/tmp/core-processing"
    
    class Config:
        """Pydantic config."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    """Get application settings.
    
    Returns:
        Settings: Application settings
    """
    return Settings() 