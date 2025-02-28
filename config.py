"""Configuration management module for the Core Processing API.

This module handles all configuration settings for the application using Pydantic's BaseSettings.
It supports loading configuration from environment variables and .env files.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional

class Settings(BaseSettings):
    """Application settings and configuration.
    
    This class defines all configuration parameters for the application.
    It uses Pydantic's BaseSettings for automatic environment variable loading
    and validation.

    Attributes:
        API_V1_PREFIX (str): API version prefix for all endpoints
        PROJECT_NAME (str): Name of the project
        DEBUG (bool): Debug mode flag
        SUPABASE_URL (str): Supabase instance URL
        SUPABASE_KEY (str): Supabase API key
        AWS_ACCESS_KEY_ID (str): AWS access key for S3
        AWS_SECRET_ACCESS_KEY (str): AWS secret key for S3
        AWS_DEFAULT_REGION (str): Default AWS region
        S3_BUCKET_NAME (str): S3 bucket for storing processed files
        GEOSERVER_URL (str): GeoServer instance URL
        GEOSERVER_USERNAME (str): GeoServer admin username
        GEOSERVER_PASSWORD (str): GeoServer admin password
        GEOSERVER_WORKSPACE (str): Default GeoServer workspace
        MAX_WORKERS (int): Maximum number of concurrent processing workers
        TEMP_DIR (str): Directory for temporary file storage
    """
    
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
    AWS_DEFAULT_REGION: str = "eu-west-1"
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
        """Pydantic configuration class.
        
        Specifies configuration for the settings class, including:
        - .env file support
        - Case sensitivity for environment variables
        """
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings.
    
    Returns:
        Settings: Application settings instance
        
    Note:
        Uses lru_cache to prevent multiple instantiations of Settings class
        and improve performance by caching the settings.
    """
    return Settings() 