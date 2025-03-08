"""
Supabase Raster Manager

This module provides utilities for storing and retrieving raster metadata in Supabase.
It handles the interaction with the Supabase database for GeoTIFF files and processed rasters.
"""

import os
import json
from typing import Dict, List, Any, Optional, Union
import asyncpg
from loguru import logger

# Import the singleton Supabase client
from lib.supabase_client import get_supabase

async def register_geotiff_metadata(
    project_id: str,
    file_name: str,
    s3_url: str,
    metadata: Dict[str, Any]
) -> Optional[str]:
    """
    Register GeoTIFF metadata in Supabase.
    
    Args:
        project_id: UUID of the project
        file_name: Name of the GeoTIFF file
        s3_url: S3 URL where the file is stored
        metadata: Dictionary containing GeoTIFF metadata
        
    Returns:
        UUID of the created raster_file record or None if failed
    """
    try:
        # Add file_size if not present
        if 'file_size' not in metadata:
            metadata['file_size'] = 0  # Default value
            
        # Get the singleton Supabase client
        supabase = get_supabase()
        
        # Call the Supabase function to register metadata
        response = supabase.rpc(
            'register_geotiff_metadata',
            {
                'p_project_id': project_id,
                'p_file_name': file_name,
                'p_s3_url': s3_url,
                'p_metadata': metadata
            }
        ).execute()
        
        if hasattr(response, 'error') and response.error:
            logger.error(f"Error registering GeoTIFF metadata: {response.error}")
            return None
            
        # Return the UUID of the created raster_file
        return response.data
        
    except Exception as e:
        logger.error(f"Failed to register GeoTIFF metadata: {str(e)}")
        return None

async def register_processed_raster(
    processing_job_id: str,
    raster_file_id: str,
    output_type: str,
    s3_url: str,
    metadata: Dict[str, Any]
) -> Optional[str]:
    """
    Register processed raster metadata in Supabase.
    
    Args:
        processing_job_id: UUID of the processing job
        raster_file_id: UUID of the source raster file
        output_type: Type of the output (e.g., 'ndvi', 'evi')
        s3_url: S3 URL where the processed file is stored
        metadata: Dictionary containing processed raster metadata
        
    Returns:
        UUID of the created processed_raster record
        
    Raises:
        Exception: If there is an error registering the processed raster
    """
    try:
        # Get the singleton Supabase client
        supabase = get_supabase()
        
        # Call the Supabase function to register processed raster
        response = supabase.rpc(
            'register_processed_raster',
            {
                'p_processing_job_id': processing_job_id,
                'p_raster_file_id': raster_file_id,
                'p_output_type': output_type,
                'p_s3_url': s3_url,
                'p_metadata': metadata
            }
        ).execute()
        
        if hasattr(response, 'error') and response.error:
            raise Exception(f"Error registering processed raster: {response.error}")
            
        # Return the UUID of the created processed_raster
        return response.data
        
    except Exception as e:
        logger.error(f"Failed to register processed raster: {str(e)}")
        raise

async def get_raster_file_metadata(raster_file_id: str) -> Optional[Dict[str, Any]]:
    """
    Get metadata for a raster file.
    
    Args:
        raster_file_id: UUID of the raster file
        
    Returns:
        Dictionary containing raster file metadata or None if not found
    """
    try:
        # Get the singleton Supabase client
        supabase = get_supabase()
        
        # Query the raster_files table
        response = supabase.table('raster_files').select('*').eq('id', raster_file_id).execute()
        
        if hasattr(response, 'error') and response.error:
            logger.error(f"Error getting raster file metadata: {response.error}")
            return None
            
        if not response.data:
            logger.warning(f"Raster file not found: {raster_file_id}")
            return None
            
        raster_file = response.data[0]
        
        # Get band information
        bands_response = supabase.table('raster_bands').select('*').eq('raster_file_id', raster_file_id).execute()
        
        if hasattr(bands_response, 'error') and bands_response.error:
            logger.error(f"Error getting raster bands: {bands_response.error}")
        else:
            raster_file['bands'] = bands_response.data
            
        return raster_file
        
    except Exception as e:
        logger.error(f"Failed to get raster file metadata: {str(e)}")
        return None

async def get_processed_raster_metadata(processed_raster_id: str) -> Optional[Dict[str, Any]]:
    """
    Get metadata for a processed raster.
    
    Args:
        processed_raster_id: UUID of the processed raster
        
    Returns:
        Dictionary containing processed raster metadata or None if not found
    """
    try:
        # Get the singleton Supabase client
        supabase = get_supabase()
        
        # Query the processed_rasters table
        response = supabase.table('processed_rasters').select('*').eq('id', processed_raster_id).execute()
        
        if hasattr(response, 'error') and response.error:
            logger.error(f"Error getting processed raster metadata: {response.error}")
            return None
            
        if not response.data:
            logger.warning(f"Processed raster not found: {processed_raster_id}")
            return None
            
        return response.data[0]
        
    except Exception as e:
        logger.error(f"Failed to get processed raster metadata: {str(e)}")
        return None

async def get_band_mappings(project_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Get band mappings for a project or system-defined mappings.
    
    Args:
        project_id: Optional UUID of the project
        
    Returns:
        List of band mappings
    """
    try:
        # Get the singleton Supabase client
        supabase = get_supabase()
        
        # Query the band_mappings table
        query = supabase.table('band_mappings').select('*')
        
        if project_id:
            # Get system mappings and project-specific mappings
            query = query.or_(f'is_system.eq.true,project_id.eq.{project_id}')
        else:
            # Get only system mappings
            query = query.eq('is_system', True)
            
        response = query.execute()
        
        if hasattr(response, 'error') and response.error:
            logger.error(f"Error getting band mappings: {response.error}")
            return []
            
        return response.data
        
    except Exception as e:
        logger.error(f"Failed to get band mappings: {str(e)}")
        return []

async def get_health_indices() -> List[Dict[str, Any]]:
    """
    Get available health indices.
    
    Returns:
        List of health indices
    """
    try:
        # Get the singleton Supabase client
        supabase = get_supabase()
        
        # Query the health_indices table
        response = supabase.table('health_indices').select('*').execute()
        
        if hasattr(response, 'error') and response.error:
            logger.error(f"Error getting health indices: {response.error}")
            return []
            
        return response.data
        
    except Exception as e:
        logger.error(f"Failed to get health indices: {str(e)}")
        return []

async def create_custom_band_mapping(
    name: str,
    mapping: Dict[str, int],
    project_id: str,
    description: Optional[str] = None
) -> Optional[str]:
    """
    Create a custom band mapping for a project.
    
    Args:
        name: Name of the band mapping
        mapping: Dictionary mapping band names to band numbers
        project_id: UUID of the project
        description: Optional description
        
    Returns:
        UUID of the created band mapping or None if failed
    """
    try:
        # Get the singleton Supabase client
        supabase = get_supabase()
        
        # Insert into the band_mappings table
        response = supabase.table('band_mappings').insert({
            'name': name,
            'description': description,
            'mapping': mapping,
            'is_system': False,
            'project_id': project_id
        }).execute()
        
        if hasattr(response, 'error') and response.error:
            logger.error(f"Error creating custom band mapping: {response.error}")
            return None
            
        if not response.data:
            logger.warning("No data returned from band mapping creation")
            return None
            
        return response.data[0]['id']
        
    except Exception as e:
        logger.error(f"Failed to create custom band mapping: {str(e)}")
        return None

async def get_project_raster_files(project_id: str) -> List[Dict[str, Any]]:
    """Get all non-deleted raster files for a project.
    
    Args:
        project_id: UUID of the project
        
    Returns:
        List[Dict[str, Any]]: List of active raster files
    """
    try:
        supabase = get_supabase()
        
        # Query raster files for this project using the new deleted column
        query = (supabase.table("raster_files")
                .select("*")
                .eq("project_id", project_id)
                .eq("deleted", False))  # Use the new column
        
        result = query.execute()
        
        logger.info(f"Retrieved {len(result.data)} active raster files for project {project_id}")
        return result.data
        
    except Exception as e:
        logger.error(f"Error getting project raster files: {str(e)}")
        raise Exception(f"Error getting project raster files: {str(e)}")

async def get_project_processed_rasters(project_id: str) -> List[Dict[str, Any]]:
    """
    Get all processed rasters for a project.
    
    Args:
        project_id: UUID of the project
        
    Returns:
        List of processed rasters
    """
    try:
        # Get the singleton Supabase client
        supabase = get_supabase()
        
        # Query the processed_rasters table via processing_jobs
        response = supabase.table('processed_rasters').select(
            'processed_rasters.*, processing_jobs.project_id'
        ).eq('processing_jobs.project_id', project_id).execute()
        
        if hasattr(response, 'error') and response.error:
            logger.error(f"Error getting project processed rasters: {response.error}")
            return []
            
        return response.data
        
    except Exception as e:
        logger.error(f"Failed to get project processed rasters: {str(e)}")
        return [] 