from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from typing import Dict, Any
from app.core.logger import logger
from app.core.database import get_db
from app.schemas.raster import RasterDetails

router = APIRouter(prefix="/api/v1/raster-files", tags=["raster-files"])

async def fetch_raster_details(db: AsyncSession, raster_id: str) -> Dict[str, Any]:
    """
    Fetch raster details from the database.
    
    Args:
        db: AsyncSession - Database session
        raster_id: str - ID of the raster file
        
    Returns:
        Dict containing raster details
        
    Raises:
        HTTPException: If raster not found or database error occurs
    """
    query = text("""
        WITH raster_info AS (
          SELECT 
            rf.id,
            rf.file_name,
            rf.file_size,
            rf.width,
            rf.height,
            rf.band_count,
            rf.driver,
            rf.projection,
            rf.geotransform,
            rf.bounds,
            rf.metadata,
            rf.created_at,
            rf.updated_at,
            rf.type,
            rf.process_type,
            CASE 
              WHEN pr.id IS NOT NULL THEN 'processed'
              ELSE 'raw'
            END as raster_type
          FROM raster_files rf
          LEFT JOIN processed_rasters pr ON rf.id = pr.raster_file_id
          LEFT JOIN processing_jobs pj ON pr.processing_job_id = pj.id
          WHERE rf.id = :raster_id
          AND (rf.deleted IS NULL OR rf.deleted = false)
        ),
        band_info AS (
          SELECT 
            rb.*
          FROM raster_bands rb
          WHERE rb.raster_file_id = :raster_id
          ORDER BY rb.band_number
        )
        SELECT 
          json_build_object(
            'basic_info', json_build_object(
              'id', ri.id,
              'file_name', ri.file_name,
              'file_size', ri.file_size,
              'created_at', ri.created_at,
              'updated_at', ri.updated_at,
              'type', ri.raster_type,
              'process_type', ri.process_type
            ),
            'spatial_info', json_build_object(
              'width', ri.width,
              'height', ri.height,
              'band_count', ri.band_count,
              'driver', ri.driver,
              'projection', ri.projection,
              'bounds', ri.bounds,
              'geotransform', ri.geotransform
            ),
            'bands', (
              SELECT json_agg(
                json_build_object(
                  'band_number', bi.band_number,
                  'data_type', bi.data_type,
                  'min_value', bi.min_value,
                  'max_value', bi.max_value,
                  'mean_value', bi.mean_value,
                  'stddev_value', bi.stddev_value,
                  'nodata_value', bi.nodata_value,
                  'color_interpretation', bi.color_interpretation,
                  'wavelength', bi.wavelength,
                  'wavelength_unit', bi.wavelength_unit,
                  'band_name', bi.band_name
                )
              )
              FROM band_info bi
            ),
            'metadata', ri.metadata
          ) as raster_details
        FROM raster_info ri;
    """)
    
    result = await db.execute(query, {"raster_id": raster_id})
    row = result.fetchone()
    
    if not row:
        raise HTTPException(
            status_code=404,
            detail=f"Raster file with id {raster_id} not found"
        )
        
    return row.raster_details

@router.get("/{raster_id}/details", response_model=RasterDetails)
async def get_raster_details(
    raster_id: str,
    db: AsyncSession = Depends(get_db)
) -> RasterDetails:
    """
    Get detailed information about a specific raster file.
    
    Args:
        raster_id: str - The ID of the raster file
        db: AsyncSession - Database session (injected)
        
    Returns:
        RasterDetails object containing all raster information
        
    Raises:
        HTTPException: 404 if raster not found, 500 for server errors
    """
    try:
        return await fetch_raster_details(db, raster_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching raster details for {raster_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to fetch raster details"
        ) 