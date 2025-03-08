-- Function to populate raster_files from file_uploads
CREATE OR REPLACE FUNCTION public.populate_raster_files_from_uploads()
RETURNS TRIGGER AS $$
BEGIN
  -- Only process GeoTIFF files that are completed
  IF NEW.status = 'completed' AND 
     (NEW.filename LIKE '%.tif' OR NEW.filename LIKE '%.tiff' OR 
      NEW.content_type = 'image/tiff' OR NEW.content_type = 'image/geotiff') THEN
    
    -- Check if this file is already in raster_files
    IF NOT EXISTS (SELECT 1 FROM raster_files WHERE s3_url = 's3://' || NEW.s3_key) THEN
      -- Insert into raster_files with minimal information
      -- The actual metadata will be populated by the processing service later
      INSERT INTO raster_files (
        project_id,
        file_name,
        s3_url,
        file_size,
        width,
        height,
        band_count,
        driver,
        bounds,
        metadata,
        created_by
      ) VALUES (
        NEW.project_id,
        NEW.filename,
        's3://' || NEW.s3_key,
        NEW.file_size,
        0, -- Default width, will be updated by processing service
        0, -- Default height, will be updated by processing service
        0, -- Default band_count, will be updated by processing service
        'GTiff',
        '{"minx": 0, "miny": 0, "maxx": 0, "maxy": 0}'::jsonb, -- Default bounds
        '{}'::jsonb, -- Empty metadata
        auth.uid()
      );
    END IF;
  END IF;
  
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Trigger to populate raster_files when a file upload is completed
DROP TRIGGER IF EXISTS populate_raster_files_on_upload ON public.file_uploads;
CREATE TRIGGER populate_raster_files_on_upload
AFTER INSERT OR UPDATE OF status ON public.file_uploads
FOR EACH ROW
WHEN (NEW.status = 'completed')
EXECUTE FUNCTION public.populate_raster_files_from_uploads();

-- Function to populate raster_files from processing_jobs
CREATE OR REPLACE FUNCTION public.populate_raster_files_from_processing()
RETURNS TRIGGER AS $$
BEGIN
  -- Only process completed jobs with a result
  IF NEW.status = 'completed' AND NEW.result IS NOT NULL THEN
    -- Check if this processed file has an output path
    IF NEW.result->>'output_path' IS NOT NULL THEN
      -- Insert into raster_files with information from the processing job
      INSERT INTO raster_files (
        project_id,
        file_name,
        s3_url,
        file_size,
        width,
        height,
        band_count,
        driver,
        bounds,
        metadata,
        created_by
      ) VALUES (
        NEW.project_id,
        COALESCE(NEW.result->>'output_name', 'processed_' || NEW.id) || '.tif',
        NEW.result->>'output_path',
        0, -- Default file_size, will be updated by processing service
        COALESCE((NEW.result->>'width')::integer, 0),
        COALESCE((NEW.result->>'height')::integer, 0),
        COALESCE((NEW.result->>'band_count')::integer, 0),
        'GTiff',
        COALESCE(NEW.result->'bounds', '{"minx": 0, "miny": 0, "maxx": 0, "maxy": 0}'::jsonb),
        NEW.result,
        auth.uid()
      );
    END IF;
  END IF;
  
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Trigger to populate raster_files when a processing job is completed
DROP TRIGGER IF EXISTS populate_raster_files_on_processing ON public.processing_jobs;
CREATE TRIGGER populate_raster_files_on_processing
AFTER INSERT OR UPDATE OF status ON public.processing_jobs
FOR EACH ROW
WHEN (NEW.status = 'completed')
EXECUTE FUNCTION public.populate_raster_files_from_processing();

-- Function to handle soft deletion of raster files
CREATE OR REPLACE FUNCTION public.handle_raster_file_deletion()
RETURNS TRIGGER AS $$
BEGIN
  -- Add a deleted_at timestamp instead of actually deleting
  UPDATE raster_files
  SET metadata = jsonb_set(metadata, '{deleted}', 'true'::jsonb),
      metadata = jsonb_set(metadata, '{deleted_at}', to_jsonb(now()))
  WHERE id = OLD.id;
  
  -- Return NULL to prevent the actual deletion
  RETURN NULL;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Trigger to handle soft deletion of raster files
DROP TRIGGER IF EXISTS soft_delete_raster_files ON public.raster_files;
CREATE TRIGGER soft_delete_raster_files
BEFORE DELETE ON public.raster_files
FOR EACH ROW
EXECUTE FUNCTION public.handle_raster_file_deletion();

-- One-time migration to populate raster_files from existing file_uploads
INSERT INTO raster_files (
  project_id,
  file_name,
  s3_url,
  file_size,
  width,
  height,
  band_count,
  driver,
  bounds,
  metadata,
  created_at
)
SELECT 
  project_id,
  filename,
  's3://' || s3_key,
  file_size,
  0, -- Default width
  0, -- Default height
  0, -- Default band_count
  'GTiff',
  '{"minx": 0, "miny": 0, "maxx": 0, "maxy": 0}'::jsonb, -- Default bounds
  '{}'::jsonb, -- Empty metadata
  created_at
FROM file_uploads
WHERE status = 'completed'
AND (filename LIKE '%.tif' OR filename LIKE '%.tiff' OR content_type = 'image/tiff' OR content_type = 'image/geotiff')
AND NOT EXISTS (
  SELECT 1 FROM raster_files WHERE s3_url = 's3://' || file_uploads.s3_key
);

-- One-time migration to populate raster_files from existing processing_jobs
INSERT INTO raster_files (
  project_id,
  file_name,
  s3_url,
  file_size,
  width,
  height,
  band_count,
  driver,
  bounds,
  metadata,
  created_at
)
SELECT 
  project_id,
  COALESCE(result->>'output_name', 'processed_' || id) || '.tif',
  result->>'output_path',
  0, -- Default file_size
  COALESCE((result->>'width')::integer, 0),
  COALESCE((result->>'height')::integer, 0),
  COALESCE((result->>'band_count')::integer, 0),
  'GTiff',
  COALESCE(result->'bounds', '{"minx": 0, "miny": 0, "maxx": 0, "maxy": 0}'::jsonb),
  result,
  created_at
FROM processing_jobs
WHERE status = 'completed'
AND result IS NOT NULL
AND result->>'output_path' IS NOT NULL
AND NOT EXISTS (
  SELECT 1 FROM raster_files WHERE s3_url = processing_jobs.result->>'output_path'
); 