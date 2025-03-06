-- Schema extension for storing GeoTIFF and raster processing metadata

-- Enable the auth schema extensions if not already enabled
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Organizations table
CREATE TABLE organizations (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_by UUID REFERENCES auth.users(id)
);

-- Organization members table
CREATE TABLE organization_members (
    organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    role VARCHAR(50) NOT NULL DEFAULT 'member',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (organization_id, user_id)
);

-- Projects table
CREATE TABLE projects (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_by UUID REFERENCES auth.users(id)
);

-- Enable Row Level Security
ALTER TABLE organizations ENABLE ROW LEVEL SECURITY;
ALTER TABLE organization_members ENABLE ROW LEVEL SECURITY;
ALTER TABLE projects ENABLE ROW LEVEL SECURITY;

-- Organizations policies
CREATE POLICY "Users can view organizations they are members of"
ON organizations FOR SELECT
TO authenticated
USING (
    id IN (
        SELECT organization_id 
        FROM organization_members 
        WHERE user_id = auth.uid()
    )
);

CREATE POLICY "Users can create organizations"
ON organizations FOR INSERT
TO authenticated
WITH CHECK (
    created_by = auth.uid()
);

-- Organization members policies
CREATE POLICY "Users can view members of their organizations"
ON organization_members FOR SELECT
TO authenticated
USING (
    organization_id IN (
        SELECT organization_id 
        FROM organization_members 
        WHERE user_id = auth.uid()
    )
);

CREATE POLICY "Organization owners can manage members"
ON organization_members FOR ALL
TO authenticated
USING (
    organization_id IN (
        SELECT organization_id 
        FROM organization_members 
        WHERE user_id = auth.uid() 
        AND role = 'owner'
    )
);

-- Projects policies
CREATE POLICY "Users can view projects in their organizations"
ON projects FOR SELECT
TO authenticated
USING (
    organization_id IN (
        SELECT organization_id 
        FROM organization_members 
        WHERE user_id = auth.uid()
    )
);

CREATE POLICY "Users can create projects in their organizations"
ON projects FOR INSERT
TO authenticated
WITH CHECK (
    organization_id IN (
        SELECT organization_id 
        FROM organization_members 
        WHERE user_id = auth.uid()
    )
    AND created_by = auth.uid()
);

-- Raster files table to store metadata about uploaded GeoTIFF files
CREATE TABLE raster_files (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
    file_name VARCHAR(255) NOT NULL,
    s3_url VARCHAR(255) NOT NULL,
    file_size BIGINT NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    band_count INTEGER NOT NULL,
    driver VARCHAR(50) NOT NULL,
    projection TEXT,
    geotransform JSONB,
    bounds JSONB NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_by UUID REFERENCES auth.users(id)
);

-- Raster bands table to store detailed information about each band in a raster file
CREATE TABLE raster_bands (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    raster_file_id UUID REFERENCES raster_files(id) ON DELETE CASCADE,
    band_number INTEGER NOT NULL,
    data_type VARCHAR(50) NOT NULL,
    min_value DOUBLE PRECISION,
    max_value DOUBLE PRECISION,
    mean_value DOUBLE PRECISION,
    stddev_value DOUBLE PRECISION,
    nodata_value DOUBLE PRECISION,
    color_interpretation VARCHAR(50),
    wavelength DOUBLE PRECISION,
    wavelength_unit VARCHAR(20),
    band_name VARCHAR(50),
    metadata JSONB DEFAULT '{}'::jsonb,
    UNIQUE (raster_file_id, band_number)
);

-- Band mappings table to store band mappings for different sensors and custom mappings
CREATE TABLE band_mappings (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    mapping JSONB NOT NULL, -- e.g., {"R": 3, "G": 2, "B": 1, "N": 4}
    is_system BOOLEAN DEFAULT FALSE, -- True for system-defined mappings (e.g., for known sensors)
    project_id UUID REFERENCES projects(id) ON DELETE CASCADE, -- NULL for system-defined mappings
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_by UUID REFERENCES auth.users(id),
    UNIQUE (name, project_id) -- Allow same name across different projects
);

-- Processed rasters table to store metadata about processed raster outputs
CREATE TABLE processed_rasters (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    processing_job_id UUID REFERENCES processing_jobs(id) ON DELETE CASCADE,
    raster_file_id UUID REFERENCES raster_files(id),
    output_type VARCHAR(50) NOT NULL, -- 'ndvi', 'evi', 'landcover', etc.
    s3_url VARCHAR(255) NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    band_count INTEGER NOT NULL,
    driver VARCHAR(50) NOT NULL,
    projection TEXT,
    geotransform JSONB,
    bounds JSONB NOT NULL,
    min_value DOUBLE PRECISION,
    max_value DOUBLE PRECISION,
    mean_value DOUBLE PRECISION,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Health indices definitions table to store information about available health indices
CREATE TABLE health_indices (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    name VARCHAR(50) NOT NULL UNIQUE,
    formula VARCHAR(255) NOT NULL,
    description TEXT,
    min_value DOUBLE PRECISION,
    max_value DOUBLE PRECISION,
    required_bands VARCHAR[] NOT NULL,
    is_system BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_by UUID REFERENCES auth.users(id)
);

-- Insert system-defined health indices
INSERT INTO health_indices (name, formula, description, min_value, max_value, required_bands)
VALUES
    ('NDVI', '(N - R) / (N + R)', 'Normalized Difference Vegetation Index shows the amount of green vegetation.', -1, 1, ARRAY['N', 'R']),
    ('NDRE', '(N - Re) / (N + Re)', 'Normalized Difference Red Edge Index shows the amount of green vegetation of permanent or later stage crops.', -1, 1, ARRAY['N', 'Re']),
    ('EVI', '2.5 * ((N - R) / (N + 6 * R - 7.5 * B + 1))', 'Enhanced Vegetation Index is optimized to enhance the vegetation signal with improved sensitivity in high biomass regions.', -1, 1, ARRAY['N', 'R', 'B']),
    ('NDWI', '(G - N) / (G + N)', 'Normalized Difference Water Index is used to monitor changes in water content of leaves.', -1, 1, ARRAY['G', 'N']),
    ('NDYI', '(G - B) / (G + B)', 'Normalized difference yellowness index (NDYI), best model variability in relative yield potential in Canola.', -1, 1, ARRAY['G', 'B']);

-- Insert system-defined band mappings for common sensors
INSERT INTO band_mappings (name, description, mapping, is_system)
VALUES
    ('Sentinel-2', 'Sentinel-2 MSI standard band mapping', '{"B": 2, "G": 3, "R": 4, "Re": 5, "N": 8, "SWIR1": 11, "SWIR2": 12}'::jsonb, TRUE),
    ('Landsat-8', 'Landsat-8 OLI standard band mapping', '{"B": 2, "G": 3, "R": 4, "N": 5, "SWIR1": 6, "SWIR2": 7}'::jsonb, TRUE),
    ('WorldView-3', 'WorldView-3 standard band mapping', '{"C": 1, "B": 2, "G": 3, "Y": 4, "R": 5, "Re": 6, "N": 7, "N2": 8}'::jsonb, TRUE);

-- Create Row Level Security (RLS) policies

-- Raster files RLS
ALTER TABLE raster_files ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view raster files in their projects" 
ON raster_files FOR SELECT 
TO authenticated 
USING (
    project_id IN (
        SELECT id FROM projects
        WHERE organization_id IN (
            SELECT organization_id FROM organization_members
            WHERE user_id = auth.uid()
        )
    )
);

CREATE POLICY "Users can insert raster files in their projects" 
ON raster_files FOR INSERT 
TO authenticated 
WITH CHECK (
    project_id IN (
        SELECT id FROM projects
        WHERE organization_id IN (
            SELECT organization_id FROM organization_members
            WHERE user_id = auth.uid()
        )
    )
);

-- Raster bands RLS
ALTER TABLE raster_bands ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view raster bands in their projects" 
ON raster_bands FOR SELECT 
TO authenticated 
USING (
    raster_file_id IN (
        SELECT id FROM raster_files
        WHERE project_id IN (
            SELECT id FROM projects
            WHERE organization_id IN (
                SELECT organization_id FROM organization_members
                WHERE user_id = auth.uid()
            )
        )
    )
);

-- Band mappings RLS
ALTER TABLE band_mappings ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view system band mappings" 
ON band_mappings FOR SELECT 
TO authenticated 
USING (
    is_system = TRUE OR
    project_id IN (
        SELECT id FROM projects
        WHERE organization_id IN (
            SELECT organization_id FROM organization_members
            WHERE user_id = auth.uid()
        )
    )
);

CREATE POLICY "Users can insert band mappings in their projects" 
ON band_mappings FOR INSERT 
TO authenticated 
WITH CHECK (
    project_id IN (
        SELECT id FROM projects
        WHERE organization_id IN (
            SELECT organization_id FROM organization_members
            WHERE user_id = auth.uid()
        )
    )
);

-- Processed rasters RLS
ALTER TABLE processed_rasters ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view processed rasters in their projects" 
ON processed_rasters FOR SELECT 
TO authenticated 
USING (
    processing_job_id IN (
        SELECT id FROM processing_jobs
        WHERE project_id IN (
            SELECT id FROM projects
            WHERE organization_id IN (
                SELECT organization_id FROM organization_members
                WHERE user_id = auth.uid()
            )
        )
    )
);

-- Health indices RLS
ALTER TABLE health_indices ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view all health indices" 
ON health_indices FOR SELECT 
TO authenticated 
USING (TRUE);

CREATE POLICY "Users can insert custom health indices" 
ON health_indices FOR INSERT 
TO authenticated 
WITH CHECK (
    is_system = FALSE
);

-- Create a function to update processing_jobs when a processed raster is created
CREATE OR REPLACE FUNCTION update_processing_job_on_processed_raster()
RETURNS TRIGGER AS $$
BEGIN
    -- Update the processing_job with the result URL
    UPDATE processing_jobs
    SET 
        result_url = NEW.s3_url,
        status = 'COMPLETED',
        completed_at = CURRENT_TIMESTAMP
    WHERE id = NEW.processing_job_id;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

CREATE TRIGGER after_processed_raster_insert
AFTER INSERT ON processed_rasters
FOR EACH ROW
EXECUTE FUNCTION update_processing_job_on_processed_raster();

-- Create a function to analyze a GeoTIFF file and store its metadata
CREATE OR REPLACE FUNCTION register_geotiff_metadata(
    p_project_id UUID,
    p_file_name TEXT,
    p_s3_url TEXT,
    p_metadata JSONB
)
RETURNS UUID
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    v_raster_file_id UUID;
    v_band JSONB;
BEGIN
    -- Insert the raster file metadata
    INSERT INTO raster_files (
        project_id,
        file_name,
        s3_url,
        file_size,
        width,
        height,
        band_count,
        driver,
        projection,
        geotransform,
        bounds,
        metadata,
        created_by
    ) VALUES (
        p_project_id,
        p_file_name,
        p_s3_url,
        (p_metadata->>'file_size')::BIGINT,
        (p_metadata->>'width')::INTEGER,
        (p_metadata->>'height')::INTEGER,
        (p_metadata->>'band_count')::INTEGER,
        p_metadata->>'driver',
        p_metadata->>'projection',
        p_metadata->'geotransform',
        p_metadata->'bounds',
        p_metadata->'metadata',
        auth.uid()
    )
    RETURNING id INTO v_raster_file_id;
    
    -- Insert band information
    FOR v_band IN SELECT * FROM jsonb_array_elements(p_metadata->'bands')
    LOOP
        INSERT INTO raster_bands (
            raster_file_id,
            band_number,
            data_type,
            min_value,
            max_value,
            mean_value,
            stddev_value,
            nodata_value,
            color_interpretation,
            wavelength,
            metadata
        ) VALUES (
            v_raster_file_id,
            (v_band->>'band_number')::INTEGER,
            v_band->>'data_type',
            (v_band->>'min')::DOUBLE PRECISION,
            (v_band->>'max')::DOUBLE PRECISION,
            (v_band->>'mean')::DOUBLE PRECISION,
            (v_band->>'stddev')::DOUBLE PRECISION,
            (v_band->>'nodata_value')::DOUBLE PRECISION,
            v_band->>'color_interpretation',
            (v_band->>'wavelength')::DOUBLE PRECISION,
            v_band->'metadata'
        );
    END LOOP;
    
    RETURN v_raster_file_id;
END;
$$;

-- Create a function to register a processed raster
CREATE OR REPLACE FUNCTION register_processed_raster(
    p_processing_job_id UUID,
    p_raster_file_id UUID,
    p_output_type TEXT,
    p_s3_url TEXT,
    p_metadata JSONB
)
RETURNS UUID
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    v_processed_raster_id UUID;
BEGIN
    -- Insert the processed raster metadata
    INSERT INTO processed_rasters (
        processing_job_id,
        raster_file_id,
        output_type,
        s3_url,
        width,
        height,
        band_count,
        driver,
        projection,
        geotransform,
        bounds,
        min_value,
        max_value,
        mean_value,
        metadata
    ) VALUES (
        p_processing_job_id,
        p_raster_file_id,
        p_output_type,
        p_s3_url,
        (p_metadata->>'width')::INTEGER,
        (p_metadata->>'height')::INTEGER,
        (p_metadata->>'band_count')::INTEGER,
        p_metadata->>'driver',
        p_metadata->>'projection',
        p_metadata->'geotransform',
        p_metadata->'bounds',
        (p_metadata->>'min_value')::DOUBLE PRECISION,
        (p_metadata->>'max_value')::DOUBLE PRECISION,
        (p_metadata->>'mean_value')::DOUBLE PRECISION,
        p_metadata->'metadata'
    )
    RETURNING id INTO v_processed_raster_id;
    
    RETURN v_processed_raster_id;
END;
$$;

-- Processing jobs table to store job information and status
CREATE TABLE processing_jobs (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
    process_type VARCHAR(50) NOT NULL,
    input_file VARCHAR(255) NOT NULL,
    parameters JSONB DEFAULT '{}'::jsonb,
    status VARCHAR(20) NOT NULL DEFAULT 'PENDING',
    error TEXT,
    result JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_by UUID REFERENCES auth.users(id)
);

-- Add RLS policies for processing_jobs
ALTER TABLE processing_jobs ENABLE ROW LEVEL SECURITY;

-- Drop existing policies if they exist
DROP POLICY IF EXISTS "Users can view processing jobs in their projects" ON processing_jobs;
DROP POLICY IF EXISTS "Users can create processing jobs in their projects" ON processing_jobs;
DROP POLICY IF EXISTS "Users can update processing jobs in their projects" ON processing_jobs;

-- Create new policies with proper checks
CREATE POLICY "Users can view processing jobs in their projects" 
ON processing_jobs FOR SELECT 
TO authenticated 
USING (
    project_id IN (
        SELECT p.id FROM projects p
        INNER JOIN organization_members om ON p.organization_id = om.organization_id
        WHERE om.user_id = auth.uid()
    )
);

CREATE POLICY "Users can create processing jobs in their projects" 
ON processing_jobs FOR INSERT 
TO authenticated 
WITH CHECK (
    project_id IN (
        SELECT p.id FROM projects p
        INNER JOIN organization_members om ON p.organization_id = om.organization_id
        WHERE om.user_id = auth.uid()
    )
    AND created_by = auth.uid()
);

CREATE POLICY "Users can update processing jobs in their projects" 
ON processing_jobs FOR UPDATE 
TO authenticated 
USING (
    project_id IN (
        SELECT p.id FROM projects p
        INNER JOIN organization_members om ON p.organization_id = om.organization_id
        WHERE om.user_id = auth.uid()
    )
);

-- Create a function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create a trigger to automatically update the updated_at column
CREATE TRIGGER update_processing_jobs_updated_at
    BEFORE UPDATE ON processing_jobs
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column(); 