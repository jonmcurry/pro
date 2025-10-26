-- Migration: 030_create_import_headers_table
-- Description: Create import_headers table to track data element headers for imports
-- Date: 2025-10-24

-- Create import_headers table in claims schema
CREATE TABLE claims.import_headers (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Project and facility identification
    project_id UUID, -- Nullable to support facilities without project assignment
    facility_id UUID NOT NULL REFERENCES claims.facility(facility_id) ON DELETE CASCADE,

    -- File type classification
    file_type VARCHAR(10) NOT NULL CHECK (file_type IN ('hl7', 'csv', 'edi')),

    -- Timestamp tracking
    date TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    -- Header data
    header VARCHAR(255) NOT NULL,

    -- Status flag
    is_active BOOLEAN DEFAULT true,

    -- Audit trail
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100),
    updated_by VARCHAR(100),

    -- Ensure no duplicate headers within same project/facility/file_type
    UNIQUE(project_id, facility_id, file_type, header)
);

-- Create indexes for performance
CREATE INDEX idx_import_headers_project_facility_type
    ON claims.import_headers(project_id, facility_id, file_type);

CREATE INDEX idx_import_headers_facility
    ON claims.import_headers(facility_id);

CREATE INDEX idx_import_headers_file_type
    ON claims.import_headers(file_type);

CREATE INDEX idx_import_headers_date
    ON claims.import_headers(date DESC);

CREATE INDEX idx_import_headers_active
    ON claims.import_headers(is_active) WHERE is_active = true;

-- Add trigger for automatic updated_at timestamp
CREATE TRIGGER update_import_headers_updated_at
    BEFORE UPDATE ON claims.import_headers
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Add table and column comments for documentation
COMMENT ON TABLE claims.import_headers IS
'Tracks data element headers for file imports across projects, facilities, and file types';

COMMENT ON COLUMN claims.import_headers.id IS
'Unique identifier for each header record';

COMMENT ON COLUMN claims.import_headers.project_id IS
'Optional project identifier - allows grouping headers by project (references organization.project_id)';

COMMENT ON COLUMN claims.import_headers.facility_id IS
'Facility where this header is used (foreign key to claims.facility)';

COMMENT ON COLUMN claims.import_headers.file_type IS
'Type of import file: hl7, csv, or edi';

COMMENT ON COLUMN claims.import_headers.date IS
'When this header was defined or imported';

COMMENT ON COLUMN claims.import_headers.header IS
'The data element header value (e.g., column name, field identifier)';

COMMENT ON COLUMN claims.import_headers.is_active IS
'Soft delete flag - false indicates header is no longer in use';

-- Mark this migration as applied
INSERT INTO staging.schema_migrations (migration_name, applied_at, checksum, description)
VALUES ('030_create_import_headers_table.sql', NOW(), 'v1', 'Create import_headers table to track data element headers')
ON CONFLICT (migration_name) DO NOTHING;
