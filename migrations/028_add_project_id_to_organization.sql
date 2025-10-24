-- Migration: 028_add_project_id_to_organization
-- Description: Add project_id column to organization table
-- Date: 2025-10-24

-- Add project_id column to organization table
ALTER TABLE claims.organization
ADD COLUMN project_id UUID;

-- Add index for project_id lookups
CREATE INDEX idx_organization_project_id ON claims.organization(project_id);

-- Add comment explaining the column
COMMENT ON COLUMN claims.organization.project_id IS
'Optional project identifier for grouping organizations into projects';

-- Mark this migration as applied
INSERT INTO staging.schema_migrations (migration_name, applied_at, checksum, description)
VALUES ('028_add_project_id_to_organization.sql', NOW(), 'v1', 'Add project_id column to organization table')
ON CONFLICT (migration_name) DO NOTHING;
