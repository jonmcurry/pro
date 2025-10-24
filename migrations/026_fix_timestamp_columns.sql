-- Migration 026: Fix TIMESTAMP columns to TIMESTAMPTZ
-- This migration fixes the column types in version tracking tables

-- Fix application_version.installed_at
ALTER TABLE staging.application_version
ALTER COLUMN installed_at TYPE TIMESTAMPTZ USING installed_at AT TIME ZONE 'UTC';

-- Fix schema_migrations.applied_at
ALTER TABLE staging.schema_migrations
ALTER COLUMN applied_at TYPE TIMESTAMPTZ USING applied_at AT TIME ZONE 'UTC';

-- Add comments
COMMENT ON COLUMN staging.application_version.installed_at IS
'Installation timestamp with timezone (fixed in v1.2.6)';

COMMENT ON COLUMN staging.schema_migrations.applied_at IS
'Migration application timestamp with timezone (fixed in v1.2.6)';
