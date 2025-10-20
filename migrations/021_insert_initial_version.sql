-- Migration: 021_insert_initial_version
-- Description: Insert initial version information for baseline installations
-- Date: 2025-10-20

-- Insert version 1.0.0 as baseline
-- For fresh installs, this represents the first version with version tracking
-- For upgrades from 1.0.0, this represents the legacy version
INSERT INTO staging.application_version (version, installed_at, upgraded_from, notes)
VALUES (
    '1.0.0',
    NOW(),
    NULL,
    'Baseline installation with 19 migrations (001-019). Version tracking added in 1.1.0.'
)
ON CONFLICT (version) DO NOTHING;

-- Mark this migration as applied
INSERT INTO staging.schema_migrations (migration_name, applied_at, checksum, description)
VALUES ('021_insert_initial_version.sql', NOW(), 'legacy', 'Insert initial version information for baseline installations')
ON CONFLICT (migration_name) DO NOTHING;
