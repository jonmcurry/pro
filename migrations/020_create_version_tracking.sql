-- Migration: 020_create_version_tracking
-- Description: Create version tracking tables for upgrade management
-- Date: 2025-10-20

-- Create schema_migrations table to track which migrations have been applied
CREATE TABLE IF NOT EXISTS staging.schema_migrations (
    migration_name VARCHAR(255) PRIMARY KEY,
    applied_at TIMESTAMP NOT NULL DEFAULT NOW(),
    checksum TEXT NOT NULL,
    execution_time_ms INTEGER,
    description TEXT
);

-- Create application_version table to track application versions
CREATE TABLE IF NOT EXISTS staging.application_version (
    version VARCHAR(50) PRIMARY KEY,
    installed_at TIMESTAMP NOT NULL DEFAULT NOW(),
    upgraded_from VARCHAR(50),
    notes TEXT
);

-- Add indexes for performance
CREATE INDEX IF NOT EXISTS idx_schema_migrations_applied_at
    ON staging.schema_migrations(applied_at);

CREATE INDEX IF NOT EXISTS idx_application_version_installed_at
    ON staging.application_version(installed_at DESC);

-- Add comments
COMMENT ON TABLE staging.schema_migrations IS
    'Tracks which database migrations have been applied';
COMMENT ON COLUMN staging.schema_migrations.migration_name IS
    'Name of the migration file (e.g., 001_create_schemas.sql)';
COMMENT ON COLUMN staging.schema_migrations.checksum IS
    'SHA-256 checksum of the migration file for integrity verification';
COMMENT ON COLUMN staging.schema_migrations.execution_time_ms IS
    'Time taken to execute the migration in milliseconds';

COMMENT ON TABLE staging.application_version IS
    'Tracks application version history for upgrade management';
COMMENT ON COLUMN staging.application_version.version IS
    'Semantic version number (e.g., 1.0.0, 1.1.0)';
COMMENT ON COLUMN staging.application_version.upgraded_from IS
    'Previous version if this was an upgrade, NULL for fresh install';

-- Backfill schema_migrations with existing migrations
-- This is for installations upgrading from pre-version-tracking state
-- We mark all migrations 001-019 as already applied
INSERT INTO staging.schema_migrations (migration_name, applied_at, checksum, description)
VALUES
    ('001_create_schemas.sql', NOW(), 'legacy', 'Create the three main schemas for the application'),
    ('002_create_organization_tables.sql', NOW(), 'legacy', 'Create organization and facility tables'),
    ('003_create_provider_tables.sql', NOW(), 'legacy', 'Create provider and practitioner tables'),
    ('004_create_encounter_tables.sql', NOW(), 'legacy', 'Create encounter and claim tables'),
    ('005_create_diagnosis_procedure_tables.sql', NOW(), 'legacy', 'Create diagnosis and procedure tables'),
    ('006_create_flag_tables.sql', NOW(), 'legacy', 'Create flag and validation tables'),
    ('007_create_staging_tables.sql', NOW(), 'legacy', 'Create staging tables for file processing'),
    ('008_create_audit_tables.sql', NOW(), 'legacy', 'Create audit and logging tables'),
    ('009_create_rvu_tables.sql', NOW(), 'legacy', 'Create RVU reference tables'),
    ('010_create_denial_tables.sql', NOW(), 'legacy', 'Create denial tracking tables'),
    ('011_create_schedule_tables.sql', NOW(), 'legacy', 'Create schedule and appointment tables'),
    ('012_create_ml_tables.sql', NOW(), 'legacy', 'Create machine learning tables'),
    ('013_create_dashboard_views.sql', NOW(), 'legacy', 'Create dashboard and reporting views'),
    ('014_create_utility_functions.sql', NOW(), 'legacy', 'Create utility functions'),
    ('015_create_fifo_queue.sql', NOW(), 'legacy', 'Create FIFO queue for processing'),
    ('016_phase5_performance_indexes.sql', NOW(), 'legacy', 'Phase 5 performance indexes'),
    ('017_streaming_progress_tracking.sql', NOW(), 'legacy', 'Streaming progress tracking'),
    ('018_phase6_strategic_indexes.sql', NOW(), 'legacy', 'Phase 6 strategic indexes'),
    ('019_phase6_materialized_views.sql', NOW(), 'legacy', 'Phase 6 materialized views')
ON CONFLICT (migration_name) DO NOTHING;

-- Mark migration 020 as applied
INSERT INTO staging.schema_migrations (migration_name, applied_at, checksum, description)
VALUES ('020_create_version_tracking.sql', NOW(), 'legacy', 'Create version tracking tables for upgrade management')
ON CONFLICT (migration_name) DO NOTHING;
