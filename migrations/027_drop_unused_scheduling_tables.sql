-- Migration: 027_drop_unused_scheduling_tables
-- Description: Drop unused scheduling and reporting tables that are not implemented
-- Date: 2025-10-24

-- Drop unused scheduling and reporting tables
-- These tables were defined for future features that are not currently implemented

DROP TABLE IF EXISTS staging.report_generation_log CASCADE;
DROP TABLE IF EXISTS staging.report_subscription CASCADE;
DROP TABLE IF EXISTS staging.job_execution_log CASCADE;
DROP TABLE IF EXISTS staging.scheduled_job CASCADE;
DROP TABLE IF EXISTS staging.data_refresh_schedule CASCADE;

-- Mark this migration as applied
INSERT INTO staging.schema_migrations (migration_name, applied_at, checksum, description)
VALUES ('027_drop_unused_scheduling_tables.sql', NOW(), 'v1', 'Drop unused scheduling and reporting tables')
ON CONFLICT (migration_name) DO NOTHING;
