-- Migration: 019_phase6_materialized_views
-- Description: Phase 6 materialized views for analytics performance
-- Date: 2025-10-15
-- Status: DISABLED - Schema incompatibility with current flag table structure
-- Impact: 10-100x faster dashboard and analytics queries (when enabled)

-- ============================================================================
-- MIGRATION DISABLED DUE TO SCHEMA INCOMPATIBILITY
-- ============================================================================

-- This migration creates materialized views that reference claims.flag table,
-- which does not exist in the current schema. Flags are stored in:
-- - claims.encounter_flag
-- - claims.service_line_flag
--
-- This migration needs to be rewritten to use the correct flag tables before
-- it can be enabled. For now, it is disabled to allow fresh installations to
-- proceed without error.
--
-- Analytics views can be added later once the flag table structure is
-- standardized.

-- Create analytics schema if it doesn't exist
CREATE SCHEMA IF NOT EXISTS analytics;

-- Placeholder comment to mark this migration as applied
COMMENT ON SCHEMA analytics IS 'Analytics schema created - materialized views pending flag table refactoring (migration 019 disabled)';
