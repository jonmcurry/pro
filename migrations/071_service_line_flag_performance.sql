-- Migration: 071_service_line_flag_performance
-- Description: Add performance indexes for service_line_flag queries
-- Problem: SELECT * FROM claims.service_line_flag taking 5+ minutes
-- Date: 2025-01-15

-- ============================================================================
-- ROOT CAUSE ANALYSIS:
-- 1. service_line_flag table has TEXT columns (flag_reason, resolution_note)
-- 2. SELECT * forces full table scan including large TEXT data
-- 3. v_service_line_flags_detail view has 4 INNER JOINs
-- 4. No composite index for the JOIN pattern
-- 5. No unique constraint - reprocessing creates duplicate flags
-- ============================================================================

-- ============================================================================
-- CRITICAL: Add unique constraint to prevent duplicate flags
-- ============================================================================
-- A service line should only have ONE flag per issue_id with OPEN status
-- This prevents duplicate flags when claims are reprocessed

CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS idx_service_line_flag_unique_open
ON claims.service_line_flag (service_line_id, issue_id)
WHERE flag_status = 'OPEN';

COMMENT ON INDEX claims.idx_service_line_flag_unique_open IS
'Prevents duplicate OPEN flags for the same service_line and issue combination';

-- ============================================================================
-- Performance Indexes
-- ============================================================================

-- Composite index for the view's primary JOIN pattern
-- This allows index-only scan for the most common columns
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_service_line_flag_view_lookup
ON claims.service_line_flag (service_line_id, issue_id)
INCLUDE (flag_id, severity, flag_status, flag_type, created_at, flagged_element);

COMMENT ON INDEX claims.idx_service_line_flag_view_lookup IS
'Covering index for v_service_line_flags_detail view JOINs - avoids heap access for common columns';

-- Index for recent flags (dashboards typically show recent first)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_service_line_flag_recent
ON claims.service_line_flag (created_at DESC)
INCLUDE (service_line_id, issue_id, severity, flag_status)
WHERE flag_status = 'OPEN';

COMMENT ON INDEX claims.idx_service_line_flag_recent IS
'Optimizes recent open flags queries for dashboards';

-- Index for flag status filtering with service line lookup
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_service_line_flag_status_lookup
ON claims.service_line_flag (flag_status, service_line_id)
INCLUDE (issue_id, severity, created_at);

COMMENT ON INDEX claims.idx_service_line_flag_status_lookup IS
'Optimizes flag status queries with service line context';

-- ============================================================================
-- Service Line Index for Reverse Lookup
-- ============================================================================
-- The view JOINs service_line_flag -> service_line -> encounter
-- This index helps the service_line -> encounter lookup

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_service_line_encounter_lookup
ON claims.service_line (encounter_id)
INCLUDE (service_line_id, procedure_code, line_item_charge_amount)
WHERE line_status = 'ACTIVE';

COMMENT ON INDEX claims.idx_service_line_encounter_lookup IS
'Covering index for service_line to encounter JOINs in flag views';

-- ============================================================================
-- Analyze Tables After Index Creation
-- ============================================================================

ANALYZE claims.service_line_flag;
ANALYZE claims.service_line;

-- ============================================================================
-- USAGE RECOMMENDATIONS (for application code):
--
-- Instead of: SELECT * FROM claims.service_line_flag
-- Use:        SELECT flag_id, service_line_id, issue_id, severity, flag_status,
--                    flagged_element, created_at
--             FROM claims.service_line_flag
--             WHERE flag_status = 'OPEN'
--             ORDER BY created_at DESC
--             LIMIT 1000
--
-- The TEXT columns (flag_reason, resolution_note) should only be fetched
-- when viewing a specific flag's details, not in list queries.
-- ============================================================================
