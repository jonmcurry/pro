-- Migration: 074_service_line_flag_performance_tuning
-- Description: Performance tuning for service_line_flag with 900K+ rows
-- Problem: SELECT * FROM claims.service_line_flag takes 20 minutes
-- Date: 2025-01-16

-- ============================================================================
-- ROOT CAUSE ANALYSIS:
-- 1. SELECT * returns ALL columns including TEXT (flag_reason, resolution_note)
-- 2. TEXT columns are TOAST-compressed, requiring decompression for each row
-- 3. 900K rows × TEXT decompression = massive I/O
-- 4. No index helps full table scans (SELECT *)
--
-- SOLUTIONS:
-- 1. Create a "fast" view that excludes TEXT columns for list queries
-- 2. Add BRIN index for time-range queries (very efficient for append-only data)
-- 3. Ensure statistics are up-to-date
-- ============================================================================

-- ============================================================================
-- 1. Create Fast View for List Queries (excludes TEXT columns)
-- ============================================================================
-- Use this view for dashboards/lists; only fetch flag_reason when viewing details

CREATE OR REPLACE VIEW claims.v_service_line_flag_list AS
SELECT
    slf.flag_id,
    slf.service_line_id,
    slf.issue_id,
    slf.flag_type,
    slf.severity,
    slf.flagged_element,
    slf.proposed_code,
    slf.proposed_modifier,
    slf.proposed_quantity,
    slf.flag_status,
    slf.resolved_at,
    slf.resolved_by,
    slf.created_at,
    slf.created_by
    -- EXCLUDES: flag_reason (TEXT), resolution_note (TEXT)
FROM claims.service_line_flag slf;

COMMENT ON VIEW claims.v_service_line_flag_list IS
'Fast view for service_line_flag - excludes TEXT columns (flag_reason, resolution_note) for list queries. Use direct table access only when TEXT data is needed.';

-- ============================================================================
-- 2. Create Detail View with JOINs (for single flag lookup)
-- ============================================================================
CREATE OR REPLACE VIEW claims.v_service_line_flag_detail AS
SELECT
    slf.flag_id,
    slf.service_line_id,
    slf.issue_id,
    slf.flag_type,
    slf.severity,
    slf.flag_reason,
    slf.flagged_element,
    slf.proposed_code,
    slf.proposed_modifier,
    slf.proposed_quantity,
    slf.flag_status,
    slf.resolution_note,
    slf.resolved_at,
    slf.resolved_by,
    slf.created_at,
    slf.created_by,
    -- Join data
    fi.issue_code,
    fi.issue_description,
    fc.category_code,
    fc.category_name,
    sl.procedure_code,
    sl.line_item_charge_amount,
    e.encounter_id,
    e.date_of_service_from
FROM claims.service_line_flag slf
JOIN claims.flag_issue fi ON slf.issue_id = fi.issue_id
JOIN claims.flag_category fc ON fi.category_id = fc.category_id
JOIN claims.service_line sl ON slf.service_line_id = sl.service_line_id
JOIN claims.encounter e ON sl.encounter_id = e.encounter_id;

COMMENT ON VIEW claims.v_service_line_flag_detail IS
'Full detail view for single flag lookup - includes TEXT columns and JOINs to related tables.';

-- ============================================================================
-- 3. BRIN Index for Time-Range Queries
-- ============================================================================
-- BRIN (Block Range INdex) is extremely efficient for naturally ordered data
-- service_line_flag is append-only, so created_at is naturally ordered
-- BRIN is 100-1000x smaller than B-tree and very fast for range scans

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_service_line_flag_created_brin
ON claims.service_line_flag USING BRIN (created_at)
WITH (pages_per_range = 32);

COMMENT ON INDEX claims.idx_service_line_flag_created_brin IS
'BRIN index for time-range queries - extremely efficient for append-only tables';

-- ============================================================================
-- 4. Add index for flag_id primary key lookups with INCLUDE
-- ============================================================================
-- When fetching a single flag by ID, include common columns to avoid heap access

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_service_line_flag_pk_covering
ON claims.service_line_flag (flag_id)
INCLUDE (service_line_id, issue_id, severity, flag_status, created_at);

COMMENT ON INDEX claims.idx_service_line_flag_pk_covering IS
'Covering index for single flag lookups by flag_id';

-- ============================================================================
-- 5. Ensure Statistics are Up-to-Date
-- ============================================================================
ANALYZE claims.service_line_flag;

-- ============================================================================
-- 6. Set Table Storage Parameters for Better TOAST Performance
-- ============================================================================
-- Increase TOAST threshold to reduce TOAST storage for smaller flag_reason values
-- Default is 2KB; we'll try 4KB to keep more data inline

ALTER TABLE claims.service_line_flag
SET (toast_tuple_target = 4096);

-- ============================================================================
-- USAGE RECOMMENDATIONS:
--
-- ❌ SLOW (20+ minutes for 900K rows):
--    SELECT * FROM claims.service_line_flag;
--
-- ✅ FAST (for lists/dashboards):
--    SELECT * FROM claims.v_service_line_flag_list
--    WHERE flag_status = 'OPEN'
--    ORDER BY created_at DESC
--    LIMIT 1000;
--
-- ✅ FAST (for single flag detail):
--    SELECT * FROM claims.v_service_line_flag_detail
--    WHERE flag_id = 12345;
--
-- ✅ FAST (for time-range queries using BRIN):
--    SELECT * FROM claims.v_service_line_flag_list
--    WHERE created_at >= '2025-01-01'
--    ORDER BY created_at DESC;
--
-- ✅ FAST (count queries - no TEXT access):
--    SELECT flag_status, COUNT(*)
--    FROM claims.service_line_flag
--    GROUP BY flag_status;
-- ============================================================================
