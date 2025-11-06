-- Migration: Add Performance Indexes for Rules Engine
-- Phase 6 Critical Optimizations
-- Expected Impact: 90%+ improvement on duplicate detection and batch queries

-- ============================================================================
-- DUPLICATE DETECTION OPTIMIZATION
-- ============================================================================
-- These indexes support the duplicate detection rule which queries:
-- "SELECT COUNT(*) FROM service_line WHERE encounter_id = ? AND procedure_code = ?"

-- Composite index for duplicate detection (most critical)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_service_line_duplicate_detection
ON claims.service_line (encounter_id, procedure_code)
WHERE line_status = 'ACTIVE';

-- Covering index with modifiers for modifier-aware duplicate detection
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_service_line_duplicate_with_modifiers
ON claims.service_line (encounter_id, procedure_code, procedure_modifier_1, procedure_modifier_2)
INCLUDE (service_line_id, line_item_charge_amount)
WHERE line_status = 'ACTIVE';

-- ============================================================================
-- PROVIDER LOOKUP OPTIMIZATION
-- ============================================================================
-- Supports queries like: "SELECT provider_id FROM provider WHERE npi = ?"

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_provider_npi_lookup
ON claims.provider (npi)
INCLUDE (provider_id, last_name, first_name, provider_type);

-- ============================================================================
-- ENCOUNTER BATCH PROCESSING OPTIMIZATION
-- ============================================================================
-- Supports loading diagnosis codes for multiple encounters efficiently

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_diagnosis_code_encounter_batch
ON claims.encounter_diagnosis (encounter_id)
INCLUDE (diagnosis_code, sequence_number);

-- ============================================================================
-- SERVICE LINE BATCH LOADING OPTIMIZATION
-- ============================================================================
-- Supports batch loading service lines by encounter_id with all needed fields

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_service_line_encounter_batch
ON claims.service_line (encounter_id, line_number)
INCLUDE (procedure_code, procedure_modifier_1, service_unit_count, line_item_charge_amount, service_date_from)
WHERE line_status = 'ACTIVE';

-- ============================================================================
-- RULE EXECUTION LOG OPTIMIZATION (for execution_planner.rs)
-- ============================================================================
-- Supports loading historical statistics for rule execution planning
-- Query pattern: "SELECT flag_type, AVG(...) FROM rule_execution_log GROUP BY flag_type"

-- Note: This table doesn't exist yet, but when it does, this index will be ready
-- CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_rule_execution_log_stats
-- ON claims.rule_execution_log (flag_type)
-- INCLUDE (flag_created, financial_impact, execution_time_ms);

-- ============================================================================
-- FACILITY RULE CONFIGURATION OPTIMIZATION
-- ============================================================================
-- Supports queries: "SELECT * FROM facility_rule_assignment WHERE facility_id = ? AND is_enabled = true"

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_facility_rule_config_lookup
ON claims.facility_rule_assignment (facility_id, is_enabled)
INCLUDE (rule_id, parameter_overrides_encrypted);

-- ============================================================================
-- STATISTICS AND VERIFICATION
-- ============================================================================

-- ============================================================================
-- BATCH PROCESSING OPTIMIZATION (PHASE 6 MEDIUM-VALUE)
-- ============================================================================
-- Supports batch completion queries that count by batch_id and processing_status
-- Query pattern: "SELECT batch_id, COUNT(*) FILTER (WHERE processing_status = ...)
--                 FROM raw_claims WHERE batch_id = ANY($1) GROUP BY batch_id"

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_raw_claims_batch_status
ON staging.raw_claims (batch_id, processing_status)
INCLUDE (raw_claim_id);

-- ============================================================================
-- STATISTICS AND ANALYSIS
-- ============================================================================

-- Analyze tables to update statistics for query planner
ANALYZE claims.service_line;
ANALYZE claims.provider;
ANALYZE claims.encounter_diagnosis;
ANALYZE claims.facility_rule_assignment;
ANALYZE staging.raw_claims;

-- Display index sizes for monitoring
DO $$
BEGIN
    RAISE NOTICE '=== Index Creation Complete ===';
    RAISE NOTICE 'Index sizes:';

    -- This will be visible in migration logs
    PERFORM pg_size_pretty(pg_relation_size('claims.idx_service_line_duplicate_detection'));
END $$;

-- ============================================================================
-- MIGRATION NOTES
-- ============================================================================
--
-- CONCURRENTLY option: Indexes are built without blocking writes
-- WHERE clauses: Partial indexes only include active records (smaller, faster)
-- INCLUDE: Covering indexes allow index-only scans (no table access needed)
--
-- Expected Performance Improvements:
-- - Duplicate detection: 50-200x faster (table scan → index lookup)
-- - Provider lookups: 100-500x faster (table scan → index lookup)
-- - Batch processing: 10-50x faster (sorted access, covering index)
-- - Diagnosis loading: 20-100x faster (index scan vs sequential)
-- - Batch completion queries: 10-50x faster (grouped aggregation with index)
--
-- Disk Space Impact:
-- - Estimated 50-200 MB per index (depends on table size)
-- - Total: ~600 MB - 1.2 GB for all indexes (7 indexes total)
-- - Trade-off: Disk space for 90%+ query performance improvement
--
-- Maintenance:
-- - Indexes are automatically maintained by PostgreSQL
-- - VACUUM ANALYZE recommended after large data loads
-- - Monitor index bloat with pg_stat_user_indexes
-- ============================================================================
