-- Migration: Phase 5 Performance Indexes
-- Purpose: Add indexes for cache population and queue processing optimization
-- Expected Impact: 30-50% reduction in cache population time
-- IMPORTANT: Uses CREATE INDEX CONCURRENTLY for zero-downtime deployment

-- ============================================================================
-- Service Line Indexes (Duplicate Detection)
-- ============================================================================
-- Used by: crates/pro-rules/src/rule_engine.rs:202-237 (populate_duplicate_checks)
-- Purpose: Speed up duplicate service line lookups for DuplicateServiceRule
-- Impact: Most critical index for Phase 3/4 cache population

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_service_line_duplicate_lookup
ON claims.service_line (procedure_code, service_date_from, rendering_provider_id)
WHERE service_date_from IS NOT NULL
  AND is_active = true;

COMMENT ON INDEX claims.idx_service_line_duplicate_lookup IS
'Phase 5: Optimizes duplicate service line detection in rule cache population. Used by DuplicateServiceRule.';

-- Additional index for date-range queries (encounter history)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_service_line_date_range
ON claims.service_line (service_date_from DESC, service_date_to)
WHERE service_date_from IS NOT NULL
  AND is_active = true;

COMMENT ON INDEX claims.idx_service_line_date_range IS
'Phase 5: Optimizes date-range queries for service line history lookups.';

-- ============================================================================
-- Provider Indexes (Provider Credential Lookups)
-- ============================================================================
-- Used by: crates/pro-rules/src/rule_engine.rs:283-330 (populate_provider_info)
-- Purpose: Speed up provider credential lookups by NPI
-- Impact: Critical for provider validation rules

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_provider_npi_lookup
ON claims.provider (npi)
WHERE is_active = true
  AND npi IS NOT NULL;

COMMENT ON INDEX claims.idx_provider_npi_lookup IS
'Phase 5: Optimizes provider lookups by NPI in cache population.';

-- Index for provider specialty lookups (for future ML features)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_provider_specialty
ON claims.provider (specialty_code, provider_type)
WHERE is_active = true
  AND specialty_code IS NOT NULL;

COMMENT ON INDEX claims.idx_provider_specialty IS
'Phase 5: Optimizes provider specialty lookups for analytics and ML features.';

-- ============================================================================
-- Encounter Indexes (Subscriber History)
-- ============================================================================
-- Used by: crates/pro-rules/src/rule_engine.rs:332-380 (populate_encounter_history)
-- Purpose: Speed up encounter history lookups by subscriber
-- Impact: Important for temporal pattern detection

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_encounter_subscriber_history
ON claims.encounter (subscriber_id, date_of_service_from DESC)
WHERE is_active = true
  AND soft_deleted = false;

COMMENT ON INDEX claims.idx_encounter_subscriber_history IS
'Phase 5: Optimizes encounter history lookups by subscriber for temporal pattern detection.';

-- Composite index for claim status queries (dashboard performance)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_encounter_status_date
ON claims.encounter (claim_status, date_of_service_from DESC, organization_id)
WHERE is_active = true
  AND soft_deleted = false;

COMMENT ON INDEX claims.idx_encounter_status_date IS
'Phase 5: Optimizes claim status queries for dashboards and reporting.';

-- Index for facility-based queries (common filter)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_encounter_facility_date
ON claims.encounter (facility_id, date_of_service_from DESC)
WHERE is_active = true
  AND soft_deleted = false;

COMMENT ON INDEX claims.idx_encounter_facility_date IS
'Phase 5: Optimizes facility-based encounter queries.';

-- ============================================================================
-- Queue Processing Indexes (FIFO Optimization)
-- ============================================================================
-- Used by: crates/pro-worker/src/queue_manager.rs:140, 172 (dequeue operations)
-- Purpose: Speed up FIFO queue processing with proper ordering
-- Impact: Critical for queue manager performance

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_queue_global_fifo
ON staging.file_processing_queue (priority ASC, queued_at ASC)
WHERE queue_status IN ('QUEUED', 'RETRY');

COMMENT ON INDEX staging.idx_queue_global_fifo IS
'Phase 5: Optimizes global FIFO queue processing with priority support.';

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_queue_facility_fifo
ON staging.file_processing_queue (facility_id, priority ASC, queued_at ASC)
WHERE queue_status IN ('QUEUED', 'RETRY');

COMMENT ON INDEX staging.idx_queue_facility_fifo IS
'Phase 5: Optimizes per-facility FIFO queue processing with priority support.';

-- Index for queue status monitoring
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_queue_status_monitoring
ON staging.file_processing_queue (queue_status, queued_at DESC)
WHERE queue_status = 'PROCESSING';

COMMENT ON INDEX staging.idx_queue_status_monitoring IS
'Phase 5: Optimizes queue status monitoring queries for currently processing files.';

-- ============================================================================
-- Diagnosis Indexes (Rule Evaluation)
-- ============================================================================
-- Used by: Rule engine for diagnosis code validation
-- Purpose: Speed up diagnosis code lookups and validation
-- Impact: Moderate - improves diagnosis-related rule performance

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_encounter_diagnosis_code
ON claims.encounter_diagnosis (diagnosis_code, encounter_id)
WHERE diagnosis_code IS NOT NULL;

COMMENT ON INDEX claims.idx_encounter_diagnosis_code IS
'Phase 5: Optimizes diagnosis code lookups for validation rules.';

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_encounter_diagnosis_sequence
ON claims.encounter_diagnosis (encounter_id, sequence_number, is_principal)
WHERE is_principal = true;

COMMENT ON INDEX claims.idx_encounter_diagnosis_sequence IS
'Phase 5: Optimizes principal diagnosis lookups.';

-- ============================================================================
-- Flag Indexes (Rule Result Queries)
-- ============================================================================
-- Used by: Dashboard queries and flag management
-- Purpose: Speed up flag queries by encounter and status
-- Impact: Moderate - improves dashboard and reporting performance

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_flag_encounter_severity
ON claims.flag (encounter_id, severity_level, flag_status)
WHERE flag_status IN ('ACTIVE', 'PENDING');

COMMENT ON INDEX claims.idx_flag_encounter_severity IS
'Phase 5: Optimizes flag queries by encounter and severity.';

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_flag_service_line_severity
ON claims.flag (service_line_id, severity_level, flag_status)
WHERE service_line_id IS NOT NULL
  AND flag_status IN ('ACTIVE', 'PENDING');

COMMENT ON INDEX claims.idx_flag_service_line_severity IS
'Phase 5: Optimizes flag queries by service line and severity.';

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_flag_category_date
ON claims.flag (flag_category, created_at DESC)
WHERE flag_status IN ('ACTIVE', 'PENDING');

COMMENT ON INDEX claims.idx_flag_category_date IS
'Phase 5: Optimizes flag category queries for reporting.';

-- ============================================================================
-- Import Batch Indexes (Job Processing)
-- ============================================================================
-- Used by: Job status tracking and file processing
-- Purpose: Speed up batch status queries
-- Impact: Low - improves job monitoring performance

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_import_batch_status_date
ON staging.import_batch (status, created_at DESC)
WHERE status IN ('PENDING', 'PROCESSING');

COMMENT ON INDEX staging.idx_import_batch_status_date IS
'Phase 5: Optimizes import batch status queries for monitoring.';

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_import_batch_org_date
ON staging.import_batch (organization_id, created_at DESC);

COMMENT ON INDEX staging.idx_import_batch_org_date IS
'Phase 5: Optimizes organization-based batch queries.';

-- ============================================================================
-- Verification and Statistics
-- ============================================================================

-- Analyze tables to update statistics after index creation
ANALYZE claims.service_line;
ANALYZE claims.provider;
ANALYZE claims.encounter;
ANALYZE claims.encounter_diagnosis;
ANALYZE claims.flag;
ANALYZE staging.file_processing_queue;
ANALYZE staging.import_batch;

-- Report index sizes for monitoring
DO $$
DECLARE
    index_record RECORD;
    total_size BIGINT := 0;
BEGIN
    RAISE NOTICE '=== Phase 5 Index Size Report ===';

    FOR index_record IN
        SELECT
            schemaname,
            tablename,
            indexname,
            pg_size_pretty(pg_relation_size(indexname::regclass)) as size,
            pg_relation_size(indexname::regclass) as bytes
        FROM pg_indexes
        WHERE schemaname IN ('claims', 'staging')
          AND indexname LIKE 'idx_%'
        ORDER BY pg_relation_size(indexname::regclass) DESC
    LOOP
        total_size := total_size + index_record.bytes;
        RAISE NOTICE 'Index: %.% - Size: %',
            index_record.tablename,
            index_record.indexname,
            index_record.size;
    END LOOP;

    RAISE NOTICE 'Total index size: %', pg_size_pretty(total_size);
END $$;

-- ============================================================================
-- Migration Complete
-- ============================================================================

-- Log completion
INSERT INTO audit.schema_version_log (version, description, applied_at)
VALUES ('016', 'Phase 5: Performance indexes for cache population and queue processing', CURRENT_TIMESTAMP)
ON CONFLICT DO NOTHING;
