-- Migration 065: Expression indexes for CTE-based batch acquisition
-- Problem: CTE batch acquisition query taking 2.5+ seconds per batch
-- Root cause: No indexes on JSONB expressions used in GROUP BY, JOIN, and ORDER BY
--
-- The acquire_next_batch() CTE query uses these JSONB expressions:
--   - encounter_fields->>'patient_control_number' (GROUP BY, JOIN, ORDER BY)
--   - encounter_fields->>'date_of_service_from' (GROUP BY, JOIN)
--   - MIN(ingested_at) with ORDER BY for FIFO processing
--
-- PostgreSQL best practice: Create expression indexes for JSONB field access
-- See: https://www.postgresql.org/docs/current/indexes-expressional.html

-- ============================================================================
-- EXPRESSION INDEXES FOR BATCH ACQUISITION CTE
-- ============================================================================

-- Index 1: Expression index on patient_control_number for grouping and joining
-- This index enables index scans on the JSONB expression instead of full table scans
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_raw_claims_pcn_expr
ON staging.raw_claims ((encounter_fields->>'patient_control_number'))
WHERE processing_status = 'PENDING' AND batch_sequence_number IS NULL;

-- Index 2: Expression index on date_of_service_from for grouping and joining
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_raw_claims_dos_expr
ON staging.raw_claims ((encounter_fields->>'date_of_service_from'))
WHERE processing_status = 'PENDING' AND batch_sequence_number IS NULL;

-- Index 3: Composite expression index for the encounter grouping in CTE
-- This index covers both GROUP BY columns and allows efficient aggregate operations
-- The ingested_at column enables the FIFO ordering without a separate sort step
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_raw_claims_encounter_fifo
ON staging.raw_claims (
    (encounter_fields->>'patient_control_number'),
    (encounter_fields->>'date_of_service_from'),
    ingested_at ASC
)
WHERE processing_status = 'PENDING' AND batch_sequence_number IS NULL;

-- Index 4: Expression index for NULL checks on JSONB fields
-- The CTE uses IS NOT NULL filters on both JSONB expressions
-- This partial index pre-filters rows where both fields have values
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_raw_claims_encounter_notnull
ON staging.raw_claims (ingested_at ASC)
WHERE processing_status = 'PENDING'
  AND batch_sequence_number IS NULL
  AND encounter_fields->>'patient_control_number' IS NOT NULL
  AND encounter_fields->>'date_of_service_from' IS NOT NULL;

-- ============================================================================
-- ANALYZE TABLE FOR QUERY PLANNER STATISTICS
-- ============================================================================
-- Update statistics after index creation for optimal query planning
ANALYZE staging.raw_claims;

-- ============================================================================
-- DOCUMENTATION
-- ============================================================================
COMMENT ON INDEX staging.idx_raw_claims_pcn_expr IS
'Expression index for batch acquisition CTE - enables index scan on patient_control_number JSONB extraction';

COMMENT ON INDEX staging.idx_raw_claims_dos_expr IS
'Expression index for batch acquisition CTE - enables index scan on date_of_service_from JSONB extraction';

COMMENT ON INDEX staging.idx_raw_claims_encounter_fifo IS
'Composite expression index for batch acquisition CTE - covers GROUP BY and ORDER BY for FIFO processing';

COMMENT ON INDEX staging.idx_raw_claims_encounter_notnull IS
'Partial index for batch acquisition CTE - pre-filters valid encounter records for FIFO ordering';
