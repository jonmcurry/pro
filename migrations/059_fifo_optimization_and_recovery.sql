-- Migration 059: FIFO optimization indexes and stuck sequence recovery support
-- Identified in code review 2025-11-26
-- Purpose: Improve FIFO processing performance and enable automatic recovery

-- ============================================================================
-- FIFO PERFORMANCE INDEXES
-- ============================================================================

-- GAP 1.1: Efficient sequence-based claim fetching for workers
-- Current index includes NULL values inefficiently
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_raw_claims_sequence_processing
ON staging.raw_claims(batch_sequence_number ASC, processing_status)
WHERE batch_sequence_number IS NOT NULL
  AND processing_status IN ('PROCESSING', 'COMPLETED');

-- GAP 1.4: FIFO compliance verification
-- Enables efficient querying of claims processed out of order
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_raw_claims_ingested_processed
ON staging.raw_claims(ingested_at ASC, processed_at ASC)
WHERE processing_status IN ('COMPLETED', 'FAILED');

-- ============================================================================
-- DENIAL QUERY OPTIMIZATION
-- ============================================================================

-- GAP 2.1: Coder-based denial queries with date ordering
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_denial_coder_date_desc
ON claims.denial_event(coder_id, denial_date DESC)
WHERE denial_status != 'RESOLVED';

-- GAP 2.1: Provider-based denial queries with date ordering
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_denial_provider_date_desc
ON claims.denial_event(provider_id, denial_date DESC)
WHERE denial_status != 'RESOLVED';

-- ============================================================================
-- RULE STATS OPTIMIZATION
-- ============================================================================

-- GAP 2.2: Rule stats trending by facility
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_rule_stats_facility_date
ON claims.rule_execution_stats(facility_id, stat_date DESC)
WHERE execution_count > 0;

-- ============================================================================
-- STUCK SEQUENCE RECOVERY SUPPORT
-- ============================================================================

-- Update the processing_stage constraint to include 'RECOVERY' status
-- This allows marking stuck sequences as recovered for audit trail
ALTER TABLE staging.batch_sequences
DROP CONSTRAINT IF EXISTS ck_batch_sequences_stage;

ALTER TABLE staging.batch_sequences
ADD CONSTRAINT ck_batch_sequences_stage CHECK (
    processing_stage IN ('STAGE2', 'VALIDATION', 'RULES', 'COMPLETION', 'RECOVERY')
);

-- Add index for monitoring stuck sequences recovery
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_batch_sequences_recovery
ON staging.batch_sequences(completed_at DESC, processing_stage)
WHERE processing_stage = 'RECOVERY';

-- Function to recover all stuck sequences (manual trigger)
CREATE OR REPLACE FUNCTION staging.recover_stuck_sequences(threshold_minutes INTEGER DEFAULT 5)
RETURNS TABLE (
    sequence_number INTEGER,
    claims_reset BIGINT,
    wait_time_seconds INTEGER
)
LANGUAGE plpgsql
AS $$
DECLARE
    rec RECORD;
    reset_count BIGINT;
BEGIN
    FOR rec IN
        SELECT bs.sequence_number, bs.assigned_at, bs.claim_count
        FROM staging.batch_sequences bs
        WHERE bs.completed_at IS NULL
        AND bs.assigned_at < CURRENT_TIMESTAMP - (threshold_minutes || ' minutes')::INTERVAL
        ORDER BY bs.sequence_number ASC
    LOOP
        -- Reset claims in this sequence
        UPDATE staging.raw_claims
        SET processing_status = 'PENDING',
            batch_sequence_number = NULL,
            updated_at = CURRENT_TIMESTAMP
        WHERE batch_sequence_number = rec.sequence_number
        AND processing_status IN ('PENDING', 'PROCESSING');

        GET DIAGNOSTICS reset_count = ROW_COUNT;

        -- Mark sequence as recovered
        UPDATE staging.batch_sequences
        SET completed_at = CURRENT_TIMESTAMP,
            processing_stage = 'RECOVERY',
            errors = jsonb_build_object(
                'recovery_reason', 'manual_recovery',
                'original_claim_count', rec.claim_count,
                'claims_reset', reset_count,
                'recovered_at', CURRENT_TIMESTAMP
            )
        WHERE staging.batch_sequences.sequence_number = rec.sequence_number;

        -- Return info about this recovery
        sequence_number := rec.sequence_number;
        claims_reset := reset_count;
        wait_time_seconds := EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - rec.assigned_at))::INTEGER;
        RETURN NEXT;
    END LOOP;
END;
$$;

COMMENT ON FUNCTION staging.recover_stuck_sequences IS
'Manually recover all stuck sequences. Call with: SELECT * FROM staging.recover_stuck_sequences(5);
The threshold_minutes parameter sets how long a sequence must be waiting to be considered stuck.';

-- View for monitoring stuck sequence recovery history
CREATE OR REPLACE VIEW staging.vw_sequence_recovery_history AS
SELECT
    sequence_number,
    assigned_at,
    completed_at,
    batch_id,
    claim_count,
    worker_id,
    errors->>'recovery_reason' AS recovery_reason,
    (errors->>'claims_reset')::INTEGER AS claims_reset,
    (errors->>'wait_time_seconds')::INTEGER AS wait_time_seconds,
    errors->>'recovered_at' AS recovered_at
FROM staging.batch_sequences
WHERE processing_stage = 'RECOVERY'
ORDER BY completed_at DESC;

COMMENT ON VIEW staging.vw_sequence_recovery_history IS
'History of recovered stuck sequences for monitoring and debugging worker failures';
