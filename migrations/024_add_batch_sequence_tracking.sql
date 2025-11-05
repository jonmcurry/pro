-- Migration 024: Add batch sequence tracking for strict FIFO ordering
-- Purpose: Enable Sequential Completion Manager for multi-worker pipeline
-- Architecture: Aegis-inspired sequence-controlled batch processing

-- Add batch_sequence_number to raw_claims for strict FIFO ordering
ALTER TABLE staging.raw_claims
ADD COLUMN IF NOT EXISTS batch_sequence_number INTEGER;

-- Index for sequence-based queries (used by workers to fetch batches)
CREATE INDEX IF NOT EXISTS idx_raw_claims_sequence_status
ON staging.raw_claims(batch_sequence_number, processing_status)
WHERE processing_status IN ('PENDING', 'PROCESSING', 'COMPLETED');

-- Table to track batch sequences (audit trail + coordination)
CREATE TABLE IF NOT EXISTS staging.batch_sequences (
    sequence_number INTEGER PRIMARY KEY,
    assigned_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMPTZ,

    -- Batch metadata
    batch_id BIGINT NOT NULL REFERENCES staging.import_batch(batch_id) ON DELETE CASCADE,
    claim_count INTEGER NOT NULL,
    processing_stage TEXT NOT NULL DEFAULT 'STAGE2',

    -- Worker tracking
    worker_id TEXT,

    -- Processing metrics
    processing_time_seconds REAL,
    success_count INTEGER DEFAULT 0,
    failure_count INTEGER DEFAULT 0,

    -- Error tracking
    errors JSONB,

    CONSTRAINT ck_batch_sequences_stage CHECK (
        processing_stage IN ('STAGE2', 'VALIDATION', 'RULES', 'COMPLETION')
    )
);

-- Index for finding incomplete sequences (monitoring)
CREATE INDEX IF NOT EXISTS idx_batch_sequences_incomplete
ON staging.batch_sequences(sequence_number)
WHERE completed_at IS NULL;

-- Index for performance analysis
CREATE INDEX IF NOT EXISTS idx_batch_sequences_performance
ON staging.batch_sequences(assigned_at, completed_at, processing_time_seconds);

-- Add comments
COMMENT ON TABLE staging.batch_sequences IS 'Tracks batch sequence numbers for strict FIFO ordering with multi-worker processing';
COMMENT ON COLUMN staging.batch_sequences.sequence_number IS 'Monotonic sequence number assigned by SequencedBatchAcquirer (1, 2, 3...)';
COMMENT ON COLUMN staging.batch_sequences.completed_at IS 'When batch was committed to production (set by SequentialCompletionManager)';
COMMENT ON COLUMN staging.raw_claims.batch_sequence_number IS 'Sequence number for FIFO ordering (assigned at batch acquisition time)';

-- Create view for monitoring sequence processing
CREATE OR REPLACE VIEW staging.vw_sequence_processing_status AS
SELECT
    bs.sequence_number,
    bs.assigned_at,
    bs.completed_at,
    bs.batch_id,
    b.file_path,
    bs.claim_count,
    bs.processing_stage,
    bs.worker_id,
    bs.processing_time_seconds,
    bs.success_count,
    bs.failure_count,

    -- Calculate wait time for incomplete sequences
    CASE
        WHEN bs.completed_at IS NULL THEN
            EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - bs.assigned_at))
        ELSE NULL
    END AS wait_time_seconds,

    -- Identify stuck sequences (waiting > 5 minutes)
    CASE
        WHEN bs.completed_at IS NULL AND bs.assigned_at < CURRENT_TIMESTAMP - INTERVAL '5 minutes' THEN
            TRUE
        ELSE FALSE
    END AS is_stuck,

    -- Count pending claims in this sequence
    (SELECT COUNT(*) FROM staging.raw_claims
     WHERE batch_sequence_number = bs.sequence_number
     AND processing_status = 'PENDING') AS pending_claims,

    -- Count processing claims in this sequence
    (SELECT COUNT(*) FROM staging.raw_claims
     WHERE batch_sequence_number = bs.sequence_number
     AND processing_status = 'PROCESSING') AS processing_claims,

    -- Count completed claims in this sequence
    (SELECT COUNT(*) FROM staging.raw_claims
     WHERE batch_sequence_number = bs.sequence_number
     AND processing_status = 'COMPLETED') AS completed_claims,

    -- Count failed claims in this sequence
    (SELECT COUNT(*) FROM staging.raw_claims
     WHERE batch_sequence_number = bs.sequence_number
     AND processing_status = 'FAILED') AS failed_claims

FROM staging.batch_sequences bs
JOIN staging.import_batch b ON bs.batch_id = b.batch_id
ORDER BY bs.sequence_number DESC;

COMMENT ON VIEW staging.vw_sequence_processing_status IS 'Monitor batch sequence processing status and identify stuck sequences';

-- Function to find the next expected sequence number (for Sequential Completion Manager)
CREATE OR REPLACE FUNCTION staging.get_next_expected_sequence()
RETURNS INTEGER
LANGUAGE plpgsql
AS $$
DECLARE
    next_seq INTEGER;
BEGIN
    -- Find the first sequence that hasn't been completed yet
    SELECT COALESCE(MIN(sequence_number), 1)
    INTO next_seq
    FROM staging.batch_sequences
    WHERE completed_at IS NULL;

    RETURN next_seq;
END;
$$;

COMMENT ON FUNCTION staging.get_next_expected_sequence IS 'Returns the next sequence number expected for completion (used by SequentialCompletionManager)';

-- Function to detect stuck sequences
CREATE OR REPLACE FUNCTION staging.detect_stuck_sequences(threshold_minutes INTEGER DEFAULT 5)
RETURNS TABLE (
    sequence_number INTEGER,
    assigned_at TIMESTAMPTZ,
    wait_time_minutes REAL,
    claim_count INTEGER,
    worker_id TEXT,
    pending_claims BIGINT,
    processing_claims BIGINT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        bs.sequence_number,
        bs.assigned_at,
        EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - bs.assigned_at))::REAL / 60.0 AS wait_time_minutes,
        bs.claim_count,
        bs.worker_id,
        (SELECT COUNT(*) FROM staging.raw_claims WHERE batch_sequence_number = bs.sequence_number AND processing_status = 'PENDING') AS pending_claims,
        (SELECT COUNT(*) FROM staging.raw_claims WHERE batch_sequence_number = bs.sequence_number AND processing_status = 'PROCESSING') AS processing_claims
    FROM staging.batch_sequences bs
    WHERE bs.completed_at IS NULL
    AND bs.assigned_at < CURRENT_TIMESTAMP - (threshold_minutes || ' minutes')::INTERVAL
    ORDER BY bs.sequence_number ASC;
END;
$$;

COMMENT ON FUNCTION staging.detect_stuck_sequences IS 'Detects sequences that have been waiting too long (default: 5 minutes) - indicates potential worker failure';

-- Add configuration table for FIFO mode
CREATE TABLE IF NOT EXISTS staging.processing_configuration (
    config_key TEXT PRIMARY KEY,
    config_value TEXT NOT NULL,
    description TEXT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Insert default FIFO mode configuration
INSERT INTO staging.processing_configuration (config_key, config_value, description)
VALUES
    ('fifo_mode', 'strict', 'FIFO ordering mode: strict (sequential completion), soft (best effort), or none'),
    ('stage2_worker_count', '8', 'Number of concurrent Stage 2 workers'),
    ('batch_size', '750', 'Claims per batch (optimal: 750 based on Aegis production data)')
ON CONFLICT (config_key) DO NOTHING;

COMMENT ON TABLE staging.processing_configuration IS 'Runtime configuration for claims processing pipeline';
