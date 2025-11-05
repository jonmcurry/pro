-- PHASE 5: Streaming Processing - Progress Tracking
-- Real-time progress tracking for streaming file processing

-- Progress tracking table for real-time updates
CREATE TABLE IF NOT EXISTS staging.file_processing_progress (
    id BIGINT GENERATED ALWAYS AS IDENTITY (CACHE 100) PRIMARY KEY,
    queue_id BIGINT NOT NULL REFERENCES staging.file_processing_queue(queue_id) ON DELETE CASCADE,

    -- Claim counts
    total_claims INTEGER NOT NULL DEFAULT 0,
    processed_claims INTEGER NOT NULL DEFAULT 0,
    failed_claims INTEGER NOT NULL DEFAULT 0,

    -- Flag statistics
    flags_created INTEGER NOT NULL DEFAULT 0,
    critical_flags INTEGER NOT NULL DEFAULT 0,

    -- Timing information
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ,
    estimated_completion_at TIMESTAMPTZ,

    -- Performance metrics
    claims_per_second DECIMAL(10, 2),
    average_processing_time_ms INTEGER,

    -- Metadata
    is_active BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Create indexes for progress tracking
CREATE INDEX IF NOT EXISTS idx_progress_queue_id ON staging.file_processing_progress(queue_id);
CREATE INDEX IF NOT EXISTS idx_progress_active ON staging.file_processing_progress(is_active, updated_at DESC) WHERE is_active = true;

-- Add trigger to update updated_at timestamp
CREATE TRIGGER update_file_processing_progress_updated_at
    BEFORE UPDATE ON staging.file_processing_progress
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Failed claims table for streaming error handling
CREATE TABLE IF NOT EXISTS staging.failed_claims (
    id BIGINT GENERATED ALWAYS AS IDENTITY (CACHE 100) PRIMARY KEY,
    queue_id BIGINT NOT NULL REFERENCES staging.file_processing_queue(queue_id) ON DELETE CASCADE,
    progress_id BIGINT REFERENCES staging.file_processing_progress(id) ON DELETE CASCADE,

    -- Claim identification
    claim_number VARCHAR(50),
    subscriber_id_from_file VARCHAR(50),
    provider_npi VARCHAR(20),

    -- Error information
    error_message TEXT NOT NULL,
    error_type VARCHAR(100),
    stack_trace TEXT,

    -- Claim data (JSON for debugging)
    claim_data JSONB,

    -- Retry information
    retry_count INTEGER NOT NULL DEFAULT 0,
    last_retry_at TIMESTAMPTZ,
    can_retry BOOLEAN NOT NULL DEFAULT true,

    -- Metadata
    failed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Create indexes for failed claims
CREATE INDEX IF NOT EXISTS idx_failed_claims_queue_id ON staging.failed_claims(queue_id);
CREATE INDEX IF NOT EXISTS idx_failed_claims_progress_id ON staging.failed_claims(progress_id);
CREATE INDEX IF NOT EXISTS idx_failed_claims_can_retry ON staging.failed_claims(can_retry, retry_count) WHERE can_retry = true;
CREATE INDEX IF NOT EXISTS idx_failed_claims_error_type ON staging.failed_claims(error_type);

-- Add new queue statuses for streaming by updating the CHECK constraint
DO $$
BEGIN
    -- Drop existing constraint if it exists
    IF EXISTS (
        SELECT 1 FROM information_schema.constraint_column_usage
        WHERE table_schema = 'staging'
        AND table_name = 'file_processing_queue'
        AND constraint_name = 'valid_queue_status'
    ) THEN
        ALTER TABLE staging.file_processing_queue DROP CONSTRAINT valid_queue_status;
    END IF;

    -- Add updated constraint with new statuses
    ALTER TABLE staging.file_processing_queue ADD CONSTRAINT valid_queue_status CHECK (
        queue_status = ANY (ARRAY['QUEUED'::text, 'PROCESSING'::text, 'COMPLETED'::text, 'FAILED'::text, 'RETRY'::text, 'STREAMING'::text, 'PARTIAL_SUCCESS'::text])
    );
END $$;

-- Comments for documentation
COMMENT ON TABLE staging.file_processing_progress IS 'Real-time progress tracking for streaming file processing (PHASE 5)';
COMMENT ON COLUMN staging.file_processing_progress.total_claims IS 'Total claims in the file (may be estimated initially)';
COMMENT ON COLUMN staging.file_processing_progress.processed_claims IS 'Number of claims successfully processed';
COMMENT ON COLUMN staging.file_processing_progress.failed_claims IS 'Number of claims that failed processing';
COMMENT ON COLUMN staging.file_processing_progress.claims_per_second IS 'Real-time processing rate';
COMMENT ON COLUMN staging.file_processing_progress.estimated_completion_at IS 'Estimated completion time based on current processing rate';

COMMENT ON TABLE staging.failed_claims IS 'Individual claim failures during streaming processing (PHASE 5)';
COMMENT ON COLUMN staging.failed_claims.claim_data IS 'Full claim data as JSON for debugging and retry';
COMMENT ON COLUMN staging.failed_claims.can_retry IS 'Whether this claim can be retried (false for validation errors)';
