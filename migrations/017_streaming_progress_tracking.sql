-- PHASE 5: Streaming Processing - Progress Tracking
-- Real-time progress tracking for streaming file processing

-- Progress tracking table for real-time updates
CREATE TABLE staging.file_processing_progress (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    queue_id UUID NOT NULL REFERENCES staging.file_processing_queue(id) ON DELETE CASCADE,

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
CREATE INDEX idx_progress_queue_id ON staging.file_processing_progress(queue_id);
CREATE INDEX idx_progress_active ON staging.file_processing_progress(is_active, updated_at DESC) WHERE is_active = true;

-- Add trigger to update updated_at timestamp
CREATE TRIGGER update_file_processing_progress_updated_at
    BEFORE UPDATE ON staging.file_processing_progress
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Failed claims table for streaming error handling
CREATE TABLE staging.failed_claims (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    queue_id UUID NOT NULL REFERENCES staging.file_processing_queue(id) ON DELETE CASCADE,
    progress_id UUID REFERENCES staging.file_processing_progress(id) ON DELETE CASCADE,

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
CREATE INDEX idx_failed_claims_queue_id ON staging.failed_claims(queue_id);
CREATE INDEX idx_failed_claims_progress_id ON staging.failed_claims(progress_id);
CREATE INDEX idx_failed_claims_can_retry ON staging.failed_claims(can_retry, retry_count) WHERE can_retry = true;
CREATE INDEX idx_failed_claims_error_type ON staging.failed_claims(error_type);

-- Add new queue statuses for streaming
ALTER TYPE staging.queue_status ADD VALUE IF NOT EXISTS 'STREAMING';
ALTER TYPE staging.queue_status ADD VALUE IF NOT EXISTS 'PARTIAL_SUCCESS';

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
