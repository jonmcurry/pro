-- Migration: 042_create_provider_enrichment_queue
-- Description: Create queue table for asynchronous NPI provider enrichment from CMS NPI Registry API
-- Date: 2025-11-05

-- Provider Enrichment Queue Table
-- Manages asynchronous enrichment of provider data from NPI Registry API
-- Ensures claims processing is never blocked by external API calls
CREATE TABLE IF NOT EXISTS claims.provider_enrichment_queue (
    queue_id BIGINT GENERATED ALWAYS AS IDENTITY (CACHE 100) PRIMARY KEY,
    provider_id BIGINT NOT NULL REFERENCES claims.provider(provider_id) ON DELETE CASCADE,
    npi VARCHAR(10) NOT NULL,

    -- Status tracking
    status VARCHAR(20) NOT NULL DEFAULT 'PENDING', -- PENDING, IN_PROGRESS, COMPLETED, FAILED
    priority INTEGER NOT NULL DEFAULT 5, -- 1-10 (10=highest priority, 1=lowest)
    retry_count INTEGER NOT NULL DEFAULT 0,
    max_retries INTEGER NOT NULL DEFAULT 3,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    next_retry_at TIMESTAMPTZ,

    -- Error tracking
    last_error TEXT,
    last_error_at TIMESTAMPTZ,

    -- API response cache (for audit trail and debugging)
    api_response JSONB,

    -- Ensure each provider is only queued once
    CONSTRAINT unique_provider_enrichment UNIQUE(provider_id),

    -- Validate status values
    CONSTRAINT valid_status CHECK (status IN ('PENDING', 'IN_PROGRESS', 'COMPLETED', 'FAILED')),

    -- Validate priority range
    CONSTRAINT valid_priority CHECK (priority BETWEEN 1 AND 10)
);

-- Index for finding pending/failed items to process
CREATE INDEX IF NOT EXISTS idx_enrichment_status_pending
    ON claims.provider_enrichment_queue(status, priority DESC, created_at ASC)
    WHERE status IN ('PENDING', 'FAILED');

-- Index for finding items ready to retry
CREATE INDEX IF NOT EXISTS idx_enrichment_retry
    ON claims.provider_enrichment_queue(next_retry_at, priority DESC)
    WHERE status = 'FAILED' AND next_retry_at IS NOT NULL;

-- Index for finding items by NPI
CREATE INDEX IF NOT EXISTS idx_enrichment_npi
    ON claims.provider_enrichment_queue(npi);

-- Index for finding recently completed items
CREATE INDEX IF NOT EXISTS idx_enrichment_completed
    ON claims.provider_enrichment_queue(completed_at DESC)
    WHERE status = 'COMPLETED';

-- Index for monitoring worker activity
CREATE INDEX IF NOT EXISTS idx_enrichment_in_progress
    ON claims.provider_enrichment_queue(started_at DESC)
    WHERE status = 'IN_PROGRESS';

COMMENT ON TABLE claims.provider_enrichment_queue IS 'Queue for asynchronous provider data enrichment from CMS NPI Registry API - ensures claims processing is never blocked by external API calls';
COMMENT ON COLUMN claims.provider_enrichment_queue.status IS 'Current processing status: PENDING (waiting), IN_PROGRESS (being processed), COMPLETED (successfully enriched), FAILED (failed after retries)';
COMMENT ON COLUMN claims.provider_enrichment_queue.priority IS 'Processing priority 1-10, where 10 is highest priority. Used to prioritize recently-seen providers over historical ones.';
COMMENT ON COLUMN claims.provider_enrichment_queue.retry_count IS 'Number of failed attempts. After max_retries, item is marked as permanently FAILED.';
COMMENT ON COLUMN claims.provider_enrichment_queue.next_retry_at IS 'Timestamp when failed item should be retried. Uses exponential backoff: 1hr, 2hr, 4hr.';
COMMENT ON COLUMN claims.provider_enrichment_queue.api_response IS 'Full NPI Registry API response stored as JSONB for audit trail and debugging. Contains provider demographics, taxonomies, and addresses.';

-- Trigger to automatically set updated_at when provider is enriched
CREATE OR REPLACE FUNCTION claims.on_provider_enrichment_completed()
RETURNS TRIGGER AS $$
BEGIN
    -- When enrichment completes successfully, update the provider's updated_at timestamp
    IF NEW.status = 'COMPLETED' AND OLD.status != 'COMPLETED' THEN
        UPDATE claims.provider
        SET updated_at = CURRENT_TIMESTAMP,
            updated_by = 'NPI_ENRICHMENT'
        WHERE provider_id = NEW.provider_id;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_provider_enrichment_completed ON claims.provider_enrichment_queue;
CREATE TRIGGER trigger_provider_enrichment_completed
    AFTER UPDATE ON claims.provider_enrichment_queue
    FOR EACH ROW
    WHEN (NEW.status = 'COMPLETED' AND OLD.status IS DISTINCT FROM 'COMPLETED')
    EXECUTE FUNCTION claims.on_provider_enrichment_completed();

COMMENT ON FUNCTION claims.on_provider_enrichment_completed() IS 'Automatically updates provider.updated_at timestamp when enrichment completes successfully';

-- View for monitoring enrichment queue health
CREATE OR REPLACE VIEW claims.v_enrichment_queue_summary AS
SELECT
    status,
    COUNT(*) as count,
    AVG(retry_count) as avg_retries,
    MIN(created_at) as oldest_item,
    MAX(created_at) as newest_item,
    COUNT(*) FILTER (WHERE retry_count >= max_retries) as permanently_failed
FROM claims.provider_enrichment_queue
GROUP BY status;

COMMENT ON VIEW claims.v_enrichment_queue_summary IS 'Summary view of enrichment queue status for monitoring and alerting';

-- View for recent failures
CREATE OR REPLACE VIEW claims.v_enrichment_recent_failures AS
SELECT
    q.npi,
    p.last_name,
    p.first_name,
    q.last_error,
    q.retry_count,
    q.max_retries,
    q.last_error_at,
    q.next_retry_at,
    q.created_at
FROM claims.provider_enrichment_queue q
JOIN claims.provider p ON q.provider_id = p.provider_id
WHERE q.status = 'FAILED'
ORDER BY q.last_error_at DESC
LIMIT 100;

COMMENT ON VIEW claims.v_enrichment_recent_failures IS 'Recent enrichment failures with provider details for debugging';
