-- Migration: 073_create_rules_processing_queue
-- Description: Create queue for deferred rules processing
-- Date: 2025-01-15
-- Purpose: Enable high-throughput import by deferring rules execution to background

-- ============================================================================
-- RULES PROCESSING QUEUE
-- ============================================================================
-- When DEFER_RULES_EXECUTION=true, encounters are queued here for background
-- rule processing. This separates import throughput from rules processing.

CREATE TABLE IF NOT EXISTS staging.rules_processing_queue (
    queue_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    encounter_id BIGINT NOT NULL,
    organization_id BIGINT NOT NULL,
    batch_id BIGINT,
    status VARCHAR(20) NOT NULL DEFAULT 'PENDING',
    priority INTEGER NOT NULL DEFAULT 5,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    flags_created INTEGER,
    error_message TEXT,
    worker_id VARCHAR(50),

    CONSTRAINT fk_rules_queue_encounter
        FOREIGN KEY (encounter_id)
        REFERENCES claims.encounter(encounter_id)
        ON DELETE CASCADE,
    CONSTRAINT fk_rules_queue_organization
        FOREIGN KEY (organization_id)
        REFERENCES claims.organization(organization_id)
        ON DELETE CASCADE
);

-- Index for efficient queue polling
CREATE INDEX IF NOT EXISTS idx_rules_queue_status_priority
ON staging.rules_processing_queue(status, priority, created_at)
WHERE status = 'PENDING';

-- Index for finding stale processing items
CREATE INDEX IF NOT EXISTS idx_rules_queue_processing
ON staging.rules_processing_queue(status, started_at)
WHERE status = 'PROCESSING';

-- Index for encounter lookup
CREATE INDEX IF NOT EXISTS idx_rules_queue_encounter
ON staging.rules_processing_queue(encounter_id);

COMMENT ON TABLE staging.rules_processing_queue IS
'Queue for deferred rules processing. Enables high-throughput import by separating data ingestion from rule execution.';

COMMENT ON COLUMN staging.rules_processing_queue.status IS
'PENDING=awaiting processing, PROCESSING=being processed, COMPLETED=done, FAILED=error occurred';

-- ============================================================================
-- QUEUE MANAGEMENT FUNCTIONS
-- ============================================================================

-- Function to enqueue an encounter for rules processing
CREATE OR REPLACE FUNCTION staging.enqueue_for_rules_processing(
    p_encounter_id BIGINT,
    p_organization_id BIGINT,
    p_batch_id BIGINT DEFAULT NULL,
    p_priority INTEGER DEFAULT 5
)
RETURNS BIGINT AS $$
DECLARE
    v_queue_id BIGINT;
BEGIN
    INSERT INTO staging.rules_processing_queue (
        encounter_id, organization_id, batch_id, priority
    )
    VALUES (p_encounter_id, p_organization_id, p_batch_id, p_priority)
    ON CONFLICT DO NOTHING
    RETURNING queue_id INTO v_queue_id;

    RETURN v_queue_id;
END;
$$ LANGUAGE plpgsql;

-- Function to acquire next batch of encounters for processing
CREATE OR REPLACE FUNCTION staging.acquire_rules_processing_batch(
    p_worker_id VARCHAR(50),
    p_batch_size INTEGER DEFAULT 100
)
RETURNS TABLE (
    queue_id BIGINT,
    encounter_id BIGINT,
    organization_id BIGINT
) AS $$
BEGIN
    RETURN QUERY
    WITH acquired AS (
        UPDATE staging.rules_processing_queue rpq
        SET status = 'PROCESSING',
            started_at = CURRENT_TIMESTAMP,
            worker_id = p_worker_id
        WHERE rpq.queue_id IN (
            SELECT q.queue_id
            FROM staging.rules_processing_queue q
            WHERE q.status = 'PENDING'
            ORDER BY q.priority DESC, q.created_at ASC
            LIMIT p_batch_size
            FOR UPDATE SKIP LOCKED
        )
        RETURNING rpq.queue_id, rpq.encounter_id, rpq.organization_id
    )
    SELECT * FROM acquired;
END;
$$ LANGUAGE plpgsql;

-- Function to mark queue item as completed
CREATE OR REPLACE FUNCTION staging.complete_rules_processing(
    p_queue_id BIGINT,
    p_flags_created INTEGER
)
RETURNS VOID AS $$
BEGIN
    UPDATE staging.rules_processing_queue
    SET status = 'COMPLETED',
        completed_at = CURRENT_TIMESTAMP,
        flags_created = p_flags_created
    WHERE queue_id = p_queue_id;
END;
$$ LANGUAGE plpgsql;

-- Function to mark queue item as failed
CREATE OR REPLACE FUNCTION staging.fail_rules_processing(
    p_queue_id BIGINT,
    p_error_message TEXT
)
RETURNS VOID AS $$
BEGIN
    UPDATE staging.rules_processing_queue
    SET status = 'FAILED',
        completed_at = CURRENT_TIMESTAMP,
        error_message = p_error_message
    WHERE queue_id = p_queue_id;
END;
$$ LANGUAGE plpgsql;

-- Function to recover stale processing items (stuck > 5 minutes)
CREATE OR REPLACE FUNCTION staging.recover_stale_rules_processing()
RETURNS INTEGER AS $$
DECLARE
    v_recovered INTEGER;
BEGIN
    UPDATE staging.rules_processing_queue
    SET status = 'PENDING',
        started_at = NULL,
        worker_id = NULL
    WHERE status = 'PROCESSING'
      AND started_at < CURRENT_TIMESTAMP - INTERVAL '5 minutes';

    GET DIAGNOSTICS v_recovered = ROW_COUNT;
    RETURN v_recovered;
END;
$$ LANGUAGE plpgsql;
