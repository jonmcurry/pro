-- Migration 015: Create FIFO Processing Queue
-- This migration adds facility-aware FIFO processing capabilities

-- ============================================================================
-- File Processing Queue
-- ============================================================================
-- Ensures files are processed in chronological order per facility

CREATE TABLE IF NOT EXISTS staging.file_processing_queue (
    queue_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    facility_id UUID NOT NULL REFERENCES claims.facility(facility_id),
    import_batch_id UUID NOT NULL REFERENCES staging.import_batch(import_batch_id),
    file_path TEXT NOT NULL,
    file_hash TEXT NOT NULL,
    file_format TEXT NOT NULL,
    organization_id UUID NOT NULL REFERENCES claims.organization(organization_id),

    -- FIFO ordering
    queued_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    processing_started_at TIMESTAMPTZ,
    processing_completed_at TIMESTAMPTZ,

    -- Status tracking
    queue_status TEXT NOT NULL DEFAULT 'QUEUED',
    -- Valid values: QUEUED, PROCESSING, COMPLETED, FAILED, RETRY

    -- Priority (lower number = higher priority, default = 100)
    priority INTEGER NOT NULL DEFAULT 100,

    -- Retry handling
    retry_count INTEGER NOT NULL DEFAULT 0,
    max_retries INTEGER NOT NULL DEFAULT 3,
    last_error TEXT,

    -- Audit trail
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_by TEXT DEFAULT 'SYSTEM',
    updated_by TEXT DEFAULT 'SYSTEM',

    -- Constraints
    CONSTRAINT valid_queue_status CHECK (queue_status IN ('QUEUED', 'PROCESSING', 'COMPLETED', 'FAILED', 'RETRY')),
    CONSTRAINT valid_priority CHECK (priority >= 0 AND priority <= 1000),
    CONSTRAINT valid_retry_count CHECK (retry_count >= 0 AND retry_count <= max_retries)
);

-- ============================================================================
-- Indexes for FIFO Processing
-- ============================================================================

-- Primary index for FIFO retrieval by facility
-- Returns oldest queued file for a facility
CREATE INDEX idx_queue_fifo_by_facility
ON staging.file_processing_queue (facility_id, priority ASC, queued_at ASC)
WHERE queue_status = 'QUEUED';

-- Index for global FIFO (across all facilities)
CREATE INDEX idx_queue_fifo_global
ON staging.file_processing_queue (priority ASC, queued_at ASC)
WHERE queue_status = 'QUEUED';

-- Index for finding currently processing jobs
CREATE INDEX idx_queue_processing
ON staging.file_processing_queue (queue_status, processing_started_at DESC)
WHERE queue_status = 'PROCESSING';

-- Index for finding failed jobs
CREATE INDEX idx_queue_failed
ON staging.file_processing_queue (queue_status, queued_at DESC)
WHERE queue_status = 'FAILED';

-- Index for retry jobs
CREATE INDEX idx_queue_retry
ON staging.file_processing_queue (queue_status, retry_count ASC, queued_at ASC)
WHERE queue_status = 'RETRY' AND retry_count < max_retries;

-- Index for facility statistics
CREATE INDEX idx_queue_facility_stats
ON staging.file_processing_queue (facility_id, queue_status, created_at DESC);

-- Index for organization-level queries
CREATE INDEX idx_queue_organization
ON staging.file_processing_queue (organization_id, queue_status, queued_at ASC);

-- ============================================================================
-- Service Date Index on Encounter
-- ============================================================================
-- Ensures efficient queries for FIFO validation

CREATE INDEX IF NOT EXISTS idx_encounter_service_date_facility
ON claims.encounter (facility_id, date_of_service_from ASC, import_date ASC)
WHERE is_active = true;

-- Index for finding latest processed claim by facility
CREATE INDEX IF NOT EXISTS idx_encounter_import_date_facility
ON claims.encounter (facility_id, import_date DESC, date_of_service_from DESC)
WHERE is_active = true;

-- ============================================================================
-- Monitoring Views
-- ============================================================================

-- Queue health monitoring view
CREATE OR REPLACE VIEW staging.v_queue_health AS
SELECT
    f.facility_id,
    f.facility_code,
    f.facility_name,
    o.organization_name,
    COUNT(*) FILTER (WHERE q.queue_status = 'QUEUED') as queued_count,
    COUNT(*) FILTER (WHERE q.queue_status = 'PROCESSING') as processing_count,
    COUNT(*) FILTER (WHERE q.queue_status = 'COMPLETED') as completed_count,
    COUNT(*) FILTER (WHERE q.queue_status = 'FAILED') as failed_count,
    COUNT(*) FILTER (WHERE q.queue_status = 'RETRY') as retry_count,
    MIN(q.queued_at) FILTER (WHERE q.queue_status = 'QUEUED') as oldest_queued,
    MAX(q.queued_at) FILTER (WHERE q.queue_status = 'QUEUED') as newest_queued,
    AVG(EXTRACT(EPOCH FROM (q.processing_completed_at - q.processing_started_at)))
        FILTER (WHERE q.queue_status = 'COMPLETED') as avg_processing_seconds,
    MAX(EXTRACT(EPOCH FROM (q.processing_completed_at - q.processing_started_at)))
        FILTER (WHERE q.queue_status = 'COMPLETED') as max_processing_seconds
FROM claims.facility f
JOIN claims.organization o ON f.organization_id = o.organization_id
LEFT JOIN staging.file_processing_queue q ON f.facility_id = q.facility_id
    AND q.created_at > CURRENT_TIMESTAMP - INTERVAL '24 hours'
WHERE f.is_active = true
GROUP BY f.facility_id, f.facility_code, f.facility_name, o.organization_name
ORDER BY queued_count DESC, facility_code;

COMMENT ON VIEW staging.v_queue_health IS 'Real-time view of file processing queue health by facility (last 24 hours)';

-- FIFO violation detection view
CREATE OR REPLACE VIEW claims.v_fifo_violations AS
SELECT
    e1.encounter_id as earlier_encounter_id,
    e1.patient_control_number as earlier_pcn,
    e1.date_of_service_from as earlier_service_date,
    e1.import_date as earlier_import_date,
    e2.encounter_id as later_encounter_id,
    e2.patient_control_number as later_pcn,
    e2.date_of_service_from as later_service_date,
    e2.import_date as later_import_date,
    e1.facility_id,
    f.facility_code,
    f.facility_name,
    EXTRACT(EPOCH FROM (e1.import_date - e2.import_date)) as import_gap_seconds,
    EXTRACT(DAYS FROM (e2.date_of_service_from - e1.date_of_service_from)) as service_date_gap_days
FROM claims.encounter e1
JOIN claims.encounter e2 ON e1.facility_id = e2.facility_id
JOIN claims.facility f ON e1.facility_id = f.facility_id
WHERE e1.date_of_service_from > e2.date_of_service_from  -- Later service date
  AND e1.import_date < e2.import_date                      -- But earlier import = FIFO violation
  AND e1.is_active = true
  AND e2.is_active = true
  AND e1.encounter_id != e2.encounter_id
  -- Only look at recent imports (last 30 days)
  AND e1.import_date > CURRENT_TIMESTAMP - INTERVAL '30 days'
ORDER BY e1.facility_id, e1.import_date DESC;

COMMENT ON VIEW claims.v_fifo_violations IS 'Detects cases where claims were processed out of service date order (FIFO violations)';

-- Queue processing statistics view
CREATE OR REPLACE VIEW staging.v_queue_statistics AS
SELECT
    DATE_TRUNC('hour', q.queued_at) as hour,
    f.facility_code,
    f.facility_name,
    COUNT(*) as total_files,
    COUNT(*) FILTER (WHERE q.queue_status = 'COMPLETED') as completed_files,
    COUNT(*) FILTER (WHERE q.queue_status = 'FAILED') as failed_files,
    AVG(EXTRACT(EPOCH FROM (q.processing_completed_at - q.queued_at)))
        FILTER (WHERE q.queue_status = 'COMPLETED') as avg_total_seconds,
    AVG(EXTRACT(EPOCH FROM (q.processing_started_at - q.queued_at)))
        FILTER (WHERE q.queue_status = 'COMPLETED') as avg_queue_wait_seconds,
    AVG(EXTRACT(EPOCH FROM (q.processing_completed_at - q.processing_started_at)))
        FILTER (WHERE q.queue_status = 'COMPLETED') as avg_processing_seconds
FROM staging.file_processing_queue q
JOIN claims.facility f ON q.facility_id = f.facility_id
WHERE q.created_at > CURRENT_TIMESTAMP - INTERVAL '7 days'
GROUP BY DATE_TRUNC('hour', q.queued_at), f.facility_code, f.facility_name
ORDER BY hour DESC, facility_code;

COMMENT ON VIEW staging.v_queue_statistics IS 'Hourly statistics on file processing queue performance';

-- ============================================================================
-- Functions
-- ============================================================================

-- Function to automatically update updated_at timestamp
CREATE OR REPLACE FUNCTION staging.update_queue_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to update timestamp
CREATE TRIGGER trg_queue_updated_at
BEFORE UPDATE ON staging.file_processing_queue
FOR EACH ROW
EXECUTE FUNCTION staging.update_queue_updated_at();

-- Function to clean up old completed/failed queue entries
CREATE OR REPLACE FUNCTION staging.cleanup_old_queue_entries(retention_days INTEGER DEFAULT 90)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM staging.file_processing_queue
    WHERE queue_status IN ('COMPLETED', 'FAILED')
      AND processing_completed_at < CURRENT_TIMESTAMP - (retention_days || ' days')::INTERVAL;

    GET DIAGNOSTICS deleted_count = ROW_COUNT;

    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION staging.cleanup_old_queue_entries IS 'Removes completed/failed queue entries older than specified days (default 90)';

-- ============================================================================
-- Grants
-- ============================================================================

GRANT SELECT, INSERT, UPDATE, DELETE ON staging.file_processing_queue TO pro_user;
GRANT SELECT ON staging.v_queue_health TO pro_user;
GRANT SELECT ON staging.v_queue_statistics TO pro_user;
GRANT SELECT ON claims.v_fifo_violations TO pro_user;
GRANT EXECUTE ON FUNCTION staging.cleanup_old_queue_entries TO pro_user;

-- ============================================================================
-- Comments
-- ============================================================================

COMMENT ON TABLE staging.file_processing_queue IS 'FIFO queue for file processing ensuring chronological order per facility';
COMMENT ON COLUMN staging.file_processing_queue.priority IS 'Priority level (0-1000, lower = higher priority, default = 100)';
COMMENT ON COLUMN staging.file_processing_queue.queue_status IS 'Current status: QUEUED, PROCESSING, COMPLETED, FAILED, RETRY';
COMMENT ON COLUMN staging.file_processing_queue.queued_at IS 'Timestamp when file was added to queue (used for FIFO ordering)';
COMMENT ON COLUMN staging.file_processing_queue.retry_count IS 'Number of times this file has been retried after failure';
