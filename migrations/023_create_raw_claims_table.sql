-- Migration 023: Create staging.raw_claims table for two-stage processing pipeline
-- Purpose: Decouple file ingestion from claim validation
-- Stage 1: Files -> raw_claims (fast ingestion)
-- Stage 2: raw_claims -> encounters/errors (validated processing)

-- Create raw_claims table for storing parsed but unvalidated claims
CREATE TABLE staging.raw_claims (
    raw_claim_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    batch_id UUID NOT NULL REFERENCES staging.import_batch(batch_id) ON DELETE CASCADE,
    queue_id UUID NOT NULL REFERENCES staging.file_processing_queue(queue_id) ON DELETE CASCADE,

    -- Original parsed data stored as JSONB for flexibility
    -- Allows storing any claim format without schema changes
    encounter_fields JSONB NOT NULL,
    service_line_fields JSONB,
    diagnosis_fields JSONB,

    -- Metadata from original file
    row_number INTEGER NOT NULL,
    facility_code TEXT,

    -- Processing status tracking
    processing_status TEXT NOT NULL DEFAULT 'PENDING',
    -- PENDING: Not yet processed by Stage 2
    -- PROCESSING: Currently being processed by Stage 2
    -- COMPLETED: Successfully inserted to claims.encounter
    -- FAILED: Validation failed, logged to staging.import_error_log

    -- Timestamps for tracking pipeline latency
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMPTZ,

    -- Error tracking (populated if processing_status = FAILED)
    error_message TEXT,

    -- FIFO ordering field (extracted from encounter_fields for performance)
    date_of_service_from DATE,

    -- Ensure valid status values
    CONSTRAINT ck_raw_claims_status CHECK (
        processing_status IN ('PENDING', 'PROCESSING', 'COMPLETED', 'FAILED')
    )
);

-- Primary index for Stage 2 processor to find next batch of pending claims
-- Partial index only on PENDING/PROCESSING for efficiency
CREATE INDEX idx_raw_claims_pending ON staging.raw_claims(ingested_at ASC)
    WHERE processing_status IN ('PENDING', 'PROCESSING');

-- Index for tracking claims by batch (useful for batch status queries)
CREATE INDEX idx_raw_claims_batch ON staging.raw_claims(batch_id, processing_status);

-- Index for tracking claims by queue (useful for queue metrics)
CREATE INDEX idx_raw_claims_queue ON staging.raw_claims(queue_id, processing_status);

-- Index for FIFO compliance within facilities (if needed for prioritization)
CREATE INDEX idx_raw_claims_fifo ON staging.raw_claims(facility_code, date_of_service_from ASC, ingested_at ASC)
    WHERE processing_status = 'PENDING';

-- Index for finding stale PROCESSING claims (for recovery after crashes)
CREATE INDEX idx_raw_claims_stale ON staging.raw_claims(processing_status, ingested_at)
    WHERE processing_status = 'PROCESSING';

-- Add comment explaining table purpose
COMMENT ON TABLE staging.raw_claims IS 'Two-stage processing pipeline: Stage 1 stores raw parsed claims here before Stage 2 validates and inserts to encounters';
COMMENT ON COLUMN staging.raw_claims.encounter_fields IS 'Patient, subscriber, facility, payer data as JSONB';
COMMENT ON COLUMN staging.raw_claims.service_line_fields IS 'Procedure codes, charges, dates as JSONB array';
COMMENT ON COLUMN staging.raw_claims.diagnosis_fields IS 'ICD diagnosis codes as JSONB array';
COMMENT ON COLUMN staging.raw_claims.processing_status IS 'PENDING (not processed) | PROCESSING (in progress) | COMPLETED (inserted to encounters) | FAILED (validation error)';

-- Update import_batch table to support new INGESTING/INGESTED states for Stage 1
-- Current states: PENDING, PROCESSING, COMPLETED, FAILED
-- New states needed: INGESTING (Stage 1 in progress), INGESTED (Stage 1 done, Stage 2 pending)

-- First, check if we need to update the constraint
DO $$
BEGIN
    -- Drop existing constraint if it exists
    IF EXISTS (
        SELECT 1 FROM information_schema.constraint_column_usage
        WHERE table_schema = 'staging'
        AND table_name = 'import_batch'
        AND constraint_name = 'ck_import_batch_status'
    ) THEN
        ALTER TABLE staging.import_batch DROP CONSTRAINT ck_import_batch_status;
    END IF;

    -- Add new constraint with additional states
    ALTER TABLE staging.import_batch ADD CONSTRAINT ck_import_batch_status CHECK (
        batch_status IN ('PENDING', 'INGESTING', 'INGESTED', 'PROCESSING', 'COMPLETED', 'FAILED')
    );
END $$;

COMMENT ON CONSTRAINT ck_import_batch_status ON staging.import_batch IS 'INGESTING: Stage 1 reading file | INGESTED: Stage 1 complete, raw_claims populated | PROCESSING: Stage 2 validating claims | COMPLETED: All claims processed | FAILED: Pipeline error';

-- Add new column to processing_metrics to track which stage is being measured
ALTER TABLE staging.processing_metrics ADD COLUMN IF NOT EXISTS processing_stage TEXT DEFAULT 'IMPORT';
-- IMPORT: Legacy single-stage processing
-- INGEST: Stage 1 (file -> raw_claims)
-- PROCESS: Stage 2 (raw_claims -> encounters/errors)

COMMENT ON COLUMN staging.processing_metrics.processing_stage IS 'IMPORT (legacy) | INGEST (Stage 1: file->staging) | PROCESS (Stage 2: staging->encounters)';

-- Create view for monitoring raw_claims processing status
CREATE OR REPLACE VIEW staging.vw_raw_claims_status AS
SELECT
    b.batch_id,
    b.file_name,
    b.batch_status,
    COUNT(*) AS total_claims,
    COUNT(*) FILTER (WHERE rc.processing_status = 'PENDING') AS pending_claims,
    COUNT(*) FILTER (WHERE rc.processing_status = 'PROCESSING') AS processing_claims,
    COUNT(*) FILTER (WHERE rc.processing_status = 'COMPLETED') AS completed_claims,
    COUNT(*) FILTER (WHERE rc.processing_status = 'FAILED') AS failed_claims,
    MIN(rc.ingested_at) AS first_ingested_at,
    MAX(rc.processed_at) AS last_processed_at,
    -- Calculate average processing time for completed claims
    AVG(EXTRACT(EPOCH FROM (rc.processed_at - rc.ingested_at))) FILTER (WHERE rc.processing_status = 'COMPLETED') AS avg_processing_seconds
FROM staging.import_batch b
LEFT JOIN staging.raw_claims rc ON b.batch_id = rc.batch_id
GROUP BY b.batch_id, b.file_name, b.batch_status;

COMMENT ON VIEW staging.vw_raw_claims_status IS 'Monitoring view: raw_claims processing progress by batch';

-- Create function to mark stale PROCESSING claims back to PENDING (for recovery after crashes)
CREATE OR REPLACE FUNCTION staging.recover_stale_raw_claims(stale_threshold_minutes INTEGER DEFAULT 30)
RETURNS INTEGER
LANGUAGE plpgsql
AS $$
DECLARE
    recovered_count INTEGER;
BEGIN
    -- Find claims that have been in PROCESSING state for too long
    -- These are likely from crashed workers
    UPDATE staging.raw_claims
    SET processing_status = 'PENDING',
        error_message = 'Recovered from stale PROCESSING state at ' || CURRENT_TIMESTAMP
    WHERE processing_status = 'PROCESSING'
    AND ingested_at < CURRENT_TIMESTAMP - (stale_threshold_minutes || ' minutes')::INTERVAL;

    GET DIAGNOSTICS recovered_count = ROW_COUNT;

    RETURN recovered_count;
END;
$$;

COMMENT ON FUNCTION staging.recover_stale_raw_claims IS 'Recovery function: marks stale PROCESSING claims back to PENDING (default: 30 min threshold)';
