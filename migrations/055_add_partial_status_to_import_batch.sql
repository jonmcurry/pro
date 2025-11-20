-- Migration 055: Add 'PARTIAL' status to import_batch constraint
-- Purpose: Fix constraint violation when batch has both successful and failed claims
-- Issue: Code sets import_status='PARTIAL' but constraint doesn't allow it

-- Drop and recreate constraint with PARTIAL status
DO $$
BEGIN
    -- Drop existing constraint if it exists
    IF EXISTS (
        SELECT 1 FROM information_schema.table_constraints
        WHERE table_schema = 'staging'
        AND table_name = 'import_batch'
        AND constraint_name = 'ck_import_batch_import_status'
    ) THEN
        ALTER TABLE staging.import_batch DROP CONSTRAINT ck_import_batch_import_status;
    END IF;

    -- Add new constraint with PARTIAL status
    ALTER TABLE staging.import_batch ADD CONSTRAINT ck_import_batch_import_status CHECK (
        import_status IN ('PENDING', 'QUEUED', 'INGESTING', 'INGESTED', 'PROCESSING', 'COMPLETED', 'PARTIAL', 'FAILED')
    );
END $$;

COMMENT ON CONSTRAINT ck_import_batch_import_status ON staging.import_batch
IS 'Valid import statuses: PENDING, QUEUED, INGESTING, INGESTED, PROCESSING, COMPLETED, PARTIAL (some succeeded/some failed), FAILED';
