-- Migration 025: Rename duration_milliseconds to duration_seconds in processing_metrics
-- This migration renames the column and converts existing data from milliseconds to seconds

-- Check if old column exists and rename it (idempotent)
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'staging'
        AND table_name = 'processing_metrics'
        AND column_name = 'duration_milliseconds'
    ) THEN
        -- Rename the column
        ALTER TABLE staging.processing_metrics
        RENAME COLUMN duration_milliseconds TO duration_seconds;

        -- Update existing values to convert from milliseconds to seconds
        UPDATE staging.processing_metrics
        SET duration_seconds = duration_seconds / 1000.0
        WHERE duration_seconds IS NOT NULL;
    END IF;
END $$;

-- Add comment for clarity (safe to run multiple times)
COMMENT ON COLUMN staging.processing_metrics.duration_seconds IS
'Processing duration in seconds (converted from milliseconds in v1.2.5)';
