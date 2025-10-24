-- Migration 025: Rename duration_milliseconds to duration_seconds in processing_metrics
-- This migration renames the column and converts existing data from milliseconds to seconds

-- Rename the column
ALTER TABLE staging.processing_metrics
RENAME COLUMN duration_milliseconds TO duration_seconds;

-- Update existing values to convert from milliseconds to seconds
UPDATE staging.processing_metrics
SET duration_seconds = duration_seconds / 1000.0
WHERE duration_seconds IS NOT NULL;

-- Add comment for clarity
COMMENT ON COLUMN staging.processing_metrics.duration_seconds IS
'Processing duration in seconds (converted from milliseconds in v1.2.5)';
