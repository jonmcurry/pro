-- Migration 064: Make date fields nullable
-- Don't default dates to 1900-01-01 when not provided in 837P

-- Make subscriber_birth_date nullable (was NOT NULL)
ALTER TABLE claims.encounter
ALTER COLUMN subscriber_birth_date DROP NOT NULL;

COMMENT ON COLUMN claims.encounter.subscriber_birth_date IS 'Patient date of birth (nullable - may not be provided in all 837P files)';
