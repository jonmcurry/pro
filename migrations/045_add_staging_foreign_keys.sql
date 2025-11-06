-- Migration: 045_add_staging_foreign_keys
-- Description: Add missing foreign key constraints for staging/import tracking columns
-- Date: 2025-11-05

-- Add foreign key constraint from encounter.import_batch_id to staging.import_batch
-- This ensures encounter import tracking references valid batch records
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'fk_encounter_import_batch'
        AND conrelid = 'claims.encounter'::regclass
    ) THEN
        ALTER TABLE claims.encounter
        ADD CONSTRAINT fk_encounter_import_batch
        FOREIGN KEY (import_batch_id)
        REFERENCES staging.import_batch(batch_id)
        ON DELETE SET NULL
        ON UPDATE CASCADE;
    END IF;
END $$;

-- Add foreign key constraint from encounter.import_configuration_id to staging.import_configuration
-- This ensures encounter configuration tracking references valid configuration records
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'fk_encounter_import_configuration'
        AND conrelid = 'claims.encounter'::regclass
    ) THEN
        ALTER TABLE claims.encounter
        ADD CONSTRAINT fk_encounter_import_configuration
        FOREIGN KEY (import_configuration_id)
        REFERENCES staging.import_configuration(configuration_id)
        ON DELETE SET NULL
        ON UPDATE CASCADE;
    END IF;
END $$;

-- Add foreign key constraint from import_batch.configuration_id to staging.import_configuration
-- This ensures batch processing references valid configuration records
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'fk_import_batch_configuration'
        AND conrelid = 'staging.import_batch'::regclass
    ) THEN
        ALTER TABLE staging.import_batch
        ADD CONSTRAINT fk_import_batch_configuration
        FOREIGN KEY (configuration_id)
        REFERENCES staging.import_configuration(configuration_id)
        ON DELETE SET NULL
        ON UPDATE CASCADE;
    END IF;
END $$;

-- Add index on import_batch.configuration_id for faster lookups (if not exists)
CREATE INDEX IF NOT EXISTS idx_import_batch_configuration_id
ON staging.import_batch(configuration_id);

COMMENT ON CONSTRAINT fk_encounter_import_batch ON claims.encounter IS
'Foreign key to staging.import_batch - tracks which batch imported this encounter';

COMMENT ON CONSTRAINT fk_encounter_import_configuration ON claims.encounter IS
'Foreign key to staging.import_configuration - tracks which configuration was used to import this encounter';

COMMENT ON CONSTRAINT fk_import_batch_configuration ON staging.import_batch IS
'Foreign key to staging.import_configuration - tracks which configuration was used for this batch';
