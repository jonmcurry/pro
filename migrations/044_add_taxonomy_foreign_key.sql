-- Migration: 044_add_taxonomy_foreign_key
-- Description: Add foreign key constraint from claims.provider.taxonomy_code to claims.provider_taxonomy
-- Date: 2025-11-05

-- Add foreign key constraint to enforce referential integrity
-- This ensures taxonomy_code values in claims.provider must exist in claims.provider_taxonomy
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'fk_provider_taxonomy'
        AND conrelid = 'claims.provider'::regclass
    ) THEN
        ALTER TABLE claims.provider
        ADD CONSTRAINT fk_provider_taxonomy
        FOREIGN KEY (taxonomy_code)
        REFERENCES claims.provider_taxonomy(taxonomy_code)
        ON DELETE RESTRICT
        ON UPDATE CASCADE;
    END IF;
END $$;

-- Add index on taxonomy_code for faster lookups (if not already exists)
-- This improves performance when joining provider to provider_taxonomy
CREATE INDEX IF NOT EXISTS idx_provider_taxonomy_code ON claims.provider(taxonomy_code);

COMMENT ON CONSTRAINT fk_provider_taxonomy ON claims.provider IS
'Foreign key to provider_taxonomy table - ensures taxonomy codes are valid NUCC codes';
