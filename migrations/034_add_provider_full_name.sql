-- Migration 034: Add full_name column to provider table
-- This column stores the provider's complete name for cases where the full name is provided
-- If full_name is provided, it will be parsed into first_name and last_name

ALTER TABLE claims.provider
ADD COLUMN full_name VARCHAR(255);

-- Create a computed index for full name search (combining first, middle, last)
CREATE INDEX idx_provider_full_name_computed ON claims.provider
USING gin(to_tsvector('english', COALESCE(first_name, '') || ' ' || COALESCE(middle_name, '') || ' ' || COALESCE(last_name, '')));

-- Add comment
COMMENT ON COLUMN claims.provider.full_name IS 'Full name of provider as provided in source data. Individual name components (first_name, last_name) are still required for claim processing.';
