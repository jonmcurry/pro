-- Migration: 052_add_npi_registry_link
-- Description: Add NPI Registry link column to provider table for direct NPPES API access
-- Date: 2025-11-13

-- Add npi_registry_link column to provider table
-- This will store the direct link to the NPPES NPI Registry API for each provider
ALTER TABLE claims.provider
ADD COLUMN IF NOT EXISTS npi_registry_link TEXT;

-- Add comment explaining the column
COMMENT ON COLUMN claims.provider.npi_registry_link IS
'Direct link to NPPES NPI Registry API for this provider. Format: https://nppesapi.cms.hhs.gov/api/?version=2.1&number={NPI}. Auto-populated during NPI enrichment.';

-- Add index for faster lookups if needed (though not commonly queried)
CREATE INDEX IF NOT EXISTS idx_provider_npi_link
ON claims.provider(npi_registry_link)
WHERE npi_registry_link IS NOT NULL;

-- Backfill existing providers with NPI links
-- This generates the link for all existing providers that have an NPI
UPDATE claims.provider
SET npi_registry_link = 'https://nppesapi.cms.hhs.gov/api/?version=2.1&number=' || npi
WHERE npi IS NOT NULL
  AND npi_registry_link IS NULL
  AND LENGTH(npi) = 10;

-- Note: New providers will have npi_registry_link populated automatically during creation
-- and during NPI enrichment when provider data is updated from the NPPES API
