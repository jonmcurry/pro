-- Migration 063: Add billing_date column to claims.encounter
-- This field stores the transaction creation date from the BHT segment (BHT04)
-- which represents when the claim was created/submitted

-- Add billing_date column
ALTER TABLE claims.encounter
ADD COLUMN IF NOT EXISTS billing_date DATE;

-- Add comment explaining the field
COMMENT ON COLUMN claims.encounter.billing_date IS 'Transaction Set Creation Date from EDI BHT04 segment - represents when the claim was created/submitted';

-- Add index for billing_date queries
CREATE INDEX IF NOT EXISTS idx_encounter_billing_date ON claims.encounter(billing_date);

-- Add composite index for common query patterns
CREATE INDEX IF NOT EXISTS idx_encounter_facility_billing_date ON claims.encounter(facility_id, billing_date DESC)
WHERE is_active = true AND soft_deleted = false;
