-- Migration: 029_drop_charge_amount_constraints
-- Description: Drop obsolete check constraints on charge amounts
-- Date: 2025-10-24

-- Drop check constraint on encounter.total_claim_charge_amount
-- This constraint limited amounts to 99999.99 which is too restrictive for real claims
ALTER TABLE claims.encounter
DROP CONSTRAINT IF EXISTS encounter_total_claim_charge_amount_check;

-- Drop check constraint on service_line.line_item_charge_amount
-- The NOT NULL constraint already ensures we have values, additional check not needed
ALTER TABLE claims.service_line
DROP CONSTRAINT IF EXISTS service_line_line_item_charge_amount_check;

-- Mark this migration as applied
INSERT INTO staging.schema_migrations (migration_name, applied_at, checksum, description)
VALUES ('029_drop_charge_amount_constraints.sql', NOW(), 'v1', 'Drop obsolete check constraints on charge amounts')
ON CONFLICT (migration_name) DO NOTHING;
