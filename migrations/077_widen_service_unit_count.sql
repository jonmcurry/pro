-- Migration: 077_widen_service_unit_count
-- Description: Widen the CHECK constraint on claims.service_line.service_unit_count.
--
-- Background: the column is NUMERIC(15,1), but an inline CHECK from migration
-- 005 capped it at 9999.9. The X12 837P SV104 quantity element imposes no such
-- limit, and HCPCS/drug (J-code) unit counts legitimately exceed 9999.9.
-- Claims with larger unit counts were being clamped down to 9999.9 on import,
-- silently corrupting the billed quantity.
--
-- Fix: drop the 9999.9 ceiling and keep only the positivity check. The
-- NUMERIC(15,1) column type remains the upper bound (max 99999999999999.9).
--
-- Idempotent: DROP CONSTRAINT IF EXISTS / re-add the same named constraint.

DO $$
BEGIN
    -- The original constraint was created inline and auto-named by PostgreSQL.
    ALTER TABLE claims.service_line
        DROP CONSTRAINT IF EXISTS service_line_service_unit_count_check;

    -- A previous run of this migration may have left the new named constraint.
    ALTER TABLE claims.service_line
        DROP CONSTRAINT IF EXISTS chk_service_unit_count_positive;

    ALTER TABLE claims.service_line
        ADD CONSTRAINT chk_service_unit_count_positive
        CHECK (service_unit_count > 0);

    RAISE NOTICE 'service_unit_count CHECK widened: 9999.9 ceiling removed, NUMERIC(15,1) is now the only upper bound';
END $$;
