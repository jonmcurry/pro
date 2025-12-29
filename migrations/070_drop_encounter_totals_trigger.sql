-- Migration: Drop sync_encounter_totals triggers for performance optimization
--
-- PERFORMANCE FIX: The sync_encounter_totals triggers execute for EVERY service line insert,
-- causing ~60,000 extra DB operations for 10,000 claims (SELECT SUM + UPDATE per service line).
--
-- The total_claim_charge_amount is now calculated in the Rust application before encounter insert,
-- so these triggers are redundant and significantly hurt throughput.
--
-- Impact: Removes ~6 extra queries per encounter (2 queries × 3 service lines avg)
-- Expected improvement: From ~190 claims/sec to 600+ claims/sec

-- Drop the triggers (keep the function for potential future use)
DROP TRIGGER IF EXISTS sync_encounter_totals_insert ON claims.service_line;
DROP TRIGGER IF EXISTS sync_encounter_totals_update ON claims.service_line;
DROP TRIGGER IF EXISTS sync_encounter_totals_delete ON claims.service_line;

-- Add comment explaining why triggers were removed
COMMENT ON FUNCTION update_encounter_totals IS
'DEPRECATED: This trigger function is no longer used.
The total_claim_charge_amount is calculated in the Rust application before encounter INSERT.
Triggers were removed in migration 070 for performance optimization.
If you need to recalculate totals for existing data, run:
UPDATE claims.encounter e SET total_claim_charge_amount = (
    SELECT COALESCE(SUM(line_item_charge_amount), 0) FROM claims.service_line sl WHERE sl.encounter_id = e.encounter_id
);';
