-- Migration 058: Change CASCADE to RESTRICT on critical foreign keys
--
-- Purpose: Prevent accidental data loss by requiring explicit deletion of child records
-- before parent records can be deleted. This is a safety measure for production data.
--
-- Critical tables affected:
-- - encounter_diagnosis (child of encounter)
-- - service_line (child of encounter)
-- - service_line_diagnosis_pointer (child of service_line and diagnosis)
-- - encounter_flag (child of encounter)
-- - service_line_flag (child of service_line)
-- - denial_event (child of encounter)
-- - service_line_adjustment (child of service_line)
--
-- NOTE: Some CASCADE relationships are intentionally kept:
-- - staging tables (import_batch children) - temporary data, OK to cascade
-- - audit tables - audit data should cascade with parent
-- - file_processing tables - queue data can cascade

-- ============================================================================
-- CLAIMS.ENCOUNTER_DIAGNOSIS - Critical: Contains diagnosis codes
-- ============================================================================
ALTER TABLE claims.encounter_diagnosis
DROP CONSTRAINT IF EXISTS encounter_diagnosis_encounter_id_fkey;

ALTER TABLE claims.encounter_diagnosis
ADD CONSTRAINT encounter_diagnosis_encounter_id_fkey
FOREIGN KEY (encounter_id) REFERENCES claims.encounter(encounter_id) ON DELETE RESTRICT;

-- ============================================================================
-- CLAIMS.SERVICE_LINE - Critical: Contains billing/procedure data
-- ============================================================================
ALTER TABLE claims.service_line
DROP CONSTRAINT IF EXISTS service_line_encounter_id_fkey;

ALTER TABLE claims.service_line
ADD CONSTRAINT service_line_encounter_id_fkey
FOREIGN KEY (encounter_id) REFERENCES claims.encounter(encounter_id) ON DELETE RESTRICT;

-- ============================================================================
-- CLAIMS.SERVICE_LINE_DIAGNOSIS_POINTER - Links service lines to diagnoses
-- ============================================================================
ALTER TABLE claims.service_line_diagnosis_pointer
DROP CONSTRAINT IF EXISTS service_line_diagnosis_pointer_service_line_id_fkey;

ALTER TABLE claims.service_line_diagnosis_pointer
ADD CONSTRAINT service_line_diagnosis_pointer_service_line_id_fkey
FOREIGN KEY (service_line_id) REFERENCES claims.service_line(service_line_id) ON DELETE RESTRICT;

ALTER TABLE claims.service_line_diagnosis_pointer
DROP CONSTRAINT IF EXISTS service_line_diagnosis_pointer_diagnosis_id_fkey;

ALTER TABLE claims.service_line_diagnosis_pointer
ADD CONSTRAINT service_line_diagnosis_pointer_diagnosis_id_fkey
FOREIGN KEY (diagnosis_id) REFERENCES claims.encounter_diagnosis(diagnosis_id) ON DELETE RESTRICT;

-- ============================================================================
-- CLAIMS.ENCOUNTER_FLAG - Important: Business rule results
-- ============================================================================
ALTER TABLE claims.encounter_flag
DROP CONSTRAINT IF EXISTS encounter_flag_encounter_id_fkey;

ALTER TABLE claims.encounter_flag
ADD CONSTRAINT encounter_flag_encounter_id_fkey
FOREIGN KEY (encounter_id) REFERENCES claims.encounter(encounter_id) ON DELETE RESTRICT;

-- ============================================================================
-- CLAIMS.SERVICE_LINE_FLAG - Important: Business rule results
-- ============================================================================
ALTER TABLE claims.service_line_flag
DROP CONSTRAINT IF EXISTS service_line_flag_service_line_id_fkey;

ALTER TABLE claims.service_line_flag
ADD CONSTRAINT service_line_flag_service_line_id_fkey
FOREIGN KEY (service_line_id) REFERENCES claims.service_line(service_line_id) ON DELETE RESTRICT;

-- ============================================================================
-- CLAIMS.DENIAL_EVENT - Important: Financial data
-- ============================================================================
ALTER TABLE claims.denial_event
DROP CONSTRAINT IF EXISTS denial_event_encounter_id_fkey;

ALTER TABLE claims.denial_event
ADD CONSTRAINT denial_event_encounter_id_fkey
FOREIGN KEY (encounter_id) REFERENCES claims.encounter(encounter_id) ON DELETE RESTRICT;

-- Service line FK can remain cascade or be changed to restrict
ALTER TABLE claims.denial_event
DROP CONSTRAINT IF EXISTS denial_event_service_line_id_fkey;

ALTER TABLE claims.denial_event
ADD CONSTRAINT denial_event_service_line_id_fkey
FOREIGN KEY (service_line_id) REFERENCES claims.service_line(service_line_id) ON DELETE RESTRICT;

-- ============================================================================
-- CLAIMS.SERVICE_LINE_ADJUSTMENT - Important: Financial/billing data
-- ============================================================================
ALTER TABLE claims.service_line_adjustment
DROP CONSTRAINT IF EXISTS service_line_adjustment_service_line_id_fkey;

ALTER TABLE claims.service_line_adjustment
ADD CONSTRAINT service_line_adjustment_service_line_id_fkey
FOREIGN KEY (service_line_id) REFERENCES claims.service_line(service_line_id) ON DELETE RESTRICT;

-- ============================================================================
-- CLAIMS.DENIAL_APPEAL - Child of denial_event
-- ============================================================================
ALTER TABLE claims.denial_appeal
DROP CONSTRAINT IF EXISTS denial_appeal_denial_id_fkey;

ALTER TABLE claims.denial_appeal
ADD CONSTRAINT denial_appeal_denial_id_fkey
FOREIGN KEY (denial_id) REFERENCES claims.denial_event(denial_id) ON DELETE RESTRICT;

-- ============================================================================
-- ORGANIZATION CHILDREN - Keep as RESTRICT (already safe pattern)
-- These prevent accidental org deletion when facilities/regions exist
-- ============================================================================
-- claims.facility -> organization already uses RESTRICT or CASCADE
ALTER TABLE claims.facility
DROP CONSTRAINT IF EXISTS facility_organization_id_fkey;

ALTER TABLE claims.facility
ADD CONSTRAINT facility_organization_id_fkey
FOREIGN KEY (organization_id) REFERENCES claims.organization(organization_id) ON DELETE RESTRICT;

ALTER TABLE claims.region
DROP CONSTRAINT IF EXISTS region_organization_id_fkey;

ALTER TABLE claims.region
ADD CONSTRAINT region_organization_id_fkey
FOREIGN KEY (organization_id) REFERENCES claims.organization(organization_id) ON DELETE RESTRICT;

-- ============================================================================
-- Add helper function for safe encounter deletion
-- ============================================================================
CREATE OR REPLACE FUNCTION claims.safe_delete_encounter(p_encounter_id BIGINT)
RETURNS BOOLEAN AS $$
DECLARE
    v_deleted BOOLEAN := FALSE;
BEGIN
    -- Delete child records in proper order
    DELETE FROM claims.service_line_diagnosis_pointer sld
    USING claims.service_line sl
    WHERE sld.service_line_id = sl.service_line_id
    AND sl.encounter_id = p_encounter_id;

    DELETE FROM claims.service_line_adjustment sla
    USING claims.service_line sl
    WHERE sla.service_line_id = sl.service_line_id
    AND sl.encounter_id = p_encounter_id;

    DELETE FROM claims.service_line_flag slf
    USING claims.service_line sl
    WHERE slf.service_line_id = sl.service_line_id
    AND sl.encounter_id = p_encounter_id;

    DELETE FROM claims.denial_appeal da
    USING claims.denial_event de
    WHERE da.denial_id = de.denial_id
    AND de.encounter_id = p_encounter_id;

    DELETE FROM claims.denial_event WHERE encounter_id = p_encounter_id;
    DELETE FROM claims.service_line WHERE encounter_id = p_encounter_id;
    DELETE FROM claims.encounter_flag WHERE encounter_id = p_encounter_id;
    DELETE FROM claims.encounter_diagnosis WHERE encounter_id = p_encounter_id;

    -- Finally delete the encounter
    DELETE FROM claims.encounter WHERE encounter_id = p_encounter_id;

    GET DIAGNOSTICS v_deleted = ROW_COUNT;
    RETURN v_deleted > 0;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION claims.safe_delete_encounter IS
'Safely delete an encounter and all its child records.
Use this instead of direct DELETE to handle RESTRICT constraints properly.';
