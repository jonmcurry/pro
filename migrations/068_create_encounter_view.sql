-- Migration: 068_create_encounter_view
-- Description: Create encounter view with provider, payer, and diagnosis details
-- Date: 2025-12-26

-- Encounter view joining encounter data with providers, payers, and diagnoses
CREATE OR REPLACE VIEW claims.encounter_view AS
SELECT
    e.encounter_id,
    e.billing_date,
    e.submitter_id,
    e.patient_control_number,
    e.subscriber_birth_date,
    e.patient_date_of_birth,
    e.patient_gender,
    -- Primary payer
    epp.payer_id AS primary_payer_id,
    epp.payer_name AS primary_payer_name,
    epp.claim_filing_indicator AS primary_claim_filing_indicator,
    epp.is_billing_payer AS primary_is_billing_payer,
    -- Secondary payer
    eps.payer_id AS secondary_payer_id,
    eps.payer_name AS secondary_payer_name,
    eps.claim_filing_indicator AS secondary_claim_filing_indicator,
    eps.is_billing_payer AS secondary_is_billing_payer,
    -- Tertiary payer
    ept.payer_id AS tertiary_payer_id,
    ept.payer_name AS tertiary_payer_name,
    ept.claim_filing_indicator AS tertiary_claim_filing_indicator,
    ept.is_billing_payer AS tertiary_is_billing_payer,
    -- Claim details
    e.total_claim_charge_amount,
    e.place_of_service_code,
    e.date_of_service_from,
    e.date_of_service_to,
    -- Billing provider
    e.billing_provider_id,
    bpid.npi AS billing_provider_npi,
    bpid.last_name AS billing_provider_last,
    bpid.taxonomy_code AS billing_provider_taxonomy_code,
    bpid.specialty AS billing_provider_specialty,
    -- Referring provider
    e.referring_provider_id,
    rpid.npi AS referring_provider_npi,
    rpid.last_name AS referring_provider_last,
    rpid.taxonomy_code AS referring_provider_taxonomy_code,
    rpid.specialty AS referring_provider_specialty,
    -- Rendering provider
    e.rendering_provider_id,
    repid.npi AS rendering_provider_npi,
    repid.last_name AS rendering_provider_last,
    repid.taxonomy_code AS rendering_provider_taxonomy_code,
    repid.specialty AS rendering_provider_specialty,
    -- Supervising provider
    e.supervising_provider_id,
    spid.npi AS supervising_provider_npi,
    spid.last_name AS supervising_provider_last,
    spid.taxonomy_code AS supervising_provider_taxonomy_code,
    spid.specialty AS supervising_provider_specialty,
    -- Diagnosis codes (aggregated)
    dx.diagnosis_codes,
    -- Service facility
    e.service_facility_npi,
    e.service_facility_name,
    e.service_facility_city,
    e.service_facility_state
FROM claims.encounter e
-- Diagnosis codes aggregation
LEFT JOIN (
    SELECT
        encounter_id,
        string_agg(diagnosis_code, ', ' ORDER BY sequence_number) AS diagnosis_codes
    FROM claims.encounter_diagnosis
    WHERE diagnosis_code IS NOT NULL
    GROUP BY encounter_id
) dx ON dx.encounter_id = e.encounter_id
-- Provider joins
LEFT JOIN claims.provider bpid ON e.billing_provider_id = bpid.provider_id
LEFT JOIN claims.provider rpid ON e.referring_provider_id = rpid.provider_id
LEFT JOIN claims.provider repid ON e.rendering_provider_id = repid.provider_id
LEFT JOIN claims.provider spid ON e.supervising_provider_id = spid.provider_id
-- Payer joins by responsibility code
LEFT JOIN claims.encounter_payer epp ON e.encounter_id = epp.encounter_id AND epp.payer_responsibility_code = 'P'
LEFT JOIN claims.encounter_payer eps ON e.encounter_id = eps.encounter_id AND eps.payer_responsibility_code = 'S'
LEFT JOIN claims.encounter_payer ept ON e.encounter_id = ept.encounter_id AND ept.payer_responsibility_code = 'T';

COMMENT ON VIEW claims.encounter_view IS 'Denormalized encounter view with provider details, payer hierarchy, and aggregated diagnosis codes';
