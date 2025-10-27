-- Migration 032: Create comprehensive claims detail view
-- This view consolidates patient, provider, encounter, procedure, diagnosis, and financial information

CREATE OR REPLACE VIEW claims.v_claims_detail AS
SELECT
    -- Encounter identifiers
    e.encounter_id,
    e.patient_control_number,
    e.transaction_set_control_number,

    -- Patient demographic information
    e.subscriber_id AS patient_id,
    e.subscriber_last_name AS patient_last_name,
    e.subscriber_first_name AS patient_first_name,
    e.subscriber_middle_name AS patient_middle_name,
    e.subscriber_name_suffix AS patient_name_suffix,
    e.subscriber_gender AS patient_gender,
    e.subscriber_birth_date AS patient_birth_date,
    EXTRACT(YEAR FROM AGE(e.date_of_service_from, e.subscriber_birth_date)) AS patient_age_at_service,
    e.subscriber_address_line1 AS patient_address_line1,
    e.subscriber_address_line2 AS patient_address_line2,
    e.subscriber_city AS patient_city,
    e.subscriber_state AS patient_state,
    e.subscriber_postal_code AS patient_postal_code,

    -- Insurance information
    e.payer_id,
    e.payer_name,
    e.payer_responsibility_code,
    e.claim_filing_indicator,

    -- Encounter information
    e.date_of_service_from AS encounter_date_from,
    e.date_of_service_to AS encounter_date_to,
    e.place_of_service_code AS encounter_place_of_service,
    CASE e.place_of_service_code
        WHEN '11' THEN 'Office'
        WHEN '02' THEN 'Telehealth'
        WHEN '21' THEN 'Inpatient Hospital'
        WHEN '22' THEN 'Outpatient Hospital'
        WHEN '23' THEN 'Emergency Room'
        WHEN '81' THEN 'Independent Laboratory'
        ELSE 'Other (' || e.place_of_service_code || ')'
    END AS encounter_place_of_service_description,
    e.claim_frequency_code,
    e.total_claim_charge_amount AS encounter_total_charge,

    -- Service line information
    sl.service_line_id,
    sl.line_number,
    sl.service_date_from,
    sl.service_date_to,
    sl.place_of_service_code AS service_line_place_of_service,

    -- Rendering Provider information (from service line)
    rp.provider_id AS rendering_provider_id,
    rp.npi AS rendering_provider_npi,
    rp.first_name || ' ' || rp.last_name AS rendering_provider_name,
    rp.taxonomy_code AS rendering_provider_taxonomy,
    rp.specialty AS rendering_provider_specialty,
    CASE
        WHEN rp.taxonomy_code LIKE '213E%' THEN 'Podiatry'
        WHEN rp.taxonomy_code LIKE '2086S0129%' THEN 'Hand Surgery'
        WHEN rp.taxonomy_code LIKE '207X%' THEN 'Orthopedic Surgery'
        WHEN rp.taxonomy_code LIKE '208600%' THEN 'General Surgery'
        ELSE rp.specialty
    END AS rendering_provider_specialty_category,

    -- Billing Provider information (from encounter)
    bp.provider_id AS billing_provider_id,
    bp.npi AS billing_provider_npi,
    bp.first_name || ' ' || bp.last_name AS billing_provider_name,

    -- CPT/HCPCS Procedure codes
    sl.product_service_id_qualifier,
    sl.procedure_code,
    sl.procedure_description,

    -- Modifiers
    sl.procedure_modifier_1,
    sl.procedure_modifier_2,
    sl.procedure_modifier_3,
    sl.procedure_modifier_4,
    CASE
        WHEN sl.procedure_modifier_1 IS NOT NULL OR
             sl.procedure_modifier_2 IS NOT NULL OR
             sl.procedure_modifier_3 IS NOT NULL OR
             sl.procedure_modifier_4 IS NOT NULL
        THEN CONCAT_WS(', ',
            NULLIF(sl.procedure_modifier_1, ''),
            NULLIF(sl.procedure_modifier_2, ''),
            NULLIF(sl.procedure_modifier_3, ''),
            NULLIF(sl.procedure_modifier_4, '')
        )
        ELSE NULL
    END AS all_modifiers,

    -- Diagnosis codes (concatenated from encounter diagnoses)
    (
        SELECT STRING_AGG(
            ed.diagnosis_code ||
            CASE WHEN ed.is_principal THEN ' (Principal)' ELSE '' END,
            ', '
            ORDER BY ed.sequence_number
        )
        FROM claims.encounter_diagnosis ed
        WHERE ed.encounter_id = e.encounter_id
    ) AS all_diagnosis_codes,

    -- Principal diagnosis
    (
        SELECT ed.diagnosis_code
        FROM claims.encounter_diagnosis ed
        WHERE ed.encounter_id = e.encounter_id
          AND ed.is_principal = true
        LIMIT 1
    ) AS principal_diagnosis_code,

    -- Diagnosis pointers for this service line
    (
        SELECT STRING_AGG(ed.diagnosis_code, ', ' ORDER BY sldp.pointer_sequence)
        FROM claims.service_line_diagnosis_pointer sldp
        JOIN claims.encounter_diagnosis ed ON sldp.diagnosis_id = ed.diagnosis_id
        WHERE sldp.service_line_id = sl.service_line_id
    ) AS service_line_diagnosis_codes,

    -- Financial information
    sl.service_unit_count AS units,
    sl.line_item_charge_amount AS line_charge_amount,
    sl.unit_basis_measurement_code,

    -- Expected reimbursement (if available - placeholder for now)
    NULL::numeric(18,2) AS expected_reimbursement,
    NULL::numeric(18,2) AS expected_payment_amount,
    NULL::character varying(50) AS reimbursement_basis,

    -- Additional service line details
    sl.emergency_indicator,
    sl.epsdt_indicator,
    sl.prior_authorization_number,
    sl.referral_number,
    sl.revenue_code,
    sl.ndc_code,

    -- Facility information
    f.facility_name,
    f.npi AS facility_npi,
    o.organization_name,

    -- Timestamps
    e.created_at AS encounter_created_at,
    sl.created_at AS service_line_created_at

FROM claims.encounter e
INNER JOIN claims.service_line sl ON e.encounter_id = sl.encounter_id
LEFT JOIN claims.provider rp ON sl.rendering_provider_id = rp.provider_id
LEFT JOIN claims.provider bp ON e.billing_provider_id = bp.provider_id
LEFT JOIN claims.facility f ON e.facility_id = f.facility_id
LEFT JOIN claims.organization o ON e.organization_id = o.organization_id

ORDER BY e.date_of_service_from DESC, e.encounter_id, sl.line_number;

-- Add comment to the view
COMMENT ON VIEW claims.v_claims_detail IS 'Comprehensive claims detail view showing patient demographics, insurance, provider info (NPI, specialty), encounter details (DOS, place of service), procedure codes (CPT/HCPCS), diagnosis codes, modifiers, and financial information (units, charges, expected reimbursement)';
