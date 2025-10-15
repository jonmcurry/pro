-- Migration: 013_create_dashboard_views
-- Description: Create comprehensive views for dashboards and reporting
-- Date: 2025-10-14

-- ====================================================================================
-- MANAGEMENT DASHBOARD VIEWS
-- ====================================================================================

-- Overview metrics view
CREATE OR REPLACE VIEW claims.v_management_overview AS
SELECT
    e.organization_id,
    e.facility_id,
    DATE_TRUNC('month', e.date_of_service_from) AS month,

    -- Volume metrics
    COUNT(DISTINCT e.encounter_id) AS total_encounters,
    COUNT(DISTINCT sl.service_line_id) AS total_service_lines,
    COUNT(DISTINCT e.billing_provider_id) AS active_providers,
    COUNT(DISTINCT e.coder_id) AS active_coders,

    -- Financial metrics
    SUM(e.total_claim_charge_amount) AS total_billed_amount,
    AVG(e.total_claim_charge_amount) AS avg_claim_amount,
    SUM(sl.line_item_charge_amount) AS total_line_charges,

    -- RVU metrics
    SUM(slr.total_rvu) AS total_rvus,
    SUM(slr.total_medicare_payment) AS estimated_medicare_payment,

    -- Coding metrics
    COUNT(DISTINCT CASE WHEN ef.flag_id IS NOT NULL THEN e.encounter_id END) AS encounters_with_flags,
    COUNT(DISTINCT ef.flag_id) AS total_flags,
    COUNT(DISTINCT CASE WHEN ef.severity = 'HIGH' THEN ef.flag_id END) AS high_severity_flags,
    COUNT(DISTINCT CASE WHEN ef.severity = 'MEDIUM' THEN ef.flag_id END) AS medium_severity_flags,
    COUNT(DISTINCT CASE WHEN ef.severity = 'LOW' THEN ef.flag_id END) AS low_severity_flags,

    -- Flag rate
    ROUND(100.0 * COUNT(DISTINCT CASE WHEN ef.flag_id IS NOT NULL THEN e.encounter_id END) /
        NULLIF(COUNT(DISTINCT e.encounter_id), 0), 2) AS flag_rate_percentage,

    -- Denial metrics
    COUNT(DISTINCT de.denial_id) AS total_denials,
    SUM(de.denied_amount) AS total_denied_amount,
    ROUND(100.0 * COUNT(DISTINCT de.denial_id) / NULLIF(COUNT(DISTINCT e.encounter_id), 0), 2) AS denial_rate_percentage

FROM claims.encounter e
LEFT JOIN claims.service_line sl ON e.encounter_id = sl.encounter_id
LEFT JOIN claims.service_line_reimbursement slr ON sl.service_line_id = slr.service_line_id
LEFT JOIN claims.encounter_flag ef ON e.encounter_id = ef.encounter_id AND ef.flag_status = 'OPEN'
LEFT JOIN claims.denial_event de ON e.encounter_id = de.encounter_id
WHERE e.is_active = true
    AND e.soft_deleted = false
GROUP BY e.organization_id, e.facility_id, DATE_TRUNC('month', e.date_of_service_from);

COMMENT ON VIEW claims.v_management_overview IS 'High-level metrics for management dashboard';

-- Claim status summary
CREATE OR REPLACE VIEW claims.v_claim_status_summary AS
SELECT
    e.organization_id,
    e.facility_id,
    e.claim_status,
    COUNT(DISTINCT e.encounter_id) AS encounter_count,
    SUM(e.total_claim_charge_amount) AS total_amount,
    AVG(e.total_claim_charge_amount) AS avg_amount,
    MIN(e.date_of_service_from) AS earliest_dos,
    MAX(e.date_of_service_from) AS latest_dos
FROM claims.encounter e
WHERE e.is_active = true
    AND e.soft_deleted = false
GROUP BY e.organization_id, e.facility_id, e.claim_status;

COMMENT ON VIEW claims.v_claim_status_summary IS 'Summary of claims by status';

-- ====================================================================================
-- CODING ACCURACY VIEWS
-- ====================================================================================

-- Coder performance view
CREATE OR REPLACE VIEW claims.v_coder_performance AS
SELECT
    c.coder_id,
    c.coder_code,
    c.last_name,
    c.first_name,
    c.organization_id,

    -- Current month metrics
    COUNT(DISTINCT e.encounter_id) AS encounters_coded,
    COUNT(DISTINCT sl.service_line_id) AS service_lines_coded,
    SUM(e.total_claim_charge_amount) AS total_amount_coded,

    -- RVU productivity
    SUM(slr.total_rvu) AS total_rvus,
    ROUND(AVG(slr.total_rvu), 3) AS avg_rvu_per_service,

    -- Accuracy metrics (from audits)
    COUNT(DISTINCT ae.audit_encounter_id) AS encounters_audited,
    COUNT(DISTINCT CASE WHEN ae.has_errors THEN ae.audit_encounter_id END) AS encounters_with_errors,
    ROUND(100.0 * (1 - COUNT(DISTINCT CASE WHEN ae.has_errors THEN ae.audit_encounter_id END)::NUMERIC /
        NULLIF(COUNT(DISTINCT ae.audit_encounter_id), 0)), 2) AS accuracy_rate,

    -- Error breakdown
    SUM(ae.severity_high_count) AS high_severity_errors,
    SUM(ae.severity_medium_count) AS medium_severity_errors,
    SUM(ae.severity_low_count) AS low_severity_errors,

    -- Financial impact
    SUM(ae.overpayment_amount) AS total_overpayment,
    SUM(ae.underpayment_amount) AS total_underpayment,
    SUM(ae.net_financial_impact) AS net_financial_impact,

    -- Flag metrics
    COUNT(DISTINCT ef.flag_id) AS total_flags_generated,
    COUNT(DISTINCT CASE WHEN ef.severity = 'HIGH' THEN ef.flag_id END) AS high_severity_flags,

    -- Productivity
    ROUND(COUNT(DISTINCT e.encounter_id)::NUMERIC / NULLIF(COUNT(DISTINCT e.coding_date), 0), 2) AS avg_encounters_per_day

FROM claims.coder c
LEFT JOIN claims.encounter e ON c.coder_id = e.coder_id
    AND e.coding_date >= CURRENT_DATE - INTERVAL '30 days'
LEFT JOIN claims.service_line sl ON e.encounter_id = sl.encounter_id
LEFT JOIN claims.service_line_reimbursement slr ON sl.service_line_id = slr.service_line_id
LEFT JOIN claims.audit_encounter ae ON e.encounter_id = ae.encounter_id
LEFT JOIN claims.encounter_flag ef ON e.encounter_id = ef.encounter_id AND ef.created_by = 'SYSTEM'
WHERE c.is_active = true
GROUP BY c.coder_id, c.coder_code, c.last_name, c.first_name, c.organization_id;

COMMENT ON VIEW claims.v_coder_performance IS 'Coder performance and accuracy metrics';

-- Provider documentation accuracy
CREATE OR REPLACE VIEW claims.v_provider_documentation_accuracy AS
SELECT
    p.provider_id,
    p.npi,
    p.last_name,
    p.first_name,
    p.specialty,
    p.organization_id,

    -- Volume
    COUNT(DISTINCT e.encounter_id) AS encounters_billed,
    COUNT(DISTINCT sl.service_line_id) AS service_lines_billed,

    -- Audited metrics
    COUNT(DISTINCT ae.audit_encounter_id) AS encounters_audited,
    COUNT(DISTINCT CASE WHEN ae.has_errors THEN ae.audit_encounter_id END) AS encounters_with_errors,
    ROUND(100.0 * (1 - COUNT(DISTINCT CASE WHEN ae.has_errors THEN ae.audit_encounter_id END)::NUMERIC /
        NULLIF(COUNT(DISTINCT ae.audit_encounter_id), 0)), 2) AS documentation_accuracy_rate,

    -- Error types
    SUM(ae.severity_high_count) AS high_severity_errors,
    SUM(ae.severity_medium_count) AS medium_severity_errors,

    -- Common issues
    COUNT(DISTINCT CASE WHEN sle.evaluation_result = 'OVERCODED' THEN sle.evaluation_id END) AS overcoding_instances,
    COUNT(DISTINCT CASE WHEN sle.evaluation_result = 'UNDERCODED' THEN sle.evaluation_id END) AS undercoding_instances,
    COUNT(DISTINCT CASE WHEN sle.evaluation_result = 'UNSUPPORTED' THEN sle.evaluation_id END) AS unsupported_instances,

    -- Financial impact
    SUM(ae.overpayment_amount) AS total_overpayment,
    SUM(ae.underpayment_amount) AS total_underpayment,
    SUM(ae.net_financial_impact) AS net_financial_impact

FROM claims.provider p
LEFT JOIN claims.encounter e ON p.provider_id = e.rendering_provider_id
    AND e.date_of_service_from >= CURRENT_DATE - INTERVAL '90 days'
LEFT JOIN claims.service_line sl ON e.encounter_id = sl.encounter_id
LEFT JOIN claims.audit_encounter ae ON e.encounter_id = ae.encounter_id
LEFT JOIN claims.service_line_evaluation sle ON sl.service_line_id = sle.service_line_id
WHERE p.is_active = true
GROUP BY p.provider_id, p.npi, p.last_name, p.first_name, p.specialty, p.organization_id;

COMMENT ON VIEW claims.v_provider_documentation_accuracy IS 'Provider documentation and coding accuracy metrics';

-- ====================================================================================
-- FLAG ANALYSIS VIEWS
-- ====================================================================================

-- Flags by category
CREATE OR REPLACE VIEW claims.v_flags_by_category AS
SELECT
    e.organization_id,
    e.facility_id,
    fc.category_code,
    fc.category_name,
    fi.issue_code,
    fi.issue_description,
    fi.severity,
    DATE_TRUNC('month', ef.created_at) AS month,

    -- Flag counts
    COUNT(DISTINCT ef.flag_id) AS flag_count,
    COUNT(DISTINCT ef.encounter_id) AS affected_encounters,

    -- Resolution metrics
    COUNT(DISTINCT CASE WHEN ef.flag_status = 'OPEN' THEN ef.flag_id END) AS open_flags,
    COUNT(DISTINCT CASE WHEN ef.flag_status = 'RESOLVED' THEN ef.flag_id END) AS resolved_flags,
    COUNT(DISTINCT CASE WHEN ef.flag_status = 'ACCEPTED' THEN ef.flag_id END) AS accepted_flags,
    COUNT(DISTINCT CASE WHEN ef.flag_status = 'REJECTED' THEN ef.flag_id END) AS rejected_flags,

    -- Resolution rate
    ROUND(100.0 * COUNT(DISTINCT CASE WHEN ef.flag_status IN ('RESOLVED', 'ACCEPTED') THEN ef.flag_id END) /
        NULLIF(COUNT(DISTINCT ef.flag_id), 0), 2) AS resolution_rate_percentage,

    -- Timing
    AVG(EXTRACT(EPOCH FROM (ef.resolved_at - ef.created_at))/3600) AS avg_resolution_hours

FROM claims.encounter_flag ef
INNER JOIN claims.flag_issue fi ON ef.issue_id = fi.issue_id
INNER JOIN claims.flag_category fc ON fi.category_id = fc.category_id
INNER JOIN claims.encounter e ON ef.encounter_id = e.encounter_id
WHERE e.is_active = true
GROUP BY e.organization_id, e.facility_id, fc.category_code, fc.category_name,
    fi.issue_code, fi.issue_description, fi.severity, DATE_TRUNC('month', ef.created_at);

COMMENT ON VIEW claims.v_flags_by_category IS 'Flag statistics by category and issue type';

-- Service line flags detail
CREATE OR REPLACE VIEW claims.v_service_line_flags_detail AS
SELECT
    slf.flag_id,
    e.organization_id,
    e.facility_id,
    e.encounter_id,
    e.patient_control_number,
    sl.service_line_id,
    sl.procedure_code,
    sl.procedure_description,
    sl.line_item_charge_amount,
    fi.issue_code,
    fi.issue_description,
    slf.severity,
    slf.flag_reason,
    slf.flagged_element,
    slf.proposed_code,
    slf.proposed_modifier,
    slf.proposed_quantity,
    slf.flag_status,
    slf.created_at AS flagged_at,
    slf.resolved_at,
    slf.resolution_note,
    e.coder_id,
    e.rendering_provider_id
FROM claims.service_line_flag slf
INNER JOIN claims.service_line sl ON slf.service_line_id = sl.service_line_id
INNER JOIN claims.encounter e ON sl.encounter_id = e.encounter_id
INNER JOIN claims.flag_issue fi ON slf.issue_id = fi.issue_id
WHERE e.is_active = true
    AND e.soft_deleted = false;

COMMENT ON VIEW claims.v_service_line_flags_detail IS 'Detailed view of service line flags';

-- ====================================================================================
-- DENIAL ANALYSIS VIEWS
-- ====================================================================================

-- Denial summary by payer
CREATE OR REPLACE VIEW claims.v_denial_by_payer AS
SELECT
    de.organization_id,
    de.facility_id,
    de.payer_id,
    de.payer_name,
    DATE_TRUNC('month', de.denial_date) AS month,

    -- Volume metrics
    COUNT(DISTINCT de.denial_id) AS denial_count,
    COUNT(DISTINCT de.encounter_id) AS affected_encounters,

    -- Financial metrics
    SUM(de.denied_amount) AS total_denied_amount,
    SUM(de.billed_amount) AS total_billed_amount,
    SUM(de.paid_amount) AS total_paid_amount,
    ROUND(100.0 * SUM(de.denied_amount) / NULLIF(SUM(de.billed_amount), 0), 2) AS denial_rate_percentage,

    -- Root cause
    COUNT(DISTINCT CASE WHEN de.root_cause_category = 'CODING' THEN de.denial_id END) AS coding_denials,
    COUNT(DISTINCT CASE WHEN de.root_cause_category = 'DOCUMENTATION' THEN de.denial_id END) AS documentation_denials,
    COUNT(DISTINCT CASE WHEN de.root_cause_category = 'AUTHORIZATION' THEN de.denial_id END) AS authorization_denials,
    COUNT(DISTINCT CASE WHEN de.root_cause_category = 'ELIGIBILITY' THEN de.denial_id END) AS eligibility_denials,
    COUNT(DISTINCT CASE WHEN de.root_cause_category = 'MEDICAL_NECESSITY' THEN de.denial_id END) AS medical_necessity_denials,

    -- Preventability
    COUNT(DISTINCT CASE WHEN de.is_preventable THEN de.denial_id END) AS preventable_denials,
    SUM(CASE WHEN de.is_preventable THEN de.denied_amount ELSE 0 END) AS preventable_denied_amount,
    ROUND(100.0 * COUNT(DISTINCT CASE WHEN de.is_preventable THEN de.denial_id END) /
        NULLIF(COUNT(DISTINCT de.denial_id), 0), 2) AS preventable_percentage,

    -- Appeals
    COUNT(DISTINCT CASE WHEN de.appeal_filed THEN de.denial_id END) AS appeals_filed,
    COUNT(DISTINCT CASE WHEN de.resolution_status = 'OVERTURNED' THEN de.denial_id END) AS appeals_won,
    ROUND(100.0 * COUNT(DISTINCT CASE WHEN de.resolution_status = 'OVERTURNED' THEN de.denial_id END) /
        NULLIF(COUNT(DISTINCT CASE WHEN de.appeal_filed THEN de.denial_id END), 0), 2) AS appeal_success_rate

FROM claims.denial_event de
GROUP BY de.organization_id, de.facility_id, de.payer_id, de.payer_name, DATE_TRUNC('month', de.denial_date);

COMMENT ON VIEW claims.v_denial_by_payer IS 'Denial statistics by payer';

-- Denial by reason code
CREATE OR REPLACE VIEW claims.v_denial_by_reason AS
SELECT
    de.organization_id,
    de.facility_id,
    de.claim_adjustment_reason_code AS carc,
    drc.short_description AS carc_description,
    drc.category AS denial_category,
    DATE_TRUNC('month', de.denial_date) AS month,

    -- Volume
    COUNT(DISTINCT de.denial_id) AS denial_count,
    COUNT(DISTINCT de.encounter_id) AS affected_encounters,

    -- Financial
    SUM(de.denied_amount) AS total_denied_amount,
    AVG(de.denied_amount) AS avg_denied_amount,

    -- Preventability
    COUNT(DISTINCT CASE WHEN de.is_preventable THEN de.denial_id END) AS preventable_count,
    SUM(CASE WHEN de.is_preventable THEN de.denied_amount ELSE 0 END) AS preventable_amount,

    -- Resolution
    COUNT(DISTINCT CASE WHEN de.resolution_status = 'OVERTURNED' THEN de.denial_id END) AS overturned_count,
    COUNT(DISTINCT CASE WHEN de.resolution_status = 'WRITTEN_OFF' THEN de.denial_id END) AS written_off_count

FROM claims.denial_event de
LEFT JOIN claims.denial_reason_code drc ON de.claim_adjustment_reason_code = drc.reason_code
    AND drc.code_type = 'CARC'
GROUP BY de.organization_id, de.facility_id, de.claim_adjustment_reason_code,
    drc.short_description, drc.category, DATE_TRUNC('month', de.denial_date);

COMMENT ON VIEW claims.v_denial_by_reason IS 'Denial statistics by CARC reason code';

-- ====================================================================================
-- PRACTICE ANALYTICS VIEWS
-- ====================================================================================

-- Procedure volume analysis
CREATE OR REPLACE VIEW claims.v_procedure_volume AS
SELECT
    e.organization_id,
    e.facility_id,
    sl.procedure_code,
    sl.procedure_description,
    DATE_TRUNC('month', sl.service_date_from) AS month,

    -- Volume
    COUNT(DISTINCT sl.service_line_id) AS procedure_count,
    COUNT(DISTINCT e.encounter_id) AS encounter_count,
    COUNT(DISTINCT e.rendering_provider_id) AS provider_count,
    SUM(sl.service_unit_count) AS total_units,

    -- Financial
    SUM(sl.line_item_charge_amount) AS total_charges,
    AVG(sl.line_item_charge_amount) AS avg_charge,

    -- RVU metrics
    SUM(slr.total_rvu) AS total_rvus,
    AVG(slr.total_rvu) AS avg_rvu,
    SUM(slr.total_medicare_payment) AS estimated_payment,

    -- Quality metrics
    COUNT(DISTINCT CASE WHEN slf.flag_id IS NOT NULL THEN sl.service_line_id END) AS flagged_lines,
    ROUND(100.0 * COUNT(DISTINCT CASE WHEN slf.flag_id IS NOT NULL THEN sl.service_line_id END) /
        NULLIF(COUNT(DISTINCT sl.service_line_id), 0), 2) AS flag_rate_percentage

FROM claims.service_line sl
INNER JOIN claims.encounter e ON sl.encounter_id = e.encounter_id
LEFT JOIN claims.service_line_reimbursement slr ON sl.service_line_id = slr.service_line_id
LEFT JOIN claims.service_line_flag slf ON sl.service_line_id = slf.service_line_id AND slf.flag_status = 'OPEN'
WHERE e.is_active = true
    AND e.soft_deleted = false
GROUP BY e.organization_id, e.facility_id, sl.procedure_code, sl.procedure_description,
    DATE_TRUNC('month', sl.service_date_from);

COMMENT ON VIEW claims.v_procedure_volume IS 'Procedure volume and performance analysis';

-- Provider productivity
CREATE OR REPLACE VIEW claims.v_provider_productivity AS
SELECT
    p.provider_id,
    p.npi,
    p.last_name,
    p.first_name,
    p.specialty,
    e.organization_id,
    e.facility_id,
    DATE_TRUNC('month', e.date_of_service_from) AS month,

    -- Volume metrics
    COUNT(DISTINCT e.encounter_id) AS encounter_count,
    COUNT(DISTINCT sl.service_line_id) AS service_line_count,
    AVG(COUNT(DISTINCT e.encounter_id)) OVER (
        PARTITION BY p.provider_id, DATE_TRUNC('month', e.date_of_service_from)
    ) AS avg_daily_encounters,

    -- Financial metrics
    SUM(e.total_claim_charge_amount) AS total_charges,
    AVG(e.total_claim_charge_amount) AS avg_charge_per_encounter,
    SUM(sl.line_item_charge_amount) AS total_line_charges,

    -- RVU metrics
    SUM(slr.total_rvu) AS total_work_rvus,
    AVG(slr.total_rvu) AS avg_rvu_per_service,
    SUM(slr.total_medicare_payment) AS estimated_collections,

    -- E/M distribution
    COUNT(DISTINCT CASE WHEN sl.procedure_code LIKE '99___' THEN sl.service_line_id END) AS em_visit_count,
    COUNT(DISTINCT CASE WHEN sl.procedure_code NOT LIKE '99___' THEN sl.service_line_id END) AS non_em_procedure_count

FROM claims.provider p
LEFT JOIN claims.encounter e ON p.provider_id = e.rendering_provider_id
LEFT JOIN claims.service_line sl ON e.encounter_id = sl.encounter_id
LEFT JOIN claims.service_line_reimbursement slr ON sl.service_line_id = slr.service_line_id
WHERE p.is_active = true
    AND e.is_active = true
GROUP BY p.provider_id, p.npi, p.last_name, p.first_name, p.specialty,
    e.organization_id, e.facility_id, DATE_TRUNC('month', e.date_of_service_from);

COMMENT ON VIEW claims.v_provider_productivity IS 'Provider productivity and RVU analysis';

-- ====================================================================================
-- AUDIT TRACKING VIEWS
-- ====================================================================================

-- Audit assignment status
CREATE OR REPLACE VIEW claims.v_audit_assignment_status AS
SELECT
    aa.audit_id,
    aa.audit_name,
    aa.audit_type,
    aa.organization_id,
    aa.facility_id,
    aa.reviewer_id,
    r.last_name AS reviewer_last_name,
    r.first_name AS reviewer_first_name,
    aa.audit_status,
    aa.due_date,

    -- Progress metrics
    aa.sample_size,
    aa.encounters_reviewed,
    aa.completion_percentage,
    ROUND(100.0 * aa.encounters_reviewed / NULLIF(aa.sample_size, 0), 2) AS actual_completion_percentage,

    -- Quality metrics
    aa.encounters_with_errors,
    aa.total_flags_found,
    aa.error_rate,

    -- Financial metrics
    aa.total_billed_amount,
    aa.total_overpayment_amount,
    aa.total_underpayment_amount,
    aa.net_financial_impact,

    -- Timing
    aa.assigned_at,
    aa.completed_at,
    EXTRACT(EPOCH FROM (COALESCE(aa.completed_at, CURRENT_TIMESTAMP) - aa.assigned_at))/86400 AS days_in_progress,
    EXTRACT(EPOCH FROM (aa.due_date - CURRENT_DATE)) AS days_until_due

FROM claims.audit_assignment aa
LEFT JOIN claims.reviewer r ON aa.reviewer_id = r.reviewer_id;

COMMENT ON VIEW claims.v_audit_assignment_status IS 'Audit assignment status and progress tracking';

-- ====================================================================================
-- FINANCIAL ANALYSIS VIEWS
-- ====================================================================================

-- Reimbursement analysis
CREATE OR REPLACE VIEW claims.v_reimbursement_analysis AS
SELECT
    e.organization_id,
    e.facility_id,
    DATE_TRUNC('month', e.date_of_service_from) AS month,

    -- Volume
    COUNT(DISTINCT e.encounter_id) AS encounter_count,
    COUNT(DISTINCT sl.service_line_id) AS service_line_count,

    -- Charges
    SUM(e.total_claim_charge_amount) AS total_billed,
    SUM(sl.line_item_charge_amount) AS total_line_charges,

    -- RVU-based estimates
    SUM(slr.total_rvu) AS total_rvus,
    SUM(slr.total_medicare_payment) AS estimated_medicare_payment,

    -- Payment-to-charge ratio
    ROUND(SUM(slr.total_medicare_payment) / NULLIF(SUM(sl.line_item_charge_amount), 0), 4) AS payment_to_charge_ratio,

    -- Denials impact
    SUM(COALESCE(de.denied_amount, 0)) AS total_denied,
    ROUND(100.0 * SUM(COALESCE(de.denied_amount, 0)) / NULLIF(SUM(e.total_claim_charge_amount), 0), 2) AS denial_percentage,

    -- Net expected
    SUM(slr.total_medicare_payment) - SUM(COALESCE(de.denied_amount, 0)) AS net_expected_payment

FROM claims.encounter e
LEFT JOIN claims.service_line sl ON e.encounter_id = sl.encounter_id
LEFT JOIN claims.service_line_reimbursement slr ON sl.service_line_id = slr.service_line_id
LEFT JOIN claims.denial_event de ON e.encounter_id = de.encounter_id
WHERE e.is_active = true
    AND e.soft_deleted = false
GROUP BY e.organization_id, e.facility_id, DATE_TRUNC('month', e.date_of_service_from);

COMMENT ON VIEW claims.v_reimbursement_analysis IS 'Comprehensive reimbursement and financial analysis';
