-- Migration: 057_create_dashboard_materialized_views
-- Description: Create materialized views for dashboard performance optimization
-- Date: 2025-11-26
-- Purpose: Pre-compute expensive aggregations for dashboard queries

-- ==============================================================================
-- MATERIALIZED VIEW: Management Overview
-- ==============================================================================
-- Refreshes: Daily or on-demand
-- Pre-computes: Monthly metrics across 6+ tables with 15+ aggregations

CREATE MATERIALIZED VIEW IF NOT EXISTS claims.mv_management_overview AS
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
    ROUND(100.0 * COUNT(DISTINCT de.denial_id) / NULLIF(COUNT(DISTINCT e.encounter_id), 0), 2) AS denial_rate_percentage,

    -- Metadata
    CURRENT_TIMESTAMP AS last_refreshed

FROM claims.encounter e
LEFT JOIN claims.service_line sl ON e.encounter_id = sl.encounter_id
LEFT JOIN claims.service_line_reimbursement slr ON sl.service_line_id = slr.service_line_id
LEFT JOIN claims.encounter_flag ef ON e.encounter_id = ef.encounter_id AND ef.flag_status = 'OPEN'
LEFT JOIN claims.denial_event de ON e.encounter_id = de.encounter_id
WHERE e.is_active = true
    AND e.soft_deleted = false
GROUP BY e.organization_id, e.facility_id, DATE_TRUNC('month', e.date_of_service_from);

-- Create unique index for concurrent refresh
CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_management_overview_pk
ON claims.mv_management_overview(organization_id, facility_id, month);

-- Create indexes for common filters
CREATE INDEX IF NOT EXISTS idx_mv_management_overview_org
ON claims.mv_management_overview(organization_id);

CREATE INDEX IF NOT EXISTS idx_mv_management_overview_month
ON claims.mv_management_overview(month DESC);

COMMENT ON MATERIALIZED VIEW claims.mv_management_overview IS
'Pre-computed management dashboard metrics. Refresh daily with: REFRESH MATERIALIZED VIEW CONCURRENTLY claims.mv_management_overview';

-- ==============================================================================
-- MATERIALIZED VIEW: Denial by Payer
-- ==============================================================================
-- Refreshes: Daily or on-demand
-- Pre-computes: Monthly denial metrics by payer with root cause breakdown

CREATE MATERIALIZED VIEW IF NOT EXISTS claims.mv_denial_by_payer AS
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
        NULLIF(COUNT(DISTINCT CASE WHEN de.appeal_filed THEN de.denial_id END), 0), 2) AS appeal_success_rate,

    -- Metadata
    CURRENT_TIMESTAMP AS last_refreshed

FROM claims.denial_event de
GROUP BY de.organization_id, de.facility_id, de.payer_id, de.payer_name, DATE_TRUNC('month', de.denial_date);

-- Create unique index for concurrent refresh
CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_denial_by_payer_pk
ON claims.mv_denial_by_payer(organization_id, facility_id, COALESCE(payer_id, ''), month);

-- Create indexes for common filters
CREATE INDEX IF NOT EXISTS idx_mv_denial_by_payer_org
ON claims.mv_denial_by_payer(organization_id);

CREATE INDEX IF NOT EXISTS idx_mv_denial_by_payer_payer
ON claims.mv_denial_by_payer(payer_id);

CREATE INDEX IF NOT EXISTS idx_mv_denial_by_payer_month
ON claims.mv_denial_by_payer(month DESC);

COMMENT ON MATERIALIZED VIEW claims.mv_denial_by_payer IS
'Pre-computed denial metrics by payer. Refresh daily with: REFRESH MATERIALIZED VIEW CONCURRENTLY claims.mv_denial_by_payer';

-- ==============================================================================
-- MATERIALIZED VIEW: Procedure Volume
-- ==============================================================================
-- Refreshes: Daily or on-demand
-- Pre-computes: Monthly procedure volume with RVU and flag metrics

CREATE MATERIALIZED VIEW IF NOT EXISTS claims.mv_procedure_volume AS
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
        NULLIF(COUNT(DISTINCT sl.service_line_id), 0), 2) AS flag_rate_percentage,

    -- Metadata
    CURRENT_TIMESTAMP AS last_refreshed

FROM claims.service_line sl
INNER JOIN claims.encounter e ON sl.encounter_id = e.encounter_id
LEFT JOIN claims.service_line_reimbursement slr ON sl.service_line_id = slr.service_line_id
LEFT JOIN claims.service_line_flag slf ON sl.service_line_id = slf.service_line_id AND slf.flag_status = 'OPEN'
WHERE e.is_active = true
    AND e.soft_deleted = false
GROUP BY e.organization_id, e.facility_id, sl.procedure_code, sl.procedure_description,
    DATE_TRUNC('month', sl.service_date_from);

-- Create unique index for concurrent refresh
CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_procedure_volume_pk
ON claims.mv_procedure_volume(organization_id, facility_id, procedure_code, month);

-- Create indexes for common filters
CREATE INDEX IF NOT EXISTS idx_mv_procedure_volume_org
ON claims.mv_procedure_volume(organization_id);

CREATE INDEX IF NOT EXISTS idx_mv_procedure_volume_proc
ON claims.mv_procedure_volume(procedure_code);

CREATE INDEX IF NOT EXISTS idx_mv_procedure_volume_month
ON claims.mv_procedure_volume(month DESC);

COMMENT ON MATERIALIZED VIEW claims.mv_procedure_volume IS
'Pre-computed procedure volume metrics. Refresh daily with: REFRESH MATERIALIZED VIEW CONCURRENTLY claims.mv_procedure_volume';

-- ==============================================================================
-- MATERIALIZED VIEW: Provider Productivity
-- ==============================================================================
-- Refreshes: Daily or on-demand
-- Pre-computes: Monthly provider productivity with RVU analysis

CREATE MATERIALIZED VIEW IF NOT EXISTS claims.mv_provider_productivity AS
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
    COUNT(DISTINCT e.date_of_service_from) AS unique_service_days,

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
    COUNT(DISTINCT CASE WHEN sl.procedure_code NOT LIKE '99___' THEN sl.service_line_id END) AS non_em_procedure_count,

    -- Calculated productivity
    ROUND(COUNT(DISTINCT e.encounter_id)::NUMERIC / NULLIF(COUNT(DISTINCT e.date_of_service_from), 0), 2) AS avg_encounters_per_day,
    ROUND(SUM(slr.total_rvu) / NULLIF(COUNT(DISTINCT e.date_of_service_from), 0), 2) AS avg_rvus_per_day,

    -- Metadata
    CURRENT_TIMESTAMP AS last_refreshed

FROM claims.provider p
INNER JOIN claims.encounter e ON p.provider_id = e.rendering_provider_id
LEFT JOIN claims.service_line sl ON e.encounter_id = sl.encounter_id
LEFT JOIN claims.service_line_reimbursement slr ON sl.service_line_id = slr.service_line_id
WHERE p.is_active = true
    AND e.is_active = true
    AND e.soft_deleted = false
GROUP BY p.provider_id, p.npi, p.last_name, p.first_name, p.specialty,
    e.organization_id, e.facility_id, DATE_TRUNC('month', e.date_of_service_from);

-- Create unique index for concurrent refresh
CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_provider_productivity_pk
ON claims.mv_provider_productivity(provider_id, organization_id, facility_id, month);

-- Create indexes for common filters
CREATE INDEX IF NOT EXISTS idx_mv_provider_productivity_org
ON claims.mv_provider_productivity(organization_id);

CREATE INDEX IF NOT EXISTS idx_mv_provider_productivity_provider
ON claims.mv_provider_productivity(provider_id);

CREATE INDEX IF NOT EXISTS idx_mv_provider_productivity_month
ON claims.mv_provider_productivity(month DESC);

COMMENT ON MATERIALIZED VIEW claims.mv_provider_productivity IS
'Pre-computed provider productivity metrics. Refresh daily with: REFRESH MATERIALIZED VIEW CONCURRENTLY claims.mv_provider_productivity';

-- ==============================================================================
-- REFRESH FUNCTION
-- ==============================================================================
-- Refreshes all dashboard materialized views

CREATE OR REPLACE FUNCTION claims.refresh_dashboard_views()
RETURNS TABLE (
    view_name TEXT,
    refresh_status TEXT,
    duration_ms INTEGER
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_start TIMESTAMP;
    v_duration INTEGER;
BEGIN
    -- Refresh management overview
    v_start := clock_timestamp();
    BEGIN
        REFRESH MATERIALIZED VIEW CONCURRENTLY claims.mv_management_overview;
        v_duration := EXTRACT(MILLISECONDS FROM (clock_timestamp() - v_start))::INTEGER;
        view_name := 'mv_management_overview';
        refresh_status := 'SUCCESS';
        duration_ms := v_duration;
        RETURN NEXT;
    EXCEPTION WHEN OTHERS THEN
        view_name := 'mv_management_overview';
        refresh_status := 'FAILED: ' || SQLERRM;
        duration_ms := 0;
        RETURN NEXT;
    END;

    -- Refresh denial by payer
    v_start := clock_timestamp();
    BEGIN
        REFRESH MATERIALIZED VIEW CONCURRENTLY claims.mv_denial_by_payer;
        v_duration := EXTRACT(MILLISECONDS FROM (clock_timestamp() - v_start))::INTEGER;
        view_name := 'mv_denial_by_payer';
        refresh_status := 'SUCCESS';
        duration_ms := v_duration;
        RETURN NEXT;
    EXCEPTION WHEN OTHERS THEN
        view_name := 'mv_denial_by_payer';
        refresh_status := 'FAILED: ' || SQLERRM;
        duration_ms := 0;
        RETURN NEXT;
    END;

    -- Refresh procedure volume
    v_start := clock_timestamp();
    BEGIN
        REFRESH MATERIALIZED VIEW CONCURRENTLY claims.mv_procedure_volume;
        v_duration := EXTRACT(MILLISECONDS FROM (clock_timestamp() - v_start))::INTEGER;
        view_name := 'mv_procedure_volume';
        refresh_status := 'SUCCESS';
        duration_ms := v_duration;
        RETURN NEXT;
    EXCEPTION WHEN OTHERS THEN
        view_name := 'mv_procedure_volume';
        refresh_status := 'FAILED: ' || SQLERRM;
        duration_ms := 0;
        RETURN NEXT;
    END;

    -- Refresh provider productivity
    v_start := clock_timestamp();
    BEGIN
        REFRESH MATERIALIZED VIEW CONCURRENTLY claims.mv_provider_productivity;
        v_duration := EXTRACT(MILLISECONDS FROM (clock_timestamp() - v_start))::INTEGER;
        view_name := 'mv_provider_productivity';
        refresh_status := 'SUCCESS';
        duration_ms := v_duration;
        RETURN NEXT;
    EXCEPTION WHEN OTHERS THEN
        view_name := 'mv_provider_productivity';
        refresh_status := 'FAILED: ' || SQLERRM;
        duration_ms := 0;
        RETURN NEXT;
    END;
END;
$$;

COMMENT ON FUNCTION claims.refresh_dashboard_views IS
'Refreshes all dashboard materialized views. Call daily via pg_cron or scheduled task.
Usage: SELECT * FROM claims.refresh_dashboard_views();';


-- ==============================================================================
-- USAGE EXAMPLES
-- ==============================================================================
-- View current data:
--   SELECT * FROM claims.mv_management_overview WHERE organization_id = 1 ORDER BY month DESC LIMIT 12;
--   SELECT * FROM claims.mv_denial_by_payer WHERE organization_id = 1 ORDER BY month DESC LIMIT 12;
--   SELECT * FROM claims.mv_procedure_volume WHERE organization_id = 1 ORDER BY procedure_count DESC LIMIT 100;
--   SELECT * FROM claims.mv_provider_productivity WHERE organization_id = 1 ORDER BY total_work_rvus DESC LIMIT 50;
--
-- Refresh all views:
--   SELECT * FROM claims.refresh_dashboard_views();
--
-- Manual refresh single view:
--   REFRESH MATERIALIZED VIEW CONCURRENTLY claims.mv_management_overview;
--
-- Check when views were last refreshed:
--   SELECT 'mv_management_overview' AS view_name, MAX(last_refreshed) FROM claims.mv_management_overview
--   UNION ALL
--   SELECT 'mv_denial_by_payer', MAX(last_refreshed) FROM claims.mv_denial_by_payer
--   UNION ALL
--   SELECT 'mv_procedure_volume', MAX(last_refreshed) FROM claims.mv_procedure_volume
--   UNION ALL
--   SELECT 'mv_provider_productivity', MAX(last_refreshed) FROM claims.mv_provider_productivity;
--
-- Schedule daily refresh with pg_cron (if installed):
--   SELECT cron.schedule('refresh-dashboard-views', '0 2 * * *', 'SELECT * FROM claims.refresh_dashboard_views()');
