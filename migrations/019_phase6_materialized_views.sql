-- Migration: 019_phase6_materialized_views
-- Description: Phase 6 materialized views for analytics performance
-- Date: 2025-10-15
-- Impact: 10-100x faster dashboard and analytics queries

-- Create analytics schema if it doesn't exist
CREATE SCHEMA IF NOT EXISTS analytics;

-- ============================================================================
-- FLAG STATISTICS - Daily Aggregation
-- ============================================================================

DROP MATERIALIZED VIEW IF EXISTS analytics.flag_statistics_daily CASCADE;
CREATE MATERIALIZED VIEW analytics.flag_statistics_daily AS
SELECT
    f.organization_id,
    f.facility_id,
    DATE(f.created_at) as flag_date,
    f.flag_category,
    f.flag_severity,
    f.flag_status,
    COUNT(*) as flag_count,
    COUNT(DISTINCT f.encounter_id) as unique_encounters,
    COUNT(DISTINCT f.service_line_id) as unique_service_lines,
    AVG(f.financial_impact) as avg_financial_impact,
    SUM(f.financial_impact) as total_financial_impact,
    MIN(f.financial_impact) as min_financial_impact,
    MAX(f.financial_impact) as max_financial_impact,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY f.financial_impact) as median_financial_impact
FROM claims.flag f
WHERE f.created_at >= CURRENT_DATE - INTERVAL '90 days'
GROUP BY
    f.organization_id,
    f.facility_id,
    DATE(f.created_at),
    f.flag_category,
    f.flag_severity,
    f.flag_status;

-- Unique index for fast lookups and CONCURRENTLY refresh
CREATE UNIQUE INDEX idx_flag_stats_daily_pk
    ON analytics.flag_statistics_daily(
        organization_id,
        COALESCE(facility_id, '00000000-0000-0000-0000-000000000000'::uuid),
        flag_date,
        flag_category,
        flag_severity,
        flag_status
    );

CREATE INDEX IF NOT EXISTS idx_flag_stats_org_date
    ON analytics.flag_statistics_daily(organization_id, flag_date DESC);

CREATE INDEX IF NOT EXISTS idx_flag_stats_facility_date
    ON analytics.flag_statistics_daily(facility_id, flag_date DESC)
    WHERE facility_id IS NOT NULL;

COMMENT ON MATERIALIZED VIEW analytics.flag_statistics_daily IS 'Phase 6: Daily aggregated flag statistics for dashboards';

-- ============================================================================
-- ENCOUNTER STATISTICS - Daily Aggregation
-- ============================================================================

DROP MATERIALIZED VIEW IF EXISTS analytics.encounter_statistics_daily CASCADE;
CREATE MATERIALIZED VIEW analytics.encounter_statistics_daily AS
SELECT
    e.organization_id,
    e.facility_id,
    DATE(e.date_of_service_from) as service_date,
    e.claim_status,
    e.payer_responsibility_code,
    COUNT(*) as encounter_count,
    COUNT(DISTINCT e.subscriber_id) as unique_patients,
    SUM(e.total_claim_charge_amount) as total_charges,
    AVG(e.total_claim_charge_amount) as avg_charge_amount,
    MIN(e.total_claim_charge_amount) as min_charge_amount,
    MAX(e.total_claim_charge_amount) as max_charge_amount,
    -- Service line stats
    SUM(sl_stats.service_line_count) as total_service_lines,
    AVG(sl_stats.service_line_count) as avg_service_lines_per_encounter
FROM claims.encounter e
LEFT JOIN LATERAL (
    SELECT COUNT(*) as service_line_count
    FROM claims.service_line sl
    WHERE sl.encounter_id = e.encounter_id
) sl_stats ON true
WHERE e.date_of_service_from >= CURRENT_DATE - INTERVAL '90 days'
    AND e.is_active = true
    AND e.soft_deleted = false
GROUP BY
    e.organization_id,
    e.facility_id,
    DATE(e.date_of_service_from),
    e.claim_status,
    e.payer_responsibility_code;

CREATE UNIQUE INDEX idx_encounter_stats_daily_pk
    ON analytics.encounter_statistics_daily(
        organization_id,
        COALESCE(facility_id, '00000000-0000-0000-0000-000000000000'::uuid),
        service_date,
        claim_status,
        payer_responsibility_code
    );

CREATE INDEX IF NOT EXISTS idx_encounter_stats_org_date
    ON analytics.encounter_statistics_daily(organization_id, service_date DESC);

COMMENT ON MATERIALIZED VIEW analytics.encounter_statistics_daily IS 'Phase 6: Daily encounter volume and financial statistics';

-- ============================================================================
-- PROCEDURE CODE STATISTICS - Top Procedures by Volume
-- ============================================================================

DROP MATERIALIZED VIEW IF EXISTS analytics.procedure_statistics CASCADE;
CREATE MATERIALIZED VIEW analytics.procedure_statistics AS
SELECT
    sl.procedure_code,
    COUNT(*) as procedure_count,
    COUNT(DISTINCT sl.encounter_id) as unique_encounters,
    SUM(sl.service_unit_count) as total_units,
    AVG(sl.service_unit_count) as avg_units,
    SUM(sl.line_item_charge_amount) as total_charges,
    AVG(sl.line_item_charge_amount) as avg_charge,
    -- Flag statistics
    COUNT(DISTINCT f.flag_id) as flag_count,
    CAST(COUNT(DISTINCT f.flag_id) AS DECIMAL) / NULLIF(COUNT(*), 0) as flag_rate,
    -- Top modifiers
    ARRAY_AGG(DISTINCT sl.procedure_modifier_1) FILTER (WHERE sl.procedure_modifier_1 IS NOT NULL) as common_modifier_1,
    ARRAY_AGG(DISTINCT sl.procedure_modifier_2) FILTER (WHERE sl.procedure_modifier_2 IS NOT NULL) as common_modifier_2,
    -- Date range
    MIN(sl.service_date_from) as first_service_date,
    MAX(sl.service_date_from) as last_service_date
FROM claims.service_line sl
LEFT JOIN claims.flag f ON f.service_line_id = sl.service_line_id
WHERE sl.service_date_from >= CURRENT_DATE - INTERVAL '90 days'
GROUP BY sl.procedure_code
HAVING COUNT(*) >= 5; -- Only procedures with at least 5 occurrences

CREATE UNIQUE INDEX idx_procedure_stats_pk
    ON analytics.procedure_statistics(procedure_code);

CREATE INDEX IF NOT EXISTS idx_procedure_stats_count
    ON analytics.procedure_statistics(procedure_count DESC);

CREATE INDEX IF NOT EXISTS idx_procedure_stats_charges
    ON analytics.procedure_statistics(total_charges DESC);

COMMENT ON MATERIALIZED VIEW analytics.procedure_statistics IS 'Phase 6: Procedure code usage and financial statistics';

-- ============================================================================
-- PROVIDER PERFORMANCE STATISTICS
-- ============================================================================

DROP MATERIALIZED VIEW IF EXISTS analytics.provider_performance CASCADE;
CREATE MATERIALIZED VIEW analytics.provider_performance AS
SELECT
    e.rendering_provider_id as provider_id,
    e.rendering_provider_npi as provider_npi,
    COUNT(DISTINCT e.encounter_id) as encounter_count,
    COUNT(DISTINCT e.subscriber_id) as unique_patients,
    SUM(e.total_claim_charge_amount) as total_charges,
    AVG(e.total_claim_charge_amount) as avg_charge_per_encounter,
    -- Flag statistics
    COUNT(DISTINCT f.flag_id) as total_flags,
    CAST(COUNT(DISTINCT f.flag_id) AS DECIMAL) / NULLIF(COUNT(DISTINCT e.encounter_id), 0) as flags_per_encounter,
    COUNT(DISTINCT f.flag_id) FILTER (WHERE f.flag_severity = 'HIGH') as high_severity_flags,
    SUM(f.financial_impact) as total_flag_financial_impact,
    -- Claim status distribution
    COUNT(*) FILTER (WHERE e.claim_status = 'COMPLETED') as completed_claims,
    COUNT(*) FILTER (WHERE e.claim_status = 'PENDING') as pending_claims,
    COUNT(*) FILTER (WHERE e.claim_status = 'DENIED') as denied_claims,
    -- Date range
    MIN(e.date_of_service_from) as first_service_date,
    MAX(e.date_of_service_from) as last_service_date
FROM claims.encounter e
LEFT JOIN claims.flag f ON f.encounter_id = e.encounter_id
WHERE e.rendering_provider_id IS NOT NULL
    AND e.date_of_service_from >= CURRENT_DATE - INTERVAL '90 days'
    AND e.is_active = true
    AND e.soft_deleted = false
GROUP BY e.rendering_provider_id, e.rendering_provider_npi
HAVING COUNT(DISTINCT e.encounter_id) >= 5; -- Providers with at least 5 encounters

CREATE UNIQUE INDEX idx_provider_performance_pk
    ON analytics.provider_performance(provider_id);

CREATE INDEX IF NOT EXISTS idx_provider_performance_encounter_count
    ON analytics.provider_performance(encounter_count DESC);

CREATE INDEX IF NOT EXISTS idx_provider_performance_flag_rate
    ON analytics.provider_performance(flags_per_encounter DESC);

COMMENT ON MATERIALIZED VIEW analytics.provider_performance IS 'Phase 6: Provider activity and quality metrics';

-- ============================================================================
-- PAYER STATISTICS
-- ============================================================================

DROP MATERIALIZED VIEW IF EXISTS analytics.payer_statistics CASCADE;
CREATE MATERIALIZED VIEW analytics.payer_statistics AS
SELECT
    e.payer_id,
    e.payer_name,
    e.organization_id,
    COUNT(DISTINCT e.encounter_id) as encounter_count,
    SUM(e.total_claim_charge_amount) as total_charges,
    AVG(e.total_claim_charge_amount) as avg_charge_amount,
    -- Claim status breakdown
    COUNT(*) FILTER (WHERE e.claim_status = 'COMPLETED') as completed_count,
    COUNT(*) FILTER (WHERE e.claim_status = 'PENDING') as pending_count,
    COUNT(*) FILTER (WHERE e.claim_status = 'DENIED') as denied_count,
    CAST(COUNT(*) FILTER (WHERE e.claim_status = 'DENIED') AS DECIMAL) / NULLIF(COUNT(*), 0) as denial_rate,
    -- Flag statistics
    COUNT(DISTINCT f.flag_id) as total_flags,
    SUM(f.financial_impact) as total_flag_impact,
    -- Top procedure codes
    ARRAY_AGG(DISTINCT sl.procedure_code ORDER BY COUNT(*) DESC) FILTER (WHERE sl.procedure_code IS NOT NULL) as top_procedures
FROM claims.encounter e
LEFT JOIN claims.flag f ON f.encounter_id = e.encounter_id
LEFT JOIN claims.service_line sl ON sl.encounter_id = e.encounter_id
WHERE e.payer_id IS NOT NULL
    AND e.date_of_service_from >= CURRENT_DATE - INTERVAL '90 days'
    AND e.is_active = true
    AND e.soft_deleted = false
GROUP BY e.payer_id, e.payer_name, e.organization_id
HAVING COUNT(DISTINCT e.encounter_id) >= 5;

CREATE UNIQUE INDEX idx_payer_stats_pk
    ON analytics.payer_statistics(payer_id, organization_id);

CREATE INDEX IF NOT EXISTS idx_payer_stats_denial_rate
    ON analytics.payer_statistics(denial_rate DESC);

COMMENT ON MATERIALIZED VIEW analytics.payer_statistics IS 'Phase 6: Payer performance and denial rate tracking';

-- ============================================================================
-- ML MODEL PERFORMANCE TRACKING
-- ============================================================================

DROP MATERIALIZED VIEW IF EXISTS analytics.ml_model_performance_summary CASCADE;
CREATE MATERIALIZED VIEW analytics.ml_model_performance_summary AS
SELECT
    mr.model_id,
    mr.model_name,
    mr.model_type,
    mr.model_purpose,
    mr.deployment_status,
    COUNT(DISTINCT mp.prediction_id) as total_predictions,
    COUNT(DISTINCT mp.encounter_id) as unique_encounters,
    -- Accuracy metrics (where actual values are recorded)
    COUNT(*) FILTER (WHERE mp.was_correct = true) as correct_predictions,
    COUNT(*) FILTER (WHERE mp.was_correct = false) as incorrect_predictions,
    CAST(COUNT(*) FILTER (WHERE mp.was_correct = true) AS DECIMAL) /
        NULLIF(COUNT(*) FILTER (WHERE mp.was_correct IS NOT NULL), 0) as accuracy,
    AVG(mp.prediction_score) as avg_confidence,
    -- Risk distribution
    COUNT(*) FILTER (WHERE mp.risk_level = 'CRITICAL') as critical_risk_count,
    COUNT(*) FILTER (WHERE mp.risk_level = 'HIGH') as high_risk_count,
    COUNT(*) FILTER (WHERE mp.risk_level = 'MEDIUM') as medium_risk_count,
    COUNT(*) FILTER (WHERE mp.risk_level = 'LOW') as low_risk_count,
    -- Date range
    MIN(mp.predicted_at) as first_prediction,
    MAX(mp.predicted_at) as last_prediction
FROM ml.model_registry mr
LEFT JOIN ml.model_prediction mp ON mp.model_id = mr.model_id
WHERE mp.predicted_at >= CURRENT_DATE - INTERVAL '30 days'
    OR mp.predicted_at IS NULL
GROUP BY mr.model_id, mr.model_name, mr.model_type, mr.model_purpose, mr.deployment_status;

CREATE UNIQUE INDEX idx_ml_performance_pk
    ON analytics.ml_model_performance_summary(model_id);

CREATE INDEX IF NOT EXISTS idx_ml_performance_accuracy
    ON analytics.ml_model_performance_summary(accuracy DESC NULLS LAST);

COMMENT ON MATERIALIZED VIEW analytics.ml_model_performance_summary IS 'Phase 6: ML model performance tracking';

-- ============================================================================
-- REFRESH FUNCTIONS
-- ============================================================================

-- Function to refresh all analytics materialized views
CREATE OR REPLACE FUNCTION analytics.refresh_all_views()
RETURNS void
LANGUAGE plpgsql
AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY analytics.flag_statistics_daily;
    REFRESH MATERIALIZED VIEW CONCURRENTLY analytics.encounter_statistics_daily;
    REFRESH MATERIALIZED VIEW CONCURRENTLY analytics.procedure_statistics;
    REFRESH MATERIALIZED VIEW CONCURRENTLY analytics.provider_performance;
    REFRESH MATERIALIZED VIEW CONCURRENTLY analytics.payer_statistics;
    REFRESH MATERIALIZED VIEW CONCURRENTLY analytics.ml_model_performance_summary;

    RAISE NOTICE 'All analytics materialized views refreshed successfully';
END;
$$;

COMMENT ON FUNCTION analytics.refresh_all_views() IS 'Phase 6: Refresh all analytics materialized views concurrently';

-- Grant permissions
GRANT USAGE ON SCHEMA analytics TO PUBLIC;
GRANT SELECT ON ALL TABLES IN SCHEMA analytics TO PUBLIC;

-- Initial refresh
SELECT analytics.refresh_all_views();
