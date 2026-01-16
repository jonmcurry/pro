-- Migration: 072_processing_metrics_rollup
-- Description: Add rollup views for processing throughput metrics
-- Date: 2025-01-15

-- ============================================================================
-- Hourly Rollup View
-- ============================================================================

CREATE OR REPLACE VIEW staging.v_processing_metrics_hourly AS
SELECT
    date_trunc('hour', started_at) AS hour,
    metric_type,
    processing_stage,
    COUNT(*) AS batch_count,
    SUM(records_processed) AS total_records,
    ROUND(AVG(records_per_second)::numeric, 2) AS avg_records_per_sec,
    ROUND(MAX(records_per_second)::numeric, 2) AS max_records_per_sec,
    ROUND(MIN(records_per_second)::numeric, 2) AS min_records_per_sec,
    ROUND(SUM(duration_seconds)::numeric, 2) AS total_duration_sec,
    ROUND(AVG(duration_seconds)::numeric, 2) AS avg_duration_sec,
    SUM(success_count) AS total_success,
    SUM(error_count) AS total_errors,
    ROUND(
        CASE WHEN SUM(success_count) + SUM(error_count) > 0
        THEN (SUM(success_count)::numeric / (SUM(success_count) + SUM(error_count)) * 100)
        ELSE 100 END, 2
    ) AS success_rate_pct
FROM staging.processing_metrics
GROUP BY date_trunc('hour', started_at), metric_type, processing_stage;

COMMENT ON VIEW staging.v_processing_metrics_hourly IS
'Hourly aggregated processing metrics for throughput analysis';

-- ============================================================================
-- Daily Rollup View
-- ============================================================================

CREATE OR REPLACE VIEW staging.v_processing_metrics_daily AS
SELECT
    date_trunc('day', started_at) AS day,
    metric_type,
    processing_stage,
    COUNT(*) AS batch_count,
    SUM(records_processed) AS total_records,
    ROUND(AVG(records_per_second)::numeric, 2) AS avg_records_per_sec,
    ROUND(MAX(records_per_second)::numeric, 2) AS max_records_per_sec,
    ROUND(MIN(records_per_second)::numeric, 2) AS min_records_per_sec,
    ROUND(SUM(duration_seconds)::numeric, 2) AS total_duration_sec,
    ROUND(AVG(duration_seconds)::numeric, 2) AS avg_duration_sec,
    SUM(success_count) AS total_success,
    SUM(error_count) AS total_errors,
    ROUND(
        CASE WHEN SUM(success_count) + SUM(error_count) > 0
        THEN (SUM(success_count)::numeric / (SUM(success_count) + SUM(error_count)) * 100)
        ELSE 100 END, 2
    ) AS success_rate_pct,
    -- Calculate effective throughput (records / total time including gaps)
    ROUND(
        CASE WHEN EXTRACT(EPOCH FROM (MAX(completed_at) - MIN(started_at))) > 0
        THEN SUM(records_processed)::numeric / EXTRACT(EPOCH FROM (MAX(completed_at) - MIN(started_at)))
        ELSE 0 END, 2
    ) AS effective_records_per_sec
FROM staging.processing_metrics
GROUP BY date_trunc('day', started_at), metric_type, processing_stage;

COMMENT ON VIEW staging.v_processing_metrics_daily IS
'Daily aggregated processing metrics for throughput analysis';

-- ============================================================================
-- Overall Summary View (Last 24 Hours)
-- ============================================================================

CREATE OR REPLACE VIEW staging.v_processing_summary AS
SELECT
    metric_type,
    processing_stage,
    COUNT(*) AS batch_count,
    SUM(records_processed) AS total_records,
    ROUND(AVG(records_per_second)::numeric, 2) AS avg_records_per_sec,
    ROUND(MAX(records_per_second)::numeric, 2) AS peak_records_per_sec,
    ROUND(SUM(duration_seconds)::numeric, 2) AS total_processing_time_sec,
    ROUND(SUM(duration_seconds) / 60.0, 2) AS total_processing_time_min,
    SUM(success_count) AS total_success,
    SUM(error_count) AS total_errors,
    ROUND(
        CASE WHEN SUM(success_count) + SUM(error_count) > 0
        THEN (SUM(success_count)::numeric / (SUM(success_count) + SUM(error_count)) * 100)
        ELSE 100 END, 2
    ) AS success_rate_pct,
    MIN(started_at) AS first_batch,
    MAX(completed_at) AS last_batch,
    -- Wall clock time from first to last batch
    ROUND(EXTRACT(EPOCH FROM (MAX(completed_at) - MIN(started_at))) / 60.0, 2) AS wall_clock_time_min
FROM staging.processing_metrics
WHERE started_at > NOW() - INTERVAL '24 hours'
GROUP BY metric_type, processing_stage
ORDER BY metric_type, processing_stage;

COMMENT ON VIEW staging.v_processing_summary IS
'Last 24 hours processing summary with throughput and success rates';

-- ============================================================================
-- Stage 2 Specific View (Claims Processing with Rules)
-- ============================================================================

CREATE OR REPLACE VIEW staging.v_stage2_throughput AS
SELECT
    date_trunc('hour', started_at) AS hour,
    COUNT(*) AS batches_processed,
    SUM(records_processed) AS claims_processed,
    ROUND(AVG(records_per_second)::numeric, 2) AS avg_claims_per_sec,
    ROUND(MAX(records_per_second)::numeric, 2) AS peak_claims_per_sec,
    SUM(success_count) AS successful,
    SUM(error_count) AS failed,
    ROUND(SUM(duration_seconds)::numeric, 2) AS processing_time_sec
FROM staging.processing_metrics
WHERE metric_type = 'batch_processing'
  AND processing_stage = 'sequenced_batch_stage2'
  AND started_at > NOW() - INTERVAL '24 hours'
GROUP BY date_trunc('hour', started_at)
ORDER BY hour DESC;

COMMENT ON VIEW staging.v_stage2_throughput IS
'Hourly Stage 2 (claims processing) throughput for the last 24 hours';
