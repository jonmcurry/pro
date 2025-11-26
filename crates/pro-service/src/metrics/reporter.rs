//! Metrics reporter implementation
//!
//! Formats and outputs collected metrics in various formats.
//!
//! NOTE: This module is scaffolding for future observability features.

#![allow(dead_code)]

use super::collector::MetricsCollector;
use serde::Serialize;

/// Metrics reporter for formatting output
pub struct MetricsReporter;

impl MetricsReporter {
    /// Generate a summary report as JSON
    pub fn to_json(metrics: &MetricsCollector) -> serde_json::Value {
        let uptime = metrics.uptime();
        serde_json::json!({
            "uptime_seconds": uptime.as_secs(),
            "processing": {
                "total_processed": metrics.processing.total_processed,
                "successful": metrics.processing.successful,
                "failed": metrics.processing.failed,
                "success_rate_pct": format!("{:.2}", metrics.processing.success_rate()),
                "claims_per_second": format!("{:.2}", metrics.processing.claims_per_second(uptime)),
                "avg_processing_time_ms": format!("{:.2}", metrics.processing.avg_processing_time_ms()),
                "p50_ms": metrics.processing.p50_processing_time_ms(),
                "p95_ms": metrics.processing.p95_processing_time_ms(),
                "p99_ms": metrics.processing.p99_processing_time_ms(),
                "queue_depth": metrics.processing.queue_depth,
                "peak_queue_depth": metrics.processing.peak_queue_depth
            },
            "database": {
                "total_queries": metrics.database.total_queries(),
                "total_failed": metrics.database.total_failed(),
                "pool_size": metrics.database.pool_size,
                "pool_idle": metrics.database.pool_idle,
                "pool_in_use": metrics.database.pool_in_use,
                "query_counts": metrics.database.query_counts,
                "avg_times_ms": Self::compute_avg_times(&metrics.database)
            },
            "rules": {
                "total_executions": metrics.rules.total_executions,
                "successful": metrics.rules.successful,
                "failed": metrics.rules.failed,
                "total_flags_raised": metrics.rules.total_flags_raised,
                "flag_rate": format!("{:.3}", metrics.rules.flag_rate())
            },
            "memory": {
                "heap_used_mb": format!("{:.2}", metrics.memory.heap_used_mb()),
                "peak_heap_mb": format!("{:.2}", metrics.memory.peak_heap_mb()),
                "allocations": metrics.memory.allocations
            },
            "errors": {
                "total": metrics.errors.total,
                "per_minute": format!("{:.2}", metrics.errors.errors_per_minute(uptime)),
                "by_type": metrics.errors.by_type,
                "recent_count": metrics.errors.recent.len()
            },
            "enrichment": {
                "total_attempts": metrics.enrichment.total_attempts,
                "successful": metrics.enrichment.successful,
                "already_enriched": metrics.enrichment.already_enriched,
                "not_found": metrics.enrichment.not_found,
                "errors": metrics.enrichment.errors,
                "success_rate_pct": format!("{:.2}", metrics.enrichment.success_rate())
            }
        })
    }

    /// Generate a compact summary string
    pub fn summary(metrics: &MetricsCollector) -> String {
        let uptime = metrics.uptime();
        format!(
            "Uptime: {}s | Processed: {} ({:.1}/s) | Success: {:.1}% | Queue: {} | Flags: {} | Errors: {}",
            uptime.as_secs(),
            metrics.processing.total_processed,
            metrics.processing.claims_per_second(uptime),
            metrics.processing.success_rate(),
            metrics.processing.queue_depth,
            metrics.rules.total_flags_raised,
            metrics.errors.total
        )
    }

    /// Generate detailed processing stats
    pub fn processing_stats(metrics: &MetricsCollector) -> ProcessingStats {
        let uptime = metrics.uptime();
        ProcessingStats {
            uptime_seconds: uptime.as_secs(),
            total_processed: metrics.processing.total_processed,
            successful: metrics.processing.successful,
            failed: metrics.processing.failed,
            success_rate: metrics.processing.success_rate(),
            claims_per_second: metrics.processing.claims_per_second(uptime),
            avg_time_ms: metrics.processing.avg_processing_time_ms(),
            p50_ms: metrics.processing.p50_processing_time_ms(),
            p95_ms: metrics.processing.p95_processing_time_ms(),
            p99_ms: metrics.processing.p99_processing_time_ms(),
            queue_depth: metrics.processing.queue_depth,
            peak_queue_depth: metrics.processing.peak_queue_depth,
        }
    }

    /// Generate database stats
    pub fn database_stats(metrics: &MetricsCollector) -> DatabaseStats {
        DatabaseStats {
            total_queries: metrics.database.total_queries(),
            failed_queries: metrics.database.total_failed(),
            pool_size: metrics.database.pool_size,
            pool_idle: metrics.database.pool_idle,
            pool_in_use: metrics.database.pool_in_use,
            query_types: metrics.database.query_counts.clone(),
        }
    }

    fn compute_avg_times(db: &super::collector::DatabaseMetrics) -> std::collections::HashMap<String, String> {
        db.query_counts
            .keys()
            .map(|k| (k.clone(), format!("{:.2}", db.avg_query_time_ms(k))))
            .collect()
    }
}

/// Serializable processing statistics
#[derive(Debug, Clone, Serialize)]
pub struct ProcessingStats {
    pub uptime_seconds: u64,
    pub total_processed: u64,
    pub successful: u64,
    pub failed: u64,
    pub success_rate: f64,
    pub claims_per_second: f64,
    pub avg_time_ms: f64,
    pub p50_ms: u64,
    pub p95_ms: u64,
    pub p99_ms: u64,
    pub queue_depth: usize,
    pub peak_queue_depth: usize,
}

/// Serializable database statistics
#[derive(Debug, Clone, Serialize)]
pub struct DatabaseStats {
    pub total_queries: u64,
    pub failed_queries: u64,
    pub pool_size: u32,
    pub pool_idle: u32,
    pub pool_in_use: u32,
    pub query_types: std::collections::HashMap<String, u64>,
}
