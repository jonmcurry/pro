//! Comprehensive metrics collection for Professional SMART
//!
//! This module provides detailed performance and business metrics tracking
//! for monitoring system health, performance, and processing efficiency.
//!
//! NOTE: This module is scaffolding for future observability features. Currently unused
//! but retained for planned integration with the claims processing pipeline.

#![allow(dead_code)]
#![allow(unused_imports)]

pub mod collector;
pub mod reporter;

pub use collector::{MetricsCollector, ProcessingMetrics, DatabaseMetrics, RuleMetrics};
pub use reporter::MetricsReporter;

use std::sync::Arc;
use tokio::sync::RwLock;

/// Global metrics instance
static METRICS: once_cell::sync::OnceCell<Arc<RwLock<MetricsCollector>>> =
    once_cell::sync::OnceCell::new();

/// Initialize the global metrics collector
pub fn init() {
    let _ = METRICS.set(Arc::new(RwLock::new(MetricsCollector::new())));
}

/// Get the global metrics collector
pub fn get() -> Arc<RwLock<MetricsCollector>> {
    METRICS
        .get()
        .cloned()
        .unwrap_or_else(|| Arc::new(RwLock::new(MetricsCollector::new())))
}

/// Record a processing event
pub async fn record_processing(duration_ms: u64, success: bool) {
    let metrics = get();
    let mut m = metrics.write().await;
    m.processing.record(duration_ms, success);
}

/// Record a database query
pub async fn record_db_query(query_type: &str, duration_ms: u64, success: bool) {
    let metrics = get();
    let mut m = metrics.write().await;
    m.database.record_query(query_type, duration_ms, success);
}

/// Record a rule execution
pub async fn record_rule_execution(rule_id: i64, duration_ms: u64, success: bool, flags_raised: u32) {
    let metrics = get();
    let mut m = metrics.write().await;
    m.rules.record_execution(rule_id, duration_ms, success, flags_raised);
}

/// Record queue depth
pub async fn record_queue_depth(depth: usize) {
    let metrics = get();
    let mut m = metrics.write().await;
    m.processing.queue_depth = depth;
}
