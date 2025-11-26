//! Metrics collector implementation
//!
//! Collects and aggregates performance metrics for various system components.
//!
//! NOTE: This module is scaffolding for future observability features.

#![allow(dead_code)]

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Main metrics collector
#[derive(Debug)]
pub struct MetricsCollector {
    /// Processing throughput metrics
    pub processing: ProcessingMetrics,
    /// Database performance metrics
    pub database: DatabaseMetrics,
    /// Rule execution metrics
    pub rules: RuleMetrics,
    /// Memory usage tracking
    pub memory: MemoryMetrics,
    /// Error tracking
    pub errors: ErrorMetrics,
    /// Provider enrichment metrics
    pub enrichment: EnrichmentMetrics,
    /// When collection started
    pub started_at: Instant,
}

impl MetricsCollector {
    /// Create a new metrics collector
    pub fn new() -> Self {
        Self {
            processing: ProcessingMetrics::new(),
            database: DatabaseMetrics::new(),
            rules: RuleMetrics::new(),
            memory: MemoryMetrics::new(),
            errors: ErrorMetrics::new(),
            enrichment: EnrichmentMetrics::new(),
            started_at: Instant::now(),
        }
    }

    /// Get uptime duration
    pub fn uptime(&self) -> Duration {
        self.started_at.elapsed()
    }

    /// Reset all metrics
    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// Processing throughput metrics
#[derive(Debug, Default)]
pub struct ProcessingMetrics {
    /// Total claims processed
    pub total_processed: u64,
    /// Successfully processed claims
    pub successful: u64,
    /// Failed claims
    pub failed: u64,
    /// Total processing time in milliseconds
    pub total_processing_time_ms: u64,
    /// Current queue depth
    pub queue_depth: usize,
    /// Peak queue depth observed
    pub peak_queue_depth: usize,
    /// Processing times histogram (for percentile calculations)
    processing_times: Vec<u64>,
}

impl ProcessingMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a processing event
    pub fn record(&mut self, duration_ms: u64, success: bool) {
        self.total_processed += 1;
        self.total_processing_time_ms += duration_ms;

        if success {
            self.successful += 1;
        } else {
            self.failed += 1;
        }

        // Keep last 1000 processing times for percentile calculations
        if self.processing_times.len() >= 1000 {
            self.processing_times.remove(0);
        }
        self.processing_times.push(duration_ms);

        // Update peak queue depth
        if self.queue_depth > self.peak_queue_depth {
            self.peak_queue_depth = self.queue_depth;
        }
    }

    /// Calculate claims per second
    pub fn claims_per_second(&self, uptime: Duration) -> f64 {
        let secs = uptime.as_secs_f64();
        if secs > 0.0 {
            self.total_processed as f64 / secs
        } else {
            0.0
        }
    }

    /// Get average processing time in milliseconds
    pub fn avg_processing_time_ms(&self) -> f64 {
        if self.total_processed > 0 {
            self.total_processing_time_ms as f64 / self.total_processed as f64
        } else {
            0.0
        }
    }

    /// Get P50 (median) processing time
    pub fn p50_processing_time_ms(&self) -> u64 {
        self.percentile(50)
    }

    /// Get P95 processing time
    pub fn p95_processing_time_ms(&self) -> u64 {
        self.percentile(95)
    }

    /// Get P99 processing time
    pub fn p99_processing_time_ms(&self) -> u64 {
        self.percentile(99)
    }

    fn percentile(&self, p: u8) -> u64 {
        if self.processing_times.is_empty() {
            return 0;
        }
        let mut sorted = self.processing_times.clone();
        sorted.sort_unstable();
        let idx = (sorted.len() as f64 * p as f64 / 100.0).ceil() as usize;
        sorted.get(idx.saturating_sub(1)).copied().unwrap_or(0)
    }

    /// Get success rate as a percentage
    pub fn success_rate(&self) -> f64 {
        if self.total_processed > 0 {
            (self.successful as f64 / self.total_processed as f64) * 100.0
        } else {
            100.0
        }
    }
}

/// Database performance metrics
#[derive(Debug, Default)]
pub struct DatabaseMetrics {
    /// Query counts by type
    pub query_counts: HashMap<String, u64>,
    /// Total query time by type (ms)
    pub query_times_ms: HashMap<String, u64>,
    /// Failed queries by type
    pub failed_queries: HashMap<String, u64>,
    /// Connection pool stats
    pub pool_size: u32,
    pub pool_idle: u32,
    pub pool_in_use: u32,
}

impl DatabaseMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a database query
    pub fn record_query(&mut self, query_type: &str, duration_ms: u64, success: bool) {
        *self.query_counts.entry(query_type.to_string()).or_insert(0) += 1;
        *self.query_times_ms.entry(query_type.to_string()).or_insert(0) += duration_ms;

        if !success {
            *self.failed_queries.entry(query_type.to_string()).or_insert(0) += 1;
        }
    }

    /// Get average query time for a type
    pub fn avg_query_time_ms(&self, query_type: &str) -> f64 {
        let count = self.query_counts.get(query_type).copied().unwrap_or(0);
        let time = self.query_times_ms.get(query_type).copied().unwrap_or(0);
        if count > 0 {
            time as f64 / count as f64
        } else {
            0.0
        }
    }

    /// Get total queries
    pub fn total_queries(&self) -> u64 {
        self.query_counts.values().sum()
    }

    /// Get total failed queries
    pub fn total_failed(&self) -> u64 {
        self.failed_queries.values().sum()
    }

    /// Update pool statistics
    pub fn update_pool_stats(&mut self, size: u32, idle: u32, in_use: u32) {
        self.pool_size = size;
        self.pool_idle = idle;
        self.pool_in_use = in_use;
    }
}

/// Rule execution metrics
#[derive(Debug, Default)]
pub struct RuleMetrics {
    /// Total rule executions
    pub total_executions: u64,
    /// Successful executions
    pub successful: u64,
    /// Failed executions
    pub failed: u64,
    /// Total flags raised
    pub total_flags_raised: u64,
    /// Execution counts by rule ID
    pub executions_by_rule: HashMap<i64, u64>,
    /// Execution time by rule ID (ms)
    pub time_by_rule_ms: HashMap<i64, u64>,
    /// Flags raised by rule ID
    pub flags_by_rule: HashMap<i64, u64>,
}

impl RuleMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a rule execution
    pub fn record_execution(&mut self, rule_id: i64, duration_ms: u64, success: bool, flags_raised: u32) {
        self.total_executions += 1;
        self.total_flags_raised += flags_raised as u64;

        if success {
            self.successful += 1;
        } else {
            self.failed += 1;
        }

        *self.executions_by_rule.entry(rule_id).or_insert(0) += 1;
        *self.time_by_rule_ms.entry(rule_id).or_insert(0) += duration_ms;
        *self.flags_by_rule.entry(rule_id).or_insert(0) += flags_raised as u64;
    }

    /// Get average execution time for a rule
    pub fn avg_rule_time_ms(&self, rule_id: i64) -> f64 {
        let count = self.executions_by_rule.get(&rule_id).copied().unwrap_or(0);
        let time = self.time_by_rule_ms.get(&rule_id).copied().unwrap_or(0);
        if count > 0 {
            time as f64 / count as f64
        } else {
            0.0
        }
    }

    /// Get flag rate (flags per execution)
    pub fn flag_rate(&self) -> f64 {
        if self.total_executions > 0 {
            self.total_flags_raised as f64 / self.total_executions as f64
        } else {
            0.0
        }
    }
}

/// Memory usage metrics
#[derive(Debug, Default)]
pub struct MemoryMetrics {
    /// Current heap usage in bytes
    pub heap_used: usize,
    /// Peak heap usage observed
    pub peak_heap_used: usize,
    /// Number of allocations
    pub allocations: u64,
}

impl MemoryMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    /// Update memory statistics
    pub fn update(&mut self, heap_used: usize) {
        self.heap_used = heap_used;
        if heap_used > self.peak_heap_used {
            self.peak_heap_used = heap_used;
        }
        self.allocations += 1;
    }

    /// Get heap usage in MB
    pub fn heap_used_mb(&self) -> f64 {
        self.heap_used as f64 / (1024.0 * 1024.0)
    }

    /// Get peak heap usage in MB
    pub fn peak_heap_mb(&self) -> f64 {
        self.peak_heap_used as f64 / (1024.0 * 1024.0)
    }
}

/// Error tracking metrics
#[derive(Debug, Default)]
pub struct ErrorMetrics {
    /// Error counts by type/category
    pub by_type: HashMap<String, u64>,
    /// Recent errors (last 100)
    pub recent: Vec<ErrorEntry>,
    /// Total errors
    pub total: u64,
}

/// A single error entry
#[derive(Debug, Clone)]
pub struct ErrorEntry {
    pub error_type: String,
    pub message: String,
    pub timestamp_ms: u64,
}

impl ErrorMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record an error
    pub fn record(&mut self, error_type: &str, message: &str) {
        self.total += 1;
        *self.by_type.entry(error_type.to_string()).or_insert(0) += 1;

        // Keep last 100 errors
        if self.recent.len() >= 100 {
            self.recent.remove(0);
        }
        self.recent.push(ErrorEntry {
            error_type: error_type.to_string(),
            message: message.to_string(),
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
        });
    }

    /// Get error rate per minute (requires uptime)
    pub fn errors_per_minute(&self, uptime: Duration) -> f64 {
        let mins = uptime.as_secs_f64() / 60.0;
        if mins > 0.0 {
            self.total as f64 / mins
        } else {
            0.0
        }
    }
}

/// Provider enrichment metrics
#[derive(Debug, Default)]
pub struct EnrichmentMetrics {
    /// Total enrichment attempts
    pub total_attempts: u64,
    /// Successful enrichments
    pub successful: u64,
    /// Already enriched (skipped)
    pub already_enriched: u64,
    /// Not found in reference data
    pub not_found: u64,
    /// Errors during enrichment
    pub errors: u64,
}

impl EnrichmentMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record an enrichment attempt
    pub fn record(&mut self, result: EnrichmentResult) {
        self.total_attempts += 1;
        match result {
            EnrichmentResult::Success => self.successful += 1,
            EnrichmentResult::AlreadyEnriched => self.already_enriched += 1,
            EnrichmentResult::NotFound => self.not_found += 1,
            EnrichmentResult::Error => self.errors += 1,
        }
    }

    /// Get enrichment success rate
    pub fn success_rate(&self) -> f64 {
        if self.total_attempts > 0 {
            (self.successful as f64 / self.total_attempts as f64) * 100.0
        } else {
            0.0
        }
    }
}

/// Result of an enrichment attempt
#[derive(Debug, Clone, Copy)]
pub enum EnrichmentResult {
    Success,
    AlreadyEnriched,
    NotFound,
    Error,
}
