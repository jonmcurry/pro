//! Batch Manager Module
//!
//! Handles batch status management and metrics logging.
//! Extracted from ClaimsProcessor as part of god object refactoring.
//!
//! NOTE: This module is scaffolding for future refactoring. Currently unused
//! but retained for planned integration with the claims processing pipeline.

#![allow(dead_code)]

use anyhow::{Context, Result};
use sqlx::PgPool;
use std::collections::HashSet;
use tracing::warn;

/// Manager for batch status and metrics
pub struct BatchManager {
    pool: PgPool,
}

impl BatchManager {
    /// Create a new batch manager
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Get reference to the database pool
    pub fn pool(&self) -> &PgPool {
        &self.pool
    }

    /// Update batch status to PROCESSING for the given batch IDs
    pub async fn mark_batches_processing(&self, batch_ids: &[i64]) -> Result<()> {
        if batch_ids.is_empty() {
            return Ok(());
        }

        sqlx::query(
            r#"
            UPDATE staging.import_batch
            SET import_status = 'PROCESSING'
            WHERE batch_id = ANY($1) AND import_status = 'INGESTED'
            "#,
        )
        .bind(batch_ids)
        .execute(&self.pool)
        .await
        .context("Failed to update batch status to PROCESSING")?;

        Ok(())
    }

    /// Update batch completion status based on claim processing results
    pub async fn update_batch_completion(&self, batch_ids: &[i64]) -> Result<()> {
        if batch_ids.is_empty() {
            return Ok(());
        }

        // Single query to update all batches based on their claim counts
        sqlx::query(
            r#"
            UPDATE staging.import_batch ib
            SET import_status = CASE
                    WHEN counts.failed_count = 0 THEN 'COMPLETED'
                    WHEN counts.successful_count > 0 THEN 'PARTIAL'
                    ELSE 'FAILED'
                END,
                successful_records = counts.successful_count,
                failed_records = counts.failed_count,
                completed_at = CURRENT_TIMESTAMP
            FROM (
                SELECT
                    batch_id,
                    COUNT(*) FILTER (WHERE processing_status = 'COMPLETED') as successful_count,
                    COUNT(*) FILTER (WHERE processing_status = 'FAILED') as failed_count
                FROM staging.raw_claims
                WHERE batch_id = ANY($1)
                GROUP BY batch_id
            ) counts
            WHERE ib.batch_id = counts.batch_id
            "#,
        )
        .bind(batch_ids)
        .execute(&self.pool)
        .await
        .context("Failed to update batch completion status")?;

        Ok(())
    }

    /// Log a processing metric to staging.processing_metrics
    pub async fn log_processing_metric(
        &self,
        batch_id: i64,
        metric_type: &str,
        metric_name: &str,
        started_at: chrono::DateTime<chrono::Utc>,
        completed_at: chrono::DateTime<chrono::Utc>,
        records_processed: i32,
        success_count: i32,
        error_count: i32,
        details: Option<serde_json::Value>,
        processing_stage: &str,
    ) -> Result<()> {
        let duration_ms = (completed_at - started_at).num_milliseconds();
        let duration_sec = duration_ms as f64 / 1000.0;
        let records_per_second = if duration_sec > 0.0 {
            records_processed as f64 / duration_sec
        } else {
            0.0
        };

        sqlx::query(
            r#"
            INSERT INTO staging.processing_metrics (
                metric_id,
                batch_id,
                metric_type,
                metric_name,
                started_at,
                completed_at,
                duration_seconds,
                records_processed,
                records_per_second,
                success_count,
                error_count,
                details,
                processing_stage
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            "#,
        )
        .bind(0i64) // metric_id will be auto-generated or use default
        .bind(batch_id)
        .bind(metric_type)
        .bind(metric_name)
        .bind(started_at)
        .bind(completed_at)
        .bind(duration_sec)
        .bind(records_processed)
        .bind(
            rust_decimal::Decimal::from_f64_retain(records_per_second)
                .unwrap_or(rust_decimal::Decimal::ZERO),
        )
        .bind(success_count)
        .bind(error_count)
        .bind(details)
        .bind(processing_stage)
        .execute(&self.pool)
        .await
        .context("Failed to insert processing metric")?;

        Ok(())
    }

    /// Log metrics for multiple batches
    pub async fn log_batch_metrics(
        &self,
        batch_ids: &HashSet<i64>,
        total_processed: usize,
        successful: usize,
        failed: usize,
        process_start: chrono::DateTime<chrono::Utc>,
        process_end: chrono::DateTime<chrono::Utc>,
    ) -> Result<()> {
        for batch_id in batch_ids {
            // Distribute counts across batches (approximate)
            let batch_claim_count = total_processed / batch_ids.len();
            let batch_success = successful / batch_ids.len();
            let batch_failed = failed / batch_ids.len();

            if let Err(e) = self
                .log_processing_metric(
                    *batch_id,
                    "PROCESS",
                    "Claims Processing",
                    process_start,
                    process_end,
                    batch_claim_count as i32,
                    batch_success as i32,
                    batch_failed as i32,
                    Some(serde_json::json!({
                        "stage": "PROCESS",
                        "batch_size": batch_claim_count,
                        "successful": batch_success,
                        "failed": batch_failed
                    })),
                    "PROCESS",
                )
                .await
            {
                warn!("Failed to log PROCESS metric for batch {}: {}", batch_id, e);
            }
        }

        Ok(())
    }

    /// Get unique batch IDs from a list of claim batch_ids
    pub fn get_unique_batch_ids(&self, batch_ids: Vec<i64>) -> HashSet<i64> {
        batch_ids.into_iter().collect()
    }
}
