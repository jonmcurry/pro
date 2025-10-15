//! Real-time Progress Tracking for Streaming Processing
//!
//! PHASE 5: Track and broadcast file processing progress for real-time updates

use chrono::{DateTime, Utc};
use pro_common::{Error, Result};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::broadcast;
use uuid::Uuid;

/// Progress event types for broadcasting
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ProgressEvent {
    Started {
        queue_id: Uuid,
        progress_id: Uuid,
        total_claims: usize,
        started_at: DateTime<Utc>,
    },
    Progress {
        queue_id: Uuid,
        progress_id: Uuid,
        total_claims: usize,
        processed_claims: usize,
        failed_claims: usize,
        flags_created: usize,
        percent_complete: f64,
        claims_per_second: f64,
        estimated_completion_seconds: Option<i64>,
    },
    ClaimProcessed {
        queue_id: Uuid,
        progress_id: Uuid,
        claim_number: Option<String>,
        processing_time_ms: u64,
    },
    ClaimFailed {
        queue_id: Uuid,
        progress_id: Uuid,
        claim_number: Option<String>,
        error: String,
    },
    Completed {
        queue_id: Uuid,
        progress_id: Uuid,
        total_claims: usize,
        processed_claims: usize,
        failed_claims: usize,
        flags_created: usize,
        duration_seconds: i64,
    },
    Failed {
        queue_id: Uuid,
        progress_id: Uuid,
        error: String,
    },
}

/// Progress statistics snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressSnapshot {
    pub progress_id: Uuid,
    pub queue_id: Uuid,
    pub total_claims: usize,
    pub processed_claims: usize,
    pub failed_claims: usize,
    pub flags_created: usize,
    pub critical_flags: usize,
    pub percent_complete: f64,
    pub claims_per_second: Option<f64>,
    pub average_processing_time_ms: Option<u64>,
    pub estimated_completion_at: Option<DateTime<Utc>>,
    pub started_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub is_active: bool,
}

/// Real-time progress tracker
#[derive(Clone)]
pub struct ProgressTracker {
    progress_id: Uuid,
    queue_id: Uuid,
    pool: PgPool,
    broadcaster: broadcast::Sender<ProgressEvent>,

    // Atomic counters for lock-free updates
    total_claims: Arc<AtomicU64>,
    processed_claims: Arc<AtomicU64>,
    failed_claims: Arc<AtomicU64>,
    flags_created: Arc<AtomicU64>,
    critical_flags: Arc<AtomicU64>,

    // Timing
    started_at: DateTime<Utc>,
}

impl ProgressTracker {
    /// Create a new progress tracker
    pub async fn new(
        queue_id: Uuid,
        total_claims: usize,
        pool: PgPool,
        broadcaster: broadcast::Sender<ProgressEvent>,
    ) -> Result<Self> {
        let progress_id = Uuid::new_v4();
        let started_at = Utc::now();

        // Insert initial progress record
        sqlx::query(
            r#"
            INSERT INTO staging.file_processing_progress (
                id, queue_id, total_claims, started_at, is_active
            ) VALUES ($1, $2, $3, $4, true)
            "#
        )
        .bind(progress_id)
        .bind(queue_id)
        .bind(total_claims as i32)
        .bind(started_at)
        .execute(&pool)
        .await
        .map_err(Error::Database)?;

        let tracker = Self {
            progress_id,
            queue_id,
            pool,
            broadcaster,
            total_claims: Arc::new(AtomicU64::new(total_claims as u64)),
            processed_claims: Arc::new(AtomicU64::new(0)),
            failed_claims: Arc::new(AtomicU64::new(0)),
            flags_created: Arc::new(AtomicU64::new(0)),
            critical_flags: Arc::new(AtomicU64::new(0)),
            started_at,
        };

        // Broadcast started event
        let _ = tracker.broadcaster.send(ProgressEvent::Started {
            queue_id,
            progress_id,
            total_claims,
            started_at,
        });

        Ok(tracker)
    }

    /// Record a successful claim processing
    pub async fn record_claim_processed(
        &self,
        claim_number: Option<String>,
        flags_count: usize,
        critical_flags_count: usize,
        processing_time_ms: u64,
    ) -> Result<()> {
        self.processed_claims.fetch_add(1, Ordering::Relaxed);
        self.flags_created.fetch_add(flags_count as u64, Ordering::Relaxed);
        self.critical_flags
            .fetch_add(critical_flags_count as u64, Ordering::Relaxed);

        // Broadcast claim processed event
        let _ = self.broadcaster.send(ProgressEvent::ClaimProcessed {
            queue_id: self.queue_id,
            progress_id: self.progress_id,
            claim_number,
            processing_time_ms,
        });

        // Update progress periodically (every 10 claims or if last claim)
        let processed = self.processed_claims.load(Ordering::Relaxed);
        if processed % 10 == 0 || processed == self.total_claims.load(Ordering::Relaxed) {
            self.update_progress().await?;
        }

        Ok(())
    }

    /// Record a failed claim processing
    pub async fn record_claim_failed(
        &self,
        claim_number: Option<String>,
        error: String,
        claim_data: Option<serde_json::Value>,
        error_type: Option<String>,
        can_retry: bool,
    ) -> Result<()> {
        self.failed_claims.fetch_add(1, Ordering::Relaxed);

        // Store failed claim in database
        sqlx::query(
            r#"
            INSERT INTO staging.failed_claims (
                queue_id, progress_id, claim_number, error_message,
                error_type, claim_data, can_retry
            ) VALUES ($1, $2, $3, $4, $5, $6, $7)
            "#
        )
        .bind(self.queue_id)
        .bind(self.progress_id)
        .bind(claim_number.as_deref())
        .bind(error.clone())
        .bind(error_type.as_deref())
        .bind(claim_data)
        .bind(can_retry)
        .execute(&self.pool)
        .await
        .map_err(Error::Database)?;

        // Broadcast claim failed event
        let _ = self.broadcaster.send(ProgressEvent::ClaimFailed {
            queue_id: self.queue_id,
            progress_id: self.progress_id,
            claim_number,
            error,
        });

        // Update progress
        self.update_progress().await?;

        Ok(())
    }

    /// Update progress in database and broadcast
    async fn update_progress(&self) -> Result<()> {
        let total = self.total_claims.load(Ordering::Relaxed);
        let processed = self.processed_claims.load(Ordering::Relaxed);
        let failed = self.failed_claims.load(Ordering::Relaxed);
        let flags = self.flags_created.load(Ordering::Relaxed);
        let critical = self.critical_flags.load(Ordering::Relaxed);

        let elapsed = Utc::now() - self.started_at;
        let elapsed_seconds = elapsed.num_seconds().max(1) as f64;
        let claims_per_second = processed as f64 / elapsed_seconds;

        // Calculate estimated completion
        let remaining = total.saturating_sub(processed);
        let estimated_completion_seconds = if claims_per_second > 0.0 && remaining > 0 {
            Some((remaining as f64 / claims_per_second) as i64)
        } else {
            None
        };

        let estimated_completion_at = estimated_completion_seconds
            .map(|secs| Utc::now() + chrono::Duration::seconds(secs));

        // Update database
        sqlx::query(
            r#"
            UPDATE staging.file_processing_progress
            SET processed_claims = $1,
                failed_claims = $2,
                flags_created = $3,
                critical_flags = $4,
                claims_per_second = $5,
                estimated_completion_at = $6,
                updated_at = now()
            WHERE id = $7
            "#
        )
        .bind(processed as i32)
        .bind(failed as i32)
        .bind(flags as i32)
        .bind(critical as i32)
        .bind(rust_decimal::Decimal::try_from(claims_per_second).ok())
        .bind(estimated_completion_at)
        .bind(self.progress_id)
        .execute(&self.pool)
        .await
        .map_err(Error::Database)?;

        // Broadcast progress event
        let percent_complete = if total > 0 {
            (processed as f64 / total as f64) * 100.0
        } else {
            0.0
        };

        let _ = self.broadcaster.send(ProgressEvent::Progress {
            queue_id: self.queue_id,
            progress_id: self.progress_id,
            total_claims: total as usize,
            processed_claims: processed as usize,
            failed_claims: failed as usize,
            flags_created: flags as usize,
            percent_complete,
            claims_per_second,
            estimated_completion_seconds,
        });

        Ok(())
    }

    /// Mark processing as completed
    pub async fn complete(&self) -> Result<()> {
        let total = self.total_claims.load(Ordering::Relaxed);
        let processed = self.processed_claims.load(Ordering::Relaxed);
        let failed = self.failed_claims.load(Ordering::Relaxed);
        let flags = self.flags_created.load(Ordering::Relaxed);

        let duration = Utc::now() - self.started_at;
        let duration_seconds = duration.num_seconds();

        // Update database
        sqlx::query(
            r#"
            UPDATE staging.file_processing_progress
            SET completed_at = now(),
                is_active = false,
                updated_at = now()
            WHERE id = $1
            "#
        )
        .bind(self.progress_id)
        .execute(&self.pool)
        .await
        .map_err(Error::Database)?;

        // Broadcast completed event
        let _ = self.broadcaster.send(ProgressEvent::Completed {
            queue_id: self.queue_id,
            progress_id: self.progress_id,
            total_claims: total as usize,
            processed_claims: processed as usize,
            failed_claims: failed as usize,
            flags_created: flags as usize,
            duration_seconds,
        });

        Ok(())
    }

    /// Mark processing as failed
    pub async fn fail(&self, error: String) -> Result<()> {
        sqlx::query(
            r#"
            UPDATE staging.file_processing_progress
            SET is_active = false,
                updated_at = now()
            WHERE id = $1
            "#
        )
        .bind(self.progress_id)
        .execute(&self.pool)
        .await
        .map_err(Error::Database)?;

        // Broadcast failed event
        let _ = self.broadcaster.send(ProgressEvent::Failed {
            queue_id: self.queue_id,
            progress_id: self.progress_id,
            error,
        });

        Ok(())
    }

    /// Get current progress snapshot
    pub async fn snapshot(&self) -> Result<ProgressSnapshot> {
        let row: (Uuid, Uuid, i32, i32, i32, i32, i32, Option<rust_decimal::Decimal>, DateTime<Utc>, DateTime<Utc>, bool) = sqlx::query_as(
            r#"
            SELECT
                id, queue_id, total_claims, processed_claims, failed_claims,
                flags_created, critical_flags, claims_per_second,
                started_at, updated_at, is_active
            FROM staging.file_processing_progress
            WHERE id = $1
            "#
        )
        .bind(self.progress_id)
        .fetch_one(&self.pool)
        .await
        .map_err(Error::Database)?;

        let total = row.2 as usize;
        let processed = row.3 as usize;
        let percent_complete = if total > 0 {
            (processed as f64 / total as f64) * 100.0
        } else {
            0.0
        };

        Ok(ProgressSnapshot {
            progress_id: row.0,
            queue_id: row.1,
            total_claims: total,
            processed_claims: processed,
            failed_claims: row.4 as usize,
            flags_created: row.5 as usize,
            critical_flags: row.6 as usize,
            percent_complete,
            claims_per_second: row.7.map(|d| d.to_string().parse().unwrap_or(0.0)),
            average_processing_time_ms: None, // TODO: Calculate from timing data
            estimated_completion_at: None,    // TODO: Calculate based on rate
            started_at: row.8,
            updated_at: row.9,
            is_active: row.10,
        })
    }

    /// Get progress ID
    pub fn progress_id(&self) -> Uuid {
        self.progress_id
    }

    /// Get queue ID
    pub fn queue_id(&self) -> Uuid {
        self.queue_id
    }

    /// Subscribe to progress events
    pub fn subscribe(&self) -> broadcast::Receiver<ProgressEvent> {
        self.broadcaster.subscribe()
    }
}

/// Query progress by queue ID
pub async fn get_progress_by_queue_id(pool: &PgPool, queue_id: Uuid) -> Result<Option<ProgressSnapshot>> {
    let row: Option<(Uuid, Uuid, i32, i32, i32, i32, i32, Option<rust_decimal::Decimal>, DateTime<Utc>, DateTime<Utc>, bool)> = sqlx::query_as(
        r#"
        SELECT
            id, queue_id, total_claims, processed_claims, failed_claims,
            flags_created, critical_flags, claims_per_second,
            started_at, updated_at, is_active
        FROM staging.file_processing_progress
        WHERE queue_id = $1
        ORDER BY started_at DESC
        LIMIT 1
        "#
    )
    .bind(queue_id)
    .fetch_optional(pool)
    .await
    .map_err(Error::Database)?;

    Ok(row.map(|row| {
        let total = row.2 as usize;
        let processed = row.3 as usize;
        let percent_complete = if total > 0 {
            (processed as f64 / total as f64) * 100.0
        } else {
            0.0
        };

        ProgressSnapshot {
            progress_id: row.0,
            queue_id: row.1,
            total_claims: total,
            processed_claims: processed,
            failed_claims: row.4 as usize,
            flags_created: row.5 as usize,
            critical_flags: row.6 as usize,
            percent_complete,
            claims_per_second: row.7.map(|d| d.to_string().parse().unwrap_or(0.0)),
            average_processing_time_ms: None,
            estimated_completion_at: None,
            started_at: row.8,
            updated_at: row.9,
            is_active: row.10,
        }
    }))
}
