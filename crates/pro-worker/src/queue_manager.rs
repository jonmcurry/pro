//! FIFO Queue Manager for File Processing
//!
//! Ensures files are processed in chronological order per facility.
//! This is CRITICAL for healthcare claims processing to maintain proper sequencing.

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use sqlx::PgPool;
use tracing::{info, warn};


use crate::types::FileFormat;

/// Queued file ready for processing
#[derive(Debug, Clone)]
pub struct QueuedFile {
    pub queue_id: i64,
    pub facility_id: i64,
    pub import_batch_id: i64,
    pub file_path: String,
    pub file_hash: String,
    pub file_format: FileFormat,
    pub organization_id: i64,
    pub queued_at: DateTime<Utc>,
    pub priority: i32,
}

/// Queue status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueueStatus {
    Queued,
    Processing,
    Completed,
    Failed,
    Retry,
}

impl QueueStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            QueueStatus::Queued => "QUEUED",
            QueueStatus::Processing => "PROCESSING",
            QueueStatus::Completed => "COMPLETED",
            QueueStatus::Failed => "FAILED",
            QueueStatus::Retry => "RETRY",
        }
    }
}

/// Manages FIFO queue for file processing
#[derive(Clone)]
pub struct QueueManager {
    pool: PgPool,
}

impl QueueManager {
    /// Create a new queue manager
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Enqueue a file for processing
    ///
    /// Files are queued in chronological order per facility.
    /// Lower priority numbers are processed first.
    pub async fn enqueue_file(
        &self,
        facility_id: i64,
        import_batch_id: i64,
        file_path: String,
        file_hash: String,
        file_format: FileFormat,
        organization_id: i64,
        priority: Option<i32>,
    ) -> Result<i64> {
        let priority = priority.unwrap_or(100); // Default priority

        // Convert FileFormat to database string format
        let file_format_str = match file_format {
            FileFormat::Edi837p => "EDI837p",
            FileFormat::Csv => "CSV",
        };

        let queue_id: i64 = sqlx::query_scalar(
            r#"
            INSERT INTO staging.file_processing_queue (
                facility_id,
                import_batch_id,
                file_path,
                file_hash,
                file_format,
                organization_id,
                priority,
                queue_status,
                queued_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, 'QUEUED', CURRENT_TIMESTAMP)
            RETURNING queue_id
            "#,
        )
        .bind(facility_id)
        .bind(import_batch_id)
        .bind(&file_path)
        .bind(&file_hash)
        .bind(file_format_str)
        .bind(organization_id)
        .bind(priority)
        .fetch_one(&self.pool)
        .await
        .context("Failed to enqueue file")?;

        info!(
            "Enqueued file {} for facility {} (queue_id: {}, priority: {})",
            file_path, facility_id, queue_id, priority
        );

        Ok(queue_id)
    }

    /// Get next file to process for a specific facility (FIFO)
    ///
    /// Returns the oldest queued file for the facility, respecting priority.
    pub async fn dequeue_next_for_facility(
        &self,
        facility_id: i64,
    ) -> Result<Option<QueuedFile>> {
        let result = sqlx::query_as::<_, QueuedFileRow>(
            r#"
            SELECT
                queue_id,
                facility_id,
                import_batch_id,
                file_path,
                file_hash,
                file_format,
                organization_id,
                queued_at,
                priority
            FROM staging.file_processing_queue
            WHERE facility_id = $1
              AND queue_status = 'QUEUED'
            ORDER BY priority ASC, queued_at ASC
            LIMIT 1
            FOR UPDATE SKIP LOCKED
            "#,
        )
        .bind(facility_id)
        .fetch_optional(&self.pool)
        .await
        .context("Failed to dequeue file for facility")?;

        Ok(result.map(|row| row.into()))
    }

    /// Get next file to process across all facilities (global FIFO)
    ///
    /// Returns the oldest queued file across all facilities, respecting priority.
    /// Skips files that are locked by other workers.
    pub async fn dequeue_next_global(&self) -> Result<Option<QueuedFile>> {
        let result = sqlx::query_as::<_, QueuedFileRow>(
            r#"
            SELECT
                queue_id,
                facility_id,
                import_batch_id,
                file_path,
                file_hash,
                file_format,
                organization_id,
                queued_at,
                priority
            FROM staging.file_processing_queue
            WHERE queue_status = 'QUEUED'
            ORDER BY priority ASC, queued_at ASC
            LIMIT 1
            FOR UPDATE SKIP LOCKED
            "#,
        )
        .fetch_optional(&self.pool)
        .await
        .context("Failed to dequeue file globally")?;

        Ok(result.map(|row| row.into()))
    }

    /// Mark a file as processing
    pub async fn mark_processing(&self, queue_id: i64) -> Result<()> {
        sqlx::query(
            r#"
            UPDATE staging.file_processing_queue
            SET queue_status = 'PROCESSING',
                processing_started_at = CURRENT_TIMESTAMP,
                updated_by = 'WORKER'
            WHERE queue_id = $1
            "#,
        )
        .bind(queue_id)
        .execute(&self.pool)
        .await
        .context("Failed to mark file as processing")?;

        info!("Marked queue entry {} as processing", queue_id);

        Ok(())
    }

    /// Mark a file as completed
    pub async fn mark_completed(&self, queue_id: i64) -> Result<()> {
        sqlx::query(
            r#"
            UPDATE staging.file_processing_queue
            SET queue_status = 'COMPLETED',
                processing_completed_at = CURRENT_TIMESTAMP,
                updated_by = 'WORKER'
            WHERE queue_id = $1
            "#,
        )
        .bind(queue_id)
        .execute(&self.pool)
        .await
        .context("Failed to mark file as completed")?;

        info!("Marked queue entry {} as completed", queue_id);

        Ok(())
    }

    /// Mark a file as failed
    pub async fn mark_failed(&self, queue_id: i64, error: &str) -> Result<()> {
        sqlx::query(
            r#"
            UPDATE staging.file_processing_queue
            SET queue_status = 'FAILED',
                processing_completed_at = CURRENT_TIMESTAMP,
                last_error = $2,
                updated_by = 'WORKER'
            WHERE queue_id = $1
            "#,
        )
        .bind(queue_id)
        .bind(error)
        .execute(&self.pool)
        .await
        .context("Failed to mark file as failed")?;

        warn!("Marked queue entry {} as failed: {}", queue_id, error);

        Ok(())
    }

    /// Requeue a file for retry after failure
    pub async fn requeue_for_retry(&self, queue_id: i64) -> Result<bool> {
        let rows_affected = sqlx::query(
            r#"
            UPDATE staging.file_processing_queue
            SET queue_status = 'RETRY',
                retry_count = retry_count + 1,
                queued_at = CURRENT_TIMESTAMP,
                processing_started_at = NULL,
                processing_completed_at = NULL,
                updated_by = 'WORKER'
            WHERE queue_id = $1
              AND retry_count < max_retries
            "#,
        )
        .bind(queue_id)
        .execute(&self.pool)
        .await
        .context("Failed to requeue for retry")?
        .rows_affected();

        if rows_affected > 0 {
            info!("Requeued file {} for retry", queue_id);
            Ok(true)
        } else {
            warn!("File {} exceeded max retries, not requeued", queue_id);
            Ok(false)
        }
    }

    /// Get queue depth for a facility
    pub async fn get_queue_depth_by_facility(&self, facility_id: i64) -> Result<usize> {
        let count: i64 = sqlx::query_scalar(
            r#"
            SELECT COUNT(*)
            FROM staging.file_processing_queue
            WHERE facility_id = $1
              AND queue_status IN ('QUEUED', 'RETRY')
            "#,
        )
        .bind(facility_id)
        .fetch_one(&self.pool)
        .await
        .context("Failed to get queue depth")?;

        Ok(count as usize)
    }

    /// Get total queue depth across all facilities
    pub async fn get_total_queue_depth(&self) -> Result<usize> {
        let count: i64 = sqlx::query_scalar(
            r#"
            SELECT COUNT(*)
            FROM staging.file_processing_queue
            WHERE queue_status IN ('QUEUED', 'RETRY')
            "#,
        )
        .fetch_one(&self.pool)
        .await
        .context("Failed to get total queue depth")?;

        Ok(count as usize)
    }

    /// Get processing count (currently being processed)
    pub async fn get_processing_count(&self) -> Result<usize> {
        let count: i64 = sqlx::query_scalar(
            r#"
            SELECT COUNT(*)
            FROM staging.file_processing_queue
            WHERE queue_status = 'PROCESSING'
            "#,
        )
        .fetch_one(&self.pool)
        .await
        .context("Failed to get processing count")?;

        Ok(count as usize)
    }

    /// Clean up old completed/failed queue entries
    pub async fn cleanup_old_entries(&self, retention_days: i32) -> Result<usize> {
        let deleted: i32 = sqlx::query_scalar(
            "SELECT staging.cleanup_old_queue_entries($1)"
        )
        .bind(retention_days)
        .fetch_one(&self.pool)
        .await
        .context("Failed to cleanup old queue entries")?;

        if deleted > 0 {
            info!("Cleaned up {} old queue entries", deleted);
        }

        Ok(deleted as usize)
    }
}

/// Database row structure for queued files
#[derive(Debug, sqlx::FromRow)]
struct QueuedFileRow {
    queue_id: i64,
    facility_id: i64,
    import_batch_id: i64,
    file_path: String,
    file_hash: String,
    file_format: String,
    organization_id: i64,
    queued_at: DateTime<Utc>,
    priority: i32,
}

impl From<QueuedFileRow> for QueuedFile {
    fn from(row: QueuedFileRow) -> Self {
        Self {
            queue_id: row.queue_id,
            facility_id: row.facility_id,
            import_batch_id: row.import_batch_id,
            file_path: row.file_path,
            file_hash: row.file_hash,
            file_format: FileFormat::from_str(&row.file_format),
            organization_id: row.organization_id,
            queued_at: row.queued_at,
            priority: row.priority,
        }
    }
}

impl FileFormat {
    fn from_str(s: &str) -> Self {
        match s {
            "EDI837p" => FileFormat::Edi837p,
            "CSV" => FileFormat::Csv,
            _ => FileFormat::Edi837p, // Default
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_queue_manager_creation() {
        let pool = PgPool::connect_lazy("postgres://dummy").unwrap();
        let _manager = QueueManager::new(pool);
    }
}
