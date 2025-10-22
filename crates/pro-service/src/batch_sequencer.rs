//! Batch Sequencing for Strict FIFO Ordering
//!
//! This module implements Aegis-style sequential completion to maintain strict FIFO
//! ordering with multi-worker parallel processing.
//!
//! Architecture:
//! - SequencedBatchAcquirer: Single-threaded batch acquisition with sequence assignment
//! - SequentialCompletionManager: Buffers results and commits in sequence order
//! - Workers: Process batches in parallel, report to completion manager
//!
//! Flow:
//! 1. Acquirer assigns monotonic sequence numbers (1, 2, 3...)
//! 2. Workers process batches concurrently (may finish out of order)
//! 3. Completion manager buffers results and commits only in sequence order

use anyhow::{Context, Result};
use sqlx::PgPool;
use std::collections::HashMap;
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// Batch information with sequence number
#[derive(Debug, Clone)]
pub struct SequencedBatch {
    pub sequence_number: i32,
    pub batch_id: Uuid,
    pub claim_ids: Vec<Uuid>,
    pub assigned_at: chrono::DateTime<chrono::Utc>,
}

/// Result from processing a sequenced batch
#[derive(Debug, Clone)]
pub struct BatchResult {
    pub sequence_number: i32,
    pub batch_id: Uuid,
    pub success_count: usize,
    pub failure_count: usize,
    pub processing_time_seconds: f64,
    pub worker_id: String,
    pub errors: Vec<String>,
}

/// Atomic sequence counter for thread-safe sequence number generation
pub struct SequenceCounter {
    counter: AtomicI32,
}

impl SequenceCounter {
    pub fn new(start_value: i32) -> Self {
        Self {
            counter: AtomicI32::new(start_value),
        }
    }

    /// Get next sequence number atomically
    pub fn next(&self) -> i32 {
        self.counter.fetch_add(1, Ordering::SeqCst)
    }

    /// Get current value without incrementing
    pub fn current(&self) -> i32 {
        self.counter.load(Ordering::SeqCst)
    }
}

/// Sequenced Batch Acquirer - Single-threaded batch acquisition with sequence assignment
///
/// This component is responsible for:
/// - Fetching batches of PENDING raw_claims in FIFO order
/// - Assigning monotonic sequence numbers
/// - Creating batch_sequences tracking records
/// - Sending batches to worker pool via channel
pub struct SequencedBatchAcquirer {
    pool: PgPool,
    batch_size: usize,
    sequence_counter: Arc<SequenceCounter>,
}

impl SequencedBatchAcquirer {
    pub fn new(pool: PgPool, batch_size: usize) -> Self {
        Self {
            pool,
            batch_size,
            sequence_counter: Arc::new(SequenceCounter::new(1)),
        }
    }

    /// Start the batch acquisition loop
    /// Sends acquired batches to the provided channel
    pub async fn start(
        &self,
        batch_tx: mpsc::Sender<SequencedBatch>,
        mut shutdown_rx: mpsc::Receiver<()>,
    ) -> Result<()> {
        info!("Starting SequencedBatchAcquirer (batch_size: {})", self.batch_size);

        loop {
            tokio::select! {
                _ = shutdown_rx.recv() => {
                    info!("SequencedBatchAcquirer shutting down");
                    break;
                }
                _ = tokio::time::sleep(tokio::time::Duration::from_secs(1)) => {
                    // Try to acquire next batch
                    match self.acquire_next_batch().await {
                        Ok(Some(batch)) => {
                            info!(
                                "Acquired batch sequence {} ({} claims)",
                                batch.sequence_number, batch.claim_ids.len()
                            );

                            // Send to worker pool (only ONE worker will receive it)
                            if let Err(e) = batch_tx.send(batch).await {
                                error!("Failed to send batch to workers: {}", e);
                            }
                        }
                        Ok(None) => {
                            // No pending claims, wait briefly
                            debug!("No pending claims available");
                        }
                        Err(e) => {
                            error!("Failed to acquire batch: {}", e);
                            tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Acquire next batch of PENDING claims and assign sequence number
    async fn acquire_next_batch(&self) -> Result<Option<SequencedBatch>> {
        let mut tx = self.pool.begin().await
            .context("Failed to begin transaction")?;

        // Get next batch of PENDING claims in FIFO order (oldest first)
        // Use FOR UPDATE to lock the claims (prevents other acquirers from getting same batch)
        let claims: Vec<(Uuid, Uuid)> = sqlx::query_as(
            r#"
            SELECT raw_claim_id, batch_id
            FROM staging.raw_claims
            WHERE processing_status = 'PENDING'
            AND batch_sequence_number IS NULL
            ORDER BY ingested_at ASC
            LIMIT $1
            FOR UPDATE
            "#
        )
        .bind(self.batch_size as i64)
        .fetch_all(&mut *tx)
        .await
        .context("Failed to fetch pending claims")?;

        if claims.is_empty() {
            tx.rollback().await?;
            return Ok(None);
        }

        // Get batch_id (all claims in this acquisition should have same batch_id from Stage 1)
        let batch_id = claims[0].1;
        let claim_ids: Vec<Uuid> = claims.iter().map(|(id, _)| *id).collect();

        // Assign sequence number
        let sequence_number = self.sequence_counter.next();
        let assigned_at = chrono::Utc::now();

        // Update claims with sequence number and mark as PROCESSING
        sqlx::query(
            r#"
            UPDATE staging.raw_claims
            SET batch_sequence_number = $1,
                processing_status = 'PROCESSING'
            WHERE raw_claim_id = ANY($2)
            "#
        )
        .bind(sequence_number)
        .bind(&claim_ids)
        .execute(&mut *tx)
        .await
        .context("Failed to assign sequence number to claims")?;

        // Create batch_sequences tracking record
        sqlx::query(
            r#"
            INSERT INTO staging.batch_sequences (
                sequence_number,
                batch_id,
                claim_count,
                assigned_at,
                processing_stage
            )
            VALUES ($1, $2, $3, $4, $5)
            "#
        )
        .bind(sequence_number)
        .bind(batch_id)
        .bind(claim_ids.len() as i32)
        .bind(assigned_at)
        .bind("STAGE2")
        .execute(&mut *tx)
        .await
        .context("Failed to create batch_sequences record")?;

        // Commit transaction
        tx.commit().await
            .context("Failed to commit batch acquisition")?;

        Ok(Some(SequencedBatch {
            sequence_number,
            batch_id,
            claim_ids,
            assigned_at,
        }))
    }
}

/// Sequential Completion Manager - Buffers results and commits in FIFO order
///
/// This component is responsible for:
/// - Receiving batch results from workers (via channel)
/// - Buffering out-of-order results in memory
/// - Committing batches to production only when sequence number matches expected
/// - Updating batch_sequences records with completion metadata
pub struct SequentialCompletionManager {
    pool: PgPool,
    pending_completions: Arc<Mutex<HashMap<i32, BatchResult>>>,
    next_expected_sequence: Arc<Mutex<i32>>,
}

impl SequentialCompletionManager {
    pub fn new(pool: PgPool) -> Self {
        Self {
            pool,
            pending_completions: Arc::new(Mutex::new(HashMap::new())),
            next_expected_sequence: Arc::new(Mutex::new(1)),
        }
    }

    /// Start the completion manager loop
    /// Receives batch results from workers and commits in sequence order
    pub async fn start(
        &self,
        mut result_rx: mpsc::Receiver<BatchResult>,
        mut shutdown_rx: mpsc::Receiver<()>,
    ) -> Result<()> {
        info!("Starting SequentialCompletionManager");

        loop {
            tokio::select! {
                _ = shutdown_rx.recv() => {
                    info!("SequentialCompletionManager shutting down");
                    break;
                }
                Some(batch_result) = result_rx.recv() => {
                    if let Err(e) = self.handle_batch_result(batch_result).await {
                        error!("Failed to handle batch result: {}", e);
                    }
                }
                _ = tokio::time::sleep(tokio::time::Duration::from_secs(30)) => {
                    // Periodically check for stuck sequences
                    if let Err(e) = self.check_stuck_sequences().await {
                        error!("Failed to check stuck sequences: {}", e);
                    }
                }
            }
        }

        Ok(())
    }

    /// Handle a completed batch result
    async fn handle_batch_result(&self, batch_result: BatchResult) -> Result<()> {
        let sequence_num = batch_result.sequence_number;

        // Add to pending completions
        {
            let mut pending = self.pending_completions.lock().await;
            pending.insert(sequence_num, batch_result.clone());
        }

        // Get current next expected sequence
        let next_expected = *self.next_expected_sequence.lock().await;

        // Check if this is out of order
        if sequence_num != next_expected {
            debug!(
                "Batch {} arrived out of order (expecting {}), buffering",
                sequence_num, next_expected
            );
        }

        // Try to process all consecutive ready sequences
        self.process_ready_sequences().await?;

        Ok(())
    }

    /// Process all consecutive sequences that are ready for completion
    async fn process_ready_sequences(&self) -> Result<()> {
        loop {
            // Get next expected sequence
            let next_expected = *self.next_expected_sequence.lock().await;

            // Check if we have this sequence in pending completions
            let batch_result = {
                let mut pending = self.pending_completions.lock().await;
                pending.remove(&next_expected)
            };

            match batch_result {
                Some(result) => {
                    // Commit this batch to production
                    info!(
                        "Committing batch {} in FIFO order ({} successful, {} failed)",
                        result.sequence_number, result.success_count, result.failure_count
                    );

                    if let Err(e) = self.commit_batch_to_production(&result).await {
                        error!("Failed to commit batch {}: {}", result.sequence_number, e);

                        // Put it back in pending and stop processing to maintain order
                        let mut pending = self.pending_completions.lock().await;
                        pending.insert(next_expected, result);
                        break;
                    }

                    // Advance to next sequence
                    let mut next_expected_lock = self.next_expected_sequence.lock().await;
                    *next_expected_lock += 1;
                }
                None => {
                    // No more consecutive sequences ready
                    break;
                }
            }
        }

        Ok(())
    }

    /// Commit a batch to production (final step in FIFO order)
    async fn commit_batch_to_production(&self, batch_result: &BatchResult) -> Result<()> {
        let completed_at = chrono::Utc::now();

        // Update batch_sequences record with completion metadata
        sqlx::query(
            r#"
            UPDATE staging.batch_sequences
            SET completed_at = $1,
                processing_time_seconds = $2,
                worker_id = $3,
                success_count = $4,
                failure_count = $5,
                errors = $6
            WHERE sequence_number = $7
            "#
        )
        .bind(completed_at)
        .bind(batch_result.processing_time_seconds as f32)
        .bind(&batch_result.worker_id)
        .bind(batch_result.success_count as i32)
        .bind(batch_result.failure_count as i32)
        .bind(serde_json::to_value(&batch_result.errors).ok())
        .bind(batch_result.sequence_number)
        .execute(&self.pool)
        .await
        .context("Failed to update batch_sequences record")?;

        // Update batch status if all claims completed
        sqlx::query(
            r#"
            UPDATE staging.import_batch
            SET import_status = CASE
                WHEN (SELECT COUNT(*) FROM staging.raw_claims WHERE batch_id = $1 AND processing_status = 'FAILED') > 0
                    THEN 'PARTIAL'
                ELSE 'COMPLETED'
            END,
            completed_at = $2
            WHERE batch_id = $1
            AND import_status NOT IN ('COMPLETED', 'PARTIAL', 'FAILED')
            "#
        )
        .bind(batch_result.batch_id)
        .bind(completed_at)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Check for stuck sequences (waiting too long)
    async fn check_stuck_sequences(&self) -> Result<()> {
        // Query database for stuck sequences
        let stuck: Vec<(i32, chrono::DateTime<chrono::Utc>, i32)> = sqlx::query_as(
            r#"
            SELECT sequence_number, assigned_at, claim_count
            FROM staging.batch_sequences
            WHERE completed_at IS NULL
            AND assigned_at < CURRENT_TIMESTAMP - INTERVAL '5 minutes'
            ORDER BY sequence_number ASC
            LIMIT 10
            "#
        )
        .fetch_all(&self.pool)
        .await?;

        for (seq_num, assigned_at, claim_count) in stuck {
            let wait_time = (chrono::Utc::now() - assigned_at).num_seconds();
            warn!(
                "Stuck sequence detected: #{} ({} claims, waiting {} seconds)",
                seq_num, claim_count, wait_time
            );
        }

        Ok(())
    }

    /// Get statistics about pending completions
    pub async fn get_stats(&self) -> (usize, i32) {
        let pending = self.pending_completions.lock().await;
        let next_expected = *self.next_expected_sequence.lock().await;
        (pending.len(), next_expected)
    }
}
