//! Parallel Stage 1 Pipeline with FIFO Guarantee
//!
//! Splits Stage 1 ingestion into a parallel parse phase and a single serial commit
//! phase. Multiple parser workers read + parse 837P files concurrently; a single
//! committer applies their results to `staging.raw_claims` in strict `queue_id`
//! order using a reorder buffer.
//!
//! ### FIFO invariant
//!
//! INSERTs into `staging.raw_claims` happen in strictly ascending `queue_id` order.
//! Stage 2's `SequencedBatchAcquirer` then preserves order downstream.
//!
//! ### Concurrency model
//!
//! ```text
//!  Dispatcher (1) ──> parse_tx (bounded mpsc)
//!                       │
//!                       ▼
//!                   ┌───────────────┐
//!                   │ Parser worker │ × N    ──> commit_tx (bounded mpsc)
//!                   └───────────────┘                │
//!                                                    ▼
//!                                            Committer (1)
//!                                              ├─ reorder buffer
//!                                              └─ commit in queue_id order
//! ```
//!
//! Set `STAGE1_PARSE_WORKERS=1` to use the legacy serial loop in `main.rs` instead.

use anyhow::{Context, Result};
use sqlx::PgPool;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

use pro_worker::queue_manager::QueueManager;
use pro_worker::types::FileFormat;

use crate::claims_importer::ClaimsImporter;

/// Configuration for the parallel Stage 1 pipeline.
#[derive(Debug, Clone)]
pub struct Stage1Config {
    /// Number of parallel parser workers. Must be >= 2 to use this pipeline.
    pub parse_workers: usize,
    /// Maximum number of parsed-but-not-yet-committed entries held in the
    /// reorder buffer. Provides back-pressure so memory cannot grow unbounded.
    pub reorder_buffer_max: usize,
    /// How long the dispatcher sleeps when no queued files are available.
    pub idle_backoff: Duration,
}

impl Stage1Config {
    pub fn from_env(parse_workers: usize) -> Self {
        let reorder_buffer_max = std::env::var("STAGE1_REORDER_BUFFER_MAX")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(parse_workers.saturating_mul(4).max(8));

        Self {
            parse_workers,
            reorder_buffer_max,
            idle_backoff: Duration::from_secs(2),
        }
    }
}

/// A unit of parse work dispatched from the dequeue loop to a parser worker.
#[derive(Debug, Clone)]
enum ParseTask {
    Edi {
        queue_id: i64,
        file_path: String,
    },
    /// CSV files pass through the parser pool without parse-parallelism (the legacy
    /// CSV path mixes parse + DB writes and is not on the EDI hot path). The
    /// committer invokes the existing `ingest_file_to_staging` serially when its
    /// turn comes up — this keeps CSVs in FIFO order without blocking EDI parsing.
    Csv {
        queue_id: i64,
        file_path: String,
    },
}

/// Result of a parser worker, delivered to the committer in arbitrary order.
enum ParsedResult {
    EdiOk {
        queue_id: i64,
        file_path: String,
        transaction: Box<pro_parser_edi::Transaction837p>,
        parse_start: chrono::DateTime<chrono::Utc>,
        parse_end: chrono::DateTime<chrono::Utc>,
    },
    EdiFailed {
        queue_id: i64,
        file_path: String,
        error: String,
    },
    /// CSV: committer does both parse and commit when this slot's turn arrives.
    Csv {
        queue_id: i64,
        file_path: String,
    },
}

impl ParsedResult {
    fn queue_id(&self) -> i64 {
        match self {
            ParsedResult::EdiOk { queue_id, .. } => *queue_id,
            ParsedResult::EdiFailed { queue_id, .. } => *queue_id,
            ParsedResult::Csv { queue_id, .. } => *queue_id,
        }
    }
}

/// Spawns the dispatcher, N parser workers, and the committer. Returns join
/// handles so the caller can await shutdown.
pub struct Stage1Pipeline {
    pub config: Stage1Config,
    pub pool: PgPool,
    pub importer: Arc<ClaimsImporter>,
    pub processed_dir: PathBuf,
    pub error_dir: PathBuf,
}

pub struct Stage1Handles {
    pub dispatcher: tokio::task::JoinHandle<()>,
    pub workers: Vec<tokio::task::JoinHandle<()>>,
    pub committer: tokio::task::JoinHandle<()>,
}

impl Stage1Pipeline {
    /// Spawn the full pipeline. The caller owns the join handles and is responsible
    /// for shutdown coordination (e.g., aborting on Ctrl+C).
    pub fn spawn(self) -> Stage1Handles {
        let parse_workers = self.config.parse_workers;
        let buffer_max = self.config.reorder_buffer_max;

        // Channel capacities sized to back-pressure naturally:
        //  - parse channel: 2x workers so parsers stay fed without unbounded queueing
        //  - commit channel: equal to reorder buffer so a full buffer parks parsers
        let (parse_tx, parse_rx) = mpsc::channel::<ParseTask>(parse_workers.saturating_mul(2).max(4));
        let (commit_tx, commit_rx) = mpsc::channel::<ParsedResult>(buffer_max.max(4));

        let parse_rx = Arc::new(tokio::sync::Mutex::new(parse_rx));

        // --- Dispatcher ---
        let dispatcher_pool = self.pool.clone();
        let dispatcher_backoff = self.config.idle_backoff;
        let dispatcher_workers = parse_workers;
        let dispatcher_parse_tx = parse_tx.clone();
        let dispatcher = tokio::spawn(async move {
            run_dispatcher(
                QueueManager::new(dispatcher_pool),
                dispatcher_parse_tx,
                dispatcher_workers,
                dispatcher_backoff,
            )
            .await;
        });
        // Drop our copy of parse_tx so the channel closes after the dispatcher exits.
        drop(parse_tx);

        // --- Parser workers ---
        let mut workers = Vec::with_capacity(parse_workers);
        for worker_id in 0..parse_workers {
            let parse_rx = parse_rx.clone();
            let commit_tx = commit_tx.clone();
            let importer = self.importer.clone();
            workers.push(tokio::spawn(async move {
                run_parser_worker(worker_id, parse_rx, commit_tx, importer).await;
            }));
        }
        drop(commit_tx); // last sender drops when all workers exit

        // --- Committer ---
        let committer_pool = self.pool.clone();
        let committer_importer = self.importer.clone();
        let processed_dir = self.processed_dir.clone();
        let error_dir = self.error_dir.clone();
        let committer = tokio::spawn(async move {
            run_committer(
                committer_pool,
                committer_importer,
                commit_rx,
                processed_dir,
                error_dir,
            )
            .await;
        });

        Stage1Handles {
            dispatcher,
            workers,
            committer,
        }
    }
}

// ============================================================================
// Dispatcher
// ============================================================================

async fn run_dispatcher(
    queue_manager: QueueManager,
    parse_tx: mpsc::Sender<ParseTask>,
    parse_workers: usize,
    idle_backoff: Duration,
) {
    info!(
        "Stage 1 dispatcher starting (parse_workers={}, idle_backoff={:?})",
        parse_workers, idle_backoff
    );

    // Pull a small window of files at a time so we never grossly outrun the
    // workers. The window scales with worker count.
    let window = parse_workers.saturating_mul(2).max(2);
    let mut consecutive_empty: u32 = 0;

    loop {
        let claimed = match queue_manager.dequeue_next_n(window).await {
            Ok(rows) => rows,
            Err(e) => {
                error!("Stage 1 dispatcher: dequeue_next_n failed: {}", e);
                tokio::time::sleep(Duration::from_secs(5)).await;
                continue;
            }
        };

        if claimed.is_empty() {
            consecutive_empty = consecutive_empty.saturating_add(1);
            let backoff = std::cmp::min(
                idle_backoff.saturating_mul(consecutive_empty.min(5)),
                Duration::from_secs(30),
            );
            debug!(
                "Stage 1 dispatcher: queue empty (consecutive_empty={}), sleeping {:?}",
                consecutive_empty, backoff
            );
            tokio::time::sleep(backoff).await;
            continue;
        }

        consecutive_empty = 0;

        for queued_file in claimed {
            // Mark PROCESSING before dispatching so an external observer can
            // distinguish in-flight files from queued ones.
            if let Err(e) = queue_manager.mark_processing(queued_file.queue_id).await {
                error!(
                    "Stage 1 dispatcher: failed to mark queue_id={} as PROCESSING: {}",
                    queued_file.queue_id, e
                );
                continue;
            }

            let task = match queued_file.file_format {
                FileFormat::Edi837p => ParseTask::Edi {
                    queue_id: queued_file.queue_id,
                    file_path: queued_file.file_path,
                },
                FileFormat::Csv => ParseTask::Csv {
                    queue_id: queued_file.queue_id,
                    file_path: queued_file.file_path,
                },
            };

            if let Err(e) = parse_tx.send(task).await {
                error!(
                    "Stage 1 dispatcher: parse channel closed unexpectedly: {}",
                    e
                );
                return;
            }
        }
    }
}

// ============================================================================
// Parser worker
// ============================================================================

async fn run_parser_worker(
    worker_id: usize,
    parse_rx: Arc<tokio::sync::Mutex<mpsc::Receiver<ParseTask>>>,
    commit_tx: mpsc::Sender<ParsedResult>,
    importer: Arc<ClaimsImporter>,
) {
    info!("Stage 1 parser worker {} starting", worker_id);

    loop {
        let task = {
            let mut rx = parse_rx.lock().await;
            match rx.recv().await {
                Some(t) => t,
                None => {
                    info!("Stage 1 parser worker {}: channel closed, exiting", worker_id);
                    return;
                }
            }
        };

        let result = match task {
            ParseTask::Edi { queue_id, file_path } => {
                let path = PathBuf::from(&file_path);
                match importer.parse_edi_file_blocking(path).await {
                    Ok((transaction, parse_start, parse_end)) => ParsedResult::EdiOk {
                        queue_id,
                        file_path,
                        transaction: Box::new(transaction),
                        parse_start,
                        parse_end,
                    },
                    Err(e) => {
                        warn!(
                            "Stage 1 parser worker {}: parse FAILED queue_id={} path={}: {}",
                            worker_id, queue_id, file_path, e
                        );
                        ParsedResult::EdiFailed {
                            queue_id,
                            file_path,
                            error: e.to_string(),
                        }
                    }
                }
            }
            ParseTask::Csv { queue_id, file_path } => {
                // No parsing here — committer handles CSV serially when its turn arrives.
                ParsedResult::Csv { queue_id, file_path }
            }
        };

        if let Err(e) = commit_tx.send(result).await {
            error!(
                "Stage 1 parser worker {}: commit channel closed: {}",
                worker_id, e
            );
            return;
        }
    }
}

// ============================================================================
// Committer (serial, FIFO)
// ============================================================================

async fn run_committer(
    pool: PgPool,
    importer: Arc<ClaimsImporter>,
    mut commit_rx: mpsc::Receiver<ParsedResult>,
    processed_dir: PathBuf,
    error_dir: PathBuf,
) {
    let queue_manager = QueueManager::new(pool.clone());

    let mut next_expected = match resolve_next_expected_queue_id(&pool).await {
        Ok(n) => n,
        Err(e) => {
            error!(
                "Stage 1 committer: failed to resolve initial next_expected queue_id: {}. Defaulting to 1.",
                e
            );
            1
        }
    };

    info!(
        "Stage 1 committer starting (next_expected_queue_id={})",
        next_expected
    );

    let mut buffer: HashMap<i64, ParsedResult> = HashMap::new();

    while let Some(result) = commit_rx.recv().await {
        let qid = result.queue_id();
        if qid < next_expected {
            // Stale result — shouldn't happen unless a recovery pass re-emitted
            // a completed entry. Log loudly and drop.
            warn!(
                "Stage 1 committer: discarding stale result for queue_id={} (next_expected={})",
                qid, next_expected
            );
            continue;
        }
        buffer.insert(qid, result);

        // Drain all consecutive ready entries.
        while let Some(ready) = buffer.remove(&next_expected) {
            commit_one(
                &queue_manager,
                &importer,
                ready,
                &processed_dir,
                &error_dir,
            )
            .await;
            next_expected += 1;
        }
    }

    info!("Stage 1 committer: channel closed, exiting");
}

async fn resolve_next_expected_queue_id(pool: &PgPool) -> Result<i64> {
    // Lowest queue_id still pending or in-flight is the next we expect to commit.
    let pending: Option<i64> = sqlx::query_scalar(
        r#"
        SELECT MIN(queue_id)
        FROM staging.file_processing_queue
        WHERE queue_status IN ('QUEUED', 'PROCESSING', 'RETRY')
        "#,
    )
    .fetch_one(pool)
    .await
    .context("Failed to resolve next_expected queue_id (pending)")?;

    if let Some(qid) = pending {
        return Ok(qid);
    }

    // No pending rows — start one past the highest queue_id we've seen.
    let max_seen: Option<i64> = sqlx::query_scalar(
        "SELECT MAX(queue_id) FROM staging.file_processing_queue",
    )
    .fetch_one(pool)
    .await
    .context("Failed to resolve next_expected queue_id (max_seen)")?;

    Ok(max_seen.unwrap_or(0) + 1)
}

async fn commit_one(
    queue_manager: &QueueManager,
    importer: &ClaimsImporter,
    result: ParsedResult,
    processed_dir: &Path,
    error_dir: &Path,
) {
    match result {
        ParsedResult::EdiOk {
            queue_id,
            file_path,
            transaction,
            parse_start,
            parse_end,
        } => {
            let path = PathBuf::from(&file_path);
            match importer
                .commit_parsed_edi_to_staging(&path, queue_id, *transaction, parse_start, parse_end)
                .await
            {
                Ok(ingest_result) => {
                    info!(
                        "Stage 1 committer: STAGE 1 COMPLETE queue_id={} batch_id={} claims={}",
                        queue_id, ingest_result.batch_id, ingest_result.total_rows
                    );
                    if let Err(e) = queue_manager.mark_completed(queue_id).await {
                        error!(
                            "Stage 1 committer: failed to mark queue_id={} COMPLETED: {}",
                            queue_id, e
                        );
                    }
                    move_file(&path, processed_dir, "processed");
                }
                Err(e) => {
                    error!(
                        "Stage 1 committer: commit FAILED queue_id={} path={}: {}",
                        queue_id, file_path, e
                    );
                    if let Err(mark_err) =
                        queue_manager.mark_failed(queue_id, &e.to_string()).await
                    {
                        error!(
                            "Stage 1 committer: also failed to mark queue_id={} FAILED: {}",
                            queue_id, mark_err
                        );
                    }
                    move_file_with_error(&path, error_dir, &e.to_string());
                }
            }
        }
        ParsedResult::EdiFailed {
            queue_id,
            file_path,
            error,
        } => {
            let path = PathBuf::from(&file_path);
            warn!(
                "Stage 1 committer: parser-reported FAILED queue_id={} path={}: {}",
                queue_id, file_path, error
            );
            if let Err(mark_err) = queue_manager.mark_failed(queue_id, &error).await {
                error!(
                    "Stage 1 committer: failed to mark queue_id={} FAILED: {}",
                    queue_id, mark_err
                );
            }
            move_file_with_error(&path, error_dir, &error);
        }
        ParsedResult::Csv { queue_id, file_path } => {
            let path = PathBuf::from(&file_path);
            // CSV path: do the full ingest serially here. This preserves FIFO
            // because we're already inside the committer's single-task loop.
            match importer
                .ingest_file_to_staging(&path, Some(queue_id))
                .await
            {
                Ok(ingest_result) => {
                    info!(
                        "Stage 1 committer: CSV STAGE 1 COMPLETE queue_id={} batch_id={} rows={}",
                        queue_id, ingest_result.batch_id, ingest_result.total_rows
                    );
                    if let Err(e) = queue_manager.mark_completed(queue_id).await {
                        error!(
                            "Stage 1 committer: failed to mark CSV queue_id={} COMPLETED: {}",
                            queue_id, e
                        );
                    }
                    move_file(&path, processed_dir, "processed");
                }
                Err(e) => {
                    error!(
                        "Stage 1 committer: CSV commit FAILED queue_id={} path={}: {}",
                        queue_id, file_path, e
                    );
                    if let Err(mark_err) =
                        queue_manager.mark_failed(queue_id, &e.to_string()).await
                    {
                        error!(
                            "Stage 1 committer: also failed to mark CSV queue_id={} FAILED: {}",
                            queue_id, mark_err
                        );
                    }
                    move_file_with_error(&path, error_dir, &e.to_string());
                }
            }
        }
    }
}

fn move_file(file_path: &Path, dest_dir: &Path, label: &str) {
    let Some(filename) = file_path.file_name() else {
        warn!("Stage 1 committer: no filename to move for {}", file_path.display());
        return;
    };
    let dest = dest_dir.join(filename);
    if let Err(e) = std::fs::rename(file_path, &dest) {
        // If the file is already gone, this is OK on retry paths; otherwise log loudly.
        if file_path.exists() {
            error!(
                "Stage 1 committer: failed to move {} -> {} ({}): {}",
                file_path.display(),
                dest.display(),
                label,
                e
            );
        } else {
            debug!(
                "Stage 1 committer: source file already moved ({}): {}",
                label,
                file_path.display()
            );
        }
    } else {
        info!(
            "Stage 1 committer: moved {} -> {} ({})",
            file_path.display(),
            dest.display(),
            label
        );
    }
}

fn move_file_with_error(file_path: &Path, error_dir: &Path, error_message: &str) {
    move_file(file_path, error_dir, "error");
    if let Some(filename) = file_path.file_name() {
        let error_file = error_dir.join(filename).with_extension("error");
        if let Err(e) = std::fs::write(&error_file, error_message) {
            warn!(
                "Stage 1 committer: failed to write error sidecar {}: {}",
                error_file.display(),
                e
            );
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Verifies the reorder buffer logic in isolation: out-of-order arrivals
    /// must be committed in queue_id order via the same `next_expected` +
    /// drain-consecutive pattern used in `run_committer`.
    #[test]
    fn reorder_buffer_drains_in_order() {
        let mut buffer: HashMap<i64, i64> = HashMap::new();
        let mut next_expected: i64 = 1;
        let mut committed: Vec<i64> = Vec::new();

        // Simulate arrivals in the order: 3, 1, 4, 2, 5
        let arrivals = [3, 1, 4, 2, 5];
        for qid in arrivals {
            buffer.insert(qid, qid);
            while let Some(ready) = buffer.remove(&next_expected) {
                committed.push(ready);
                next_expected += 1;
            }
        }

        assert_eq!(committed, vec![1, 2, 3, 4, 5]);
        assert!(buffer.is_empty());
        assert_eq!(next_expected, 6);
    }

    /// A failed parse must not block successors: the committer advances
    /// `next_expected` past the failed slot exactly the same way it advances
    /// past a successful one.
    #[test]
    fn reorder_buffer_advances_past_failures() {
        #[derive(Debug, PartialEq, Eq)]
        enum Outcome {
            Ok(i64),
            Failed(i64),
        }

        let mut buffer: HashMap<i64, Outcome> = HashMap::new();
        let mut next_expected: i64 = 1;
        let mut committed: Vec<Outcome> = Vec::new();

        // Arrivals: 2 (fails) first, then 1 (ok), then 3 (ok)
        buffer.insert(2, Outcome::Failed(2));
        while let Some(ready) = buffer.remove(&next_expected) {
            committed.push(ready);
            next_expected += 1;
        }
        assert!(committed.is_empty(), "must not commit out of order");

        buffer.insert(1, Outcome::Ok(1));
        while let Some(ready) = buffer.remove(&next_expected) {
            committed.push(ready);
            next_expected += 1;
        }
        assert_eq!(committed, vec![Outcome::Ok(1), Outcome::Failed(2)]);

        buffer.insert(3, Outcome::Ok(3));
        while let Some(ready) = buffer.remove(&next_expected) {
            committed.push(ready);
            next_expected += 1;
        }
        assert_eq!(
            committed,
            vec![Outcome::Ok(1), Outcome::Failed(2), Outcome::Ok(3)]
        );
    }
}
