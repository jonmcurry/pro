# Plan: Parallel Stage 1 Ingestion with FIFO Guarantee

## Problem

Stage 1 (file dequeue + parse + insert into `staging.raw_claims`) is currently a single
serial loop in [crates/pro-service/src/main.rs:389-498](../crates/pro-service/src/main.rs#L389-L498).
For a 6,000-file production run, this becomes the dominant bottleneck once Stage 2's
backlog drains: only one file is parsed at a time, and the 12 Stage 2 workers go idle.

Stage 1 must preserve **FIFO compliance** at the claim level — claims from `queue_id=N`
must appear in `staging.raw_claims` with lower `raw_claim_id` than any claim from
`queue_id=N+1`. Stage 2's existing `SequencedBatchAcquirer` then preserves order
downstream.

## Solution: Parse-Parallel, Commit-Serial

Split Stage 1 into two phases:

1. **Parse phase (parallel, N workers):** read file from disk, parse EDI to in-memory
   `Transaction837p`. Pure CPU/IO, no DB writes. Runs on tokio's blocking pool to avoid
   stalling async workers.
2. **Commit phase (single serial committer):** consume parsed results in queue_id order
   via a reorder buffer, do the transform + INSERT into `staging.raw_claims`, mark
   queue entry `COMPLETED`, move file to `processed/`.

```
                                    ┌─ parse worker 1 ─┐
                                    │                  │
Dispatcher (single)                 ├─ parse worker 2 ─┤  ──>  Reorder
  dequeue_next_n(parse_workers*2)   │                  │       Buffer       ──>  Sequential
  -> mark PROCESSING                ├─ parse worker 3 ─┤    HashMap<queue_id,    Committer
  -> send to parsers                │                  │     ParsedResult>      (single task)
                                    └─ parse worker N ─┘                           │
                                                                                   ├─ commit_parsed_edi_to_staging
                                                                                   ├─ mark COMPLETED / FAILED
                                                                                   └─ move file
```

**Invariant:** `INSERT` operations into `staging.raw_claims` happen in strictly
ascending order of `queue_id`. Out-of-order parse completions sit in the reorder
buffer until their predecessor commits.

## Components

### 1. `QueueManager::dequeue_next_n(limit)`

New method in [crates/pro-worker/src/queue_manager.rs](../crates/pro-worker/src/queue_manager.rs).
Atomically claims up to `limit` rows:

```sql
SELECT ... FROM staging.file_processing_queue
WHERE queue_status = 'QUEUED'
ORDER BY priority ASC, queued_at ASC, queue_id ASC
LIMIT $1
FOR UPDATE SKIP LOCKED
```

Returns `Vec<QueuedFile>` in priority/queued_at/queue_id order. The transaction is
held only long enough to read the rows; the dispatcher then issues `UPDATE ...
SET queue_status='PROCESSING'` for each. Acceptable race: two service instances
might claim the same window if started simultaneously — `SKIP LOCKED` prevents that.

### 2. `ClaimsImporter::parse_edi_file_blocking(file_path)`

New method in [crates/pro-service/src/claims_importer.rs](../crates/pro-service/src/claims_importer.rs).
Wraps the existing `EdiParser::parse_file` in `tokio::task::spawn_blocking` so the
sync parser doesn't block tokio's async runtime when N workers parse concurrently.

```rust
pub async fn parse_edi_file_blocking(&self, file_path: PathBuf)
    -> Result<(Transaction837p, DateTime<Utc>, DateTime<Utc>)>
```

Returns the parsed transaction plus parse start/end timestamps (preserves metric
logging fidelity).

### 3. `ClaimsImporter::commit_parsed_edi_to_staging(...)`

New method that does everything the current `ingest_edi_to_staging` does AFTER the
parse step: batch_id lookup, field transformation, JSONB construction, INSERT into
`staging.raw_claims`. Refactored from current lines 678–end of method.

`ingest_edi_to_staging` becomes a thin wrapper:
```rust
pub async fn ingest_edi_to_staging(&self, file_path: &Path, queue_id: Option<i64>)
    -> Result<IngestResult>
{
    let queue_id = queue_id.ok_or_else(...)?;
    let (transaction, parse_start, parse_end) = self.parse_edi_file_blocking(file_path.to_path_buf()).await?;
    self.commit_parsed_edi_to_staging(file_path, queue_id, transaction, parse_start, parse_end).await
}
```

This guarantees the single-worker path remains byte-for-byte identical in behavior.

### 4. New module `crates/pro-service/src/stage1_pipeline.rs`

Contains:
- `Stage1Pipeline` struct (holds DB pool, importer, channels, config)
- `ParseTask` enum: `Edi(QueuedFile)` | `Csv(QueuedFile)` (CSV passes through without
  parse-parallelism; goes directly to committer)
- `ParsedResult` enum: `EdiOk { queue_id, file_path, parsed, parse_start, parse_end }`,
  `EdiFailed { queue_id, file_path, error }`, `Csv { queue_id, file_path }`
- `spawn()` method: spawns dispatcher + N parsers + 1 committer, returns join handles

### 5. Dispatcher loop

```rust
loop {
    let batch = queue_manager.dequeue_next_n(parse_workers * 2).await?;
    if batch.is_empty() {
        backoff_sleep().await;
        continue;
    }
    for queued_file in batch {
        queue_manager.mark_processing(queued_file.queue_id).await?;
        parse_tx.send(ParseTask::from(queued_file)).await?;
    }
}
```

### 6. Parser worker loop (N copies)

```rust
loop {
    let task = parse_rx.recv().await?;
    match task {
        ParseTask::Edi(qf) => {
            let result = importer.parse_edi_file_blocking(PathBuf::from(&qf.file_path)).await;
            commit_tx.send(ParsedResult::from_edi(qf, result)).await?;
        }
        ParseTask::Csv(qf) => {
            // CSV pass-through: defer all work to committer
            commit_tx.send(ParsedResult::Csv { queue_id: qf.queue_id, file_path: qf.file_path }).await?;
        }
    }
}
```

### 7. Sequential committer

Mirrors [batch_sequencer.rs:344-385](../crates/pro-service/src/batch_sequencer.rs#L344-L385)
exactly — same reorder buffer pattern:

```rust
let mut next_expected_queue_id = recover_next_expected().await?;
let mut buffer: HashMap<i64, ParsedResult> = HashMap::new();

loop {
    let result = commit_rx.recv().await?;
    buffer.insert(result.queue_id(), result);

    while let Some(ready) = buffer.remove(&next_expected_queue_id) {
        commit_one(ready).await;   // INSERT to staging, mark done, move file
        next_expected_queue_id += 1;
    }
}
```

`commit_one` calls `importer.commit_parsed_edi_to_staging` for EDI or
`importer.ingest_file_to_staging` for CSV, then `queue_manager.mark_completed` (or
`mark_failed` on error) and moves the file to `processed/` or `error/`.

### 8. Startup recovery sweep

Before spawning the pipeline, reset any orphaned PROCESSING entries from a prior
crash:

```sql
UPDATE staging.file_processing_queue
SET queue_status = 'QUEUED',
    processing_started_at = NULL,
    updated_by = 'STARTUP_RECOVERY'
WHERE queue_status = 'PROCESSING';
```

Recovery is loud — log at INFO level the count of recovered entries (Rule 3).

Also handle the "DB committed but file not yet moved" edge case: on startup, scan
the input directory; for any file matching a `COMPLETED` queue entry, move it to
`processed/`.

### 9. Reorder-buffer determination of `next_expected_queue_id`

On startup, the committer queries:

```sql
SELECT COALESCE(MIN(queue_id), 1)
FROM staging.file_processing_queue
WHERE queue_status IN ('QUEUED', 'PROCESSING', 'RETRY')
```

That's the lowest queue_id we still need to commit. If there are no in-flight rows,
default to the next expected based on the highest COMPLETED queue_id + 1.

## Configuration

Two new env vars in [crates/pro-service/src/main.rs](../crates/pro-service/src/main.rs):

| Var | Default | Description |
|---|---|---|
| `STAGE1_PARSE_WORKERS` | `1` | Number of parallel parser workers. `1` = legacy serial loop (zero behavior change). Recommended 4–8. |
| `STAGE1_REORDER_BUFFER_MAX` | `parse_workers * 4` | Max parsed-but-not-yet-committed files held in RAM. Dispatcher backs off when full. |

When `STAGE1_PARSE_WORKERS = 1`, the new pipeline still runs but with one parser and
a trivially small reorder buffer — semantically identical to the legacy loop. (Or we
keep the legacy loop entirely as a fallback path. Decision: **keep both paths during
rollout** — `STAGE1_PARSE_WORKERS=1` uses the legacy serial code; `>1` uses the new
pipeline. This minimizes blast radius until we've validated the new path on prod.)

## FIFO invariant — proof sketch

1. Dispatcher pulls rows in `ORDER BY priority, queued_at, queue_id`. Each call to
   `dequeue_next_n` returns a contiguous window in that order.
2. Dispatcher hands tasks to parser workers in that order. Tasks may complete out of
   order (some files parse faster than others).
3. Committer maintains `next_expected_queue_id` — INSERT into `staging.raw_claims`
   happens only when the current `next_expected` is ready. Out-of-order parsed results
   sit in the buffer.
4. Since INSERTs into `staging.raw_claims` are serialized in `queue_id` order, the
   `raw_claim_id` sequence-generated PK is also monotonic in `queue_id`.
5. Stage 2's `SequencedBatchAcquirer` reads from `raw_claims` ordered by `raw_claim_id`,
   so claim-level FIFO holds end-to-end. ∎

## Edge cases

| Case | Handling |
|---|---|
| Parser worker crashes mid-parse | Queue row stays `PROCESSING` until startup recovery sweep resets it. |
| Parse fails (corrupt file, etc.) | Worker sends `ParsedResult::EdiFailed`; committer advances past it (mark `FAILED`, move to `error/`), so the failed file does not block successors. |
| Service crash after DB commit but before file move | Startup sweep moves files matching `COMPLETED` queue entries. |
| Reorder buffer full (slow file blocking buffer) | Parser workers block on bounded `commit_tx.send`. Dispatcher blocks on bounded `parse_tx.send`. Natural back-pressure. |
| New file arrives while pipeline draining | Dispatcher picks it up on next `dequeue_next_n` poll. |
| CSV file inside an EDI stream | Passes through committer; effectively serial for CSV but doesn't block EDI parallel parsing. |

## Memory bounds

Worst case: `STAGE1_REORDER_BUFFER_MAX` parsed `Transaction837p` structs in RAM.
Typical 837P file ~1–2 KB on disk; parsed structure ~5–10× larger = ~20 KB.
With buffer=32, worst case ~640 KB. Negligible.

## Risk summary

- **Refactor scope**: splitting `ingest_edi_to_staging` is the high-risk change.
  Mitigated by keeping the thin wrapper that calls parse-then-commit — single-worker
  behavior is byte-identical.
- **Rollback**: `STAGE1_PARSE_WORKERS=1` reverts to legacy serial loop. No DB schema
  changes needed; no migration coupling.
- **Recovery sweep semantics**: resetting all `PROCESSING` → `QUEUED` at startup is
  safe only if Stage 1 commits are idempotent or guarded. They are: a row in
  `staging.raw_claims` is keyed by `(batch_id, row_number)` (verify), so re-running
  ingest for the same batch_id won't duplicate. (Will verify during implementation.)

## Checklist

### Phase 1 — Refactor importer
- [x] Extract `parse_edi_file_blocking(file_path) -> (Transaction837p, parse_start, parse_end)` from `ingest_edi_to_staging`
- [x] Extract `commit_parsed_edi_to_staging(file_path, queue_id, transaction, parse_start, parse_end) -> IngestResult` from `ingest_edi_to_staging`
- [x] Rewrite `ingest_edi_to_staging` as a thin wrapper that calls both
- [x] Verify existing tests still pass (`cargo test -p pro-service`)

### Phase 2 — QueueManager
- [x] Add `dequeue_next_n(limit: usize) -> Result<Vec<QueuedFile>>` to QueueManager
- [x] Unit test for the new method (mock pool or integration test)

### Phase 3 — Recovery sweep
- [x] Add `QueueManager::reset_stuck_processing() -> Result<usize>` (UPDATE PROCESSING → QUEUED, returns count)
- [x] Add `QueueManager::reconcile_completed_files(input_dir, processed_dir)` helper for the move-on-startup case

### Phase 4 — Stage1Pipeline module
- [x] Create `crates/pro-service/src/stage1_pipeline.rs`
- [x] Define `ParseTask`, `ParsedResult` enums
- [x] Implement dispatcher loop with `dequeue_next_n` + mark_processing + send
- [x] Implement parser worker (calls `parse_edi_file_blocking` or pass-through for CSV)
- [x] Implement sequential committer with reorder buffer + `next_expected_queue_id`
- [x] Implement startup `recover_next_expected()` query

### Phase 5 — Wire into main.rs
- [x] Add `STAGE1_PARSE_WORKERS` env var (default 1)
- [x] Add `STAGE1_REORDER_BUFFER_MAX` env var (default 4 * parse_workers)
- [x] When `parse_workers > 1`: spawn `Stage1Pipeline` instead of legacy loop
- [x] When `parse_workers == 1`: keep existing legacy loop (zero-risk fallback)
- [x] Call recovery sweep before spawning either path

### Phase 6 — Tests
- [x] Unit test: reorder buffer correctly serializes out-of-order arrivals
- [x] Unit test: failed parse advances `next_expected_queue_id` without blocking successors
- [ ] Integration test (manual, user-run): seed 100 EDI files with monotonic patient_control_number, run with workers=4, verify `SELECT patient_control_number FROM staging.raw_claims ORDER BY raw_claim_id` is monotonic
- [x] `cargo test -p pro-service -p pro-worker -p pro-parser-edi` (pre-existing `test_multi_claim_hierarchy` failure on `pro-parser-edi` is unrelated — missing fixture file)
- [x] `cargo build --workspace --release` succeeds

### Phase 7 — Release
- [x] Update `CHANGELOG.md` with `[2.13.0.0]` entry (Y bump — new feature: parallel Stage 1)
- [x] Bump `installer/version.txt` to `2.13.0.0`
- [x] Rebuild installer: `.\build-msi.ps1 -Version "2.13.0.0"`
- [x] Commit + push

## Validation on production

1. Deploy with `STAGE1_PARSE_WORKERS=1` (legacy path) — confirm no regression.
2. Switch to `STAGE1_PARSE_WORKERS=4` — observe throughput on a 6,000-file run.
3. Verify FIFO invariant after run:
   ```sql
   SELECT raw_claim_id, batch_id, encounter_fields->>'patient_control_number'
   FROM staging.raw_claims
   WHERE batch_id IN (<recent batches>)
   ORDER BY raw_claim_id;
   ```
   `raw_claim_id` and `batch_id` should both increase monotonically.

## Version bump rationale (Rule 11)

Y-bump (`2.12.75.0` → `2.13.0.0`): new feature (parallel Stage 1 pipeline). No
breaking changes, no schema migration. Default config preserves existing behavior.

## Out of scope

- Parallelizing the CSV ingest path (`ingest_file_to_staging`) — CSV files are
  infrequent master-data loads, not in the hot path. They pass through the committer
  serially in v1.
- Parallelizing the per-claim transform inside `commit_parsed_edi_to_staging`. The
  inner JSONB construction loop could itself be parallelized later if profiling shows
  it dominates.
- Restructuring Stage 2 — the existing `SequentialCompletionManager` is correct for
  its layer; this work only changes the layer above it.
