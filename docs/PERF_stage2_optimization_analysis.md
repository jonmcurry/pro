# Stage 2 Performance Optimization Analysis

## Current Architecture

Stage 2 takes claims from `staging.raw_claims` and processes them into
`claims.encounter` + `claims.service_line` + related tables.

**Flow:**
```
SequencedBatchAcquirer (single-threaded, assigns sequence numbers)
  -> mpsc channel -> N workers (STAGE2_WORKER_COUNT=4)
    -> Each worker: group by encounter key, then parallel encounters (MAX_CONCURRENT_ENCOUNTERS=4)
      -> Per-encounter transaction: INSERT encounter, diagnoses, service lines, modifiers, payers, rules
  -> SequentialCompletionManager (commits batches in strict FIFO order)
```

**Current defaults:** 4 workers x 4 concurrent encounters = 16 max active DB connections

---

## Identified Bottlenecks

### 1. FAILED CLAIMS: Sequential per-row UPDATE + INSERT (HIGH IMPACT)

**Location:** `claims_processor.rs:3596-3632`

When claims fail, each one gets an individual UPDATE + INSERT in a loop:
```rust
for (raw_claim_id, ...) in &failed_claims {
    sqlx::query("UPDATE staging.raw_claims SET ... WHERE raw_claim_id = $1")...
    sqlx::query("INSERT INTO staging.import_error_log ...")...
}
```

This is 2 round-trips per failed claim. With 50 failures in a batch, that's
100 sequential queries holding up batch completion.

**Fix:** Batch both operations using `ANY($1)` for the UPDATE (same pattern as
the success path at line 3578) and a single multi-row INSERT for error logs.

---

### 2. ENCOUNTER FIELDS CLONE on first_line (MEDIUM IMPACT)

**Location:** `claims_processor.rs:680`

```rust
let encounter_fields = EncounterFieldsWrapper::new(first_line.encounter_fields.clone())
```

This clones the entire ~50KB JsonValue for every encounter. The `RawClaim`
struct owns the data, so a reference-based approach (borrow instead of clone)
would eliminate this allocation.

---

### 3. PROVIDER PREWARM: Redundant per-encounter prewarm after batch prewarm (LOW IMPACT)

**Location:** `claims_processor.rs:686`

The batch-level prewarm at line 3408 already inserts ALL providers for the
entire batch. Then each encounter calls `prewarm_provider_cache()` again
(line 686) which re-collects NPIs and checks the cache. If the batch prewarm
succeeded, the per-encounter prewarm is pure overhead (cache lookups + NPI
string collection).

**Fix:** Skip per-encounter prewarm if batch-level prewarm succeeded. Pass a
flag from `process_sequenced_batch` indicating batch prewarm status.

---

### 4. FACILITY LOOKUP: OR condition prevents index-only scan (LOW-MEDIUM IMPACT)

**Location:** `claims_processor.rs:699-701`

```sql
SELECT facility_id, organization_id, region_id
FROM claims.facility
WHERE facility_code = $1 OR npi = $1
```

The `OR` condition forces PostgreSQL to evaluate two index scans (or a seq scan)
even though the result is cached. On first encounter per facility, this could
be slower than separate lookups with proper indexes.

**Fix:** Already cached after first hit - low priority unless many unique
facilities per batch.

---

### 5. TOKIO::SPAWN per encounter: Task overhead (LOW IMPACT)

**Location:** `claims_processor.rs:3484`

Each encounter is spawned as a separate tokio task. With 250 claims grouping
into ~50-100 encounters per batch, that's 50-100 task spawns per batch. Tokio
spawn overhead is minimal (~200ns each) but the real cost is the task wakeup
scheduling when all encounters are immediately ready.

**Fix:** Consider `FuturesUnordered` with semaphore instead of individual
spawns. This avoids task creation overhead and keeps the work on the same
executor thread (better cache locality).

---

### 6. SINGLE ACQUIRER THREAD: Batch acquisition is sequential (MEDIUM IMPACT)

**Location:** `batch_sequencer.rs` - SequencedBatchAcquirer

Only one batch is acquired at a time. If workers finish fast, they idle waiting
for the next batch. The acquirer does:
1. SELECT FOR UPDATE SKIP LOCKED (fast)
2. UPDATE set PROCESSING + batch_sequence_number (slower with 250 rows)
3. INSERT into batch_sequences (fast)

Steps 2-3 could overlap with worker processing of the previous batch if the
acquirer pre-fetches the next batch while workers are busy.

**Fix:** Double-buffer: acquire batch N+1 while workers process batch N. Use a
bounded channel with capacity 2 instead of 1.

---

### 7. RULES ENGINE INLINE EXECUTION (CONFIGURABLE - HIGH IMPACT when enabled)

**Location:** `claims_processor.rs:155` (`defer_rules` flag)

When `DEFER_RULES_EXECUTION=false` (default), rules execute inside each
encounter transaction. With 500+ rules, even with CPT-indexed filtering, this
adds significant latency per encounter.

**Status:** Already configurable via env var. Recommend defaulting to `true`
for production throughput.

---

## Recommended Changes (Priority Order)

| # | Fix | Impact | Effort | Risk |
|---|-----|--------|--------|------|
| 1 | Batch failed claims UPDATE + INSERT | HIGH | Low | Low |
| 2 | Double-buffer batch acquisition | MEDIUM | Medium | Low |
| 3 | Skip per-encounter prewarm when batch prewarm succeeded | LOW-MED | Low | Low |
| 4 | Replace tokio::spawn with FuturesUnordered | LOW-MED | Medium | Medium |
| 5 | Avoid encounter_fields.clone() (borrow instead) | LOW-MED | Medium | Medium |
| 6 | Default DEFER_RULES_EXECUTION=true | HIGH | Trivial | None (config) |

---

## Estimated Impact

**Fix #1 alone:** In batches with failures (common during initial imports),
eliminates 2N sequential DB round-trips. For a batch with 50 failures, saves
~500ms-2s of sequential I/O.

**Fix #2:** Eliminates worker idle time between batches. Workers currently
block on channel receive while the acquirer does a ~50ms SELECT+UPDATE. With
4 workers, one is always blocked. Pre-fetching eliminates this gap entirely.

**Fix #6:** Deferring rules removes the dominant per-encounter cost when rules
are enabled. Typical improvement: 40-60% reduction in per-encounter processing
time.

**Combined estimate:** 20-40% throughput improvement for typical workloads;
higher for batches with many failures or when rules are enabled inline.

---

## Current Performance Baseline

- Target: 10,000 claims / 15 seconds = 666 claims/sec
- Batch size: 250 claims
- Batch time: ~30-60s (reported in logs)
- Current throughput: ~250 claims / 30-60s = 4-8 claims/sec (well below target)

The gap suggests the bottleneck is not CPU but DB I/O per encounter. Each
encounter requires ~8-12 INSERT/UPDATE operations in sequence within its
transaction. With 4 concurrent encounters per worker and 4 workers, peak
parallelism is 16 encounters simultaneously - but each encounter still waits
on its own sequential DB operations internally.
