# Stage 2 Optimization Analysis - 10K Claims in 30 Seconds

## Target
- **Goal**: 10,000 claims / 30 seconds = **333 rec/sec**
- **Current**: 50 rec/sec
- **Improvement Needed**: 6.7x

## CRITICAL: FIFO Requirement

The SRD requires **strict FIFO ordering**. The current architecture implements this via:

### FIFO Architecture (batch_sequencer.rs)
```
1. SequencedBatchAcquirer: Assigns monotonic sequence numbers (1, 2, 3...)
2. Multiple Workers: Process batches in PARALLEL (batches may finish out of order)
3. SequentialCompletionManager: Buffers results, commits ONLY in sequence order
```

**Key Insight**: FIFO is maintained at the **BATCH level**, not the encounter level.
- Batch 1 must complete before Batch 2 is "committed to production"
- Within a batch, encounters are already processed in an arbitrary order (HashMap iteration)

## Current Architecture Analysis

### Processing Flow (per batch)
```
1. SequencedBatchAcquirer assigns sequence number
2. Worker receives batch via channel
3. Query raw_claims from staging (1 query)
4. Group claims by encounter (HashMap - no ordering guarantee!)
5. FOR EACH encounter (SEQUENTIAL within batch):
   a. BEGIN TRANSACTION
   b. process_encounter_with_service_lines()
   c. COMMIT TRANSACTION
6. Report BatchResult to SequentialCompletionManager
7. CompletionManager commits in sequence order
```

### Identified Bottlenecks

#### 1. Sequential Encounter Processing (MAJOR)
**Current**: Each encounter processed one-by-one in a for loop
**Impact**: Cannot utilize multiple CPU cores or DB connections
**Line**: `claims_processor.rs:3291` - `for ((patient_control_number, ...`

#### 2. Per-Encounter Transaction (MINOR)
**Current**: BEGIN/COMMIT for each encounter
**Impact**: Transaction overhead (~1-2ms per encounter)
**Reasoning**: This is intentional - prevents cascading failures

#### 3. Rule Engine Lock Acquisition (MODERATE)
**Current**: `self.rule_engine.read().await` per encounter
**Impact**: RwLock acquisition overhead
**Line**: `claims_processor.rs:3732`

#### 4. String Allocations in Flag Collection (MINOR)
**Current**: Multiple `.to_string()` calls per flag
**Impact**: ~4-6 allocations per triggered flag
**Line**: `claims_processor.rs:3785-3789`

## Safe Optimization Options

### Option A: Parallel Encounter Processing WITHIN Batch (HIGHEST IMPACT)
**Description**: Process multiple encounters concurrently within a single batch
**Expected Speedup**: 4-8x (limited by DB connection pool)
**Risk**: LOW - Does NOT break FIFO

**WHY THIS IS FIFO-SAFE**:
1. FIFO is enforced at BATCH level by SequentialCompletionManager
2. Within a batch, encounters are already in arbitrary order (HashMap iteration)
3. The batch_result (success/failure counts) is reported AFTER all encounters complete
4. CompletionManager commits batches in strict sequence order

**Implementation**:
```rust
// Instead of sequential:
for encounter in encounters { process_encounter() }

// Use parallel WITHIN the batch:
let handles: Vec<_> = encounters.into_iter()
    .map(|enc| tokio::spawn(process_encounter(enc)))
    .collect();
join_all(handles).await;  // Wait for ALL encounters in this batch

// Then report batch result (unchanged)
result_tx.send(batch_result).await;  // CompletionManager handles FIFO
```

**Considerations**:
- Each encounter has its own transaction (already isolated)
- Rule engine uses RwLock (supports concurrent reads)
- DB pool needs sufficient connections (current default: 10?)
- Batch completion waits for ALL encounters - FIFO ordering maintained

### Option B: Batch Rule Execution Across Encounters (MODERATE IMPACT)
**Description**: Collect all service lines, execute rules in batch, then batch insert flags
**Expected Speedup**: 2-3x
**Risk**: LOW - Purely computational optimization
**Implementation**:
- Collect service line contexts from ALL encounters first
- Execute rules for all service lines (still per-service-line, but without lock/unlock overhead)
- Batch insert all flags in single query

### Option C: Pre-acquire Rule Engine Lock (LOW IMPACT)
**Description**: Acquire rule engine read lock once per batch, not per encounter
**Expected Speedup**: 1.1-1.2x
**Risk**: LOW
**Implementation**: Pass `&RuleEngine` to process_encounter instead of acquiring in execute_rules_for_service_lines

### Option D: Increase DB Connection Pool (ENABLES PARALLELISM)
**Description**: Increase pool from 10 to 32-64 connections
**Expected Speedup**: Enables Option A to reach full potential
**Risk**: LOW - Just configuration
**Implementation**: Set `MAX_CONNECTIONS=32` in .env

### Option E: Pipeline Stage 1 and Stage 2 (ARCHITECTURAL)
**Description**: Run Stage 1 (ingestion) and Stage 2 (processing) in parallel on different batches
**Expected Speedup**: 2x throughput
**Risk**: MEDIUM - Requires coordination
**Status**: May already be implemented?

## Recommended Approach

### Phase 1: Quick Wins (Low Risk)
1. **Option C**: Pre-acquire rule engine lock per batch
2. **Option D**: Increase DB connection pool to 32

### Phase 2: Parallel Processing (Medium Risk, High Reward)
3. **Option A**: Parallel encounter processing with tokio::spawn
   - Start with 4 concurrent encounters
   - Monitor for deadlocks or connection exhaustion
   - Scale up to 8-16 if stable

### Expected Results
| Phase | Optimization | Expected Speed |
|-------|--------------|----------------|
| Current | - | 50 rec/sec |
| Phase 1 | Lock + Pool | 55-60 rec/sec |
| Phase 2 | 4x Parallel | 200-240 rec/sec |
| Phase 2+ | 8x Parallel | 300-400 rec/sec |

## Risks and Mitigations

### Risk 1: DB Connection Exhaustion
**Mitigation**: Use semaphore to limit concurrent encounters
```rust
let semaphore = Arc::new(Semaphore::new(8)); // Max 8 concurrent
```

### Risk 2: Memory Pressure with Many Concurrent Encounters
**Mitigation**: Process in chunks of 100-200 encounters

### Risk 3: Transaction Deadlocks
**Mitigation**: Per-encounter transactions are already isolated; low risk

## DO NOT DO (Based on Previous Issues)

1. **DO NOT** defer rules to background queue (broke flag insertion)
2. **DO NOT** use `execute_all()` instead of `execute_all_indexed()` (loses CPT optimization)
3. **DO NOT** remove per-encounter transactions (will cause cascading failures)
4. **DO NOT** batch commits across encounters (same reason)

## Implementation Order

1. [ ] Verify current DB pool size
2. [ ] Increase pool to 32 connections
3. [ ] Pre-acquire rule engine lock per batch (not per encounter)
4. [ ] Add semaphore-bounded parallel encounter processing
5. [ ] Test with 4 concurrent encounters
6. [ ] Scale to 8 concurrent encounters
7. [ ] Measure and verify 333 rec/sec target
