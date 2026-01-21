# Stage 2 Performance Optimizations Plan

## Target
- **Goal**: 10,000 claims / 30 seconds = **333 rec/sec**
- **Current**: 228 avg rec/sec (68% of target)
- **Peak**: 714 rec/sec (proves system CAN achieve target)
- **Gap**: Need +105 rec/sec improvement

## Version
Starting from: 2.12.73.52
Target version: 2.12.74.0 (minor version bump for performance improvements)

## Identified Bottlenecks (Priority Order)

### 1. Rule Engine Lock Contention (CRITICAL) - Est. +100-150 rec/sec
**Location:** `crates/pro-service/src/claims_processor.rs:3879`

**Problem:**
```rust
let rule_engine = self.rule_engine.read().await;
```
- Lock acquired INSIDE each parallel encounter task
- 24 concurrent tasks all competing for same RwLock
- Lock held for entire rule execution (~150ms per encounter)

**Solution:** Clone Arc<RuleEngine> BEFORE spawning parallel tasks, pass to each task.

### 2. Sequential Async Rule Execution (CRITICAL) - Est. +50-100 rec/sec
**Location:** `crates/pro-rules/src/rule_engine.rs:605-658`

**Problem:**
- 543 rules x ~3 service lines = ~1,600 async task switches per claim
- Each `rule.execute().await` has scheduler overhead
- Pure CPU rules (COMPOSITE, pattern matching) don't need async

**Solution:** Use `execute_all_indexed_sync()` for CPU-only rules. Only use async for database-dependent rules.

### 3. JSON Clone Storms (HIGH) - Est. +50-100 rec/sec
**Location:** `crates/pro-service/src/claims_processor.rs:353, 1268`

**Problem:**
```rust
serde_json::from_value(raw_claim.encounter_fields.clone())  // CLONE
```
- ~5 clones per claim x 50KB = 250KB unnecessary allocations per claim
- At 228 rec/sec = 57 MB/sec of unnecessary memory copies

**Solution:** Use reference-based deserialization where possible.

### 4. RuleExecutionContext Allocations (MEDIUM) - Est. +40-60 rec/sec
**Location:** `crates/pro-service/src/claims_processor.rs:3901-3915`

**Problem:**
- New context created per service line with String clones
- 315 rules x 3 service lines = ~5,670 allocations per claim

**Solution:** Use Arc<RuleExecutionContext> or accept references in rule execution.

## Implementation Checklist

### Phase 1: Rule Engine Lock Fix (v2.12.73.53) - COMPLETED
- [x] Read claims_processor.rs to understand current lock pattern
- [x] Identify where rule_engine lock is acquired in parallel tasks
- [x] **BETTER SOLUTION:** Removed RwLock entirely - rules never modified at runtime
- [x] Changed `Arc<RwLock<RuleEngine>>` to `Arc<RuleEngine>`
- [x] Removed `.read().await` - direct Arc reference now
- [x] Build verified - compiles successfully
- [x] Update CHANGELOG.md
- [ ] Test with 10K claims import
- [ ] Rebuild installer

### Phase 2: Sync Rule Execution Path (v2.12.73.54)
- [ ] Read rule_engine.rs to understand execute_all_indexed() vs execute_all_indexed_sync()
- [ ] Identify which rules are CPU-only (no database queries)
- [ ] Add flag or detection to use sync path for CPU-only rules
- [ ] Test with 10K claims import
- [ ] Verify all flags still generated correctly
- [ ] Update CHANGELOG.md
- [ ] Rebuild installer

### Phase 3: JSON Clone Reduction (v2.12.73.55)
- [ ] Identify all `.clone()` calls on JsonValue in hot paths
- [ ] Refactor to use references where possible
- [ ] Use Cow<str> for field extraction where appropriate
- [ ] Test with 10K claims import
- [ ] Update CHANGELOG.md
- [ ] Rebuild installer

### Phase 4: Context Allocation Optimization (v2.12.73.56)
- [ ] Refactor RuleExecutionContext to accept references
- [ ] Reduce String allocations in context creation
- [ ] Test with 10K claims import
- [ ] Update CHANGELOG.md
- [ ] Rebuild installer

## Success Criteria
- [ ] Average throughput >= 333 rec/sec
- [ ] No regression in flag accuracy (same flags generated)
- [ ] FIFO ordering maintained
- [ ] No increase in memory usage
- [ ] Web app remains responsive during imports (no DB timeouts)

## Testing Plan
1. Truncate encounter data before each test
2. Import same 10K claim file for consistent comparison
3. Record avg_records_per_sec and peak_records_per_sec
4. Verify flag counts match baseline
5. Check web app responsiveness during import

## Rollback Plan
If performance regresses or bugs are introduced:
1. Revert to v2.12.73.52 binaries
2. Document what went wrong in CHANGELOG.md
3. Re-analyze and adjust approach

## Notes
- FIFO is maintained at BATCH level, not encounter level
- Within a batch, encounters are already in arbitrary order (HashMap iteration)
- Peak of 714 rec/sec proves the system CAN achieve target when not blocked
