# Inline Rules Performance Plan

## Goal
Execute rules inline during ingestion (not deferred) while maintaining 10,000 claims / 15-30 seconds (333-666 rec/sec)

## Current State
- 537 COMPOSITE rules loaded from database
- Current inline performance: ~69 rec/sec (with v2.12.73.32 optimizations)
- Target: 333-666 rec/sec (5-10x improvement needed)
- Flags NOT being inserted (separate issue to investigate)

## Performance Analysis

### Bottleneck 1: Rule Execution Architecture
Current flow per service line:
1. Build `RuleExecutionContext` (stack allocation, minimal)
2. Call `rule_engine.execute_all_indexed(&ctx).await`
3. For each rule result, collect flag data
4. After all service lines, batch INSERT flags

The `.await` on `execute_all_indexed` is suspicious - COMPOSITE rules are CPU-only, no DB access needed.

### Bottleneck 2: Async Overhead
Even though COMPOSITE rules use `execute_sync()`, the rule engine still uses async infrastructure:
- `execute_all_indexed` is async
- Each rule iteration may have async overhead
- RwLock acquisition is async

### Bottleneck 3: Rule Count Still Too High
537 rules executing on every service line (even with short-circuit optimization)
- If each rule takes 1-2 microseconds, 537 rules = 0.5-1ms per service line
- 3 service lines per encounter = 1.5-3ms just for rules
- At 3ms/encounter, max throughput = 333 rec/sec (barely hitting target)

## Optimization Strategy

### Phase 1: Eliminate Async Overhead for CPU-Only Rules
- [x] Create `execute_all_indexed_sync()` method for rules that don't need DB
- [x] Check `requires_db_access()` - if ALL rules are CPU-only, use sync path
- [x] Eliminate tokio task switching overhead

### Phase 2: Parallel Rule Execution (SIMD-style) - FUTURE
- [ ] Group rules by condition type for cache-friendly execution
- [ ] Execute DxIn rules together (share diagnosis code iteration)
- [ ] Use rayon for parallel rule evaluation within a batch

### Phase 3: Pre-filter Rules by First Condition - FUTURE
- [ ] Most rules fail on first condition (AND logic)
- [ ] Index rules by their FIRST condition type
- [ ] Skip rules whose first condition cannot match

### Phase 4: Batch Rule Execution Across Service Lines - FUTURE
- [ ] Instead of: for each SL { execute all rules }
- [ ] Do: for each rule { execute across all SLs }
- [ ] Better CPU cache utilization

## Checklist

- [ ] Investigate why flags aren't being inserted (separate from performance)
- [x] Profile current execution to identify actual bottleneck
- [x] Implement sync execution path
- [ ] Test performance after each optimization
- [ ] Achieve 333+ rec/sec target
- [x] Rebuild installer after each change
- [x] Update CHANGELOG.md

## Measurements
| Version | Rec/Sec | Notes |
|---------|---------|-------|
| Pre-optimization | 5-15 | O(N*M) DxIn, no short-circuit |
| v2.12.73.31 | ~69 | HashSet + short-circuit |
| v2.12.73.35 | TBD | Fully sync execution path |
| Target | 333-666 | 10K claims / 15-30 sec |

## Completed Optimizations (v2.12.73.35)

### 1. O(1) HashSet Lookups (v2.12.73.31)
- [x] Replaced Vec<String> with FxHashSet<String> for CptIn, DxIn, PosIn, ModifierIn
- [x] All codes normalized to UPPERCASE at compile time
- [x] DxIn: O(N*M) -> O(N) complexity

### 2. Short-Circuit Evaluation (v2.12.73.31)
- [x] AND conditions stop on first false
- [x] OR conditions stop on first true
- [x] Most rules fail early, avoiding unnecessary condition evaluation

### 3. Fully Synchronous Execution (v2.12.73.35)
- [x] Added `execute_all_indexed_sync()` method
- [x] Cached `all_sync_capable` flag computed once at index build
- [x] Zero async overhead when all rules are CPU-only
- [x] Pre-allocated result vectors
- [x] Minimal branching in hot loop

## Remaining Investigation
- [ ] Flags not being inserted - need to check logs for issue_code mismatch warnings
