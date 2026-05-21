# Rules Engine Performance Deep Dive

## Current State

The rules engine executes **537 rules** against each service line during Stage 2
claim processing. After CPT indexing filters out ~80% of rules, approximately
**65-95 rules evaluate per service line**. With ~3 service lines per encounter
and 250 claims per batch, this is ~750 rule-set evaluations per batch.

All COMPOSITE rules (91.6% of the rule set) are synchronous CPU-only operations
with no database access. The engine is already heavily optimized through 7 phases:

- CPT indexing (O(1) rule lookup, 80% filtering)
- Synchronous execution (zero async overhead)
- Pre-computed uppercase values (no allocations in hot loop)
- Condition reordering (cheap checks first for short-circuit)
- Batch flag insertion (single UNNEST query)
- FxHashSet for O(1) code lookups

---

## Where Time Is Actually Spent

### Per-service-line breakdown (estimated):

| Operation | Time | Notes |
|-----------|------|-------|
| `ctx.finalize()` (uppercase precompute) | ~1-2 us | 1 call per SL |
| CPT index lookup + universal merge | ~1-3 us | Vec allocation + sort/dedup |
| Rule evaluation loop (65-95 rules) | ~30-150 us | Dominated by DxPattern regex |
| Flag result collection | ~1-5 us | Vec push, string clone |
| **Total per service line** | **~35-160 us** | |

### Per-encounter breakdown:

| Operation | Time | Notes |
|-----------|------|-------|
| Rule execution (3 SLs x ~100 us) | ~300 us | CPU-only |
| `diagnosis_codes.to_vec()` per SL | ~0.5-2 us | Cloning 1-12 strings x 3 |
| Flag batch INSERT (UNNEST query) | ~2-5 ms | Single DB round-trip |
| **Total rules phase per encounter** | **~2.5-5.5 ms** | DB I/O dominates |

### Per-batch (250 claims, ~100 encounters):

| Operation | Time | Notes |
|-----------|------|-------|
| Rule evaluation (100 encounters) | ~30 ms | CPU-only, parallelized |
| Flag insertion (100 encounters) | ~250-500 ms | Sequential within each encounter tx |
| **Total rules phase per batch** | **~280-530 ms** | |

---

## Optimization Opportunities

### 1. SKIP REDUNDANT CPT_IN CONDITION (HIGH IMPACT, LOW EFFORT)

**Problem:** When a rule is selected via CPT index (because it has a `cpt_in`
condition listing CPT "99213"), the first condition evaluated is still `CptIn`
— re-checking what the index already confirmed.

**Location:** `composite_rule.rs:302-308` — the `CptIn` condition evaluates even
though the CPT index already matched this rule to this procedure code.

**Fix:** At rule instantiation, if the rule was selected via CPT index, mark the
`CptIn` condition as `pre_verified = true`. In the evaluate loop, skip
pre-verified conditions.

**Impact:** Saves 1 condition evaluation per CPT-indexed rule. With ~50
CPT-indexed rules per service line, saves ~50 HashSet lookups = ~2-5 us per SL.
Small but free.

---

### 2. AVOID diagnosis_codes.to_vec() PER SERVICE LINE (MEDIUM IMPACT, LOW EFFORT)

**Problem:** Line 4196 clones the diagnosis codes vector for every service line:
```rust
ctx.diagnosis_codes = diagnosis_codes.to_vec();
```

All service lines in an encounter share the same diagnosis codes. The clone is
unnecessary if `RuleExecutionContext` can borrow instead of own.

**Location:** `claims_processor.rs:4196`

**Fix:** Change `RuleExecutionContext.diagnosis_codes` from `Vec<String>` to
accept a reference (`&[String]`). This avoids N allocations per encounter where
N = service_lines * diagnosis_codes.

**Challenge:** `RuleExecutionContext` is also used in contexts where it needs to
own the data. Could use `Cow<'_, [String]>` or a separate borrowed context type
for the hot path.

**Impact:** Eliminates 3-12 string clones x 3 service lines = 9-36 allocations
per encounter. ~5-15 us saved per encounter.

---

### 3. PARALLEL SERVICE LINE RULE EXECUTION (HIGH IMPACT, MEDIUM EFFORT)

**Problem:** Service lines within an encounter are evaluated sequentially
(line 4184: `for sl_ctx in service_line_contexts`). Each service line's rules
are independent — they read shared diagnosis codes but produce independent flag
results.

**Location:** `claims_processor.rs:4184-4236`

**Fix:** Use `rayon::par_iter()` or manually partition service lines across
threads. Since all rule execution is CPU-only and synchronous, this maps
perfectly to data parallelism.

```rust
// Instead of sequential:
for sl_ctx in service_line_contexts { ... }

// Use rayon parallel iterator:
let all_flags: Vec<_> = service_line_contexts
    .par_iter()
    .flat_map(|sl_ctx| {
        let mut ctx = RuleExecutionContext::new(organization_id);
        // ... populate ctx ...
        ctx.finalize();
        rule_engine.execute_all_indexed_sync(&ctx)
            .unwrap_or_default()
            .into_iter()
            .map(|result| (sl_ctx.service_line_id, ...))
            .collect::<Vec<_>>()
    })
    .collect();
```

**Impact:** With 3 service lines, limited benefit (~1.5x). With encounters that
have 10+ service lines (surgical cases), significant benefit (3-5x). Average
improvement: ~20-40% for the rule evaluation phase.

**Risk:** Requires `rayon` dependency. The `RuleEngine` and rules are already
`Arc`-wrapped and `Send+Sync`. Context is stack-allocated per thread.

---

### 4. BATCH RULE EXECUTION ACROSS SERVICE LINES (HIGH IMPACT, HIGH EFFORT)

**Problem:** Each service line builds its own `RuleExecutionContext`, calls
`finalize()`, and iterates through rule indices independently. The index lookup,
sort, and dedup happen per service line even though many service lines share the
same CPT code (e.g., multiple units of 99213).

**Location:** `rule_engine.rs:820-836` — index lookup repeated per SL

**Fix:** Group service lines by CPT code, compute rule indices once per unique
CPT, then evaluate all service lines with the same CPT against the same rule set.

```rust
// Group by CPT for batch evaluation
let mut by_cpt: HashMap<&str, Vec<&ServiceLineRuleContext>> = HashMap::new();
for sl in service_line_contexts {
    by_cpt.entry(&sl.procedure_code).or_default().push(sl);
}

for (cpt, sls) in &by_cpt {
    // Compute rule indices ONCE for this CPT
    let rule_indices = rule_engine.get_rule_indices_for_cpt(cpt);
    
    // Evaluate all SLs with same CPT against same rule set
    for sl in sls {
        let ctx = build_context(sl, diagnosis_codes);
        for &idx in &rule_indices {
            // ... evaluate ...
        }
    }
}
```

**Impact:** When an encounter has multiple service lines with the same CPT code
(common: 3x 99213, 2x 99214), this saves the index lookup + sort + dedup for
each duplicate. Saves ~3-5 us per duplicate CPT group. Modest but compounds.

---

### 5. ELIMINATE SORT+DEDUP IN HOT PATH (MEDIUM IMPACT, LOW EFFORT)

**Problem:** `rule_indices.sort_unstable(); rule_indices.dedup();` runs every
evaluation. This exists because a rule could theoretically appear in both the
CPT index and universal rules.

**Location:** `rule_engine.rs:834-836`

**Fix:** Pre-compute at index build time: ensure universal rules are never also
in the CPT index (or use a `BitSet` for O(1) dedup). At startup, remove
universal rule indices from all CPT index entries. Then at runtime, just
concatenate without sort/dedup.

**Impact:** Eliminates a sort of ~95 elements per service line. With 750
evaluations per batch: saves ~50-100 us per batch total. Small.

---

### 6. ARENA-ALLOCATE FLAG RESULTS (LOW-MEDIUM IMPACT, MEDIUM EFFORT)

**Problem:** Each triggered rule allocates a `RuleResult` with multiple `String`
fields (description, details, issue_code, severity). With 3-10 flags per service
line and 750 evaluations per batch, that's 2,000-7,500 heap allocations for
strings that live only until the batch INSERT.

**Location:** `rule_engine.rs:860` — `results.push(result)` and
`claims_processor.rs:4229` — `flags_to_insert.push(...)`

**Fix:** Use a bump allocator (e.g., `bumpalo`) for the per-encounter rule
execution phase. All strings allocated from the arena, freed in one shot after
the batch flag INSERT.

**Impact:** Reduces allocator pressure. Modern allocators (jemalloc/mimalloc)
handle this well, so improvement is ~5-10% of rule execution time.

---

### 7. CACHE DxPattern REGEX MATCH RESULTS (MEDIUM IMPACT, LOW EFFORT)

**Problem:** Many COMPOSITE rules check the same regex pattern against the same
diagnosis codes. For example, 20 rules might all have `dx_pattern: "^E11"` in
their conditions. Each evaluates the regex against all 12 diagnosis codes
independently = 240 regex matches that could be 12.

**Location:** `composite_rule.rs:321-328`

**Fix:** Before the rule evaluation loop, pre-compute a small cache:
`HashMap<&str, bool>` mapping `(regex_pattern, dx_code) -> matched`. Since
diagnosis codes are shared across all rules in an encounter, this deduplicates
regex evaluation.

**Implementation:** At the `execute_rules_for_service_lines` level:
```rust
// Pre-compute DX pattern matches for this encounter's diagnosis codes
let dx_pattern_cache: HashMap<(regex_id, dx_idx), bool> = ...;
// Pass cache into rule execution context
ctx.dx_pattern_cache = Some(&dx_pattern_cache);
```

**Impact:** If 20 rules share the same DxPattern and there are 12 diagnosis
codes, this reduces 240 regex matches to 12 (20x reduction for that condition
type). DxPattern is the most expensive condition (cost rank 6). Could save
~20-50 us per service line when many DxPattern rules apply.

---

### 8. DEFAULT DEFER_RULES_EXECUTION=true (HIGHEST IMPACT, ZERO EFFORT)

**Problem:** The default is `false`, meaning rules execute inline during import.
This blocks the encounter transaction until all rules complete and flags are
inserted.

**Location:** `claims_processor.rs:178-181`

**Fix:** Change default from `"false"` to `"true"`. Rules execute in a
background worker after import completes. Import throughput jumps immediately
because the ~2-5 ms rules phase per encounter is eliminated from the critical
path.

**Impact:** Eliminates 100% of rules execution time from import. Flags appear
with a delay (seconds to minutes) but claims are imported faster.

**Trade-off:** Flags are not immediately visible after import. If downstream
workflows depend on flags being present at import time, this breaks them.

---

## Recommended Priority

| # | Optimization | Impact | Effort | Risk |
|---|-------------|--------|--------|------|
| 8 | Default DEFER_RULES=true | Eliminates bottleneck | Trivial | Low (config) |
| 3 | Parallel SL execution (rayon) | 20-40% rule phase | Medium | Low |
| 7 | Cache DxPattern results | 10-30% for DxPattern rules | Low | None |
| 2 | Borrow diagnosis_codes | 5-15 us/encounter | Low | Medium |
| 1 | Skip pre-verified CptIn | 2-5 us/SL | Low | None |
| 5 | Eliminate sort+dedup | Marginal | Low | None |
| 4 | Batch by CPT code | Moderate | High | Low |
| 6 | Arena allocation | 5-10% | Medium | Low |

---

## Quick Win Recommendation

If the application can tolerate eventual flag availability (flags appear seconds
after import rather than atomically with it), **change the default to
`DEFER_RULES_EXECUTION=true`**. This single config change eliminates the rules
bottleneck entirely from the import critical path.

If flags must be synchronous, the combination of **#3 (rayon parallel SLs)** +
**#7 (DxPattern cache)** would yield the best improvement with moderate effort:
~30-50% reduction in the rules phase time.
