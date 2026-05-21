# Plan: Rules Engine Performance Optimization Implementation

## Summary

Implement the top 6 optimizations from PERF_rules_engine_deep_dive.md to improve
rule execution throughput during Stage 2 claim processing.

## Checklist

- [x] Opt #8: Default DEFER_RULES_EXECUTION=true (eliminates rules from import critical path)
- [x] Opt #7: Cache DxPattern regex results (deduplicate regex across shared diagnosis codes)
- [x] Opt #1: Skip pre-verified CptIn condition (skip redundant check on CPT-indexed rules)
- [x] Opt #5: Eliminate sort+dedup in hot path (pre-clean universal rules from CPT index at build time)
- [x] Opt #2: Borrow diagnosis_codes (use shared uppercase dx codes, avoid to_vec per SL)
- [x] Opt #3: Parallel service line execution with rayon (data parallelism for CPU-only rules)
- [x] Update CHANGELOG.md (version 2.18.0.0 - performance feature)
- [ ] Commit and push

## Files Modified

1. `crates/pro-service/src/claims_processor.rs` - DEFER default, rayon parallel SLs, borrow dx codes
2. `crates/pro-rules/src/rule_engine.rs` - Remove sort+dedup, add DxPattern cache support
3. `crates/pro-rules/src/templates/composite_rule.rs` - CptIn skip, DxPattern cache evaluation
4. `crates/pro-rules/Cargo.toml` - Add rayon dependency
5. `crates/pro-service/Cargo.toml` - Add rayon dependency
6. `CHANGELOG.md` - Version entry

## Version

2.18.0.0 - Minor version bump (new performance feature: parallel rule execution + caching)
