# Rule Performance Fix Plan - v2.12.73.39

## Problem
Stage 2 processing at 4 rec/sec with 500+ COMPOSITE rules (should be 300-500+ rec/sec)

## Root Causes Identified

### 1. INFO-level Logging in Hot Path (CRITICAL)
- Line 3769-3772: `info!("RULES: service_line {} triggered {} rules")`
- Line 3784: `info!("RULES: Flag collected: issue_code={}")`
- With 500 rules × 3 service lines × 1000 claims = 1.5 million log writes
- Each `info!()` call involves mutex locks, formatting, I/O

### 2. Double Condition Evaluation When Triggered
- In composite_rule.rs lines 406-425:
  - First: `self.conditions.iter().all(|c| c.evaluate(ctx))` to check if triggered
  - Then: `self.conditions.iter().filter(|c| c.evaluate(ctx))` to build description
- Re-evaluates ALL conditions a second time for triggered rules

### 3. OR Rules with Always-True Date Conditions (from home.txt analysis)
- Many rules have: `{"operator":"OR","conditions":[{"type":"date_gte","min_date":"1900-01-01"},...]}`
- `date_gte: 1900-01-01` is ALWAYS TRUE (every claim is after 1900)
- OR logic means these rules trigger on EVERY service line regardless of other conditions
- This is a data issue but causes performance impact

## Fixes

### Fix 1: Downgrade Hot Path Logging to DEBUG
- Change `info!()` to `debug!()` for per-service-line and per-flag logging
- Keep summary logging at INFO level (total flags inserted per encounter)
- **Impact**: Eliminates millions of unnecessary log writes

### Fix 2: Remove Double Evaluation in CompositeRule
- Store which conditions matched during first evaluation
- Reuse that result for description building
- **Impact**: ~2x faster rule evaluation for triggered rules

### Fix 3: (NOT IN THIS VERSION - requires user to fix home.txt)
- User should convert OR rules with universal date conditions to AND logic
- OR rules with `date_gte: 1900-01-01` defeat the purpose of other conditions

## Checklist
- [ ] Downgrade claims_processor.rs logging from info! to debug!
- [ ] Optimize CompositeRule evaluate() to avoid double evaluation
- [ ] Update CHANGELOG.md
- [ ] Rebuild installer v2.12.73.39
- [ ] Test that flags still insert correctly

## Critical: DO NOT BREAK
- Flag insertion into claims.service_line_flag
- Flag insertion into claims.encounter_flag
- Rule execution flow
- issue_code matching with flag_issue table
