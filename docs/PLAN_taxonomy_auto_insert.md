# Plan: Auto-insert unknown taxonomy codes into provider_taxonomy

## Problem

The `claims.provider_taxonomy` table is seeded with only 383 NUCC codes. The full
NUCC code set has 800+ entries and is updated 1-2x/year. When a taxonomy code
arrives (from an 837 file or the CMS NPI Registry API) that is not in the table:

1. **claims_processor** (`lookup_taxonomy`): returns `(None, None)`, discarding
   the valid code from the source data. The provider row gets NULL taxonomy/specialty.
2. **enrichment worker** (`enrich_provider`): binds the code into an UPDATE that
   hits `fk_provider_taxonomy`, failing the entire enrichment for that provider.

Both failures are silent data loss (Rule 3 violation).

## Root Cause

No mechanism exists to add taxonomy codes at runtime. The table is static and
incomplete. Missing codes are treated as invalid rather than unrecognized.

## Fix (v2.17.0.0 - minor, new feature)

### 1. Enrichment worker (`crates/pro-npi-enrichment/src/worker.rs`)

After the specialty lookup (line 182-196), if the taxonomy code was NOT found in
the table, INSERT it with:
- `taxonomy_code`: the raw code from CMS API
- `provider_type`: derive from CMS API taxonomy entry (Individual/Organization)
- `classification`: from CMS API `desc` field (or "Unknown" if not available)
- `specialization`: NULL
- `specialty_display`: from CMS API `desc` field (or the raw code as fallback)
- `definition`: NULL
- `is_active`: true

Then re-query the specialty_display so it populates the provider record.

### 2. Claims processor (`crates/pro-service/src/claims_processor.rs`)

In `lookup_taxonomy` (line 262-279), when the code is NOT in the cache:
- INSERT into `claims.provider_taxonomy` with minimal metadata:
  - `provider_type`: "Unknown"
  - `classification`: "Auto-inserted from claim data"
  - `specialty_display`: the raw taxonomy code
- Add to the in-memory cache
- Return `(Some(code), Some(code))` so the FK-safe code propagates
- Log at WARN level so missing codes are visible (Rule 3)

### 3. No new migration needed

Both changes use runtime INSERT at the application layer. The table schema is
unchanged. Per Rule 15, no migration file is required.

## Checklist

- [x] Modify `enrich_provider` in worker.rs to auto-insert unknown taxonomy codes
- [x] Modify `lookup_taxonomy` in claims_processor.rs to auto-insert and cache unknown codes
- [x] Update CHANGELOG.md with v2.17.0.0 entry
- [x] Update version to 2.17.0.0 (installer/version.txt, installer/Product.wxs)
- [x] Git commit and push
- [ ] Rebuild installer (Windows-only, skip on Mac dev)
