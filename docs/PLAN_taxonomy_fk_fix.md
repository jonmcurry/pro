# Plan: Fix `fk_provider_taxonomy` FK Violation During Provider Prewarm

## Problem

After installing 2.14.0.0, the provider-cache prewarm started failing with:

```
failed to batch insert providers during prewarm:
  error returned from database:
  insert or update on table "provider" violates foreign key constraint
  "fk_provider_taxonomy"
```

`fk_provider_taxonomy` (migrations/044_add_taxonomy_foreign_key.sql) requires
`claims.provider.taxonomy_code` to reference an existing row in
`claims.provider_taxonomy(taxonomy_code)`. Source 837p / CSV files routinely
carry taxonomy codes that are not in the NUCC reference set (typos, padding,
deprecated codes, clearinghouse data quality issues).

Because the prewarm uses a single batch INSERT, one bad taxonomy code rejects
the entire batch - every other provider in that batch is also lost, and the
encounter transactions that referenced them then fail with
`service_line_*_provider_id_fkey`.

## Root cause

`crates/pro-service/src/claims_processor.rs::upsert_providers_in_own_tx`
collects taxonomy codes from encounter fields and binds them directly into the
provider INSERT without validating them against `claims.provider_taxonomy`.

The codebase already has a validation primitive: `lookup_taxonomy()` returns
`(validated_code, specialty)`, where `validated_code` is `None` if the input
is not in the in-memory taxonomy cache (which is loaded from
`claims.provider_taxonomy`). The prewarm was destructuring the tuple as
`(_, spec)`, capturing only the specialty and throwing the validated code
away, then binding the original (unvalidated) input.

## Fix

Capture the validated code from `lookup_taxonomy()` and bind that instead of
the raw input. Unknown codes become `NULL`, the row inserts cleanly, and the
NPI enrichment worker can populate the correct taxonomy later from the NPPI
registry.

This is not a silent fallback (Rule 3): `lookup_taxonomy()` already emits
`warn!("Taxonomy code '{}' not found in cache", ...)` for every unknown code.

## Checklist

- [x] `upsert_providers_in_own_tx`: capture validated taxonomy code, bind
      `validated_taxonomies` instead of raw `new_providers.taxonomy_code`.
- [x] Append migration 075 to baseline `000_baseline_v2.12.sql` (Rule 15).
- [x] Bump `BASELINE_COVERS_THROUGH` 74 -> 75 in `embedded_migrations.rs`.
- [x] Update baseline header to reference migrations 001-075.
- [x] CHANGELOG entry for 2.14.1.0 (patch bump, Rule 11 - bug fix only,
      no new feature or migration).
- [x] Rebuild MSI (Rule 10).
- [x] Commit + push.
