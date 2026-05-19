# Plan: Rendering Provider Specialty - Three-Bug Fix + Backfill

## Symptom

Reported in prod after 2.14.3.0: many `claims.provider` rows have
`taxonomy_code` and/or `specialty` NULL even though the source 837p files
carried valid `PRV*PE*PXC*<taxonomy>` segments.

## Root cause - three compounding bugs in `claims_processor.rs`

### Bug A - Existing providers are never re-evaluated

`upsert_providers_in_own_tx` filters out NPIs already in `self.provider_cache`
before the upsert runs. Once an NPI is in the cache, the prewarm skips it
entirely. If the provider was originally inserted with `taxonomy_code = NULL`
(first claim happened to omit `PRV*PE*PXC` or had an unrecognized taxonomy),
subsequent claims with valid taxonomy info never get a chance to fill it in.
This is the dominant cause of "lots of missing specialties on long-known
providers".

### Bug B - `ON CONFLICT` only updates `updated_at`

```sql
ON CONFLICT (npi) DO UPDATE SET updated_at = CURRENT_TIMESTAMP
```

Even if Bug A is fixed and an existing provider reaches the upsert, this
clause throws away every new column value, including taxonomy and specialty.

### Bug C - `entry().or_insert()` keeps the first sample, drops the rest

Inside `collect_providers_from_encounter` / `collect_providers_from_service_lines`,
the same NPI appearing across many raw_claims in one batch resolves to a
single `ProviderData` entry - the FIRST one inserted. If raw_claim #1 had
`rendering_provider_taxonomy = ""` and raw_claim #47 had the real value,
#47's taxonomy is silently discarded because `or_insert` is a no-op when the
key already exists. So taxonomy info can be lost WITHIN one batch, before we
ever touch the DB.

## Fix

### Code (Bugs A + B + C)

1. **Bug C** - `collect_providers_from_*` helpers: change from `or_insert` to
   an explicit `entry().and_modify(...).or_insert(...)` so that, when the
   existing entry has `None`/empty taxonomy and the new sample has a
   non-empty value, we fill it in. Same treatment for `last_name` (when
   existing is the `"Unknown"` placeholder) and `first_name`.

2. **Bug A** - `upsert_providers_in_own_tx` cache filter:
   ```rust
   // Old: filter out everything in cache
   // New: filter out only providers we have nothing new to teach
   .filter(|(npi, data)| {
       data.has_useful_taxonomy() || !cache.contains_key(npi)
   })
   ```
   Providers carrying a taxonomy ALWAYS go through the upsert so ON CONFLICT
   can fill missing fields on existing rows. Providers with no new info still
   get cache-filtered for perf.

3. **Bug B** - `ON CONFLICT` clause:
   ```sql
   ON CONFLICT (npi) DO UPDATE SET
     updated_at    = CURRENT_TIMESTAMP,
     taxonomy_code = COALESCE(claims.provider.taxonomy_code, EXCLUDED.taxonomy_code),
     specialty     = COALESCE(claims.provider.specialty,     EXCLUDED.specialty)
   ```
   Preserves any already-set value (including NPI-enrichment results from
   NPPI), only fills genuine NULLs.

### Migration 076 - backfill existing rows

Code fix only repairs FORWARD - providers inserted from this point on. We
also need to repair rows that already got into DB under the buggy code.

`migrations/076_backfill_provider_taxonomy.sql`: idempotent one-shot UPDATE
that picks a taxonomy code for each NULL-taxonomy provider from any
`claims.encounter.{rendering,referring,supervising}_provider_taxonomy` that
references the provider, falling back to `claims.service_line.rendering_provider_taxonomy`,
preferring the most recent. Joins against `claims.provider_taxonomy` to also
fill `specialty`.

Safe properties:
- Only updates rows where `provider.taxonomy_code IS NULL OR specialty IS NULL`.
- Joins to `provider_taxonomy` so only valid codes propagate.
- Idempotent - re-running produces the same result.
- `RAISE NOTICE` reports the row count for visibility.

Per Rule 15: append to `000_baseline_v2.12.sql`, bump
`BASELINE_COVERS_THROUGH` 75 -> 76.

## Version bump (Rule 11)

New migration -> Y bump -> **`2.14.3.0` -> `2.15.0.0`**. The fix prior to
2.14.x (e.g. 2.14.0.0) followed the same rule with migration 075.

## Checklist

- [x] `collect_providers_from_encounter` - merge entry instead of `or_insert`.
- [x] `collect_providers_from_service_lines` - same.
- [x] `upsert_providers_in_own_tx` cache filter - admit taxonomy-bearing providers.
- [x] `upsert_providers_in_own_tx` ON CONFLICT - COALESCE-fill taxonomy/specialty.
- [x] `migrations/076_backfill_provider_taxonomy.sql` - new file.
- [x] `embedded_migrations.rs` - register 076; bump `BASELINE_COVERS_THROUGH` to 76.
- [x] `000_baseline_v2.12.sql` - append 076; update header.
- [x] CHANGELOG entry for 2.15.0.0.
- [x] Rebuild MSI (Rule 10).
- [x] Commit + push.
