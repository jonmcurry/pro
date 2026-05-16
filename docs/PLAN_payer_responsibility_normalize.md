# Plan: Normalize payer_responsibility_code Bind to Satisfy `chk_payer_responsibility`

## Problem

After 2.14.1.0 deployed to prod, claims with non-`P`/`S` payer responsibility
codes fail with:

```
failed to insert encounter:
  error returned from database:
  new row for relation "encounter" violates check constraint "chk_payer_responsibility"
```

`chk_payer_responsibility` on `claims.encounter`
(migrations/004_create_encounter_tables.sql:152) restricts the column to `'P'`
or `'S'`. The 837p parser extracts the value from SBR01, which per the X12
standard can be `P`, `S`, `T` (tertiary), and other codes (`A`, `B`, `C`, ...).
Anything other than P/S trips the constraint and fails the encounter insert.

## Why the schema is this way

The schema is intentional: `encounter.payer_responsibility_code` is the
PRIMARY OBLIGATION for this single claim submission (typically `P`, sometimes
`S` for a resubmission to the secondary). Full COB - including tertiary
payers - lives in `claims.encounter_payer`, whose constraint
(migration 062, `chk_payer_responsibility_code`) explicitly allows `P`/`S`/`T`.

Relaxing the encounter-table constraint would erode that design distinction
and violate Rule 9 (no shortcuts). Normalizing the bind at the call site is
the right answer.

## Current bind sites

Three near-identical chunks of bind logic; each only truncates to 1 char and
defaults to `"P"` if missing, but does NOT validate the character:

- `crates/pro-service/src/claims_processor.rs:689-695`
  (in `process_encounter_with_service_lines`)
- `crates/pro-service/src/claims_processor.rs:1977-1980`
  (in `process_raw_claim`)
- `crates/pro-service/src/builders/encounter_builder.rs:73-76`
  (dead-code scaffolding, per the module's own header comment)

The first two also contain a latent multi-byte UTF-8 panic: `&s[..1]` is
byte-indexed and panics at runtime on a multi-byte first char. Worth fixing
while in the area.

## Fix

Add `normalize_payer_responsibility_code(raw: &str) -> &'static str` in
`builders/mod.rs` and call it from all three sites:

- `"P"` -> `"P"`
- `"S"` -> `"S"`
- `"T"` -> `"S"` (tertiary maps down to secondary on the main encounter row;
  the full tertiary record is preserved separately in `encounter_payer`)
- empty / unrecognized (including SBR01 codes `A`/`B`/`C`/`G`/`H`/`I`) ->
  `"P"` with `warn!` (most common case; default to primary)

Not a silent fallback (Rule 3): every coercion away from the source value
emits a warning naming the offending raw code, so data-quality issues are
visible in the logs.

## Checklist

- [x] `builders/mod.rs`: add `pub fn normalize_payer_responsibility_code`.
- [x] `claims_processor.rs` (~L689, ~L1977): replace bind logic with helper call.
- [x] `builders/encounter_builder.rs` (~L73): use helper for consistency
      (dead code today but kept in sync with the live sites).
- [x] CHANGELOG entry for 2.14.2.0 (Z bump - bug fix only, no new migration).
- [x] Rebuild MSI (Rule 10).
- [x] Commit + push.
