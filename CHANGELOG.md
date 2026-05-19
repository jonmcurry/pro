# Changelog

## [2.16.0.0] - 2026-05-19

### Bug fix 1 - Service units clamped to 9999.9

Reported in prod: a service line imported with 22,500 units stored `9999.9`
in `claims.service_line.service_unit_count`.

The column is `NUMERIC(15,1)`, but an artificial `9999.9` ceiling existed in
five places. `claims_processor.rs` clamped any larger parsed value down to
`9999.9` before insert, silently corrupting the billed quantity. The X12 837P
SV104 quantity element imposes no such cap, and HCPCS/drug (J-code) unit
counts legitimately exceed 9999.9.

Fix - the cap is removed everywhere; `NUMERIC(15,1)` (max 99999999999999.9) is
now the only upper bound:

- `migrations/077_widen_service_unit_count.sql` (new): drops the
  `service_unit_count <= 9999.9` CHECK, replaces it with
  `chk_service_unit_count_positive CHECK (service_unit_count > 0)`.
- `crates/pro-service/src/claims_processor.rs`: removed the upper-clamp branch
  in `build_service_line_rule_context`. The `<= 0 -> 1` coercion is kept.
- `crates/pro-common/src/constants.rs`: `MAX_SERVICE_UNITS` raised to the
  column capacity.
- `crates/pro-parser-csv/src/mapping.rs`: CSV `Units` `Range` max raised to
  the column capacity.
- `crates/pro-parser-edi/src/validator.rs`: EDI service-line validation
  threshold raised to the column capacity; it now rejects only values that
  genuinely overflow `NUMERIC(15,1)`, with a clear message.

Existing rows already clamped to `9999.9` are not backfilled - the true value
survives in `staging.raw_claims`, so re-importing the affected file repairs
them.

### Bug fix 2 - Primary payer dropped for dependent-patient claims

Reported in prod: `encounter_view.primary_payer_*` blank while
`secondary_payer_*` had data.

`parse_claim_info` in `crates/pro-parser-edi/src/loops.rs` decided whether an
`SBR` segment was the subscriber/billing payer (Loop 2000B) or a COB payer
(Loop 2320) by testing `claim.subscriber_relationship_code.is_empty()`. That
field is SBR02, which is legitimately blank in Loop 2000B whenever the patient
is a dependent carried in a separate Loop 2000C. With SBR02 blank, the COB
Loop 2320 `SBR` was misdetected as a second "first SBR": it overwrote
`payer_responsibility_code` and the COB payer was never appended to
`other_insurance`. The encounter then received a single `encounter_payer` row
and `encounter_view` showed a blank primary. The test fixture did not catch
this because its patient is the subscriber (`SBR*P*18*...`).

Fix - `parse_claim_info` now tracks a dedicated `subscriber_sbr_seen` boolean
for first-SBR detection instead of relying on the SBR02 emptiness test.

Mis-imported encounters are not backfilled; re-importing the affected file
repairs the `payer_responsibility_code` and COB rows.

### Technical Changes

- `crates/pro-parser-edi/src/loops.rs`: added `subscriber_sbr_seen` flag.
- `crates/pro-service/src/claims_processor.rs`: removed service-unit clamp.
- `crates/pro-common/src/constants.rs`, `crates/pro-parser-csv/src/mapping.rs`,
  `crates/pro-parser-edi/src/validator.rs`: raised service-unit caps.
- `migrations/077_widen_service_unit_count.sql` (new).
- `crates/pro-upgrade-manager/src/embedded_migrations.rs`: registered 077;
  bumped `BASELINE_COVERS_THROUGH` to 77.
- `migrations/000_baseline_v2.12.sql`: widened the inline `service_unit_count`
  constraint; appended 077; header bumped to 001-077.

## [2.15.0.0] - 2026-05-19

### Bug fix - Rendering provider specialty/taxonomy missing on many provider rows

Reported in prod after 2.14.3.0: many `claims.provider` rows had
`taxonomy_code` and/or `specialty` set to NULL, even though source 837p
files carried valid `PRV*PE*PXC*<taxonomy>` segments. The taxonomy was being
extracted by the parser and written into `staging.raw_claims.encounter_fields`
correctly, but three compounding bugs in
`claims_processor.rs::upsert_providers_in_own_tx` prevented the master
provider records from being kept in sync.

### Three bugs, all in the provider prewarm

**Bug A - Existing providers never re-evaluated.** The cache filter at
~`L3795` removed every NPI already in `self.provider_cache` before the
upsert ran. If a provider was originally inserted with NULL taxonomy
(because the first claim to mention them happened to omit `PRV*PE*PXC`),
no subsequent claim with valid taxonomy could ever fill the gap. This was
the dominant cause - existing provider rows in a mature DB were stuck with
whatever values the first claim happened to carry.

**Bug B - `ON CONFLICT` only touched `updated_at`.** Even if A were fixed
and an existing provider reached the INSERT, the conflict clause threw
away every new column value:

```sql
-- Was:
ON CONFLICT (npi) DO UPDATE SET updated_at = CURRENT_TIMESTAMP
```

**Bug C - `entry().or_insert()` kept the first sample, dropped the rest.**
The `add` closures in `collect_providers_from_encounter` /
`collect_providers_from_service_lines` collapsed duplicate NPIs to whichever
was inserted first. If raw_claim #1 had `rendering_provider_taxonomy = ""`
and raw_claim #47 had the real value, #47's taxonomy was silently lost
because `or_insert` is a no-op when the key exists. Taxonomy could be lost
WITHIN one batch, before we ever touched the DB.

### Fix

1. **Bug C**: both `add` closures replaced with `entry().and_modify(...).or_insert_with(...)`.
   A helper `merge_provider_field` fills `None`/empty optional fields with
   non-empty new samples; never overwrites already-set values.
   `last_name == "Unknown"` placeholder is also upgraded to a real name
   when one appears later in the batch.

2. **Bug A**: cache filter widened to admit providers with useful taxonomy
   regardless of cache state:
   ```rust
   .filter(|(npi, data)| data.has_useful_taxonomy() || !cache.contains_key(npi))
   ```
   Providers WITHOUT new taxonomy still get cache-filtered for perf.

3. **Bug B**: `ON CONFLICT` now COALESCE-fills the columns that can change:
   ```sql
   ON CONFLICT (npi) DO UPDATE SET
     updated_at    = CURRENT_TIMESTAMP,
     taxonomy_code = COALESCE(claims.provider.taxonomy_code, EXCLUDED.taxonomy_code),
     specialty     = COALESCE(claims.provider.specialty,     EXCLUDED.specialty)
   ```
   Preserves NPI-enrichment results and any other already-set value; only
   fills genuine NULLs.

Restructured the upsert into a single INSERT-with-ON-CONFLICT round-trip
(was: SELECT-then-INSERT-only-for-new). Enrichment queue still only
enqueues genuinely new providers, identified by comparing the
returned-from-RETURNING NPIs against the pre-batch SELECT set.

### Migration 076 - one-shot backfill

The code fix only repairs FORWARD - providers seen from this point on. To
repair existing rows that got into DB under the buggy code, migration 076
runs an idempotent UPDATE that picks a taxonomy code per provider from any
referencing `claims.encounter.{rendering,referring,supervising}_provider_taxonomy`
or `claims.service_line.rendering_provider_taxonomy` (most recent wins),
joins to `claims.provider_taxonomy` to resolve `specialty_display`, and
fills NULL columns on `claims.provider` without overwriting existing
values. `RAISE NOTICE` reports the row count for visibility (Rule 3).

Per Rule 15, migration 076 is also appended to
`migrations/000_baseline_v2.12.sql` and `BASELINE_COVERS_THROUGH` is bumped
from 75 to 76.

### Technical Changes

- `crates/pro-service/src/claims_processor.rs`:
  - New `ProviderData::has_useful_taxonomy()` and
    `merge_provider_field()` helpers.
  - `collect_providers_from_encounter` and `..._from_service_lines`:
    `entry().and_modify(...).or_insert_with(...)` merge logic.
  - `upsert_providers_in_own_tx`: widened cache filter; single-round-trip
    upsert with COALESCE ON CONFLICT; enrichment queue scoped to genuinely
    new providers via post-RETURNING diff.
- `migrations/076_backfill_provider_taxonomy.sql` (new).
- `crates/pro-upgrade-manager/src/embedded_migrations.rs`: registered 076;
  bumped `BASELINE_COVERS_THROUGH` to 76.
- `migrations/000_baseline_v2.12.sql`: appended 076; header bumped to 001-076.

### Version bump rationale (Rule 11)

Y bump (`2.14.3.0` -> `2.15.0.0`): new migration (076). Per the project's
versioning rule, migrations count as features.

## [2.14.3.0] - 2026-05-16

### Bug fix - Encounter DOS range loaded only first service line's dates

Symptom (observed in prod after 2.14.2.0): claims with multiple
`DTP*472*RD8` segments at the service-line level showed only the FIRST
service line's date range on `claims.encounter`. End dates from any later
service lines were invisible at the encounter level - they only existed on
the individual `claims.service_line` rows.

Root cause: the post-parse fallback in `loops.rs` (the only place that
populated encounter dates from service lines) just copied `service_lines[0]`
into `claim.date_of_service_from` / `..._to`. No MIN/MAX, no span. Three
related defects with the same shape:

1. **Per-line `DTP*472`** (the common 837P pattern): only line 1's dates
   reached the encounter; line 2..N's dates were silently dropped.
2. **Multiple claim-level `DTP*472*RD8`**: each one OVERWROTE the previous
   value, so the last-claim-level-DTP wins regardless of which was correct.
3. **Mixed `D8` + `RD8` lines**: D8 single-date lines have
   `service_date_to == None`. Even with a naive MIN/MAX they would have
   under-counted the upper bound.

This is a Rule 3 violation: data was loaded but wrong, with no log
indicating why.

### Fix (Option A - line-derived span)

Replaced the first-line fallback with an unconditional MIN/MAX computation
over `claim.service_lines`:

```text
date_of_service_from = MIN(line.service_date_from)
date_of_service_to   = MAX(line.service_date_to.unwrap_or(line.service_date_from))
```

Properties:

* Single-date (`D8`) lines participate correctly via the `unwrap_or` fallback.
* `chk_dos_range` (`date_of_service_to >= date_of_service_from`) is satisfied
  by construction.
* Any claim-level `DTP*472` is overridden by the line-derived span. A
  `debug!` log fires when the override changes a previously-set value, so
  the discrepancy is auditable (Rule 3).
* Safety net: the parser validator already rejects service lines whose
  `service_date_from` is the `DEFAULT_DATE` sentinel, so MIN/MAX over
  `service_lines` never sees a garbage value.

Rejected alternative (Option B - "defensive widen" that included the
claim-level DTP value as another sample point) for two reasons: 837P doesn't
have a separate "statement period" concept, and widening based on a
submitter-asserted range that disagrees with the actual line dates risks
over-trusting bad data.

### Technical Changes

- `crates/pro-parser-edi/src/loops.rs` (`compute_encounter_dos_span`):
  new `pub(crate)` helper computing MIN/MAX in a single pass. Extracted so
  the logic is unit-testable without constructing a full 837P transaction.
- `crates/pro-parser-edi/src/loops.rs` (post-loop block, was L1138-1143):
  replaced first-line fallback with a call to the helper; emits a `debug!`
  when overriding a non-default claim-level DTP value.
- `crates/pro-parser-edi/src/loops.rs` (tests): 5 new unit tests covering
  empty, single D8, single RD8, multiple RD8, and mixed D8/RD8 cases. The
  `d8_line_with_later_from_extends_to_via_from` test is a direct regression
  guard for the original bug.

### Version bump rationale (Rule 11)

Z bump (`2.14.2.0` -> `2.14.3.0`): parser correctness fix only. No schema
change, no new migration.

## [2.14.2.0] - 2026-05-16

### Bug fix - Encounter insert rejected by `chk_payer_responsibility`

Symptom (reported after 2.14.1.0 deployed to prod):

```
failed to insert encounter:
  error returned from database:
  new row for relation "encounter" violates check constraint "chk_payer_responsibility"
```

Root cause: `chk_payer_responsibility` on `claims.encounter`
(migration 004) restricts `payer_responsibility_code` to `'P'` or `'S'`. The
837p parser extracts the value from SBR01, which per the X12 standard can
also be `T` (tertiary) and other codes (`A`/`B`/`C`/...). The bind site was
only truncating to one character without validating the value, so anything
other than `P`/`S` tripped the constraint.

The encounter table's narrow constraint is intentional - the primary
obligation for a single claim submission is typically `P` or `S`. Full COB
(including tertiary payers) is preserved separately in `claims.encounter_payer`,
whose own constraint allows `P`/`S`/`T`. Relaxing the encounter constraint
would erode that design distinction; normalizing the bind is the right answer.

Fix: added `builders::normalize_payer_responsibility_code` and called it from
both live bind sites in `claims_processor.rs`:

- `P` / `p` -> `P`
- `S` / `s` -> `S`
- `T` / `t` -> `S` (warn; tertiary maps down on the main encounter row)
- empty / unrecognized (including `A`/`B`/`C`/`01`/...) -> `P` (warn)

Every coercion away from the source value emits a `warn!` naming the raw
code, so data-quality drift stays visible in the logs (Rule 3 - no silent
fallback).

Also fixes a latent multi-byte UTF-8 panic in the prior `&s[..1]` form
(byte-indexed slice on a `&str` containing multi-byte chars). The helper uses
`chars().next()` instead.

### Technical Changes

- `crates/pro-service/src/builders/mod.rs` (new helper): `pub fn
  normalize_payer_responsibility_code(raw: &str) -> &'static str` with 4 unit
  tests covering known values, padding/case, unknown defaults, multi-byte.
- `crates/pro-service/src/claims_processor.rs` (~L689 and ~L1977): replaced
  truncate-only bind logic with calls to the helper.
- `crates/pro-service/src/builders/encounter_builder.rs` (~L73): same
  (dead-code scaffolding kept in sync).

### Version bump rationale (Rule 11)

Z bump (`2.14.1.0` -> `2.14.2.0`): bug fix only. No new migration, no new
feature.

## [2.14.1.0] - 2026-05-16

### Bug fix - Provider prewarm batch INSERT rejected by `fk_provider_taxonomy`

Symptom (reported after 2.14.0.0 deployed to prod): every encounter in the
batch was failing with

```
failed to batch insert providers during prewarm:
  error returned from database:
  insert or update on table "provider" violates foreign key constraint
  "fk_provider_taxonomy"
```

Root cause: `fk_provider_taxonomy` (migration 044) requires
`claims.provider.taxonomy_code` to reference a row in
`claims.provider_taxonomy(taxonomy_code)`. Source 837p / CSV files carry codes
that are not in the NUCC reference set. The prewarm was binding those raw
codes straight into the provider INSERT, and because the prewarm uses a single
batch INSERT, one bad code rejected every provider in the batch - which then
cascaded into `service_line_*_provider_id_fkey` failures on every encounter
that referenced any of those providers.

Fix: `upsert_providers_in_own_tx` now uses the existing `lookup_taxonomy()`
validator, which checks against `claims.provider_taxonomy` (loaded into an
in-memory cache at startup). The earlier code destructured the tuple as
`(_, spec)` - capturing only the specialty and throwing the validated code
away. Now both elements are captured: invalid codes become `NULL`, the row
inserts cleanly, and NPI enrichment populates the correct taxonomy later from
the NPPI registry.

Not a silent fallback (Rule 3): `lookup_taxonomy()` already emits
`warn!("Taxonomy code '{}' not found in cache", ...)` for every unknown code.
If the warning volume is too noisy in practice, it can be deduplicated to log
each unknown code once per process lifetime.

### Baseline cleanup

The 2.14.0.0 ship was missing the baseline append for migration 075 (Rule 15).
Folded in now:

- `migrations/000_baseline_v2.12.sql`: appended migration 075 SQL; header
  updated from "001-073" to "001-075".
- `crates/pro-upgrade-manager/src/embedded_migrations.rs`: bumped
  `BASELINE_COVERS_THROUGH` from 74 to 75 so fresh installs do not re-run
  migration 075 after the baseline has already applied it.

### Technical Changes

- `crates/pro-service/src/claims_processor.rs::upsert_providers_in_own_tx`:
  capture `validated_taxonomies` from `lookup_taxonomy()`; bind those instead
  of raw `new_providers.taxonomy_code`.
- `migrations/000_baseline_v2.12.sql`: appended migration 075; updated header.
- `crates/pro-upgrade-manager/src/embedded_migrations.rs`: bumped
  `BASELINE_COVERS_THROUGH`.

### Version bump rationale (Rule 11)

Z bump (`2.14.0.0` -> `2.14.1.0`): bug fix only. No new migration, no new
feature. Baseline catch-up does not change behavior - the SQL it contains was
already shipping in 2.14.0.0 as a separate file.

## [2.14.0.0] - 2026-05-15

### Feature - HARDWARE TUNING FOR 8 VCPU + PROVIDER FK RACE FIX

Two related changes shipped together: a tuning pass sizing the system for the
target deployment profile (8 vCPU box co-located with Postgres, 64 GB RAM), and
a Stage 2 bug fix that eliminates an FK race seen between concurrent encounter
transactions.

### Hardware tuning

Defaults were previously sized for a much larger machine and created severe
connection-pool pressure on the target hardware:

| Setting | Before | After | Rationale |
|---|---|---|---|
| `STAGE2_WORKER_COUNT` | 12 | 4 | Match vCPU count to avoid scheduler thrashing |
| `MAX_CONCURRENT_ENCOUNTERS` | 40 | 4 | Keep `workers * encounters` under the pool size |
| `DB_MAX_CONNECTIONS` | 75 | 24 | `workers * encounters (16)` + headroom for prewarm/status |
| `DB_MIN_CONNECTIONS` | 0 | 4 | One warm connection per worker; avoids first-batch latency |

**Invariant:** `STAGE2_WORKER_COUNT * MAX_CONCURRENT_ENCOUNTERS < DB_MAX_CONNECTIONS`,
and `DB_MAX_CONNECTIONS` must stay under Postgres `max_connections` minus the
web app's reservation. Operators on larger hardware should raise all four
together via env vars.

### New migration: 075_tune_postgresql_for_hardware

Configures Postgres for the 8 vCPU / 64 GB profile via `ALTER SYSTEM SET`:

- `shared_buffers = 16GB` (25% of RAM, community sweet spot)
- `effective_cache_size = 40GB` (planner hint for OS page cache)
- `maintenance_work_mem = 1GB`
- `max_connections = 50` (app pool 24 + web + admin + headroom)
- `max_worker_processes = 8`, `max_parallel_workers = 4` (cap to avoid
  starving the ingestion pipeline)
- `wal_buffers = 16MB`, `checkpoint_completion_target = 0.9`
- `random_page_cost = 1.1` (SSD)

Memory and connection settings require a Postgres restart to take effect.

### Bug fix - Provider FK race between concurrent encounter transactions

Symptom: under parallel Stage 2 processing, encounter inserts intermittently
failed with `service_line_*_provider_id_fkey` errors when two concurrent
encounter transactions both tried to upsert the same provider — one would
commit, the other would see the row before the FK target was visible in its
snapshot.

Fix: added `prewarm_provider_cache_for_batch` in `claims_processor.rs` that
collects every provider NPI referenced anywhere in the batch (encounter-level
and service-line-level) and commits them all in a single dedicated transaction
BEFORE spawning the per-encounter parallel tasks. This guarantees FK targets
are visible to every concurrent encounter transaction.

Per-encounter `prewarm_provider_cache` is preserved as a fallback so a missing
provider doesn't abort the encounter, but the batch-level prewarm is what
prevents the race.

### Other reliability fixes

- **service_unit_count clamping**: the DB constraint requires
  `0 < service_unit_count <= 9999.9`. Out-of-range or unparseable values from
  EDI files now clamp to the boundary (and warn) instead of failing the row.
- **Better error messages**: encounter processing failures now log the full
  anyhow chain (`{:#}` formatter), surfacing the underlying Postgres error
  alongside the top-level context instead of just the outer description.

### Technical Changes

- `migrations/075_tune_postgresql_for_hardware.sql` (new): `ALTER SYSTEM SET`
  for the tuning above, plus `pg_reload_conf()` for reload-capable settings.
- `crates/pro-upgrade-manager/src/embedded_migrations.rs`: registered migration 075.
- `crates/pro-db/src/connection.rs`: new defaults for `DB_MAX_CONNECTIONS` (24)
  and `DB_MIN_CONNECTIONS` (4); updated comments documenting the sizing math.
- `crates/pro-service/src/service.rs`: `STAGE2_WORKER_COUNT` default 12 -> 4.
- `crates/pro-service/src/claims_processor.rs`:
  - `MAX_CONCURRENT_ENCOUNTERS` default 40 -> 4
  - New `ProviderData` struct
  - New `prewarm_provider_cache_for_batch` + helpers
    (`collect_providers_from_encounter`, `collect_providers_from_service_lines`,
    `upsert_providers_in_own_tx`)
  - `service_unit_count` clamping
  - Switched error logging to `{:#}` formatter for full anyhow chains
- `.env.example`, `installer/env.template`, `installer/WriteConfig.vbs`:
  documented invariant and new defaults; clarified the upgrade path for larger
  hardware.

### Version bump rationale (Rule 11)

Y bump (`2.13.0.0` -> `2.14.0.0`): new migration (075) plus a real bug fix.
Migrations count as features under the project's versioning rule.

## [2.13.0.0] - 2026-05-15

### Feature - PARALLEL STAGE 1 INGESTION WITH FIFO GUARANTEE

Stage 1 (EDI file -> `staging.raw_claims`) now supports parallel parsing while
preserving strict FIFO order of claims in staging. Production runs of 6,000+ 837P
files were bottlenecked by the prior single-threaded ingest loop; near the tail of
a run, after Stage 2's backlog drained, throughput dropped to one file at a time
because only one thread was doing any work.

### Architecture

Splits Stage 1 into two phases coordinated by a reorder buffer:

1. **Parse phase (N parallel workers)** — read file from disk, parse 837P to
   in-memory `Transaction837p`. CPU/IO bound. Runs on tokio's blocking pool so
   the sync EDI parser does not stall async workers.
2. **Commit phase (single serial committer)** — receives parsed results in
   arbitrary order, holds them in a HashMap-backed reorder buffer keyed by
   `queue_id`, and applies them to `staging.raw_claims` in strict `queue_id`
   ascending order.

```text
   Dispatcher (1)  -->  parse channel  -->  Parser worker x N  -->  commit channel  -->  Committer (1)
   (dequeue_next_n,                                                                       (reorder buffer +
    mark PROCESSING)                                                                       FIFO commit)
```

**FIFO invariant:** every claim from `queue_id = N` lands in `staging.raw_claims`
with a lower `raw_claim_id` than any claim from `queue_id = N+1`. Stage 2's
existing `SequencedBatchAcquirer` preserves order downstream.

### Configuration

Behavior is unchanged by default. Two new environment variables enable the
parallel pipeline:

| Var | Default | Purpose |
|---|---|---|
| `STAGE1_PARSE_WORKERS` | `1` | Set `>1` to enable parallel pipeline. `1` keeps legacy serial loop. Recommended `4-8`. |
| `STAGE1_REORDER_BUFFER_MAX` | `4 * parse_workers` | Max parsed-but-not-yet-committed files held in RAM. Provides back-pressure. |

Setting `STAGE1_PARSE_WORKERS=1` (the default) is byte-identical to the prior
serial behavior. The parallel pipeline is opt-in.

### Recovery sweep

On every service start, queue entries left in `PROCESSING` by a prior crash are
reset to `QUEUED` (loud at WARN level, per CLAUDE.md Rule 3). This is safe
because Stage 1 commits are idempotent at the `staging.raw_claims` row level.

### Technical Changes

- `crates/pro-parser-edi`: no change.
- `crates/pro-worker/src/queue_manager.rs`:
  - Added `dequeue_next_n(limit)` — atomically claims up to N queued rows with
    `FOR UPDATE SKIP LOCKED` in priority/queued_at/queue_id order.
  - Added `reset_stuck_processing()` — startup recovery sweep for stuck rows.
- `crates/pro-service/src/claims_importer.rs`:
  - Added `parse_edi_file_blocking(file_path)` — pure parse, no DB, wraps the
    sync `EdiParser::parse_file` in `tokio::task::spawn_blocking`.
  - Added `commit_parsed_edi_to_staging(file_path, queue_id, transaction,
    parse_start, parse_end)` — DB-bound transform + INSERT.
  - `ingest_edi_to_staging` is now a thin wrapper that calls both, so the
    single-worker path is byte-identical to v2.12.75.0.
- `crates/pro-service/src/stage1_pipeline.rs` (new ~660 lines):
  - `Stage1Config`, `Stage1Pipeline`, `Stage1Handles` API.
  - Dispatcher / parser worker / committer task functions.
  - `resolve_next_expected_queue_id` for startup recovery of the FIFO counter
    (queries `MIN(queue_id) WHERE queue_status IN ('QUEUED','PROCESSING','RETRY')`).
  - Two unit tests for reorder buffer correctness (drain in order; advance past
    failures).
- `crates/pro-service/src/main.rs`:
  - Reads `STAGE1_PARSE_WORKERS` (default `1`).
  - Runs `reset_stuck_processing` on startup before either path.
  - `> 1`: spawns `Stage1Pipeline`. `== 1`: spawns legacy serial loop. Behavior
    fully preserved for the default config.

### Impact

- Default deployments (no env vars set) get the loud startup recovery sweep and
  no other changes - zero risk to ingestion correctness.
- Operators with large file backlogs can opt in with `STAGE1_PARSE_WORKERS=4`
  (or higher) to parse files concurrently while keeping FIFO ordering in
  `staging.raw_claims`.
- The "files left -> one at a time" tail-drain behavior persists when remaining
  work is less than one parse worker, but peak and steady-state throughput
  during a large backlog improve roughly linearly in `STAGE1_PARSE_WORKERS`
  until the DB pool saturates.

## [2.12.75.0] - 2026-05-13

### Bug Fix - EDI COMPONENT ELEMENT SEPARATOR NOT HONORED FROM ISA16

Fixed an EDI 837P parsing bug where composite elements were always split on `:`
regardless of the component element separator declared in ISA segment position 104
(ISA16). Production files declaring a different separator (e.g. `>`) had service
units, procedure codes, modifiers, and diagnosis pointers silently lost.

### Root Cause
`EdiParser::extract_delimiters` correctly read the component separator from ISA
position 104 and stored it on `EdiEnvelope.component_element_separator`, but every
composite-parsing call site in `crates/pro-parser-edi/src/segments.rs` was hard-coded
to `composite.split(':')`. The declared separator was never plumbed down to the
segment parsers. A file with `ISA*...*P*>~` and `SV1*HC>99213>25*...` would parse
the entire `HC>99213>25` string as the qualifier, leaving procedure code and
modifiers empty.

### User-reported Symptom
"prod file shows P*>~ and the service units/line items are not being processed
correctly."

### Solution
Carry the component separator on each `EdiSegment` so composite parsers cannot
forget it:
- Added `component_separator: char` field to `EdiSegment` (default `':'`).
- Added `EdiSegment::split_composite(index)` helper that splits using the segment's
  own declared separator.
- `EdiParser::split_segments` now propagates `self.component_separator` onto every
  segment it builds.
- Replaced every hard-coded `composite.split(':')` in segments.rs (CLM05, CLM10,
  SV1 procedure composite, SV1 diagnosis pointers, HI diagnosis, SVD procedure
  composite) with the new helper.

### Technical Changes
- `crates/pro-parser-edi/src/types.rs`: added `component_separator` field +
  `split_composite` helper + `Default` impl for test ergonomics.
- `crates/pro-parser-edi/src/parser.rs`: propagated separator in `split_segments`;
  added regression test using `>` separator; fixed pre-existing broken
  `test_split_segments` whose ISA was too short (47 chars) for `extract_delimiters`
  to read position 104.
- `crates/pro-parser-edi/src/segments.rs`: six composite-split sites now use
  `segment.split_composite(idx)`. Also corrected a pre-existing latent bug in
  `SvdSegment::parse` where `procedure_modifier_1` read index 3 (should be index 2)
  and index 3 was duplicated; now reads 2, 3, 4, 5 matching SV1.
- `crates/pro-parser-edi/src/loops.rs`, `validator.rs`: updated 2 test
  `EdiSegment { ... }` struct literals with `..Default::default()`.

### Before
```rust
let composite = segment.get_or_empty(0);
let parts: Vec<&str> = composite.split(':').collect(); // hard-coded
```

### After
```rust
let parts = segment.split_composite(0); // uses the separator declared in ISA16
```

### Impact
- Production 837 files declaring `>` (or any non-colon character) as the component
  element separator now parse SV1 procedure codes, modifiers, diagnosis pointers,
  CLM05/CLM10 composites, HI diagnosis codes, and SVD adjudication composites
  correctly.
- Files using the default `:` separator are unaffected (behavior identical).

## [2.12.74.0] - 2025-01-26

### Bug Fix - TOTAL_CLAIM_CHARGE_AMOUNT NOT POPULATED FROM CLM02

Fixed issue where `total_claim_charge_amount` column was empty in `claims.encounter` table
despite the value being present in CLM02 segment of 837 files.

### Root Cause
The `claims_processor.rs` was calculating `total_claim_charge_amount` by summing service line
charges instead of using the authoritative CLM02 value. Additionally, there was a field naming
mismatch:
- Importer stored: `service_line_1_charge_amount`
- Processor looked for: `service_line_1_line_item_charge_amount`

This caused the sum to always be zero when processing EDI files.

### Solution
Changed `claims_processor.rs` to read `total_claim_charge_amount` directly from
`encounter_fields` (which contains the CLM02 value) instead of calculating from service lines.

### Technical Changes
- `crates/pro-service/src/claims_processor.rs`:
  - Line ~680: Changed from service line summation to direct CLM02 value lookup
  - Line ~1970: Same fix for secondary processing path

### Before
```rust
// Calculated from service lines (broken due to field naming mismatch)
let mut total_claim_charge = rust_decimal::Decimal::ZERO;
for service_line in service_lines {
    if let Some(charge_str) = slf_value.get("service_line_1_line_item_charge_amount")...
}
```

### After
```rust
// Use CLM02 value directly (authoritative value from 837 file)
let total_claim_charge = encounter_fields.get("total_claim_charge_amount")
    .and_then(|s| s.parse::<rust_decimal::Decimal>().ok())
    .unwrap_or(rust_decimal::Decimal::ZERO);
```

### Impact
- Production 837 files with CLM segments like `CLM*99999999999*264.66***11:B:1*Y*A*Y*Y~`
  will now correctly populate `total_claim_charge_amount` with `264.66`

## [2.12.73.86] - 2025-01-21

### Performance - BATCH PROVIDER OPERATIONS (ELIMINATE 16+ SEQUENTIAL DB CALLS)

Major performance optimization that batches all provider operations per encounter into 3 queries
instead of 16+ sequential operations.

### Problem Analysis
Performance gap analysis showed:
- **Peak**: 995 rec/sec (proves system CAN achieve target)
- **Average**: 183 rec/sec (18% of peak)
- **Gap cause**: Sequential DB operations within each encounter

Each encounter was making 16+ sequential `ensure_provider_exists()` calls:
- 4 encounter-level providers (rendering, referring, supervising, billing)
- 4 × N service-line-level providers (per service line with provider NPIs)
- Each call: cache read lock → DB upsert → cache write lock → enrichment queue insert

With 3 service lines average: 4 + (4 × 3) = **16 DB round-trips per encounter**

### Solution
Restructured provider handling into batch operations:

**Before (per encounter):**
```
prewarm_provider_cache() → 2 queries (SELECT existing, INSERT new NPIs only)
ensure_provider_exists() × 16 → 16 DB upserts (sequential)
```

**After (per encounter):**
```
prewarm_provider_cache() → 3 queries total:
  1. SELECT existing providers by NPI (batch)
  2. INSERT new providers with full metadata (batch with UNNEST)
  3. INSERT enrichment queue entries (batch with UNNEST)
ensure_provider_exists() → 0 DB calls (cache-only lookup)
```

### Technical Changes
- `crates/pro-service/src/claims_processor.rs`:
  - **`prewarm_provider_cache()`**: Now collects ALL provider data (NPI, type, names, taxonomy)
    from encounter and service lines, then batch processes:
    - Single SELECT for existing providers
    - Single INSERT for new providers with full metadata
    - Single INSERT for enrichment queue
  - **`ensure_provider_exists()`**: Converted to cache-only lookup (no DB operations)
    - If NPI not in cache after prewarm, returns None (claim proceeds with NULL provider_id)
    - This is safe - NULL provider_id is already handled gracefully

### Expected Impact
- **Per-encounter reduction**: 16+ DB round-trips → 3 batch queries
- **Estimated improvement**: 3-5x reduction in per-encounter DB time
- **Target**: Sustain closer to peak throughput (995 rec/sec)

### FIFO Safety
This optimization is FIFO-safe:
- All provider operations happen within a single encounter's transaction scope
- No cross-batch dependencies introduced
- Batch-level FIFO maintained by SequentialCompletionManager

## [2.12.73.85] - 2025-01-21

### Performance - OPTIMIZE RULE CONDITION EVALUATION ORDER

Major performance optimization that reorders rule conditions for optimal short-circuit evaluation.

### Problem Analysis
Reviewing the example rule (AHRQOP001A):
```json
{"operator":"AND","conditions":[
  {"type":"date_gte","min_date":"2017-07-01"},
  {"type":"dx_in","codes":[...126 diagnosis codes...]},
  {"type":"cpt_in","codes":["99281","99282","99283","99284","99285","99291"]}
]}
```

With AND logic and short-circuit evaluation, conditions are evaluated in order until one fails.
The original order was determined by the JSON definition order:
1. `date_gte` - cheap (single comparison)
2. `dx_in` with 126 codes - **EXPENSIVE** (iterates ALL diagnosis codes against HashSet)
3. `cpt_in` - cheap (O(1) HashSet lookup) - but evaluated LAST!

For most service lines that don't match, the expensive `dx_in` check runs before discovering
the CPT doesn't match. With 543 rules × 30,000 service lines, this adds significant overhead.

### Solution
Added `evaluation_cost()` method to `CompiledCondition` and sort conditions by cost during
rule compilation:

| Cost | Condition Type | Reason |
|------|---------------|--------|
| 1 | DateGte, DateLte | Single comparison |
| 2 | CptIn, PosIn | O(1) HashSet lookup |
| 3 | ModifierIn, ModifierNotIn | O(N) where N = modifiers (0-4) |
| 4 | CptPattern, PosPattern | Single regex match |
| 5 | DxIn | O(N) diagnosis × O(1) HashSet |
| 6 | DxPattern, DxPatternExclude | O(N) regex matches |

After optimization, conditions are reordered to:
1. `date_gte` (cost 1) - cheap, evaluated first
2. `cpt_in` (cost 2) - cheap, evaluated second (fails fast for non-matching CPTs)
3. `dx_in` (cost 5) - expensive, evaluated LAST (only if date and CPT match)

### Technical Changes
- `crates/pro-rules/src/templates/composite_rule.rs`:
  - Added `evaluation_cost()` method to `CompiledCondition`
  - Sort conditions by cost during `CompositeRuleTemplate::create()`
  - Enables short-circuit to skip expensive DX checks when cheaper conditions fail

### Expected Impact
For rules with AND logic (most rules):
- Non-matching service lines fail fast on cheap checks (date, CPT)
- Expensive DX iteration only runs when cheaper conditions pass
- Estimated 20-40% reduction in rule evaluation time

## [2.12.73.84] - 2025-01-21

### Bugfix - REVERTED DIAGNOSIS CODE OPTIMIZATION

Reverted the diagnosis code sharing optimization from v83 to fix FK constraint violation on encounter insert.

### Issue
v83 caused FK constraint violation: `encounter_rendering_provider_id_fkey`
No claims were being processed into the claims schema.

### Fix
- Reverted to using `ctx.finalize()` instead of `ctx.finalize_with_shared_dx()`
- Removed pre-computation of `diagnosis_codes_upper` per encounter
- Kept the CPT uppercase optimization in rule_engine (uses ctx.procedure_code_upper)
- Kept the direct sync rule execution from v82

### Retained Optimizations
- Direct sync rule execution (v82)
- Use of pre-computed `ctx.procedure_code_upper` in rule engine index lookup

## [2.12.73.83] - 2025-01-21

### Performance - ELIMINATE REDUNDANT ALLOCATIONS IN RULE EXECUTION (REVERTED IN v84)

**NOTE: This version had a bug causing FK constraint violations. See v84.**

Major performance optimization eliminating redundant string allocations in rule execution hot path.

### Root Cause Analysis
After v82 showed no improvement (177 rec/sec), profiled the actual rule execution path:
1. **Redundant CPT uppercase**: `execute_all_indexed_sync()` was calling `to_uppercase()` on every call despite `ctx.procedure_code_upper` already being pre-computed
2. **Repeated diagnosis code allocations**: For each of 30,000 service lines:
   - `diagnosis_codes.to_vec()` - clones all diagnosis codes (~8 per encounter)
   - `finalize()` then calls `diagnosis_codes.iter().map(|s| s.to_uppercase()).collect()` - uppercase all codes again
3. With ~3 service lines per encounter, diagnosis uppercase was computed 3x per encounter instead of 1x

### Solution
1. **Use pre-computed uppercase CPT**: Changed `execute_all_indexed_sync()` to use `ctx.procedure_code_upper` directly
2. **Share uppercase diagnosis codes**: Added `finalize_with_shared_dx()` method that accepts pre-computed uppercase diagnosis codes
3. **Pre-compute once per encounter**: Compute uppercase diagnosis codes once per encounter, share across all service lines

### Technical Changes
- `crates/pro-rules/src/rule_engine.rs`:
  - `execute_all_indexed_sync()`: Uses `ctx.procedure_code_upper` instead of computing `to_uppercase()`
  - `execute_all_indexed()`: Same fix for fallback path
  - Added `finalize_with_shared_dx()` method for shared diagnosis codes

- `crates/pro-service/src/claims_processor.rs`:
  - Pre-compute `diagnosis_codes_upper` once per encounter
  - Call `finalize_with_shared_dx()` for each service line

### Expected Impact
- **Saved allocations per 10K claims**:
  - CPT uppercase: 30,000 string allocations eliminated
  - Diagnosis codes: ~160,000 string allocations eliminated (8 dx × 10K claims × 2 extra service lines)
- Estimated 5-15% throughput improvement from reduced allocation pressure

## [2.12.73.82] - 2025-01-21

### Performance - ELIMINATE ASYNC OVERHEAD FOR SYNC RULE EXECUTION

Major performance optimization eliminating async/await overhead for rule execution.

### Root Cause Analysis
After extensive testing, the actual bottleneck was identified:
- All 543 composite rules are sync-capable (no database access needed)
- But rule execution was called via async wrapper: `rule_engine.execute_all_indexed(&ctx).await`
- Each `.await` triggers tokio scheduler overhead even for sync-only code
- With 30,000 service lines per 10K claim batch, this adds 9-30 seconds of pure scheduler overhead

### Solution
- Changed from async `execute_all_indexed()` to direct sync `execute_all_indexed_sync()` call
- Eliminates 30,000 unnecessary async yield points per 10K claim batch
- No functional change - both methods produce identical results

### Technical Changes
- `crates/pro-service/src/claims_processor.rs`:
  - Replaced: `match rule_engine.execute_all_indexed(&ctx).await`
  - With: `let results = rule_engine.execute_all_indexed_sync(&ctx)`
  - Removed `.await` for rule execution entirely

### Expected Impact
- **Estimated improvement**: 33-50% throughput increase (based on async overhead analysis)
- Target: 333 rec/sec (10K claims in 30 seconds)
- Previous average: ~180 rec/sec
- Expected new average: ~240-270 rec/sec

### Performance Analysis Background
| Component | Time/SL | Count (10K claims) | Total Time |
|-----------|---------|-------------------|------------|
| Async overhead (before) | 0.3-1.0ms | 30,000 SLs | 9-30s |
| Async overhead (after) | 0ms | 30,000 SLs | 0s |

## [2.12.73.79] - 2025-01-21

### Cleanup - REMOVED UNUSED ENCOUNTERS_PER_TRANSACTION CODE

Cleaned up unused code and configuration from transaction batching experiment.

### Changes
- Removed unused `DEFAULT_ENCOUNTERS_PER_TRANSACTION` constant
- Removed unused `get_encounters_per_transaction()` function
- Removed `ENCOUNTERS_PER_TRANSACTION` from .env and WriteConfig.vbs
- Updated WriteConfig.vbs comments

### Configuration Summary (Final)
- `MAX_CONCURRENT_ENCOUNTERS=40` (increased from original 24)
- `DB_MAX_CONNECTIONS=75`
- `BATCH_SIZE=250`
- `STAGE2_WORKER_COUNT=12`

### Performance Testing Summary
The transaction batching experiment (v69-v77) showed that batching multiple encounters
per transaction hurt average throughput due to sequential processing within batches.
The code has been restored to the original simple structure where each encounter
gets its own parallel transaction.

## [2.12.73.77] - 2025-01-21

### Performance - RESTORED ORIGINAL SIMPLE ENCOUNTER LOOP

Restored the original simple encounter processing loop, eliminating transaction batching overhead.

### Problem
- v76 showed 182 avg rec/sec (still below original 220)
- Transaction batching code had overhead even with `ENCOUNTERS_PER_TRANSACTION=1`:
  - `encounter_groups.into_iter().collect()` - HashMap to Vec conversion
  - `.chunks(1).map(|chunk| chunk.to_vec()).collect()` - unnecessary chunking and cloning
  - Extra loop nesting and Vec allocations

### Solution
- Reverted to original simple encounter loop structure (matching commit c1802d2)
- Direct iteration over `encounter_groups` HashMap
- Each encounter spawns its own task with its own transaction
- No intermediate Vec conversions or chunking
- Removed `ENCOUNTERS_PER_TRANSACTION` config variable (no longer used)

### Technical Changes
- `crates/pro-service/src/claims_processor.rs`:
  - Removed transaction batching code path entirely
  - Restored original direct iteration: `for ((pcn, dos), service_lines) in encounter_groups`
  - Each encounter gets its own BEGIN → process → COMMIT

### Expected Impact
- Should restore performance to original ~220 avg rec/sec baseline
- Cleaner, simpler code that's easier to maintain

## [2.12.73.75] - 2025-01-21

### Performance - SKIP SAVEPOINTS WHEN ENCOUNTERS_PER_TRANSACTION=1

Eliminated unnecessary savepoint overhead when using single-encounter transactions.

### Problem
- v74 still showed only 162 avg rec/sec (vs original 220)
- Even with `ENCOUNTERS_PER_TRANSACTION=1`, code was creating SAVEPOINTs
- Each encounter had 3 extra DB round-trips: SAVEPOINT, RELEASE SAVEPOINT, COMMIT
- SAVEPOINTs are only useful when batching multiple encounters per transaction

### Solution
- Skip SAVEPOINT creation/release when `tx_batch_encounters.len() == 1`
- Only use savepoints when actually batching multiple encounters per transaction
- Removes 2 unnecessary DB round-trips per encounter in single-encounter mode

### Technical Changes
- `crates/pro-service/src/claims_processor.rs`:
  - Added `let use_savepoints = tx_batch_encounters.len() > 1`
  - Only execute SAVEPOINT/RELEASE when `use_savepoints` is true

### Expected Impact
- Should restore performance closer to original 220 avg rec/sec
- Eliminates ~2 DB round-trips per encounter when `ENCOUNTERS_PER_TRANSACTION=1`

## [2.12.73.73] - 2025-01-21

### Performance - REVERT TO PARALLEL ENCOUNTERS (ENCOUNTERS_PER_TRANSACTION=1)

Reverted to 1 encounter per transaction after testing showed batch transactions hurt average throughput.

### Test Results Summary
| Version | ENCOUNTERS_PER_TRANSACTION | Peak rec/sec | Avg rec/sec |
|---------|---------------------------|--------------|-------------|
| Original | 1 | 754 | 220 |
| v70 | 10 | 951 | 151 |
| v72 | 5 | 894 | 157 |

### Analysis
- Batch transactions increased peak (less transaction overhead during bursts)
- But sequential processing within batches killed average throughput
- The parallelism loss outweighed the transaction overhead reduction
- Original approach (1 encounter per transaction) had best average throughput

### Solution
- Reverted `ENCOUNTERS_PER_TRANSACTION` to 1 (original behavior)
- Each encounter processes in its own transaction with full parallelism
- Keep `MAX_CONCURRENT_ENCOUNTERS=40` for higher parallelism than original 24

### Technical Changes
- `crates/pro-service/src/claims_processor.rs`: Changed default from 5 to 1
- `.env`: Updated `ENCOUNTERS_PER_TRANSACTION=1`
- `installer/WriteConfig.vbs`: Updated `ENCOUNTERS_PER_TRANSACTION=1`

### Next Steps for Further Optimization
The batch transaction approach needs redesign to process encounters in PARALLEL within each batch,
not sequentially. This requires a different architecture that maintains transaction boundaries
while allowing concurrent DB operations within those boundaries.

## [2.12.73.71] - 2025-01-21

### Performance - TUNING ENCOUNTERS_PER_TRANSACTION

Reduced `ENCOUNTERS_PER_TRANSACTION` from 10 to 5 to increase parallelism.

### Problem
- v2.12.73.69/70 showed peak 951 rec/sec but average only 151 rec/sec
- Encounters within a transaction batch are processed sequentially (share same tx)
- With 10 encounters per tx and ~80 encounters per batch = only 8 parallel transaction batches
- Underutilized the 40 concurrent permit limit

### Solution
- Reduced `ENCOUNTERS_PER_TRANSACTION`: 10 → 5
- With 5 encounters per tx and ~80 encounters per batch = ~16 parallel transaction batches
- Better utilization of concurrent permits while still reducing transaction overhead

### Technical Changes
- `crates/pro-service/src/claims_processor.rs`: Changed default from 10 to 5
- `.env`: Updated `ENCOUNTERS_PER_TRANSACTION=5`
- `installer/WriteConfig.vbs`: Updated `ENCOUNTERS_PER_TRANSACTION=5`

### Tradeoff Analysis
- Old (1 encounter/tx): 80 parallel, 80 BEGIN/COMMIT - high parallelism, high overhead
- v2.12.73.70 (10 enc/tx): 8 parallel, 8 BEGIN/COMMIT - low parallelism, low overhead
- v2.12.73.71 (5 enc/tx): 16 parallel, 16 BEGIN/COMMIT - balanced parallelism and overhead

## [2.12.73.69] - 2025-01-21

### Performance - BATCH TRANSACTIONS AND CONCURRENCY TUNING

Major performance optimization for Stage 2 processing implementing transaction batching.

### Problem
- Each encounter processed in its own transaction (50-80 BEGIN/COMMIT per batch)
- Transaction overhead dominated processing time (~113ms per DB operation)
- Peak of 754 rec/sec proved CPU/rules fast enough; bottleneck was I/O

### Solution - Phase 1: Batch Transactions
Instead of 1 transaction per encounter, batch 10 encounters into a single transaction:
- Reduces transaction BEGIN/COMMIT cycles by 10x
- Uses PostgreSQL SAVEPOINTs for per-encounter rollback granularity
- If one encounter fails, only that encounter rolls back; others commit

### Solution - Phase 3: Increased Concurrency
- `MAX_CONCURRENT_ENCOUNTERS`: 24 → 40 (concurrent transaction batches)
- With 10 encounters per transaction batch, this processes up to 400 encounters in parallel

### New Configuration Options
- `ENCOUNTERS_PER_TRANSACTION=10` - encounters per database transaction (default: 10)
  - Higher values reduce overhead but increase rollback scope on errors
  - Lower values provide finer rollback granularity but more transaction overhead

### Technical Changes
- `crates/pro-service/src/claims_processor.rs`:
  - Added `get_encounters_per_transaction()` configuration function
  - Refactored `process_sequenced_batch()` to group encounters into transaction batches
  - Each transaction batch creates savepoints for individual encounter rollback
  - Changed default `MAX_CONCURRENT_ENCOUNTERS` from 24 to 40
- `.env`: Added `ENCOUNTERS_PER_TRANSACTION=10`, updated `MAX_CONCURRENT_ENCOUNTERS=40`
- `installer/WriteConfig.vbs`: Added `ENCOUNTERS_PER_TRANSACTION=10`, updated `MAX_CONCURRENT_ENCOUNTERS=40`

### Expected Impact
- +100-150 rec/sec from reduced transaction overhead (Phase 1)
- +30-50 rec/sec from increased parallelism (Phase 3)
- Target: 333 rec/sec (10K claims / 30 seconds)

### Migration Notes
- No database changes required
- Existing deployments can tune `ENCOUNTERS_PER_TRANSACTION` and `MAX_CONCURRENT_ENCOUNTERS` via .env

## [2.12.73.67] - 2025-01-21

### Fix - REVERTED AGGRESSIVE POOL SETTINGS (PERFORMANCE RESTORATION)
Reverted connection pool settings that caused performance regression from 228 to 171 rec/sec.

### Problem
- Performance dropped from 228 rec/sec to 171 rec/sec (~25% regression)
- `idle_timeout=10s` was too aggressive, causing connection churn during processing
- `max_connections=40` was too low for optimal parallelism

### Solution
Reverted to balanced settings that allow connection release while maintaining performance:
- `max_connections`: 40 → 75 (restored for parallel throughput)
- `min_connections`: 0 → 5 (keep small warm pool, still allows shrinkage to 5)
- `idle_timeout`: 10s → 60s (prevents connection churn during processing)

### Connection Release Strategy
With the exponential backoff from v2.12.73.65 still in place:
- During processing: Pool scales to 75 connections as needed
- After processing: Background loops back off to 30s polling
- Connections idle > 60s will be closed (down to min of 5)
- Web app can use remaining ~95 connections (assuming PostgreSQL max_connections=100)

### Technical Changes
- `crates/pro-db/src/connection.rs`: Updated defaults
- `installer/WriteConfig.vbs`: Updated defaults

## [2.12.73.65] - 2025-01-21

### Fix - STAGE 1 QUEUE PROCESSOR POLLING + IDLE CONNECTIONS
Root cause: Stage 1 queue processor was polling every 1 second even when idle, keeping connections active.

### Problem
- `pg_stat_activity` showed 51 connections in `idle` state with last query being `SELECT ... FROM staging.file_processing_queue WHERE queue_status = 'QUEUED'`
- Stage 1 queue processor polled every 1 second when no files were queued
- Frequent polling kept "touching" connections, preventing idle timeout from closing them
- `test_before_acquire=true` added extra query overhead per connection acquisition

### Solution
1. **Stage 1 Queue Processor**: Added exponential backoff when no files queued
   - When files exist: Process immediately
   - When idle: Backoff from 2s → 4s → 8s → 16s → 30s (max)
   - Resets to fast polling when new files arrive

2. **Disabled test_before_acquire**: Reduces connection overhead
   - PostgreSQL connections are reliable; testing adds latency
   - Removes extra `SELECT 1` query on each connection acquisition

### Expected Impact
- Connections should now release within 10-30 seconds after processing
- Reduced database load when idle: ~1 query per 30 seconds (was 1 per second)
- Slight performance improvement from removing test_before_acquire overhead

### Technical Changes
- `crates/pro-service/src/main.rs`: Added exponential backoff to Stage 1 queue processor
- `crates/pro-service/src/service.rs`: Added exponential backoff to Stage 1 queue processor
- `crates/pro-db/src/connection.rs`: Set `test_before_acquire: false`

## [2.12.73.63] - 2025-01-20

### Fix - AGGRESSIVE CONNECTION POOL SETTINGS
Reduced max_connections and idle_timeout to force faster connection release.

### Problem
- Even with polling backoff, connections weren't releasing
- 75 max_connections was excessive for 12 workers
- 60 second idle timeout was too long for web app coexistence

### Solution
1. **Reduced max_connections**: 75 → 40
   - 12 workers + completion manager + acquirer = ~15 components
   - 40 connections is sufficient with headroom for parallel queries

2. **Reduced idle_timeout**: 60s → 10s
   - Aggressively close connections after they become idle
   - Forces pool to shrink quickly when batch processing completes

### Technical Changes
- `crates/pro-db/src/connection.rs`:
  - `max_connections` default: 75 → 40
  - `idle_timeout_seconds` default: 60 → 10
- `installer/WriteConfig.vbs`: Updated defaults
- `.env`: Updated settings

### Notes
- During batch processing, connections scale up to 40 as needed
- After processing, connections should drop within 10-20 seconds
- If you need more connections during processing, set `DB_MAX_CONNECTIONS` higher

## [2.12.73.61] - 2025-01-20

### Fix - CONNECTION POOL RELEASE (REDUCED POLLING FREQUENCY)
Connections were staying active because background loops polled the database too frequently.

### Problem
- `pg_stat_activity` showed 75 connections running `SELECT sequence_number, assigned_at... FROM staging.batch_sequences`
- **SequentialCompletionManager** checked for stuck sequences every 100ms (600 queries/minute)
- **SequencedBatchAcquirer** polled for pending claims every 10ms (6000 queries/minute when idle)
- This constant polling prevented connection pool from releasing idle connections

### Solution
1. **SequentialCompletionManager**: Changed stuck sequence check from 100ms to 30 seconds
   - Stuck sequences are rare (timeout is 5 minutes anyway)
   - 30 second interval is sufficient for detection

2. **SequencedBatchAcquirer**: Implemented exponential backoff when no claims
   - When claims exist: 50ms between acquisitions (fast processing)
   - When idle: Backoff from 2s → 4s → 8s → 16s → 30s (max)
   - Resets to fast polling when new claims arrive

### Expected Impact
- Connections should release within 60 seconds after processing completes
- Minimal impact on processing throughput (only affects idle periods)
- Reduced database load when idle from ~6600 queries/min to ~3 queries/min

### Technical Changes
- `crates/pro-service/src/batch_sequencer.rs`:
  - Line ~295: Changed stuck check interval from 100ms to 30s
  - Lines ~113-145: Added exponential backoff for idle acquisition loop

## [2.12.73.59] - 2025-01-20

### Performance - JSON CLONE REDUCTION IN ENCOUNTER GROUPING
Eliminated unnecessary JSON deserialization/cloning in the encounter grouping hot path.

### Problem
- Each raw claim's `encounter_fields` (~50KB) was being cloned and fully deserialized
- Just to extract two fields: `patient_control_number` and `date_of_service_from`
- At 250 claims/batch = 12.5 MB of unnecessary memory allocations per batch
- This was happening in both the parallel and sequential processing paths

### Solution
- Added `get_field_from_json()` helper that extracts fields directly from JsonValue
- Removed `serde_json::from_value(raw_claim.encounter_fields.clone())` calls
- Now uses `get_field_from_json(&raw_claim.encounter_fields, "field_name")` instead
- Zero-copy access to the two required fields

### Expected Impact
- Reduced memory churn by ~12.5 MB per batch
- Faster encounter grouping phase (CPU-bound)
- Potential +20-50 rec/sec improvement

### Technical Changes
- `crates/pro-service/src/claims_processor.rs`:
  - Added `get_field_from_json()` helper function
  - Updated parallel batch processing path (line ~3339)
  - Updated sequential batch processing path (line ~358)

## [2.12.73.57] - 2025-01-20

### Fix - CONNECTION POOL MIN_CONNECTIONS = 0
Changed default `min_connections` from 5 to 0 to allow full pool shrinkage after batch processing.

### Problem
- Even with `idle_timeout=60s`, all 75 connections stayed open after batch processing
- SQLx's idle connection reaper doesn't aggressively close connections above `min_connections`
- Connections were not being released back to PostgreSQL for web app usage

### Solution
- Changed default `DB_MIN_CONNECTIONS` from 5 to 0
- Pool can now shrink to zero connections when idle
- Connections are created on-demand when processing starts
- After 60 seconds of idle time, all connections will close

### Technical Changes
- `crates/pro-db/src/connection.rs`: Changed min_connections default from 5 to 0
- `installer/WriteConfig.vbs`: Updated `DB_MIN_CONNECTIONS=0`
- `.env`: Updated `DB_MIN_CONNECTIONS=0`

### Notes
- Set `DB_MIN_CONNECTIONS` > 0 if you want to keep warm connections for faster startup
- No performance impact during batch processing (connections still scale to max_connections)

## [2.12.73.55] - 2025-01-20

### Performance - SYNC EXECUTION PATH FOR CPU-ONLY RULES
Enabled synchronous execution for all CPU-only template rules, eliminating async overhead.

### Problem
- RuleEngine has `all_sync_capable` flag that enables sync execution path
- Only `CompositeRule` had `requires_db_access() -> false` implemented
- Other CPU-only templates (Threshold, MissingField, FieldPattern, CrossField) used default `true`
- If ANY rule returns `requires_db_access() -> true`, ALL rules use async path
- This forced unnecessary async overhead for rules that do pure CPU evaluation

### Solution
Added sync execution support to all CPU-only template rules:
- `ThresholdRule` - numeric threshold comparisons
- `MissingFieldRule` - required field validation
- `FieldPatternRule` - regex pattern matching
- `CrossFieldRule` - cross-field comparisons

Each now implements:
```rust
fn requires_db_access(&self) -> bool { false }
fn execute_sync(&self, ctx: &RuleExecutionContext) -> Result<Option<RuleResult>>
```

### Skipped
- `DuplicateRule` - genuinely requires DB access (checks for duplicates via SQL queries)

### Expected Impact
- If ruleset contains NO DuplicateRules, entire execution path becomes synchronous
- Eliminates async runtime overhead for CPU-bound rule evaluation
- Potential +50-100 rec/sec improvement depending on rule composition

### Technical Changes
- `crates/pro-rules/src/templates/threshold_rule.rs`: Added sync support
- `crates/pro-rules/src/templates/missing_field_rule.rs`: Added sync support
- `crates/pro-rules/src/templates/field_pattern_rule.rs`: Added sync support
- `crates/pro-rules/src/templates/cross_field_rule.rs`: Added sync support

## [2.12.73.54] - 2025-01-20

### Fix - CONNECTION POOL IDLE TIMEOUT
Reduced idle timeout from 600s to 60s so connections are released faster after batch processing.

### Problem
- After batch processing, all 75 connections remained open for 10 minutes
- Web app could not acquire connections during this window
- `pg_stat_activity` showed 75 idle connections blocking new requests

### Solution
- Added `DB_IDLE_TIMEOUT` environment variable (default: 60 seconds)
- Idle connections now close within 1 minute after processing completes
- No impact on processing performance (only affects idle connections)

### Technical Changes
- `crates/pro-db/src/connection.rs`: Made idle_timeout configurable via `DB_IDLE_TIMEOUT` env var
- `installer/WriteConfig.vbs`: Added `DB_IDLE_TIMEOUT=60` to installer config
- `.env.example`: Updated documentation for connection pool settings

### New Environment Variable
| Variable | Default | Description |
|----------|---------|-------------|
| `DB_IDLE_TIMEOUT` | 60 | Seconds before idle connections are closed |

## [2.12.73.53] - 2025-01-20

### Performance - ELIMINATED RULE ENGINE LOCK CONTENTION
Removed RwLock from rule_engine to eliminate lock contention in parallel encounter processing.

### Problem
- With 24 parallel encounters, each task was acquiring `rule_engine.read().await`
- This caused 24 concurrent lock acquisitions per batch
- Lock contention serialized what should be parallel work
- Contributing to avg 228 rec/sec (target: 333 rec/sec)

### Solution
- Changed `rule_engine: Arc<RwLock<RuleEngine>>` to `rule_engine: Arc<RuleEngine>`
- Rules are loaded once at startup and never modified during runtime
- Write lock was never used - only read locks were acquired
- Direct Arc access eliminates ALL lock acquisition overhead

### Expected Impact
- Estimated +100-150 rec/sec improvement
- Zero lock contention for rule execution
- No functional changes - rules execute identically

### Technical Changes
- `crates/pro-service/src/claims_processor.rs`:
  - Line 96: Changed type from `Arc<RwLock<RuleEngine>>` to `Arc<RuleEngine>`
  - Line 159: Removed `RwLock::new()` wrapper
  - Line 3880: Removed `.read().await` - now direct Arc reference

## [2.12.73.52] - 2025-01-16

### Installer - UPDATED DEFAULT CONFIGURATION
Updated installer to include Stage 2 performance tuning variables in new installations.

### Changes
- `installer/WriteConfig.vbs`: Updated defaults and added new env vars
  - STAGE2_WORKER_COUNT=12 (was 8)
  - BATCH_SIZE=250 (was 750)
  - MAX_CONCURRENT_ENCOUNTERS=24 (new)
  - DB_MAX_CONNECTIONS=75 (new)
  - DB_MIN_CONNECTIONS=5 (new)
- `.env.example`: Added Stage 2 tuning section with documentation
- `.env`: Added Stage 2 tuning variables

### Note for Existing Installations
The installer preserves existing `.env` files during upgrade. To get new variables,
manually add to your `.env`:
```
STAGE2_WORKER_COUNT=12
BATCH_SIZE=250
MAX_CONCURRENT_ENCOUNTERS=24
DB_MAX_CONNECTIONS=75
DB_MIN_CONNECTIONS=5
```

## [2.12.73.51] - 2025-01-16

### Feature - CONFIGURABLE PERFORMANCE PARAMETERS VIA ENV VARS
All performance tuning parameters are now configurable via environment variables.
No need to rebuild the MSI installer to adjust these settings.

### New Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_CONCURRENT_ENCOUNTERS` | 24 | Concurrent encounters per batch |
| `DB_MAX_CONNECTIONS` | 75 | Database connection pool size |
| `DB_MIN_CONNECTIONS` | 5 | Minimum idle connections |
| `STAGE2_WORKER_COUNT` | 12 | Number of Stage 2 workers |
| `BATCH_SIZE` | 250 | Claims per batch |

### Technical Changes
- `crates/pro-service/src/claims_processor.rs`: MAX_CONCURRENT_ENCOUNTERS now reads from env var
- `crates/pro-db/src/connection.rs`: max_connections and min_connections now read from env vars

### Example .env Configuration
```
# Stage 2 Performance Tuning
STAGE2_WORKER_COUNT=12
BATCH_SIZE=250
MAX_CONCURRENT_ENCOUNTERS=24
DB_MAX_CONNECTIONS=75
DB_MIN_CONNECTIONS=5
```

## [2.12.73.50] - 2025-01-16

### Performance - INCREASED CONCURRENT ENCOUNTERS TO 24
- **Problem**: Stage 2 avg 213 rec/sec, target 333 rec/sec (64% of target)
- **Analysis**: With 75 DB connections now available, can increase per-batch parallelism
- **Solution**: Increase MAX_CONCURRENT_ENCOUNTERS from 16 to 24

### Performance Analysis (v2.12.73.49)
| Metric | Value | % of Target |
|--------|-------|-------------|
| Average | 213.00 rec/sec | 64% |
| Peak | 591.00 rec/sec | 177% |

### Technical Changes
- `crates/pro-service/src/claims_processor.rs`: MAX_CONCURRENT_ENCOUNTERS 16 -> 24

### Configuration Summary
| Parameter | Value | Env Var |
|-----------|-------|---------|
| Workers | 12 | STAGE2_WORKER_COUNT |
| Batch Size | 250 | BATCH_SIZE |
| Concurrent Encounters | 24 | (constant) |
| DB Pool | 75 | (hardcoded) |

## [2.12.73.49] - 2025-01-16

### Performance - REVERTED WORKERS + INCREASED DB POOL
- **Problem**: v2.12.73.48 (16 workers) showed decreased performance vs v2.12.73.47 (12 workers)
- **Analysis**: 16 workers caused DB connection pool contention (50 connections)
- **Solution**: Revert to 12 workers (optimal) + increase DB pool from 50 to 75 connections

### Performance Analysis (v2.12.73.48 - 16 workers was worse)
| Metric | v2.12.73.47 (12 workers) | v2.12.73.48 (16 workers) |
|--------|--------------------------|--------------------------|
| Average | 204.99 rec/sec | 200.94 rec/sec (-2%) |
| Peak | 619.47 rec/sec | 599.06 rec/sec (-3%) |

### Technical Changes
- `crates/pro-service/src/service.rs`: STAGE2_WORKER_COUNT 16 -> 12 (reverted)
- `crates/pro-service/src/main.rs`: STAGE2_WORKER_COUNT 16 -> 12 (reverted)
- `crates/pro-db/src/connection.rs`: max_connections 50 -> 75

### Configuration Summary
| Parameter | Value | Env Var |
|-----------|-------|---------|
| Workers | 12 | STAGE2_WORKER_COUNT |
| Batch Size | 250 | BATCH_SIZE |
| Concurrent Encounters | 16 | (constant) |
| DB Pool | 75 | (hardcoded) |

## [2.12.73.48] - 2025-01-16

### Performance - INCREASED WORKER COUNT TO 16
- **Problem**: Stage 2 avg 205 rec/sec, target 333 rec/sec (62% of target)
- **Root Cause**: Still have capacity for more parallel batch processing
- **Solution**: Increase default worker count from 12 to 16

### Performance Analysis (v2.12.73.47)
| Metric | Value | % of Target |
|--------|-------|-------------|
| Average | 204.99 rec/sec | 62% |
| Peak | 619.47 rec/sec | 186% |
| Gap Ratio | 3.0x | Room for improvement |

### Technical Changes
- `crates/pro-service/src/service.rs`: Default STAGE2_WORKER_COUNT 12 -> 16
- `crates/pro-service/src/main.rs`: Default STAGE2_WORKER_COUNT 12 -> 16

### Configuration Summary
| Parameter | Value | Env Var |
|-----------|-------|---------|
| Workers | 16 | STAGE2_WORKER_COUNT |
| Batch Size | 250 | BATCH_SIZE |
| Concurrent Encounters | 16 | (constant) |

## [2.12.73.47] - 2025-01-16

### Performance - INCREASED WORKER COUNT
- **Problem**: Stage 2 avg 191 rec/sec, target 333 rec/sec (58% of target)
- **Root Cause**: With smaller batches (250), need more workers to maintain throughput
- **Solution**: Increase default worker count from 8 to 12

### Performance Analysis (v2.12.73.46)
| Metric | Value | % of Target |
|--------|-------|-------------|
| Average | 191.61 rec/sec | 58% |
| Peak | 555.56 rec/sec | 167% |
| Gap Ratio | 2.9x | Improved from 3.7x |

### Why More Workers Help
- More workers = more batches processed in parallel
- Smaller batches (250) complete faster, need more workers to keep pipeline full
- DB pool (50 connections) can support: 12 workers * ~4 connections = 48 connections
- FIFO maintained by SequentialCompletionManager regardless of worker count

### Technical Changes
- `crates/pro-service/src/service.rs`: Default STAGE2_WORKER_COUNT 8 -> 12
- `crates/pro-service/src/main.rs`: Default STAGE2_WORKER_COUNT 8 -> 12

### Configuration Summary
| Parameter | Value | Env Var |
|-----------|-------|---------|
| Workers | 12 | STAGE2_WORKER_COUNT |
| Batch Size | 250 | BATCH_SIZE |
| Concurrent Encounters | 16 | (constant) |

## [2.12.73.46] - 2025-01-16

### Performance - REDUCED BATCH SIZE FOR CONSISTENT THROUGHPUT
- **Problem**: Stage 2 avg 186 rec/sec, peak 687 rec/sec - large 3.7x gap indicates batch variance
- **Root Cause**: Large batches (750 claims) have highly variable encounter counts/complexity
- **Solution**: Reduce batch size from 750 to 250 for more consistent processing times

### Performance Analysis (v2.12.73.45)
| Metric | Value | Issue |
|--------|-------|-------|
| Average | 186.08 rec/sec | 56% of target |
| Peak | 686.57 rec/sec | 206% of target |
| Gap Ratio | 3.7x | Batch variance too high |

### Why Smaller Batches Help
- Smaller batches = more consistent processing times
- Faster completion of "easy" batches (few encounters)
- More granular FIFO ordering (batch-level)
- Better utilization of 16 concurrent encounters per batch

### Technical Changes
- `crates/pro-service/src/service.rs`: Default BATCH_SIZE 750 -> 250
- `crates/pro-service/src/main.rs`: Default BATCH_SIZE 750 -> 250

### Configuration
- Batch size configurable via `BATCH_SIZE` env var (default: 250)
- Concurrent encounters per batch: 16 (unchanged)

## [2.12.73.45] - 2025-01-16

### Performance - INCREASED PARALLEL CONCURRENCY
- **Problem**: Stage 2 averaging 189.68 rec/sec, target is 333 rec/sec
- **Peak Performance**: 545.95 rec/sec proves system CAN achieve target
- **Root Cause**: Conservative concurrency limit (8) leaving capacity unused

### Solution
- Increased `MAX_CONCURRENT_ENCOUNTERS` from 8 to 16
- With 50 DB connections and 2 workers: 16 * 2 = 32 connections max (well under limit)

### Performance Analysis
| Metric | Value | % of Target |
|--------|-------|-------------|
| Previous Average | 189.68 rec/sec | 57% |
| Previous Peak | 545.95 rec/sec | 164% |
| Target | 333 rec/sec | 100% |

### Expected Impact
Doubling concurrency from 8 to 16 should bring average throughput closer to peak performance.

### Technical Changes
- `crates/pro-service/src/claims_processor.rs`:
  - Changed `MAX_CONCURRENT_ENCOUNTERS` from 8 to 16

## [2.12.73.44] - 2025-01-16

### Performance - PARALLEL ENCOUNTER PROCESSING
- **Problem**: Stage 2 at 50 rec/sec, target is 333 rec/sec (10K claims / 30 seconds)
- **Root Cause**: Sequential encounter processing within each batch
- **Solution**: Process encounters in PARALLEL within each batch

### FIFO Safety Analysis
This change is FIFO-safe because:
1. FIFO is enforced at **BATCH level** by `SequentialCompletionManager`
2. Within a batch, encounters were already in arbitrary order (HashMap iteration)
3. The `BatchResult` is reported AFTER all encounters complete
4. `CompletionManager` commits batches in strict sequence order

### Implementation Details
- Added `MAX_CONCURRENT_ENCOUNTERS = 8` constant to limit parallelism
- Uses `tokio::sync::Semaphore` to control concurrent encounter processing
- Thread-safe facility cache using `Arc<RwLock<HashMap>>`
- Uses `futures::future::join_all()` to wait for all encounters before reporting batch result

### Technical Changes
- `crates/pro-service/src/claims_processor.rs`:
  - Added `MAX_CONCURRENT_ENCOUNTERS` constant (8 concurrent)
  - Added `process_encounter_with_service_lines_parallel()` method
  - Converted sequential `for` loop to parallel `tokio::spawn()` with semaphore
  - Results aggregated after all parallel tasks complete

### Expected Performance Impact
| Scenario | Speed | Notes |
|----------|-------|-------|
| Before (sequential) | 50 rec/sec | 1 encounter at a time |
| After (8x parallel) | 200-400 rec/sec | 8 concurrent encounters |
| Target | 333 rec/sec | 10K claims / 30 sec |

### Configuration
- DB connection pool: 50 connections (unchanged)
- Concurrent encounters per worker: 8 (configurable via `MAX_CONCURRENT_ENCOUNTERS`)

## [2.12.73.43] - 2025-01-16

### Database - SERVICE_LINE_FLAG QUERY PERFORMANCE
- **Problem**: `SELECT * FROM claims.service_line_flag` taking ~20 minutes for 900K rows
- **Root Cause**: TEXT columns (`flag_reason`, `resolution_note`) require TOAST decompression for every row

### Solution: Migration 074
Added migration `074_service_line_flag_performance_tuning.sql`:

1. **Fast List View** (`v_service_line_flag_list`)
   - Excludes TEXT columns for dashboard/list queries
   - Returns in seconds instead of minutes

2. **Detail View** (`v_service_line_flag_detail`)
   - Includes TEXT columns and JOINs
   - For single flag lookup only

3. **BRIN Index** (`idx_service_line_flag_created_brin`)
   - Block Range Index for time-range queries
   - 100-1000x smaller than B-tree, very fast for append-only tables

4. **Covering Index** (`idx_service_line_flag_pk_covering`)
   - For single flag lookups by flag_id
   - Includes common columns to avoid heap access

5. **TOAST Tuning**
   - Increased `toast_tuple_target` to 4096 bytes
   - Keeps smaller flag_reason values inline

### Usage
```sql
-- ❌ SLOW (20+ minutes):
SELECT * FROM claims.service_line_flag;

-- ✅ FAST (for lists):
SELECT * FROM claims.v_service_line_flag_list
WHERE flag_status = 'OPEN'
ORDER BY created_at DESC
LIMIT 1000;

-- ✅ FAST (for single flag):
SELECT * FROM claims.v_service_line_flag_detail
WHERE flag_id = 12345;
```

## [2.12.73.42] - 2025-01-16

### Performance - CPT INDEXING ENABLED
- **Problem**: Stage 2 at 12 rec/sec, evaluating 537 rules × 3 service lines = 1,611 evaluations per claim
- **Solution**: Re-enabled CPT indexing for rule execution

### Analysis of home_fixed.txt Rules
| Metric | Count |
|--------|-------|
| Total rules | 537 |
| Rules WITH cpt_in (indexable) | 492 (91.6%) |
| Rules WITHOUT cpt_in (universal) | 45 (8.4%) |

### How CPT Indexing Works
- **Before**: All 537 rules execute for every service line
- **After**: Only rules matching the service line's CPT code execute
  - 45 universal rules (no cpt_in) always run
  - ~50-70 CPT-specific rules run per service line (varies by CPT code)
  - **Expected reduction: ~80% fewer rule evaluations**

### Expected Performance Impact
| Scenario | Evaluations/Claim | Est. Time |
|----------|-------------------|-----------|
| Without indexing | 1,611 | ~12 rec/sec |
| With indexing | ~315 | 50-60+ rec/sec |

### Technical Details
- `crates/pro-service/src/claims_processor.rs`:
  - Changed from `execute_all()` to `execute_all_indexed()`
  - CPT index maps CPT codes → applicable rule indices
  - Universal rules (no cpt_in) always execute

## [2.12.73.41] - 2025-01-16

### Performance - STRING ALLOCATION OPTIMIZATION
- **Problem**: Stage 2 still at 12 rec/sec with 50+ rules triggering per claim
- **Root Cause**: Excessive string allocations when rules trigger
  - Each triggered rule was building detailed condition descriptions
  - Flag collection was formatting strings multiple times
  - Per-encounter logging adding overhead

### Fixes Applied
1. **Simplified Rule Description**
   - Changed from detailed "X conditions matched (cond1; cond2; ...)" to just rule name
   - Eliminates: `description()` call per condition, `join()`, `format!()`
   - Saves ~4-6 string allocations per triggered rule

2. **Reduced Flag Collection Allocations**
   - Use `issue_code` directly instead of cloning
   - Simplified flag_reason to use details directly
   - Removed debug logging from flag collection loop

3. **Downgraded Per-Encounter Logging**
   - Changed "RULES: Executing X rules for encounter Y" from `info!()` to `debug!()`
   - Saves one log operation per encounter

### Technical Details
- `crates/pro-rules/src/templates/composite_rule.rs`:
  - Simplified `evaluate()` to just clone rule_name for description
  - Removed condition description collection and joining

- `crates/pro-service/src/claims_processor.rs`:
  - Simplified flag collection to minimize string operations
  - Downgraded per-encounter log to debug level

## [2.12.73.40] - 2025-01-16

### Performance - MAJOR RULE EXECUTION OPTIMIZATION
- **Problem**: Stage 2 still slow at 11 rec/sec after v2.12.73.39 (target: 300+ rec/sec)
- **Root Causes Identified**:
  1. Async overhead in `execute_all()` even for sync-capable rules
  2. Repeated `to_uppercase()` allocations in hot loop (537 rules x 3 service lines = 1,611 calls per claim)

### Fixes Applied
1. **Fully Synchronous `execute_all()` Path**
   - Added `execute_all_sync()` method that bypasses ALL async overhead
   - `execute_all()` now auto-detects when all rules are sync-capable and uses sync path
   - Eliminates tokio task switching and state machine overhead for CPU-only rules

2. **Pre-computed Uppercase Values**
   - Added `finalize()` method to `RuleExecutionContext`
   - Pre-computes uppercase for: procedure_code, diagnosis_codes, place_of_service, modifiers
   - Called ONCE before rule execution instead of thousands of times in hot loop
   - Eliminates ~1,611 string allocations per claim

3. **Inlined Hot Path Functions**
   - Added `#[inline]` to `evaluate()` and `finalize()` methods
   - Reduces function call overhead in tight loops

### Technical Details
- `crates/pro-rules/src/rule_engine.rs`:
  - Added `execute_all_sync()` for zero-async-overhead execution
  - `execute_all()` auto-routes to sync path when `all_sync_capable` is true
  - Added `procedure_code_upper`, `diagnosis_codes_upper`, `place_of_service_upper`, `modifiers_upper` fields
  - Added `finalize()` method to pre-compute uppercase values

- `crates/pro-rules/src/templates/composite_rule.rs`:
  - Updated condition evaluation to use pre-computed uppercase values
  - Added `#[inline]` attribute to `evaluate()` method

- `crates/pro-service/src/claims_processor.rs`:
  - Calls `ctx.finalize()` before rule execution

## [2.12.73.39] - 2025-01-16

### Performance - RULE EXECUTION OPTIMIZATION
- **Problem**: Stage 2 processing at 4 rec/sec with 500+ rules (target: 300+ rec/sec)
- **Root Causes Identified**:
  1. INFO-level logging in hot path - millions of log writes with high rule counts
  2. Double condition evaluation - re-evaluating conditions when building flag descriptions

### Fixes Applied
1. **Downgraded Hot Path Logging to DEBUG Level**
   - `claims_processor.rs`: Changed per-service-line and per-flag logging from `info!()` to `debug!()`
   - Eliminates millions of unnecessary log operations
   - Summary logging (flags inserted per encounter) remains at INFO level

2. **Eliminated Double Condition Evaluation in CompositeRule**
   - `composite_rule.rs`: Captures matched condition indices during initial evaluation
   - Reuses captured indices for description building (no re-evaluation)
   - ~2x faster for rules that trigger

### Technical Details
- `crates/pro-service/src/claims_processor.rs`:
  - Lines 3769-3786: Changed `info!()` to `debug!()` for rule execution logging
  - Line 3798: Changed `info!()` to `debug!()` for "no flags collected" message

- `crates/pro-rules/src/templates/composite_rule.rs`:
  - Refactored `evaluate()` method to track matched indices during evaluation
  - Builds description from indices instead of re-filtering conditions

## [2.12.73.38] - 2025-01-16

### Fixed - FLAGS NOT INSERTING INTO service_line_flag
- **Root Cause**: AUTO_DEFER_RULE_THRESHOLD logic (added in v2.12.73.30) was auto-deferring rules when count >= 100
  - With 537 rules and DEFER_RULES_EXECUTION not set, rules were being queued to `rules_processing_queue`
  - No background worker exists to process that queue, so flags never got inserted
  - This broke the behavior that was working in v2.12.73.27
- **Solution**: Removed AUTO_DEFER_RULE_THRESHOLD auto-detection logic
  - Rules now execute inline by default (DEFER_RULES_EXECUTION defaults to false)
  - Explicit `DEFER_RULES_EXECUTION=true` still available if user wants deferred processing
  - Matches v2.12.73.27 behavior where rules executed inline regardless of count

### Technical Details
- `crates/pro-service/src/claims_processor.rs`:
  - Removed `AUTO_DEFER_RULE_THRESHOLD` constant
  - Removed auto-defer logic that checked rule count against threshold
  - `defer_rules` now simply reads `DEFER_RULES_EXECUTION` env var (defaults to false)

## [2.12.73.37] - 2025-01-16

### Fixed - RULES NOT TRIGGERING
- **Critical Bug Fix**: Reverted from `execute_all_indexed()` to `execute_all()` for rule execution
  - Problem: `execute_all_indexed()` (introduced in v2.12.73.27) was skipping rules due to CPT indexing
  - Rules with `cpt_in` conditions were only executing for exact CPT matches in the index
  - If a claim's CPT code wasn't in any rule's `cpt_in` list, NO rules would execute
  - Solution: Revert to `execute_all()` which iterates all rules directly without CPT filtering
  - Trade-off: Slightly slower (executes all 537 rules per service line) but rules actually trigger

### Technical Details
- `crates/pro-service/src/claims_processor.rs`:
  - Changed `rule_engine.execute_all_indexed(&ctx)` to `rule_engine.execute_all(&ctx)`
  - All rules now execute for every service line (correct behavior)

## [2.12.73.36] - 2025-01-16

### Diagnostics - INFO-LEVEL LOGGING FOR RULES ENGINE
- **Enhanced Logging**: Added info-level logging to rule execution path for debugging flag insertion issues
  - Logs rule count, encounter ID, and service line count at start of rule execution
  - Logs triggered rule count per service line with CPT code
  - Logs each flag collected with issue_code and severity
  - Logs when no flags are collected for an encounter
  - Logs successful flag insertions
  - Warns when flags are collected but 0 inserted (issue_code mismatch)

### Technical Details
- `crates/pro-service/src/claims_processor.rs`:
  - Changed debug! to info! for key rule execution logging
  - Added per-service-line trigger counts
  - Added individual flag collection logging

## [2.12.73.34] - 2025-01-16

### Performance - FULLY SYNCHRONOUS RULE EXECUTION PATH
- **Critical Performance Fix**: Added zero-async-overhead execution path for CPU-only rules
  - Problem: Even with sync-capable rules, `execute_all_indexed` was async and had tokio overhead
  - Each rule execution involved async state machine, task switching overhead
  - With 537 rules × 3 service lines = 1,611 async operations per encounter

- **Solution: `execute_all_indexed_sync()` method**
  - New fully synchronous execution path bypasses all async infrastructure
  - `all_sync_capable` flag computed once at index build time (not per-call)
  - Pre-allocated result vectors to avoid reallocations in hot loop
  - Minimal branching in tight execution loop
  - When all rules are CPU-only (like COMPOSITE), uses sync path automatically

- **Debug Logging for Flag Investigation**
  - Added logging when flags are collected but INSERT returns 0 rows
  - Helps identify issue_code mismatches with flag_issue table

- **Expected Performance**:
  - Eliminates ~50-100 microseconds async overhead per rule
  - With 537 rules: saves 27-54ms per service line
  - Target: 150-300 rec/sec with inline execution

### Technical Details
- `crates/pro-rules/src/rule_engine.rs`:
  - Added `all_sync_capable: bool` field to RuleEngine
  - `build_cpt_index()` computes sync capability once
  - `execute_all_indexed()` calls sync path when all rules are CPU-only
  - `execute_all_indexed_sync()` - tight loop, no async, no fallbacks

## [2.12.73.31] - 2025-01-16

### Performance - O(1) HASHSET LOOKUPS + SHORT-CIRCUIT EVALUATION
- **Critical Performance Fix**: Optimized COMPOSITE rule evaluation from O(N*M) to O(N*1)
  - Problem: DxIn condition used nested iteration: `codes.iter().any(|c| c.eq_ignore_ascii_case(dx))`
  - For 10 diagnosis codes × 50 allowed codes = 500 string comparisons per condition
  - With 537 rules × 3 conditions avg = 805,500 string comparisons per service line!

- **Solution 1: FxHashSet for O(1) lookups**
  - Replaced `Vec<String>` with `FxHashSet<String>` for all code lookups
  - CptIn, DxIn, PosIn, ModifierIn, ModifierNotIn now use O(1) HashSet.contains()
  - All codes normalized to UPPERCASE at compile time (no runtime case conversion)
  - DxIn complexity: O(N*M) → O(N) where N = diagnosis codes per claim

- **Solution 2: Short-circuit evaluation for AND/OR**
  - AND conditions: Stop evaluating on first FALSE (most rules fail early)
  - OR conditions: Stop evaluating on first TRUE
  - Previously evaluated ALL conditions before checking result
  - Most rules with 3+ AND conditions now exit after 1-2 checks

- **Expected Performance**:
  - COMPOSITE rule evaluation: **10-50x faster** per rule
  - Combined with CPT indexing: May enable inline execution for 500+ rules
  - Target: Inline execution at 200-400 rec/sec (vs 5-15 rec/sec before)

### Technical Details
- `crates/pro-rules/src/templates/composite_rule.rs`:
  - `CompiledCondition` enum uses `FxHashSet<String>` instead of `Vec<String>`
  - `compile()` converts codes to uppercase and collects into HashSet
  - `evaluate()` uses `codes.contains(&code.to_uppercase())` for O(1) lookup
  - `CompositeRule::evaluate()` uses short-circuit `.all()` and `.any()` iterators

## [2.12.73.30] - 2025-01-16

### Performance - RESTORE AUTO-DEFER FOR HIGH RULE COUNTS
- **Critical Fix**: Restored auto-defer threshold that was incorrectly removed in v2.12.73.27
  - Problem: Version 2.12.73.27 removed auto-defer assuming CPT indexing would solve performance
  - Reality: With 537 universal rules (no `cpt_in` filter), CPT indexing doesn't help
  - All 537 rules still execute on every service line = 1,611 evaluations per encounter
  - This caused throughput to drop from 300-500 rec/sec to 5-15 rec/sec
  - Solution: Restored `AUTO_DEFER_RULE_THRESHOLD = 100`
  - When rule count >= 100 and `DEFER_RULES_EXECUTION` not set, auto-defers to background
  - Expected improvement: **20-40x throughput** (from 15 rec/sec back to 300-500 rec/sec)

### Key Insight
- CPT indexing only helps when rules have `cpt_in` conditions
- Universal rules (rules without `cpt_in`) execute on EVERY service line regardless of CPT code
- With 537 universal rules, inline execution is fundamentally too slow for the 10K/15-30sec target

## [2.12.73.29] - 2025-01-16

### Performance - ELIMINATE MODIFIER SELECT QUERY
- **Critical Performance Fix**: Eliminated expensive SELECT DISTINCT ... LATERAL query for modifiers
  - Problem: After inserting service lines, a SELECT query with LATERAL was reading them back (~40-50ms)
  - Root cause: `insert_encounter_procedure_modifiers()` queried service_line table to get modifiers
  - Solution: Collect modifiers from `ServiceLineRuleContext` (already in memory from import)
  - New function `insert_encounter_procedure_modifiers_fast()` uses pre-collected data
  - Eliminates 1 SELECT + 1 INSERT per encounter, replaced with just 1 INSERT
  - Expected improvement: **3-5x throughput increase** (from 14 rec/sec to 42-70 rec/sec)

## [2.12.73.28] - 2025-01-15

### Performance - BATCH PROVIDER INSERTION
- **Critical Performance Fix**: Batch insert ALL providers in `prewarm_provider_cache()`
  - Problem: 4+ sequential `ensure_provider_exists()` calls per encounter (each a DB round-trip)
  - For 10K claims × 2.5 service lines × 4 providers = ~100,000 sequential INSERT operations
  - Solution: Batch INSERT all new providers in a single query using UNNEST
  - All providers (existing + new) are now cached BEFORE encounter processing begins
  - `ensure_provider_exists()` calls become instant cache hits (no DB query)
  - Expected improvement: **3-5x throughput increase** (from 13 rec/sec to 40-65 rec/sec)

## [2.12.73.27] - 2025-01-15

### Performance - CPT INDEXING + SYNC EXECUTION
- **Critical Performance Fix**: Switched from `execute_all()` to `execute_all_indexed()` in claims processor
  - Problem: All 600+ rules were being evaluated for every service line (O(n) where n = total rules)
  - Solution: Use CPT code index to filter rules (O(k) where k = rules applicable to this CPT + universal rules)
  - Rules with `cpt_in` conditions are indexed at load time for O(1) lookup
  - Only rules matching the service line's CPT code are executed
  - Expected improvement: 5-20x speedup depending on how many rules have `cpt_in` filters

- **Sync Execution for COMPOSITE Rules**: COMPOSITE rules now use synchronous execution path
  - COMPOSITE rules are CPU-only (no database access required)
  - `requires_db_access()` returns false, enabling `execute_sync()` path
  - Avoids async/await overhead for pure CPU rule evaluation
  - Expected improvement: 2-5x speedup for COMPOSITE rule execution

- **Performance Warning for Universal Rules**: Added warning when >50 universal rules loaded
  - Universal rules (no `cpt_in` filter) run on every service line
  - Warning helps identify rules that should have CPT filters added
  - Target: <50 universal rules for optimal throughput

### Changed
- **Removed Auto-Defer Threshold**: Inline execution is now default even with 600+ rules
  - CPT indexing + sync execution makes inline execution viable for high rule counts
  - `DEFER_RULES_EXECUTION` still available for explicit deferral if needed
  - Previous auto-defer at 100 rules was a workaround, now properly fixed

## [2.12.73.25] - 2025-01-15

### Performance - AUTO-DEFER RULES
- **Smart Auto-Detection for Deferred Rules**: System now automatically defers rules when count >= 100
  - Problem: Users with 500+ rules were experiencing 5 rec/sec throughput without knowing to set `DEFER_RULES_EXECUTION=true`
  - Solution: Auto-detect high rule counts and automatically enable deferred mode
  - Threshold: 100+ rules triggers auto-deferral (configurable via code constant)
  - Override: Users can still force inline execution with `DEFER_RULES_EXECUTION=false`
  - Logs clear warnings when auto-deferring: "AUTO-DEFERRING rules execution - X rules exceeds threshold of 100"

### Configuration
- `DEFER_RULES_EXECUTION` behavior updated:
  - **Not set (recommended)**: Auto-defers if rule count >= 100
  - `true`: Always defer rules to background processing
  - `false`: Always execute rules inline (slow with many rules)

### Documentation
- Updated `.env.example` with comprehensive rules engine configuration section
- Added `ENABLE_DATABASE_RULES`, `DEFER_RULES_EXECUTION`, and `RULE_ENCRYPTION_KEY` documentation

## [2.12.73.24] - 2025-01-15

### Performance - CRITICAL
- **Deferred Rules Execution Mode**: Added `DEFER_RULES_EXECUTION` environment variable for high-throughput import
  - Root cause: Sequential rule execution consuming 200-300ms per claim (537 rules x 3 service lines = 1,611 rule evaluations per encounter)
  - With 10K claims target in 30 seconds, inline rule execution is the bottleneck (theoretical max: 3-5 claims/sec)
  - Solution: When `DEFER_RULES_EXECUTION=true`, rules are queued for background processing instead of inline execution
  - Encounters are queued to `staging.rules_processing_queue` for async rule processing
  - Expected throughput improvement: **50-100x** when enabled (from ~5 claims/sec to 300-500 claims/sec)
  - Trade-off: Flags appear after import completes rather than immediately

### Added
- **Rules Processing Queue** (migration 073):
  - `staging.rules_processing_queue` table for deferred rule execution
  - `staging.enqueue_for_rules_processing()` - queue encounter for background processing
  - `staging.acquire_rules_processing_batch()` - FIFO batch acquisition with SKIP LOCKED
  - `staging.complete_rules_processing()` - mark completed with flag count
  - `staging.fail_rules_processing()` - mark failed with error message
  - `staging.recover_stale_rules_processing()` - recover stuck items (>5 min)

### Configuration
- New environment variables:
  - `DEFER_RULES_EXECUTION=true` - Enable deferred rules for maximum import throughput
  - `DEFER_RULES_EXECUTION=false` (default) - Inline rules execution (slower but immediate flags)

## [2.12.73.23] - 2025-01-15

### Performance
- **Batch Provider Cache Pre-warming**: Added `prewarm_provider_cache()` optimization for Stage 2 processing
  - Root cause: Each service line was doing 4 sequential database queries for provider lookups (rendering, ordering, supervising, referring)
  - With 5 service lines per encounter, that's up to 20 DB round-trips per encounter
  - Solution: Before processing service lines, collect all unique NPIs and query existing providers in ONE batch query
  - Pre-populates provider cache so subsequent `ensure_provider_exists()` calls are instant cache hits
  - Expected throughput improvement: 2-4x for encounters with multiple service lines
  - Impact: Reduces DB round-trips from O(service_lines * 4) to O(1) per encounter for existing providers

## [2.12.73.22] - 2025-01-15

### Fixed
- **Baseline SQL Syntax Error Blocking Views**: Removed invalid `COMMENT ON DATABASE current_database()` statement
  - Root cause: This statement is syntactically invalid (cannot use function call in DDL statement)
  - This caused the baseline execution to fail, preventing all subsequent statements from executing
  - Migration 072 views (`v_processing_summary`, `v_stage2_throughput`, etc.) were never created
  - Removed the invalid statement from baseline SQL

## [2.12.73.21] - 2025-01-15

### Fixed
- **Baseline Migration Tracking Conflict**: Removed incorrect INSERT statements from baseline SQL
  - Root cause: Baseline SQL had hardcoded INSERT statements for migrations 031-072 with wrong filenames
  - These conflicted with the programmatic registration in `apply_baseline()` which uses correct embedded migration names
  - Removed the incorrect INSERT block (migrations 031-072) from baseline
  - Removed non-existent `011_create_schedule_tables.sql` from migration tracking
  - Migration tracking is now handled entirely by `apply_baseline()` in migration.rs which iterates through embedded migrations

## [2.12.73.20] - 2025-01-15

### Fixed
- **Fresh Install Missing Views**: Fixed `BASELINE_COVERS_THROUGH` constant not updated to 72
  - Fresh installs were missing migration 072 views (`v_processing_summary`, `v_stage2_throughput`, etc.)
  - Updated constant from 64 to 72 so baseline properly includes all migrations through 072

## [2.12.73.19] - 2025-01-15

### Performance
- **Sync Execution for Universal Rules**: Fixed slow throughput (3-25 records/sec) when all rules are universal
  - Root cause: `execute_all()` was only using sync execution when CPT index was populated
  - With 537 universal COMPOSITE rules (no `cpt_in` filter), CPT index was empty
  - Modified `execute_all()` to always use `execute_sync()` for rules that don't require database access
  - Expected 2-5x throughput improvement for universal rule workloads

- **CPT Index Logging Fix**: Changed CPT index statistics from `eprintln!` to `tracing::info!`
  - Now properly appears in service.log instead of stderr

### Added
- **Processing Metrics Rollup Views** (migration 072):
  - `staging.v_processing_metrics_hourly` - Hourly aggregated throughput metrics
  - `staging.v_processing_metrics_daily` - Daily aggregated throughput metrics
  - `staging.v_processing_summary` - Last 24 hours summary with success rates
  - `staging.v_stage2_throughput` - Hourly Stage 2 claims processing throughput

## [2.12.73.17] - 2025-01-15

### Performance
- **Service Line Flag Query Performance**: Fixed extremely slow queries on `claims.service_line_flag` (5+ minutes)
  - Added migration 071 with performance indexes for flag queries
  - Added covering index `idx_service_line_flag_view_lookup` for view JOINs
  - Added partial index `idx_service_line_flag_recent` for open flag dashboards
  - Added `idx_service_line_flag_status_lookup` for status filtering
  - Added `idx_service_line_encounter_lookup` for service_line to encounter JOINs

### Fixed
- **Duplicate Flag Prevention**: Added unique constraint and ON CONFLICT handling to prevent duplicate flags
  - Root cause: Reprocessing claims would create duplicate flags for the same service_line + issue combination
  - Added unique index `idx_service_line_flag_unique_open` on `(service_line_id, issue_id) WHERE flag_status = 'OPEN'`
  - Modified flag INSERT to use `ON CONFLICT DO NOTHING` - skips if flag already exists
  - Prevents flag table bloat from reprocessing the same claims multiple times

## [2.12.73.14] - 2025-01-15

### Fixed
- **Processing Metrics INSERT Fix**: Fixed critical bug preventing `staging.processing_metrics` records from being inserted
  - Root cause: Code was providing a value (`0i64`) for `metric_id` column which is defined as `GENERATED ALWAYS AS IDENTITY`
  - PostgreSQL rejects INSERT statements that explicitly provide values for GENERATED ALWAYS columns
  - Metrics logging was silently failing, leaving `processing_metrics` table empty
  - Fixed all 4 affected functions across 3 files:
    - `claims_processor.rs::log_processing_metric()`
    - `batch_manager.rs::log_processing_metric()`
    - `claims_importer.rs::log_processing_metric()`
    - `claims_importer.rs::log_processing_metric_with_stage()`
  - Processing throughput metrics will now be properly recorded for performance monitoring

## [2.12.73.12] - 2025-01-15

### Performance
- **CPT Code Index for Rule Engine**: Major performance optimization for large rule sets (500+ rules)
  - Added CPT-based rule index that maps procedure codes to applicable rules
  - Rules with `cpt_in` conditions are indexed by their CPT codes for O(1) lookup
  - Instead of executing all 560 rules per service line, now only executes rules that match the procedure code
  - Typical performance improvement: 95%+ reduction in rule evaluations per service line
  - Added `build_cpt_index()` method called automatically after loading rules from database

- **Synchronous Execution for COMPOSITE Rules**: Eliminated async overhead for CPU-only rules
  - COMPOSITE rules now implement `execute_sync()` for direct synchronous execution
  - Added `requires_db_access() = false` for COMPOSITE rules to use sync path
  - Avoids tokio async machinery overhead when no database access is needed

- **Rule Trait Extensions**: New trait methods for optimization
  - `requires_db_access()`: Returns whether rule needs database during execution
  - `applicable_cpt_codes()`: Returns CPT codes for index-based filtering
  - `execute_sync()`: Synchronous execution path for CPU-only rules

## [2.12.73.11] - 2025-01-15

### Fixed
- **Remove PHASE 8 Rule Logging**: Removed incomplete PHASE 8 rule execution logging code from `execute_all_with_cache()`
  - The code was calling non-existent `claims.log_rule_execution()` stored procedure
  - Spawned background tasks for every rule execution (triggered or not), causing potential connection pool exhaustion
  - While not directly called by claims_processor (which uses `execute_all`), the code path exists and could cause issues

## [2.12.73.10] - 2025-01-15

### Fixed
- **Rules Engine Flag INSERT JOIN Fix**: Fixed critical bug where flags were not being inserted into `claims.service_line_flag`
  - Root cause: `RuleResult.flag_type.code()` returned hardcoded enum codes like `"OTH-003"` but the database `claims.flag_issue.issue_code` contains custom values like `"TEST_99213_SA"`
  - The INSERT JOIN `ON fi.issue_code = fd.issue_code` was failing because codes didn't match
  - Added `issue_code: Option<String>` field to `RuleResult` struct
  - Added `with_issue_code()` builder method to `RuleResult`
  - Updated `RuleTemplate::instantiate()` signature to accept `issue_code` parameter
  - Updated loader to extract `issue_code` from database row and pass to all templates
  - Updated all template rules to store and return `issue_code` in their `execute()` methods:
    - `composite_rule.rs` (COMPOSITE template)
    - `threshold_rule.rs` (THRESHOLD template)
    - `duplicate_rule.rs` (DUPLICATE template)
    - `missing_field_rule.rs` (MISSING_FIELD template)
    - `field_pattern_rule.rs` (FIELD_PATTERN template)
    - `cross_field_rule.rs` (CROSS_FIELD template)
  - Updated `claims_processor.rs` to use `result.issue_code` when available, falling back to `result.flag_type.code()` for legacy rules
  - Added debug logging in loader to show `issue_code` during rule instantiation

## [2.12.73.5] - 2025-01-13

### Fixed
- **Rules Engine Flag Persistence**: Fixed critical bug where flags were not being inserted
  - Rule engine was trying to insert into non-existent `claims.flag` table
  - Now correctly inserts into `claims.encounter_flag` for encounter-level flags
  - Now correctly inserts into `claims.service_line_flag` for service line-level flags
  - Flags now link to `flag_issue` table via `issue_id` using the `issue_code` lookup
  - Added proper routing based on whether `service_line_id` or `encounter_id` is present

## [2.12.73.4] - 2025-01-13

### Fixed
- **Rule Converter GUI Performance**: Optimized for large datasets (500+ rules)
  - Fixed crash when clicking "Select All" with 553 rules
  - Added `set_redraw(false/true)` wrapper for bulk ListView operations
  - Removed redundant selection tracking (HashSet) - now queries ListView directly
  - Removed `on_selection_changed` event handler that was triggering 553 times
  - Added `bulk_operation` flag to prevent event handling during batch operations
  - Pre-allocate SQL string buffer for faster export
  - Safe string truncation at character boundaries (not byte boundaries)

## [2.12.73.3] - 2025-01-13

### Fixed
- **Rule Converter GUI Definition Column**: Added missing definition column to ListView
  - Now shows Rule Code, Rule Name, Description, and Definition columns
  - Definition is truncated to 80 chars in display (full text used for export)
- **Rule Converter GUI Export Crash**: Fixed application crash when clicking "Export Selected to SQL"
  - Added extensive debug logging to diagnose export issues
  - Added proper error handling for empty definitions
  - Shows warning in log and SQL output for rules that fail to convert
  - Shows success/error counts after export

## [2.12.73.1] - 2025-01-13

### Fixed
- **Rule Converter GUI MS SQL Connection**: Fixed connection to MS SQL Server
  - Now uses ADO.NET connection string with `Encrypt=false` and `TrustServerCertificate=true`
  - Added Username and Password input fields to GUI
  - Properly handles SQL Server Authentication

## [2.12.73.0] - 2025-01-13

### Added
- **Rule Converter GUI**: New GUI tool to convert legacy filter rules from MS SQL Server to COMPOSITE template SQL
  - Connects to MS SQL Server using tiberius crate with SQL Server Authentication
  - Configurable SQL query via `rule-converter-config.toml` file
  - ListView with multi-select for choosing rules to export
  - Exports selected rules as SQL INSERT statements with COMPOSITE template JSON parameters
  - Added Start Menu shortcut "Rule Converter (MS SQL)"
  - Files added:
    - `crates/pro-rule-converter-gui/Cargo.toml` - Package configuration
    - `crates/pro-rule-converter-gui/src/main.rs` - NWG-based GUI application
    - `crates/pro-rule-converter-gui/src/converter.rs` - Rule parsing and SQL generation
    - `crates/pro-rule-converter-gui/src/mssql.rs` - MS SQL Server client using tiberius
    - `crates/pro-rule-converter-gui/rule-converter-config.toml` - Configuration file with SQL query
    - `crates/pro-rule-converter-gui/build.rs` - Build script for Windows resources
    - `crates/pro-rule-converter-gui/windows-manifest.rc` - Windows manifest resource
    - `crates/pro-rule-converter-gui/windows-manifest.xml` - DPI awareness manifest

## [2.12.72.0] - 2025-01-12

### Added
- **Rule Converter Tool**: New CLI tool to convert legacy filter rules to COMPOSITE template SQL
  - Parses legacy `Parser.In()` syntax for DX, CPT, Date, POS fields
  - Generates SQL INSERT statements with proper COMPOSITE JSON parameters
  - Supports file input or inline rules via `--inline` flag
  - Usage: `pro-rule-converter -i rules.txt -o output.sql`
  - Files added:
    - `crates/pro-rule-converter/Cargo.toml`
    - `crates/pro-rule-converter/src/main.rs`

## [2.12.71.1] - 2025-01-12

### Added
- **AHRQOP001A Rule in Baseline**: Added AHRQ Opioid ED Visit rule to mandatory baseline
  - Added QM (Quality Measures) flag category
  - Added QM_OPIOID_ED flag issue
  - Added AHRQOP001A rule definition using COMPOSITE template
  - Rule flags ED visits (CPT 99281-99285, 99291) with opioid-related diagnosis (F11.x except F11.21, T40.x)

## [2.12.71.0] - 2025-01-12

### Added
- **COMPOSITE Rule Template**: New template for creating compound rules without recompilation
  - Supports AND/OR logic for combining multiple conditions
  - Condition types: cpt_in, cpt_pattern, dx_in, dx_pattern, dx_pattern_exclude, date_gte, date_lte, pos_in, pos_pattern, modifier_in, modifier_not_in
  - Enables database-only configuration of complex AHRQ quality indicators
  - Files added:
    - `crates/pro-rules/src/templates/composite_rule.rs` - Template implementation
    - `migrations/seed_data/ahrqop001a_opioid_ed_rule.sql` - Example AHRQ rule
  - Files modified:
    - `crates/pro-rules/src/templates/mod.rs` - Export new template
    - `crates/pro-rules/src/loader.rs` - Register COMPOSITE template
    - `migrations/046_create_rule_configuration_system.sql` - Add template to database
    - `migrations/000_baseline_v2.12.sql` - Add template to baseline

## [2.12.70.2] - 2025-01-01

### Changed
- **Project ID Auto-Generation**: Changed `projects.project.id` column to use IDENTITY instead of SERIAL
  - Updated `smartproaudit/000_baseline.sql` to use `GENERATED BY DEFAULT AS IDENTITY`
  - Allows auto-generation of IDs while still permitting explicit values when needed

## [2.12.70.1] - 2025-12-30

### Changed
- **FDW Password Authentication**: Updated Foreign Data Wrapper to use password authentication instead of peer authentication
  - Added password option to USER MAPPING in migration 069
  - Updated baseline 000 with the same change
  - Default credentials: user `postgres`, password `postgres`
  - Updated FDW_HOWTO.md documentation

## [2.12.70.0] - 2025-12-30

### Fixed
- **Reverted egui/eframe Migration**: Reverted GUI framework back to NWG (Native Windows GUI)
  - egui/eframe with wgpu backend requires DirectX 12/Vulkan which is not available on Windows Server 2019 without GPU
  - NWG uses Win32 GDI controls that work on all Windows versions without GPU requirements
  - Cleaned up temporary backup files

## [2.12.68.0] - 2025-12-30

### Enhanced
- **GUI 2025 UX Polish**: Applied modern design principles for a polished, professional appearance
  - **Increased Dimensions**: Larger windows with more generous spacing
    - Data Loader: 920×720 (was 900×680)
    - Project Manager: 1000×720 (was 960×680)
  - **Improved Typography**: Larger, more readable fonts with clear hierarchy
    - Header font: Segoe UI Semibold 17pt (was 15pt)
    - Body font: Segoe UI 14pt (was 13pt)
    - Log font: Consolas 13pt (was 12pt)
  - **Better Spacing**: Increased margins (20px from 16px), row heights (38px), and control heights (28px)
  - **Fixed Button Truncation**: Widened action buttons (180px from 150px) to fit "Load from Directory..." text
  - **Comfortable Touch Targets**: Larger button heights (34px) for easier clicking
  - **Consistent Font Application**: Body font applied to all form labels for uniformity
  - **Files Changed**:
    - `pro-data-loader-gui/src/main.rs`: Dimension and font updates
    - `pro-project/src/gui/app.rs`: Dimension and font updates

## [2.12.67.0] - 2025-12-30

### Enhanced
- **GUI Modernization - Full Visual Refresh**: Complete visual overhaul of both GUI applications
  - **Custom Font System**: Added distinct fonts for different UI elements
    - Header font: Segoe UI Semibold 15pt for section headers
    - Body font: Segoe UI 13pt for labels and text
    - Log font: Consolas 12pt for monospace log display
  - **Colored Status Indicators**: Traffic light style status colors
    - Green (Forest Green): Success/Connected/Up to date
    - Yellow (Dark Goldenrod): Warnings/Pending
    - Red (Firebrick): Errors/Failed
    - Blue (Steel Blue): Info/Processing
  - **RichLabel Status Displays**: Replaced plain Labels with RichLabel for colored, styled status text
  - **RichTextBox Activity Logs**: Replaced ListBox with RichTextBox for colored, formatted log entries
    - Log level indicators (INFO, SUCCESS, WARNING, ERROR) now colored and bold
  - **Improved Layout & Spacing**: Increased margins (16px), larger controls, better visual rhythm
  - **Files Changed**:
    - `pro-data-loader-gui/Cargo.toml`: Added `rich-textbox` feature
    - `pro-data-loader-gui/src/main.rs`: Full modernization
    - `pro-project/Cargo.toml`: Added `rich-textbox` feature
    - `pro-project/src/gui/app.rs`: Full modernization

## [2.12.66.0] - 2025-12-30

### Fixed
- **Project Database Manager Console Window - Complete Fix**: Use FreeConsole to completely detach from console
  - **Issue**: Previous ShowWindow(SW_HIDE) fix only minimized the console window; it remained visible in taskbar
  - **Solution**: Changed from `ShowWindow(SW_HIDE)` to `FreeConsole()` which completely detaches the process from its console
  - **Technical Details**: `FreeConsole()` is the proper Windows API for console detachment - it releases the console rather than just hiding it
  - **File Changed**: `pro-project/src/gui/mod.rs`

## [2.12.65.0] - 2025-12-30

### Fixed
- **Project Database Manager Console Window**: Hide console window when running in GUI mode
  - **Issue**: Black command prompt window appeared behind the GUI window
  - **Root Cause**: `pro-project` is compiled as a console application (no `windows_subsystem = "windows"`) to support CLI mode
  - **Solution**: Added `hide_console_window()` function that calls Windows API `ShowWindow(SW_HIDE)` when GUI mode starts
  - **File Changed**: `pro-project/src/gui/mod.rs`

## [2.12.64.0] - 2025-12-30

### Fixed
- **NWG GUI DPI Scaling - Feature Flag**: Enabled `high-dpi` feature in native-windows-gui
  - **Root Cause**: `scale_factor()` returns 1.0 (no scaling) unless `high-dpi` feature is enabled
  - **Solution**: Added `high-dpi` feature to both GUI crates' Cargo.toml
  - **Files Changed**:
    - `pro-data-loader-gui/Cargo.toml`: Added `features = ["high-dpi"]`
    - `pro-project/Cargo.toml`: Added `features = ["list-view", "high-dpi"]`
  - **Reference**: https://github.com/gabdube/native-windows-gui/blob/master/native-windows-gui/src/win32/high_dpi.rs

## [2.12.63.0] - 2025-12-30

### Fixed
- **NWG GUI DPI Scaling**: Implemented runtime DPI-aware layout for both GUI applications
  - **Root Cause**: Per-Monitor V2 DPI awareness in Windows manifest meant controls received physical pixels, but fixed pixel values designed for 96 DPI were too small at higher DPI settings (125%, 150%, etc.)
  - **Solution**: Query `nwg::scale_factor()` at runtime and scale all control dimensions proportionally
  - **Data Loader GUI** (`pro-data-loader-gui`):
    - Added `apply_dpi_scaling()` function that runs on init
    - All labels, buttons, text inputs, and layout positions scaled by DPI factor
    - Base window 880x620 at 96 DPI, scales appropriately at higher DPI
  - **Project Manager GUI** (`pro-project`):
    - Added `apply_dpi_scaling()` function that runs on init
    - Connection controls, toolbar, ListView, and log section all scale properly
    - ListView column widths also scaled for proper text display
    - Base window 920x600 at 96 DPI, scales appropriately at higher DPI
  - **Technical Details**:
    - Scale factor: 1.0 at 96 DPI (100%), 1.25 at 120 DPI (125%), 1.5 at 144 DPI (150%)
    - All pixel values multiplied by scale factor at runtime
    - Windows manifests retained Per-Monitor V2 for crisp text rendering

## [2.12.57.0] - 2025-12-29

### Fixed
- **NWG GUI Layout Rewrite**: Simplified GUI layout for better rendering
  - **Project Manager GUI**:
    - Removed Frame containers that caused black backgrounds
    - Added ListView `ex_flags` (FULL_ROW_SELECT, GRID) for proper rendering
    - Simplified layout with all controls directly on window
    - Reduced window size to 900x600 for better default display
  - Both GUIs now use flat layout without nested Frame containers

## [2.12.56.0] - 2025-12-29

### Fixed
- **NWG GUI Layout Issues**: Fixed control sizing and ListView rendering
  - **Data Loader GUI**: Widened labels and buttons to prevent text truncation
    - Increased window width from 900 to 950
    - Widened labels ("Organizations:", "Regions (Optional):", etc.)
    - Widened action buttons ("Load from Directory...", "Generate Templates...")
  - **Project Manager GUI**:
    - Widened toolbar buttons to show full text
    - Enabled `list-view` feature for `native-windows-gui` to fix black screen where ListView should appear

## [2.12.55.0] - 2025-12-29

### Fixed
- **NWG FileDialog Filter Format**: Fixed incorrect filter format causing GUI to crash on startup
  - **Error**: `Failed to build UI: FileDialogError("Bad extension filter format")`
  - **Root Cause**: NWG FileDialog filter format uses pipe to separate different filters, not pattern repetition
  - **Solution**: Changed from `"CSV Files (*.csv)|*.csv"` to `"CSV Files(*.csv)|All Files(*.*)"`

## [2.12.54.0] - 2025-12-29

### Fixed
- **NWG GetWindowSubclass Error - Complete Fix**: Added Windows manifest to resolve "Entry Point Not Found" error
  - **Error**: `The procedure entry point GetWindowSubclass could not be located in comctl32.dll`
  - **Root Cause**: `windows-sys` crate (pulled by sqlx via etcetera) requires a manifest declaring Common Controls v6
  - **Solution**: Embed Windows manifest in both GUI executables using `embed-resource` crate
  - **Files Added**:
    - `pro-data-loader-gui/pro-data-loader-gui.exe.manifest`: Common Controls v6 declaration
    - `pro-data-loader-gui/pro-data-loader-gui-manifest.rc`: Resource file
    - `pro-data-loader-gui/build.rs`: Build script to embed manifest
    - `pro-project/pro-project.exe.manifest`: Common Controls v6 declaration
    - `pro-project/pro-project-manifest.rc`: Resource file
    - `pro-project/build.rs`: Build script to embed manifest
  - **Reference**: [NWG Issue #251](https://github.com/gabdube/native-windows-gui/issues/251)

## [2.12.53.0] - 2025-12-29

### Fixed
- **NWG GetWindowSubclass Error**: Attempted fix by pinning chrono (incomplete - see 2.12.54.0)
  - **Error**: `The procedure entry point GetWindowSubclass could not be located in comctl32.dll`
  - **Root Cause**: Conflict between `native-windows-gui` and `chrono` 0.4.27+ which pulls in `windows-targets`
  - **Solution**: Pin chrono to use `default-features = false` to exclude `windows-iana` feature
  - **Reference**: [NWG Issue #282](https://github.com/gabdube/native-windows-gui/issues/282)

## [2.12.52.0] - 2025-12-29

### Changed
- **Windows Server 2019 GUI: Migrated to Native Windows GUI (NWG)**
  - **Solution**: Replaced egui/eframe (wgpu-based) with Native Windows GUI which uses Win32 GDI controls
  - **Why**: wgpu/WARP/DX12 still failed on Windows Server 2019 RDS sessions - no GPU adapter available
  - **NWG Benefits**:
    - Uses Win32 GDI controls - pure software rendering built into Windows
    - No GPU/OpenGL/DirectX/Vulkan requirements
    - Works on all Windows versions (Vista+) including headless Windows Server
    - Lighter dependencies - no wgpu, winit, or graphics drivers needed
  - **Files Changed**:
    - `pro-project/Cargo.toml`: Replaced eframe/egui with native-windows-gui/native-windows-derive
    - `pro-project/src/gui/mod.rs`: NWG initialization
    - `pro-project/src/gui/app.rs`: Complete rewrite using NWG controls
    - `pro-data-loader-gui/Cargo.toml`: Replaced eframe/egui with native-windows-gui
    - `pro-data-loader-gui/src/main.rs`: Complete rewrite using NWG controls
  - **Reference**: [Native Windows GUI](https://github.com/gabdube/native-windows-gui)
  - **Plan Document**: [NWG_GUI_MIGRATION_PLAN.md](docs/NWG_GUI_MIGRATION_PLAN.md)

## [2.12.51.0] - 2025-12-29

### Fixed
- **Windows Server 2019 GUI: DirectX 12 WARP Configuration**: Proper configuration for WARP software rendering
  - **Change**: Explicitly configure `WgpuConfiguration` with DX12 backend and `LowPower` preference
  - **Technical Details**:
    - Set `supported_backends: wgpu::Backends::DX12` in code (not just env var)
    - Set `power_preference: wgpu::PowerPreference::LowPower` to help select WARP
    - WARP (Windows Advanced Rasterization Platform) is built into Windows Server 2019
  - **Files**: `pro-project/src/gui/mod.rs`, `pro-data-loader-gui/src/main.rs`
  - **Reference**: [Microsoft WARP Guide](https://learn.microsoft.com/en-us/windows/win32/direct3darticles/directx-warp)
  - **Note**: This did not work - see v2.12.52.0 for the successful NWG solution

## [2.12.50.0] - 2025-12-29

### Changed
- **Windows Server 2019: CLI-First Architecture**: Resolved GUI issues on headless Windows Server by making CLI the primary interface
  - **Root Cause**: Windows Server 2019 headless environments lack GPU/graphics support required by modern GUI frameworks (wgpu, OpenGL 3.3+, Vulkan, DirectX 12)
  - **Solution**: CLI-first design - all functionality available via command line
  - **pro-project.exe**: Now runs as CLI by default (console window visible), use `--gui` flag for GUI mode
  - **pro-data-loader.exe**: Pure CLI tool with full functionality
  - **pro-data-loader-gui.exe**: Shows helpful error message directing to CLI if GUI fails
  - **Error messages**: Now include specific CLI examples when GUI unavailable
  - See: [WINDOWS_SERVER_GUI_SOLUTION.md](docs/WINDOWS_SERVER_GUI_SOLUTION.md) for full documentation

### Usage on Windows Server
```powershell
# Project management
pro-project.exe list
pro-project.exe create --name MyProject
pro-project.exe switch --name MyProject
pro-project.exe status

# Master data loading
pro-data-loader.exe --csv-dir C:\data\master
```

## [2.12.49.0] - 2025-12-29

### Fixed
- **Windows Server 2019 GUI Compatibility (v3)**: Comprehensive fix for GUI applications on headless Windows Server
  - **Changes**:
    - Try multiple backends: Vulkan → DX12 → GL (in order of preference)
    - Disabled multisampling (`multisampling: 0`) - required for software renderers
    - Disabled depth buffer (`depth_buffer: 0`) - reduces GPU requirements
    - Set `WGPU_POWER_PREF=low` - prefer integrated/software rendering
    - Set `WGPU_ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER=1` - allow software renderers
    - Added user-friendly error message with solution instructions if GUI fails to start
  - **If GUI still fails**: Install Mesa3D for Windows from https://github.com/pal1000/mesa-dist-win
    - Download mesa3d release, extract `opengl32.dll` and `libgallium_wgl.dll`
    - Copy both DLLs to `C:\Program Files\Professional SMART\bin\`
  - Reference: [egui software rendering issue](https://github.com/emilk/egui/issues/957)

## [2.12.48.0] - 2025-12-29

### Fixed
- **Windows Server 2019 GUI Compatibility (v2)**: Additional fix for GUI applications not loading on Windows Server 2019
  - **Root Cause**: Windows Server RDS sessions don't expose GPU by default, and wgpu needs explicit DirectX 12 backend selection
  - **Solution**: Added `WGPU_BACKEND=dx12` environment variable at startup to force DirectX 12 with WARP software rendering fallback
  - Files: `pro-project/src/gui/mod.rs`, `pro-data-loader-gui/src/main.rs`
  - Reference: [wgpu DX12 WARP issue](https://github.com/gfx-rs/wgpu/issues/2503)

## [2.12.47.0] - 2025-12-29

### Fixed
- **Windows Server 2019 GUI Compatibility**: Fixed Project Database Manager and Master Data Loader GUI applications not loading on Windows Server 2019
  - **Root Cause**: The `glow` (OpenGL) backend requires OpenGL 2.0+ which is not available on headless Windows Server environments
  - **Solution**: Switched to `wgpu` backend which can use DirectX 12 WARP (software renderer) when GPU is unavailable
  - Files: `pro-project/Cargo.toml`, `pro-data-loader-gui/Cargo.toml`, GUI initialization code updated
  - Reference: [egui_glow NoAvailablePixelFormat issue](https://github.com/emilk/egui/issues/957)

## [2.12.46.0] - 2025-12-29

### Performance
- **Provider Cache Optimization**: Implemented in-memory NPI → provider_id cache in ClaimsProcessor
  - **Root Cause**: Same provider NPI appears up to 16 times per encounter (4 encounter-level + 4 per service line × ~3 lines)
  - **Impact**: Each `ensure_provider_exists` call was executing 2 DB queries (upsert + enrichment queue) for every occurrence
  - **Solution**: Cache provider_id after first lookup, subsequent lookups return from cache instantly
  - **Verified Result**: **1,284 claims/second** (192.8% of SRD target)
  - Files: `claims_processor.rs` updated

### SRD Performance Target ACHIEVED
- **Target**: 10,000 claims in 15 seconds (666.67 claims/sec)
- **Actual**: 9,971 claims in 7.76 seconds (**1,284 claims/sec**)
- **Performance**: 192.8% of target (nearly 2x requirement)

| Version | Throughput | Notes |
|---------|------------|-------|
| v2.12.44.0 | ~190 claims/sec | Baseline with default config |
| v2.12.45.0 | ~195 claims/sec | Trigger removal (+2.6%) |
| **v2.12.46.0** | **1,284 claims/sec** | Provider cache (+558%) |

## [2.12.45.0] - 2025-12-29

### Performance
- **CRITICAL: Removed sync_encounter_totals Triggers**: Dropped the `sync_encounter_totals_insert`, `sync_encounter_totals_update`, and `sync_encounter_totals_delete` triggers from `claims.service_line` table.
  - **Root Cause**: These triggers fired for EVERY service line insert, executing a `SELECT SUM()` and `UPDATE` on the encounter table each time
  - **Impact**: For 10,000 claims with ~30,000 service lines, this added ~60,000 extra database operations
  - **Why Safe**: The `total_claim_charge_amount` is already calculated in Rust before the encounter INSERT, making the triggers redundant
  - **Expected Improvement**: From ~190 claims/sec to 600+ claims/sec (eliminating 6 queries per encounter)
  - Files: `070_drop_encounter_totals_trigger.sql`, `000_baseline_v2.12.sql` updated

## [2.12.44.0] - 2025-12-29

### Performance
- **Installer Default Configuration**: Updated default worker configuration for new installs to achieve 666+ claims/sec SRD target:
  - Changed `WORKER_THREADS=4` to `STAGE2_WORKER_COUNT=8` (correct environment variable name)
  - Changed `BATCH_SIZE=100` to `BATCH_SIZE=750` (proven optimal for throughput)
  - Added inline documentation comments explaining each setting
  - Files updated: `env.template`, `WriteConfig.vbs`

## [2.12.43.0] - 2025-12-28

### Code Quality (MEDIUM Priority Fixes)
- **Registry Service Methods**: Added `#[allow(dead_code)]` to `get_active_project()` and `project_exists()` (reserved for future project switching UI)
- **Windows Service Manager**: Added `#[allow(dead_code)]` to `restart()` (reserved for future service management UI)
- **WebSocket State**: Added `#[allow(dead_code)]` to `broadcaster()` (reserved for future progress tracking integration)
- **Pipeline Wrapper Methods**: Added `#[allow(dead_code)]` to `extract_diagnoses_from_csv()`, `extract_service_lines_from_csv()`, and `process_claim_in_transaction()` (wrapper methods superseded by improved implementations)
- **Unused Imports Cleanup**: Removed unused `DEFAULT_DATE` import from claims_processor.rs
- **Variable Prefixes**: Fixed unused variable warnings (`_data` in websocket.rs, `_fac_id` in dashboard.rs, removed `mut` from `batch_rx` in service.rs)

### Code Review Status
- MEDIUM priority items from CODE-REVIEW-2025-12-28.md addressed
- Verified code review item "parser.rs unused apply_transformations" - import does not exist (false positive)
- Verified code review item "transformers.rs unused regex::Regex" - import does not exist (false positive)
- Verified code review item "models.rs unused sqlx::types::Uuid" - import does not exist (false positive)
- Reviewed `.ok()` patterns in claims_processor.rs - acceptable for optional JSON field parsing with default fallback

## [2.12.42.0] - 2025-12-28

### Code Quality
- **Iterator Pattern Improvements**: Refactored batch INSERT placeholder generation to use idiomatic `map().collect()` instead of `for i in 0..len()` loops in `import_encounter_payers_from_cob()` and `import_other_insurance()`
- **Service Constants Documentation**: Added `#[allow(dead_code)]` to `SERVICE_DISPLAY_NAME` and `SERVICE_DESCRIPTION` constants (referenced via function parameters)
- **Connection Pool Documentation**: Added documentation to `DatabaseService` explaining that fresh connections per operation are acceptable for infrequent project management operations

### Schema Review
- **Migration 018 Verified**: Confirmed flag table indexes are properly commented out with documentation explaining `claims.flag` table doesn't exist
- **Migration 019 Verified**: Confirmed materialized views migration is properly disabled with documentation for future flag table refactoring

## [2.12.41.0] - 2025-12-28

### Changed
- **Additional Unused Code Documentation**: Added `#[allow(dead_code)]` annotations with documentation for reserved/future-use code:
  - `pro-rules`: Removed unused `FlagContext` import from threshold_rule.rs, `Error` from missing_field_rule.rs, `Rule` from hot_reload.rs
  - `pro-worker`: Removed unused `Error` and `Encounter` imports from claim_processor.rs, `ClaimProcessingResult` from file_processor.rs
  - `pro-worker`: Documented `facility_id` field in `IngestionPipeline` as reserved for future facility-specific rule loading
  - `pro-project`: Documented `ProjectRow` and `TaskMessage` structs as GUI data models
  - `pro-setup`: Documented PostgreSQL auto-installer functions as reserved for future feature

## [2.12.40.0] - 2025-12-28

### Performance
- **Batch COB Payer Inserts**: Optimized `import_encounter_payers_from_cob()` to use batch INSERT instead of individual inserts per payer.
  - Reduces database round-trips from N to 1 for COB payer imports
  - Improves throughput for claims with multiple insurance payers
- **Batch Other Insurance Inserts**: Optimized `import_other_insurance()` to use batch INSERT instead of individual inserts.
  - Reduces database round-trips from N to 1 for other insurance records

### Fixed
- **Silent Provider Lookup Errors**: Changed provider lookup `.unwrap_or(None)` to `.unwrap_or_else()` with warning logging.
  - Ensures unexpected errors from `ensure_provider_exists()` are logged instead of silently dropped
  - Affects rendering, referring, supervising, and billing provider lookups in claims processor

### Removed
- **Unused Encounter Repository Methods**: Removed unused `list_by_organization()`, `list_by_facility()`, and `list_by_date_range()` methods from `EncounterRepository`.
  - These methods were never called and used `SELECT *` pattern

### Changed
- **Unused Code Documentation**: Added `#[allow(dead_code)]` with documentation comments to utility methods reserved for future use:
  - `BackupService::verify()`, `BackupService::list_backups()`, `BackupInfo` struct
  - `ConfigService::exists()`, `DbParams::connection_string*()` methods
  - `MigrationService::get_baseline()`, `apply_all_pending()`, `update_application_version()`
  - `ProjectStatus::Error`, `ProjectStatus::Checking` enum variants
  - `MigrationResult` struct
- **Cleanup Unused Imports**: Removed unused imports from `pro-rules` crate (template.rs, loader.rs, composite_rule.rs, hot_reload.rs)
- **Data Loader Validation Comment**: Added comment clarifying facility validation in provider import

## [2.12.39.0] - 2025-12-28

### Fixed
- **Installer Schema Version Query**: Fixed psql command construction for schema version query when psql is in system PATH.
  - Added proper handling for PATH vs full-path psql executable scenarios
  - Added detailed logging for schema version query debugging
  - Schema version is now correctly calculated from highest migration number

## [2.12.38.0] - 2025-12-28

### Fixed
- **Installer Schema Version Registration**: Fixed MSI installer using build version instead of actual schema version when registering projects.
  - `CreateDatabase.vbs` now queries `staging.schema_migrations` after applying migrations to get the actual schema version
  - Fresh installs now correctly register with schema version 2.12.69.0 instead of the build version
  - Schema version is calculated from highest migration number (e.g., migration 069 = 2.12.69.0)

## [2.12.37.0] - 2025-12-28

### Fixed
- **Baseline Missing Migration Registrations**: Fixed baseline not registering migrations 031-069 in `schema_migrations` table.
  - Added INSERT statements for all 39 missing migrations (031-069) at the end of `000_baseline_v2.12.sql`
  - Fresh installs now correctly show schema version as 2.12.69.0

## [2.12.36.0] - 2025-12-28

### Fixed
- **Fresh Install Schema Version**: Fixed fresh database installs incorrectly setting schema version to build version instead of migration-based version.
  - `get_schema_version()` now calculates version from `staging.schema_migrations` table (highest migration number)
  - Fresh installs now correctly report schema version as 2.12.69.0 (based on 69 migrations in baseline)
  - Removed hardcoded version fallbacks in favor of dynamic calculation from embedded migrations

## [2.12.35.0] - 2025-12-27

### Fixed
- **Windows Server GUI Compatibility**: Fixed GUI applications not loading on Windows Server 2019+.
  - Switched from wgpu to glow (OpenGL) renderer backend for software rendering fallback
  - Explicitly set `renderer: eframe::Renderer::Glow` for both pro-project and pro-data-loader-gui
  - GUI tools now work on servers without GPU acceleration

## [2.12.34.0] - 2025-12-27

### Fixed
- **Project Database Manager Version Update**: Fixed GUI upgrade not updating `database_version` in SmartProAudit registry after applying migrations.
  - GUI now correctly updates `projects.project.database_version` after successful schema upgrade
  - Version is computed from highest applied migration number (e.g., migration 069 -> 2.12.69.0)
  - Both CLI and GUI upgrade paths now consistently update the registry
- **Multi-statement Migration Execution**: Fixed migration application failing for SQL files with multiple statements.
  - Added `split_sql_statements()` function to properly parse SQL files
  - Handles dollar-quoted strings (`$$`) in PostgreSQL functions correctly
  - Uses `sqlx::raw_sql()` instead of `sqlx::query()` for statement execution
- **Migration Column Name Fix**: Fixed MigrationService querying wrong column name.
  - Changed from `version` to `migration_name` column in `staging.schema_migrations`
  - Version is now extracted from migration filename (e.g., "069" from "069_setup_smartproaudit_fdw.sql")

## [2.12.33.0] - 2025-12-27

### Added
- **Project Database Manager Tool (`pro-project.exe`)**: New CLI and GUI tool for managing multiple Professional SMART project databases.
  - **CLI Commands:**
    - `pro-project create --name <NAME> [--switch]` - Create new project database with full schema
    - `pro-project switch --name <NAME> [--no-restart]` - Switch active database and restart service
    - `pro-project list [--format table|json|csv]` - List all registered project databases
    - `pro-project info [--name <NAME>]` - Show detailed project information
    - `pro-project delete --name <NAME> [--force] [--backup]` - Delete project database
    - `pro-project backup [--name <NAME>] [--output <PATH>]` - Create pg_dump backup
    - `pro-project status` - Show schema upgrade status for all projects
    - `pro-project upgrade [--name <NAME>|--all] [--backup] [--dry-run]` - Apply pending migrations
  - **GUI Mode:**
    - `pro-project gui` - Launch graphical interface
    - Data grid showing all SmartProAudit registered projects
    - Checkbox selection for batch operations
    - Status indicators (up to date, pending upgrades, errors)
    - "Upgrade Selected" and "Backup & Upgrade" actions
    - Real-time progress and log display
  - **Services:**
    - `RegistryService` - Query/update SmartProAudit project registry
    - `DatabaseService` - PostgreSQL operations (create, drop, schema check)
    - `ConfigService` - Atomic .env file updates with backup
    - `WindowsServiceManager` - Stop/start ProfessionalSMART service
    - `MigrationService` - Detect and apply pending migrations
    - `BackupService` - pg_dump backup operations
  - **Installer Integration:**
    - Added `pro-project.exe` to installer package
    - Start Menu shortcut: "Project Database Manager"

## [2.12.32.0] - 2025-12-27

### Fixed
- **Database Name Case Preservation**: Fixed PostgreSQL case sensitivity issue where database names with mixed case were being lowercased during creation.
  - Added proper SQL identifier quoting in `CREATE DATABASE` commands in `CreateDatabase.vbs`
  - Database names like `professional_smart_clientA` now preserve case correctly
  - Service was failing to start because it looked for `professional_smart_clientA` but database was created as `professional_smart_clienta`

## [2.12.31.0] - 2025-12-27

### Fixed
- **Encounter View Column Bug**: Removed non-existent `encounter_group_id` column from `claims.encounter_view`.
  - Column was referenced in migration 068 but doesn't exist in `claims.encounter` table
  - Fixed in both `068_create_encounter_view.sql` and `000_baseline_v2.12.sql`

## [2.12.30.0] - 2025-12-27

### Fixed
- **SmartProAudit Database Name Case**: Fixed PostgreSQL case sensitivity issue with database name.
  - Changed `SmartProAudit` to lowercase `smartproaudit` throughout installer and migrations
  - PostgreSQL lowercases unquoted identifiers, causing connection failures with mixed-case names
  - Affected files: `CreateDatabase.vbs`, `069_setup_smartproaudit_fdw.sql`, `000_baseline_v2.12.sql`

## [2.12.29.0] - 2025-12-26

### Changed
- **Unique Username Constraint**: Changed `idx_security_user_name` from regular index to UNIQUE index.
  - Enforces unique usernames in `security.security_user` table
  - Prevents duplicate user registrations

## [2.12.28.0] - 2025-12-26

### Added
- **SmartProAudit Foreign Data Wrapper**: Added cross-database querying capability for project databases.
  - Migration: `migrations/069_setup_smartproaudit_fdw.sql`
  - Enables `postgres_fdw` extension in each project database
  - Creates `smartproaudit` schema with foreign tables linked to SmartProAudit database
  - Foreign tables: `security_role`, `security_user`, `security_user_role`, `lookup_field_definitions`, `project`
  - Convenience view: `smartproaudit.user_roles` - Shows users with their assigned roles
  - Helper function: `smartproaudit.user_has_role(user_name, role_name)` - Check user permissions
  - Helper function: `smartproaudit.get_field_definitions(table_name)` - Get friendly column names
  - Allows project databases (professional_smart_clientA, etc.) to query centralized security and field definitions

## [2.12.27.0] - 2025-12-26

### Added
- **Security Schema Indexes**: Added performance and integrity indexes to security tables.
  - `idx_security_role_name` (UNIQUE) - Fast role lookup by name, enforces unique role names
  - `idx_security_user_role_unique` (UNIQUE) - Prevents duplicate user-role assignments

## [2.12.26.0] - 2025-12-26

### Added
- **Security Schema**: Added `security` schema to SmartProAudit master database for user authentication and role-based access control.
  - `security.security_role` table with role_name and role_description
  - `security.security_user` table with user_name and active status
  - `security.security_user_role` junction table for user-role assignments
  - Pre-populated with Admin, Super User, and User roles
  - Default user 'MWELLINGTO002' with Admin role

## [2.12.25.0] - 2025-12-26

### Added
- **SmartProAudit Master Database**: New PostgreSQL database for centralized project management.
  - Schema file: `migrations/smartproaudit/000_baseline.sql`
  - **projects schema**: Tracks all Professional SMART project databases
    - `projects.project` table with project_name, organization, versions, connection info
    - `projects.schema_migrations` table for SmartProAudit upgrades
  - **fields schema**: Field metadata for claims data display and export
    - `fields.lookup_field_definitions` table with friendly names for columns
    - Pre-populated with encounter, service_line, encounter_diagnosis, encounter_payer, encounter_view fields
  - Replaces file-based `projects.json` registry with PostgreSQL-based registry
  - Automatically created during installation if it doesn't exist
  - Each project database is registered in SmartProAudit during creation

### Changed
- **Installer**: Modified `CreateDatabase.vbs` to create and initialize SmartProAudit database
- **Product.wxs**: Added SmartProAudit migration files to installer package

## [2.12.24.0] - 2025-12-26

### Added
- **Encounter View**: Added `claims.encounter_view` for denormalized access to encounter data.
  - Migration: `migrations/068_create_encounter_view.sql`
  - Joins encounter with all provider types (billing, referring, rendering, supervising)
  - Includes payer hierarchy (primary, secondary, tertiary) from `encounter_payer` table
  - Aggregates diagnosis codes as comma-separated list ordered by sequence
  - Includes service facility details
  - Useful for reporting and data export without complex joins

## [2.12.23.0] - 2025-12-23

### Added
- **Encounter Procedure Modifiers Table**: Added new table `claims.encounter_procedure_modifier` to store aggregated procedure modifiers at encounter level.
  - Migration: `migrations/067_create_encounter_procedure_modifiers.sql`
  - Stores comma-separated list of unique modifiers from all service lines (e.g., "24,25,59")
  - VARCHAR(20) column for modifiers, deduplicated and sorted
  - Foreign key reference to `claims.encounter` with CASCADE delete
  - GIN index for pattern matching (e.g., finding encounters with modifier "25")
  - Automatically populated during claim ingestion from service line modifiers
  - File: `crates/pro-service/src/claims_processor.rs` - added `insert_encounter_procedure_modifiers()` function

### Fixed
- **Embedded Migrations**: Added migrations 066 and 067 to `crates/pro-upgrade-manager/src/embedded_migrations.rs` so they are included in the installer.
- **Baseline Migration**: Updated `migrations/000_baseline_v2.12.sql` to include migrations 065-067 (now covers 001-067).
- **CLAUDE.md Documentation**: Added installer build process documentation with step-by-step guide for adding new migrations.

## [2.12.22.0] - 2025-12-23

### Added
- **PostgreSQL Settings Enforcement Migration**: Added migration 066 to automatically enforce critical PostgreSQL settings during install/upgrade.
  - Migration: `migrations/066_enforce_postgresql_settings.sql`
  - Ensures `autovacuum = 'on'` to prevent table bloat (fixes issue from v2.12.19.0)
  - Sets `work_mem = '64MB'` to prevent memory exhaustion (fixes issue from v2.12.19.0)
  - Reloads PostgreSQL configuration automatically
  - Verifies settings were applied with NOTICE logging
  - Previously these were manual fixes; now enforced automatically on every install/upgrade

### Fixed
- **Build Script WiX Path**: Fixed `build-msi.ps1` to automatically add WiX Toolset to PATH and pass SolutionDir variable to candle.

## [2.12.21.0] - 2025-12-22

### Fixed
- **Removed Provider Advisory Locks**: Removed `pg_try_advisory_xact_lock` mechanism that was causing 96% claim failure rate.
  - File: `crates/pro-service/src/claims_processor.rs`
  - Root cause: Advisory locks prevented concurrent processing of claims with the same provider NPI. With test data using a single billing provider (NPI 1234567890), only 1 of 8 workers could proceed - the other 7 failed with "provider locked".
  - Impact: 9,627 of 10,000 claims (96%) marked as FAILED due to lock conflicts.
  - Fix: Removed advisory lock mechanism entirely. The `ensure_provider_exists` function already uses `INSERT ON CONFLICT DO NOTHING` which is safe for concurrent access.
  - Result: All workers can now process claims concurrently without lock contention.

### Performance Results
- **Target: 666 claims/second - ACHIEVED (123.5%)**
- Test: 10,000 claims (29,626 service lines) processed in 36.02 seconds
- Throughput: **822.5 claims/second** (274 encounters/second)
- Sustained rate: 290-340 encounters/second (870-1020 claims/second)
- Success rate: 98.7% (9,871 completed, 129 failed due to future DOS dates in test data)

## [2.12.20.0] - 2025-12-22

### Fixed
- **FIFO Batch Result Scatter Bug**: Fixed critical bug where provider lock conflicts caused claims to be reset and re-acquired across multiple batch sequences, breaking FIFO ordering.
  - File: `crates/pro-service/src/claims_processor.rs`
  - Root cause: When `pg_try_advisory_xact_lock` failed to acquire a provider lock, claims were reset to `PENDING` with `batch_sequence_number = NULL`, allowing them to be re-acquired by different batches.
  - Impact: Batches expected 100 claims but only retained 0-8 claims each. Workers returned near-empty results. SequentialCompletionManager never received meaningful batch completions. All batches ended up in RECOVERY state after 5-minute timeout.
  - Fix: Instead of resetting claims on lock conflict, mark them as failed within the same batch. This preserves batch integrity and allows proper FIFO completion tracking.
  - Note: This fix was superseded by v2.12.21.0 which removes advisory locks entirely.

### Performance
- Previous test: 163 encounters/second (~490 claims/second) but with broken batch tracking
- Expected after fix: Improved throughput with proper FIFO completion

## [2.12.19.0] - 2025-12-22

### Fixed
- **PostgreSQL Autovacuum Disabled**: Re-enabled autovacuum which was disabled in postgresql.auto.conf, causing 717k dead tuples (71x table bloat) on staging.raw_claims table.
  - Root cause: `autovacuum = 'off'` in postgresql.auto.conf
  - Impact: Table bloat from 30 MB to 238 MB, degraded query performance
  - Fix: `ALTER SYSTEM SET autovacuum = 'on'`

- **Excessive work_mem Setting**: Reduced work_mem from 512MB to 64MB to prevent memory exhaustion.
  - Root cause: `work_mem = '512MB'` with 300 max_connections could consume 150GB+ RAM
  - Fix: `ALTER SYSTEM SET work_mem = '64MB'`

- **Table Bloat Cleanup**: Ran VACUUM FULL ANALYZE on staging.raw_claims to reclaim space.
  - Before: 238 MB (717,632 dead tuples)
  - After: 30 MB (0 dead tuples)
  - Reduction: 87%

### Performance
- Measured baseline: 159 encounters/second (~477 claims/second)
- Target: 666 claims/second (per SRD.md specification)

## [2.12.18.0] - 2025-12-22

### Performance
- **Simplified FIFO Batch Acquisition**: Replaced complex CTE-based encounter grouping with simple FIFO-ordered claim acquisition.
  - File: `crates/pro-service/src/batch_sequencer.rs`
  - Problem: CTE with JSONB expression extraction, GROUP BY, and JOIN still taking 1-3 seconds per batch despite indexes
  - Root cause: Expression indexes help but don't eliminate JSONB extraction overhead; partial index invalidation during updates; 266k dead tuples causing bloat
  - Solution: Simplified query that:
    1. Selects claims by `ingested_at ASC` order (simple btree scan)
    2. Locks and updates atomically with `FOR UPDATE SKIP LOCKED`
    3. Relies on application-layer encounter grouping (already in claims_processor.rs)
  - Benefits:
    - No JSONB extraction in query
    - No GROUP BY or JOIN operations
    - Uses simple btree index on ingested_at
    - Avoids partial index invalidation issues
  - Expected: 10-20x faster batch acquisition

### Target
- Performance target: 666 claims/second (per SRD.md specification)
- Previous measured: 73 claims/second (v2.12.17.0 with indexed CTE)
- Expected: 400+ claims/second with simplified acquisition

## [2.12.17.0] - 2025-12-22

### Performance
- **CTE Batch Acquisition Index Optimization**: Added expression indexes on JSONB fields to optimize the CTE-based batch acquisition query.
  - Migration: `migrations/065_cte_batch_acquisition_indexes.sql`
  - Problem: CTE query taking 2.5+ seconds per batch due to missing indexes on JSONB expressions
  - Root cause: `encounter_fields->>'patient_control_number'` and `encounter_fields->>'date_of_service_from'` were being extracted without index support, requiring full table scans for GROUP BY and JOIN operations
  - Solution: Created 4 expression indexes:
    1. `idx_raw_claims_pcn_expr` - Expression index on patient_control_number
    2. `idx_raw_claims_dos_expr` - Expression index on date_of_service_from
    3. `idx_raw_claims_encounter_fifo` - Composite expression index for GROUP BY and FIFO ordering
    4. `idx_raw_claims_encounter_notnull` - Partial index with NOT NULL filters for pre-filtering
  - Result: Query time reduced from 2.5s to ~95ms (26x improvement)
  - PostgreSQL best practice: See https://www.postgresql.org/docs/current/indexes-expressional.html

### Target
- Performance target: 666 claims/second (per SRD.md specification)
- Previous measured: 38 claims/second (with slow CTE query)
- Expected: Significant improvement with indexed batch acquisition

## [2.12.16.0] - 2025-12-22

### Fixed
- **Batch Acquisition Re-acquisition Bug**: Rewrote `acquire_next_batch()` using a single atomic CTE (Common Table Expression) to eliminate the 20x re-acquisition overhead.
  - File: `crates/pro-service/src/batch_sequencer.rs`
  - Root cause: Previous implementation selected 2x batch_size claims with `FOR UPDATE SKIP LOCKED`, but only updated a subset to PROCESSING. The remaining claims were unlocked on commit and immediately re-acquired by the next iteration.
  - Evidence: 206,794 claims batched for 10,000 actual claims (20.7x overhead)
  - Fix: Single CTE that atomically:
    1. Identifies N distinct encounter groups in FIFO order
    2. Selects ALL claims belonging to those encounter groups
    3. Updates ALL selected claims to PROCESSING in one operation
  - Benefits: Atomic operation, no race conditions, complete encounter integrity, standard PostgreSQL best practice

### Performance
- Expected improvement: ~20x reduction in batch acquisition overhead
- Target: 666 claims/second (per SRD.md specification)
- Previous measured: 114 claims/second (with re-acquisition bug)

## [2.12.15.0] - 2025-12-22

### Performance
- **Parser Logging Optimization**: Downgraded all `[LOOP_DEBUG]` logging in `identify_loops()` from INFO to DEBUG level. These messages were generating 80,000+ log entries per 10k claims (8 messages per claim), causing massive I/O overhead.
  - File: `crates/pro-parser-edi/src/parser.rs`
  - 10 `info!()` calls changed to `debug!()`
  - Expected impact: Significant reduction in logging I/O overhead during parsing phase

### Target
- Performance target: 666 claims/second (per SRD.md specification)
- Previous measured: 94 claims/second (with excessive parser logging)

## [2.12.14.0] - 2025-12-22

### Fixed
- **Worker Transaction Handling**: Applied per-encounter transaction fix to `process_sequenced_batch()` (the multi-worker code path). Previously, when one encounter failed (e.g., `validate_dos()` trigger for future dates), the entire batch transaction was aborted, and all subsequent claims in the batch remained stuck in PROCESSING state.
  - File: `crates/pro-service/src/claims_processor.rs`
  - Root cause: "current transaction is aborted" error cascade when one encounter fails within a batch
  - Fix: Each encounter now has its own transaction; failures don't cascade to other encounters in the batch

### Performance
- Performance baseline maintained at ~142 claims/second for successful claims

## [2.12.13.0] - 2025-12-22

### Performance
- **Phase 1: Stuck Claims Recovery**: Added automatic recovery of stale PROCESSING claims (stuck > 5 minutes) at startup. This prevents claims from being permanently stuck if a previous run crashed.
- **Phase 2: Per-Encounter Transactions**: Changed from batched transactions to per-encounter transactions. Failures in one encounter no longer cause cascading rollbacks of successful encounters.
- **Phase 3: Batch Status Updates**: Batch update of claim statuses using `UPDATE ... WHERE raw_claim_id = ANY($1)` instead of individual UPDATE queries per claim.
- **Phase 4: Reduced JSON Cloning**:
  - Changed `process_encounter_with_service_lines()` to take `&[RawClaim]` reference instead of `Vec<RawClaim>` (avoiding clone)
  - Added `count_service_lines_in_json_value()` to work directly with JsonValue (avoiding deserialize + clone)
  - Changed total_claim_charge calculation to use `get()` on JsonValue directly
  - File: `crates/pro-service/src/claims_processor.rs`

### Fixed
- **5,044 claims stuck in PROCESSING**: Root cause was transaction rollback followed by failed error logging transaction. Claims are now properly marked COMPLETED or FAILED after processing.

## [2.12.12.0] - 2025-12-22

### Performance
- **Logging Optimization**: Downgraded diagnosis pointer logging from INFO to DEBUG level. The INFO-level logging was generating 233,108+ log messages per 10k claims run, causing significant I/O overhead and reducing performance from ~130 to ~113 claims/second.
  - File: `crates/pro-service/src/claims_processor.rs`
  - The diagnostic logging remains available at DEBUG level for troubleshooting

## [2.12.11.0] - 2025-12-22

### Fixed
- **Operation Order Bug**: Moved `import_diagnoses()` to execute BEFORE `import_service_line()` in `process_encounter_with_service_lines()`. Previously, service line diagnosis pointers were trying to reference diagnoses that hadn't been inserted yet, causing 0 rows to be inserted into the junction table.
  - File: `crates/pro-service/src/claims_processor.rs`
  - Root cause: Diagnoses must exist before service lines can reference them via diagnosis pointers

## [2.12.10.0] - 2025-12-22

### Added
- **Diagnostic Logging**: Added INFO-level logging to `import_service_line_diagnosis_pointers()` to trace diagnosis pointer insertion issues.
  - Logs when pointers are empty (all None)
  - Logs pointer count before processing
  - Logs rows_affected after INSERT execution

## [2.12.9.0] - 2025-12-22

### Fixed
- **Diagnosis Pointer Insert Bug**: Fixed `param_idx` starting at 2 instead of 3 in `import_service_line_diagnosis_pointers()` function. This caused `$2` to be used for both `encounter_id` and the first `pointer_sequence`, resulting in no rows being inserted into `service_line_diagnosis_pointer` junction table.
  - File: `crates/pro-service/src/claims_processor.rs`
  - Line: 2532

## [2.12.8.0] - 2025-12-22

### Fixed
- **ON CONFLICT Clause Mismatch**: Fixed `ON CONFLICT (service_line_id, diagnosis_id, pointer_sequence)` to match actual unique constraint `uk_line_diag_pointer` which is `(service_line_id, pointer_sequence)`. The previous clause caused batch insert failures with error "there is no unique or exclusion constraint matching the ON CONFLICT specification".
  - File: `crates/pro-service/src/claims_processor.rs`
  - Line: 2551

## [2.12.7.0] - 2025-12-22

### Added
- **Phase 3c: Savepoint Removal**: Removed unnecessary savepoints from `ensure_provider_exists()` function to reduce transaction overhead.

### Optimized
- **Phase 2c: Taxonomy Cache**: Added in-memory taxonomy cache (`Arc<RwLock<HashMap>>`) for provider lookups, reducing database queries for repeated taxonomy codes.
- **Phase 2b: Batch Diagnosis Pointer INSERT**: Replaced N individual INSERT statements with single INSERT...SELECT using UNION ALL for diagnosis pointers.
- **Phase 2a: Batch Diagnosis INSERT**: Replaced N individual INSERT statements with single multi-row INSERT for encounter diagnoses.
- **Phase 1a: Hot Path Logging**: Reduced excessive debug logging in critical processing paths.

### Performance
- Processing rate: ~130 claims/second (up from ~50 claims/second baseline)
