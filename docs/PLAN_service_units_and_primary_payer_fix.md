# Plan: Service units cap + primary payer dropped on dependent claims

Two unrelated import defects, fixed together (v2.16.0.0).

## Issue 1 - Service units clamped to 9999.9

### Symptom
A service line imported with 22,500 units stores `9999.9` in
`claims.service_line.service_unit_count`.

### Root cause
An artificial `9999.9` ceiling exists in five places. The column itself is
`NUMERIC(15,1)`, which already bounds the value; the X12 837P SV104 quantity
element imposes no such cap, and HCPCS/drug (J-code) unit counts legitimately
run far past 9999.9.

- `claims.service_line` CHECK constraint `service_unit_count <= 9999.9`
- `crates/pro-service/src/claims_processor.rs` - clamps the parsed value to
  `9999.9` before insert
- `crates/pro-common/src/constants.rs` - `MAX_SERVICE_UNITS = 9999.9`
- `crates/pro-parser-csv/src/mapping.rs` - CSV `Range` validation max `9999.9`
- `crates/pro-parser-edi/src/validator.rs` - EDI validation rejects `> 9999.9`

### Fix - match the column capacity
- [ ] Migration `077_widen_service_unit_count.sql`: drop the old CHECK, add
      `CHECK (service_unit_count > 0)` - `NUMERIC(15,1)` is the upper bound.
- [ ] Register 077 in `embedded_migrations.rs`; bump `BASELINE_COVERS_THROUGH`
      75->76 is already done; bump 76->77.
- [ ] Append 077 source to `migrations/000_baseline_v2.12.sql`; fix the inline
      constraint at the `service_line` CREATE TABLE; header range -> 001-077.
- [ ] `claims_processor.rs`: remove the upper clamp branch (keep `<= 0 -> 1`).
- [ ] `constants.rs`: `MAX_SERVICE_UNITS` -> `NUMERIC(15,1)` capacity.
- [ ] `mapping.rs`: CSV `Range` max -> column capacity.
- [ ] `validator.rs`: EDI threshold -> column capacity (loud reject past it).

### Existing data
Already-clamped rows are not backfilled - the true value survives in
`staging.raw_claims`, so re-importing the affected file corrects them.

## Issue 2 - Primary payer blank, secondary populated

### Symptom
For some client claims `encounter_view.primary_payer_*` is blank while
`secondary_payer_*` has data.

### Root cause
`parse_claim_info` in `crates/pro-parser-edi/src/loops.rs` decides whether an
`SBR` segment is the subscriber/billing payer (Loop 2000B) or a COB payer
(Loop 2320) by testing `claim.subscriber_relationship_code.is_empty()`.

`subscriber_relationship_code` is SBR02. SBR02 is blank in Loop 2000B whenever
the patient is a dependent carried in a separate Loop 2000C. With SBR02 blank,
the COB Loop 2320 SBR is misdetected as another "first SBR": it overwrites
`payer_responsibility_code` and the COB payer is never appended to
`other_insurance`. The encounter then gets a single `encounter_payer` row
(the billing payer) and `encounter_view` shows a blank primary.

### Fix
- [ ] Add an explicit `subscriber_sbr_seen: bool` flag in `parse_claim_info`
      and use it for first-SBR detection instead of the SBR02 emptiness test.

### Existing data
Mis-imported encounters keep the wrong `payer_responsibility_code` and miss the
COB row; re-importing the affected file corrects them.

## Build
- [ ] CHANGELOG.md entry for v2.16.0.0 (new migration -> minor bump).
- [ ] `.\build-msi.ps1 -Version "2.16.0.0"`.
