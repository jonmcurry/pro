# Fix EDI 837 Claims facility_code Column Not Populated

## Issue
The `staging.raw_claims.facility_code` column is not being populated when processing EDI 837 claims, even though the `facility_code` value is correctly computed and stored in the `encounter_fields` JSONB column.

## Root Cause Analysis
In `crates/pro-service/src/claims_importer.rs`:

**CSV Ingestion (lines 536-565)** - CORRECT:
- Includes `facility_code` in the INSERT column list
- Extracts facility_code from encounter_fields and binds it to the column

**EDI Ingestion (lines 977-1001)** - MISSING:
- The INSERT statement does NOT include `facility_code` column
- The `facility_code` variable IS computed (lines 870-887)
- The `facility_code` IS added to `encounter_fields` JSON (line 889)
- But it is NOT bound to the dedicated database column

## Impact
- FIFO ordering by facility may not work correctly for EDI claims
- Any queries relying on the `facility_code` column will return NULL for EDI claims
- Stage 2 processing may not correctly associate claims with facilities

## Fix Plan

- [x] 1. Modify the EDI INSERT statement (line 977-991) to include `facility_code` column
- [x] 2. Add binding for the `facility_code` variable (already computed at line 870-887)
- [x] 3. Update version number (2.8.7.0 -> 2.8.8.0 - bug fix)
- [x] 4. Rebuild the installer (ProfessionalSMART.msi - 11MB)

## Code Changes Required

### File: `crates/pro-service/src/claims_importer.rs`

Change the EDI INSERT from:
```rust
sqlx::query(
    r#"
    INSERT INTO staging.raw_claims (
        batch_id,
        queue_id,
        encounter_fields,
        service_line_fields,
        diagnosis_fields,
        row_number,
        processing_status,
        date_of_service_from
    )
    VALUES ($1, $2, $3, $4, $5, $6, 'PENDING', $7)
    "#
)
.bind(batch_id)
.bind(queue_id)
.bind(&encounter_fields_json)
.bind(&service_line_fields_json)
.bind(&diagnosis_fields_json)
.bind(row_number)
.bind(claim.date_of_service_from)
```

To:
```rust
sqlx::query(
    r#"
    INSERT INTO staging.raw_claims (
        batch_id,
        queue_id,
        encounter_fields,
        service_line_fields,
        diagnosis_fields,
        row_number,
        facility_code,
        processing_status,
        date_of_service_from
    )
    VALUES ($1, $2, $3, $4, $5, $6, $7, 'PENDING', $8)
    "#
)
.bind(batch_id)
.bind(queue_id)
.bind(&encounter_fields_json)
.bind(&service_line_fields_json)
.bind(&diagnosis_fields_json)
.bind(row_number)
.bind(&facility_code)
.bind(claim.date_of_service_from)
```

## Testing
- Process an EDI 837 file through the system
- Verify `staging.raw_claims.facility_code` is populated
- Verify value matches what's in `encounter_fields->>'facility_code'`
