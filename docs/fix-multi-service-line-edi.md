# Fix: Multiple Service Lines Not Captured from EDI 837P Files

## Problem
When an 837P EDI file contains multiple service lines (LX segments) for a single claim, only the first service line is being inserted into `claims.service_line` table.

Example hierarchy showing 2 service lines (LX*1, LX*2):
```
CLM*...*...
HI:ABK*...
LX*1
SV1*HC*99213...
DTP*472...
LX*2
SV1*HC*99214...
DTP*472...
```

## Root Cause
1. **Ingestion is correct**: `claims_importer.rs` stores ALL service lines with prefixes:
   - `service_line_1_procedure_code`, `service_line_1_charge_amount`, etc.
   - `service_line_2_procedure_code`, `service_line_2_charge_amount`, etc.

2. **Processing is wrong**: `claims_processor.rs::import_service_line()` at line 1817 has:
   ```rust
   let prefix = "service_line_1_";  // HARDCODED - only reads first service line!
   ```

3. The comment at line 1814-1816 is incorrect for EDI files:
   ```rust
   // IMPORTANT: Each RawClaim from staging represents ONE service line...
   ```
   This is only true for CSV. For EDI, one raw_claim contains ALL service lines.

## Solution

### Option A: Change import_service_line to accept prefix parameter (Chosen)
- [x] Modify `import_service_line` to accept `service_line_prefix: &str` parameter
- [x] Add helper function `count_service_lines_in_jsonb()` to count service lines in JSONB
- [x] Update `process_encounter_with_service_lines` to iterate through all service lines
- [x] Update deprecated `process_raw_claim` to also handle multiple service lines
- [x] Keep CSV path working (one service line per raw_claim)
- [x] Build succeeded
- [x] Version updated: 2.11.4.0 -> 2.11.5.0
- [x] Installer rebuilt: ProfessionalSMART.msi (11MB)

### Changes Required

#### File: `crates/pro-service/src/claims_processor.rs`

1. Add helper function to count service lines:
```rust
fn count_service_lines(service_line_fields: &HashMap<String, String>) -> usize {
    let mut max_line = 0;
    for key in service_line_fields.keys() {
        if let Some(num_str) = key.strip_prefix("service_line_")
            .and_then(|s| s.split('_').next())
        {
            if let Ok(num) = num_str.parse::<usize>() {
                max_line = max_line.max(num);
            }
        }
    }
    max_line
}
```

2. Modify `import_service_line` signature:
```rust
async fn import_service_line(
    &self,
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    encounter_id: i64,
    organization_id: i64,
    raw_claim: &RawClaim,
    line_number: i32,
    prefix: &str,  // NEW: dynamic prefix like "service_line_1_" or "service_line_2_"
) -> Result<()>
```

3. Update callers to iterate through all service lines in a single raw_claim

## Expected Outcome
- All service lines from EDI 837P files will be properly captured
- CSV processing continues to work as before
- No data loss during claims import

## Testing
- Test with provided sample 837P file with multiple LX segments
- Verify both service lines appear in `claims.service_line` table
