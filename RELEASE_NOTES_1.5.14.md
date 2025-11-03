# Release Notes - Professional SMART v1.5.14.0

**Release Date**: November 3, 2025
**Type**: Patch Release (Bug Fix)

## Summary

This release fixes a critical bug in EDI 837p date parsing that caused processing failures when encountering date range formats (RD8) in DTP*472 segments.

## Bug Fix

### DTP*472 Date Range Parsing Error

**Issue**: Processing EDI/837p files containing DTP segments with RD8 format (date ranges) threw parsing errors and caused claim processing to fail.

**Example Segment**: `DTP*472*RD8*20250619-20250619~`

**Error**: The parser attempted to parse the date range string "20250619-20250619" as a single 8-digit date, causing an "Invalid date format" error.

**Root Cause**: The DTP*472 parsing code at line 421 in loops.rs did not check the `date_time_period_format_qualifier` field before calling `parse_date()`. It assumed all DTP*472 segments used D8 format (single date) and did not handle RD8 format (date range).

**Impact**:
- All claims with date range formats failed to process
- Common in claims where service spans multiple days
- Affected both claim-level and service line-level dates

**Files Changed**:
- `crates/pro-parser-edi/src/loops.rs:421-444`

**Code Change**:
```rust
// Before (v1.5.13):
if dtp.date_time_qualifier == "472" {
    if let Some(ref mut line) = current_service_line {
        line.service_date_from = dtp.parse_date()?;
    } else {
        claim.date_of_service_from = dtp.parse_date()?;
    }
}

// After (v1.5.14):
if dtp.date_time_qualifier == "472" {
    // Check date format: D8 (single date) or RD8 (date range)
    if dtp.date_time_period_format_qualifier == "D8" {
        if let Some(ref mut line) = current_service_line {
            line.service_date_from = dtp.parse_date()?;
        } else {
            claim.date_of_service_from = dtp.parse_date()?;
        }
    } else if dtp.date_time_period_format_qualifier == "RD8" {
        // Date range format: CCYYMMDD-CCYYMMDD
        let (from, to) = dtp.parse_date_range()?;
        if let Some(ref mut line) = current_service_line {
            line.service_date_from = from;
            line.service_date_to = Some(to);
        } else {
            claim.date_of_service_from = from;
            claim.date_of_service_to = Some(to);
        }
    }
}
```

**Resolution**: Added format qualifier check to distinguish between D8 (single date) and RD8 (date range) formats, calling the appropriate parsing function for each.

## Technical Details

### DTP Segment Format

The DTP (Date or Time or Period) segment in X12 837p can use different format qualifiers:

- **D8**: Single date in CCYYMMDD format (e.g., "20250619")
- **RD8**: Date range in CCYYMMDD-CCYYMMDD format (e.g., "20250619-20250619")

DTP*472 specifically indicates "Service Date" and can appear at:
- Claim level (Loop 2300): Indicates date of service for entire claim
- Service line level (Loop 2400): Indicates date for specific service line

### Parsing Logic

The fix now properly handles both formats:

1. **D8 Format**: Parse as single date using `parse_date()`
   - Sets `service_date_from` or `date_of_service_from`

2. **RD8 Format**: Parse as date range using `parse_date_range()`
   - Sets both `from` and `to` dates
   - Extracts start and end dates from the range string

### Date Range Parsing

The `parse_date_range()` function splits on '-' delimiter:
```
Input:  "20250619-20250619"
Output: (NaiveDate(2025-06-19), NaiveDate(2025-06-19))
```

## Verification

After applying fix:
- DTP*472*D8*20250619~ - Single date parsed correctly
- DTP*472*RD8*20250619-20250619~ - Date range parsed correctly
- Both claim-level and service-line level dates handled
- Date range `to` field populated in database

## Testing

### Test Case: Single Date (D8)
```
DTP*472*D8*20250619~
```
**Expected**:
- `service_date_from` = 2025-06-19
- `service_date_to` = NULL

### Test Case: Date Range (RD8)
```
DTP*472*RD8*20250619-20250625~
```
**Expected**:
- `service_date_from` = 2025-06-19
- `service_date_to` = 2025-06-25

### SQL Verification
```sql
-- Check date range handling
SELECT
    encounter_fields->>'patient_control_number' as claim_id,
    encounter_fields->>'date_of_service_from' as dos_from,
    encounter_fields->>'date_of_service_to' as dos_to,
    service_line_fields->>'service_line_1_date_from' as svc_from,
    service_line_fields->>'service_line_1_date_to' as svc_to,
    processing_status
FROM staging.raw_claims
WHERE batch_id = (
    SELECT batch_id FROM staging.import_batch
    WHERE file_format = '837P'
    ORDER BY created_at DESC
    LIMIT 1
);
```

## Installation

### Fresh Install
```powershell
msiexec /i ProfessionalSMART.msi /l*v install.log
```

### Upgrade from v1.5.13.0 or earlier
```powershell
msiexec /i ProfessionalSMART.msi /l*v upgrade.log
```

The installer will automatically:
1. Stop the Windows service
2. Update binaries
3. Restart the service

## Known Issues

None

## Compatibility

- PostgreSQL 12+
- Windows Server 2016+, Windows 10/11
- .NET Framework 4.8+
- Backward compatible with v1.5.13.0

## Migration Notes

No database migrations required. Existing data is not affected.

## Files Modified

- `crates/pro-parser-edi/src/loops.rs` - Fixed DTP*472 date parsing
- `installer/Product.wxs` - Version updated to 1.5.14.0
- `installer/ProfessionalSMART.msi` - Rebuilt with fix

## Build Information

- Build Date: November 3, 2025
- Compiler: rustc 1.x.x
- WiX Toolset: 3.14.1.8722
- MSI Size: 9.2 MB
- Build Time: 1m 43s

## Compliance

Conforms to:
- X12 837p specification Version 005010X222A1
- ASC X12N TR3 Implementation Guide
- DTP segment format qualifiers per X12 standards

## Support

For issues or questions, please file an issue in the project repository.

## Upgrade Impact

- **Risk Level**: Low (isolated bug fix)
- **Testing Required**: Process claims with date range formats
- **Downtime**: ~5 minutes during service restart
- **Rollback**: Simple (reinstall v1.5.13.0 if needed)

## Previous Releases

- v1.5.13.0 - Fixed facility NPI extraction and subscriber birth date field
- v1.5.12.0 - Fixed Windows service installation
- v1.5.11.0 - Added service line parsing
