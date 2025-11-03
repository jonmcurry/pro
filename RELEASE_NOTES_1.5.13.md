# Release Notes - Professional SMART v1.5.13.0

**Release Date**: November 3, 2025
**Type**: Patch Release (Bug Fixes)

## Summary

This release fixes two critical issues in EDI 837p processing that prevented claims from being successfully processed through the complete pipeline.

## Bug Fixes

### 1. EDI Test Data - Missing NM1 Name Suffix Field

**Issue**: Test EDI files had malformed NM1*77 (Service Facility) segments missing the name_suffix field (NM107), causing the parser to read incorrect element indices for identification_code_qualifier and identification_code.

**Impact**: Facility NPI was not being extracted from EDI files, causing all claims to fail with "Facility not found:" errors.

**Files Changed**:
- `test_data/*.edi` - Added missing asterisk to create proper 4-field empty name section

**Example**:
```
Before: NM1*77*2*North Region Medical Center****XX*7319437180~
After:  NM1*77*2*North Region Medical Center*****XX*7319437180~
```

**Resolution**: Updated all test EDI files to conform to X12 837p specification requiring NM101-NM109 elements, with empty fields properly represented.

### 2. Subscriber Birth Date Field Name Mismatch

**Issue**: Field name inconsistency between Stage 1 (ingestion) and Stage 2 (processing):
- Stage 1 stored as: `subscriber_date_of_birth`
- Stage 2 expected: `subscriber_birth_date`

**Impact**: All claims failed Stage 2 processing with error "Missing subscriber_birth_date" even when DMG segments were present and correctly parsed.

**Files Changed**:
- `crates/pro-service/src/claims_importer.rs:702`

**Code Change**:
```rust
// Before:
encounter_fields.insert("subscriber_date_of_birth".to_string(),
    serde_json::json!(claim.subscriber_date_of_birth.map(|d| d.format("%Y-%m-%d").to_string()).unwrap_or_default()));

// After:
encounter_fields.insert("subscriber_birth_date".to_string(),
    serde_json::json!(claim.subscriber_date_of_birth.map(|d| d.format("%Y-%m-%d").to_string()).unwrap_or_default()));
```

**Resolution**: Standardized field name to `subscriber_birth_date` to match claims_processor expectations.

## Technical Details

### Root Cause Analysis

**Issue #1 - NM1 Segment Format**:
The X12 837p specification defines NM1 segments with 9 data elements:
- NM101: Entity Identifier Code
- NM102: Entity Type Qualifier
- NM103: Name Last or Organization Name
- NM104: Name First
- NM105: Name Middle
- NM106: Name Prefix
- NM107: Name Suffix
- NM108: Identification Code Qualifier
- NM109: Identification Code

For organization entities (NM102=2), fields NM104-NM107 should all be empty, represented as `****` (4 consecutive delimiters creating 4 empty fields). The test data had only `***` (3 delimiters), causing the parser to read NM107 as the qualifier and NM108 as the code, with NM109 missing entirely.

**Issue #2 - Field Name Inconsistency**:
The EDI parser correctly extracted subscriber birth date from DMG segments and stored it in the Claim struct as `subscriber_date_of_birth`. However, when mapping to JSONB for staging.raw_claims, the field was stored with the same name. The claims_processor expected `subscriber_birth_date` (without "of"), causing validation to fail.

## Verification

After applying both fixes:
- Facility NPI correctly extracted: `7319437180`
- Subscriber birth date correctly mapped: `1982-06-22`, `1969-12-04`, etc.
- DMG segment parsing confirmed working (gender field also populated)
- Claims ready for Stage 2 processing once service restarted with updated binary

## Installation

### Fresh Install
Run the installer as administrator:
```powershell
msiexec /i ProfessionalSMART.msi /l*v install.log
```

### Upgrade from v1.5.12.0 or earlier
The installer will automatically upgrade the existing installation:
```powershell
msiexec /i ProfessionalSMART.msi /l*v upgrade.log
```

The Windows service will be stopped, binaries updated, and service restarted automatically.

## Testing

### Verify Facility NPI Extraction
```sql
-- Check staging claims have facility_npi populated
SELECT
    encounter_fields->>'patient_control_number' as claim_id,
    encounter_fields->>'facility_npi' as facility_npi,
    processing_status
FROM staging.raw_claims
WHERE batch_id = (
    SELECT batch_id FROM staging.import_batch
    WHERE file_format = '837P'
    ORDER BY created_at DESC
    LIMIT 1
);

-- Expected: facility_npi = '7319437180' for test files
```

### Verify Subscriber Birth Date
```sql
-- Check staging claims have subscriber_birth_date populated
SELECT
    encounter_fields->>'patient_control_number' as claim_id,
    encounter_fields->>'subscriber_birth_date' as birth_date,
    encounter_fields->>'subscriber_gender' as gender,
    processing_status,
    error_message
FROM staging.raw_claims
WHERE batch_id = (
    SELECT batch_id FROM staging.import_batch
    WHERE file_format = '837P'
    ORDER BY created_at DESC
    LIMIT 1
);

-- Expected: birth_date populated (e.g., '1982-06-22'), no "Missing subscriber_birth_date" errors
```

## Known Issues

None

## Compatibility

- PostgreSQL 12+
- Windows Server 2016+, Windows 10/11
- .NET Framework 4.8+
- Requires administrator privileges for installation

## Migration Notes

No database migrations required. Existing data is not affected.

## Files Modified

- `installer/Product.wxs` - Version updated to 1.5.13.0
- `crates/pro-service/src/claims_importer.rs` - Fixed field name for subscriber_birth_date
- `test_data/claims_*.edi` - Fixed NM1*77 segment format (all 8 test files)

## Build Information

- Build Date: November 3, 2025 08:47
- Compiler: rustc 1.x.x
- WiX Toolset: 3.14.1.8722
- MSI Size: 9.2 MB

## Support

For issues or questions, please file an issue in the project repository.
