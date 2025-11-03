# Release v1.5.13.0 - Summary

**Release Date**: November 3, 2025
**Type**: Patch Release
**Commit**: 1f7e323

## What Was Fixed

### Issue 1: Facility NPI Not Extracted from EDI Files
**Problem**: All EDI claims failed with "Facility not found:" errors because the facility NPI wasn't being extracted.

**Root Cause**: Test EDI files had malformed NM1*77 segments missing the name_suffix field, causing the parser to read wrong element indices.

**Solution**: Updated all test EDI files to include proper 9-element NM1 segments per X12 837p specification.

**Files Changed**:
- `test_data/claims_*.edi` (all 7 files)

### Issue 2: Subscriber Birth Date Field Name Mismatch
**Problem**: All EDI claims failed Stage 2 processing with "Missing subscriber_birth_date" even though DMG segments were present.

**Root Cause**: Stage 1 stored field as `subscriber_date_of_birth` but Stage 2 expected `subscriber_birth_date`.

**Solution**: Standardized field name to `subscriber_birth_date` in claims importer.

**Files Changed**:
- `crates/pro-service/src/claims_importer.rs`

## Installation

### For Testing/Development
```powershell
cd C:\Users\jonmc\dev\pro\installer
msiexec /i ProfessionalSMART.msi /l*v install.log
```

### For Production
1. Stop existing service (if upgrading)
2. Run installer
3. Verify service restarted
4. Test with sample EDI file

## Verification Steps

After installation, verify both fixes work:

```sql
-- 1. Check facility NPI extraction
SELECT
    encounter_fields->>'facility_npi' as facility_npi,
    COUNT(*) as count
FROM staging.raw_claims
WHERE batch_id = (SELECT MAX(batch_id) FROM staging.import_batch WHERE file_format = '837P')
GROUP BY encounter_fields->>'facility_npi';
-- Expected: facility_npi populated (e.g., '7319437180')

-- 2. Check subscriber birth date field
SELECT
    encounter_fields->>'subscriber_birth_date' as birth_date,
    processing_status,
    error_message
FROM staging.raw_claims
WHERE batch_id = (SELECT MAX(batch_id) FROM staging.import_batch WHERE file_format = '837P')
LIMIT 5;
-- Expected: birth_date populated, no "Missing subscriber_birth_date" errors
```

## Documentation

- [Full Release Notes](../RELEASE_NOTES_1.5.13.md)
- [Testing Guide](TESTING_V1.5.13.md)
- [EDI Processing Documentation](TESTING_EDI_PROCESSING.md)

## Git Information

**Commit Message**:
```
Release v1.5.13.0 - Fix EDI 837p processing bugs

This release fixes two critical bugs preventing successful EDI claim processing
```

**Branch**: main
**Commit Hash**: 1f7e323

## Build Information

- **Compiler**: rustc 1.x.x
- **WiX Toolset**: 3.14.1.8722
- **MSI Size**: 9.2 MB
- **Build Time**: November 3, 2025 08:47
- **Auto-rebuild**: Yes (post-commit hook triggered)

## Testing Checklist

Before deploying to production:

- [ ] Install v1.5.13.0 on test system
- [ ] Load master data (organizations, regions, facilities, providers)
- [ ] Process test EDI file (claims_ORG001-R1-F1.edi)
- [ ] Verify facility_npi field populated
- [ ] Verify subscriber_birth_date field populated
- [ ] Verify no "Facility not found" errors
- [ ] Verify no "Missing subscriber_birth_date" errors
- [ ] Check encounters created in claims.encounter table
- [ ] Check service_lines created in claims.service_line table
- [ ] Process multiple EDI files
- [ ] Verify CSV processing still works (regression test)
- [ ] Check service logs for errors
- [ ] Verify service is stable and running

## Known Limitations

- Requires service restart to load new binary (Windows service limitation)
- Test data must be loaded before processing EDI files
- No database migrations required

## Next Steps

After deploying v1.5.13.0:

1. Monitor service logs for any unexpected errors
2. Verify production EDI files process successfully
3. Check processing metrics (throughput, error rates)
4. Gather feedback from users
5. Plan next release based on priority issues

## Support

If issues occur:
1. Check service logs: `C:\ProgramData\Professional SMART\logs\`
2. Run diagnostic queries from Testing Guide
3. Collect failed claims data
4. File issue in repository with logs and error details

## Compliance

This release maintains compliance with:
- X12 837p specification Version 005010X222A1
- CMS 1500 claim form requirements
- HIPAA transaction standards

## Performance Impact

No performance degradation expected. Changes are:
- Bug fixes only (no new features)
- Single field name change (minimal overhead)
- Test data corrections (no production impact)

Build times and runtime performance remain unchanged.
