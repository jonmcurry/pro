# Testing Guide for v1.5.13.0

**Version**: 1.5.13.0
**Date**: November 3, 2025
**Purpose**: Verify EDI 837p processing bug fixes

## Prerequisites

1. Fresh installation of Professional SMART v1.5.13.0
2. PostgreSQL 12+ installed and running
3. Database `professional_smart` created
4. Test data loaded from `test_data/setup/*.csv`

## Test Setup

### 1. Install v1.5.13.0

```powershell
# As Administrator
cd C:\Users\jonmc\dev\pro\installer
msiexec /i ProfessionalSMART.msi /l*v install.log

# Verify service is running
sc query ProfessionalSMART
# Expected: STATE: 4 RUNNING
```

### 2. Load Master Data

```powershell
# Using GUI Data Loader
& "C:\Program Files\Professional SMART\bin\pro-data-loader-gui.exe"
# Select: C:\Users\jonmc\dev\pro\test_data\setup
# Click "Load Data"
```

Verify data loaded:
```sql
SELECT COUNT(*) FROM claims.organization;  -- Expected: 2
SELECT COUNT(*) FROM claims.region;        -- Expected: 4
SELECT COUNT(*) FROM claims.facility;      -- Expected: 8
SELECT COUNT(*) FROM claims.provider;      -- Expected: varies
```

## Test Case 1: Facility NPI Extraction

**Objective**: Verify facility NPI is correctly extracted from NM1*77 segments

**Test File**: `test_data/claims_ORG001-R1-F1.edi`

### Steps

1. Copy EDI file to input directory:
```powershell
Copy-Item "C:\Users\jonmc\dev\pro\test_data\claims_ORG001-R1-F1.edi" `
          -Destination "C:\ProgramData\Professional SMART\data\input\"
```

2. Wait 10 seconds for service to process

3. Check results:
```sql
-- Get latest batch
SELECT batch_id, file_format, import_status, total_records, successful_records, failed_records
FROM staging.import_batch
WHERE file_format = '837P'
ORDER BY created_at DESC
LIMIT 1;
```

4. Verify facility NPI extracted:
```sql
SELECT
    raw_claim_id,
    encounter_fields->>'patient_control_number' as claim_id,
    encounter_fields->>'facility_npi' as facility_npi,
    processing_status,
    error_message
FROM staging.raw_claims
WHERE batch_id = (
    SELECT batch_id FROM staging.import_batch
    WHERE file_format = '837P'
    ORDER BY created_at DESC
    LIMIT 1
)
ORDER BY row_number;
```

### Expected Results

- All claims have `facility_npi = '7319437180'`
- No "Facility not found:" errors
- Processing status may be FAILED for other reasons, but facility_npi must be populated

### Pass Criteria

- Facility NPI extracted: PASS
- No facility lookup errors: PASS

## Test Case 2: Subscriber Birth Date Field

**Objective**: Verify subscriber_birth_date field is correctly mapped for Stage 2 processing

**Test File**: Same as Test Case 1

### Steps

1. Check subscriber_birth_date field:
```sql
SELECT
    encounter_fields->>'patient_control_number' as claim_id,
    encounter_fields->>'subscriber_last_name' as last_name,
    encounter_fields->>'subscriber_first_name' as first_name,
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
)
ORDER BY row_number;
```

2. Verify field name (not subscriber_date_of_birth):
```sql
SELECT
    COUNT(*) FILTER (WHERE encounter_fields ? 'subscriber_birth_date') as has_birth_date,
    COUNT(*) FILTER (WHERE encounter_fields ? 'subscriber_date_of_birth') as has_date_of_birth
FROM staging.raw_claims
WHERE batch_id = (
    SELECT batch_id FROM staging.import_batch
    WHERE file_format = '837P'
    ORDER BY created_at DESC
    LIMIT 1
);
```

### Expected Results

- `subscriber_birth_date` field populated with dates (e.g., '1975-03-15', '1982-06-22')
- `has_birth_date` = 10 (all claims)
- `has_date_of_birth` = 0 (old field name not used)
- `subscriber_gender` also populated ('M' or 'F')
- No "Missing subscriber_birth_date" errors

### Pass Criteria

- Correct field name used: PASS
- Birth dates populated: PASS
- No birth date validation errors: PASS

## Test Case 3: End-to-End Processing

**Objective**: Verify complete claim processing pipeline

### Steps

1. Check if claims completed Stage 2 processing:
```sql
SELECT
    e.encounter_id,
    e.patient_control_number,
    e.subscriber_last_name,
    e.subscriber_first_name,
    e.date_of_service_from,
    f.facility_code,
    f.npi as facility_npi
FROM claims.encounter e
JOIN claims.facility f ON e.facility_id = f.facility_id
WHERE e.created_at > NOW() - INTERVAL '1 hour'
ORDER BY e.created_at DESC
LIMIT 20;
```

2. Check service lines:
```sql
SELECT
    e.patient_control_number,
    sl.line_number,
    sl.procedure_code,
    sl.charge_amount,
    sl.service_date
FROM claims.service_line sl
JOIN claims.encounter e ON sl.encounter_id = e.encounter_id
WHERE e.created_at > NOW() - INTERVAL '1 hour'
ORDER BY e.patient_control_number, sl.line_number;
```

### Expected Results

For `claims_ORG001-R1-F1.edi`:
- 10 encounters created in claims.encounter
- Facility NPI: 7319437180
- Facility code: ORG001-R1-F1
- 10 service lines in claims.service_line
- Procedure codes: 99214, 85025, 93000, 99215, etc.

### Pass Criteria

- Encounters created: PASS
- Service lines created: PASS
- Facility correctly linked: PASS
- No orphaned claims in staging: PASS

## Test Case 4: Multiple EDI Files

**Objective**: Verify batch processing of multiple files

### Steps

1. Process all test files:
```powershell
Get-ChildItem "C:\Users\jonmc\dev\pro\test_data\claims_*.edi" | ForEach-Object {
    Copy-Item $_.FullName -Destination "C:\ProgramData\Professional SMART\data\input\"
    Start-Sleep -Seconds 15
}
```

2. Check batch summary:
```sql
SELECT
    file_format,
    import_status,
    COUNT(*) as batch_count,
    SUM(total_records) as total_claims,
    SUM(successful_records) as successful,
    SUM(failed_records) as failed
FROM staging.import_batch
WHERE file_format = '837P'
  AND created_at > NOW() - INTERVAL '1 hour'
GROUP BY file_format, import_status;
```

### Expected Results

- 8 EDI files processed
- All files: import_status = 'INGESTED' or 'COMPLETED'
- Total claims: ~63 (varies by test data)
- No file-level failures

### Pass Criteria

- All files processed: PASS
- No file-level errors: PASS

## Regression Testing

### Verify Previous Functionality

1. CSV file processing still works:
```powershell
Copy-Item "C:\Users\jonmc\dev\pro\test_data\sample_claims.csv" `
          -Destination "C:\ProgramData\Professional SMART\data\input\"
```

2. Service stability:
```powershell
# Check service hasn't crashed
sc query ProfessionalSMART

# Check for errors in logs
Get-Content "C:\ProgramData\Professional SMART\logs\pro-service.log" -Tail 50
```

## Cleanup

After testing, clean up test data:

```sql
-- Remove test batches
DELETE FROM staging.raw_claims
WHERE batch_id IN (
    SELECT batch_id FROM staging.import_batch
    WHERE file_format = '837P'
      AND created_at > NOW() - INTERVAL '1 hour'
);

DELETE FROM staging.file_processing_queue
WHERE created_at > NOW() - INTERVAL '1 hour';

DELETE FROM staging.import_batch
WHERE file_format = '837P'
  AND created_at > NOW() - INTERVAL '1 hour';

-- Optional: Remove test encounters
DELETE FROM claims.service_line
WHERE encounter_id IN (
    SELECT encounter_id FROM claims.encounter
    WHERE created_at > NOW() - INTERVAL '1 hour'
);

DELETE FROM claims.encounter
WHERE created_at > NOW() - INTERVAL '1 hour';
```

## Issue Reporting

If any test fails, collect the following information:

1. Service logs:
```powershell
Copy-Item "C:\ProgramData\Professional SMART\logs\*" -Destination "C:\temp\logs\" -Recurse
```

2. Failed claims:
```sql
\copy (SELECT * FROM staging.raw_claims WHERE processing_status = 'FAILED' AND batch_id IN (SELECT batch_id FROM staging.import_batch WHERE created_at > NOW() - INTERVAL '1 hour')) TO 'C:\temp\failed_claims.csv' CSV HEADER;
```

3. Database state:
```sql
-- Save to file
\o C:\temp\db_state.txt
SELECT * FROM staging.import_batch WHERE created_at > NOW() - INTERVAL '1 hour';
SELECT queue_status, COUNT(*) FROM staging.file_processing_queue WHERE created_at > NOW() - INTERVAL '1 hour' GROUP BY queue_status;
\o
```

## Success Criteria Summary

All test cases must pass:
- Test Case 1: Facility NPI Extraction - PASS
- Test Case 2: Subscriber Birth Date Field - PASS
- Test Case 3: End-to-End Processing - PASS
- Test Case 4: Multiple EDI Files - PASS
- Regression Testing - PASS

## Notes

- Test data files in `test_data/` have been corrected with proper NM1*77 format
- The fix affects only EDI 837p processing, CSV processing is unchanged
- No database migrations required for this release
- Service automatically restarts during upgrade installation
