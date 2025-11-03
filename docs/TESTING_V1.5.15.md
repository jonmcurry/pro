# Testing Guide for v1.5.15.0

**Version**: 1.5.15.0
**Date**: November 3, 2025
**Purpose**: Verify multiple EDI file processing bug fix

## What's Being Tested

This release fixes a critical bug where only one EDI file was processed when multiple files were placed in the input directory. We need to verify that:

1. Multiple EDI files are all processed (not just the first one)
2. Files are processed in FIFO order
3. Each file creates a separate batch
4. All files are moved to `processed/` directory
5. Queue processor continues running after processing multiple files

## Prerequisites

1. Professional SMART v1.5.15.0 installed
2. PostgreSQL running with `professional_smart` database
3. Test data loaded (organizations, regions, facilities, providers)
4. Service running in console mode OR as Windows service

## Test Environment Setup

### Option A: Console Mode (Recommended for Testing)

```powershell
# Stop Windows service if running
Stop-Service -Name ProfessionalSMART -ErrorAction SilentlyContinue

# Set environment variables
$env:DATABASE_URL = "postgresql://postgres:ClearToFly1@localhost/professional_smart"
$env:INPUT_DIR = "C:\ProgramData\Professional SMART\data\input"

# Clear any existing files from input directory
Remove-Item "C:\ProgramData\Professional SMART\data\input\*.edi" -ErrorAction SilentlyContinue

# Run in console mode
cd "C:\Program Files\Professional SMART\bin"
.\pro-service.exe console
```

### Option B: Windows Service Mode

```powershell
# Ensure service is running
Start-Service -Name ProfessionalSMART

# Verify status
Get-Service -Name ProfessionalSMART
# Expected: Status = Running
```

## Test Case 1: Process Multiple EDI Files Simultaneously

**Objective**: Verify all files are processed when placed in input directory at once

### Setup

```sql
-- Note the current batch count before test
SELECT COUNT(*) as batch_count_before
FROM staging.import_batch
WHERE file_format = '837P';
```

### Execution

```powershell
# Copy 5 EDI files to input directory simultaneously
Copy-Item "C:\Users\jonmc\dev\pro\test_data\claims_ORG001-R1-F1.edi" `
          -Destination "C:\ProgramData\Professional SMART\data\input\"
Copy-Item "C:\Users\jonmc\dev\pro\test_data\claims_ORG001-R1-F2.edi" `
          -Destination "C:\ProgramData\Professional SMART\data\input\"
Copy-Item "C:\Users\jonmc\dev\pro\test_data\claims_ORG001-R2-F1.edi" `
          -Destination "C:\ProgramData\Professional SMART\data\input\"
Copy-Item "C:\Users\jonmc\dev\pro\test_data\claims_ORG002-R1-F1.edi" `
          -Destination "C:\ProgramData\Professional SMART\data\input\"
Copy-Item "C:\Users\jonmc\dev\pro\test_data\claims_ORG002-R1-F2.edi" `
          -Destination "C:\ProgramData\Professional SMART\data\input\"

# Wait for processing (allow 60 seconds)
Start-Sleep -Seconds 60
```

### Verification

```sql
-- Check that 5 new batches were created
SELECT
    batch_id,
    file_name,
    file_format,
    import_status,
    total_records,
    successful_records,
    failed_records,
    created_at
FROM staging.import_batch
WHERE file_format = '837P'
  AND created_at > NOW() - INTERVAL '5 minutes'
ORDER BY created_at ASC;
-- Expected: 5 rows, one for each file
-- Expected: All have import_status = 'INGESTED' or 'COMPLETED'
```

```powershell
# Verify files moved to processed directory
Get-ChildItem "C:\ProgramData\Professional SMART\data\processed\" -Filter "*.edi" |
    Where-Object { $_.LastWriteTime -gt (Get-Date).AddMinutes(-5) } |
    Select-Object Name, LastWriteTime

# Expected: All 5 EDI files present with recent timestamp
```

```powershell
# Check service logs for processing messages
Get-Content "C:\ProgramData\Professional SMART\logs\pro-service.log" -Tail 200 |
    Select-String -Pattern "STAGE 1 COMPLETE \(EDI\)"

# Expected: 5 log entries showing successful processing
```

### Pass Criteria

- ✓ All 5 files created separate batches in staging.import_batch
- ✓ All batches show import_status = 'INGESTED' or 'COMPLETED'
- ✓ All 5 files moved to processed/ directory
- ✓ No files remaining in input/ directory
- ✓ Service logs show 5 successful "STAGE 1 COMPLETE (EDI)" messages

## Test Case 2: Process Files One at a Time

**Objective**: Verify queue processor continues running and processes files added sequentially

### Execution

```powershell
# Add first file
Copy-Item "C:\Users\jonmc\dev\pro\test_data\claims_ORG001-R1-F1.edi" `
          -Destination "C:\ProgramData\Professional SMART\data\input\"

Start-Sleep -Seconds 15

# Add second file
Copy-Item "C:\Users\jonmc\dev\pro\test_data\claims_ORG001-R1-F2.edi" `
          -Destination "C:\ProgramData\Professional SMART\data\input\"

Start-Sleep -Seconds 15

# Add third file
Copy-Item "C:\Users\jonmc\dev\pro\test_data\claims_ORG001-R2-F1.edi" `
          -Destination "C:\ProgramData\Professional SMART\data\input\"

Start-Sleep -Seconds 15
```

### Verification

```sql
-- All 3 files should have been processed
SELECT
    file_name,
    import_status,
    created_at
FROM staging.import_batch
WHERE file_format = '837P'
  AND created_at > NOW() - INTERVAL '5 minutes'
ORDER BY created_at ASC;
-- Expected: 3 rows with sequential timestamps
```

### Pass Criteria

- ✓ All 3 files processed successfully
- ✓ Files processed in order they were added
- ✓ No files stuck in input directory

## Test Case 3: Large Batch - 8 Files

**Objective**: Stress test with all available EDI test files

### Execution

```powershell
# Copy all test EDI files at once
Get-ChildItem "C:\Users\jonmc\dev\pro\test_data\claims_*.edi" | ForEach-Object {
    Copy-Item $_.FullName -Destination "C:\ProgramData\Professional SMART\data\input\"
}

# Wait for processing (allow 2 minutes for larger batch)
Start-Sleep -Seconds 120
```

### Verification

```sql
-- Check all files processed
SELECT
    COUNT(*) as files_processed,
    SUM(total_records) as total_claims,
    MIN(created_at) as first_file,
    MAX(created_at) as last_file
FROM staging.import_batch
WHERE file_format = '837P'
  AND created_at > NOW() - INTERVAL '5 minutes';
-- Expected: files_processed = 8 (or however many test files exist)
```

```sql
-- Check processing status distribution
SELECT
    import_status,
    COUNT(*) as count
FROM staging.import_batch
WHERE file_format = '837P'
  AND created_at > NOW() - INTERVAL '5 minutes'
GROUP BY import_status;
-- Expected: All files either INGESTED or COMPLETED
```

### Pass Criteria

- ✓ All EDI test files processed
- ✓ No files with FAILED status (unless test data has errors)
- ✓ All files moved to processed/ directory
- ✓ Total claims match expected count from test files

## Test Case 4: Error Handling - Malformed File

**Objective**: Verify queue processor continues after encountering error

### Setup

Create a malformed EDI file:

```powershell
# Create invalid EDI file
@"
ISA*00*          *00*          *ZZ*SENDER         *ZZ*RECEIVER       *250103*1200*^*00501*000000001*0*P*:~
INVALID_SEGMENT_HERE
"@ | Out-File "C:\ProgramData\Professional SMART\data\input\bad_file.edi" -Encoding ASCII
```

### Execution

```powershell
# Add valid file before bad file
Copy-Item "C:\Users\jonmc\dev\pro\test_data\claims_ORG001-R1-F1.edi" `
          -Destination "C:\ProgramData\Professional SMART\data\input\good1.edi"

Start-Sleep -Seconds 10

# Bad file added
# (already created above)

Start-Sleep -Seconds 10

# Add valid file after bad file
Copy-Item "C:\Users\jonmc\dev\pro\test_data\claims_ORG001-R1-F2.edi" `
          -Destination "C:\ProgramData\Professional SMART\data\input\good2.edi"

Start-Sleep -Seconds 10
```

### Verification

```sql
-- Check processing results
SELECT
    file_name,
    import_status,
    error_message,
    created_at
FROM staging.import_batch
WHERE created_at > NOW() - INTERVAL '5 minutes'
ORDER BY created_at ASC;
-- Expected:
--   good1.edi: INGESTED/COMPLETED
--   bad_file.edi: FAILED (with error message)
--   good2.edi: INGESTED/COMPLETED
```

```powershell
# Check error directory for bad file
Get-ChildItem "C:\ProgramData\Professional SMART\data\error\" -Filter "bad_file*"
# Expected: bad_file.edi present with error suffix
```

### Pass Criteria

- ✓ Valid files before error processed successfully
- ✓ Bad file marked as FAILED with error message
- ✓ Bad file moved to error/ directory
- ✓ Valid files after error processed successfully
- ✓ Queue processor did not stop due to error

## Test Case 5: Service Restart - Queue Persistence

**Objective**: Verify files enqueued before restart are processed after restart

### Execution

```powershell
# Stop service (or Ctrl+C if console mode)
Stop-Service -Name ProfessionalSMART

# Add files while service is stopped
Copy-Item "C:\Users\jonmc\dev\pro\test_data\claims_ORG001-R1-F1.edi" `
          -Destination "C:\ProgramData\Professional SMART\data\input\"
Copy-Item "C:\Users\jonmc\dev\pro\test_data\claims_ORG001-R1-F2.edi" `
          -Destination "C:\ProgramData\Professional SMART\data\input\"

# Restart service
Start-Service -Name ProfessionalSMART

# Wait for processing
Start-Sleep -Seconds 30
```

### Verification

```sql
-- Files should be processed after service restart
SELECT
    file_name,
    import_status,
    created_at
FROM staging.import_batch
WHERE file_format = '837P'
  AND created_at > NOW() - INTERVAL '5 minutes'
ORDER BY created_at ASC;
-- Expected: 2 rows with timestamps after service restart
```

### Pass Criteria

- ✓ Files present during startup are processed
- ✓ File watcher calls process_existing_files() on startup
- ✓ All files moved to processed/ directory

## Performance Metrics

During testing, collect these metrics:

```sql
-- Average processing time per file
SELECT
    AVG(EXTRACT(EPOCH FROM (updated_at - created_at))) as avg_processing_seconds,
    MIN(EXTRACT(EPOCH FROM (updated_at - created_at))) as min_processing_seconds,
    MAX(EXTRACT(EPOCH FROM (updated_at - created_at))) as max_processing_seconds
FROM staging.import_batch
WHERE file_format = '837P'
  AND created_at > NOW() - INTERVAL '1 hour'
  AND import_status IN ('INGESTED', 'COMPLETED');
```

**Expected Performance**:
- Small files (10 claims): 1-5 seconds
- Medium files (50 claims): 5-15 seconds
- Large files (100+ claims): 15-30 seconds

## Regression Testing

Verify previous functionality still works:

### CSV File Processing

```powershell
Copy-Item "C:\Users\jonmc\dev\pro\test_data\sample_claims.csv" `
          -Destination "C:\ProgramData\Professional SMART\data\input\"

Start-Sleep -Seconds 15
```

```sql
-- Verify CSV processed
SELECT * FROM staging.import_batch
WHERE file_format = 'CSV'
  AND created_at > NOW() - INTERVAL '5 minutes';
-- Expected: CSV file processed successfully
```

### DTP Date Range Parsing (v1.5.14 fix)

```sql
-- Verify date ranges still parsed correctly
SELECT
    encounter_fields->>'date_of_service_from' as dos_from,
    encounter_fields->>'date_of_service_to' as dos_to
FROM staging.raw_claims
WHERE batch_id IN (
    SELECT batch_id FROM staging.import_batch
    WHERE file_format = '837P'
      AND created_at > NOW() - INTERVAL '1 hour'
)
  AND encounter_fields->>'date_of_service_to' IS NOT NULL
LIMIT 5;
-- Expected: Both from and to dates populated for date range segments
```

## Cleanup

After testing, clean up test data:

```sql
-- Remove test batches and claims
DELETE FROM staging.raw_claims
WHERE batch_id IN (
    SELECT batch_id FROM staging.import_batch
    WHERE created_at > NOW() - INTERVAL '1 hour'
);

DELETE FROM staging.file_processing_queue
WHERE created_at > NOW() - INTERVAL '1 hour';

DELETE FROM staging.import_batch
WHERE created_at > NOW() - INTERVAL '1 hour';

-- Optional: Remove test encounters
DELETE FROM claims.service_line
WHERE encounter_id IN (
    SELECT encounter_id FROM claims.encounter
    WHERE created_at > NOW() - INTERVAL '1 hour'
);

DELETE FROM claims.encounter
WHERE created_at > NOW() - INTERVAL '1 hour';
```

```powershell
# Clear processed and error directories
Remove-Item "C:\ProgramData\Professional SMART\data\processed\*.edi"
Remove-Item "C:\ProgramData\Professional SMART\data\error\*.edi"
```

## Issue Reporting

If any test fails, collect:

1. **Service Logs**:
   ```powershell
   Copy-Item "C:\ProgramData\Professional SMART\logs\*" -Destination "C:\temp\test_failure_logs\" -Recurse
   ```

2. **Database State**:
   ```sql
   -- Export failed batches
   \copy (SELECT * FROM staging.import_batch WHERE created_at > NOW() - INTERVAL '1 hour') TO 'C:\temp\test_batches.csv' CSV HEADER;

   -- Export queue status
   \copy (SELECT * FROM staging.file_processing_queue WHERE created_at > NOW() - INTERVAL '1 hour') TO 'C:\temp\test_queue.csv' CSV HEADER;
   ```

3. **File System State**:
   ```powershell
   Get-ChildItem "C:\ProgramData\Professional SMART\data\input\" |
       Select-Object Name, Length, LastWriteTime |
       Export-Csv "C:\temp\stuck_files.csv"
   ```

## Success Criteria Summary

All test cases must pass:
- ✓ Test Case 1: Multiple files simultaneously - PASS
- ✓ Test Case 2: Files one at a time - PASS
- ✓ Test Case 3: Large batch (8 files) - PASS
- ✓ Test Case 4: Error handling - PASS
- ✓ Test Case 5: Service restart persistence - PASS
- ✓ Regression: CSV processing - PASS
- ✓ Regression: DTP date ranges - PASS

## Notes

- The fix changes only the queue processor logic in main.rs
- No database schema changes
- No configuration changes required
- Service automatically processes files on startup via process_existing_files()
- Queue processor runs in infinite loop with 1-second sleep when no files available
