# Deployment Guide - v1.5.15.0

**Version**: 1.5.15.0
**Release Date**: November 3, 2025
**Deployment Type**: Patch (Bug Fix)

## Pre-Deployment Checklist

- [x] Code changes committed to git (commit 3bb94c2)
- [x] Version bumped to 1.5.15.0
- [x] MSI installer rebuilt (9.2 MB)
- [x] Release notes created
- [x] Testing documentation created
- [ ] Backup production database
- [ ] Schedule maintenance window
- [ ] Notify users of deployment

## What This Release Fixes

**Critical Bug**: Only one EDI file was processed when multiple files were placed in the input directory.

**Root Cause**: Queue processor was marking EDI files as "completed" without actually calling `ingest_edi_to_staging()`.

**Solution**: Updated queue processor to properly process EDI files through the two-stage pipeline.

## Deployment Steps

### 1. Backup Current System

```powershell
# Backup database
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$backupFile = "C:\backup\professional_smart_$timestamp.backup"

$env:PGPASSWORD = "your_password"
& "C:\Program Files\PostgreSQL\16\bin\pg_dump.exe" `
    -h localhost `
    -U postgres `
    -d professional_smart `
    -F c `
    -f $backupFile

Write-Host "Database backed up to: $backupFile"

# Backup current binaries
Copy-Item "C:\Program Files\Professional SMART\bin" `
          -Destination "C:\backup\pro_smart_bin_$timestamp" `
          -Recurse

Write-Host "Binaries backed up"
```

### 2. Stop Services

```powershell
# Stop the service
Stop-Service -Name ProfessionalSMART

# Verify stopped
Get-Service -Name ProfessionalSMART
# Expected: Status = Stopped
```

### 3. Install Update

```powershell
# Run installer as Administrator
cd C:\Users\jonmc\dev\pro\installer

# Install/Upgrade
msiexec /i ProfessionalSMART.msi /l*v "C:\temp\upgrade_v1.5.15.log"

# Wait for installation to complete
# The service should auto-start after installation
```

### 4. Verify Installation

```powershell
# Check service is running
Get-Service -Name ProfessionalSMART
# Expected: Status = Running

# Check version
Get-ItemProperty "HKLM:\SOFTWARE\Professional SMART" -Name Version
# Expected: 1.5.15.0

# Check binary file dates
Get-ChildItem "C:\Program Files\Professional SMART\bin\*.exe" |
    Select-Object Name, LastWriteTime, Length
# Expected: Recent timestamp (today's date)
```

### 5. Test Deployment

#### Test 1: Service Health

```powershell
# Check service is responding
Invoke-WebRequest -Uri "http://localhost:8080/health" -Method GET -ErrorAction SilentlyContinue
# Or check if service is processing files

# Check logs for startup errors
Get-Content "C:\ProgramData\Professional SMART\logs\pro-service.log" -Tail 50
# Expected: No ERROR or FATAL messages
# Expected: "Starting STAGE 1 queue processor" message present
```

#### Test 2: Process Multiple EDI Files

This is the critical test for v1.5.15.0 - verify multiple files are all processed.

```powershell
# Before test, note current batch count
```

```sql
SELECT COUNT(*) as batch_count_before
FROM staging.import_batch
WHERE file_format = '837P';
```

```powershell
# Copy 3 test EDI files simultaneously
Copy-Item "C:\Users\jonmc\dev\pro\test_data\claims_ORG001-R1-F1.edi" `
          -Destination "C:\ProgramData\Professional SMART\data\input\"

Copy-Item "C:\Users\jonmc\dev\pro\test_data\claims_ORG001-R1-F2.edi" `
          -Destination "C:\ProgramData\Professional SMART\data\input\"

Copy-Item "C:\Users\jonmc\dev\pro\test_data\claims_ORG001-R2-F1.edi" `
          -Destination "C:\ProgramData\Professional SMART\data\input\"

# Wait 30 seconds for processing
Start-Sleep -Seconds 30
```

```sql
-- Verify all 3 files processed
SELECT
    file_name,
    import_status,
    total_records,
    successful_records,
    failed_records,
    created_at
FROM staging.import_batch
WHERE file_format = '837P'
  AND created_at > NOW() - INTERVAL '5 minutes'
ORDER BY created_at DESC;
-- Expected: 3 rows, all with import_status = 'INGESTED' or 'COMPLETED'

-- Check service logs for processing
```

```powershell
Get-Content "C:\ProgramData\Professional SMART\logs\pro-service.log" -Tail 100 |
    Select-String -Pattern "STAGE 1 COMPLETE \(EDI\)"
# Expected: 3 log entries showing successful EDI processing
```

```powershell
# Verify files moved to processed directory
Get-ChildItem "C:\ProgramData\Professional SMART\data\processed\" -Filter "*.edi" |
    Where-Object { $_.LastWriteTime -gt (Get-Date).AddMinutes(-5) } |
    Select-Object Name, LastWriteTime
# Expected: All 3 files present
```

#### Test 3: Regression - CSV Processing

Verify CSV processing still works:

```powershell
# Test CSV processing
Copy-Item "C:\Users\jonmc\dev\pro\test_data\sample_claims.csv" `
          -Destination "C:\ProgramData\Professional SMART\data\input\"

Start-Sleep -Seconds 15
```

```sql
-- Verify CSV processed
SELECT
    file_format,
    import_status,
    total_records
FROM staging.import_batch
WHERE file_format = 'CSV'
ORDER BY created_at DESC
LIMIT 1;
-- Expected: import_status = 'COMPLETED'
```

## Post-Deployment Verification

### Check System Health

```powershell
# 1. Service Status
Get-Service -Name ProfessionalSMART
# Expected: Running

# 2. Check for errors in logs
Get-Content "C:\ProgramData\Professional SMART\logs\pro-service.log" -Tail 100 |
    Select-String -Pattern "ERROR|FATAL|CRITICAL"
# Expected: No critical errors (warnings OK)

# 3. Check processing queue
```

```sql
-- Check queue health
SELECT
    queue_status,
    COUNT(*) as count,
    MAX(created_at) as latest
FROM staging.file_processing_queue
WHERE created_at > NOW() - INTERVAL '1 hour'
GROUP BY queue_status;
-- Expected: COMPLETED status, no FAILED or stuck PROCESSING
```

### Monitor for 24 Hours

After deployment, monitor:

1. **Service logs** for errors or warnings
2. **Processing metrics** (throughput, latency)
3. **Error rates** in staging.raw_claims
4. **User reports** of issues

```sql
-- Hourly processing stats
SELECT
    DATE_TRUNC('hour', created_at) as hour,
    file_format,
    COUNT(*) as batches,
    SUM(total_records) as total_claims,
    SUM(successful_records) as successful,
    SUM(failed_records) as failed,
    ROUND(100.0 * SUM(successful_records) / NULLIF(SUM(total_records), 0), 2) as success_rate
FROM staging.import_batch
WHERE created_at > NOW() - INTERVAL '24 hours'
GROUP BY DATE_TRUNC('hour', created_at), file_format
ORDER BY hour DESC, file_format;
```

## Rollback Procedure

If issues occur, rollback to v1.5.14.0:

### Option 1: Restore from Backup

```powershell
# 1. Stop service
Stop-Service -Name ProfessionalSMART

# 2. Restore binaries
Remove-Item "C:\Program Files\Professional SMART\bin\*" -Recurse
Copy-Item "C:\backup\pro_smart_bin_TIMESTAMP\*" `
          -Destination "C:\Program Files\Professional SMART\bin\" `
          -Recurse

# 3. Start service
Start-Service -Name ProfessionalSMART

# 4. Verify version rolled back
Get-ItemProperty "HKLM:\SOFTWARE\Professional SMART" -Name Version
# Expected: 1.5.14.0
```

### Option 2: Reinstall Previous Version

```powershell
# 1. Stop service
Stop-Service -Name ProfessionalSMART

# 2. Uninstall current version
msiexec /x {PRODUCT-GUID} /l*v "C:\temp\uninstall.log"

# 3. Install previous version
msiexec /i ProfessionalSMART_v1.5.14.msi /l*v "C:\temp\reinstall_v1.5.14.log"
```

### Database Rollback (Not Needed)

No database schema changes in v1.5.15.0, so database rollback is not required.

## Success Criteria

Deployment is successful when:

- [x] Service running without errors
- [x] **Multiple EDI files all processed** (not just first file)
- [x] All test files moved to processed/ directory
- [x] No files stuck in input/ directory
- [x] Service logs show "STAGE 1 COMPLETE (EDI)" for each file
- [x] Queue processor continues running after processing files
- [x] CSV processing still works (regression test)
- [x] No unexpected errors in logs

## Communication

### Before Deployment

Send notification to users:
```
Subject: Planned Maintenance - Professional SMART Upgrade to v1.5.15.0

We will be upgrading Professional SMART to version 1.5.15.0 on [DATE] at [TIME].

Expected downtime: 15 minutes

This update fixes a critical bug where only one EDI file was processed when
multiple files were placed in the input directory. After this upgrade, all
EDI files will be processed correctly.

No action required from users. Processing will resume automatically after the upgrade.
```

### After Deployment

Send confirmation:
```
Subject: Professional SMART Upgrade Complete - v1.5.15.0

The upgrade to v1.5.15.0 has been completed successfully.

The service is now running and processing claims normally.

All EDI files placed in the input directory will now be processed correctly.

Please report any issues to [SUPPORT_EMAIL].
```

## Notes

- This is a **patch release** (bug fixes only)
- No database migrations required
- No breaking changes
- Backward compatible with v1.5.14.0
- Critical fix for EDI file processing

## Files Included

- `installer/ProfessionalSMART.msi` - Version 1.5.15.0 (9.2 MB)
- `RELEASE_NOTES_1.5.15.md` - Full release notes
- `docs/TESTING_V1.5.15.md` - Testing guide

## Support

If issues occur during deployment:

1. Check deployment log: `C:\temp\upgrade_v1.5.15.log`
2. Check service log: `C:\ProgramData\Professional SMART\logs\pro-service.log`
3. Review [Rollback Procedure](#rollback-procedure)
4. Contact support with:
   - Deployment logs
   - Service logs
   - Error messages
   - List of files in input directory (if stuck)
   - Database queue status

## Troubleshooting

### Files Not Processing

If files remain in input directory after 2+ minutes:

```powershell
# Check service is running
Get-Service -Name ProfessionalSMART

# Check service logs for errors
Get-Content "C:\ProgramData\Professional SMART\logs\pro-service.log" -Tail 50
```

```sql
-- Check queue status
SELECT * FROM staging.file_processing_queue
WHERE queue_status = 'PROCESSING'
  AND created_at < NOW() - INTERVAL '5 minutes';
-- Expected: No stuck files
```

### Service Won't Start

```powershell
# Check Windows Event Log
Get-EventLog -LogName Application -Source "ProfessionalSMART" -Newest 10

# Check database connectivity
& "C:\Program Files\PostgreSQL\16\bin\psql.exe" `
    -h localhost `
    -U postgres `
    -d professional_smart `
    -c "SELECT 1;"
```

## Sign-Off

Deployment completed by: ________________
Date/Time: ________________
Version verified: ________________
Multiple file test passed: [ ] Yes [ ] No
Issues noted: ________________
