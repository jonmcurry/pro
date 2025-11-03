# Deployment Guide - v1.5.13.0

**Version**: 1.5.13.0
**Release Date**: November 3, 2025
**Deployment Type**: Patch (Bug Fix)

## Pre-Deployment Checklist

- [x] Code changes committed to git (commit 1f7e323)
- [x] Version bumped to 1.5.13.0
- [x] MSI installer rebuilt
- [x] Release notes created
- [x] Testing documentation updated
- [ ] Backup production database
- [ ] Schedule maintenance window
- [ ] Notify users of deployment

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
msiexec /i ProfessionalSMART.msi /l*v "C:\temp\upgrade_v1.5.13.log"

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
# Expected: 1.5.13.0

# Check binary file dates
Get-ChildItem "C:\Program Files\Professional SMART\bin\*.exe" |
    Select-Object Name, LastWriteTime, Length
# Expected: Recent timestamp (today's date)
```

### 5. Test Deployment

#### Test 1: Service Health
```powershell
# Check service is responding
Invoke-WebRequest -Uri "http://localhost:8080/health" -Method GET
# Or check if service is processing files

# Check logs for startup errors
Get-Content "C:\ProgramData\Professional SMART\logs\pro-service.log" -Tail 50
```

#### Test 2: Process Test EDI File
```sql
-- Before copying file, note current batch count
SELECT COUNT(*) FROM staging.import_batch;
```

```powershell
# Copy test file
Copy-Item "C:\Users\jonmc\dev\pro\test_data\claims_ORG001-R1-F2.edi" `
          -Destination "C:\ProgramData\Professional SMART\data\input\"

# Wait 15 seconds
Start-Sleep -Seconds 15
```

```sql
-- Check file was processed
SELECT
    file_format,
    import_status,
    total_records,
    successful_records,
    failed_records,
    error_message
FROM staging.import_batch
ORDER BY created_at DESC
LIMIT 1;
-- Expected: import_status = 'INGESTED' or 'COMPLETED', no errors

-- Check facility NPI extracted (Fix #1)
SELECT
    encounter_fields->>'patient_control_number' as claim_id,
    encounter_fields->>'facility_npi' as facility_npi,
    processing_status
FROM staging.raw_claims
WHERE batch_id = (
    SELECT batch_id FROM staging.import_batch
    ORDER BY created_at DESC LIMIT 1
)
LIMIT 5;
-- Expected: facility_npi populated (e.g., '4967786886')

-- Check subscriber birth date field (Fix #2)
SELECT
    encounter_fields->>'subscriber_birth_date' as birth_date,
    encounter_fields->>'subscriber_gender' as gender,
    processing_status,
    error_message
FROM staging.raw_claims
WHERE batch_id = (
    SELECT batch_id FROM staging.import_batch
    ORDER BY created_at DESC LIMIT 1
)
LIMIT 5;
-- Expected: birth_date populated, no "Missing subscriber_birth_date" errors
```

#### Test 3: Regression - CSV Processing
```powershell
# Test CSV processing still works
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

# 2. Check for errors in logs
Get-Content "C:\ProgramData\Professional SMART\logs\pro-service.log" -Tail 100 |
    Select-String -Pattern "ERROR|FATAL|CRITICAL"

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
    COUNT(*) as batches,
    SUM(total_records) as total_claims,
    SUM(successful_records) as successful,
    SUM(failed_records) as failed,
    ROUND(100.0 * SUM(successful_records) / NULLIF(SUM(total_records), 0), 2) as success_rate
FROM staging.import_batch
WHERE created_at > NOW() - INTERVAL '24 hours'
GROUP BY DATE_TRUNC('hour', created_at)
ORDER BY hour DESC;
```

## Rollback Procedure

If issues occur, rollback to previous version:

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
```

### Option 2: Reinstall Previous Version

```powershell
# 1. Stop service
Stop-Service -Name ProfessionalSMART

# 2. Uninstall current version
msiexec /x {PRODUCT-GUID} /l*v "C:\temp\uninstall.log"

# 3. Install previous version
msiexec /i ProfessionalSMART_v1.5.12.msi /l*v "C:\temp\reinstall_v1.5.12.log"
```

### Database Rollback (if needed)

```powershell
# Only if database schema changed (not needed for v1.5.13)
$env:PGPASSWORD = "your_password"
& "C:\Program Files\PostgreSQL\16\bin\pg_restore.exe" `
    -h localhost `
    -U postgres `
    -d professional_smart `
    -c `
    "C:\backup\professional_smart_TIMESTAMP.backup"
```

## Success Criteria

Deployment is successful when:

- [x] Service running without errors
- [x] Test EDI file processed successfully
- [x] Facility NPI extracted correctly
- [x] Subscriber birth date field populated
- [x] No "Facility not found" errors
- [x] No "Missing subscriber_birth_date" errors
- [x] CSV processing still works
- [x] No regression issues reported
- [x] Processing throughput maintained
- [x] No unexpected errors in logs

## Communication

### Before Deployment
Send notification to users:
```
Subject: Planned Maintenance - Professional SMART Upgrade

We will be upgrading Professional SMART to version 1.5.13.0 on [DATE] at [TIME].

Expected downtime: 15 minutes

This update fixes critical EDI processing issues:
- Facility NPI extraction from 837p files
- Subscriber birth date validation

No action required from users. Processing will resume automatically after the upgrade.
```

### After Deployment
Send confirmation:
```
Subject: Professional SMART Upgrade Complete - v1.5.13.0

The upgrade to v1.5.13.0 has been completed successfully.

The service is now running and processing claims normally.

Please report any issues to [SUPPORT_EMAIL].
```

## Notes

- This is a **patch release** (bug fixes only)
- No database migrations required
- No breaking changes
- Backward compatible with v1.5.12
- Test data files have been corrected (if using test data, copy new versions)

## Files Included

- `installer/ProfessionalSMART.msi` - Version 1.5.13.0
- `RELEASE_NOTES_1.5.13.md` - Full release notes
- `docs/TESTING_V1.5.13.md` - Testing guide
- `test_data/claims_*.edi` - Corrected test files

## Support

If issues occur during deployment:

1. Check deployment log: `C:\temp\upgrade_v1.5.13.log`
2. Check service log: `C:\ProgramData\Professional SMART\logs\pro-service.log`
3. Review [Rollback Procedure](#rollback-procedure)
4. Contact support with:
   - Deployment logs
   - Service logs
   - Error messages
   - System information

## Sign-Off

Deployment completed by: ________________
Date/Time: ________________
Version verified: ________________
Tests passed: [ ] Yes [ ] No
Issues noted: ________________
