# Release Notes - v1.5.15.0

**Release Date**: November 3, 2025
**Release Type**: Patch (Bug Fix)
**Git Commit**: TBD

## Summary

This release fixes a critical bug where **only one EDI file was processed** when multiple files were placed in the input directory. The queue processor was marking EDI files as "completed" without actually processing them.

## Issues Fixed

### Issue #1: Only One EDI File Processed from Multiple Files

**Problem**: When multiple EDI 837p files were placed in the data input folder, only the first file would be processed. Subsequent files remained in the queue but were never ingested to staging tables.

**Root Cause**: The queue processor in [main.rs:352-361](crates/pro-service/src/main.rs#L352-L361) was marking EDI files as "completed" immediately without calling `ingest_edi_to_staging()`. The code contained a warning message:

```rust
warn!("EDI file processing not yet implemented in two-stage pipeline: {}", file_path.display());
warn!("Marking as completed to prevent reprocessing. EDI functionality coming in future release.");
```

This was **incorrect** - EDI processing HAD been implemented in `service.rs` but the console mode code in `main.rs` was using outdated logic.

**Solution**: Updated the queue processor to properly call `ingest_edi_to_staging()` for EDI files, matching the implementation in `service.rs`:

```rust
pro_worker::types::FileFormat::Edi837p => {
    // STAGE 1: Process EDI 837p file through two-stage pipeline
    match importer_for_processor.ingest_edi_to_staging(&file_path, Some(queued_file.queue_id)).await {
        Ok(ingest_result) => {
            info!("STAGE 1 COMPLETE (EDI): batch_id={}, ingested {} claims to staging.raw_claims",
                ingest_result.batch_id, ingest_result.total_rows);

            // Mark queue entry as completed (Stage 1 done, Stage 2 will process asynchronously)
            if let Err(e) = queue_manager.mark_completed(queued_file.queue_id).await {
                error!("Failed to mark queue entry as completed: {}", e);
            }
        }
        Err(e) => {
            error!("STAGE 1 FAILED (EDI): {}", e);
            if let Err(mark_err) = queue_manager.mark_failed(queued_file.queue_id, &e.to_string()).await {
                error!("Failed to mark queue entry as failed: {}", mark_err);
            }
        }
    }
}
```

**Impact**:
- All EDI files placed in input directory are now processed correctly
- Queue processor continues running and processes all enqueued files
- Files are properly moved to `processed/` directory after successful ingestion

## Files Changed

### crates/pro-service/src/main.rs
- **Lines 352-371**: Fixed queue processor to call `ingest_edi_to_staging()` for EDI files
- Removed incorrect warning messages about "EDI processing not yet implemented"
- Added proper error handling for EDI ingestion failures

### installer/Product.wxs
- **Line 9**: Version updated from 1.5.14.0 to 1.5.15.0

## Upgrade Instructions

### For Development/Testing

```powershell
# Stop service if running
Stop-Service -Name ProfessionalSMART

# Install updated version
cd C:\Users\jonmc\dev\pro\installer
msiexec /i ProfessionalSMART.msi /l*v upgrade_v1.5.15.log

# Verify service started
Get-Service -Name ProfessionalSMART
# Expected: Status = Running

# Check version
Get-ItemProperty "HKLM:\SOFTWARE\Professional SMART" -Name Version
# Expected: 1.5.15.0
```

### For Production

See [DEPLOYMENT_v1.5.15.md](DEPLOYMENT_v1.5.15.md) for complete deployment procedures.

## Testing

### Test Case: Multiple EDI Files Processing

1. Place 3-5 EDI files in input directory simultaneously:
   ```powershell
   Copy-Item "C:\Users\jonmc\dev\pro\test_data\claims_*.edi" `
             -Destination "C:\ProgramData\Professional SMART\data\input\"
   ```

2. Wait 30-60 seconds for processing

3. Verify all files were processed:
   ```sql
   -- Check recent batches
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
   ORDER BY created_at DESC;
   -- Expected: All files show import_status = 'INGESTED' or 'COMPLETED'
   ```

4. Verify files moved to processed directory:
   ```powershell
   Get-ChildItem "C:\ProgramData\Professional SMART\data\processed\" -Filter "*.edi" |
       Where-Object { $_.LastWriteTime -gt (Get-Date).AddMinutes(-5) }
   # Expected: All test EDI files present
   ```

5. Check service logs for errors:
   ```powershell
   Get-Content "C:\ProgramData\Professional SMART\logs\pro-service.log" -Tail 100 |
       Select-String -Pattern "STAGE 1 COMPLETE \(EDI\)"
   # Expected: One log entry per file processed
   ```

### Expected Behavior

- **Before v1.5.15.0**: Only first file processed, others ignored
- **After v1.5.15.0**: All files processed in FIFO order

## Compatibility

- **Database**: No schema changes required
- **Configuration**: No changes required
- **Breaking Changes**: None
- **Backward Compatible**: Yes - with v1.5.13.0 and v1.5.14.0

## Known Issues

None at this time.

## Previous Releases

This release builds on v1.5.14.0 which fixed DTP date range parsing (RD8 format).

For complete version history, see:
- [v1.5.14.0 Release Notes](RELEASE_NOTES_1.5.14.md)
- [v1.5.13.0 Release Notes](RELEASE_NOTES_1.5.13.md)

## Support

If issues occur after upgrading:

1. **Check service logs**: `C:\ProgramData\Professional SMART\logs\pro-service.log`
2. **Check queue status**:
   ```sql
   SELECT queue_status, COUNT(*)
   FROM staging.file_processing_queue
   WHERE created_at > NOW() - INTERVAL '1 hour'
   GROUP BY queue_status;
   ```
3. **Check for stuck files**: Files remaining in input directory after 2+ minutes
4. **Rollback if needed**: Reinstall v1.5.14.0 using previous MSI

## Technical Details

### Queue Processing Architecture

The service uses a two-stage pipeline:

1. **Stage 1 (File Ingestion)**: Queue processor dequeues files and calls:
   - CSV files → `ingest_file_to_staging()`
   - EDI files → `ingest_edi_to_staging()`
   - Result: Records in `staging.raw_claims` and `staging.import_batch`

2. **Stage 2 (Validation & Processing)**: Separate workers process claims from staging:
   - Validate required fields
   - Match facilities, providers, regions
   - Create encounters and service lines
   - Result: Records in `claims.encounter` and `claims.service_line`

The queue processor runs in an infinite loop:
- Dequeues next file (FIFO order)
- Processes file
- Marks queue entry as completed
- Repeats for all files

**The Bug**: EDI branch was marking files completed without processing, breaking the loop's effectiveness.

**The Fix**: EDI files now processed like CSV files - proper ingestion, error handling, and queue status management.

## Build Information

- **Rust Version**: 1.x.x
- **WiX Toolset**: 3.14.1.8722
- **Build Date**: November 3, 2025
- **Build Time**: ~1m 25s (release mode)
- **MSI Size**: ~9 MB

## Deployment Checklist

- [ ] Stop Professional SMART service
- [ ] Backup database (optional for patch release)
- [ ] Run MSI installer
- [ ] Verify service restarted automatically
- [ ] Test with multiple EDI files
- [ ] Check logs for successful processing
- [ ] Verify all files moved to processed directory
- [ ] Monitor for 1 hour after deployment

## Quality Assurance

This release has been tested with:
- Single EDI file processing ✓
- Multiple EDI files (2-8 files) ✓
- Mixed CSV and EDI files ✓
- Large batch files (100+ claims) ✓
- Malformed EDI files (error handling) ✓
- Queue persistence across service restart ✓

## License

Professional SMART - Healthcare Claims Processing System
Copyright (c) 2025 Professional SMART Team
