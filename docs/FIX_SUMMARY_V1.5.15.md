# Fix Summary - v1.5.15.0: Multiple EDI File Processing

**Issue**: Only one EDI file processed when multiple files placed in input directory
**Date Fixed**: November 3, 2025
**Commit**: 3bb94c2

## The Problem

When a user placed multiple EDI 837p files into the data input folder, only the first file would be processed. The subsequent files would remain in the input directory and never get ingested into the staging tables.

### User Report

> "when multiple files are put into the data folder only one file gets processed. the others do not get picked up"

## Root Cause Analysis

### Investigation Steps

1. **Examined file_watcher.rs** (lines 110-154)
   - Found `process_existing_files()` correctly iterates through all files
   - Each file is enqueued via `enqueue_file()`
   - Files are added to `staging.file_processing_queue` table

2. **Examined claims_importer.rs**
   - Confirmed `enqueue_file()` adds all files to queue correctly
   - Queue uses FIFO ordering via sequence numbers

3. **Examined main.rs queue processor** (lines 310-374)
   - **FOUND THE BUG**: Lines 352-361 contained outdated code:

```rust
pro_worker::types::FileFormat::Edi837p => {
    // EDI files: Mark as completed immediately (no Stage 1 ingestion needed)
    // EDI files are processed directly from the file system, not through staging tables
    warn!("EDI file processing not yet implemented in two-stage pipeline: {}", file_path.display());
    warn!("Marking as completed to prevent reprocessing. EDI functionality coming in future release.");

    if let Err(e) = queue_manager.mark_completed(queued_file.queue_id).await {
        error!("Failed to mark EDI queue entry as completed: {}", e);
    }
}
```

**The Bug**: EDI files were being marked as "completed" without actually being processed!

4. **Examined service.rs** (lines 318-346)
   - Found **correct implementation** that calls `ingest_edi_to_staging()`
   - This code path was not being used in console mode

### Root Cause

The queue processor in [main.rs](../crates/pro-service/src/main.rs) had **outdated code** that:
1. Marked EDI files as "completed" immediately
2. Never called `ingest_edi_to_staging()` to actually process them
3. Left files in input directory

This was a **remnant from earlier development** before EDI processing was fully implemented. The correct implementation existed in `service.rs` but wasn't being used.

## The Solution

### Code Changes

Updated [crates/pro-service/src/main.rs](../crates/pro-service/src/main.rs) lines 352-371:

**BEFORE** (v1.5.14.0):
```rust
pro_worker::types::FileFormat::Edi837p => {
    // EDI files: Mark as completed immediately (no Stage 1 ingestion needed)
    warn!("EDI file processing not yet implemented in two-stage pipeline: {}", file_path.display());
    warn!("Marking as completed to prevent reprocessing. EDI functionality coming in future release.");

    if let Err(e) = queue_manager.mark_completed(queued_file.queue_id).await {
        error!("Failed to mark EDI queue entry as completed: {}", e);
    }
}
```

**AFTER** (v1.5.15.0):
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

### What Changed

1. **Added**: Call to `importer_for_processor.ingest_edi_to_staging()`
2. **Added**: Proper error handling for ingestion failures
3. **Added**: Success logging with batch_id and claim count
4. **Removed**: Misleading warning messages about "not yet implemented"
5. **Fixed**: Queue status now reflects actual processing outcome

## Impact

### Before v1.5.15.0
```
Input Directory: [file1.edi, file2.edi, file3.edi]
Queue Processor:
  1. Dequeues file1.edi
  2. Marks as "completed" ❌ (without processing)
  3. Dequeues file2.edi
  4. Marks as "completed" ❌ (without processing)
  5. Dequeues file3.edi
  6. Marks as "completed" ❌ (without processing)

Result: All files still in input/ directory, no claims in staging tables
```

### After v1.5.15.0
```
Input Directory: [file1.edi, file2.edi, file3.edi]
Queue Processor:
  1. Dequeues file1.edi
  2. Calls ingest_edi_to_staging() ✓
  3. Creates batch, parses claims, inserts to staging ✓
  4. Marks as "completed" ✓
  5. File moved to processed/ directory ✓

  6. Dequeues file2.edi
  7. Calls ingest_edi_to_staging() ✓
  8. Creates batch, parses claims, inserts to staging ✓
  9. Marks as "completed" ✓
  10. File moved to processed/ directory ✓

  (continues for all files...)

Result: All claims in staging.raw_claims, all files in processed/ directory
```

## Verification

### Test Scenario

Place 5 EDI files in input directory simultaneously:

```powershell
Copy-Item "test_data\claims_*.edi" -Destination "C:\ProgramData\Professional SMART\data\input\"
```

### Expected Behavior (v1.5.15.0)

**Database**:
```sql
SELECT file_name, import_status, total_records
FROM staging.import_batch
WHERE file_format = '837P'
  AND created_at > NOW() - INTERVAL '5 minutes'
ORDER BY created_at ASC;
```

Expected Output:
```
file_name                      | import_status | total_records
------------------------------|---------------|---------------
claims_ORG001-R1-F1.edi      | INGESTED      | 10
claims_ORG001-R1-F2.edi      | INGESTED      | 10
claims_ORG001-R2-F1.edi      | INGESTED      | 9
claims_ORG002-R1-F1.edi      | INGESTED      | 8
claims_ORG002-R1-F2.edi      | INGESTED      | 10
```

**Service Logs**:
```
[INFO] STAGE 1: Processing queued file: ...\claims_ORG001-R1-F1.edi (queue_id=...)
[INFO] STAGE 1 COMPLETE (EDI): batch_id=..., ingested 10 claims to staging.raw_claims
[INFO] STAGE 1: Processing queued file: ...\claims_ORG001-R1-F2.edi (queue_id=...)
[INFO] STAGE 1 COMPLETE (EDI): batch_id=..., ingested 10 claims to staging.raw_claims
...
```

**File System**:
```
Input Directory: (empty)
Processed Directory: claims_ORG001-R1-F1.edi, claims_ORG001-R1-F2.edi, ...
```

## Technical Details

### Queue Processing Flow

1. **File Watcher** detects files in input directory
2. **Enqueue**: Files added to `staging.file_processing_queue` with status PENDING
3. **Queue Processor** (infinite loop):
   - Dequeue next file (FIFO order)
   - Mark as PROCESSING
   - Route to appropriate handler:
     - CSV → `ingest_file_to_staging()`
     - EDI → `ingest_edi_to_staging()` ✓ **FIXED**
   - On success: Mark as COMPLETED, move to processed/
   - On failure: Mark as FAILED, move to error/
4. **Stage 2 Workers** process claims from staging asynchronously

### Why This Bug Existed

The bug existed because:
1. Early in development, EDI processing wasn't fully implemented
2. Placeholder code was added to prevent reprocessing
3. When EDI processing was later implemented, the placeholder wasn't updated
4. Two code paths existed: `service.rs` (correct) and `main.rs` (incorrect)
5. Console mode used `main.rs` code path with the bug

## Testing Performed

- ✅ Single EDI file processing
- ✅ Multiple EDI files simultaneously (5 files)
- ✅ Multiple EDI files sequentially (added one at a time)
- ✅ Large batch (8 files)
- ✅ Error handling (malformed file doesn't stop queue)
- ✅ Service restart (files enqueued before restart are processed after)
- ✅ Regression: CSV processing still works
- ✅ Regression: DTP date range parsing (v1.5.14 fix) still works

## Related Releases

- **v1.5.13.0**: Fixed facility NPI extraction and subscriber birth date field
- **v1.5.14.0**: Fixed DTP*472 date range parsing (RD8 format)
- **v1.5.15.0**: Fixed multiple EDI file processing ← **This Release**

## Documentation

- [Release Notes](../RELEASE_NOTES_1.5.15.md) - Complete release notes
- [Testing Guide](TESTING_V1.5.15.md) - Comprehensive test cases
- [Deployment Guide](../DEPLOYMENT_v1.5.15.md) - Production deployment procedures

## Lessons Learned

1. **Remove placeholder code once feature is implemented**
   - The "not yet implemented" warning should have been removed

2. **Consolidate duplicate code paths**
   - `service.rs` and `main.rs` had duplicate queue processor logic
   - Should be refactored into shared module

3. **Integration testing needed**
   - Unit tests passed because individual components worked
   - Integration test with multiple files would have caught this

4. **Code review on all branches**
   - Both console mode and service mode should be reviewed together

## Future Work

### Short Term
- Add integration test for multiple file processing
- Refactor queue processor into shared module used by both service.rs and main.rs

### Long Term
- Add automated testing with file system operations
- Add performance benchmarks for batch processing
- Monitor queue processor metrics (throughput, latency)

## Deployment Status

- **Development**: ✅ Tested
- **Staging**: ⏳ Pending
- **Production**: ⏳ Pending

## Support

For issues related to this fix:

1. Check service logs for "STAGE 1 COMPLETE (EDI)" messages
2. Verify files moved to processed/ directory
3. Check queue status in database:
   ```sql
   SELECT queue_status, COUNT(*) FROM staging.file_processing_queue
   WHERE created_at > NOW() - INTERVAL '1 hour'
   GROUP BY queue_status;
   ```

## Conclusion

This was a **critical bug** that prevented EDI file batch processing from working correctly. The fix is **simple** (call the existing ingestion function) but **essential** for production use.

Users can now confidently place multiple EDI files in the input directory and all will be processed correctly in FIFO order.
