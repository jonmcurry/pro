# Two-Stage Processing Pipeline - Implementation Progress

## Status: Phase 1-4 Complete (Sequential Completion Integrated)

Date: 2025-10-22
Last Updated: 2025-10-22 (Sequential Completion Integration)

## Completed Work

### Phase 1: Database Schema ✅ COMPLETE
- [x] Created migration `023_create_raw_claims_table.sql`
- [x] Added `staging.raw_claims` table with JSONB fields for flexible storage
- [x] Created indexes for FIFO ordering and performance:
  - `idx_raw_claims_pending` - For Stage 2 to find next batch
  - `idx_raw_claims_batch` - Tracking by batch
  - `idx_raw_claims_queue` - Tracking by queue
  - `idx_raw_claims_fifo` - FIFO compliance within facilities
  - `idx_raw_claims_stale` - Recovery after crashes
- [x] Updated `staging.import_batch` constraint to support INGESTING/INGESTED states
- [x] Added `processing_stage` column to `staging.processing_metrics` (IMPORT/INGEST/PROCESS)
- [x] Created monitoring view: `staging.vw_raw_claims_status`
- [x] Created recovery function: `staging.recover_stale_raw_claims()`
- [x] Tested migration successfully on local database

**Files Created:**
- `C:\Users\jonmc\dev\pro\migrations\023_create_raw_claims_table.sql`

### Phase 2: Stage 1 Implementation (File Ingestion) ✅ COMPLETE
- [x] Created new method: `ClaimsImporter::ingest_file_to_staging()`
- [x] Parses CSV/EDI files to `Vec<ParsedRow>` (reusing existing parser logic)
- [x] Serializes parsed data to JSONB format:
  - `encounter_fields` - Patient, subscriber, facility, payer data
  - `service_line_fields` - Procedure codes, charges, dates
  - `diagnosis_fields` - ICD diagnosis codes
- [x] Batch inserts to `staging.raw_claims` (1000 per commit)
- [x] Updates batch status: INGESTING → INGESTED
- [x] Logs INGEST performance metric with `processing_stage = 'INGEST'`
- [x] Returns `IngestResult` with batch_id, total_rows, ingested_at

**Code Changes:**
- `crates/pro-service/src/claims_importer.rs`:
  - Added `ingest_file_to_staging()` method (lines 221-520)
  - Added `log_processing_metric_with_stage()` helper (lines 522-582)
  - Added `IngestResult` struct (lines 1292-1298)
  - Marked `import_file_with_queue()` as LEGACY

### Phase 3: Stage 2 Implementation (Claim Processing) ✅ COMPLETE
- [x] Created new module: `pro-claims-processor`
- [x] Implemented `ClaimsProcessor::process_pending_claims()`:
  - Queries `staging.raw_claims` WHERE status = PENDING (FIFO order)
  - Marks as PROCESSING (with FOR UPDATE SKIP LOCKED)
  - Deserializes JSONB to claim structures
  - Validates and inserts to `claims.encounter` OR `staging.import_error_log`
  - Updates `raw_claims.processing_status` atomically (COMPLETED/FAILED)
  - Batch commits (1000 per transaction)
- [x] Facility lookup with HashMap caching (performance optimization)
- [x] Updates batch status: INGESTED → PROCESSING → COMPLETED/PARTIAL/FAILED
- [x] Logs PROCESS metric with `processing_stage = 'PROCESS'`
- [x] Returns `ProcessResult` with total_processed, successful, failed counts

**Files Created:**
- `crates/pro-service/src/claims_processor.rs` (664 lines)

**Code Changes:**
- `crates/pro-service/src/main.rs`:
  - Added `mod claims_processor;` (line 26)

### Phase 4: Multi-Worker Sequential Completion ✅ COMPLETE
- [x] Created migration 024: `migrations/024_add_batch_sequence_tracking.sql`
  - Added `batch_sequence_number` column to `staging.raw_claims`
  - Created `staging.batch_sequences` tracking table
  - Added monitoring view `vw_sequence_processing_status`
  - Created `detect_stuck_sequences()` function
  - Added `processing_configuration` table with defaults
- [x] Created `batch_sequencer.rs` module:
  - `SequenceCounter` - Atomic counter for thread-safe sequence generation
  - `SequencedBatchAcquirer` - Single-threaded batch acquisition with sequence assignment
  - `SequentialCompletionManager` - Buffers out-of-order results and commits in sequence
  - `SequencedBatch` and `BatchResult` structs
- [x] Updated `claims_processor.rs`:
  - Added `process_sequenced_batch()` method for multi-worker processing
- [x] Integrated into `main.rs` (console mode):
  - Replaced single Stage 2 processor with multi-worker architecture
  - Configuration: 8 workers (default), 750 batch size
  - Channel-based communication (broadcast + mpsc)
  - Spawned SequencedBatchAcquirer, 8 workers, and SequentialCompletionManager
  - Updated shutdown sequence
- [x] Integrated into `service.rs` (Windows service mode):
  - Same multi-worker architecture as console mode
- [x] Code compiles successfully with Rust 1.83.0
- [x] Build time: 1 minute 14 seconds
- [x] Warnings are expected (legacy single-worker methods preserved for rollback)

## Architecture Overview

### Two-Stage Pipeline Flow

```
Input File (.csv/.edi)
    |
    v
File Watcher (detects file)
    |
    v
Enqueue to staging.file_processing_queue (QUEUED)
    |
    v
Queue Manager (dequeue FIFO)
    |
    v
╔═══════════════════════════════════════════════════════════╗
║                    STAGE 1: INGESTION                     ║
║  ClaimsImporter::ingest_file_to_staging()                ║
╠═══════════════════════════════════════════════════════════╣
║  1. Parse CSV/EDI → Vec<ParsedRow>                       ║
║  2. Serialize to JSONB (encounter/service/diagnosis)     ║
║  3. Batch INSERT to staging.raw_claims (1000/commit)     ║
║  4. Update batch: INGESTING → INGESTED                   ║
║  5. Log INGEST metric (Stage 1 throughput)               ║
╚═══════════════════════════════════════════════════════════╝
    |
    | staging.raw_claims (processing_status = PENDING)
    |
    v
╔═══════════════════════════════════════════════════════════╗
║                   STAGE 2: PROCESSING                     ║
║  ClaimsProcessor::process_pending_claims()               ║
╠═══════════════════════════════════════════════════════════╣
║  1. SELECT raw_claims WHERE status = PENDING (FIFO)      ║
║  2. Mark as PROCESSING (FOR UPDATE SKIP LOCKED)          ║
║  3. Deserialize JSONB → claim structures                 ║
║  4. Validate + Facility lookup (cached)                  ║
║  5. IF VALID:                                            ║
║       INSERT claims.encounter                            ║
║       UPDATE raw_claims SET status = COMPLETED           ║
║     IF INVALID:                                          ║
║       INSERT staging.import_error_log                    ║
║       UPDATE raw_claims SET status = FAILED              ║
║  6. Batch COMMIT (1000 per transaction)                  ║
║  7. Update batch: PROCESSING → COMPLETED/PARTIAL         ║
║  8. Log PROCESS metric (Stage 2 throughput)              ║
╚═══════════════════════════════════════════════════════════╝
    |
    v
claims.encounter (successful) + staging.import_error_log (failed)
```

### Performance Optimizations

1. **Batch Processing**: 1000 claims per commit in both stages
2. **JSONB Storage**: Efficient PostgreSQL native format
3. **Indexed Queries**: Partial indexes on PENDING status
4. **Facility Caching**: HashMap cache to avoid repeated DB lookups
5. **FOR UPDATE SKIP LOCKED**: Enables concurrent Stage 2 workers
6. **Separate Metrics**: Independent throughput tracking for each stage

### Database Tables

**New Table: `staging.raw_claims`**
- `raw_claim_id` (UUID, PK)
- `batch_id` (UUID, FK → staging.import_batch)
- `queue_id` (UUID, FK → staging.file_processing_queue)
- `encounter_fields` (JSONB) - Patient/subscriber/facility/payer
- `service_line_fields` (JSONB) - Procedures/charges/dates
- `diagnosis_fields` (JSONB) - ICD codes
- `row_number` (INTEGER) - Original file row
- `facility_code` (TEXT) - For FIFO ordering
- `processing_status` (TEXT) - PENDING/PROCESSING/COMPLETED/FAILED
- `ingested_at` (TIMESTAMPTZ) - Stage 1 timestamp
- `processed_at` (TIMESTAMPTZ) - Stage 2 timestamp
- `error_message` (TEXT) - If failed
- `date_of_service_from` (DATE) - For FIFO ordering

**Updated Tables:**
- `staging.import_batch` - New statuses: INGESTING, INGESTED, PROCESSING
- `staging.processing_metrics` - New column: `processing_stage` (INGEST/PROCESS)

**New Views:**
- `staging.vw_raw_claims_status` - Monitoring dashboard

**New Functions:**
- `staging.recover_stale_raw_claims()` - Recovery after crashes

## Remaining Work

### Phase 4: Orchestration ✅ COMPLETE (Sequential Completion with 8 Workers)

### Phase 5: Performance Metrics (Partial - DB schema done, runtime integration pending)
- [x] Added `processing_stage` column to `staging.processing_metrics`
- [ ] Verify Stage 1 throughput logging works in practice
- [ ] Verify Stage 2 throughput logging works in practice
- [ ] Calculate end-to-end latency (file arrival to final insertion)
- [ ] Add metrics to dashboard/monitoring UI

### Phase 6: Testing (Not Started)
- [ ] Unit tests for Stage 1 ingestion
- [ ] Unit tests for Stage 2 processing
- [ ] Integration test: full pipeline with sample CSV
- [ ] Performance test: 10,000 claims in 15 seconds (666.67 claims/sec)
- [ ] Edge case: partial failures (some claims valid, some invalid)
- [ ] Edge case: file with all invalid claims
- [ ] Edge case: Stage 2 restarts mid-processing (idempotency)
- [ ] Edge case: concurrent Stage 2 workers processing same batch

### Phase 7: Deployment (Not Started)
- [ ] Rebuild Rust binaries with new logic
- [ ] Test fresh install with migration 023
- [ ] Test upgrade from existing installation
- [ ] Rebuild MSI installer with updated binaries
- [ ] Update user documentation

## Performance Target

**Requirement**: 10,000 claims / 15 seconds = **666.67 claims/sec**

**Current Architecture Support:**
- Batch processing: 1000 claims per commit
- Facility caching: Reduces DB round-trips
- JSONB storage: Fast PostgreSQL native format
- Indexed queries: O(log n) lookups on PENDING status

**Expected Performance:**
- Stage 1 (Ingestion): **>1000 claims/sec** (minimal validation, pure inserts)
- Stage 2 (Processing): **666+ claims/sec** (meets target with caching)

**Bottleneck Analysis:**
- Stage 1: File I/O + JSON serialization (fast)
- Stage 2: Facility lookups (cached), encounter inserts (batched)
- Overall: Should meet or exceed 666.67 claims/sec target

## Success Criteria

- [x] Database schema created and tested ✅
- [x] Stage 1 code implemented ✅
- [x] Stage 2 code implemented ✅
- [x] Code compiles successfully ✅
- [ ] Performance: 10,000 claims processed in 15 seconds (pending testing)
- [ ] Metrics: Stage 1 and Stage 2 throughput captured accurately (pending testing)
- [ ] No data loss: All claims accounted for (encounter OR error log) (pending testing)
- [ ] FIFO compliance: Claims processed in correct order (pending testing)
- [ ] Idempotency: Stage 2 can restart without duplicate inserts (pending testing)
- [ ] All existing tests pass (pending)
- [ ] New integration tests cover two-stage flow (pending)

## Next Steps

1. **Orchestration Integration**:
   - Modify service startup to call `ingest_file_to_staging()` instead of `import_file_with_queue()`
   - Create background task to poll and process PENDING raw_claims
   - Add configuration for two-stage vs legacy pipeline

2. **Testing**:
   - Create test harness with sample CSV files
   - Measure actual throughput for both stages
   - Verify 10k/15s performance target

3. **Deployment**:
   - Rebuild binaries
   - Test MSI upgrade scenario
   - Update installer

## Timeline Estimate

- ~~Phase 1 (Schema): 1 hour~~ ✅ DONE
- ~~Phase 2 (Stage 1): 3-4 hours~~ ✅ DONE
- ~~Phase 3 (Stage 2): 4-5 hours~~ ✅ DONE
- **Phase 4 (Orchestration): 2-3 hours** ← NEXT
- Phase 5 (Metrics): 2 hours
- Phase 6 (Testing): 3-4 hours
- Phase 7 (Deployment): 1-2 hours

**Remaining**: ~5-7 hours of testing + deployment

## Files Modified/Created

### Created:
- `docs/TWO_STAGE_PROCESSING_PIPELINE.md` - Design document
- `docs/TWO_STAGE_PIPELINE_PROGRESS.md` - This status document
- `docs/AEGIS_VS_PROFESSIONAL_SMART_COMPARISON.md` - Aegis architecture comparison
- `docs/FIFO_ORDERING_ANALYSIS.md` - FIFO ordering analysis
- `docs/SEQUENTIAL_COMPLETION_INTEGRATION.md` - Integration guide
- `migrations/023_create_raw_claims_table.sql` - Database schema (Stage 1 + 2)
- `migrations/024_add_batch_sequence_tracking.sql` - Database schema (Sequential Completion)
- `crates/pro-service/src/claims_processor.rs` - Stage 2 processor (NEW MODULE)
- `crates/pro-service/src/batch_sequencer.rs` - Sequential completion system (NEW MODULE)

### Modified:
- `crates/pro-service/src/claims_importer.rs`:
  - Added `ingest_file_to_staging()` - Stage 1 method
  - Added `log_processing_metric_with_stage()` helper
  - Added `IngestResult` struct
  - Kept `import_file_with_queue()` as LEGACY
- `crates/pro-service/src/main.rs`:
  - Registered `mod claims_processor;` and `mod batch_sequencer;`
  - Replaced single Stage 2 processor with 8-worker sequential completion system (console mode)
  - Added environment variable configuration (STAGE2_WORKER_COUNT, BATCH_SIZE)
  - Implemented channel-based worker coordination
  - Updated shutdown sequence for multi-component architecture
- `crates/pro-service/src/service.rs`:
  - Same multi-worker sequential completion integration (Windows service mode)

## Rollback Strategy

If two-stage approach causes issues:
1. Legacy `ClaimsImporter::import_file_with_queue()` still exists unchanged
2. Can add feature flag: `USE_TWO_STAGE_PIPELINE` (default: true)
3. Switch back to single-stage via configuration or environment variable
4. Migration 023 is additive (doesn't break existing tables)
5. Can disable Stage 2 worker and only run Stage 1 (claims stay in raw_claims for manual processing)

## Notes

- **No breaking changes**: Legacy single-stage method preserved
- **Backward compatible**: Existing code still works
- **Incremental deployment**: Can test Stage 1 first, then Stage 2
- **Observability**: Separate metrics for each stage enable bottleneck analysis
- **Scalability**: FOR UPDATE SKIP LOCKED enables multiple Stage 2 workers
- **Recovery**: Stale PROCESSING claims can be recovered to PENDING state
