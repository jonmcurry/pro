# Two-Stage Processing Pipeline Refactor

## Problem Statement

Current processing pipeline reads EDI/CSV files and directly processes claims to either:
- **Staging database** (if claim fails validation)
- **Encounters database** (if claim succeeds)

This direct file-to-database approach can cause slowdowns because:
1. File parsing and database insertion happen in same transaction
2. No decoupling between ingestion and validation
3. Cannot prioritize failed claims for retry without re-parsing files
4. Limited ability to parallelize processing across multiple workers

## Solution: Two-Stage Pipeline

### Stage 1: File Ingestion (Fast)
**Input**: EDI/CSV files from `C:\Program Files\Professional SMART\data\input`
**Output**: `staging.raw_claims` table (all claims stored as-is)
**Performance Target**: Maximize throughput - minimal validation, just parsing and storage

### Stage 2: Claim Processing (Validated)
**Input**: `staging.raw_claims` table
**Output**:
- `claims.encounter` (valid claims)
- `staging.import_error_log` (invalid claims)
**Performance Target**: Maintain 10,000 claims / 15 seconds (666.67 claims/sec)

## Performance Requirements

- **Overall Target**: 10,000 claims / 15 seconds = 666.67 claims/sec
- **Metrics Capture**:
  - Stage 1 throughput (file read + parse + insert to raw_claims)
  - Stage 2 throughput (raw_claims processing to encounters/errors)
  - End-to-end latency (file arrival to final insertion)
- **No Performance Degradation**: Two-stage approach must maintain or improve current performance

## Architecture Design

### New Table: `staging.raw_claims`

```sql
CREATE TABLE staging.raw_claims (
    raw_claim_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    batch_id UUID NOT NULL REFERENCES staging.import_batch(batch_id),
    queue_id UUID NOT NULL REFERENCES staging.file_processing_queue(queue_id),

    -- Original parsed data (JSON for flexibility)
    encounter_fields JSONB NOT NULL,
    service_line_fields JSONB,
    diagnosis_fields JSONB,

    -- Metadata
    row_number INTEGER NOT NULL,
    facility_code TEXT,

    -- Processing status
    processing_status TEXT NOT NULL DEFAULT 'PENDING',
        -- PENDING: Not yet processed
        -- PROCESSING: Currently being processed
        -- COMPLETED: Successfully inserted to encounters
        -- FAILED: Validation failed, logged to error table

    -- Timestamps
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMPTZ,

    -- Error tracking (if failed)
    error_message TEXT,

    -- FIFO ordering
    date_of_service_from DATE,

    CONSTRAINT ck_processing_status CHECK (
        processing_status IN ('PENDING', 'PROCESSING', 'COMPLETED', 'FAILED')
    )
);

CREATE INDEX idx_raw_claims_processing ON staging.raw_claims(processing_status, ingested_at)
    WHERE processing_status IN ('PENDING', 'PROCESSING');

CREATE INDEX idx_raw_claims_batch ON staging.raw_claims(batch_id);
CREATE INDEX idx_raw_claims_queue ON staging.raw_claims(queue_id);
```

### Updated Processing Flow

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
===== STAGE 1: FILE INGESTION =====
ClaimsImporter::ingest_file()
    |
    +-- Parse File (CSV/EDI)
    |       |
    |       v
    |   Vec<ParsedRow>
    |       |
    +-- Create Import Batch (status: INGESTING)
    |       |
    +-- Batch Insert to staging.raw_claims
    |       - encounter_fields: JSONB
    |       - service_line_fields: JSONB
    |       - diagnosis_fields: JSONB
    |       - processing_status: PENDING
    |       - Commit every 1000 rows
    |       |
    +-- Update batch (status: INGESTED)
    |       |
    +-- Log INGEST metric (records_per_second)
    |
    v
===== STAGE 2: CLAIM PROCESSING =====
ClaimsProcessor::process_pending_claims()
    |
    +-- Query staging.raw_claims WHERE status = PENDING
    |       ORDER BY ingested_at ASC (FIFO)
    |       LIMIT 10000
    |       |
    +-- For each raw_claim (batched 1000):
    |       |
    |       +-- Lookup Facility (cached HashMap)
    |       |       |
    |       +-- Validate Claim Data
    |       |       |
    |       +-- IF VALID:
    |       |       |
    |       |       +-- INSERT claims.encounter
    |       |       +-- UPDATE raw_claims SET status = COMPLETED
    |       |       |
    |       +-- IF INVALID:
    |       |       |
    |       |       +-- INSERT staging.import_error_log
    |       |       +-- UPDATE raw_claims SET status = FAILED
    |       |       |
    |       +-- Update progress tracker (every 10)
    |       +-- Broadcast event
    |       +-- COMMIT every 1000
    |       |
    +-- Log PROCESS metric (records_per_second)
    |
    v
Mark Queue COMPLETED
```

## Implementation Checklist

### Phase 1: Database Schema
- [ ] Create migration `023_create_raw_claims_table.sql`
- [ ] Add `staging.raw_claims` table with indexes
- [ ] Update `staging.import_batch` to support INGESTING/INGESTED states
- [ ] Test migration on local database

### Phase 2: Stage 1 Implementation (File Ingestion)
- [ ] Create new method: `ClaimsImporter::ingest_file_to_staging()`
- [ ] Parse file to `Vec<ParsedRow>` (existing logic)
- [ ] Serialize parsed data to JSONB format
- [ ] Batch insert to `staging.raw_claims` (1000 per commit)
- [ ] Log INGEST performance metric
- [ ] Update progress tracker for Stage 1

### Phase 3: Stage 2 Implementation (Claim Processing)
- [ ] Create new crate or module: `pro-claims-processor`
- [ ] Implement `ClaimsProcessor::process_pending_claims()`
- [ ] Query `staging.raw_claims` WHERE status = PENDING (FIFO order)
- [ ] Deserialize JSONB to claim structures
- [ ] Facility lookup with caching (reuse existing HashMap)
- [ ] Validate and insert to `claims.encounter` OR `staging.import_error_log`
- [ ] Update `raw_claims.processing_status` atomically
- [ ] Log PROCESS performance metric
- [ ] Update progress tracker for Stage 2

### Phase 4: Orchestration
- [ ] Update `pro-service` to trigger Stage 1 (file ingestion)
- [ ] Create background worker to continuously run Stage 2
- [ ] Implement polling mechanism: check for PENDING raw_claims every N seconds
- [ ] Ensure Stage 2 can run independently (e.g., after restart)
- [ ] Add graceful shutdown handling for both stages

### Phase 5: Performance Metrics
- [ ] Add `stage` column to `staging.processing_metrics` (INGEST vs PROCESS)
- [ ] Track Stage 1 throughput: file_size, rows parsed, duration
- [ ] Track Stage 2 throughput: claims processed, duration
- [ ] Track end-to-end latency: file arrival to final insertion
- [ ] Add metrics to dashboard/monitoring

### Phase 6: Testing
- [ ] Unit tests for Stage 1 ingestion
- [ ] Unit tests for Stage 2 processing
- [ ] Integration test: full pipeline with sample CSV
- [ ] Performance test: 10,000 claims in 15 seconds
- [ ] Edge case: partial failures (some claims valid, some invalid)
- [ ] Edge case: file with all invalid claims
- [ ] Edge case: Stage 2 restarts mid-processing (idempotency)

### Phase 7: Deployment
- [ ] Rebuild Rust binaries with new logic
- [ ] Test fresh install with new migration
- [ ] Test upgrade from existing installation
- [ ] Rebuild MSI installer with updated binaries
- [ ] Update documentation

## Benefits of Two-Stage Approach

### Performance
- **Decoupled Processing**: File I/O doesn't block database validation
- **Parallelization**: Multiple Stage 2 workers can process from raw_claims table
- **Faster Ingestion**: Stage 1 optimized for speed (minimal validation)
- **Batch Optimization**: Stage 2 can process in optimal batch sizes

### Reliability
- **Retry Without Re-parsing**: Failed claims can be reprocessed from raw_claims
- **Partial Recovery**: If Stage 2 fails, data is safe in raw_claims table
- **Audit Trail**: Complete history of raw ingested data
- **Idempotent Processing**: Stage 2 can safely restart/resume

### Observability
- **Granular Metrics**: Separate throughput for ingestion vs processing
- **Bottleneck Identification**: Identify if slowdowns are in parsing or validation
- **Queue Depth Monitoring**: Track PENDING raw_claims as backlog metric

### Scalability
- **Horizontal Scaling**: Multiple Stage 2 workers can process claims concurrently
- **Priority Processing**: Can prioritize certain facilities or claim types
- **Load Shedding**: Can throttle ingestion if processing falls behind

## Risk Mitigation

### Performance Regression
- **Risk**: Two-stage approach adds overhead (extra database writes)
- **Mitigation**:
  - Batch inserts (1000 rows per commit) for both stages
  - Use JSONB for efficient storage/retrieval
  - Index optimization for PENDING status queries
  - Performance testing before deployment

### Data Integrity
- **Risk**: Raw claims could be orphaned if Stage 2 fails
- **Mitigation**:
  - Transaction safety: atomically update status with encounter insert
  - Monitoring: alert on high PENDING counts
  - Cleanup job: mark stale PROCESSING claims back to PENDING

### Increased Storage
- **Risk**: Storing all claims in raw_claims table increases storage
- **Mitigation**:
  - Cleanup policy: delete COMPLETED/FAILED raw_claims after N days
  - Compression: JSONB is efficient, consider further compression
  - Archive: move old raw_claims to archive table periodically

## Rollback Plan

If two-stage approach causes issues:
1. Keep original `ClaimsImporter::import_file_with_queue()` as fallback
2. Add feature flag: `USE_TWO_STAGE_PIPELINE` (default: true)
3. Can switch back to single-stage via configuration
4. Migration 023 is additive (doesn't break existing tables)

## Success Criteria

- [ ] Performance: 10,000 claims processed in 15 seconds (maintained)
- [ ] Metrics: Stage 1 and Stage 2 throughput captured accurately
- [ ] No data loss: All claims accounted for (encounter OR error log)
- [ ] FIFO compliance: Claims processed in correct order
- [ ] Idempotency: Stage 2 can restart without duplicate inserts
- [ ] All existing tests pass
- [ ] New integration tests cover two-stage flow

## Timeline Estimate

- Phase 1 (Schema): 1 hour
- Phase 2 (Stage 1): 3-4 hours
- Phase 3 (Stage 2): 4-5 hours
- Phase 4 (Orchestration): 2-3 hours
- Phase 5 (Metrics): 2 hours
- Phase 6 (Testing): 3-4 hours
- Phase 7 (Deployment): 1-2 hours

**Total**: 16-21 hours of development + testing

## Next Steps

1. Review and approve this plan
2. Create migration 023_create_raw_claims_table.sql
3. Implement Stage 1 (file ingestion to staging.raw_claims)
4. Implement Stage 2 (staging.raw_claims to encounters/errors)
5. Test performance meets 10k/15s requirement
6. Deploy and monitor
