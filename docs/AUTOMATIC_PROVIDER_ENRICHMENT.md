# Automatic Provider Enrichment Flow

**Date**: 2025-11-05
**Purpose**: Explain how providers are automatically enriched during claims processing

## Overview

When claims are processed through the pipeline, providers are **automatically** created and queued for enrichment. This happens in the background and does NOT block claims processing.

## Complete Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                      CLAIMS PROCESSING PIPELINE                     │
└─────────────────────────────────────────────────────────────────────┘

Step 1: EDI File Ingested
  └─> staging.raw_claims (PENDING status)

Step 2: Claims Processor Runs
  └─> process_encounter_with_service_lines()
      │
      ├─> Parse billing provider NPI from encounter data
      │   └─> ensure_provider_exists(npi, type, name, taxonomy)
      │       │
      │       ├─> Check: Does provider exist?
      │       │   SELECT provider_id FROM claims.provider WHERE npi = ?
      │       │
      │       ├─> NO: Create new provider
      │       │   │
      │       │   ├─> Lookup specialty from taxonomy (if provided)
      │       │   │   SELECT specialty_display
      │       │   │   FROM claims.provider_taxonomy
      │       │   │   WHERE taxonomy_code = ?
      │       │   │
      │       │   ├─> INSERT INTO claims.provider
      │       │   │   (npi, provider_type, last_name, first_name, taxonomy_code, specialty)
      │       │   │   RETURNING provider_id
      │       │   │
      │       │   └─> INSERT INTO claims.provider_enrichment_queue  ⭐
      │       │       (provider_id, npi, priority)
      │       │       VALUES (?, ?, 5)
      │       │       ON CONFLICT (provider_id) DO NOTHING
      │       │
      │       └─> YES: Return existing provider_id
      │
      ├─> Parse rendering provider NPI
      │   └─> ensure_provider_exists(...)
      │       └─> [Same flow as above]
      │
      ├─> Parse referring provider NPI
      │   └─> ensure_provider_exists(...)
      │       └─> [Same flow as above]
      │
      └─> Parse supervising provider NPI
          └─> ensure_provider_exists(...)
              └─> [Same flow as above]

Step 3: Encounter Inserted into claims.encounter
  └─> With provider_ids populated
  └─> Claims processing continues WITHOUT WAITING

┌─────────────────────────────────────────────────────────────────────┐
│                 BACKGROUND ENRICHMENT WORKER                        │
│                  (Runs in separate async task)                      │
└─────────────────────────────────────────────────────────────────────┘

Step 4: EnrichmentWorker polls queue every 30 seconds
  └─> SELECT queue_id, provider_id, npi
      FROM claims.provider_enrichment_queue
      WHERE status = 'PENDING'
      ORDER BY priority DESC, created_at ASC
      LIMIT 10
      FOR UPDATE SKIP LOCKED

Step 5: For each pending provider:
  │
  ├─> Mark as IN_PROGRESS
  │
  ├─> Call NPI Registry API
  │   GET https://npiregistry.cms.hhs.gov/api/?version=2.1&number={npi}
  │   └─> Returns: name, taxonomies, addresses, licenses
  │
  ├─> Extract primary taxonomy
  │   └─> Find taxonomy where primary = true
  │
  ├─> Lookup specialty display name
  │   SELECT specialty_display
  │   FROM claims.provider_taxonomy
  │   WHERE taxonomy_code = ?
  │
  ├─> UPDATE claims.provider SET
  │   first_name = API.first_name,
  │   last_name = API.last_name,
  │   taxonomy_code = API.primary_taxonomy.code,
  │   specialty = TAXONOMY.specialty_display,
  │   license_number = API.license,
  │   address_line1 = API.location_address.address_1,
  │   city = API.location_address.city,
  │   state_code = API.location_address.state,
  │   postal_code = API.location_address.postal_code,
  │   phone = API.location_address.telephone,
  │   updated_by = 'NPI_ENRICHMENT'
  │   WHERE provider_id = ?
  │
  ├─> Store full API response in queue for audit
  │   UPDATE claims.provider_enrichment_queue
  │   SET api_response = ?
  │   WHERE provider_id = ?
  │
  └─> Mark as COMPLETED
      UPDATE claims.provider_enrichment_queue
      SET status = 'COMPLETED', completed_at = NOW()
      WHERE queue_id = ?

Step 6: Provider now fully enriched!
  └─> Has: name, taxonomy_code, specialty, address, phone
  └─> Future encounters using this NPI will reference enriched data
```

## Key Points

### 1. Automatic Queue Population ⭐

**When**: Every time a NEW provider is created during claims processing

**Where**: [claims_processor.rs:1641-1654](../crates/pro-service/src/claims_processor.rs#L1641-L1654)

```rust
// Enqueue provider for background NPI enrichment (fire-and-forget)
// This does not block claim processing if it fails
let _ = sqlx::query(
    r#"
    INSERT INTO claims.provider_enrichment_queue (provider_id, npi, priority)
    VALUES ($1, $2, $3)
    ON CONFLICT (provider_id) DO NOTHING
    "#
)
.bind(provider_id)
.bind(npi)
.bind(5) // Default priority
.execute(&mut **tx)
.await;
```

**Important**:
- Uses `let _ =` (fire-and-forget) - enrichment queue insertion failure does NOT stop claims processing
- `ON CONFLICT (provider_id) DO NOTHING` - each provider is only queued once (UNIQUE constraint)
- Default priority = 5 (medium priority on scale of 1-10)

### 2. Non-Blocking Design

Claims processing is **NEVER** blocked by:
- NPI Registry API calls
- Network issues
- API rate limits
- Enrichment failures

The flow is:
1. Create provider with basic info from EDI file ✅
2. Queue for enrichment ✅
3. Continue processing claim immediately ✅
4. Worker enriches in background ⏰
5. Future claims use enriched data ✅

### 3. Taxonomy Lookup Happens Twice

**First Lookup** (During claims processing):
```sql
-- If taxonomy_code is provided in EDI file
SELECT specialty_display
FROM claims.provider_taxonomy
WHERE taxonomy_code = ?
```
- Uses taxonomy code from EDI file (if present)
- Happens synchronously during claim processing
- Result stored in `provider.specialty` field

**Second Lookup** (During background enrichment):
```sql
-- Using primary taxonomy from NPI Registry API
SELECT specialty_display
FROM claims.provider_taxonomy
WHERE taxonomy_code = ?
```
- Uses primary taxonomy from CMS NPI Registry
- Happens asynchronously in background worker
- Overwrites `provider.specialty` field with authoritative data

### 4. Data Priority (COALESCE strategy)

The enrichment worker uses `COALESCE` to preserve existing data:

```sql
UPDATE claims.provider
SET
    first_name = COALESCE($api_first_name, first_name),
    last_name = COALESCE($api_last_name, last_name),
    taxonomy_code = COALESCE($api_taxonomy, taxonomy_code),
    specialty = COALESCE($api_specialty, specialty),
    ...
```

This means:
- If EDI file had good data, it's kept if API returns NULL
- If API has better data, it overwrites EDI data
- No data is lost unnecessarily

### 5. When Providers Are Created

Providers are created via `ensure_provider_exists()` for:

1. **Billing Provider** (Loop 2010AA)
   - Line 672: `self.ensure_provider_exists(..., "Billing", ...)`

2. **Rendering Provider** (Loop 2310B)
   - Line 687: `self.ensure_provider_exists(..., "Rendering", ...)`

3. **Referring Provider** (Loop 2310A)
   - Line 702: `self.ensure_provider_exists(..., "Referring", ...)`

4. **Supervising Provider** (Loop 2310F)
   - Line 730: `self.ensure_provider_exists(..., "Supervising", ...)`

Each distinct NPI creates ONE provider record, queued ONCE for enrichment.

## Monitoring Enrichment

### Check Queue Status

```sql
-- See current queue state
SELECT
    status,
    COUNT(*) as count,
    MIN(created_at) as oldest,
    MAX(created_at) as newest
FROM claims.provider_enrichment_queue
GROUP BY status
ORDER BY status;
```

### Check Recently Enriched Providers

```sql
-- See providers enriched in last 24 hours
SELECT
    p.provider_id,
    p.npi,
    p.last_name,
    p.first_name,
    p.taxonomy_code,
    pt.specialty_display,
    p.updated_by,
    p.updated_at,
    pq.status,
    pq.completed_at
FROM claims.provider p
INNER JOIN claims.provider_enrichment_queue pq
    ON p.provider_id = pq.provider_id
LEFT JOIN claims.provider_taxonomy pt
    ON p.taxonomy_code = pt.taxonomy_code
WHERE p.updated_by = 'NPI_ENRICHMENT'
AND p.updated_at > NOW() - INTERVAL '24 hours'
ORDER BY p.updated_at DESC;
```

### Check Pending Enrichments

```sql
-- See what's waiting to be processed
SELECT
    pq.queue_id,
    pq.provider_id,
    pq.npi,
    pq.priority,
    pq.retry_count,
    pq.created_at,
    EXTRACT(EPOCH FROM (NOW() - pq.created_at))/60 as minutes_waiting,
    p.last_name,
    p.first_name
FROM claims.provider_enrichment_queue pq
INNER JOIN claims.provider p ON pq.provider_id = p.provider_id
WHERE pq.status = 'PENDING'
ORDER BY pq.priority DESC, pq.created_at ASC
LIMIT 20;
```

### Check Failed Enrichments

```sql
-- See what failed and why
SELECT
    pq.queue_id,
    pq.npi,
    pq.retry_count,
    pq.max_retries,
    pq.last_error,
    pq.last_error_at,
    pq.next_retry_at,
    p.last_name,
    p.first_name
FROM claims.provider_enrichment_queue pq
INNER JOIN claims.provider p ON pq.provider_id = p.provider_id
WHERE pq.status = 'FAILED'
ORDER BY pq.last_error_at DESC
LIMIT 20;
```

## Worker Configuration

The enrichment worker can be configured:

```rust
WorkerConfig {
    batch_size: 10,              // Process 10 providers at a time
    poll_interval: 30 seconds,   // Check queue every 30s when empty
    rate_limit_delay: 200ms,     // 5 req/sec max to NPI Registry
    enabled: true,               // Can disable if needed
}
```

## Retry Logic

If enrichment fails, the worker:
1. Increments `retry_count`
2. Sets `status = 'FAILED'`
3. Stores `last_error` message
4. Calculates `next_retry_at` (exponential backoff)
5. Retries up to `max_retries` (default: 3)

After 3 failures, the provider stays in FAILED status and must be manually requeued or fixed.

## Performance Characteristics

### Throughput
- **Batches**: 10 providers per batch
- **Rate limit**: 5 API calls per second (200ms delay)
- **Theoretical max**: ~18,000 providers per hour (5 req/sec × 3600s)
- **Practical max**: ~10,000-15,000 per hour (accounting for API response time)

### Latency
- **Queue insertion**: <1ms (during claims processing)
- **First enrichment attempt**: 0-30 seconds (depending on poll timing)
- **API call duration**: 200-500ms per provider
- **Total enrichment time**: ~30-60 seconds from creation to completion

### Impact on Claims Processing
- **Zero blocking**: Claims processing never waits for enrichment
- **Zero failures**: Enrichment queue insertion failure does not stop claims
- **Background only**: All API calls happen in separate async task

## Testing Automatic Enrichment

### Test 1: Process a Claim with New Providers

```bash
# 1. Place EDI file in inbox
cp test_data/claims_test.edi /path/to/inbox/

# 2. Wait for Stage 1 (ingestion) to complete
# Check: staging.raw_claims should have PENDING rows

# 3. Wait for Stage 2 (processing) to complete
# Check: claims.encounter should have new rows

# 4. Check if providers were created and queued
SELECT
    p.provider_id,
    p.npi,
    p.last_name,
    pq.status,
    pq.created_at
FROM claims.provider p
LEFT JOIN claims.provider_enrichment_queue pq
    ON p.provider_id = pq.provider_id
WHERE p.created_at > NOW() - INTERVAL '5 minutes'
ORDER BY p.created_at DESC;

# 5. Wait 30-60 seconds for enrichment worker

# 6. Check enrichment results
SELECT
    p.npi,
    p.taxonomy_code,
    pt.specialty_display,
    p.updated_by,
    pq.status
FROM claims.provider p
LEFT JOIN claims.provider_enrichment_queue pq ON p.provider_id = pq.provider_id
LEFT JOIN claims.provider_taxonomy pt ON p.taxonomy_code = pt.taxonomy_code
WHERE p.created_at > NOW() - INTERVAL '5 minutes'
ORDER BY p.created_at DESC;
```

### Test 2: Verify No Duplicates

```sql
-- Each provider should only be queued once
SELECT
    provider_id,
    COUNT(*) as queue_count
FROM claims.provider_enrichment_queue
GROUP BY provider_id
HAVING COUNT(*) > 1;
-- Should return 0 rows (UNIQUE constraint prevents duplicates)
```

### Test 3: Verify Claims Don't Block

This is automatic - claims processing completes even if:
- NPI Registry API is down
- Network is unavailable
- Enrichment queue insertion fails

The provider record is created with basic data, and the claim proceeds.

## Troubleshooting

### No Providers Being Enriched

**Check 1: Is the worker running?**
```sql
-- If you see recent IN_PROGRESS or COMPLETED, worker is running
SELECT status, MAX(updated_at) as last_activity
FROM claims.provider_enrichment_queue
GROUP BY status;
```

**Check 2: Are providers being queued?**
```sql
-- Should see PENDING entries after processing claims
SELECT COUNT(*) FROM claims.provider_enrichment_queue WHERE status = 'PENDING';
```

**Check 3: Check worker logs**
```powershell
Get-Content "C:\ProgramData\Professional SMART\logs\service.log" | Select-String "enrichment"
```

### Providers Stuck in PENDING

**Possible causes**:
1. Worker is disabled in configuration
2. Worker crashed or isn't running
3. Database connection issue

**Solution**: Restart the service
```powershell
net stop ProfessionalSMART
net start ProfessionalSMART
```

### Many Failed Enrichments

**Check errors**:
```sql
SELECT last_error, COUNT(*) as count
FROM claims.provider_enrichment_queue
WHERE status = 'FAILED'
GROUP BY last_error
ORDER BY count DESC;
```

**Common issues**:
- `"NPI not found"` - NPI doesn't exist in CMS registry (EDI file error)
- `"Invalid NPI format"` - EDI file has malformed NPI
- `"API returned error: 429"` - Rate limit exceeded (reduce batch size)
- `"API returned error: 503"` - CMS API temporarily unavailable (will retry)

## Summary

✅ **Automatic**: Providers are automatically queued when created
✅ **Non-blocking**: Claims processing never waits for enrichment
✅ **Fire-and-forget**: Enrichment queue failures don't stop claims
✅ **Unique**: Each provider is only queued once (UNIQUE constraint)
✅ **Background**: Worker processes queue asynchronously every 30s
✅ **Resilient**: Retries up to 3 times with exponential backoff
✅ **Auditable**: Full API response stored in queue for debugging

The system is designed to prioritize claims processing speed while still enriching provider data in the background for better reporting and analytics.

## Related Documentation

- [TESTING_NPI_ENRICHMENT.md](TESTING_NPI_ENRICHMENT.md) - Manual testing guide
- [Migration 042](../migrations/042_create_provider_enrichment_queue.sql) - Queue table definition
- [claims_processor.rs:1536-1657](../crates/pro-service/src/claims_processor.rs#L1536-L1657) - ensure_provider_exists function
- [worker.rs](../crates/pro-npi-enrichment/src/worker.rs) - Background worker implementation

---

**Last Updated**: 2025-11-26
**Version**: v2.8.4.0
