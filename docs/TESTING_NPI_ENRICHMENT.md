# Testing NPI Enrichment

**Version:** 2.8.4.0
**Last Updated:** 2025-11-26

---

## Overview

The NPI enrichment feature automatically enriches provider records with data from the CMS NPI Registry API. This guide covers how to test the feature at different levels.

---

## Quick Test - Command Line Lookup

The fastest way to test NPI lookup is using the built-in example:

```powershell
# Navigate to project root
cd C:\Users\jonmc\dev\pro

# Run the test tool with any valid NPI
cargo run --example test_npi_lookup 1234567890
```

**Note:** Replace `1234567890` with a real NPI. To find real NPIs:
1. Visit https://npiregistry.cms.hhs.gov
2. Search for a provider (e.g., "John Smith" in your state)
3. Copy the 10-digit NPI number

### Expected Output

```
Looking up NPI: 1234567890
API: https://npiregistry.cms.hhs.gov/api/

SUCCESS - Found 1 result(s)

PROVIDER INFORMATION
  NPI: 1234567890
  Type: NPI-1
  Name: John Smith, MD
  Status: Active
  Enumeration Date: 2006-05-23

TAXONOMIES (SPECIALTIES)
  1. Code: 207Q00000X
     Description: Family Medicine
     Primary: Yes
```

---

## Database Integration Test

### Step 1: Create a Test Provider

```sql
-- Insert a test provider without enrichment data
INSERT INTO claims.provider (
    organization_id,
    provider_type,
    npi,
    last_name,
    first_name
) VALUES (
    1,
    'Billing',
    '1234567890',  -- Use a real NPI
    'Test',
    'Provider'
);
```

### Step 2: Queue for Enrichment

```sql
-- Queue the provider for NPI enrichment
INSERT INTO claims.provider_enrichment_queue (
    provider_id,
    npi,
    priority,
    status
)
SELECT
    provider_id,
    npi,
    10,  -- High priority
    'PENDING'
FROM claims.provider
WHERE npi = '1234567890';
```

### Step 3: Verify Enrichment (after 30-60 seconds)

```sql
-- Check provider was enriched
SELECT
    p.provider_id,
    p.npi,
    p.last_name,
    p.first_name,
    p.credential,
    p.specialty,
    p.taxonomy_code,
    pt.specialty_display,
    p.npi_registry_link,
    p.enrichment_status,
    p.last_enriched_at
FROM claims.provider p
LEFT JOIN claims.provider_taxonomy pt ON p.taxonomy_code = pt.taxonomy_code
WHERE p.npi = '1234567890';

-- Check queue status
SELECT * FROM claims.provider_enrichment_queue
WHERE npi = '1234567890'
ORDER BY created_at DESC
LIMIT 1;
```

### Expected Results After Enrichment

| Field | Before | After |
|-------|--------|-------|
| last_name | 'Test' | 'Smith' (real name from API) |
| first_name | 'Provider' | 'John' (real name from API) |
| credential | NULL | 'MD' |
| taxonomy_code | NULL | '207Q00000X' |
| specialty | NULL | 'Family Medicine' |
| enrichment_status | NULL | 'ENRICHED' |
| npi_registry_link | NULL | 'https://nppesapi.cms.hhs.gov/api/?version=2.1&number=1234567890' |

---

## Batch Enrichment Test

To test enrichment of multiple providers:

```sql
-- Queue multiple providers for enrichment
INSERT INTO claims.provider_enrichment_queue (provider_id, npi, priority, status)
SELECT provider_id, npi, 5, 'PENDING'
FROM claims.provider
WHERE taxonomy_code IS NULL
  AND npi IS NOT NULL
  AND LENGTH(npi) = 10
LIMIT 100;

-- Monitor progress
SELECT
    status,
    COUNT(*) as count
FROM claims.provider_enrichment_queue
GROUP BY status;
```

---

## Troubleshooting

### Provider Not Being Enriched

1. **Check worker is running:**
   ```powershell
   Get-Service "Professional SMART" | Select-Object Status, DisplayName
   ```

2. **Check queue status:**
   ```sql
   SELECT * FROM claims.provider_enrichment_queue
   WHERE status = 'FAILED'
   ORDER BY updated_at DESC
   LIMIT 10;
   ```

3. **Check for rate limiting:**
   The CMS API has rate limits. The worker automatically throttles requests.

### Invalid NPI

```sql
-- NPIs must be exactly 10 digits
SELECT * FROM claims.provider
WHERE npi IS NOT NULL
  AND LENGTH(npi) != 10;
```

### Network Issues

Test network connectivity to CMS API:
```powershell
Invoke-WebRequest -Uri "https://npiregistry.cms.hhs.gov/api/?version=2.1&number=1234567890"
```

---

## API Rate Limits

The CMS NPI Registry API has the following limits:
- **Requests per second:** ~2-3 (recommended)
- **Requests per day:** No hard limit, but be respectful

The enrichment worker automatically:
- Throttles requests to avoid rate limiting
- Retries failed requests with exponential backoff
- Processes in batches of 50 providers

---

## Monitoring Enrichment

### View Enrichment Statistics

```sql
-- Overall enrichment status
SELECT
    enrichment_status,
    COUNT(*) as provider_count
FROM claims.provider
WHERE npi IS NOT NULL
GROUP BY enrichment_status;

-- Recent enrichments
SELECT
    provider_id,
    npi,
    last_name,
    first_name,
    specialty,
    last_enriched_at
FROM claims.provider
WHERE last_enriched_at > CURRENT_TIMESTAMP - INTERVAL '1 hour'
ORDER BY last_enriched_at DESC;
```

### View Queue Performance

```sql
-- Queue processing rate (last hour)
SELECT
    DATE_TRUNC('minute', updated_at) as minute,
    COUNT(*) FILTER (WHERE status = 'COMPLETED') as completed,
    COUNT(*) FILTER (WHERE status = 'FAILED') as failed
FROM claims.provider_enrichment_queue
WHERE updated_at > CURRENT_TIMESTAMP - INTERVAL '1 hour'
GROUP BY DATE_TRUNC('minute', updated_at)
ORDER BY minute DESC;
```

---

## Related Documentation

- [AUTOMATIC_PROVIDER_ENRICHMENT.md](AUTOMATIC_PROVIDER_ENRICHMENT.md) - Architecture and configuration
- [DATABASE_SCHEMA_REFERENCE.md](DATABASE_SCHEMA_REFERENCE.md) - Provider and enrichment queue tables
