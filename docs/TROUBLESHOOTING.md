# Troubleshooting Guide

**Version:** 2.8.4.0
**Last Updated:** 2025-11-26

---

## Table of Contents

1. [Service Issues](#service-issues)
2. [Database Issues](#database-issues)
3. [File Processing Issues](#file-processing-issues)
4. [Migration Issues](#migration-issues)
5. [Performance Issues](#performance-issues)
6. [Common Error Messages](#common-error-messages)

---

## Service Issues

### Service Won't Start

**Symptoms:** Windows service fails to start or stops immediately.

**Check 1: Service Status**
```powershell
Get-Service "Professional SMART" | Select-Object *
```

**Check 2: Event Logs**
```powershell
Get-EventLog -LogName Application -Source "Professional SMART" -Newest 10
```

**Check 3: Log Files**
```powershell
Get-Content "C:\ProgramData\Professional SMART\logs\pro-service.log" -Tail 50
```

**Common Causes:**
1. **Database not accessible:** Verify PostgreSQL is running and DATABASE_URL is correct
2. **Port in use:** Check if port 8080 is available
3. **Missing .env file:** Ensure configuration exists at `C:\Program Files\Professional SMART\.env`

**Solutions:**
```powershell
# Check PostgreSQL service
Get-Service postgresql* | Select-Object Status, DisplayName

# Test database connection
psql -U postgres -d professional_smart -c "SELECT 1"

# Check if port is in use
netstat -ano | findstr :8080
```

### Service Stops After Processing Files

**Symptoms:** Service processes a file then stops.

**Cause:** Unhandled error in file processing.

**Solution:**
1. Check error logs for the specific file
2. Move problematic file to error directory
3. Restart service

```powershell
# Check for crashed files
Get-ChildItem "C:\Program Files\Professional SMART\data\error" -Recurse

# Restart service
Restart-Service "Professional SMART"
```

---

## Database Issues

### Connection Refused

**Error:** `connection refused` or `could not connect to server`

**Check PostgreSQL status:**
```powershell
Get-Service postgresql* | Select-Object Status, DisplayName
```

**Check PostgreSQL is listening:**
```powershell
netstat -ano | findstr :5432
```

**Solutions:**
1. Start PostgreSQL service
2. Check `pg_hba.conf` allows local connections
3. Verify DATABASE_URL in .env file

### Too Many Connections

**Error:** `too many connections for role`

**Check current connections:**
```sql
SELECT count(*) FROM pg_stat_activity;
SELECT * FROM pg_stat_activity WHERE datname = 'professional_smart';
```

**Solutions:**
1. Increase `max_connections` in postgresql.conf
2. Reduce `DB_MAX_CONNECTIONS` in .env
3. Kill idle connections:

```sql
SELECT pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE datname = 'professional_smart'
  AND state = 'idle'
  AND pid != pg_backend_pid();
```

### Migration Failed

**Error:** Migration fails during installation or upgrade.

**Check migration status:**
```sql
SELECT * FROM staging.schema_migrations ORDER BY version;
```

**Manual migration:**
```powershell
cd "C:\Program Files\Professional SMART\bin"
.\pro-upgrade.exe apply-migrations --db-host localhost --db-port 5432 --db-name professional_smart --db-user postgres --db-password YOUR_PASSWORD
```

**Rollback specific migration:** See [MIGRATION_STATUS.md](MIGRATION_STATUS.md) for rollback procedures.

---

## File Processing Issues

### File Not Being Processed

**Symptoms:** Files in input directory are not processed.

**Check 1: File location**
```powershell
Get-ChildItem "C:\Program Files\Professional SMART\data\input"
```

**Check 2: File permissions**
```powershell
icacls "C:\Program Files\Professional SMART\data\input\*.edi"
```

**Check 3: Queue status**
```sql
SELECT * FROM staging.file_processing_queue
WHERE processing_status = 'PENDING'
ORDER BY queued_at DESC
LIMIT 10;
```

**Solutions:**
1. Ensure files have correct extension (.edi or .csv)
2. Check file permissions allow read/write
3. Verify INPUT_DIR environment variable is correct

### File Processing Fails

**Symptoms:** Files moved to error directory.

**Check error details:**
```sql
-- Check import batch errors
SELECT * FROM staging.import_batch
WHERE import_status = 'FAILED'
ORDER BY created_at DESC
LIMIT 10;

-- Check error log
SELECT * FROM staging.import_error_log
ORDER BY created_at DESC
LIMIT 20;
```

**Common file issues:**
1. **Invalid EDI format:** Missing required segments (ISA, GS, ST)
2. **Encoding issues:** File should be UTF-8 or ASCII
3. **Missing required fields:** Check error_details for specific field

### Claims Not Appearing in Database

**Check staging tables:**
```sql
-- Raw claims in staging
SELECT COUNT(*) FROM staging.raw_claims WHERE processing_status = 'PENDING';

-- Check if claims were processed
SELECT * FROM staging.raw_claims
WHERE processing_status = 'FAILED'
ORDER BY ingested_at DESC
LIMIT 10;
```

---

## Migration Issues

### Migration Version Mismatch

**Error:** `Database version does not match expected version`

**Check versions:**
```sql
SELECT * FROM staging.schema_migrations ORDER BY version DESC LIMIT 5;
```

**Check application version:**
```powershell
"C:\Program Files\Professional SMART\bin\pro-upgrade.exe" --version
```

### Missing Tables After Migration

**Verify tables exist:**
```sql
SELECT table_schema, table_name
FROM information_schema.tables
WHERE table_schema IN ('claims', 'staging', 'ml', 'archive')
ORDER BY table_schema, table_name;
```

**Re-run migrations:**
```powershell
"C:\Program Files\Professional SMART\bin\pro-upgrade.exe" apply-migrations --db-host localhost --db-port 5432 --db-name professional_smart --db-user postgres --db-password YOUR_PASSWORD
```

---

## Performance Issues

### Slow File Processing

**Check processing metrics:**
```sql
SELECT
    DATE_TRUNC('hour', created_at) as hour,
    COUNT(*) as batches,
    SUM(total_claims) as total_claims,
    AVG(EXTRACT(EPOCH FROM (completed_at - created_at))) as avg_seconds
FROM staging.import_batch
WHERE created_at > CURRENT_TIMESTAMP - INTERVAL '24 hours'
GROUP BY DATE_TRUNC('hour', created_at)
ORDER BY hour DESC;
```

**Optimize database:**
```sql
-- Analyze tables
ANALYZE claims.encounter;
ANALYZE claims.service_line;
ANALYZE staging.raw_claims;

-- Check for bloat
SELECT relname, pg_size_pretty(pg_total_relation_size(relid))
FROM pg_catalog.pg_statio_user_tables
ORDER BY pg_total_relation_size(relid) DESC
LIMIT 10;
```

### High Memory Usage

**Check connection pool:**
```sql
SELECT count(*), state FROM pg_stat_activity
WHERE datname = 'professional_smart'
GROUP BY state;
```

**Reduce pool size in .env:**
```env
DB_MAX_CONNECTIONS=50  # Reduce from 100
DB_MIN_CONNECTIONS=5   # Reduce from 10
```

### Dashboard Queries Slow

**Refresh materialized views:**
```sql
SELECT * FROM claims.refresh_dashboard_views();
```

**Check when views were last refreshed:**
```sql
SELECT MAX(last_refreshed) FROM claims.mv_management_overview;
```

---

## Common Error Messages

### "queue_id required for EDI ingestion"

**Cause:** File was not properly queued before processing.

**Solution:** Ensure file watcher is running and file has correct extension.

### "Failed to get batch_id from queue"

**Cause:** Queue entry missing import_batch_id.

**Check:**
```sql
SELECT * FROM staging.file_processing_queue
WHERE import_batch_id IS NULL;
```

### "Invalid date format"

**Cause:** Date in claim doesn't match expected format.

**Check claim dates:**
```sql
SELECT * FROM staging.raw_claims
WHERE processing_status = 'FAILED'
  AND error_message LIKE '%date%';
```

### "Provider not found"

**Cause:** Provider NPI not in database.

**Solution:**
```sql
-- Check if provider exists
SELECT * FROM claims.provider WHERE npi = 'THE_NPI';

-- Insert missing provider
INSERT INTO claims.provider (organization_id, provider_type, npi, last_name, first_name)
VALUES (1, 'Billing', 'THE_NPI', 'Unknown', 'Provider');
```

### "Foreign key constraint violation"

**Cause:** Referenced record doesn't exist.

**Check constraint name in error for the specific relationship.**

---

## Getting Help

### Collect Diagnostic Information

Before reporting issues, gather:

```powershell
# System info
systeminfo > diagnostics.txt

# Service logs (last 1000 lines)
Get-Content "C:\ProgramData\Professional SMART\logs\pro-service.log" -Tail 1000 >> diagnostics.txt

# Database version
psql -U postgres -d professional_smart -c "SELECT version();" >> diagnostics.txt

# Migration status
psql -U postgres -d professional_smart -c "SELECT * FROM staging.schema_migrations ORDER BY version;" >> diagnostics.txt
```

### Log Locations

| Log | Location |
|-----|----------|
| Service log | `C:\ProgramData\Professional SMART\logs\pro-service.log` |
| Windows Event Log | Application log, source "Professional SMART" |
| PostgreSQL log | `C:\Program Files\PostgreSQL\16\data\log\` |

---

## Related Documentation

- [INSTALLATION.md](INSTALLATION.md) - Installation and setup
- [MIGRATION_STATUS.md](MIGRATION_STATUS.md) - Migration details and rollback
- [UPGRADE_GUIDE.md](UPGRADE_GUIDE.md) - Upgrade procedures
