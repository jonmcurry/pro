# Database Setup Guide

This guide covers database setup and migrations for Professional SMART.

## Prerequisites

- PostgreSQL 13 or higher
- Database superuser access (for initial setup)
- Network connectivity to PostgreSQL server

## Initial Database Setup

### 1. Create Database

```sql
CREATE DATABASE professional_smart
    WITH OWNER = postgres
    ENCODING = 'UTF8'
    LC_COLLATE = 'en_US.UTF-8'
    LC_CTYPE = 'en_US.UTF-8'
    TEMPLATE = template0;
```

### 2. Create Schemas

```sql
\c professional_smart

CREATE SCHEMA IF NOT EXISTS claims;
CREATE SCHEMA IF NOT EXISTS staging;
CREATE SCHEMA IF NOT EXISTS audit;
```

### 3. Enable Required Extensions

```sql
CREATE EXTENSION IF NOT EXISTS pgcrypto;  -- For encryption
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";  -- For UUID generation
```

## Running Migrations

Professional SMART uses numbered migration files in the `migrations/` directory.

### Automatic Migration (Recommended)

The service automatically runs pending migrations on startup. No manual intervention required.

### Manual Migration

To apply migrations manually:

```bash
# Apply all pending migrations
psql -U postgres -d professional_smart -f migrations/001_initial_schema.sql
psql -U postgres -d professional_smart -f migrations/002_add_encounters.sql
# ... continue for each migration in order
```

### Recent Performance Optimizations (Phase 6-8)

The following migrations include critical performance optimizations:

**Migration 050: Performance Indexes** (Phase 6)
- Duplicate detection optimization (90%+ improvement)
- Provider lookup indexes (100-500x faster)
- Batch processing optimization (10-50x faster)
- Diagnosis loading indexes (20-100x faster)

```bash
psql -U postgres -d professional_smart -f migrations/050_add_performance_indexes.sql
```

**Migration 051: Rule Execution Statistics** (Phase 8)
- Historical statistics tracking
- Materialized view for fast aggregation
- Automatic cleanup (90-day retention)

```bash
psql -U postgres -d professional_smart -f migrations/051_add_rule_execution_stats.sql
```

## Database Configuration

### Connection Settings

The service reads database configuration from environment variables or config file:

```bash
# Environment variables
export DATABASE_URL="postgresql://user:password@localhost/professional_smart"

# Or in config file (config.toml)
[database]
host = "localhost"
port = 5432
database = "professional_smart"
username = "postgres"
password = "your_password"
max_connections = 10
```

### Performance Tuning

For optimal performance with Phase 6-8 optimizations:

```sql
-- Increase shared buffers (25% of RAM)
ALTER SYSTEM SET shared_buffers = '4GB';

-- Increase work memory for aggregations
ALTER SYSTEM SET work_mem = '64MB';

-- Enable parallel query execution
ALTER SYSTEM SET max_parallel_workers_per_gather = 4;

-- Optimize for write-heavy workloads
ALTER SYSTEM SET checkpoint_timeout = '15min';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;

-- Apply changes
SELECT pg_reload_conf();
```

### Maintenance Schedule

**Daily Tasks**:
- Cleanup old rule execution statistics:
  ```sql
  SELECT claims.cleanup_old_rule_execution_stats();
  ```

**Hourly Tasks**:
- Refresh rule execution statistics view:
  ```sql
  SELECT claims.refresh_rule_execution_stats();
  ```

**Weekly Tasks**:
- Vacuum and analyze:
  ```sql
  VACUUM ANALYZE;
  ```

**Monthly Tasks**:
- Reindex to prevent bloat:
  ```sql
  REINDEX DATABASE professional_smart;
  ```

## Monitoring

### Check Migration Status

```sql
-- View applied migrations (if migration tracking table exists)
SELECT * FROM schema_migrations ORDER BY version;
```

### Monitor Index Usage

```sql
-- Check index sizes
SELECT
    schemaname,
    tablename,
    indexname,
    pg_size_pretty(pg_relation_size(indexname::regclass)) as size
FROM pg_indexes
WHERE schemaname IN ('claims', 'staging')
ORDER BY pg_relation_size(indexname::regclass) DESC;

-- Check index usage statistics
SELECT
    schemaname,
    tablename,
    indexname,
    idx_scan as scans,
    idx_tup_read as tuples_read,
    idx_tup_fetch as tuples_fetched
FROM pg_stat_user_indexes
WHERE schemaname IN ('claims', 'staging')
ORDER BY idx_scan DESC;
```

### Monitor Rule Execution Statistics

```sql
-- Daily execution counts
SELECT
    DATE(executed_at) as date,
    COUNT(*) as total_executions,
    COUNT(*) FILTER (WHERE triggered) as flags_created,
    AVG(execution_time_ms) as avg_time_ms
FROM claims.rule_execution_stats
WHERE executed_at >= NOW() - INTERVAL '7 days'
GROUP BY DATE(executed_at)
ORDER BY date DESC;

-- Top slowest rules
SELECT
    flag_type,
    rule_code,
    COUNT(*) as executions,
    AVG(execution_time_ms) as avg_time_ms,
    MAX(execution_time_ms) as max_time_ms
FROM claims.rule_execution_stats
WHERE executed_at >= NOW() - INTERVAL '7 days'
GROUP BY flag_type, rule_code
ORDER BY avg_time_ms DESC
LIMIT 10;
```

## Backup and Recovery

### Backup

```bash
# Full database backup
pg_dump -U postgres -d professional_smart -F c -f backup_$(date +%Y%m%d).dump

# Schema-only backup
pg_dump -U postgres -d professional_smart --schema-only -f schema_backup.sql

# Data-only backup
pg_dump -U postgres -d professional_smart --data-only -f data_backup.sql
```

### Restore

```bash
# Restore from custom format backup
pg_restore -U postgres -d professional_smart -c backup_20251106.dump

# Restore from SQL backup
psql -U postgres -d professional_smart -f schema_backup.sql
```

## Troubleshooting

### Connection Issues

**Problem**: Cannot connect to database

**Solutions**:
1. Check PostgreSQL is running: `systemctl status postgresql`
2. Verify connection settings in config file
3. Check pg_hba.conf allows connections from your IP
4. Test connection: `psql -U postgres -h localhost -d professional_smart`

### Slow Queries

**Problem**: Queries taking too long

**Solutions**:
1. Verify indexes are applied:
   ```sql
   SELECT indexname FROM pg_indexes
   WHERE schemaname = 'claims'
   AND indexname LIKE 'idx_%';
   ```
2. Run ANALYZE to update statistics:
   ```sql
   ANALYZE claims.service_line;
   ANALYZE claims.encounter;
   ```
3. Check for missing indexes using pg_stat_statements

### Disk Space Issues

**Problem**: Database growing too large

**Solutions**:
1. Check table sizes:
   ```sql
   SELECT
       schemaname || '.' || tablename as table,
       pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
   FROM pg_tables
   WHERE schemaname IN ('claims', 'staging')
   ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
   ```
2. Run cleanup for old statistics:
   ```sql
   SELECT claims.cleanup_old_rule_execution_stats();
   ```
3. Archive old data to cold storage

## Security

### User Permissions

Create application user with minimal permissions:

```sql
-- Create application user
CREATE USER pro_app WITH PASSWORD 'secure_password';

-- Grant schema access
GRANT USAGE ON SCHEMA claims TO pro_app;
GRANT USAGE ON SCHEMA staging TO pro_app;

-- Grant table permissions
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA claims TO pro_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA staging TO pro_app;

-- Grant sequence permissions
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA claims TO pro_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA staging TO pro_app;
```

### Connection Encryption

Enable SSL connections in postgresql.conf:

```
ssl = on
ssl_cert_file = 'server.crt'
ssl_key_file = 'server.key'
ssl_ca_file = 'root.crt'
```

Update connection string:

```
DATABASE_URL="postgresql://user:password@localhost/professional_smart?sslmode=require"
```

## Support

For additional help:
- See [INSTALLATION.md](INSTALLATION.md) for service installation
- See [CONFIGURATION.md](CONFIGURATION.md) for configuration options
- See [PERFORMANCE_TUNING.md](PERFORMANCE_TUNING.md) for optimization tips
- Check service logs: `C:\ProgramData\Professional SMART\logs\service.log`

## Appendix: Complete Migration List

```
001_initial_schema.sql           - Base schema and tables
002_add_encounters.sql           - Encounter tracking
...
049_add_flag_issue_helpers.sql   - Flag type mapping (Phase 4)
050_add_performance_indexes.sql  - Critical indexes (Phase 6)
051_add_rule_execution_stats.sql - Historical statistics (Phase 8)
```

For complete migration history and details, see the `migrations/` directory.
