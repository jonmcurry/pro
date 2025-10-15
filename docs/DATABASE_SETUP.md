# Database Setup Guide

This guide provides detailed instructions for setting up and configuring the PostgreSQL database for the Professional SMART claims processing system.

## Prerequisites

- PostgreSQL 14 or later installed
- Database created (see [INSTALLATION.md](INSTALLATION.md))
- Pro_user database user created with appropriate permissions

## Database Architecture

### Schemas
The system uses three schemas for separation of concerns:

1. **staging**: Temporary data during import/processing
2. **claims**: Core business data (encounters, service lines, flags)
3. **ml**: Machine learning and analytics data

### Key Tables

#### Claims Schema (30+ tables)
- **Organization hierarchy**: organization, region, facility
- **Personnel**: provider, coder, reviewer
- **Clinical data**: encounter, service_line, encounter_diagnosis
- **Flags and rules**: encounter_flag, service_line_flag, flag_issue
- **Financial**: rvu_data, conversion_factor, denial_event
- **Scheduling**: schedule, coding_assignment

#### Staging Schema
- **import_batch**: Tracks file imports and statistics
- **edi_transaction**: Raw EDI data
- **claim_staging**: Staged claims before processing

#### ML Schema
- **prediction**: ML model predictions
- **model_metadata**: Model versioning and configuration
- **training_data**: Historical data for training

## Migration Files

Execute these SQL files in order:

| # | File | Description |
|---|------|-------------|
| 001 | create_schemas.sql | Creates staging, claims, and ml schemas |
| 002 | create_organization_tables.sql | Organization hierarchy tables |
| 003 | create_provider_tables.sql | Provider and personnel tables |
| 004 | create_encounter_tables.sql | Encounter and claim tables |
| 005 | create_diagnosis_procedure_tables.sql | Diagnosis and procedure tables |
| 006 | create_flag_tables.sql | Flag and audit issue tables |
| 007 | create_staging_tables.sql | Import staging tables |
| 008 | create_audit_tables.sql | Audit trail tables |
| 009 | create_rvu_tables.sql | RVU and payment tables |
| 010 | create_denial_tables.sql | Denial tracking tables |
| 011 | create_schedule_tables.sql | Scheduling and assignment tables |
| 012 | create_ml_tables.sql | Machine learning tables |
| 013 | create_dashboard_views.sql | Dashboard and reporting views |
| 014 | create_utility_functions.sql | Database functions and procedures |

## Running Migrations

### Method 1: Individual Files (Recommended)

```cmd
cd C:\Users\YourUsername\pro\migrations

psql -U pro_user -d professional_smart -f 001_create_schemas.sql
psql -U pro_user -d professional_smart -f 002_create_organization_tables.sql
psql -U pro_user -d professional_smart -f 003_create_provider_tables.sql
psql -U pro_user -d professional_smart -f 004_create_encounter_tables.sql
psql -U pro_user -d professional_smart -f 005_create_diagnosis_procedure_tables.sql
psql -U pro_user -d professional_smart -f 006_create_flag_tables.sql
psql -U pro_user -d professional_smart -f 007_create_staging_tables.sql
psql -U pro_user -d professional_smart -f 008_create_audit_tables.sql
psql -U pro_user -d professional_smart -f 009_create_rvu_tables.sql
psql -U pro_user -d professional_smart -f 010_create_denial_tables.sql
psql -U pro_user -d professional_smart -f 011_create_schedule_tables.sql
psql -U pro_user -d professional_smart -f 012_create_ml_tables.sql
psql -U pro_user -d professional_smart -f 013_create_dashboard_views.sql
psql -U pro_user -d professional_smart -f 014_create_utility_functions.sql
```

### Method 2: Batch Script

Create `run_migrations.bat`:

```batch
@echo off
setlocal

set PGUSER=pro_user
set PGDATABASE=professional_smart
set PGPASSWORD=your_password

echo Running database migrations...

for %%f in (*.sql) do (
    echo Processing %%f...
    psql -f %%f
    if errorlevel 1 (
        echo ERROR: Migration %%f failed!
        exit /b 1
    )
)

echo All migrations completed successfully!
```

Run:
```cmd
cd C:\Users\YourUsername\pro\migrations
run_migrations.bat
```

## Verify Migration Success

### Check Schemas
```sql
SELECT schema_name
FROM information_schema.schemata
WHERE schema_name IN ('staging', 'claims', 'ml');
```

Expected output:
```
 schema_name
-------------
 claims
 ml
 staging
```

### Check Table Count
```sql
SELECT
    schemaname,
    COUNT(*) as table_count
FROM pg_tables
WHERE schemaname IN ('staging', 'claims', 'ml')
GROUP BY schemaname
ORDER BY schemaname;
```

Expected output (approximately):
```
 schemaname | table_count
------------+-------------
 claims     |          30
 ml         |           3
 staging    |           3
```

### Check Indexes
```sql
SELECT
    schemaname,
    COUNT(*) as index_count
FROM pg_indexes
WHERE schemaname IN ('staging', 'claims', 'ml')
GROUP BY schemaname
ORDER BY schemaname;
```

Should show 50+ indexes across all schemas.

### Check Views
```sql
SELECT
    schemaname,
    viewname
FROM pg_views
WHERE schemaname = 'claims'
ORDER BY viewname;
```

Should show 13 dashboard views.

## Initial Data Setup

### 1. Insert Flag Issue Types

The flag issue types define the 27 different types of flags that can be raised:

```sql
-- This is typically done by migration 006, but verify:
SELECT COUNT(*) FROM claims.flag_issue;
```

Expected: 27 rows

### 2. Create Initial Organization

```sql
INSERT INTO claims.organization (
    organization_id,
    organization_code,
    organization_name,
    tax_id,
    address_line1,
    city,
    state_code,
    zip_code,
    phone,
    is_active,
    created_at,
    created_by
) VALUES (
    gen_random_uuid(),
    'ORG001',
    'Sample Healthcare System',
    '12-3456789',
    '123 Healthcare Drive',
    'Medical City',
    'CA',
    '90210',
    '555-123-4567',
    true,
    CURRENT_TIMESTAMP,
    'system'
);
```

### 3. Create Sample Facility

```sql
-- Get organization_id from previous insert
INSERT INTO claims.facility (
    facility_id,
    organization_id,
    facility_code,
    facility_name,
    facility_type,
    npi,
    address_line1,
    city,
    state_code,
    zip_code,
    phone,
    is_active,
    created_at,
    created_by
) VALUES (
    gen_random_uuid(),
    (SELECT organization_id FROM claims.organization WHERE organization_code = 'ORG001'),
    'FAC001',
    'Main Hospital',
    'HOSPITAL',
    '1234567890',
    '123 Healthcare Drive',
    'Medical City',
    'CA',
    '90210',
    '555-123-4567',
    true,
    CURRENT_TIMESTAMP,
    'system'
);
```

### 4. Create Sample Provider

```sql
INSERT INTO claims.provider (
    provider_id,
    organization_id,
    npi,
    first_name,
    last_name,
    provider_type,
    specialty,
    taxonomy_code,
    is_active,
    created_at,
    created_by
) VALUES (
    gen_random_uuid(),
    (SELECT organization_id FROM claims.organization WHERE organization_code = 'ORG001'),
    '9876543210',
    'John',
    'Smith',
    'PHYSICIAN',
    'Internal Medicine',
    '207R00000X',
    true,
    CURRENT_TIMESTAMP,
    'system'
);
```

## Import RVU Data

### 1. Prepare RVU Data File

Create a CSV file `rvu_data_2024.csv` with 2024 RVU values:

```csv
hcpcs_code,year,work_rvu,pe_rvu_nonfacility,pe_rvu_facility,mp_rvu,description
99202,2024,0.93,1.19,0.60,0.09,Office Visit New Patient Level 2
99203,2024,1.60,1.76,0.96,0.16,Office Visit New Patient Level 3
99204,2024,2.60,2.40,1.43,0.25,Office Visit New Patient Level 4
99205,2024,3.50,3.14,1.94,0.33,Office Visit New Patient Level 5
99211,2024,0.18,0.55,0.15,0.03,Office Visit Established Patient Level 1
99212,2024,0.70,0.89,0.48,0.07,Office Visit Established Patient Level 2
99213,2024,1.30,1.30,0.82,0.13,Office Visit Established Patient Level 3
99214,2024,1.92,1.79,1.17,0.19,Office Visit Established Patient Level 4
99215,2024,2.80,2.40,1.56,0.27,Office Visit Established Patient Level 5
99221,2024,1.50,1.14,1.14,0.13,Hospital Initial Care Level 1
99222,2024,2.43,1.96,1.96,0.22,Hospital Initial Care Level 2
99223,2024,3.57,2.80,2.80,0.30,Hospital Initial Care Level 3
```

### 2. Import RVU Data

```sql
\copy claims.rvu_data(hcpcs_code, year, work_rvu, pe_rvu_nonfacility, pe_rvu_facility, mp_rvu, description)
FROM 'C:/path/to/rvu_data_2024.csv'
CSV HEADER;
```

### 3. Insert Conversion Factor

```sql
INSERT INTO claims.conversion_factor (
    conversion_factor_id,
    year,
    conversion_factor,
    budget_neutrality_adjustment,
    effective_date,
    termination_date,
    created_at,
    created_by
) VALUES (
    gen_random_uuid(),
    2024,
    33.2875,
    1.0000,
    '2024-01-01',
    '2024-12-31',
    CURRENT_TIMESTAMP,
    'system'
);
```

## PostgreSQL Configuration

### Recommended Settings

Edit `postgresql.conf` (located in `C:\Program Files\PostgreSQL\14\data\`):

```ini
# Memory Settings
shared_buffers = 4GB                    # 25% of total RAM
effective_cache_size = 12GB             # 75% of total RAM
work_mem = 16MB                         # For sorting operations
maintenance_work_mem = 512MB            # For maintenance operations

# Connection Settings
max_connections = 100                   # Maximum concurrent connections

# Performance Settings
random_page_cost = 1.1                  # SSD optimization
effective_io_concurrency = 200          # SSD optimization
max_worker_processes = 8                # Number of CPU cores
max_parallel_workers_per_gather = 4     # Parallel query workers
max_parallel_workers = 8                # Total parallel workers

# Write-Ahead Log
wal_buffers = 16MB
checkpoint_completion_target = 0.9
max_wal_size = 2GB
min_wal_size = 1GB

# Query Planning
default_statistics_target = 100
```

### Apply Configuration
After editing `postgresql.conf`:

```cmd
# Restart PostgreSQL service
sc stop postgresql-x64-14
sc start postgresql-x64-14
```

## Database Maintenance

### Vacuum and Analyze

Run regularly (weekly recommended):

```sql
VACUUM ANALYZE;
```

### Reindex

Run monthly or after large data loads:

```sql
REINDEX DATABASE professional_smart;
```

### Update Statistics

After importing large amounts of data:

```sql
ANALYZE claims.encounter;
ANALYZE claims.service_line;
ANALYZE claims.encounter_flag;
ANALYZE claims.service_line_flag;
```

## Backup and Restore

### Create Backup

```cmd
pg_dump -U pro_user -d professional_smart -F c -f "C:\Backups\professional_smart_%date:~-4,4%%date:~-10,2%%date:~-7,2%.backup"
```

### Create Automated Backup Script

Create `backup.bat`:

```batch
@echo off
set PGPASSWORD=your_password
set BACKUP_DIR=C:\Backups\Professional_SMART
set DATE=%date:~-4,4%%date:~-10,2%%date:~-7,2%

mkdir %BACKUP_DIR% 2>nul

pg_dump -U pro_user -d professional_smart -F c -f "%BACKUP_DIR%\professional_smart_%DATE%.backup"

echo Backup completed: professional_smart_%DATE%.backup
```

Schedule with Windows Task Scheduler to run daily.

### Restore from Backup

```cmd
pg_restore -U pro_user -d professional_smart -c "C:\Backups\professional_smart_20241014.backup"
```

## Monitoring

### Check Database Size

```sql
SELECT
    pg_size_pretty(pg_database_size('professional_smart')) as database_size;
```

### Check Table Sizes

```sql
SELECT
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as total_size
FROM pg_tables
WHERE schemaname IN ('staging', 'claims', 'ml')
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
LIMIT 20;
```

### Check Active Connections

```sql
SELECT
    count(*) as active_connections,
    usename,
    application_name
FROM pg_stat_activity
WHERE datname = 'professional_smart'
GROUP BY usename, application_name;
```

### Check Slow Queries

```sql
SELECT
    query,
    calls,
    total_time,
    mean_time,
    max_time
FROM pg_stat_statements
ORDER BY mean_time DESC
LIMIT 10;
```

## Troubleshooting

### Migration Failed

1. Check PostgreSQL logs:
```cmd
type "C:\Program Files\PostgreSQL\14\data\log\postgresql-*.log"
```

2. Roll back and retry:
```sql
DROP SCHEMA claims CASCADE;
DROP SCHEMA staging CASCADE;
DROP SCHEMA ml CASCADE;
```

Then re-run migrations from 001.

### Connection Issues

Verify `pg_hba.conf` allows local connections:
```
# IPv4 local connections:
host    all             all             127.0.0.1/32            md5
```

### Performance Issues

1. Check if vacuum is needed:
```sql
SELECT schemaname, tablename, last_vacuum, last_autovacuum
FROM pg_stat_user_tables
WHERE schemaname IN ('staging', 'claims', 'ml')
ORDER BY last_vacuum;
```

2. Check for missing indexes:
```sql
SELECT schemaname, tablename, indexname
FROM pg_indexes
WHERE schemaname IN ('staging', 'claims', 'ml')
ORDER BY tablename, indexname;
```

## Next Steps

- [Configuration Guide](CONFIGURATION.md)
- [Performance Tuning Guide](PERFORMANCE_TUNING.md)
- [Troubleshooting Guide](TROUBLESHOOTING.md)
