# Upgrade Guide

**Version:** 2.12.32.0
**Last Updated:** 2025-12-27

---

## Overview

This guide covers upgrading Professional SMART between versions. The MSI installer handles most upgrades automatically, including database migrations.

---

## Quick Upgrade (Recommended)

### Step 1: Backup Database

```powershell
# Create backup
pg_dump -U postgres -Fc professional_smart > "C:\Backups\professional_smart_$(Get-Date -Format 'yyyyMMdd_HHmmss').dump"
```

### Step 2: Stop Service

```powershell
Stop-Service "Professional SMART"
```

### Step 3: Run New MSI Installer

```powershell
# Run installer (will upgrade in place)
msiexec /i ProfessionalSMART.msi
```

The installer automatically:
- Stops the existing service
- Backs up configuration
- Installs new binaries
- Runs pending database migrations
- Restarts the service

### Step 4: Verify Upgrade

```powershell
# Check service status
Get-Service "Professional SMART" | Select-Object Status, DisplayName

# Check version
"C:\Program Files\Professional SMART\bin\pro-upgrade.exe" --version

# Check migrations
psql -U postgres -d professional_smart -c "SELECT * FROM staging.schema_migrations ORDER BY version DESC LIMIT 5;"
```

---

## Version-Specific Upgrade Notes

### Upgrading to 2.12.x from 2.11.x or earlier

**New Features (2.12.23.0 - 2.12.32.0):**
- SmartProAudit master database for centralized project management
- Foreign Data Wrapper (FDW) for cross-database security queries
- Encounter view for denormalized reporting
- Encounter procedure modifiers table
- Security schema with user/role management

**Database Changes:**
- New database: `smartproaudit` (master database)
- New schema in project DB: `smartproaudit` (FDW foreign tables)
- New view: `claims.encounter_view`
- New table: `claims.encounter_procedure_modifier`

**Important Fixes:**
- Database name case preservation (v2.12.32.0)
- SmartProAudit database name case fix (v2.12.30.0)
- Encounter view column bug fix (v2.12.31.0)

**Post-Upgrade Tasks:**
```sql
-- Verify SmartProAudit database exists
\l smartproaudit

-- Verify FDW tables in project database
\dt smartproaudit.*

-- Verify encounter view
SELECT COUNT(*) FROM claims.encounter_view;
```

### Upgrading to 2.8.x from 2.7.x

**New Features:**
- Archive system for data archival (migration 056)
- Materialized views for dashboards (migration 057)
- DEFAULT_DATE constant for safer date handling

**Database Changes:**
- New schema: `archive`
- New materialized views: `claims.mv_*`
- New function: `claims.refresh_dashboard_views()`

**Post-Upgrade Tasks:**
```sql
-- Populate materialized views
SELECT * FROM claims.refresh_dashboard_views();
```

### Upgrading to 2.7.x from 2.6.x

**New Features:**
- NPI Registry link on providers (migration 052)
- 837p v2 comprehensive fields (migration 053)
- Medicare specialty codes (migration 054)
- PARTIAL import status (migration 055)

**Database Changes:**
- New table: `claims.specialty`
- New columns on `claims.encounter`: ambulance, paperwork, condition codes
- New columns on `claims.service_line`: allowed_amount, saving_amount

### Upgrading from 1.x to 2.x

**Major Changes:**
- Complete schema restructuring
- New staging pipeline (raw_claims table)
- Rule configuration system
- Provider enrichment queue

**Recommended:** Perform a fresh install with data migration.

---

## Manual Upgrade Process

Use this process if the MSI upgrade fails or for custom deployments.

### Step 1: Create Full Backup

```powershell
# Database backup
pg_dump -U postgres -Fc professional_smart > backup.dump

# Configuration backup
Copy-Item "C:\Program Files\Professional SMART\.env" "C:\Backups\.env.backup"
```

### Step 2: Stop Service

```powershell
Stop-Service "Professional SMART"
```

### Step 3: Replace Binaries

```powershell
# Copy new binaries
Copy-Item "target\release\pro-service.exe" "C:\Program Files\Professional SMART\bin\" -Force
Copy-Item "target\release\pro-upgrade.exe" "C:\Program Files\Professional SMART\bin\" -Force
```

### Step 4: Run Migrations

```powershell
cd "C:\Program Files\Professional SMART\bin"
.\pro-upgrade.exe apply-migrations `
    --db-host localhost `
    --db-port 5432 `
    --db-name professional_smart `
    --db-user postgres `
    --db-password YOUR_PASSWORD
```

### Step 5: Start Service

```powershell
Start-Service "Professional SMART"
```

---

## Rollback Procedure

### If Upgrade Fails

**Step 1: Stop Service**
```powershell
Stop-Service "Professional SMART" -Force
```

**Step 2: Restore Database**
```powershell
# Drop and recreate database
psql -U postgres -c "DROP DATABASE professional_smart;"
psql -U postgres -c "CREATE DATABASE professional_smart;"

# Restore from backup
pg_restore -U postgres -d professional_smart backup.dump
```

**Step 3: Reinstall Previous Version**
```powershell
msiexec /x ProfessionalSMART.msi /qn
# Install previous version MSI if available
```

### Partial Rollback (Migrations Only)

If only specific migrations need to be rolled back:

```sql
-- Example: Rollback materialized views (migration 057)
DROP MATERIALIZED VIEW IF EXISTS claims.mv_management_overview;
DROP MATERIALIZED VIEW IF EXISTS claims.mv_denial_by_payer;
DROP MATERIALIZED VIEW IF EXISTS claims.mv_procedure_volume;
DROP MATERIALIZED VIEW IF EXISTS claims.mv_provider_productivity;
DROP FUNCTION IF EXISTS claims.refresh_dashboard_views();

DELETE FROM staging.schema_migrations WHERE version = '057';
```

See [MIGRATION_STATUS.md](MIGRATION_STATUS.md) for specific rollback SQL.

---

## Pre-Upgrade Checklist

- [ ] Database backup created and tested
- [ ] Configuration backup created
- [ ] Sufficient disk space available
- [ ] Users notified of maintenance window
- [ ] Processing queue is empty (no pending files)
- [ ] Service can be stopped safely

### Check Processing Queue

```sql
-- Ensure no pending work
SELECT COUNT(*) FROM staging.file_processing_queue WHERE processing_status IN ('PENDING', 'PROCESSING');
SELECT COUNT(*) FROM staging.raw_claims WHERE processing_status = 'PENDING';
```

---

## Post-Upgrade Checklist

- [ ] Service is running
- [ ] Version is correct
- [ ] Migrations applied successfully
- [ ] Test file processes correctly
- [ ] Dashboard views refreshed (if applicable)
- [ ] Logs show no errors

### Verify Service Health

```powershell
# Service status
Get-Service "Professional SMART"

# Recent log entries
Get-Content "C:\ProgramData\Professional SMART\logs\pro-service.log" -Tail 20
```

### Verify Database

```sql
-- Check migration status
SELECT * FROM staging.schema_migrations ORDER BY version DESC LIMIT 10;

-- Check table counts
SELECT
    schemaname,
    COUNT(*) as table_count
FROM pg_tables
WHERE schemaname IN ('claims', 'staging', 'ml', 'archive')
GROUP BY schemaname;
```

---

## Troubleshooting Upgrades

### Migration Fails with "already exists"

**Cause:** Partial migration from previous attempt.

**Solution:**
```sql
-- Check what was created
SELECT * FROM pg_tables WHERE tablename = 'TABLE_NAME';

-- Manually mark migration as complete (if objects exist)
INSERT INTO staging.schema_migrations (version) VALUES ('XXX');
```

### Service Won't Start After Upgrade

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md#service-wont-start)

### Data Missing After Upgrade

**Check archive tables:**
```sql
SELECT * FROM archive.v_archive_statistics;
```

If data was archived, restore if needed:
```sql
SELECT * FROM archive.restore_encounters(ARRAY[encounter_ids]);
```

---

## Silent/Unattended Upgrade

For automated deployments:

```powershell
# Silent install with logging
msiexec /i ProfessionalSMART.msi /qn /l*v upgrade.log `
    DB_HOST=localhost `
    DB_PORT=5432 `
    DB_NAME=professional_smart `
    DB_USER=postgres `
    DB_PASSWORD=YOUR_PASSWORD

# Check exit code
if ($LASTEXITCODE -ne 0) {
    Write-Error "Upgrade failed. Check upgrade.log for details."
}
```

---

## Related Documentation

- [INSTALLATION.md](INSTALLATION.md) - Fresh installation
- [MIGRATION_STATUS.md](MIGRATION_STATUS.md) - Migration details
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Common issues
- [DATABASE_SCHEMA_REFERENCE.md](DATABASE_SCHEMA_REFERENCE.md) - Schema documentation
