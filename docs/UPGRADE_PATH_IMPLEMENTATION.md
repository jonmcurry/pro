# Upgrade Path Implementation Plan

## Overview
This document outlines the implementation plan for adding a proper upgrade path to Professional SMART installations. Currently, all deployments require full uninstallation (including database destruction) before redeployment. This plan enables seamless upgrades that preserve data and configuration.

## Current State Analysis

### Installer
- MSI using WiX 3.x toolset
- Version: 1.0.0.0 (hardcoded in Product.wxs)
- UpgradeCode: 8F8D8F8D-8F8D-8F8D-8F8D-8F8D8F8D8F8D (stable across versions)
- MajorUpgrade element exists but only prevents downgrades
- Product Id uses "*" (generates new GUID each build)

### Database
- PostgreSQL database created during installation
- 19 migration files (001-019) applied in order during fresh install
- No migration tracking table exists
- No version metadata stored in database
- Uninstall completely drops database

### Services
- Windows service "ProfessionalSMART" installed via ServiceInstall
- Service stopped during uninstall via VBScript
- No upgrade-aware service handling

## Implementation Checklist

### Phase 1: Database Version Tracking
- [ ] Create migration 020_create_version_tracking.sql
  - [ ] Create staging.schema_migrations table
    - [ ] migration_name VARCHAR(255) PRIMARY KEY
    - [ ] applied_at TIMESTAMP NOT NULL DEFAULT NOW()
    - [ ] checksum TEXT (for integrity verification)
  - [ ] Create staging.application_version table
    - [ ] version VARCHAR(50) PRIMARY KEY
    - [ ] installed_at TIMESTAMP NOT NULL DEFAULT NOW()
    - [ ] upgraded_from VARCHAR(50)
  - [ ] Backfill schema_migrations with existing migrations 001-019

- [ ] Create migration 021_insert_initial_version.sql
  - [ ] Insert version 1.0.0 into staging.application_version
  - [ ] Mark as baseline installation

### Phase 2: Version Detection and Backup Tools
- [ ] Create Rust crate: pro-upgrade-manager
  - [ ] Implement version detection
    - [ ] Query staging.application_version table
    - [ ] Return current version or None if not exists
  - [ ] Implement database backup functionality
    - [ ] Use pg_dump via Command API
    - [ ] Store backups in %PROGRAMDATA%\Professional SMART\backups\
    - [ ] Include timestamp in backup filename
    - [ ] Compress backup using gzip
    - [ ] Verify backup integrity after creation
  - [ ] Implement migration tracker
    - [ ] Check which migrations have been applied
    - [ ] Return list of pending migrations
    - [ ] Validate migration checksums
  - [ ] Implement rollback capability
    - [ ] Restore from backup file
    - [ ] Use pg_restore via Command API
    - [ ] Log rollback operations

- [ ] Create executable: pro-upgrade.exe
  - [ ] Command: check-version
  - [ ] Command: backup-database
  - [ ] Command: restore-database <backup-file>
  - [ ] Command: list-pending-migrations
  - [ ] Command: apply-migrations
  - [ ] Add to installer bin folder

### Phase 3: Migration System Enhancement
- [ ] Update CreateDatabase.vbs to detect existing installation
  - [ ] Query staging.application_version table
  - [ ] If table doesn't exist: Fresh install (run all migrations)
  - [ ] If table exists: Upgrade mode (run pending migrations only)
  - [ ] Check schema_migrations to determine which migrations to skip
  - [ ] Apply only new migrations

- [ ] Create UpgradeDatabase.vbs
  - [ ] Call pro-upgrade.exe check-version
  - [ ] Display current version and target version
  - [ ] Call pro-upgrade.exe backup-database
  - [ ] Wait for backup completion
  - [ ] Apply pending migrations only
  - [ ] Update staging.application_version table
  - [ ] Log all upgrade operations
  - [ ] Handle rollback on failure

### Phase 4: Installer Upgrade Logic
- [ ] Update Product.wxs
  - [ ] Change version to 1.1.0.0
  - [ ] Keep UpgradeCode identical (critical for upgrade detection)
  - [ ] Update MajorUpgrade element
    - [ ] Set AllowSameVersionUpgrades="yes"
    - [ ] Set Schedule="afterInstallInitialize"
    - [ ] Add proper upgrade messages
  - [ ] Add Property for detecting existing installation
    - [ ] Use RegistrySearch or FileSearch
    - [ ] Set PREVIOUSVERSION property

- [ ] Create DetectInstallation.vbs
  - [ ] Check if database exists
  - [ ] Check if version tracking tables exist
  - [ ] Set MSI properties: INSTALLMODE (FRESH or UPGRADE)
  - [ ] Store detected version in property DETECTEDVERSION

- [ ] Update InstallExecuteSequence
  - [ ] Add DetectInstallationAction (before CreateDatabase)
  - [ ] Condition CreateDatabaseAction on fresh install only
  - [ ] Add UpgradeDatabaseAction (conditioned on upgrade)
  - [ ] Add BackupDatabaseAction (before upgrade, optional via checkbox)
  - [ ] Ensure service is stopped before file replacement
  - [ ] Restart service after successful upgrade

### Phase 5: User Interface Enhancements
- [ ] Create UpgradeOptionsDlg.wxs
  - [ ] Display current version (DETECTEDVERSION)
  - [ ] Display target version (from Product/@Version)
  - [ ] Checkbox: "Create database backup before upgrade" (default: checked)
  - [ ] Checkbox: "Preserve existing configuration" (default: checked)
  - [ ] Show backup location
  - [ ] Warning text about upgrade process

- [ ] Update UI flow in Product.wxs
  - [ ] Show UpgradeOptionsDlg only when INSTALLMODE=UPGRADE
  - [ ] Skip database config dialog during upgrade
  - [ ] Add progress messages for upgrade steps

### Phase 6: Configuration Preservation
- [ ] Update WriteConfig.vbs
  - [ ] Check if .env file exists in ProgramData
  - [ ] If exists and preserve flag set: Skip overwriting
  - [ ] If exists: Create .env.backup with timestamp
  - [ ] Merge new settings with existing (add new keys only)
  - [ ] Log configuration preservation actions

- [ ] Create MergeConfig.vbs
  - [ ] Read existing .env file
  - [ ] Read new .env.template
  - [ ] Identify new configuration keys
  - [ ] Append new keys with default values
  - [ ] Preserve all existing values
  - [ ] Add comments showing which keys are new

### Phase 7: Testing and Validation
- [ ] Create test migration files (022, 023) for testing
- [ ] Test fresh installation (baseline)
  - [ ] Verify all 21 migrations applied
  - [ ] Verify version tracking tables exist
  - [ ] Verify version is 1.1.0
- [ ] Test upgrade from 1.0.0 to 1.1.0
  - [ ] Install version 1.0.0 first (19 migrations)
  - [ ] Manually add version tracking (simulate 1.0.x with tracking)
  - [ ] Run upgrade installer
  - [ ] Verify only new migrations applied (020, 021)
  - [ ] Verify data preserved
  - [ ] Verify configuration preserved
  - [ ] Verify service restarts correctly
- [ ] Test upgrade from pre-version-tracking (simulate current state)
  - [ ] Install without version tracking tables
  - [ ] Run upgrade installer
  - [ ] Verify upgrade detects no version table
  - [ ] Verify backfill of schema_migrations
  - [ ] Verify only new migrations applied
- [ ] Test backup and restore
  - [ ] Create backup via pro-upgrade.exe
  - [ ] Verify backup file created
  - [ ] Test restore from backup
  - [ ] Verify data integrity after restore
- [ ] Test rollback on failed migration
  - [ ] Create intentionally failing migration
  - [ ] Verify rollback triggered
  - [ ] Verify database restored from backup
- [ ] Test downgrade prevention
  - [ ] Try to install older version over newer
  - [ ] Verify proper error message shown

### Phase 8: Documentation
- [ ] Update INSTALLATION.md
  - [ ] Add upgrade instructions section
  - [ ] Document backup location
  - [ ] Document manual rollback procedure
  - [ ] Add troubleshooting for upgrade issues
- [ ] Update DATABASE_SETUP.md
  - [ ] Document version tracking tables
  - [ ] Document migration checksum system
  - [ ] Document how to manually apply migrations
- [ ] Create UPGRADE_GUIDE.md
  - [ ] Step-by-step upgrade instructions
  - [ ] Pre-upgrade checklist
  - [ ] Post-upgrade verification steps
  - [ ] Rollback procedure
  - [ ] FAQ for common upgrade issues
- [ ] Update README.md
  - [ ] Mention upgrade capability
  - [ ] Link to UPGRADE_GUIDE.md

### Phase 9: Cleanup and Finalization
- [ ] Remove any temporary test files
- [ ] Clean up VBScript logging to production level
- [ ] Verify all error messages are user-friendly
- [ ] Ensure all file paths use proper escaping
- [ ] Test on clean Windows Server environment
- [ ] Test on Windows 10/11 workstation
- [ ] Rebuild installer with final changes
- [ ] Version final MSI as 1.1.0.0
- [ ] Tag release in git

## Technical Design Details

### Version Tracking Schema

```sql
-- staging.schema_migrations
CREATE TABLE staging.schema_migrations (
    migration_name VARCHAR(255) PRIMARY KEY,
    applied_at TIMESTAMP NOT NULL DEFAULT NOW(),
    checksum TEXT NOT NULL,
    execution_time_ms INTEGER
);

-- staging.application_version
CREATE TABLE staging.application_version (
    version VARCHAR(50) PRIMARY KEY,
    installed_at TIMESTAMP NOT NULL DEFAULT NOW(),
    upgraded_from VARCHAR(50),
    notes TEXT
);
```

### Backup File Naming Convention
```
professional_smart_backup_YYYYMMDD_HHMMSS.sql.gz
Example: professional_smart_backup_20251020_143052.sql.gz
```

### Migration Checksum Calculation
- Use SHA-256 hash of migration file content
- Store in schema_migrations table
- Verify on upgrade to detect manual changes
- Warn if checksum mismatch detected

### Upgrade Detection Logic
```
1. Check registry: HKLM\SOFTWARE\ProfessionalSMART\Version
2. If not found, check database: SELECT version FROM staging.application_version ORDER BY installed_at DESC LIMIT 1
3. If both not found, check if database exists with claims schema
4. If database exists but no version: Treat as 1.0.0 (legacy)
5. If database doesn't exist: Fresh install
```

### Migration Order Enforcement
- Migrations must be named: NNN_description.sql (where NNN is zero-padded number)
- Sorted alphabetically before execution
- Each migration is atomic (wrapped in transaction where possible)
- Failed migration stops upgrade process
- Successful migrations are immediately recorded

## Risk Mitigation

### Backup Strategy
- Automatic backup before upgrade (default: enabled)
- Backup stored outside installation directory
- Backup retained for 30 days (configurable)
- User can opt-out but must confirm

### Rollback Strategy
- Automatic rollback on migration failure
- Manual rollback tool provided (pro-upgrade.exe restore-database)
- Service stopped during rollback
- Clear error messages guide user through recovery

### Data Integrity
- All migrations run in transactions where possible
- Checksum verification prevents corrupted migrations
- Foreign key constraints maintained throughout
- Indexes rebuilt after major schema changes

### Compatibility
- Maintain backward compatibility with 1.0.0 database
- Support upgrades from any 1.x version to any newer 1.x version
- Breaking changes only in major version (2.0.0)
- API endpoint compatibility maintained within major version

## Success Criteria

1. User can upgrade from 1.0.0 to 1.1.0 without data loss
2. All existing claims, providers, and configuration preserved
3. Service restarts automatically after upgrade
4. Backup created and verified before upgrade
5. Rollback works if upgrade fails
6. Clear progress indicators during upgrade
7. Comprehensive error messages on failure
8. Documentation covers all upgrade scenarios
9. No manual SQL scripts required
10. Upgrade completes in under 5 minutes for typical database

## Timeline Estimate

- Phase 1: Database Version Tracking - 4 hours
- Phase 2: Version Detection and Backup Tools - 8 hours
- Phase 3: Migration System Enhancement - 6 hours
- Phase 4: Installer Upgrade Logic - 8 hours
- Phase 5: User Interface Enhancements - 6 hours
- Phase 6: Configuration Preservation - 4 hours
- Phase 7: Testing and Validation - 12 hours
- Phase 8: Documentation - 4 hours
- Phase 9: Cleanup and Finalization - 4 hours

**Total Estimated Time: 56 hours (7 working days)**

## Dependencies

- WiX Toolset 3.x installed
- PostgreSQL client tools (pg_dump, pg_restore)
- Rust toolchain for building pro-upgrade.exe
- Test environment with PostgreSQL database
- Windows SDK for testing service operations

## Questions to Address

1. Should we support upgrading from installations without version tracking?
   - **Recommendation: Yes, detect legacy 1.0.0 and backfill version info**

2. How many backup files should we retain?
   - **Recommendation: Keep last 5 backups or 30 days, whichever is greater**

3. Should we add a GUI for upgrade management?
   - **Recommendation: Phase 2 enhancement, CLI first**

4. What happens if user modifies migration files?
   - **Recommendation: Checksum validation fails, warn user, allow override flag**

5. Should we support downgrades?
   - **Recommendation: No, too risky. Block with clear error message**

6. How do we handle service downtime during upgrade?
   - **Recommendation: Stop service, upgrade, restart. Typical downtime: 1-2 minutes**

7. What if backup fails during upgrade?
   - **Recommendation: Abort upgrade, require manual intervention**

8. Should we compress backups?
   - **Recommendation: Yes, use gzip. Typical 10:1 compression ratio**
