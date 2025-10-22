# Update Mechanism Implementation Plan

## Overview
This plan implements a robust update mechanism for Professional SMART that allows in-place upgrades without requiring uninstall/reinstall. The system will update both application binaries and database schemas automatically.

## Current State Analysis

### What Already Exists
- ✅ `pro-upgrade.exe` tool with commands: check-version, backup-database, restore-database, apply-migrations, etc.
- ✅ `UpgradeDatabase.vbs` - VBScript for database upgrade during MSI upgrade
- ✅ `DetectInstallation.vbs` - Detects if this is fresh install or upgrade
- ✅ `MajorUpgrade` element in Product.wxs with `AllowSameVersionUpgrades="yes"`
- ✅ Migration system using staging.schema_migrations table
- ✅ Working fresh install process (CreateDatabase.vbs)

### What Works Now
- Fresh installation creates database and applies all migrations successfully
- All schemas (claims, ml, staging) and tables are created correctly
- Service installation and configuration works

### What Needs to Be Fixed
- Upgrade process is configured in Product.wxs but not fully tested
- Need to verify UpgradeDatabase.vbs works correctly without breaking fresh installs
- Need proper version detection and upgrade path validation
- Service must be stopped before upgrade and restarted after

## Implementation Plan

### Phase 1: Verify and Test Current Infrastructure
**Goal**: Ensure existing upgrade infrastructure works without breaking fresh installs

- [ ] Test current upgrade logic by:
  - [ ] Install current version (1.2.2.0) fresh
  - [ ] Verify database and schemas created correctly
  - [ ] Make a code change (increment version to 1.2.3.0)
  - [ ] Run installer again as upgrade
  - [ ] Verify UpgradeDatabase.vbs executes correctly
  - [ ] Verify service stops/starts properly
  - [ ] Verify new binaries are deployed
  - [ ] Verify migrations are applied

- [ ] Document any issues found during testing

### Phase 2: Fix Identified Issues
**Goal**: Fix any problems found in Phase 1 without breaking fresh installs

- [ ] Fix UpgradeDatabase.vbs if needed (similar to CreateDatabase.vbs issues)
  - [ ] Ensure no duplicate variable declarations
  - [ ] Verify environment variable handling
  - [ ] Add proper error logging
  - [ ] Ensure pro-upgrade.exe is called correctly

- [ ] Verify Product.wxs custom action sequencing:
  - [ ] Fresh Install: DetectInstallation → WriteConfig → CreateDatabase
  - [ ] Upgrade: DetectInstallation → WriteConfig → UpgradeDatabase
  - [ ] Ensure service stops before file replacement
  - [ ] Ensure service restarts after upgrade completes

- [ ] Add registry version tracking:
  - [ ] Store current version in HKLM\SOFTWARE\ProfessionalSMART\Version
  - [ ] Update during both fresh install and upgrade
  - [ ] Use for upgrade detection

### Phase 3: Enhance Upgrade Process
**Goal**: Add robust upgrade features

- [ ] Improve backup functionality:
  - [ ] Verify backup-database command works
  - [ ] Test restore-database command
  - [ ] Implement automatic backup cleanup (keep last 5)
  - [ ] Add backup directory to ProgramData

- [ ] Add migration validation:
  - [ ] Use list-pending-migrations before upgrade
  - [ ] Log which migrations will be applied
  - [ ] Verify no schema conflicts

- [ ] Improve error handling:
  - [ ] If migrations fail, provide clear rollback instructions
  - [ ] Don't fail installer if database is inaccessible (log warning)
  - [ ] Provide detailed logs in C:\ProgramData\Professional SMART\logs\

### Phase 4: Service Management
**Goal**: Ensure service stops/starts cleanly during upgrade

- [ ] Verify StopService.vbs works correctly
- [ ] Add timeout for service stop (30 seconds)
- [ ] If service won't stop, fail upgrade with clear message
- [ ] Restart service after successful upgrade
- [ ] Verify service uses new binaries after restart

### Phase 5: Testing and Validation
**Goal**: Comprehensive testing of all scenarios

- [ ] Test fresh install (should still work exactly as before)
- [ ] Test upgrade from 1.2.2.0 to 1.2.3.0
- [ ] Test upgrade with database changes (new migrations)
- [ ] Test upgrade with no database changes
- [ ] Test upgrade failure scenarios:
  - [ ] Database unreachable
  - [ ] Migration fails
  - [ ] Service won't stop
- [ ] Test rollback using backup
- [ ] Test multiple sequential upgrades (1.2.2 → 1.2.3 → 1.2.4)

### Phase 6: Documentation
**Goal**: Document the upgrade process for users and developers

- [ ] Create UPGRADE_GUIDE.md for users
- [ ] Document version numbering strategy
- [ ] Document how to add new migrations
- [ ] Document rollback procedures
- [ ] Update installer build documentation

## Success Criteria

### Must Have
1. ✅ Fresh install continues to work exactly as it does now (no regressions)
2. Upgrade from any version to newer version works automatically
3. Database schemas are updated during upgrade
4. Application binaries are updated during upgrade
5. Service stops before upgrade and restarts after
6. Automatic database backup before migrations
7. Clear error messages if upgrade fails

### Should Have
1. Registry tracks installed version
2. Backup cleanup (keep last 5)
3. Migration validation before application
4. Detailed upgrade logs
5. Graceful handling of database connection failures

### Nice to Have
1. Upgrade UI showing progress
2. Automatic rollback on failure
3. Pre-upgrade validation checks
4. Post-upgrade health checks

## Technical Implementation Details

### Version Detection Logic
```
IF registry HKLM\SOFTWARE\ProfessionalSMART\Version exists THEN
    upgrade_mode = true
    previous_version = registry_value
ELSE
    upgrade_mode = false (fresh install)
END IF
```

### Custom Action Sequence

#### Fresh Install
```
1. FindRelatedProducts
2. DetectInstallationAction (sets INSTALLMODE=FRESH)
3. CostFinalize
4. InstallFiles (copy all files)
5. WriteConfigAction (create .env)
6. CreateDatabaseAction (create DB + apply all migrations)
7. InstallFinalize
8. Start service
```

#### Upgrade
```
1. FindRelatedProducts (sets PREVIOUSVERSION if found)
2. StopService (stop before files replaced)
3. DetectInstallationAction (sets INSTALLMODE=UPGRADE)
4. CostFinalize
5. RemoveExistingProducts (per MajorUpgrade schedule)
6. InstallFiles (copy new files)
7. WriteConfigAction (preserve existing .env, update if needed)
8. UpgradeDatabaseAction (backup + apply new migrations)
9. InstallFinalize
10. Start service
```

### Migration Management
- Migrations stored in: C:\Program Files\Professional SMART\migrations\
- Migration tracking table: staging.schema_migrations
- Each migration file: XXX_description.sql (e.g., 001_create_schemas.sql)
- pro-upgrade.exe automatically detects and applies pending migrations
- Migrations are transactional (rollback on failure)

### Backup Management
- Backups stored in: C:\ProgramData\Professional SMART\backups\
- Backup filename format: professional_smart_YYYYMMDD_HHMMSS.sql
- Keep last 5 backups automatically
- User can manually restore using: pro-upgrade.exe restore-database <backup-file>

## Risk Mitigation

### Risk: Breaking Fresh Install
**Mitigation**: Test fresh install after every change. Have rollback commits ready.

### Risk: Data Loss During Upgrade
**Mitigation**: Automatic backup before migrations. Test migrations in dev environment first.

### Risk: Service Won't Stop
**Mitigation**: Implement timeout. Fail upgrade with clear message rather than corrupt installation.

### Risk: Migration Fails Mid-Way
**Mitigation**: Use transactions. Log detailed error. Provide restore instructions.

### Risk: Installer Cached
**Mitigation**: Increment version number. Clear C:\Windows\Installer cache during testing.

## Testing Strategy

### Test Matrix
| Scenario | Expected Result | Status |
|----------|----------------|--------|
| Fresh install on clean system | DB created, all schemas/tables present | ✅ PASS |
| Upgrade 1.2.2 → 1.2.3 (no DB changes) | Binaries updated, no migrations run | ⏳ TODO |
| Upgrade 1.2.2 → 1.2.3 (with new migration) | Binaries updated, new migration applied | ⏳ TODO |
| Upgrade while service running | Service stops, upgrade succeeds, service restarts | ⏳ TODO |
| Upgrade with DB unreachable | Binaries updated, error logged, install succeeds | ⏳ TODO |
| Upgrade with migration failure | Upgrade fails, backup preserved, clear error | ⏳ TODO |
| Multiple upgrades (1.2.2→1.2.3→1.2.4) | All work sequentially | ⏳ TODO |

### Test Environment Setup
1. Create VM snapshot before testing
2. Install version 1.2.2.0 fresh
3. Add test data to database
4. Build version 1.2.3.0 with test migration
5. Run upgrade
6. Verify results
7. Restore VM snapshot for next test

## Rollback Plan

If implementation breaks fresh install or upgrade:
1. Restore CreateDatabase.vbs from commit ad123c8
2. Git revert any Product.wxs changes
3. Rebuild installer
4. Test fresh install
5. Re-approach with smaller, incremental changes

## Timeline

- Phase 1: 1 hour (testing current state)
- Phase 2: 2 hours (fix issues)
- Phase 3: 2 hours (enhancements)
- Phase 4: 1 hour (service management)
- Phase 5: 3 hours (comprehensive testing)
- Phase 6: 1 hour (documentation)

**Total Estimated Time**: 10 hours

## Notes

- Follow Rule 10: Rebuild installer after every change
- Follow Rule 9: No shortcuts - fully resolve issues
- Follow Rule 3: All failures should be loud and proud (no silent fallbacks)
- Preserve the working fresh install functionality at all costs
- Make incremental changes and test after each change
- Git commit after each successful phase
