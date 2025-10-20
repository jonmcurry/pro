# Fresh Install Test Plan

## Current Situation
- User reports schemas still not being created
- Last install log is from 18:08:14 (6:08 PM) - BEFORE latest fixes
- MSI was rebuilt at ~21:54 (9:54 PM) with all fixes
- User has NOT tested the newly rebuilt MSI yet

## Root Cause of Previous Failures
The old MSI had CreateDatabase.vbs that returned success even when migrations failed (silent failure, violates Rule 3).

## What Changed in Latest Build

### Fix 1: No More Silent Failures
- **File**: `installer/CreateDatabase.vbs`
- **Change**: Now returns error code 3 when migrations fail, causing installation to abort
- **Result**: User will see "Installation failed" instead of silent success

### Fix 2: Migration Framework Fixes
- **File**: `crates/pro-upgrade-manager/src/migration.rs`
- **Changes**:
  - Creates staging schema and schema_migrations table automatically
  - Splits SQL statements properly
  - Handles PostgreSQL dollar-quoted functions
  - Executes each statement individually

## Pre-Test Verification

### Step 1: Verify Current Database State
```sql
-- Check if database exists
SELECT datname FROM pg_database WHERE datname = 'professional_smart';

-- Check current schemas
SELECT schema_name FROM information_schema.schemata
WHERE schema_name IN ('claims', 'staging', 'ml')
ORDER BY schema_name;

-- Count tables
SELECT schemaname, COUNT(*)
FROM pg_tables
WHERE schemaname IN ('claims', 'staging', 'ml')
GROUP BY schemaname
ORDER BY schemaname;
```

### Step 2: Clean Start (REQUIRED)
Since there may be leftover state from failed installs, we need a clean slate:

```sql
-- Drop all schemas to start fresh
DROP SCHEMA IF EXISTS claims CASCADE;
DROP SCHEMA IF EXISTS staging CASCADE;
DROP SCHEMA IF EXISTS ml CASCADE;
```

## Test Steps

### Step 1: Uninstall Any Existing Installation
```cmd
-- Check if Professional SMART is installed
wmic product where "name like '%Professional%SMART%'" get name,version

-- If installed, uninstall
wmic product where "name like '%Professional%SMART%'" call uninstall
```

### Step 2: Run Fresh Install with Latest MSI
```cmd
cd c:\Users\jonmc\dev\pro\installer
msiexec /i ProfessionalSMART.msi /l*v C:\temp\fresh_install_NEW.log
```

### Step 3: Complete Installation
- Follow the wizard
- Enter database credentials:
  - Host: localhost
  - Port: 5432
  - Database: professional_smart
  - User: postgres
  - Password: ClearToFly1

## Expected Outcomes

### Success Scenario
1. Installer completes successfully
2. All schemas created (claims, staging, ml)
3. All tables created:
   - claims: ~31 tables
   - staging: ~12 tables
   - ml: ~6 tables
4. Log shows: "CreateDatabase: SUCCESS - All migrations applied successfully"

### Failure Scenario (Loud Failure - Rule 3)
1. Installer shows error dialog: "Installation failed"
2. Log shows: "CreateDatabase: ERROR - Migration application failed"
3. Log shows: "CreateDatabase: INSTALLATION WILL FAIL"
4. Installation rolls back
5. NO schemas are created (clean rollback)

## Verification After Install

### If Installation Succeeded
```sql
-- Verify schemas
SELECT schema_name FROM information_schema.schemata
WHERE schema_name IN ('claims', 'staging', 'ml');
-- Expected: 3 rows

-- Verify tables
SELECT schemaname, COUNT(*)
FROM pg_tables
WHERE schemaname IN ('claims', 'staging', 'ml')
GROUP BY schemaname;
-- Expected: claims=31, staging=12, ml=6 (approximately)

-- Check migration tracking
SELECT COUNT(*) FROM staging.schema_migrations;
-- Expected: 12-21 (number of migrations applied)
```

### If Installation Failed
```cmd
-- Check the log file
type C:\temp\fresh_install_NEW.log | findstr /i "CreateDatabase ERROR"

-- Verify nothing was left behind
```sql
SELECT schema_name FROM information_schema.schemata
WHERE schema_name IN ('claims', 'staging', 'ml');
-- Expected: 0 rows (clean rollback)
```

## Troubleshooting

### If Installer Fails with "pro-upgrade.exe not found"
- Check: `C:\Program Files\Professional SMART\bin\pro-upgrade.exe` exists
- Check: File size is ~2-5 MB (not 0 bytes)
- Solution: Copy from `c:\Users\jonmc\dev\pro\target\release\pro-upgrade.exe`

### If Installer Fails with "Migration application failed"
- Check log for specific SQL error
- Common issues:
  - PostgreSQL not running: `net start postgresql-x64-16`
  - Wrong password: Double-check credentials
  - Database doesn't exist: Create it first

### If Installer Succeeds but Schemas Missing
- **This should NOT happen anymore** (was the bug we fixed)
- If it does happen, check log for: "CreateDatabase: SUCCESS"
- If shows SUCCESS but no schemas: **BUG IN OUR FIX** - investigate

## Next Steps After Successful Test
1. Re-enable RegistrySearch in Product.wxs (currently disabled)
2. Revert UpgradeCode to original value
3. Test upgrade path from this version
4. Document the working upgrade process
