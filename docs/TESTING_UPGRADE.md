# Testing the Upgrade Path

## Current Situation

You're seeing "Change, Modify, or Repair" because Windows Installer detects version 1.1.0 is already installed, and you're trying to install the same version again.

## How to Test Upgrade Properly

### Option 1: Simulate 1.0.0 to 1.1.0 Upgrade (Recommended)

Since you're currently on 1.1.0, you need to simulate having 1.0.0 installed:

#### Step 1: Prepare a 1.0.0 MSI

1. **Revert Product.wxs to 1.0.0:**
   ```cmd
   cd C:\Users\jonmc\dev\pro\installer
   ```

2. **Edit Product.wxs temporarily:**
   - Change line 8: `Version="1.0.0.0"` (instead of 1.1.0.0)
   - Remove lines 113-118 (UpgradeExecutable component)
   - Remove lines 196-197 (Migration020 and Migration021 files)
   - Remove line 301 (UpgradeExecutable ComponentRef)
   - Change line 495: `Value="1.0.0.0"` (in VersionRegistryKey)

3. **Build the 1.0.0 MSI:**
   ```cmd
   cargo build --release
   cd installer
   candle Product.wxs DatabaseConfigDlg.wxs PrerequisiteDlg.wxs
   light -ext WixUIExtension Product.wixobj DatabaseConfigDlg.wixobj PrerequisiteDlg.wixobj -out ProfessionalSMART_1.0.0.msi
   ```

#### Step 2: Install 1.0.0

1. **Uninstall current version:**
   ```cmd
   # Via Control Panel or:
   wmic product where "name='Professional SMART'" call uninstall
   ```

2. **Install the 1.0.0 MSI:**
   ```cmd
   msiexec /i ProfessionalSMART_1.0.0.msi /l*v install_1.0.0.log
   ```

3. **Verify 1.0.0 is installed:**
   ```cmd
   reg query "HKLM\SOFTWARE\ProfessionalSMART" /v Version
   ```
   Should show: `1.0.0.0`

#### Step 3: Build and Install 1.1.0 (Upgrade)

1. **Revert Product.wxs back to 1.1.0:**
   - Restore all changes (or use git to revert)
   - Ensure Version="1.1.0.0"

2. **Rebuild 1.1.0 MSI:**
   ```cmd
   cargo build --release
   cd installer
   candle Product.wxs DatabaseConfigDlg.wxs PrerequisiteDlg.wxs
   light -ext WixUIExtension Product.wixobj DatabaseConfigDlg.wixobj PrerequisiteDlg.wixobj -out ProfessionalSMART_1.1.0.msi
   ```

3. **Run the upgrade:**
   ```cmd
   msiexec /i ProfessionalSMART_1.1.0.msi /l*v upgrade_1.0_to_1.1.log
   ```

4. **Verify upgrade:**
   ```cmd
   reg query "HKLM\SOFTWARE\ProfessionalSMART" /v Version
   cd "C:\Program Files\Professional SMART\bin"
   pro-upgrade.exe check-version
   ```

### Option 2: Force Reinstall with Upgrade Logic (Quick Test)

If you want to test without rolling back to 1.0.0:

1. **Uninstall current version:**
   ```cmd
   msiexec /x ProfessionalSMART.msi /l*v uninstall.log
   # Or use Add/Remove Programs
   ```

2. **Delete registry key (to simulate fresh state):**
   ```cmd
   reg delete "HKLM\SOFTWARE\ProfessionalSMART" /f
   ```

3. **Manually adjust database to simulate 1.0.0:**
   ```sql
   -- Connect to database
   psql -h localhost -U postgres -d professional_smart

   -- Drop version tracking tables (to simulate 1.0.0)
   DROP TABLE IF EXISTS staging.application_version CASCADE;
   DROP TABLE IF EXISTS staging.schema_migrations CASCADE;

   -- Exit
   \q
   ```

4. **Install 1.1.0 MSI:**
   ```cmd
   msiexec /i ProfessionalSMART_1.1.0.msi /l*v install.log
   ```

5. **Verify it detected legacy installation:**
   - Check install.log for "Legacy installation" or "Upgrade detected" messages

### Option 3: Version Bump Test (Simplest for Future)

For testing future upgrades (1.1.0 to 1.2.0):

1. **Change version to 1.2.0 in Product.wxs:**
   ```xml
   Version="1.2.0.0"
   ```

2. **Update VersionRegistryKey value:**
   ```xml
   Value="1.2.0.0"
   ```

3. **Rebuild MSI:**
   ```cmd
   cargo build --release
   cd installer
   candle Product.wxs DatabaseConfigDlg.wxs PrerequisiteDlg.wxs
   light -ext WixUIExtension Product.wixobj DatabaseConfigDlg.wixobj PrerequisiteDlg.wixobj -out ProfessionalSMART_1.2.0.msi
   ```

4. **Install (should detect 1.1.0 and upgrade):**
   ```cmd
   msiexec /i ProfessionalSMART_1.2.0.msi /l*v upgrade_to_1.2.log
   ```

## What You Should See During Upgrade

### Successful Upgrade Signs:

1. **MSI Log shows:**
   ```
   Professional SMART Installer: DetectInstallation: Found existing installation in registry: 1.0.0.0
   Professional SMART Installer: DetectInstallation: Installation mode = UPGRADE
   Professional SMART Installer: UpgradeDatabase: Starting database upgrade process
   Professional SMART Installer: UpgradeDatabase: Backup is enabled, creating database backup...
   Professional SMART Installer: UpgradeDatabase: SUCCESS - Backup created
   Professional SMART Installer: UpgradeDatabase: Applying pending database migrations...
   Professional SMART Installer: UpgradeDatabase: SUCCESS - All migrations applied successfully
   ```

2. **Registry updated:**
   ```cmd
   reg query "HKLM\SOFTWARE\ProfessionalSMART" /v Version
   # Shows new version
   ```

3. **Database has version tracking:**
   ```sql
   SELECT * FROM staging.application_version;
   SELECT COUNT(*) FROM staging.schema_migrations;
   ```

4. **Service is running:**
   ```cmd
   sc query ProfessionalSMART
   ```

## Troubleshooting

### Still Shows "Change, Modify, Repair"

**Cause:** Same version (1.1.0) already installed.

**Solution:** Either:
- Use Option 1 to test 1.0.0 → 1.1.0
- Use Option 3 to test 1.1.0 → 1.2.0

### "Another version of this product is already installed"

**Cause:** Different Product GUID but same UpgradeCode.

**Solution:** This is expected! The MajorUpgrade element should handle it automatically. Check the log for:
```
Action start: RemoveExistingProducts
```

If this is missing, the upgrade isn't being triggered properly.

### Upgrade Runs But Version Not Updated

**Cause:** VersionRegistryKey not being updated.

**Solution:** Check that VersionRegistryKey component is in MainApplication feature and has the correct version number.

## Quick Verification Commands

After any installation/upgrade:

```cmd
# Check installed version in registry
reg query "HKLM\SOFTWARE\ProfessionalSMART" /v Version

# Check database version
cd "C:\Program Files\Professional SMART\bin"
set DB_PASSWORD=your_password
pro-upgrade.exe check-version

# Check if pro-upgrade.exe exists (1.1.0+ only)
dir "C:\Program Files\Professional SMART\bin\pro-upgrade.exe"

# Check migrations in database
set DB_PASSWORD=your_password
pro-upgrade.exe list-pending-migrations

# Check if backups directory exists (1.1.0+ only)
dir "C:\ProgramData\Professional SMART\backups"

# Check service status
sc query ProfessionalSMART
```

## Current State Analysis

Based on your question, you're currently:
- **Installed Version:** 1.1.0.0
- **Trying to Install:** 1.1.0.0 (same version)
- **Result:** Windows shows "Change, Modify, or Repair"

This is correct behavior! To test upgrades, you need to either:
1. Install from a lower version (1.0.0)
2. Or bump to a higher version (1.2.0)

## Recommended Next Steps

1. **To test the upgrade path from 1.0.0:**
   - Follow Option 1 above (simulate 1.0.0 → 1.1.0)

2. **To test for production:**
   - Keep 1.0.0 MSI for existing clients
   - They can directly run 1.1.0 MSI and it will upgrade automatically

3. **For future development:**
   - When you make changes for version 1.2.0, change the version number
   - Current 1.1.0 installations will smoothly upgrade to 1.2.0

## Summary

**You're not doing anything wrong!** Windows Installer correctly detects that 1.1.0 is installed and won't upgrade to the same version. To test the upgrade:
- Either roll back to 1.0.0 and upgrade to 1.1.0
- Or bump to 1.2.0 and upgrade from 1.1.0

The upgrade infrastructure is working correctly - you just need a version difference to trigger the upgrade path.
