# Solution: Uninstall Old Version Before Fresh Install

## Problem

Fresh installation is failing because:
1. Old version 1.2.0.0 is still installed (Product Code: `{D1FEA2D9-872B-4058-940D-D31FC0BFECC6}`)
2. The old 1.2.0 MSI has the broken UpgradeDatabase.vbs that returns ERROR when pro-upgrade.exe is missing
3. During RemoveExistingProducts, the old MSI tries to run UpgradeDatabaseAction and fails with error code 3
4. This causes the entire installation to fail

## Why We Can't Fix the Old MSI

**The old 1.2.0 MSI is already installed and can't be changed.** Our fixes only apply to NEW installations (1.2.1+).

## The Solution

**MUST uninstall the old version completely BEFORE installing the new one.**

## Step-by-Step Instructions

### Step 1: Uninstall Old Version

Run the provided batch file:
```cmd
cd c:\Users\jonmc\dev\pro\installer
UNINSTALL_OLD_VERSION.bat
```

Or manually:
```cmd
msiexec /x {D1FEA2D9-872B-4058-940D-D31FC0BFECC6} /quiet /l*v C:\temp\uninstall.log
```

### Step 2: Verify Uninstall

Check that nothing remains:
```powershell
# Check registry
reg query "HKLM\SOFTWARE\ProfessionalSMART" /v Version
# Should return: ERROR: The system was unable to find the specified registry key or value.

# Check installation directory
dir "C:\Program Files\Professional SMART"
# Should return: File Not Found

# Check ProgramData
dir "C:\ProgramData\Professional SMART"
# Should exist but be empty or have backups only
```

### Step 3: Clean Up (Optional but Recommended)

Remove all remnants:
```cmd
rmdir /s /q "C:\Program Files\Professional SMART"
rmdir /s /q "C:\ProgramData\Professional SMART"
```

### Step 4: Install Fresh 1.2.1.0

Now run the new MSI:
```cmd
cd c:\Users\jonmc\dev\pro\installer
msiexec /i ProfessionalSMART.msi /l*v C:\temp\fresh_install.log
```

**EXPECTED BEHAVIOR:**
1. No previous version detected
2. Database dialog will appear (this is correct - no .env exists)
3. Enter credentials: localhost, 5432, professional_smart, postgres, ClearToFly1
4. Installation completes successfully
5. .env file created at `C:\ProgramData\Professional SMART\config\.env`

### Step 5: Verify Fresh Install

```cmd
# Check version
reg query "HKLM\SOFTWARE\ProfessionalSMART" /v Version
# Should show: 1.2.1.0

# Check .env created
type "C:\ProgramData\Professional SMART\config\.env"
# Should show database credentials

# Check service
sc query ProfessionalSMART
# Should show: SERVICE_NAME: ProfessionalSMART
```

### Step 6: Test Upgrade (Optional)

Now you can test the credential loading feature:

```cmd
# Build version 1.2.2.0
cd c:\Users\jonmc\dev\pro\installer
powershell -ExecutionPolicy Bypass -File .\build-msi.ps1 -Patch -NoBuild

# Install upgrade
msiexec /i ProfessionalSMART.msi /l*v C:\temp\upgrade_test.log
```

**EXPECTED BEHAVIOR:**
1. Previous version 1.2.1.0 detected
2. .env file found
3. Credentials loaded automatically
4. Database dialog SKIPPED
5. Upgrade completes without prompts

## Why This Happened

The old 1.2.0 installation:
- Has Product Code `{D1FEA2D9-872B-4058-940D-D31FC0BFECC6}`
- Contains UpgradeDatabase.vbs that returns error code 3 when pro-upgrade.exe missing
- Can't be fixed retroactively

The new 1.2.1+ installations:
- Will have different Product Codes (generated dynamically with `Product Id="*"`)
- Contain fixed UpgradeDatabase.vbs that returns success code 1 when pro-upgrade.exe missing
- Will upgrade cleanly in the future

## Alternative: Force Uninstall via Registry

If the batch file doesn't work, force remove:

```cmd
# Remove from Add/Remove Programs
reg delete "HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\{D1FEA2D9-872B-4058-940D-D31FC0BFECC6}" /f

# Remove product registry
reg delete "HKLM\SOFTWARE\ProfessionalSMART" /f

# Remove files manually
rmdir /s /q "C:\Program Files\Professional SMART"
rmdir /s /q "C:\ProgramData\Professional SMART"
```

## Summary

**Problem:** Can't upgrade from broken 1.2.0 installation
**Root Cause:** Old MSI has unfixable bug
**Solution:** Uninstall old version completely, then install fresh
**Future:** All 1.2.1+ versions will upgrade cleanly from each other

## Files Created

- `UNINSTALL_OLD_VERSION.bat` - Automated uninstall script
- This document - Solution explanation

## Next Steps

1. Run `UNINSTALL_OLD_VERSION.bat`
2. Verify old version completely removed
3. Install fresh 1.2.1.0
4. Test that it works
5. Build 1.2.2.0 and test upgrade (optional)
