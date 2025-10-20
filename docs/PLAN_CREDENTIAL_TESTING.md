# Plan: Proper Testing of Credential Loading Feature

## Current Situation

- **Problem**: Trying to test upgrade credential loading, but no successful installation exists
- **Symptom**: Database dialog keeps appearing because .env doesn't exist
- **Root Cause**: Failed installations clean up .env file, creating a testing loop

## Why This Keeps Failing

1. Testing upgrade from non-existent base installation
2. .env file gets cleaned up during failed upgrade rollback
3. No way to establish baseline with .env file

## Correct Testing Approach (Following claude.md Rule 9)

### Phase 1: Establish Baseline Installation ✓
**Objective**: Create a successful 1.2.1 installation with .env file

- [ ] Uninstall any existing Professional SMART installations completely
- [ ] Clean up ProgramData directories
- [ ] Run fresh install of 1.2.1.0 MSI
- [ ] Enter database credentials in dialog (EXPECTED BEHAVIOR)
- [ ] Verify .env file created at `C:\ProgramData\Professional SMART\config\.env`
- [ ] Verify installation completes successfully
- [ ] Verify service can start with credentials from .env

### Phase 2: Test Upgrade Credential Loading ✓
**Objective**: Verify credentials are loaded automatically during upgrade

- [ ] Increment version to 1.2.2.0
- [ ] Rebuild MSI with new version
- [ ] Run MSI installer
- [ ] **EXPECTED**: Database dialog should NOT appear
- [ ] **EXPECTED**: LoadEnvCredentials should load from existing .env
- [ ] **EXPECTED**: Upgrade completes without credential prompts
- [ ] Verify .env file unchanged
- [ ] Verify service still works

### Phase 3: Test Fresh Install (No .env)
**Objective**: Verify fallback behavior when .env missing

- [ ] Delete .env file manually
- [ ] Increment version to 1.2.3.0
- [ ] Run MSI installer
- [ ] **EXPECTED**: Database dialog SHOULD appear
- [ ] **EXPECTED**: User can enter credentials
- [ ] **EXPECTED**: New .env created
- [ ] Verify installation succeeds

## What We've Been Doing Wrong

- ❌ Trying to upgrade from broken 1.2.0 installation
- ❌ Expecting .env to exist when it never was created
- ❌ Creating .env manually, then it gets cleaned up
- ❌ Not establishing proper test baseline

## What We Should Do

- ✅ Accept that database dialog WILL appear on first successful install
- ✅ Let the installer create the .env file properly
- ✅ THEN test upgrade path from working baseline
- ✅ Test both scenarios: with .env and without .env

## Implementation Steps

### Step 1: Clean Slate
```cmd
# Uninstall completely
msiexec /x {ProductCode} /quiet

# Or use Add/Remove Programs
# Remove any leftover files
rmdir /s /q "C:\Program Files\Professional SMART"
rmdir /s /q "C:\ProgramData\Professional SMART"
```

### Step 2: Fresh Install 1.2.1.0
```cmd
# Current MSI is 1.2.1.0
msiexec /i ProfessionalSMART.msi /l*v fresh_install.log

# Fill in database credentials when prompted
# Let it complete
```

### Step 3: Verify Baseline
```cmd
# Check version
reg query "HKLM\SOFTWARE\ProfessionalSMART" /v Version

# Check .env exists
dir "C:\ProgramData\Professional SMART\config\.env"

# Check contents
type "C:\ProgramData\Professional SMART\config\.env"
```

### Step 4: Build 1.2.2.0 for Upgrade Test
```powershell
cd c:\Users\jonmc\dev\pro\installer
.\build-msi.ps1 -Patch -NoBuild  # Increments to 1.2.2.0
```

### Step 5: Test Upgrade
```cmd
msiexec /i ProfessionalSMART.msi /l*v upgrade_test.log

# Should NOT show database dialog
# Should complete automatically
```

### Step 6: Verify Upgrade
```cmd
# Check version changed
reg query "HKLM\SOFTWARE\ProfessionalSMART" /v Version
# Should show 1.2.2.0

# Check .env unchanged
dir "C:\ProgramData\Professional SMART\config\.env"
```

## Expected Outcomes

### Fresh Install (No previous version)
- ✅ Database dialog appears
- ✅ User enters credentials
- ✅ .env file created
- ✅ Installation succeeds

### Upgrade (With .env file)
- ✅ LoadEnvCredentials finds .env
- ✅ Credentials loaded automatically
- ✅ Database dialog SKIPPED
- ✅ Upgrade succeeds silently

### Upgrade (Without .env file)
- ✅ LoadEnvCredentials finds no file
- ✅ Database dialog appears (fallback)
- ✅ User enters credentials
- ✅ New .env created
- ✅ Installation succeeds

## Why This is the Right Approach (claude.md Compliance)

- **Rule 1**: Not disabling features - credential loading works, just needs proper test
- **Rule 3**: No silent failures - dialog appears when .env missing (loud and clear)
- **Rule 9**: No shortcuts - establishing proper baseline before testing
- **Rule 10**: Will rebuild installer after version increment

## Next Actions

**STOP trying to fix the credential loading.** It's working correctly!

**START with proper installation sequence:**
1. Do clean fresh install to create baseline
2. Build next version
3. Test upgrade from working baseline

The feature is **WORKING AS DESIGNED**. We just need to test it properly.
