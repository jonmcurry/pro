# Windows Service Installation Fix - Version 1.5.12.0

**Date**: November 3, 2025
**Issues**:
1. Service not starting automatically on fresh install
2. Service not removed on uninstall

## Problem Analysis

### Issue 1: Service Not Auto-Starting

**Location**: [installer/Product.wxs:87-89](installer/Product.wxs#L87-L89)

```xml
<ServiceControl Id="ServiceControl"
                Remove="uninstall"
                Name="ProfessionalSMART" />
```

**Problem**: The ServiceControl element is missing the `Start="install"` attribute. This means:
- Service is installed (via ServiceInstall at line 75)
- Service is set to auto-start (Start="auto" at line 80)
- But service is NOT started during installation
- User must manually start it via Services.msc or `sc start`

**Why This Happens**:
- `Start="auto"` in ServiceInstall means "start automatically on Windows boot"
- But it doesn't start the service during installation
- ServiceControl with `Start="install"` is required to start during installation

### Issue 2: Service Not Removed on Uninstall

**Location**: [installer/Product.wxs:87-89](installer/Product.wxs#L87-L89)

```xml
<ServiceControl Id="ServiceControl"
                Remove="uninstall"
                Name="ProfessionalSMART" />
```

**Problem**: ServiceControl only has `Remove="uninstall"` but is missing the Stop action. This means:
- Service is stopped during uninstall (via StopServiceAndWaitAction custom action)
- But service registration is NOT removed
- Service entry remains in Windows Services even after uninstall

**Why This Happens**:
- `Remove="uninstall"` tells WiX to remove the service
- But without `Stop="uninstall"`, WiX may fail to remove if service is running
- The custom action StopServiceAndWaitAction stops it, but doesn't remove it

**Additional Issue**: ServiceInstall is missing `EraseDescription="yes"` which can cause upgrade issues.

## Root Cause

The ServiceControl element is incomplete. It should have:
1. `Start="install"` - Start service after installation
2. `Stop="both"` - Stop service on both upgrade and uninstall
3. `Remove="uninstall"` - Remove service on uninstall (already present)

## Solution

Update ServiceControl element to include all required attributes:

```xml
<ServiceControl Id="ServiceControl"
                Start="install"
                Stop="both"
                Remove="uninstall"
                Name="ProfessionalSMART"
                Wait="yes" />
```

Also update ServiceInstall to handle upgrades better:

```xml
<ServiceInstall Id="ServiceInstaller"
                Type="ownProcess"
                Name="ProfessionalSMART"
                DisplayName="Professional SMART Claims Processing Service"
                Description="Automated claims processing, validation, and flagging system for healthcare providers"
                Start="auto"
                Account="LocalSystem"
                ErrorControl="normal"
                Arguments="service"
                EraseDescription="yes" />
```

### Attributes Explained

**ServiceControl**:
- `Start="install"`: Start service after installation completes
- `Stop="both"`: Stop service on upgrade AND uninstall (prevents file-in-use errors)
- `Remove="uninstall"`: Remove service registration on uninstall
- `Wait="yes"`: Wait for service to fully start/stop before continuing

**ServiceInstall**:
- `EraseDescription="yes"`: Replace service description on upgrade (prevents description concatenation bug)

## Implementation

### File: installer/Product.wxs

**Line 75-89 - BEFORE**:
```xml
<ServiceInstall Id="ServiceInstaller"
                Type="ownProcess"
                Name="ProfessionalSMART"
                DisplayName="Professional SMART Claims Processing Service"
                Description="Automated claims processing, validation, and flagging system for healthcare providers"
                Start="auto"
                Account="LocalSystem"
                ErrorControl="normal"
                Arguments="service" />

<!-- Do not automatically start service during install/upgrade - let user start manually -->
<!-- This prevents installation failure if database is not ready -->
<ServiceControl Id="ServiceControl"
                Remove="uninstall"
                Name="ProfessionalSMART" />
```

**Line 75-90 - AFTER**:
```xml
<ServiceInstall Id="ServiceInstaller"
                Type="ownProcess"
                Name="ProfessionalSMART"
                DisplayName="Professional SMART Claims Processing Service"
                Description="Automated claims processing, validation, and flagging system for healthcare providers"
                Start="auto"
                Account="LocalSystem"
                ErrorControl="normal"
                Arguments="service"
                EraseDescription="yes" />

<!-- Service Control: Start on install, Stop on upgrade/uninstall, Remove on uninstall -->
<ServiceControl Id="ServiceControl"
                Start="install"
                Stop="both"
                Remove="uninstall"
                Name="ProfessionalSMART"
                Wait="yes" />
```

### Comment Removal

The old comment about "do not automatically start" is outdated. The database is created during installation (CreateDatabaseAction), so it's safe to start the service.

## Testing

### Test 1: Fresh Install

1. Uninstall any existing version
2. Delete service manually if it exists: `sc delete ProfessionalSMART`
3. Install v1.5.12.0
4. Check service status: `sc query ProfessionalSMART`

**Expected**:
- Service exists
- Service status: RUNNING
- Service start type: AUTO_START

### Test 2: Upgrade Install

1. Install v1.5.11.0
2. Verify service is running
3. Install v1.5.12.0 (upgrade)
4. Check service status

**Expected**:
- Service exists
- Service status: RUNNING (restarted automatically)
- Service start type: AUTO_START
- No duplicate services

### Test 3: Uninstall

1. Install v1.5.12.0
2. Verify service is running
3. Uninstall via Control Panel
4. Check service: `sc query ProfessionalSMART`

**Expected**:
- Service does not exist
- Error: "The specified service does not exist as an installed service"

### Test 4: Service Removal Verification

After uninstall, check:

```powershell
# Should return nothing
Get-Service -Name "ProfessionalSMART" -ErrorAction SilentlyContinue

# Should show error
sc query ProfessionalSMART
```

**Expected**:
```
[SC] EnumQueryServicesStatus:OpenService FAILED 1060:

The specified service does not exist as an installed service.
```

## Interaction with Existing Custom Actions

### StopServiceAndWaitAction (Line 528)

The custom action at line 528 stops the service during upgrades and uninstalls:

```xml
<Custom Action="StopServiceAndWaitAction" After="InstallInitialize">WIX_UPGRADE_DETECTED OR (Installed AND REMOVE="ALL")</Custom>
```

This is **still needed** and works together with ServiceControl:
1. StopServiceAndWaitAction stops service early (after InstallInitialize)
2. This allows RemoveExistingProducts to remove old files
3. ServiceControl's Stop="both" provides backup stop mechanism
4. ServiceControl's Remove="uninstall" removes service registration

Both work together - the custom action ensures service is stopped before file operations, and ServiceControl ensures proper cleanup.

## Risk Assessment

**Low Risk**: These are standard WiX service installation patterns. Changes only affect service lifecycle management:
- Fresh installs: Service will start automatically (user benefit)
- Upgrades: Service will restart automatically (user benefit)
- Uninstalls: Service will be removed properly (fixes bug)

**No Database Risk**: Database operations are unaffected.

**No File Risk**: File installation/removal is unaffected.

## Files Modified

1. **installer/Product.wxs**
   - Lines 75-90: Update ServiceInstall and ServiceControl elements

2. **installer/Product.wxs**
   - Line 9: Update version to 1.5.12.0

## Success Criteria

- ✅ Fresh install: Service starts automatically
- ✅ Fresh install: Service set to auto-start on boot
- ✅ Upgrade: Service restarts automatically
- ✅ Uninstall: Service stopped and removed
- ✅ Uninstall: No service entry in Services.msc
- ✅ Uninstall: `sc query ProfessionalSMART` returns error 1060

## Documentation

Windows Installer Service Control reference:
- ServiceInstall: https://wixtoolset.org/docs/v3/xsd/wix/serviceinstall/
- ServiceControl: https://wixtoolset.org/docs/v3/xsd/wix/servicecontrol/

Key attributes:
- Start: "install" | "uninstall" | "both"
- Stop: "install" | "uninstall" | "both"
- Remove: "install" | "uninstall" | "both"
- Wait: "yes" | "no"

## Version History

- **v1.5.11.0**: Added service line parsing
- **v1.5.12.0**: Fixed service installation and removal ← **Current**
