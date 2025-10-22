# Uninstaller Cleanup Fix Plan

## Problem
After uninstalling Professional SMART through Windows Settings, the `bin` and `data` folders remain in `C:\Program Files\Professional SMART\`.

## Root Cause Analysis
The custom action `RemoveInstallFolderAction` that was added has a condition `REMOVE~="ALL"` which is not triggering properly during uninstall.

### Evidence
From install log analysis:
```
MSI (s) (E8:04) [15:17:35:463]: Skipping action: SetRemoveInstallFolderData (condition is false)
MSI (s) (E8:04) [15:17:35:463]: Skipping action: RemoveInstallFolderAction (condition is false)
```

## Investigation Steps
- [ ] Check what REMOVE property value is during uninstall
- [ ] Verify the RemoveInstallFolder.vbs script is correctly embedded in MSI
- [ ] Test different condition syntaxes for custom action
- [ ] Verify custom action sequencing is correct

## Solution Approach

### Option 1: Fix Custom Action Condition
Change from `REMOVE~="ALL"` to a more reliable condition like:
- `(REMOVE="ALL") AND (NOT UPGRADINGPRODUCTCODE)`
- `($ComponentName=3)`

### Option 2: Use RemoveFile Element
Instead of VBScript, use WiX's built-in `RemoveFile` element with proper wildcards.

### Option 3: Enhanced VBScript with Logging
Add diagnostic logging to VBScript to understand why it's not executing.

## Implementation Checklist
- [x] Add diagnostic logging to understand REMOVE property value
- [x] Test RemoveFile element approach for bin and data folders
- [x] Removed VBScript approach - was unreliable
- [x] Used WiX RemoveFile and RemoveFolder elements instead
- [x] Added components for all runtime directories
- [x] Rebuild MSI after fix
- [x] Document final solution

## Final Solution Implemented

### Approach: Native WiX RemoveFile/RemoveFolder Elements

Instead of using unreliable VBScript custom actions, used WiX's built-in elements:

1. **Added RemoveFile/RemoveFolder to data subdirectories:**
   - InputDir component with `RemoveFile Id="RemoveInputFiles" Name="*.*"`
   - ProcessedDir component with `RemoveFile Id="RemoveProcessedFiles" Name="*.*"`
   - ErrorDir component with `RemoveFile Id="RemoveErrorFiles" Name="*.*"`
   - DataDir component with `RemoveFolder Id="RemoveDataFolder"`

2. **Added RemoveFolder to logs directory:**
   - LogsDir component with `RemoveFile Id="RemoveLogFiles" Name="*.*"`

3. **Added RemoveFolder to bin directory:**
   - BinDir component with `RemoveFolder Id="RemoveBinFolder"`

4. **Added RemoveFolder to root installation folder:**
   - InstallFolderComponent with `RemoveFolder Id="RemoveINSTALLFOLDER"`

5. **Added registry keys as KeyPath for each component:**
   - Required for components without installed files
   - Pattern: `HKCU\Software\ProfessionalSMART\Folders\{FolderName}`

6. **Removed broken VBScript custom action:**
   - Deleted RemoveInstallFolder.vbs
   - Removed RemoveInstallFolderAction custom action
   - Removed custom action scheduling from InstallExecuteSequence

## Testing Plan
1. Install MSI
2. Process a CSV file (creates files in bin/data folders)
3. Uninstall through Windows Settings
4. Verify `C:\Program Files\Professional SMART` is completely removed
5. Check for orphaned registry keys
6. Verify clean reinstall works

## Success Criteria
- Complete removal of `C:\Program Files\Professional SMART` folder
- No registry keys remain
- Clean reinstall possible without conflicts
