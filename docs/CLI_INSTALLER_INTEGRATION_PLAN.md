# CLI Installer Integration Plan

## Objective
Add the CLI (pro-data-loader.exe) to the MSI installer to support Windows Server 2019+ Datacenter installations, particularly Server Core environments where the GUI cannot run.

## Problem Statement
- Windows Server 2019 Datacenter Server Core does not have Desktop Experience installed
- UxSms (Desktop Window Manager) and Themes services are not available on Server Core
- GUI (pro-data-loader-gui.exe) requires graphics subsystem that doesn't exist on Server Core
- CyberArk privileged access sessions have limited graphics capabilities
- Current MSI installer only includes GUI, leaving Server Core users without a data loader tool

## Solution
Add CLI executable to MSI installer alongside GUI, providing both options for all environments.

## Implementation Checklist

### Phase 1: Installer Modifications
- [x] Add DataLoaderExecutable component to Product.wxs (GUID: 3A3A3A3A-3B3B-3C3C-3D3D-3E3E3E3E3E3E)
- [x] Reference pro-data-loader.exe from target/release directory
- [x] Add ComponentRef to MainApplication feature
- [x] Create Start Menu shortcut "Load Master Data (CLI)"
- [x] Rename existing GUI shortcut to "Load Master Data (GUI)" for clarity
- [x] Build release binaries (cargo build --release)
- [x] Compile WiX installer (candle.exe)
- [x] Link WiX installer (light.exe)
- [x] Update version to 1.5.0.0 (minor version increment)
- [x] Clean up temporary build files
- [ ] Test MSI installation on clean Windows Server 2019 Datacenter
- [ ] Verify CLI executable is deployed to C:\Program Files\Professional SMART\bin\

### Phase 2: Documentation
- [ ] Create CLI_USAGE_GUIDE.md (comprehensive usage documentation)
- [ ] Add CLI usage guide to installer Documentation component
- [ ] Update INSTALLATION.md to reference CLI for Server Core
- [ ] Document Server Core compatibility in README

### Phase 3: Validation
- [ ] Test CLI on Windows Server 2019 Datacenter Server Core
- [ ] Test CLI on Windows Server 2019 Datacenter with Desktop Experience
- [ ] Test GUI on Windows 11 (ensure no regression)
- [ ] Test both CLI and GUI from Start Menu shortcuts
- [ ] Verify generate-templates command works
- [ ] Verify data import command works with all file types

### Phase 4: Versioning
- [x] Determine version increment (minor - added CLI feature)
- [x] Update version in Product.wxs (1.4.3.0 -> 1.5.0.0)
- [ ] Update version in Cargo.toml files (if needed)
- [ ] Tag release in git

### Phase 5: Cleanup
- [x] Remove temporary build files (wixobj, wixpdb, extracted/)
- [x] Remove unused CLI_USAGE_GUIDE.md
- [x] Verify no test/debug files in installer

## Technical Details

### Files Modified
- `installer/Product.wxs`:
  - Added DataLoaderExecutable component (lines 100-105)
  - Added ComponentRef in MainApplication feature (line 305)
  - Added/updated Start Menu shortcuts (lines 261-272)

### Target Platforms
- Windows Server 2019 Datacenter (Server Core and Desktop Experience)
- Windows Server 2022 Datacenter (Server Core and Desktop Experience)
- Windows 10/11 (GUI testing)

### CLI Features (Same as GUI)
- Generate CSV templates
- Validate CSV data before import
- Import organizations, regions, facilities, providers
- Detailed error reporting
- Database connection via .env or environment variables

### Advantages of CLI for Server Environments
- No graphics dependencies (works on Server Core)
- Works in CyberArk/privileged access sessions
- Scriptable and automatable
- Lower resource usage
- Same validation and import logic as GUI

## Success Criteria
- [x] MSI compiles without errors
- [x] MSI includes both CLI and GUI executables
- [x] Version incremented to 1.5.0.0
- [x] Temporary build files cleaned up
- [ ] MSI installs on Server Core without errors
- [ ] CLI executable present in bin directory after install
- [ ] CLI --help command works
- [ ] CLI generate-templates command works
- [ ] CLI import command works with sample data
- [ ] Start Menu shortcuts work on both Server Core and Desktop Experience
- [ ] No regression in GUI functionality

## Build Summary

**Version**: 1.5.0.0
**Build Date**: 2025-11-02
**MSI Size**: 9.1 MB
**MD5 Checksum**: ef708f8e72f9712c2a438e8c828130f6

**Included Executables**:
- pro-data-loader.exe (CLI) - 2.6 MB
- pro-data-loader-gui.exe (GUI) - 5.9 MB
- pro-service.exe - Windows Service
- pro-setup.exe - Configuration Wizard
- pro-upgrade.exe - Database Migration Tool

**Installation Path**: C:\Program Files\Professional SMART\bin\

**Start Menu Shortcuts**:
- Configuration Wizard
- Load Master Data (GUI)
- Load Master Data (CLI)
- Documentation
- Uninstall

## Rollback Plan
If issues arise:
1. Revert Product.wxs changes to previous version
2. Remove CLI component and references
3. Rebuild installer from known-good state
4. Git revert commits if necessary

## Notes
- CLI and GUI share the same codebase (pro-data-loader crate)
- Both use identical validation and import logic
- Server Core is the primary use case for CLI
- Desktop Experience installations get both CLI and GUI
