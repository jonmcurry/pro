# Installer Build Guide

Quick guide for building the Professional SMART MSI installer with automatic version management.

## Quick Start

```cmd
cd installer
.\build-simple.bat
```

This will auto-increment the build number and create `ProfessionalSMART.msi`.

## Version Control

### Auto-increment Build Number (Default)
```cmd
.\build-simple.bat
```
Increments: `1.1.0.0` → `1.1.0.1` → `1.1.0.2`

### Increment Minor Version (New Features)
```cmd
.\build-simple.bat -Minor
```
Increments: `1.1.0.x` → `1.2.0.0`

### Increment Patch Version (Bug Fixes)
```cmd
.\build-simple.bat -Patch
```
Increments: `1.1.0.x` → `1.1.1.0`

### Increment Major Version (Breaking Changes)
```cmd
.\build-simple.bat -Major
```
Increments: `1.x.x.x` → `2.0.0.0`

### Set Specific Version
```cmd
.\build-simple.bat 1.5.0.0
```

## Smart Build (Git-Aware)

Analyzes commit messages to suggest version increment:

```powershell
# Interactive (prompts for confirmation)
powershell -ExecutionPolicy Bypass .\smart-build.ps1 -Auto

# Automatic (no prompts)
powershell -ExecutionPolicy Bypass .\smart-build.ps1 -Auto -Force
```

## When to Increment What

See [../docs/VERSIONING_GUIDE.md](../docs/VERSIONING_GUIDE.md) for details.

**Quick Reference:**
- **Major** (2.0.0.0): Breaking changes, schema overhaul
- **Minor** (1.2.0.0): New features, new migrations
- **Patch** (1.1.1.0): Bug fixes, security patches
- **Build** (1.1.0.1): Each rebuild (auto)

## Testing Upgrades

To test the upgrade path, you need different version numbers:

```cmd
# Option 1: Bump to next minor version
.\build-simple.bat -Minor

# Option 2: Bump to next patch version
.\build-simple.bat -Patch

# Then install over existing version
msiexec /i ProfessionalSMART.msi /l*v upgrade.log
```

See [../docs/TESTING_UPGRADE.md](../docs/TESTING_UPGRADE.md) for complete testing guide.

## Prerequisites

1. **WiX Toolset 3.x**: https://wixtoolset.org/
2. **Rust**: https://rustup.rs/
3. **PowerShell**: Built into Windows

## Build Process

The build script automatically:
1. Increments version in `version.txt`
2. Updates `Product.wxs` with new version
3. Builds Rust binaries (`cargo build --release`)
4. Compiles WiX installer (`candle` + `light`)
5. Creates `ProfessionalSMART.msi`

## Files

- **build-simple.bat** - Easy-to-use batch wrapper
- **build-msi.ps1** - Main PowerShell build script
- **smart-build.ps1** - Git-aware intelligent builder
- **version.txt** - Current version tracker (auto-managed)
- **Product.wxs** - Main installer definition
- **DatabaseConfigDlg.wxs** - Database config dialog
- **PrerequisiteDlg.wxs** - Prerequisites check dialog
- **CreateDatabase.vbs** - Database setup script
- **UpgradeDatabase.vbs** - Upgrade workflow script
- **DetectInstallation.vbs** - Installation type detection
- **WriteConfig.vbs** - Configuration file creation

## Troubleshooting

### "Change, Modify, Repair" Dialog

**Cause:** Same version already installed.

**Solution:** Increment version:
```cmd
.\build-simple.bat -Minor
```

### Candle/Light Not Found

**Cause:** WiX not in PATH.

**Solution:** Add WiX to PATH or install from https://wixtoolset.org/

### Build Fails

```cmd
# Check Rust build separately
cd ..
cargo build --release

# If that works, rebuild MSI
cd installer
.\build-simple.bat
```

## See Also

- [VERSIONING_GUIDE.md](../docs/VERSIONING_GUIDE.md) - Semantic versioning rules
- [UPGRADE_GUIDE.md](../docs/UPGRADE_GUIDE.md) - User upgrade instructions
- [TESTING_UPGRADE.md](../docs/TESTING_UPGRADE.md) - Testing upgrade paths
