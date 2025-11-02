# Professional SMART Release Notes - Version 1.5.0

**Release Date**: November 2, 2025
**Build**: 1.5.0.0
**Installer**: ProfessionalSMART.msi (9.1 MB)
**Checksum (MD5)**: ef708f8e72f9712c2a438e8c828130f6

## What's New

### CLI Data Loader Added

This release adds the command-line interface (CLI) version of the data loader to support Windows Server 2019+ Datacenter installations, particularly Server Core environments.

**New Features**:
- `pro-data-loader.exe` now included in MSI installer
- Start Menu shortcut: "Load Master Data (CLI)"
- Full parity with GUI version (same validation and import logic)
- Works on Windows Server Core (no Desktop Experience required)
- Compatible with CyberArk and other privileged access sessions

### Why This Matters

**Windows Server Core Support**:
- Server Core installations don't have UxSms (Desktop Window Manager) or Themes services
- GUI applications cannot run without graphics subsystem
- CLI provides identical functionality without graphics dependencies

**Enterprise Access Scenarios**:
- CyberArk privileged sessions have limited OpenGL/DirectX support
- Remote Desktop (RDP) sessions may have graphics limitations
- CLI works reliably in all remote access scenarios

**Automation Ready**:
- CLI can be scripted with PowerShell
- Batch processing of multiple CSV files
- Integration with deployment pipelines
- Lower resource usage for server environments

## Installation

### Standard Installation
Run `ProfessionalSMART.msi` as Administrator. Both CLI and GUI will be installed to:
```
C:\Program Files\Professional SMART\bin\
```

### Server Core Installation
The MSI installer works on Server Core. Only the CLI will function:
```powershell
# Install MSI
msiexec /i ProfessionalSMART.msi /qn

# Verify CLI works
cd "C:\Program Files\Professional SMART\bin"
.\pro-data-loader.exe --help
```

## CLI Usage

### Generate Templates
```powershell
cd "C:\Program Files\Professional SMART\bin"

# Generate CSV templates in current directory
.\pro-data-loader.exe generate-templates

# Or specify output directory
.\pro-data-loader.exe generate-templates --output-dir "C:\Templates"
```

### Import Data
```powershell
# Import all master data
.\pro-data-loader.exe `
  --org-file "organizations.csv" `
  --region-file "regions.csv" `
  --facility-file "facilities.csv" `
  --provider-file "providers.csv"

# Import specific files only
.\pro-data-loader.exe --org-file "organizations.csv"
```

### Get Help
```powershell
.\pro-data-loader.exe --help
.\pro-data-loader.exe generate-templates --help
```

## Start Menu Shortcuts

After installation, the following shortcuts are available in **Start Menu → Professional SMART**:

- **Configuration Wizard** - Configure database connection and settings
- **Load Master Data (GUI)** - Graphical interface for data import
- **Load Master Data (CLI)** - Opens command prompt ready for CLI usage
- **Documentation** - Installation and configuration guides
- **Uninstall** - Remove Professional SMART

## Compatibility

### Supported Platforms
- Windows Server 2019 Datacenter (Server Core and Desktop Experience)
- Windows Server 2022 Datacenter (Server Core and Desktop Experience)
- Windows 10 Pro/Enterprise (64-bit)
- Windows 11 Pro/Enterprise (64-bit)

### CLI vs GUI Availability

| Platform | CLI | GUI |
|----------|-----|-----|
| Server Core | Yes | No (missing Desktop Experience) |
| Desktop Experience | Yes | Yes |
| Windows 10/11 | Yes | Yes |
| CyberArk Session | Yes | Limited |

## What's Included

### Executables
- **pro-service.exe** - Claims processing Windows Service
- **pro-setup.exe** - Configuration wizard
- **pro-data-loader.exe** - CLI data loader (NEW)
- **pro-data-loader-gui.exe** - GUI data loader
- **pro-upgrade.exe** - Database migration tool

### Documentation
- INSTALLATION.md - Installation guide
- CONFIGURATION.md - Configuration reference
- DATABASE_SETUP.md - PostgreSQL setup guide
- PERFORMANCE_TUNING.md - Performance optimization
- LICENSE - Software license

## Upgrade Instructions

### From 1.4.x
The MSI will automatically upgrade from version 1.4.x:

1. Run `ProfessionalSMART.msi` as Administrator
2. Follow the upgrade wizard
3. Database will be automatically migrated
4. Service will be updated
5. CLI will be added to bin directory

**Note**: Configuration from `.env` file is preserved during upgrade.

### Database Migration
The installer automatically runs database migrations. A backup is created at:
```
C:\ProgramData\Professional SMART\backups\
```

## Server Core Quick Start

For Windows Server 2019/2022 Datacenter Server Core:

```powershell
# 1. Install MSI (silent)
msiexec /i ProfessionalSMART.msi /qn

# 2. Navigate to bin directory
cd "C:\Program Files\Professional SMART\bin"

# 3. Run configuration wizard
.\pro-setup.exe

# 4. Generate CSV templates
.\pro-data-loader.exe generate-templates --output-dir "C:\Data"

# 5. Edit templates with your data
# (Use notepad or transfer from another machine)

# 6. Import data
.\pro-data-loader.exe `
  --org-file "C:\Data\organizations.csv" `
  --region-file "C:\Data\regions.csv" `
  --facility-file "C:\Data\facilities.csv" `
  --provider-file "C:\Data\providers.csv"

# 7. Start service
Start-Service ProfessionalSMART
```

## Known Issues

### GUI on Server Core
The GUI application (`pro-data-loader-gui.exe`) will not run on Windows Server Core installations due to missing graphics subsystem. Use the CLI version instead.

**Error Symptoms**:
- Process exits immediately with code 1
- No error message displayed
- UxSms service not found

**Solution**: Use `pro-data-loader.exe` (CLI) which has identical functionality.

### CyberArk Graphics Limitations
When accessing via CyberArk Privileged Session Manager, the GUI may not display correctly due to OpenGL restrictions. Use the CLI version for reliable operation in privileged access scenarios.

## Breaking Changes

None. This is a feature addition release with full backward compatibility.

## Bug Fixes

None. This release focuses on Server Core support.

## Technical Details

### Version Information
- **Product Version**: 1.5.0.0
- **Previous Version**: 1.4.3.0
- **Version Type**: Minor (feature addition)
- **Upgrade Code**: 9A9E9B9C-9E9F-9A9B-9C9D-9E9F9A9B9C9E (unchanged)

### Build Information
- **Compiler**: WiX Toolset 3.14
- **Rust Version**: Latest stable
- **Build Date**: November 2, 2025
- **Build Type**: Release (optimized)

### File Sizes
- **MSI Installer**: 9.1 MB
- **CLI Executable**: 2.6 MB
- **GUI Executable**: 5.9 MB

## Support

### Getting Help
- **CLI Help**: Run `pro-data-loader.exe --help`
- **Documentation**: Check `C:\Program Files\Professional SMART\docs\`
- **Support Email**: support@professional-smart.com

### Reporting Issues
If you encounter issues:
1. Check Event Viewer (Windows Logs → Application)
2. Review installation logs in `%TEMP%`
3. Verify database connectivity
4. Contact support with system details

## Next Steps

After installation:
1. Run Configuration Wizard to set up database connection
2. Use CLI or GUI to generate CSV templates
3. Fill templates with your organization data
4. Import data using CLI or GUI
5. Start the Professional SMART service
6. Begin processing claims

## Credits

Developed by the Professional SMART Team.

This release specifically addresses feedback from Windows Server Core deployments and enterprise environments using privileged access management solutions.
