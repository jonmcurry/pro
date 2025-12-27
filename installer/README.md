# Professional SMART Windows Installer

This directory contains the WiX Toolset configuration for building the Professional SMART Windows installer (.msi package).

**Current Version:** 2.12.33.0
**Last Updated:** 2025-12-27

## Prerequisites

1. **WiX Toolset 3.11 or later**
   - Download from: https://wixtoolset.org/
   - Add to PATH during installation

2. **Visual Studio Build Tools 2019+**
   - Required for WiX compilation

3. **Rust Release Binaries**
   - Run `cargo build --release` from project root
   - Creates binaries in `target/release/`

## Building the Installer

### Using PowerShell (Recommended)

```powershell
cd installer
.\build-msi.ps1 -Version "2.12.33.0"
```

Options:
- `-Version "X.Y.Z.W"` - Specify version number
- `-NoBuild` - Skip Rust compilation (use existing binaries)

### Windows Command Prompt

```cmd
cd installer
build.bat
```

This will:
1. Verify WiX Toolset is installed
2. Check for release binaries
3. Compile WiX source files
4. Create `ProfessionalSMART.msi`

### Manual Build

```cmd
REM Compile WiX source
candle.exe -dSolutionDir="..\\" -out Product.wixobj Product.wxs

REM Link to create MSI
light.exe -ext WixUIExtension -out ProfessionalSMART.msi Product.wixobj
```

## Installer Features

### What Gets Installed

**Binaries** (`C:\Program Files\Professional SMART\bin\`):
- `pro-service.exe` - Windows service executable
- `pro-setup.exe` - Configuration wizard

**Documentation** (`C:\Program Files\Professional SMART\docs\`):
- Installation guide
- Configuration reference
- Database setup instructions
- Troubleshooting guide
- Performance tuning guide
- License and README

**Configuration** (`C:\ProgramData\Professional SMART\config\`):
- `.env` file (created by wizard)
- Configuration templates

**Data Directories** (`C:\Program Files\Professional SMART\data\`):
- `input/` - Claims files to process
- `processed/` - Successfully processed files
- `error/` - Failed files

**Logs** (`C:\ProgramData\Professional SMART\logs\`):
- Service logs
- Application logs

**Start Menu**:
- Configuration Wizard shortcut
- Documentation folder shortcut
- Uninstall shortcut

### Database Creation

The installer creates two PostgreSQL databases:

1. **Project Database** (user-specified name, e.g., `professional_smart_clientA`)
   - Main claims processing database
   - All 69 migrations applied via `000_baseline_v2.12.sql`
   - Schemas: `claims`, `staging`, `analytics`, `archive`, `ml`, `smartproaudit`

2. **SmartProAudit** (master database, lowercase)
   - Centralized project registry
   - Created from `migrations/smartproaudit/000_baseline.sql`
   - Schemas: `projects`, `fields`, `security`

**Database Creation Script:** `CreateDatabase.vbs`
- Validates PostgreSQL credentials before proceeding
- Creates databases with proper SQL identifier quoting (preserves case)
- Applies baseline migrations using `pro-upgrade.exe`
- Registers the project in SmartProAudit

### Windows Service

The installer automatically:
1. Installs Windows service: `ProfessionalSMART`
2. Sets startup type to Automatic
3. Configures recovery options (restart on failure)
4. Adds PostgreSQL as service dependency

Service can be managed via:
```cmd
professional-smart start
professional-smart stop
professional-smart install
professional-smart uninstall
```

Or through Windows Services (services.msc).

## Installation

### Interactive Installation

Double-click `ProfessionalSMART.msi` and follow the wizard.

### Silent Installation

```cmd
msiexec /i ProfessionalSMART.msi /quiet /qn
```

### With Logging

```cmd
msiexec /i ProfessionalSMART.msi /l*v install.log
```

### Installation Sequence

1. Welcome screen
2. License agreement
3. Feature selection
4. Installation directory
5. Copy files
6. Install Windows service
7. Run configuration wizard (optional)
8. Complete

## Uninstallation

### Interactive

- Programs and Features (Add/Remove Programs)
- Start Menu → Professional SMART → Uninstall

### Silent

```cmd
msiexec /x ProfessionalSMART.msi /quiet /qn
```

### What Gets Removed

- All installed files
- Windows service
- Start Menu shortcuts
- Registry entries

### What Remains (Optional)

- Configuration files in `C:\ProgramData\Professional SMART\`
- Database data
- Log files

User can manually delete these if desired.

## Customization

### Changing Installation Path

Edit `Product.wxs` and modify:
```xml
<Directory Id="INSTALLFOLDER" Name="Professional SMART">
```

### Adding Components

1. Add component to `Product.wxs`
2. Add ComponentRef to appropriate Feature
3. Regenerate GUIDs (use `uuidgen` or online tool)

### Changing Service Properties

Edit in `Product.wxs`:
```xml
<ServiceInstall Id="ServiceInstaller"
                Type="ownProcess"
                Name="ProfessionalSMART"
                DisplayName="Professional SMART Claims Processing Service"
                Description="..."
                Start="auto"
                Account="LocalSystem"
                ErrorControl="normal"
                Arguments="service">
```

### Custom Actions

To add custom actions (e.g., database setup):
```xml
<CustomAction Id="MyAction"
              FileKey="MyExecutable"
              ExeCommand="arguments"
              Execute="deferred"
              Return="check" />

<InstallExecuteSequence>
  <Custom Action="MyAction" After="InstallFiles">NOT Installed</Custom>
</InstallExecuteSequence>
```

## Troubleshooting

### Error: WiX Toolset not found

- Install WiX Toolset from https://wixtoolset.org/
- Add to PATH: `C:\Program Files (x86)\WiX Toolset v3.11\bin`
- Restart Command Prompt

### Error: Release binaries not found

```cmd
cd ..
cargo build --release --package pro-service
cargo build --release --package pro-setup
```

### Error: Permission denied during installation

- Run installer as Administrator
- Right-click → "Run as administrator"

### Error: Service failed to start

- Check PostgreSQL is installed and running
- Verify database connection in `.env`
- Check service logs: `C:\ProgramData\Professional SMART\logs\`
- View Windows Event Log (Application)

### Error: ICE validation errors

These are warnings from WiX. Most can be safely ignored. To suppress:
```cmd
light.exe -sval -out ProfessionalSMART.msi Product.wixobj
```

## File Structure

```
installer/
  Product.wxs           # Main WiX source file
  License.rtf           # License agreement text
  build.bat             # Build automation script
  README.md             # This file
  icon.ico              # Application icon (to be created)
  Product.wixobj        # Compiled object (generated)
  ProfessionalSMART.msi # Final installer (generated)
```

## Advanced Topics

### Code Signing

To sign the installer (recommended for production):

1. Obtain code signing certificate
2. Sign MSI:
   ```cmd
   signtool sign /f certificate.pfx /p password /t http://timestamp.digicert.com ProfessionalSMART.msi
   ```

### Including PostgreSQL

To bundle PostgreSQL installer:

1. Download PostgreSQL installer
2. Add as component in Product.wxs
3. Create custom action to run installer
4. Handle silent installation parameters

### Multiple Languages

To support multiple languages:

1. Create .wxl files for each language
2. Use loc elements in Product.wxs
3. Build separate MSI for each language

### Upgrade Scenarios

Current configuration supports:
- Major upgrades (automatically removes old version)
- Prevents downgrades

To customize:
```xml
<MajorUpgrade DowngradeErrorMessage="..." />
```

## Testing Checklist

- [ ] Clean Windows 10 installation
- [ ] Clean Windows 11 installation
- [ ] System with PostgreSQL already installed
- [ ] System with insufficient permissions
- [ ] Installation to non-default directory
- [ ] Silent installation
- [ ] Uninstallation (clean removal)
- [ ] Upgrade from previous version
- [ ] Service starts automatically after reboot
- [ ] Configuration wizard works correctly
- [ ] All shortcuts created
- [ ] All documentation accessible

## References

- WiX Toolset Documentation: https://wixtoolset.org/documentation/
- Windows Installer Reference: https://docs.microsoft.com/en-us/windows/win32/msi/
- Service Installation: https://wixtoolset.org/documentation/manual/v3/xsd/wix/serviceinstall.html
- Custom Actions: https://wixtoolset.org/documentation/manual/v3/xsd/wix/customaction.html
