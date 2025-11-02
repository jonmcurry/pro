# Professional SMART Release Notes - Version 1.5.2

**Release Date**: November 2, 2025
**Build**: 1.5.2.0
**Installer**: ProfessionalSMART.msi (9.1 MB)
**Checksum (MD5)**: a3be40f632cb94c2d4dfc6453f28c9c9
**Release Type**: Patch (Bug Fix)

## What's Fixed

### Native .837p File Extension Support

**Problem**: The file watcher required users to rename `.837p` files to `.edi` before processing.

**Fixed**: File watcher now processes `.837p` files natively, in addition to `.csv` and `.edi` files.

**Impact**: 837p EDI files with their native `.837p` extension are now automatically picked up and processed when placed in:
```
C:\Program Files\Professional SMART\data\input\
```

No more manual file renaming required!

### Technical Details

**Modified File**: `crates/pro-service/src/file_watcher.rs`

**Change**:
```rust
// Before (1.5.1):
if ext_lower == "csv" || ext_lower == "edi" {
    return true;
}

// After (1.5.2):
if ext_lower == "csv" || ext_lower == "edi" || ext_lower == "837p" {
    return true;
}
```

**Supported File Types**:
- `.csv` - Master data files (organizations, regions, facilities, providers)
- `.edi` - 837p EDI claims files (alternative extension)
- `.837p` - 837p EDI claims files (native extension) **NEW**
- All extensions are case-insensitive (`.CSV`, `.EDI`, `.837P` also work)

## Upgrade Instructions

### From 1.5.1 or 1.5.0

Simply install the new MSI. The installer will automatically upgrade:

```powershell
# Install as Administrator
.\ProfessionalSMART.msi
```

### From 1.4.x

Same upgrade process - MSI handles the migration automatically.

## Usage

### Processing .837p Files (Native Extension)

1. **Place .837p files directly in input directory**:
   ```powershell
   Copy-Item "C:\YourFiles\*.837p" "C:\Program Files\Professional SMART\data\input\"
   ```

2. **Service processes automatically**:
   - File watcher detects `.837p` files (no renaming needed!)
   - Parses 837p format
   - Validates claims
   - Inserts into database
   - Moves to processed directory

3. **Monitor processing**:
   ```powershell
   # View service logs
   Get-EventLog -LogName Application -Source "ProfessionalSMART" -Newest 20

   # Check directories
   Get-ChildItem "C:\Program Files\Professional SMART\data\input"
   Get-ChildItem "C:\Program Files\Professional SMART\data\processed"
   Get-ChildItem "C:\Program Files\Professional SMART\data\error"
   ```

### File Extensions Supported

| Extension | Purpose | Auto-Processed | Notes |
|-----------|---------|----------------|-------|
| `.csv` | Master data | Yes | Organizations, facilities, providers |
| `.edi` | 837p claims | Yes | Alternative EDI extension |
| `.837p` | 837p claims | Yes | **NEW** - Native 837p extension |
| `.txt` | Text files | No | Not processed |

**All extensions are case-insensitive**: `.CSV`, `.EDI`, `.837P` work just as well.

## Breaking Changes

None. This is a backward-compatible bug fix.

## What Changed from 1.5.1

Version 1.5.1 added `.edi` support but still required users to rename `.837p` files. This release eliminates that requirement by adding native `.837p` extension support.

**Timeline**:
- **1.5.0**: Only `.csv` files processed
- **1.5.1**: Added `.edi` support (users had to rename `.837p` → `.edi`)
- **1.5.2**: Added native `.837p` support (no renaming required!)

## Known Issues

None specific to this release.

### Master Data Required

EDI claim files require master data (organizations and facilities) to be loaded first. If master data is missing, claims will fail validation and move to the error directory.

**Solution**: Load master data before processing claims:
```powershell
# Use GUI or CLI to load master data first
.\pro-data-loader.exe `
  --org-file "organizations.csv" `
  --region-file "regions.csv" `
  --facility-file "facilities.csv"

# Then process EDI/837p files
Copy-Item "*.837p" "C:\Program Files\Professional SMART\data\input\"
```

## What's Included

Same executables as 1.5.1:
- **pro-service.exe** - Claims processing service (UPDATED)
- **pro-setup.exe** - Configuration wizard
- **pro-data-loader.exe** - CLI data loader
- **pro-data-loader-gui.exe** - GUI data loader
- **pro-upgrade.exe** - Database migration tool

## Testing

### Verify .837p Processing Works

```powershell
# 1. Ensure service is running
Get-Service ProfessionalSMART

# 2. Place test .837p file (no renaming needed!)
Copy-Item "test_data\sample.837p" "C:\Program Files\Professional SMART\data\input\"

# 3. Wait 2-3 seconds for processing

# 4. Check if file moved
Test-Path "C:\Program Files\Professional SMART\data\processed\sample.837p"

# 5. Check logs
Get-EventLog -LogName Application -Source "ProfessionalSMART" -Newest 5 |
    Where-Object { $_.Message -like "*sample.837p*" }
```

### Expected Behavior

**Success Path**:
1. `.837p` file appears in `input\` directory
2. Service logs: "Detected file: sample.837p"
3. Service logs: "Processing new file: sample.837p"
4. Service logs: "Successfully processed file: sample.837p"
5. File moves to `processed\` directory (keeps `.837p` extension)

**Error Path**:
1. `.837p` file appears in `input\` directory
2. Service logs: "Detected file: sample.837p"
3. Service logs: "Failed to process file sample.837p: [error message]"
4. File moves to `error\` directory
5. Companion `sample.error` file created with error details

## Version History

- **1.5.2** (Nov 2, 2025) - Added native .837p file extension support
- **1.5.1** (Nov 2, 2025) - Added .edi file processing support
- **1.5.0** (Nov 2, 2025) - Added CLI data loader for Server Core
- **1.4.3** - Previous release

## Support

### Troubleshooting

If .837p files aren't being processed:

1. **Check service status**:
   ```powershell
   Get-Service ProfessionalSMART
   ```
   Must be "Running"

2. **Check Event Log**:
   ```powershell
   Get-EventLog -LogName Application -Source "ProfessionalSMART" -EntryType Error -Newest 10
   ```

3. **Check master data loaded**:
   ```sql
   -- Connect to PostgreSQL
   SELECT COUNT(*) FROM claims.organization;
   SELECT COUNT(*) FROM claims.facility;
   ```
   Both must return > 0

4. **Check error directory**:
   ```powershell
   Get-ChildItem "C:\Program Files\Professional SMART\data\error" -Filter "*.error" |
       ForEach-Object { Get-Content $_.FullName }
   ```

### Getting Help

- **Diagnostic Script**: `scripts\diagnose-837p-processing.ps1`
- **Documentation**: [docs/SERVER_837P_TROUBLESHOOTING.md](docs/SERVER_837P_TROUBLESHOOTING.md)
- **Event Logs**: `eventvwr.msc` → Windows Logs → Application → ProfessionalSMART
- **Support Email**: support@professional-smart.com

## Installation Paths

- **Binaries**: `C:\Program Files\Professional SMART\bin\`
- **Configuration**: `C:\ProgramData\Professional SMART\config\.env`
- **Input Directory**: `C:\Program Files\Professional SMART\data\input\`
- **Processed Directory**: `C:\Program Files\Professional SMART\data\processed\`
- **Error Directory**: `C:\Program Files\Professional SMART\data\error\`
- **Logs**: Windows Event Viewer (Application log)

## Credits

Developed by the Professional SMART Team.

This patch eliminates the manual file renaming step that was required in v1.5.1, making the system more user-friendly for healthcare providers working with native 837p files.
