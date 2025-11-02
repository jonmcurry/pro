# Professional SMART Release Notes - Version 1.5.1

**Release Date**: November 2, 2025
**Build**: 1.5.1.0
**Installer**: ProfessionalSMART.msi (9.1 MB)
**Checksum (MD5)**: ff96915c58465ed0049a864e2f8103b7
**Release Type**: Patch (Bug Fix)

## What's Fixed

### EDI File Processing Support

**Problem**: The file watcher only processed `.csv` files, ignoring `.edi` files (837p claims) placed in the input directory.

**Fixed**: File watcher now processes both `.csv` and `.edi` files automatically.

**Impact**: 837p EDI files (`.edi` extension) are now automatically picked up and processed when placed in:
```
C:\Program Files\Professional SMART\data\input\
```

### Technical Details

**Modified File**: `crates/pro-service/src/file_watcher.rs`

**Change**:
```rust
// Before (1.5.0):
if ext_lower == "csv" {
    return true;
}

// After (1.5.1):
if ext_lower == "csv" || ext_lower == "edi" {
    return true;
}
```

**Supported File Types**:
- `.csv` - Master data files (organizations, regions, facilities, providers)
- `.edi` - 837p EDI claims files
- `.CSV` - Case insensitive
- `.EDI` - Case insensitive

## Upgrade Instructions

### From 1.5.0
Simply install the new MSI. The installer will automatically upgrade:

```powershell
# Install as Administrator
.\ProfessionalSMART.msi
```

### From 1.4.x
Same upgrade process - MSI handles the migration automatically.

## Usage

### Processing EDI Files

1. **Place EDI files in input directory**:
   ```powershell
   Copy-Item "C:\YourFiles\*.edi" "C:\Program Files\Professional SMART\data\input\"
   ```

2. **Service processes automatically**:
   - File watcher detects `.edi` files
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

| Extension | Purpose | Auto-Processed |
|-----------|---------|----------------|
| `.csv` | Master data | Yes |
| `.edi` | 837p claims | Yes (NEW) |
| `.837p` | 837p claims | No - rename to .edi |
| `.txt` | Text files | No |

**Note**: If you have `.837p` files, rename them to `.edi`:
```powershell
Get-ChildItem "C:\Program Files\Professional SMART\data\input" -Filter "*.837p" |
    Rename-Item -NewName { $_.Name -replace '\.837p$','.edi' }
```

## Breaking Changes

None. This is a backward-compatible bug fix.

## Known Issues

### File Extension Limitation
Files with `.837p` extension are NOT automatically processed. They must be renamed to `.edi`.

**Workaround**:
```powershell
# Rename .837p to .edi
cd "C:\Program Files\Professional SMART\data\input"
Get-ChildItem -Filter "*.837p" | Rename-Item -NewName { $_.Name -replace '\.837p$','.edi' }
```

### Master Data Required
EDI claim files require master data (organizations and facilities) to be loaded first. If master data is missing, claims will fail validation and move to the error directory.

**Solution**: Load master data before processing claims:
```powershell
# Use GUI or CLI to load master data first
.\pro-data-loader.exe `
  --org-file "organizations.csv" `
  --region-file "regions.csv" `
  --facility-file "facilities.csv"

# Then process EDI files
Copy-Item "*.edi" "C:\Program Files\Professional SMART\data\input\"
```

## What's Included

Same executables as 1.5.0:
- **pro-service.exe** - Claims processing service (UPDATED)
- **pro-setup.exe** - Configuration wizard
- **pro-data-loader.exe** - CLI data loader
- **pro-data-loader-gui.exe** - GUI data loader
- **pro-upgrade.exe** - Database migration tool

## Testing

### Verify EDI Processing Works

```powershell
# 1. Ensure service is running
Get-Service ProfessionalSMART

# 2. Place test EDI file
Copy-Item "test_data\sample.edi" "C:\Program Files\Professional SMART\data\input\"

# 3. Wait 2-3 seconds for processing

# 4. Check if file moved
Test-Path "C:\Program Files\Professional SMART\data\processed\sample.edi"

# 5. Check logs
Get-EventLog -LogName Application -Source "ProfessionalSMART" -Newest 5 |
    Where-Object { $_.Message -like "*sample.edi*" }
```

### Expected Behavior

**Success Path**:
1. EDI file appears in `input\` directory
2. Service logs: "Detected file: sample.edi"
3. Service logs: "Processing new file: sample.edi"
4. Service logs: "Successfully processed file: sample.edi"
5. File moves to `processed\` directory

**Error Path**:
1. EDI file appears in `input\` directory
2. Service logs: "Detected file: sample.edi"
3. Service logs: "Failed to process file sample.edi: [error message]"
4. File moves to `error\` directory
5. Companion `sample.error` file created with error details

## Version History

- **1.5.1** (Nov 2, 2025) - Added .edi file processing support
- **1.5.0** (Nov 2, 2025) - Added CLI data loader for Server Core
- **1.4.3** - Previous release

## Support

### Troubleshooting EDI Processing

If EDI files aren't being processed:

1. **Check file extension**:
   ```powershell
   Get-ChildItem "C:\Program Files\Professional SMART\data\input" | Select-Object Name, Extension
   ```
   Must be `.edi` (not `.837p`)

2. **Check service status**:
   ```powershell
   Get-Service ProfessionalSMART
   ```
   Must be "Running"

3. **Check Event Log**:
   ```powershell
   Get-EventLog -LogName Application -Source "ProfessionalSMART" -EntryType Error -Newest 10
   ```

4. **Check master data loaded**:
   ```sql
   -- Connect to PostgreSQL
   SELECT COUNT(*) FROM claims.organization;
   SELECT COUNT(*) FROM claims.facility;
   ```
   Both must return > 0

5. **Check error directory**:
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

This patch specifically addresses the file processing limitation discovered during Windows Server 2019 deployment testing.
