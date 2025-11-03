# Release Notes - v1.5.15.2

**Release Date**: November 3, 2025
**Release Type**: Patch (Critical Bug Fix)
**Git Commit**: 918d040

## Summary

This release fixes a **critical installer bug** where the `.env` configuration file was generated with **quotes around Windows paths**, causing the dotenvy library to fail parsing and preventing the file watcher from detecting existing files.

## Critical Issue Fixed

### Installer Generated Invalid .env File Format

**Problem**: The installer's `WriteConfig.vbs` script was adding quotes around Windows paths:

```ini
# WRONG - Causes parse errors
INPUT_DIR="C:\Program Files\Professional SMART\data\input"
PROCESSED_DIR="C:\Program Files\Professional SMART\data\processed"
ERROR_DIR="C:\Program Files\Professional SMART\data\error"
```

**Impact**:
1. The `dotenvy` library threw parse errors when loading the `.env` file
2. Service logged: `Failed to load .env file... error at line index: 4 (error kind: LineParse)`
3. File watcher couldn't find the input directory → "No existing files found to process"
4. **Only files created AFTER service start** were detected and processed
5. **Existing files in the input directory were completely ignored**

**Root Cause**: VBScript string concatenation in `WriteConfig.vbs` was using `""""` to add literal quotes:
```vbscript
configFile.WriteLine "INPUT_DIR=""" & installFolder & "data\input" & """"
```

**Solution**: Removed quotes - the dotenvy library doesn't require or support quotes for paths with spaces:
```ini
# CORRECT - Works properly
INPUT_DIR=C:\Program Files\Professional SMART\data\input
PROCESSED_DIR=C:\Program Files\Professional SMART\data\processed
ERROR_DIR=C:\Program Files\Professional SMART\data\error
```

## Files Changed

### installer/WriteConfig.vbs
**Lines 133-135**: Removed `""""` quote escaping from path variables

**Before (v1.5.15.1)**:
```vbscript
configFile.WriteLine "INPUT_DIR=""" & installFolder & "data\input" & """"
configFile.WriteLine "PROCESSED_DIR=""" & installFolder & "data\processed" & """"
configFile.WriteLine "ERROR_DIR=""" & installFolder & "data\error" & """"
```

**After (v1.5.15.2)**:
```vbscript
configFile.WriteLine "INPUT_DIR=" & installFolder & "data\input"
configFile.WriteLine "PROCESSED_DIR=" & installFolder & "data\processed"
configFile.WriteLine "ERROR_DIR=" & installFolder & "data\error"
```

### installer/Product.wxs
**Line 9**: Version updated from 1.5.15.1 → 1.5.15.2

## Upgrade Instructions

### For Fresh Installations

Simply install v1.5.15.2 - the `.env` file will be generated correctly without quotes.

```powershell
cd C:\Users\jonmc\dev\pro\installer
msiexec /i ProfessionalSMART.msi /l*v install_v1.5.15.2.log
```

### For Existing Installations (Manual Fix Required)

**Important**: The installer preserves existing `.env` files during upgrades, so you must **manually fix** the `.env` file if you're upgrading from v1.5.15.0 or v1.5.15.1.

**Option 1: Edit .env File (Recommended)**

1. Stop the service:
   ```powershell
   Stop-Service ProfessionalSMART
   ```

2. Edit `C:\ProgramData\Professional SMART\config\.env`

3. Remove quotes from these lines:
   ```ini
   # Change FROM:
   INPUT_DIR="C:\Program Files\Professional SMART\data\input"
   PROCESSED_DIR="C:\Program Files\Professional SMART\data\processed"
   ERROR_DIR="C:\Program Files\Professional SMART\data\error"

   # Change TO:
   INPUT_DIR=C:\Program Files\Professional SMART\data\input
   PROCESSED_DIR=C:\Program Files\Professional SMART\data\processed
   ERROR_DIR=C:\Program Files\Professional SMART\data\error
   ```

4. Start the service:
   ```powershell
   Start-Service ProfessionalSMART
   ```

**Option 2: Delete .env and Reinstall**

1. Uninstall Professional SMART
2. Delete `C:\ProgramData\Professional SMART\config\.env`
3. Install v1.5.15.2 fresh

## Verification

After upgrading or fixing the `.env` file:

### 1. Check .env File Format
```powershell
Get-Content "C:\ProgramData\Professional SMART\config\.env" | Select-String "INPUT_DIR"
```

**Expected Output**:
```
INPUT_DIR=C:\Program Files\Professional SMART\data\input
```

**NOT** (with quotes):
```
INPUT_DIR="C:\Program Files\Professional SMART\data\input"
```

### 2. Check Service Logs

```powershell
Get-Content "C:\ProgramData\Professional SMART\logs\service.log.2025-11-03" -Tail 50 |
    Select-String "Scanning for existing files"
```

**Expected**: Should show files being found if any exist in input directory:
```
Scanning for existing files in input directory...
Found existing file: C:\Program Files\Professional SMART\data\input\claims_ORG001-R1-F1.edi
Found existing file: C:\Program Files\Professional SMART\data\input\claims_ORG001-R1-F2.edi
...
```

**NOT**:
```
Scanning for existing files in input directory...
No existing files found to process
```

### 3. Test File Processing

Place multiple EDI files in the input directory and verify ALL are processed:

```powershell
# Copy test files
Copy-Item "C:\Users\jonmc\dev\pro\test_data\claims_*.edi" `
          -Destination "C:\Program Files\Professional SMART\data\input\"

# Wait 30 seconds
Start-Sleep -Seconds 30

# Check input directory is empty
Get-ChildItem "C:\Program Files\Professional SMART\data\input\" -Filter "*.edi"
# Expected: No files

# Check processed directory has all files
Get-ChildItem "C:\Program Files\Professional SMART\data\processed\" -Filter "*.edi"
# Expected: All test files present
```

## Compatibility

- **Database**: No schema changes required
- **Configuration**: Manual `.env` file fix required for upgrades
- **Breaking Changes**: None
- **Backward Compatible**: Yes (after `.env` file is corrected)

## Related Issues

This fix resolves the following symptoms:
- "No existing files found to process" even when files are in input directory
- Only one file (newly added) gets processed, others ignored
- Files remain in input directory indefinitely
- .env parse errors in service logs
- File watcher not detecting existing files on startup

## Previous Releases

- **v1.5.15.0**: Initial fix attempt (added file move logic, but .env issue prevented it from working)
- **v1.5.15.1**: Added file move logic to main.rs (but service uses service.rs, so no effect)
- **v1.5.15.2**: Fixed root cause - installer .env generation ← **This Release**

## Known Issues

None at this time.

## Testing Checklist

- [x] Fresh install generates .env file without quotes
- [x] File watcher detects existing files on startup
- [x] Multiple EDI files all processed correctly
- [x] Files moved to processed/ directory after ingestion
- [x] No .env parse errors in logs
- [x] Service starts successfully
- [x] All file extensions (.edi, .837p, .csv) detected

## Support

If issues persist after upgrading:

1. **Verify .env file format** (no quotes around paths)
2. **Restart the service** after fixing .env
3. **Check service logs** for parse errors or file detection issues
4. **Check file permissions** on input directory
5. Contact support with:
   - `.env` file contents (sanitize passwords)
   - Service logs
   - List of files in input directory

## Technical Details

### Why Quotes Were Problematic

The `dotenvy` crate (Rust's .env file parser) follows the dotenv file format specification:

- **Plain values**: `KEY=value with spaces` - The entire value including spaces is used as-is
- **Quoted values**: `KEY="value"` - Quotes are treated as **literal characters**, not delimiters

For Windows paths with spaces like `C:\Program Files\...`:
- ✅ **Correct**: `INPUT_DIR=C:\Program Files\Professional SMART\data\input`
- ❌ **Wrong**: `INPUT_DIR="C:\Program Files\Professional SMART\data\input"` (includes literal quotes in the value)

The parser saw the value as: `"C:\Program Files\..."` (with quotes) instead of `C:\Program Files\...`

### Why File Watcher Failed

When the service tried to use `INPUT_DIR`, it got the literal string:
```
"C:\Program Files\Professional SMART\data\input"
```

Windows file system operations failed because:
1. Directory names cannot contain `"` characters (invalid)
2. The path didn't exist (quotes made it invalid)
3. File watcher couldn't scan a non-existent directory
4. Result: "No existing files found to process"

## Build Information

- **Rust Version**: 1.x.x
- **WiX Toolset**: 3.14.1.8722
- **MSI Size**: ~9 MB
- **Build Date**: November 3, 2025
- **Commit**: 918d040

## Quality Assurance

Tested with:
- Fresh installation ✓
- Upgrade from v1.5.15.1 with manual .env fix ✓
- Multiple EDI files in input directory ✓
- File watcher detection on startup ✓
- All file extensions (.edi, .837p, .csv) ✓
- Paths with spaces ✓
- Service restart after .env changes ✓

## License

Professional SMART - Healthcare Claims Processing System
Copyright (c) 2025 Professional SMART Team
