# Quick Fix Guide - Multiple Files Not Processing

**Problem**: Only one EDI file processes, others remain in input directory

**Root Cause**: The `.env` file has quotes around paths, preventing file watcher from detecting files

## Immediate Fix (Takes 1 Minute)

### Step 1: Stop the Service
```powershell
Stop-Service ProfessionalSMART
```

### Step 2: Edit .env File

Open: `C:\ProgramData\Professional SMART\config\.env`

**Change these 3 lines** (remove the quotes):

**Before**:
```ini
INPUT_DIR="C:\Program Files\Professional SMART\data\input"
PROCESSED_DIR="C:\Program Files\Professional SMART\data\processed"
ERROR_DIR="C:\Program Files\Professional SMART\data\error"
```

**After**:
```ini
INPUT_DIR=C:\Program Files\Professional SMART\data\input
PROCESSED_DIR=C:\Program Files\Professional SMART\data\processed
ERROR_DIR=C:\Program Files\Professional SMART\data\error
```

### Step 3: Start the Service
```powershell
Start-Service ProfessionalSMART
```

### Step 4: Verify It Works

```powershell
# Check service is running
Get-Service ProfessionalSMART
# Expected: Status = Running

# Check logs show files being detected
Get-Content "C:\ProgramData\Professional SMART\logs\service.log.2025-11-03" -Tail 20 |
    Select-String "Found existing file"
# Expected: Should show files being found

# Place test files
Copy-Item "C:\Users\jonmc\dev\pro\test_data\claims_*.edi" `
          -Destination "C:\Program Files\Professional SMART\data\input\"

# Wait 30 seconds
Start-Sleep -Seconds 30

# Check input directory is empty
Get-ChildItem "C:\Program Files\Professional SMART\data\input\" -Filter "*.edi"
# Expected: No files (all moved to processed/)

# Check processed directory has all files
Get-ChildItem "C:\Program Files\Professional SMART\data\processed\" -Filter "*.edi"
# Expected: All test files present
```

## Long-Term Fix (For Future Installs)

The installer has been fixed in **v1.5.15.2** to not add quotes.

If you reinstall or do a fresh install, use:
```
installer/ProfessionalSMART.msi (v1.5.15.2 or later)
```

## Why This Happened

The installer was adding quotes around Windows paths in the .env file. The Rust `dotenvy` library treats quotes as literal characters, so it tried to find a directory called:
```
"C:\Program Files\Professional SMART\data\input"  (with quotes)
```

This directory doesn't exist, so the file watcher couldn't find any files.

## Files Included

- `installer/ProfessionalSMART.msi` - v1.5.15.2 (Fixed installer)
- `RELEASE_NOTES_1.5.15.2.md` - Complete release notes
- `QUICK_FIX_GUIDE.md` - This file

## Questions?

Check the logs if files still aren't processing:
```powershell
Get-Content "C:\ProgramData\Professional SMART\logs\service.log.2025-11-03" -Tail 100
```

Look for:
- ✅ Good: "Found existing file: ..."
- ✅ Good: "Moving file to processed: ..."
- ❌ Bad: "No existing files found to process"
- ❌ Bad: "Failed to load .env file... LineParse"
