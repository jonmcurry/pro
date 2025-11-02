# Windows Server 837p File Processing Troubleshooting

## Problem: 837p files not being processed

When .837p files are placed in the input folder but aren't being processed by the service.

## Supported File Extensions (v1.5.2+)

The service automatically processes:
- **`.csv`** - Master data files (organizations, facilities, providers)
- **`.edi`** - 837p EDI claims files (alternative extension)
- **`.837p`** - 837p EDI claims files (native extension)

## Root Causes

### 1. Wrong File Location

**Problem**: Files placed in incorrect directory won't be processed.

**Default Path**: `C:\Program Files\Professional SMART\data\input`

**Location**: `crates/pro-service/src/file_watcher.rs:200-210`
```rust
fn is_processable_file(&self, path: &Path) -> bool {
    if let Some(ext) = path.extension() {
        let ext_lower = ext.to_string_lossy().to_lowercase();
        // Process CSV files (master data) and EDI files (837p claims with .edi or .837p extension)
        if ext_lower == "csv" || ext_lower == "edi" || ext_lower == "837p" {
            return true;
        }
    }
    false
}
```

**Solution**: Move files to correct directory

```powershell
# Check current location
Get-ChildItem "C:\Program Files\Professional SMART" -Recurse -Filter "*.837p"

# Move .837p files to input directory
Move-Item "C:\Path\To\Your\*.837p" "C:\Program Files\Professional SMART\data\input\"

# Also works with .edi files
Move-Item "C:\Path\To\Your\*.edi" "C:\Program Files\Professional SMART\data\input\"

# Verify
Get-ChildItem "C:\Program Files\Professional SMART\data\input"
```

### 2. Service Not Running

**Problem**: File watcher only runs when the Windows service is active.

**Check service status**:
```powershell
# Check if service is running
Get-Service ProfessionalSMART

# If stopped, check why it won't start
Get-EventLog -LogName Application -Source ProfessionalSMART -Newest 20 | Format-List

# Try starting
Start-Service ProfessionalSMART

# Check status again
Get-Service ProfessionalSMART

# If still not running, check exit code
Get-WinEvent -FilterHashtable @{LogName='System'; ID=7034,7031,7030} -MaxEvents 10 |
    Where-Object { $_.Message -like "*ProfessionalSMART*" }
```

### 3. Database Connection Issue

**Problem**: Service starts but immediately stops due to database connection failure.

**Check database connectivity**:
```powershell
# Check if PostgreSQL is running
Get-Service -Name "postgresql*"

# Test connection from Professional SMART machine
# Install psql if not available, or use this PowerShell test:
$env:PGPASSWORD="your_password"
psql -h localhost -U postgres -d professional_smart -c "SELECT 1;"
```

**Check .env configuration**:
```powershell
# View configuration
Get-Content "C:\ProgramData\Professional SMART\config\.env"

# Required variables:
# DB_HOST=localhost
# DB_PORT=5432
# DB_NAME=professional_smart
# DB_USER=postgres
# DB_PASSWORD=YourPassword
```

**Solution**: Fix database configuration

```powershell
# Navigate to config directory
cd "C:\ProgramData\Professional SMART\config"

# Edit .env file
notepad .env

# Update these values:
# DATABASE_URL=postgres://postgres:YourPassword@localhost:5432/professional_smart

# Restart service
Restart-Service ProfessionalSMART
```

### 4. Permission Issues

**Problem**: Service cannot read/write files in input directory.

**Check permissions**:
```powershell
# Check who owns the input directory
Get-Acl "C:\Program Files\Professional SMART\data\input" | Format-List

# Service runs as SYSTEM, verify it has access
icacls "C:\Program Files\Professional SMART\data\input"
```

**Solution**: Fix permissions

```powershell
# Grant SYSTEM full control
icacls "C:\Program Files\Professional SMART\data\input" /grant "SYSTEM:(OI)(CI)F"

# Also grant to processed and error directories
icacls "C:\Program Files\Professional SMART\data\processed" /grant "SYSTEM:(OI)(CI)F"
icacls "C:\Program Files\Professional SMART\data\error" /grant "SYSTEM:(OI)(CI)F"

# Restart service
Restart-Service ProfessionalSMART
```

## Complete Diagnostic Procedure

Run this PowerShell script as Administrator:

```powershell
Write-Host "=== Professional SMART 837p Processing Diagnostic ===" -ForegroundColor Cyan

# 1. Check service status
Write-Host "`n[1/7] Checking service status..." -ForegroundColor Cyan
$service = Get-Service ProfessionalSMART -ErrorAction SilentlyContinue
if ($service) {
    Write-Host "  Status: $($service.Status)" -ForegroundColor $(if($service.Status -eq "Running"){"Green"}else{"Red"})
} else {
    Write-Host "  [X] Service not found" -ForegroundColor Red
}

# 2. Check input directory
Write-Host "`n[2/7] Checking input directory..." -ForegroundColor Cyan
$inputDir = "C:\Program Files\Professional SMART\data\input"
if (Test-Path $inputDir) {
    Write-Host "  [OK] Input directory exists" -ForegroundColor Green
    $files = Get-ChildItem $inputDir
    Write-Host "  Files found: $($files.Count)" -ForegroundColor White

    # Check for .837p files (v1.5.2+)
    $x837pFiles = Get-ChildItem $inputDir -Filter "*.837p"
    if ($x837pFiles) {
        Write-Host "  [OK] Found $($x837pFiles.Count) .837p files ready for processing" -ForegroundColor Green
        $x837pFiles | ForEach-Object { Write-Host "      - $($_.Name)" -ForegroundColor Gray }
    }

    # Check for .edi files (v1.5.1+)
    $ediFiles = Get-ChildItem $inputDir -Filter "*.edi"
    if ($ediFiles) {
        Write-Host "  [OK] Found $($ediFiles.Count) .edi files ready for processing" -ForegroundColor Green
        $ediFiles | ForEach-Object { Write-Host "      - $($_.Name)" -ForegroundColor Gray }
    }

    # Check for .csv files
    $csvFiles = Get-ChildItem $inputDir -Filter "*.csv"
    if ($csvFiles) {
        Write-Host "  [OK] Found $($csvFiles.Count) .csv files ready for processing" -ForegroundColor Green
        $csvFiles | ForEach-Object { Write-Host "      - $($_.Name)" -ForegroundColor Gray }
    }
} else {
    Write-Host "  [X] Input directory does not exist" -ForegroundColor Red
}

# 3. Check database configuration
Write-Host "`n[3/7] Checking database configuration..." -ForegroundColor Cyan
$envFile = "C:\ProgramData\Professional SMART\config\.env"
if (Test-Path $envFile) {
    Write-Host "  [OK] Configuration file exists" -ForegroundColor Green
    $dbUrl = Select-String -Path $envFile -Pattern "DATABASE_URL" | Select-Object -First 1
    if ($dbUrl) {
        $sanitized = $dbUrl -replace 'password=[^@]+@', 'password=***@'
        Write-Host "  DB Config: $sanitized" -ForegroundColor Gray
    }
} else {
    Write-Host "  [X] Configuration file not found" -ForegroundColor Red
}

# 4. Check PostgreSQL service
Write-Host "`n[4/7] Checking PostgreSQL..." -ForegroundColor Cyan
$postgres = Get-Service -Name "postgresql*" -ErrorAction SilentlyContinue
if ($postgres) {
    Write-Host "  [OK] PostgreSQL service found: $($postgres.Status)" -ForegroundColor Green
} else {
    Write-Host "  [X] PostgreSQL service not found" -ForegroundColor Red
}

# 5. Check recent errors in Event Log
Write-Host "`n[5/7] Checking Event Log for errors..." -ForegroundColor Cyan
$errors = Get-EventLog -LogName Application -Source "ProfessionalSMART" -EntryType Error -Newest 5 -ErrorAction SilentlyContinue
if ($errors) {
    Write-Host "  [!] Found $($errors.Count) recent errors:" -ForegroundColor Yellow
    $errors | ForEach-Object {
        Write-Host "      [$($_.TimeGenerated)] $($_.Message.Substring(0,[Math]::Min(100,$_.Message.Length)))" -ForegroundColor Gray
    }
} else {
    Write-Host "  [OK] No recent errors found" -ForegroundColor Green
}

# 6. Check permissions
Write-Host "`n[6/7] Checking directory permissions..." -ForegroundColor Cyan
$acl = Get-Acl $inputDir -ErrorAction SilentlyContinue
if ($acl) {
    $systemAccess = $acl.Access | Where-Object { $_.IdentityReference -like "*SYSTEM*" }
    if ($systemAccess) {
        Write-Host "  [OK] SYSTEM has access" -ForegroundColor Green
    } else {
        Write-Host "  [!] SYSTEM may not have proper access" -ForegroundColor Yellow
    }
} else {
    Write-Host "  [X] Could not check permissions" -ForegroundColor Red
}

# 7. Check processed/error directories
Write-Host "`n[7/7] Checking output directories..." -ForegroundColor Cyan
$processedDir = "C:\Program Files\Professional SMART\data\processed"
$errorDir = "C:\Program Files\Professional SMART\data\error"

if (Test-Path $processedDir) {
    $processed = Get-ChildItem $processedDir
    Write-Host "  Processed: $($processed.Count) files" -ForegroundColor White
}

if (Test-Path $errorDir) {
    $errors = Get-ChildItem $errorDir
    Write-Host "  Errors: $($errors.Count) files" -ForegroundColor $(if($errors.Count -gt 0){"Yellow"}else{"Green"})

    # Show error files
    $errorFiles = Get-ChildItem $errorDir -Filter "*.error"
    if ($errorFiles) {
        Write-Host "`n  Recent error details:" -ForegroundColor Yellow
        $errorFiles | Select-Object -First 3 | ForEach-Object {
            Write-Host "    File: $($_.BaseName)" -ForegroundColor Gray
            $errorContent = Get-Content $_.FullName -First 3
            $errorContent | ForEach-Object { Write-Host "      $_" -ForegroundColor DarkGray }
        }
    }
}

Write-Host "`n=== RECOMMENDATIONS ===" -ForegroundColor Cyan
Write-Host "1. Check service is running:" -ForegroundColor White
Write-Host "   Get-Service ProfessionalSMART" -ForegroundColor Gray
Write-Host "`n2. Restart service:" -ForegroundColor White
Write-Host "   Restart-Service ProfessionalSMART" -ForegroundColor Gray
Write-Host "`n3. Monitor logs:" -ForegroundColor White
Write-Host "   Get-EventLog -LogName Application -Source 'ProfessionalSMART' -Newest 20 | Format-List" -ForegroundColor Gray
```

## Quick Fix Script

Save as `fix-837p-processing.ps1`:

```powershell
# Quick Fix for 837p File Processing (v1.5.2+)
# Run as Administrator

$ErrorActionPreference = "Stop"

$inputDir = "C:\Program Files\Professional SMART\data\input"

Write-Host "Professional SMART - 837p Processing Fix" -ForegroundColor Cyan

# Step 1: Verify processable files exist
Write-Host "`nStep 1: Checking for processable files..." -ForegroundColor Yellow
$x837pFiles = Get-ChildItem $inputDir -Filter "*.837p"
$ediFiles = Get-ChildItem $inputDir -Filter "*.edi"
$csvFiles = Get-ChildItem $inputDir -Filter "*.csv"
$totalFiles = $x837pFiles.Count + $ediFiles.Count + $csvFiles.Count

if ($totalFiles -gt 0) {
    Write-Host "  Found $totalFiles file(s) ready for processing" -ForegroundColor Green
    if ($x837pFiles) {
        Write-Host "    .837p files: $($x837pFiles.Count)" -ForegroundColor White
        $x837pFiles | ForEach-Object { Write-Host "      - $($_.Name)" -ForegroundColor Gray }
    }
    if ($ediFiles) {
        Write-Host "    .edi files: $($ediFiles.Count)" -ForegroundColor White
        $ediFiles | ForEach-Object { Write-Host "      - $($_.Name)" -ForegroundColor Gray }
    }
    if ($csvFiles) {
        Write-Host "    .csv files: $($csvFiles.Count)" -ForegroundColor White
        $csvFiles | ForEach-Object { Write-Host "      - $($_.Name)" -ForegroundColor Gray }
    }
} else {
    Write-Host "  [!] No processable files found (.837p, .edi, or .csv)" -ForegroundColor Red
    exit 1
}

# Step 2: Restart service
Write-Host "`nStep 2: Restarting service..." -ForegroundColor Yellow
try {
    Restart-Service ProfessionalSMART -ErrorAction Stop
    Start-Sleep -Seconds 2
    $service = Get-Service ProfessionalSMART
    Write-Host "  Service status: $($service.Status)" -ForegroundColor $(if($service.Status -eq "Running"){"Green"}else{"Red"})
} catch {
    Write-Host "  [!] Failed to restart service: $_" -ForegroundColor Red
}

Write-Host "`nDone! Monitor processing:" -ForegroundColor Cyan
Write-Host "  Get-EventLog -LogName Application -Source 'ProfessionalSMART' -Newest 20" -ForegroundColor Gray
```

## Monitoring File Processing

Watch files being processed in real-time:

```powershell
# Watch input directory
while ($true) {
    Clear-Host
    Write-Host "=== File Processing Monitor ===" -ForegroundColor Cyan
    Write-Host "Press Ctrl+C to stop`n"

    $input = Get-ChildItem "C:\Program Files\Professional SMART\data\input" -File
    $processed = Get-ChildItem "C:\Program Files\Professional SMART\data\processed" -File
    $errors = Get-ChildItem "C:\Program Files\Professional SMART\data\error" -File

    Write-Host "Input:     $($input.Count) files" -ForegroundColor Yellow
    Write-Host "Processed: $($processed.Count) files" -ForegroundColor Green
    Write-Host "Errors:    $($errors.Count) files" -ForegroundColor Red

    Write-Host "`nRecent events:"
    Get-EventLog -LogName Application -Source "ProfessionalSMART" -Newest 5 -ErrorAction SilentlyContinue |
        ForEach-Object {
            $color = if ($_.EntryType -eq "Error") {"Red"} elseif ($_.EntryType -eq "Warning") {"Yellow"} else {"Green"}
            Write-Host "  [$($_.TimeGenerated.ToString('HH:mm:ss'))] $($_.Message.Substring(0,[Math]::Min(80,$_.Message.Length)))" -ForegroundColor $color
        }

    Start-Sleep -Seconds 5
}
```

## Summary

**No Renaming Required**: As of v1.5.2, .837p files are processed natively!

**Supported Extensions (v1.5.2+)**:
- `.837p` - 837p EDI claims files (native extension)
- `.edi` - 837p EDI claims files (alternative extension)
- `.csv` - Master data or claims files

**Quick Fix** (if files aren't processing):
```powershell
# Ensure service is running
Restart-Service ProfessionalSMART

# Check Event Log for errors
Get-EventLog -LogName Application -Source "ProfessionalSMART" -EntryType Error -Newest 10
```

**Verify**:
```powershell
Get-EventLog -LogName Application -Source "ProfessionalSMART" -Newest 10 | Format-List
```

**Version Check**:
To see if you have v1.5.2+ with native .837p support:
```powershell
Get-ItemProperty "HKLM:\SOFTWARE\ProfessionalSMART" -Name Version
```
Should show `1.5.2.0` or higher.

**Upgrade Path**:
- v1.5.0 and earlier: Only .csv files processed
- v1.5.1: Added .edi support (required renaming .837p → .edi)
- v1.5.2: Added native .837p support (no renaming required!)
