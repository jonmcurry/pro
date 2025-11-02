# Professional SMART - 837p File Processing Diagnostic
# Run as Administrator on Windows Server

Write-Host "`n=== Professional SMART 837p Processing Diagnostic ===" -ForegroundColor Cyan
Write-Host "Timestamp: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')`n"

$inputDir = "C:\Program Files\Professional SMART\data\input"
$processedDir = "C:\Program Files\Professional SMART\data\processed"
$errorDir = "C:\Program Files\Professional SMART\data\error"

# 1. Check service status
Write-Host "[1/7] Checking service status..." -ForegroundColor Cyan
$service = Get-Service ProfessionalSMART -ErrorAction SilentlyContinue
if ($service) {
    if ($service.Status -eq "Running") {
        Write-Host "  [OK] Service is running" -ForegroundColor Green
    } else {
        Write-Host "  [X] Service is $($service.Status)" -ForegroundColor Red
    }
} else {
    Write-Host "  [X] Service not found - MSI may not be installed" -ForegroundColor Red
    exit 1
}

# 2. Check input directory and files
Write-Host "`n[2/7] Checking input directory..." -ForegroundColor Cyan
if (Test-Path $inputDir) {
    Write-Host "  [OK] Input directory exists: $inputDir" -ForegroundColor Green

    # Check for .837p files (v1.5.2+ - native support!)
    $x837pFiles = Get-ChildItem $inputDir -Filter "*.837p" -ErrorAction SilentlyContinue
    if ($x837pFiles) {
        Write-Host "  [OK] Found $($x837pFiles.Count) .837p files ready for processing" -ForegroundColor Green
        $x837pFiles | ForEach-Object { Write-Host "      - $($_.Name)" -ForegroundColor Gray }
    }

    # Check for .edi files (v1.5.1+)
    $ediFiles = Get-ChildItem $inputDir -Filter "*.edi" -ErrorAction SilentlyContinue
    if ($ediFiles) {
        Write-Host "  [OK] Found $($ediFiles.Count) .edi files ready for processing" -ForegroundColor Green
        $ediFiles | ForEach-Object { Write-Host "      - $($_.Name)" -ForegroundColor Gray }
    }

    # Check for .csv files
    $csvFiles = Get-ChildItem $inputDir -Filter "*.csv" -ErrorAction SilentlyContinue
    if ($csvFiles) {
        Write-Host "  [OK] Found $($csvFiles.Count) .csv files ready for processing" -ForegroundColor Green
        $csvFiles | ForEach-Object { Write-Host "      - $($_.Name)" -ForegroundColor Gray }
    }

    # Check if directory is empty
    $allFiles = Get-ChildItem $inputDir -File -ErrorAction SilentlyContinue
    if ($allFiles.Count -eq 0) {
        Write-Host "  [!] Input directory is empty" -ForegroundColor Yellow
    } elseif (-not $x837pFiles -and -not $ediFiles -and -not $csvFiles) {
        Write-Host "  [!] No processable files found (.837p, .edi, or .csv)" -ForegroundColor Yellow
    }
} else {
    Write-Host "  [X] Input directory does not exist: $inputDir" -ForegroundColor Red
}

# 3. Check database configuration
Write-Host "`n[3/7] Checking database configuration..." -ForegroundColor Cyan
$envFile = "C:\ProgramData\Professional SMART\config\.env"
if (Test-Path $envFile) {
    Write-Host "  [OK] Configuration file exists" -ForegroundColor Green

    $content = Get-Content $envFile
    $dbUrl = $content | Where-Object { $_ -like "DATABASE_URL=*" } | Select-Object -First 1
    if ($dbUrl) {
        # Sanitize password
        $sanitized = $dbUrl -replace ':([^:@]+)@', ':***@'
        Write-Host "  Database: $sanitized" -ForegroundColor Gray
    } else {
        Write-Host "  [!] DATABASE_URL not found in .env" -ForegroundColor Yellow
    }
} else {
    Write-Host "  [X] Configuration file not found: $envFile" -ForegroundColor Red
    Write-Host "      Run Configuration Wizard: pro-setup.exe" -ForegroundColor Yellow
}

# 4. Check PostgreSQL service
Write-Host "`n[4/7] Checking PostgreSQL..." -ForegroundColor Cyan
$postgres = Get-Service -Name "postgresql*" -ErrorAction SilentlyContinue
if ($postgres) {
    foreach ($pg in $postgres) {
        $statusColor = if ($pg.Status -eq "Running") {"Green"} else {"Yellow"}
        Write-Host "  $($pg.DisplayName): $($pg.Status)" -ForegroundColor $statusColor
    }
} else {
    Write-Host "  [!] PostgreSQL service not found on this machine" -ForegroundColor Yellow
    Write-Host "      Database may be on remote server" -ForegroundColor Gray
}

# 5. Check recent logs in Event Viewer
Write-Host "`n[5/7] Checking Event Log..." -ForegroundColor Cyan
try {
    $recentEvents = Get-EventLog -LogName Application -Source "ProfessionalSMART" -Newest 10 -ErrorAction SilentlyContinue
    if ($recentEvents) {
        $errorCount = ($recentEvents | Where-Object { $_.EntryType -eq "Error" }).Count
        $warningCount = ($recentEvents | Where-Object { $_.EntryType -eq "Warning" }).Count

        Write-Host "  Found $($recentEvents.Count) recent events:" -ForegroundColor White
        Write-Host "    Errors: $errorCount" -ForegroundColor $(if($errorCount -gt 0){"Red"}else{"Green"})
        Write-Host "    Warnings: $warningCount" -ForegroundColor $(if($warningCount -gt 0){"Yellow"}else{"Green"})

        Write-Host "`n  Last 3 events:" -ForegroundColor Gray
        $recentEvents | Select-Object -First 3 | ForEach-Object {
            $color = if ($_.EntryType -eq "Error") {"Red"} elseif ($_.EntryType -eq "Warning") {"Yellow"} else {"White"}
            $msg = if ($_.Message.Length -gt 100) { $_.Message.Substring(0,100) + "..." } else { $_.Message }
            Write-Host "    [$($_.TimeGenerated.ToString('HH:mm:ss'))] $($_.EntryType): $msg" -ForegroundColor $color
        }
    } else {
        Write-Host "  [!] No events found - service may not have logged anything yet" -ForegroundColor Yellow
    }
} catch {
    Write-Host "  [!] Could not read Event Log: $_" -ForegroundColor Yellow
}

# 6. Check directory permissions
Write-Host "`n[6/7] Checking directory permissions..." -ForegroundColor Cyan
try {
    $acl = Get-Acl $inputDir -ErrorAction Stop
    $systemAccess = $acl.Access | Where-Object { $_.IdentityReference -like "*SYSTEM*" }
    if ($systemAccess) {
        $rights = $systemAccess.FileSystemRights
        Write-Host "  [OK] SYSTEM has access: $rights" -ForegroundColor Green
    } else {
        Write-Host "  [!] SYSTEM may not have proper access" -ForegroundColor Yellow
    }
} catch {
    Write-Host "  [X] Could not check permissions: $_" -ForegroundColor Red
}

# 7. Check processed and error directories
Write-Host "`n[7/7] Checking output directories..." -ForegroundColor Cyan

if (Test-Path $processedDir) {
    $processed = Get-ChildItem $processedDir -File -ErrorAction SilentlyContinue
    Write-Host "  Processed directory: $($processed.Count) files" -ForegroundColor $(if($processed.Count -gt 0){"Green"}else{"Gray"})
    if ($processed) {
        $processed | Select-Object -First 5 | ForEach-Object { Write-Host "    - $($_.Name)" -ForegroundColor Gray }
    }
} else {
    Write-Host "  [X] Processed directory does not exist" -ForegroundColor Red
}

if (Test-Path $errorDir) {
    $errors = Get-ChildItem $errorDir -File -ErrorAction SilentlyContinue
    Write-Host "  Error directory: $($errors.Count) files" -ForegroundColor $(if($errors.Count -gt 0){"Red"}else{"Green"})

    $errorFiles = Get-ChildItem $errorDir -Filter "*.error" -ErrorAction SilentlyContinue
    if ($errorFiles) {
        Write-Host "`n  Recent errors:" -ForegroundColor Red
        $errorFiles | Select-Object -First 3 | ForEach-Object {
            Write-Host "    File: $($_.BaseName)" -ForegroundColor Yellow
            $errorContent = Get-Content $_.FullName -First 2 -ErrorAction SilentlyContinue
            $errorContent | ForEach-Object { Write-Host "      $_" -ForegroundColor DarkGray }
        }
    }
} else {
    Write-Host "  [X] Error directory does not exist" -ForegroundColor Red
}

# Summary and recommendations
Write-Host "`n=== DIAGNOSIS SUMMARY ===" -ForegroundColor Cyan

$issues = @()
$fixes = @()

if ($service.Status -ne "Running") {
    $issues += "Service is not running"
    $fixes += "Start service: Start-Service ProfessionalSMART"
}

if (-not (Test-Path $envFile)) {
    $issues += "Configuration file missing"
    $fixes += "Run: pro-setup.exe"
}

# Check if there are processable files but no processing happening
$hasFiles = ($x837pFiles.Count -gt 0) -or ($ediFiles.Count -gt 0) -or ($csvFiles.Count -gt 0)
if ($hasFiles -and $service.Status -eq "Running") {
    Write-Host "[!] Files are present and service is running - check Event Log for processing errors" -ForegroundColor Yellow
}

if ($issues.Count -eq 0) {
    Write-Host "[OK] No obvious issues found" -ForegroundColor Green
    Write-Host "`nIf files still aren't processing, check:" -ForegroundColor Yellow
    Write-Host "  1. Database connection is working"
    Write-Host "  2. Master data is loaded (organizations, facilities)"
    Write-Host "  3. Event Log for specific error messages"
} else {
    Write-Host "[X] Found $($issues.Count) issue(s):" -ForegroundColor Red
    $issues | ForEach-Object { Write-Host "  - $_" -ForegroundColor Yellow }

    Write-Host "`nRECOMMENDED FIXES:" -ForegroundColor Cyan
    $fixes | ForEach-Object { Write-Host "  $_" -ForegroundColor White }
}

Write-Host "`n=== QUICK FIX COMMANDS ===" -ForegroundColor Cyan
Write-Host "1. Check service status:" -ForegroundColor White
Write-Host "   Get-Service ProfessionalSMART" -ForegroundColor Gray

Write-Host "`n2. Restart service:" -ForegroundColor White
Write-Host "   Restart-Service ProfessionalSMART" -ForegroundColor Gray

Write-Host "`n3. Monitor logs:" -ForegroundColor White
Write-Host "   Get-EventLog -LogName Application -Source 'ProfessionalSMART' -Newest 20 | Format-List" -ForegroundColor Gray

Write-Host "`n4. Watch file processing:" -ForegroundColor White
Write-Host "   Get-ChildItem '$inputDir' -File | Measure-Object" -ForegroundColor Gray
Write-Host "   Get-ChildItem '$processedDir' -File | Measure-Object" -ForegroundColor Gray

Write-Host "`n=== SUPPORTED FILE EXTENSIONS (v1.5.2+) ===" -ForegroundColor Cyan
Write-Host ".837p - 837p EDI claims files (native extension)" -ForegroundColor White
Write-Host ".edi  - 837p EDI claims files (alternative extension)" -ForegroundColor White
Write-Host ".csv  - Master data files (organizations, facilities, providers)" -ForegroundColor White
Write-Host "`nNo file renaming required!" -ForegroundColor Green

Write-Host "`n"
