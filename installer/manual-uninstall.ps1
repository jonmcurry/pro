# Manual Uninstall Script for Professional SMART
# Run this as Administrator

Write-Host "Professional SMART Manual Uninstall Script" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Stop the service
Write-Host "1. Stopping service..." -ForegroundColor Yellow
try {
    $service = Get-Service -Name "ProfessionalSMART" -ErrorAction SilentlyContinue
    if ($service) {
        if ($service.Status -eq 'Running') {
            Write-Host "   Service is running, stopping it..."
            Stop-Service -Name "ProfessionalSMART" -Force -ErrorAction Stop
            Start-Sleep -Seconds 2
            Write-Host "   Service stopped successfully" -ForegroundColor Green
        } else {
            Write-Host "   Service is already stopped" -ForegroundColor Green
        }
    } else {
        Write-Host "   Service does not exist" -ForegroundColor Gray
    }
} catch {
    Write-Host "   Warning: Could not stop service: $_" -ForegroundColor Red
}

# Kill any running processes
Write-Host ""
Write-Host "2. Killing any running processes..." -ForegroundColor Yellow
try {
    $processes = Get-Process -Name "pro-service" -ErrorAction SilentlyContinue
    if ($processes) {
        $processes | ForEach-Object {
            Write-Host "   Killing process $($_.Id)..."
            $_ | Stop-Process -Force
        }
        Start-Sleep -Seconds 2
        Write-Host "   Processes killed successfully" -ForegroundColor Green
    } else {
        Write-Host "   No running processes found" -ForegroundColor Gray
    }
} catch {
    Write-Host "   Warning: $_" -ForegroundColor Red
}

# Delete the service
Write-Host ""
Write-Host "3. Deleting service..." -ForegroundColor Yellow
try {
    $service = Get-Service -Name "ProfessionalSMART" -ErrorAction SilentlyContinue
    if ($service) {
        sc.exe delete ProfessionalSMART
        Write-Host "   Service deleted successfully" -ForegroundColor Green
    } else {
        Write-Host "   Service does not exist" -ForegroundColor Gray
    }
} catch {
    Write-Host "   Warning: Could not delete service: $_" -ForegroundColor Red
}

# Remove Program Files directory
Write-Host ""
Write-Host "4. Removing Program Files..." -ForegroundColor Yellow
$programFiles = "C:\Program Files\Professional SMART"
if (Test-Path $programFiles) {
    try {
        Remove-Item -Path $programFiles -Recurse -Force -ErrorAction Stop
        Write-Host "   Program Files removed successfully" -ForegroundColor Green
    } catch {
        Write-Host "   Warning: Could not remove Program Files: $_" -ForegroundColor Red
        Write-Host "   Path: $programFiles" -ForegroundColor Red
    }
} else {
    Write-Host "   Program Files directory does not exist" -ForegroundColor Gray
}

# Remove ProgramData directory
Write-Host ""
Write-Host "5. Removing ProgramData..." -ForegroundColor Yellow
$programData = "C:\ProgramData\Professional SMART"
if (Test-Path $programData) {
    try {
        Remove-Item -Path $programData -Recurse -Force -ErrorAction Stop
        Write-Host "   ProgramData removed successfully" -ForegroundColor Green
    } catch {
        Write-Host "   Warning: Could not remove ProgramData: $_" -ForegroundColor Red
        Write-Host "   Path: $programData" -ForegroundColor Red
    }
} else {
    Write-Host "   ProgramData directory does not exist" -ForegroundColor Gray
}

# Remove registry keys
Write-Host ""
Write-Host "6. Removing registry keys..." -ForegroundColor Yellow
try {
    $regPath = "HKLM:\SOFTWARE\ProfessionalSMART"
    if (Test-Path $regPath) {
        Remove-Item -Path $regPath -Recurse -Force -ErrorAction Stop
        Write-Host "   Registry keys removed successfully" -ForegroundColor Green
    } else {
        Write-Host "   Registry keys do not exist" -ForegroundColor Gray
    }
} catch {
    Write-Host "   Warning: Could not remove registry keys: $_" -ForegroundColor Red
}

# Remove user registry keys
Write-Host ""
Write-Host "7. Removing user registry keys..." -ForegroundColor Yellow
try {
    $regPath = "HKCU:\Software\ProfessionalSMART"
    if (Test-Path $regPath) {
        Remove-Item -Path $regPath -Recurse -Force -ErrorAction Stop
        Write-Host "   User registry keys removed successfully" -ForegroundColor Green
    } else {
        Write-Host "   User registry keys do not exist" -ForegroundColor Gray
    }
} catch {
    Write-Host "   Warning: Could not remove user registry keys: $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "=========================================="  -ForegroundColor Cyan
Write-Host "Manual uninstall completed!" -ForegroundColor Cyan
Write-Host "You can now install the new version." -ForegroundColor Cyan
Write-Host ""
