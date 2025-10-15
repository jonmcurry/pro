# Professional SMART - Build Release Script
# This script builds the release binaries on Windows (not WSL)

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Building Professional SMART - Release Mode" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Check if cargo is available
if (!(Get-Command cargo -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: Cargo not found in PATH" -ForegroundColor Red
    Write-Host "Please install Rust from https://rustup.rs/" -ForegroundColor Yellow
    exit 1
}

Write-Host "Cargo detected: $(cargo --version)" -ForegroundColor Green
Write-Host ""

# Set working directory
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

Write-Host "Project root: $projectRoot" -ForegroundColor Gray
Write-Host ""

# Clean previous build artifacts (optional)
$cleanBuild = $false
if ($args -contains "--clean") {
    $cleanBuild = $true
    Write-Host "Cleaning previous build artifacts..." -ForegroundColor Yellow
    cargo clean
    Write-Host ""
}

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Building Release Binaries..." -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "This may take 5-10 minutes on first build..." -ForegroundColor Yellow
Write-Host ""

# Build release binaries
$buildStart = Get-Date

$buildResult = cargo build --release --bin pro-service --bin pro-setup 2>&1

$buildEnd = Get-Date
$buildDuration = $buildEnd - $buildStart

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "================================================" -ForegroundColor Red
    Write-Host "BUILD FAILED!" -ForegroundColor Red
    Write-Host "================================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "Error output:" -ForegroundColor Red
    Write-Host $buildResult -ForegroundColor Gray
    exit 1
}

Write-Host ""
Write-Host "================================================" -ForegroundColor Green
Write-Host "BUILD SUCCESSFUL!" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Build time: $($buildDuration.Minutes)m $($buildDuration.Seconds)s" -ForegroundColor Gray
Write-Host ""

# Check output binaries
$serviceBinary = Join-Path $projectRoot "target\release\pro-service.exe"
$setupBinary = Join-Path $projectRoot "target\release\pro-setup.exe"

if (Test-Path $serviceBinary) {
    $serviceSize = (Get-Item $serviceBinary).Length / 1MB
    Write-Host "pro-service.exe:" -ForegroundColor Cyan -NoNewline
    Write-Host " $([math]::Round($serviceSize, 2)) MB" -ForegroundColor Gray
} else {
    Write-Host "WARNING: pro-service.exe not found!" -ForegroundColor Yellow
}

if (Test-Path $setupBinary) {
    $setupSize = (Get-Item $setupBinary).Length / 1MB
    Write-Host "pro-setup.exe:" -ForegroundColor Cyan -NoNewline
    Write-Host " $([math]::Round($setupSize, 2)) MB" -ForegroundColor Gray
} else {
    Write-Host "WARNING: pro-setup.exe not found!" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Next Steps" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "To build the MSI installer:" -ForegroundColor White
Write-Host "  cd installer" -ForegroundColor Gray
Write-Host "  .\build.bat" -ForegroundColor Gray
Write-Host ""
Write-Host "To run the service:" -ForegroundColor White
Write-Host "  .\target\release\pro-service.exe" -ForegroundColor Gray
Write-Host ""
Write-Host "To run database setup:" -ForegroundColor White
Write-Host "  .\target\release\pro-setup.exe migrate" -ForegroundColor Gray
Write-Host ""
