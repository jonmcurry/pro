# build-msi.ps1 - Automated MSI build with version increment
# Usage: .\build-msi.ps1 [major.minor.patch]
# If no version specified, auto-increments the build number

param(
    [string]$Version = "",
    [switch]$Major,
    [switch]$Minor,
    [switch]$Patch,
    [switch]$NoBuild
)

$ErrorActionPreference = "Stop"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Professional SMART MSI Builder" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Version file to track build numbers
$versionFile = ".\version.txt"
$productWxs = ".\Product.wxs"

# Read current version from version.txt or initialize
if (Test-Path $versionFile) {
    $currentVersion = Get-Content $versionFile -Raw
    $currentVersion = $currentVersion.Trim()
    Write-Host "Current version: $currentVersion" -ForegroundColor Yellow
} else {
    $currentVersion = "1.1.0.0"
    Write-Host "Initializing version: $currentVersion" -ForegroundColor Yellow
}

# Parse current version
$versionParts = $currentVersion.Split('.')
$majorVer = [int]$versionParts[0]
$minorVer = [int]$versionParts[1]
$patchVer = [int]$versionParts[2]
$buildVer = [int]$versionParts[3]

# Determine new version
if ($Version -ne "") {
    # User specified exact version
    $newVersion = $Version
    if ($newVersion -notmatch '^\d+\.\d+\.\d+\.\d+$') {
        Write-Error "Invalid version format. Use: major.minor.patch.build (e.g., 1.2.0.0)"
        exit 1
    }
    Write-Host "Using specified version: $newVersion" -ForegroundColor Green
} elseif ($Major) {
    # Increment major version
    $majorVer++
    $minorVer = 0
    $patchVer = 0
    $buildVer = 0
    $newVersion = "$majorVer.$minorVer.$patchVer.$buildVer"
    Write-Host "Incrementing MAJOR version: $newVersion" -ForegroundColor Green
} elseif ($Minor) {
    # Increment minor version
    $minorVer++
    $patchVer = 0
    $buildVer = 0
    $newVersion = "$majorVer.$minorVer.$patchVer.$buildVer"
    Write-Host "Incrementing MINOR version: $newVersion" -ForegroundColor Green
} elseif ($Patch) {
    # Increment patch version
    $patchVer++
    $buildVer = 0
    $newVersion = "$majorVer.$minorVer.$patchVer.$buildVer"
    Write-Host "Incrementing PATCH version: $newVersion" -ForegroundColor Green
} else {
    # Auto-increment build number (default)
    $buildVer++
    $newVersion = "$majorVer.$minorVer.$patchVer.$buildVer"
    Write-Host "Auto-incrementing BUILD number: $newVersion" -ForegroundColor Green
}

Write-Host ""

# Update Product.wxs with new version
Write-Host "Updating Product.wxs..." -ForegroundColor Cyan

$productContent = Get-Content $productWxs -Raw

# Update Product Version attribute
$productContent = $productContent -replace 'Version="\d+\.\d+\.\d+\.\d+"', "Version=`"$newVersion`""

# Update VersionRegistryKey Value attribute
$productContent = $productContent -replace '<RegistryValue Root="HKLM"\s+Key="SOFTWARE\\ProfessionalSMART"\s+Name="Version"\s+Type="string"\s+Value="\d+\.\d+\.\d+\.\d+"', "<RegistryValue Root=`"HKLM`"`n                     Key=`"SOFTWARE\ProfessionalSMART`"`n                     Name=`"Version`"`n                     Type=`"string`"`n                     Value=`"$newVersion`""

# Write updated content
Set-Content -Path $productWxs -Value $productContent -NoNewline

Write-Host "Product.wxs updated with version $newVersion" -ForegroundColor Green
Write-Host ""

# Save new version to version.txt
Set-Content -Path $versionFile -Value $newVersion -NoNewline
Write-Host "Version saved to $versionFile" -ForegroundColor Green
Write-Host ""

# Build Rust binaries unless -NoBuild specified
if (-not $NoBuild) {
    Write-Host "Building Rust binaries..." -ForegroundColor Cyan
    Push-Location ..

    try {
        & cargo build --release 2>&1 | ForEach-Object {
            Write-Host $_ -ForegroundColor Gray
        }

        if ($LASTEXITCODE -ne 0) {
            Write-Error "Cargo build failed with exit code $LASTEXITCODE"
            Pop-Location
            exit $LASTEXITCODE
        }

        Write-Host "Rust binaries built successfully" -ForegroundColor Green
    } finally {
        Pop-Location
    }

    Write-Host ""
} else {
    Write-Host "Skipping Rust build (NoBuild flag specified)" -ForegroundColor Yellow
    Write-Host ""
}

# Build MSI with WiX
Write-Host "Compiling WiX files..." -ForegroundColor Cyan

$candle = "candle"
$light = "light"

# Check if WiX is in PATH
try {
    & $candle -? > $null 2>&1
} catch {
    Write-Error "WiX toolset not found in PATH. Please install WiX Toolset from https://wixtoolset.org/"
    exit 1
}

# Compile .wxs files
Write-Host "Running candle..." -ForegroundColor Gray
& $candle Product.wxs DatabaseConfigDlg.wxs PrerequisiteDlg.wxs 2>&1 | ForEach-Object {
    Write-Host $_ -ForegroundColor DarkGray
}

if ($LASTEXITCODE -ne 0) {
    Write-Error "Candle compilation failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

Write-Host "Candle completed successfully" -ForegroundColor Green
Write-Host ""

# Link .wixobj files
Write-Host "Running light..." -ForegroundColor Gray
& $light -ext WixUIExtension Product.wixobj DatabaseConfigDlg.wixobj PrerequisiteDlg.wixobj -out ProfessionalSMART.msi 2>&1 | ForEach-Object {
    Write-Host $_ -ForegroundColor DarkGray
}

if ($LASTEXITCODE -ne 0) {
    Write-Error "Light linking failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

Write-Host "Light completed successfully" -ForegroundColor Green
Write-Host ""

# Get MSI file info
if (Test-Path ".\ProfessionalSMART.msi") {
    $msiFile = Get-Item ".\ProfessionalSMART.msi"
    $msiSizeMB = [math]::Round($msiFile.Length / 1MB, 2)

    Write-Host "============================================" -ForegroundColor Cyan
    Write-Host "BUILD SUCCESSFUL!" -ForegroundColor Green
    Write-Host "============================================" -ForegroundColor Cyan
    Write-Host "Version:  $newVersion" -ForegroundColor White
    Write-Host "File:     $($msiFile.FullName)" -ForegroundColor White
    Write-Host "Size:     $msiSizeMB MB" -ForegroundColor White
    Write-Host "Modified: $($msiFile.LastWriteTime)" -ForegroundColor White
    Write-Host "============================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "To install/upgrade:" -ForegroundColor Yellow
    Write-Host "  msiexec /i ProfessionalSMART.msi /l*v install.log" -ForegroundColor White
    Write-Host ""
    Write-Host "To uninstall:" -ForegroundColor Yellow
    Write-Host "  msiexec /x ProfessionalSMART.msi /l*v uninstall.log" -ForegroundColor White
    Write-Host ""
} else {
    Write-Error "MSI file was not created"
    exit 1
}

# Cleanup .wixobj and .wixpdb files (optional)
Write-Host "Cleaning up build artifacts..." -ForegroundColor Cyan
Remove-Item -Path "*.wixobj" -ErrorAction SilentlyContinue
Remove-Item -Path "*.wixpdb" -ErrorAction SilentlyContinue
Write-Host "Cleanup complete" -ForegroundColor Green
Write-Host ""

Write-Host "Build process completed successfully!" -ForegroundColor Green
