# smart-build.ps1 - Intelligent version increment based on Git changes
# Analyzes git commits to suggest appropriate version increment

param(
    [switch]$Auto,      # Automatically determine version increment
    [switch]$Force,     # Force the suggested increment
    [string]$Version = "",
    [switch]$Major,
    [switch]$Minor,
    [switch]$Patch,
    [switch]$NoBuild
)

$ErrorActionPreference = "Stop"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Professional SMART Smart Builder" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Check if git is available
$gitAvailable = $false
try {
    git --version > $null 2>&1
    $gitAvailable = $true
} catch {
    Write-Host "Git not available - falling back to manual versioning" -ForegroundColor Yellow
}

# Analyze changes if Auto mode and git available
if ($Auto -and $gitAvailable) {
    Write-Host "Analyzing recent changes..." -ForegroundColor Cyan
    Write-Host ""

    # Get commits since last tag
    $lastTag = git describe --tags --abbrev=0 2>$null
    if ($lastTag) {
        Write-Host "Last tagged version: $lastTag" -ForegroundColor Yellow
        $commitRange = "$lastTag..HEAD"
    } else {
        Write-Host "No previous tags found, analyzing all commits" -ForegroundColor Yellow
        $commitRange = "HEAD"
    }

    # Get commit messages
    $commits = git log $commitRange --pretty=format:"%s" 2>$null

    if (-not $commits) {
        Write-Host "No new commits since last tag" -ForegroundColor Yellow
        if (-not $Force) {
            Write-Host "Use -Force to rebuild anyway" -ForegroundColor Gray
            exit 0
        }
    }

    Write-Host "Recent commits:" -ForegroundColor Cyan
    $commits | ForEach-Object { Write-Host "  $_" -ForegroundColor Gray }
    Write-Host ""

    # Analyze commit messages for keywords
    $hasMajor = $false
    $hasMinor = $false
    $hasPatch = $false

    $majorKeywords = @("BREAKING", "breaking change", "major feature", "remove support", "incompatible")
    $minorKeywords = @("feature", "add", "new", "enhancement", "support for", "implement")
    $patchKeywords = @("fix", "bug", "patch", "correct", "repair", "hotfix", "security")

    foreach ($commit in $commits) {
        $commitLower = $commit.ToLower()

        foreach ($keyword in $majorKeywords) {
            if ($commitLower -match $keyword.ToLower()) {
                $hasMajor = $true
                break
            }
        }

        foreach ($keyword in $minorKeywords) {
            if ($commitLower -match $keyword.ToLower()) {
                $hasMinor = $true
            }
        }

        foreach ($keyword in $patchKeywords) {
            if ($commitLower -match $keyword.ToLower()) {
                $hasPatch = $true
            }
        }
    }

    # Check for migration files
    $migrationChanges = git diff $commitRange --name-only migrations/ 2>$null
    if ($migrationChanges) {
        Write-Host "Migration changes detected:" -ForegroundColor Yellow
        $migrationChanges | ForEach-Object { Write-Host "  $_" -ForegroundColor Gray }
        $hasMinor = $true
    }

    # Suggest version increment
    Write-Host ""
    Write-Host "Analysis Results:" -ForegroundColor Cyan
    if ($hasMajor) {
        Write-Host "  [!] MAJOR version increment recommended" -ForegroundColor Red
        Write-Host "      Reason: Breaking changes detected" -ForegroundColor Gray
        $suggestedIncrement = "Major"
    } elseif ($hasMinor) {
        Write-Host "  [+] MINOR version increment recommended" -ForegroundColor Green
        Write-Host "      Reason: New features or migrations detected" -ForegroundColor Gray
        $suggestedIncrement = "Minor"
    } elseif ($hasPatch) {
        Write-Host "  [~] PATCH version increment recommended" -ForegroundColor Yellow
        Write-Host "      Reason: Bug fixes detected" -ForegroundColor Gray
        $suggestedIncrement = "Patch"
    } else {
        Write-Host "  [.] BUILD number increment (default)" -ForegroundColor Cyan
        Write-Host "      Reason: No significant changes detected" -ForegroundColor Gray
        $suggestedIncrement = "Build"
    }

    Write-Host ""

    if ($Force) {
        Write-Host "Auto-applying suggested increment: $suggestedIncrement" -ForegroundColor Green
        switch ($suggestedIncrement) {
            "Major" { $Major = $true }
            "Minor" { $Minor = $true }
            "Patch" { $Patch = $true }
        }
    } else {
        Write-Host "Suggested increment: $suggestedIncrement" -ForegroundColor Yellow
        Write-Host "Use -Force to auto-apply, or specify -Major, -Minor, or -Patch manually" -ForegroundColor Gray
        Write-Host ""
        $response = Read-Host "Apply suggested increment? (Y/n)"
        if ($response -eq "" -or $response -eq "Y" -or $response -eq "y") {
            switch ($suggestedIncrement) {
                "Major" { $Major = $true }
                "Minor" { $Minor = $true }
                "Patch" { $Patch = $true }
            }
        } else {
            Write-Host "Increment cancelled" -ForegroundColor Yellow
            exit 0
        }
    }
}

# Call the main build script with determined parameters
$buildArgs = @()
if ($Version) { $buildArgs += $Version }
if ($Major) { $buildArgs += "-Major" }
if ($Minor) { $buildArgs += "-Minor" }
if ($Patch) { $buildArgs += "-Patch" }
if ($NoBuild) { $buildArgs += "-NoBuild" }

Write-Host ""
Write-Host "Calling build-msi.ps1..." -ForegroundColor Cyan
& .\build-msi.ps1 @buildArgs
