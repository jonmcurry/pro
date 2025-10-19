#!/bin/bash
# Automatic MSI Rebuild Script
# Triggers Windows PowerShell build from WSL
# Per CLAUDE.md Rule 10: "Rebuild the installer after every change"

set -e  # Exit on error

echo "================================================"
echo "Automatic MSI Rebuild (WSL -> Windows)"
echo "================================================"
echo ""

# Convert WSL path to Windows path
PROJECT_ROOT="/mnt/c/Users/jonmc/dev/pro"
WIN_PROJECT_ROOT="C:\\Users\\jonmc\\dev\\pro"

echo "Project: $PROJECT_ROOT"
echo ""

# Step 1: Build release binaries
echo "================================================"
echo "Step 1: Building Release Binaries..."
echo "================================================"
echo ""

powershell.exe -ExecutionPolicy Bypass -File "$WIN_PROJECT_ROOT\\build-release.ps1"

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Release build failed!"
    exit 1
fi

echo ""
echo "Release binaries built successfully"
echo ""

# Step 2: Build MSI installer
echo "================================================"
echo "Step 2: Building MSI Installer..."
echo "================================================"
echo ""

cmd.exe /c "cd /d $WIN_PROJECT_ROOT\\installer && build.bat"

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: MSI build failed!"
    exit 1
fi

echo ""
echo "================================================"
echo "MSI Rebuild Complete!"
echo "================================================"
echo ""

# Show MSI details
if [ -f "$PROJECT_ROOT/installer/ProfessionalSMART.msi" ]; then
    MSI_SIZE=$(du -h "$PROJECT_ROOT/installer/ProfessionalSMART.msi" | cut -f1)
    MSI_DATE=$(stat -c %y "$PROJECT_ROOT/installer/ProfessionalSMART.msi" | cut -d'.' -f1)
    echo "MSI File: ProfessionalSMART.msi"
    echo "Size: $MSI_SIZE"
    echo "Modified: $MSI_DATE"
    echo ""
    echo "Location: C:\\Users\\jonmc\\dev\\pro\\installer\\ProfessionalSMART.msi"
else
    echo "WARNING: MSI file not found!"
fi

echo ""
