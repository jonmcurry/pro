@echo off
REM build-simple.bat - Simple wrapper to call PowerShell build script
REM Usage:
REM   build-simple.bat              - Auto-increment build number
REM   build-simple.bat -Major       - Increment major version (2.0.0.0)
REM   build-simple.bat -Minor       - Increment minor version (1.2.0.0)
REM   build-simple.bat -Patch       - Increment patch version (1.1.1.0)
REM   build-simple.bat 1.5.0.0      - Set specific version

echo Professional SMART MSI Builder
echo.

REM Check if PowerShell is available
where powershell >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: PowerShell not found in PATH
    echo Please ensure PowerShell is installed
    exit /b 1
)

REM Run PowerShell build script with arguments
if "%~1"=="" (
    powershell -ExecutionPolicy Bypass -File .\build-msi.ps1
) else (
    powershell -ExecutionPolicy Bypass -File .\build-msi.ps1 %*
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo BUILD FAILED
    exit /b %ERRORLEVEL%
)

echo.
echo Build complete! See ProfessionalSMART.msi
pause
