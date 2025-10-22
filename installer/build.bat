@echo off
REM Professional SMART Installer Build Script
REM This script builds the Windows installer using WiX Toolset

setlocal

echo ================================================
echo Building Professional SMART Installer
echo ================================================
echo.

REM Check if WiX is installed (try PATH first, then common locations)
where candle.exe >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo WiX Toolset detected in PATH
    set "CANDLE=candle.exe"
    set "LIGHT=light.exe"
    goto :wix_found
)

REM Try WiX v3.14
if exist "C:\Program Files (x86)\WiX Toolset v3.14\bin\candle.exe" (
    echo WiX Toolset v3.14 detected
    set "CANDLE=C:\Program Files (x86)\WiX Toolset v3.14\bin\candle.exe"
    set "LIGHT=C:\Program Files (x86)\WiX Toolset v3.14\bin\light.exe"
    goto :wix_found
)

REM Try WiX v3.11
if exist "C:\Program Files (x86)\WiX Toolset v3.11\bin\candle.exe" (
    echo WiX Toolset v3.11 detected
    set "CANDLE=C:\Program Files (x86)\WiX Toolset v3.11\bin\candle.exe"
    set "LIGHT=C:\Program Files (x86)\WiX Toolset v3.11\bin\light.exe"
    goto :wix_found
)

echo ERROR: WiX Toolset not found in PATH or common locations
echo Please install WiX Toolset from https://wixtoolset.org/
echo Or add WiX bin directory to PATH
exit /b 1

:wix_found
echo.

REM Set solution directory (parent of installer directory)
set SOLUTION_DIR=%~dp0..

echo Solution directory: %SOLUTION_DIR%
echo.

REM Check if release binaries exist
if not exist "%SOLUTION_DIR%\target\release\pro-service.exe" (
    echo ERROR: pro-service.exe not found
    echo Please build the project first: cargo build --release
    exit /b 1
)

if not exist "%SOLUTION_DIR%\target\release\pro-setup.exe" (
    echo ERROR: pro-setup.exe not found
    echo Please build the project first: cargo build --release
    exit /b 1
)

echo Release binaries found
echo.

REM Create .env.example if it doesn't exist
if not exist "%SOLUTION_DIR%\.env.example" (
    echo Creating .env.example template...
    copy /Y "%SOLUTION_DIR%\docs\CONFIGURATION.md" "%SOLUTION_DIR%\.env.example" >nul
)

REM Create LICENSE file if it doesn't exist
if not exist "%SOLUTION_DIR%\LICENSE" (
    echo Creating LICENSE file...
    echo MIT License > "%SOLUTION_DIR%\LICENSE"
)

REM Create icon if it doesn't exist
if not exist "%~dp0icon.ico" (
    echo WARNING: icon.ico not found, using default
    REM In production, you would create or copy an actual icon file
)

echo.
echo ================================================
echo Auto-generating migration files list...
echo ================================================
echo.

REM Auto-generate MigrationsFragment.wxs from migrations folder using Heat.exe
REM This scans the migrations folder and creates WiX components for all .sql files
REM This way you never have to manually list migration files!

REM Find Heat.exe (should be in same directory as candle/light)
set "HEAT=%CANDLE:candle.exe=heat.exe%"

if not exist "%HEAT%" (
    echo ERROR: heat.exe not found at %HEAT%
    exit /b 1
)

echo Running Heat.exe to harvest migrations folder...
"%HEAT%" dir "%SOLUTION_DIR%\migrations" -cg MigrationsComponentGroup -dr MigrationsFolder -var var.SolutionDir -gg -sfrag -srd -platform x64 -out "%~dp0MigrationsFragment_temp.wxs"
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to run Heat.exe
    exit /b 1
)

echo Processing Heat.exe output to fix paths and add Win64 attribute...

REM Use PowerShell script to fix the generated file
powershell -ExecutionPolicy Bypass -File "%~dp0fix_migrations_fragment.ps1" -InputFile "%~dp0MigrationsFragment_temp.wxs" -OutputFile "%~dp0MigrationsFragment.wxs"
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: PowerShell processing failed, using temp file as-is
    copy /Y "%~dp0MigrationsFragment_temp.wxs" "%~dp0MigrationsFragment.wxs" >nul
)

del "%~dp0MigrationsFragment_temp.wxs" 2>nul

echo MigrationsFragment.wxs generated successfully
echo.

echo ================================================
echo Compiling WiX source files...
echo ================================================
echo.

REM Compile Product.wxs
"%CANDLE%" -dSolutionDir="%SOLUTION_DIR%\\" -out "%~dp0Product.wixobj" "%~dp0Product.wxs"
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to compile Product.wxs
    exit /b 1
)

echo Product.wxs compiled successfully

REM Compile DatabaseConfigDlg.wxs
"%CANDLE%" -dSolutionDir="%SOLUTION_DIR%\\" -out "%~dp0DatabaseConfigDlg.wixobj" "%~dp0DatabaseConfigDlg.wxs"
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to compile DatabaseConfigDlg.wxs
    exit /b 1
)

echo DatabaseConfigDlg.wxs compiled successfully

REM Compile PrerequisiteDlg.wxs
"%CANDLE%" -dSolutionDir="%SOLUTION_DIR%\\" -out "%~dp0PrerequisiteDlg.wixobj" "%~dp0PrerequisiteDlg.wxs"
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to compile PrerequisiteDlg.wxs
    exit /b 1
)

echo PrerequisiteDlg.wxs compiled successfully

REM Compile MigrationsFragment.wxs
"%CANDLE%" -dSolutionDir="%SOLUTION_DIR%\\" -out "%~dp0MigrationsFragment.wixobj" "%~dp0MigrationsFragment.wxs"
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to compile MigrationsFragment.wxs
    exit /b 1
)

echo MigrationsFragment.wxs compiled successfully
echo.

echo ================================================
echo Linking installer...
echo ================================================
echo.

REM Link to create MSI
"%LIGHT%" -ext WixUIExtension -out "%~dp0ProfessionalSMART.msi" "%~dp0Product.wixobj" "%~dp0DatabaseConfigDlg.wixobj" "%~dp0PrerequisiteDlg.wixobj" "%~dp0MigrationsFragment.wixobj"
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to link installer
    exit /b 1
)

echo.
echo ================================================
echo Installer build complete!
echo ================================================
echo.
echo Output: %~dp0ProfessionalSMART.msi
echo.
echo To install:
echo   msiexec /i ProfessionalSMART.msi
echo.
echo To install silently:
echo   msiexec /i ProfessionalSMART.msi /quiet /qn
echo.

endlocal
