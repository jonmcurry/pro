@echo off
echo ====================================
echo Uninstalling Old Professional SMART
echo ====================================
echo.

REM Uninstall using the old product code found in log
echo Attempting to uninstall version 1.2.0.0...
msiexec /x {D1FEA2D9-872B-4058-940D-D31FC0BFECC6} /quiet /l*v C:\temp\uninstall.log

echo.
echo Uninstall completed. Check C:\temp\uninstall.log for details.
echo.
pause
