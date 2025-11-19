@echo off
set DB_HOST=localhost
set DB_PORT=5432
set DB_NAME=professional_smart
set DB_USER=postgres
set DB_PASSWORD=ClearToFly1
set INSTALLER_VERSION=2.7.0.0
echo Testing INSTALLER_VERSION=%INSTALLER_VERSION%
"C:\Users\jonmc\dev\pro\target\release\pro-upgrade.exe" apply-migrations
