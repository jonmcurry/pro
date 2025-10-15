# Fresh Server Installation Guide

This guide explains how to install Professional SMART on a fresh Windows server using the MSI installer.

## Prerequisites

Before running the MSI installer, ensure the following is installed on the server:

### 1. PostgreSQL 14 or Later

**Download:** https://www.postgresql.org/download/windows/

**Installation:**
1. Run the PostgreSQL installer as Administrator
2. Accept the default installation directory
3. Set a strong password for the `postgres` superuser **and remember it**
4. Keep the default port: 5432
5. Complete the installation

**Verify Installation:**
```cmd
psql --version
```

Expected output: `psql (PostgreSQL) 14.x` or later

**Verify Service is Running:**
```cmd
sc query postgresql-x64-14
```

Expected: Status should show `RUNNING`

### 2. Disk Space

Ensure at least 500 MB free space on C:\ drive (the installer checks this)

## Installation Process

### Step 1: Copy the MSI File

Copy `ProfessionalSMART.msi` to the server (any location is fine)

### Step 2: Run the Installer

**Interactive Installation:**
```cmd
msiexec /i ProfessionalSMART.msi
```

**Silent Installation:**
```cmd
msiexec /i ProfessionalSMART.msi /quiet /qn
```

### Step 3: Follow the Installation Wizard

1. **Welcome Screen** - Click Next
2. **Prerequisite Check** - The installer will verify:
   - PostgreSQL is installed
   - PostgreSQL service is running
   - Sufficient disk space is available
3. **Feature Selection** - Accept defaults or customize
4. **Database Configuration** - Enter your database settings:
   - Host: `localhost` (or IP address if remote)
   - Port: `5432` (default PostgreSQL port)
   - Database Name: `professional_smart` (or your preferred name)
   - Username: `postgres` (or another PostgreSQL user)
   - Password: The password you set during PostgreSQL installation

### Step 4: Automatic Database Setup

The installer will automatically:
1. Validate the PostgreSQL credentials
2. Create the database if it doesn't exist
3. Run all migration scripts to create the schema
4. Create the Windows service
5. Create configuration files

## Post-Installation Verification

### Verify Service Installation

```cmd
sc query ProfessionalSMART
```

Expected: Service should exist (may be stopped initially)

### Verify Database Creation

```cmd
psql -U postgres -d professional_smart -c "\dn"
```

Expected: You should see the schemas: `claims`, `ml`, `staging`

### Verify Configuration File

Check that the config file was created:
```cmd
type "C:\ProgramData\Professional SMART\config\.env"
```

Expected: File should contain database connection settings

## Common Issues

### Issue: "PostgreSQL authentication failed"

**Cause:** Wrong password or user doesn't exist

**Solution:**
1. Verify you're using the correct PostgreSQL password
2. Test the password manually:
   ```cmd
   psql -U postgres -d postgres
   ```
3. If the user doesn't exist, create it:
   ```sql
   CREATE USER your_user WITH PASSWORD 'your_password';
   ALTER USER your_user WITH SUPERUSER;
   ```

### Issue: "Migrations directory not found"

**Cause:** Installation path issue

**Solution:**
1. Check if migrations were copied:
   ```cmd
   dir "C:\Program Files\Professional SMART\migrations"
   ```
2. If missing, manually copy migrations from the source

### Issue: "Migration failed"

**Cause:** Database permissions or syntax errors

**Solution:**
1. Check the MSI installation log:
   ```cmd
   msiexec /i ProfessionalSMART.msi /l*v install.log
   ```
2. Look for "CreateDatabase:" entries to see which migration failed
3. Run the failed migration manually:
   ```cmd
   psql -U postgres -d professional_smart -f "C:\Program Files\Professional SMART\migrations\001_create_schemas.sql"
   ```

### Issue: "Service won't start"

**Cause:** Database connection issues or missing configuration

**Solution:**
1. Check the configuration file exists:
   ```cmd
   type "C:\ProgramData\Professional SMART\config\.env"
   ```
2. Verify DATABASE_URL is correct
3. Test database connection manually:
   ```cmd
   psql -U postgres -d professional_smart -c "SELECT 1"
   ```

## Manual Database Setup (If Installer Fails)

If the automatic database setup fails, you can create it manually:

### Create Database and User

```cmd
psql -U postgres
```

```sql
CREATE DATABASE professional_smart;
CREATE USER pro_user WITH ENCRYPTED PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE professional_smart TO pro_user;
\q
```

### Run Migrations Manually

```cmd
cd "C:\Program Files\Professional SMART\migrations"

psql -U postgres -d professional_smart -f 001_create_schemas.sql
psql -U postgres -d professional_smart -f 002_create_organization_tables.sql
psql -U postgres -d professional_smart -f 003_create_provider_tables.sql
psql -U postgres -d professional_smart -f 004_create_encounter_tables.sql
psql -U postgres -d professional_smart -f 005_create_diagnosis_procedure_tables.sql
psql -U postgres -d professional_smart -f 006_create_flag_tables.sql
psql -U postgres -d professional_smart -f 007_create_staging_tables.sql
psql -U postgres -d professional_smart -f 008_create_audit_tables.sql
psql -U postgres -d professional_smart -f 009_create_rvu_tables.sql
psql -U postgres -d professional_smart -f 010_create_denial_tables.sql
psql -U postgres -d professional_smart -f 011_create_schedule_tables.sql
psql -U postgres -d professional_smart -f 012_create_ml_tables.sql
psql -U postgres -d professional_smart -f 013_create_dashboard_views.sql
psql -U postgres -d professional_smart -f 014_create_utility_functions.sql
psql -U postgres -d professional_smart -f 015_create_fifo_queue.sql
psql -U postgres -d professional_smart -f 016_phase5_performance_indexes.sql
psql -U postgres -d professional_smart -f 017_streaming_progress_tracking.sql
psql -U postgres -d professional_smart -f 018_phase6_strategic_indexes.sql
psql -U postgres -d professional_smart -f 019_phase6_materialized_views.sql
```

### Update Configuration File

Create or edit: `C:\ProgramData\Professional SMART\config\.env`

```env
DATABASE_URL=postgres://postgres:your_password@localhost:5432/professional_smart
DB_HOST=localhost
DB_PORT=5432
DB_NAME=professional_smart
DB_USER=postgres
DB_PASSWORD=your_password

LOG_LEVEL=info
RUST_LOG=info

INPUT_DIR=C:\Program Files\Professional SMART\data\input
PROCESSED_DIR=C:\Program Files\Professional SMART\data\processed
ERROR_DIR=C:\Program Files\Professional SMART\data\error

STREAMING_ENABLE_WEBSOCKET=true
WEBSOCKET_HOST=127.0.0.1:8080

WORKER_THREADS=4
BATCH_SIZE=100
```

## Viewing Installation Logs

To see detailed logs during installation:

```cmd
msiexec /i ProfessionalSMART.msi /l*v install.log
```

Then open `install.log` in a text editor and search for:
- "Professional SMART Installer:" - All custom action logs
- "CreateDatabase:" - Database creation logs
- "WriteConfig:" - Configuration file creation logs

## Starting the Service

After successful installation:

```cmd
sc start ProfessionalSMART
```

Or use the Services control panel (services.msc)

## Uninstallation

To uninstall:

```cmd
msiexec /x ProfessionalSMART.msi
```

Or use "Programs and Features" in Control Panel

Note: The database is NOT automatically deleted during uninstallation. To remove it:

```cmd
psql -U postgres -c "DROP DATABASE professional_smart;"
```
