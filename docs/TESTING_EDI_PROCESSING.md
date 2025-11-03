# Testing EDI 837p Processing - Complete Guide

**Date**: November 3, 2025
**Version**: 1.5.12.0

## Problem: "Facility not found:"

When processing EDI files from `test_data/`, the error "Row 6: Facility not found: " occurs because the facility lookup is failing.

### Root Cause

The test data from `test_data/setup/*.csv` has NOT been loaded into the database. The EDI parser correctly extracts the facility NPI (7319437180), but the facility doesn't exist in the database yet.

### Solution

Load the test data BEFORE processing EDI files.

## Complete Testing Procedure

### Prerequisites

1. ✅ PostgreSQL installed and running
2. ✅ Professional SMART v1.5.12.0 installed
3. ✅ Database created (either via installer or manually)
4. ✅ Service running

### Step 1: Verify Database Exists

```powershell
# Check if database exists
$env:PGPASSWORD="your_password"
psql -h localhost -U postgres -d professional_smart -c "SELECT version();"
```

If database doesn't exist, create it:

```powershell
# Set password
$env:PGPASSWORD="your_password"

# Create database
psql -h localhost -U postgres -c "CREATE DATABASE professional_smart;"

# Run migrations
cd "C:\Program Files\Professional SMART\bin"
.\pro-upgrade.exe apply-migrations
```

### Step 2: Load Test Master Data

The test data must be loaded in this **specific order** due to foreign key constraints:

1. Organizations
2. Regions
3. Facilities
4. Providers

**Option A: Using GUI Data Loader (Recommended)**

```powershell
# Navigate to test data
cd C:\Users\jonmc\dev\pro\test_data\setup

# Launch GUI data loader
& "C:\Program Files\Professional SMART\bin\pro-data-loader-gui.exe"

# In the GUI:
# 1. Select the setup folder: C:\Users\jonmc\dev\pro\test_data\setup
# 2. Click "Load Data"
# 3. Wait for success message
```

**Option B: Using CLI Data Loader**

```powershell
cd "C:\Program Files\Professional SMART\bin"

# Set database connection
$env:PGPASSWORD="your_password"
$env:DB_HOST="localhost"
$env:DB_PORT="5432"
$env:DB_NAME="professional_smart"
$env:DB_USER="postgres"

# Load test data
.\pro-data-loader.exe --directory "C:\Users\jonmc\dev\pro\test_data\setup"
```

**Option C: Manual SQL Loading**

```sql
-- Connect to database
\c professional_smart

-- Load organizations
\copy claims.organization FROM 'C:\Users\jonmc\dev\pro\test_data\setup\organizations.csv' WITH (FORMAT csv, HEADER true);

-- Load regions
\copy claims.region FROM 'C:\Users\jonmc\dev\pro\test_data\setup\regions.csv' WITH (FORMAT csv, HEADER true);

-- Load facilities
\copy claims.facility FROM 'C:\Users\jonmc\dev\pro\test_data\setup\facilities.csv' WITH (FORMAT csv, HEADER true);

-- Load providers
\copy claims.provider FROM 'C:\Users\jonmc\dev\pro\test_data\setup\providers.csv' WITH (FORMAT csv, HEADER true);
```

### Step 3: Verify Data Loaded

```sql
-- Check organizations
SELECT organization_code, organization_name FROM claims.organization;
-- Expected: ORG001, ORG002

-- Check regions
SELECT region_code, region_name FROM claims.region;
-- Expected: ORG001-R1, ORG001-R2, ORG002-R1, ORG002-R2

-- Check facilities
SELECT facility_code, facility_name, npi FROM claims.facility;
-- Expected: 8 facilities including NPI 7319437180

-- Verify the specific facility from EDI file
SELECT facility_id, facility_code, facility_name, npi
FROM claims.facility
WHERE npi = '7319437180';
-- Expected: ORG001-R1-F1, North Region Medical Center, 7319437180
```

Expected output:
```
           facility_id            | facility_code |       facility_name       |    npi
--------------------------------------+---------------+---------------------------+------------
 2457bb67-b073-461a-bc92-4e27e67bdf3d | ORG001-R1-F1  | North Region Medical Center | 7319437180
```

### Step 4: Process EDI Files

Now that master data is loaded, process the EDI files.

**Option A: Automatic Processing (Service Watches Folder)**

```powershell
# Copy EDI file to input folder
Copy-Item "C:\Users\jonmc\dev\pro\test_data\claims_ORG001-R1-F1.edi" `
          -Destination "C:\ProgramData\Professional SMART\data\input\"

# Service will automatically detect and process
# Check logs at: C:\ProgramData\Professional SMART\logs\
```

**Option B: Manual SQL Import for Testing**

```sql
-- Insert file into import queue
INSERT INTO staging.import_queue (
    file_path,
    file_name,
    file_format,
    queue_status,
    created_at
) VALUES (
    'C:\Users\jonmc\dev\pro\test_data\claims_ORG001-R1-F1.edi',
    'claims_ORG001-R1-F1.edi',
    '837P',
    'PENDING',
    NOW()
);

-- Service will pick it up automatically
```

### Step 5: Monitor Processing

**Check Import Queue:**
```sql
SELECT
    queue_id,
    file_name,
    file_format,
    queue_status,
    error_message,
    created_at,
    processed_at
FROM staging.import_queue
WHERE file_name LIKE '%ORG001%'
ORDER BY created_at DESC;
```

**Check Staging Claims:**
```sql
SELECT
    raw_claim_id,
    encounter_fields->>'patient_control_number' as claim_id,
    encounter_fields->>'facility_npi' as facility_npi,
    encounter_fields->>'subscriber_last_name' as subscriber,
    service_line_fields->>'service_line_1_procedure_code' as proc_code,
    processing_status,
    error_message
FROM staging.raw_claims
WHERE batch_id IN (
    SELECT batch_id FROM staging.import_batch WHERE file_name LIKE '%ORG001%'
)
ORDER BY row_number;
```

Expected after v1.5.12.0 parser:
```
 claim_id          | facility_npi | subscriber | proc_code | processing_status | error_message
-------------------+--------------+------------+-----------+-------------------+---------------
 ORG001-R1-F1-0002 | 7319437180   | Smith      | 99214     | COMPLETED         |
 ORG001-R1-F1-0003 | 7319437180   | Johnson    | 99213     | COMPLETED         |
 ...
```

**Check Final Encounters:**
```sql
SELECT
    e.encounter_id,
    e.patient_control_number,
    e.facility_id,
    f.facility_code,
    f.npi as facility_npi,
    e.subscriber_last_name,
    e.total_charge_amount
FROM claims.encounters e
JOIN claims.facility f ON e.facility_id = f.facility_id
WHERE e.batch_id IN (
    SELECT batch_id FROM staging.import_batch WHERE file_name LIKE '%ORG001%'
)
ORDER BY e.patient_control_number;
```

Expected:
```
 patient_control_number | facility_code |    npi     | subscriber_last_name | total_charge_amount
------------------------+---------------+------------+----------------------+---------------------
 ORG001-R1-F1-0002      | ORG001-R1-F1  | 7319437180 | Smith                |              125.00
 ORG001-R1-F1-0003      | ORG001-R1-F1  | 7319437180 | Johnson              |               85.00
 ...
```

**Check Service Lines:**
```sql
SELECT
    e.patient_control_number,
    sl.line_number,
    sl.procedure_code,
    sl.charge_amount,
    sl.service_units,
    sl.service_date
FROM claims.service_lines sl
JOIN claims.encounters e ON sl.encounter_id = e.encounter_id
WHERE e.batch_id IN (
    SELECT batch_id FROM staging.import_batch WHERE file_name LIKE '%ORG001%'
)
ORDER BY e.patient_control_number, sl.line_number;
```

Expected for 7 claims:
```
 patient_control_number | line_number | procedure_code | charge_amount | service_units | service_date
------------------------+-------------+----------------+---------------+---------------+--------------
 ORG001-R1-F1-0002      |           1 | 99214          |        125.00 |          1.00 | 2025-06-28
 ORG001-R1-F1-0003      |           1 | 99213          |         85.00 |          1.00 | 2025-07-14
 ...
(7 rows)
```

## Troubleshooting

### Error: "Facility not found: "

**Cause**: Facility data not loaded in database

**Solution**: Load test data from `test_data/setup/*.csv` (see Step 2)

**Verification**:
```sql
SELECT COUNT(*) FROM claims.facility WHERE npi = '7319437180';
-- Should return: 1
```

### Error: "Missing facility_code or facility_npi"

**Cause**: EDI parser not extracting facility NPI (v1.5.9.0 or earlier)

**Solution**: Upgrade to v1.5.10.0+ which includes NM1*77 segment parsing

**Verification**: Check parser version
```sql
SELECT name, value FROM claims.system_config WHERE name = 'version';
```

### Service Lines Empty

**Cause**: EDI parser not extracting service lines (v1.5.10.0 or earlier)

**Solution**: Upgrade to v1.5.11.0+ which includes LX/SV1 segment parsing

**Verification**:
```sql
SELECT
    service_line_fields->>'service_line_1_procedure_code' as proc_code
FROM staging.raw_claims
WHERE batch_id = (SELECT MAX(batch_id) FROM staging.import_batch WHERE file_format = '837P');
-- Should show: 99214, 99213, etc. (not empty)
```

### Service Not Starting

**Cause**: Service installation issue (v1.5.11.0 or earlier)

**Solution**: Upgrade to v1.5.12.0 which includes proper ServiceControl configuration

**Verification**:
```powershell
sc query ProfessionalSMART
# Should show: STATE: 4 RUNNING
```

## Test Data Details

### Test File: claims_ORG001-R1-F1.edi

**Contains**: 7 professional claims for organization ORG001, region R1, facility F1

**Claims**:
| # | Patient | NPI | Procedure | Charge | DOS | Dx |
|---|---------|-----|-----------|--------|-----|-----|
| 1 | Smith, John | 7319437180 | 99214 | $125.00 | 2025-06-28 | F41.9 |
| 2 | Johnson, Mary | 7319437180 | 99213 | $85.00 | 2025-07-14 | E11.9 |
| 3 | Williams, James | 7319437180 | 99215 | $155.00 | 2025-08-02 | I10 |
| 4 | Brown, Patricia | 7319437180 | 99214 | $125.00 | 2025-08-19 | Z00.00 |
| 5 | Jones, Michael | 7319437180 | 99213 | $85.00 | 2025-09-05 | M79.3 |
| 6 | Garcia, Linda | 7319437180 | 99215 | $155.00 | 2025-09-23 | J06.9 |
| 7 | Martinez, David | 7319437180 | 99214 | $125.00 | 2025-10-10 | K21.9 |

**Total Charges**: $855.00

### Master Data Files

**organizations.csv**: 2 organizations (ORG001, ORG002)
**regions.csv**: 4 regions (2 per organization)
**facilities.csv**: 8 facilities (2 per region)
**providers.csv**: Provider data

## Complete End-to-End Test

```powershell
# 1. Load master data
cd "C:\Program Files\Professional SMART\bin"
.\pro-data-loader-gui.exe
# Select: C:\Users\jonmc\dev\pro\test_data\setup

# 2. Verify facility exists
$env:PGPASSWORD="your_password"
psql -h localhost -U postgres -d professional_smart -c "SELECT facility_code, npi FROM claims.facility WHERE npi = '7319437180';"

# 3. Process EDI file
Copy-Item "C:\Users\jonmc\dev\pro\test_data\claims_ORG001-R1-F1.edi" `
          -Destination "C:\ProgramData\Professional SMART\data\input\"

# 4. Wait 5-10 seconds for processing

# 5. Check results
psql -h localhost -U postgres -d professional_smart -c "
SELECT
    e.patient_control_number,
    f.facility_code,
    e.subscriber_last_name,
    sl.procedure_code,
    sl.charge_amount
FROM claims.encounters e
JOIN claims.facility f ON e.facility_id = f.facility_id
JOIN claims.service_lines sl ON e.encounter_id = sl.encounter_id
WHERE e.batch_id = (SELECT MAX(batch_id) FROM staging.import_batch WHERE file_format = '837P')
ORDER BY e.patient_control_number;
"

# Expected: 7 encounters with service lines, total charges = $855.00
```

## Success Criteria

After following this guide:

- ✅ 2 organizations loaded
- ✅ 4 regions loaded
- ✅ 8 facilities loaded (including NPI 7319437180)
- ✅ EDI file ingested without errors
- ✅ 7 claims in staging.raw_claims with COMPLETED status
- ✅ 7 encounters in claims.encounters
- ✅ 7 service lines in claims.service_lines
- ✅ All procedure codes populated (99214, 99213, 99215)
- ✅ All diagnosis codes populated (F41.9, E11.9, I10, etc.)
- ✅ Total charges = $855.00

## Version Requirements

**Minimum versions for full EDI processing**:

- ✅ v1.5.9.0 or later: 3-column JSONB architecture
- ✅ v1.5.10.0 or later: NM1*77 facility parsing
- ✅ v1.5.11.0 or later: LX/SV1 service line parsing
- ✅ v1.5.12.0 or later: Service auto-start on install

**Recommended**: v1.5.12.0 (current release)
