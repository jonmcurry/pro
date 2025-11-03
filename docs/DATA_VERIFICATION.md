# Test Data Verification Report

**Date**: November 3, 2025
**Status**: ✅ **VERIFIED - All NPIs Match**

## Summary

All facility NPIs in the EDI test files match the NPIs in facilities.csv. The data is consistent and correct.

## Facility NPIs in EDI Files

| EDI File | Facility Name | NPI | Claims Count |
|----------|---------------|-----|--------------|
| claims_ORG001-R1-F1.edi | North Region Medical Center | 7319437180 | 7 |
| claims_ORG001-R1-F2.edi | North Region Clinic | 4967786886 | 10 |
| claims_ORG001-R2-F1.edi | South Region Medical Center | 9057397983 | 10 |
| claims_ORG001-R2-F2.edi | South Region Clinic | 1292573024 | 7 |
| claims_ORG002-R1-F1.edi | North Region Medical Center | 1997033985 | 9 |
| claims_ORG002-R1-F2.edi | North Region Clinic | 7531225741 | 6 |
| claims_ORG002-R2-F1.edi | South Region Medical Center | 9782808653 | 8 |
| claims_ORG002-R2-F2.edi | South Region Clinic | 8327420318 | 6 |

## Facility NPIs in facilities.csv

| Row | Facility Code | Facility Name | NPI | Org | Region |
|-----|---------------|---------------|-----|-----|--------|
| 2 | ORG001-R1-F1 | North Region Medical Center | 7319437180 | ORG001 | ORG001-R1 |
| 3 | ORG001-R1-F2 | North Region Clinic | 4967786886 | ORG001 | ORG001-R1 |
| 4 | ORG001-R2-F1 | South Region Medical Center | 9057397983 | ORG001 | ORG001-R2 |
| 5 | ORG001-R2-F2 | South Region Clinic | 1292573024 | ORG001 | ORG001-R2 |
| 6 | ORG002-R1-F1 | North Region Medical Center | 1997033985 | ORG002 | ORG002-R1 |
| 7 | ORG002-R1-F2 | North Region Clinic | 7531225741 | ORG002 | ORG002-R1 |
| 8 | ORG002-R2-F1 | South Region Medical Center | 9782808653 | ORG002 | ORG002-R2 |
| 9 | ORG002-R2-F2 | South Region Clinic | 8327420318 | ORG002 | ORG002-R2 |

## Comparison Result

✅ **Perfect Match**: All 8 NPIs from EDI files exist in facilities.csv

```
EDI NPIs:
1292573024
1997033985
4967786886
7319437180
7531225741
8327420318
9057397983
9782808653

CSV NPIs:
1292573024
1997033985
4967786886
7319437180
7531225741
8327420318
9057397983
9782808653

Diff: (none)
```

## Root Cause of "Facility not found:" Error

Since the test data files are correct, the error "Facility not found:" means the CSV data has NOT been successfully loaded into the database.

## Verification Steps

### Step 1: Check if facilities exist in database

```sql
-- Connect to database
psql -h localhost -U postgres -d professional_smart

-- Check facility count
SELECT COUNT(*) FROM claims.facility;
-- Expected: 8

-- Check if specific NPIs exist
SELECT facility_code, facility_name, npi
FROM claims.facility
WHERE npi IN (
    '7319437180',
    '4967786886',
    '9057397983',
    '1292573024',
    '1997033985',
    '7531225741',
    '9782808653',
    '8327420318'
)
ORDER BY npi;
-- Expected: 8 rows
```

### Step 2: If no facilities found, check if tables exist

```sql
-- Check if facility table exists
SELECT table_name FROM information_schema.tables
WHERE table_schema = 'claims' AND table_name = 'facility';
-- Expected: 1 row

-- Check table structure
\d claims.facility

-- Check all claims schema tables
SELECT table_name FROM information_schema.tables
WHERE table_schema = 'claims'
ORDER BY table_name;
```

### Step 3: If tables don't exist, run migrations

```powershell
cd "C:\Program Files\Professional SMART\bin"
.\pro-upgrade.exe apply-migrations
```

### Step 4: Load test data

**Using GUI Data Loader**:
```powershell
& "C:\Program Files\Professional SMART\bin\pro-data-loader-gui.exe"
# Select folder: C:\Users\jonmc\dev\pro\test_data\setup
# Click "Load Data"
```

**Using CLI Data Loader**:
```powershell
cd "C:\Program Files\Professional SMART\bin"
$env:PGPASSWORD="your_password"
.\pro-data-loader.exe --directory "C:\Users\jonmc\dev\pro\test_data\setup"
```

## Common Issues

### Issue 1: Data Loader Fails with "relation does not exist"

**Cause**: Database migrations haven't been run

**Solution**: Run migrations first
```powershell
cd "C:\Program Files\Professional SMART\bin"
.\pro-upgrade.exe apply-migrations
```

### Issue 2: Data Loader Succeeds but Facilities Still Not Found

**Cause**: Data loaded to wrong database

**Solution**: Verify database name
```sql
-- Check current database
SELECT current_database();
-- Should be: professional_smart

-- Check if data exists
SELECT COUNT(*) FROM claims.facility;
```

### Issue 3: Foreign Key Constraint Violations

**Cause**: Loading data in wrong order

**Solution**: Load in this order:
1. organizations.csv
2. regions.csv
3. facilities.csv
4. providers.csv

The data loader should handle this automatically, but manual loads must follow this order.

## Testing After Data Load

After loading facilities, test with a simple query:

```sql
-- Test facility lookup (same query the processor uses)
SELECT facility_id, organization_id, region_id
FROM claims.facility
WHERE facility_code = '7319437180' OR npi = '7319437180';
-- Expected: 1 row with facility ORG001-R1-F1
```

If this returns 1 row, then processing EDI files should work.

## Expected Database State

After loading all test data:

```sql
-- Organizations
SELECT COUNT(*) FROM claims.organization;  -- Expected: 2

-- Regions
SELECT COUNT(*) FROM claims.region;  -- Expected: 4

-- Facilities
SELECT COUNT(*) FROM claims.facility;  -- Expected: 8

-- Providers
SELECT COUNT(*) FROM claims.provider;  -- Expected: varies
```

## Processing Test

After verifying data is loaded, process a test file:

```powershell
# Copy EDI file to input folder
Copy-Item "C:\Users\jonmc\dev\pro\test_data\claims_ORG001-R1-F1.edi" `
          -Destination "C:\ProgramData\Professional SMART\data\input\"

# Wait 5-10 seconds for service to process

# Check results
$env:PGPASSWORD="your_password"
psql -h localhost -U postgres -d professional_smart -c "
SELECT
    processing_status,
    COUNT(*),
    error_message
FROM staging.raw_claims
WHERE batch_id = (SELECT MAX(batch_id) FROM staging.import_batch)
GROUP BY processing_status, error_message;
"
```

**Expected**: 7 claims with `processing_status = 'COMPLETED'` and `error_message = NULL`

If you still see "Facility not found:", the facilities are not in the database despite loading the CSV.

## Next Steps if Issue Persists

1. **Verify data loader ran without errors**: Check data loader output/logs
2. **Verify correct database**: Make sure service and data loader use same database
3. **Check .env file**: Verify DB_NAME is `professional_smart`
4. **Manual verification**: Run the SQL queries above to confirm data presence
5. **Check service logs**: Look for connection issues in service logs

## Conclusion

The test data files (CSV and EDI) are perfectly aligned. All 8 facility NPIs match. The "Facility not found:" error indicates the CSV data has not been successfully loaded into the database, not a data mismatch issue.
