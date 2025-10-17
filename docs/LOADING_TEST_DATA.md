# Loading Test Data Guide

Complete guide for loading test data into Professional SMART.

---

## Overview

This guide covers loading the generated test data in the correct order:

1. **Master Data** - Organizations, Regions, Facilities (via SQL/script)
2. **Claims Data** - 80,000 claims in CSV format (via file processing)

### Important: Facility is the Anchor for Claims

For claims import, **only the facility must exist** in the database. The organization and region are automatically derived from the facility lookup:

- Claims specify: `facility_code` or `facility_npi`
- System derives: `organization_id` (from facility) and `region_id` (from facility, can be NULL)

This means claims CSVs **do not need** organization_code or region_code columns.

---

## Prerequisites

- PostgreSQL database installed and running
- Database migrations completed
- `DATABASE_URL` environment variable set or in `.env` file
- Python 3.7+ installed (for loader script)
- `psycopg2` Python package installed

### Install Python Dependencies

```bash
pip install psycopg2-binary
```

Or if you have PostgreSQL development libraries:

```bash
pip install psycopg2
```

---

## Method 1: Load Master Data with Python Script (Recommended)

### Step 1: Verify Database Connection

Make sure your `.env` file has the correct DATABASE_URL:

```bash
# .env file
DATABASE_URL=postgres://pro_user:your_password@localhost:5432/professional_smart
```

### Step 2: Run the Loader Script

```bash
# From the project root directory
python3 scripts/load_master_data.py test_data
```

**Expected Output**:

```
======================================================================
Professional SMART Master Data Loader
======================================================================
Data directory: /path/to/test_data

Connecting to database...
  ✓ Connected successfully

Loading organizations from test_data/organizations.csv...
  Found 2 organizations
  ✓ Loaded: ORG001 - Regional Health System
  ✓ Loaded: ORG002 - Metropolitan Medical Group
  ✓ Successfully loaded 2 organizations

Loading regions from test_data/regions.csv...
  Found 4 regions
  ✓ Loaded: ORG001-R1 - North Region
  ✓ Loaded: ORG001-R2 - South Region
  ✓ Loaded: ORG002-R1 - North Region
  ✓ Loaded: ORG002-R2 - South Region
  ✓ Successfully loaded 4 regions

Loading facilities from test_data/facilities.csv...
  Found 8 facilities
  ✓ Loaded: ORG001-R1-F1 - North Region Medical Center
  ✓ Loaded: ORG001-R1-F2 - North Region Clinic
  ✓ Loaded: ORG001-R2-F1 - South Region Medical Center
  ✓ Loaded: ORG001-R2-F2 - South Region Clinic
  ✓ Loaded: ORG002-R1-F1 - North Region Medical Center
  ✓ Loaded: ORG002-R1-F2 - North Region Clinic
  ✓ Loaded: ORG002-R2-F1 - South Region Medical Center
  ✓ Loaded: ORG002-R2-F2 - South Region Clinic
  ✓ Successfully loaded 8 facilities

Verifying loaded data...
  Organizations in database: 2
  Regions in database: 4
  Facilities in database: 8

Organization Hierarchy:
  ORG001: Regional Health System
    - 2 regions
    - 4 facilities
  ORG002: Metropolitan Medical Group
    - 2 regions
    - 4 facilities

======================================================================
Loading Complete!
======================================================================
Summary:
  Organizations loaded: 2
  Regions loaded: 4
  Facilities loaded: 8
```

### Step 3: Verify Data in Database

```bash
# Connect to database
psql -U pro_user -d professional_smart

# Check organizations
SELECT * FROM claims.organization;

# Check regions
SELECT * FROM claims.region;

# Check facilities
SELECT * FROM claims.facility;

# View hierarchy
SELECT
    o.organization_name,
    r.region_name,
    f.facility_code,
    f.facility_name
FROM claims.organization o
LEFT JOIN claims.region r ON o.organization_id = r.organization_id
LEFT JOIN claims.facility f ON o.organization_id = f.organization_id
ORDER BY o.organization_code, r.region_code, f.facility_code;
```

---

## Method 2: Load Master Data with SQL COPY (Alternative)

If you prefer direct SQL loading:

### Step 1: Prepare SQL Script

Create a file `load_master_data.sql`:

```sql
-- Load organizations
COPY claims.organization (
    organization_id, organization_code, organization_name, tax_id, npi,
    address_line1, city, state_code, postal_code, phone, email
)
FROM '/absolute/path/to/test_data/organizations.csv'
WITH (FORMAT csv, HEADER true);

-- Load regions
COPY claims.region (
    region_id, organization_id, region_code, region_name, description
)
FROM '/absolute/path/to/test_data/regions.csv'
WITH (FORMAT csv, HEADER true);

-- Load facilities
COPY claims.facility (
    facility_id, organization_id, region_id, facility_code, facility_name,
    npi, tax_id, facility_type, address_line1, city, state_code,
    postal_code, phone, email, ehr_system
)
FROM '/absolute/path/to/test_data/facilities.csv'
WITH (FORMAT csv, HEADER true);
```

### Step 2: Run SQL Script

```bash
psql -U pro_user -d professional_smart -f load_master_data.sql
```

**Note**: This method requires:
- Absolute file paths
- PostgreSQL user must have file read permissions
- May have issues with UUID generation

---

## Loading Claims Data

Once master data is loaded, you can import claims.

### Option 1: Via Professional SMART Service (Recommended)

The service automatically processes CSV files from the input directory.

**Step 1**: Configure input directory in `.env`:

```bash
INPUT_DIRECTORY=C:\Claims\Input
```

**Step 2**: Copy claims files to input directory:

```bash
# Copy one facility for testing
cp test_data/claims_ORG001-R1-F1.csv /path/to/input/

# Or copy all facilities
cp test_data/claims_*.csv /path/to/input/
```

**Step 3**: Start the service:

```bash
professional-smart console
```

**Step 4**: Monitor progress in logs:

```bash
tail -f logs/pro-service.log
```

The service will:
1. Auto-detect Athena CSV format
2. Map headers to database fields
3. Validate data
4. Import encounters, service lines, and diagnoses
5. Move processed files to `processed/` directory

### Option 2: Direct Database Import

For faster testing, you can bulk-load claims directly:

**WARNING**: This bypasses validation and rules engine!

```sql
-- Create temporary staging table
CREATE TEMP TABLE staging_claims AS
SELECT * FROM claims.encounter LIMIT 0;

-- Load CSV
COPY staging_claims (
    patient_control_number, date_of_service_from, ...
)
FROM '/path/to/claims.csv'
WITH (FORMAT csv, HEADER true);

-- Insert into main tables with transformations
-- (This is complex - use the service instead!)
```

**Recommendation**: Use the Professional SMART service for proper data validation and processing.

---

## Verification Queries

After loading all data, verify the import:

### Check Master Data

```sql
-- Organizations
SELECT COUNT(*) FROM claims.organization;
-- Expected: 2

-- Regions
SELECT COUNT(*) FROM claims.region;
-- Expected: 4

-- Facilities
SELECT COUNT(*) FROM claims.facility;
-- Expected: 8
```

### Check Claims Data (After Import)

```sql
-- Total encounters
SELECT COUNT(*) FROM claims.encounter;
-- Expected: 80,000 (if all 8 facilities loaded)

-- Total service lines
SELECT COUNT(*) FROM claims.service_line;
-- Expected: ~120,000-200,000 (1-5 per claim)

-- Total diagnoses
SELECT COUNT(*) FROM claims.encounter_diagnosis;
-- Expected: ~120,000-320,000 (1-4 per claim)

-- Claims by facility
SELECT
    f.facility_code,
    f.facility_name,
    COUNT(*) as claim_count
FROM claims.encounter e
JOIN claims.facility f ON e.facility_id = f.facility_id
GROUP BY f.facility_code, f.facility_name
ORDER BY f.facility_code;
-- Expected: ~10,000 claims per facility
```

### Check Data Quality

```sql
-- Verify all claims have facilities
SELECT COUNT(*) FROM claims.encounter WHERE facility_id IS NULL;
-- Expected: 0

-- Verify service lines linked correctly
SELECT COUNT(*) FROM claims.service_line sl
WHERE NOT EXISTS (
    SELECT 1 FROM claims.encounter e WHERE e.encounter_id = sl.encounter_id
);
-- Expected: 0

-- Check for orphaned diagnoses
SELECT COUNT(*) FROM claims.encounter_diagnosis ed
WHERE NOT EXISTS (
    SELECT 1 FROM claims.encounter e WHERE e.encounter_id = ed.encounter_id
);
-- Expected: 0
```

---

## Troubleshooting

### Error: "psycopg2 not found"

```bash
pip install psycopg2-binary
```

### Error: "DATABASE_URL not set"

Make sure your `.env` file exists and contains:

```
DATABASE_URL=postgres://username:password@localhost:5432/database_name
```

### Error: "Permission denied" for CSV files

- Make sure CSV files exist in the specified directory
- Check file permissions: `chmod 644 test_data/*.csv`
- For SQL COPY: Use absolute paths

### Error: "Facility not found" during claims import

This means master data wasn't loaded first, or the facility_code/facility_npi in the claims data doesn't match any existing facility.

**Solution:**
```bash
# Load master data first
python3 scripts/load_master_data.py test_data

# Verify facilities exist
psql -U pro_user -d professional_smart -c "SELECT facility_code, facility_name FROM claims.facility;"
```

**Remember:** Claims only need the facility to exist. Organization and region are auto-derived from the facility record.

### Error: "Duplicate key violation"

Master data already exists. Options:

1. **Skip duplicates** (loader script handles this automatically with `ON CONFLICT`)
2. **Delete existing data**:

```sql
-- CAUTION: Deletes all data!
TRUNCATE TABLE claims.facility CASCADE;
TRUNCATE TABLE claims.region CASCADE;
TRUNCATE TABLE claims.organization CASCADE;
```

3. **Use different organization codes** in CSV files

---

## Performance Tips

### For Large Datasets

If loading many claims files:

1. **Increase batch size** in configuration:

```env
BATCH_SIZE=5000
```

2. **Use multiple workers**:

```env
MAX_WORKERS=8
```

3. **Disable rules engine during import** (re-run later):

```env
ENABLE_RULES_ENGINE=false
```

4. **Monitor database connections**:

```sql
SELECT count(*) FROM pg_stat_activity;
```

### For Testing Individual Facilities

Load one facility at a time:

```bash
# Load just one facility's claims
cp test_data/claims_ORG001-R1-F1.csv input/
```

This allows faster iteration during testing.

---

## Clean Up Test Data

### Remove All Test Data from Database

```sql
-- Remove all claims (cascades to service_lines and diagnoses)
TRUNCATE TABLE claims.encounter CASCADE;

-- Remove import batches
TRUNCATE TABLE staging.import_batch CASCADE;

-- Remove master data
TRUNCATE TABLE claims.facility CASCADE;
TRUNCATE TABLE claims.region CASCADE;
TRUNCATE TABLE claims.organization CASCADE;

-- Reset sequences if needed
-- (UUIDs don't use sequences, so this is not needed)
```

### Remove Files

```bash
# Remove generated test data
rm -rf test_data/

# Remove processed files
rm -rf data/processed/*

# Remove error files
rm -rf data/error/*
```

---

## Regenerating Test Data

To generate fresh test data with different parameters:

```bash
# Edit the script to change parameters
nano scripts/generate_test_data.py

# Modify the generate_all() call:
generator.generate_all(
    org_count=3,              # Change from 2
    regions_per_org=3,        # Change from 2
    facilities_per_region=3,  # Change from 2
    providers_count=100,      # Change from 50
    claims_per_facility=5000  # Change from 10,000
)

# Generate new data
python3 scripts/generate_test_data.py new_test_data

# Load new data
python3 scripts/load_master_data.py new_test_data
```

---

## Next Steps

After loading test data:

1. **Test Dashboard Analytics**
   ```sql
   SELECT * FROM claims.v_management_overview;
   ```

2. **Test REST API**
   ```bash
   curl http://localhost:8080/api/v1/dashboard/management-overview
   ```

3. **Run Rules Engine**
   ```sql
   -- Check for flags
   SELECT COUNT(*) FROM claims.service_line_flag;
   ```

4. **View Import History**
   ```sql
   SELECT * FROM staging.import_batch ORDER BY created_at DESC;
   ```

---

## Summary

**Loading Master Data**:
```bash
python3 scripts/load_master_data.py test_data
```

**Loading Claims Data**:
```bash
cp test_data/claims_*.csv /path/to/input/
professional-smart console
```

**Verification**:
```sql
SELECT COUNT(*) FROM claims.organization;  -- 2
SELECT COUNT(*) FROM claims.region;        -- 4
SELECT COUNT(*) FROM claims.facility;      -- 8
SELECT COUNT(*) FROM claims.encounter;     -- 80,000
```

---

## See Also

- [Test Data README](../test_data/README.md) - Test data documentation
- [CSV Mapping Guide](CSV_MAPPING_GUIDE.md) - CSV header mapping
- [Configuration Guide](CONFIGURATION.md) - System configuration
- [API Documentation](../API_DOCUMENTATION.md) - REST API reference
