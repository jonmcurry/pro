# Professional SMART Master Data Loader

Standalone utility for loading master data (organizations, regions, facilities, providers) into the Professional SMART database from CSV files.

## Features

- Reads database configuration from installed `.env` file
- Validates all data before import
- Uses database transactions (all-or-nothing import)
- Supports upsert (updates existing records)
- Generates CSV templates
- Clear error messages with row numbers

## Usage

### Generate CSV Templates

```bash
pro-data-loader generate-templates ./templates
```

This creates template CSV files in the specified directory:
- `organizations.csv`
- `regions.csv`
- `facilities.csv`
- `providers.csv`

### Import from Directory

```bash
pro-data-loader --csv-dir C:\Data\MasterData
```

Looks for the four CSV files in the specified directory.

### Import from Individual Files

```bash
pro-data-loader \
  --organizations orgs.csv \
  --regions regions.csv \
  --facilities facs.csv \
  --providers providers.csv
```

### Import from Current Directory

```bash
# Place CSV files in current directory
pro-data-loader
```

Looks for CSV files in the current directory.

## CSV File Formats

### organizations.csv

```csv
organization_code,organization_name,tax_id,contact_email,address_line1,address_line2,city,state,zip_code,active
ORG001,Regional Health System,12-3456789,admin@regional.health,123 Main St,,Springfield,IL,62701,true
```

**Required fields**: `organization_code`, `organization_name`

### regions.csv

```csv
organization_code,region_code,region_name,manager_name,manager_email,active
ORG001,R1,North Region,John Smith,john@regional.health,true
```

**Required fields**: `organization_code`, `region_code`, `region_name`

### facilities.csv

```csv
organization_code,region_code,facility_code,facility_name,facility_npi,tax_id,address_line1,address_line2,city,state,zip_code,phone,ehr_system,active
ORG001,R1,F1,North Medical Center,1234567890,12-3456789,100 Hospital Dr,,Chicago,IL,60601,555-1234,Athena,true
```

**Required fields**: `organization_code`, `region_code`, `facility_code`, `facility_name`

### providers.csv

```csv
facility_code,provider_npi,first_name,last_name,middle_name,credentials,specialty,taxonomy_code,email,phone,active
F1,1234567890,John,Smith,,MD,Family Medicine,207Q00000X,jsmith@example.health,555-1000,true
```

**Required fields**: `facility_code`, `provider_npi`, `first_name`, `last_name`

**NPI validation**: Must be exactly 10 digits

## Configuration

The tool automatically finds the `.env` file in these locations (in order):

1. `C:\ProgramData\Professional SMART\config\.env`
2. `C:\Program Files\Professional SMART\config\.env`
3. `.env` (current directory)

The `.env` file must contain:
```
DATABASE_URL=postgres://username:password@localhost:5432/professional_smart
```

## Validation

Before import, the tool validates:

1. **Required fields**: All required fields must be present and non-empty
2. **Duplicate detection**: Checks for duplicate codes/NPIs
3. **Referential integrity**: Regions must reference valid organizations, facilities must reference valid regions, etc.
4. **Data format**: NPIs must be 10 digits

If validation fails, the tool reports the exact row number and issue.

## Import Behavior

- **Upsert**: Existing records are updated based on unique keys
  - Organizations: `organization_code`
  - Regions: `(organization_id, region_code)`
  - Facilities: `facility_code`
  - Providers: `provider_npi`

- **Transaction**: All imports happen in a single transaction. If any step fails, all changes are rolled back.

- **Dependency order**: Data is imported in the correct order:
  1. Organizations
  2. Regions (require organizations)
  3. Facilities (require regions)
  4. Providers (require facilities)

## Examples

### Example 1: Generate Templates and Import

```bash
# Generate templates
pro-data-loader generate-templates C:\Temp\MasterData

# Edit the CSV files in Excel or text editor
# ...

# Import the data
pro-data-loader --csv-dir C:\Temp\MasterData
```

### Example 2: Update Existing Data

```bash
# Export current organizations
# Edit and add new ones
# Import (will update existing and add new)
pro-data-loader --csv-dir C:\Updates
```

### Example 3: Bulk Provider Import

```bash
# Import providers for new facility
# (assuming organizations, regions, facilities already exist)
pro-data-loader \
  --organizations empty.csv \
  --regions empty.csv \
  --facilities empty.csv \
  --providers new_providers.csv
```

Note: CSV files must exist but can be empty (just headers) if you're only importing one entity type.

## Error Handling

The tool provides clear, actionable error messages:

```
Error: Failed to parse organization at row 5
Caused by: missing field `organization_name`
```

```
Error: Duplicate organization code 'ORG001' at row 12
```

```
Error: Region at row 8 references unknown organization code 'ORG999'
```

## Building

```bash
cargo build --release --bin pro-data-loader
```

Output: `target/release/pro-data-loader.exe`

## Installation Location

After building, the executable should be copied to:
```
C:\Program Files\Professional SMART\bin\pro-data-loader.exe
```

## Support

For issues:
1. Verify the `.env` file exists and contains DATABASE_URL
2. Check CSV files have correct headers
3. Review error messages for specific row numbers
4. Ensure database is running and accessible
