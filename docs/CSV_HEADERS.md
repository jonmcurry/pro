# CSV File Headers Reference

## Overview

The Professional SMART Data Loader supports **dynamic CSV header parsing**. The system automatically reads headers from the first row of each CSV file and maps them to the corresponding data fields using case-sensitive exact name matching.

## 🎯 Claims Import: Only Facility Required!

**Important:** For claims processing, **only the facility needs to exist** in master data. The organization and region are automatically derived from the facility.

- ✅ **Facility** (REQUIRED for claims) - identified by `facility_code` or `facility_npi`
- ✅ **Organization** (auto-derived from facility.organization_id)
- ✅ **Region** (auto-derived from facility.region_id, can be NULL)

When importing claims (CSV or EDI 837P), you only need to specify the facility identifier. The system will:
1. Look up the facility by code or NPI
2. Automatically populate organization_id from facility.organization_id
3. Automatically populate region_id from facility.region_id (NULL if no region)

**This means:** Claims data does NOT need organization_code or region_code - only facility_code/facility_npi!

## Header Parsing Behavior

- ✅ **Dynamic Headers**: Headers are read from the first row of the CSV file
- ✅ **Case Sensitive**: Header names must match exactly (e.g., `organization_code` not `Organization_Code`)
- ✅ **Order Independent**: Columns can appear in any order
- ✅ **Optional Fields**: Fields marked as `Option<T>` can be omitted or empty
- ✅ **Default Values**: Some boolean fields default to `true` if not provided

---

## Claims CSV (For Importing Claims Data)

**File**: `claims.csv` (or similar names like `claims_*.csv`)

**Purpose**: Import encounter/claim data into the system.

### Required Headers for Claims Import

| Header | Type | Description | Example |
|--------|------|-------------|---------|
| `facility_code` | String | **REQUIRED** - Facility identifier | `F1` |
| `patient_control_number` | String | Unique claim identifier | `CLAIM-001` |
| `subscriber_last_name` | String | Patient last name | `Smith` |
| `subscriber_first_name` | String | Patient first name | `John` |
| `subscriber_birth_date` | Date | Patient date of birth | `1980-01-15` |
| `date_of_service_from` | Date | Service date | `2025-01-15` |
| `procedure_code` | String | CPT/HCPCS code | `99213` |
| `line_item_charge_amount` | Decimal | Charge amount | `150.00` |
| `diagnosis_code` | String | Primary diagnosis (ICD-10) | `Z00.00` |

### Optional Headers (Common)
- `facility_npi` - Alternative to facility_code
- `provider_npi` - Rendering provider NPI
- `payer_id` - Insurance payer identifier
- `payer_name` - Insurance payer name
- Many more (see COMPLETE_HEADER_LIST.md for full list)

### Critical Note for Claims CSV

**Only include:**
- ✅ `facility_code` OR `facility_npi` - System will lookup and auto-populate organization and region

---

## Master Data CSVs (For Loading Organizations, Regions, Facilities)

The following sections describe CSV formats for loading **master data** (the organizational hierarchy). These are separate from claims import.

---

## Organizations CSV

**File**: `organizations.csv`

### Required Headers
| Header | Type | Description | Example |
|--------|------|-------------|---------|
| `organization_code` | String | Unique organization identifier | `ORG001` |
| `organization_name` | String | Organization name | `Regional Health System` |

### Optional Headers
| Header | Type | Description | Example |
|--------|------|-------------|---------|
| `tax_id` | String | Federal tax ID (EIN) | `12-3456789` |
| `contact_email` | String | Primary contact email | `contact@org.com` |
| `address_line1` | String | Street address line 1 | `123 Medical Plaza` |
| `address_line2` | String | Street address line 2 | `Suite 100` |
| `city` | String | City | `Houston` |
| `state` | String | State/Province code | `TX` |
| `zip_code` | String | Postal/ZIP code | `77001` |
| `active` | Boolean | Active status (default: `true`) | `true`, `false` |

### Example
```csv
organization_code,organization_name,tax_id,contact_email,address_line1,city,state,zip_code,active
ORG001,Regional Health System,12-3456789,contact@org001.com,123 Medical Plaza,Houston,TX,77001,true
ORG002,Metro Medical Group,98-7654321,info@org002.com,456 Healthcare Blvd,Chicago,IL,60601,true
```

---

## Regions CSV

**File**: `regions.csv`

### Required Headers
| Header | Type | Description | Example |
|--------|------|-------------|---------|
| `organization_code` | String | Parent organization code | `ORG001` |
| `region_code` | String | Unique region identifier | `R1` |
| `region_name` | String | Region name | `North Region` |

### Optional Headers
| Header | Type | Description | Example |
|--------|------|-------------|---------|
| `manager_name` | String | Region manager name | `John Smith` |
| `manager_email` | String | Region manager email | `john.smith@org.com` |
| `active` | Boolean | Active status (default: `true`) | `true`, `false` |

### Example
```csv
organization_code,region_code,region_name,manager_name,manager_email,active
ORG001,R1,North Region,John Smith,john.smith@org1.com,true
ORG001,R2,South Region,Jane Doe,jane.doe@org1.com,true
```

### Notes
- Region is **optional** - facilities can exist without regions
- If regions are provided, they must reference valid organizations
- `region_code` is unique within an organization, but can be reused across organizations

---

## Facilities CSV

**File**: `facilities.csv`

### Required Headers
| Header | Type | Description | Example |
|--------|------|-------------|---------|
| `organization_code` | String | Parent organization code | `ORG001` |
| `region_code` | String | Parent region code | `R1` |
| `facility_code` | String | Unique facility identifier | `F1` |
| `facility_name` | String | Facility name | `North Medical Center` |

### Optional Headers
| Header | Type | Description | Example |
|--------|------|-------------|---------|
| `facility_npi` | String | 10-digit NPI | `1234567890` |
| `tax_id` | String | Federal tax ID | `12-3456789` |
| `address_line1` | String | Street address line 1 | `100 Healthcare Dr` |
| `address_line2` | String | Street address line 2 | `Building A` |
| `city` | String | City | `Houston` |
| `state` | String | State/Province code | `TX` |
| `zip_code` | String | Postal/ZIP code | `77001` |
| `phone` | String | Phone number | `(555) 123-4567` |
| `ehr_system` | String | EHR system name | `EPIC`, `ATHENA`, `CERNER` |
| `active` | Boolean | Active status (default: `true`) | `true`, `false` |

### Example
```csv
organization_code,region_code,facility_code,facility_name,facility_npi,tax_id,address_line1,city,state,zip_code,phone,ehr_system,active
ORG001,R1,F1,North Medical Center,1234567890,12-3456789,100 Healthcare Dr,Houston,TX,77001,(555) 123-4567,EPIC,true
ORG001,R1,F2,North Clinic,1234567891,12-3456789,200 Medical Plaza,Houston,TX,77002,(555) 123-4568,ATHENA,true
```

### Notes
- `region_code` can be empty if facilities don't use regions
- If regions are used, the region must exist for the same organization
- `facility_code` is unique globally across all facilities

---

## Providers CSV

**File**: `providers.csv`

### Required Headers
| Header | Type | Description | Example |
|--------|------|-------------|---------|
| `facility_code` | String | Parent facility code | `F1` |
| `provider_npi` | String | 10-digit NPI | `1234567890` |
| `first_name` | String | Provider first name | `John` |
| `last_name` | String | Provider last name | `Smith` |

### Optional Headers
| Header | Type | Description | Example |
|--------|------|-------------|---------|
| `middle_name` | String | Middle name/initial | `A` |
| `credentials` | String | Professional credentials | `MD`, `DO`, `NP`, `PA` |
| `specialty` | String | Medical specialty | `Family Medicine` |
| `taxonomy_code` | String | Healthcare Provider Taxonomy | `207Q00000X` |
| `email` | String | Provider email | `john.smith@facility.com` |
| `phone` | String | Provider phone | `(555) 100-0001` |
| `active` | Boolean | Active status (default: `true`) | `true`, `false` |

### Example
```csv
facility_code,provider_npi,first_name,last_name,middle_name,credentials,specialty,taxonomy_code,email,phone,active
F1,1234567890,John,Smith,A,MD,Family Medicine,207Q00000X,john.smith@facility1.com,(555) 100-0001,true
F1,1234567891,Jane,Doe,B,DO,Internal Medicine,207R00000X,jane.doe@facility1.com,(555) 100-0002,true
```

### Notes
- Provider is **optional** - facilities can exist without providers
- `provider_npi` must be unique globally
- The referenced `facility_code` must exist

---

## Header Mapping Rules

### Boolean Fields
Boolean fields accept the following values (case-insensitive):
- **True**: `true`, `1`, `yes`, `t`, `y`
- **False**: `false`, `0`, `no`, `f`, `n`
- **Default**: If field is missing or empty, defaults to `true` for `active` fields

### String Fields
- Empty strings are treated as `None` for optional fields
- Leading/trailing whitespace is preserved (trim your data if needed)
- No length limits enforced at parse time (database constraints apply)

### Null/Missing Values
- Optional fields can be:
  - Omitted entirely (column not present)
  - Empty string (`""`)
  - Left blank in CSV (`,,,`)

### Case Sensitivity
⚠️ **Important**: Header names are case-sensitive and must match exactly:
- ✅ Correct: `organization_code`
- ❌ Wrong: `Organization_Code`, `ORGANIZATION_CODE`, `OrganizationCode`

### Column Order
✅ Columns can appear in **any order**. The parser matches by header name, not position.

Example - both are valid:
```csv
organization_code,organization_name,city
ORG001,Health System,Houston
```

```csv
city,organization_name,organization_code
Houston,Health System,ORG001
```

---

## Validation

CSV headers are validated during import:

1. **Header Presence**: All required headers must be present
2. **Header Names**: Must match exactly (case-sensitive)
3. **Data Types**: Values must be convertible to the specified type
4. **Referential Integrity**: Foreign keys validated after parsing
   - Regions reference organizations
   - Facilities reference organizations and regions
   - Providers reference facilities

### Error Messages

If headers don't match:
```
Failed to parse organization at row 2: missing field `organization_code`
```

If data type is wrong:
```
Failed to parse organization at row 3: invalid type: string "abc", expected bool for field `active`
```

---

## Best Practices

### Header Naming
- Use exact names from this documentation
- Use lowercase with underscores (snake_case)
- Don't add spaces or special characters

### File Encoding
- UTF-8 encoding required
- Unix (LF) or Windows (CRLF) line endings supported

### Data Format
- Dates: Not used in master data files
- Booleans: Use `true`/`false` for clarity
- Codes: Use consistent formatting (e.g., always `ORG001` not `org001`)

### Testing Headers
Use the template generator to create files with correct headers:
```bash
# From GUI: File > Generate Templates
# Or from code: pro_data_loader::templates::generate_templates()
```

---

## Template Generation

To generate CSV templates with correct headers:

### Using GUI
1. Open Data Loader GUI
2. Click "Generate Templates..."
3. Select output directory
4. Templates are created with all headers

### Using Code
```rust
use pro_data_loader::templates;

templates::generate_templates("./output")?;
```

This creates:
- `organizations_template.csv`
- `regions_template.csv`
- `facilities_template.csv`
- `providers_template.csv`

Each with proper headers and example data.

---

## Summary: Required Headers Quick Reference

### Organizations (2 required)
- ✅ `organization_code`
- ✅ `organization_name`

### Regions (3 required)
- ✅ `organization_code`
- ✅ `region_code`
- ✅ `region_name`

### Facilities (4 required)
- ✅ `organization_code`
- ✅ `region_code`
- ✅ `facility_code`
- ✅ `facility_name`

### Providers (4 required)
- ✅ `facility_code`
- ✅ `provider_npi`
- ✅ `first_name`
- ✅ `last_name`

All other fields are optional and can be omitted or left empty.
