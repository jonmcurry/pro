# CSV Header Mapping Guide

This guide explains how the Professional SMART system handles CSV files with different header formats through dynamic header mapping.

## Overview

The CSV parser uses a **dynamic header mapping system** that:
- Supports multiple EHR systems (Athena, Epic, Cerner, Generic)
- Auto-detects which mapping best matches your CSV file
- Allows case-insensitive header matching
- Supports alternate header names for the same field
- Validates data types and formats
- Applies transformations to standardize data

## How It Works

1. **Upload CSV File** - Drop your CSV file into the input directory
2. **Auto-Detection** - System reads headers and scores each predefined mapping
3. **Best Match Selection** - Highest scoring mapping is selected
4. **Field Mapping** - CSV columns are mapped to database fields
5. **Validation** - Data is validated according to mapping rules
6. **Transformation** - Data is transformed to standard format
7. **Import** - Claims are imported into the database

## Predefined Mappings

### Athena Health

**Source System**: `ATHENA`

**Supported Headers**:

| CSV Header | Alternate Headers | Database Field | Table | Required |
|------------|-------------------|----------------|-------|----------|
| Patient ID | PatientID, Patient # | patient_control_number | encounter | Yes |
| DOS | Date of Service, Service Date | date_of_service_from | encounter | Yes |
| Provider NPI | Rendering NPI | rendering_provider_npi | encounter | No |
| CPT | CPT Code, Procedure Code | procedure_code | service_line | Yes |
| Modifier 1 | Mod 1 | procedure_modifier_1 | service_line | No |
| Units | | service_unit_count | service_line | Yes |
| Charges | Charge Amount, Billed Amount | line_item_charge_amount | service_line | Yes |
| Diagnosis 1 | DX1, Primary Diagnosis | diagnosis_code | encounter_diagnosis | Yes |
| Patient Last Name | Last Name | subscriber_last_name | encounter | Yes |
| Patient First Name | First Name | subscriber_first_name | encounter | Yes |
| DOB | Date of Birth, Birth Date | subscriber_birth_date | encounter | Yes |
| Gender | Sex | subscriber_gender | encounter | No |
| POS | Place of Service | place_of_service_code | encounter | No |

**Example Athena CSV**:
```csv
Patient ID,DOS,Provider NPI,CPT,Modifier 1,Units,Charges,Diagnosis 1,Patient Last Name,Patient First Name,DOB,Gender,POS
12345,2024-01-15,1234567890,99213,,,150.00,Z00.00,Smith,John,1980-05-12,M,11
```

### Epic

**Source System**: `EPIC`

**Supported Headers**:

| CSV Header | Alternate Headers | Database Field | Table | Required |
|------------|-------------------|----------------|-------|----------|
| ACCOUNT NUMBER | Account | patient_control_number | encounter | Yes |
| SERV DT | SERVICE DATE | date_of_service_from | encounter | Yes |
| PROC CD | PROCEDURE | procedure_code | service_line | Yes |
| CHG | CHARGE | line_item_charge_amount | service_line | Yes |

**Example Epic CSV**:
```csv
ACCOUNT NUMBER,SERV DT,PROC CD,CHG
E12345,01/15/2024,99213,150.00
```

### Cerner

**Source System**: `CERNER`

**Supported Headers**:

| CSV Header | Alternate Headers | Database Field | Table | Required |
|------------|-------------------|----------------|-------|----------|
| Encounter_ID | EncounterID | patient_control_number | encounter | Yes |
| Service_Date | | date_of_service_from | encounter | Yes |
| CPT_Code | | procedure_code | service_line | Yes |

**Example Cerner CSV**:
```csv
Encounter_ID,Service_Date,CPT_Code
C98765,2024-01-15,99213
```

### Generic

**Source System**: `GENERIC`

The Generic mapping is the most flexible and accepts many common header variations.

**Supported Headers**:

| CSV Header | Alternate Headers | Database Field | Table | Required |
|------------|-------------------|----------------|-------|----------|
| Patient Control Number | Patient ID, Account, Encounter, Claim | patient_control_number | encounter | Yes |
| Date of Service | DOS, Service Date, Serv Date | date_of_service_from | encounter | Yes |
| Procedure Code | CPT, CPT Code, HCPCS, Procedure | procedure_code | service_line | Yes |
| Charge Amount | Charges, Billed, Amount | line_item_charge_amount | service_line | Yes |

**Example Generic CSV**:
```csv
Patient ID,Date of Service,CPT Code,Charges
ABC123,2024-01-15,99213,150.00
```

## Data Types

The system supports the following data types:

| Data Type | Description | Example |
|-----------|-------------|---------|
| String | Text data | "John Smith" |
| Integer | Whole numbers | 42 |
| Decimal | Numbers with decimals | 150.00 |
| Date | Date only (YYYY-MM-DD) | 2024-01-15 |
| DateTime | Date and time | 2024-01-15 10:30:00 |
| Boolean | True/false | true |
| Uuid | Universally unique identifier | 550e8400-e29b-41d4-a716-446655440000 |

## Validation Rules

The system validates data according to these rules:

| Rule | Description | Example |
|------|-------------|---------|
| MinLength | Minimum string length | MinLength(3) |
| MaxLength | Maximum string length | MaxLength(50) |
| Regex | Pattern matching | Regex("^[A-Z]{2}[0-9]{3}$") |
| Range | Numeric range | Range { min: 0.0, max: 9999.99 } |
| OneOf | Must be one of specified values | OneOf(["M", "F", "U"]) |
| NotEmpty | Cannot be empty or whitespace | NotEmpty |
| Npi | Valid 10-digit NPI | Npi |
| Icd10 | Valid ICD-10 diagnosis code | Icd10 |
| CptHcpcs | Valid CPT or HCPCS code | CptHcpcs |
| Mbi | Valid Medicare Beneficiary Identifier | Mbi |

## Transformations

Data can be automatically transformed during import:

| Transformation | Description | Example |
|----------------|-------------|---------|
| Uppercase | Convert to uppercase | "abc" → "ABC" |
| Lowercase | Convert to lowercase | "ABC" → "abc" |
| Trim | Remove leading/trailing whitespace | " text " → "text" |
| RemoveSpaces | Remove all spaces | "A B C" → "ABC" |
| RemoveNonAlphanumeric | Keep only letters and numbers | "A-B-C-123" → "ABC123" |
| PadLeft | Pad left with characters | "42" → "0042" |
| PadRight | Pad right with characters | "42" → "4200" |
| Replace | Replace pattern with text | "A-B" → "A_B" |
| Split | Split and take part | "First,Last" → "First" |
| DateFormat | Convert date format | "01/15/2024" → "2024-01-15" |
| Concat | Combine multiple fields | ["First", "Last"] → "First Last" |
| Custom | Apply custom function | normalize_npi, clean_icd10, etc. |

## Creating Custom Mappings (Future Feature)

While the system is designed to support custom mappings, this feature is not yet fully implemented. The infrastructure exists to:

1. **Create mappings programmatically** (in code)
2. **Serialize to JSON** for storage
3. **Deserialize from JSON** to load custom mappings
4. **Store in database** for organization-specific configurations

### Example JSON Mapping (Reference)

This is the JSON format that custom mappings will use when the feature is implemented:

```json
{
  "mapping_id": null,
  "mapping_name": "Custom Clinic Format",
  "source_system": "CUSTOM",
  "field_mappings": [
    {
      "csv_header": "Account Number",
      "alternate_headers": ["Acct", "Account #"],
      "target_field": "patient_control_number",
      "target_table": "encounter",
      "data_type": "String",
      "is_required": true,
      "default_value": null,
      "validation_rules": [
        { "MaxLength": 38 }
      ]
    },
    {
      "csv_header": "Service Date",
      "alternate_headers": ["DOS", "Date"],
      "target_field": "date_of_service_from",
      "target_table": "encounter",
      "data_type": "Date",
      "is_required": true,
      "default_value": null,
      "validation_rules": []
    },
    {
      "csv_header": "Procedure",
      "alternate_headers": ["CPT"],
      "target_field": "procedure_code",
      "target_table": "service_line",
      "data_type": "String",
      "is_required": true,
      "default_value": null,
      "validation_rules": [
        "CptHcpcs"
      ]
    },
    {
      "csv_header": "Amount",
      "alternate_headers": ["Charge", "Billed"],
      "target_field": "line_item_charge_amount",
      "target_table": "service_line",
      "data_type": "Decimal",
      "is_required": true,
      "default_value": null,
      "validation_rules": [
        { "Range": { "min": 0.0, "max": 999999.99 } }
      ]
    }
  ],
  "transformations": [
    {
      "target_field": "patient_control_number",
      "transformation_type": "Trim"
    },
    {
      "target_field": "procedure_code",
      "transformation_type": "Uppercase"
    }
  ]
}
```

## Troubleshooting

### CSV Not Being Detected

**Problem**: CSV file is uploaded but not processed

**Solutions**:
- Verify file is in the correct input directory
- Check file has `.csv` extension
- Ensure file is not locked by another program
- Check logs for error messages

### Headers Not Matching

**Problem**: Headers are not being recognized

**Solutions**:
- Review the predefined mappings above
- Ensure headers are in the first row
- Check for typos in header names
- Use alternate headers if primary names don't match
- Consider using the Generic mapping which is most flexible

### Data Validation Errors

**Problem**: Data is rejected during validation

**Solutions**:
- Check data types match expected formats
- Verify dates are in YYYY-MM-DD format (or MM/DD/YYYY for auto-conversion)
- Ensure required fields are not empty
- Validate NPI numbers are 10 digits
- Validate ICD-10 codes are in correct format
- Check decimal numbers use period (.) not comma (,)

### Auto-Detection Choosing Wrong Mapping

**Problem**: System selects incorrect mapping

**Solutions**:
- Review which headers are present in your CSV
- Add more standard headers to improve detection
- Use more specific header names that match one system
- Contact administrator about creating custom mapping for your format

## Best Practices

### CSV File Preparation

1. **Use Standard Headers**: Match predefined headers when possible
2. **Include All Required Fields**: See tables above for required fields
3. **Use Consistent Formats**: Keep date, decimal, and text formats consistent
4. **Clean Data First**: Remove extra spaces, special characters
5. **Validate Before Upload**: Check file opens correctly in Excel/text editor

### Header Naming

1. **Be Descriptive**: "Patient ID" better than "ID"
2. **Follow Conventions**: Use system-standard headers when available
3. **Avoid Special Characters**: Stick to letters, numbers, spaces
4. **Use Spaces or Underscores**: "Date of Service" or "Date_of_Service"
5. **Case Doesn't Matter**: "PATIENT ID" = "Patient ID" = "patient id"

### Data Quality

1. **Validate NPIs**: Ensure 10-digit provider NPIs are correct
2. **Check Procedure Codes**: Verify CPT/HCPCS codes are valid
3. **Verify Dates**: Use YYYY-MM-DD or MM/DD/YYYY formats
4. **Confirm Amounts**: Use decimal format with period (150.00 not 150,00)
5. **Complete Required Fields**: All required fields must have values

## CSV Template Downloads

### Athena Health Template

[Download: athena_template.csv](/docs/examples/athena_template.csv)

### Epic Template

[Download: epic_template.csv](/docs/examples/epic_template.csv)

### Cerner Template

[Download: cerner_template.csv](/docs/examples/cerner_template.csv)

### Generic Template

[Download: generic_template.csv](/docs/examples/generic_template.csv)

## Complete Field Reference

See [FIELD_MAPPING_REFERENCE.md](FIELD_MAPPING_REFERENCE.md) for a complete list of all database fields and their descriptions.

## Getting Help

If you need assistance with CSV mapping:

1. Review this guide and examples
2. Check logs in the configured log directory
3. Verify your CSV matches one of the predefined formats
4. Contact your system administrator about creating a custom mapping

## Future Enhancements

Planned features for CSV mapping:

- [ ] Web UI to create and manage custom mappings
- [ ] API endpoints for mapping management
- [ ] Organization-specific mapping overrides
- [ ] Facility-specific mapping configurations
- [ ] Visual field mapping editor
- [ ] Mapping testing and validation tools
- [ ] Import/export mapping configurations
- [ ] Mapping templates library
