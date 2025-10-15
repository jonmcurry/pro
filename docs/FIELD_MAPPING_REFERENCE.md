# Field Mapping Reference

Complete reference of all database fields that can be mapped from CSV files.

## Encounter Fields (encounter table)

These fields are claim/encounter-level data (one per claim).

| Target Field | Description | Data Type | Max Length | Required | Example |
|--------------|-------------|-----------|------------|----------|---------|
| patient_control_number | Unique claim/encounter identifier | String | 38 | Yes | "CLM-12345" |
| date_of_service_from | Service date start | Date | - | Yes | 2024-01-15 |
| date_of_service_to | Service date end | Date | - | No | 2024-01-15 |
| rendering_provider_npi | Rendering provider NPI | String | 10 | No | "1234567890" |
| billing_provider_npi | Billing provider NPI | String | 10 | No | "9876543210" |
| referring_provider_npi | Referring provider NPI | String | 10 | No | "5555555555" |
| facility_npi | Facility NPI | String | 10 | No | "1111111111" |
| subscriber_last_name | Patient last name | String | 60 | Yes | "Smith" |
| subscriber_first_name | Patient first name | String | 35 | Yes | "John" |
| subscriber_middle_name | Patient middle name | String | 25 | No | "Robert" |
| subscriber_birth_date | Patient date of birth | Date | - | Yes | 1980-05-12 |
| subscriber_gender | Patient gender | String | 1 | No | "M" (M/F/U) |
| subscriber_member_id | Insurance member ID | String | 80 | No | "ABC123456789" |
| subscriber_group_number | Insurance group number | String | 50 | No | "GRP-001" |
| place_of_service_code | Place of service (2-digit) | String | 2 | No | "11" |
| claim_filing_indicator_code | Insurance type code | String | 2 | No | "CI" |
| payer_responsibility_sequence | Primary/Secondary/Tertiary | String | 1 | No | "P" |
| claim_frequency_type_code | Claim frequency | String | 1 | No | "1" |
| patient_relationship_to_subscriber | Patient relationship | String | 2 | No | "18" |
| total_claim_charge_amount | Total claim charges | Decimal | - | No | 500.00 |

### Valid Gender Codes
- `M` - Male
- `F` - Female
- `U` - Unknown

### Common Place of Service Codes
- `11` - Office
- `21` - Inpatient Hospital
- `22` - On Campus Outpatient Hospital
- `23` - Emergency Room Hospital
- `31` - Skilled Nursing Facility
- `81` - Independent Laboratory

### Common Claim Filing Indicator Codes
- `CI` - Commercial Insurance
- `MB` - Medicare Part B
- `MC` - Medicaid
- `16` - Blue Cross Blue Shield
- `12` - Preferred Provider Organization (PPO)
- `HM` - Health Maintenance Organization (HMO)

## Service Line Fields (service_line table)

These fields are procedure/service-level data (one or more per claim).

| Target Field | Description | Data Type | Max Length | Required | Example |
|--------------|-------------|-----------|------------|----------|---------|
| service_line_number | Line number on claim | Integer | - | Yes | 1 |
| procedure_code | CPT or HCPCS code | String | 5 | Yes | "99213" |
| procedure_modifier_1 | First modifier | String | 2 | No | "25" |
| procedure_modifier_2 | Second modifier | String | 2 | No | "59" |
| procedure_modifier_3 | Third modifier | String | 2 | No | "GT" |
| procedure_modifier_4 | Fourth modifier | String | 2 | No | "76" |
| line_item_charge_amount | Charge for this service | Decimal | - | Yes | 150.00 |
| service_unit_count | Number of units | Decimal | - | No | 1.0 |
| service_date | Service date (if different) | Date | - | No | 2024-01-15 |
| revenue_center_code | Revenue code (institutional) | String | 4 | No | "0450" |
| national_drug_code | NDC code (for drugs) | String | 11 | No | "12345-6789-01" |

### Common Procedure Modifiers
- `25` - Significant, separately identifiable E/M service
- `26` - Professional component
- `50` - Bilateral procedure
- `59` - Distinct procedural service
- `76` - Repeat procedure by same physician
- `77` - Repeat procedure by another physician
- `GT` - Via interactive telecommunications
- `TC` - Technical component

## Diagnosis Fields (encounter_diagnosis table)

These fields are diagnosis codes linked to the claim.

| Target Field | Description | Data Type | Max Length | Required | Example |
|--------------|-------------|-----------|------------|----------|---------|
| diagnosis_code | ICD-10 diagnosis code | String | 7 | Yes | "Z00.00" |
| diagnosis_sequence | Position (1-12) | Integer | - | Yes | 1 |
| diagnosis_type | Admitting/Principal/Other | String | 3 | No | "ABF" |

### Diagnosis Types
- `ABF` - Admitting diagnosis
- `BF` - Principal diagnosis
- `ABK` - Other diagnosis

## Common CSV Header Mappings

### Patient Information

| Common CSV Headers | Target Field | Table |
|-------------------|--------------|-------|
| Patient ID, PatientID, Account, Encounter, Claim | patient_control_number | encounter |
| Patient Last Name, Last Name, Surname, LName | subscriber_last_name | encounter |
| Patient First Name, First Name, FName, Given Name | subscriber_first_name | encounter |
| DOB, Date of Birth, Birth Date, Birthdate | subscriber_birth_date | encounter |
| Gender, Sex, Patient Sex | subscriber_gender | encounter |
| Member ID, Insurance ID, Subscriber ID | subscriber_member_id | encounter |
| Group Number, Group ID, Group # | subscriber_group_number | encounter |

### Service Information

| Common CSV Headers | Target Field | Table |
|-------------------|--------------|-------|
| DOS, Date of Service, Service Date, Serv Date | date_of_service_from | encounter |
| CPT, CPT Code, Procedure Code, HCPCS, Procedure | procedure_code | service_line |
| Modifier 1, Mod1, Modifier | procedure_modifier_1 | service_line |
| Modifier 2, Mod2 | procedure_modifier_2 | service_line |
| Units, Unit Count, Qty, Quantity | service_unit_count | service_line |
| Charges, Charge Amount, Billed, Amount | line_item_charge_amount | service_line |
| POS, Place of Service | place_of_service_code | encounter |

### Provider Information

| Common CSV Headers | Target Field | Table |
|-------------------|--------------|-------|
| Provider NPI, Rendering NPI, Prov NPI, NPI | rendering_provider_npi | encounter |
| Billing NPI, Billing Provider NPI | billing_provider_npi | encounter |
| Facility NPI, Facility ID | facility_npi | encounter |

### Diagnosis Information

| Common CSV Headers | Target Field | Table |
|-------------------|--------------|-------|
| Diagnosis 1, DX1, Primary Diagnosis, Diagnosis | diagnosis_code | encounter_diagnosis |
| Diagnosis 2, DX2 | diagnosis_code | encounter_diagnosis |
| Diagnosis 3, DX3 | diagnosis_code | encounter_diagnosis |
| Diagnosis 4, DX4 | diagnosis_code | encounter_diagnosis |

## Data Format Requirements

### Dates
**Accepted Formats**:
- `YYYY-MM-DD` (preferred): 2024-01-15
- `MM/DD/YYYY`: 01/15/2024
- `M/D/YYYY`: 1/15/2024

### Decimals
**Format**:
- Use period (`.`) as decimal separator: `150.00`
- Do NOT use comma (`,`) as decimal separator: ~~`150,00`~~
- Up to 2 decimal places for currency
- No currency symbols: `150.00` not `$150.00`

### NPIs (National Provider Identifier)
**Format**:
- Exactly 10 digits
- Example: `1234567890`
- No dashes or spaces

### ICD-10 Codes
**Format**:
- 3-7 characters
- Letter + numbers + optional decimal
- Examples: `Z00.00`, `I10`, `E11.9`
- Decimal point optional but recommended

### CPT/HCPCS Codes
**Format**:
- 5 characters
- Numbers only (CPT) or letter + numbers (HCPCS)
- Examples: `99213`, `J1234`, `G0001`

### Modifiers
**Format**:
- Exactly 2 characters
- Numbers or letters
- Examples: `25`, `GT`, `59`

### Gender
**Accepted Values**:
- `M` or `Male` → M
- `F` or `Female` → F
- `U` or `Unknown` or blank → U

## Validation Examples

### Valid Data
```csv
Patient ID,DOS,CPT,Charges,Provider NPI,Diagnosis 1
CLM123,2024-01-15,99213,150.00,1234567890,Z00.00
```

### Invalid Data (with corrections)

| Issue | Invalid | Valid | Reason |
|-------|---------|-------|--------|
| Missing required field | `DOS,CPT` | `CLM123,DOS,CPT` | patient_control_number required |
| Invalid date | `01-15-2024` | `2024-01-15` or `01/15/2024` | Wrong date format |
| Invalid decimal | `150,00` | `150.00` | Use period not comma |
| Invalid NPI | `123456789` | `1234567890` | NPI must be 10 digits |
| Invalid gender | `Male` | `M` | Use single character code |
| Invalid CPT | `9921` | `99213` | CPT must be 5 characters |

## Transformation Examples

### Uppercase
```
Input: "cpt123"
Output: "CPT123"
```

### Trim
```
Input: "  123  "
Output: "123"
```

### Normalize NPI
```
Input: "123-456-7890"
Output: "1234567890"

Input: "1234567890  "
Output: "1234567890"
```

### Clean ICD-10
```
Input: "Z0000"
Output: "Z00.00"

Input: "z00.00"
Output: "Z00.00"
```

### Standardize Gender
```
Input: "Male", "MALE", "m", "M"
Output: "M"

Input: "Female", "FEMALE", "f", "F"
Output: "F"

Input: "Unknown", "U", "", null
Output: "U"
```

## Complete Mapping Example

This example shows how a CSV row maps to database fields:

**CSV Row**:
```csv
Account: "CLM-12345"
Service Date: "01/15/2024"
CPT: "99213"
Modifier 1: "25"
Units: "1"
Charges: "150.00"
Provider NPI: "1234567890"
Diagnosis 1: "Z00.00"
Patient Last: "Smith"
Patient First: "John"
DOB: "05/12/1980"
Gender: "M"
POS: "11"
```

**Database Mapping**:

**Encounter Table**:
```
patient_control_number: "CLM-12345"
date_of_service_from: 2024-01-15
rendering_provider_npi: "1234567890"
subscriber_last_name: "Smith"
subscriber_first_name: "John"
subscriber_birth_date: 1980-05-12
subscriber_gender: "M"
place_of_service_code: "11"
```

**Service Line Table**:
```
service_line_number: 1
procedure_code: "99213"
procedure_modifier_1: "25"
service_unit_count: 1.0
line_item_charge_amount: 150.00
```

**Encounter Diagnosis Table**:
```
diagnosis_code: "Z00.00"
diagnosis_sequence: 1
```

## Need Help?

- Review [CSV_MAPPING_GUIDE.md](CSV_MAPPING_GUIDE.md) for mapping configuration
- Check [CONFIGURATION.md](CONFIGURATION.md) for system settings
- See [docs/examples/](examples/) for CSV templates
- Contact your administrator for custom mapping needs
