# Complete CSV Header Mapping Reference

This document lists ALL possible database fields that can be mapped from CSV imports, organized by category.

**Last Updated**: 2025-01-15

---

## Table of Contents

1. [Encounter (Claim-Level) Fields](#encounter-claim-level-fields)
2. [Service Line (Procedure-Level) Fields](#service-line-procedure-level-fields)
3. [Diagnosis Fields](#diagnosis-fields)
4. [Organization Fields](#organization-fields)
5. [Facility Fields](#facility-fields)
6. [Provider Fields](#provider-fields)
7. [Payer Fields](#payer-fields)
8. [Import Batch Tracking](#import-batch-tracking)
9. [Quick Reference: Common Header Variations](#quick-reference-common-header-variations)

---

## Encounter (Claim-Level) Fields

**Table**: `claims.encounter`

**Description**: These are claim/encounter-level fields - one set per claim. Most CSV files will have these columns.

### Control Numbers and Identifiers

| Database Field | Data Type | Required | Max Length | Description | Common CSV Headers |
|----------------|-----------|----------|------------|-------------|--------------------|
| patient_control_number | String | Yes | 38 | Unique claim identifier | Patient ID, PatientID, Account, Account Number, Encounter, Claim, Claim Number, Patient Control Number, PCN, Account #, Acct |
| subscriber_id | String | Yes | 50 | Patient/Member ID | Member ID, Subscriber ID, Insurance ID, Patient Member ID, Member Number, Insurance Number |
| transaction_set_control_number | String | No | 9 | EDI transaction control number | Transaction Control Number, Control Number, TSC |
| submitter_id | String | Yes | 80 | Submitter identifier | Submitter ID, Submitter Code |
| submitter_name | String | No | 60 | Submitter name | Submitter Name, Submitter |

### Patient/Subscriber Information

| Database Field | Data Type | Required | Max Length | Description | Common CSV Headers |
|----------------|-----------|----------|------------|-------------|--------------------|
| subscriber_last_name | String | Yes | 60 | Patient last name | Patient Last Name, Last Name, Surname, LName, Patient Surname |
| subscriber_first_name | String | Yes | 35 | Patient first name | Patient First Name, First Name, FName, Given Name, Patient Given Name |
| subscriber_middle_name | String | No | 25 | Patient middle name | Patient Middle Name, Middle Name, MName, Middle Initial, MI |
| subscriber_name_suffix | String | No | 10 | Name suffix (Jr, Sr, III) | Name Suffix, Suffix, Patient Suffix |
| subscriber_gender | String | No | 1 | Gender (M/F/U) | Gender, Sex, Patient Sex, Patient Gender |
| subscriber_birth_date | Date | Yes | - | Date of birth | DOB, Date of Birth, Birth Date, Birthdate, Patient DOB, Patient Birth Date |
| subscriber_address_line1 | String | No | 55 | Address line 1 | Address 1, Address Line 1, Street Address, Patient Address, Address |
| subscriber_address_line2 | String | No | 55 | Address line 2 | Address 2, Address Line 2, Apt, Suite, Unit |
| subscriber_city | String | No | 30 | City | City, Patient City |
| subscriber_state | String | No | 2 | State code | State, Patient State, State Code |
| subscriber_postal_code | String | No | 15 | Zip/postal code | Zip, Zip Code, Postal Code, Patient Zip, ZIP |
| subscriber_country | String | No | 3 | Country code | Country, Country Code |

### Service Dates

| Database Field | Data Type | Required | Max Length | Description | Common CSV Headers |
|----------------|-----------|----------|------------|-------------|--------------------|
| date_of_service_from | Date | Yes | - | Service start date | DOS, Date of Service, Service Date, Serv Date, From Date, Start Date, Service From |
| date_of_service_to | Date | No | - | Service end date | Service To, End Date, To Date, Through Date |

### Payer Information

| Database Field | Data Type | Required | Max Length | Description | Common CSV Headers |
|----------------|-----------|----------|------------|-------------|--------------------|
| payer_responsibility_code | String | Yes | 1 | P/S/T (Primary/Secondary/Tertiary) | Payer Responsibility, Responsibility, Payer Sequence |
| payer_id | String | No | 80 | Payer identifier | Payer ID, Insurance ID, Payer Code, Insurance Company ID |
| payer_name | String | No | 60 | Payer name | Payer Name, Insurance Name, Insurance Company, Payer, Insurance |
| claim_filing_indicator | String | No | 2 | Insurance type code | Claim Filing Indicator, Filing Indicator, Insurance Type |

### Provider Information (Claim Level)

| Database Field | Data Type | Required | Max Length | Description | Common CSV Headers |
|----------------|-----------|----------|------------|-------------|--------------------|
| rendering_provider_npi | String | No | 10 | Rendering provider NPI | Provider NPI, Rendering NPI, Prov NPI, NPI, Rendering Provider NPI, Doctor NPI |
| billing_provider_npi | String | No | 10 | Billing provider NPI | Billing NPI, Billing Provider NPI, Bill Prov NPI |
| billing_provider_tax_id | String | No | 12 | Billing provider Tax ID | Billing Tax ID, Tax ID, TIN, EIN |
| billing_provider_name | String | No | 60 | Billing provider name | Billing Provider, Billing Provider Name |
| referring_provider_npi | String | No | 10 | Referring provider NPI | Referring NPI, Referring Provider NPI, Ref Prov NPI |
| supervising_provider_npi | String | No | 10 | Supervising provider NPI | Supervising NPI, Supervising Provider NPI |
| service_facility_npi | String | No | 10 | Service facility NPI | Facility NPI, Service Facility NPI, Facility ID |

### Claim Information

| Database Field | Data Type | Required | Max Length | Description | Common CSV Headers |
|----------------|-----------|----------|------------|-------------|--------------------|
| total_claim_charge_amount | Decimal | Yes | - | Total claim charges | Total Charges, Claim Total, Total Amount, Total Billed, Claim Charges |
| place_of_service_code | String | No | 2 | Place of service (2-digit) | POS, Place of Service, Place, Service Location |
| claim_frequency_code | String | No | 1 | Claim frequency (1-9) | Claim Frequency, Frequency Code, Frequency |
| claim_status | String | Yes | 20 | Current claim status | Claim Status, Status |
| case_status | String | No | 20 | Case status | Case Status |
| financial_class | String | No | 10 | Financial class | Financial Class, Fin Class, FC |

### Coder Information

| Database Field | Data Type | Required | Max Length | Description | Common CSV Headers |
|----------------|-----------|----------|------------|-------------|--------------------|
| coder_id | UUID | No | - | Coder identifier (internal) | Coder ID, Coder Code, Coded By |
| coding_date | Date | No | - | Date coded | Coding Date, Coded Date, Date Coded |

---

## Service Line (Procedure-Level) Fields

**Table**: `claims.service_line`

**Description**: These are procedure/service-level fields - one or more per claim. Can have multiple rows per encounter.

### Line Identification

| Database Field | Data Type | Required | Max Length | Description | Common CSV Headers |
|----------------|-----------|----------|------------|-------------|--------------------|
| line_number | Integer | Yes | - | Line number (1, 2, 3...) | Line Number, Line #, Line, Service Line Number, Seq |

### Procedure Information

| Database Field | Data Type | Required | Max Length | Description | Common CSV Headers |
|----------------|-----------|----------|------------|-------------|--------------------|
| procedure_code | String | Yes | 5 | CPT or HCPCS code | CPT, CPT Code, Procedure Code, HCPCS, Procedure, Proc Code, Proc CD, Code |
| procedure_modifier_1 | String | No | 2 | First modifier | Modifier 1, Mod1, Modifier, Mod, Modifier I |
| procedure_modifier_2 | String | No | 2 | Second modifier | Modifier 2, Mod2, Modifier II |
| procedure_modifier_3 | String | No | 2 | Third modifier | Modifier 3, Mod3, Modifier III |
| procedure_modifier_4 | String | No | 2 | Fourth modifier | Modifier 4, Mod4, Modifier IV |
| procedure_description | String | No | 80 | Procedure description | Procedure Description, Description, Proc Description, Service Description |
| product_service_id_qualifier | String | No | 2 | Code qualifier | Qualifier, Service Qualifier |

### Charges and Units

| Database Field | Data Type | Required | Max Length | Description | Common CSV Headers |
|----------------|-----------|----------|------------|-------------|--------------------|
| line_item_charge_amount | Decimal | Yes | - | Charge for this line | Charges, Charge Amount, Charge, Billed, Amount, Line Charge, Billed Amount |
| service_unit_count | Decimal | Yes | - | Number of units | Units, Unit Count, Qty, Quantity, Unit, Service Units |
| unit_basis_measurement_code | String | No | 2 | Unit basis code | Unit Basis, Measurement Code |

### Service Dates (Line Level)

| Database Field | Data Type | Required | Max Length | Description | Common CSV Headers |
|----------------|-----------|----------|------------|-------------|--------------------|
| service_date_from | Date | Yes | - | Line service start date | Service Date, Line Date, DOS, Date of Service, Service From |
| service_date_to | Date | No | - | Line service end date | Service To, Line Date To, Through Date |

### Place of Service (Line Level)

| Database Field | Data Type | Required | Max Length | Description | Common CSV Headers |
|----------------|-----------|----------|------------|-------------|--------------------|
| place_of_service_code | String | No | 2 | Place of service (line level) | POS, Place of Service, Place |

### Provider Information (Line Level)

| Database Field | Data Type | Required | Max Length | Description | Common CSV Headers |
|----------------|-----------|----------|------------|-------------|--------------------|
| rendering_provider_npi | String | No | 10 | Line-level rendering NPI | Rendering NPI, Provider NPI, NPI |
| supervising_provider_npi | String | No | 10 | Line-level supervising NPI | Supervising NPI, Supervisor NPI |
| ordering_provider_npi | String | No | 10 | Ordering provider NPI | Ordering NPI, Ordering Provider NPI |
| referring_provider_npi | String | No | 10 | Line-level referring NPI | Referring NPI, Referring Provider NPI |
| service_facility_npi | String | No | 10 | Line-level facility NPI | Facility NPI, Service Facility NPI |

### Authorization and Referral

| Database Field | Data Type | Required | Max Length | Description | Common CSV Headers |
|----------------|-----------|----------|------------|-------------|--------------------|
| prior_authorization_number | String | No | 50 | Prior auth number | Prior Auth, Authorization Number, Auth Number, PA Number, Prior Authorization |
| referral_number | String | No | 50 | Referral number | Referral Number, Referral, Referral # |

### Revenue Code

| Database Field | Data Type | Required | Max Length | Description | Common CSV Headers |
|----------------|-----------|----------|------------|-------------|--------------------|
| revenue_code | String | No | 4 | Revenue center code (institutional) | Revenue Code, Rev Code, Revenue |

### NDC (National Drug Code)

| Database Field | Data Type | Required | Max Length | Description | Common CSV Headers |
|----------------|-----------|----------|------------|-------------|--------------------|
| ndc_code | String | No | 11 | National drug code | NDC, NDC Code, Drug Code, National Drug Code |
| ndc_unit_count | Decimal | No | - | NDC unit count | NDC Units, NDC Quantity, Drug Units |
| ndc_measurement_unit | String | No | 2 | NDC measurement unit | NDC Unit, Drug Unit, Unit Type |

### Diagnosis Pointers

| Database Field | Data Type | Required | Max Length | Description | Common CSV Headers |
|----------------|-----------|----------|------------|-------------|--------------------|
| diagnosis_code_pointer_1 | Integer | No | - | Points to diagnosis 1-12 | Diagnosis Pointer 1, DX Pointer 1, Pointer 1, Diag Ptr 1 |
| diagnosis_code_pointer_2 | Integer | No | - | Points to diagnosis 1-12 | Diagnosis Pointer 2, DX Pointer 2, Pointer 2, Diag Ptr 2 |
| diagnosis_code_pointer_3 | Integer | No | - | Points to diagnosis 1-12 | Diagnosis Pointer 3, DX Pointer 3, Pointer 3, Diag Ptr 3 |
| diagnosis_code_pointer_4 | Integer | No | - | Points to diagnosis 1-12 | Diagnosis Pointer 4, DX Pointer 4, Pointer 4, Diag Ptr 4 |

### Notes and Status

| Database Field | Data Type | Required | Max Length | Description | Common CSV Headers |
|----------------|-----------|----------|------------|-------------|--------------------|
| line_note | String | No | 255 | Line-level note | Line Note, Note, Comments, Service Note |
| line_status | String | Yes | 20 | Line status | Line Status, Status |

---

## Diagnosis Fields

**Table**: `claims.encounter_diagnosis`

**Description**: Diagnosis codes linked to the claim. Can have multiple diagnoses per encounter (typically 1-12).

| Database Field | Data Type | Required | Max Length | Description | Common CSV Headers |
|----------------|-----------|----------|------------|-------------|--------------------|
| sequence_number | Integer | Yes | - | Position (1-12) | Diagnosis Sequence, DX Sequence, Sequence, Position |
| diagnosis_code | String | Yes | 7 | ICD-10 diagnosis code | Diagnosis 1, DX1, Diagnosis, Diagnosis Code, ICD10, ICD-10, Primary Diagnosis |
| diagnosis_code_qualifier | String | No | 3 | Code qualifier (ABK, ABF) | Diagnosis Qualifier, DX Qualifier, Qualifier |
| diagnosis_description | String | No | 255 | Diagnosis description | Diagnosis Description, DX Description, Description |
| is_principal | Boolean | No | - | Principal diagnosis flag | Is Principal, Principal, Principal Diagnosis |
| is_admitting | Boolean | No | - | Admitting diagnosis flag | Is Admitting, Admitting, Admitting Diagnosis |
| is_external_cause | Boolean | No | - | External cause flag | External Cause, E-Code |
| is_patient_reason | Boolean | No | - | Patient reason for visit | Patient Reason, Reason for Visit |
| present_on_admission_indicator | String | No | 1 | POA indicator (Y/N/U/W) | POA, Present on Admission, POA Indicator |
| hcc_indicator | Boolean | No | - | HCC flag | HCC, HCC Indicator, Risk Adjustment |
| hcc_category | String | No | 10 | HCC category | HCC Category, HCC Code, RAF Category |

### Common Diagnosis CSV Header Patterns

For multiple diagnoses, CSVs typically use numbered columns:

| Position | Common Headers |
|----------|---------------|
| 1st Diagnosis | Diagnosis 1, DX1, Primary Diagnosis, Diagnosis, Principal Diagnosis |
| 2nd Diagnosis | Diagnosis 2, DX2, Secondary Diagnosis |
| 3rd Diagnosis | Diagnosis 3, DX3 |
| 4th Diagnosis | Diagnosis 4, DX4 |
| 5th Diagnosis | Diagnosis 5, DX5 |
| 6th Diagnosis | Diagnosis 6, DX6 |
| 7th Diagnosis | Diagnosis 7, DX7 |
| 8th Diagnosis | Diagnosis 8, DX8 |
| 9th Diagnosis | Diagnosis 9, DX9 |
| 10th Diagnosis | Diagnosis 10, DX10 |
| 11th Diagnosis | Diagnosis 11, DX11 |
| 12th Diagnosis | Diagnosis 12, DX12 |

---

## Organization Fields

**Table**: `master.organization`

**Description**: Organization-level information. Usually not in claim CSVs, but can be provided for reference.

| Database Field | Data Type | Required | Max Length | Description | Common CSV Headers |
|----------------|-----------|----------|------------|-------------|--------------------|
| organization_code | String | Yes | 20 | Organization code | Organization Code, Org Code, Organization ID |
| organization_name | String | Yes | 100 | Organization name | Organization Name, Organization, Org Name, Company Name |
| tax_id | String | No | 12 | Organization tax ID | Tax ID, TIN, EIN, Organization Tax ID |
| npi | String | No | 10 | Organization NPI | Organization NPI, Org NPI |
| address_line1 | String | No | 100 | Address line 1 | Organization Address, Org Address 1, Address 1 |
| address_line2 | String | No | 100 | Address line 2 | Org Address 2, Address 2 |
| city | String | No | 50 | City | Organization City, Org City, City |
| state_code | String | No | 2 | State code | Organization State, Org State, State |
| postal_code | String | No | 15 | Zip code | Organization Zip, Org Zip, Zip |
| country_code | String | No | 3 | Country code | Country |
| phone | String | No | 20 | Phone number | Organization Phone, Org Phone, Phone |
| email | String | No | 100 | Email address | Organization Email, Org Email, Email |

---

## Facility Fields

**Table**: `master.facility`

**Description**: Facility-level information. Sometimes included in claim CSVs for facility identification.

| Database Field | Data Type | Required | Max Length | Description | Common CSV Headers |
|----------------|-----------|----------|------------|-------------|--------------------|
| facility_code | String | Yes | 20 | Facility code | Facility Code, Fac Code, Facility ID |
| facility_name | String | Yes | 100 | Facility name | Facility Name, Facility, Fac Name |
| npi | String | No | 10 | Facility NPI | Facility NPI, Fac NPI |
| tax_id | String | No | 12 | Facility tax ID | Facility Tax ID, Fac TIN |
| facility_type | String | No | 20 | Facility type | Facility Type, Type |
| address_line1 | String | No | 100 | Address line 1 | Facility Address, Fac Address 1 |
| address_line2 | String | No | 100 | Address line 2 | Fac Address 2 |
| city | String | No | 50 | City | Facility City, Fac City |
| state_code | String | No | 2 | State code | Facility State, Fac State |
| postal_code | String | No | 15 | Zip code | Facility Zip, Fac Zip |
| country_code | String | No | 3 | Country code | Country |
| phone | String | No | 20 | Phone number | Facility Phone, Fac Phone |
| email | String | No | 100 | Email address | Facility Email, Fac Email |
| ehr_system | String | No | 50 | EHR system name | EHR System, EHR, EMR System |

---

## Provider Fields

**Table**: `master.provider`

**Description**: Provider master information. Usually referenced by NPI in claims, but full provider data can be imported separately.

| Database Field | Data Type | Required | Max Length | Description | Common CSV Headers |
|----------------|-----------|----------|------------|-------------|--------------------|
| npi | String | Yes | 10 | Provider NPI | NPI, Provider NPI |
| provider_type | String | Yes | 20 | Individual/Organization | Provider Type, Type |
| last_name | String | Yes | 60 | Last name | Provider Last Name, Last Name, Surname |
| first_name | String | Yes | 35 | First name | Provider First Name, First Name |
| middle_name | String | No | 25 | Middle name | Provider Middle Name, Middle Name |
| name_suffix | String | No | 10 | Name suffix | Provider Suffix, Suffix |
| taxonomy_code | String | No | 10 | Taxonomy code | Taxonomy, Taxonomy Code, Specialty Code |
| license_number | String | No | 20 | License number | License Number, License #, License |
| license_state | String | No | 2 | License state | License State |
| specialty | String | No | 50 | Specialty | Specialty, Provider Specialty |
| provider_group | String | No | 50 | Provider group | Provider Group, Group |
| address_line1 | String | No | 100 | Address line 1 | Provider Address, Address 1 |
| address_line2 | String | No | 100 | Address line 2 | Address 2 |
| city | String | No | 50 | City | Provider City, City |
| state_code | String | No | 2 | State code | Provider State, State |
| postal_code | String | No | 15 | Zip code | Provider Zip, Zip |
| country_code | String | No | 3 | Country code | Country |
| phone | String | No | 20 | Phone number | Provider Phone, Phone |
| email | String | No | 100 | Email address | Provider Email, Email |

---

## Payer Fields

**Table**: Reference data, typically stored in `master.payer` (if exists)

**Description**: Payer information. Usually referenced by payer ID in claims.

| Field Concept | Common CSV Headers |
|---------------|-------------------|
| Payer ID | Payer ID, Insurance ID, Payer Code, Insurance Company ID |
| Payer Name | Payer Name, Insurance Name, Insurance Company, Payer, Insurance |
| Payer Address | Payer Address, Insurance Address |
| Payer City | Payer City |
| Payer State | Payer State |
| Payer Zip | Payer Zip, Payer Postal Code |
| Payer Phone | Payer Phone |
| Payer Type | Payer Type, Insurance Type |
| Plan Name | Plan Name, Insurance Plan |
| Group Number | Group Number, Group ID, Group # |

---

## Import Batch Tracking

**Table**: `staging.import_batch`

**Description**: Metadata about the import process. Usually auto-generated, not in CSV.

These fields track the import process and are typically NOT provided in CSV files:

| Database Field | Description |
|----------------|-------------|
| batch_id | Unique batch identifier (auto-generated) |
| batch_name | Name of the import batch |
| batch_type | Type of import (CSV, EDI, HL7, etc.) |
| file_format | File format (CSV, X12, etc.) |
| original_filename | Original filename |
| file_path | Path to imported file |
| file_size_bytes | File size in bytes |
| file_hash | SHA-256 hash for duplicate detection |
| import_status | Status (Queued, Processing, Completed, Failed) |
| total_records | Total records in file |
| processed_records | Records processed |
| successful_records | Successfully imported records |
| failed_records | Failed records |
| skipped_records | Skipped records |
| duplicate_records | Duplicate records |
| started_at | Import start timestamp |
| completed_at | Import completion timestamp |
| processing_duration_seconds | Processing duration |
| error_message | Error message if failed |

---

## Quick Reference: Common Header Variations

### Most Common Required Fields

These are the minimum required fields for a basic claim import:

| Purpose | Field | Common Headers |
|---------|-------|---------------|
| Claim ID | patient_control_number | Patient ID, Account, Claim |
| Patient Name | subscriber_last_name | Last Name, Patient Last Name |
| Patient Name | subscriber_first_name | First Name, Patient First Name |
| Patient DOB | subscriber_birth_date | DOB, Date of Birth |
| Service Date | date_of_service_from | DOS, Service Date, Date of Service |
| Procedure | procedure_code | CPT, CPT Code, Procedure Code |
| Charge | line_item_charge_amount | Charges, Charge Amount, Billed |
| Diagnosis | diagnosis_code | Diagnosis 1, DX1, Diagnosis |

### Provider Identifiers

| Purpose | Field | Common Headers |
|---------|-------|---------------|
| Rendering Provider | rendering_provider_npi | Provider NPI, Rendering NPI, NPI |
| Billing Provider | billing_provider_npi | Billing NPI, Billing Provider NPI |
| Referring Provider | referring_provider_npi | Referring NPI, Referring Provider NPI |
| Facility | service_facility_npi | Facility NPI, Service Facility NPI |

### Modifiers

| Position | Common Headers |
|----------|---------------|
| 1st Modifier | Modifier 1, Mod1, Modifier, Mod |
| 2nd Modifier | Modifier 2, Mod2 |
| 3rd Modifier | Modifier 3, Mod3 |
| 4th Modifier | Modifier 4, Mod4 |

### Units and Quantities

| Purpose | Field | Common Headers |
|---------|-------|---------------|
| Service Units | service_unit_count | Units, Unit Count, Qty, Quantity |
| NDC Units | ndc_unit_count | NDC Units, Drug Units, Drug Quantity |

### Amounts

| Purpose | Field | Common Headers |
|---------|-------|---------------|
| Line Charge | line_item_charge_amount | Charges, Charge Amount, Billed, Amount |
| Total Claim | total_claim_charge_amount | Total Charges, Claim Total, Total Amount |

### Codes

| Purpose | Field | Common Headers |
|---------|-------|---------------|
| Place of Service | place_of_service_code | POS, Place of Service, Place |
| Revenue Code | revenue_code | Revenue Code, Rev Code, Revenue |
| NDC | ndc_code | NDC, NDC Code, Drug Code |

---

## Data Type Reference

| Data Type | Description | Example Values |
|-----------|-------------|----------------|
| String | Text data | "John Smith", "99213" |
| Integer | Whole numbers | 1, 42, 100 |
| Decimal | Numbers with decimals | 150.00, 1.5, 99.99 |
| Date | Date only (YYYY-MM-DD or MM/DD/YYYY) | 2024-01-15, 01/15/2024 |
| DateTime | Date and time | 2024-01-15 14:30:00 |
| Boolean | True/false | true, false, 1, 0, Y, N |
| UUID | Universally unique identifier | 550e8400-e29b-41d4-a716-446655440000 |

---

## Notes

### Field Naming Conventions

- **encounter** = claim-level data (one per claim)
- **service_line** = procedure-level data (one or more per claim)
- **encounter_diagnosis** = diagnosis-level data (one or more per claim)
- **_npi** suffix = National Provider Identifier (10 digits)
- **_code** suffix = Usually a short code (2-5 characters)
- **_date** suffix = Date field
- **_amount** suffix = Currency/decimal field
- **_id** suffix = Identifier field (usually UUID internally)

### Required vs Optional

- **Required** = Must be present in CSV for successful import
- **Optional** = Can be blank or omitted
- Fields marked as required may have defaults if not provided in some cases

### Multiple Rows Per Claim

Some fields can have multiple values per claim:
- **Service Lines**: Multiple procedures per claim (separate rows or columns)
- **Diagnoses**: Multiple diagnosis codes per claim (typically Diagnosis 1-12 columns)
- **Modifiers**: Up to 4 modifiers per procedure (Modifier 1-4 columns)

### Auto-Generated Fields

These are typically NOT in CSV files and are generated by the system:
- UUIDs (_id fields)
- Timestamps (created_at, updated_at)
- Audit fields (created_by, updated_by)
- Calculated fields (processing metrics)

---

## See Also

- [CSV Mapping Guide](CSV_MAPPING_GUIDE.md) - How to configure mappings
- [Field Mapping Reference](FIELD_MAPPING_REFERENCE.md) - Detailed field documentation
- [CSV Templates](examples/) - Download sample CSV files

---

**Document Version**: 1.0
**Last Updated**: 2025-01-15
**Total Fields Documented**: 150+
