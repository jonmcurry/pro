# Rules Engine Fields Reference

Complete reference for all available fields that can be used in the Professional SMART rules engine, with practical examples and sample rules.

**Last Updated**: 2025-11-07
**Version**: 1.8.0

---

## Table of Contents

1. [Overview](#overview)
2. [Field Access by Rule Template](#field-access-by-rule-template)
3. [Encounter-Level Fields](#encounter-level-fields)
4. [Service-Line-Level Fields](#service-line-level-fields)
5. [Provider Context Fields](#provider-context-fields)
6. [Sample Rules by Use Case](#sample-rules-by-use-case)
7. [Field Availability Matrix](#field-availability-matrix)
8. [Best Practices](#best-practices)

---

## Overview

The Professional SMART rules engine executes rules at two levels:
- **ENCOUNTER level**: Rules that analyze the entire claim/encounter
- **SERVICE_LINE level**: Rules that analyze individual procedure lines
- **BOTH**: Rules that can run at either level

Each rule has access to different fields depending on its execution level and the data context.

### Rule Execution Context

All rules receive a `RuleExecutionContext` which contains:

```rust
pub struct RuleExecutionContext {
    // Core identifiers
    pub organization_id: i64,
    pub facility_id: Option<i64>,
    pub encounter_id: Option<i64>,
    pub service_line_id: Option<i64>,
    pub provider_id: Option<i64>,
    pub coder_id: Option<i64>,

    // Service line data (when execution_level = SERVICE_LINE)
    pub procedure_code: Option<String>,
    pub modifiers: Vec<String>,
    pub service_unit_count: Option<Decimal>,
    pub line_item_charge_amount: Option<Decimal>,
    pub date_of_service: Option<NaiveDate>,
    pub place_of_service_code: Option<String>,

    // Encounter data (when execution_level = ENCOUNTER)
    pub total_claim_charge_amount: Option<Decimal>,
    pub date_of_service_from: Option<NaiveDate>,
    pub date_of_service_to: Option<NaiveDate>,

    // Provider context
    pub provider_type: Option<String>,
    pub provider_specialty: Option<String>,

    // Diagnosis codes (available at both levels)
    pub diagnosis_codes: Vec<String>,
}
```

---

## Field Access by Rule Template

### THRESHOLD Template

**Available Fields**:
- `units` (service_unit_count)
- `line_item_charge_amount` (service line charge)
- `total_claim_charge_amount` (total encounter charge)

**Parameters**:
```json
{
  "field": "total_claim_charge_amount",
  "operator": ">",
  "threshold": 10000
}
```

**Operators**: `>`, `>=`, `<`, `<=`, `==`, `!=`

### MISSING_FIELD Template

**Available Fields**:
```
Service Line Fields:
- procedure_code
- service_unit_count
- line_item_charge_amount
- date_of_service
- place_of_service_code

Encounter Fields:
- encounter_id
- facility_id
- provider_id
- total_claim_charge_amount
- date_of_service_from
- date_of_service_to

Provider Fields:
- provider_type
- provider_specialty
```

**Parameters**:
```json
{
  "fields": ["procedure_code", "service_unit_count"],
  "check_empty": true
}
```

### FIELD_PATTERN Template

**Available Fields**: Any string field in context

**Parameters**:
```json
{
  "field": "procedure_code",
  "pattern": "^9921[0-5]$",
  "negate": false
}
```

### DUPLICATE Template

**Available Fields**: Any field combination for matching

**Parameters**:
```json
{
  "match_fields": ["procedure_code", "provider_id", "date_of_service"],
  "timeframe_days": 7
}
```

### CROSS_FIELD Template

**Available Fields**: Any two numeric or date fields

**Parameters**:
```json
{
  "field1": "date_of_service",
  "operator": ">",
  "field2": "date_of_birth"
}
```

---

## Encounter-Level Fields

These fields contain claim/encounter-level data (one record per claim).

### Patient/Subscriber Information

| Field Name | Type | Description | Example |
|------------|------|-------------|---------|
| `subscriber_id` | String | Insurance member ID | "ABC123456789" |
| `subscriber_last_name` | String | Patient last name | "Smith" |
| `subscriber_first_name` | String | Patient first name | "John" |
| `subscriber_middle_name` | String | Patient middle name | "Robert" |
| `subscriber_birth_date` | Date | Patient date of birth | 1980-05-12 |
| `subscriber_gender` | String(1) | Patient gender (M/F/U) | "M" |
| `subscriber_member_id` | String | Member ID on insurance card | "987654321" |
| `subscriber_group_number` | String | Insurance group number | "GRP-001" |
| `subscriber_address_line1` | String | Patient street address | "123 Main St" |
| `subscriber_address_line2` | String | Patient address line 2 | "Apt 4B" |
| `subscriber_city` | String | Patient city | "Springfield" |
| `subscriber_state` | String(2) | Patient state | "IL" |
| `subscriber_postal_code` | String | Patient ZIP code | "62701" |

### Claim Identifiers

| Field Name | Type | Description | Example |
|------------|------|-------------|---------|
| `patient_control_number` | String | Unique claim identifier | "CLM-12345-2024" |
| `submitter_id` | String | Submitter/clearinghouse ID | "SUBM001" |
| `transaction_set_control_number` | String | 837 transaction control # | "0001" |

### Service Dates

| Field Name | Type | Description | Example |
|------------|------|-------------|---------|
| `date_of_service_from` | Date | Service start date (primary) | 2024-01-15 |
| `date_of_service_to` | Date | Service end date | 2024-01-15 |
| `onset_of_illness_date` | Date | Illness onset date | 2024-01-10 |
| `initial_treatment_date` | Date | First treatment date | 2024-01-11 |
| `last_seen_date` | Date | Last seen date | 2024-01-15 |
| `acute_manifestation_date` | Date | Acute symptom date | 2024-01-12 |
| `accident_date` | Date | Date of accident | 2024-01-09 |
| `last_menstrual_period_date` | Date | LMP date (OB) | 2023-10-01 |
| `last_xray_date` | Date | Last X-ray date | 2024-01-05 |
| `prescription_date` | Date | Prescription written date | 2024-01-15 |
| `disability_from_date` | Date | Disability start | 2024-01-10 |
| `disability_to_date` | Date | Disability end | 2024-02-10 |
| `last_worked_date` | Date | Last day worked | 2024-01-09 |
| `authorized_return_to_work_date` | Date | RTW authorization date | 2024-02-11 |
| `admission_date` | Date | Inpatient admission date | 2024-01-15 |
| `discharge_date` | Date | Inpatient discharge date | 2024-01-17 |
| `assumed_care_date` | Date | Assumed care date | 2024-01-15 |
| `relinquished_care_date` | Date | Relinquished care date | 2024-01-17 |

### Financial Fields

| Field Name | Type | Description | Example |
|------------|------|-------------|---------|
| `total_claim_charge_amount` | Decimal | Total charges for claim | 1500.00 |
| `patient_amount_paid` | Decimal | Amount paid by patient | 50.00 |
| `other_payer_paid_amount` | Decimal | Amount paid by other payer | 0.00 |

### Claim Status and Indicators

| Field Name | Type | Description | Example Values |
|------------|------|-------------|----------------|
| `claim_status` | String | Current claim status | "NEW", "PENDING", "FLAGGED", "REVIEWED", "ACCEPTED", "REJECTED" |
| `claim_frequency_code` | Char | Claim type indicator | "1" (Original), "7" (Replacement), "8" (Void) |
| `case_status` | String | Case status | "OPEN", "CLOSED" |
| `financial_class` | String | Financial classification | "COMMERCIAL", "MEDICARE" |
| `signature_indicator` | Boolean | Signature on file | true/false |
| `assignment_indicator` | Boolean | Accept assignment | true/false |
| `benefits_assignment_indicator` | Char | Benefits assignment | "Y", "N" |
| `release_of_information_code` | String | Release of info code | "Y", "I" |
| `patient_signature_code` | String | Patient signature status | "P", "B" |

### Claim Details

| Field Name | Type | Description | Example Values |
|------------|------|-------------|----------------|
| `place_of_service_code` | String(2) | Where service rendered | "11" (Office), "21" (Inpatient), "22" (Outpatient), "23" (ER) |
| `claim_filing_indicator` | String(2) | Insurance type | "CI" (Commercial), "MB" (Medicare B), "MC" (Medicaid) |
| `service_authorization_code` | String | Prior authorization # | "PA-123456" |
| `special_program_code` | String | Special program | "03" (EPSDT) |
| `delay_reason_code` | String | Reason for delay | "1" through "11" |

### Payer Information

| Field Name | Type | Description | Example |
|------------|------|-------------|---------|
| `payer_responsibility_code` | Char | Payer level | "P" (Primary), "S" (Secondary), "T" (Tertiary) |
| `payer_id` | String | Primary payer ID | "12345" |
| `payer_name` | String | Primary payer name | "Blue Cross Blue Shield" |
| `other_payer_id` | String | Secondary payer ID | "67890" |
| `other_payer_name` | String | Secondary payer name | "Medicare" |
| `other_payer_claim_number` | String | Other payer's claim # | "1234567890123" |
| `other_payer_claim_filing_indicator` | String | Other payer type | "MB" |

### Provider Information (Claim-Level)

| Field Name | Type | Description | Example |
|------------|------|-------------|---------|
| `billing_provider_id` | i64 | Billing provider FK | 123 |
| `billing_provider_npi` | String(10) | Billing provider NPI | "1234567890" |
| `billing_provider_name` | String | Billing provider name | "Springfield Medical Group" |
| `billing_provider_tax_id` | String | Tax ID (EIN/SSN) | "12-3456789" |
| `rendering_provider_id` | i64 | Rendering provider FK | 456 |
| `rendering_provider_npi` | String(10) | Rendering provider NPI | "9876543210" |
| `rendering_provider_name` | String | Rendering provider name | "Dr. John Smith" |
| `referring_provider_id` | i64 | Referring provider FK | 789 |
| `referring_provider_npi` | String(10) | Referring provider NPI | "5555555555" |
| `referring_provider_name` | String | Referring provider name | "Dr. Jane Doe" |
| `supervising_provider_id` | i64 | Supervising provider FK | 234 |
| `supervising_provider_npi` | String(10) | Supervising provider NPI | "1111111111" |
| `supervising_provider_name` | String | Supervising provider name | "Dr. Mary Johnson" |

### Service Facility Information

| Field Name | Type | Description | Example |
|------------|------|-------------|---------|
| `service_facility_id` | i64 | Service facility FK | 1 |
| `service_facility_npi` | String(10) | Facility NPI | "1234567890" |
| `service_facility_name` | String | Facility name | "Springfield Hospital" |
| `service_facility_address_line1` | String | Facility address | "100 Hospital Dr" |
| `service_facility_address_line2` | String | Facility address line 2 | "Suite 200" |
| `service_facility_city` | String | Facility city | "Springfield" |
| `service_facility_state` | String(2) | Facility state | "IL" |
| `service_facility_postal_code` | String | Facility ZIP | "62701" |

### Ambulance Information (Claim-Level)

| Field Name | Type | Description | Example |
|------------|------|-------------|---------|
| `ambulance_transport_reason_code` | String | Reason for transport | "A" |
| `ambulance_transport_distance` | Decimal | Miles transported | 15.5 |
| `ambulance_patient_weight` | Decimal | Patient weight (lbs) | 180.0 |
| `ambulance_patient_count` | i32 | Number of patients | 1 |

### Audit Trail

| Field Name | Type | Description | Example |
|------------|------|-------------|---------|
| `coder_id` | i64 | Coder who worked claim | 5 |
| `coding_date` | Date | Date coded | 2024-01-16 |
| `import_batch_id` | i64 | Import batch FK | 42 |
| `import_date` | Date | Date imported | 2024-01-15 |
| `is_active` | Boolean | Active record | true |
| `soft_deleted` | Boolean | Soft deleted | false |
| `created_at` | Timestamp | Record created | 2024-01-15 10:30:00 |
| `updated_at` | Timestamp | Last updated | 2024-01-15 11:00:00 |
| `created_by` | String | Created by user | "admin" |
| `updated_by` | String | Updated by user | "coder1" |

---

## Service-Line-Level Fields

These fields contain procedure/service-level data (multiple records per claim).

### Core Service Line Fields

| Field Name | Type | Description | Required | Example |
|------------|------|-------------|----------|---------|
| `service_line_id` | i64 | Service line PK | Yes | 1001 |
| `encounter_id` | i64 | Parent encounter FK | Yes | 500 |
| `line_number` | i32 | Line # on claim (1-50) | Yes | 1 |

### Procedure/Service Information

| Field Name | Type | Description | Required | Example |
|------------|------|-------------|----------|---------|
| `procedure_code` | String(5) | CPT/HCPCS code | Yes | "99213" |
| `procedure_modifier_1` | String(2) | First modifier | No | "25" |
| `procedure_modifier_2` | String(2) | Second modifier | No | "59" |
| `procedure_modifier_3` | String(2) | Third modifier | No | "GT" |
| `procedure_modifier_4` | String(2) | Fourth modifier | No | "76" |
| `procedure_description` | Text | Procedure description | No | "Office Visit Level 3" |
| `product_service_id_qualifier` | String(2) | Code set qualifier | No | "HC" (HCPCS) |
| `unit_basis_measurement_code` | String(2) | Unit type | No | "UN" (units), "MJ" (minutes) |

**Common Modifiers**:
- `25` - Significant, separately identifiable E/M service
- `26` - Professional component
- `50` - Bilateral procedure
- `59` - Distinct procedural service
- `76` - Repeat procedure by same physician
- `77` - Repeat procedure by another physician
- `GT` - Via interactive telecommunications
- `TC` - Technical component
- `LT` - Left side
- `RT` - Right side

### Financial Fields (Line-Level)

| Field Name | Type | Description | Required | Example |
|------------|------|-------------|----------|---------|
| `line_item_charge_amount` | Decimal | Charge for service | Yes | 150.00 |
| `service_unit_count` | Decimal | Number of units | Yes | 1.0 |

**Valid Range**: 1.0 - 9999.9 units

### Date Fields (Line-Level)

| Field Name | Type | Description | Required | Example |
|------------|------|-------------|----------|---------|
| `service_date_from` | Date | Service start date | Yes | 2024-01-15 |
| `service_date_to` | Date | Service end date | No | 2024-01-15 |

### Service Line Details

| Field Name | Type | Description | Example |
|------------|------|-------------|---------|
| `place_of_service_code` | String(2) | POS (overrides encounter) | "11" |
| `revenue_code` | String(4) | Revenue center code | "0450" (ER) |
| `line_note` | Text | Line-level note | "Patient fell" |
| `prior_authorization_number` | String | PA number for this line | "PA-789456" |
| `referral_number` | String | Referral number | "REF-12345" |

**Common Revenue Codes**:
- `0450` - Emergency room
- `0360` - Operating room
- `0250` - Pharmacy
- `0730` - EKG/ECG
- `0260` - IV therapy

### Diagnosis Pointers

| Field Name | Type | Description | Example |
|------------|------|-------------|---------|
| `diagnosis_code_pointer_1` | i16 | Primary diagnosis pointer | 1 |
| `diagnosis_code_pointer_2` | i16 | Second diagnosis pointer | 2 |
| `diagnosis_code_pointer_3` | i16 | Third diagnosis pointer | 3 |
| `diagnosis_code_pointer_4` | i16 | Fourth diagnosis pointer | 4 |
| `diagnosis_code_pointer_5` through `_12` | i16 | Additional pointers | 5-12 |

**Note**: Pointers reference the diagnosis sequence number (1-12) on the encounter.

### Provider Information (Line-Level)

| Field Name | Type | Description | Example |
|------------|------|-------------|---------|
| `rendering_provider_id` | i64 | Rendering provider FK | 456 |
| `rendering_provider_npi` | String(10) | Rendering NPI | "9876543210" |
| `supervising_provider_id` | i64 | Supervising provider FK | 234 |
| `supervising_provider_npi` | String(10) | Supervising NPI | "1111111111" |
| `ordering_provider_id` | i64 | Ordering provider FK | 567 |
| `ordering_provider_npi` | String(10) | Ordering NPI | "2222222222" |
| `referring_provider_id` | i64 | Referring provider FK | 789 |
| `referring_provider_npi` | String(10) | Referring NPI | "5555555555" |
| `service_facility_id` | i64 | Service facility FK | 1 |
| `service_facility_npi` | String(10) | Facility NPI | "1234567890" |

### Service Line Indicators

| Field Name | Type | Description | Example |
|------------|------|-------------|---------|
| `emergency_indicator` | Boolean | Emergency service | true |
| `epsdt_indicator` | Boolean | EPSDT service | false |
| `family_planning_indicator` | Boolean | Family planning | false |

### Drug Information (NDC)

| Field Name | Type | Description | Example |
|------------|------|-------------|---------|
| `ndc_code` | String(11) | National Drug Code | "12345-678-90" |
| `ndc_unit_count` | Decimal | Drug quantity | 30.0 |
| `ndc_measurement_unit` | String(2) | Unit of measure | "UN", "ML", "GR" |

**Common NDC Units**:
- `UN` - Unit
- `ML` - Milliliter
- `GR` - Gram
- `F2` - International Unit

### DME (Durable Medical Equipment)

| Field Name | Type | Description | Example |
|------------|------|-------------|---------|
| `dme_rental_price` | Decimal | Monthly rental price | 150.00 |
| `dme_purchase_price` | Decimal | Purchase price | 1800.00 |
| `dme_frequency_code` | String | Rental frequency | "1" (Monthly) |

### Anesthesia

| Field Name | Type | Description | Example |
|------------|------|-------------|---------|
| `anesthesia_minutes` | i32 | Anesthesia time | 120 |
| `obstetric_additional_units` | Decimal | OB additional units | 0.0 |

### Test Results

| Field Name | Type | Description | Example |
|------------|------|-------------|---------|
| `test_result_value` | Decimal | Test result value | 98.6 |
| `test_result_measurement_code` | String | Unit of measure | "DG" (Degrees) |

### Ambulance Information (Line-Level)

| Field Name | Type | Description | Example |
|------------|------|-------------|---------|
| `ambulance_patient_count` | i32 | Number of patients | 1 |
| `ambulance_transport_distance` | Decimal | Distance in miles | 15.5 |
| `ambulance_patient_weight` | Decimal | Patient weight (lbs) | 180.0 |

### Coordination of Benefits (COB)

| Field Name | Type | Description | Example |
|------------|------|-------------|---------|
| `other_payer_line_paid_amount` | Decimal | Other payer paid | 100.00 |
| `other_payer_line_service_id` | String | Other payer line ID | "LINE-001" |

### Status and Audit

| Field Name | Type | Description | Example |
|------------|------|-------------|---------|
| `line_status` | String | Line status | "ACTIVE" |
| `created_at` | Timestamp | Record created | 2024-01-15 10:30:00 |
| `updated_at` | Timestamp | Last updated | 2024-01-15 11:00:00 |
| `created_by` | String | Created by user | "admin" |
| `updated_by` | String | Updated by user | "coder1" |

---

## Provider Context Fields

These fields provide provider context when rules are executed.

| Field Name | Type | Description | Example |
|------------|------|-------------|---------|
| `provider_id` | i64 | Provider FK | 123 |
| `provider_type` | String | Provider credential | "MD", "PA", "NP", "CRNA", "PT", "OT" |
| `provider_specialty` | String | Medical specialty | "Cardiology", "Internal Medicine", "Surgery" |
| `taxonomy_code` | String(10) | Provider taxonomy | "207R00000X" |

**Common Provider Types**:
- `MD` - Medical Doctor
- `DO` - Doctor of Osteopathy
- `PA` - Physician Assistant
- `NP` - Nurse Practitioner
- `CRNA` - Certified Registered Nurse Anesthetist
- `PT` - Physical Therapist
- `OT` - Occupational Therapist
- `SLP` - Speech Language Pathologist

**Common Specialties**:
- `Internal Medicine`
- `Family Practice`
- `Cardiology`
- `Orthopedic Surgery`
- `Emergency Medicine`
- `Anesthesiology`
- `Radiology`

---

## Sample Rules by Use Case

### 1. High-Value Claims Without Authorization

**Use Case**: Flag claims over $10,000 without prior authorization

**Template**: THRESHOLD
**Execution Level**: ENCOUNTER

```sql
INSERT INTO claims.rule_definition (
    rule_code, rule_name, rule_description,
    template_id, rule_parameters_encrypted,
    flag_issue_id, execution_order, execution_level,
    default_severity, is_active
) VALUES (
    'HIGH_VALUE_NO_AUTH',
    'High Value Claims Without Authorization',
    'Flags encounters over $10,000 that lack prior authorization',
    (SELECT template_id FROM claims.rule_template WHERE template_code = 'THRESHOLD'),
    pgp_sym_encrypt('{"field": "total_claim_charge_amount", "operator": ">", "threshold": 10000}', '${RULE_ENCRYPTION_KEY}'),
    (SELECT issue_id FROM claims.flag_issue WHERE issue_code = 'DOC_MISSING'),
    5,
    'ENCOUNTER',
    'HIGH',
    true
);
```

**Fields Used**:
- `total_claim_charge_amount` - Total charge for comparison
- `service_authorization_code` - Checked separately for authorization

---

### 2. Excessive Service Units

**Use Case**: Flag service lines with more than 100 units

**Template**: THRESHOLD
**Execution Level**: SERVICE_LINE

```sql
INSERT INTO claims.rule_definition (
    rule_code, rule_name, rule_description,
    template_id, rule_parameters_encrypted,
    flag_issue_id, execution_order, execution_level,
    default_severity, is_active
) VALUES (
    'EXCESSIVE_UNITS',
    'Excessive Service Units',
    'Flags service lines with unit count exceeding 100',
    (SELECT template_id FROM claims.rule_template WHERE template_code = 'THRESHOLD'),
    pgp_sym_encrypt('{"field": "units", "operator": ">", "threshold": 100}', '${RULE_ENCRYPTION_KEY}'),
    (SELECT issue_id FROM claims.flag_issue WHERE issue_code = 'QTY_FEWER'),
    20,
    'SERVICE_LINE',
    'HIGH',
    true
);
```

**Fields Used**:
- `service_unit_count` - Unit count for comparison

---

### 3. Missing NPI Numbers

**Use Case**: Flag encounters missing rendering provider NPI

**Template**: MISSING_FIELD
**Execution Level**: ENCOUNTER

```sql
INSERT INTO claims.rule_definition (
    rule_code, rule_name, rule_description,
    template_id, rule_parameters_encrypted,
    flag_issue_id, execution_order, execution_level,
    default_severity, is_active
) VALUES (
    'MISSING_PROVIDER_NPI',
    'Missing Rendering Provider NPI',
    'Flags encounters without a rendering provider NPI',
    (SELECT template_id FROM claims.rule_template WHERE template_code = 'MISSING_FIELD'),
    pgp_sym_encrypt('{"fields": ["rendering_provider_npi"], "check_empty": true}', '${RULE_ENCRYPTION_KEY}'),
    (SELECT issue_id FROM claims.flag_issue WHERE issue_code = 'OTH_PROVIDER'),
    8,
    'ENCOUNTER',
    'MEDIUM',
    true
);
```

**Fields Used**:
- `rendering_provider_npi` - Checked for presence

---

### 4. E/M Codes Without Place of Service

**Use Case**: Flag E/M procedures missing place of service code

**Template**: MISSING_FIELD + FIELD_PATTERN
**Execution Level**: SERVICE_LINE

```sql
-- First rule: Check for E/M codes
INSERT INTO claims.rule_definition (
    rule_code, rule_name, rule_description,
    template_id, rule_parameters_encrypted,
    flag_issue_id, execution_order, execution_level,
    default_severity, is_active
) VALUES (
    'EM_MISSING_POS',
    'E/M Services Missing Place of Service',
    'Flags E/M procedure codes (9920x-9929x) without place of service',
    (SELECT template_id FROM claims.rule_template WHERE template_code = 'MISSING_FIELD'),
    pgp_sym_encrypt('{"fields": ["place_of_service_code"], "check_empty": true}', '${RULE_ENCRYPTION_KEY}'),
    (SELECT issue_id FROM claims.flag_issue WHERE issue_code = 'DOC_MISSING'),
    25,
    'SERVICE_LINE',
    'MEDIUM',
    true
);
```

**Fields Used**:
- `procedure_code` - To identify E/M codes
- `place_of_service_code` - Checked for presence

---

### 5. Duplicate Services Within 7 Days

**Use Case**: Flag duplicate procedures by same provider within 7 days

**Template**: DUPLICATE
**Execution Level**: SERVICE_LINE

```sql
INSERT INTO claims.rule_definition (
    rule_code, rule_name, rule_description,
    template_id, rule_parameters_encrypted,
    flag_issue_id, execution_order, execution_level,
    default_severity, is_active
) VALUES (
    'DUPLICATE_7DAY',
    'Duplicate Service Within 7 Days',
    'Flags duplicate procedures by same provider within 7-day window',
    (SELECT template_id FROM claims.rule_template WHERE template_code = 'DUPLICATE'),
    pgp_sym_encrypt('{"match_fields": ["procedure_code", "rendering_provider_npi", "patient_control_number"], "timeframe_days": 7}', '${RULE_ENCRYPTION_KEY}'),
    (SELECT issue_id FROM claims.flag_issue WHERE issue_code = 'OTH_DUPLICATE'),
    15,
    'SERVICE_LINE',
    'HIGH',
    true
);
```

**Fields Used**:
- `procedure_code` - For matching
- `rendering_provider_npi` - For matching
- `patient_control_number` - Patient identifier
- `service_date_from` - For timeframe calculation

---

### 6. Bilateral Procedures Without Modifier 50

**Use Case**: Flag bilateral procedures missing modifier 50

**Template**: FIELD_PATTERN
**Execution Level**: SERVICE_LINE

```sql
INSERT INTO claims.rule_definition (
    rule_code, rule_name, rule_description,
    template_id, rule_parameters_encrypted,
    flag_issue_id, execution_order, execution_level,
    default_severity, is_active
) VALUES (
    'BILATERAL_MISSING_MOD50',
    'Bilateral Procedures Missing Modifier 50',
    'Flags procedures known to be bilateral without modifier 50',
    (SELECT template_id FROM claims.rule_template WHERE template_code = 'FIELD_PATTERN'),
    pgp_sym_encrypt('{"field": "procedure_code", "pattern": "^(27447|27486|27487)$", "negate": false}', '${RULE_ENCRYPTION_KEY}'),
    (SELECT issue_id FROM claims.flag_issue WHERE issue_code = 'MOD_MISSING'),
    30,
    'SERVICE_LINE',
    'MEDIUM',
    true
);
```

**Fields Used**:
- `procedure_code` - Pattern match for bilateral codes
- `procedure_modifier_1` through `_4` - Checked for "50"

---

### 7. Inpatient Services in Office Setting

**Use Case**: Flag inpatient CPT codes with office place of service

**Template**: FIELD_PATTERN + CROSS_FIELD (custom logic needed)
**Execution Level**: SERVICE_LINE

```sql
INSERT INTO claims.rule_definition (
    rule_code, rule_name, rule_description,
    template_id, rule_parameters_encrypted,
    flag_issue_id, execution_order, execution_level,
    default_severity, is_active
) VALUES (
    'INPATIENT_IN_OFFICE',
    'Inpatient Codes Billed in Office',
    'Flags inpatient procedure codes (9923x) with place of service 11 (Office)',
    (SELECT template_id FROM claims.rule_template WHERE template_code = 'FIELD_PATTERN'),
    pgp_sym_encrypt('{"field": "procedure_code", "pattern": "^9923[0-9]$", "negate": false}', '${RULE_ENCRYPTION_KEY}'),
    (SELECT issue_id FROM claims.flag_issue WHERE issue_code = 'COD_INCORRECT'),
    35,
    'SERVICE_LINE',
    'HIGH',
    true
);
```

**Fields Used**:
- `procedure_code` - For inpatient code pattern
- `place_of_service_code` - Checked for "11"

---

### 8. Charges Below Minimum Threshold

**Use Case**: Flag service lines with suspiciously low charges (under $1)

**Template**: THRESHOLD
**Execution Level**: SERVICE_LINE

```sql
INSERT INTO claims.rule_definition (
    rule_code, rule_name, rule_description,
    template_id, rule_parameters_encrypted,
    flag_issue_id, execution_order, execution_level,
    default_severity, is_active
) VALUES (
    'SUSPICIOUSLY_LOW_CHARGE',
    'Suspiciously Low Service Charge',
    'Flags service lines with charges under $1.00',
    (SELECT template_id FROM claims.rule_template WHERE template_code = 'THRESHOLD'),
    pgp_sym_encrypt('{"field": "line_item_charge_amount", "operator": "<", "threshold": 1.00}', '${RULE_ENCRYPTION_KEY}'),
    (SELECT issue_id FROM claims.flag_issue WHERE issue_code = 'OTH_GENERAL'),
    40,
    'SERVICE_LINE',
    'LOW',
    true
);
```

**Fields Used**:
- `line_item_charge_amount` - For charge comparison

---

### 9. Range-Based Charge Validation

**Use Case**: Flag charges outside acceptable range ($100-$500)

**Template**: THRESHOLD (with min/max)
**Execution Level**: SERVICE_LINE

```sql
INSERT INTO claims.rule_definition (
    rule_code, rule_name, rule_description,
    template_id, rule_parameters_encrypted,
    flag_issue_id, execution_order, execution_level,
    default_severity, is_active
) VALUES (
    'CHARGE_OUT_OF_RANGE',
    'Charge Amount Out of Expected Range',
    'Flags service charges outside $100-$500 range for specific codes',
    (SELECT template_id FROM claims.rule_template WHERE template_code = 'THRESHOLD'),
    pgp_sym_encrypt('{"field": "line_item_charge_amount", "operator": ">", "threshold": 500, "min_threshold": 100, "max_threshold": 500}', '${RULE_ENCRYPTION_KEY}'),
    (SELECT issue_id FROM claims.flag_issue WHERE issue_code = 'OTH_GENERAL'),
    45,
    'SERVICE_LINE',
    'MEDIUM',
    true
);
```

**Fields Used**:
- `line_item_charge_amount` - For range validation

---

### 10. Emergency Room Services Without Emergency Indicator

**Use Case**: Flag ER procedures without emergency indicator set

**Template**: MISSING_FIELD (custom combination needed)
**Execution Level**: SERVICE_LINE

```sql
INSERT INTO claims.rule_definition (
    rule_code, rule_name, rule_description,
    template_id, rule_parameters_encrypted,
    flag_issue_id, execution_order, execution_level,
    default_severity, is_active
) VALUES (
    'ER_MISSING_EMERGENCY_IND',
    'Emergency Room Services Without Indicator',
    'Flags ER services (POS 23) without emergency indicator',
    (SELECT template_id FROM claims.rule_template WHERE template_code = 'MISSING_FIELD'),
    pgp_sym_encrypt('{"fields": ["emergency_indicator"], "check_empty": false}', '${RULE_ENCRYPTION_KEY}'),
    (SELECT issue_id FROM claims.flag_issue WHERE issue_code = 'DOC_MISSING'),
    50,
    'SERVICE_LINE',
    'MEDIUM',
    true
);
```

**Fields Used**:
- `place_of_service_code` - Checked for "23" (ER)
- `emergency_indicator` - Checked for presence

---

## Field Availability Matrix

This table shows which fields are available at each execution level.

| Field Category | ENCOUNTER | SERVICE_LINE | Notes |
|----------------|-----------|--------------|-------|
| **Identifiers** |
| organization_id | Yes | Yes | Always available |
| facility_id | Yes | Yes | Always available |
| encounter_id | Yes | Yes | Always available |
| service_line_id | No | Yes | Only at service line level |
| provider_id | Yes | Yes | Context-dependent |
| coder_id | Yes | Yes | Context-dependent |
| **Financial** |
| total_claim_charge_amount | Yes | No | Encounter-level only |
| line_item_charge_amount | No | Yes | Service-line-level only |
| service_unit_count | No | Yes | Service-line-level only |
| patient_amount_paid | Yes | No | Encounter-level only |
| **Dates** |
| date_of_service_from | Yes | No | Encounter-level only |
| date_of_service_to | Yes | No | Encounter-level only |
| service_date_from | No | Yes | Service-line-level only |
| service_date_to | No | Yes | Service-line-level only |
| **Procedures** |
| procedure_code | No | Yes | Service-line-level only |
| modifiers | No | Yes | Service-line-level only |
| **Patient** |
| subscriber_* | Yes | No | Encounter-level only |
| **Providers** |
| rendering_provider_* | Yes | Yes | Both levels |
| billing_provider_* | Yes | No | Encounter-level only |
| referring_provider_* | Yes | Yes | Both levels |
| supervising_provider_* | Yes | Yes | Both levels |
| ordering_provider_* | No | Yes | Service-line-level only |
| **Context** |
| provider_type | Yes | Yes | Both levels |
| provider_specialty | Yes | Yes | Both levels |
| place_of_service_code | Yes | Yes | Can be at both levels |
| diagnosis_codes | Yes | Yes | Available at both levels |

---

## Best Practices

### 1. Choose the Correct Execution Level

**ENCOUNTER Level**: Use when the rule analyzes claim-wide data
- Total charge validation
- Patient demographics checks
- Missing claim-level fields
- Authorization checks

**SERVICE_LINE Level**: Use when the rule analyzes individual procedures
- Procedure code validation
- Modifier checks
- Per-service charge validation
- Duplicate procedure detection

**BOTH**: Rarely used - for rules that need to run at both levels

### 2. Use Appropriate Field Names

Always reference fields by their exact database column names:
- `total_claim_charge_amount` (not `total_charge`)
- `line_item_charge_amount` (not `line_charge`)
- `service_unit_count` (not `units` - though templates may map this)

### 3. Consider NULL Values

Many fields are optional. Rules should handle NULL gracefully:
- Use `check_empty: true` in MISSING_FIELD rules to catch both NULL and empty strings
- Threshold rules automatically skip NULL values
- Write rules that assume fields may not be present

### 4. Optimize Execution Order

Lower execution order = runs first:
- **1-10**: Fast, critical rules (missing required fields)
- **11-30**: Moderate complexity rules (duplicate detection)
- **31-50**: Standard validation rules
- **51-90**: Complex rules (cross-field validation)
- **91-99**: Low-priority rules

### 5. Use Descriptive Rule Codes

Good rule codes are:
- Uppercase with underscores: `HIGH_VALUE_NO_AUTH`
- Descriptive: Explain what they check
- Unique: No duplicates
- Consistent: Follow naming patterns

### 6. Document Your Rules

Always provide:
- Clear `rule_name`
- Detailed `rule_description`
- Appropriate `default_severity` (HIGH/MEDIUM/LOW)

### 7. Test with Real Data

Before activating globally:
1. Test on a single facility first
2. Use `facility_rule_assignment` to enable for test facility
3. Monitor `rule_execution_stats` for performance
4. Check `flag_triggered_count` to verify it's working
5. Gradually roll out to more facilities

### 8. Use Parameter Overrides Wisely

Different facilities may need different thresholds:
```sql
-- Global default: $10,000
INSERT INTO claims.rule_definition (...)
VALUES (..., pgp_sym_encrypt('{"threshold": 10000}', 'key'), ...);

-- Facility 123 override: $15,000
INSERT INTO claims.facility_rule_assignment (facility_id, rule_id, parameter_overrides_encrypted)
VALUES (123, rule_id, pgp_sym_encrypt('{"threshold": 15000}', 'key'));
```

### 9. Monitor Rule Performance

Regularly check execution statistics:
```sql
SELECT
    rd.rule_code,
    res.execution_count,
    res.flag_triggered_count,
    res.avg_execution_time_ms,
    ROUND(100.0 * res.flag_triggered_count / NULLIF(res.execution_count, 0), 2) AS trigger_rate_pct
FROM claims.rule_execution_stats res
JOIN claims.rule_definition rd ON res.rule_id = rd.rule_id
WHERE res.stat_date >= CURRENT_DATE - INTERVAL '7 days'
ORDER BY res.avg_execution_time_ms DESC;
```

### 10. Keep Encryption Keys Safe

- Store `RULE_ENCRYPTION_KEY` in environment variables
- Never commit keys to version control
- Use different keys per environment
- Rotate keys periodically
- Document key rotation procedures

---

## Related Documentation

- [RULE_CONFIGURATION_GUIDE.md](RULE_CONFIGURATION_GUIDE.md) - Complete rule configuration guide
- [FACILITY_RULE_CONFIGURATION_GUIDE.md](FACILITY_RULE_CONFIGURATION_GUIDE.md) - Facility-specific configuration
- [FIELD_MAPPING_REFERENCE.md](FIELD_MAPPING_REFERENCE.md) - CSV import field mappings
- [DATABASE_SCHEMA_REFERENCE.md](DATABASE_SCHEMA_REFERENCE.md) - Complete database schema
- [migrations/046_create_rule_configuration_system.sql](../migrations/046_create_rule_configuration_system.sql) - Rule tables schema

---

## Quick Reference

### Most Commonly Used Fields

**For Charge Validation**:
- `total_claim_charge_amount` (ENCOUNTER)
- `line_item_charge_amount` (SERVICE_LINE)
- `service_unit_count` (SERVICE_LINE)

**For Procedure Validation**:
- `procedure_code` (SERVICE_LINE)
- `procedure_modifier_1` through `_4` (SERVICE_LINE)

**For Provider Validation**:
- `rendering_provider_npi`
- `billing_provider_npi`
- `provider_type`
- `provider_specialty`

**For Date Validation**:
- `date_of_service_from` (ENCOUNTER)
- `date_of_service_to` (ENCOUNTER)
- `service_date_from` (SERVICE_LINE)

**For Authorization Checks**:
- `service_authorization_code` (ENCOUNTER)
- `prior_authorization_number` (SERVICE_LINE)

**For Place of Service**:
- `place_of_service_code` (both levels)

**For Patient Demographics**:
- `subscriber_birth_date`
- `subscriber_gender`
- `subscriber_member_id`

---

**For questions or support**: Check service logs at `C:\ProgramData\Professional SMART\logs\service.log` or review the database schema in [DATABASE_SCHEMA_REFERENCE.md](DATABASE_SCHEMA_REFERENCE.md).
