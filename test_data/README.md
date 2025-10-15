# Professional SMART Test Data

This directory contains realistic test data for the Professional SMART claims processing system.

**Generated**: 2025-01-15

---

## Overview

This test dataset includes:
- **2 Organizations** - Multi-region healthcare systems
- **4 Regions** - Geographic service areas (2 per organization)
- **8 Facilities** - Medical centers and clinics (2 per region)
- **50 Providers** - Various specialties
- **80,000 Claims** - 10,000 claims per facility with service lines and diagnoses

Total service lines: ~120,000-200,000 (1-5 service lines per claim)

---

## Files Generated

### Master Data Files

| File | Records | Description |
|------|---------|-------------|
| `organizations.csv` | 2 | Healthcare organizations/systems |
| `regions.csv` | 4 | Geographic regions within organizations |
| `facilities.csv` | 8 | Medical facilities (hospitals, clinics, etc.) |

### Claims Files (Athena Health Format)

| File | Records | Facility |
|------|---------|----------|
| `claims_ORG001-R1-F1.csv` | 10,000 | Regional Health System - North Region Medical Center |
| `claims_ORG001-R1-F2.csv` | 10,000 | Regional Health System - North Region Clinic |
| `claims_ORG001-R2-F1.csv` | 10,000 | Regional Health System - South Region Medical Center |
| `claims_ORG001-R2-F2.csv` | 10,000 | Regional Health System - South Region Clinic |
| `claims_ORG002-R1-F1.csv` | 10,000 | Metropolitan Medical Group - North Region Medical Center |
| `claims_ORG002-R1-F2.csv` | 10,000 | Metropolitan Medical Group - North Region Clinic |
| `claims_ORG002-R2-F1.csv` | 10,000 | Metropolitan Medical Group - South Region Medical Center |
| `claims_ORG002-R2-F2.csv` | 10,000 | Metropolitan Medical Group - South Region Clinic |

**Total: 80,000 claims**

---

## Data Characteristics

### Organizations

**Organization 1: Regional Health System**
- 2 Regions (North, South)
- 4 Facilities total
- 40,000 claims

**Organization 2: Metropolitan Medical Group**
- 2 Regions (North, South)
- 4 Facilities total
- 40,000 claims

### Facilities

Each facility has:
- Unique NPI (10-digit National Provider Identifier)
- Tax ID
- Complete address
- EHR system designation (Athena, Epic, or Cerner)
- 10,000 claims

### Providers

- **50 providers** across 10 specialties:
  - Family Medicine
  - Internal Medicine
  - Pediatrics
  - Cardiology
  - Orthopedics
  - Dermatology
  - Neurology
  - Psychiatry
  - General Surgery
  - Emergency Medicine

- Each provider has unique NPI

### Claims Data

**Date Range**: Last 365 days from generation date

**Service Distribution**:
- 50% single-procedure claims
- 30% two-procedure claims
- 15% three-procedure claims
- 5% four or more procedures

**CPT Codes**: 30 common procedures including:
- Office visits (99213-99215, 99203-99205)
- Hospital care (99232-99233)
- Emergency visits (99284-99285)
- Lab tests (80053, 85025)
- Imaging (X-rays, CT, MRI, Ultrasound)
- Procedures (injections, repairs, splints)

**ICD-10 Diagnoses**: 30 common conditions including:
- Preventive care (Z00.00, Z23)
- Chronic conditions (I10, E11.9, E78.5)
- Acute conditions (J06.9, N39.0)
- Musculoskeletal (M25.511, M54.5)
- Injuries (S93.401A, S83.201A)

**Payers**: 10 major insurance companies:
- Blue Cross Blue Shield
- Aetna
- UnitedHealthcare
- Cigna
- Humana
- Medicare
- Medicaid
- Anthem
- Centene
- Molina Healthcare

**Demographics**:
- Patients aged 1-90 years
- Realistic first and last names
- Gender distribution
- Valid date of birth

**Financial Data**:
- Charges vary by CPT code
- Realistic pricing with +/- 20% variation
- Units: typically 1, sometimes 2-3
- Total charges per claim: $25 - $1,500+

---

## CSV Format

### Claims File Format (Athena Health)

All claims files use standard Athena Health CSV format with these columns:

| Column | Description | Example |
|--------|-------------|---------|
| Patient ID | Unique claim identifier | ORG001-R1-F1-000001 |
| DOS | Date of service | 2024-01-15 |
| Provider NPI | Rendering provider NPI | 1234567890 |
| CPT | Procedure code | 99213 |
| Modifier 1 | First modifier | 25 |
| Modifier 2 | Second modifier | (blank) |
| Units | Service units | 1 |
| Charges | Charge amount | 150.00 |
| Diagnosis 1 | Primary diagnosis | I10 |
| Diagnosis 2 | Secondary diagnosis | E11.9 |
| Diagnosis 3 | Third diagnosis | (blank) |
| Diagnosis 4 | Fourth diagnosis | (blank) |
| Patient Last Name | Last name | Smith |
| Patient First Name | First name | John |
| DOB | Date of birth | 1980-05-12 |
| Gender | M/F | M |
| POS | Place of service | 11 |
| Payer ID | Payer identifier | BCBS001 |
| Payer Name | Payer name | Blue Cross Blue Shield |
| Member ID | Insurance member ID | MEM123456789 |
| Facility Code | Facility code | ORG001-R1-F1 |
| Facility Name | Facility name | North Region Medical Center |

---

## Usage

### Import Individual Facility

To import a single facility's claims:

```bash
# Copy the facility CSV to the input directory
cp test_data/claims_ORG001-R1-F1.csv /path/to/input/

# Or use the Professional SMART service
professional-smart console
```

### Import All Claims

To import all test data:

```bash
# Copy all claims files
cp test_data/claims_*.csv /path/to/input/

# Monitor processing
tail -f /path/to/logs/pro-service.log
```

### Sample Queries

After importing, test with these SQL queries:

**Total Claims**:
```sql
SELECT COUNT(*) FROM claims.encounter;
-- Expected: 80,000
```

**Claims by Facility**:
```sql
SELECT
    f.facility_name,
    COUNT(*) as claim_count,
    SUM(e.total_claim_charge_amount) as total_charges
FROM claims.encounter e
JOIN master.facility f ON e.facility_id = f.facility_id
GROUP BY f.facility_name
ORDER BY claim_count DESC;
-- Expected: 10,000 claims per facility
```

**Claims by Payer**:
```sql
SELECT
    payer_name,
    COUNT(*) as claim_count,
    SUM(total_claim_charge_amount) as total_charges,
    AVG(total_claim_charge_amount) as avg_charge
FROM claims.encounter
GROUP BY payer_name
ORDER BY claim_count DESC;
```

**Top Procedures**:
```sql
SELECT
    sl.procedure_code,
    COUNT(*) as procedure_count,
    SUM(sl.line_item_charge_amount) as total_charges
FROM claims.service_line sl
GROUP BY sl.procedure_code
ORDER BY procedure_count DESC
LIMIT 10;
```

**Provider Productivity**:
```sql
SELECT
    p.last_name || ', ' || p.first_name as provider_name,
    p.specialty,
    COUNT(DISTINCT e.encounter_id) as encounter_count,
    COUNT(sl.service_line_id) as service_line_count
FROM claims.encounter e
JOIN master.provider p ON e.rendering_provider_id = p.provider_id
JOIN claims.service_line sl ON e.encounter_id = sl.encounter_id
GROUP BY p.provider_id, p.last_name, p.first_name, p.specialty
ORDER BY encounter_count DESC
LIMIT 20;
```

---

## Regenerating Test Data

To generate fresh test data with different parameters:

```bash
# Generate with custom parameters
python3 scripts/generate_test_data.py ./my_test_data

# Or edit the script to change:
# - Number of organizations
# - Regions per organization
# - Facilities per region
# - Number of providers
# - Claims per facility
```

Edit `scripts/generate_test_data.py` and modify the `generate_all()` call:

```python
generator.generate_all(
    org_count=3,              # 3 organizations
    regions_per_org=3,        # 3 regions each
    facilities_per_region=3,  # 3 facilities per region
    providers_count=100,      # 100 providers
    claims_per_facility=5000  # 5,000 claims per facility
)
# Total: 3 orgs × 3 regions × 3 facilities × 5,000 claims = 135,000 claims
```

---

## File Sizes

Approximate file sizes:

| File Type | Size per 10,000 Claims | Total Size |
|-----------|------------------------|------------|
| Claims CSV | ~2.5 MB | ~20 MB for 8 files |
| Organizations CSV | < 1 KB | < 1 KB |
| Regions CSV | < 1 KB | < 1 KB |
| Facilities CSV | < 1 KB | < 1 KB |

**Total Dataset**: ~20 MB

---

## Data Quality

### Realistic Features

✓ **Valid NPIs** - 10-digit numbers
✓ **Realistic charges** - Based on Medicare fee schedules
✓ **Proper modifiers** - Common modifier combinations
✓ **Valid ICD-10 codes** - Real diagnosis codes
✓ **Age-appropriate** - Patients 1-90 years old
✓ **Date consistency** - Service dates within last year
✓ **Service line distribution** - Realistic multi-line claims
✓ **Diagnosis linking** - 1-4 diagnoses per claim

### Limitations

⚠ **Not real data** - Computer-generated, not actual patient records
⚠ **Simplified payer logic** - Random payer assignment
⚠ **No authorization data** - Prior auth numbers not included
⚠ **No remittance data** - No payment/denial information
⚠ **Random provider assignment** - Not specialty-matched to procedures

---

## Testing Scenarios

Use this data to test:

1. **Import Pipeline**
   - CSV parsing and validation
   - Auto-detection of Athena format
   - Duplicate detection
   - Error handling

2. **Rules Engine**
   - Coding validation
   - Modifier rules
   - Diagnosis coding
   - Medical necessity

3. **Dashboard Analytics**
   - Management overview
   - Provider productivity
   - Facility performance
   - Payer analysis

4. **Performance Testing**
   - Import 80,000 claims
   - Query performance on large dataset
   - Concurrent processing

5. **API Testing**
   - REST endpoint responses
   - Query filtering
   - Pagination
   - WebSocket streaming

---

## Cleanup

To remove all test data:

```bash
rm -rf test_data/
```

To remove from database:

```sql
-- CAUTION: This deletes ALL data, not just test data!
TRUNCATE TABLE claims.encounter CASCADE;
TRUNCATE TABLE claims.service_line CASCADE;
TRUNCATE TABLE claims.encounter_diagnosis CASCADE;
TRUNCATE TABLE staging.import_batch CASCADE;
```

---

## Support

For issues with test data generation:

1. Check Python version: `python3 --version` (3.7+ required)
2. Review script: `scripts/generate_test_data.py`
3. Check generation logs above

For issues with importing test data:

1. Verify CSV format matches Athena template
2. Check service logs: `logs/pro-service.log`
3. Review [CSV Mapping Guide](../docs/CSV_MAPPING_GUIDE.md)

---

**Generated by**: Professional SMART Test Data Generator v1.0
**Script**: `/scripts/generate_test_data.py`
**Documentation**: `/docs/CSV_MAPPING_GUIDE.md`
