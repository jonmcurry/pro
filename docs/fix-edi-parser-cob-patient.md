# EDI Parser COB and Patient Fix - v2.9.0.0

## Summary

Fixed three critical issues in the 837P EDI parser:

1. **COB Data Corruption** - Secondary payer/subscriber data was overwriting primary payer/subscriber
2. **Patient Capture** - Patient information (NM1*QC) when different from subscriber was not being captured
3. **COB Adjudication** - SVD/CAS/OI segments were not being parsed

## Changes

### Priority 1: Fix COB Data Corruption

**Problem**: When an 837P file contained COB (Coordination of Benefits) data in Loop 2320/2330, the secondary payer's NM1*PR and NM1*IL segments would overwrite the primary payer and subscriber fields.

**Solution**:
- Added COB loop detection triggered by second SBR segment
- Track `in_cob_loop` state to route NM1*IL/PR to `OtherInsurance` struct instead of primary fields
- Preserve primary payer/subscriber data from Loop 2010BA/2010BB

**Files Changed**:
- `crates/pro-parser-edi/src/loops.rs` - Added COB loop tracking
- `crates/pro-parser-edi/src/types.rs` - Added `OtherInsurance` struct

### Priority 2: Capture Patient When Different From Subscriber

**Problem**: When patient is different from subscriber (HL level 23 present), NM1*QC segment contains patient name/address but was not being captured.

**Solution**:
- Added patient fields to `ParsedClaim` struct
- Added NM1*QC handler in parser
- Added PAT segment handler for patient relationship code
- Updated DMG handling to populate patient fields when in patient loop

**Files Changed**:
- `crates/pro-parser-edi/src/types.rs` - Added 15 patient fields
- `crates/pro-parser-edi/src/loops.rs` - Added QC handler, PAT handler, patient loop tracking
- `crates/pro-parser-edi/src/segments.rs` - Added PatSegment parser
- `crates/pro-service/src/claims_importer.rs` - Added patient fields to encounter_fields
- `crates/pro-service/src/claims_processor.rs` - Added patient field extraction and INSERT
- `migrations/060_add_patient_fields.sql` - Database schema update

### Priority 3: Add COB Adjudication Segments

**Problem**: SVD (Service Line Adjudication), CAS (Claim Adjustment), and OI (Other Insurance) segments were not being parsed.

**Solution**:
- Added segment parsers for OI, CAS, SVD
- Added `LineAdjudication` struct for SVD data
- Added `ClaimAdjustment` struct for CAS data
- Integrated parsers into loops.rs

**Files Changed**:
- `crates/pro-parser-edi/src/segments.rs` - Added OiSegment, CasSegment, SvdSegment parsers
- `crates/pro-parser-edi/src/types.rs` - Added LineAdjudication, ClaimAdjustment structs
- `crates/pro-parser-edi/src/loops.rs` - Added SVD/CAS/OI handlers

## New Database Fields (Migration 060)

```sql
-- Patient fields added to claims.encounter
patient_last_name VARCHAR(255)
patient_first_name VARCHAR(255)
patient_middle_name VARCHAR(255)
patient_name_suffix VARCHAR(50)
patient_date_of_birth DATE
patient_gender CHAR(1)
patient_address_line1 VARCHAR(255)
patient_address_line2 VARCHAR(255)
patient_city VARCHAR(100)
patient_state CHAR(2)
patient_postal_code VARCHAR(15)
patient_relationship_code VARCHAR(2)
```

## New Types

### OtherInsurance
Represents COB data from Loop 2320/2330:
- Payer responsibility sequence (P/S/T)
- Other subscriber name/ID/address
- Other payer name/ID/address
- OI segment fields
- AMT amounts
- CAS adjustments

### ClaimAdjustment
Represents CAS segment data:
- Adjustment group code (CO/CR/OA/PI/PR)
- Adjustment reason code (CARC)
- Adjustment amount
- Adjustment quantity

### LineAdjudication
Represents SVD segment data:
- Payer ID
- Paid amount
- Procedure code
- Paid units
- Adjudication date
- Associated CAS adjustments

## Version

- **Version**: 2.9.0.0
- **MSI**: installer/ProfessionalSMART.msi (10.7 MB)
- **Date**: 2025-12-15

## Testing Notes

The parser now correctly handles:
1. Single-payer claims (Example 1 hierarchy) - unchanged behavior
2. COB claims (Example 2 hierarchy) - primary payer preserved, secondary goes to other_insurance
3. Patient ≠ Subscriber scenarios - patient fields populated
4. Line-level adjudication from prior payers - stored in line_adjudications
