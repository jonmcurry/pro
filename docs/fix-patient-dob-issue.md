# Fix: Patient Date of Birth Not Reaching claims.encounter

## Problem
Patient date of birth from 837P EDI files (specifically for newborn claims) is not making it into the claims.encounter table.

## Investigation Summary

### Parser Verification
- Created test file `test_data/test_newborn.edi` with proper HL*23 patient loop structure
- Parser correctly extracts:
  - `subscriber_date_of_birth` from DMG in subscriber loop (HL*22)
  - `patient_date_of_birth` from DMG in patient loop (HL*23)
- Test passed: Parser returns correct patient DOB (2023-12-10) for newborn claim

### EDI 837P Structure for Newborn Claims
For patient DOB to be captured, the EDI must have:
```
HL*x*x*22*1~           <- Subscriber level with child_code=1 (has dependent)
SBR*P*18*...~          <- Subscriber information
NM1*IL*1*SMITH*JANE... <- Subscriber name
DMG*D8*19900115*F~     <- Subscriber DOB
NM1*PR*2*...~          <- Payer
HL*x*x*23*0~           <- Patient level (REQUIRED for newborn)
PAT*19~                <- Patient relationship code (19=child)
NM1*QC*1*SMITH*BABY... <- Patient name (REQUIRED)
N3*...~                <- Patient address
N4*...~                <- Patient city/state
DMG*D8*20231210*M~     <- Patient DOB (REQUIRED)
CLM*...~               <- Claim information
```

### Data Flow (All Working Correctly)
1. **Parser** (`pro-parser-edi`): Extracts `patient_date_of_birth` from DMG after NM1*QC
2. **Importer** (`claims_importer.rs:763`): Stores in `encounter_fields["patient_date_of_birth"]`
3. **Processor** (`claims_processor.rs:523-524`): Reads and inserts into encounter table

### Root Cause
The test file `test_cob_secondary.edi` that was imported does NOT have an HL*23 patient loop - it only has HL*22 (subscriber level) where subscriber=patient (relationship code 18=self).

For newborn claims, the EDI MUST include:
- `HL*x*x*23*0` - Patient hierarchical level
- `NM1*QC` - Patient name segment
- `DMG` after NM1*QC - Patient demographics (including DOB)

## Resolution
No code changes required. The system correctly handles patient DOB when the EDI file has proper structure.

### Action Items
- [x] Verify parser correctly extracts patient DOB (test passed)
- [x] Verify importer stores patient DOB in raw_claims
- [x] Verify processor inserts patient DOB into encounter
- [ ] Request sample of actual newborn EDI file from user to verify structure
- [ ] If EDI structure is non-standard, may need parser enhancement

## Test File Created
`test_data/test_newborn.edi` - Sample newborn claim with proper HL*23 structure
