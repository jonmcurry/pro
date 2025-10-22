Professional SMART Test Data Files
===================================

Generated: 2025-10-22

Summary:
- 2 Organizations (ORG001, ORG002)
- 4 Regions (2 per organization)
- 8 Facilities (2 per region)
- 40 Providers (5 per facility)
- 80,000 Total Encounters (10,000 per facility)

CSV Files (8 total):
- claims_ORG001-R1-F1.csv (10,000 encounters, ~17,750 service lines)
- claims_ORG001-R1-F2.csv (10,000 encounters, ~17,600 service lines)
- claims_ORG001-R2-F1.csv (10,000 encounters, ~17,560 service lines)
- claims_ORG001-R2-F2.csv (10,000 encounters, ~17,520 service lines)
- claims_ORG002-R1-F1.csv (10,000 encounters, ~17,740 service lines)
- claims_ORG002-R1-F2.csv (10,000 encounters, ~17,520 service lines)
- claims_ORG002-R2-F1.csv (10,000 encounters, ~17,690 service lines)
- claims_ORG002-R2-F2.csv (10,000 encounters, ~17,610 service lines)

EDI Files (8 total):
- Corresponding EDI X12 837P files for each CSV
- Each EDI file contains 10,000 encounters

CSV Column Names (compatible with two-stage pipeline):
- Patient Control Number
- Date of Service
- Provider NPI
- Procedure Code
- Modifier 1, Modifier 2
- Units
- Charge Amount
- Diagnosis 1, Diagnosis 2, Diagnosis 3, Diagnosis 4
- Patient Last Name, Patient First Name
- DOB
- Gender
- Place of Service
- Payer ID, Payer Name
- Member ID
- Facility Code, Facility Name

Notes:
- Each encounter may have multiple service lines (procedures)
- All data is synthetic and randomly generated
- Dates range from 2024-01-01 to 2025-10-22
- Includes realistic CPT codes, ICD-10 codes, and modifiers
