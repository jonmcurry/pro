# Professional SMART Test Data

Generated: October 17, 2025

## Master Data Files (4 files)
- organizations.csv (2 organizations)
- regions.csv (4 regions)
- facilities.csv (8 facilities)
- providers.csv (16 providers)

## Claims Data Files

### CSV Format (80,000 total claims)
10,000 claims per facility with multiple service lines each:
- claims_ORG001-R1-F1.csv through claims_ORG002-R2-F8.csv
- Format: Athena Health compatible CSV

### EDI X12 837P Format (808 total encounters)
100 encounters per facility:
- claims_ORG001-R1-F1.edi through claims_ORG002-R2-F8.edi
- Format: HIPAA 5010 X12 837P Professional Claims

## Regeneration
cd /mnt/c/Users/jonmc/dev/pro
python3 scripts/regenerate_claims.py
python3 scripts/generate_edi_from_csv.py
