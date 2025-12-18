# Add Billing Date from BHT Segment

## Problem
Billing date (transaction creation date) from 837P EDI files is not being captured. This date comes from the BHT segment (BHT04).

## BHT Segment Structure
```
BHT*0019*00*244579*20190501*1036*CH~
```
- BHT01: Hierarchical Structure Code (0019)
- BHT02: Transaction Set Purpose Code (00=Original, 18=Reissue)
- BHT03: Reference Identification (Batch Control Number)
- BHT04: Transaction Set Creation Date (CCYYMMDD) - **BILLING DATE**
- BHT05: Transaction Set Creation Time (HHMM)
- BHT06: Transaction Type Code (CH=Chargeable)

## Solution

### Step 1: Add BHT segment parser
- [x] Add `BhtSegment` struct to `pro-parser-edi/src/segments.rs`
- [x] Parse BHT04 as transaction_date

### Step 2: Add billing_date to ParsedClaim type
- [x] Add `billing_date` field to `ParsedClaim` in `pro-parser-edi/src/types.rs`
- [x] Initialize billing_date in `pro-parser-edi/src/loops.rs`
- [x] Initialize billing_date in `pro-worker/src/converters.rs` (CSV converter)

### Step 3: Parse BHT in main parser
- [x] Extract BHT segment in `pro-parser-edi/src/parser.rs`
- [x] Store transaction_date in each ParsedClaim
- [x] Also updated streaming parser (parse_stream)

### Step 4: Add billing_date column to encounter table
- [x] Create migration 063_add_billing_date_to_encounter.sql

### Step 5: Store billing_date during import
- [x] Update `claims_importer.rs` to include billing_date in encounter_fields
- [x] Update `claims_processor.rs` to insert billing_date into encounter table

### Step 6: Build and version
- [x] Build all binaries
- [x] Update version (2.11.6.0 -> 2.11.7.0)
- [x] Rebuild installer (ProfessionalSMART.msi)
