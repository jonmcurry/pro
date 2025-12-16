# Fix Service Facility Fields Not Populating in Encounter Table

## Issue
Service facility NPI and name are present in `staging.raw_claims.encounter_fields` but not being populated in `claims.encounter` table.

## Root Cause
The `process_encounter_with_service_lines` function (used during batch processing) was missing the service facility field extraction and INSERT bindings. These fields were only present in the deprecated `process_raw_claim` function.

## Changes Made

### 1. claims_processor.rs - process_encounter_with_service_lines function

- [x] Added service facility field extraction (lines 496-503):
  - `service_facility_npi`
  - `service_facility_name`
  - `service_facility_address_line1`
  - `service_facility_address_line2`
  - `service_facility_city`
  - `service_facility_state`
  - `service_facility_postal_code`

- [x] Added debug logging for service facility fields (lines 505-508)

- [x] Added service facility columns to INSERT statement (lines 676-682)

- [x] Added bind statements for service facility values (lines 723-729)

- [x] Added service facility to error logging (line 744)

### 2. service.rs - Logging configuration

- [x] Changed default log level from "error" to "info" so service facility logs are visible (line 875)

### 3. claims_importer.rs (previous session)

- [x] Added service facility address fields capture from EDI parser (lines 871-875)

## Testing Checklist

- [ ] Install updated MSI on production server
- [ ] Process EDI file with NM1*77 segment containing service facility data
- [ ] Verify log shows: `Service facility from encounter_fields: npi=Some("..."), name=Some("...")`
- [ ] Verify `claims.encounter.service_facility_npi` is populated
- [ ] Verify `claims.encounter.service_facility_name` is populated
- [ ] Verify address fields are populated when present in EDI

## Version
- Previous: 2.9.2.0
- New: 2.9.3.0 (minor version - bug fix for missing field population)
