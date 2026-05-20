# Plan: Fix date_of_service_to not persisted in process_encounter_with_service_lines

## Problem

`process_encounter_with_service_lines` (claims_processor.rs:1168) binds `dos_from` for
BOTH `date_of_service_from` ($19) and `date_of_service_to` ($20). The parser correctly
computes `date_of_service_to = MAX(service_line.service_date_to)` and stores it in
encounter_fields, but this code path never reads it.

The other code path (`process_raw_claim`, line 2238) correctly parses and binds
`date_of_service_to`.

## Root Cause

The comment at line 1168 says `"date_of_service_to same as from for now"` -- this was
a placeholder that was never updated when the DOS span computation was added in v2.14.3.0.

## Fix (v2.16.2.0 - patch)

- [x] Parse `date_of_service_to` from encounter_fields after `dos_from` (line ~734)
- [x] Bind the parsed value (or fall back to `dos_from`) at $20 instead of `dos_from`
- [x] Update CHANGELOG.md
- [x] Update version to 2.16.2.0
- [ ] Rebuild installer (Windows-only, skip on Mac dev)
