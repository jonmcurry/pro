# Production Processing Errors - Fix Plan

## Issue Summary

4228 claims are stuck in staging as "PROCESSING" status. Analysis of logs reveals multiple root causes:

## Root Cause Analysis

### Issue 1: "value too long for type character varying(2)"
**Error**: Database rejecting service line inserts
**Evidence**:
- `procedure_code:` (empty in logs)
- Error is `varchar(2)` overflow

**Root Cause**: Claims were imported to staging with empty procedure codes. When the processor tries to insert these into the database, something is misaligned OR the claims have malformed SV1 segments.

**Investigation**: The SV1 segment composite parsing may be failing for certain file formats.

### Issue 2: Provider taxonomy foreign key constraint
**Error**: `insert or update on table "provider" violates foreign key constraint "fk_provider_taxonomy"`
**Evidence**: NPI=1760917884 failing repeatedly

**Root Cause**: Provider has a taxonomy code that doesn't exist in the `provider_taxonomy` reference table.

**Fix**: Need to handle missing taxonomy codes gracefully or add them to the reference table.

### Issue 3: Migration 056 failing
**Error**: `relation "encounter" already exists` in archive schema
**Evidence**: Migration 056_create_archive_system.sql failing at statement 3

**Root Cause**: Migration uses `CREATE TABLE` without `IF NOT EXISTS`. Partial migration was applied previously, leaving archive.encounter table but migration not marked complete.

**Fix**: Add IF NOT EXISTS to all CREATE TABLE statements in migration 056.

### Issue 4: ANSI 275 files incorrectly named as .837p
**Error**: `Invalid transaction set identifier. Expected 837, got 275`
**Evidence**: Files like `15902_ANSI275.837p`

**Root Cause**: These are X12 275 (Patient Information) files, not 837P claims. Not a code bug - upstream system is sending wrong file types.

**Status**: Not a code fix - these files are correctly being moved to error folder.

---

## Fix Checklist

### Phase 1: Fix Migration 056
- [x] Add IF NOT EXISTS to archive table creation statements (using DO blocks)
- [x] Make migration idempotent

### Phase 2: Handle missing provider taxonomy gracefully
- [x] Validate taxonomy_code exists in provider_taxonomy before INSERT
- [x] Set taxonomy_code to NULL if not found (avoids FK violation)
- [x] Log warning when taxonomy code is not found

### Phase 3: Investigate empty procedure codes
- [ ] Check SV1 parsing for edge cases (deferred - may be data issue)
- [ ] Add better error logging for malformed segments (deferred)

### Phase 4: Rebuild and deploy
- [x] Build all binaries
- [x] Update version to 2.8.11.0
- [x] Build MSI (installer/ProfessionalSMART.msi - 10.7 MB)
- [ ] Deploy to production

### Phase 5: Reset stuck claims (on production after deploying v2.8.11.0)
- [ ] SQL: UPDATE staging.raw_claims SET processing_status = 'PENDING' WHERE processing_status = 'PROCESSING'

---

## Investigation Notes

- varchar(2) columns in service_line: product_service_id_qualifier, procedure_modifier_1-4, unit_basis_measurement_code, place_of_service_code, ndc_measurement_unit
- Empty procedure_code suggests SV1 composite parsing failed or claims have malformed data
- Provider taxonomy FK error cascades and aborts the transaction, causing "current transaction is aborted" errors
