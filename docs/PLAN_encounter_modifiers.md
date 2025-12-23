# Plan: Encounter Procedure Modifiers Table

## Overview
Add a new table to store aggregated procedure modifiers at the encounter level as a comma-separated list.

## Requirements
- New table `claims.encounter_procedure_modifier` with FK to `claims.encounter`
- VARCHAR(20) column for comma-separated modifiers (e.g., "24,25,59")
- Populated during claim ingestion by aggregating unique modifiers from all service lines
- Deduplicated and sorted for consistency

## Implementation Checklist

### 1. Database Migration
- [x] Create migration `067_create_encounter_procedure_modifiers.sql`
- [x] Create table `claims.encounter_procedure_modifier`
  - `encounter_id` BIGINT PK/FK to claims.encounter
  - `modifiers` VARCHAR(20) - comma-separated list
  - `created_at` TIMESTAMPTZ
  - `updated_at` TIMESTAMPTZ
- [x] Add index on encounter_id (PK)
- [x] Add index on modifiers for searching
- [x] Add GIN index for pattern matching

### 2. Rust Model Updates
- [x] Added `insert_encounter_procedure_modifiers()` function to claims_processor.rs
- [x] Function queries service_line table for all unique modifiers
- [x] Uses LATERAL VALUES to unpivot modifier columns

### 3. Processing Logic
- [x] After inserting service lines, collect all non-null modifiers via SQL query
- [x] Deduplicate and sort modifiers (ORDER BY in query)
- [x] Join with comma separator
- [x] Insert into encounter_procedure_modifier table with ON CONFLICT UPDATE

### 4. Build & Test
- [x] Rebuild installer with new migration
- [x] Version increment to 2.12.23.0

## Files Modified
1. `migrations/067_create_encounter_procedure_modifiers.sql` (new)
2. `crates/pro-service/src/claims_processor.rs` - added `insert_encounter_procedure_modifiers()` function
3. `CHANGELOG.md` - added v2.12.23.0 entry
