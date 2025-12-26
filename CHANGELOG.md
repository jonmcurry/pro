# Changelog

## [2.12.29.0] - 2025-12-26

### Changed
- **Unique Username Constraint**: Changed `idx_security_user_name` from regular index to UNIQUE index.
  - Enforces unique usernames in `security.security_user` table
  - Prevents duplicate user registrations

## [2.12.28.0] - 2025-12-26

### Added
- **SmartProAudit Foreign Data Wrapper**: Added cross-database querying capability for project databases.
  - Migration: `migrations/069_setup_smartproaudit_fdw.sql`
  - Enables `postgres_fdw` extension in each project database
  - Creates `smartproaudit` schema with foreign tables linked to SmartProAudit database
  - Foreign tables: `security_role`, `security_user`, `security_user_role`, `lookup_field_definitions`, `project`
  - Convenience view: `smartproaudit.user_roles` - Shows users with their assigned roles
  - Helper function: `smartproaudit.user_has_role(user_name, role_name)` - Check user permissions
  - Helper function: `smartproaudit.get_field_definitions(table_name)` - Get friendly column names
  - Allows project databases (professional_smart_clientA, etc.) to query centralized security and field definitions

## [2.12.27.0] - 2025-12-26

### Added
- **Security Schema Indexes**: Added performance and integrity indexes to security tables.
  - `idx_security_role_name` (UNIQUE) - Fast role lookup by name, enforces unique role names
  - `idx_security_user_role_unique` (UNIQUE) - Prevents duplicate user-role assignments

## [2.12.26.0] - 2025-12-26

### Added
- **Security Schema**: Added `security` schema to SmartProAudit master database for user authentication and role-based access control.
  - `security.security_role` table with role_name and role_description
  - `security.security_user` table with user_name and active status
  - `security.security_user_role` junction table for user-role assignments
  - Pre-populated with Admin, Super User, and User roles
  - Default user 'MWELLINGTO002' with Admin role

## [2.12.25.0] - 2025-12-26

### Added
- **SmartProAudit Master Database**: New PostgreSQL database for centralized project management.
  - Schema file: `migrations/smartproaudit/000_baseline.sql`
  - **projects schema**: Tracks all Professional SMART project databases
    - `projects.project` table with project_name, organization, versions, connection info
    - `projects.schema_migrations` table for SmartProAudit upgrades
  - **fields schema**: Field metadata for claims data display and export
    - `fields.lookup_field_definitions` table with friendly names for columns
    - Pre-populated with encounter, service_line, encounter_diagnosis, encounter_payer, encounter_view fields
  - Replaces file-based `projects.json` registry with PostgreSQL-based registry
  - Automatically created during installation if it doesn't exist
  - Each project database is registered in SmartProAudit during creation

### Changed
- **Installer**: Modified `CreateDatabase.vbs` to create and initialize SmartProAudit database
- **Product.wxs**: Added SmartProAudit migration files to installer package

## [2.12.24.0] - 2025-12-26

### Added
- **Encounter View**: Added `claims.encounter_view` for denormalized access to encounter data.
  - Migration: `migrations/068_create_encounter_view.sql`
  - Joins encounter with all provider types (billing, referring, rendering, supervising)
  - Includes payer hierarchy (primary, secondary, tertiary) from `encounter_payer` table
  - Aggregates diagnosis codes as comma-separated list ordered by sequence
  - Includes service facility details
  - Useful for reporting and data export without complex joins

## [2.12.23.0] - 2025-12-23

### Added
- **Encounter Procedure Modifiers Table**: Added new table `claims.encounter_procedure_modifier` to store aggregated procedure modifiers at encounter level.
  - Migration: `migrations/067_create_encounter_procedure_modifiers.sql`
  - Stores comma-separated list of unique modifiers from all service lines (e.g., "24,25,59")
  - VARCHAR(20) column for modifiers, deduplicated and sorted
  - Foreign key reference to `claims.encounter` with CASCADE delete
  - GIN index for pattern matching (e.g., finding encounters with modifier "25")
  - Automatically populated during claim ingestion from service line modifiers
  - File: `crates/pro-service/src/claims_processor.rs` - added `insert_encounter_procedure_modifiers()` function

### Fixed
- **Embedded Migrations**: Added migrations 066 and 067 to `crates/pro-upgrade-manager/src/embedded_migrations.rs` so they are included in the installer.
- **Baseline Migration**: Updated `migrations/000_baseline_v2.12.sql` to include migrations 065-067 (now covers 001-067).
- **CLAUDE.md Documentation**: Added installer build process documentation with step-by-step guide for adding new migrations.

## [2.12.22.0] - 2025-12-23

### Added
- **PostgreSQL Settings Enforcement Migration**: Added migration 066 to automatically enforce critical PostgreSQL settings during install/upgrade.
  - Migration: `migrations/066_enforce_postgresql_settings.sql`
  - Ensures `autovacuum = 'on'` to prevent table bloat (fixes issue from v2.12.19.0)
  - Sets `work_mem = '64MB'` to prevent memory exhaustion (fixes issue from v2.12.19.0)
  - Reloads PostgreSQL configuration automatically
  - Verifies settings were applied with NOTICE logging
  - Previously these were manual fixes; now enforced automatically on every install/upgrade

### Fixed
- **Build Script WiX Path**: Fixed `build-msi.ps1` to automatically add WiX Toolset to PATH and pass SolutionDir variable to candle.

## [2.12.21.0] - 2025-12-22

### Fixed
- **Removed Provider Advisory Locks**: Removed `pg_try_advisory_xact_lock` mechanism that was causing 96% claim failure rate.
  - File: `crates/pro-service/src/claims_processor.rs`
  - Root cause: Advisory locks prevented concurrent processing of claims with the same provider NPI. With test data using a single billing provider (NPI 1234567890), only 1 of 8 workers could proceed - the other 7 failed with "provider locked".
  - Impact: 9,627 of 10,000 claims (96%) marked as FAILED due to lock conflicts.
  - Fix: Removed advisory lock mechanism entirely. The `ensure_provider_exists` function already uses `INSERT ON CONFLICT DO NOTHING` which is safe for concurrent access.
  - Result: All workers can now process claims concurrently without lock contention.

### Performance Results
- **Target: 666 claims/second - ACHIEVED (123.5%)**
- Test: 10,000 claims (29,626 service lines) processed in 36.02 seconds
- Throughput: **822.5 claims/second** (274 encounters/second)
- Sustained rate: 290-340 encounters/second (870-1020 claims/second)
- Success rate: 98.7% (9,871 completed, 129 failed due to future DOS dates in test data)

## [2.12.20.0] - 2025-12-22

### Fixed
- **FIFO Batch Result Scatter Bug**: Fixed critical bug where provider lock conflicts caused claims to be reset and re-acquired across multiple batch sequences, breaking FIFO ordering.
  - File: `crates/pro-service/src/claims_processor.rs`
  - Root cause: When `pg_try_advisory_xact_lock` failed to acquire a provider lock, claims were reset to `PENDING` with `batch_sequence_number = NULL`, allowing them to be re-acquired by different batches.
  - Impact: Batches expected 100 claims but only retained 0-8 claims each. Workers returned near-empty results. SequentialCompletionManager never received meaningful batch completions. All batches ended up in RECOVERY state after 5-minute timeout.
  - Fix: Instead of resetting claims on lock conflict, mark them as failed within the same batch. This preserves batch integrity and allows proper FIFO completion tracking.
  - Note: This fix was superseded by v2.12.21.0 which removes advisory locks entirely.

### Performance
- Previous test: 163 encounters/second (~490 claims/second) but with broken batch tracking
- Expected after fix: Improved throughput with proper FIFO completion

## [2.12.19.0] - 2025-12-22

### Fixed
- **PostgreSQL Autovacuum Disabled**: Re-enabled autovacuum which was disabled in postgresql.auto.conf, causing 717k dead tuples (71x table bloat) on staging.raw_claims table.
  - Root cause: `autovacuum = 'off'` in postgresql.auto.conf
  - Impact: Table bloat from 30 MB to 238 MB, degraded query performance
  - Fix: `ALTER SYSTEM SET autovacuum = 'on'`

- **Excessive work_mem Setting**: Reduced work_mem from 512MB to 64MB to prevent memory exhaustion.
  - Root cause: `work_mem = '512MB'` with 300 max_connections could consume 150GB+ RAM
  - Fix: `ALTER SYSTEM SET work_mem = '64MB'`

- **Table Bloat Cleanup**: Ran VACUUM FULL ANALYZE on staging.raw_claims to reclaim space.
  - Before: 238 MB (717,632 dead tuples)
  - After: 30 MB (0 dead tuples)
  - Reduction: 87%

### Performance
- Measured baseline: 159 encounters/second (~477 claims/second)
- Target: 666 claims/second (per SRD.md specification)

## [2.12.18.0] - 2025-12-22

### Performance
- **Simplified FIFO Batch Acquisition**: Replaced complex CTE-based encounter grouping with simple FIFO-ordered claim acquisition.
  - File: `crates/pro-service/src/batch_sequencer.rs`
  - Problem: CTE with JSONB expression extraction, GROUP BY, and JOIN still taking 1-3 seconds per batch despite indexes
  - Root cause: Expression indexes help but don't eliminate JSONB extraction overhead; partial index invalidation during updates; 266k dead tuples causing bloat
  - Solution: Simplified query that:
    1. Selects claims by `ingested_at ASC` order (simple btree scan)
    2. Locks and updates atomically with `FOR UPDATE SKIP LOCKED`
    3. Relies on application-layer encounter grouping (already in claims_processor.rs)
  - Benefits:
    - No JSONB extraction in query
    - No GROUP BY or JOIN operations
    - Uses simple btree index on ingested_at
    - Avoids partial index invalidation issues
  - Expected: 10-20x faster batch acquisition

### Target
- Performance target: 666 claims/second (per SRD.md specification)
- Previous measured: 73 claims/second (v2.12.17.0 with indexed CTE)
- Expected: 400+ claims/second with simplified acquisition

## [2.12.17.0] - 2025-12-22

### Performance
- **CTE Batch Acquisition Index Optimization**: Added expression indexes on JSONB fields to optimize the CTE-based batch acquisition query.
  - Migration: `migrations/065_cte_batch_acquisition_indexes.sql`
  - Problem: CTE query taking 2.5+ seconds per batch due to missing indexes on JSONB expressions
  - Root cause: `encounter_fields->>'patient_control_number'` and `encounter_fields->>'date_of_service_from'` were being extracted without index support, requiring full table scans for GROUP BY and JOIN operations
  - Solution: Created 4 expression indexes:
    1. `idx_raw_claims_pcn_expr` - Expression index on patient_control_number
    2. `idx_raw_claims_dos_expr` - Expression index on date_of_service_from
    3. `idx_raw_claims_encounter_fifo` - Composite expression index for GROUP BY and FIFO ordering
    4. `idx_raw_claims_encounter_notnull` - Partial index with NOT NULL filters for pre-filtering
  - Result: Query time reduced from 2.5s to ~95ms (26x improvement)
  - PostgreSQL best practice: See https://www.postgresql.org/docs/current/indexes-expressional.html

### Target
- Performance target: 666 claims/second (per SRD.md specification)
- Previous measured: 38 claims/second (with slow CTE query)
- Expected: Significant improvement with indexed batch acquisition

## [2.12.16.0] - 2025-12-22

### Fixed
- **Batch Acquisition Re-acquisition Bug**: Rewrote `acquire_next_batch()` using a single atomic CTE (Common Table Expression) to eliminate the 20x re-acquisition overhead.
  - File: `crates/pro-service/src/batch_sequencer.rs`
  - Root cause: Previous implementation selected 2x batch_size claims with `FOR UPDATE SKIP LOCKED`, but only updated a subset to PROCESSING. The remaining claims were unlocked on commit and immediately re-acquired by the next iteration.
  - Evidence: 206,794 claims batched for 10,000 actual claims (20.7x overhead)
  - Fix: Single CTE that atomically:
    1. Identifies N distinct encounter groups in FIFO order
    2. Selects ALL claims belonging to those encounter groups
    3. Updates ALL selected claims to PROCESSING in one operation
  - Benefits: Atomic operation, no race conditions, complete encounter integrity, standard PostgreSQL best practice

### Performance
- Expected improvement: ~20x reduction in batch acquisition overhead
- Target: 666 claims/second (per SRD.md specification)
- Previous measured: 114 claims/second (with re-acquisition bug)

## [2.12.15.0] - 2025-12-22

### Performance
- **Parser Logging Optimization**: Downgraded all `[LOOP_DEBUG]` logging in `identify_loops()` from INFO to DEBUG level. These messages were generating 80,000+ log entries per 10k claims (8 messages per claim), causing massive I/O overhead.
  - File: `crates/pro-parser-edi/src/parser.rs`
  - 10 `info!()` calls changed to `debug!()`
  - Expected impact: Significant reduction in logging I/O overhead during parsing phase

### Target
- Performance target: 666 claims/second (per SRD.md specification)
- Previous measured: 94 claims/second (with excessive parser logging)

## [2.12.14.0] - 2025-12-22

### Fixed
- **Worker Transaction Handling**: Applied per-encounter transaction fix to `process_sequenced_batch()` (the multi-worker code path). Previously, when one encounter failed (e.g., `validate_dos()` trigger for future dates), the entire batch transaction was aborted, and all subsequent claims in the batch remained stuck in PROCESSING state.
  - File: `crates/pro-service/src/claims_processor.rs`
  - Root cause: "current transaction is aborted" error cascade when one encounter fails within a batch
  - Fix: Each encounter now has its own transaction; failures don't cascade to other encounters in the batch

### Performance
- Performance baseline maintained at ~142 claims/second for successful claims

## [2.12.13.0] - 2025-12-22

### Performance
- **Phase 1: Stuck Claims Recovery**: Added automatic recovery of stale PROCESSING claims (stuck > 5 minutes) at startup. This prevents claims from being permanently stuck if a previous run crashed.
- **Phase 2: Per-Encounter Transactions**: Changed from batched transactions to per-encounter transactions. Failures in one encounter no longer cause cascading rollbacks of successful encounters.
- **Phase 3: Batch Status Updates**: Batch update of claim statuses using `UPDATE ... WHERE raw_claim_id = ANY($1)` instead of individual UPDATE queries per claim.
- **Phase 4: Reduced JSON Cloning**:
  - Changed `process_encounter_with_service_lines()` to take `&[RawClaim]` reference instead of `Vec<RawClaim>` (avoiding clone)
  - Added `count_service_lines_in_json_value()` to work directly with JsonValue (avoiding deserialize + clone)
  - Changed total_claim_charge calculation to use `get()` on JsonValue directly
  - File: `crates/pro-service/src/claims_processor.rs`

### Fixed
- **5,044 claims stuck in PROCESSING**: Root cause was transaction rollback followed by failed error logging transaction. Claims are now properly marked COMPLETED or FAILED after processing.

## [2.12.12.0] - 2025-12-22

### Performance
- **Logging Optimization**: Downgraded diagnosis pointer logging from INFO to DEBUG level. The INFO-level logging was generating 233,108+ log messages per 10k claims run, causing significant I/O overhead and reducing performance from ~130 to ~113 claims/second.
  - File: `crates/pro-service/src/claims_processor.rs`
  - The diagnostic logging remains available at DEBUG level for troubleshooting

## [2.12.11.0] - 2025-12-22

### Fixed
- **Operation Order Bug**: Moved `import_diagnoses()` to execute BEFORE `import_service_line()` in `process_encounter_with_service_lines()`. Previously, service line diagnosis pointers were trying to reference diagnoses that hadn't been inserted yet, causing 0 rows to be inserted into the junction table.
  - File: `crates/pro-service/src/claims_processor.rs`
  - Root cause: Diagnoses must exist before service lines can reference them via diagnosis pointers

## [2.12.10.0] - 2025-12-22

### Added
- **Diagnostic Logging**: Added INFO-level logging to `import_service_line_diagnosis_pointers()` to trace diagnosis pointer insertion issues.
  - Logs when pointers are empty (all None)
  - Logs pointer count before processing
  - Logs rows_affected after INSERT execution

## [2.12.9.0] - 2025-12-22

### Fixed
- **Diagnosis Pointer Insert Bug**: Fixed `param_idx` starting at 2 instead of 3 in `import_service_line_diagnosis_pointers()` function. This caused `$2` to be used for both `encounter_id` and the first `pointer_sequence`, resulting in no rows being inserted into `service_line_diagnosis_pointer` junction table.
  - File: `crates/pro-service/src/claims_processor.rs`
  - Line: 2532

## [2.12.8.0] - 2025-12-22

### Fixed
- **ON CONFLICT Clause Mismatch**: Fixed `ON CONFLICT (service_line_id, diagnosis_id, pointer_sequence)` to match actual unique constraint `uk_line_diag_pointer` which is `(service_line_id, pointer_sequence)`. The previous clause caused batch insert failures with error "there is no unique or exclusion constraint matching the ON CONFLICT specification".
  - File: `crates/pro-service/src/claims_processor.rs`
  - Line: 2551

## [2.12.7.0] - 2025-12-22

### Added
- **Phase 3c: Savepoint Removal**: Removed unnecessary savepoints from `ensure_provider_exists()` function to reduce transaction overhead.

### Optimized
- **Phase 2c: Taxonomy Cache**: Added in-memory taxonomy cache (`Arc<RwLock<HashMap>>`) for provider lookups, reducing database queries for repeated taxonomy codes.
- **Phase 2b: Batch Diagnosis Pointer INSERT**: Replaced N individual INSERT statements with single INSERT...SELECT using UNION ALL for diagnosis pointers.
- **Phase 2a: Batch Diagnosis INSERT**: Replaced N individual INSERT statements with single multi-row INSERT for encounter diagnoses.
- **Phase 1a: Hot Path Logging**: Reduced excessive debug logging in critical processing paths.

### Performance
- Processing rate: ~130 claims/second (up from ~50 claims/second baseline)
