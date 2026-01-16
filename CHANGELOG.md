# Changelog

## [2.12.73.29] - 2025-01-16

### Performance - ELIMINATE MODIFIER SELECT QUERY
- **Critical Performance Fix**: Eliminated expensive SELECT DISTINCT ... LATERAL query for modifiers
  - Problem: After inserting service lines, a SELECT query with LATERAL was reading them back (~40-50ms)
  - Root cause: `insert_encounter_procedure_modifiers()` queried service_line table to get modifiers
  - Solution: Collect modifiers from `ServiceLineRuleContext` (already in memory from import)
  - New function `insert_encounter_procedure_modifiers_fast()` uses pre-collected data
  - Eliminates 1 SELECT + 1 INSERT per encounter, replaced with just 1 INSERT
  - Expected improvement: **3-5x throughput increase** (from 14 rec/sec to 42-70 rec/sec)

## [2.12.73.28] - 2025-01-15

### Performance - BATCH PROVIDER INSERTION
- **Critical Performance Fix**: Batch insert ALL providers in `prewarm_provider_cache()`
  - Problem: 4+ sequential `ensure_provider_exists()` calls per encounter (each a DB round-trip)
  - For 10K claims × 2.5 service lines × 4 providers = ~100,000 sequential INSERT operations
  - Solution: Batch INSERT all new providers in a single query using UNNEST
  - All providers (existing + new) are now cached BEFORE encounter processing begins
  - `ensure_provider_exists()` calls become instant cache hits (no DB query)
  - Expected improvement: **3-5x throughput increase** (from 13 rec/sec to 40-65 rec/sec)

## [2.12.73.27] - 2025-01-15

### Performance - CPT INDEXING + SYNC EXECUTION
- **Critical Performance Fix**: Switched from `execute_all()` to `execute_all_indexed()` in claims processor
  - Problem: All 600+ rules were being evaluated for every service line (O(n) where n = total rules)
  - Solution: Use CPT code index to filter rules (O(k) where k = rules applicable to this CPT + universal rules)
  - Rules with `cpt_in` conditions are indexed at load time for O(1) lookup
  - Only rules matching the service line's CPT code are executed
  - Expected improvement: 5-20x speedup depending on how many rules have `cpt_in` filters

- **Sync Execution for COMPOSITE Rules**: COMPOSITE rules now use synchronous execution path
  - COMPOSITE rules are CPU-only (no database access required)
  - `requires_db_access()` returns false, enabling `execute_sync()` path
  - Avoids async/await overhead for pure CPU rule evaluation
  - Expected improvement: 2-5x speedup for COMPOSITE rule execution

- **Performance Warning for Universal Rules**: Added warning when >50 universal rules loaded
  - Universal rules (no `cpt_in` filter) run on every service line
  - Warning helps identify rules that should have CPT filters added
  - Target: <50 universal rules for optimal throughput

### Changed
- **Removed Auto-Defer Threshold**: Inline execution is now default even with 600+ rules
  - CPT indexing + sync execution makes inline execution viable for high rule counts
  - `DEFER_RULES_EXECUTION` still available for explicit deferral if needed
  - Previous auto-defer at 100 rules was a workaround, now properly fixed

## [2.12.73.25] - 2025-01-15

### Performance - AUTO-DEFER RULES
- **Smart Auto-Detection for Deferred Rules**: System now automatically defers rules when count >= 100
  - Problem: Users with 500+ rules were experiencing 5 rec/sec throughput without knowing to set `DEFER_RULES_EXECUTION=true`
  - Solution: Auto-detect high rule counts and automatically enable deferred mode
  - Threshold: 100+ rules triggers auto-deferral (configurable via code constant)
  - Override: Users can still force inline execution with `DEFER_RULES_EXECUTION=false`
  - Logs clear warnings when auto-deferring: "AUTO-DEFERRING rules execution - X rules exceeds threshold of 100"

### Configuration
- `DEFER_RULES_EXECUTION` behavior updated:
  - **Not set (recommended)**: Auto-defers if rule count >= 100
  - `true`: Always defer rules to background processing
  - `false`: Always execute rules inline (slow with many rules)

### Documentation
- Updated `.env.example` with comprehensive rules engine configuration section
- Added `ENABLE_DATABASE_RULES`, `DEFER_RULES_EXECUTION`, and `RULE_ENCRYPTION_KEY` documentation

## [2.12.73.24] - 2025-01-15

### Performance - CRITICAL
- **Deferred Rules Execution Mode**: Added `DEFER_RULES_EXECUTION` environment variable for high-throughput import
  - Root cause: Sequential rule execution consuming 200-300ms per claim (537 rules x 3 service lines = 1,611 rule evaluations per encounter)
  - With 10K claims target in 30 seconds, inline rule execution is the bottleneck (theoretical max: 3-5 claims/sec)
  - Solution: When `DEFER_RULES_EXECUTION=true`, rules are queued for background processing instead of inline execution
  - Encounters are queued to `staging.rules_processing_queue` for async rule processing
  - Expected throughput improvement: **50-100x** when enabled (from ~5 claims/sec to 300-500 claims/sec)
  - Trade-off: Flags appear after import completes rather than immediately

### Added
- **Rules Processing Queue** (migration 073):
  - `staging.rules_processing_queue` table for deferred rule execution
  - `staging.enqueue_for_rules_processing()` - queue encounter for background processing
  - `staging.acquire_rules_processing_batch()` - FIFO batch acquisition with SKIP LOCKED
  - `staging.complete_rules_processing()` - mark completed with flag count
  - `staging.fail_rules_processing()` - mark failed with error message
  - `staging.recover_stale_rules_processing()` - recover stuck items (>5 min)

### Configuration
- New environment variables:
  - `DEFER_RULES_EXECUTION=true` - Enable deferred rules for maximum import throughput
  - `DEFER_RULES_EXECUTION=false` (default) - Inline rules execution (slower but immediate flags)

## [2.12.73.23] - 2025-01-15

### Performance
- **Batch Provider Cache Pre-warming**: Added `prewarm_provider_cache()` optimization for Stage 2 processing
  - Root cause: Each service line was doing 4 sequential database queries for provider lookups (rendering, ordering, supervising, referring)
  - With 5 service lines per encounter, that's up to 20 DB round-trips per encounter
  - Solution: Before processing service lines, collect all unique NPIs and query existing providers in ONE batch query
  - Pre-populates provider cache so subsequent `ensure_provider_exists()` calls are instant cache hits
  - Expected throughput improvement: 2-4x for encounters with multiple service lines
  - Impact: Reduces DB round-trips from O(service_lines * 4) to O(1) per encounter for existing providers

## [2.12.73.22] - 2025-01-15

### Fixed
- **Baseline SQL Syntax Error Blocking Views**: Removed invalid `COMMENT ON DATABASE current_database()` statement
  - Root cause: This statement is syntactically invalid (cannot use function call in DDL statement)
  - This caused the baseline execution to fail, preventing all subsequent statements from executing
  - Migration 072 views (`v_processing_summary`, `v_stage2_throughput`, etc.) were never created
  - Removed the invalid statement from baseline SQL

## [2.12.73.21] - 2025-01-15

### Fixed
- **Baseline Migration Tracking Conflict**: Removed incorrect INSERT statements from baseline SQL
  - Root cause: Baseline SQL had hardcoded INSERT statements for migrations 031-072 with wrong filenames
  - These conflicted with the programmatic registration in `apply_baseline()` which uses correct embedded migration names
  - Removed the incorrect INSERT block (migrations 031-072) from baseline
  - Removed non-existent `011_create_schedule_tables.sql` from migration tracking
  - Migration tracking is now handled entirely by `apply_baseline()` in migration.rs which iterates through embedded migrations

## [2.12.73.20] - 2025-01-15

### Fixed
- **Fresh Install Missing Views**: Fixed `BASELINE_COVERS_THROUGH` constant not updated to 72
  - Fresh installs were missing migration 072 views (`v_processing_summary`, `v_stage2_throughput`, etc.)
  - Updated constant from 64 to 72 so baseline properly includes all migrations through 072

## [2.12.73.19] - 2025-01-15

### Performance
- **Sync Execution for Universal Rules**: Fixed slow throughput (3-25 records/sec) when all rules are universal
  - Root cause: `execute_all()` was only using sync execution when CPT index was populated
  - With 537 universal COMPOSITE rules (no `cpt_in` filter), CPT index was empty
  - Modified `execute_all()` to always use `execute_sync()` for rules that don't require database access
  - Expected 2-5x throughput improvement for universal rule workloads

- **CPT Index Logging Fix**: Changed CPT index statistics from `eprintln!` to `tracing::info!`
  - Now properly appears in service.log instead of stderr

### Added
- **Processing Metrics Rollup Views** (migration 072):
  - `staging.v_processing_metrics_hourly` - Hourly aggregated throughput metrics
  - `staging.v_processing_metrics_daily` - Daily aggregated throughput metrics
  - `staging.v_processing_summary` - Last 24 hours summary with success rates
  - `staging.v_stage2_throughput` - Hourly Stage 2 claims processing throughput

## [2.12.73.17] - 2025-01-15

### Performance
- **Service Line Flag Query Performance**: Fixed extremely slow queries on `claims.service_line_flag` (5+ minutes)
  - Added migration 071 with performance indexes for flag queries
  - Added covering index `idx_service_line_flag_view_lookup` for view JOINs
  - Added partial index `idx_service_line_flag_recent` for open flag dashboards
  - Added `idx_service_line_flag_status_lookup` for status filtering
  - Added `idx_service_line_encounter_lookup` for service_line to encounter JOINs

### Fixed
- **Duplicate Flag Prevention**: Added unique constraint and ON CONFLICT handling to prevent duplicate flags
  - Root cause: Reprocessing claims would create duplicate flags for the same service_line + issue combination
  - Added unique index `idx_service_line_flag_unique_open` on `(service_line_id, issue_id) WHERE flag_status = 'OPEN'`
  - Modified flag INSERT to use `ON CONFLICT DO NOTHING` - skips if flag already exists
  - Prevents flag table bloat from reprocessing the same claims multiple times

## [2.12.73.14] - 2025-01-15

### Fixed
- **Processing Metrics INSERT Fix**: Fixed critical bug preventing `staging.processing_metrics` records from being inserted
  - Root cause: Code was providing a value (`0i64`) for `metric_id` column which is defined as `GENERATED ALWAYS AS IDENTITY`
  - PostgreSQL rejects INSERT statements that explicitly provide values for GENERATED ALWAYS columns
  - Metrics logging was silently failing, leaving `processing_metrics` table empty
  - Fixed all 4 affected functions across 3 files:
    - `claims_processor.rs::log_processing_metric()`
    - `batch_manager.rs::log_processing_metric()`
    - `claims_importer.rs::log_processing_metric()`
    - `claims_importer.rs::log_processing_metric_with_stage()`
  - Processing throughput metrics will now be properly recorded for performance monitoring

## [2.12.73.12] - 2025-01-15

### Performance
- **CPT Code Index for Rule Engine**: Major performance optimization for large rule sets (500+ rules)
  - Added CPT-based rule index that maps procedure codes to applicable rules
  - Rules with `cpt_in` conditions are indexed by their CPT codes for O(1) lookup
  - Instead of executing all 560 rules per service line, now only executes rules that match the procedure code
  - Typical performance improvement: 95%+ reduction in rule evaluations per service line
  - Added `build_cpt_index()` method called automatically after loading rules from database

- **Synchronous Execution for COMPOSITE Rules**: Eliminated async overhead for CPU-only rules
  - COMPOSITE rules now implement `execute_sync()` for direct synchronous execution
  - Added `requires_db_access() = false` for COMPOSITE rules to use sync path
  - Avoids tokio async machinery overhead when no database access is needed

- **Rule Trait Extensions**: New trait methods for optimization
  - `requires_db_access()`: Returns whether rule needs database during execution
  - `applicable_cpt_codes()`: Returns CPT codes for index-based filtering
  - `execute_sync()`: Synchronous execution path for CPU-only rules

## [2.12.73.11] - 2025-01-15

### Fixed
- **Remove PHASE 8 Rule Logging**: Removed incomplete PHASE 8 rule execution logging code from `execute_all_with_cache()`
  - The code was calling non-existent `claims.log_rule_execution()` stored procedure
  - Spawned background tasks for every rule execution (triggered or not), causing potential connection pool exhaustion
  - While not directly called by claims_processor (which uses `execute_all`), the code path exists and could cause issues

## [2.12.73.10] - 2025-01-15

### Fixed
- **Rules Engine Flag INSERT JOIN Fix**: Fixed critical bug where flags were not being inserted into `claims.service_line_flag`
  - Root cause: `RuleResult.flag_type.code()` returned hardcoded enum codes like `"OTH-003"` but the database `claims.flag_issue.issue_code` contains custom values like `"TEST_99213_SA"`
  - The INSERT JOIN `ON fi.issue_code = fd.issue_code` was failing because codes didn't match
  - Added `issue_code: Option<String>` field to `RuleResult` struct
  - Added `with_issue_code()` builder method to `RuleResult`
  - Updated `RuleTemplate::instantiate()` signature to accept `issue_code` parameter
  - Updated loader to extract `issue_code` from database row and pass to all templates
  - Updated all template rules to store and return `issue_code` in their `execute()` methods:
    - `composite_rule.rs` (COMPOSITE template)
    - `threshold_rule.rs` (THRESHOLD template)
    - `duplicate_rule.rs` (DUPLICATE template)
    - `missing_field_rule.rs` (MISSING_FIELD template)
    - `field_pattern_rule.rs` (FIELD_PATTERN template)
    - `cross_field_rule.rs` (CROSS_FIELD template)
  - Updated `claims_processor.rs` to use `result.issue_code` when available, falling back to `result.flag_type.code()` for legacy rules
  - Added debug logging in loader to show `issue_code` during rule instantiation

## [2.12.73.5] - 2025-01-13

### Fixed
- **Rules Engine Flag Persistence**: Fixed critical bug where flags were not being inserted
  - Rule engine was trying to insert into non-existent `claims.flag` table
  - Now correctly inserts into `claims.encounter_flag` for encounter-level flags
  - Now correctly inserts into `claims.service_line_flag` for service line-level flags
  - Flags now link to `flag_issue` table via `issue_id` using the `issue_code` lookup
  - Added proper routing based on whether `service_line_id` or `encounter_id` is present

## [2.12.73.4] - 2025-01-13

### Fixed
- **Rule Converter GUI Performance**: Optimized for large datasets (500+ rules)
  - Fixed crash when clicking "Select All" with 553 rules
  - Added `set_redraw(false/true)` wrapper for bulk ListView operations
  - Removed redundant selection tracking (HashSet) - now queries ListView directly
  - Removed `on_selection_changed` event handler that was triggering 553 times
  - Added `bulk_operation` flag to prevent event handling during batch operations
  - Pre-allocate SQL string buffer for faster export
  - Safe string truncation at character boundaries (not byte boundaries)

## [2.12.73.3] - 2025-01-13

### Fixed
- **Rule Converter GUI Definition Column**: Added missing definition column to ListView
  - Now shows Rule Code, Rule Name, Description, and Definition columns
  - Definition is truncated to 80 chars in display (full text used for export)
- **Rule Converter GUI Export Crash**: Fixed application crash when clicking "Export Selected to SQL"
  - Added extensive debug logging to diagnose export issues
  - Added proper error handling for empty definitions
  - Shows warning in log and SQL output for rules that fail to convert
  - Shows success/error counts after export

## [2.12.73.1] - 2025-01-13

### Fixed
- **Rule Converter GUI MS SQL Connection**: Fixed connection to MS SQL Server
  - Now uses ADO.NET connection string with `Encrypt=false` and `TrustServerCertificate=true`
  - Added Username and Password input fields to GUI
  - Properly handles SQL Server Authentication

## [2.12.73.0] - 2025-01-13

### Added
- **Rule Converter GUI**: New GUI tool to convert legacy filter rules from MS SQL Server to COMPOSITE template SQL
  - Connects to MS SQL Server using tiberius crate with SQL Server Authentication
  - Configurable SQL query via `rule-converter-config.toml` file
  - ListView with multi-select for choosing rules to export
  - Exports selected rules as SQL INSERT statements with COMPOSITE template JSON parameters
  - Added Start Menu shortcut "Rule Converter (MS SQL)"
  - Files added:
    - `crates/pro-rule-converter-gui/Cargo.toml` - Package configuration
    - `crates/pro-rule-converter-gui/src/main.rs` - NWG-based GUI application
    - `crates/pro-rule-converter-gui/src/converter.rs` - Rule parsing and SQL generation
    - `crates/pro-rule-converter-gui/src/mssql.rs` - MS SQL Server client using tiberius
    - `crates/pro-rule-converter-gui/rule-converter-config.toml` - Configuration file with SQL query
    - `crates/pro-rule-converter-gui/build.rs` - Build script for Windows resources
    - `crates/pro-rule-converter-gui/windows-manifest.rc` - Windows manifest resource
    - `crates/pro-rule-converter-gui/windows-manifest.xml` - DPI awareness manifest

## [2.12.72.0] - 2025-01-12

### Added
- **Rule Converter Tool**: New CLI tool to convert legacy filter rules to COMPOSITE template SQL
  - Parses legacy `Parser.In()` syntax for DX, CPT, Date, POS fields
  - Generates SQL INSERT statements with proper COMPOSITE JSON parameters
  - Supports file input or inline rules via `--inline` flag
  - Usage: `pro-rule-converter -i rules.txt -o output.sql`
  - Files added:
    - `crates/pro-rule-converter/Cargo.toml`
    - `crates/pro-rule-converter/src/main.rs`

## [2.12.71.1] - 2025-01-12

### Added
- **AHRQOP001A Rule in Baseline**: Added AHRQ Opioid ED Visit rule to mandatory baseline
  - Added QM (Quality Measures) flag category
  - Added QM_OPIOID_ED flag issue
  - Added AHRQOP001A rule definition using COMPOSITE template
  - Rule flags ED visits (CPT 99281-99285, 99291) with opioid-related diagnosis (F11.x except F11.21, T40.x)

## [2.12.71.0] - 2025-01-12

### Added
- **COMPOSITE Rule Template**: New template for creating compound rules without recompilation
  - Supports AND/OR logic for combining multiple conditions
  - Condition types: cpt_in, cpt_pattern, dx_in, dx_pattern, dx_pattern_exclude, date_gte, date_lte, pos_in, pos_pattern, modifier_in, modifier_not_in
  - Enables database-only configuration of complex AHRQ quality indicators
  - Files added:
    - `crates/pro-rules/src/templates/composite_rule.rs` - Template implementation
    - `migrations/seed_data/ahrqop001a_opioid_ed_rule.sql` - Example AHRQ rule
  - Files modified:
    - `crates/pro-rules/src/templates/mod.rs` - Export new template
    - `crates/pro-rules/src/loader.rs` - Register COMPOSITE template
    - `migrations/046_create_rule_configuration_system.sql` - Add template to database
    - `migrations/000_baseline_v2.12.sql` - Add template to baseline

## [2.12.70.2] - 2025-01-01

### Changed
- **Project ID Auto-Generation**: Changed `projects.project.id` column to use IDENTITY instead of SERIAL
  - Updated `smartproaudit/000_baseline.sql` to use `GENERATED BY DEFAULT AS IDENTITY`
  - Allows auto-generation of IDs while still permitting explicit values when needed

## [2.12.70.1] - 2025-12-30

### Changed
- **FDW Password Authentication**: Updated Foreign Data Wrapper to use password authentication instead of peer authentication
  - Added password option to USER MAPPING in migration 069
  - Updated baseline 000 with the same change
  - Default credentials: user `postgres`, password `postgres`
  - Updated FDW_HOWTO.md documentation

## [2.12.70.0] - 2025-12-30

### Fixed
- **Reverted egui/eframe Migration**: Reverted GUI framework back to NWG (Native Windows GUI)
  - egui/eframe with wgpu backend requires DirectX 12/Vulkan which is not available on Windows Server 2019 without GPU
  - NWG uses Win32 GDI controls that work on all Windows versions without GPU requirements
  - Cleaned up temporary backup files

## [2.12.68.0] - 2025-12-30

### Enhanced
- **GUI 2025 UX Polish**: Applied modern design principles for a polished, professional appearance
  - **Increased Dimensions**: Larger windows with more generous spacing
    - Data Loader: 920×720 (was 900×680)
    - Project Manager: 1000×720 (was 960×680)
  - **Improved Typography**: Larger, more readable fonts with clear hierarchy
    - Header font: Segoe UI Semibold 17pt (was 15pt)
    - Body font: Segoe UI 14pt (was 13pt)
    - Log font: Consolas 13pt (was 12pt)
  - **Better Spacing**: Increased margins (20px from 16px), row heights (38px), and control heights (28px)
  - **Fixed Button Truncation**: Widened action buttons (180px from 150px) to fit "Load from Directory..." text
  - **Comfortable Touch Targets**: Larger button heights (34px) for easier clicking
  - **Consistent Font Application**: Body font applied to all form labels for uniformity
  - **Files Changed**:
    - `pro-data-loader-gui/src/main.rs`: Dimension and font updates
    - `pro-project/src/gui/app.rs`: Dimension and font updates

## [2.12.67.0] - 2025-12-30

### Enhanced
- **GUI Modernization - Full Visual Refresh**: Complete visual overhaul of both GUI applications
  - **Custom Font System**: Added distinct fonts for different UI elements
    - Header font: Segoe UI Semibold 15pt for section headers
    - Body font: Segoe UI 13pt for labels and text
    - Log font: Consolas 12pt for monospace log display
  - **Colored Status Indicators**: Traffic light style status colors
    - Green (Forest Green): Success/Connected/Up to date
    - Yellow (Dark Goldenrod): Warnings/Pending
    - Red (Firebrick): Errors/Failed
    - Blue (Steel Blue): Info/Processing
  - **RichLabel Status Displays**: Replaced plain Labels with RichLabel for colored, styled status text
  - **RichTextBox Activity Logs**: Replaced ListBox with RichTextBox for colored, formatted log entries
    - Log level indicators (INFO, SUCCESS, WARNING, ERROR) now colored and bold
  - **Improved Layout & Spacing**: Increased margins (16px), larger controls, better visual rhythm
  - **Files Changed**:
    - `pro-data-loader-gui/Cargo.toml`: Added `rich-textbox` feature
    - `pro-data-loader-gui/src/main.rs`: Full modernization
    - `pro-project/Cargo.toml`: Added `rich-textbox` feature
    - `pro-project/src/gui/app.rs`: Full modernization

## [2.12.66.0] - 2025-12-30

### Fixed
- **Project Database Manager Console Window - Complete Fix**: Use FreeConsole to completely detach from console
  - **Issue**: Previous ShowWindow(SW_HIDE) fix only minimized the console window; it remained visible in taskbar
  - **Solution**: Changed from `ShowWindow(SW_HIDE)` to `FreeConsole()` which completely detaches the process from its console
  - **Technical Details**: `FreeConsole()` is the proper Windows API for console detachment - it releases the console rather than just hiding it
  - **File Changed**: `pro-project/src/gui/mod.rs`

## [2.12.65.0] - 2025-12-30

### Fixed
- **Project Database Manager Console Window**: Hide console window when running in GUI mode
  - **Issue**: Black command prompt window appeared behind the GUI window
  - **Root Cause**: `pro-project` is compiled as a console application (no `windows_subsystem = "windows"`) to support CLI mode
  - **Solution**: Added `hide_console_window()` function that calls Windows API `ShowWindow(SW_HIDE)` when GUI mode starts
  - **File Changed**: `pro-project/src/gui/mod.rs`

## [2.12.64.0] - 2025-12-30

### Fixed
- **NWG GUI DPI Scaling - Feature Flag**: Enabled `high-dpi` feature in native-windows-gui
  - **Root Cause**: `scale_factor()` returns 1.0 (no scaling) unless `high-dpi` feature is enabled
  - **Solution**: Added `high-dpi` feature to both GUI crates' Cargo.toml
  - **Files Changed**:
    - `pro-data-loader-gui/Cargo.toml`: Added `features = ["high-dpi"]`
    - `pro-project/Cargo.toml`: Added `features = ["list-view", "high-dpi"]`
  - **Reference**: https://github.com/gabdube/native-windows-gui/blob/master/native-windows-gui/src/win32/high_dpi.rs

## [2.12.63.0] - 2025-12-30

### Fixed
- **NWG GUI DPI Scaling**: Implemented runtime DPI-aware layout for both GUI applications
  - **Root Cause**: Per-Monitor V2 DPI awareness in Windows manifest meant controls received physical pixels, but fixed pixel values designed for 96 DPI were too small at higher DPI settings (125%, 150%, etc.)
  - **Solution**: Query `nwg::scale_factor()` at runtime and scale all control dimensions proportionally
  - **Data Loader GUI** (`pro-data-loader-gui`):
    - Added `apply_dpi_scaling()` function that runs on init
    - All labels, buttons, text inputs, and layout positions scaled by DPI factor
    - Base window 880x620 at 96 DPI, scales appropriately at higher DPI
  - **Project Manager GUI** (`pro-project`):
    - Added `apply_dpi_scaling()` function that runs on init
    - Connection controls, toolbar, ListView, and log section all scale properly
    - ListView column widths also scaled for proper text display
    - Base window 920x600 at 96 DPI, scales appropriately at higher DPI
  - **Technical Details**:
    - Scale factor: 1.0 at 96 DPI (100%), 1.25 at 120 DPI (125%), 1.5 at 144 DPI (150%)
    - All pixel values multiplied by scale factor at runtime
    - Windows manifests retained Per-Monitor V2 for crisp text rendering

## [2.12.57.0] - 2025-12-29

### Fixed
- **NWG GUI Layout Rewrite**: Simplified GUI layout for better rendering
  - **Project Manager GUI**:
    - Removed Frame containers that caused black backgrounds
    - Added ListView `ex_flags` (FULL_ROW_SELECT, GRID) for proper rendering
    - Simplified layout with all controls directly on window
    - Reduced window size to 900x600 for better default display
  - Both GUIs now use flat layout without nested Frame containers

## [2.12.56.0] - 2025-12-29

### Fixed
- **NWG GUI Layout Issues**: Fixed control sizing and ListView rendering
  - **Data Loader GUI**: Widened labels and buttons to prevent text truncation
    - Increased window width from 900 to 950
    - Widened labels ("Organizations:", "Regions (Optional):", etc.)
    - Widened action buttons ("Load from Directory...", "Generate Templates...")
  - **Project Manager GUI**:
    - Widened toolbar buttons to show full text
    - Enabled `list-view` feature for `native-windows-gui` to fix black screen where ListView should appear

## [2.12.55.0] - 2025-12-29

### Fixed
- **NWG FileDialog Filter Format**: Fixed incorrect filter format causing GUI to crash on startup
  - **Error**: `Failed to build UI: FileDialogError("Bad extension filter format")`
  - **Root Cause**: NWG FileDialog filter format uses pipe to separate different filters, not pattern repetition
  - **Solution**: Changed from `"CSV Files (*.csv)|*.csv"` to `"CSV Files(*.csv)|All Files(*.*)"`

## [2.12.54.0] - 2025-12-29

### Fixed
- **NWG GetWindowSubclass Error - Complete Fix**: Added Windows manifest to resolve "Entry Point Not Found" error
  - **Error**: `The procedure entry point GetWindowSubclass could not be located in comctl32.dll`
  - **Root Cause**: `windows-sys` crate (pulled by sqlx via etcetera) requires a manifest declaring Common Controls v6
  - **Solution**: Embed Windows manifest in both GUI executables using `embed-resource` crate
  - **Files Added**:
    - `pro-data-loader-gui/pro-data-loader-gui.exe.manifest`: Common Controls v6 declaration
    - `pro-data-loader-gui/pro-data-loader-gui-manifest.rc`: Resource file
    - `pro-data-loader-gui/build.rs`: Build script to embed manifest
    - `pro-project/pro-project.exe.manifest`: Common Controls v6 declaration
    - `pro-project/pro-project-manifest.rc`: Resource file
    - `pro-project/build.rs`: Build script to embed manifest
  - **Reference**: [NWG Issue #251](https://github.com/gabdube/native-windows-gui/issues/251)

## [2.12.53.0] - 2025-12-29

### Fixed
- **NWG GetWindowSubclass Error**: Attempted fix by pinning chrono (incomplete - see 2.12.54.0)
  - **Error**: `The procedure entry point GetWindowSubclass could not be located in comctl32.dll`
  - **Root Cause**: Conflict between `native-windows-gui` and `chrono` 0.4.27+ which pulls in `windows-targets`
  - **Solution**: Pin chrono to use `default-features = false` to exclude `windows-iana` feature
  - **Reference**: [NWG Issue #282](https://github.com/gabdube/native-windows-gui/issues/282)

## [2.12.52.0] - 2025-12-29

### Changed
- **Windows Server 2019 GUI: Migrated to Native Windows GUI (NWG)**
  - **Solution**: Replaced egui/eframe (wgpu-based) with Native Windows GUI which uses Win32 GDI controls
  - **Why**: wgpu/WARP/DX12 still failed on Windows Server 2019 RDS sessions - no GPU adapter available
  - **NWG Benefits**:
    - Uses Win32 GDI controls - pure software rendering built into Windows
    - No GPU/OpenGL/DirectX/Vulkan requirements
    - Works on all Windows versions (Vista+) including headless Windows Server
    - Lighter dependencies - no wgpu, winit, or graphics drivers needed
  - **Files Changed**:
    - `pro-project/Cargo.toml`: Replaced eframe/egui with native-windows-gui/native-windows-derive
    - `pro-project/src/gui/mod.rs`: NWG initialization
    - `pro-project/src/gui/app.rs`: Complete rewrite using NWG controls
    - `pro-data-loader-gui/Cargo.toml`: Replaced eframe/egui with native-windows-gui
    - `pro-data-loader-gui/src/main.rs`: Complete rewrite using NWG controls
  - **Reference**: [Native Windows GUI](https://github.com/gabdube/native-windows-gui)
  - **Plan Document**: [NWG_GUI_MIGRATION_PLAN.md](docs/NWG_GUI_MIGRATION_PLAN.md)

## [2.12.51.0] - 2025-12-29

### Fixed
- **Windows Server 2019 GUI: DirectX 12 WARP Configuration**: Proper configuration for WARP software rendering
  - **Change**: Explicitly configure `WgpuConfiguration` with DX12 backend and `LowPower` preference
  - **Technical Details**:
    - Set `supported_backends: wgpu::Backends::DX12` in code (not just env var)
    - Set `power_preference: wgpu::PowerPreference::LowPower` to help select WARP
    - WARP (Windows Advanced Rasterization Platform) is built into Windows Server 2019
  - **Files**: `pro-project/src/gui/mod.rs`, `pro-data-loader-gui/src/main.rs`
  - **Reference**: [Microsoft WARP Guide](https://learn.microsoft.com/en-us/windows/win32/direct3darticles/directx-warp)
  - **Note**: This did not work - see v2.12.52.0 for the successful NWG solution

## [2.12.50.0] - 2025-12-29

### Changed
- **Windows Server 2019: CLI-First Architecture**: Resolved GUI issues on headless Windows Server by making CLI the primary interface
  - **Root Cause**: Windows Server 2019 headless environments lack GPU/graphics support required by modern GUI frameworks (wgpu, OpenGL 3.3+, Vulkan, DirectX 12)
  - **Solution**: CLI-first design - all functionality available via command line
  - **pro-project.exe**: Now runs as CLI by default (console window visible), use `--gui` flag for GUI mode
  - **pro-data-loader.exe**: Pure CLI tool with full functionality
  - **pro-data-loader-gui.exe**: Shows helpful error message directing to CLI if GUI fails
  - **Error messages**: Now include specific CLI examples when GUI unavailable
  - See: [WINDOWS_SERVER_GUI_SOLUTION.md](docs/WINDOWS_SERVER_GUI_SOLUTION.md) for full documentation

### Usage on Windows Server
```powershell
# Project management
pro-project.exe list
pro-project.exe create --name MyProject
pro-project.exe switch --name MyProject
pro-project.exe status

# Master data loading
pro-data-loader.exe --csv-dir C:\data\master
```

## [2.12.49.0] - 2025-12-29

### Fixed
- **Windows Server 2019 GUI Compatibility (v3)**: Comprehensive fix for GUI applications on headless Windows Server
  - **Changes**:
    - Try multiple backends: Vulkan → DX12 → GL (in order of preference)
    - Disabled multisampling (`multisampling: 0`) - required for software renderers
    - Disabled depth buffer (`depth_buffer: 0`) - reduces GPU requirements
    - Set `WGPU_POWER_PREF=low` - prefer integrated/software rendering
    - Set `WGPU_ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER=1` - allow software renderers
    - Added user-friendly error message with solution instructions if GUI fails to start
  - **If GUI still fails**: Install Mesa3D for Windows from https://github.com/pal1000/mesa-dist-win
    - Download mesa3d release, extract `opengl32.dll` and `libgallium_wgl.dll`
    - Copy both DLLs to `C:\Program Files\Professional SMART\bin\`
  - Reference: [egui software rendering issue](https://github.com/emilk/egui/issues/957)

## [2.12.48.0] - 2025-12-29

### Fixed
- **Windows Server 2019 GUI Compatibility (v2)**: Additional fix for GUI applications not loading on Windows Server 2019
  - **Root Cause**: Windows Server RDS sessions don't expose GPU by default, and wgpu needs explicit DirectX 12 backend selection
  - **Solution**: Added `WGPU_BACKEND=dx12` environment variable at startup to force DirectX 12 with WARP software rendering fallback
  - Files: `pro-project/src/gui/mod.rs`, `pro-data-loader-gui/src/main.rs`
  - Reference: [wgpu DX12 WARP issue](https://github.com/gfx-rs/wgpu/issues/2503)

## [2.12.47.0] - 2025-12-29

### Fixed
- **Windows Server 2019 GUI Compatibility**: Fixed Project Database Manager and Master Data Loader GUI applications not loading on Windows Server 2019
  - **Root Cause**: The `glow` (OpenGL) backend requires OpenGL 2.0+ which is not available on headless Windows Server environments
  - **Solution**: Switched to `wgpu` backend which can use DirectX 12 WARP (software renderer) when GPU is unavailable
  - Files: `pro-project/Cargo.toml`, `pro-data-loader-gui/Cargo.toml`, GUI initialization code updated
  - Reference: [egui_glow NoAvailablePixelFormat issue](https://github.com/emilk/egui/issues/957)

## [2.12.46.0] - 2025-12-29

### Performance
- **Provider Cache Optimization**: Implemented in-memory NPI → provider_id cache in ClaimsProcessor
  - **Root Cause**: Same provider NPI appears up to 16 times per encounter (4 encounter-level + 4 per service line × ~3 lines)
  - **Impact**: Each `ensure_provider_exists` call was executing 2 DB queries (upsert + enrichment queue) for every occurrence
  - **Solution**: Cache provider_id after first lookup, subsequent lookups return from cache instantly
  - **Verified Result**: **1,284 claims/second** (192.8% of SRD target)
  - Files: `claims_processor.rs` updated

### SRD Performance Target ACHIEVED
- **Target**: 10,000 claims in 15 seconds (666.67 claims/sec)
- **Actual**: 9,971 claims in 7.76 seconds (**1,284 claims/sec**)
- **Performance**: 192.8% of target (nearly 2x requirement)

| Version | Throughput | Notes |
|---------|------------|-------|
| v2.12.44.0 | ~190 claims/sec | Baseline with default config |
| v2.12.45.0 | ~195 claims/sec | Trigger removal (+2.6%) |
| **v2.12.46.0** | **1,284 claims/sec** | Provider cache (+558%) |

## [2.12.45.0] - 2025-12-29

### Performance
- **CRITICAL: Removed sync_encounter_totals Triggers**: Dropped the `sync_encounter_totals_insert`, `sync_encounter_totals_update`, and `sync_encounter_totals_delete` triggers from `claims.service_line` table.
  - **Root Cause**: These triggers fired for EVERY service line insert, executing a `SELECT SUM()` and `UPDATE` on the encounter table each time
  - **Impact**: For 10,000 claims with ~30,000 service lines, this added ~60,000 extra database operations
  - **Why Safe**: The `total_claim_charge_amount` is already calculated in Rust before the encounter INSERT, making the triggers redundant
  - **Expected Improvement**: From ~190 claims/sec to 600+ claims/sec (eliminating 6 queries per encounter)
  - Files: `070_drop_encounter_totals_trigger.sql`, `000_baseline_v2.12.sql` updated

## [2.12.44.0] - 2025-12-29

### Performance
- **Installer Default Configuration**: Updated default worker configuration for new installs to achieve 666+ claims/sec SRD target:
  - Changed `WORKER_THREADS=4` to `STAGE2_WORKER_COUNT=8` (correct environment variable name)
  - Changed `BATCH_SIZE=100` to `BATCH_SIZE=750` (proven optimal for throughput)
  - Added inline documentation comments explaining each setting
  - Files updated: `env.template`, `WriteConfig.vbs`

## [2.12.43.0] - 2025-12-28

### Code Quality (MEDIUM Priority Fixes)
- **Registry Service Methods**: Added `#[allow(dead_code)]` to `get_active_project()` and `project_exists()` (reserved for future project switching UI)
- **Windows Service Manager**: Added `#[allow(dead_code)]` to `restart()` (reserved for future service management UI)
- **WebSocket State**: Added `#[allow(dead_code)]` to `broadcaster()` (reserved for future progress tracking integration)
- **Pipeline Wrapper Methods**: Added `#[allow(dead_code)]` to `extract_diagnoses_from_csv()`, `extract_service_lines_from_csv()`, and `process_claim_in_transaction()` (wrapper methods superseded by improved implementations)
- **Unused Imports Cleanup**: Removed unused `DEFAULT_DATE` import from claims_processor.rs
- **Variable Prefixes**: Fixed unused variable warnings (`_data` in websocket.rs, `_fac_id` in dashboard.rs, removed `mut` from `batch_rx` in service.rs)

### Code Review Status
- MEDIUM priority items from CODE-REVIEW-2025-12-28.md addressed
- Verified code review item "parser.rs unused apply_transformations" - import does not exist (false positive)
- Verified code review item "transformers.rs unused regex::Regex" - import does not exist (false positive)
- Verified code review item "models.rs unused sqlx::types::Uuid" - import does not exist (false positive)
- Reviewed `.ok()` patterns in claims_processor.rs - acceptable for optional JSON field parsing with default fallback

## [2.12.42.0] - 2025-12-28

### Code Quality
- **Iterator Pattern Improvements**: Refactored batch INSERT placeholder generation to use idiomatic `map().collect()` instead of `for i in 0..len()` loops in `import_encounter_payers_from_cob()` and `import_other_insurance()`
- **Service Constants Documentation**: Added `#[allow(dead_code)]` to `SERVICE_DISPLAY_NAME` and `SERVICE_DESCRIPTION` constants (referenced via function parameters)
- **Connection Pool Documentation**: Added documentation to `DatabaseService` explaining that fresh connections per operation are acceptable for infrequent project management operations

### Schema Review
- **Migration 018 Verified**: Confirmed flag table indexes are properly commented out with documentation explaining `claims.flag` table doesn't exist
- **Migration 019 Verified**: Confirmed materialized views migration is properly disabled with documentation for future flag table refactoring

## [2.12.41.0] - 2025-12-28

### Changed
- **Additional Unused Code Documentation**: Added `#[allow(dead_code)]` annotations with documentation for reserved/future-use code:
  - `pro-rules`: Removed unused `FlagContext` import from threshold_rule.rs, `Error` from missing_field_rule.rs, `Rule` from hot_reload.rs
  - `pro-worker`: Removed unused `Error` and `Encounter` imports from claim_processor.rs, `ClaimProcessingResult` from file_processor.rs
  - `pro-worker`: Documented `facility_id` field in `IngestionPipeline` as reserved for future facility-specific rule loading
  - `pro-project`: Documented `ProjectRow` and `TaskMessage` structs as GUI data models
  - `pro-setup`: Documented PostgreSQL auto-installer functions as reserved for future feature

## [2.12.40.0] - 2025-12-28

### Performance
- **Batch COB Payer Inserts**: Optimized `import_encounter_payers_from_cob()` to use batch INSERT instead of individual inserts per payer.
  - Reduces database round-trips from N to 1 for COB payer imports
  - Improves throughput for claims with multiple insurance payers
- **Batch Other Insurance Inserts**: Optimized `import_other_insurance()` to use batch INSERT instead of individual inserts.
  - Reduces database round-trips from N to 1 for other insurance records

### Fixed
- **Silent Provider Lookup Errors**: Changed provider lookup `.unwrap_or(None)` to `.unwrap_or_else()` with warning logging.
  - Ensures unexpected errors from `ensure_provider_exists()` are logged instead of silently dropped
  - Affects rendering, referring, supervising, and billing provider lookups in claims processor

### Removed
- **Unused Encounter Repository Methods**: Removed unused `list_by_organization()`, `list_by_facility()`, and `list_by_date_range()` methods from `EncounterRepository`.
  - These methods were never called and used `SELECT *` pattern

### Changed
- **Unused Code Documentation**: Added `#[allow(dead_code)]` with documentation comments to utility methods reserved for future use:
  - `BackupService::verify()`, `BackupService::list_backups()`, `BackupInfo` struct
  - `ConfigService::exists()`, `DbParams::connection_string*()` methods
  - `MigrationService::get_baseline()`, `apply_all_pending()`, `update_application_version()`
  - `ProjectStatus::Error`, `ProjectStatus::Checking` enum variants
  - `MigrationResult` struct
- **Cleanup Unused Imports**: Removed unused imports from `pro-rules` crate (template.rs, loader.rs, composite_rule.rs, hot_reload.rs)
- **Data Loader Validation Comment**: Added comment clarifying facility validation in provider import

## [2.12.39.0] - 2025-12-28

### Fixed
- **Installer Schema Version Query**: Fixed psql command construction for schema version query when psql is in system PATH.
  - Added proper handling for PATH vs full-path psql executable scenarios
  - Added detailed logging for schema version query debugging
  - Schema version is now correctly calculated from highest migration number

## [2.12.38.0] - 2025-12-28

### Fixed
- **Installer Schema Version Registration**: Fixed MSI installer using build version instead of actual schema version when registering projects.
  - `CreateDatabase.vbs` now queries `staging.schema_migrations` after applying migrations to get the actual schema version
  - Fresh installs now correctly register with schema version 2.12.69.0 instead of the build version
  - Schema version is calculated from highest migration number (e.g., migration 069 = 2.12.69.0)

## [2.12.37.0] - 2025-12-28

### Fixed
- **Baseline Missing Migration Registrations**: Fixed baseline not registering migrations 031-069 in `schema_migrations` table.
  - Added INSERT statements for all 39 missing migrations (031-069) at the end of `000_baseline_v2.12.sql`
  - Fresh installs now correctly show schema version as 2.12.69.0

## [2.12.36.0] - 2025-12-28

### Fixed
- **Fresh Install Schema Version**: Fixed fresh database installs incorrectly setting schema version to build version instead of migration-based version.
  - `get_schema_version()` now calculates version from `staging.schema_migrations` table (highest migration number)
  - Fresh installs now correctly report schema version as 2.12.69.0 (based on 69 migrations in baseline)
  - Removed hardcoded version fallbacks in favor of dynamic calculation from embedded migrations

## [2.12.35.0] - 2025-12-27

### Fixed
- **Windows Server GUI Compatibility**: Fixed GUI applications not loading on Windows Server 2019+.
  - Switched from wgpu to glow (OpenGL) renderer backend for software rendering fallback
  - Explicitly set `renderer: eframe::Renderer::Glow` for both pro-project and pro-data-loader-gui
  - GUI tools now work on servers without GPU acceleration

## [2.12.34.0] - 2025-12-27

### Fixed
- **Project Database Manager Version Update**: Fixed GUI upgrade not updating `database_version` in SmartProAudit registry after applying migrations.
  - GUI now correctly updates `projects.project.database_version` after successful schema upgrade
  - Version is computed from highest applied migration number (e.g., migration 069 -> 2.12.69.0)
  - Both CLI and GUI upgrade paths now consistently update the registry
- **Multi-statement Migration Execution**: Fixed migration application failing for SQL files with multiple statements.
  - Added `split_sql_statements()` function to properly parse SQL files
  - Handles dollar-quoted strings (`$$`) in PostgreSQL functions correctly
  - Uses `sqlx::raw_sql()` instead of `sqlx::query()` for statement execution
- **Migration Column Name Fix**: Fixed MigrationService querying wrong column name.
  - Changed from `version` to `migration_name` column in `staging.schema_migrations`
  - Version is now extracted from migration filename (e.g., "069" from "069_setup_smartproaudit_fdw.sql")

## [2.12.33.0] - 2025-12-27

### Added
- **Project Database Manager Tool (`pro-project.exe`)**: New CLI and GUI tool for managing multiple Professional SMART project databases.
  - **CLI Commands:**
    - `pro-project create --name <NAME> [--switch]` - Create new project database with full schema
    - `pro-project switch --name <NAME> [--no-restart]` - Switch active database and restart service
    - `pro-project list [--format table|json|csv]` - List all registered project databases
    - `pro-project info [--name <NAME>]` - Show detailed project information
    - `pro-project delete --name <NAME> [--force] [--backup]` - Delete project database
    - `pro-project backup [--name <NAME>] [--output <PATH>]` - Create pg_dump backup
    - `pro-project status` - Show schema upgrade status for all projects
    - `pro-project upgrade [--name <NAME>|--all] [--backup] [--dry-run]` - Apply pending migrations
  - **GUI Mode:**
    - `pro-project gui` - Launch graphical interface
    - Data grid showing all SmartProAudit registered projects
    - Checkbox selection for batch operations
    - Status indicators (up to date, pending upgrades, errors)
    - "Upgrade Selected" and "Backup & Upgrade" actions
    - Real-time progress and log display
  - **Services:**
    - `RegistryService` - Query/update SmartProAudit project registry
    - `DatabaseService` - PostgreSQL operations (create, drop, schema check)
    - `ConfigService` - Atomic .env file updates with backup
    - `WindowsServiceManager` - Stop/start ProfessionalSMART service
    - `MigrationService` - Detect and apply pending migrations
    - `BackupService` - pg_dump backup operations
  - **Installer Integration:**
    - Added `pro-project.exe` to installer package
    - Start Menu shortcut: "Project Database Manager"

## [2.12.32.0] - 2025-12-27

### Fixed
- **Database Name Case Preservation**: Fixed PostgreSQL case sensitivity issue where database names with mixed case were being lowercased during creation.
  - Added proper SQL identifier quoting in `CREATE DATABASE` commands in `CreateDatabase.vbs`
  - Database names like `professional_smart_clientA` now preserve case correctly
  - Service was failing to start because it looked for `professional_smart_clientA` but database was created as `professional_smart_clienta`

## [2.12.31.0] - 2025-12-27

### Fixed
- **Encounter View Column Bug**: Removed non-existent `encounter_group_id` column from `claims.encounter_view`.
  - Column was referenced in migration 068 but doesn't exist in `claims.encounter` table
  - Fixed in both `068_create_encounter_view.sql` and `000_baseline_v2.12.sql`

## [2.12.30.0] - 2025-12-27

### Fixed
- **SmartProAudit Database Name Case**: Fixed PostgreSQL case sensitivity issue with database name.
  - Changed `SmartProAudit` to lowercase `smartproaudit` throughout installer and migrations
  - PostgreSQL lowercases unquoted identifiers, causing connection failures with mixed-case names
  - Affected files: `CreateDatabase.vbs`, `069_setup_smartproaudit_fdw.sql`, `000_baseline_v2.12.sql`

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
