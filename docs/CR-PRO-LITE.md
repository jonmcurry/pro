# Change Request: Professional SMART Lite
# Embedded SQLite Thin Client Branch

**CR Number**: CR-2026-001
**Version**: 1.0
**Date**: 2026-03-10
**Status**: Draft
**Requested By**: Engineering
**Priority**: High
**Branch**: `pro-lite`
**Parent Project**: Professional SMART Claims Processing System v0.3.2

---

## 1. Executive Summary

### 1.1 Purpose

Create a new `pro-lite` branch of Professional SMART that produces a **single, self-contained executable** capable of processing 837P professional claims with an embedded SQLite database. The binary requires **zero installation** on the target client server — no PostgreSQL, no Windows Service registration, no MSI installer, no runtime dependencies.

### 1.2 Business Need

Certain client engagements require claims processing to occur on the **client's own infrastructure** where:

- No software installation is permitted on the client server
- Protected Health Information (PHI) cannot leave the client's network
- The client provides only a shared folder and the ability to run an executable
- Multiple end users need browser-based UI access to view processed results

### 1.3 Scope

| In Scope | Out of Scope |
|----------|-------------|
| New `pro-lite` crate (entry point) | Changes to existing `pro-service` (main branch) |
| SQLite repository layer (`pro-db-lite`) | ML schema and features |
| Embedded Axum web server with static frontend | Archive/retention management |
| Reduced SQLite schema (~25 tables) | Denial workflow tables |
| Bundled RVU/GPCI reference data | NPI enrichment (no outbound API calls) |
| EDI 837P and CSV parsing (unchanged) | Windows Service registration |
| Rules engine (unchanged) | MSI installer |
| RVU calculation (unchanged) | Audit assignment/review workflow |
| File watcher for input directory | Coder/reviewer accuracy tables |
| WAL mode concurrent access | PostgreSQL support in this branch |

---

## 2. Architecture

### 2.1 Current Architecture (main branch)

```
┌─────────────────────────────────────────────────────────────────┐
│                     Windows Service (MSI Install)                │
│                     pro-service.exe (15 MB)                      │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PostgreSQL 14+ (Installed)                    │
│                    79 tables, 4 schemas, 400+ indexes            │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Separate Frontend UI                          │
│                    (connects to PostgreSQL directly)             │
└─────────────────────────────────────────────────────────────────┘
```

**Requirements**: PostgreSQL installation, Windows Service registration, MSI installer, separate frontend deployment.

### 2.2 Pro Lite Architecture (pro-lite branch)

```
┌─────────────────────────────────────────────────────────────────┐
│                     pro-lite.exe (~18-23 MB)                     │
│                     Single binary, zero install                  │
│                                                                  │
│  ┌────────────────┐  ┌──────────────┐  ┌──────────────────────┐ │
│  │ File Watcher   │  │ Claims       │  │ Axum Web Server      │ │
│  │                │  │ Processor    │  │ (REST API + Static   │ │
│  │ Monitors       │  │              │  │  Frontend Assets)    │ │
│  │ ./data/input/  │  │ Parse → Val  │  │                      │ │
│  │ for .edi/.csv  │  │ → Rules →   │  │ :8080                │ │
│  │                │  │ RVU → Store  │  │                      │ │
│  └───────┬────────┘  └──────┬───────┘  └──────────┬───────────┘ │
│          │                  │                      │             │
│          └──────────────────┼──────────────────────┘             │
│                             │                                    │
│                      ┌──────▼───────┐                           │
│                      │   SQLite     │                           │
│                      │   (embedded) │                           │
│                      │              │                           │
│                      │ WAL mode     │                           │
│                      │ ~25 tables   │                           │
│                      │ ~50 indexes  │                           │
│                      └──────────────┘                           │
│                             │                                    │
│                      ┌──────▼───────┐                           │
│                      │ pro-lite.db  │  ← single file on disk   │
│                      └──────────────┘                           │
└─────────────────────────────────────────────────────────────────┘
```

**Requirements**: Copy the .exe to a folder. Run it. Open a browser.

### 2.3 Client Deployment Layout

```
\\client-server\claims\               ← shared folder, no install needed
├── pro-lite.exe                       ← single binary (~18-23 MB)
├── pro-lite.db                        ← created on first run
├── pro-lite.db-wal                    ← SQLite WAL file (auto-created)
├── .env                               ← optional config overrides
├── data/
│   ├── input/                         ← client drops .edi/.csv here
│   ├── processed/                     ← files moved after processing
│   └── errors/                        ← files that failed parsing
├── reference/                         ← optional: external RVU/GPCI data
└── logs/
    └── pro-lite.log                   ← rolling log file
```

### 2.4 End User Access

```
┌───────────────────────────────────────────────────────┐
│  Client Network                                       │
│                                                       │
│  ┌─────────────┐                                     │
│  │ pro-lite.exe│ ← runs on client server             │
│  │ :8080       │                                     │
│  └──────┬──────┘                                     │
│         │                                            │
│    ┌────┴─────────────────────────┐                  │
│    │         LAN / Intranet       │                  │
│    │                              │                  │
│  ┌─▼──────┐ ┌────────┐ ┌────────┐                   │
│  │Browser │ │Browser │ │Browser │  ← end users      │
│  │User A  │ │User B  │ │User C  │    on same network│
│  └────────┘ └────────┘ └────────┘                    │
│                                                       │
│  Access: http://<client-server>:8080                  │
└───────────────────────────────────────────────────────┘
```

Multiple users access the embedded web UI by navigating to the server's IP/hostname on port 8080. The Axum web server serves both the REST API and the static frontend assets from within the same binary.

---

## 3. Technical Specification

### 3.1 New Crates

| Crate | Purpose | Estimated LOC |
|-------|---------|---------------|
| `pro-lite` | Entry point binary — replaces `pro-service` for lite deployments | ~800-1,200 |
| `pro-db-lite` | SQLite repository layer — mirrors `pro-db` interface for SQLite | ~4,000-5,000 |

### 3.2 Shared Crates (Unchanged)

These crates are shared between `main` and `pro-lite` branches with zero modification:

| Crate | LOC | Purpose |
|-------|-----|---------|
| `pro-parser-edi` | 2,263 | EDI 837P parser |
| `pro-parser-csv` | — | CSV parser with dynamic header mapping |
| `pro-rules` | 6,168 | Rules engine (27 rules, 11 categories) |
| `pro-rvu` | 1,567 | RVU payment calculation |
| `pro-common` | 543 | Shared types, errors, validation |
| `pro-worker` | 2,727 | Processing pipeline orchestration |

### 3.3 Modified Crates

| Crate | Change | Reason |
|-------|--------|--------|
| `pro-worker` | Abstract database calls behind trait | Allow SQLite or PostgreSQL backends |
| `pro-rules` | Replace `&PgPool` with generic pool trait | Rules execute queries against either backend |

### 3.4 Dropped Crates (Not in pro-lite)

| Crate | Reason |
|-------|--------|
| `pro-npi-enrichment` | No outbound API calls from client server |
| `pro-setup` | No installation wizard needed |
| `pro-data-loader` / `pro-data-loader-gui` | Reference data bundled in binary |
| `pro-upgrade-manager` / `pro-upgrade` | Single schema init, no migrations |
| `pro-ml` | ML features not in lite |
| `pro-rule-converter` / `pro-rule-converter-gui` | Not needed on client |

### 3.5 New Dependencies

| Dependency | Version | Feature Flags | Purpose | Size Impact |
|------------|---------|---------------|---------|-------------|
| `rusqlite` | 0.31+ | `bundled` | Embedded SQLite (compiles SQLite C source into binary) | +1.5-2 MB |
| `rust-embed` | 8.x | `compression` | Embed static frontend assets into binary | +2-10 MB (depends on frontend) |

### 3.6 Removed Dependencies

| Dependency | Reason | Size Savings |
|------------|--------|-------------|
| `sqlx` (postgres feature) | No PostgreSQL | -2-3 MB |
| `reqwest` + `rustls` | No outbound HTTP (NPI enrichment removed) | -1-2 MB |
| `windows-service` | Not running as a service | -0.3 MB |
| `deadpool-postgres` | No PostgreSQL connection pool | -0.2 MB |

---

## 4. Database Schema (SQLite)

### 4.1 Schema Overview

| Metric | Pro (PostgreSQL) | Pro Lite (SQLite) |
|--------|-----------------|-------------------|
| Tables | 79 | 25 |
| Schemas | 4 (claims, staging, ml, archive) | 1 (single namespace) |
| Indexes | 400+ | ~50 |
| Reference data rows | Loaded via data-loader | Bundled in binary |

### 4.2 Tables Included (25 tables)

**Organization Hierarchy (3)**:
- `organization`
- `region`
- `facility`

**Providers (2)**:
- `provider`
- `provider_taxonomy` (pre-loaded with 383 taxonomy codes)

**Claims Core (5)**:
- `encounter`
- `service_line`
- `encounter_diagnosis`
- `service_line_diagnosis_pointer`
- `encounter_payer`

**Auditing (4)**:
- `encounter_flag`
- `service_line_flag`
- `flag_category` (pre-loaded with 11 categories)
- `flag_issue` (pre-loaded with 24+ issue types)

**Reimbursement (4)**:
- `rvu_reference` (pre-loaded with ~10,000 CPT codes)
- `gpci_reference` (pre-loaded with 89 localities)
- `conversion_factor` (pre-loaded)
- `modifier_adjustment` (pre-loaded)
- `service_line_reimbursement`

**Staging (3)**:
- `raw_claims`
- `import_batch`
- `import_error_log`

**Configuration (2)**:
- `rules_configuration`
- `import_configuration`

**Users (1)**:
- `app_user` (basic auth for UI access)

### 4.3 Tables Excluded (54 tables)

| Excluded Table Group | Count | Reason |
|---------------------|-------|--------|
| Archive schema | 8 | No long-term retention on client server |
| ML schema | 6 | ML features not in lite |
| Denial workflow | 5 | Denial tracking not in lite |
| Audit assignment/review | 4 | Full audit workflow not in lite |
| Coder/reviewer accuracy | 3 | Accuracy tracking not in lite |
| Provider enrichment queue | 2 | No NPI API calls |
| Batch sequences | 2 | Simplified queue (single instance) |
| Materialized views | 4 | SQLite doesn't support; use regular views |
| Rule execution stats | 2 | Simplified metrics |
| File processing queue | 2 | Simplified to single-file processing |
| Remaining staging tables | 6 | Not needed for lite |
| Other | 10 | Various supporting tables |

### 4.4 PostgreSQL → SQLite Type Mapping

| PostgreSQL | SQLite | Notes |
|-----------|--------|-------|
| `UUID` | `TEXT` | Store as 36-char string `xxxxxxxx-xxxx-...` |
| `BIGSERIAL` | `INTEGER PRIMARY KEY` | SQLite auto-increment |
| `TIMESTAMPTZ` | `TEXT` | Store as ISO 8601 string |
| `NUMERIC(12,2)` | `REAL` | Or `TEXT` for exact decimal precision |
| `JSONB` | `TEXT` | Store as JSON string, query with `json_extract()` |
| `CITEXT` | `TEXT COLLATE NOCASE` | Case-insensitive comparison |
| `TEXT[]` (arrays) | `TEXT` | Store as JSON array string |
| `VARCHAR(n)` | `TEXT` | SQLite has no length enforcement |
| `SMALLINT` | `INTEGER` | SQLite uses dynamic typing |
| `BOOLEAN` | `INTEGER` | 0 = false, 1 = true |

### 4.5 SQLite Pragma Configuration

```sql
-- Set on every connection open
PRAGMA journal_mode = WAL;           -- Write-Ahead Logging for concurrent reads
PRAGMA busy_timeout = 5000;          -- Wait 5s for write lock before failing
PRAGMA synchronous = NORMAL;         -- Balance durability vs performance
PRAGMA cache_size = -64000;          -- 64 MB page cache
PRAGMA foreign_keys = ON;            -- Enforce FK constraints
PRAGMA temp_store = MEMORY;          -- Temp tables in RAM
PRAGMA mmap_size = 268435456;        -- 256 MB memory-mapped I/O
PRAGMA page_size = 4096;             -- 4 KB pages (default)
```

### 4.6 Concurrency Model (WAL Mode)

SQLite in WAL mode supports:
- **Multiple concurrent readers** while one writer is active
- **Writer does not block readers** (readers see last committed state)
- **Write lock held only during commit** (microseconds for typical operations)

This is sufficient for Pro Lite because:
- The claims processor is the **single writer** (batch inserts during processing)
- Multiple browser users are **concurrent readers** (dashboard queries, search, reports)
- Write contention only occurs during active file processing
- At typical client volumes (hundreds to low thousands of claims/day), WAL write duration is negligible

---

## 5. Binary Size Estimate

### 5.1 Component Breakdown

| Component | Size | Notes |
|-----------|------|-------|
| **Rust application code** | | |
| Tokio async runtime | ~2.5 MB | Required for async processing + web server |
| Axum + Tower + Hyper | ~1.5 MB | Web server framework |
| Serde + serde_json | ~1.0 MB | Serialization |
| rusqlite (bundled SQLite) | ~1.5 MB | SQLite C library compiled in |
| Clap | ~0.5 MB | CLI argument parsing |
| Tracing ecosystem | ~0.5 MB | Logging |
| Chrono, UUID, Regex, SHA2, Blake3 | ~1.5 MB | Utility crates |
| Application logic (parsers, rules, RVU, worker) | ~3.0 MB | Core Pro processing code |
| **Subtotal (binary without assets)** | **~12-15 MB** | |
| | | |
| **Embedded assets** | | |
| Frontend SPA (compressed) | ~2-8 MB | Depends on frontend complexity |
| Bundled RVU reference data (~10K rows) | ~0.5 MB | Embedded CSV or const data |
| Bundled GPCI reference data (89 rows) | < 0.1 MB | Embedded |
| Bundled taxonomy data (383 rows) | < 0.1 MB | Embedded |
| Bundled flag categories/issues | < 0.1 MB | Embedded |
| SQLite init schema | < 0.1 MB | Embedded SQL |
| **Subtotal (embedded assets)** | **~3-9 MB** | |
| | | |
| **Total estimated binary** | **~15-23 MB** | |

### 5.2 Comparison

| Binary | Size | Database | Install Required |
|--------|------|----------|-----------------|
| pro-service.exe (current) | 15 MB | PostgreSQL (separate) | Yes (MSI + PG) |
| **pro-lite.exe (proposed)** | **18-23 MB** | **SQLite (embedded)** | **No** |

### 5.3 Build Configuration

```toml
[profile.release]
opt-level = 3          # Maximum optimization
lto = "fat"            # Full link-time optimization
codegen-units = 1      # Single codegen unit (smaller binary)
strip = true           # Strip debug symbols
debug = false          # No debug info
panic = "abort"        # Smaller than unwind tables
```

---

## 6. Server Requirements

### 6.1 Minimum Server Requirements

| Resource | Minimum | Recommended | Notes |
|----------|---------|-------------|-------|
| **CPU** | 2 cores | 4 cores | Processing is I/O-bound to SQLite |
| **RAM** | 1 GB | 2 GB | No PostgreSQL overhead |
| **Disk** | 500 MB free | 10 GB free | DB grows ~50-100 MB per 10K claims |
| **OS** | Windows Server 2016+ | Windows Server 2019+ | Or Linux (x86_64) |
| **Network** | LAN access, port 8080 | Same | For browser UI access |
| **.NET / Java / Runtime** | **None** | **None** | Zero runtime dependencies |
| **Admin rights** | **Not required** | Not required | Runs as unprivileged user |
| **Installation** | **None** | None | xcopy deploy |

### 6.2 Memory Breakdown

| Consumer | Idle | Active Processing | Peak (10K claims) |
|----------|------|-------------------|-------------------|
| Pro Lite process | 30 MB | 150 MB | 400 MB |
| SQLite page cache (64 MB configured) | 10 MB | 50 MB | 64 MB |
| SQLite WAL file (memory-mapped) | 1 MB | 20 MB | 50 MB |
| Provider/taxonomy caches | 5 MB | 50 MB | 100 MB |
| Rule result cache | 0 MB | 20 MB | 50 MB |
| Axum web server (10 concurrent users) | 5 MB | 15 MB | 30 MB |
| OS overhead (working set) | 50 MB | 100 MB | 150 MB |
| **Total** | **~100 MB** | **~400 MB** | **~850 MB** |

### 6.3 CPU Utilization

| Scenario | CPU Usage | Notes |
|----------|-----------|-------|
| Idle (file watcher + web server) | < 1% | Polling, serving cached pages |
| Steady processing | 2-3 cores | 4 workers, SQLite serialized writes |
| Peak burst (large file) | 3-4 cores | Parallel parsing + sequential DB writes |
| UI queries (10 concurrent users) | < 0.5 cores | Read-only, SQLite handles well |

### 6.4 Disk Growth

| Claims Volume | SQLite DB Size | WAL File (peak) | Logs (30 days) | Total Disk |
|---------------|---------------|-----------------|----------------|-----------|
| 1K claims | ~10 MB | ~5 MB | ~10 MB | ~50 MB |
| 10K claims | ~100 MB | ~20 MB | ~50 MB | ~200 MB |
| 100K claims | ~1 GB | ~50 MB | ~100 MB | ~1.5 GB |
| 1M claims | ~8 GB | ~100 MB | ~200 MB | ~9 GB |

### 6.5 Processing Performance Estimate

SQLite write throughput constrains Pro Lite compared to PostgreSQL:

| Metric | Pro (PostgreSQL) | Pro Lite (SQLite) | Notes |
|--------|-----------------|-------------------|-------|
| Claims/second (peak) | 1,284 | ~200-400 | SQLite single-writer bottleneck |
| Claims/second (sustained) | 870-1,020 | ~150-300 | WAL mode, batch inserts |
| 10K claims processing time | 7.76 sec | ~30-60 sec | Still well within acceptable range |
| Concurrent UI users | Unlimited (PG handles) | 10-20 | SQLite reader concurrency is adequate |
| Worker count | 12 | 4 | Fewer workers since writes serialize |

**Note**: Pro Lite throughput is lower than Pro but more than sufficient for client deployments processing hundreds to low thousands of claims per day. A 10K-claim file processing in ~30-60 seconds is operationally acceptable.

### 6.6 Pro Lite Processing Configuration Defaults

```env
# Pro Lite defaults (reduced from Pro full)
STAGE2_WORKER_COUNT=4              # vs 12 (SQLite single-writer)
BATCH_SIZE=100                     # vs 250 (smaller batches for SQLite)
MAX_CONCURRENT_ENCOUNTERS=8        # vs 40 (less parallelism needed)
RULE_CACHE_TTL=3600                # Same as Pro
ENABLE_PARALLEL_RULES=true         # Same as Pro
SERVER_HOST=0.0.0.0                # Bind all interfaces (LAN access)
SERVER_PORT=8080                   # Web UI port
LOG_LEVEL=info                     # Default log level
```

---

## 7. Implementation Plan

### 7.1 Phase 1: Branch Setup & Database Abstraction (2-3 weeks)

**Objective**: Create the `pro-lite` branch and introduce a database abstraction layer that allows shared crates to work with either PostgreSQL or SQLite.

**Tasks**:

1. **Create `pro-lite` branch** from `main`
2. **Define repository traits** in `pro-common` or new `pro-db-traits` crate:
   ```rust
   #[async_trait]
   pub trait EncounterRepository: Send + Sync {
       async fn insert_encounter(&self, encounter: &NewEncounter) -> Result<i64>;
       async fn get_encounter(&self, id: i64) -> Result<Option<Encounter>>;
       async fn update_status(&self, id: i64, status: &str) -> Result<()>;
       // ... other methods
   }

   #[async_trait]
   pub trait ServiceLineRepository: Send + Sync {
       async fn insert_service_line(&self, sl: &NewServiceLine) -> Result<i64>;
       async fn get_by_encounter(&self, encounter_id: i64) -> Result<Vec<ServiceLine>>;
       // ... other methods
   }

   // Similar traits for Flag, Provider, RVU, etc.
   ```
3. **Refactor `pro-rules`** to accept trait references instead of `&PgPool`
4. **Refactor `pro-worker`** to accept trait references
5. **Verify main branch still compiles** with PostgreSQL repository implementations

### 7.2 Phase 2: SQLite Repository Layer (3-4 weeks)

**Objective**: Implement the SQLite-backed repository layer.

**Tasks**:

1. **Create `pro-db-lite` crate** with `rusqlite` dependency
2. **Write SQLite schema initialization** (25 tables, ~50 indexes)
3. **Implement all repository traits** for SQLite:
   - `SqliteEncounterRepository`
   - `SqliteServiceLineRepository`
   - `SqliteDiagnosisRepository`
   - `SqliteFlagRepository`
   - `SqliteProviderRepository`
   - `SqliteRvuRepository`
   - `SqliteStagingRepository`
   - `SqliteConfigRepository`
4. **Handle SQLite-specific concerns**:
   - WAL mode initialization
   - Busy timeout configuration
   - UUID generation (no `uuid_generate_v4()` in SQLite)
   - Timestamp formatting (ISO 8601 strings)
   - JSON storage/extraction
   - Case-insensitive text comparison
5. **Implement connection management**:
   - Single write connection (mutex-protected)
   - Read connection pool (multiple readers via WAL)
6. **Bundle reference data** (RVU, GPCI, taxonomy, flag categories) as embedded resources
7. **Write unit tests** against SQLite

### 7.3 Phase 3: Pro Lite Entry Point & Web Server (2-3 weeks)

**Objective**: Build the `pro-lite` binary with embedded web server and frontend.

**Tasks**:

1. **Create `pro-lite` crate** (binary entry point):
   - CLI argument parsing (port, data directory, log level)
   - SQLite database initialization on first run
   - Reference data seeding
   - File watcher startup
   - Claims processor startup (using SQLite repositories)
   - Axum web server startup
2. **Embed frontend static assets** using `rust-embed`
3. **Implement REST API endpoints** for frontend:
   - `GET /api/dashboard` — summary statistics
   - `GET /api/encounters` — paginated encounter list with filters
   - `GET /api/encounters/:id` — encounter detail with service lines and flags
   - `GET /api/service-lines/:id` — service line detail
   - `GET /api/flags` — flag list with filters
   - `GET /api/providers` — provider list
   - `GET /api/imports` — import batch history
   - `GET /api/rules` — active rules configuration
   - `PUT /api/rules/:id` — enable/disable rules per facility
   - `POST /api/auth/login` — basic authentication
4. **Implement basic auth** (username/password stored in `app_user` table, bcrypt hashed)
5. **Configure CORS** for local network access
6. **Add graceful shutdown** (Ctrl+C handler)

### 7.4 Phase 4: Frontend Bundling (2-3 weeks)

**Objective**: Package the existing frontend UI for embedding in the binary.

**Tasks**:

1. **Adapt frontend** to use REST API instead of direct PostgreSQL connection
2. **Build frontend** as static SPA (HTML/JS/CSS bundle)
3. **Embed in binary** via `rust-embed` with compression
4. **Serve from Axum** at root path (`/`)
5. **Verify all UI features work** against the REST API layer
6. **Test with multiple concurrent browser sessions**

### 7.5 Phase 5: Testing & Hardening (1-2 weeks)

**Objective**: Validate Pro Lite end-to-end.

**Tasks**:

1. **Process test EDI files** (use existing `test_data/comprehensive_837p_test.edi`)
2. **Process performance test** (use existing `test_data/perf_test_10k.edi`)
3. **Verify all 27 rules fire correctly** against SQLite
4. **Verify RVU calculations match** Pro (PostgreSQL) results
5. **Test concurrent access** (processing + multiple UI users)
6. **Test on clean Windows Server** with no pre-installed software
7. **Test on minimal Linux server** (optional target)
8. **Verify file watcher** detects and processes new files
9. **Verify error handling** (invalid EDI, corrupt files, disk full)
10. **Document deployment instructions**

---

## 8. Risk Assessment

### 8.1 Technical Risks

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|-----------|
| SQLite write performance insufficient for large files | Medium | Low | Batch inserts, WAL mode, reduced worker count; typical client volumes are well within SQLite capability |
| Repository abstraction breaks existing PostgreSQL code | High | Medium | Comprehensive test suite, feature-flagged compilation, CI runs both backends |
| Frontend requires significant rework for REST API | Medium | Medium | Depends on current frontend architecture; may need API adapter layer |
| SQLite concurrent read/write contention under load | Medium | Low | WAL mode handles this well for read-heavy workloads; only one writer thread |
| Binary size exceeds client constraints | Low | Low | 18-23 MB is small; could strip frontend for API-only mode |
| SQLite database corruption on unexpected shutdown | Medium | Low | WAL mode + `synchronous = NORMAL` provides crash recovery; SQLite is well-tested for durability |

### 8.2 Operational Risks

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|-----------|
| Client firewall blocks port 8080 | Medium | Medium | Make port configurable; document firewall requirements |
| Client server lacks sufficient disk space | Medium | Low | Document requirements; add disk space check on startup |
| No way to remotely troubleshoot issues | High | Medium | Comprehensive logging; include diagnostic endpoint (`/api/health`) |
| Database grows unbounded | Medium | Medium | Add configurable retention/purge; warn at size thresholds |
| No automated updates | Medium | High | Manual binary replacement; document update procedure |

---

## 9. Workspace Configuration

### 9.1 Branch Cargo.toml (pro-lite branch)

```toml
[workspace]
members = [
    "crates/pro-common",
    "crates/pro-db-lite",          # NEW: SQLite repository layer
    "crates/pro-parser-edi",       # SHARED: unchanged
    "crates/pro-parser-csv",       # SHARED: unchanged
    "crates/pro-rules",            # SHARED: modified for trait abstraction
    "crates/pro-rvu",              # SHARED: unchanged
    "crates/pro-worker",           # SHARED: modified for trait abstraction
    "crates/pro-lite",             # NEW: binary entry point
]
resolver = "2"

[workspace.package]
version = "0.1.0"
edition = "2021"
rust-version = "1.75"

[workspace.dependencies]
# Async runtime
tokio = { version = "1.40", features = ["full"] }

# Web framework
axum = { version = "0.7", features = ["macros"] }
tower = { version = "0.5", features = ["full"] }
tower-http = { version = "0.5", features = ["fs", "cors", "trace", "compression-full"] }

# SQLite (replaces sqlx postgres)
rusqlite = { version = "0.31", features = ["bundled", "column_decltype"] }

# Embedded static assets
rust-embed = { version = "8", features = ["compression"] }

# Serialization
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

# Date/Time
chrono = { version = "0.4.38", default-features = false, features = ["serde", "std", "clock"] }

# UUIDs
uuid = { version = "1.11", features = ["serde", "v4"] }

# Logging
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter", "json"] }
tracing-appender = "0.2"

# Error handling
anyhow = "1.0"
thiserror = "1.0"

# CLI
clap = { version = "4.5", features = ["derive", "env"] }

# Hashing
sha2 = "0.10"
blake3 = "1.5"

# CSV
csv = "1.3"

# Regex
regex = "1.11"

# Auth
bcrypt = "0.15"

[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
strip = true
debug = false
panic = "abort"
```

### 9.2 Crate Dependency Graph

```
pro-lite (binary)
├── pro-common
├── pro-db-lite
│   ├── rusqlite (bundled)
│   ├── pro-common
│   └── uuid, chrono, serde_json
├── pro-parser-edi
│   └── pro-common
├── pro-parser-csv
│   └── pro-common
├── pro-rules
│   └── pro-common (trait-based, no DB dependency)
├── pro-rvu
│   └── pro-common
├── pro-worker
│   └── pro-common (trait-based, no DB dependency)
├── axum + tower-http
├── rust-embed
├── tokio
├── clap
└── tracing
```

---

## 10. Acceptance Criteria

### 10.1 Functional Requirements

| # | Requirement | Verification |
|---|-------------|-------------|
| F1 | Single `.exe` file runs without any installation | Deploy to clean Windows Server, execute, confirm startup |
| F2 | Creates SQLite database on first run | Verify `pro-lite.db` is created in working directory |
| F3 | Seeds reference data (RVU, GPCI, taxonomy, flags) on first run | Query tables, confirm row counts match expected |
| F4 | File watcher detects `.edi` and `.csv` files in input directory | Drop test file, confirm processing begins within 5 seconds |
| F5 | Processes 837P EDI files with same parsing accuracy as Pro | Compare parsed output against Pro (PostgreSQL) for identical input |
| F6 | All 27 rules execute and produce correct flags | Run test EDI file, compare flag output against Pro baseline |
| F7 | RVU calculations match Pro output | Compare service_line_reimbursement values for same input |
| F8 | Web UI accessible at `http://<host>:8080` | Open browser from another machine on same LAN |
| F9 | Multiple users can access UI simultaneously | 5+ concurrent browser sessions viewing dashboard and searching |
| F10 | Basic authentication required to access UI | Verify login prompt, verify unauthenticated requests are rejected |
| F11 | Processed files moved to `data/processed/` directory | Confirm file is moved after successful processing |
| F12 | Failed files moved to `data/errors/` directory | Process malformed file, confirm it moves to errors |
| F13 | Rolling log file created in `logs/` directory | Verify log file exists and contains processing entries |

### 10.2 Non-Functional Requirements

| # | Requirement | Target | Verification |
|---|-------------|--------|-------------|
| NF1 | Binary size | < 25 MB | Measure release build output |
| NF2 | Cold startup time | < 3 seconds | Time from execution to web server ready |
| NF3 | Memory usage (idle) | < 150 MB | Monitor with Task Manager |
| NF4 | Memory usage (processing 10K claims) | < 1 GB | Monitor peak during processing |
| NF5 | Process 10K claims | < 120 seconds | Time full processing of perf test file |
| NF6 | Concurrent UI users | 10+ without degradation | Load test with 10 browser sessions |
| NF7 | No installation required | Zero | Run on clean server, no admin rights needed |
| NF8 | No outbound network connections | Zero | Monitor with netstat during operation |
| NF9 | Graceful shutdown on Ctrl+C | Clean exit, no DB corruption | Interrupt during processing, verify DB integrity |

---

## 11. Future Considerations

### 11.1 Potential Enhancements (Not in initial scope)

- **Data export**: Generate de-identified aggregate reports that can be taken off-site
- **Rule updates**: Mechanism to update rules without replacing the binary (sidecar rule file)
- **Multi-client management**: Central dashboard that shows status of all deployed Pro Lite instances (heartbeat/health check only, no PHI transmitted)
- **Linux cross-compilation**: Build for `x86_64-unknown-linux-musl` for fully static Linux binary
- **Encrypted database**: SQLite Encryption Extension (SEE) or SQLCipher for at-rest encryption
- **Auto-purge**: Configurable retention period to prevent unbounded database growth

### 11.2 Branch Maintenance Strategy

The `pro-lite` branch diverges from `main` at the data access layer. Shared crates (parsers, rules, RVU) should be kept in sync:

- **Shared crate changes on `main`** → cherry-pick or merge to `pro-lite`
- **Rule additions on `main`** → automatically available in `pro-lite` (same rules crate)
- **Schema changes on `main`** → evaluate if needed in SQLite schema, apply manually
- **Pro Lite-specific changes** → stay on `pro-lite` branch only

---

## 12. Appendix

### A. Glossary

| Term | Definition |
|------|-----------|
| **837P** | ASC X12N standard for professional healthcare claims |
| **CPT** | Current Procedural Terminology — 5-digit procedure codes |
| **EDI** | Electronic Data Interchange |
| **GPCI** | Geographic Practice Cost Index |
| **ICD-10** | International Classification of Diseases, 10th Revision |
| **NPI** | National Provider Identifier (10-digit) |
| **PHI** | Protected Health Information |
| **RVU** | Relative Value Unit — Medicare payment measure |
| **WAL** | Write-Ahead Logging — SQLite concurrency mode |
| **xcopy deploy** | Deployment by simply copying files, no installer |

### B. Reference: SQLite Limits

| Limit | Value | Pro Lite Impact |
|-------|-------|----------------|
| Max database size | 281 TB | Not a concern |
| Max row size | 1 GB | Not a concern |
| Max columns per table | 2,000 | Not a concern (max ~30 columns) |
| Max concurrent readers (WAL) | Unlimited | Sufficient for 10-20 UI users |
| Concurrent writers | 1 | Processor is single writer; acceptable |
| Max attached databases | 10 | Not using attached databases |

### C. Reference: File Sizes

| File | Typical Size |
|------|-------------|
| pro-lite.exe | 18-23 MB |
| pro-lite.db (10K claims) | ~100 MB |
| pro-lite.db (100K claims) | ~1 GB |
| pro-lite.db-wal (peak) | ~50 MB |
| pro-lite.log (daily) | ~5-10 MB |
| Input EDI file (10K claims) | ~15-20 MB |
