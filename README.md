# Professional SMART

**Version**: 2.12.57.0 | **Rust Edition**: 2021 | **Min Rust Version**: 1.75

A high-performance healthcare claims processing system for 837P professional claims with automated auditing, rules-based flagging, RVU reimbursement estimation, and COB (Coordination of Benefits) tracking.

**Performance**: 1,284 claims/second (192.8% of SRD target) - 10,000 claims in 7.76 seconds

## Overview

Professional SMART is a Windows-native application designed to help healthcare organizations:

- Process and validate 837P EDI claims and CSV imports
- Identify coding issues, documentation gaps, and billing errors
- Track multiple payers per encounter (primary, secondary, tertiary)
- Estimate Medicare reimbursement using CMS Physician Fee Schedule
- Manage denials with root cause analysis and appeal tracking
- Conduct retrospective audits with coder accuracy metrics

## System Architecture

```
+-------------------+     +-------------------+     +-------------------+
|   File Watcher    | --> |  Claims Importer  | --> |  Staging Tables   |
|   (EDI/CSV)       |     |  (Parser + Queue) |     |  (raw_claims)     |
+-------------------+     +-------------------+     +-------------------+
                                                            |
                                                            v
+-------------------+     +-------------------+     +-------------------+
|   Rules Engine    | <-- | Claims Processor  | <-- | Batch Sequencer   |
|   (Flagging)      |     |  (FIFO Workers)   |     |  (Strict Order)   |
+-------------------+     +-------------------+     +-------------------+
                                  |
                                  v
+-------------------+     +-------------------+     +-------------------+
|   encounter       |     |  service_line     |     | encounter_payer   |
|   (claims.*)      |     |  (procedures)     |     |  (COB tracking)   |
+-------------------+     +-------------------+     +-------------------+
```

## Key Features

### Claims Processing
- **837P EDI Parsing**: Full ASC X12N 005010X222A1 compliance with loop identification
- **CSV Import**: Dynamic header mapping with configurable profiles
- **FIFO Processing**: Strict ordering with batch sequencing and stuck recovery
- **Multi-threaded**: 8 concurrent workers with encounter-level grouping

### Coordination of Benefits (COB)
- **encounter_payer table**: Tracks all payers across claim submissions
- **Primary/Secondary/Tertiary**: Full payer responsibility chain
- **Payment tracking**: Prior payer amounts, adjustments, and claim control numbers
- **Claim Filing Indicators**: Medicare (MA/MB), Medicaid (MC), Commercial (CI/BL), etc.

### Rules Engine
- **11 Flag Categories**: Coding, Documentation, E/M levels, Modifiers, Diagnosis
- **24+ Issue Types**: Specific flagging rules per category
- **Facility-level Configuration**: Enable/disable rules per facility
- **Hot Reload**: Update rules without service restart

### RVU Calculation
- **CMS Physician Fee Schedule**: Work, Practice Expense, Malpractice RVUs
- **GPCI Adjustments**: Geographic Practice Cost Index by locality
- **Conversion Factors**: Annual Medicare rates (2024: $32.74)
- **Modifier Support**: -26, -TC, global billing

### Provider Management
- **NPI Enrichment**: Automatic lookup from NPPES registry
- **Taxonomy Codes**: Provider specialty classification
- **Provider Types**: Rendering, Referring, Supervising, Billing, Ordering

## Project Structure

```
pro/
├── Cargo.toml                      # Workspace configuration
├── migrations/                     # PostgreSQL migrations (62 files)
│   ├── 001-010_*.sql              # Core schema and tables
│   ├── 011-030_*.sql              # Views, indexes, ML schema
│   ├── 031-050_*.sql              # Rules, enrichment, optimization
│   ├── 051-062_*.sql              # Archive, patient fields, COB
├── crates/
│   ├── pro-common/                # Shared types, errors, validation
│   ├── pro-db/                    # Database models and queries
│   ├── pro-parser-edi/            # 837P EDI parser with loop handling
│   ├── pro-parser-csv/            # Dynamic CSV parser
│   ├── pro-rules/                 # Rules engine and flagging
│   ├── pro-rvu/                   # RVU calculation engine
│   ├── pro-worker/                # Background job processing
│   ├── pro-ml/                    # Machine learning models
│   ├── pro-service/               # Windows service (main application)
│   ├── pro-upgrade/               # CLI upgrade tool
│   ├── pro-upgrade-manager/       # Migration management with embedded SQL
│   ├── pro-npi-enrichment/        # NPI registry lookup worker
│   ├── pro-setup/                 # First-time setup utility
│   ├── pro-data-loader/           # Master data import CLI
│   └── pro-data-loader-gui/       # Master data import GUI
├── installer/
│   ├── Product.wxs                # WiX installer definition
│   ├── ProfessionalSMART.msi      # Built installer (~11 MB)
│   └── *.wxs                      # Dialog definitions
├── docs/                          # Documentation (21 files)
│   ├── CLAUDE.md                  # Development rules and guidelines
│   ├── INSTALLATION.md            # Installation guide
│   ├── API_DOCUMENTATION.md       # WebSocket API reference
│   ├── RULE_CONFIGURATION_GUIDE.md
│   ├── DATABASE_SCHEMA_REFERENCE.md
│   └── srd.md                     # Software Requirements Document
└── test_data/
    ├── setup/                     # Master data CSVs
    │   ├── organizations.csv
    │   ├── facilities.csv
    │   ├── providers.csv
    │   └── payers.csv
    └── *.edi                      # Test EDI files
```

## Database Schema

PostgreSQL with 4 schemas and 77 tables:

### staging (15 tables)
| Table | Purpose |
|-------|---------|
| `raw_claims` | Parsed claims awaiting processing |
| `import_batch` | File import tracking with metrics |
| `file_processing_queue` | FIFO file queue with priorities |
| `batch_sequences` | Processing order tracking |
| `import_configuration` | CSV/EDI import profiles |
| `rules_configuration` | Facility-level rule settings |

### claims (48 tables)
| Table | Purpose |
|-------|---------|
| `organization` | Top-level org hierarchy |
| `region` | Regional groupings |
| `facility` | Billing facilities with NPI |
| `provider` | Providers with taxonomy codes |
| `encounter` | Main claim record (837P CLM segment) |
| `service_line` | Procedures (837P SV1 segments) |
| `encounter_diagnosis` | ICD-10 codes (837P HI segments) |
| `encounter_payer` | **COB tracking - all payers per encounter** |
| `encounter_flag` | Claim-level flags |
| `service_line_flag` | Line-level flags |
| `flag_category` | 11 flag categories |
| `flag_issue` | 24+ specific issue types |
| `denial_event` | Denial tracking with root cause |
| `audit_assignment` | Retrospective audit workflow |
| `rvu_reference` | CMS fee schedule data |

### ml (6 tables)
| Table | Purpose |
|-------|---------|
| `model_registry` | ML model versioning |
| `model_prediction` | Predictions with explanations |
| `feature_definition` | Feature engineering metadata |

### archive (8 tables)
| Table | Purpose |
|-------|---------|
| `archived_encounter` | Historical claim data |
| `archived_service_line` | Historical procedures |
| `archive_batch` | Archive job tracking |

## Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Rust 2021 Edition |
| Database | PostgreSQL 14+ |
| Web Framework | Axum 0.7 with WebSocket |
| Async Runtime | Tokio (multi-threaded) |
| Database Access | SQLx with compile-time verification |
| Serialization | Serde (JSON, YAML) |
| Logging | Tracing with rolling file appender |
| CLI | Clap 4.5 |
| Installer | WiX Toolset 3.14 |

## Installation

### Production (MSI Installer)

1. **Prerequisites**
   - Windows 10/11 (64-bit)
   - PostgreSQL 14 or higher
   - 4 GB RAM minimum, 8 GB recommended

2. **Install**
   ```
   Run installer/ProfessionalSMART.msi as Administrator
   ```

3. **Configure Database**
   - The installer prompts for database credentials
   - Migrations run automatically on first start

4. **Load Master Data**
   - Use Data Loader GUI from Start Menu
   - Import organizations, facilities, providers, payers

### Development

```bash
# Prerequisites
# - Rust 1.75+
# - PostgreSQL 14+

# Clone and build
git clone <repository>
cd pro
cargo build --release

# Set environment
export DATABASE_URL=postgresql://user:pass@localhost/professional_smart
export RUST_LOG=info

# Run migrations
cargo run -p pro-upgrade -- apply-migrations

# Run service
cargo run -p pro-service --release
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DB_HOST` | localhost | PostgreSQL host |
| `DB_PORT` | 5432 | PostgreSQL port |
| `DB_NAME` | professional_smart | Database name |
| `DB_USER` | postgres | Database user |
| `DB_PASSWORD` | (required) | Database password |
| `LOG_LEVEL` | info | Logging level |
| `WORKER_COUNT` | 8 | Processing workers |
| `BATCH_SIZE` | 100 | Claims per batch |

### File Locations

| Path | Purpose |
|------|---------|
| `C:\Program Files\Professional SMART\bin` | Executables |
| `C:\Program Files\Professional SMART\data\input` | Drop folder for EDI/CSV |
| `C:\Program Files\Professional SMART\data\processed` | Processed files |
| `C:\Program Files\Professional SMART\data\error` | Failed files |
| `C:\ProgramData\Professional SMART\logs` | Log files |

## Flag Categories

| Code | Category | Description |
|------|----------|-------------|
| COD | Coding Issues | Bundled, incorrect, missed charges |
| DOC | Documentation | Missing or insufficient documentation |
| EMO | E/M Over-coded | Evaluation higher than supported |
| EMU | E/M Under-coded | Evaluation lower than supported |
| EMI | E/M Incorrect | Wrong E/M category |
| EMT | E/M Time | Time documentation issues |
| MOD | Modifier Issues | Incorrect, missing, unnecessary |
| OTH | Other Issues | Provider, date, signature |
| QTY | Quantity Issues | Unit count discrepancies |
| SUP | Supervision | Incident-to, split-shared, teaching |
| DX | Diagnosis Issues | Additional, unsupported, incorrect |

## API Endpoints

WebSocket API on port 3000:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ws` | WebSocket | Real-time updates |
| `/api/encounters` | GET | List encounters |
| `/api/encounters/:id` | GET | Encounter details |
| `/api/encounters/:id/flags` | GET | Encounter flags |
| `/api/service-lines/:id` | GET | Service line details |
| `/api/rules` | GET | Active rules |
| `/api/facilities/:id/rules` | GET/PUT | Facility rule config |

## Performance

- **Throughput**: 822.5 claims/second (123.5% of 666 target)
- **Test Results**: 10,000 claims (29,626 service lines) in 36.02 seconds
- **Success Rate**: 98.7% (9,871 completed, 129 failed due to future DOS dates in test data)
- **Sustained Rate**: 290-340 encounters/second (870-1,020 claims/second)
- **Processing**: 8 concurrent workers with strict FIFO ordering
- **Database**: Optimized indexes, partial indexes for status columns
- **Concurrency**: Lock-free provider insertion using INSERT ON CONFLICT DO NOTHING

## Healthcare Standards

| Standard | Implementation |
|----------|----------------|
| 837P | ASC X12N 005010X222A1 |
| ICD-10-CM | Diagnosis validation |
| CPT/HCPCS | Procedure validation |
| NPI | 10-digit validation with check digit |
| MBI | Medicare Beneficiary Identifier format |
| HIPAA | Audit trails, data encryption |

## Development Guidelines

See [docs/CLAUDE.md](docs/CLAUDE.md) for complete rules:

- Never disable features to fix bugs
- No silent fallbacks or swallowed errors
- Always version MSI builds (Product.wxs)
- Run `cargo build --release` after changes
- Test with real EDI files before release
- Document database schema changes

## Documentation

| Document | Description |
|----------|-------------|
| [INSTALLATION.md](docs/INSTALLATION.md) | Installation guide |
| [API_DOCUMENTATION.md](docs/API_DOCUMENTATION.md) | WebSocket API |
| [RULE_CONFIGURATION_GUIDE.md](docs/RULE_CONFIGURATION_GUIDE.md) | Rules setup |
| [DATABASE_SCHEMA_REFERENCE.md](docs/DATABASE_SCHEMA_REFERENCE.md) | Schema details |
| [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | Common issues |
| [srd.md](docs/srd.md) | Software Requirements |

## License

MIT

## Support

- Documentation: `docs/` directory
- Issues: Check logs at `C:\ProgramData\Professional SMART\logs`
- 837P Reference: `docs/837p_compguide.pdf`
