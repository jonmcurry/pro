# Professional SMART

**Version**: 1.5.30.0 | **Release Date**: November 4, 2025

A comprehensive healthcare claims auditing and analytics system for processing 837p professional claims and CSV files with automated flagging, rules engine, RVU-based reimbursement estimation, and denial tracking.

## Overview

Professional SMART is a Windows-based application designed to help healthcare organizations improve coding accuracy, reduce denials, and optimize reimbursement. The system processes claims data from multiple sources, applies sophisticated rules to identify potential issues, and provides comprehensive analytics through PostgreSQL views.

## Key Features

- **Multi-format Import**: Process 837p EDI files and CSV files with dynamic header mapping
- **Comprehensive Rules Engine**: Automated flagging across 11 categories with 24+ specific issue types
- **RVU-based Estimation**: Medicare reimbursement estimation using CMS Physician Fee Schedule data
- **Denial Tracking**: Root cause analysis, preventability assessment, and appeal tracking
- **Audit Management**: Retrospective audit workflow with coder and provider accuracy tracking
- **Machine Learning**: Predictive models for denial risk, coding suggestions, and audit prioritization
- **Performance**: Designed to process 10,000 claims in 15 seconds
- **Seamless Upgrades**: In-place upgrades with automatic data preservation and database backups (v1.1.0+)

## Project Structure

```
pro/
├── Cargo.toml                    # Workspace configuration
├── migrations/                   # PostgreSQL migration files
│   ├── 001_create_schemas.sql
│   ├── 002_create_organization_tables.sql
│   ├── 003_create_provider_tables.sql
│   ├── 004_create_encounter_tables.sql
│   ├── 005_create_diagnosis_procedure_tables.sql
│   ├── 006_create_flag_tables.sql
│   ├── 007_create_staging_tables.sql
│   ├── 008_create_audit_tables.sql
│   ├── 009_create_rvu_tables.sql
│   ├── 010_create_denial_tables.sql
│   ├── 011_create_schedule_tables.sql
│   ├── 012_create_ml_tables.sql
│   ├── 013_create_dashboard_views.sql
│   └── 014_create_utility_functions.sql
├── crates/
│   ├── pro-common/              # Shared types, errors, validation
│   ├── pro-db/                  # Database access layer
│   ├── pro-parser-edi/          # 837p EDI parser
│   ├── pro-parser-csv/          # Dynamic CSV parser
│   ├── pro-rules/               # Rules engine and flagging
│   ├── pro-rvu/                 # RVU calculation and reimbursement
│   ├── pro-api/                 # REST API with Axum
│   ├── pro-worker/              # Background processing worker
│   └── pro-ml/                  # Machine learning models
├── docs/
│   ├── CLAUDE.md                # Project rules and guidelines
│   ├── INSTALLATION.md          # Installation guide (MSI and manual)
│   ├── EDI_PARSING.md          # 837P EDI parser documentation
│   ├── CHANGELOG.md            # Version history and release notes
│   ├── CONFIGURATION.md        # System configuration guide
│   ├── DATABASE_SETUP.md       # Database setup and migration guide
│   ├── srd.md                  # Software Requirements Document
│   └── 837p_compguide.pdf      # CMS 837P Companion Guide
├── installer/
│   └── ProfessionalSMART.msi   # Windows installer package
└── test_data/                  # Test EDI and CSV files

```

## Database Schema

The application uses PostgreSQL with three schemas:

### staging
- **import_batch**: File import tracking and processing metrics
- **file_upload**: Multi-part upload management
- **import_configuration**: CSV/EDI import profiles with header mappings
- **rules_configuration**: Rules engine configuration
- **scheduled_job**: Automated job scheduling
- **report_subscription**: Scheduled report delivery

### claims
- **organization/region/facility**: Organizational hierarchy
- **provider/coder/reviewer**: Personnel management
- **encounter**: Main claim table with all 837p data elements
- **service_line**: Procedure/service line details
- **encounter_diagnosis**: Diagnosis codes (up to 12 per encounter)
- **flag_category/flag_issue**: Predefined flag definitions
- **encounter_flag/service_line_flag**: Issue flagging
- **audit_assignment**: Retrospective audit workflow
- **denial_event**: Denial tracking with root cause
- **rvu_reference**: CMS Physician Fee Schedule data
- **conversion_factor**: Annual Medicare rates

### ml
- **model_registry**: ML model versioning
- **model_prediction**: Predictions with explanations
- **feature_definition**: Feature engineering
- **training_dataset**: Training data management
- **model_performance_log**: Drift detection

## Technology Stack

- **Language**: Rust (edition 2021)
- **Database**: PostgreSQL 14+
- **Web Framework**: Axum 0.7
- **Async Runtime**: Tokio
- **ORM**: SQLx with compile-time query checking
- **Serialization**: Serde
- **Validation**: Validator + custom validation
- **Logging**: Tracing
- **CLI**: Clap

## Quick Start

### Production Installation (Recommended)

1. Download `ProfessionalSMART.msi` from the `installer` directory
2. Run the MSI installer as Administrator
3. Follow the installation wizard
4. See [INSTALLATION.md](docs/INSTALLATION.md) for detailed instructions

### Documentation

- **[Installation Guide](docs/INSTALLATION.md)** - MSI installer and manual installation
- **[EDI Parsing](docs/EDI_PARSING.md)** - 837P format, loop structure, and troubleshooting
- **[Changelog](docs/CHANGELOG.md)** - Version history and release notes
- **[Configuration](docs/CONFIGURATION.md)** - System configuration options
- **[Database Setup](docs/DATABASE_SETUP.md)** - Database migrations and setup
- **[Development Rules](docs/CLAUDE.md)** - Project guidelines and best practices

## Getting Started (Development)

### Prerequisites

- Rust 1.75 or higher
- PostgreSQL 14 or higher
- Windows 10/11 (primary target platform)

### Database Setup

1. Create a PostgreSQL database:
```bash
createdb professional_smart
```

2. Run migrations in order:
```bash
for file in migrations/*.sql; do
    psql -U username -d professional_smart -f "$file"
done
```

3. Verify installation:
```sql
SELECT schema_name FROM information_schema.schemata
WHERE schema_name IN ('staging', 'claims', 'ml');
```

### Build

```bash
# Build all crates
cargo build --release

# Build specific crate
cargo build -p pro-api --release

# Run tests
cargo test --workspace
```

### Configuration

Create a `.env` file in the project root:

```env
DATABASE_URL=postgresql://username:password@localhost/professional_smart
RUST_LOG=info
SERVER_PORT=3000
```

## Current Status

### Completed
- [x] Database schema design with 14 migration files
- [x] Three-schema architecture (staging, claims, ml)
- [x] Comprehensive views for dashboards
- [x] Utility functions and performance indexes
- [x] Rust workspace structure
- [x] Common types and error handling
- [x] Validation functions for healthcare standards

### In Progress
- [x] Database access layer (pro-db)
- [x] 837p EDI parser (pro-parser-edi) - **v1.5.30.0 with critical loop identification fix**
- [x] Dynamic CSV parser (pro-parser-csv)
- [x] Windows service implementation (pro-service)
- [x] Claims importer with facility lookup
- [x] Claims processor with service line handling
- [x] File watcher and batch processing
- [x] MSI installer with automated setup

### Pending
- [ ] Rules engine implementation
- [ ] Flagging system
- [ ] RVU calculation engine
- [ ] REST API endpoints
- [ ] Background worker service
- [ ] ML model integration
- [ ] Windows GUI installer
- [ ] Comprehensive test suite
- [ ] Deployment documentation

## Flag Categories

The system supports 11 flag categories from the SRD:

1. **COD** - Coding Issues: Bundled, incorrect, missed charges
2. **DOC** - Documentation Issues: Missing or insufficient
3. **EMO** - E/M Over-coded: 1+ levels higher than supported
4. **EMU** - E/M Under-coded: 1+ levels lower than supported
5. **EMI** - E/M Incorrect Category: Wrong E/M category
6. **EMT** - E/M Time: Time not documented
7. **MOD** - Modifier Issues: Incorrect, missing, unnecessary
8. **OTH** - Other Issues: Provider, date, signature problems
9. **QTY** - Quantity Issues: Unit count discrepancies
10. **SUP** - Supervision: Incident-to, split-shared, teaching
11. **DX** - Diagnosis Issues: Additional, unsupported, incorrect

## Performance Targets

- **Import Speed**: 10,000 claims in 15 seconds (666 claims/sec)
- **Database**: Optimized with composite and partial indexes
- **Concurrency**: Async processing with Tokio
- **Caching**: Materialized views for dashboard performance

## Installation and Upgrade

### Fresh Installation
For detailed installation instructions, see [docs/INSTALLATION.md](docs/INSTALLATION.md)

Quick start:
1. Install PostgreSQL 14+
2. Download and run the `ProfessionalSMART.msi` installer
3. Follow the installation wizard
4. Configure database settings
5. Load master data using the Data Loader tool

### Upgrading
Starting with version 1.1.0, Professional SMART supports seamless in-place upgrades. See [docs/UPGRADE_GUIDE.md](docs/UPGRADE_GUIDE.md) for complete upgrade instructions.

Quick upgrade:
1. Download the latest MSI installer
2. Run the installer (it auto-detects existing installations)
3. Your data and configuration are preserved automatically
4. Database backup created before upgrade (recommended)

## Healthcare Standards Compliance

- **837p Professional Claims**: ASC X12N Version 005010X222A1
- **ICD-10-CM**: Diagnosis code validation
- **CPT/HCPCS**: Procedure code validation
- **NPI**: National Provider Identifier validation
- **MBI**: Medicare Beneficiary Identifier validation
- **HIPAA**: Compliant data handling and audit trails

## Development Guidelines

See `docs/CLAUDE.md` for project rules:

- Never disable or remove features to fix bugs
- No silent fallbacks or failures
- Always check official documentation
- Clean up temporary files
- No Docker (Windows-only application)
- No emoji in code or documentation
- Create markdown plans for new features
- Resolve issues properly without shortcuts

## License

MIT

## Support

For questions or issues, refer to:
- Software Requirements Document: `docs/srd.md`
- 837P Companion Guide: `docs/837p_compguide.pdf`
- Migration README: `migrations/README.md`
