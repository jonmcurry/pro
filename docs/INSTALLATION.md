# Installation Guide

This guide provides step-by-step instructions for installing the Professional SMART claims processing system on Windows.

**Current Version**: 2.12.32.0
**Release Date**: December 27, 2025

## Quick Start (MSI Installer)

For production deployments, use the MSI installer for automated installation:

1. **Download** `ProfessionalSMART.msi` from the `installer` directory
2. **Run** the MSI installer as Administrator
3. **Follow** the installation wizard:
   - Configure database connection (server, database name, credentials)
   - Set data directories (input, processed, error)
   - Configure service startup options
4. **Verify** installation by checking Windows Services for "Professional SMART"

The MSI installer automatically:
- Installs the service executable and dependencies
- Creates necessary directories (`data\input`, `data\processed`, `data\error`, `logs`)
- Creates the project database with all 69 migrations (001-069)
- Creates the SmartProAudit master database for centralized management
- Configures the Windows service
- Generates `.env` configuration file
- Registers the project in SmartProAudit

### Databases Created

The installer creates two databases:

1. **Project Database** (e.g., `professional_smart_clientA`)
   - Main claims processing database
   - Schemas: `claims`, `staging`, `analytics`, `archive`, `ml`, `smartproaudit`
   - All 69 migrations applied via baseline

2. **SmartProAudit** (master database)
   - Centralized project registry
   - Schemas: `projects`, `fields`, `security`
   - Shared across all project databases

Skip to [Post-Installation Verification](#installation-verification) after MSI installation.

---

## Upgrading from Previous Version

**If you already have Professional SMART installed**, please see the [UPGRADE_GUIDE.md](UPGRADE_GUIDE.md) for upgrade instructions. Starting with version 1.1.0, you can upgrade in-place without losing your data.

**Quick upgrade steps:**
1. Download the latest MSI installer
2. Run the installer (it will detect your existing installation)
3. Follow the prompts to complete the upgrade
4. Your data and configuration will be preserved automatically

For detailed upgrade instructions, troubleshooting, and rollback procedures, see the complete [Upgrade Guide](UPGRADE_GUIDE.md).

---

## Fresh Installation

The following instructions are for fresh installations on systems that have never had Professional SMART installed.

## System Requirements

### Minimum Requirements
- **Operating System**: Windows 10 or Windows Server 2019 (or later)
- **RAM**: 8GB minimum, 16GB recommended
- **Disk Space**: 10GB for application and database
- **CPU**: 4 cores minimum, 8 cores recommended for optimal performance

### Software Requirements
- **PostgreSQL**: Version 14 or later
- **Rust**: Latest stable version (1.70+)
- **Git**: For cloning the repository

## Step 1: Install PostgreSQL

### Download PostgreSQL
1. Visit https://www.postgresql.org/download/windows/
2. Download the PostgreSQL installer for Windows (version 14 or later)
3. Run the installer as Administrator

### PostgreSQL Installation
1. Accept the default installation directory: `C:\Program Files\PostgreSQL\14\`
2. Select components to install:
   - [x] PostgreSQL Server
   - [x] pgAdmin 4
   - [x] Command Line Tools
   - [ ] Stack Builder (optional)
3. Set data directory: `C:\Program Files\PostgreSQL\14\data\`
4. Set a strong password for the `postgres` superuser (remember this!)
5. Set port: `5432` (default)
6. Set locale: `English, United States`
7. Complete the installation

### Verify PostgreSQL Installation
Open Command Prompt and run:
```cmd
psql --version
```

Expected output:
```
psql (PostgreSQL) 14.x
```

## Step 2: Install Rust Toolchain

### Download Rust
1. Visit https://www.rust-lang.org/tools/install
2. Download `rustup-init.exe`
3. Run the installer

### Rust Installation
1. Select option 1: "Proceed with installation (default)"
2. Wait for installation to complete
3. Close and reopen Command Prompt

### Verify Rust Installation
```cmd
rustc --version
cargo --version
```

Expected output:
```
rustc 1.xx.x
cargo 1.xx.x
```

## Step 3: Clone the Repository

### Using Git
```cmd
cd C:\Users\YourUsername\
git clone https://path/to/pro.git
cd pro
```

### Alternative: Download ZIP
1. Download the project as ZIP
2. Extract to `C:\Users\YourUsername\pro\`
3. Open Command Prompt in that directory

## Step 4: Create PostgreSQL Database

### Using psql
```cmd
psql -U postgres
```

Enter your PostgreSQL password when prompted.

### Create Database
```sql
CREATE DATABASE professional_smart;
CREATE USER pro_user WITH ENCRYPTED PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE professional_smart TO pro_user;
\q
```

### Verify Database Creation
```cmd
psql -U pro_user -d professional_smart
```

If successful, you'll see:
```
professional_smart=>
```

Type `\q` to exit.

## Step 5: Run Database Migrations

### Navigate to Migration Directory
```cmd
cd C:\Users\YourUsername\pro\migrations
```

### Execute Migrations in Order
```cmd
psql -U pro_user -d professional_smart -f 001_create_schemas.sql
psql -U pro_user -d professional_smart -f 002_create_organization_tables.sql
psql -U pro_user -d professional_smart -f 003_create_provider_tables.sql
psql -U pro_user -d professional_smart -f 004_create_encounter_tables.sql
psql -U pro_user -d professional_smart -f 005_create_diagnosis_procedure_tables.sql
psql -U pro_user -d professional_smart -f 006_create_flag_tables.sql
psql -U pro_user -d professional_smart -f 007_create_staging_tables.sql
psql -U pro_user -d professional_smart -f 008_create_audit_tables.sql
psql -U pro_user -d professional_smart -f 009_create_rvu_tables.sql
psql -U pro_user -d professional_smart -f 010_create_denial_tables.sql
psql -U pro_user -d professional_smart -f 011_create_schedule_tables.sql
psql -U pro_user -d professional_smart -f 012_create_ml_tables.sql
psql -U pro_user -d professional_smart -f 013_create_dashboard_views.sql
psql -U pro_user -d professional_smart -f 014_create_utility_functions.sql
```

### Verify Schema Creation
```cmd
psql -U pro_user -d professional_smart -c "\dn"
```

Expected output should show three schemas:
```
  Name   | Owner
---------+----------
 claims  | pro_user
 ml      | pro_user
 staging | pro_user
```

## Step 6: Configure Environment Variables

### Create Environment File
Create a file named `.env` in the project root directory:

```
C:\Users\YourUsername\pro\.env
```

### Add Configuration
Copy from `.env.example` and customize. Key settings:

```env
# Database Configuration (required)
DATABASE_URL=postgres://pro_user:your_secure_password@localhost:5432/professional_smart

# Database pool settings
DB_MAX_CONNECTIONS=100
DB_MIN_CONNECTIONS=10
DB_ACQUIRE_TIMEOUT=30
DB_IDLE_TIMEOUT=600
DB_MAX_LIFETIME=1800

# Logging (optional)
RUST_LOG=info
LOG_FORMAT=pretty

# Environment
ENVIRONMENT=production

# Worker Configuration
BATCH_SIZE=100
MAX_CONCURRENT_BATCHES=4

# File processing paths (used by Windows service)
INPUT_DIR=C:\Program Files\Professional SMART\data\input

# Performance Tuning
RULE_CACHE_TTL=3600
ENABLE_PARALLEL_RULES=true
```

See `.env.example` for complete configuration options including:
- Streaming processing (ENABLE_STREAMING)
- WebSocket server (WEBSOCKET_HOST)
- Security settings (JWT_SECRET, rate limiting)
- Development settings (SQLX_LOGGING, DEV_DISABLE_AUTH)

## Step 7: Build the Application

### Navigate to Project Root
```cmd
cd C:\Users\YourUsername\pro
```

### Build in Release Mode
```cmd
cargo build --release
```

This will take 5-10 minutes on first build. Expected output:
```
   Compiling pro-common v0.1.0
   Compiling pro-db v0.1.0
   Compiling pro-parser-edi v0.1.0
   ...
    Finished release [optimized] target(s) in 8m 32s
```

### Verify Build
```cmd
dir target\release
```

You should see compiled binaries in this directory.

## Step 8: Run Tests

### Run Unit Tests
```cmd
cargo test
```

Expected: All 78 tests should pass.

### Run Performance Benchmarks (Optional)
```cmd
cargo bench
```

This validates that the system meets performance targets (666 claims/sec).

## Step 9: Initial Configuration

See [CONFIGURATION.md](CONFIGURATION.md) for detailed configuration options.

### Create Initial Organization
```cmd
psql -U pro_user -d professional_smart
```

```sql
INSERT INTO claims.organization (
    organization_id,
    organization_code,
    organization_name,
    tax_id,
    is_active
) VALUES (
    gen_random_uuid(),
    'ORG001',
    'Sample Healthcare Organization',
    '12-3456789',
    true
);
```

## Installation Verification

### Check Database Connection
```cmd
psql -U pro_user -d professional_smart -c "SELECT COUNT(*) FROM claims.organization;"
```

Expected output:
```
 count
-------
     1
```

### Check Application Build
```cmd
cargo test --release
```

All tests should pass.

## Next Steps

1. **Configure the System**: See [CONFIGURATION.md](CONFIGURATION.md)
2. **Set Up Database**: See [DATABASE_SETUP.md](DATABASE_SETUP.md)
3. **Import RVU Data**: See [DATABASE_SETUP.md](DATABASE_SETUP.md#import-rvu-data)
4. **Configure Organizations**: Add facilities, providers, and coders
5. **Understand EDI Parsing**: See [EDI_PARSING.md](EDI_PARSING.md) for 837P format details

## Troubleshooting

If you encounter issues during installation, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

### Common Issues

**PostgreSQL Connection Failed**
- Verify PostgreSQL service is running: `sc query postgresql-x64-14`
- Check firewall settings allow port 5432
- Verify password is correct

**Rust Compilation Errors**
- Update Rust: `rustup update`
- Clear build cache: `cargo clean`
- Rebuild: `cargo build --release`

**Migration Errors**
- Verify migrations run in correct order
- Check PostgreSQL logs: `C:\Program Files\PostgreSQL\14\data\log\`
- Drop and recreate database if needed (development only)

## Support

For additional help:
- Review [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- Check application logs
- Review PostgreSQL logs
- Consult the project documentation

## Security Notes

1. **Change default password**: Never use default passwords in production
2. **Restrict access**: Configure PostgreSQL `pg_hba.conf` appropriately
3. **Enable SSL**: For production deployments, enable SSL for database connections
4. **Backup regularly**: Set up automated database backups
5. **Update regularly**: Keep PostgreSQL and Rust toolchain updated

## Performance Notes

- For optimal performance, allocate at least 16GB RAM
- Use SSD storage for database files
- Configure PostgreSQL shared_buffers to 25% of RAM
- Monitor system resources during operation
- Run performance benchmarks to validate targets (666 claims/sec)
