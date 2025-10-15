# Configuration Guide

This guide covers all configuration options for the Professional SMART claims processing system.

## Configuration File

The primary configuration is stored in `.env` file in the project root directory.

### Location
```
C:\Users\YourUsername\pro\.env
```

## Database Configuration

### DATABASE_URL
**Description**: PostgreSQL connection string
**Format**: `postgres://username:password@host:port/database`
**Default**: None (required)
**Example**:
```env
DATABASE_URL=postgres://pro_user:secure_password@localhost:5432/professional_smart
```

### DATABASE_MAX_CONNECTIONS
**Description**: Maximum number of database connections in the pool
**Type**: Integer
**Default**: 50
**Range**: 10-100
**Recommended**: 50 for production, 20 for development

**Example**:
```env
DATABASE_MAX_CONNECTIONS=50
```

**Tuning Notes**:
- Higher values allow more concurrent operations
- Set based on PostgreSQL `max_connections` setting
- Monitor connection usage with `pg_stat_activity`

### DATABASE_MIN_CONNECTIONS
**Description**: Minimum number of database connections to maintain
**Type**: Integer
**Default**: 5
**Range**: 1-20
**Recommended**: 5 for production, 2 for development

**Example**:
```env
DATABASE_MIN_CONNECTIONS=5
```

### DATABASE_CONNECT_TIMEOUT
**Description**: Connection timeout in seconds
**Type**: Integer
**Default**: 30
**Range**: 5-300

**Example**:
```env
DATABASE_CONNECT_TIMEOUT=30
```

## Application Configuration

### RUST_LOG
**Description**: Logging level for the application
**Type**: String
**Default**: info
**Options**: `error`, `warn`, `info`, `debug`, `trace`

**Example**:
```env
RUST_LOG=info
```

**Module-Specific Logging**:
```env
RUST_LOG=pro_worker=debug,pro_parser_edi=info,sqlx=warn
```

### RUST_BACKTRACE
**Description**: Enable detailed error backtraces
**Type**: Integer/String
**Default**: 0
**Options**: `0` (disabled), `1` (enabled), `full` (detailed)

**Example** (development):
```env
RUST_BACKTRACE=1
```

**Example** (production):
```env
RUST_BACKTRACE=0
```

## Performance Configuration

### BATCH_SIZE
**Description**: Number of claims to process in each batch
**Type**: Integer
**Default**: 1000
**Range**: 100-10000
**Recommended**: 1000 for balanced performance

**Example**:
```env
BATCH_SIZE=1000
```

**Tuning Guidelines**:
- Smaller batches (100-500): Lower memory usage, more frequent database commits
- Larger batches (2000-5000): Higher throughput, more memory usage
- Very large batches (10000+): May cause memory issues

### MAX_WORKERS
**Description**: Number of parallel worker threads
**Type**: Integer
**Default**: Number of CPU cores
**Range**: 1-32
**Recommended**: Number of CPU cores

**Example**:
```env
MAX_WORKERS=8
```

**Tuning Guidelines**:
- Set to number of CPU cores for optimal CPU utilization
- Reduce if memory constrained
- Increase database connection pool if increasing workers

### WORKER_TIMEOUT_SECONDS
**Description**: Timeout for individual worker operations
**Type**: Integer
**Default**: 300 (5 minutes)
**Range**: 60-3600

**Example**:
```env
WORKER_TIMEOUT_SECONDS=300
```

## File Processing Configuration

### INPUT_DIRECTORY
**Description**: Directory to monitor for incoming files
**Type**: Path
**Default**: `./data/input`

**Example**:
```env
INPUT_DIRECTORY=C:\Claims\Input
```

### PROCESSED_DIRECTORY
**Description**: Directory to move successfully processed files
**Type**: Path
**Default**: `./data/processed`

**Example**:
```env
PROCESSED_DIRECTORY=C:\Claims\Processed
```

### ERROR_DIRECTORY
**Description**: Directory to move files with errors
**Type**: Path
**Default**: `./data/error`

**Example**:
```env
ERROR_DIRECTORY=C:\Claims\Error
```

### AUTO_PROCESS_FILES
**Description**: Automatically process files in input directory
**Type**: Boolean
**Default**: false
**Options**: `true`, `false`

**Example**:
```env
AUTO_PROCESS_FILES=true
```

## Rules Engine Configuration

### ENABLE_RULES_ENGINE
**Description**: Enable/disable rules engine processing
**Type**: Boolean
**Default**: true

**Example**:
```env
ENABLE_RULES_ENGINE=true
```

### RULES_SEVERITY_THRESHOLD
**Description**: Minimum severity level to create flags
**Type**: String
**Default**: Low
**Options**: `Low`, `Medium`, `High`

**Example**:
```env
RULES_SEVERITY_THRESHOLD=Medium
```

### ENABLE_AUTO_CODING_SUGGESTIONS
**Description**: Generate automatic coding suggestions for under-coded claims
**Type**: Boolean
**Default**: true

**Example**:
```env
ENABLE_AUTO_CODING_SUGGESTIONS=true
```

## RVU Calculation Configuration

### ENABLE_RVU_CALCULATION
**Description**: Enable/disable RVU payment calculations
**Type**: Boolean
**Default**: true

**Example**:
```env
ENABLE_RVU_CALCULATION=true
```

### DEFAULT_GPCI_LOCALITY
**Description**: Default GPCI locality code when not specified
**Type**: String
**Default**: `00` (National Average)

**Example**:
```env
DEFAULT_GPCI_LOCALITY=00
```

**Common Localities**:
- `00`: National Average
- `01`: Manhattan, NY
- `02`: Queens, NY
- `03`: Nassau/Suffolk, NY
- `05`: Los Angeles, CA
- `07`: San Francisco, CA
- `16`: Chicago, IL
- `18`: Boston, MA
- `23`: Miami, FL
- `26`: Dallas, TX
- `27`: Rest of Texas

### RVU_YEAR
**Description**: RVU data year to use for calculations
**Type**: Integer
**Default**: Current year

**Example**:
```env
RVU_YEAR=2024
```

## Validation Configuration

### ENABLE_FILE_HASH_VALIDATION
**Description**: Check for duplicate files using SHA-256 hash
**Type**: Boolean
**Default**: true

**Example**:
```env
ENABLE_FILE_HASH_VALIDATION=true
```

### ENABLE_PCN_VALIDATION
**Description**: Check for duplicate patient control numbers
**Type**: Boolean
**Default**: true

**Example**:
```env
ENABLE_PCN_VALIDATION=true
```

### ENABLE_SERVICE_LINE_VALIDATION
**Description**: Check for duplicate service lines
**Type**: Boolean
**Default**: true

**Example**:
```env
ENABLE_SERVICE_LINE_VALIDATION=true
```

## Audit and Logging Configuration

### ENABLE_AUDIT_TRAIL
**Description**: Record all data modifications in audit tables
**Type**: Boolean
**Default**: true

**Example**:
```env
ENABLE_AUDIT_TRAIL=true
```

### LOG_DIRECTORY
**Description**: Directory for application log files
**Type**: Path
**Default**: `./logs`

**Example**:
```env
LOG_DIRECTORY=C:\Claims\Logs
```

### LOG_ROTATION_SIZE_MB
**Description**: Size in MB before rotating log files
**Type**: Integer
**Default**: 100

**Example**:
```env
LOG_ROTATION_SIZE_MB=100
```

### LOG_RETENTION_DAYS
**Description**: Number of days to retain log files
**Type**: Integer
**Default**: 30

**Example**:
```env
LOG_RETENTION_DAYS=30
```

## Example Configuration Files

### Development Configuration
```env
# Database
DATABASE_URL=postgres://pro_user:dev_password@localhost:5432/professional_smart
DATABASE_MAX_CONNECTIONS=20
DATABASE_MIN_CONNECTIONS=2

# Logging
RUST_LOG=debug
RUST_BACKTRACE=1

# Performance
BATCH_SIZE=100
MAX_WORKERS=2

# Processing
INPUT_DIRECTORY=C:\Dev\Claims\Input
PROCESSED_DIRECTORY=C:\Dev\Claims\Processed
ERROR_DIRECTORY=C:\Dev\Claims\Error
AUTO_PROCESS_FILES=false

# Features
ENABLE_RULES_ENGINE=true
ENABLE_RVU_CALCULATION=true
ENABLE_AUTO_CODING_SUGGESTIONS=true
```

### Production Configuration
```env
# Database
DATABASE_URL=postgres://pro_user:strong_prod_password@db-server:5432/professional_smart
DATABASE_MAX_CONNECTIONS=50
DATABASE_MIN_CONNECTIONS=10
DATABASE_CONNECT_TIMEOUT=30

# Logging
RUST_LOG=info
RUST_BACKTRACE=0

# Performance
BATCH_SIZE=1000
MAX_WORKERS=8
WORKER_TIMEOUT_SECONDS=300

# Processing
INPUT_DIRECTORY=E:\Claims\Input
PROCESSED_DIRECTORY=E:\Claims\Processed
ERROR_DIRECTORY=E:\Claims\Error
AUTO_PROCESS_FILES=true

# Rules Engine
ENABLE_RULES_ENGINE=true
RULES_SEVERITY_THRESHOLD=Low
ENABLE_AUTO_CODING_SUGGESTIONS=true

# RVU Calculation
ENABLE_RVU_CALCULATION=true
DEFAULT_GPCI_LOCALITY=00
RVU_YEAR=2024

# Validation
ENABLE_FILE_HASH_VALIDATION=true
ENABLE_PCN_VALIDATION=true
ENABLE_SERVICE_LINE_VALIDATION=true

# Audit
ENABLE_AUDIT_TRAIL=true
LOG_DIRECTORY=E:\Claims\Logs
LOG_ROTATION_SIZE_MB=100
LOG_RETENTION_DAYS=90
```

## Configuration Validation

After creating your `.env` file, validate the configuration:

### Check Database Connection
```cmd
psql -U pro_user -d professional_smart -c "SELECT version();"
```

### Test Configuration
```cmd
cargo test --release
```

### Verify Directories Exist
```cmd
mkdir C:\Claims\Input
mkdir C:\Claims\Processed
mkdir C:\Claims\Error
mkdir C:\Claims\Logs
```

## Security Best Practices

1. **Never commit `.env` to version control**
   - Add `.env` to `.gitignore`

2. **Use strong passwords**
   - Minimum 16 characters
   - Mix of uppercase, lowercase, numbers, symbols

3. **Restrict file permissions**
   - Limit access to `.env` file
   - Only application service account should have read access

4. **Rotate passwords regularly**
   - Change database passwords quarterly
   - Update `.env` file accordingly

5. **Use separate configurations**
   - Different passwords for dev/staging/production
   - Different database servers per environment

## Performance Tuning

See [PERFORMANCE_TUNING.md](PERFORMANCE_TUNING.md) for detailed performance optimization guidance.

### Quick Tuning Tips

**For High Throughput**:
```env
BATCH_SIZE=2000
MAX_WORKERS=16
DATABASE_MAX_CONNECTIONS=80
```

**For Low Memory**:
```env
BATCH_SIZE=500
MAX_WORKERS=4
DATABASE_MAX_CONNECTIONS=20
```

**For Development**:
```env
BATCH_SIZE=100
MAX_WORKERS=2
DATABASE_MAX_CONNECTIONS=10
RUST_LOG=debug
```

## Troubleshooting Configuration

**Configuration not loading**:
- Verify `.env` file exists in project root
- Check for syntax errors (no spaces around `=`)
- Verify file encoding is UTF-8

**Database connection fails**:
- Verify `DATABASE_URL` format
- Test connection with `psql`
- Check PostgreSQL service status

**Performance issues**:
- Increase `BATCH_SIZE` for higher throughput
- Increase `MAX_WORKERS` for parallel processing
- Increase `DATABASE_MAX_CONNECTIONS` for more concurrent operations

## Next Steps

- [Database Setup Guide](DATABASE_SETUP.md)
- [Performance Tuning Guide](PERFORMANCE_TUNING.md)
- [Troubleshooting Guide](TROUBLESHOOTING.md)
