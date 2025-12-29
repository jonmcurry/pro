# Performance Tuning Guide

This guide provides detailed instructions for optimizing the Professional SMART claims processing system to meet or exceed the target performance of **666 claims per second**.

## Current Performance (v2.12.46.0)

| Metric | Achieved | Target | Status |
|--------|----------|--------|--------|
| **Throughput** | **1,284 claims/sec** | 666 claims/sec | ✅ **192.8%** |
| **Processing Time** | **7.76 seconds** | 15 seconds | 9,971 claims |
| **Success Rate** | 99.7%+ | 95%+ | ✅ |
| **SRD Target** | **ACHIEVED** | 10k claims in 15s | ✅ |

### Performance History

| Version | Throughput | Key Change |
|---------|------------|------------|
| v2.12.44.0 | ~190 claims/sec | Baseline with default config |
| v2.12.45.0 | ~195 claims/sec | Trigger removal (+2.6%) |
| **v2.12.46.0** | **1,284 claims/sec** | Provider cache (+558%) |

### Key Optimizations Applied

1. **Provider Cache (v2.12.46.0)** - In-memory NPI → provider_id cache eliminates redundant DB upserts. Same provider NPI appearing up to 16 times per encounter (4 encounter-level + 4 per service line × ~3 lines) now only hits DB once. **This single optimization delivered 558% improvement.**

2. **Removed sync_encounter_totals Triggers (v2.12.45.0)** - Dropped triggers that fired for every service line INSERT, eliminating ~60,000 extra DB operations for 10k claims.

3. **Removed Provider Advisory Locks** - Advisory locks caused 96% failure rate when multiple workers processed claims with the same provider NPI. The `ensure_provider_exists` function uses `INSERT ON CONFLICT DO UPDATE` which is safe for concurrent access.

4. **Simplified FIFO Batch Acquisition** - Replaced complex CTE-based encounter grouping with simple FIFO-ordered claim acquisition using `FOR UPDATE SKIP LOCKED`.

5. **Per-Encounter Transactions** - Each encounter has its own transaction; failures don't cascade to other encounters in the batch.

6. **Parser Logging Optimization** - Downgraded loop debug logging from INFO to DEBUG level, eliminating 80,000+ log entries per 10k claims.

7. **PostgreSQL Configuration** - Enabled autovacuum, reduced work_mem from 512MB to 64MB, configured synchronous_commit=off for throughput.

8. **Optimal Default Configuration (v2.12.44.0)** - Set `STAGE2_WORKER_COUNT=8` and `BATCH_SIZE=750` as installer defaults.

## Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Throughput** | 666 claims/sec | 10,000 claims in ≤ 15 seconds |
| **Parser Performance** | < 1ms per claim | EDI/CSV parsing time |
| **Rules Engine** | < 5ms per claim | All 27 rules execution |
| **RVU Calculation** | < 1ms per service line | Payment calculation |
| **Database Operations** | < 10ms per claim | Insert encounter + lines + flags |
| **Memory Usage** | < 2GB for 10K claims | Peak memory during processing |

## Baseline Performance Measurement

### Run Performance Benchmarks

```cmd
cd C:\Users\YourUsername\pro
cargo bench
```

This generates:
- Detailed benchmark results
- HTML reports in `target/criterion/`
- Performance comparisons

### Key Benchmarks to Review

1. **parser_benchmarks**:
   - `edi_parser/parse/1000`: Time to parse 1000 claims
   - `csv_parser/parse/1000`: Time to parse 1000 rows
   - `throughput_validation/10k_claims_target`: Validates 666 claims/sec target

2. **pipeline_benchmarks**:
   - `rvu_calculation/calculate_payment`: RVU calculation time
   - `throughput_claims_per_second/parsing_throughput/10000`: Overall throughput
   - `memory_pressure/10k_claims_memory`: Memory usage pattern

### Interpret Results

Criterion output format:
```
edi_parser/parse/1000   time:   [1.2345 s 1.2456 s 1.2567 s]
                        thrpt:  [795.82 elem/s 802.91 elem/s 810.00 elem/s]
```

- **time**: Time to complete (lower is better)
- **thrpt**: Throughput (higher is better)
- **elem/s**: Elements (claims) per second

**Good Performance**:
- 1000 claims parse in < 1.5 seconds (666+ claims/sec)
- Single claim parse in < 1.5ms
- 10,000 claims parse in < 15 seconds

**Poor Performance**:
- 1000 claims parse in > 2 seconds (< 500 claims/sec)
- Single claim parse in > 2ms
- 10,000 claims parse in > 20 seconds

## Application Tuning

### 1. Batch Size Optimization

The `BATCH_SIZE` parameter significantly impacts performance.

**Test Different Batch Sizes**:

```env
# Small batches (memory constrained)
BATCH_SIZE=500

# Medium batches (balanced)
BATCH_SIZE=1000

# Large batches (high throughput)
BATCH_SIZE=2000

# Very large batches (maximum throughput)
BATCH_SIZE=5000
```

**Guidelines**:
- **Small systems** (8GB RAM): 500-1000
- **Medium systems** (16GB RAM): 1000-2000
- **Large systems** (32GB+ RAM): 2000-5000

**Trade-offs**:
- Larger batches: Higher throughput, more memory, less frequent commits
- Smaller batches: Lower memory, more frequent commits, lower throughput

### 2. Worker Thread Optimization

The `MAX_WORKERS` parameter controls parallel processing.

**Optimal Settings**:
```env
# For CPU-bound workloads (parsing, rules)
MAX_WORKERS={number of CPU cores}

# For I/O-bound workloads (database heavy)
MAX_WORKERS={2x number of CPU cores}

# Examples
MAX_WORKERS=4   # 4-core system
MAX_WORKERS=8   # 8-core system
MAX_WORKERS=16  # 16-core system
```

**Guidelines**:
- Start with number of CPU cores
- Increase if CPU usage < 80%
- Decrease if context switching is high
- Monitor with Task Manager

### 3. Database Connection Pool Tuning

Balance connection pool with worker threads:

```env
# Formula: MAX_CONNECTIONS = MAX_WORKERS * 2-4
MAX_WORKERS=8
DATABASE_MAX_CONNECTIONS=32

# Conservative (fewer connections per worker)
DATABASE_MAX_CONNECTIONS=16

# Aggressive (more connections per worker)
DATABASE_MAX_CONNECTIONS=48
```

**Guidelines**:
- More workers need more connections
- PostgreSQL `max_connections` must be higher than `DATABASE_MAX_CONNECTIONS`
- Monitor with: `SELECT count(*) FROM pg_stat_activity;`

## Database Tuning

### 1. PostgreSQL Memory Configuration

Edit `postgresql.conf`:

```ini
# Memory Settings (for 16GB RAM system)
shared_buffers = 4GB                    # 25% of RAM
effective_cache_size = 12GB             # 75% of RAM
work_mem = 32MB                         # Per-operation memory
maintenance_work_mem = 1GB              # For VACUUM, CREATE INDEX
```

**Calculation Guidelines**:
- `shared_buffers`: 25% of total RAM (max 8GB on Windows)
- `effective_cache_size`: 75% of total RAM
- `work_mem`: (Total RAM - shared_buffers) / (max_connections × 2-4)
- `maintenance_work_mem`: 5-10% of RAM

### 2. PostgreSQL Performance Settings

```ini
# Parallelism
max_worker_processes = 8                # Number of CPU cores
max_parallel_workers_per_gather = 4
max_parallel_workers = 8

# Query Planning
random_page_cost = 1.1                  # SSD optimization (default 4.0)
effective_io_concurrency = 200          # SSD: 200, HDD: 2

# Checkpoints
checkpoint_completion_target = 0.9      # Spread out checkpoint I/O
wal_buffers = 16MB
max_wal_size = 4GB                      # Allow larger WAL before checkpoint
min_wal_size = 1GB

# Logging (for tuning only, disable in production)
log_min_duration_statement = 1000       # Log queries > 1 second
log_line_prefix = '%t [%p]: [%l-1] user=%u,db=%d,app=%a,client=%h '
log_checkpoints = on
log_connections = on
log_disconnections = on
```

### 3. Index Optimization

**Verify All Indexes Exist**:

```sql
SELECT
    schemaname,
    tablename,
    indexname
FROM pg_indexes
WHERE schemaname IN ('staging', 'claims', 'ml')
ORDER BY tablename, indexname;
```

Should show 50+ indexes.

**Check for Missing Indexes**:

```sql
-- Find sequential scans that might benefit from indexes
SELECT
    schemaname,
    tablename,
    seq_scan,
    seq_tup_read,
    idx_scan,
    seq_tup_read / seq_scan AS avg_seq_tup
FROM pg_stat_user_tables
WHERE schemaname IN ('staging', 'claims', 'ml')
AND seq_scan > 0
ORDER BY seq_tup_read DESC
LIMIT 10;
```

**Add Missing Indexes**:

```sql
-- Example: If many lookups by patient_control_number
CREATE INDEX IF NOT EXISTS idx_encounter_pcn_lookup
ON claims.encounter (patient_control_number)
WHERE is_active = true;

-- Example: If many lookups by service date
CREATE INDEX IF NOT EXISTS idx_service_line_date_range
ON claims.service_line (service_date_from, service_date_to)
WHERE is_active = true;
```

### 4. Query Optimization

**Identify Slow Queries**:

```sql
-- Enable pg_stat_statements
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- Find slowest queries
SELECT
    substring(query, 1, 100) AS short_query,
    calls,
    total_exec_time,
    mean_exec_time,
    max_exec_time
FROM pg_stat_statements
WHERE query NOT LIKE '%pg_stat_statements%'
ORDER BY mean_exec_time DESC
LIMIT 10;
```

**Analyze Query Plans**:

```sql
EXPLAIN (ANALYZE, BUFFERS, VERBOSE)
SELECT * FROM claims.encounter WHERE patient_control_number = 'PCN123';
```

Look for:
- Sequential Scans (should use indexes)
- High cost estimates
- Large buffer usage

### 5. Maintenance Operations

**Regular VACUUM**:

```sql
-- Analyze frequency of dead tuples
SELECT
    schemaname,
    tablename,
    n_dead_tup,
    n_live_tup,
    n_dead_tup * 100 / NULLIF(n_live_tup + n_dead_tup, 0) AS dead_pct
FROM pg_stat_user_tables
WHERE schemaname IN ('staging', 'claims', 'ml')
ORDER BY n_dead_tup DESC;
```

Run VACUUM if dead_pct > 10%:

```sql
VACUUM ANALYZE claims.encounter;
VACUUM ANALYZE claims.service_line;
VACUUM ANALYZE claims.encounter_flag;
VACUUM ANALYZE claims.service_line_flag;
```

**Automated Maintenance**:

```ini
# In postgresql.conf
autovacuum = on
autovacuum_max_workers = 3
autovacuum_naptime = 1min
```

## System-Level Tuning

### 1. Windows Performance Settings

**Disable Unnecessary Services**:
- Windows Search (if not needed)
- Windows Update (schedule for off-hours)
- Antivirus (exclude database and application directories)

**Adjust Virtual Memory**:
1. System Properties → Advanced → Performance Settings
2. Advanced tab → Virtual Memory
3. Set custom size:
   - Initial: 1.5x RAM
   - Maximum: 3x RAM

### 2. Disk I/O Optimization

**Use SSD for Database**:
- Move PostgreSQL data directory to SSD
- Significant performance improvement (5-10x)

**Check Disk Performance**:
```cmd
winsat disk -drive c
```

**Optimize Disk**:
```cmd
# Disable disk indexing on data drives
fsutil behavior set disablelastaccess 1

# Defragment (HDD only, not needed for SSD)
defrag C: /O

# TRIM for SSD
fsutil behavior set disabledeletenotify 0
```

### 3. Network Optimization (if using remote database)

```cmd
# Disable Nagle's algorithm for lower latency
netsh int tcp set global autotuninglevel=normal
netsh int tcp set global chimney=enabled
```

## Monitoring Performance

### 1. Application Monitoring

**Log Performance Metrics**:

```env
RUST_LOG=pro_worker=info
```

Look for log entries showing processing times:
```
INFO pro_worker: Processed 1000 claims in 1.2 seconds (833 claims/sec)
```

### 2. Database Monitoring

**Active Connections**:

```sql
SELECT
    count(*),
    state
FROM pg_stat_activity
WHERE datname = 'professional_smart'
GROUP BY state;
```

**Lock Contention**:

```sql
SELECT
    locktype,
    relation::regclass,
    mode,
    granted,
    count(*)
FROM pg_locks
WHERE database = (SELECT oid FROM pg_database WHERE datname = 'professional_smart')
GROUP BY locktype, relation, mode, granted;
```

**Cache Hit Ratio** (should be > 99%):

```sql
SELECT
    sum(heap_blks_read) as heap_read,
    sum(heap_blks_hit) as heap_hit,
    sum(heap_blks_hit) * 100 / (sum(heap_blks_hit) + sum(heap_blks_read)) as cache_hit_ratio
FROM pg_statio_user_tables
WHERE schemaname IN ('staging', 'claims', 'ml');
```

### 3. System Monitoring

**CPU Usage**:
```cmd
wmic cpu get loadpercentage
```

**Memory Usage**:
```cmd
wmic OS get FreePhysicalMemory,TotalVisibleMemorySize /value
```

**Disk I/O**:
```cmd
typeperf "\PhysicalDisk(_Total)\Disk Reads/sec" "\PhysicalDisk(_Total)\Disk Writes/sec" -sc 10
```

## Performance Optimization Checklist

### Quick Wins (Immediate Impact)

- [ ] Set `BATCH_SIZE=2000`
- [ ] Set `MAX_WORKERS` to CPU core count
- [ ] Set `DATABASE_MAX_CONNECTIONS` to `MAX_WORKERS * 4`
- [ ] Run `VACUUM ANALYZE` on all tables
- [ ] Verify all indexes exist
- [ ] Set PostgreSQL `shared_buffers = 4GB`
- [ ] Set PostgreSQL `effective_cache_size = 12GB`
- [ ] Set `random_page_cost = 1.1` for SSD

### Medium Impact (Requires Testing)

- [ ] Tune `work_mem` for your workload
- [ ] Enable parallel workers in PostgreSQL
- [ ] Increase `max_wal_size` to reduce checkpoints
- [ ] Adjust `checkpoint_completion_target = 0.9`
- [ ] Partition large tables (if > 10M rows)
- [ ] Add application-specific indexes based on query patterns

### Long-term Optimization (System Upgrade)

- [ ] Upgrade to SSD storage
- [ ] Increase RAM to 32GB+
- [ ] Upgrade to faster CPU
- [ ] Use dedicated database server
- [ ] Implement connection pooler (PgBouncer)
- [ ] Consider read replicas for reporting

## Performance Testing Scenarios

### Test 1: Single File Processing

**Objective**: Validate parsing and processing speed

```cmd
# Process a file with 1000 claims
# Expected: < 1.5 seconds (666+ claims/sec)
```

### Test 2: Batch Processing

**Objective**: Validate sustained throughput

```cmd
# Process 10 files with 1000 claims each
# Expected: Consistent 666+ claims/sec across all files
```

### Test 3: Large File Processing

**Objective**: Validate memory management

```cmd
# Process a file with 10,000 claims
# Expected: < 15 seconds, < 2GB memory
```

### Test 4: Concurrent Processing

**Objective**: Validate parallel processing

```cmd
# Process multiple files simultaneously
# Expected: Linear scaling with worker count
```

## Troubleshooting Performance Issues

### Symptom: Low Throughput (< 500 claims/sec)

**Diagnosis**:
1. Run benchmarks: `cargo bench`
2. Check CPU usage: Should be 70-90%
3. Check database: Run slow query analysis

**Solutions**:
1. Increase `BATCH_SIZE`
2. Increase `MAX_WORKERS`
3. Run `VACUUM ANALYZE`
4. Check for missing indexes

### Symptom: High Memory Usage

**Diagnosis**:
1. Monitor memory during processing
2. Check batch size

**Solutions**:
1. Decrease `BATCH_SIZE`
2. Decrease `MAX_WORKERS`
3. Increase system RAM

### Symptom: Database Bottleneck

**Diagnosis**:
1. Check active connections
2. Check lock contention
3. Check slow queries

**Solutions**:
1. Increase `DATABASE_MAX_CONNECTIONS`
2. Optimize slow queries
3. Add missing indexes
4. Increase PostgreSQL `shared_buffers`

## Performance Benchmarking Results

### v2.12.46.0 Benchmark (December 29, 2025) - CURRENT

```
System Configuration:
- OS: Windows 10/11 (64-bit)
- Database: PostgreSQL 17
- Workers: 8 concurrent

Configuration:
- STAGE2_WORKER_COUNT: 8
- BATCH_SIZE: 750
- DB_MAX_CONNECTIONS: 100

Test Data:
- 9,971 claims processed
- ~3 service lines per claim average
- EDI 837P professional claims

Performance Results:
- Throughput: 1,284 claims/sec (192.8% of target)
- Processing Time: 7.76 seconds
- Success Rate: 99.7%+
- SRD Target: ✅ ACHIEVED (10k claims in <15s)

Key Optimizations:
- Provider NPI → provider_id cache (558% improvement)
- sync_encounter_totals triggers removed
- Batch INSERT for diagnoses and diagnosis pointers

Status: ✅ Significantly exceeds target (666 claims/sec)
```

### v2.12.21.0 Benchmark (Historical Reference)

```
Configuration:
- BATCH_SIZE: 100
- MAX_CONCURRENT_BATCHES: 4

Performance Results:
- Throughput: 822.5 claims/sec (123.5% of target)
- Processing Time: 36.02 seconds
- Success Rate: 98.7% (9,871 completed)

Status: ✅ Exceeded target (superseded by v2.12.46.0)
```

### Key Learnings

1. **Cache Shared Resources** - Provider NPIs repeat across many claims. Caching provider_id after first DB lookup eliminated ~150,000 redundant queries for 10k claims. This was the single biggest optimization (558% improvement).

2. **Remove Redundant Triggers** - Database triggers that recalculate values already computed in application code add unnecessary overhead. The sync_encounter_totals triggers were firing 3x per encounter for values already calculated in Rust.

3. **Avoid Advisory Locks on Shared Resources** - Provider NPIs are shared across many claims. Advisory locks caused 96% failure rate when 8 workers competed for the same provider.

4. **Use INSERT ON CONFLICT for Concurrent Inserts** - `INSERT ON CONFLICT DO UPDATE` is safe for concurrent access without locks.

5. **Keep Autovacuum Enabled** - Disabling autovacuum caused 717k dead tuples and 71x table bloat.

6. **Reduce work_mem for High Concurrency** - 512MB work_mem with 300 connections could exhaust 150GB+ RAM.

## Next Steps

- [Troubleshooting Guide](TROUBLESHOOTING.md) - If performance issues persist
- [Configuration Guide](CONFIGURATION.md) - Detailed configuration options
- [Database Setup Guide](DATABASE_SETUP.md) - Database optimization
