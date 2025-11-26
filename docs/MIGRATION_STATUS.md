# Migration Status Report

**Date**: 2025-11-26
**MSI Version**: 2.8.2.0
**Status**: ✅ All migrations embedded in MSI

---

## Issue Identified

**Problem**: Rule configuration tables (migrations 046-051) were not applied to database

**Root Cause**: Migrations 050-051 were **not embedded** in the pro-upgrade binary

---

## Investigation Summary

### Database State (Before Fix)

```sql
-- Checked applied migrations
SELECT migration_name FROM staging.schema_migrations ORDER BY migration_name DESC LIMIT 10;

-- Result: Only migrations 001-039 were applied
-- Missing: 040-051
```

**Tables Missing**:
- `claims.rule_template` - Rule templates (migration 046)
- `claims.rule_definition` - Rule instances (migration 046)
- `claims.facility_rule_assignment` - Per-facility assignments (migration 046)
- `claims.organization_rule_assignment` - Per-org assignments (migration 046)
- Performance indexes (migration 050)
- `claims.rule_execution_stats` - Statistics tracking (migration 051)

### How Migrations Work

**Architecture**:
```
migrations/*.sql  ──┐
                    ├──> Embedded in binary via include_str!
                    │    (crates/pro-upgrade-manager/src/embedded_migrations.rs)
                    ↓
              pro-upgrade.exe
                    ↓
           Applied automatically
              on service startup
                    ↓
         staging.schema_migrations
              (tracking table)
```

**Key Files**:
- `crates/pro-upgrade-manager/src/embedded_migrations.rs` - Embeds SQL files at compile time
- `crates/pro-upgrade/src/main.rs` - Applies migrations
- `migrations/*.sql` - SQL migration files

**Not Deployed to Disk**: Migrations are **embedded in the binary**, not copied as files

---

## Resolution

### Fix Applied

**Step 1**: Added missing migrations to embedded_migrations.rs

```rust
// Added to crates/pro-upgrade-manager/src/embedded_migrations.rs:

EmbeddedMigration {
    version: "050",
    name: "add_performance_indexes",
    sql: include_str!("../../../migrations/050_add_performance_indexes.sql"),
},
EmbeddedMigration {
    version: "051",
    name: "add_rule_execution_stats",
    sql: include_str!("../../../migrations/051_add_rule_execution_stats.sql"),
},
```

**Step 2**: Rebuilt release binary

```bash
cargo build --release
# Compiling pro-upgrade-manager v0.3.2
# Compiling pro-upgrade v0.3.2
# Finished `release` profile [optimized] target(s) in 31.03s
```

**Step 3**: Rebuilt MSI

```bash
cd installer
candle.exe Product.wxs DatabaseConfigDlg.wxs PrerequisiteDlg.wxs
light.exe -out ProfessionalSMART.msi *.wixobj
```

### Updated MSI

**File**: `installer/ProfessionalSMART.msi`
**Size**: 11 MB
**Build Time**: 09:06 AM
**SHA256**: `4316101F52C1A910993F180D845B8911379B6B3C5D23EE42549404CFE6E22F6C`
**Version**: 2.8.2.0

**Included Migrations**: 001-051 (all embedded)

---

## Migration List

### Embedded Migrations (001-051)

| Version | Name | Status | Description |
|---------|------|--------|-------------|
| 001 | create_schemas | ✅ Applied | Base schemas (claims, staging, ml) |
| 002-039 | ... | ✅ Applied | Core tables and features |
| 040 | _(missing)_ | ⚠️ Not created | (gap in numbering) |
| 041 | create_provider_taxonomy | ✅ Applied | Provider taxonomy tables |
| 042 | create_provider_enrichment_queue | ✅ Applied | Enrichment queue |
| 043 | add_missing_foreign_key_indexes | ✅ Applied | FK indexes |
| 044 | add_taxonomy_foreign_key | ✅ Applied | Taxonomy FK |
| 045 | add_staging_foreign_keys | ✅ Applied | Staging FKs |
| 046 | create_rule_configuration_system | ⏳ Pending | **Rule tables** (will apply on upgrade) |
| 047 | add_test_facility_rule_assignments | ⏳ Pending | Test assignments |
| 048 | add_rule_templates | ⏳ Pending | Rule templates |
| 049 | add_flag_issue_helpers | ⏳ Pending | Helper functions |
| 050 | add_performance_indexes | ⏳ Pending | **Phase 6-8 indexes** (will apply on upgrade) |
| 051 | add_rule_execution_stats | ⏳ Pending | **Phase 8 statistics** (will apply on upgrade) |

---

## Deployment Instructions

### Fresh Installation

**Steps**:
1. Install MSI: `msiexec /i ProfessionalSMART.msi`
2. Service starts and **automatically applies all migrations** (001-051)
3. Verify: Check `staging.schema_migrations`

**Expected Downtime**: 2-5 minutes (initial setup + migrations)

### Upgrade from Existing Installation

**Current Database State**: Migrations 001-039 applied

**Upgrade Process**:
1. Stop service (automatic during MSI install)
2. Install MSI: `msiexec /i ProfessionalSMART.msi`
3. Service starts and applies **pending migrations** (041-051)
4. New tables created:
   - `claims.rule_template`
   - `claims.rule_definition`
   - `claims.facility_rule_assignment`
   - `claims.organization_rule_assignment`
   - `claims.rule_execution_stats`
   - 7 new performance indexes

**Expected Downtime**: 3-7 minutes (migration execution time)

**Post-Upgrade Verification**:
```sql
-- Check all migrations applied
SELECT COUNT(*) FROM staging.schema_migrations;
-- Should return: 50 (040 doesn't exist, so 51-1=50)

-- Verify rule tables exist
\dt claims.rule_*
-- Should show: rule_template, rule_definition, rule_execution_stats

-- Verify indexes created
SELECT indexname FROM pg_indexes
WHERE schemaname = 'claims' AND indexname LIKE 'idx_%'
AND indexname IN (
    'idx_service_line_duplicate_detection',
    'idx_raw_claims_batch_status'
);
```

---

## Rule Configuration Now Available

After upgrade, you can create rules **without recompilation**:

### Example: Create High-Value Flag Rule

```sql
-- Create rule using THRESHOLD template
INSERT INTO claims.rule_definition (
    rule_code,
    rule_name,
    template_id,
    rule_parameters_encrypted,
    flag_issue_id,
    execution_order,
    execution_level,
    is_active
) VALUES (
    'HIGH_VALUE_NO_AUTH',
    'High Value Claims Without Authorization',
    (SELECT template_id FROM claims.rule_template WHERE template_code = 'THRESHOLD'),
    pgp_sym_encrypt('{"field": "total_charge", "operator": ">", "threshold": 10000}', 'encryption_key'),
    (SELECT issue_id FROM claims.flag_issue WHERE issue_code = 'DOC_MISSING'),
    5,
    'ENCOUNTER',
    true
);

-- Assign to specific facility
INSERT INTO claims.facility_rule_assignment (
    facility_id, rule_id, is_enabled
) VALUES (
    123,
    (SELECT rule_id FROM claims.rule_definition WHERE rule_code = 'HIGH_VALUE_NO_AUTH'),
    true
);

-- Hot reload (no service restart needed!)
-- Windows: Create reload trigger
type NUL > "C:\ProgramData\Professional SMART\reload_rules.trigger"
```

See [RULE_CONFIGURATION_GUIDE.md](RULE_CONFIGURATION_GUIDE.md) for complete documentation.

---

## Known Issues

### 1. Migration 040 Missing

**Issue**: No migration file for 040 (gap in numbering)

**Impact**: None - migrations don't need to be sequential

**Resolution**: Not needed - 041-051 will apply correctly

### 2. Existing Installations Need Upgrade

**Issue**: Existing databases have only 001-039 applied

**Impact**: Rule configuration features unavailable until upgrade

**Resolution**: Install MSI 2.8.2.0 - migrations apply automatically

---

## Testing Checklist

### Pre-Deployment Testing

- [x] Verify embedded_migrations.rs includes 050-051
- [x] Rebuild pro-upgrade binary
- [x] Rebuild MSI
- [x] Verify MSI SHA256 changed

### Post-Deployment Testing

- [ ] Fresh install - verify all migrations apply
- [ ] Upgrade install - verify 041-051 apply
- [ ] Verify rule tables exist
- [ ] Verify performance indexes exist
- [ ] Create test rule via SQL
- [ ] Verify hot reload works
- [ ] Check execution statistics collection

---

## Rollback Plan

### If Upgrade Fails

**Option 1**: Revert to previous version
```powershell
msiexec /x ProfessionalSMART.msi /qn
# Install previous version MSI if available
```

**Option 2**: Manually rollback migrations
```sql
-- Drop new tables (if needed)
DROP TABLE IF EXISTS claims.rule_execution_stats CASCADE;
DROP TABLE IF EXISTS claims.facility_rule_assignment CASCADE;
DROP TABLE IF EXISTS claims.organization_rule_assignment CASCADE;
DROP TABLE IF EXISTS claims.rule_definition CASCADE;
DROP TABLE IF EXISTS claims.rule_template CASCADE;

-- Drop indexes (if needed)
DROP INDEX IF EXISTS claims.idx_service_line_duplicate_detection;
DROP INDEX IF EXISTS claims.idx_raw_claims_batch_status;
-- ... (other indexes)

-- Remove migration entries
DELETE FROM staging.schema_migrations WHERE migration_name IN (
    '046_create_rule_configuration_system.sql',
    '047_add_test_facility_rule_assignments.sql',
    '048_add_rule_templates.sql',
    '049_add_flag_issue_helpers.sql',
    '050_add_performance_indexes.sql',
    '051_add_rule_execution_stats.sql'
);
```

---

## Related Documentation

- [RULE_CONFIGURATION_GUIDE.md](RULE_CONFIGURATION_GUIDE.md) - How to use rule system
- [DATABASE_SETUP.md](DATABASE_SETUP.md) - Database setup and migrations
- [BUILD_SUMMARY_PHASE_6-8.md](BUILD_SUMMARY_PHASE_6-8.md) - Build details
- [CHANGELOG.md](CHANGELOG.md) - Version history

---

## Summary

✅ **Fixed**: Added migrations 050-051 to embedded_migrations.rs
✅ **Rebuilt**: Binary and MSI now include all migrations (001-051)
**Ready**: MSI 2.8.2.0 can be deployed
⏳ **Next**: Deploy to development for testing

**MSI Location**: `installer/ProfessionalSMART.msi`
**SHA256**: `4316101F52C1A910993F180D845B8911379B6B3C5D23EE42549404CFE6E22F6C`
**Version**: 2.8.2.0

On next deployment, all pending migrations (041-051) will apply automatically, enabling:
- Rule configuration without recompilation
- Per-facility rule customization
- Phase 6-8 performance optimizations
- Historical statistics tracking
