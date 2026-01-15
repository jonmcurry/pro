# Encounter Flag Not Inserting - Analysis Plan

## Status: COMPLETE - FIXED IN v2.12.73.5

## Problem
Claims are being processed but no records are being inserted into `claims.encounter_flag` table on production server.

## Root Causes Found

### Issue 1: Environment Variables Not Set
Two environment variables are required:

1. `ENABLE_DATABASE_RULES=true` - Required to load rules from database
2. `RULE_ENCRYPTION_KEY=<your-key>` - Required to decrypt rule parameters

Without `ENABLE_DATABASE_RULES=true`, the system falls back to only 6 legacy hard-coded rules.

### Issue 2: Wrong Table Name in Code (FIXED in v2.12.73.5)
The rule engine was trying to insert into `claims.flag` which doesn't exist.
The actual tables are:
- `claims.encounter_flag` - for encounter-level flags
- `claims.service_line_flag` - for service line-level flags

## Solution

### 1. Set Environment Variables

**In .env file:**
```
ENABLE_DATABASE_RULES=true
RULE_ENCRYPTION_KEY=your-secret-key
```

**In PostgreSQL:**
```sql
ALTER DATABASE your_database SET app.rule_encryption_key = 'your-secret-key';
```

### 2. Deploy v2.12.73.5

The code fix in `rule_engine.rs` now:
- Routes to `claims.encounter_flag` when `encounter_id` is present
- Routes to `claims.service_line_flag` when `service_line_id` is present
- Looks up `issue_id` from `claims.flag_issue` using `issue_code`

## Verification

After deploying, check:
```sql
SELECT COUNT(*) FROM claims.encounter_flag;
SELECT COUNT(*) FROM claims.service_line_flag;
```

And check logs for:
```
INFO Loading rules from database (facility_id: ...)
INFO Loaded 556 rule(s) from database
```

## Code References
- rule_engine.rs:763-925 - Fixed create_flag and create_flag_with_tx functions
- pipeline.rs:42-66 - ENABLE_DATABASE_RULES check
- loader.rs:49-51 - RULE_ENCRYPTION_KEY requirement

## Version
2.12.73.5
