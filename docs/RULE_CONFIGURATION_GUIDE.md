# Rule Configuration Guide

Complete guide to rules storage, facility assignment, and creating ad-hoc flags without recompilation.

**Last Updated**: 2025-11-26
**Version**: 2.8.2.0

---

## Table of Contents

1. [Overview](#overview)
2. [Where Are Rules Stored?](#where-are-rules-stored)
3. [How Are Rules Assigned to Facilities?](#how-are-rules-assigned-to-facilities)
4. [Creating Ad-Hoc Flags Without Recompilation](#creating-ad-hoc-flags-without-recompilation)
5. [Rule Templates](#rule-templates)
6. [Examples](#examples)
7. [API Reference](#api-reference)

---

## Overview

Professional SMART uses a **data-driven rule configuration system** that allows:

- ✅ **No recompilation required** for most rule changes
- ✅ **Per-facility rule customization** with parameter overrides
- ✅ **Per-organization defaults** that apply to all facilities
- ✅ **Encrypted rule parameters** for security compliance
- ✅ **Template-based rules** for rapid deployment
- ✅ **Hot reload** support for zero-downtime updates

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATABASE                                  │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────┐ │
│  │  rule_template   │  │ rule_definition  │  │  flag_issue   │ │
│  │  (Rust structs)  │  │  (Instances)     │  │  (Outcomes)   │ │
│  └──────────────────┘  └──────────────────┘  └───────────────┘ │
│           ↓                      ↓                       ↓       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │         facility_rule_assignment                          │  │
│  │         organization_rule_assignment                      │  │
│  │         (Per-facility/org enable/disable + overrides)     │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                  ↓
                    ┌─────────────────────────┐
                    │   Rust Rule Loader      │
                    │   (loader.rs)           │
                    └─────────────────────────┘
                                  ↓
                    ┌─────────────────────────┐
                    │   RuleEngine            │
                    │   (executes rules)      │
                    └─────────────────────────┘
```

---

## Where Are Rules Stored?

### Database Tables

Rules are stored in **PostgreSQL** across several tables:

#### 1. `claims.rule_template` - Pre-compiled Rule Templates

**Purpose**: Define reusable rule types (Rust implementations)

**Schema**:
```sql
CREATE TABLE claims.rule_template (
    template_id BIGINT PRIMARY KEY,
    template_code VARCHAR(50) UNIQUE,  -- 'THRESHOLD', 'DUPLICATE', etc.
    template_name VARCHAR(100),
    rust_struct_name VARCHAR(100),     -- Maps to Rust code
    parameter_schema JSONB,            -- Defines allowed parameters
    execution_level VARCHAR(20),       -- 'ENCOUNTER', 'SERVICE_LINE', 'BOTH'
    is_active BOOLEAN DEFAULT true
);
```

**Built-in Templates**:
- `LEGACY` - Existing hard-coded rules
- `THRESHOLD` - Compare field against threshold (e.g., charge > $10,000)
- `DUPLICATE` - Detect duplicate records
- `MISSING_FIELD` - Flag missing required fields
- `FIELD_PATTERN` - Validate field with regex
- `CROSS_FIELD` - Compare two fields

**Location**: [migrations/046_create_rule_configuration_system.sql](../migrations/046_create_rule_configuration_system.sql)

#### 2. `claims.rule_definition` - Rule Instances

**Purpose**: Specific rule configurations (instances of templates)

**Schema**:
```sql
CREATE TABLE claims.rule_definition (
    rule_id BIGINT PRIMARY KEY,
    rule_code VARCHAR(50) UNIQUE,          -- 'HIGH_VALUE_NO_AUTH'
    rule_name VARCHAR(200),
    template_id BIGINT,                    -- References rule_template

    -- Encrypted parameters (JSON)
    rule_parameters_encrypted BYTEA,       -- pgp_sym_encrypt(...)

    -- Execution control
    execution_order INT DEFAULT 100,       -- Lower = earlier
    execution_level VARCHAR(20),           -- Where to run
    timeout_ms INT DEFAULT 5000,

    -- Flag configuration
    flag_issue_id BIGINT,                  -- What flag to create
    default_severity VARCHAR(20),

    is_active BOOLEAN DEFAULT true
);
```

**Example Rules**:
```sql
-- Built-in rules (from migration 046)
SELECT rule_code, rule_name, execution_order FROM claims.rule_definition;

 rule_code                    | rule_name                          | execution_order
------------------------------+-----------------------------------+----------------
 DUPLICATE_SERVICE            | Duplicate Service Detection        | 10
 UNITS_EXCEED_MAX             | Units Exceed Maximum               | 20
 MISSING_REQUIRED_MODIFIER    | Missing Required Modifier          | 30
 CONFLICTING_MODIFIERS        | Conflicting Modifiers              | 40
 UNSPECIFIED_DIAGNOSIS        | Unspecified Diagnosis Code         | 50
 MISSING_DIAGNOSIS_SPECIFICITY| Missing Diagnosis Specificity      | 60
```

#### 3. `claims.flag_issue` - Flag Outcomes

**Purpose**: Define what flags can be created

**Schema**:
```sql
CREATE TABLE claims.flag_issue (
    issue_id BIGINT PRIMARY KEY,
    category_id BIGINT,                    -- References flag_category
    issue_code VARCHAR(20) UNIQUE,         -- 'COD_INCORRECT', 'MOD_MISSING'
    issue_description TEXT,
    severity VARCHAR(20) DEFAULT 'MEDIUM'
);
```

**Example Issues**:
```sql
SELECT issue_code, issue_description, severity FROM claims.flag_issue LIMIT 5;

 issue_code     | issue_description                          | severity
---------------+-------------------------------------------+----------
 COD_BUNDLED    | Bundled Service/Procedure                  | HIGH
 COD_INCORRECT  | Incorrect Procedure Code                   | HIGH
 DOC_MISSING    | Missing Documentation                      | HIGH
 MOD_MISSING    | Missing Modifier                           | MEDIUM
 QTY_FEWER      | Fewer Units Supported                      | HIGH
```

**Location**: [migrations/006_create_flag_tables.sql](../migrations/006_create_flag_tables.sql)

---

## How Are Rules Assigned to Facilities?

### Three-Level Assignment Hierarchy

Rules can be activated at three levels (most specific wins):

```
┌─────────────────────────────────────────────────────┐
│  1. GLOBAL (Default)                                 │
│     - All facilities inherit if no overrides         │
│     - Set via rule_definition.is_active = true       │
└─────────────────────────────────────────────────────┘
                         ↓ (overridden by)
┌─────────────────────────────────────────────────────┐
│  2. ORGANIZATION (All facilities in org)             │
│     - Applies to all facilities in organization      │
│     - Set via organization_rule_assignment           │
└─────────────────────────────────────────────────────┘
                         ↓ (overridden by)
┌─────────────────────────────────────────────────────┐
│  3. FACILITY (Specific facility)                     │
│     - Highest priority, overrides org and global     │
│     - Set via facility_rule_assignment               │
└─────────────────────────────────────────────────────┘
```

### Assignment Tables

#### 1. Organization-Level Assignment

```sql
CREATE TABLE claims.organization_rule_assignment (
    assignment_id BIGINT PRIMARY KEY,
    organization_id BIGINT,
    rule_id BIGINT,
    is_enabled BOOLEAN DEFAULT true,
    parameter_overrides_encrypted BYTEA,  -- Optional overrides
    effective_from DATE,
    effective_to DATE
);
```

**Example**: Enable "High Value Threshold" for all facilities in Org #1:
```sql
INSERT INTO claims.organization_rule_assignment (
    organization_id, rule_id, is_enabled, effective_from
) VALUES (
    1,  -- Organization #1
    (SELECT rule_id FROM claims.rule_definition WHERE rule_code = 'HIGH_VALUE_THRESHOLD'),
    true,
    CURRENT_DATE
);
```

#### 2. Facility-Level Assignment

```sql
CREATE TABLE claims.facility_rule_assignment (
    assignment_id BIGINT PRIMARY KEY,
    facility_id BIGINT,
    rule_id BIGINT,
    is_enabled BOOLEAN DEFAULT true,
    parameter_overrides_encrypted BYTEA,  -- Optional overrides
    effective_from DATE,
    effective_to DATE,
    UNIQUE (facility_id, rule_id)
);
```

**Example**: Override threshold for specific facility:
```sql
-- Facility #123 wants higher threshold ($15,000 instead of default $10,000)
INSERT INTO claims.facility_rule_assignment (
    facility_id, rule_id, is_enabled, parameter_overrides_encrypted, effective_from
) VALUES (
    123,
    (SELECT rule_id FROM claims.rule_definition WHERE rule_code = 'HIGH_VALUE_THRESHOLD'),
    true,
    pgp_sym_encrypt('{"threshold": 15000}', 'encryption_key'),
    CURRENT_DATE
);
```

### Viewing Active Rules

**View**: `claims.v_active_facility_rules`

```sql
-- See all active rules for facility #123
SELECT
    rule_code,
    rule_name,
    execution_order,
    assignment_level,  -- 'GLOBAL', 'ORGANIZATION', or 'FACILITY'
    is_enabled
FROM claims.v_active_facility_rules
WHERE facility_id = 123
ORDER BY execution_order;
```

**Function**: `claims.get_active_rules_for_facility()`

```sql
-- Get active rules programmatically
SELECT * FROM claims.get_active_rules_for_facility(
    123,  -- facility_id
    NULL  -- execution_level (NULL = all levels)
);
```

---

## Creating Ad-Hoc Flags Without Recompilation

### Method 1: Use Built-in Templates (Recommended)

**No Rust code needed** - Just insert database records!

#### Example 1: Flag High-Value Claims Without Authorization

**Step 1**: Create the rule definition
```sql
-- Get the THRESHOLD template
SELECT template_id FROM claims.rule_template WHERE template_code = 'THRESHOLD';
-- Returns: 2

-- Get the flag issue for unauthorized services
SELECT issue_id FROM claims.flag_issue WHERE issue_code = 'DOC_MISSING';
-- Returns: 5

-- Create new rule
INSERT INTO claims.rule_definition (
    rule_code,
    rule_name,
    rule_description,
    template_id,
    rule_parameters_encrypted,
    flag_issue_id,
    execution_order,
    execution_level,
    default_severity,
    is_active
) VALUES (
    'HIGH_VALUE_NO_AUTH',
    'High Value Claims Without Authorization',
    'Flags claims over $10,000 without prior authorization',
    2,  -- THRESHOLD template
    pgp_sym_encrypt('{"field": "total_charge", "operator": ">", "threshold": 10000}', 'your_encryption_key'),
    5,  -- DOC_MISSING issue
    5,  -- Early execution order
    'ENCOUNTER',
    'HIGH',
    true
);
```

**Step 2**: Assign to facility (optional - global by default)
```sql
-- Enable for specific facility with different threshold
INSERT INTO claims.facility_rule_assignment (
    facility_id,
    rule_id,
    is_enabled,
    parameter_overrides_encrypted
) VALUES (
    123,
    (SELECT rule_id FROM claims.rule_definition WHERE rule_code = 'HIGH_VALUE_NO_AUTH'),
    true,
    pgp_sym_encrypt('{"threshold": 15000}', 'your_encryption_key')  -- $15K for this facility
);
```

**Step 3**: Reload rules (hot reload)
```bash
# On Windows:
type NUL > C:\ProgramData\Professional SMART\reload_rules.trigger

# Service automatically detects and reloads
# OR restart service:
Restart-Service ProfessionalSMART
```

**Done!** The rule is now active. No Rust recompilation required.

#### Example 2: Flag Duplicate Services Within 7 Days

```sql
INSERT INTO claims.rule_definition (
    rule_code, rule_name, template_id,
    rule_parameters_encrypted,
    flag_issue_id, execution_order, execution_level, is_active
) VALUES (
    'DUPLICATE_7DAY',
    'Duplicate Service Within 7 Days',
    (SELECT template_id FROM claims.rule_template WHERE template_code = 'DUPLICATE'),
    pgp_sym_encrypt('{"match_fields": ["procedure_code", "provider_id"], "timeframe_days": 7}', 'key'),
    (SELECT issue_id FROM claims.flag_issue WHERE issue_code = 'OTH_DUPLICATE'),
    15,
    'SERVICE_LINE',
    true
);
```

#### Example 3: Flag Missing NPI Numbers

```sql
INSERT INTO claims.rule_definition (
    rule_code, rule_name, template_id,
    rule_parameters_encrypted,
    flag_issue_id, execution_order, execution_level, is_active
) VALUES (
    'MISSING_NPI',
    'Missing Provider NPI',
    (SELECT template_id FROM claims.rule_template WHERE template_code = 'MISSING_FIELD'),
    pgp_sym_encrypt('{"field": "provider_npi"}', 'key'),
    (SELECT issue_id FROM claims.flag_issue WHERE issue_code = 'OTH_PROVIDER'),
    8,
    'ENCOUNTER',
    true
);
```

### Method 2: Add New Template (Requires Rust Code)

If built-in templates don't cover your use case, create a new template:

**Step 1**: Implement Rust struct in `crates/pro-rules/src/templates/`

**Step 2**: Add to template registry

**Step 3**: Deploy new version

**Note**: This is only needed for complex logic not supported by existing templates.

---

## Rule Templates

### Available Templates

#### 1. THRESHOLD - Numeric Comparisons

**Use Case**: Flag when field exceeds/below threshold

**Parameters**:
```json
{
  "field": "total_charge" | "line_charge" | "units",
  "operator": ">" | "<" | ">=" | "<=" | "=" | "!=",
  "threshold": 10000
}
```

**Examples**:
- Charges over $10,000
- Units exceeding 100
- Line charges below $5

#### 2. DUPLICATE - Duplicate Detection

**Use Case**: Find duplicate records within timeframe

**Parameters**:
```json
{
  "match_fields": ["procedure_code", "provider_id", "date_of_service"],
  "timeframe_days": 30
}
```

**Examples**:
- Same procedure by same provider within 7 days
- Duplicate claims within 90 days

#### 3. MISSING_FIELD - Required Field Validation

**Use Case**: Flag when required field is null/empty

**Parameters**:
```json
{
  "field": "provider_npi",
  "when_condition": {"procedure_code": "99213"}  // Optional
}
```

**Examples**:
- Missing NPI
- Missing diagnosis codes
- Conditional: Missing modifier when procedure requires it

#### 4. FIELD_PATTERN - Regex Validation

**Use Case**: Validate field format/pattern

**Parameters**:
```json
{
  "field": "diagnosis_code",
  "pattern": "^[A-Z][0-9]{2}\\.[0-9]$",
  "negate": false
}
```

**Examples**:
- Invalid ICD-10 format
- Non-standard modifier format
- Unspecified diagnosis codes (ending in .9)

#### 5. CROSS_FIELD - Field Comparison

**Use Case**: Compare two fields

**Parameters**:
```json
{
  "field1": "date_of_service",
  "operator": ">",
  "field2": "date_of_birth"
}
```

**Examples**:
- Service date before birth date
- Units billed > units authorized
- Charge amount != (units × rate)

---

## Examples

### Complete Workflow: Create "Upcoding Detection" Rule

**Scenario**: Flag when E/M code is higher than typical for the provider

**Step 1**: Define the rule
```sql
INSERT INTO claims.rule_definition (
    rule_code,
    rule_name,
    rule_description,
    template_id,
    rule_parameters_encrypted,
    flag_issue_id,
    execution_order,
    execution_level,
    default_severity,
    is_active
) VALUES (
    'EM_UPCODING',
    'E/M Upcoding Detection',
    'Flags E/M codes that are higher than provider average',
    (SELECT template_id FROM claims.rule_template WHERE template_code = 'THRESHOLD'),
    pgp_sym_encrypt('{"field": "em_level", "operator": ">", "threshold": 4}', '${RULE_ENCRYPTION_KEY}'),
    (SELECT issue_id FROM claims.flag_issue WHERE issue_code = 'EMO_ONE_LEVEL'),
    12,
    'SERVICE_LINE',
    'HIGH',
    true
);
```

**Step 2**: Enable for specific organization
```sql
INSERT INTO claims.organization_rule_assignment (
    organization_id,
    rule_id,
    is_enabled,
    effective_from
) VALUES (
    1,  -- Org #1
    (SELECT rule_id FROM claims.rule_definition WHERE rule_code = 'EM_UPCODING'),
    true,
    CURRENT_DATE
);
```

**Step 3**: Override threshold for high-complexity facility
```sql
-- Facility #456 has complex cases, allow higher threshold
INSERT INTO claims.facility_rule_assignment (
    facility_id,
    rule_id,
    is_enabled,
    parameter_overrides_encrypted,
    effective_from
) VALUES (
    456,
    (SELECT rule_id FROM claims.rule_definition WHERE rule_code = 'EM_UPCODING'),
    true,
    pgp_sym_encrypt('{"threshold": 5}', '${RULE_ENCRYPTION_KEY}'),  -- Higher threshold
    CURRENT_DATE
);
```

**Step 4**: Verify assignment
```sql
-- Check what facilities have this rule
SELECT
    f.facility_code,
    f.facility_name,
    vafr.is_enabled,
    vafr.assignment_level
FROM claims.v_active_facility_rules vafr
JOIN claims.facility f ON vafr.facility_id = f.facility_id
WHERE vafr.rule_code = 'EM_UPCODING'
ORDER BY f.facility_code;
```

**Step 5**: Hot reload
```powershell
# Signal service to reload
New-Item -ItemType File -Path "C:\ProgramData\Professional SMART\reload_rules.trigger" -Force
```

**Step 6**: Monitor execution
```sql
-- Check rule statistics
SELECT
    stat_date,
    execution_count,
    flag_triggered_count,
    avg_execution_time_ms,
    total_financial_impact
FROM claims.rule_execution_stats
WHERE rule_id = (SELECT rule_id FROM claims.rule_definition WHERE rule_code = 'EM_UPCODING')
AND stat_date >= CURRENT_DATE - INTERVAL '7 days'
ORDER BY stat_date DESC;
```

---

## API Reference

### Database Functions

#### `get_active_rules_for_facility(facility_id, execution_level)`

Returns all active rules for a facility with parameter overrides.

```sql
SELECT * FROM claims.get_active_rules_for_facility(
    123,           -- facility_id
    'SERVICE_LINE' -- execution_level (or NULL for all)
);
```

**Returns**:
- `rule_id`: Rule identifier
- `rule_code`: Unique rule code
- `rule_name`: Human-readable name
- `template_code`: Template type
- `execution_order`: Execution order
- `parameter_overrides`: Encrypted parameters (decrypted in Rust)
- `assignment_level`: 'GLOBAL', 'ORGANIZATION', or 'FACILITY'

#### `record_rule_execution(rule_id, facility_id, execution_time_ms, ...)`

Records execution statistics for performance monitoring.

```sql
SELECT claims.record_rule_execution(
    rule_id := 123,
    facility_id := 456,
    execution_time_ms := 15,
    triggered := true,
    financial_impact := 500.00
);
```

### Rust API

#### `load_rules_from_database(pool, facility_id)`

Loads and instantiates rules from database.

```rust
use pro_rules::loader::load_rules_from_database;

let (engine, loaded_rules) = load_rules_from_database(&pool, Some(facility_id)).await?;

for info in loaded_rules {
    println!("Loaded: {} ({})", info.rule_code, info.rule_name);
}
```

**Environment Variable Required**:
```bash
RULE_ENCRYPTION_KEY="your-secret-key-32-chars"
```

---

## Security

### Encryption

Rule parameters are encrypted using PostgreSQL's `pgcrypto`:

```sql
-- Encrypt parameters
SELECT pgp_sym_encrypt('{"threshold": 10000}', 'encryption_key');

-- Decrypt parameters (in Rust loader)
SELECT pgp_sym_decrypt(rule_parameters_encrypted, 'encryption_key');
```

**Best Practices**:
- Store `RULE_ENCRYPTION_KEY` in environment variable
- Use different keys per environment (dev/staging/prod)
- Rotate keys periodically
- Never commit keys to version control

### Access Control

**Database Permissions**:
```sql
-- Read-only access for service
GRANT SELECT ON claims.rule_definition TO pro_app;
GRANT SELECT ON claims.facility_rule_assignment TO pro_app;

-- Admin access for configuration
GRANT ALL ON claims.rule_definition TO pro_admin;
GRANT ALL ON claims.facility_rule_assignment TO pro_admin;
```

---

## Troubleshooting

### Rule Not Loading

**Check 1**: Is rule active?
```sql
SELECT is_active FROM claims.rule_definition WHERE rule_code = 'MY_RULE';
```

**Check 2**: Is rule assigned to facility?
```sql
SELECT * FROM claims.v_active_facility_rules
WHERE facility_id = 123 AND rule_code = 'MY_RULE';
```

**Check 3**: Check service logs
```bash
type "C:\ProgramData\Professional SMART\logs\service.log" | findstr "rule"
```

### Rule Not Triggering

**Check 1**: Verify execution level
```sql
-- Rule must match data being processed
SELECT execution_level FROM claims.rule_definition WHERE rule_code = 'MY_RULE';
-- Should be 'ENCOUNTER', 'SERVICE_LINE', or 'BOTH'
```

**Check 2**: Check parameters
```sql
-- Decrypt and verify parameters
SELECT
    rule_code,
    pgp_sym_decrypt(rule_parameters_encrypted, 'encryption_key') AS parameters
FROM claims.rule_definition
WHERE rule_code = 'MY_RULE';
```

**Check 3**: Check execution statistics
```sql
SELECT
    execution_count,
    flag_triggered_count,
    error_count,
    timeout_count
FROM claims.rule_execution_stats
WHERE rule_id = (SELECT rule_id FROM claims.rule_definition WHERE rule_code = 'MY_RULE')
AND stat_date = CURRENT_DATE;
```

---

## Performance Considerations

### Rule Ordering

Lower `execution_order` = earlier execution:

```sql
-- Critical rules first (fast + high impact)
UPDATE claims.rule_definition SET execution_order = 10 WHERE rule_code = 'DUPLICATE';

-- Expensive rules last
UPDATE claims.rule_definition SET execution_order = 90 WHERE rule_code = 'ML_PREDICTION';
```

**Best Practices**:
- Orders 1-20: Fast, high-impact rules
- Orders 21-50: Medium complexity rules
- Orders 51-90: Expensive rules (ML, external APIs)
- Orders 91-99: Cleanup/aggregation rules

### Caching

Rules support automatic caching (Phase 5):
```sql
-- Check if template supports caching
SELECT template_code, supports_caching
FROM claims.rule_template;
```

### Statistics

Monitor performance with materialized view (Phase 8):
```sql
-- Top slowest rules
SELECT
    rule_code,
    avg_execution_time_ms,
    execution_count
FROM claims.rule_execution_stats_summary
ORDER BY avg_execution_time_ms DESC
LIMIT 10;
```

---

## Migration Guide

### From Hard-Coded to Database-Driven

**Step 1**: Identify hard-coded rule
```rust
// OLD: Hard-coded in Rust
struct MyCustomRule { /* ... */ }
```

**Step 2**: Map to template
```sql
-- NEW: Database configuration
INSERT INTO claims.rule_definition (...) VALUES (...);
```

**Step 3**: Test in development
```sql
-- Enable for test facility only
INSERT INTO claims.facility_rule_assignment (facility_id, rule_id, is_enabled)
VALUES (999, (SELECT rule_id FROM claims.rule_definition WHERE rule_code = 'NEW_RULE'), true);
```

**Step 4**: Roll out gradually
```sql
-- First: Enable for one organization
INSERT INTO claims.organization_rule_assignment ...

-- Then: Monitor and expand
SELECT * FROM claims.rule_execution_stats WHERE rule_id = ...
```

---

## Summary

**Where are rules stored?**
- Database: `claims.rule_definition` table
- Referenced by: `facility_rule_assignment` and `organization_rule_assignment`
- Loaded by: `loader.rs` at service startup or hot reload

**How are rules assigned to facilities?**
- **Global**: All facilities (via `rule_definition.is_active`)
- **Organization**: All facilities in org (via `organization_rule_assignment`)
- **Facility**: Specific facility (via `facility_rule_assignment`) - highest priority

**How to create ad-hoc flags without recompilation?**
1. Use existing templates (THRESHOLD, DUPLICATE, MISSING_FIELD, etc.)
2. INSERT into `claims.rule_definition` with encrypted parameters
3. Optionally assign to specific facilities
4. Hot reload service (or restart)
5. Monitor via `rule_execution_stats`

**No Rust code needed** for most common use cases!

---

## Related Documentation

- [migrations/046_create_rule_configuration_system.sql](../migrations/046_create_rule_configuration_system.sql) - Schema definition
- [migrations/006_create_flag_tables.sql](../migrations/006_create_flag_tables.sql) - Flag issues
- [crates/pro-rules/src/loader.rs](../crates/pro-rules/src/loader.rs) - Rule loading
- [FACILITY_RULE_CONFIGURATION_GUIDE.md](FACILITY_RULE_CONFIGURATION_GUIDE.md) - UI configuration
- [HOT_RELOAD.md](HOT_RELOAD.md) - Hot reload mechanism

For questions or support, check service logs or database query examples above.
