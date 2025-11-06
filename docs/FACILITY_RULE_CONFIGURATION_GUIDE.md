# Facility Rule Configuration Guide

**Date**: 2025-11-05
**Version**: v1.6.9.0
**Status**: Phase 2 Complete

---

## Overview

The rules engine supports **three levels of configuration**:

1. **Global** - Rules defined in `rule_definition` with `is_active = true` (default for all facilities)
2. **Organization** - Rules enabled/disabled for all facilities in an organization
3. **Facility** - Rules enabled/disabled for individual facilities (highest priority)

**Priority Resolution**:
```
Facility Assignment  >  Organization Assignment  >  Global Default
   (highest priority)      (medium priority)         (lowest priority)
```

---

## Configuration Levels

### Level 1: Global Rules (Default)

All rules in `claims.rule_definition` with `is_active = true` apply to ALL facilities by default.

**Example**:
```sql
-- Make a rule globally active
UPDATE claims.rule_definition
SET is_active = true
WHERE rule_code = 'DUPLICATE_SERVICE';

-- Result: All facilities will run this rule (unless overridden)
```

### Level 2: Organization-Wide Rules

Enable/disable rules for **all facilities** in an organization.

**Example**:
```sql
-- Enable UNSPECIFIED_DIAGNOSIS for all facilities in org 1
SELECT claims.enable_rule_for_organization(
    1,  -- organization_id
    'UNSPECIFIED_DIAGNOSIS',  -- rule_code
    'ADMIN'  -- assigned_by
);

-- Result: All facilities in organization 1 run this rule
```

**Query organization assignments**:
```sql
SELECT
    o.organization_name,
    rd.rule_code,
    rd.rule_name,
    ora.is_enabled,
    ora.assigned_by,
    ora.assigned_at
FROM claims.organization_rule_assignment ora
INNER JOIN claims.organization o ON ora.organization_id = o.organization_id
INNER JOIN claims.rule_definition rd ON ora.rule_id = rd.rule_id
WHERE o.organization_id = 1
ORDER BY rd.execution_order;
```

### Level 3: Facility-Specific Rules (Highest Priority)

Enable/disable rules for **individual facilities**. This overrides organization and global settings.

**Example**:
```sql
-- Enable DUPLICATE_SERVICE for facility 5
SELECT claims.enable_rule_for_facility(
    5,  -- facility_id
    'DUPLICATE_SERVICE',  -- rule_code
    'ADMIN'  -- assigned_by
);

-- Disable CONFLICTING_MODIFIERS for facility 5
SELECT claims.disable_rule_for_facility(
    5,  -- facility_id
    'CONFLICTING_MODIFIERS',  -- rule_code
    'ADMIN'  -- assigned_by
);
```

**Query facility assignments**:
```sql
SELECT
    f.facility_name,
    rd.rule_code,
    rd.rule_name,
    fra.is_enabled,
    fra.assigned_by,
    fra.assigned_at
FROM claims.facility_rule_assignment fra
INNER JOIN claims.facility f ON fra.facility_id = f.facility_id
INNER JOIN claims.rule_definition rd ON fra.rule_id = rd.rule_id
WHERE f.facility_id = 5
ORDER BY rd.execution_order;
```

---

## Common Use Cases

### Use Case 1: Enable Only Specific Rules for a Facility

**Scenario**: "Downtown Clinic" should ONLY run 3 rules, not all 6 global rules.

**Solution**:
```sql
-- Step 1: Enable only the rules you want
SELECT claims.enable_rule_for_facility(1, 'DUPLICATE_SERVICE', 'ADMIN');
SELECT claims.enable_rule_for_facility(1, 'UNITS_EXCEED_MAX', 'ADMIN');
SELECT claims.enable_rule_for_facility(1, 'UNSPECIFIED_DIAGNOSIS', 'ADMIN');

-- Step 2: Disable the rules you don't want
SELECT claims.disable_rule_for_facility(1, 'MISSING_REQUIRED_MODIFIER', 'ADMIN');
SELECT claims.disable_rule_for_facility(1, 'CONFLICTING_MODIFIERS', 'ADMIN');
SELECT claims.disable_rule_for_facility(1, 'MISSING_DIAGNOSIS_SPECIFICITY', 'ADMIN');

-- Verify
SELECT * FROM claims.debug_facility_rules(1);
```

**Result**: Facility 1 will only run the 3 enabled rules.

### Use Case 2: Disable One Rule for a Facility

**Scenario**: "Uptown Clinic" runs all global rules EXCEPT "CONFLICTING_MODIFIERS".

**Solution**:
```sql
-- Just disable the one rule
SELECT claims.disable_rule_for_facility(2, 'CONFLICTING_MODIFIERS', 'ADMIN');

-- All other rules remain active (inherit from global)
```

**Result**: Facility 2 runs 5 out of 6 global rules.

### Use Case 3: Organization-Wide Configuration

**Scenario**: All facilities in "ACME Healthcare" should run all rules.

**Solution**:
```sql
-- Enable all rules for organization 1
DO $$
DECLARE
    v_rule RECORD;
BEGIN
    FOR v_rule IN SELECT rule_id, rule_code FROM claims.rule_definition WHERE is_active = true
    LOOP
        PERFORM claims.enable_rule_for_organization(1, v_rule.rule_code, 'ADMIN');
    END LOOP;
END $$;
```

**Result**: All facilities in organization 1 run all active rules (unless facility override exists).

### Use Case 4: Reset Facility to Use Organization Defaults

**Scenario**: Remove facility-specific overrides so facility inherits organization settings.

**Solution**:
```sql
-- Delete all facility assignments
DELETE FROM claims.facility_rule_assignment
WHERE facility_id = 3;

-- Verify - should show organization or global assignments
SELECT * FROM claims.debug_facility_rules(3);
```

**Result**: Facility 3 now inherits rules from organization assignment or global defaults.

---

## Querying Active Rules

### Check What Rules a Facility Will Run

```sql
-- See all rules that will execute for facility 5
SELECT
    rule_code,
    rule_name,
    execution_level,
    assignment_level,  -- Shows if rule is from FACILITY, ORGANIZATION, or GLOBAL
    is_enabled
FROM claims.v_active_facility_rules
WHERE facility_id = 5
AND is_enabled = true
ORDER BY execution_order;
```

### Summary of All Facilities

```sql
-- See rule counts for all facilities
SELECT
    facility_code,
    facility_name,
    enabled_rules_count,
    total_rules_count,
    enabled_rules  -- Comma-separated list
FROM claims.v_facility_rule_summary
ORDER BY facility_code;
```

### Compare Two Facilities

```sql
-- See rule differences between facilities 1 and 2
WITH f1_rules AS (
    SELECT rule_code, is_enabled as f1_enabled
    FROM claims.v_active_facility_rules
    WHERE facility_id = 1
),
f2_rules AS (
    SELECT rule_code, is_enabled as f2_enabled
    FROM claims.v_active_facility_rules
    WHERE facility_id = 2
)
SELECT
    COALESCE(f1.rule_code, f2.rule_code) as rule_code,
    COALESCE(f1.f1_enabled, false) as facility_1_enabled,
    COALESCE(f2.f2_enabled, false) as facility_2_enabled,
    CASE
        WHEN f1.f1_enabled != f2.f2_enabled THEN 'DIFFERENT'
        ELSE 'SAME'
    END as comparison
FROM f1_rules f1
FULL OUTER JOIN f2_rules f2 ON f1.rule_code = f2.rule_code
ORDER BY rule_code;
```

---

## Service Startup Behavior

### Loading Rules for All Facilities (Global Mode)

```powershell
# Set environment variables
$env:ENABLE_DATABASE_RULES = "true"
$env:RULE_ENCRYPTION_KEY = "your-encryption-key"

# Start service (no facility_id specified)
./pro-service.exe
```

**What happens**:
1. Service loads ALL globally active rules
2. Rules apply to all claims regardless of facility
3. Logs: `"Loading rules from database (facility_id: None)"`

**Use when**: Single facility or rules are the same for all facilities.

### Loading Rules for Specific Facility

```rust
// In code - when creating pipeline
let pipeline = IngestionPipeline::new(pool, Some(facility_id)).await?;
```

**What happens**:
1. Service loads only rules active for that facility
2. Respects facility > organization > global priority
3. Logs: `"Loading rules from database (facility_id: Some(5))"`

**Use when**: Multi-tenant system with different rules per facility.

---

## Testing Facility Configuration

### Test 1: Verify Default Behavior (No Assignments)

```sql
-- Create a test facility with no assignments
INSERT INTO claims.facility (facility_code, facility_name, organization_id)
VALUES ('TEST001', 'Test Facility', 1)
RETURNING facility_id;  -- Note the ID (e.g., 999)

-- Check what rules it will run
SELECT * FROM claims.debug_facility_rules(999);

-- Expected: All global rules (is_active = true) with assignment_level = 'GLOBAL'
```

### Test 2: Enable One Rule for Facility

```sql
-- Enable just one rule
SELECT claims.enable_rule_for_facility(999, 'DUPLICATE_SERVICE', 'TEST');

-- Check again
SELECT * FROM claims.debug_facility_rules(999);

-- Expected: Only DUPLICATE_SERVICE shows with assignment_level = 'FACILITY'
```

### Test 3: Organization Override

```sql
-- Enable rule at organization level
SELECT claims.enable_rule_for_organization(1, 'UNITS_EXCEED_MAX', 'TEST');

-- Check facility (assuming no facility override for this rule)
SELECT * FROM claims.debug_facility_rules(999);

-- Expected: UNITS_EXCEED_MAX shows with assignment_level = 'ORGANIZATION'
```

### Test 4: Priority Resolution

```sql
-- Setup: Enable at organization level
SELECT claims.enable_rule_for_organization(1, 'UNSPECIFIED_DIAGNOSIS', 'TEST');

-- Override: Disable at facility level
SELECT claims.disable_rule_for_facility(999, 'UNSPECIFIED_DIAGNOSIS', 'TEST');

-- Check
SELECT rule_code, is_enabled, assignment_level
FROM claims.v_active_facility_rules
WHERE facility_id = 999 AND rule_code = 'UNSPECIFIED_DIAGNOSIS';

-- Expected: is_enabled = false, assignment_level = 'FACILITY'
-- (Facility override wins over organization)
```

---

## Monitoring and Auditing

### Who Changed What Rules?

```sql
-- Audit trail for facility rule changes
SELECT
    f.facility_name,
    rd.rule_code,
    fra.is_enabled,
    fra.assigned_by,
    fra.assigned_at,
    fra.updated_by,
    fra.updated_at
FROM claims.facility_rule_assignment fra
INNER JOIN claims.facility f ON fra.facility_id = f.facility_id
INNER JOIN claims.rule_definition rd ON fra.rule_id = rd.rule_id
WHERE fra.updated_at > CURRENT_DATE - INTERVAL '7 days'
ORDER BY fra.updated_at DESC;
```

### Rule Assignment History

```sql
-- See all changes to a specific rule
SELECT
    f.facility_name,
    fra.is_enabled,
    fra.assigned_by,
    fra.assigned_at,
    fra.updated_by,
    fra.updated_at
FROM claims.facility_rule_assignment fra
INNER JOIN claims.facility f ON fra.facility_id = f.facility_id
INNER JOIN claims.rule_definition rd ON fra.rule_id = rd.rule_id
WHERE rd.rule_code = 'DUPLICATE_SERVICE'
ORDER BY fra.updated_at DESC;
```

### Facilities Without Rule Overrides

```sql
-- Find facilities using only global/org rules
SELECT
    f.facility_id,
    f.facility_code,
    f.facility_name,
    COUNT(fra.assignment_id) as override_count
FROM claims.facility f
LEFT JOIN claims.facility_rule_assignment fra ON f.facility_id = fra.facility_id
GROUP BY f.facility_id, f.facility_code, f.facility_name
HAVING COUNT(fra.assignment_id) = 0
ORDER BY f.facility_code;
```

---

## API Examples (Future Frontend Integration)

### Enable Rule

```http
POST /api/facilities/5/rules/DUPLICATE_SERVICE/enable
Authorization: Bearer <token>

{
  "assigned_by": "admin@example.com"
}
```

**Backend Implementation**:
```sql
SELECT claims.enable_rule_for_facility(5, 'DUPLICATE_SERVICE', 'admin@example.com');
```

### Disable Rule

```http
POST /api/facilities/5/rules/DUPLICATE_SERVICE/disable
Authorization: Bearer <token>

{
  "assigned_by": "admin@example.com"
}
```

### Get Facility Rules

```http
GET /api/facilities/5/rules
Authorization: Bearer <token>
```

**Response**:
```json
{
  "facility_id": 5,
  "facility_code": "DOWNTOWN",
  "rules": [
    {
      "rule_code": "DUPLICATE_SERVICE",
      "rule_name": "Duplicate Service Detection",
      "is_enabled": true,
      "assignment_level": "FACILITY",
      "execution_order": 10
    },
    {
      "rule_code": "UNITS_EXCEED_MAX",
      "rule_name": "Units Exceed Maximum",
      "is_enabled": false,
      "assignment_level": "FACILITY",
      "execution_order": 20
    }
  ]
}
```

---

## Troubleshooting

### Problem: Facility not running expected rules

**Check 1**: Verify facility assignments
```sql
SELECT * FROM claims.debug_facility_rules(<facility_id>);
```

**Check 2**: Check priority resolution
```sql
-- See all levels
SELECT
    rd.rule_code,
    rd.is_active as global_active,
    ora.is_enabled as org_enabled,
    fra.is_enabled as facility_enabled,
    CASE
        WHEN fra.is_enabled IS NOT NULL THEN 'FACILITY'
        WHEN ora.is_enabled IS NOT NULL THEN 'ORGANIZATION'
        WHEN rd.is_active THEN 'GLOBAL'
        ELSE 'DISABLED'
    END as effective_level
FROM claims.rule_definition rd
LEFT JOIN claims.organization_rule_assignment ora ON (
    rd.rule_id = ora.rule_id AND ora.organization_id = <org_id>
)
LEFT JOIN claims.facility_rule_assignment fra ON (
    rd.rule_id = fra.rule_id AND fra.facility_id = <facility_id>
)
WHERE rd.is_active = true;
```

### Problem: Rules not loading at service startup

**Check 1**: Environment variables set?
```powershell
echo $env:ENABLE_DATABASE_RULES  # Should be "true"
echo $env:RULE_ENCRYPTION_KEY    # Should be set
```

**Check 2**: Check service logs
```powershell
Get-Content "C:\ProgramData\Professional SMART\logs\service.log" -Tail 50 | Select-String "rule"
```

**Expected log lines**:
```
Loading rules from database (facility_id: Some(5))
Loaded 3 rule(s) from database
  - DUPLICATE_SERVICE (SERVICE_LINE): Duplicate Service Detection
  - UNITS_EXCEED_MAX (SERVICE_LINE): Units Exceed Maximum
  - UNSPECIFIED_DIAGNOSIS (ENCOUNTER): Unspecified Diagnosis Code
```

---

## Best Practices

### 1. Use Organization-Level for Common Policies

✅ **DO**: Set rules at organization level for policies that apply to all facilities
```sql
-- All facilities in ACME should check for duplicates
SELECT claims.enable_rule_for_organization(1, 'DUPLICATE_SERVICE', 'ADMIN');
```

❌ **DON'T**: Set the same rule individually for 50 facilities

### 2. Use Facility-Level for Exceptions Only

✅ **DO**: Override at facility level only when needed
```sql
-- One facility has special workflow, disable this rule
SELECT claims.disable_rule_for_facility(25, 'CONFLICTING_MODIFIERS', 'ADMIN');
```

❌ **DON'T**: Configure every rule for every facility (maintenance nightmare)

### 3. Document Why Rules Are Disabled

```sql
-- Add notes in updated_by field
SELECT claims.disable_rule_for_facility(
    30,
    'UNITS_EXCEED_MAX',
    'ADMIN: Disabled per client request - ticket #1234'
);
```

### 4. Test Configuration Before Rollout

```sql
-- Always test on a single facility first
SELECT claims.enable_rule_for_facility(99, 'NEW_RULE_CODE', 'ADMIN');

-- Process some test claims
-- Verify results
-- Then roll out to more facilities
```

---

## Summary

**Phase 2 provides**:
- ✅ Three-level configuration (Global > Organization > Facility)
- ✅ Helper functions for enable/disable
- ✅ Views for querying active rules
- ✅ Audit trail via assigned_by/updated_by
- ✅ Debug functions for troubleshooting

**Next Phase** (Phase 3): Template-based rules with configurable parameters.

---

**Document Version**: 1.0
**Last Updated**: 2025-11-05
**Status**: Phase 2 Complete
