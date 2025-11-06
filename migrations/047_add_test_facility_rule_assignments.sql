-- Migration: 047_add_test_facility_rule_assignments
-- Description: Add example facility and organization rule assignments for testing
-- Date: 2025-11-05
-- Purpose: Demonstrate per-facility rule configuration

-- ============================================================================
-- 1. Example: Enable specific rules for a facility
-- ============================================================================

-- Example: Downtown Clinic (facility_id = 1) enables only 3 rules
DO $$
DECLARE
    v_rule_id BIGINT;
BEGIN
    -- Enable DUPLICATE_SERVICE for facility 1
    SELECT rule_id INTO v_rule_id FROM claims.rule_definition WHERE rule_code = 'DUPLICATE_SERVICE';
    IF v_rule_id IS NOT NULL THEN
        INSERT INTO claims.facility_rule_assignment (facility_id, rule_id, is_enabled, assigned_by)
        VALUES (1, v_rule_id, true, 'SYSTEM_INIT')
        ON CONFLICT (facility_id, rule_id) DO NOTHING;
    END IF;

    -- Enable UNITS_EXCEED_MAX for facility 1
    SELECT rule_id INTO v_rule_id FROM claims.rule_definition WHERE rule_code = 'UNITS_EXCEED_MAX';
    IF v_rule_id IS NOT NULL THEN
        INSERT INTO claims.facility_rule_assignment (facility_id, rule_id, is_enabled, assigned_by)
        VALUES (1, v_rule_id, true, 'SYSTEM_INIT')
        ON CONFLICT (facility_id, rule_id) DO NOTHING;
    END IF;

    -- Enable UNSPECIFIED_DIAGNOSIS for facility 1
    SELECT rule_id INTO v_rule_id FROM claims.rule_definition WHERE rule_code = 'UNSPECIFIED_DIAGNOSIS';
    IF v_rule_id IS NOT NULL THEN
        INSERT INTO claims.facility_rule_assignment (facility_id, rule_id, is_enabled, assigned_by)
        VALUES (1, v_rule_id, true, 'SYSTEM_INIT')
        ON CONFLICT (facility_id, rule_id) DO NOTHING;
    END IF;
END $$;

-- ============================================================================
-- 2. Example: Disable specific rule for a facility
-- ============================================================================

-- Example: Uptown Clinic (facility_id = 2) disables CONFLICTING_MODIFIERS
DO $$
DECLARE
    v_rule_id BIGINT;
BEGIN
    SELECT rule_id INTO v_rule_id FROM claims.rule_definition WHERE rule_code = 'CONFLICTING_MODIFIERS';
    IF v_rule_id IS NOT NULL THEN
        INSERT INTO claims.facility_rule_assignment (facility_id, rule_id, is_enabled, assigned_by)
        VALUES (2, v_rule_id, false, 'SYSTEM_INIT')
        ON CONFLICT (facility_id, rule_id) DO NOTHING;
    END IF;
END $$;

-- ============================================================================
-- 3. Example: Organization-wide rule assignments
-- ============================================================================

-- Example: Organization 1 (ACME) enables all rules for all its facilities
DO $$
DECLARE
    v_rule RECORD;
BEGIN
    FOR v_rule IN SELECT rule_id FROM claims.rule_definition WHERE is_active = true
    LOOP
        INSERT INTO claims.organization_rule_assignment (organization_id, rule_id, is_enabled, assigned_by)
        VALUES (1, v_rule.rule_id, true, 'SYSTEM_INIT')
        ON CONFLICT (organization_id, rule_id) DO NOTHING;
    END LOOP;
END $$;

-- ============================================================================
-- 4. View: Rule assignment summary by facility
-- ============================================================================

CREATE OR REPLACE VIEW claims.v_facility_rule_summary AS
SELECT
    f.facility_id,
    f.facility_code,
    f.facility_name,
    f.organization_id,
    o.organization_name,
    COUNT(DISTINCT CASE WHEN vafr.is_enabled THEN vafr.rule_id END) as enabled_rules_count,
    COUNT(DISTINCT rd.rule_id) as total_rules_count,
    STRING_AGG(
        DISTINCT CASE WHEN vafr.is_enabled THEN vafr.rule_code END,
        ', ' ORDER BY CASE WHEN vafr.is_enabled THEN vafr.rule_code END
    ) as enabled_rules
FROM claims.facility f
INNER JOIN claims.organization o ON f.organization_id = o.organization_id
CROSS JOIN claims.rule_definition rd
LEFT JOIN claims.v_active_facility_rules vafr ON (
    vafr.facility_id = f.facility_id
    AND vafr.rule_id = rd.rule_id
)
WHERE rd.is_active = true
GROUP BY f.facility_id, f.facility_code, f.facility_name, f.organization_id, o.organization_name
ORDER BY f.facility_code;

COMMENT ON VIEW claims.v_facility_rule_summary IS 'Summary of enabled rules per facility';

-- ============================================================================
-- 5. Function: Enable rule for facility
-- ============================================================================

CREATE OR REPLACE FUNCTION claims.enable_rule_for_facility(
    p_facility_id BIGINT,
    p_rule_code VARCHAR,
    p_assigned_by VARCHAR DEFAULT 'ADMIN'
) RETURNS VOID AS $$
DECLARE
    v_rule_id BIGINT;
BEGIN
    -- Get rule ID
    SELECT rule_id INTO v_rule_id
    FROM claims.rule_definition
    WHERE rule_code = p_rule_code AND is_active = true;

    IF v_rule_id IS NULL THEN
        RAISE EXCEPTION 'Rule not found: %', p_rule_code;
    END IF;

    -- Insert or update assignment
    INSERT INTO claims.facility_rule_assignment (
        facility_id, rule_id, is_enabled, assigned_by, updated_at, updated_by
    ) VALUES (
        p_facility_id, v_rule_id, true, p_assigned_by, CURRENT_TIMESTAMP, p_assigned_by
    )
    ON CONFLICT (facility_id, rule_id)
    DO UPDATE SET
        is_enabled = true,
        updated_at = CURRENT_TIMESTAMP,
        updated_by = p_assigned_by;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION claims.enable_rule_for_facility IS 'Enable a specific rule for a facility';

-- ============================================================================
-- 6. Function: Disable rule for facility
-- ============================================================================

CREATE OR REPLACE FUNCTION claims.disable_rule_for_facility(
    p_facility_id BIGINT,
    p_rule_code VARCHAR,
    p_assigned_by VARCHAR DEFAULT 'ADMIN'
) RETURNS VOID AS $$
DECLARE
    v_rule_id BIGINT;
BEGIN
    -- Get rule ID
    SELECT rule_id INTO v_rule_id
    FROM claims.rule_definition
    WHERE rule_code = p_rule_code AND is_active = true;

    IF v_rule_id IS NULL THEN
        RAISE EXCEPTION 'Rule not found: %', p_rule_code;
    END IF;

    -- Insert or update assignment
    INSERT INTO claims.facility_rule_assignment (
        facility_id, rule_id, is_enabled, assigned_by, updated_at, updated_by
    ) VALUES (
        p_facility_id, v_rule_id, false, p_assigned_by, CURRENT_TIMESTAMP, p_assigned_by
    )
    ON CONFLICT (facility_id, rule_id)
    DO UPDATE SET
        is_enabled = false,
        updated_at = CURRENT_TIMESTAMP,
        updated_by = p_assigned_by;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION claims.disable_rule_for_facility IS 'Disable a specific rule for a facility';

-- ============================================================================
-- 7. Function: Enable rule for organization (all facilities)
-- ============================================================================

CREATE OR REPLACE FUNCTION claims.enable_rule_for_organization(
    p_organization_id BIGINT,
    p_rule_code VARCHAR,
    p_assigned_by VARCHAR DEFAULT 'ADMIN'
) RETURNS VOID AS $$
DECLARE
    v_rule_id BIGINT;
BEGIN
    -- Get rule ID
    SELECT rule_id INTO v_rule_id
    FROM claims.rule_definition
    WHERE rule_code = p_rule_code AND is_active = true;

    IF v_rule_id IS NULL THEN
        RAISE EXCEPTION 'Rule not found: %', p_rule_code;
    END IF;

    -- Insert or update assignment
    INSERT INTO claims.organization_rule_assignment (
        organization_id, rule_id, is_enabled, assigned_by, updated_at, updated_by
    ) VALUES (
        p_organization_id, v_rule_id, true, p_assigned_by, CURRENT_TIMESTAMP, p_assigned_by
    )
    ON CONFLICT (organization_id, rule_id)
    DO UPDATE SET
        is_enabled = true,
        updated_at = CURRENT_TIMESTAMP,
        updated_by = p_assigned_by;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION claims.enable_rule_for_organization IS 'Enable a specific rule for all facilities in an organization';

-- ============================================================================
-- 8. Function: Get rules for facility (debugging)
-- ============================================================================

CREATE OR REPLACE FUNCTION claims.debug_facility_rules(
    p_facility_id BIGINT
) RETURNS TABLE (
    rule_code VARCHAR,
    rule_name VARCHAR,
    is_enabled BOOLEAN,
    assignment_level VARCHAR,
    execution_order INT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        vafr.rule_code,
        vafr.rule_name,
        vafr.is_enabled,
        vafr.assignment_level,
        vafr.execution_order
    FROM claims.v_active_facility_rules vafr
    WHERE vafr.facility_id = p_facility_id
    ORDER BY vafr.execution_order;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION claims.debug_facility_rules IS 'Debug: Show all rules and their status for a facility';
