-- Migration: 046_create_rule_configuration_system
-- Description: Create data-driven rule configuration system with encryption
-- Date: 2025-11-05
-- Purpose: Enable per-facility rule configuration without code recompilation

-- Enable pgcrypto extension for encryption
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ============================================================================
-- 1. Rule Templates (Compiled Rust code)
-- ============================================================================

CREATE TABLE IF NOT EXISTS claims.rule_template (
    template_id BIGINT GENERATED ALWAYS AS IDENTITY (CACHE 50) PRIMARY KEY,
    template_code VARCHAR(50) NOT NULL UNIQUE, -- e.g., 'THRESHOLD', 'DUPLICATE', 'MISSING_FIELD'
    template_name VARCHAR(100) NOT NULL,
    template_description TEXT,

    -- Rust implementation
    rust_struct_name VARCHAR(100) NOT NULL, -- e.g., 'ThresholdRule'

    -- Parameter schema (JSON Schema for validation)
    parameter_schema JSONB NOT NULL, -- Defines what parameters this template accepts

    -- Execution characteristics
    execution_level VARCHAR(20) NOT NULL CHECK (execution_level IN ('ENCOUNTER', 'SERVICE_LINE', 'BOTH')),
    supports_caching BOOLEAN DEFAULT true,
    estimated_execution_time_ms INT DEFAULT 5,

    -- Metadata
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100) DEFAULT 'SYSTEM'
);

CREATE INDEX IF NOT EXISTS idx_rule_template_code ON claims.rule_template(template_code);
CREATE INDEX IF NOT EXISTS idx_rule_template_active ON claims.rule_template(is_active);

COMMENT ON TABLE claims.rule_template IS 'Pre-compiled rule templates implemented in Rust';

-- ============================================================================
-- 2. Rule Definitions (Instances of templates)
-- ============================================================================

CREATE TABLE IF NOT EXISTS claims.rule_definition (
    rule_id BIGINT GENERATED ALWAYS AS IDENTITY (CACHE 100) PRIMARY KEY,

    -- Rule metadata
    rule_code VARCHAR(50) NOT NULL UNIQUE, -- e.g., 'HIGH_VALUE_NO_AUTH'
    rule_name VARCHAR(200) NOT NULL,
    rule_description TEXT,

    -- Template reference (NULL for legacy hard-coded rules)
    template_id BIGINT REFERENCES claims.rule_template(template_id),

    -- Encrypted rule logic
    -- Option 1: Template-based (most common)
    rule_parameters_encrypted BYTEA, -- pgp_sym_encrypt(JSON parameters)

    -- Option 2: DSL-based (future)
    rule_dsl_encrypted BYTEA, -- pgp_sym_encrypt(DSL JSON)

    -- Option 3: SQL-based (advanced)
    rule_sql_encrypted BYTEA, -- pgp_sym_encrypt(PostgreSQL function name)

    -- Encryption metadata
    encryption_key_id VARCHAR(50) DEFAULT 'v1', -- Which key version was used

    -- Flag configuration
    flag_issue_id BIGINT REFERENCES claims.flag_issue(issue_id),
    default_severity VARCHAR(20) DEFAULT 'MEDIUM' CHECK (default_severity IN ('HIGH', 'MEDIUM', 'LOW')),

    -- Execution control
    execution_order INT DEFAULT 100, -- Lower = earlier execution
    execution_level VARCHAR(20) NOT NULL CHECK (execution_level IN ('ENCOUNTER', 'SERVICE_LINE', 'BOTH')),
    timeout_ms INT DEFAULT 5000, -- Max execution time

    -- Enable globally by default (can be overridden per facility/org)
    is_active BOOLEAN DEFAULT true,

    -- Metadata
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100) DEFAULT 'SYSTEM',
    updated_at TIMESTAMPTZ,
    updated_by VARCHAR(100),

    -- Versioning
    version INT DEFAULT 1,
    replaces_rule_id BIGINT REFERENCES claims.rule_definition(rule_id)
);

CREATE INDEX IF NOT EXISTS idx_rule_definition_code ON claims.rule_definition(rule_code);
CREATE INDEX IF NOT EXISTS idx_rule_definition_template ON claims.rule_definition(template_id);
CREATE INDEX IF NOT EXISTS idx_rule_definition_active ON claims.rule_definition(is_active);
CREATE INDEX IF NOT EXISTS idx_rule_definition_order ON claims.rule_definition(execution_order);
CREATE INDEX IF NOT EXISTS idx_rule_definition_level ON claims.rule_definition(execution_level);

COMMENT ON TABLE claims.rule_definition IS 'Rule instances configured from templates or custom implementations';

-- ============================================================================
-- 3. Facility Rule Assignments (Enable/Disable per Facility)
-- ============================================================================

CREATE TABLE IF NOT EXISTS claims.facility_rule_assignment (
    assignment_id BIGINT GENERATED ALWAYS AS IDENTITY (CACHE 100) PRIMARY KEY,

    -- Scope
    facility_id BIGINT NOT NULL REFERENCES claims.facility(facility_id) ON DELETE CASCADE,
    rule_id BIGINT NOT NULL REFERENCES claims.rule_definition(rule_id) ON DELETE CASCADE,

    -- Activation
    is_enabled BOOLEAN DEFAULT true,

    -- Parameter overrides (optional - facility-specific parameters)
    parameter_overrides_encrypted BYTEA, -- Facility can override template parameters

    -- Effective dates
    effective_from DATE DEFAULT CURRENT_DATE,
    effective_to DATE, -- NULL = indefinite

    -- Audit
    assigned_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    assigned_by VARCHAR(100),
    updated_at TIMESTAMPTZ,
    updated_by VARCHAR(100),

    -- Prevent duplicate assignments
    UNIQUE (facility_id, rule_id)
);

CREATE INDEX IF NOT EXISTS idx_facility_rule_facility ON claims.facility_rule_assignment(facility_id);
CREATE INDEX IF NOT EXISTS idx_facility_rule_rule ON claims.facility_rule_assignment(rule_id);
CREATE INDEX IF NOT EXISTS idx_facility_rule_enabled ON claims.facility_rule_assignment(is_enabled);
CREATE INDEX IF NOT EXISTS idx_facility_rule_dates ON claims.facility_rule_assignment(effective_from, effective_to);

-- Composite index for fast rule lookup by facility
-- Note: Cannot use CURRENT_DATE in index predicate as it's not immutable
CREATE INDEX IF NOT EXISTS idx_facility_rule_active_lookup ON claims.facility_rule_assignment(facility_id, rule_id, is_enabled)
    WHERE is_enabled = true;

COMMENT ON TABLE claims.facility_rule_assignment IS 'Per-facility rule activation and parameter overrides';

-- ============================================================================
-- 4. Organization Rule Assignments (Enable/Disable per Organization)
-- ============================================================================

CREATE TABLE IF NOT EXISTS claims.organization_rule_assignment (
    assignment_id BIGINT GENERATED ALWAYS AS IDENTITY (CACHE 100) PRIMARY KEY,

    organization_id BIGINT NOT NULL REFERENCES claims.organization(organization_id) ON DELETE CASCADE,
    rule_id BIGINT NOT NULL REFERENCES claims.rule_definition(rule_id) ON DELETE CASCADE,

    is_enabled BOOLEAN DEFAULT true,
    parameter_overrides_encrypted BYTEA,

    effective_from DATE DEFAULT CURRENT_DATE,
    effective_to DATE,

    assigned_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    assigned_by VARCHAR(100),
    updated_at TIMESTAMPTZ,
    updated_by VARCHAR(100),

    UNIQUE (organization_id, rule_id)
);

CREATE INDEX IF NOT EXISTS idx_org_rule_org ON claims.organization_rule_assignment(organization_id);
CREATE INDEX IF NOT EXISTS idx_org_rule_rule ON claims.organization_rule_assignment(rule_id);
CREATE INDEX IF NOT EXISTS idx_org_rule_enabled ON claims.organization_rule_assignment(is_enabled);
CREATE INDEX IF NOT EXISTS idx_org_rule_dates ON claims.organization_rule_assignment(effective_from, effective_to);

COMMENT ON TABLE claims.organization_rule_assignment IS 'Per-organization rule activation (applies to all facilities in org unless overridden)';

-- ============================================================================
-- 5. Rule Execution Statistics
-- ============================================================================

CREATE TABLE IF NOT EXISTS claims.rule_execution_stats (
    stat_id BIGINT GENERATED ALWAYS AS IDENTITY (CACHE 100) PRIMARY KEY,

    rule_id BIGINT NOT NULL REFERENCES claims.rule_definition(rule_id) ON DELETE CASCADE,
    facility_id BIGINT REFERENCES claims.facility(facility_id) ON DELETE CASCADE,

    -- Statistics window
    stat_date DATE NOT NULL DEFAULT CURRENT_DATE,

    -- Execution metrics
    execution_count INT DEFAULT 0,
    flag_triggered_count INT DEFAULT 0,
    total_execution_time_ms BIGINT DEFAULT 0,
    avg_execution_time_ms NUMERIC(10,2),
    max_execution_time_ms INT,
    min_execution_time_ms INT,

    -- Error tracking
    error_count INT DEFAULT 0,
    timeout_count INT DEFAULT 0,

    -- Impact metrics
    total_financial_impact NUMERIC(18,2),

    -- Timestamps
    first_execution_at TIMESTAMPTZ,
    last_execution_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,

    UNIQUE (rule_id, facility_id, stat_date)
);

CREATE INDEX IF NOT EXISTS idx_rule_stats_rule ON claims.rule_execution_stats(rule_id);
CREATE INDEX IF NOT EXISTS idx_rule_stats_facility ON claims.rule_execution_stats(facility_id);
CREATE INDEX IF NOT EXISTS idx_rule_stats_date ON claims.rule_execution_stats(stat_date);
CREATE INDEX IF NOT EXISTS idx_rule_stats_date_rule ON claims.rule_execution_stats(stat_date, rule_id);

COMMENT ON TABLE claims.rule_execution_stats IS 'Daily execution statistics per rule per facility';

-- ============================================================================
-- 6. View: Active Rules by Facility
-- ============================================================================

CREATE OR REPLACE VIEW claims.v_active_facility_rules AS
SELECT
    f.facility_id,
    f.facility_code,
    f.facility_name,
    f.organization_id,
    rd.rule_id,
    rd.rule_code,
    rd.rule_name,
    rd.template_id,
    rt.template_code,
    rd.execution_order,
    rd.execution_level,
    rd.flag_issue_id,
    fi.issue_code,
    fi.severity,

    -- Assignment info
    COALESCE(fra.is_enabled, ora.is_enabled, rd.is_active) AS is_enabled,
    COALESCE(fra.effective_from, ora.effective_from, rd.created_at::DATE) AS effective_from,
    COALESCE(fra.effective_to, ora.effective_to) AS effective_to,

    -- Determine assignment level
    CASE
        WHEN fra.assignment_id IS NOT NULL THEN 'FACILITY'
        WHEN ora.assignment_id IS NOT NULL THEN 'ORGANIZATION'
        WHEN rd.is_active THEN 'GLOBAL'
        ELSE 'NONE'
    END AS assignment_level

FROM claims.facility f
INNER JOIN claims.organization o ON f.organization_id = o.organization_id
CROSS JOIN claims.rule_definition rd
LEFT JOIN claims.rule_template rt ON rd.template_id = rt.template_id
LEFT JOIN claims.flag_issue fi ON rd.flag_issue_id = fi.issue_id
LEFT JOIN claims.facility_rule_assignment fra ON (
    fra.facility_id = f.facility_id
    AND fra.rule_id = rd.rule_id
    AND (fra.effective_to IS NULL OR fra.effective_to >= CURRENT_DATE)
)
LEFT JOIN claims.organization_rule_assignment ora ON (
    ora.organization_id = o.organization_id
    AND ora.rule_id = rd.rule_id
    AND (ora.effective_to IS NULL OR ora.effective_to >= CURRENT_DATE)
    AND fra.assignment_id IS NULL -- Facility assignment takes precedence
)
WHERE rd.is_active = true
AND (rt.is_active IS NULL OR rt.is_active = true)
AND (fra.is_enabled = true OR ora.is_enabled = true OR (fra.assignment_id IS NULL AND ora.assignment_id IS NULL));

COMMENT ON VIEW claims.v_active_facility_rules IS 'All active rules for each facility (facility assignments override organization assignments)';

-- ============================================================================
-- 7. Insert Rule Templates
-- ============================================================================

INSERT INTO claims.rule_template (template_code, template_name, template_description, rust_struct_name, parameter_schema, execution_level, estimated_execution_time_ms) VALUES
-- Core templates
('LEGACY', 'Legacy Hard-coded Rule', 'Existing hard-coded Rust rule (no parameters)', 'LegacyRule',
 '{"type": "object", "properties": {}}', 'BOTH', 2),

('THRESHOLD', 'Threshold Comparison', 'Compare a numeric field against a threshold value', 'ThresholdRule',
 '{"type": "object", "properties": {"field": {"type": "string", "enum": ["total_charge", "line_charge", "units"]}, "operator": {"enum": [">", "<", ">=", "<=", "=", "!="]}, "threshold": {"type": "number"}}, "required": ["field", "operator", "threshold"]}',
 'BOTH', 1),

('DUPLICATE', 'Duplicate Detection', 'Detect duplicate records within timeframe', 'DuplicateRule',
 '{"type": "object", "properties": {"match_fields": {"type": "array", "items": {"type": "string"}}, "timeframe_days": {"type": "integer", "minimum": 1, "maximum": 365}}, "required": ["match_fields"]}',
 'SERVICE_LINE', 2),

('MISSING_FIELD', 'Missing Required Field', 'Flag when required field is null or empty', 'MissingFieldRule',
 '{"type": "object", "properties": {"field": {"type": "string"}, "when_condition": {"type": "object"}}, "required": ["field"]}',
 'BOTH', 1),

('FIELD_PATTERN', 'Field Pattern Match', 'Validate field against regex or pattern', 'FieldPatternRule',
 '{"type": "object", "properties": {"field": {"type": "string"}, "pattern": {"type": "string"}, "negate": {"type": "boolean"}}, "required": ["field", "pattern"]}',
 'BOTH', 1),

('CROSS_FIELD', 'Cross-Field Comparison', 'Compare two fields against each other', 'CrossFieldRule',
 '{"type": "object", "properties": {"field1": {"type": "string"}, "operator": {"enum": [">", "<", ">=", "<=", "=", "!="]}, "field2": {"type": "string"}}, "required": ["field1", "operator", "field2"]}',
 'BOTH', 1)
ON CONFLICT (template_code) DO NOTHING;

-- ============================================================================
-- 8. Migrate Existing Rules to Database (Legacy mapping)
-- ============================================================================

-- Get the LEGACY template ID for use in inserts
DO $$
DECLARE
    legacy_template_id BIGINT;
    oth_dup_issue_id BIGINT;
    qty_exceed_issue_id BIGINT;
    mod_missing_issue_id BIGINT;
    mod_conflict_issue_id BIGINT;
    dx_spec_issue_id BIGINT;
BEGIN
    -- Get template and issue IDs
    SELECT template_id INTO legacy_template_id FROM claims.rule_template WHERE template_code = 'LEGACY';
    SELECT issue_id INTO oth_dup_issue_id FROM claims.flag_issue WHERE issue_code = 'OTH_DUPLICATE';
    SELECT issue_id INTO qty_exceed_issue_id FROM claims.flag_issue WHERE issue_code = 'QTY_FEWER';
    SELECT issue_id INTO mod_missing_issue_id FROM claims.flag_issue WHERE issue_code = 'MOD_MISSING';
    SELECT issue_id INTO mod_conflict_issue_id FROM claims.flag_issue WHERE issue_code = 'MOD_INCORRECT';
    SELECT issue_id INTO dx_spec_issue_id FROM claims.flag_issue WHERE issue_code = 'DX_SPECIFICITY';

    -- Insert existing 6 rules as LEGACY (they remain hard-coded until migrated to templates)
    INSERT INTO claims.rule_definition (
        rule_code, rule_name, rule_description, template_id,
        flag_issue_id, execution_order, execution_level,
        default_severity, is_active
    ) VALUES
    ('DUPLICATE_SERVICE', 'Duplicate Service Detection',
     'Detects duplicate procedures on same encounter and date',
     legacy_template_id, oth_dup_issue_id, 10, 'SERVICE_LINE', 'HIGH', true),

    ('UNITS_EXCEED_MAX', 'Units Exceed Maximum',
     'Validates service unit counts against maximum threshold',
     legacy_template_id, qty_exceed_issue_id, 20, 'SERVICE_LINE', 'HIGH', true),

    ('MISSING_REQUIRED_MODIFIER', 'Missing Required Modifier',
     'Checks for required modifiers on bilateral procedures',
     legacy_template_id, mod_missing_issue_id, 30, 'SERVICE_LINE', 'MEDIUM', true),

    ('CONFLICTING_MODIFIERS', 'Conflicting Modifiers',
     'Detects mutually exclusive modifiers',
     legacy_template_id, mod_conflict_issue_id, 40, 'SERVICE_LINE', 'MEDIUM', true),

    ('UNSPECIFIED_DIAGNOSIS', 'Unspecified Diagnosis Code',
     'Flags diagnosis codes ending in .9 or .90 (unspecified)',
     legacy_template_id, dx_spec_issue_id, 50, 'ENCOUNTER', 'MEDIUM', true),

    ('MISSING_DIAGNOSIS_SPECIFICITY', 'Missing Diagnosis Specificity',
     'Checks diagnosis code length and laterality requirements',
     legacy_template_id, dx_spec_issue_id, 60, 'ENCOUNTER', 'MEDIUM', true);

END $$;

-- ============================================================================
-- 9. Function: Get Active Rules for Facility
-- ============================================================================

CREATE OR REPLACE FUNCTION claims.get_active_rules_for_facility(
    p_facility_id BIGINT,
    p_execution_level VARCHAR DEFAULT NULL -- 'ENCOUNTER', 'SERVICE_LINE', or NULL for both
) RETURNS TABLE (
    rule_id BIGINT,
    rule_code VARCHAR,
    rule_name VARCHAR,
    template_code VARCHAR,
    execution_order INT,
    execution_level VARCHAR,
    flag_issue_id BIGINT,
    parameter_overrides BYTEA,
    assignment_level VARCHAR
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        vafr.rule_id,
        vafr.rule_code,
        vafr.rule_name,
        vafr.template_code,
        vafr.execution_order,
        vafr.execution_level,
        vafr.flag_issue_id,
        COALESCE(fra.parameter_overrides_encrypted, ora.parameter_overrides_encrypted) AS parameter_overrides,
        vafr.assignment_level
    FROM claims.v_active_facility_rules vafr
    LEFT JOIN claims.facility_rule_assignment fra ON (
        fra.facility_id = vafr.facility_id AND fra.rule_id = vafr.rule_id
    )
    LEFT JOIN claims.organization_rule_assignment ora ON (
        ora.organization_id = vafr.organization_id AND ora.rule_id = vafr.rule_id AND fra.assignment_id IS NULL
    )
    WHERE vafr.facility_id = p_facility_id
    AND vafr.is_enabled = true
    AND (p_execution_level IS NULL OR vafr.execution_level = p_execution_level OR vafr.execution_level = 'BOTH')
    ORDER BY vafr.execution_order;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION claims.get_active_rules_for_facility IS 'Returns all active rules for a facility with parameter overrides';

-- ============================================================================
-- 10. Function: Record Rule Execution Stats
-- ============================================================================

CREATE OR REPLACE FUNCTION claims.record_rule_execution(
    p_rule_id BIGINT,
    p_facility_id BIGINT,
    p_execution_time_ms INT,
    p_triggered BOOLEAN DEFAULT false,
    p_financial_impact NUMERIC DEFAULT NULL,
    p_error BOOLEAN DEFAULT false,
    p_timeout BOOLEAN DEFAULT false
) RETURNS VOID AS $$
BEGIN
    INSERT INTO claims.rule_execution_stats (
        rule_id,
        facility_id,
        stat_date,
        execution_count,
        flag_triggered_count,
        total_execution_time_ms,
        avg_execution_time_ms,
        max_execution_time_ms,
        min_execution_time_ms,
        error_count,
        timeout_count,
        total_financial_impact,
        first_execution_at,
        last_execution_at
    ) VALUES (
        p_rule_id,
        p_facility_id,
        CURRENT_DATE,
        1,
        CASE WHEN p_triggered THEN 1 ELSE 0 END,
        p_execution_time_ms,
        p_execution_time_ms,
        p_execution_time_ms,
        p_execution_time_ms,
        CASE WHEN p_error THEN 1 ELSE 0 END,
        CASE WHEN p_timeout THEN 1 ELSE 0 END,
        COALESCE(p_financial_impact, 0),
        CURRENT_TIMESTAMP,
        CURRENT_TIMESTAMP
    )
    ON CONFLICT (rule_id, facility_id, stat_date)
    DO UPDATE SET
        execution_count = claims.rule_execution_stats.execution_count + 1,
        flag_triggered_count = claims.rule_execution_stats.flag_triggered_count + CASE WHEN p_triggered THEN 1 ELSE 0 END,
        total_execution_time_ms = claims.rule_execution_stats.total_execution_time_ms + p_execution_time_ms,
        avg_execution_time_ms = (claims.rule_execution_stats.total_execution_time_ms + p_execution_time_ms) / (claims.rule_execution_stats.execution_count + 1),
        max_execution_time_ms = GREATEST(claims.rule_execution_stats.max_execution_time_ms, p_execution_time_ms),
        min_execution_time_ms = LEAST(claims.rule_execution_stats.min_execution_time_ms, p_execution_time_ms),
        error_count = claims.rule_execution_stats.error_count + CASE WHEN p_error THEN 1 ELSE 0 END,
        timeout_count = claims.rule_execution_stats.timeout_count + CASE WHEN p_timeout THEN 1 ELSE 0 END,
        total_financial_impact = claims.rule_execution_stats.total_financial_impact + COALESCE(p_financial_impact, 0),
        last_execution_at = CURRENT_TIMESTAMP,
        updated_at = CURRENT_TIMESTAMP;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION claims.record_rule_execution IS 'Records execution statistics for a rule (upserts daily stats)';
