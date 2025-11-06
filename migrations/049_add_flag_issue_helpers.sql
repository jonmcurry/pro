-- Migration 049: Add helper functions for FlagIssueType mapping
-- Phase 4: Proper database-driven flag type resolution
--
-- This migration adds:
-- 1. Helper function to get FlagIssueType enum name from issue_code
-- 2. Helper view for rule loading with proper flag type mapping

-- Helper function: Get FlagIssueType enum name from issue_code
-- This maps the database issue_code to the Rust enum variant name
CREATE OR REPLACE FUNCTION claims.get_flag_issue_type_name(p_issue_code TEXT)
RETURNS TEXT
LANGUAGE plpgsql
STABLE
AS $$
BEGIN
    -- Map issue_code to Rust FlagIssueType enum name
    -- The enum names follow the pattern: Category + CamelCaseDescription
    RETURN CASE p_issue_code
        -- Coding (COD) issues
        WHEN 'COD_BUNDLED' THEN 'CodBundledService'
        WHEN 'COD_INCORRECT' THEN 'CodIncorrectProcedureCode'
        WHEN 'COD_MISSED' THEN 'CodMissedCharge'
        WHEN 'COD_UPCODING' THEN 'CodUpcoding'
        WHEN 'COD_DOWNCODING' THEN 'CodDowncoding'
        WHEN 'COD_UNBUNDLING' THEN 'CodUnbundling'
        WHEN 'COD_BILATERAL' THEN 'CodBilateralProcedure'
        WHEN 'COD_GLOBAL_PERIOD' THEN 'CodGlobalPeriod'

        -- Modifier (MOD) issues
        WHEN 'MOD_MISSING_REQUIRED' THEN 'ModMissingRequired'
        WHEN 'MOD_INCORRECT' THEN 'ModIncorrect'
        WHEN 'MOD_CONFLICTING' THEN 'ModConflicting'
        WHEN 'MOD_UNNECESSARY' THEN 'ModUnnecessary'
        WHEN 'MOD_MISSING_59' THEN 'ModMissing59'
        WHEN 'MOD_MISSING_25' THEN 'ModMissing25'

        -- Quantity/Units (QTY) issues
        WHEN 'QTY_UNITS_EXCEED_MAX' THEN 'QtyUnitsExceedMaximum'
        WHEN 'QTY_UNITS_INCONSISTENT' THEN 'QtyUnitsInconsistent'
        WHEN 'QTY_MULTIPLE_UNITS' THEN 'QtyMultipleUnitsNotSupported'
        WHEN 'QTY_TIME_BASED' THEN 'QtyTimeBasedCoding'

        -- Diagnosis (DX) issues
        WHEN 'DX_UNSPECIFIED' THEN 'DxUnspecifiedWhenSpecificAvailable'
        WHEN 'DX_MISSING_SPECIFICITY' THEN 'DxMissingSpecificity'
        WHEN 'DX_NOT_SUPPORT' THEN 'DxDoesNotSupportProcedure'
        WHEN 'DX_CHRONOLOGY' THEN 'DxChronologyIssue'
        WHEN 'DX_LATERALITY' THEN 'DxLateralityMismatch'
        WHEN 'DX_MANIFESTATION' THEN 'DxManifestationFirst'

        -- Documentation (DOC) issues
        WHEN 'DOC_INSUFFICIENT' THEN 'DocInsufficientDocumentation'
        WHEN 'DOC_MISSING_REQUIRED' THEN 'DocMissingRequiredElements'
        WHEN 'DOC_INCONSISTENT' THEN 'DocInconsistentDocumentation'
        WHEN 'DOC_MISSING_TIME' THEN 'DocMissingTimeElements'
        WHEN 'DOC_SIGNATURE' THEN 'DocSignatureIssue'

        -- Medical Necessity (MED) issues
        WHEN 'MED_NOT_REASONABLE' THEN 'MedNotReasonableOrNecessary'
        WHEN 'MED_EXPERIMENTAL' THEN 'MedExperimentalInvestigational'
        WHEN 'MED_FREQUENCY' THEN 'MedFrequencyLimitation'
        WHEN 'MED_COVERAGE' THEN 'MedCoverageRequirements'

        -- Place of Service (POS) issues
        WHEN 'POS_INCONSISTENT' THEN 'PosInconsistentWithProcedure'
        WHEN 'POS_NOT_COVERED' THEN 'PosNotCoveredForService'

        -- Provider (PRV) issues
        WHEN 'PRV_CREDENTIALS' THEN 'PrvCredentialsQualification'
        WHEN 'PRV_SPECIALTY' THEN 'PrvSpecialtyMismatch'
        WHEN 'PRV_SUPERVISION' THEN 'PrvSupervisionRequirement'

        -- Other (OTH) issues
        WHEN 'OTH_DUPLICATE_SERVICE' THEN 'OthDuplicateService'
        WHEN 'OTH_TIMELY_FILING' THEN 'OthTimelyFiling'
        WHEN 'OTH_COORDINATION_BENEFITS' THEN 'OthCoordinationOfBenefits'
        WHEN 'OTH_PRIOR_AUTH' THEN 'OthPriorAuthorizationRequired'

        -- Default fallback
        ELSE 'OthDuplicateService'
    END;
END;
$$;

COMMENT ON FUNCTION claims.get_flag_issue_type_name(TEXT) IS
'Maps database issue_code to Rust FlagIssueType enum name for rule loading';

-- Helper view: Rule definitions with FlagIssueType enum names
-- This view is used by the rule loader to get the proper enum names
CREATE OR REPLACE VIEW claims.v_rule_definitions_with_flag_types AS
SELECT
    rd.rule_definition_id,
    rd.rule_code,
    rd.rule_name,
    rd.template_id,
    rt.template_code,
    rt.template_name,
    rd.flag_issue_id,
    fi.issue_code,
    claims.get_flag_issue_type_name(fi.issue_code) AS flag_issue_type_name,
    rd.execution_order,
    rd.execution_level,
    rd.is_active,
    rd.created_at,
    rd.updated_at,
    rd.rule_parameters_encrypted,
    rd.rule_parameters_hash
FROM claims.rule_definition rd
INNER JOIN claims.flag_issue fi ON rd.flag_issue_id = fi.issue_id
LEFT JOIN claims.rule_template rt ON rd.template_id = rt.template_id
WHERE rd.is_active = true
ORDER BY rd.execution_order;

COMMENT ON VIEW claims.v_rule_definitions_with_flag_types IS
'Rule definitions with FlagIssueType enum names for easy loading in Rust';

-- Helper view: Active facility rules with flag type names
CREATE OR REPLACE VIEW claims.v_active_facility_rules_with_types AS
SELECT
    fa.facility_rule_id,
    fa.facility_id,
    fa.rule_definition_id,
    rd.rule_code,
    rd.rule_name,
    rt.template_code,
    fi.issue_code,
    claims.get_flag_issue_type_name(fi.issue_code) AS flag_issue_type_name,
    rd.execution_order,
    rd.execution_level,
    fa.parameter_overrides_encrypted,
    fa.is_active AS facility_override_active,
    rd.is_active AS rule_active
FROM claims.facility_rule_assignment fa
INNER JOIN claims.rule_definition rd ON fa.rule_definition_id = rd.rule_definition_id
INNER JOIN claims.flag_issue fi ON rd.flag_issue_id = fi.issue_id
LEFT JOIN claims.rule_template rt ON rd.template_id = rt.template_id
WHERE fa.is_active = true
  AND rd.is_active = true
ORDER BY fa.facility_id, rd.execution_order;

COMMENT ON VIEW claims.v_active_facility_rules_with_types IS
'Active facility rule assignments with FlagIssueType enum names';

-- Add index for faster lookups
CREATE INDEX IF NOT EXISTS idx_rule_definition_active ON claims.rule_definition(is_active) WHERE is_active = true;
CREATE INDEX IF NOT EXISTS idx_facility_rule_active ON claims.facility_rule_assignment(is_active, facility_id) WHERE is_active = true;

-- Grant permissions
GRANT EXECUTE ON FUNCTION claims.get_flag_issue_type_name(TEXT) TO pro_app;
GRANT SELECT ON claims.v_rule_definitions_with_flag_types TO pro_app;
GRANT SELECT ON claims.v_active_facility_rules_with_types TO pro_app;
