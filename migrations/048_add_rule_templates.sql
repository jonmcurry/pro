-- Migration: Add Rule Templates (Phase 3)
-- Description: Add template definitions and example template-based rules
-- Date: 2025-11-05

-- Insert template definitions
DO $$
DECLARE
    v_threshold_template_id BIGINT;
    v_duplicate_template_id BIGINT;
    v_missing_field_template_id BIGINT;
    v_field_pattern_template_id BIGINT;
    v_cross_field_template_id BIGINT;
BEGIN
    -- 1. THRESHOLD Template
    INSERT INTO claims.rule_template (
        template_code,
        template_name,
        rust_struct_name,
        parameter_schema,
        execution_level,
        is_active
    ) VALUES (
        'THRESHOLD',
        'Threshold Comparison Rule',
        'ThresholdRule',
        jsonb_build_object(
            'parameters', jsonb_build_array(
                jsonb_build_object(
                    'name', 'field',
                    'type', 'string',
                    'required', true,
                    'description', 'Field to check (e.g., total_charge, units)',
                    'enum', jsonb_build_array('total_charge', 'units', 'allowed_amount', 'patient_responsibility')
                ),
                jsonb_build_object(
                    'name', 'operator',
                    'type', 'string',
                    'required', true,
                    'description', 'Comparison operator',
                    'enum', jsonb_build_array('>', '>=', '<', '<=', '==', '!=')
                ),
                jsonb_build_object(
                    'name', 'threshold',
                    'type', 'number',
                    'required', true,
                    'description', 'Threshold value to compare against'
                ),
                jsonb_build_object(
                    'name', 'min_threshold',
                    'type', 'number',
                    'required', false,
                    'description', 'Optional minimum threshold for range checks'
                ),
                jsonb_build_object(
                    'name', 'max_threshold',
                    'type', 'number',
                    'required', false,
                    'description', 'Optional maximum threshold for range checks'
                )
            )
        ),
        'BOTH',
        true
    ) RETURNING template_id INTO v_threshold_template_id;

    -- 2. DUPLICATE Template
    INSERT INTO claims.rule_template (
        template_code,
        template_name,
        rust_struct_name,
        parameter_schema,
        execution_level,
        is_active
    ) VALUES (
        'DUPLICATE',
        'Duplicate Detection Rule',
        'DuplicateRule',
        jsonb_build_object(
            'parameters', jsonb_build_array(
                jsonb_build_object(
                    'name', 'table',
                    'type', 'string',
                    'required', true,
                    'description', 'Table to check for duplicates',
                    'enum', jsonb_build_array('service_line', 'encounter', 'provider')
                ),
                jsonb_build_object(
                    'name', 'match_fields',
                    'type', 'array',
                    'required', true,
                    'description', 'Fields that must match to be considered duplicate'
                ),
                jsonb_build_object(
                    'name', 'scope',
                    'type', 'string',
                    'required', false,
                    'default', 'encounter',
                    'description', 'Scope of duplicate check',
                    'enum', jsonb_build_array('encounter', 'patient', 'facility')
                ),
                jsonb_build_object(
                    'name', 'time_window_days',
                    'type', 'number',
                    'required', false,
                    'default', 0,
                    'description', 'Number of days to look back for duplicates'
                ),
                jsonb_build_object(
                    'name', 'case_sensitive',
                    'type', 'boolean',
                    'required', false,
                    'default', false,
                    'description', 'Whether string comparisons should be case-sensitive'
                )
            )
        ),
        'BOTH',
        true
    ) RETURNING template_id INTO v_duplicate_template_id;

    -- 3. MISSING_FIELD Template
    INSERT INTO claims.rule_template (
        template_code,
        template_name,
        rust_struct_name,
        parameter_schema,
        execution_level,
        is_active
    ) VALUES (
        'MISSING_FIELD',
        'Missing Required Field Rule',
        'MissingFieldRule',
        jsonb_build_object(
            'parameters', jsonb_build_array(
                jsonb_build_object(
                    'name', 'fields',
                    'type', 'array',
                    'required', true,
                    'description', 'List of required fields to check'
                ),
                jsonb_build_object(
                    'name', 'check_empty',
                    'type', 'boolean',
                    'required', false,
                    'default', true,
                    'description', 'Also flag empty strings (not just NULL)'
                ),
                jsonb_build_object(
                    'name', 'execution_level',
                    'type', 'string',
                    'required', false,
                    'default', 'service_line',
                    'description', 'Where to check',
                    'enum', jsonb_build_array('service_line', 'encounter')
                )
            )
        ),
        'BOTH',
        true
    ) RETURNING template_id INTO v_missing_field_template_id;

    -- 4. FIELD_PATTERN Template
    INSERT INTO claims.rule_template (
        template_code,
        template_name,
        rust_struct_name,
        parameter_schema,
        execution_level,
        is_active
    ) VALUES (
        'FIELD_PATTERN',
        'Field Pattern Validation Rule',
        'FieldPatternRule',
        jsonb_build_object(
            'parameters', jsonb_build_array(
                jsonb_build_object(
                    'name', 'field',
                    'type', 'string',
                    'required', true,
                    'description', 'Field to validate'
                ),
                jsonb_build_object(
                    'name', 'pattern',
                    'type', 'string',
                    'required', true,
                    'description', 'Regex pattern to match'
                ),
                jsonb_build_object(
                    'name', 'invert_match',
                    'type', 'boolean',
                    'required', false,
                    'default', false,
                    'description', 'If true, flag when pattern DOES match'
                ),
                jsonb_build_object(
                    'name', 'case_sensitive',
                    'type', 'boolean',
                    'required', false,
                    'default', true,
                    'description', 'Whether regex should be case-sensitive'
                ),
                jsonb_build_object(
                    'name', 'allow_null',
                    'type', 'boolean',
                    'required', false,
                    'default', true,
                    'description', 'If true, NULL/empty values pass validation'
                ),
                jsonb_build_object(
                    'name', 'execution_level',
                    'type', 'string',
                    'required', false,
                    'default', 'service_line',
                    'enum', jsonb_build_array('service_line', 'encounter')
                )
            )
        ),
        'BOTH',
        true
    ) RETURNING template_id INTO v_field_pattern_template_id;

    -- 5. CROSS_FIELD Template
    INSERT INTO claims.rule_template (
        template_code,
        template_name,
        rust_struct_name,
        parameter_schema,
        execution_level,
        is_active
    ) VALUES (
        'CROSS_FIELD',
        'Cross-Field Comparison Rule',
        'CrossFieldRule',
        jsonb_build_object(
            'parameters', jsonb_build_array(
                jsonb_build_object(
                    'name', 'field1',
                    'type', 'string',
                    'required', true,
                    'description', 'First field to compare'
                ),
                jsonb_build_object(
                    'name', 'operator',
                    'type', 'string',
                    'required', true,
                    'description', 'Comparison operator',
                    'enum', jsonb_build_array('>', '>=', '<', '<=', '==', '!=')
                ),
                jsonb_build_object(
                    'name', 'field2',
                    'type', 'string',
                    'required', true,
                    'description', 'Second field to compare'
                ),
                jsonb_build_object(
                    'name', 'execution_level',
                    'type', 'string',
                    'required', false,
                    'default', 'service_line',
                    'enum', jsonb_build_array('service_line', 'encounter')
                )
            )
        ),
        'BOTH',
        true
    ) RETURNING template_id INTO v_cross_field_template_id;

    -- Example template-based rules

    -- Example 1: High charge threshold rule
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
        'HIGH_CHARGE_THRESHOLD',
        'Flag charges over $10,000',
        v_threshold_template_id,
        pgp_sym_encrypt(
            '{"field": "total_charge", "operator": ">", "threshold": 10000}'::text,
            (SELECT current_setting('app.rule_encryption_key', true))
        ),
        (SELECT issue_id FROM claims.flag_issue WHERE issue_code = 'HIGH_CHARGE'),
        70,
        'SERVICE_LINE',
        false  -- Disabled by default, enable per facility as needed
    );

    -- Example 2: CPT code format validation
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
        'CPT_CODE_FORMAT',
        'Validate CPT code is 5 digits',
        v_field_pattern_template_id,
        pgp_sym_encrypt(
            '{"field": "procedure_code", "pattern": "^\\d{5}$", "allow_null": false}'::text,
            (SELECT current_setting('app.rule_encryption_key', true))
        ),
        (SELECT issue_id FROM claims.flag_issue WHERE issue_code = 'INVALID_CODE_FORMAT'),
        80,
        'SERVICE_LINE',
        false
    );

    -- Example 3: Service date range validation
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
        'INVALID_DATE_RANGE',
        'Flag when service_to_date < service_from_date',
        v_cross_field_template_id,
        pgp_sym_encrypt(
            '{"field1": "service_to_date", "operator": "<", "field2": "service_from_date"}'::text,
            (SELECT current_setting('app.rule_encryption_key', true))
        ),
        (SELECT issue_id FROM claims.flag_issue WHERE issue_code = 'INVALID_DATE'),
        90,
        'SERVICE_LINE',
        false
    );

    -- Example 4: Missing required billing fields
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
        'MISSING_BILLING_FIELDS',
        'Check for missing required billing fields',
        v_missing_field_template_id,
        pgp_sym_encrypt(
            '{"fields": ["billing_provider_id", "date_of_service", "total_charge"], "execution_level": "encounter"}'::text,
            (SELECT current_setting('app.rule_encryption_key', true))
        ),
        (SELECT issue_id FROM claims.flag_issue WHERE issue_code = 'MISSING_REQUIRED_FIELD'),
        100,
        'ENCOUNTER',
        false
    );

    -- Example 5: Duplicate service line detection (template-based)
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
        'DUPLICATE_SERVICE_TEMPLATE',
        'Detect duplicate service lines (configurable)',
        v_duplicate_template_id,
        pgp_sym_encrypt(
            '{"table": "service_line", "match_fields": ["procedure_code", "service_from_date", "units"], "scope": "encounter"}'::text,
            (SELECT current_setting('app.rule_encryption_key', true))
        ),
        (SELECT issue_id FROM claims.flag_issue WHERE issue_code = 'DUPLICATE_SERVICE'),
        110,
        'SERVICE_LINE',
        false
    );

    RAISE NOTICE 'Phase 3 complete: Added 5 rule templates and 5 example rules';
END $$;

-- Add comment for documentation
COMMENT ON TABLE claims.rule_template IS 'Phase 3: Rule templates define parameterized rule patterns that can be instantiated with different configurations';
COMMENT ON COLUMN claims.rule_template.parameter_schema IS 'JSON schema defining template parameters, validation rules, and default values';
COMMENT ON COLUMN claims.rule_definition.rule_parameters_encrypted IS 'Encrypted JSON parameters that configure the rule template instance';
