-- Migration: 014_create_utility_functions
-- Description: Create utility functions and additional indexes for performance
-- Date: 2025-10-14

-- ====================================================================================
-- UTILITY FUNCTIONS
-- ====================================================================================

-- Function to calculate age from date of birth
CREATE OR REPLACE FUNCTION calculate_age(birth_date DATE, as_of_date DATE DEFAULT CURRENT_DATE)
RETURNS INTEGER AS $$
BEGIN
    RETURN EXTRACT(YEAR FROM AGE(as_of_date, birth_date));
END;
$$ LANGUAGE plpgsql IMMUTABLE;

COMMENT ON FUNCTION calculate_age IS 'Calculate age from date of birth';

-- Function to validate NPI format (10 digits)
CREATE OR REPLACE FUNCTION is_valid_npi(npi_value VARCHAR)
RETURNS BOOLEAN AS $$
BEGIN
    RETURN npi_value ~ '^\d{10}$';
END;
$$ LANGUAGE plpgsql IMMUTABLE;

COMMENT ON FUNCTION is_valid_npi IS 'Validate NPI format (10 digits)';

-- Function to validate ICD-10-CM format
CREATE OR REPLACE FUNCTION is_valid_icd10(code VARCHAR)
RETURNS BOOLEAN AS $$
BEGIN
    -- ICD-10-CM: Letter followed by 2-6 alphanumeric characters
    RETURN code ~ '^[A-TV-Z][0-9][0-9A-TV-Z](\.[0-9A-TV-Z]{1,4})?$';
END;
$$ LANGUAGE plpgsql IMMUTABLE;

COMMENT ON FUNCTION is_valid_icd10 IS 'Validate ICD-10-CM code format';

-- Function to validate CPT/HCPCS format
CREATE OR REPLACE FUNCTION is_valid_procedure_code(code VARCHAR)
RETURNS BOOLEAN AS $$
BEGIN
    -- CPT: 5 digits, HCPCS: Letter + 4 digits
    RETURN code ~ '^\d{5}$' OR code ~ '^[A-Z]\d{4}$';
END;
$$ LANGUAGE plpgsql IMMUTABLE;

COMMENT ON FUNCTION is_valid_procedure_code IS 'Validate CPT/HCPCS procedure code format';

-- Function to calculate Medicare payment from RVU
CREATE OR REPLACE FUNCTION calculate_medicare_payment(
    work_rvu NUMERIC,
    pe_rvu NUMERIC,
    mp_rvu NUMERIC,
    work_gpci NUMERIC DEFAULT 1.0,
    pe_gpci NUMERIC DEFAULT 1.0,
    mp_gpci NUMERIC DEFAULT 1.0,
    conversion_factor NUMERIC DEFAULT 33.2875,
    modifier_adjustment NUMERIC DEFAULT 1.0,
    units NUMERIC DEFAULT 1.0
)
RETURNS NUMERIC AS $$
BEGIN
    RETURN ROUND(
        ((work_rvu * work_gpci) + (pe_rvu * pe_gpci) + (mp_rvu * mp_gpci)) *
        conversion_factor *
        modifier_adjustment *
        units,
        2
    );
END;
$$ LANGUAGE plpgsql IMMUTABLE;

COMMENT ON FUNCTION calculate_medicare_payment IS 'Calculate Medicare payment from RVU components';

-- Function to get E/M level from CPT code
CREATE OR REPLACE FUNCTION get_em_level(cpt_code VARCHAR)
RETURNS INTEGER AS $$
BEGIN
    -- Extract last digit from E/M codes (99XXX format)
    IF cpt_code ~ '^99[0-9]{3}$' THEN
        RETURN CAST(SUBSTRING(cpt_code FROM 5 FOR 1) AS INTEGER);
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

COMMENT ON FUNCTION get_em_level IS 'Extract E/M level from CPT code';

-- Function to determine if code is an E/M code
CREATE OR REPLACE FUNCTION is_em_code(cpt_code VARCHAR)
RETURNS BOOLEAN AS $$
BEGIN
    RETURN cpt_code ~ '^99[0-9]{3}$';
END;
$$ LANGUAGE plpgsql IMMUTABLE;

COMMENT ON FUNCTION is_em_code IS 'Determine if CPT code is an E/M code';

-- Function to get business days between two dates
CREATE OR REPLACE FUNCTION business_days_between(start_date DATE, end_date DATE)
RETURNS INTEGER AS $$
DECLARE
    total_days INTEGER;
    full_weeks INTEGER;
    extra_days INTEGER;
    start_day INTEGER;
    end_day INTEGER;
BEGIN
    IF start_date > end_date THEN
        RETURN -business_days_between(end_date, start_date);
    END IF;

    total_days := end_date - start_date;
    full_weeks := total_days / 7;
    extra_days := total_days % 7;

    start_day := EXTRACT(DOW FROM start_date);
    end_day := EXTRACT(DOW FROM end_date);

    -- Calculate business days
    RETURN (full_weeks * 5) +
           CASE
               WHEN extra_days = 0 THEN 0
               WHEN start_day = 0 THEN extra_days - 1  -- Start on Sunday
               WHEN start_day = 6 THEN extra_days - 1  -- Start on Saturday
               WHEN start_day + extra_days > 6 THEN extra_days - 2  -- Crosses weekend
               WHEN start_day + extra_days = 6 THEN extra_days - 1  -- Ends on Saturday
               ELSE extra_days
           END;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

COMMENT ON FUNCTION business_days_between IS 'Calculate business days between two dates';

-- Function to mask sensitive data (for display purposes)
CREATE OR REPLACE FUNCTION mask_mbi(mbi VARCHAR)
RETURNS VARCHAR AS $$
BEGIN
    IF LENGTH(mbi) < 4 THEN
        RETURN '***';
    END IF;
    RETURN '***-**-' || RIGHT(mbi, 4);
END;
$$ LANGUAGE plpgsql IMMUTABLE;

COMMENT ON FUNCTION mask_mbi IS 'Mask Medicare Beneficiary Identifier for display';

-- Function to get fiscal year from date
CREATE OR REPLACE FUNCTION get_fiscal_year(date_value DATE, fiscal_year_start_month INTEGER DEFAULT 10)
RETURNS INTEGER AS $$
BEGIN
    IF EXTRACT(MONTH FROM date_value) >= fiscal_year_start_month THEN
        RETURN EXTRACT(YEAR FROM date_value) + 1;
    ELSE
        RETURN EXTRACT(YEAR FROM date_value);
    END IF;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

COMMENT ON FUNCTION get_fiscal_year IS 'Get fiscal year from date (default starts October)';

-- Function to clean and standardize phone numbers
CREATE OR REPLACE FUNCTION standardize_phone(phone VARCHAR)
RETURNS VARCHAR AS $$
BEGIN
    -- Remove all non-numeric characters
    phone := REGEXP_REPLACE(phone, '[^0-9]', '', 'g');

    -- Format as (XXX) XXX-XXXX if 10 digits
    IF LENGTH(phone) = 10 THEN
        RETURN '(' || SUBSTRING(phone FROM 1 FOR 3) || ') ' ||
               SUBSTRING(phone FROM 4 FOR 3) || '-' ||
               SUBSTRING(phone FROM 7 FOR 4);
    END IF;

    RETURN phone;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

COMMENT ON FUNCTION standardize_phone IS 'Standardize phone number format';

-- Function to validate date of service is not in future
CREATE OR REPLACE FUNCTION validate_dos()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.date_of_service_from > CURRENT_DATE THEN
        RAISE EXCEPTION 'Date of service cannot be in the future';
    END IF;

    IF NEW.date_of_service_to IS NOT NULL AND NEW.date_of_service_to > CURRENT_DATE THEN
        RAISE EXCEPTION 'Date of service to cannot be in the future';
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION validate_dos IS 'Validate date of service is not in future';

-- Apply DOS validation trigger to encounter table
CREATE TRIGGER validate_encounter_dos
    BEFORE INSERT OR UPDATE ON claims.encounter
    FOR EACH ROW
    EXECUTE FUNCTION validate_dos();

-- Function to automatically calculate encounter totals from service lines
CREATE OR REPLACE FUNCTION update_encounter_totals()
RETURNS TRIGGER AS $$
DECLARE
    encounter_total NUMERIC(18,2);
BEGIN
    -- Calculate total from all service lines for this encounter
    SELECT COALESCE(SUM(line_item_charge_amount), 0)
    INTO encounter_total
    FROM claims.service_line
    WHERE encounter_id = COALESCE(NEW.encounter_id, OLD.encounter_id);

    -- Update encounter total
    UPDATE claims.encounter
    SET total_claim_charge_amount = encounter_total,
        updated_at = CURRENT_TIMESTAMP
    WHERE encounter_id = COALESCE(NEW.encounter_id, OLD.encounter_id);

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION update_encounter_totals IS 'Automatically recalculate encounter totals when service lines change';

-- Apply trigger to keep encounter totals in sync
CREATE TRIGGER sync_encounter_totals_insert
    AFTER INSERT ON claims.service_line
    FOR EACH ROW
    EXECUTE FUNCTION update_encounter_totals();

CREATE TRIGGER sync_encounter_totals_update
    AFTER UPDATE ON claims.service_line
    FOR EACH ROW
    WHEN (OLD.line_item_charge_amount IS DISTINCT FROM NEW.line_item_charge_amount)
    EXECUTE FUNCTION update_encounter_totals();

CREATE TRIGGER sync_encounter_totals_delete
    AFTER DELETE ON claims.service_line
    FOR EACH ROW
    EXECUTE FUNCTION update_encounter_totals();

-- ====================================================================================
-- ADDITIONAL PERFORMANCE INDEXES
-- ====================================================================================

-- Full-text search indexes for common search fields
CREATE INDEX idx_encounter_patient_control_trgm ON claims.encounter
    USING gin (patient_control_number gin_trgm_ops);

CREATE INDEX idx_encounter_subscriber_last_name_trgm ON claims.encounter
    USING gin (subscriber_last_name gin_trgm_ops);

CREATE INDEX idx_provider_last_name_trgm ON claims.provider
    USING gin (last_name gin_trgm_ops);

CREATE INDEX idx_facility_name_trgm ON claims.facility
    USING gin (facility_name gin_trgm_ops);

-- Additional composite indexes for common query patterns
CREATE INDEX idx_encounter_org_facility_dos ON claims.encounter(organization_id, facility_id, date_of_service_from);

CREATE INDEX idx_encounter_provider_status_dos ON claims.encounter(rendering_provider_id, claim_status, date_of_service_from)
    WHERE claim_status IN ('PENDING', 'FLAGGED', 'NEW');

CREATE INDEX idx_service_line_proc_date_facility ON claims.service_line(procedure_code, service_date_from, encounter_id);

CREATE INDEX idx_flag_org_severity_status ON claims.encounter_flag(severity, flag_status, created_at)
    WHERE flag_status = 'OPEN';

CREATE INDEX idx_denial_org_preventable_date ON claims.denial_event(organization_id, is_preventable, denial_date)
    WHERE is_preventable = true;

-- Indexes for audit queries
CREATE INDEX idx_audit_encounter_audit_status ON claims.audit_encounter(audit_id, review_status, has_errors);

CREATE INDEX idx_service_line_eval_result_severity ON claims.service_line_evaluation(evaluation_result, issue_severity)
    WHERE has_error = true;

-- ====================================================================================
-- MATERIALIZED VIEWS FOR PERFORMANCE
-- ====================================================================================

-- Materialized view for flag statistics (refresh daily)
CREATE MATERIALIZED VIEW claims.mv_flag_statistics AS
SELECT
    e.organization_id,
    e.facility_id,
    DATE_TRUNC('day', ef.created_at) AS flag_date,
    fc.category_code,
    ef.severity,
    COUNT(DISTINCT ef.flag_id) AS flag_count,
    COUNT(DISTINCT ef.encounter_id) AS affected_encounters
FROM claims.encounter_flag ef
INNER JOIN claims.flag_issue fi ON ef.issue_id = fi.issue_id
INNER JOIN claims.flag_category fc ON fi.category_id = fc.category_id
INNER JOIN claims.encounter e ON ef.encounter_id = e.encounter_id
WHERE e.is_active = true
GROUP BY e.organization_id, e.facility_id, DATE_TRUNC('day', ef.created_at), fc.category_code, ef.severity;

CREATE UNIQUE INDEX idx_mv_flag_stats_unique ON claims.mv_flag_statistics(organization_id, facility_id, flag_date, category_code, severity);
CREATE INDEX idx_mv_flag_stats_date ON claims.mv_flag_statistics(flag_date);

COMMENT ON MATERIALIZED VIEW claims.mv_flag_statistics IS 'Pre-aggregated flag statistics for dashboard performance';

-- Materialized view for denial statistics (refresh daily)
CREATE MATERIALIZED VIEW claims.mv_denial_statistics AS
SELECT
    de.organization_id,
    de.facility_id,
    DATE_TRUNC('day', de.denial_date) AS denial_date,
    de.payer_id,
    de.root_cause_category,
    COUNT(DISTINCT de.denial_id) AS denial_count,
    SUM(de.denied_amount) AS total_denied_amount,
    COUNT(DISTINCT CASE WHEN de.is_preventable THEN de.denial_id END) AS preventable_count,
    COUNT(DISTINCT CASE WHEN de.appeal_filed THEN de.denial_id END) AS appeal_count
FROM claims.denial_event de
GROUP BY de.organization_id, de.facility_id, DATE_TRUNC('day', de.denial_date), de.payer_id, de.root_cause_category;

CREATE UNIQUE INDEX idx_mv_denial_stats_unique ON claims.mv_denial_statistics(organization_id, facility_id, denial_date, payer_id, root_cause_category);
CREATE INDEX idx_mv_denial_stats_date ON claims.mv_denial_statistics(denial_date);

COMMENT ON MATERIALIZED VIEW claims.mv_denial_statistics IS 'Pre-aggregated denial statistics for dashboard performance';

-- ====================================================================================
-- HELPER FUNCTIONS FOR COMMON QUERIES
-- ====================================================================================

-- Function to get open flags for an encounter
CREATE OR REPLACE FUNCTION get_open_flags(p_encounter_id UUID)
RETURNS TABLE(
    flag_id UUID,
    issue_code VARCHAR,
    issue_description TEXT,
    severity VARCHAR,
    flag_reason TEXT,
    created_at TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        ef.flag_id,
        fi.issue_code,
        fi.issue_description,
        ef.severity,
        ef.flag_reason,
        ef.created_at
    FROM claims.encounter_flag ef
    INNER JOIN claims.flag_issue fi ON ef.issue_id = fi.issue_id
    WHERE ef.encounter_id = p_encounter_id
        AND ef.flag_status = 'OPEN'
    ORDER BY ef.severity DESC, ef.created_at DESC;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION get_open_flags IS 'Get all open flags for an encounter';

-- Function to get coder accuracy summary
CREATE OR REPLACE FUNCTION get_coder_accuracy_summary(
    p_coder_id UUID,
    p_start_date DATE,
    p_end_date DATE
)
RETURNS TABLE(
    encounters_coded INTEGER,
    encounters_audited INTEGER,
    encounters_with_errors INTEGER,
    accuracy_rate NUMERIC,
    total_errors INTEGER,
    high_severity_errors INTEGER,
    medium_severity_errors INTEGER,
    low_severity_errors INTEGER
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        COUNT(DISTINCT e.encounter_id)::INTEGER AS encounters_coded,
        COUNT(DISTINCT ae.audit_encounter_id)::INTEGER AS encounters_audited,
        COUNT(DISTINCT CASE WHEN ae.has_errors THEN ae.audit_encounter_id END)::INTEGER AS encounters_with_errors,
        ROUND(100.0 * (1 - COUNT(DISTINCT CASE WHEN ae.has_errors THEN ae.audit_encounter_id END)::NUMERIC /
            NULLIF(COUNT(DISTINCT ae.audit_encounter_id), 0)), 2) AS accuracy_rate,
        SUM(COALESCE(ae.error_count, 0))::INTEGER AS total_errors,
        SUM(COALESCE(ae.severity_high_count, 0))::INTEGER AS high_severity_errors,
        SUM(COALESCE(ae.severity_medium_count, 0))::INTEGER AS medium_severity_errors,
        SUM(COALESCE(ae.severity_low_count, 0))::INTEGER AS low_severity_errors
    FROM claims.encounter e
    LEFT JOIN claims.audit_encounter ae ON e.encounter_id = ae.encounter_id
    WHERE e.coder_id = p_coder_id
        AND e.coding_date BETWEEN p_start_date AND p_end_date;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION get_coder_accuracy_summary IS 'Get accuracy summary for a coder over a date range';

-- Grant permissions on all schemas to application role
-- Note: Actual role creation and permission grants should be done separately based on deployment
COMMENT ON SCHEMA staging IS 'Staging schema for import, configuration, and processing';
COMMENT ON SCHEMA claims IS 'Main schema for claims data, flags, audits, and denials';
COMMENT ON SCHEMA ml IS 'Machine learning schema for predictive models and features';
