-- Migration: 008_create_audit_tables
-- Description: Create audit and evaluation tables for retrospective reviews
-- Date: 2025-10-14

-- Audit assignments (post-bill retrospective audits)
CREATE TABLE claims.audit_assignment (
    audit_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID NOT NULL REFERENCES claims.organization(organization_id),
    facility_id UUID REFERENCES claims.facility(facility_id),

    -- Assignment details
    audit_name VARCHAR(255) NOT NULL,
    audit_type VARCHAR(50) NOT NULL, -- RANDOM, TARGETED, PROVIDER_SPECIFIC, PROCEDURE_SPECIFIC
    audit_scope VARCHAR(50) NOT NULL, -- FULL, SAMPLE, FOCUSED

    -- Selection criteria
    selection_criteria JSONB, -- Criteria used to select encounters
    total_population INTEGER, -- Total encounters matching criteria
    sample_size INTEGER, -- How many to audit
    sampling_method VARCHAR(50), -- RANDOM, STRATIFIED, SYSTEMATIC

    -- Date ranges
    dos_from DATE,
    dos_to DATE,

    -- Assigned to
    reviewer_id UUID REFERENCES claims.reviewer(reviewer_id),
    assigned_at TIMESTAMPTZ,
    due_date DATE,

    -- Status
    audit_status VARCHAR(50) DEFAULT 'ASSIGNED', -- ASSIGNED, IN_PROGRESS, COMPLETED, CANCELLED
    completed_at TIMESTAMPTZ,
    completion_percentage NUMERIC(5,2) DEFAULT 0.00,

    -- Results summary (populated after completion)
    encounters_reviewed INTEGER DEFAULT 0,
    encounters_with_errors INTEGER DEFAULT 0,
    total_flags_found INTEGER DEFAULT 0,
    error_rate NUMERIC(5,2),

    -- Financial impact
    total_billed_amount NUMERIC(18,2),
    total_overpayment_amount NUMERIC(18,2),
    total_underpayment_amount NUMERIC(18,2),
    net_financial_impact NUMERIC(18,2),

    -- Audit trail
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100),
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_by VARCHAR(100)
);

CREATE INDEX idx_audit_assignment_org ON claims.audit_assignment(organization_id);
CREATE INDEX idx_audit_assignment_facility ON claims.audit_assignment(facility_id);
CREATE INDEX idx_audit_assignment_reviewer ON claims.audit_assignment(reviewer_id);
CREATE INDEX idx_audit_assignment_status ON claims.audit_assignment(audit_status);
CREATE INDEX idx_audit_assignment_due_date ON claims.audit_assignment(due_date);
CREATE INDEX idx_audit_assignment_dos_range ON claims.audit_assignment(dos_from, dos_to);

COMMENT ON TABLE claims.audit_assignment IS 'Audit assignments for retrospective claim reviews';

CREATE TRIGGER update_audit_assignment_updated_at BEFORE UPDATE ON claims.audit_assignment
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Encounters included in audits
CREATE TABLE claims.audit_encounter (
    audit_encounter_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    audit_id UUID NOT NULL REFERENCES claims.audit_assignment(audit_id) ON DELETE CASCADE,
    encounter_id UUID NOT NULL REFERENCES claims.encounter(encounter_id) ON DELETE CASCADE,

    -- Review details
    review_status VARCHAR(50) DEFAULT 'PENDING', -- PENDING, IN_REVIEW, COMPLETED, SKIPPED
    reviewed_at TIMESTAMPTZ,
    review_duration_minutes INTEGER,

    -- Results
    has_errors BOOLEAN DEFAULT false,
    error_count INTEGER DEFAULT 0,
    severity_high_count INTEGER DEFAULT 0,
    severity_medium_count INTEGER DEFAULT 0,
    severity_low_count INTEGER DEFAULT 0,

    -- Financial impact (at encounter level)
    original_billed_amount NUMERIC(18,2),
    corrected_billed_amount NUMERIC(18,2),
    overpayment_amount NUMERIC(18,2),
    underpayment_amount NUMERIC(18,2),
    net_financial_impact NUMERIC(18,2),

    -- Reviewer notes
    reviewer_notes TEXT,

    -- Audit trail
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(audit_id, encounter_id)
);

CREATE INDEX idx_audit_encounter_audit ON claims.audit_encounter(audit_id);
CREATE INDEX idx_audit_encounter_encounter ON claims.audit_encounter(encounter_id);
CREATE INDEX idx_audit_encounter_status ON claims.audit_encounter(review_status);
CREATE INDEX idx_audit_encounter_has_errors ON claims.audit_encounter(has_errors) WHERE has_errors = true;

COMMENT ON TABLE claims.audit_encounter IS 'Encounters selected for audit review';

-- Service line evaluations (detailed audit findings)
CREATE TABLE claims.service_line_evaluation (
    evaluation_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    audit_encounter_id UUID NOT NULL REFERENCES claims.audit_encounter(audit_encounter_id) ON DELETE CASCADE,
    service_line_id UUID NOT NULL REFERENCES claims.service_line(service_line_id) ON DELETE CASCADE,
    reviewer_id UUID NOT NULL REFERENCES claims.reviewer(reviewer_id),

    -- Original values
    original_procedure_code VARCHAR(48),
    original_modifier_1 VARCHAR(2),
    original_modifier_2 VARCHAR(2),
    original_modifier_3 VARCHAR(2),
    original_modifier_4 VARCHAR(2),
    original_units NUMERIC(15,1),
    original_charge_amount NUMERIC(18,2),

    -- Corrected values
    corrected_procedure_code VARCHAR(48),
    corrected_modifier_1 VARCHAR(2),
    corrected_modifier_2 VARCHAR(2),
    corrected_modifier_3 VARCHAR(2),
    corrected_modifier_4 VARCHAR(2),
    corrected_units NUMERIC(15,1),
    corrected_charge_amount NUMERIC(18,2),

    -- Evaluation result
    evaluation_result VARCHAR(50) NOT NULL, -- CORRECT, OVERCODED, UNDERCODED, INCORRECT, UNSUPPORTED, BUNDLED
    has_error BOOLEAN DEFAULT false,

    -- Issue details
    issue_id UUID REFERENCES claims.flag_issue(issue_id),
    issue_description TEXT,
    issue_severity VARCHAR(20),

    -- Financial impact
    reimbursement_impact NUMERIC(18,2),
    impact_type VARCHAR(20), -- OVERPAYMENT, UNDERPAYMENT, NEUTRAL

    -- Documentation review
    documentation_sufficient BOOLEAN,
    documentation_notes TEXT,

    -- Reviewer confidence
    confidence_level VARCHAR(20), -- HIGH, MEDIUM, LOW
    requires_second_review BOOLEAN DEFAULT false,

    -- Audit trail
    evaluated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT chk_evaluation_result CHECK (evaluation_result IN ('CORRECT', 'OVERCODED', 'UNDERCODED', 'INCORRECT', 'UNSUPPORTED', 'BUNDLED'))
);

CREATE INDEX idx_service_line_eval_audit_enc ON claims.service_line_evaluation(audit_encounter_id);
CREATE INDEX idx_service_line_eval_line ON claims.service_line_evaluation(service_line_id);
CREATE INDEX idx_service_line_eval_reviewer ON claims.service_line_evaluation(reviewer_id);
CREATE INDEX idx_service_line_eval_result ON claims.service_line_evaluation(evaluation_result);
CREATE INDEX idx_service_line_eval_has_error ON claims.service_line_evaluation(has_error) WHERE has_error = true;
CREATE INDEX idx_service_line_eval_issue ON claims.service_line_evaluation(issue_id);

COMMENT ON TABLE claims.service_line_evaluation IS 'Detailed audit findings for individual service lines';

-- Diagnosis evaluations
CREATE TABLE claims.diagnosis_evaluation (
    diagnosis_eval_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    audit_encounter_id UUID NOT NULL REFERENCES claims.audit_encounter(audit_encounter_id) ON DELETE CASCADE,
    encounter_diagnosis_id UUID REFERENCES claims.encounter_diagnosis(diagnosis_id) ON DELETE CASCADE,
    reviewer_id UUID NOT NULL REFERENCES claims.reviewer(reviewer_id),

    -- Original diagnosis
    original_diagnosis_code VARCHAR(30),
    original_sequence_number SMALLINT,
    original_is_principal BOOLEAN,

    -- Corrected diagnosis
    corrected_diagnosis_code VARCHAR(30),
    corrected_sequence_number SMALLINT,
    corrected_is_principal BOOLEAN,

    -- Evaluation result
    evaluation_result VARCHAR(50) NOT NULL, -- CORRECT, INCORRECT, UNSUPPORTED, MISSING, ADDITIONAL_NEEDED, SPECIFICITY_ISSUE
    has_error BOOLEAN DEFAULT false,

    -- Issue details
    issue_id UUID REFERENCES claims.flag_issue(issue_id),
    issue_description TEXT,
    issue_severity VARCHAR(20),

    -- Documentation review
    documentation_sufficient BOOLEAN,
    documentation_notes TEXT,

    -- HCC impact
    hcc_impact BOOLEAN DEFAULT false,
    hcc_category_affected VARCHAR(10),

    -- Audit trail
    evaluated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT chk_dx_evaluation_result CHECK (evaluation_result IN ('CORRECT', 'INCORRECT', 'UNSUPPORTED', 'MISSING', 'ADDITIONAL_NEEDED', 'SPECIFICITY_ISSUE'))
);

CREATE INDEX idx_diagnosis_eval_audit_enc ON claims.diagnosis_evaluation(audit_encounter_id);
CREATE INDEX idx_diagnosis_eval_diagnosis ON claims.diagnosis_evaluation(encounter_diagnosis_id);
CREATE INDEX idx_diagnosis_eval_reviewer ON claims.diagnosis_evaluation(reviewer_id);
CREATE INDEX idx_diagnosis_eval_result ON claims.diagnosis_evaluation(evaluation_result);
CREATE INDEX idx_diagnosis_eval_has_error ON claims.diagnosis_evaluation(has_error) WHERE has_error = true;
CREATE INDEX idx_diagnosis_eval_hcc ON claims.diagnosis_evaluation(hcc_impact) WHERE hcc_impact = true;

COMMENT ON TABLE claims.diagnosis_evaluation IS 'Audit findings for diagnosis codes';

-- Coder accuracy tracking
CREATE TABLE claims.coder_accuracy (
    accuracy_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    coder_id UUID NOT NULL REFERENCES claims.coder(coder_id),
    organization_id UUID NOT NULL REFERENCES claims.organization(organization_id),

    -- Period
    period_start_date DATE NOT NULL,
    period_end_date DATE NOT NULL,
    period_type VARCHAR(20) NOT NULL, -- DAILY, WEEKLY, MONTHLY, QUARTERLY, YEARLY

    -- Volume metrics
    encounters_coded INTEGER DEFAULT 0,
    service_lines_coded INTEGER DEFAULT 0,

    -- Accuracy metrics
    encounters_audited INTEGER DEFAULT 0,
    encounters_with_errors INTEGER DEFAULT 0,
    service_lines_audited INTEGER DEFAULT 0,
    service_lines_with_errors INTEGER DEFAULT 0,

    -- Accuracy rates
    encounter_accuracy_rate NUMERIC(5,2),
    service_line_accuracy_rate NUMERIC(5,2),
    overall_accuracy_rate NUMERIC(5,2),

    -- Error breakdown
    high_severity_errors INTEGER DEFAULT 0,
    medium_severity_errors INTEGER DEFAULT 0,
    low_severity_errors INTEGER DEFAULT 0,

    -- Error categories
    coding_errors INTEGER DEFAULT 0,
    documentation_errors INTEGER DEFAULT 0,
    em_errors INTEGER DEFAULT 0,
    modifier_errors INTEGER DEFAULT 0,
    diagnosis_errors INTEGER DEFAULT 0,
    other_errors INTEGER DEFAULT 0,

    -- Financial impact
    total_financial_impact NUMERIC(18,2),
    overpayment_total NUMERIC(18,2),
    underpayment_total NUMERIC(18,2),

    -- Productivity
    average_encounters_per_day NUMERIC(8,2),
    average_service_lines_per_encounter NUMERIC(8,2),

    -- Calculated at
    calculated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(coder_id, period_start_date, period_type)
);

CREATE INDEX idx_coder_accuracy_coder ON claims.coder_accuracy(coder_id);
CREATE INDEX idx_coder_accuracy_org ON claims.coder_accuracy(organization_id);
CREATE INDEX idx_coder_accuracy_period ON claims.coder_accuracy(period_start_date, period_end_date);
CREATE INDEX idx_coder_accuracy_rate ON claims.coder_accuracy(overall_accuracy_rate);

COMMENT ON TABLE claims.coder_accuracy IS 'Coder accuracy metrics over time periods';

-- Provider accuracy tracking
CREATE TABLE claims.provider_accuracy (
    accuracy_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    provider_id UUID NOT NULL REFERENCES claims.provider(provider_id),
    organization_id UUID NOT NULL REFERENCES claims.organization(organization_id),

    -- Period
    period_start_date DATE NOT NULL,
    period_end_date DATE NOT NULL,
    period_type VARCHAR(20) NOT NULL, -- MONTHLY, QUARTERLY, YEARLY

    -- Volume metrics
    encounters_billed INTEGER DEFAULT 0,
    service_lines_billed INTEGER DEFAULT 0,

    -- Accuracy metrics
    encounters_audited INTEGER DEFAULT 0,
    encounters_with_errors INTEGER DEFAULT 0,
    service_lines_audited INTEGER DEFAULT 0,
    service_lines_with_errors INTEGER DEFAULT 0,

    -- Accuracy rates
    encounter_accuracy_rate NUMERIC(5,2),
    service_line_accuracy_rate NUMERIC(5,2),
    overall_accuracy_rate NUMERIC(5,2),

    -- Error patterns
    high_severity_errors INTEGER DEFAULT 0,
    medium_severity_errors INTEGER DEFAULT 0,
    low_severity_errors INTEGER DEFAULT 0,

    -- Common issues
    documentation_issues INTEGER DEFAULT 0,
    em_level_issues INTEGER DEFAULT 0,
    modifier_issues INTEGER DEFAULT 0,
    diagnosis_issues INTEGER DEFAULT 0,

    -- Financial impact
    total_financial_impact NUMERIC(18,2),
    overpayment_total NUMERIC(18,2),
    underpayment_total NUMERIC(18,2),

    -- Calculated at
    calculated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(provider_id, period_start_date, period_type)
);

CREATE INDEX idx_provider_accuracy_provider ON claims.provider_accuracy(provider_id);
CREATE INDEX idx_provider_accuracy_org ON claims.provider_accuracy(organization_id);
CREATE INDEX idx_provider_accuracy_period ON claims.provider_accuracy(period_start_date, period_end_date);
CREATE INDEX idx_provider_accuracy_rate ON claims.provider_accuracy(overall_accuracy_rate);

COMMENT ON TABLE claims.provider_accuracy IS 'Provider documentation and coding accuracy metrics';
