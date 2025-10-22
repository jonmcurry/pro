-- Migration: 009_create_rvu_tables
-- Description: Create RVU reference tables for reimbursement estimation
-- Date: 2025-10-14

-- RVU reference data (from CMS Physician Fee Schedule)
CREATE TABLE claims.rvu_reference (
    rvu_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Procedure identification
    hcpcs_code VARCHAR(5) NOT NULL,
    modifier VARCHAR(2),

    -- Effective dates
    effective_year INTEGER NOT NULL,
    effective_date DATE NOT NULL,
    termination_date DATE,

    -- RVU components (from CMS PFS)
    work_rvu NUMERIC(10,3) DEFAULT 0.000, -- Work component
    pe_rvu_nonfacility NUMERIC(10,3) DEFAULT 0.000, -- Practice Expense - Non-facility
    pe_rvu_facility NUMERIC(10,3) DEFAULT 0.000, -- Practice Expense - Facility
    mp_rvu NUMERIC(10,3) DEFAULT 0.000, -- Malpractice RVU

    -- Total RVUs
    total_rvu_nonfacility NUMERIC(10,3) DEFAULT 0.000,
    total_rvu_facility NUMERIC(10,3) DEFAULT 0.000,

    -- Status indicators
    status_code VARCHAR(3), -- A, R, T, X, etc.
    multiple_surgery_indicator CHAR(1), -- 0, 1, 2, 3, 9
    bilateral_surgery_indicator CHAR(1), -- 0, 1, 2, 3
    assistant_surgery_indicator CHAR(1), -- 0, 1, 2
    co_surgery_indicator CHAR(1), -- 0, 1, 2
    team_surgery_indicator CHAR(1), -- 0, 1, 2

    -- Global period
    global_surgery_indicator VARCHAR(3), -- 000, 010, 090, XXX, YYY, ZZZ, MMM

    -- Pre/post operative percentages
    pre_op_percentage NUMERIC(5,2),
    intra_op_percentage NUMERIC(5,2),
    post_op_percentage NUMERIC(5,2),

    -- Professional/Technical component indicators
    pc_tc_indicator CHAR(1), -- 0, 1, 2, 3

    -- Procedure description
    short_description VARCHAR(255),
    long_description TEXT,

    -- Audit trail
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT uk_rvu_code_year_modifier UNIQUE (hcpcs_code, effective_year, modifier)
);

CREATE INDEX idx_rvu_hcpcs ON claims.rvu_reference(hcpcs_code);
CREATE INDEX idx_rvu_year ON claims.rvu_reference(effective_year);
CREATE INDEX idx_rvu_effective_date ON claims.rvu_reference(effective_date);
CREATE INDEX idx_rvu_code_year ON claims.rvu_reference(hcpcs_code, effective_year);
CREATE INDEX idx_rvu_status ON claims.rvu_reference(status_code);

COMMENT ON TABLE claims.rvu_reference IS 'RVU reference data from CMS Physician Fee Schedule';

CREATE TRIGGER update_rvu_reference_updated_at BEFORE UPDATE ON claims.rvu_reference
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Conversion factors (annual updates from CMS)
CREATE TABLE claims.conversion_factor (
    conversion_factor_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Identification
    factor_year INTEGER NOT NULL UNIQUE,
    effective_date DATE NOT NULL,
    termination_date DATE,

    -- Conversion factor value
    conversion_factor NUMERIC(10,4) NOT NULL, -- 2024 = $33.2875

    -- Additional adjustments
    budget_neutrality_adjustment NUMERIC(8,6) DEFAULT 1.000000,

    -- Audit trail
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100)
);

CREATE INDEX idx_conversion_factor_year ON claims.conversion_factor(factor_year);
CREATE INDEX idx_conversion_factor_effective ON claims.conversion_factor(effective_date);

COMMENT ON TABLE claims.conversion_factor IS 'Annual Medicare conversion factors for RVU to dollar conversion';

CREATE TRIGGER update_conversion_factor_updated_at BEFORE UPDATE ON claims.conversion_factor
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert 2024 conversion factor from SRD
INSERT INTO claims.conversion_factor (factor_year, effective_date, conversion_factor, created_by)
VALUES (2024, '2024-01-01', 33.2875, 'SYSTEM');

-- Geographic Practice Cost Index (GPCI) by locality
CREATE TABLE claims.gpci_reference (
    gpci_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Locality identification
    locality_code VARCHAR(5) NOT NULL,
    locality_name VARCHAR(255) NOT NULL,
    state_code CHAR(2) NOT NULL,

    -- Effective dates
    effective_year INTEGER NOT NULL,
    effective_date DATE NOT NULL,
    termination_date DATE,

    -- GPCI values for each RVU component
    work_gpci NUMERIC(6,3) NOT NULL DEFAULT 1.000,
    pe_gpci NUMERIC(6,3) NOT NULL DEFAULT 1.000,
    mp_gpci NUMERIC(6,3) NOT NULL DEFAULT 1.000,

    -- Audit trail
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT uk_gpci_locality_year UNIQUE (locality_code, effective_year)
);

CREATE INDEX idx_gpci_locality ON claims.gpci_reference(locality_code);
CREATE INDEX idx_gpci_state ON claims.gpci_reference(state_code);
CREATE INDEX idx_gpci_year ON claims.gpci_reference(effective_year);

COMMENT ON TABLE claims.gpci_reference IS 'Geographic Practice Cost Indexes by Medicare locality';

CREATE TRIGGER update_gpci_reference_updated_at BEFORE UPDATE ON claims.gpci_reference
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Modifier reimbursement adjustments
CREATE TABLE claims.modifier_adjustment (
    adjustment_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Modifier
    modifier_code VARCHAR(2) NOT NULL UNIQUE,
    modifier_description TEXT,

    -- Payment adjustment
    payment_percentage NUMERIC(5,2), -- e.g., 50 for 50% payment
    payment_multiplier NUMERIC(5,3), -- e.g., 0.50 for 50% payment

    -- Applicability
    applies_to_professional BOOLEAN DEFAULT true,
    applies_to_technical BOOLEAN DEFAULT true,
    affects_rvu BOOLEAN DEFAULT true,

    -- Rules
    combining_rules TEXT,
    sequencing_rules TEXT,

    -- Status
    is_active BOOLEAN DEFAULT true,

    -- Audit trail
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_modifier_adjustment_code ON claims.modifier_adjustment(modifier_code);
CREATE INDEX idx_modifier_adjustment_active ON claims.modifier_adjustment(is_active) WHERE is_active = true;

COMMENT ON TABLE claims.modifier_adjustment IS 'Modifier-based reimbursement adjustment rules';

CREATE TRIGGER update_modifier_adjustment_updated_at BEFORE UPDATE ON claims.modifier_adjustment
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert common modifier adjustments
INSERT INTO claims.modifier_adjustment (modifier_code, modifier_description, payment_percentage, payment_multiplier) VALUES
('50', 'Bilateral Procedure', 150, 1.50),
('51', 'Multiple Procedures', 50, 0.50),
('52', 'Reduced Services', 50, 0.50),
('53', 'Discontinued Procedure', 50, 0.50),
('54', 'Surgical Care Only', 70, 0.70),
('55', 'Postoperative Management Only', 30, 0.30),
('56', 'Preoperative Management Only', 10, 0.10),
('62', 'Two Surgeons', 125, 1.25),
('66', 'Surgical Team', 100, 1.00),
('76', 'Repeat Procedure by Same Physician', 100, 1.00),
('77', 'Repeat Procedure by Another Physician', 100, 1.00),
('78', 'Unplanned Return to OR', 100, 1.00),
('79', 'Unrelated Procedure During Postoperative Period', 100, 1.00),
('80', 'Assistant Surgeon', 16, 0.16),
('81', 'Minimum Assistant Surgeon', 16, 0.16),
('82', 'Assistant Surgeon (when qualified resident not available)', 16, 0.16),
('AS', 'Physician Assistant, Nurse Practitioner, or CNS Services', 85, 0.85),
('TC', 'Technical Component', NULL, NULL),
('26', 'Professional Component', NULL, NULL);

-- Reimbursement estimates for service lines
CREATE TABLE claims.service_line_reimbursement (
    reimbursement_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    service_line_id UUID NOT NULL REFERENCES claims.service_line(service_line_id) ON DELETE CASCADE,

    -- RVU calculation
    rvu_id UUID REFERENCES claims.rvu_reference(rvu_id),
    conversion_factor_id UUID REFERENCES claims.conversion_factor(conversion_factor_id),
    gpci_id UUID REFERENCES claims.gpci_reference(gpci_id),

    -- RVU components used
    work_rvu NUMERIC(10,3),
    pe_rvu NUMERIC(10,3),
    mp_rvu NUMERIC(10,3),
    total_rvu NUMERIC(10,3),

    -- GPCI adjustments
    work_gpci NUMERIC(6,3) DEFAULT 1.000,
    pe_gpci NUMERIC(6,3) DEFAULT 1.000,
    mp_gpci NUMERIC(6,3) DEFAULT 1.000,

    -- Conversion factor
    conversion_factor NUMERIC(10,4),

    -- Calculated amounts
    base_medicare_payment NUMERIC(18,2),
    modifier_adjustment_percentage NUMERIC(5,2) DEFAULT 100.00,
    adjusted_medicare_payment NUMERIC(18,2),

    -- Units
    unit_count NUMERIC(15,1),
    total_medicare_payment NUMERIC(18,2),

    -- Comparison to billed
    billed_amount NUMERIC(18,2),
    payment_to_charge_ratio NUMERIC(8,4),

    -- Calculation details
    calculation_method VARCHAR(50), -- STANDARD, NON_FACILITY, FACILITY, TECHNICAL, PROFESSIONAL
    calculation_notes TEXT,
    is_estimated BOOLEAN DEFAULT true,

    -- Audit trail
    calculated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(service_line_id)
);

CREATE INDEX idx_service_line_reimb_line ON claims.service_line_reimbursement(service_line_id);
CREATE INDEX idx_service_line_reimb_rvu ON claims.service_line_reimbursement(rvu_id);
CREATE INDEX idx_service_line_reimb_calculated ON claims.service_line_reimbursement(calculated_at);

COMMENT ON TABLE claims.service_line_reimbursement IS 'Estimated Medicare reimbursement for service lines based on RVU';
