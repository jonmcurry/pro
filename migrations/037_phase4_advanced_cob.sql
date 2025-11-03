-- Migration: 037_phase4_advanced_cob
-- Description: Add tables and columns for Phase 4 - Advanced Coordination of Benefits (COB)
-- Date: 2025-11-03
-- Related: Phase 4 of 837P Full Implementation Action Plan

-- Phase 4.1: Create other_insurance table for full SBR segment data
-- Supports multiple other payers per claim (primary, secondary, tertiary)
CREATE TABLE IF NOT EXISTS claims.other_insurance (
    other_insurance_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    encounter_id UUID NOT NULL REFERENCES claims.encounter(encounter_id) ON DELETE CASCADE,

    -- SBR Segment Elements
    payer_responsibility_sequence VARCHAR(1) NOT NULL, -- P=Primary, S=Secondary, T=Tertiary
    individual_relationship_code VARCHAR(2), -- 01=Spouse, 18=Self, 19=Child, etc.
    group_policy_number VARCHAR(50),
    group_name VARCHAR(255),
    insurance_type_code VARCHAR(3), -- 12=Medicare Part A, 13=Medicare Part B, etc.
    coordination_benefits_code VARCHAR(1), -- Coordination of benefits code
    yes_no_condition_response VARCHAR(1), -- Y/N
    employment_status_code VARCHAR(1), -- Employment status
    claim_filing_indicator VARCHAR(2), -- MB=Medicare Part B, MA=Medicare Part A, etc.

    -- Payer Information (from Loop 2330B)
    payer_id VARCHAR(80),
    payer_name VARCHAR(255),
    payer_address_line1 VARCHAR(255),
    payer_address_line2 VARCHAR(255),
    payer_city VARCHAR(100),
    payer_state CHAR(2),
    payer_postal_code VARCHAR(15),

    -- Phase 4.3: OI Segment (Other Insurance Coverage Information)
    benefits_assignment_certification VARCHAR(1), -- Y/N
    patient_signature_source_code VARCHAR(1), -- P=Patient signed, etc.
    release_of_information_code VARCHAR(1), -- Y/N/I

    -- Phase 4.4: MOA Segment (Medicare Outpatient Adjudication)
    reimbursement_rate NUMERIC(18,2),
    hcpcs_payable_amount NUMERIC(18,2),
    claim_payment_remark_code_1 VARCHAR(5),
    claim_payment_remark_code_2 VARCHAR(5),
    claim_payment_remark_code_3 VARCHAR(5),
    claim_payment_remark_code_4 VARCHAR(5),
    claim_payment_remark_code_5 VARCHAR(5),
    end_stage_renal_disease_amount NUMERIC(18,2),
    non_payable_professional_component_amount NUMERIC(18,2),

    -- Payment Information
    paid_amount NUMERIC(18,2), -- Amount paid by this payer
    claim_control_number VARCHAR(50), -- Payer's claim reference number

    -- Audit trail
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,

    -- Constraints
    CONSTRAINT chk_payer_responsibility CHECK (payer_responsibility_sequence IN ('P', 'S', 'T'))
);

-- Indexes for other_insurance
CREATE INDEX IF NOT EXISTS idx_other_insurance_encounter
    ON claims.other_insurance(encounter_id);

CREATE INDEX IF NOT EXISTS idx_other_insurance_payer_responsibility
    ON claims.other_insurance(payer_responsibility_sequence);

CREATE INDEX IF NOT EXISTS idx_other_insurance_payer_id
    ON claims.other_insurance(payer_id);

CREATE INDEX IF NOT EXISTS idx_other_insurance_paid_amount
    ON claims.other_insurance(paid_amount) WHERE paid_amount IS NOT NULL;

COMMENT ON TABLE claims.other_insurance IS
    'Other insurance information for coordination of benefits (COB). Stores SBR, OI, and MOA segment data for primary/secondary payer tracking.';

COMMENT ON COLUMN claims.other_insurance.payer_responsibility_sequence IS
    'Payer sequence: P=Primary, S=Secondary, T=Tertiary';

COMMENT ON COLUMN claims.other_insurance.individual_relationship_code IS
    'Relationship of patient to insured: 01=Spouse, 18=Self, 19=Child, etc.';

-- Phase 4.2: Create claim_adjustment table for CAS segments
-- Stores claim and line-level adjustments from previous payers
CREATE TABLE IF NOT EXISTS claims.claim_adjustment (
    adjustment_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    encounter_id UUID REFERENCES claims.encounter(encounter_id) ON DELETE CASCADE,
    service_line_id UUID REFERENCES claims.service_line(service_line_id) ON DELETE CASCADE,
    other_insurance_id UUID REFERENCES claims.other_insurance(other_insurance_id) ON DELETE CASCADE,

    -- CAS Segment Elements
    adjustment_group_code VARCHAR(2) NOT NULL, -- CO=Contractual, CR=Correction, OA=Other, PI=Payer Initiated, PR=Patient Responsibility
    adjustment_reason_code VARCHAR(5) NOT NULL, -- CARC codes (1-999+)
    adjustment_amount NUMERIC(18,2) NOT NULL, -- Amount adjusted
    adjustment_quantity NUMERIC(15,3), -- Quantity adjusted (optional)

    -- Sequence tracking
    adjustment_sequence INTEGER, -- Order of adjustments within CAS segment

    -- Audit trail
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,

    -- Constraints
    CONSTRAINT chk_one_level_reference CHECK (
        -- Must reference either encounter OR service_line, not both
        (encounter_id IS NOT NULL AND service_line_id IS NULL) OR
        (encounter_id IS NULL AND service_line_id IS NOT NULL)
    ),
    CONSTRAINT chk_adjustment_amount_format CHECK (
        adjustment_amount IS NOT NULL
    )
);

-- Indexes for claim_adjustment
CREATE INDEX IF NOT EXISTS idx_claim_adjustment_encounter
    ON claims.claim_adjustment(encounter_id) WHERE encounter_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_claim_adjustment_service_line
    ON claims.claim_adjustment(service_line_id) WHERE service_line_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_claim_adjustment_other_insurance
    ON claims.claim_adjustment(other_insurance_id);

CREATE INDEX IF NOT EXISTS idx_claim_adjustment_group_code
    ON claims.claim_adjustment(adjustment_group_code);

CREATE INDEX IF NOT EXISTS idx_claim_adjustment_reason_code
    ON claims.claim_adjustment(adjustment_reason_code);

COMMENT ON TABLE claims.claim_adjustment IS
    'Claim and service line adjustments from CAS segments. Tracks denials, contractual adjustments, and patient responsibility at both claim and line levels.';

COMMENT ON COLUMN claims.claim_adjustment.adjustment_group_code IS
    'Adjustment group code: CO=Contractual Obligation, CR=Correction/Reversal, OA=Other Adjustments, PI=Payer Initiated Reductions, PR=Patient Responsibility';

COMMENT ON COLUMN claims.claim_adjustment.adjustment_reason_code IS
    'Claim Adjustment Reason Code (CARC). Standard codes 1-999+ explaining adjustment reason.';

-- Add trigger for updated_at on other_insurance
CREATE TRIGGER update_other_insurance_updated_at
    BEFORE UPDATE ON claims.other_insurance
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Validation: Ensure non-negative amounts
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'chk_other_insurance_paid_amount_nonneg') THEN
        ALTER TABLE claims.other_insurance
            ADD CONSTRAINT chk_other_insurance_paid_amount_nonneg
                CHECK (paid_amount IS NULL OR paid_amount >= 0);
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'chk_other_insurance_reimbursement_rate_nonneg') THEN
        ALTER TABLE claims.other_insurance
            ADD CONSTRAINT chk_other_insurance_reimbursement_rate_nonneg
                CHECK (reimbursement_rate IS NULL OR reimbursement_rate >= 0);
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'chk_other_insurance_hcpcs_payable_nonneg') THEN
        ALTER TABLE claims.other_insurance
            ADD CONSTRAINT chk_other_insurance_hcpcs_payable_nonneg
                CHECK (hcpcs_payable_amount IS NULL OR hcpcs_payable_amount >= 0);
    END IF;
END$$;

-- Performance: Create composite indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_other_insurance_encounter_sequence
    ON claims.other_insurance(encounter_id, payer_responsibility_sequence);

CREATE INDEX IF NOT EXISTS idx_claim_adjustment_encounter_group
    ON claims.claim_adjustment(encounter_id, adjustment_group_code)
    WHERE encounter_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_claim_adjustment_line_group
    ON claims.claim_adjustment(service_line_id, adjustment_group_code)
    WHERE service_line_id IS NOT NULL;
