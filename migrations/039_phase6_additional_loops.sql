-- Migration: 039_phase6_additional_loops
-- Description: Add tables and columns for Phase 6 - Additional Loops and Relationships
-- Date: 2025-11-03
-- Related: Phase 6 of 837P Full Implementation Action Plan

-- Phase 6.1: Patient Information (Loop 2010BC)
-- Create patient table for when patient is different from subscriber
CREATE TABLE IF NOT EXISTS claims.patient (
    patient_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    encounter_id UUID NOT NULL REFERENCES claims.encounter(encounter_id) ON DELETE CASCADE,

    -- NM1*QC Segment (Patient Name)
    patient_last_name VARCHAR(255) NOT NULL,
    patient_first_name VARCHAR(255) NOT NULL,
    patient_middle_name VARCHAR(255),
    patient_name_suffix VARCHAR(50),

    -- DMG Segment (Patient Demographics)
    patient_birth_date DATE,
    patient_gender CHAR(1), -- M, F, U
    patient_death_date DATE,
    patient_weight NUMERIC(10,2), -- Weight in pounds
    patient_pregnancy_indicator CHAR(1), -- Y/N

    -- N3/N4 Segments (Patient Address)
    patient_address_line1 VARCHAR(255),
    patient_address_line2 VARCHAR(255),
    patient_city VARCHAR(100),
    patient_state CHAR(2),
    patient_postal_code VARCHAR(15),
    patient_country CHAR(3) DEFAULT 'USA',

    -- REF Segments (Patient Identifiers)
    patient_id_qualifier VARCHAR(3), -- Y4=Property & Casualty Patient Number, etc.
    patient_identification_code VARCHAR(50),

    -- Audit trail
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,

    -- Constraints
    CONSTRAINT uq_patient_encounter UNIQUE(encounter_id)
);

-- Indexes for patient
CREATE INDEX IF NOT EXISTS idx_patient_encounter
    ON claims.patient(encounter_id);

CREATE INDEX IF NOT EXISTS idx_patient_name
    ON claims.patient(patient_last_name, patient_first_name);

CREATE INDEX IF NOT EXISTS idx_patient_birth_date
    ON claims.patient(patient_birth_date) WHERE patient_birth_date IS NOT NULL;

COMMENT ON TABLE claims.patient IS
    'Patient information from Loop 2010BC when patient is different from subscriber. One-to-one with encounter.';

COMMENT ON COLUMN claims.patient.patient_pregnancy_indicator IS
    'Pregnancy indicator: Y=Yes, N=No';

COMMENT ON COLUMN claims.patient.patient_weight IS
    'Patient weight in pounds from DMG segment';

-- Trigger for updated_at on patient
CREATE TRIGGER update_patient_updated_at
    BEFORE UPDATE ON claims.patient
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Phase 6.2: Purchased Service Provider (Loop 2420B)
-- Add columns for reference lab or purchased services
ALTER TABLE claims.service_line
    ADD COLUMN IF NOT EXISTS purchased_service_provider_id UUID REFERENCES claims.provider(provider_id),
    ADD COLUMN IF NOT EXISTS purchased_service_provider_npi VARCHAR(10),
    ADD COLUMN IF NOT EXISTS purchased_service_provider_name VARCHAR(255),
    ADD COLUMN IF NOT EXISTS purchased_service_charge_amount NUMERIC(18,2);

COMMENT ON COLUMN claims.service_line.purchased_service_provider_id IS
    'FK to provider table for purchased service provider from Loop 2420B';

COMMENT ON COLUMN claims.service_line.purchased_service_provider_npi IS
    'NPI of purchased service provider (e.g., reference laboratory)';

COMMENT ON COLUMN claims.service_line.purchased_service_charge_amount IS
    'Amount charged by purchased service provider';

-- Indexes for purchased service provider
CREATE INDEX IF NOT EXISTS idx_service_line_purchased_provider
    ON claims.service_line(purchased_service_provider_id) WHERE purchased_service_provider_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_service_line_purchased_provider_npi
    ON claims.service_line(purchased_service_provider_npi) WHERE purchased_service_provider_npi IS NOT NULL;

-- Phase 6.3: Test Results (MEA Segment)
-- Create test_result table for lab and diagnostic test results
CREATE TABLE IF NOT EXISTS claims.test_result (
    test_result_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    encounter_id UUID REFERENCES claims.encounter(encounter_id) ON DELETE CASCADE,
    service_line_id UUID REFERENCES claims.service_line(service_line_id) ON DELETE CASCADE,

    -- MEA Segment Elements
    measurement_reference_id VARCHAR(20), -- Reference ID code
    measurement_qualifier VARCHAR(3), -- TR=Test Result, OG=Original Result, etc.
    measurement_value NUMERIC(20,8), -- Numeric test result value
    measurement_unit VARCHAR(10), -- Unit of measure (mg/dL, mmHg, etc.)
    range_minimum NUMERIC(20,8), -- Normal range minimum
    range_maximum NUMERIC(20,8), -- Normal range maximum

    -- Additional context
    measurement_significance_code VARCHAR(2), -- HI=High, LO=Low, N=Normal, etc.
    measurement_description VARCHAR(80), -- Free-form description

    -- Audit trail
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,

    -- Constraints: Must reference either encounter OR service_line (or both)
    CONSTRAINT chk_test_result_reference CHECK (
        (encounter_id IS NOT NULL) OR (service_line_id IS NOT NULL)
    )
);

-- Indexes for test_result
CREATE INDEX IF NOT EXISTS idx_test_result_encounter
    ON claims.test_result(encounter_id) WHERE encounter_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_test_result_service_line
    ON claims.test_result(service_line_id) WHERE service_line_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_test_result_qualifier
    ON claims.test_result(measurement_qualifier);

CREATE INDEX IF NOT EXISTS idx_test_result_significance
    ON claims.test_result(measurement_significance_code) WHERE measurement_significance_code IS NOT NULL;

COMMENT ON TABLE claims.test_result IS
    'Test results from MEA segments in Loops 2300 and 2400. Stores lab values, vital signs, and diagnostic test results.';

COMMENT ON COLUMN claims.test_result.measurement_qualifier IS
    'Measurement qualifier: TR=Test Result, OG=Original Result, FR=Final Result, etc.';

COMMENT ON COLUMN claims.test_result.measurement_significance_code IS
    'Significance: HI=High, LO=Low, N=Normal, A=Abnormal, etc.';

-- Phase 6.4: Repricing Information (HCP Segment)
-- Add repricing and adjudication columns to encounter and service_line
ALTER TABLE claims.encounter
    ADD COLUMN IF NOT EXISTS pricing_methodology VARCHAR(2),
    ADD COLUMN IF NOT EXISTS repriced_allowed_amount NUMERIC(18,2),
    ADD COLUMN IF NOT EXISTS repriced_saving_amount NUMERIC(18,2),
    ADD COLUMN IF NOT EXISTS repricing_organization_id VARCHAR(50),
    ADD COLUMN IF NOT EXISTS repricing_rate NUMERIC(15,4),
    ADD COLUMN IF NOT EXISTS approved_drg_code VARCHAR(4),
    ADD COLUMN IF NOT EXISTS approved_drg_amount NUMERIC(18,2),
    ADD COLUMN IF NOT EXISTS reject_reason_code VARCHAR(2);

ALTER TABLE claims.service_line
    ADD COLUMN IF NOT EXISTS pricing_methodology VARCHAR(2),
    ADD COLUMN IF NOT EXISTS repriced_allowed_amount NUMERIC(18,2),
    ADD COLUMN IF NOT EXISTS repriced_saving_amount NUMERIC(18,2),
    ADD COLUMN IF NOT EXISTS unit_price NUMERIC(18,2),
    ADD COLUMN IF NOT EXISTS reject_reason_code VARCHAR(2);

COMMENT ON COLUMN claims.encounter.pricing_methodology IS
    'Pricing methodology code from HCP segment: 00=Zero Pricing, 01=Priced as Billed, 02=Priced at Standard Fee Schedule, etc.';

COMMENT ON COLUMN claims.encounter.repriced_allowed_amount IS
    'Repriced or allowed amount from HCP segment';

COMMENT ON COLUMN claims.encounter.repriced_saving_amount IS
    'Savings amount (original charge minus repriced amount)';

COMMENT ON COLUMN claims.encounter.repricing_organization_id IS
    'Organization that performed repricing';

COMMENT ON COLUMN claims.encounter.approved_drg_code IS
    'Approved DRG code from HCP segment (hospital claims)';

COMMENT ON COLUMN claims.encounter.reject_reason_code IS
    'Reject reason code from HCP segment';

COMMENT ON COLUMN claims.service_line.pricing_methodology IS
    'Line-level pricing methodology code from HCP segment';

COMMENT ON COLUMN claims.service_line.repriced_allowed_amount IS
    'Line-level repriced or allowed amount from HCP segment';

COMMENT ON COLUMN claims.service_line.unit_price IS
    'Unit price from HCP segment (price per unit of service)';

-- Indexes for repricing information
CREATE INDEX IF NOT EXISTS idx_encounter_pricing_methodology
    ON claims.encounter(pricing_methodology) WHERE pricing_methodology IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_encounter_repriced_amount
    ON claims.encounter(repriced_allowed_amount) WHERE repriced_allowed_amount IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_encounter_drg_code
    ON claims.encounter(approved_drg_code) WHERE approved_drg_code IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_service_line_pricing_methodology
    ON claims.service_line(pricing_methodology) WHERE pricing_methodology IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_service_line_repriced_amount
    ON claims.service_line(repriced_allowed_amount) WHERE repriced_allowed_amount IS NOT NULL;

-- Validation: Ensure non-negative amounts
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'chk_patient_weight_nonneg') THEN
        ALTER TABLE claims.patient
            ADD CONSTRAINT chk_patient_weight_nonneg
                CHECK (patient_weight IS NULL OR patient_weight >= 0);
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'chk_purchased_service_charge_nonneg') THEN
        ALTER TABLE claims.service_line
            ADD CONSTRAINT chk_purchased_service_charge_nonneg
                CHECK (purchased_service_charge_amount IS NULL OR purchased_service_charge_amount >= 0);
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'chk_encounter_repriced_allowed_nonneg') THEN
        ALTER TABLE claims.encounter
            ADD CONSTRAINT chk_encounter_repriced_allowed_nonneg
                CHECK (repriced_allowed_amount IS NULL OR repriced_allowed_amount >= 0);
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'chk_encounter_drg_amount_nonneg') THEN
        ALTER TABLE claims.encounter
            ADD CONSTRAINT chk_encounter_drg_amount_nonneg
                CHECK (approved_drg_amount IS NULL OR approved_drg_amount >= 0);
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'chk_sl_repriced_allowed_nonneg') THEN
        ALTER TABLE claims.service_line
            ADD CONSTRAINT chk_sl_repriced_allowed_nonneg
                CHECK (repriced_allowed_amount IS NULL OR repriced_allowed_amount >= 0);
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'chk_sl_unit_price_nonneg') THEN
        ALTER TABLE claims.service_line
            ADD CONSTRAINT chk_sl_unit_price_nonneg
                CHECK (unit_price IS NULL OR unit_price >= 0);
    END IF;
END$$;

-- Performance: Create composite indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_patient_encounter_dob
    ON claims.patient(encounter_id, patient_birth_date);

CREATE INDEX IF NOT EXISTS idx_test_result_encounter_qualifier
    ON claims.test_result(encounter_id, measurement_qualifier)
    WHERE encounter_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_test_result_line_qualifier
    ON claims.test_result(service_line_id, measurement_qualifier)
    WHERE service_line_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_encounter_repricing_org
    ON claims.encounter(repricing_organization_id) WHERE repricing_organization_id IS NOT NULL;
