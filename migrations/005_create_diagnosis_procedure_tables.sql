-- Migration: 005_create_diagnosis_procedure_tables
-- Description: Create diagnosis and procedure tables for encounter line items
-- Date: 2025-10-14

-- Diagnosis codes for encounter (Loop 2300 HI)
CREATE TABLE claims.encounter_diagnosis (
    diagnosis_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    encounter_id UUID NOT NULL REFERENCES claims.encounter(encounter_id) ON DELETE CASCADE,
    sequence_number SMALLINT NOT NULL, -- 1-12 for principal and secondary diagnoses
    diagnosis_code_qualifier VARCHAR(3) DEFAULT 'ABK', -- ABK for ICD-10-CM
    diagnosis_code VARCHAR(30) NOT NULL,
    diagnosis_description TEXT,
    is_principal BOOLEAN DEFAULT false, -- True for first/principal diagnosis
    is_admitting BOOLEAN DEFAULT false,
    is_external_cause BOOLEAN DEFAULT false,
    is_patient_reason BOOLEAN DEFAULT false,
    present_on_admission_indicator CHAR(1), -- Y, N, U, W, or null
    hcc_indicator BOOLEAN DEFAULT false,
    hcc_category VARCHAR(10),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT chk_sequence_range CHECK (sequence_number BETWEEN 1 AND 12),
    CONSTRAINT uk_encounter_diagnosis_seq UNIQUE (encounter_id, sequence_number)
);

CREATE INDEX idx_enc_diag_encounter ON claims.encounter_diagnosis(encounter_id);
CREATE INDEX idx_enc_diag_code ON claims.encounter_diagnosis(diagnosis_code);
CREATE INDEX idx_enc_diag_principal ON claims.encounter_diagnosis(encounter_id, is_principal)
    WHERE is_principal = true;
CREATE INDEX idx_enc_diag_hcc ON claims.encounter_diagnosis(hcc_indicator, hcc_category)
    WHERE hcc_indicator = true;

COMMENT ON TABLE claims.encounter_diagnosis IS 'Diagnosis codes associated with encounters (ICD-10-CM)';

-- Service lines / procedures for encounter (Loop 2400)
CREATE TABLE claims.service_line (
    service_line_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    encounter_id UUID NOT NULL REFERENCES claims.encounter(encounter_id) ON DELETE CASCADE,
    line_number SMALLINT NOT NULL, -- Service line number

    -- Service information (Loop 2400 SV1)
    product_service_id_qualifier VARCHAR(2) DEFAULT 'HC', -- HC for HCPCS
    procedure_code VARCHAR(48) NOT NULL, -- CPT/HCPCS code
    procedure_modifier_1 VARCHAR(2),
    procedure_modifier_2 VARCHAR(2),
    procedure_modifier_3 VARCHAR(2),
    procedure_modifier_4 VARCHAR(2),
    procedure_description TEXT,
    line_item_charge_amount NUMERIC(18,2) NOT NULL CHECK (line_item_charge_amount >= 0),
    unit_basis_measurement_code VARCHAR(2) DEFAULT 'UN', -- UN for units, MJ for minutes
    service_unit_count NUMERIC(15,1) NOT NULL CHECK (service_unit_count > 0 AND service_unit_count <= 9999.9),

    -- Place of service
    place_of_service_code VARCHAR(2),

    -- Dates
    service_date_from DATE NOT NULL,
    service_date_to DATE,

    -- Emergency indicator
    emergency_indicator BOOLEAN DEFAULT false,

    -- EPSDT indicator
    epsdt_indicator BOOLEAN DEFAULT false,

    -- Family planning indicator
    family_planning_indicator BOOLEAN DEFAULT false,

    -- Rendering provider at line level (Loop 2420A)
    rendering_provider_id UUID REFERENCES claims.provider(provider_id),
    rendering_provider_npi VARCHAR(10),

    -- Supervising provider at line level (Loop 2420D)
    supervising_provider_id UUID REFERENCES claims.provider(provider_id),
    supervising_provider_npi VARCHAR(10),

    -- Ordering provider at line level (Loop 2420E)
    ordering_provider_id UUID REFERENCES claims.provider(provider_id),
    ordering_provider_npi VARCHAR(10),

    -- Referring provider at line level (Loop 2420F)
    referring_provider_id UUID REFERENCES claims.provider(provider_id),
    referring_provider_npi VARCHAR(10),

    -- Service facility at line level (Loop 2420C)
    service_facility_id UUID REFERENCES claims.facility(facility_id),
    service_facility_npi VARCHAR(10),

    -- Prior authorization
    prior_authorization_number VARCHAR(50),

    -- Referral number
    referral_number VARCHAR(50),

    -- Line note/description
    line_note TEXT,

    -- Revenue code (for institutional claims, may be present on professional)
    revenue_code VARCHAR(4),

    -- NDC information (Loop 2410 for drugs)
    ndc_code VARCHAR(11),
    ndc_unit_count NUMERIC(15,3),
    ndc_measurement_unit VARCHAR(2),

    -- DME information
    dme_rental_price NUMERIC(18,2),
    dme_purchase_price NUMERIC(18,2),
    dme_frequency_code VARCHAR(1),

    -- Anesthesia information
    anesthesia_minutes INTEGER,
    obstetric_additional_units INTEGER,

    -- Test results (Loop 2400 MEA)
    test_result_value NUMERIC(20,1),
    test_result_measurement_code VARCHAR(20),

    -- Ambulance information (line level)
    ambulance_patient_count INTEGER,
    ambulance_transport_distance NUMERIC(15,4),
    ambulance_patient_weight NUMERIC(10,2),

    -- Diagnosis pointers (up to 12)
    diagnosis_code_pointer_1 SMALLINT,
    diagnosis_code_pointer_2 SMALLINT,
    diagnosis_code_pointer_3 SMALLINT,
    diagnosis_code_pointer_4 SMALLINT,
    diagnosis_code_pointer_5 SMALLINT,
    diagnosis_code_pointer_6 SMALLINT,
    diagnosis_code_pointer_7 SMALLINT,
    diagnosis_code_pointer_8 SMALLINT,
    diagnosis_code_pointer_9 SMALLINT,
    diagnosis_code_pointer_10 SMALLINT,
    diagnosis_code_pointer_11 SMALLINT,
    diagnosis_code_pointer_12 SMALLINT,

    -- Other payer information at line level (Loop 2430)
    other_payer_line_paid_amount NUMERIC(18,2),
    other_payer_line_service_id VARCHAR(48),

    -- Status
    line_status VARCHAR(50) DEFAULT 'ACTIVE',

    -- Audit trail
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100),
    updated_by VARCHAR(100),

    CONSTRAINT uk_encounter_line UNIQUE (encounter_id, line_number),
    CONSTRAINT chk_service_date_range CHECK (service_date_to IS NULL OR service_date_to >= service_date_from)
);

-- Indexes for service lines
CREATE INDEX idx_service_line_encounter ON claims.service_line(encounter_id);
CREATE INDEX idx_service_line_procedure ON claims.service_line(procedure_code);
CREATE INDEX idx_service_line_date_from ON claims.service_line(service_date_from);
CREATE INDEX idx_service_line_date_to ON claims.service_line(service_date_to);
CREATE INDEX idx_service_line_rendering_provider ON claims.service_line(rendering_provider_id);
CREATE INDEX idx_service_line_ndc ON claims.service_line(ndc_code) WHERE ndc_code IS NOT NULL;
CREATE INDEX idx_service_line_revenue ON claims.service_line(revenue_code) WHERE revenue_code IS NOT NULL;

-- Composite indexes
CREATE INDEX idx_service_line_enc_line ON claims.service_line(encounter_id, line_number);
CREATE INDEX idx_service_line_proc_date ON claims.service_line(procedure_code, service_date_from);

COMMENT ON TABLE claims.service_line IS 'Service line items (procedures) for encounters - Loop 2400 data';

-- Trigger for updated_at
CREATE TRIGGER update_service_line_updated_at BEFORE UPDATE ON claims.service_line
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Line item adjustments from other payers (Loop 2430 CAS)
CREATE TABLE claims.service_line_adjustment (
    adjustment_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    service_line_id UUID NOT NULL REFERENCES claims.service_line(service_line_id) ON DELETE CASCADE,
    claim_adjustment_group_code VARCHAR(2) NOT NULL, -- CO, CR, OA, PI, PR
    adjustment_reason_code VARCHAR(5) NOT NULL,
    adjustment_amount NUMERIC(18,2) NOT NULL,
    adjustment_quantity NUMERIC(15,3),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_service_line_adj_line ON claims.service_line_adjustment(service_line_id);
CREATE INDEX idx_service_line_adj_reason ON claims.service_line_adjustment(adjustment_reason_code);

COMMENT ON TABLE claims.service_line_adjustment IS 'Line-level claim adjustments from other payers';

-- Additional diagnosis pointer mappings (if needed for complex scenarios)
CREATE TABLE claims.service_line_diagnosis_pointer (
    pointer_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    service_line_id UUID NOT NULL REFERENCES claims.service_line(service_line_id) ON DELETE CASCADE,
    diagnosis_id UUID NOT NULL REFERENCES claims.encounter_diagnosis(diagnosis_id) ON DELETE CASCADE,
    pointer_sequence SMALLINT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT uk_line_diag_pointer UNIQUE (service_line_id, pointer_sequence)
);

CREATE INDEX idx_line_diag_ptr_line ON claims.service_line_diagnosis_pointer(service_line_id);
CREATE INDEX idx_line_diag_ptr_diag ON claims.service_line_diagnosis_pointer(diagnosis_id);

COMMENT ON TABLE claims.service_line_diagnosis_pointer IS 'Explicit mapping between service lines and diagnoses';
