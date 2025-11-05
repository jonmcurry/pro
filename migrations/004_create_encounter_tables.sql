-- Migration: 004_create_encounter_tables
-- Description: Create encounter/claim tables with all 837p elements
-- Date: 2025-10-14

-- Main encounter table (represents a claim)
CREATE TABLE claims.encounter (
    encounter_id BIGINT GENERATED ALWAYS AS IDENTITY (CACHE 100) PRIMARY KEY,

    -- Organizational references
    facility_id BIGINT NOT NULL REFERENCES claims.facility(facility_id),
    organization_id BIGINT NOT NULL REFERENCES claims.organization(organization_id),
    region_id BIGINT REFERENCES claims.region(region_id),

    -- Submitter information (Loop 1000A)
    submitter_id VARCHAR(80) NOT NULL,
    submitter_name VARCHAR(255),

    -- Control numbers
    patient_control_number VARCHAR(38) NOT NULL, -- CLM01 (only 20 chars stored/returned)
    transaction_set_control_number VARCHAR(9),

    -- Patient/Subscriber information (Loop 2000B/2010BA)
    subscriber_id VARCHAR(80) NOT NULL, -- Medicare Beneficiary Identifier (MBI)
    subscriber_last_name VARCHAR(255) NOT NULL,
    subscriber_first_name VARCHAR(255) NOT NULL,
    subscriber_middle_name VARCHAR(255),
    subscriber_name_suffix VARCHAR(50),
    subscriber_gender CHAR(1), -- M, F, U
    subscriber_birth_date DATE NOT NULL,
    subscriber_address_line1 VARCHAR(255),
    subscriber_address_line2 VARCHAR(255),
    subscriber_city VARCHAR(100),
    subscriber_state CHAR(2),
    subscriber_postal_code VARCHAR(15),
    subscriber_country CHAR(3) DEFAULT 'USA',

    -- Payer information (Loop 2010BB)
    payer_responsibility_code CHAR(1) NOT NULL, -- P (Primary) or S (Secondary)
    payer_id VARCHAR(80),
    payer_name VARCHAR(255),
    claim_filing_indicator VARCHAR(2) DEFAULT 'MB', -- Medicare Part B

    -- Billing provider (Loop 2010AA)
    billing_provider_id BIGINT REFERENCES claims.provider(provider_id),
    billing_provider_npi VARCHAR(10),
    billing_provider_tax_id VARCHAR(20),
    billing_provider_name VARCHAR(255),
    billing_provider_address_line1 VARCHAR(255),
    billing_provider_address_line2 VARCHAR(255),
    billing_provider_city VARCHAR(100),
    billing_provider_state CHAR(2),
    billing_provider_postal_code VARCHAR(15),

    -- Claim information (Loop 2300 CLM)
    total_claim_charge_amount NUMERIC(18,2) NOT NULL,
    place_of_service_code VARCHAR(2),
    claim_frequency_code CHAR(1) DEFAULT '1', -- 1 = Original
    signature_indicator CHAR(1),
    assignment_indicator CHAR(1),
    benefits_assignment_indicator CHAR(1),
    release_of_information_code CHAR(1),
    patient_signature_code CHAR(1),

    -- Dates (Loop 2300 DTP)
    date_of_service_from DATE NOT NULL,
    date_of_service_to DATE,
    onset_of_illness_date DATE,
    initial_treatment_date DATE,
    last_seen_date DATE,
    acute_manifestation_date DATE,
    accident_date DATE,
    last_menstrual_period_date DATE,
    last_xray_date DATE,
    prescription_date DATE,
    disability_from_date DATE,
    disability_to_date DATE,
    last_worked_date DATE,
    authorized_return_to_work_date DATE,
    admission_date DATE,
    discharge_date DATE,
    assumed_care_date DATE,
    relinquished_care_date DATE,

    -- Additional claim information
    delay_reason_code VARCHAR(2),
    special_program_code VARCHAR(3),
    patient_amount_paid NUMERIC(18,2),
    service_authorization_code VARCHAR(50),

    -- Referring provider (Loop 2310A)
    referring_provider_id BIGINT REFERENCES claims.provider(provider_id),
    referring_provider_npi VARCHAR(10),
    referring_provider_name VARCHAR(255),

    -- Rendering provider (Loop 2310B)
    rendering_provider_id BIGINT REFERENCES claims.provider(provider_id),
    rendering_provider_npi VARCHAR(10),
    rendering_provider_name VARCHAR(255),

    -- Service facility (Loop 2310C)
    service_facility_id BIGINT REFERENCES claims.facility(facility_id),
    service_facility_npi VARCHAR(10),
    service_facility_name VARCHAR(255),
    service_facility_address_line1 VARCHAR(255),
    service_facility_address_line2 VARCHAR(255),
    service_facility_city VARCHAR(100),
    service_facility_state CHAR(2),
    service_facility_postal_code VARCHAR(15),

    -- Supervising provider (Loop 2310D)
    supervising_provider_id BIGINT REFERENCES claims.provider(provider_id),
    supervising_provider_npi VARCHAR(10),
    supervising_provider_name VARCHAR(255),

    -- Other payer information (Loop 2320 for COB)
    other_payer_paid_amount NUMERIC(18,2),
    other_payer_id VARCHAR(80),
    other_payer_name VARCHAR(255),
    other_payer_claim_number VARCHAR(50),
    other_payer_claim_filing_indicator VARCHAR(2),

    -- Ambulance information (Loop 2300 CR1)
    ambulance_transport_reason_code CHAR(1),
    ambulance_transport_distance NUMERIC(15,4),
    ambulance_patient_weight NUMERIC(10,2),
    ambulance_patient_count INTEGER,

    -- Coder/billing information
    coder_id BIGINT REFERENCES claims.coder(coder_id),
    coding_date DATE,

    -- Status and workflow
    claim_status VARCHAR(50) DEFAULT 'NEW', -- NEW, PENDING, FLAGGED, REVIEWED, ACCEPTED, REJECTED
    case_status VARCHAR(50), -- From SRD field list
    financial_class VARCHAR(50),

    -- Import tracking
    import_batch_id BIGINT,
    import_date TIMESTAMPTZ,
    import_configuration_id BIGINT,

    -- Audit trail
    is_active BOOLEAN DEFAULT true,
    soft_deleted BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100),
    updated_by VARCHAR(100),

    -- Constraints
    CONSTRAINT chk_dos_range CHECK (date_of_service_to IS NULL OR date_of_service_to >= date_of_service_from),
    CONSTRAINT chk_payer_responsibility CHECK (payer_responsibility_code IN ('P', 'S'))
);

-- Indexes for performance (optimized for 10,000 claims / 15 seconds requirement)
CREATE INDEX idx_encounter_facility ON claims.encounter(facility_id);
CREATE INDEX idx_encounter_organization ON claims.encounter(organization_id);
CREATE INDEX idx_encounter_patient_control ON claims.encounter(patient_control_number);
CREATE INDEX idx_encounter_subscriber ON claims.encounter(subscriber_id);
CREATE INDEX idx_encounter_dos_from ON claims.encounter(date_of_service_from);
CREATE INDEX idx_encounter_dos_to ON claims.encounter(date_of_service_to);
CREATE INDEX idx_encounter_dos_range ON claims.encounter(date_of_service_from, date_of_service_to);
CREATE INDEX idx_encounter_billing_provider ON claims.encounter(billing_provider_id);
CREATE INDEX idx_encounter_rendering_provider ON claims.encounter(rendering_provider_id);
CREATE INDEX idx_encounter_referring_provider ON claims.encounter(referring_provider_id);
CREATE INDEX idx_encounter_coder ON claims.encounter(coder_id);
CREATE INDEX idx_encounter_coding_date ON claims.encounter(coding_date);
CREATE INDEX idx_encounter_status ON claims.encounter(claim_status);
CREATE INDEX idx_encounter_import_batch ON claims.encounter(import_batch_id);
CREATE INDEX idx_encounter_import_date ON claims.encounter(import_date);
CREATE INDEX idx_encounter_active ON claims.encounter(is_active) WHERE is_active = true;
CREATE INDEX idx_encounter_not_deleted ON claims.encounter(soft_deleted) WHERE soft_deleted = false;
CREATE INDEX idx_encounter_created_at ON claims.encounter(created_at);

-- Composite indexes for common query patterns
CREATE INDEX idx_encounter_org_dos ON claims.encounter(organization_id, date_of_service_from);
CREATE INDEX idx_encounter_facility_dos ON claims.encounter(facility_id, date_of_service_from);
CREATE INDEX idx_encounter_provider_dos ON claims.encounter(billing_provider_id, date_of_service_from);
CREATE INDEX idx_encounter_status_dos ON claims.encounter(claim_status, date_of_service_from);

-- Partial index for pending/flagged claims
CREATE INDEX idx_encounter_needs_review ON claims.encounter(encounter_id, claim_status)
    WHERE claim_status IN ('PENDING', 'FLAGGED');

COMMENT ON TABLE claims.encounter IS 'Main encounter/claim table containing all 837p claim-level data elements';

-- Trigger for updated_at
CREATE TRIGGER update_encounter_updated_at BEFORE UPDATE ON claims.encounter
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Table for encounter notes/comments
CREATE TABLE claims.encounter_note (
    note_id BIGINT GENERATED ALWAYS AS IDENTITY (CACHE 100) PRIMARY KEY,
    encounter_id BIGINT NOT NULL REFERENCES claims.encounter(encounter_id) ON DELETE CASCADE,
    note_type VARCHAR(50), -- GENERAL, AUDIT, BILLING, etc.
    note_text TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100)
);

CREATE INDEX idx_encounter_note_encounter ON claims.encounter_note(encounter_id);
CREATE INDEX idx_encounter_note_type ON claims.encounter_note(note_type);
CREATE INDEX idx_encounter_note_created ON claims.encounter_note(created_at);

COMMENT ON TABLE claims.encounter_note IS 'Notes and comments associated with encounters';
