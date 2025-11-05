-- Migration: 038_phase5_specialized_claims
-- Description: Add tables and columns for Phase 5 - Specialized Claim Types
-- Date: 2025-11-03
-- Related: Phase 5 of 837P Full Implementation Action Plan

-- Phase 5.1: Ambulance Claims (CR1 Segment and Loops 2310E/F)
-- Add pickup and dropoff location columns to encounter table
ALTER TABLE claims.encounter
    ADD COLUMN IF NOT EXISTS ambulance_pickup_location_name VARCHAR(255),
    ADD COLUMN IF NOT EXISTS ambulance_pickup_address_line1 VARCHAR(255),
    ADD COLUMN IF NOT EXISTS ambulance_pickup_address_line2 VARCHAR(255),
    ADD COLUMN IF NOT EXISTS ambulance_pickup_city VARCHAR(100),
    ADD COLUMN IF NOT EXISTS ambulance_pickup_state CHAR(2),
    ADD COLUMN IF NOT EXISTS ambulance_pickup_postal_code VARCHAR(15),
    ADD COLUMN IF NOT EXISTS ambulance_dropoff_location_name VARCHAR(255),
    ADD COLUMN IF NOT EXISTS ambulance_dropoff_address_line1 VARCHAR(255),
    ADD COLUMN IF NOT EXISTS ambulance_dropoff_address_line2 VARCHAR(255),
    ADD COLUMN IF NOT EXISTS ambulance_dropoff_city VARCHAR(100),
    ADD COLUMN IF NOT EXISTS ambulance_dropoff_state CHAR(2),
    ADD COLUMN IF NOT EXISTS ambulance_dropoff_postal_code VARCHAR(15);

COMMENT ON COLUMN claims.encounter.ambulance_pickup_location_name IS
    'Ambulance pickup location name from Loop 2310E';
COMMENT ON COLUMN claims.encounter.ambulance_dropoff_location_name IS
    'Ambulance dropoff location name from Loop 2310F';

-- Indexes for ambulance location searches
CREATE INDEX IF NOT EXISTS idx_encounter_ambulance_pickup_state
    ON claims.encounter(ambulance_pickup_state) WHERE ambulance_pickup_state IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_encounter_ambulance_dropoff_state
    ON claims.encounter(ambulance_dropoff_state) WHERE ambulance_dropoff_state IS NOT NULL;

-- Phase 5.2: DME Claims (CR3, SV5 Segments)
-- Add DME certification fields to service_line table
ALTER TABLE claims.service_line
    ADD COLUMN IF NOT EXISTS dme_certification_condition VARCHAR(1),
    ADD COLUMN IF NOT EXISTS dme_duration NUMERIC(15,3),
    ADD COLUMN IF NOT EXISTS dme_duration_unit VARCHAR(3),
    ADD COLUMN IF NOT EXISTS dme_certification_revision_date DATE;

COMMENT ON COLUMN claims.service_line.dme_certification_condition IS
    'DME certification condition code from CR3 segment';
COMMENT ON COLUMN claims.service_line.dme_duration IS
    'DME duration from CR3 segment (e.g., rental period)';
COMMENT ON COLUMN claims.service_line.dme_duration_unit IS
    'DME duration unit qualifier (MO=Months, DA=Days, etc.)';
COMMENT ON COLUMN claims.service_line.dme_certification_revision_date IS
    'Date DME certification was revised from CR3 segment';

-- Index for DME certification queries
CREATE INDEX IF NOT EXISTS idx_service_line_dme_certification
    ON claims.service_line(dme_certification_condition) WHERE dme_certification_condition IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_service_line_dme_revision_date
    ON claims.service_line(dme_certification_revision_date) WHERE dme_certification_revision_date IS NOT NULL;

-- Phase 5.3: Home Health Claims (CR7, HSD Segments)
-- Create home_health_plan table for Loop 2305
CREATE TABLE IF NOT EXISTS claims.home_health_plan (
    plan_id BIGINT GENERATED ALWAYS AS IDENTITY (CACHE 100) PRIMARY KEY,
    encounter_id BIGINT NOT NULL REFERENCES claims.encounter(encounter_id) ON DELETE CASCADE,

    -- CR7 Segment Elements
    discipline_type_code VARCHAR(2), -- AI=Skilled Nursing, PT=Physical Therapy, etc.
    total_visits NUMERIC(15,3), -- Total visits prescribed
    visit_period_count NUMERIC(15,3), -- Period for visits
    visit_period_qualifier VARCHAR(2), -- DA=Days, WK=Weeks, MO=Months
    prognosis_code VARCHAR(1), -- 1=Excellent, 2=Good, 3=Fair, 4=Poor, 5=Guarded

    -- HSD Segment Elements (Health Care Services Delivery)
    frequency_count NUMERIC(15,3), -- Frequency of service
    frequency_period_qualifier VARCHAR(2), -- DA=Daily, WK=Weekly, MO=Monthly
    delivery_frequency_code VARCHAR(1), -- 1=First Week, 2=Second Week, etc.
    delivery_pattern_time_code VARCHAR(2), -- Pattern of delivery

    -- Audit trail
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for home_health_plan
CREATE INDEX IF NOT EXISTS idx_home_health_plan_encounter
    ON claims.home_health_plan(encounter_id);

CREATE INDEX IF NOT EXISTS idx_home_health_plan_discipline
    ON claims.home_health_plan(discipline_type_code);

COMMENT ON TABLE claims.home_health_plan IS
    'Home health care plan information from Loop 2305 (CR7 and HSD segments)';

COMMENT ON COLUMN claims.home_health_plan.discipline_type_code IS
    'Discipline type: AI=Skilled Nursing, PT=Physical Therapy, OT=Occupational Therapy, ST=Speech Therapy, etc.';

COMMENT ON COLUMN claims.home_health_plan.prognosis_code IS
    'Patient prognosis: 1=Excellent, 2=Good, 3=Fair, 4=Poor, 5=Guarded';

-- Trigger for updated_at on home_health_plan
CREATE TRIGGER update_home_health_plan_updated_at
    BEFORE UPDATE ON claims.home_health_plan
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Phase 5.4: Spinal Manipulation (CR2 Segment)
-- Add chiropractic/spinal manipulation columns to encounter table
ALTER TABLE claims.encounter
    ADD COLUMN IF NOT EXISTS spinal_manipulation_count INTEGER,
    ADD COLUMN IF NOT EXISTS patient_condition_code_1 VARCHAR(2),
    ADD COLUMN IF NOT EXISTS patient_condition_code_2 VARCHAR(2),
    ADD COLUMN IF NOT EXISTS patient_condition_description VARCHAR(80),
    ADD COLUMN IF NOT EXISTS subluxation_level_code VARCHAR(3),
    ADD COLUMN IF NOT EXISTS subluxation_level_code_2 VARCHAR(3);

COMMENT ON COLUMN claims.encounter.spinal_manipulation_count IS
    'Number of spinal manipulation treatments from CR2 segment';
COMMENT ON COLUMN claims.encounter.patient_condition_code_1 IS
    'Patient condition code 1 from CR2 segment (e.g., acute, chronic)';
COMMENT ON COLUMN claims.encounter.patient_condition_code_2 IS
    'Patient condition code 2 from CR2 segment';
COMMENT ON COLUMN claims.encounter.patient_condition_description IS
    'Free-form description of patient condition from CR2 segment';
COMMENT ON COLUMN claims.encounter.subluxation_level_code IS
    'Subluxation level code from CR2 segment (e.g., C1, T5, L3)';

-- Index for chiropractic claims
CREATE INDEX IF NOT EXISTS idx_encounter_spinal_manipulation
    ON claims.encounter(spinal_manipulation_count) WHERE spinal_manipulation_count IS NOT NULL;

-- Phase 5.5: Oxygen Therapy (CR5 Segment)
-- Add oxygen therapy columns to service_line table
ALTER TABLE claims.service_line
    ADD COLUMN IF NOT EXISTS oxygen_equipment_type VARCHAR(3),
    ADD COLUMN IF NOT EXISTS oxygen_flow_rate NUMERIC(15,3),
    ADD COLUMN IF NOT EXISTS daily_oxygen_use_count NUMERIC(15,3),
    ADD COLUMN IF NOT EXISTS oxygen_use_period_hour_count INTEGER,
    ADD COLUMN IF NOT EXISTS arterial_blood_gas_quantity NUMERIC(15,3),
    ADD COLUMN IF NOT EXISTS oxygen_saturation_quantity NUMERIC(15,3),
    ADD COLUMN IF NOT EXISTS oxygen_test_condition_code VARCHAR(1),
    ADD COLUMN IF NOT EXISTS oxygen_test_findings_code_1 VARCHAR(2),
    ADD COLUMN IF NOT EXISTS oxygen_test_findings_code_2 VARCHAR(2),
    ADD COLUMN IF NOT EXISTS oxygen_test_findings_code_3 VARCHAR(2),
    ADD COLUMN IF NOT EXISTS oxygen_delivery_system_code VARCHAR(1),
    ADD COLUMN IF NOT EXISTS oxygen_test_date DATE;

COMMENT ON COLUMN claims.service_line.oxygen_equipment_type IS
    'Oxygen equipment type code from CR5 segment';
COMMENT ON COLUMN claims.service_line.oxygen_flow_rate IS
    'Oxygen flow rate in liters per minute from CR5 segment';
COMMENT ON COLUMN claims.service_line.daily_oxygen_use_count IS
    'Daily oxygen usage count from CR5 segment';
COMMENT ON COLUMN claims.service_line.oxygen_use_period_hour_count IS
    'Hours per day oxygen is used from CR5 segment';
COMMENT ON COLUMN claims.service_line.arterial_blood_gas_quantity IS
    'Arterial blood gas test result from CR5 segment';
COMMENT ON COLUMN claims.service_line.oxygen_saturation_quantity IS
    'Oxygen saturation percentage from CR5 segment';
COMMENT ON COLUMN claims.service_line.oxygen_test_condition_code IS
    'Test condition code: R=Rest, E=Exercise, S=Sleep';

-- Index for oxygen therapy queries
CREATE INDEX IF NOT EXISTS idx_service_line_oxygen_equipment
    ON claims.service_line(oxygen_equipment_type) WHERE oxygen_equipment_type IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_service_line_oxygen_test_date
    ON claims.service_line(oxygen_test_date) WHERE oxygen_test_date IS NOT NULL;

-- Phase 5.6: Attachment Information (PWK Segment)
-- Create claim_attachment table for Loops 2300 and 2400
CREATE TABLE IF NOT EXISTS claims.claim_attachment (
    attachment_id BIGINT GENERATED ALWAYS AS IDENTITY (CACHE 100) PRIMARY KEY,
    encounter_id BIGINT REFERENCES claims.encounter(encounter_id) ON DELETE CASCADE,
    service_line_id BIGINT REFERENCES claims.service_line(service_line_id) ON DELETE CASCADE,

    -- PWK Segment Elements
    attachment_report_type_code VARCHAR(2) NOT NULL, -- 03=Report Justifying Treatment, 04=Drugs Administered, etc.
    attachment_transmission_code VARCHAR(2), -- AA=Available on Request, BM=By Mail, EL=Electronically, etc.
    attachment_control_number VARCHAR(80), -- Identification number
    attachment_description VARCHAR(80), -- Free-form description

    -- Identification code qualifier and identifier (from PWK02-PWK03)
    identification_code_qualifier VARCHAR(2), -- AC=Attachment Control Number, etc.
    identification_code VARCHAR(80),

    -- Audit trail
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,

    -- Constraints: Must reference either encounter OR service_line (or both)
    CONSTRAINT chk_attachment_reference CHECK (
        (encounter_id IS NOT NULL) OR (service_line_id IS NOT NULL)
    )
);

-- Indexes for claim_attachment
CREATE INDEX IF NOT EXISTS idx_claim_attachment_encounter
    ON claims.claim_attachment(encounter_id) WHERE encounter_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_claim_attachment_service_line
    ON claims.claim_attachment(service_line_id) WHERE service_line_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_claim_attachment_report_type
    ON claims.claim_attachment(attachment_report_type_code);

CREATE INDEX IF NOT EXISTS idx_claim_attachment_control_number
    ON claims.claim_attachment(attachment_control_number) WHERE attachment_control_number IS NOT NULL;

COMMENT ON TABLE claims.claim_attachment IS
    'Attachment and paperwork information from PWK segments in Loops 2300 and 2400. Indicates supporting documentation for claims.';

COMMENT ON COLUMN claims.claim_attachment.attachment_report_type_code IS
    'Report type: 03=Report Justifying Treatment, 04=Drugs Administered, 05=Treatment Diagnosis, 06=Initial Assessment, etc.';

COMMENT ON COLUMN claims.claim_attachment.attachment_transmission_code IS
    'Transmission method: AA=Available on Request, BM=By Mail, EL=Electronically, EM=E-Mail, FX=By Fax';

-- Validation: Ensure non-negative numeric values where applicable
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'chk_spinal_manipulation_count_nonneg') THEN
        ALTER TABLE claims.encounter
            ADD CONSTRAINT chk_spinal_manipulation_count_nonneg
                CHECK (spinal_manipulation_count IS NULL OR spinal_manipulation_count >= 0);
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'chk_dme_duration_nonneg') THEN
        ALTER TABLE claims.service_line
            ADD CONSTRAINT chk_dme_duration_nonneg
                CHECK (dme_duration IS NULL OR dme_duration >= 0);
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'chk_oxygen_flow_rate_nonneg') THEN
        ALTER TABLE claims.service_line
            ADD CONSTRAINT chk_oxygen_flow_rate_nonneg
                CHECK (oxygen_flow_rate IS NULL OR oxygen_flow_rate >= 0);
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'chk_oxygen_use_hours_nonneg') THEN
        ALTER TABLE claims.service_line
            ADD CONSTRAINT chk_oxygen_use_hours_nonneg
                CHECK (oxygen_use_period_hour_count IS NULL OR oxygen_use_period_hour_count >= 0);
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'chk_home_health_visits_nonneg') THEN
        ALTER TABLE claims.home_health_plan
            ADD CONSTRAINT chk_home_health_visits_nonneg
                CHECK (total_visits IS NULL OR total_visits >= 0);
    END IF;
END$$;

-- Performance: Create composite indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_encounter_ambulance_route
    ON claims.encounter(ambulance_pickup_state, ambulance_dropoff_state)
    WHERE ambulance_pickup_state IS NOT NULL AND ambulance_dropoff_state IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_home_health_encounter_discipline
    ON claims.home_health_plan(encounter_id, discipline_type_code);

CREATE INDEX IF NOT EXISTS idx_claim_attachment_encounter_type
    ON claims.claim_attachment(encounter_id, attachment_report_type_code)
    WHERE encounter_id IS NOT NULL;
