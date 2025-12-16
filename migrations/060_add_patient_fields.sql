-- Migration: 060_add_patient_fields
-- Description: Add patient fields to encounter table when patient is different from subscriber
-- Date: 2025-12-15
-- Related: Fix Priority 2 - Capture Patient when different from Subscriber (NM1*QC in Loop 2010CA)

-- Add patient fields to encounter table
-- These fields are populated when the patient is different from the subscriber
-- (i.e., when HL level 23 / Loop 2000C exists in the 837P)

ALTER TABLE claims.encounter
    ADD COLUMN IF NOT EXISTS patient_last_name VARCHAR(255),
    ADD COLUMN IF NOT EXISTS patient_first_name VARCHAR(255),
    ADD COLUMN IF NOT EXISTS patient_middle_name VARCHAR(255),
    ADD COLUMN IF NOT EXISTS patient_name_suffix VARCHAR(50),
    ADD COLUMN IF NOT EXISTS patient_date_of_birth DATE,
    ADD COLUMN IF NOT EXISTS patient_gender CHAR(1),
    ADD COLUMN IF NOT EXISTS patient_address_line1 VARCHAR(255),
    ADD COLUMN IF NOT EXISTS patient_address_line2 VARCHAR(255),
    ADD COLUMN IF NOT EXISTS patient_city VARCHAR(100),
    ADD COLUMN IF NOT EXISTS patient_state CHAR(2),
    ADD COLUMN IF NOT EXISTS patient_postal_code VARCHAR(15),
    ADD COLUMN IF NOT EXISTS patient_relationship_code VARCHAR(3);

-- Add comments for documentation
COMMENT ON COLUMN claims.encounter.patient_last_name IS 'Patient last name from NM1*QC segment when patient differs from subscriber';
COMMENT ON COLUMN claims.encounter.patient_first_name IS 'Patient first name from NM1*QC segment when patient differs from subscriber';
COMMENT ON COLUMN claims.encounter.patient_middle_name IS 'Patient middle name from NM1*QC segment';
COMMENT ON COLUMN claims.encounter.patient_name_suffix IS 'Patient name suffix (Jr, Sr, III, etc) from NM1*QC segment';
COMMENT ON COLUMN claims.encounter.patient_date_of_birth IS 'Patient DOB from DMG segment in Loop 2010CA';
COMMENT ON COLUMN claims.encounter.patient_gender IS 'Patient gender from DMG segment in Loop 2010CA (M/F/U)';
COMMENT ON COLUMN claims.encounter.patient_address_line1 IS 'Patient address from N3 segment in Loop 2010CA';
COMMENT ON COLUMN claims.encounter.patient_address_line2 IS 'Patient address line 2 from N3 segment in Loop 2010CA';
COMMENT ON COLUMN claims.encounter.patient_city IS 'Patient city from N4 segment in Loop 2010CA';
COMMENT ON COLUMN claims.encounter.patient_state IS 'Patient state from N4 segment in Loop 2010CA';
COMMENT ON COLUMN claims.encounter.patient_postal_code IS 'Patient postal code from N4 segment in Loop 2010CA';
COMMENT ON COLUMN claims.encounter.patient_relationship_code IS 'Patient relationship to subscriber from PAT01 segment';

-- Create index for patient name searches
CREATE INDEX IF NOT EXISTS idx_encounter_patient_name
    ON claims.encounter(patient_last_name, patient_first_name)
    WHERE patient_last_name IS NOT NULL;
