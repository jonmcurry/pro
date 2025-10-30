-- Migration 035: Add medical_record_number column to encounter table
-- This column stores the patient's medical record number (MRN) from REF*EA segment (Loop 2010BA)
-- REF*EA is the Medical Record Number which is a static patient identifier across encounters

ALTER TABLE claims.encounter
ADD COLUMN medical_record_number VARCHAR(80);

-- Create an index for searching by MRN
CREATE INDEX idx_encounter_mrn ON claims.encounter(medical_record_number);

-- Create a composite index for tracking patient encounters by MRN + DOS
CREATE INDEX idx_encounter_mrn_dos ON claims.encounter(medical_record_number, date_of_service_from)
    WHERE medical_record_number IS NOT NULL;

-- Add comment explaining the field
COMMENT ON COLUMN claims.encounter.medical_record_number IS 'Patient Medical Record Number (MRN) from REF*EA segment in Loop 2010BA. Static patient identifier across encounters, distinct from patient_control_number (CLM01) which is unique per encounter.';
