-- Migration: 061_fix_patient_relationship_code_length
-- Description: Widen patient_relationship_code from VARCHAR(2) to VARCHAR(3)
-- Date: 2025-12-15
-- Related: Fix "value too long for type character varying(2)" error
-- PAT01 relationship codes can be up to 3 characters in some implementations

ALTER TABLE claims.encounter
    ALTER COLUMN patient_relationship_code TYPE VARCHAR(3);
