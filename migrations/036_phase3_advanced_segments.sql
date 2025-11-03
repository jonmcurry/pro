-- Migration: 036_phase3_advanced_segments
-- Description: Add columns for Phase 3 - Advanced Segments (REF, PRV, NTE, CRC, AMT)
-- Date: 2025-11-03
-- Related: Phase 3 of 837P Full Implementation Action Plan

-- Phase 3.1: Reference numbers (REF segments)
-- Store additional reference qualifiers as JSONB key-value pairs
ALTER TABLE claims.encounter
    ADD COLUMN IF NOT EXISTS reference_numbers JSONB;

COMMENT ON COLUMN claims.encounter.reference_numbers IS
    'Additional reference numbers from REF segments stored as JSONB. Example: {"9F": "REF123", "F8": "PAYER456", "D9": "CLM789"}';

-- Phase 3.2: Provider taxonomy codes (PRV segments)
-- Add taxonomy columns for provider specialty identification
ALTER TABLE claims.encounter
    ADD COLUMN IF NOT EXISTS rendering_provider_taxonomy VARCHAR(10),
    ADD COLUMN IF NOT EXISTS referring_provider_taxonomy VARCHAR(10),
    ADD COLUMN IF NOT EXISTS supervising_provider_taxonomy VARCHAR(10);

ALTER TABLE claims.service_line
    ADD COLUMN IF NOT EXISTS rendering_provider_taxonomy VARCHAR(10);

COMMENT ON COLUMN claims.encounter.rendering_provider_taxonomy IS
    'Provider taxonomy code from PRV segment (e.g., 207Q00000X for Family Medicine)';
COMMENT ON COLUMN claims.encounter.referring_provider_taxonomy IS
    'Referring provider taxonomy code from PRV segment';
COMMENT ON COLUMN claims.encounter.supervising_provider_taxonomy IS
    'Supervising provider taxonomy code from PRV segment';
COMMENT ON COLUMN claims.service_line.rendering_provider_taxonomy IS
    'Service line level rendering provider taxonomy code';

-- Phase 3.3: Enhanced note tracking (NTE segments)
-- Add columns to track multiple notes with sequence
ALTER TABLE claims.encounter_note
    ADD COLUMN IF NOT EXISTS note_sequence INTEGER,
    ADD COLUMN IF NOT EXISTS note_qualifier VARCHAR(10);

COMMENT ON COLUMN claims.encounter_note.note_sequence IS
    'Sequence number for ordering multiple notes from NTE segments';
COMMENT ON COLUMN claims.encounter_note.note_qualifier IS
    'NTE segment qualifier code indicating note type';

-- Create index for note sequence ordering
CREATE INDEX IF NOT EXISTS idx_encounter_note_sequence
    ON claims.encounter_note(encounter_id, note_sequence);

-- Phase 3.4: Condition codes (CRC segments)
-- Store condition indicators as JSONB array
ALTER TABLE claims.encounter
    ADD COLUMN IF NOT EXISTS condition_codes JSONB;

COMMENT ON COLUMN claims.encounter.condition_codes IS
    'Condition codes from CRC segments stored as JSONB array. Example: [{"qualifier": "07", "conditions": ["38", "04"]}, {"qualifier": "E1", "conditions": ["AV"]}]';

-- Phase 3.5: Supplemental amounts (AMT segments)
-- Add amount fields for financial reconciliation
ALTER TABLE claims.encounter
    ADD COLUMN IF NOT EXISTS non_covered_charges NUMERIC(18,2),
    ADD COLUMN IF NOT EXISTS patient_responsibility_amount NUMERIC(18,2);

ALTER TABLE claims.service_line
    ADD COLUMN IF NOT EXISTS approved_amount NUMERIC(18,2),
    ADD COLUMN IF NOT EXISTS non_covered_charges NUMERIC(18,2);

COMMENT ON COLUMN claims.encounter.non_covered_charges IS
    'Non-covered charges from AMT*A8 segment';
COMMENT ON COLUMN claims.encounter.patient_responsibility_amount IS
    'Patient responsibility amount from AMT*F5 segment';
COMMENT ON COLUMN claims.service_line.approved_amount IS
    'Approved amount from AMT*T segment at service line level';
COMMENT ON COLUMN claims.service_line.non_covered_charges IS
    'Non-covered charges from AMT*A8 segment at service line level';

-- Create GIN indexes for JSONB columns to enable efficient querying
CREATE INDEX IF NOT EXISTS idx_encounter_reference_numbers_gin
    ON claims.encounter USING GIN (reference_numbers);

CREATE INDEX IF NOT EXISTS idx_encounter_condition_codes_gin
    ON claims.encounter USING GIN (condition_codes);

-- Add indexes for new numeric columns
CREATE INDEX IF NOT EXISTS idx_encounter_non_covered_charges
    ON claims.encounter(non_covered_charges) WHERE non_covered_charges IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_encounter_patient_responsibility
    ON claims.encounter(patient_responsibility_amount) WHERE patient_responsibility_amount IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_service_line_approved_amount
    ON claims.service_line(approved_amount) WHERE approved_amount IS NOT NULL;

-- Validation: Ensure non-negative amounts
-- Note: Using DO block to handle constraint existence
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'chk_non_covered_charges_nonneg') THEN
        ALTER TABLE claims.encounter
            ADD CONSTRAINT chk_non_covered_charges_nonneg
                CHECK (non_covered_charges IS NULL OR non_covered_charges >= 0);
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'chk_patient_responsibility_nonneg') THEN
        ALTER TABLE claims.encounter
            ADD CONSTRAINT chk_patient_responsibility_nonneg
                CHECK (patient_responsibility_amount IS NULL OR patient_responsibility_amount >= 0);
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'chk_approved_amount_nonneg') THEN
        ALTER TABLE claims.service_line
            ADD CONSTRAINT chk_approved_amount_nonneg
                CHECK (approved_amount IS NULL OR approved_amount >= 0);
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'chk_sl_non_covered_charges_nonneg') THEN
        ALTER TABLE claims.service_line
            ADD CONSTRAINT chk_sl_non_covered_charges_nonneg
                CHECK (non_covered_charges IS NULL OR non_covered_charges >= 0);
    END IF;
END$$;
