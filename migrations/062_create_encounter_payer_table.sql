-- Migration: 062_create_encounter_payer_table
-- Description: Create encounter_payer table to track all payers across claim submissions
-- Date: 2025-12-16
-- Purpose: Track billing history - which payers were billed, when, and payment status

-- Create encounter_payer table
-- This table tracks ALL payers for an encounter across multiple submissions
-- When a claim is reprocessed with different SBR segments, new rows are added
CREATE TABLE IF NOT EXISTS claims.encounter_payer (
    encounter_payer_id BIGINT GENERATED ALWAYS AS IDENTITY (CACHE 100) PRIMARY KEY,
    encounter_id BIGINT NOT NULL REFERENCES claims.encounter(encounter_id) ON DELETE CASCADE,

    -- Payer identification (from SBR segment and NM1*PR)
    payer_responsibility_code CHAR(1) NOT NULL,  -- P=Primary, S=Secondary, T=Tertiary (SBR01)
    payer_id VARCHAR(80),                        -- Payer identifier (NM1*PR09)
    payer_name VARCHAR(255),                     -- Payer name (NM1*PR03)
    claim_filing_indicator VARCHAR(2),           -- SBR09: CI, BL, MB, MC, MA, etc.

    -- Status tracking
    is_billing_payer BOOLEAN DEFAULT false,      -- TRUE = this is the payer being billed (first SBR in file)
    submitted_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,  -- When we processed this submission

    -- Payment info (populated for payers that already paid - from Loop 2320)
    paid_amount NUMERIC(18,2),                   -- Amount paid by this payer (AMT*D)
    claim_control_number VARCHAR(50),            -- Payer's claim reference number (REF*F8)

    -- Billing provider for this submission (usually same as encounter, but could differ)
    billing_provider_id BIGINT REFERENCES claims.provider(provider_id),

    -- Audit trail
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,

    -- Constraints
    CONSTRAINT chk_payer_responsibility_code CHECK (payer_responsibility_code IN ('P', 'S', 'T'))
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_encounter_payer_encounter_id
    ON claims.encounter_payer(encounter_id);

CREATE INDEX IF NOT EXISTS idx_encounter_payer_payer_id
    ON claims.encounter_payer(payer_id)
    WHERE payer_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_encounter_payer_billing
    ON claims.encounter_payer(encounter_id, is_billing_payer)
    WHERE is_billing_payer = true;

CREATE INDEX IF NOT EXISTS idx_encounter_payer_submitted
    ON claims.encounter_payer(submitted_at DESC);

CREATE INDEX IF NOT EXISTS idx_encounter_payer_responsibility
    ON claims.encounter_payer(payer_responsibility_code);

-- Comments
COMMENT ON TABLE claims.encounter_payer IS
    'Tracks all payers for an encounter across multiple claim submissions. Each row represents a payer from an SBR segment.';

COMMENT ON COLUMN claims.encounter_payer.payer_responsibility_code IS
    'Payer sequence from SBR01: P=Primary, S=Secondary, T=Tertiary';

COMMENT ON COLUMN claims.encounter_payer.is_billing_payer IS
    'TRUE if this payer was the one being billed (first SBR in the EDI file). FALSE for other payers in COB Loop 2320.';

COMMENT ON COLUMN claims.encounter_payer.submitted_at IS
    'Timestamp when this payer record was processed/imported';

COMMENT ON COLUMN claims.encounter_payer.paid_amount IS
    'Amount already paid by this payer (from AMT*D in Loop 2320). NULL if payer has not yet paid.';

COMMENT ON COLUMN claims.encounter_payer.claim_control_number IS
    'Payer claim reference number (from REF*F8 in Loop 2320). Used to track the claim in the payer system.';
