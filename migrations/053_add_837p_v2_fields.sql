-- Migration 053: Add v2.0.0 837P Comprehensive Data Fields
-- Adds fields for previously unparsed 837P segments:
-- - REF*D9 (Claim Number)
-- - AMT*F5 (Patient Responsibility Amount)
-- - CR1 (Ambulance Transport Information)
-- - PWK (Paperwork/Attachments)
-- - CRC (Condition Indicators)
-- - HCP (Health Care Pricing - at service line level)

-- Add new encounter fields
ALTER TABLE claims.encounter
ADD COLUMN IF NOT EXISTS claim_number VARCHAR(50),
ADD COLUMN IF NOT EXISTS patient_responsibility_amount NUMERIC(10,2),
ADD COLUMN IF NOT EXISTS ambulance_transport_reason_code VARCHAR(10),
ADD COLUMN IF NOT EXISTS ambulance_transport_distance NUMERIC(10,2),
ADD COLUMN IF NOT EXISTS ambulance_patient_weight NUMERIC(10,2),
ADD COLUMN IF NOT EXISTS ambulance_patient_count INTEGER,
ADD COLUMN IF NOT EXISTS paperwork_report_type VARCHAR(10),
ADD COLUMN IF NOT EXISTS paperwork_transmission_code VARCHAR(10),
ADD COLUMN IF NOT EXISTS paperwork_control_number VARCHAR(50),
ADD COLUMN IF NOT EXISTS condition_codes TEXT;

COMMENT ON COLUMN claims.encounter.claim_number IS '837P REF*D9 - Claim Identifier';
COMMENT ON COLUMN claims.encounter.patient_responsibility_amount IS '837P AMT*F5 - Patient Responsibility Amount';
COMMENT ON COLUMN claims.encounter.ambulance_transport_reason_code IS '837P CR1 - Ambulance Transport Reason Code';
COMMENT ON COLUMN claims.encounter.ambulance_transport_distance IS '837P CR1 - Ambulance Transport Distance';
COMMENT ON COLUMN claims.encounter.ambulance_patient_weight IS '837P CR1 - Patient Weight';
COMMENT ON COLUMN claims.encounter.ambulance_patient_count IS '837P CR1 - Patient Count';
COMMENT ON COLUMN claims.encounter.paperwork_report_type IS '837P PWK - Report Type Code';
COMMENT ON COLUMN claims.encounter.paperwork_transmission_code IS '837P PWK - Report Transmission Code';
COMMENT ON COLUMN claims.encounter.paperwork_control_number IS '837P PWK - Identification Code';
COMMENT ON COLUMN claims.encounter.condition_codes IS '837P CRC - Condition Indicator Codes (comma-separated)';

-- Add new service line fields (HCP segment - Health Care Pricing)
ALTER TABLE claims.service_line
ADD COLUMN IF NOT EXISTS allowed_amount NUMERIC(10,2),
ADD COLUMN IF NOT EXISTS saving_amount NUMERIC(10,2);

COMMENT ON COLUMN claims.service_line.allowed_amount IS '837P HCP - Allowed Amount from adjudication';
COMMENT ON COLUMN claims.service_line.saving_amount IS '837P HCP - Saving Amount from adjudication';

-- Add indexes for commonly queried fields
CREATE INDEX IF NOT EXISTS idx_encounter_claim_number ON claims.encounter(claim_number) WHERE claim_number IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_encounter_ambulance ON claims.encounter(ambulance_transport_reason_code) WHERE ambulance_transport_reason_code IS NOT NULL;
