-- Migration: 006_create_flag_tables
-- Description: Create flag/issue tables for rules engine
-- Date: 2025-10-14

-- Flag categories and definitions
CREATE TABLE claims.flag_category (
    category_id BIGINT GENERATED ALWAYS AS IDENTITY (CACHE 100) PRIMARY KEY,
    category_code VARCHAR(10) NOT NULL UNIQUE, -- COD, DOC, EMO, EMU, EMI, EMT, MOD, OTH, QTY, SUP, DX
    category_name VARCHAR(100) NOT NULL,
    category_description TEXT,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_flag_category_code ON claims.flag_category(category_code);

COMMENT ON TABLE claims.flag_category IS 'Flag category definitions (COD, DOC, EMO, EMU, etc.)';

-- Insert standard flag categories from SRD
INSERT INTO claims.flag_category (category_code, category_name, category_description) VALUES
('COD', 'Coding Issues', 'Issues related to procedure coding'),
('DOC', 'Documentation Issues', 'Missing or insufficient documentation'),
('EMO', 'E/M Over-coded', 'E/M codes billed higher than supported'),
('EMU', 'E/M Under-coded', 'E/M codes billed lower than supported'),
('EMI', 'E/M Incorrect Category', 'Wrong E/M category used'),
('EMT', 'E/M Time Not Documented', 'Time-based E/M without time documentation'),
('MOD', 'Modifier Issues', 'Incorrect, missing, or unnecessary modifiers'),
('OTH', 'Other Issues', 'Provider, date, or signature issues'),
('QTY', 'Quantity Issues', 'Unit count discrepancies'),
('SUP', 'Supervision Requirements', 'Incident-to, split-shared, teaching physician issues'),
('DX', 'Diagnosis Issues', 'Diagnosis code problems');

-- Flag issue definitions
CREATE TABLE claims.flag_issue (
    issue_id BIGINT GENERATED ALWAYS AS IDENTITY (CACHE 100) PRIMARY KEY,
    category_id BIGINT NOT NULL REFERENCES claims.flag_category(category_id),
    issue_code VARCHAR(20) NOT NULL UNIQUE,
    issue_description TEXT NOT NULL,
    severity VARCHAR(20) DEFAULT 'MEDIUM', -- HIGH, MEDIUM, LOW
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_flag_issue_category ON claims.flag_issue(category_id);
CREATE INDEX idx_flag_issue_code ON claims.flag_issue(issue_code);

COMMENT ON TABLE claims.flag_issue IS 'Specific flag issue definitions with descriptions';

-- Insert standard flag issues from SRD
INSERT INTO claims.flag_issue (category_id, issue_code, issue_description, severity) VALUES
-- COD: Coding Issues
((SELECT category_id FROM claims.flag_category WHERE category_code = 'COD'), 'COD_BUNDLED', 'Bundled Service/Procedure - Documentation indicates service/procedure billed was performed but is part of, or bundled into, another service', 'HIGH'),
((SELECT category_id FROM claims.flag_category WHERE category_code = 'COD'), 'COD_INCORRECT', 'Incorrect Procedure Code - Documentation supports a different service/procedure than was billed', 'HIGH'),
((SELECT category_id FROM claims.flag_category WHERE category_code = 'COD'), 'COD_MISSED', 'Missed Charge - Documentation supports a service/procedure that was not billed', 'MEDIUM'),
((SELECT category_id FROM claims.flag_category WHERE category_code = 'COD'), 'COD_TIME_MISSING', 'Time Not Documented - Documentation does not include time and the service billed was a time-based service (non-E/M)', 'MEDIUM'),

-- DOC: Documentation Issues
((SELECT category_id FROM claims.flag_category WHERE category_code = 'DOC'), 'DOC_MISSING', 'Missing Documentation - No documentation found/provided for service billed', 'HIGH'),
((SELECT category_id FROM claims.flag_category WHERE category_code = 'DOC'), 'DOC_LIMITED', 'Limited/Insufficient Documentation - Documentation is not sufficient to support the billed service (not including missing signature)', 'HIGH'),

-- EMO: E/M Over-coded
((SELECT category_id FROM claims.flag_category WHERE category_code = 'EMO'), 'EMO_ONE_LEVEL', 'E/M Over - One Level - Documentation supports an E/M code one level lower than was billed', 'HIGH'),
((SELECT category_id FROM claims.flag_category WHERE category_code = 'EMO'), 'EMO_TWO_PLUS', 'E/M Over - Two Levels or More - Documentation supports an E/M code two or more levels lower than was billed', 'HIGH'),

-- EMU: E/M Under-coded
((SELECT category_id FROM claims.flag_category WHERE category_code = 'EMU'), 'EMU_ONE_LEVEL', 'E/M Under - One Level - Documentation supports an E/M code one level higher than was billed', 'MEDIUM'),
((SELECT category_id FROM claims.flag_category WHERE category_code = 'EMU'), 'EMU_TWO_PLUS', 'E/M Under - Two Levels or More - Documentation supports an E/M code two or more levels higher than was billed', 'MEDIUM'),

-- EMI: E/M Incorrect Category
((SELECT category_id FROM claims.flag_category WHERE category_code = 'EMI'), 'EMI_CATEGORY', 'E/M Incorrect Category - Documentation supports a different E/M category than was billed', 'HIGH'),

-- EMT: E/M Time Not Documented
((SELECT category_id FROM claims.flag_category WHERE category_code = 'EMT'), 'EMT_TIME', 'E/M Time - Not Documented - Documentation does not include time and the service billed was a time-based service or coded based on time', 'MEDIUM'),

-- MOD: Modifier Issues
((SELECT category_id FROM claims.flag_category WHERE category_code = 'MOD'), 'MOD_INCORRECT', 'Incorrect Modifier - Modifier reported is incorrect (a different modifier is required based on the documentation)', 'MEDIUM'),
((SELECT category_id FROM claims.flag_category WHERE category_code = 'MOD'), 'MOD_MISSING', 'Missing Modifier - Modifier is required for the billed service but was not reported', 'MEDIUM'),
((SELECT category_id FROM claims.flag_category WHERE category_code = 'MOD'), 'MOD_UNNECESSARY', 'Unnecessary Modifier - Modifier reported is not required for the procedure(s) billed', 'LOW'),

-- OTH: Other Issues
((SELECT category_id FROM claims.flag_category WHERE category_code = 'OTH'), 'OTH_PROVIDER', 'Incorrect Provider - Documentation indicates a different provider rendered the service than was billed', 'HIGH'),
((SELECT category_id FROM claims.flag_category WHERE category_code = 'OTH'), 'OTH_DOS', 'Incorrect Date of Service - Documentation indicates the service was performed on a different date of service than was billed', 'HIGH'),
((SELECT category_id FROM claims.flag_category WHERE category_code = 'OTH'), 'OTH_SIGNATURE', 'Missing Provider Signature - Documentation is missing the provider''s signature', 'HIGH'),

-- QTY: Quantity Issues
((SELECT category_id FROM claims.flag_category WHERE category_code = 'QTY'), 'QTY_FEWER', 'Fewer Units Supported - Documentation supports fewer units than were billed', 'HIGH'),
((SELECT category_id FROM claims.flag_category WHERE category_code = 'QTY'), 'QTY_MORE', 'More Units Supported - Documentation supports more units than were billed', 'MEDIUM'),

-- SUP: Supervision Requirements
((SELECT category_id FROM claims.flag_category WHERE category_code = 'SUP'), 'SUP_INCIDENT_TO', 'Incident to / Split Shared Requirements Not Met - Documentation does not support incident to / split shared billing requirements', 'HIGH'),
((SELECT category_id FROM claims.flag_category WHERE category_code = 'SUP'), 'SUP_TEACHING', 'Teaching Physician Guideline Requirements Not Met - Documentation does not support teaching physician guidelines billing requirements', 'HIGH'),
((SELECT category_id FROM claims.flag_category WHERE category_code = 'SUP'), 'SUP_SUPERVISION', 'Supervision Requirements Not Met - Documentation does not support required level of supervision for service type provided', 'HIGH'),

-- DX: Diagnosis Issues
((SELECT category_id FROM claims.flag_category WHERE category_code = 'DX'), 'DX_ADDITIONAL', 'Additional Diagnosis - Additional ICD-10-CM code(s) are documented but were not reported', 'MEDIUM'),
((SELECT category_id FROM claims.flag_category WHERE category_code = 'DX'), 'DX_UNSUPPORTED', 'Documentation Unsupported - Documentation does not support the reported diagnosis code(s)', 'HIGH'),
((SELECT category_id FROM claims.flag_category WHERE category_code = 'DX'), 'DX_INCORRECT', 'Incorrect Code - ICD-10-CM code(s) reported are not correct based upon documentation', 'HIGH'),
((SELECT category_id FROM claims.flag_category WHERE category_code = 'DX'), 'DX_SPECIFICITY', 'Specificity Issue - Documentation supports a more specific diagnosis code(s) than that reported', 'MEDIUM');

-- Flag assignments to encounters
CREATE TABLE claims.encounter_flag (
    flag_id BIGINT GENERATED ALWAYS AS IDENTITY (CACHE 100) PRIMARY KEY,
    encounter_id BIGINT NOT NULL REFERENCES claims.encounter(encounter_id) ON DELETE CASCADE,
    issue_id BIGINT NOT NULL REFERENCES claims.flag_issue(issue_id),

    -- Flag details
    flag_type VARCHAR(20) DEFAULT 'POST_BILL', -- POST_BILL, PRE_BILL
    severity VARCHAR(20),
    flag_reason TEXT,
    flagged_element VARCHAR(255), -- What was flagged (e.g., "CPT 99214", "Modifier 25")

    -- Proposed changes (if applicable)
    proposed_code VARCHAR(50),
    proposed_modifier VARCHAR(10),
    proposed_quantity NUMERIC(15,3),
    proposed_diagnosis_code VARCHAR(30),

    -- Status
    flag_status VARCHAR(20) DEFAULT 'OPEN', -- OPEN, REVIEWED, ACCEPTED, REJECTED, RESOLVED
    resolution_note TEXT,
    resolved_at TIMESTAMPTZ,
    resolved_by VARCHAR(100),

    -- Audit trail
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100) DEFAULT 'SYSTEM'
);

CREATE INDEX idx_encounter_flag_encounter ON claims.encounter_flag(encounter_id);
CREATE INDEX idx_encounter_flag_issue ON claims.encounter_flag(issue_id);
CREATE INDEX idx_encounter_flag_status ON claims.encounter_flag(flag_status);
CREATE INDEX idx_encounter_flag_type ON claims.encounter_flag(flag_type);
CREATE INDEX idx_encounter_flag_severity ON claims.encounter_flag(severity);
CREATE INDEX idx_encounter_flag_created ON claims.encounter_flag(created_at);

-- Composite indexes for common queries
CREATE INDEX idx_encounter_flag_enc_status ON claims.encounter_flag(encounter_id, flag_status);
CREATE INDEX idx_encounter_flag_status_created ON claims.encounter_flag(flag_status, created_at)
    WHERE flag_status = 'OPEN';

COMMENT ON TABLE claims.encounter_flag IS 'Flags assigned to encounters by the rules engine';

-- Flag assignments to service lines
CREATE TABLE claims.service_line_flag (
    flag_id BIGINT GENERATED ALWAYS AS IDENTITY (CACHE 100) PRIMARY KEY,
    service_line_id BIGINT NOT NULL REFERENCES claims.service_line(service_line_id) ON DELETE CASCADE,
    issue_id BIGINT NOT NULL REFERENCES claims.flag_issue(issue_id),

    -- Flag details
    flag_type VARCHAR(20) DEFAULT 'POST_BILL', -- POST_BILL, PRE_BILL
    severity VARCHAR(20),
    flag_reason TEXT,
    flagged_element VARCHAR(255),

    -- Proposed changes
    proposed_code VARCHAR(50),
    proposed_modifier VARCHAR(10),
    proposed_quantity NUMERIC(15,3),

    -- Status
    flag_status VARCHAR(20) DEFAULT 'OPEN',
    resolution_note TEXT,
    resolved_at TIMESTAMPTZ,
    resolved_by VARCHAR(100),

    -- Audit trail
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100) DEFAULT 'SYSTEM'
);

CREATE INDEX idx_service_line_flag_line ON claims.service_line_flag(service_line_id);
CREATE INDEX idx_service_line_flag_issue ON claims.service_line_flag(issue_id);
CREATE INDEX idx_service_line_flag_status ON claims.service_line_flag(flag_status);
CREATE INDEX idx_service_line_flag_type ON claims.service_line_flag(flag_type);
CREATE INDEX idx_service_line_flag_created ON claims.service_line_flag(created_at);

COMMENT ON TABLE claims.service_line_flag IS 'Flags assigned to service lines by the rules engine';
