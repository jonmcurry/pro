-- Migration: 010_create_denial_tables
-- Description: Create denial tracking and appeals tables
-- Date: 2025-10-14

-- Denial events from payer remittances
CREATE TABLE claims.denial_event (
    denial_id BIGINT GENERATED ALWAYS AS IDENTITY (CACHE 100) PRIMARY KEY,
    encounter_id BIGINT NOT NULL REFERENCES claims.encounter(encounter_id) ON DELETE CASCADE,
    service_line_id BIGINT REFERENCES claims.service_line(service_line_id) ON DELETE CASCADE,
    organization_id BIGINT NOT NULL REFERENCES claims.organization(organization_id),
    facility_id BIGINT REFERENCES claims.facility(facility_id),

    -- Denial identification
    denial_type VARCHAR(50) NOT NULL, -- CLAIM_LEVEL, LINE_LEVEL
    denial_category VARCHAR(50) NOT NULL, -- SOFT, HARD, PREVENTABLE, NON_PREVENTABLE

    -- Payer information
    payer_id VARCHAR(80),
    payer_name VARCHAR(255),
    claim_filing_indicator VARCHAR(2),

    -- Denial codes (CARC/RARC from 835)
    claim_adjustment_group_code VARCHAR(2), -- CO, CR, OA, PI, PR
    claim_adjustment_reason_code VARCHAR(5) NOT NULL, -- CARC
    remittance_advice_remark_code VARCHAR(5), -- RARC

    -- Denial reason
    denial_reason_description TEXT,
    payer_denial_reason TEXT,

    -- Financial impact
    denied_amount NUMERIC(18,2) NOT NULL CHECK (denied_amount >= 0),
    billed_amount NUMERIC(18,2),
    allowed_amount NUMERIC(18,2),
    paid_amount NUMERIC(18,2) DEFAULT 0,

    -- Dates
    service_date DATE NOT NULL,
    initial_submission_date DATE,
    denial_date DATE NOT NULL,
    received_date DATE,

    -- Remittance information
    remittance_advice_number VARCHAR(50),
    check_eft_number VARCHAR(50),

    -- Root cause analysis
    root_cause_category VARCHAR(50), -- CODING, DOCUMENTATION, AUTHORIZATION, ELIGIBILITY, TIMELY_FILING, MEDICAL_NECESSITY, DUPLICATE
    root_cause_subcategory VARCHAR(100),
    root_cause_details TEXT,

    -- Responsibility
    responsible_party VARCHAR(50), -- PROVIDER, CODER, BILLER, PATIENT, PAYER, OTHER
    coder_id BIGINT REFERENCES claims.coder(coder_id),
    provider_id BIGINT REFERENCES claims.provider(provider_id),

    -- Preventability
    is_preventable BOOLEAN,
    preventable_category VARCHAR(100),
    prevention_recommendations TEXT,

    -- Status and workflow
    denial_status VARCHAR(50) DEFAULT 'NEW', -- NEW, UNDER_REVIEW, APPEALING, CORRECTING, CLOSED, WRITTEN_OFF
    resolution_status VARCHAR(50), -- OVERTURNED, PARTIAL_PAYMENT, UPHELD, CORRECTED_RESUBMITTED, WRITTEN_OFF
    resolution_date DATE,

    -- Appeal tracking
    appeal_filed BOOLEAN DEFAULT false,
    appeal_level VARCHAR(20), -- FIRST, SECOND, THIRD, HEARING
    appeal_deadline DATE,

    -- Notes
    internal_notes TEXT,
    resolution_notes TEXT,

    -- Audit trail
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100),
    updated_by VARCHAR(100)
);

CREATE INDEX idx_denial_event_encounter ON claims.denial_event(encounter_id);
CREATE INDEX idx_denial_event_service_line ON claims.denial_event(service_line_id);
CREATE INDEX idx_denial_event_org ON claims.denial_event(organization_id);
CREATE INDEX idx_denial_event_facility ON claims.denial_event(facility_id);
CREATE INDEX idx_denial_event_payer ON claims.denial_event(payer_id);
CREATE INDEX idx_denial_event_denial_date ON claims.denial_event(denial_date);
CREATE INDEX idx_denial_event_service_date ON claims.denial_event(service_date);
CREATE INDEX idx_denial_event_status ON claims.denial_event(denial_status);
CREATE INDEX idx_denial_event_resolution ON claims.denial_event(resolution_status);
CREATE INDEX idx_denial_event_carc ON claims.denial_event(claim_adjustment_reason_code);
CREATE INDEX idx_denial_event_root_cause ON claims.denial_event(root_cause_category);
CREATE INDEX idx_denial_event_responsible ON claims.denial_event(responsible_party);
CREATE INDEX idx_denial_event_preventable ON claims.denial_event(is_preventable) WHERE is_preventable = true;
CREATE INDEX idx_denial_event_coder ON claims.denial_event(coder_id);
CREATE INDEX idx_denial_event_provider ON claims.denial_event(provider_id);
CREATE INDEX idx_denial_event_appeal_deadline ON claims.denial_event(appeal_deadline) WHERE appeal_filed = false AND denial_status NOT IN ('CLOSED', 'WRITTEN_OFF');

-- Composite indexes for common queries
CREATE INDEX idx_denial_event_org_status_date ON claims.denial_event(organization_id, denial_status, denial_date);
CREATE INDEX idx_denial_event_facility_date ON claims.denial_event(facility_id, denial_date);

COMMENT ON TABLE claims.denial_event IS 'Denial events from payer remittances with root cause analysis';

CREATE TRIGGER update_denial_event_updated_at BEFORE UPDATE ON claims.denial_event
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Appeal actions and correspondence
CREATE TABLE claims.denial_appeal (
    appeal_id BIGINT GENERATED ALWAYS AS IDENTITY (CACHE 100) PRIMARY KEY,
    denial_id BIGINT NOT NULL REFERENCES claims.denial_event(denial_id) ON DELETE CASCADE,

    -- Appeal details
    appeal_level VARCHAR(20) NOT NULL, -- FIRST, SECOND, THIRD, HEARING, ALJ, FEDERAL
    appeal_type VARCHAR(50), -- WRITTEN, PHONE, PEER_TO_PEER
    appeal_method VARCHAR(50), -- ONLINE_PORTAL, FAX, MAIL, EMAIL

    -- Dates
    filed_date DATE NOT NULL,
    due_date DATE,
    decision_date DATE,
    appeal_received_date DATE,

    -- Appeal content
    appeal_reason TEXT,
    supporting_documentation TEXT[], -- Array of document references
    clinical_rationale TEXT,

    -- Payer response
    payer_decision VARCHAR(50), -- OVERTURNED, PARTIAL, UPHELD, PENDING
    payer_response TEXT,
    payer_decision_reason TEXT,

    -- Financial outcome
    additional_payment_amount NUMERIC(18,2),
    final_allowed_amount NUMERIC(18,2),
    final_paid_amount NUMERIC(18,2),

    -- Status
    appeal_status VARCHAR(50) DEFAULT 'FILED', -- FILED, PENDING, DECIDED, ESCALATED

    -- Assigned to
    assigned_to VARCHAR(100),

    -- Notes
    internal_notes TEXT,

    -- Audit trail
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100),
    updated_by VARCHAR(100)
);

CREATE INDEX idx_denial_appeal_denial ON claims.denial_appeal(denial_id);
CREATE INDEX idx_denial_appeal_level ON claims.denial_appeal(appeal_level);
CREATE INDEX idx_denial_appeal_filed_date ON claims.denial_appeal(filed_date);
CREATE INDEX idx_denial_appeal_due_date ON claims.denial_appeal(due_date);
CREATE INDEX idx_denial_appeal_status ON claims.denial_appeal(appeal_status);
CREATE INDEX idx_denial_appeal_decision ON claims.denial_appeal(payer_decision);

COMMENT ON TABLE claims.denial_appeal IS 'Appeal actions and correspondence for denied claims';

CREATE TRIGGER update_denial_appeal_updated_at BEFORE UPDATE ON claims.denial_appeal
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Denial reason code reference (CARC/RARC)
CREATE TABLE claims.denial_reason_code (
    reason_code_id BIGINT GENERATED ALWAYS AS IDENTITY (CACHE 100) PRIMARY KEY,

    -- Code identification
    code_type VARCHAR(10) NOT NULL, -- CARC, RARC
    reason_code VARCHAR(5) NOT NULL,

    -- Description
    short_description VARCHAR(255),
    long_description TEXT,

    -- Categorization
    category VARCHAR(50), -- BILLING, CODING, AUTHORIZATION, ELIGIBILITY, etc.
    subcategory VARCHAR(100),

    -- Action guidance
    recommended_action TEXT,
    is_appealable BOOLEAN DEFAULT true,

    -- Prevention guidance
    prevention_tips TEXT,

    -- Status
    is_active BOOLEAN DEFAULT true,
    effective_date DATE,
    termination_date DATE,

    -- Audit trail
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(code_type, reason_code)
);

CREATE INDEX idx_denial_reason_code_type ON claims.denial_reason_code(code_type);
CREATE INDEX idx_denial_reason_code_code ON claims.denial_reason_code(reason_code);
CREATE INDEX idx_denial_reason_code_category ON claims.denial_reason_code(category);
CREATE INDEX idx_denial_reason_code_active ON claims.denial_reason_code(is_active) WHERE is_active = true;

COMMENT ON TABLE claims.denial_reason_code IS 'Reference table for CARC and RARC codes';

CREATE TRIGGER update_denial_reason_code_updated_at BEFORE UPDATE ON claims.denial_reason_code
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Denial statistics by various dimensions
CREATE TABLE claims.denial_statistics (
    statistic_id BIGINT GENERATED ALWAYS AS IDENTITY (CACHE 100) PRIMARY KEY,
    organization_id BIGINT NOT NULL REFERENCES claims.organization(organization_id),
    facility_id BIGINT REFERENCES claims.facility(facility_id),

    -- Dimension
    statistic_dimension VARCHAR(50) NOT NULL, -- PAYER, PROVIDER, CODER, PROCEDURE, DENIAL_REASON, ROOT_CAUSE
    dimension_value VARCHAR(255) NOT NULL,

    -- Period
    period_start_date DATE NOT NULL,
    period_end_date DATE NOT NULL,
    period_type VARCHAR(20) NOT NULL, -- DAILY, WEEKLY, MONTHLY, QUARTERLY, YEARLY

    -- Volume metrics
    total_denials INTEGER DEFAULT 0,
    total_denied_amount NUMERIC(18,2) DEFAULT 0,
    total_billed_amount NUMERIC(18,2) DEFAULT 0,

    -- Denial rate
    denial_rate NUMERIC(5,2), -- Percentage

    -- Appeals metrics
    appeals_filed INTEGER DEFAULT 0,
    appeals_won INTEGER DEFAULT 0,
    appeals_lost INTEGER DEFAULT 0,
    appeal_success_rate NUMERIC(5,2),

    -- Financial recovery
    amount_recovered NUMERIC(18,2) DEFAULT 0,
    amount_written_off NUMERIC(18,2) DEFAULT 0,
    recovery_rate NUMERIC(5,2),

    -- Preventability
    preventable_denials INTEGER DEFAULT 0,
    preventable_amount NUMERIC(18,2) DEFAULT 0,

    -- Calculated at
    calculated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(organization_id, facility_id, statistic_dimension, dimension_value, period_start_date, period_type)
);

CREATE INDEX idx_denial_stats_org ON claims.denial_statistics(organization_id);
CREATE INDEX idx_denial_stats_facility ON claims.denial_statistics(facility_id);
CREATE INDEX idx_denial_stats_dimension ON claims.denial_statistics(statistic_dimension);
CREATE INDEX idx_denial_stats_period ON claims.denial_statistics(period_start_date, period_end_date);
CREATE INDEX idx_denial_stats_denial_rate ON claims.denial_statistics(denial_rate);

COMMENT ON TABLE claims.denial_statistics IS 'Aggregated denial statistics by various dimensions';
