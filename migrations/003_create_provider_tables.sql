-- Migration: 003_create_provider_tables
-- Description: Create provider and coder tables
-- Date: 2025-10-14

-- Provider table (physicians, practitioners, etc.)
CREATE TABLE claims.provider (
    provider_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    npi VARCHAR(10) NOT NULL UNIQUE,
    provider_type VARCHAR(50) NOT NULL, -- Billing, Rendering, Referring, Supervising, Ordering
    last_name VARCHAR(255) NOT NULL,
    first_name VARCHAR(255) NOT NULL,
    middle_name VARCHAR(255),
    name_suffix VARCHAR(50),
    taxonomy_code VARCHAR(10),
    license_number VARCHAR(50),
    license_state CHAR(2),
    specialty VARCHAR(100),
    provider_group VARCHAR(255),
    organization_id UUID REFERENCES claims.organization(organization_id),
    address_line1 VARCHAR(255),
    address_line2 VARCHAR(255),
    city VARCHAR(100),
    state_code CHAR(2),
    postal_code VARCHAR(15),
    country_code CHAR(3) DEFAULT 'USA',
    phone VARCHAR(20),
    email citext,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100),
    updated_by VARCHAR(100)
);

CREATE INDEX idx_provider_npi ON claims.provider(npi);
CREATE INDEX idx_provider_type ON claims.provider(provider_type);
CREATE INDEX idx_provider_specialty ON claims.provider(specialty);
CREATE INDEX idx_provider_org ON claims.provider(organization_id);
CREATE INDEX idx_provider_active ON claims.provider(is_active) WHERE is_active = true;
CREATE INDEX idx_provider_name ON claims.provider(last_name, first_name);

COMMENT ON TABLE claims.provider IS 'Healthcare providers (physicians, practitioners)';

-- Coder table (individuals who code/bill claims)
CREATE TABLE claims.coder (
    coder_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    coder_code VARCHAR(50) NOT NULL UNIQUE,
    last_name VARCHAR(255) NOT NULL,
    first_name VARCHAR(255) NOT NULL,
    middle_name VARCHAR(255),
    coder_group VARCHAR(100),
    certifications TEXT[], -- Array of certifications (CPC, CCS, etc.)
    organization_id UUID REFERENCES claims.organization(organization_id),
    email citext,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100),
    updated_by VARCHAR(100)
);

CREATE INDEX idx_coder_code ON claims.coder(coder_code);
CREATE INDEX idx_coder_group ON claims.coder(coder_group);
CREATE INDEX idx_coder_org ON claims.coder(organization_id);
CREATE INDEX idx_coder_active ON claims.coder(is_active) WHERE is_active = true;
CREATE INDEX idx_coder_name ON claims.coder(last_name, first_name);

COMMENT ON TABLE claims.coder IS 'Medical coders who code/bill claims';

-- Reviewer table (individuals who perform audits)
CREATE TABLE claims.reviewer (
    reviewer_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    reviewer_code VARCHAR(50) NOT NULL UNIQUE,
    last_name VARCHAR(255) NOT NULL,
    first_name VARCHAR(255) NOT NULL,
    middle_name VARCHAR(255),
    reviewer_group VARCHAR(100),
    certifications TEXT[], -- Array of certifications
    organization_id UUID REFERENCES claims.organization(organization_id),
    email citext,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100),
    updated_by VARCHAR(100)
);

CREATE INDEX idx_reviewer_code ON claims.reviewer(reviewer_code);
CREATE INDEX idx_reviewer_group ON claims.reviewer(reviewer_group);
CREATE INDEX idx_reviewer_org ON claims.reviewer(organization_id);
CREATE INDEX idx_reviewer_active ON claims.reviewer(is_active) WHERE is_active = true;
CREATE INDEX idx_reviewer_name ON claims.reviewer(last_name, first_name);

COMMENT ON TABLE claims.reviewer IS 'Audit reviewers who perform retrospective reviews';

-- Triggers for updated_at
CREATE TRIGGER update_provider_updated_at BEFORE UPDATE ON claims.provider
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_coder_updated_at BEFORE UPDATE ON claims.coder
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_reviewer_updated_at BEFORE UPDATE ON claims.reviewer
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
