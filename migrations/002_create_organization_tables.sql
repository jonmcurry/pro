-- Migration: 002_create_organization_tables
-- Description: Create organization hierarchy tables (organization -> region -> facility)
-- Date: 2025-10-14

-- Organization table (top level)
CREATE TABLE claims.organization (
    organization_id BIGINT GENERATED ALWAYS AS IDENTITY (CACHE 50) PRIMARY KEY,
    organization_code VARCHAR(50) NOT NULL UNIQUE,
    organization_name VARCHAR(255) NOT NULL,
    tax_id VARCHAR(20),
    npi VARCHAR(10),
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

CREATE INDEX idx_organization_code ON claims.organization(organization_code);
CREATE INDEX idx_organization_active ON claims.organization(is_active) WHERE is_active = true;

COMMENT ON TABLE claims.organization IS 'Top-level organization entities';

-- Region table (middle level - optional)
CREATE TABLE claims.region (
    region_id BIGINT GENERATED ALWAYS AS IDENTITY (CACHE 50) PRIMARY KEY,
    organization_id BIGINT NOT NULL REFERENCES claims.organization(organization_id) ON DELETE CASCADE,
    region_code VARCHAR(50) NOT NULL,
    region_name VARCHAR(255) NOT NULL,
    description TEXT,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100),
    updated_by VARCHAR(100),
    UNIQUE(organization_id, region_code)
);

CREATE INDEX idx_region_org ON claims.region(organization_id);
CREATE INDEX idx_region_code ON claims.region(region_code);
CREATE INDEX idx_region_active ON claims.region(is_active) WHERE is_active = true;

COMMENT ON TABLE claims.region IS 'Regional divisions within organizations (optional level)';

-- Facility table (bottom level - required)
CREATE TABLE claims.facility (
    facility_id BIGINT GENERATED ALWAYS AS IDENTITY (CACHE 50) PRIMARY KEY,
    organization_id BIGINT NOT NULL REFERENCES claims.organization(organization_id) ON DELETE CASCADE,
    region_id BIGINT REFERENCES claims.region(region_id) ON DELETE SET NULL,
    facility_code VARCHAR(50) NOT NULL,
    facility_name VARCHAR(255) NOT NULL,
    npi VARCHAR(10),
    tax_id VARCHAR(20),
    facility_type VARCHAR(50),
    address_line1 VARCHAR(255),
    address_line2 VARCHAR(255),
    city VARCHAR(100),
    state_code CHAR(2),
    postal_code VARCHAR(15),
    country_code CHAR(3) DEFAULT 'USA',
    phone VARCHAR(20),
    email citext,
    ehr_system VARCHAR(100),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100),
    updated_by VARCHAR(100),
    UNIQUE(organization_id, facility_code)
);

CREATE INDEX idx_facility_org ON claims.facility(organization_id);
CREATE INDEX idx_facility_region ON claims.facility(region_id);
CREATE INDEX idx_facility_code ON claims.facility(facility_code);
CREATE INDEX idx_facility_npi ON claims.facility(npi);
CREATE INDEX idx_facility_active ON claims.facility(is_active) WHERE is_active = true;

COMMENT ON TABLE claims.facility IS 'Facility entities - must belong to organization, optionally to region';

-- Add constraint to prevent facility from being in multiple regions
CREATE UNIQUE INDEX idx_facility_single_region ON claims.facility(facility_id) WHERE region_id IS NOT NULL;

-- Trigger function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_organization_updated_at BEFORE UPDATE ON claims.organization
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_region_updated_at BEFORE UPDATE ON claims.region
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_facility_updated_at BEFORE UPDATE ON claims.facility
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
