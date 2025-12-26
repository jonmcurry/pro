-- ============================================================================
-- Migration: 000_baseline (SmartProAudit)
-- Description: Initial schema for SmartProAudit master database
-- Date: 2025-12-26
--
-- This is the master database for tracking all Professional SMART project databases.
-- It replaces the file-based projects.json registry with a PostgreSQL-based registry.
-- ============================================================================

-- ============================================================================
-- Schema: projects
-- Purpose: Track all project databases managed by this installation
-- ============================================================================

CREATE SCHEMA IF NOT EXISTS projects;

COMMENT ON SCHEMA projects IS 'Project registry for tracking all Professional SMART project databases';

-- Project registry table
CREATE TABLE IF NOT EXISTS projects.project (
    id SERIAL PRIMARY KEY,
    project_name VARCHAR(255) NOT NULL,
    organization VARCHAR(255),
    application_version VARCHAR(50),
    backend_version VARCHAR(50),
    database_version VARCHAR(50),
    connection_information VARCHAR(500),
    database_name VARCHAR(255) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    last_used_at TIMESTAMPTZ,
    is_active BOOLEAN DEFAULT FALSE,
    notes TEXT
);

-- Unique constraint on database_name to prevent duplicates
CREATE UNIQUE INDEX IF NOT EXISTS idx_project_database_name ON projects.project(database_name);

-- Index for quick lookup by project name
CREATE INDEX IF NOT EXISTS idx_project_name ON projects.project(project_name);

-- Trigger for updated_at
CREATE OR REPLACE FUNCTION projects.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS update_project_updated_at ON projects.project;
CREATE TRIGGER update_project_updated_at
    BEFORE UPDATE ON projects.project
    FOR EACH ROW EXECUTE FUNCTION projects.update_updated_at_column();

COMMENT ON TABLE projects.project IS 'Registry of all Professional SMART project databases';
COMMENT ON COLUMN projects.project.project_name IS 'Human-readable project name';
COMMENT ON COLUMN projects.project.organization IS 'Organization or client name';
COMMENT ON COLUMN projects.project.application_version IS 'Professional SMART application version';
COMMENT ON COLUMN projects.project.backend_version IS 'Backend service version';
COMMENT ON COLUMN projects.project.database_version IS 'Database schema version (migration level)';
COMMENT ON COLUMN projects.project.connection_information IS 'PostgreSQL connection details (host:port)';
COMMENT ON COLUMN projects.project.database_name IS 'PostgreSQL database name';
COMMENT ON COLUMN projects.project.is_active IS 'Whether this is the currently active project database';
COMMENT ON COLUMN projects.project.last_used_at IS 'Timestamp of last switch to this database';

-- Schema migrations tracking for SmartProAudit itself
CREATE TABLE IF NOT EXISTS projects.schema_migrations (
    id SERIAL PRIMARY KEY,
    migration_name VARCHAR(255) NOT NULL UNIQUE,
    applied_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    checksum VARCHAR(64),
    description TEXT
);

COMMENT ON TABLE projects.schema_migrations IS 'Tracks applied migrations for SmartProAudit database';

-- ============================================================================
-- Schema: fields
-- Purpose: Field definitions and metadata for claims data export/display
-- ============================================================================

CREATE SCHEMA IF NOT EXISTS fields;

COMMENT ON SCHEMA fields IS 'Field definitions and metadata for claims data display and export';

-- Lookup field definitions table
CREATE TABLE IF NOT EXISTS fields.lookup_field_definitions (
    id SERIAL PRIMARY KEY,
    table_name VARCHAR(255),
    column_name VARCHAR(255),
    column_sort_order INT DEFAULT 0,
    data_type VARCHAR(255),
    friendly_name VARCHAR(255),
    source VARCHAR(10) DEFAULT 'system'
);

COMMENT ON TABLE fields.lookup_field_definitions IS 'Metadata for claims table columns including friendly names for display';
COMMENT ON COLUMN fields.lookup_field_definitions.table_name IS 'Source table name (e.g., encounter, service_line)';
COMMENT ON COLUMN fields.lookup_field_definitions.column_name IS 'Database column name';
COMMENT ON COLUMN fields.lookup_field_definitions.column_sort_order IS 'Display order for columns';
COMMENT ON COLUMN fields.lookup_field_definitions.data_type IS 'PostgreSQL data type';
COMMENT ON COLUMN fields.lookup_field_definitions.friendly_name IS 'Human-readable column name for display';
COMMENT ON COLUMN fields.lookup_field_definitions.source IS 'Source of definition: system or user';

-- Index for lookups by table
CREATE INDEX IF NOT EXISTS idx_lookup_field_table ON fields.lookup_field_definitions(table_name);

-- ============================================================================
-- Seed data: Field definitions for claims tables
-- These are the standard fields from the claims schema in project databases
-- ============================================================================

-- encounter table fields
INSERT INTO fields.lookup_field_definitions (table_name, column_name, data_type, friendly_name, column_sort_order, source) VALUES
('encounter', 'encounter_id', 'bigint', 'Encounter ID', 1, 'system'),
('encounter', 'encounter_group_id', 'uuid', 'Encounter Group ID', 2, 'system'),
('encounter', 'facility_id', 'bigint', 'Facility ID', 3, 'system'),
('encounter', 'billing_date', 'date', 'Billing Date', 4, 'system'),
('encounter', 'submitter_id', 'varchar', 'Submitter ID', 5, 'system'),
('encounter', 'patient_control_number', 'varchar', 'Patient #', 6, 'system'),
('encounter', 'subscriber_id', 'varchar', 'Subscriber ID', 7, 'system'),
('encounter', 'subscriber_birth_date', 'date', 'Subscriber DOB', 8, 'system'),
('encounter', 'patient_date_of_birth', 'date', 'Patient DOB', 9, 'system'),
('encounter', 'patient_gender', 'varchar', 'Gender', 10, 'system'),
('encounter', 'total_claim_charge_amount', 'numeric', 'Total Charge', 11, 'system'),
('encounter', 'place_of_service_code', 'varchar', 'POS', 12, 'system'),
('encounter', 'date_of_service_from', 'date', 'DOS From', 13, 'system'),
('encounter', 'date_of_service_to', 'date', 'DOS To', 14, 'system'),
('encounter', 'billing_provider_id', 'bigint', 'Billing Provider ID', 15, 'system'),
('encounter', 'referring_provider_id', 'bigint', 'Referring Provider ID', 16, 'system'),
('encounter', 'rendering_provider_id', 'bigint', 'Rendering Provider ID', 17, 'system'),
('encounter', 'supervising_provider_id', 'bigint', 'Supervising Provider ID', 18, 'system'),
('encounter', 'service_facility_npi', 'varchar', 'Service Facility NPI', 19, 'system'),
('encounter', 'service_facility_name', 'varchar', 'Service Facility Name', 20, 'system'),
('encounter', 'service_facility_city', 'varchar', 'Service Facility City', 21, 'system'),
('encounter', 'service_facility_state', 'varchar', 'Service Facility State', 22, 'system');

-- encounter_view fields (denormalized view with provider/payer details)
INSERT INTO fields.lookup_field_definitions (table_name, column_name, data_type, friendly_name, column_sort_order, source) VALUES
('encounter_view', 'encounter_id', 'bigint', 'Encounter ID', 1, 'system'),
('encounter_view', 'patient_control_number', 'varchar', 'Patient #', 2, 'system'),
('encounter_view', 'date_of_service_from', 'date', 'DOS From', 3, 'system'),
('encounter_view', 'date_of_service_to', 'date', 'DOS To', 4, 'system'),
('encounter_view', 'total_claim_charge_amount', 'numeric', 'Total Charge', 5, 'system'),
('encounter_view', 'primary_payer_name', 'varchar', 'Payer', 6, 'system'),
('encounter_view', 'primary_claim_filing_indicator', 'varchar', 'FC', 7, 'system'),
('encounter_view', 'billing_provider_npi', 'varchar', 'Billing NPI', 8, 'system'),
('encounter_view', 'billing_provider_taxonomy_code', 'varchar', 'Billing Tax', 9, 'system'),
('encounter_view', 'billing_provider_specialty', 'varchar', 'Billing Spec', 10, 'system'),
('encounter_view', 'rendering_provider_npi', 'varchar', 'Rendering NPI', 11, 'system'),
('encounter_view', 'rendering_provider_taxonomy_code', 'varchar', 'Rendering Tax', 12, 'system'),
('encounter_view', 'rendering_provider_specialty', 'varchar', 'Rendering Spec', 13, 'system'),
('encounter_view', 'diagnosis_codes', 'text', 'Diagnosis Codes', 14, 'system'),
('encounter_view', 'place_of_service_code', 'varchar', 'POS', 15, 'system');

-- encounter_diagnosis fields
INSERT INTO fields.lookup_field_definitions (table_name, column_name, data_type, friendly_name, column_sort_order, source) VALUES
('encounter_diagnosis', 'encounter_diagnosis_id', 'bigint', 'Diagnosis ID', 1, 'system'),
('encounter_diagnosis', 'encounter_id', 'bigint', 'Encounter ID', 2, 'system'),
('encounter_diagnosis', 'diagnosis_code', 'varchar', 'Diagnosis Code', 3, 'system'),
('encounter_diagnosis', 'diagnosis_code_qualifier', 'varchar', 'Code Qualifier', 4, 'system'),
('encounter_diagnosis', 'sequence_number', 'int', 'Sequence', 5, 'system');

-- encounter_payer fields
INSERT INTO fields.lookup_field_definitions (table_name, column_name, data_type, friendly_name, column_sort_order, source) VALUES
('encounter_payer', 'encounter_payer_id', 'bigint', 'Payer ID', 1, 'system'),
('encounter_payer', 'encounter_id', 'bigint', 'Encounter ID', 2, 'system'),
('encounter_payer', 'payer_id', 'varchar', 'Payer Code', 3, 'system'),
('encounter_payer', 'payer_name', 'varchar', 'Payer Name', 4, 'system'),
('encounter_payer', 'payer_responsibility_code', 'varchar', 'Responsibility', 5, 'system'),
('encounter_payer', 'claim_filing_indicator', 'varchar', 'Filing Indicator', 6, 'system'),
('encounter_payer', 'is_billing_payer', 'boolean', 'Billing Payer', 7, 'system');

-- service_line fields
INSERT INTO fields.lookup_field_definitions (table_name, column_name, data_type, friendly_name, column_sort_order, source) VALUES
('service_line', 'service_line_id', 'bigint', 'Line ID', 1, 'system'),
('service_line', 'encounter_id', 'bigint', 'Encounter ID', 2, 'system'),
('service_line', 'line_number', 'int', 'Line #', 3, 'system'),
('service_line', 'procedure_code', 'varchar', 'CPT Code', 4, 'system'),
('service_line', 'modifier_1', 'varchar', 'Mod 1', 5, 'system'),
('service_line', 'modifier_2', 'varchar', 'Mod 2', 6, 'system'),
('service_line', 'modifier_3', 'varchar', 'Mod 3', 7, 'system'),
('service_line', 'modifier_4', 'varchar', 'Mod 4', 8, 'system'),
('service_line', 'units', 'numeric', 'Units', 9, 'system'),
('service_line', 'charge_amount', 'numeric', 'Charge', 10, 'system'),
('service_line', 'service_date_from', 'date', 'Service Date From', 11, 'system'),
('service_line', 'service_date_to', 'date', 'Service Date To', 12, 'system'),
('service_line', 'place_of_service', 'varchar', 'POS', 13, 'system'),
('service_line', 'rendering_provider_id', 'bigint', 'Rendering Provider ID', 14, 'system'),
('service_line', 'revenue_code', 'varchar', 'Revenue Code', 15, 'system');

-- ============================================================================
-- Schema: security
-- Purpose: User authentication and role-based access control
-- ============================================================================

CREATE SCHEMA IF NOT EXISTS security;

COMMENT ON SCHEMA security IS 'User authentication and role-based access control';

-- Security roles table
CREATE TABLE IF NOT EXISTS security.security_role (
    id BIGINT NOT NULL GENERATED BY DEFAULT AS IDENTITY (INCREMENT 1 START 1 MINVALUE 1 MAXVALUE 9223372036854775807 CACHE 1),
    role_name VARCHAR(50),
    role_description VARCHAR(100),
    PRIMARY KEY (id)
);

COMMENT ON TABLE security.security_role IS 'Available security roles for access control';
COMMENT ON COLUMN security.security_role.role_name IS 'Name of the security role';
COMMENT ON COLUMN security.security_role.role_description IS 'Description of role permissions';

-- Security users table
CREATE TABLE IF NOT EXISTS security.security_user (
    id BIGINT NOT NULL GENERATED BY DEFAULT AS IDENTITY (INCREMENT 1 START 1 MINVALUE 1 MAXVALUE 9223372036854775807 CACHE 1),
    user_name VARCHAR(100),
    active BOOLEAN DEFAULT true,
    PRIMARY KEY (id)
);

COMMENT ON TABLE security.security_user IS 'Registered users for the application';
COMMENT ON COLUMN security.security_user.user_name IS 'Username for authentication';
COMMENT ON COLUMN security.security_user.active IS 'Whether the user account is active';

-- Security user-role junction table
CREATE TABLE IF NOT EXISTS security.security_user_role (
    id BIGINT NOT NULL GENERATED BY DEFAULT AS IDENTITY (INCREMENT 1 START 1 MINVALUE 1 MAXVALUE 9223372036854775807 CACHE 1),
    user_id BIGINT,
    role_id BIGINT,
    PRIMARY KEY (id),
    CONSTRAINT fk_user_role_user FOREIGN KEY (user_id) REFERENCES security.security_user(id) ON DELETE CASCADE,
    CONSTRAINT fk_user_role_role FOREIGN KEY (role_id) REFERENCES security.security_role(id) ON DELETE CASCADE
);

COMMENT ON TABLE security.security_user_role IS 'Maps users to their assigned roles';
COMMENT ON COLUMN security.security_user_role.user_id IS 'Reference to security_user';
COMMENT ON COLUMN security.security_user_role.role_id IS 'Reference to security_role';

-- Index for user lookups by username (unique to enforce no duplicate usernames)
CREATE UNIQUE INDEX IF NOT EXISTS idx_security_user_name ON security.security_user(user_name);

-- Index for role lookups by name (e.g., find "Admin" role)
CREATE UNIQUE INDEX IF NOT EXISTS idx_security_role_name ON security.security_role(role_name);

-- Index for role lookups by user
CREATE INDEX IF NOT EXISTS idx_security_user_role_user ON security.security_user_role(user_id);
CREATE INDEX IF NOT EXISTS idx_security_user_role_role ON security.security_user_role(role_id);

-- Unique constraint to prevent duplicate user-role assignments
CREATE UNIQUE INDEX IF NOT EXISTS idx_security_user_role_unique ON security.security_user_role(user_id, role_id);

-- ============================================================================
-- Seed data: Initial security roles and default user
-- ============================================================================

-- Insert default roles
INSERT INTO security.security_role (role_name, role_description) VALUES ('Admin', 'Administrator');
INSERT INTO security.security_role (role_name, role_description) VALUES ('Super User', 'Super User');
INSERT INTO security.security_role (role_name, role_description) VALUES ('User', 'User');

-- Insert default user
INSERT INTO security.security_user (user_name, active) VALUES ('MWELLINGTO002', true);

-- Assign Admin role to default user
INSERT INTO security.security_user_role (user_id, role_id) VALUES (1, 1);

-- Record this migration
INSERT INTO projects.schema_migrations (migration_name, checksum, description)
VALUES ('000_baseline.sql', 'initial', 'Initial SmartProAudit schema with projects, fields, and security schemas');
