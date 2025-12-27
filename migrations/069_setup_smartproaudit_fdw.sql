-- ============================================================================
-- Migration: 069_setup_smartproaudit_fdw
-- Description: Set up Foreign Data Wrapper to query SmartProAudit database
-- Date: 2025-12-26
--
-- This migration creates foreign tables in the project database that link
-- to the SmartProAudit master database, allowing queries to security and
-- fields schemas without cross-database complexity.
-- ============================================================================

-- Enable the postgres_fdw extension
CREATE EXTENSION IF NOT EXISTS postgres_fdw;

-- Create a schema to hold foreign tables from SmartProAudit
CREATE SCHEMA IF NOT EXISTS smartproaudit;

COMMENT ON SCHEMA smartproaudit IS 'Foreign tables linked to SmartProAudit master database';

-- Create the foreign server connection to SmartProAudit
-- Note: Uses same host/port as current connection (localhost assumption)
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_foreign_server WHERE srvname = 'smartproaudit_server') THEN
        CREATE SERVER smartproaudit_server
            FOREIGN DATA WRAPPER postgres_fdw
            OPTIONS (host 'localhost', port '5432', dbname 'smartproaudit');
    END IF;
END $$;

-- Create user mapping for postgres user
-- This maps the current user to the same user on SmartProAudit
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_user_mappings
        WHERE srvname = 'smartproaudit_server' AND usename = current_user
    ) THEN
        EXECUTE format(
            'CREATE USER MAPPING FOR %I SERVER smartproaudit_server OPTIONS (user ''postgres'')',
            current_user
        );
    END IF;
END $$;

-- ============================================================================
-- Foreign tables for security schema
-- ============================================================================

-- security_role foreign table
CREATE FOREIGN TABLE IF NOT EXISTS smartproaudit.security_role (
    id BIGINT,
    role_name VARCHAR(50),
    role_description VARCHAR(100)
)
SERVER smartproaudit_server
OPTIONS (schema_name 'security', table_name 'security_role');

COMMENT ON FOREIGN TABLE smartproaudit.security_role IS 'Foreign table linked to SmartProAudit.security.security_role';

-- security_user foreign table
CREATE FOREIGN TABLE IF NOT EXISTS smartproaudit.security_user (
    id BIGINT,
    user_name VARCHAR(100),
    active BOOLEAN
)
SERVER smartproaudit_server
OPTIONS (schema_name 'security', table_name 'security_user');

COMMENT ON FOREIGN TABLE smartproaudit.security_user IS 'Foreign table linked to SmartProAudit.security.security_user';

-- security_user_role foreign table
CREATE FOREIGN TABLE IF NOT EXISTS smartproaudit.security_user_role (
    id BIGINT,
    user_id BIGINT,
    role_id BIGINT
)
SERVER smartproaudit_server
OPTIONS (schema_name 'security', table_name 'security_user_role');

COMMENT ON FOREIGN TABLE smartproaudit.security_user_role IS 'Foreign table linked to SmartProAudit.security.security_user_role';

-- ============================================================================
-- Foreign tables for fields schema
-- ============================================================================

-- lookup_field_definitions foreign table
CREATE FOREIGN TABLE IF NOT EXISTS smartproaudit.lookup_field_definitions (
    id INTEGER,
    table_name VARCHAR(255),
    column_name VARCHAR(255),
    column_sort_order INTEGER,
    data_type VARCHAR(255),
    friendly_name VARCHAR(255),
    source VARCHAR(10)
)
SERVER smartproaudit_server
OPTIONS (schema_name 'fields', table_name 'lookup_field_definitions');

COMMENT ON FOREIGN TABLE smartproaudit.lookup_field_definitions IS 'Foreign table linked to SmartProAudit.fields.lookup_field_definitions';

-- ============================================================================
-- Foreign tables for projects schema
-- ============================================================================

-- project registry foreign table (read-only access to project list)
CREATE FOREIGN TABLE IF NOT EXISTS smartproaudit.project (
    id INTEGER,
    project_name VARCHAR(255),
    organization VARCHAR(255),
    application_version VARCHAR(50),
    backend_version VARCHAR(50),
    database_version VARCHAR(50),
    connection_information VARCHAR(500),
    database_name VARCHAR(255),
    created_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ,
    last_used_at TIMESTAMPTZ,
    is_active BOOLEAN,
    notes TEXT
)
SERVER smartproaudit_server
OPTIONS (schema_name 'projects', table_name 'project');

COMMENT ON FOREIGN TABLE smartproaudit.project IS 'Foreign table linked to SmartProAudit.projects.project';

-- ============================================================================
-- Convenience views for common queries
-- ============================================================================

-- View to get user with their roles
CREATE OR REPLACE VIEW smartproaudit.user_roles AS
SELECT
    u.id AS user_id,
    u.user_name,
    u.active,
    r.id AS role_id,
    r.role_name,
    r.role_description
FROM smartproaudit.security_user u
LEFT JOIN smartproaudit.security_user_role ur ON u.id = ur.user_id
LEFT JOIN smartproaudit.security_role r ON ur.role_id = r.id;

COMMENT ON VIEW smartproaudit.user_roles IS 'Convenience view showing users with their assigned roles';

-- View to check if a user has a specific role
CREATE OR REPLACE FUNCTION smartproaudit.user_has_role(p_user_name VARCHAR, p_role_name VARCHAR)
RETURNS BOOLEAN AS $$
BEGIN
    RETURN EXISTS (
        SELECT 1
        FROM smartproaudit.user_roles
        WHERE user_name = p_user_name
          AND role_name = p_role_name
          AND active = true
    );
END;
$$ LANGUAGE plpgsql STABLE;

COMMENT ON FUNCTION smartproaudit.user_has_role IS 'Check if a user has a specific role';

-- Function to get field definitions for a table
CREATE OR REPLACE FUNCTION smartproaudit.get_field_definitions(p_table_name VARCHAR)
RETURNS TABLE (
    column_name VARCHAR,
    friendly_name VARCHAR,
    data_type VARCHAR,
    sort_order INTEGER
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        lfd.column_name,
        lfd.friendly_name,
        lfd.data_type,
        lfd.column_sort_order
    FROM smartproaudit.lookup_field_definitions lfd
    WHERE lfd.table_name = p_table_name
    ORDER BY lfd.column_sort_order;
END;
$$ LANGUAGE plpgsql STABLE;

COMMENT ON FUNCTION smartproaudit.get_field_definitions IS 'Get field definitions for a specific table';
