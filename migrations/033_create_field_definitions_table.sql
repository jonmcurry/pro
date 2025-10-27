-- Migration 033: Create field_definitions table
-- This table stores metadata about columns in key claims tables for dynamic field mapping

CREATE TABLE claims.field_definitions (
    table_name VARCHAR(255) NOT NULL,
    column_name VARCHAR(255) NOT NULL,
    data_type VARCHAR(255) NOT NULL,
    PRIMARY KEY (table_name, column_name)
);

-- Populate with current field definitions from service_line, encounter, and encounter_diagnosis tables
INSERT INTO claims.field_definitions (table_name, column_name, data_type)
SELECT
    table_name::VARCHAR(255),
    column_name::VARCHAR(255),
    data_type::VARCHAR(255)
FROM information_schema.columns
WHERE table_schema = 'claims'
  AND table_name IN ('service_line', 'encounter', 'encounter_diagnosis')
ORDER BY table_name, column_name;

-- Create index for faster lookups by table
CREATE INDEX idx_field_definitions_table ON claims.field_definitions(table_name);
