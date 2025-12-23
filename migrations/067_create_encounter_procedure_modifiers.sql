-- Migration: 067_create_encounter_procedure_modifiers
-- Description: Create table to store aggregated procedure modifiers at encounter level
-- Date: 2025-12-23

-- Ensure pg_trgm extension exists for GIN index
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Encounter procedure modifiers table
-- Stores comma-separated list of unique modifiers from all service lines on an encounter
CREATE TABLE claims.encounter_procedure_modifier (
    encounter_id BIGINT PRIMARY KEY REFERENCES claims.encounter(encounter_id) ON DELETE CASCADE,
    modifiers VARCHAR(20) NOT NULL, -- Comma-separated list e.g., "24,25,59"
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Index for searching by modifiers (useful for finding encounters with specific modifiers)
CREATE INDEX idx_encounter_proc_mod_modifiers ON claims.encounter_procedure_modifier(modifiers);

-- GIN index for pattern matching on modifiers (e.g., WHERE modifiers LIKE '%25%')
CREATE INDEX idx_encounter_proc_mod_modifiers_gin ON claims.encounter_procedure_modifier
    USING gin(modifiers gin_trgm_ops);

-- Trigger for updated_at
CREATE TRIGGER update_encounter_proc_mod_updated_at
    BEFORE UPDATE ON claims.encounter_procedure_modifier
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

COMMENT ON TABLE claims.encounter_procedure_modifier IS 'Aggregated procedure modifiers from all service lines on an encounter, stored as comma-separated list';
COMMENT ON COLUMN claims.encounter_procedure_modifier.modifiers IS 'Comma-separated list of unique procedure modifiers (e.g., "24,25,59"), sorted and deduplicated';
