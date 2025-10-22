-- Test migration to verify upgrade functionality
SET search_path TO staging;

CREATE TABLE IF NOT EXISTS upgrade_test (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    upgrade_version VARCHAR(20) NOT NULL,
    upgraded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    notes TEXT
);

INSERT INTO upgrade_test (upgrade_version, notes)
VALUES ('1.2.3.0', 'Test upgrade from 1.2.2.0 to 1.2.3.0');
