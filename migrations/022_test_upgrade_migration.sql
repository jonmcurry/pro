-- Test migration to verify upgrade functionality

CREATE TABLE IF NOT EXISTS staging.upgrade_test (
    id BIGINT GENERATED ALWAYS AS IDENTITY (CACHE 100) PRIMARY KEY,
    upgrade_version VARCHAR(20) NOT NULL,
    upgraded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    notes TEXT
);

INSERT INTO staging.upgrade_test (upgrade_version, notes)
VALUES ('1.2.3.0', 'Test upgrade from 1.2.2.0 to 1.2.3.0');
