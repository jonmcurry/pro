-- PostgreSQL Schema (Claims and Configuration Data)
-- ================================================

-- Create schema
CREATE SCHEMA IF NOT EXISTS edi;

-- Claims table
CREATE TABLE edi.claims (
    claim_id VARCHAR(50) PRIMARY KEY,
    patient_age INTEGER,
    provider_type VARCHAR(100),
    place_of_service VARCHAR(10),
    total_charge_amount DECIMAL(12,2),
    service_date DATE,
    processing_status VARCHAR(20) DEFAULT 'PENDING',
    processed_date TIMESTAMP,
    claim_data JSONB,
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Diagnoses table
CREATE TABLE edi.diagnoses (
    diagnosis_id SERIAL PRIMARY KEY,
    claim_id VARCHAR(50) REFERENCES edi.claims(claim_id),
    diagnosis_code VARCHAR(20),
    sequence_number INTEGER,
    is_principal BOOLEAN DEFAULT FALSE,
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Procedures table
CREATE TABLE edi.procedures (
    procedure_id SERIAL PRIMARY KEY,
    claim_id VARCHAR(50) REFERENCES edi.claims(claim_id),
    procedure_code VARCHAR(20),
    sequence_number INTEGER,
    charge_amount DECIMAL(10,2),
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Validation filters table
CREATE TABLE edi.filters (
    filter_id SERIAL PRIMARY KEY,
    filter_name VARCHAR(200) NOT NULL,
    rule_definition TEXT NOT NULL,
    description TEXT,
    rule_type VARCHAR(50) DEFAULT 'DATALOG',
    support DECIMAL(5,4),
    confidence DECIMAL(5,4),
    lift DECIMAL(5,4),
    active BOOLEAN DEFAULT TRUE,
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Validation results table (NEW - based on SQL Server schema and logs)
CREATE TABLE edi.validation_results (
    result_id BIGSERIAL PRIMARY KEY,  -- Use BIGSERIAL for auto-incrementing 64-bit integer
    claim_id VARCHAR(50) NOT NULL,    -- Consider FK to edi.claims(claim_id)
    validation_status VARCHAR(20) NOT NULL,
    predicted_filters JSONB,          -- Storing as JSONB is more flexible in PostgreSQL
    validation_details TEXT,
    processing_time DECIMAL(10,6),
    error_message TEXT,
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for edi.validation_results
CREATE INDEX IF NOT EXISTS idx_validation_results_claim_id ON edi.validation_results(claim_id);
CREATE INDEX IF NOT EXISTS idx_validation_results_status_date ON edi.validation_results(validation_status, created_date);
CREATE INDEX IF NOT EXISTS idx_validation_results_date ON edi.validation_results(created_date);

-- Note: You might want to add a foreign key constraint from edi.validation_results.claim_id to edi.claims.claim_id
-- after ensuring data consistency or as part of your data loading strategy.
-- Example:
-- ALTER TABLE edi.validation_results
-- ADD CONSTRAINT fk_validation_results_claim_id
-- FOREIGN KEY (claim_id) REFERENCES edi.claims(claim_id);

-- Processing chunks tracking
CREATE TABLE edi.processed_chunks (
    chunk_id INTEGER PRIMARY KEY,
    status VARCHAR(20) DEFAULT 'PROCESSING',
    processed_date TIMESTAMP,
    claims_count INTEGER DEFAULT 0,
    processing_duration_seconds DECIMAL(10,3),
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Retry queue for failed chunks
CREATE TABLE edi.retry_queue (
    chunk_id INTEGER PRIMARY KEY,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    resolved BOOLEAN DEFAULT FALSE,
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_retry_date TIMESTAMP
);

-- Performance indexes
CREATE INDEX CONCURRENTLY idx_claims_status_date ON edi.claims(processing_status, service_date);
CREATE INDEX CONCURRENTLY idx_claims_provider ON edi.claims(provider_type, place_of_service);
CREATE INDEX CONCURRENTLY idx_diagnoses_claim_code ON edi.diagnoses(claim_id, diagnosis_code);
CREATE INDEX CONCURRENTLY idx_procedures_claim_code ON edi.procedures(claim_id, procedure_code);
CREATE INDEX CONCURRENTLY idx_filters_active ON edi.filters(active, rule_type);

-- Sample data for testing
-- ======================

-- Insert sample filters
INSERT INTO edi.filters (filter_name, rule_definition, description, rule_type) VALUES
('Diabetes with Complications Check', 'has_diagnosis(X, ''E11.9'') & patient_age(X, Age) & (Age > 65)', 'Check for diabetes complications in elderly patients', 'DATALOG'),
('High Cost Procedure Review', 'has_procedure(X, ''99213'') & charge_amount(X, Amount) & (Amount > 500)', 'Review high-cost procedures', 'DATALOG'),
('Emergency Room Validation', 'place_of_service(X, ''23'') & provider_type(X, ''Emergency'')', 'Validate emergency room claims', 'DATALOG');

-- Performance optimization views
-- =============================

-- Claims processing view
CREATE VIEW edi.v_claims_processing_status AS
SELECT 
    processing_status,
    COUNT(*) as claim_count,
    AVG(total_charge_amount) as avg_charge,
    MIN(service_date) as earliest_service,
    MAX(service_date) as latest_service
FROM edi.claims
GROUP BY processing_status;

-- Daily processing statistics
CREATE VIEW edi.v_daily_processing_stats AS
SELECT 
    DATE(processed_date) as processing_date,
    COUNT(*) as claims_processed,
    AVG(total_charge_amount) as avg_charge_amount,
    COUNT(DISTINCT provider_type) as unique_providers
FROM edi.claims
WHERE processed_date IS NOT NULL
GROUP BY DATE(processed_date)
ORDER BY processing_date DESC;

CREATE MATERIALIZED VIEW claim_summary AS 
SELECT claim_id, array_agg(diagnosis_code) as diagnoses,
       array_agg(procedure_code) as procedures
FROM edi.claims c
LEFT JOIN edi.diagnoses d USING(claim_id)
LEFT JOIN edi.procedures p USING(claim_id)
GROUP BY claim_id;

-- Add to claims table
ALTER TABLE edi.claims ADD COLUMN patient_id VARCHAR(50);
ALTER TABLE edi.claims ADD COLUMN provider_id VARCHAR(50);

-- Add to diagnoses table  
ALTER TABLE edi.diagnoses ADD COLUMN diagnosis_type VARCHAR(20);
ALTER TABLE edi.diagnoses ADD COLUMN description TEXT;
ALTER TABLE edi.diagnoses RENAME COLUMN sequence_number TO diagnosis_sequence;

-- Add to procedures table
ALTER TABLE edi.procedures ADD COLUMN procedure_type VARCHAR(20);
ALTER TABLE edi.procedures ADD COLUMN description TEXT;
ALTER TABLE edi.procedures ADD COLUMN service_date DATE;
ALTER TABLE edi.procedures ADD COLUMN diagnosis_pointers JSONB;
ALTER TABLE edi.procedures RENAME COLUMN sequence_number TO procedure_sequence;
ALTER TABLE edi.filters ADD CONSTRAINT unique_filter_name UNIQUE (filter_name);