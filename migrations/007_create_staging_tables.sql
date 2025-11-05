-- Migration: 007_create_staging_tables
-- Description: Create staging schema tables for file import and configuration
-- Date: 2025-10-14

-- Import batch tracking
CREATE TABLE staging.import_batch (
    batch_id BIGINT GENERATED ALWAYS AS IDENTITY (CACHE 100) PRIMARY KEY,
    organization_id BIGINT NOT NULL REFERENCES claims.organization(organization_id),
    facility_id BIGINT REFERENCES claims.facility(facility_id),

    -- Batch information
    batch_name VARCHAR(255),
    batch_type VARCHAR(50) NOT NULL, -- EDI_837P, CSV, MANUAL
    file_format VARCHAR(50), -- For CSV: EXCEL, ATHENA, EPIC, CERNER, etc.

    -- File details
    original_filename VARCHAR(500),
    file_path TEXT,
    file_size_bytes BIGINT,
    file_hash VARCHAR(64), -- SHA-256 for deduplication

    -- Processing status
    import_status VARCHAR(50) DEFAULT 'PENDING', -- PENDING, PROCESSING, COMPLETED, FAILED, PARTIAL
    total_records INTEGER DEFAULT 0,
    processed_records INTEGER DEFAULT 0,
    successful_records INTEGER DEFAULT 0,
    failed_records INTEGER DEFAULT 0,
    skipped_records INTEGER DEFAULT 0,
    duplicate_records INTEGER DEFAULT 0,

    -- Timing
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    processing_duration_seconds NUMERIC(15,3),

    -- Configuration used
    configuration_id BIGINT,
    rules_applied BOOLEAN DEFAULT false,

    -- Error tracking
    error_message TEXT,
    error_details JSONB,

    -- Validation results
    validation_passed BOOLEAN,
    validation_errors JSONB,
    validation_warnings JSONB,

    -- Audit trail
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100),

    CONSTRAINT chk_batch_type CHECK (batch_type IN ('EDI_837P', 'CSV', 'MANUAL'))
);

CREATE INDEX idx_import_batch_org ON staging.import_batch(organization_id);
CREATE INDEX idx_import_batch_facility ON staging.import_batch(facility_id);
CREATE INDEX idx_import_batch_status ON staging.import_batch(import_status);
CREATE INDEX idx_import_batch_type ON staging.import_batch(batch_type);
CREATE INDEX idx_import_batch_created ON staging.import_batch(created_at);
CREATE INDEX idx_import_batch_file_hash ON staging.import_batch(file_hash);
CREATE INDEX idx_import_batch_started ON staging.import_batch(started_at);

-- Composite index for common queries
CREATE INDEX idx_import_batch_org_status_created ON staging.import_batch(organization_id, import_status, created_at);

COMMENT ON TABLE staging.import_batch IS 'Tracks file import batches and processing metrics';

-- File upload tracking (for large files uploaded in chunks)
CREATE TABLE staging.file_upload (
    upload_id BIGINT GENERATED ALWAYS AS IDENTITY (CACHE 100) PRIMARY KEY,
    organization_id BIGINT NOT NULL REFERENCES claims.organization(organization_id),

    -- Upload details
    original_filename VARCHAR(500) NOT NULL,
    file_type VARCHAR(50) NOT NULL,
    total_size_bytes BIGINT NOT NULL,
    uploaded_size_bytes BIGINT DEFAULT 0,
    chunk_count INTEGER DEFAULT 0,
    chunks_received INTEGER DEFAULT 0,

    -- Status
    upload_status VARCHAR(50) DEFAULT 'IN_PROGRESS', -- IN_PROGRESS, COMPLETED, FAILED, CANCELLED

    -- Storage
    storage_path TEXT,
    file_hash VARCHAR(64),

    -- Timing
    started_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMPTZ,
    expires_at TIMESTAMPTZ, -- For cleanup of incomplete uploads

    -- Error tracking
    error_message TEXT,

    -- Audit trail
    created_by VARCHAR(100)
);

CREATE INDEX idx_file_upload_org ON staging.file_upload(organization_id);
CREATE INDEX idx_file_upload_status ON staging.file_upload(upload_status);
CREATE INDEX idx_file_upload_expires ON staging.file_upload(expires_at) WHERE upload_status = 'IN_PROGRESS';

COMMENT ON TABLE staging.file_upload IS 'Tracks multi-part file uploads';

-- Import configuration profiles
CREATE TABLE staging.import_configuration (
    configuration_id BIGINT GENERATED ALWAYS AS IDENTITY (CACHE 100) PRIMARY KEY,
    organization_id BIGINT NOT NULL REFERENCES claims.organization(organization_id),

    -- Configuration details
    configuration_name VARCHAR(255) NOT NULL,
    configuration_type VARCHAR(50) NOT NULL, -- EDI_837P, CSV_ATHENA, CSV_EPIC, etc.
    description TEXT,

    -- CSV-specific configuration
    csv_delimiter VARCHAR(5),
    csv_quote_char VARCHAR(1),
    csv_has_header BOOLEAN DEFAULT true,
    csv_encoding VARCHAR(20) DEFAULT 'UTF-8',

    -- Header mapping (for CSV)
    header_mappings JSONB, -- JSON object mapping CSV headers to database fields

    -- Field transformations
    field_transformations JSONB, -- JSON array of transformation rules

    -- Validation rules
    validation_rules JSONB,
    required_fields TEXT[],

    -- Default values
    default_values JSONB,

    -- Deduplication strategy
    deduplication_enabled BOOLEAN DEFAULT true,
    deduplication_fields TEXT[], -- Fields to check for duplicates
    deduplication_window_days INTEGER DEFAULT 90,

    -- Auto-apply rules engine
    auto_apply_rules BOOLEAN DEFAULT true,
    rules_to_apply TEXT[], -- Array of rule codes to apply

    -- Status
    is_active BOOLEAN DEFAULT true,
    is_default BOOLEAN DEFAULT false,

    -- Audit trail
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100),
    updated_by VARCHAR(100),

    UNIQUE(organization_id, configuration_name)
);

CREATE INDEX idx_import_config_org ON staging.import_configuration(organization_id);
CREATE INDEX idx_import_config_type ON staging.import_configuration(configuration_type);
CREATE INDEX idx_import_config_active ON staging.import_configuration(is_active) WHERE is_active = true;
CREATE INDEX idx_import_config_default ON staging.import_configuration(organization_id, is_default) WHERE is_default = true;

COMMENT ON TABLE staging.import_configuration IS 'Import configuration profiles for different file types and sources';

CREATE TRIGGER update_import_config_updated_at BEFORE UPDATE ON staging.import_configuration
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Rules engine configuration
CREATE TABLE staging.rules_configuration (
    rule_config_id BIGINT GENERATED ALWAYS AS IDENTITY (CACHE 100) PRIMARY KEY,
    organization_id BIGINT NOT NULL REFERENCES claims.organization(organization_id),
    facility_id BIGINT REFERENCES claims.facility(facility_id),

    -- Rule identification
    rule_code VARCHAR(50) NOT NULL,
    rule_name VARCHAR(255) NOT NULL,
    rule_category VARCHAR(50), -- CODING, DOCUMENTATION, E_M, MODIFIER, etc.

    -- Rule logic
    rule_type VARCHAR(50) NOT NULL, -- SQL_QUERY, JAVASCRIPT, PYTHON, LOOKUP
    rule_definition TEXT NOT NULL, -- The actual rule logic
    rule_parameters JSONB, -- Parameters for the rule

    -- Rule metadata
    description TEXT,
    severity VARCHAR(20) DEFAULT 'MEDIUM', -- HIGH, MEDIUM, LOW
    auto_flag BOOLEAN DEFAULT true,

    -- Issue to create when triggered
    flag_issue_id BIGINT REFERENCES claims.flag_issue(issue_id),

    -- Conditions
    applies_to_claim_types TEXT[], -- Which claim types this applies to
    applies_to_specialties TEXT[], -- Provider specialties
    applies_to_place_of_service TEXT[], -- POS codes
    effective_date_from DATE,
    effective_date_to DATE,

    -- Performance
    execution_order INTEGER DEFAULT 100, -- Order in which rules are executed
    timeout_seconds INTEGER DEFAULT 5,

    -- Statistics
    times_triggered INTEGER DEFAULT 0,
    last_triggered_at TIMESTAMPTZ,

    -- Status
    is_active BOOLEAN DEFAULT true,
    is_deleted BOOLEAN DEFAULT false,

    -- Audit trail
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100),
    updated_by VARCHAR(100),

    UNIQUE(organization_id, rule_code)
);

CREATE INDEX idx_rules_config_org ON staging.rules_configuration(organization_id);
CREATE INDEX idx_rules_config_facility ON staging.rules_configuration(facility_id);
CREATE INDEX idx_rules_config_code ON staging.rules_configuration(rule_code);
CREATE INDEX idx_rules_config_category ON staging.rules_configuration(rule_category);
CREATE INDEX idx_rules_config_active ON staging.rules_configuration(is_active) WHERE is_active = true;
CREATE INDEX idx_rules_config_execution_order ON staging.rules_configuration(execution_order) WHERE is_active = true;

COMMENT ON TABLE staging.rules_configuration IS 'Rules engine configuration for automated flagging';

CREATE TRIGGER update_rules_config_updated_at BEFORE UPDATE ON staging.rules_configuration
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Processing metrics and performance tracking
CREATE TABLE staging.processing_metrics (
    metric_id BIGINT GENERATED ALWAYS AS IDENTITY (CACHE 100) PRIMARY KEY,
    batch_id BIGINT REFERENCES staging.import_batch(batch_id) ON DELETE CASCADE,

    -- Metric details
    metric_type VARCHAR(50) NOT NULL, -- PARSE, VALIDATE, TRANSFORM, DEDUPE, RULES, INSERT
    metric_name VARCHAR(255) NOT NULL,

    -- Performance data
    started_at TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ,
    duration_seconds NUMERIC(15,3),
    records_processed INTEGER DEFAULT 0,
    records_per_second NUMERIC(15,3),

    -- Resource usage
    memory_used_mb NUMERIC(15,2),
    cpu_time_ms NUMERIC(15,3),

    -- Results
    success_count INTEGER DEFAULT 0,
    error_count INTEGER DEFAULT 0,
    warning_count INTEGER DEFAULT 0,

    -- Additional details
    details JSONB,

    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_processing_metrics_batch ON staging.processing_metrics(batch_id);
CREATE INDEX idx_processing_metrics_type ON staging.processing_metrics(metric_type);
CREATE INDEX idx_processing_metrics_started ON staging.processing_metrics(started_at);

COMMENT ON TABLE staging.processing_metrics IS 'Performance metrics for import processing stages';

-- Error log for import failures
CREATE TABLE staging.import_error_log (
    error_id BIGINT GENERATED ALWAYS AS IDENTITY (CACHE 100) PRIMARY KEY,
    batch_id BIGINT REFERENCES staging.import_batch(batch_id) ON DELETE CASCADE,

    -- Error location
    record_number INTEGER,
    line_number INTEGER,
    field_name VARCHAR(255),

    -- Error details
    error_type VARCHAR(50) NOT NULL, -- VALIDATION, PARSE, TRANSFORM, CONSTRAINT, DUPLICATE
    error_severity VARCHAR(20) DEFAULT 'ERROR', -- ERROR, WARNING, INFO
    error_code VARCHAR(50),
    error_message TEXT NOT NULL,
    error_details JSONB,

    -- Source data
    raw_data TEXT,

    -- Resolution
    resolution_status VARCHAR(50) DEFAULT 'UNRESOLVED', -- UNRESOLVED, IGNORED, FIXED, REPROCESSED
    resolution_note TEXT,
    resolved_at TIMESTAMPTZ,
    resolved_by VARCHAR(100),

    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_import_error_batch ON staging.import_error_log(batch_id);
CREATE INDEX idx_import_error_type ON staging.import_error_log(error_type);
CREATE INDEX idx_import_error_severity ON staging.import_error_log(error_severity);
CREATE INDEX idx_import_error_status ON staging.import_error_log(resolution_status);
CREATE INDEX idx_import_error_created ON staging.import_error_log(created_at);

COMMENT ON TABLE staging.import_error_log IS 'Detailed error log for import failures';
