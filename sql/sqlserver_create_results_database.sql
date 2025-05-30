-- SQL Server Schema (Validation Results Storage)
-- ===============================================

-- Validation results table
CREATE TABLE dbo.ValidationResults (
    result_id BIGINT IDENTITY(1,1) PRIMARY KEY,
    claim_id VARCHAR(50) NOT NULL,
    validation_status VARCHAR(20) NOT NULL,
    predicted_filters NVARCHAR(MAX),
    validation_details NVARCHAR(MAX),
    processing_time DECIMAL(10,6),
    error_message NVARCHAR(MAX),
    created_date DATETIME2 DEFAULT GETDATE()
);

-- Create indexes for ValidationResults table
CREATE INDEX IX_ValidationResults_ClaimId ON dbo.ValidationResults (claim_id);
CREATE INDEX IX_ValidationResults_Status_Date ON dbo.ValidationResults (validation_status, created_date);
CREATE INDEX IX_ValidationResults_Date ON dbo.ValidationResults (created_date);

-- Archive table for old results
CREATE TABLE dbo.ValidationResultsArchive (
    result_id BIGINT,
    claim_id VARCHAR(50),
    validation_status VARCHAR(20),
    predicted_filters NVARCHAR(MAX),
    validation_details NVARCHAR(MAX),
    processing_time DECIMAL(10,6),
    error_message NVARCHAR(MAX),
    created_date DATETIME2,
    archived_date DATETIME2 DEFAULT GETDATE()
);

-- Create indexes for ValidationResultsArchive table
CREATE INDEX IX_ValidationResultsArchive_ClaimId ON dbo.ValidationResultsArchive (claim_id);
CREATE INDEX IX_ValidationResultsArchive_Date ON dbo.ValidationResultsArchive (created_date);

-- Performance statistics table
CREATE TABLE dbo.ProcessingStats (
    stat_id BIGINT IDENTITY(1,1) PRIMARY KEY,
    processing_date DATE,
    total_claims INTEGER,
    successful_validations INTEGER,
    failed_validations INTEGER,
    avg_processing_time DECIMAL(10,6),
    peak_memory_usage DECIMAL(5,2),
    peak_cpu_usage DECIMAL(5,2),
    created_date DATETIME2 DEFAULT GETDATE()
);

-- Create unique index for ProcessingStats table
CREATE UNIQUE INDEX UX_ProcessingStats_Date ON dbo.ProcessingStats (processing_date);