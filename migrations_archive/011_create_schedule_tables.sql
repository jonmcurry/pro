-- Migration: 011_create_schedule_tables
-- Description: Create schedule tables for automated processing and reports
-- Date: 2025-10-14

-- Scheduled jobs configuration
CREATE TABLE staging.scheduled_job (
    job_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID NOT NULL REFERENCES claims.organization(organization_id),

    -- Job identification
    job_name VARCHAR(255) NOT NULL,
    job_type VARCHAR(50) NOT NULL, -- IMPORT, RULES_ENGINE, REPORT, CLEANUP, CALCULATION, EXPORT
    job_description TEXT,

    -- Schedule configuration
    schedule_type VARCHAR(50) NOT NULL, -- CRON, INTERVAL, ONE_TIME
    cron_expression VARCHAR(100), -- For CRON type
    interval_minutes INTEGER, -- For INTERVAL type
    scheduled_time TIME, -- For daily jobs

    -- Execution configuration
    job_config JSONB, -- Job-specific configuration parameters
    timeout_minutes INTEGER DEFAULT 60,
    max_retries INTEGER DEFAULT 3,
    retry_delay_minutes INTEGER DEFAULT 5,

    -- Concurrency control
    allow_concurrent BOOLEAN DEFAULT false,
    max_concurrent_executions INTEGER DEFAULT 1,

    -- Status
    is_active BOOLEAN DEFAULT true,
    is_running BOOLEAN DEFAULT false,

    -- Execution history
    last_run_at TIMESTAMPTZ,
    last_run_status VARCHAR(50), -- SUCCESS, FAILURE, TIMEOUT, CANCELLED
    last_run_duration_seconds INTEGER,
    next_run_at TIMESTAMPTZ,
    total_executions INTEGER DEFAULT 0,
    successful_executions INTEGER DEFAULT 0,
    failed_executions INTEGER DEFAULT 0,

    -- Error tracking
    last_error_message TEXT,
    consecutive_failures INTEGER DEFAULT 0,

    -- Notifications
    notify_on_success BOOLEAN DEFAULT false,
    notify_on_failure BOOLEAN DEFAULT true,
    notification_emails TEXT[],

    -- Audit trail
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100),
    updated_by VARCHAR(100),

    UNIQUE(organization_id, job_name)
);

CREATE INDEX idx_scheduled_job_org ON staging.scheduled_job(organization_id);
CREATE INDEX idx_scheduled_job_type ON staging.scheduled_job(job_type);
CREATE INDEX idx_scheduled_job_active ON staging.scheduled_job(is_active) WHERE is_active = true;
CREATE INDEX idx_scheduled_job_next_run ON staging.scheduled_job(next_run_at) WHERE is_active = true AND is_running = false;
CREATE INDEX idx_scheduled_job_running ON staging.scheduled_job(is_running) WHERE is_running = true;

COMMENT ON TABLE staging.scheduled_job IS 'Configuration for scheduled automated jobs';

CREATE TRIGGER update_scheduled_job_updated_at BEFORE UPDATE ON staging.scheduled_job
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Job execution log
CREATE TABLE staging.job_execution_log (
    execution_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id UUID NOT NULL REFERENCES staging.scheduled_job(job_id) ON DELETE CASCADE,

    -- Execution details
    started_at TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ,
    duration_seconds INTEGER,

    -- Status
    execution_status VARCHAR(50) NOT NULL, -- RUNNING, SUCCESS, FAILURE, TIMEOUT, CANCELLED
    status_message TEXT,

    -- Results
    records_processed INTEGER DEFAULT 0,
    records_successful INTEGER DEFAULT 0,
    records_failed INTEGER DEFAULT 0,

    -- Performance metrics
    peak_memory_mb NUMERIC(15,2),
    cpu_time_seconds NUMERIC(15,3),

    -- Error details
    error_message TEXT,
    error_stack_trace TEXT,
    error_details JSONB,

    -- Retry information
    is_retry BOOLEAN DEFAULT false,
    retry_attempt INTEGER DEFAULT 0,
    original_execution_id UUID REFERENCES staging.job_execution_log(execution_id),

    -- Output
    execution_output JSONB,
    log_file_path TEXT,

    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_job_execution_job ON staging.job_execution_log(job_id);
CREATE INDEX idx_job_execution_status ON staging.job_execution_log(execution_status);
CREATE INDEX idx_job_execution_started ON staging.job_execution_log(started_at);
CREATE INDEX idx_job_execution_retry ON staging.job_execution_log(original_execution_id) WHERE is_retry = true;

COMMENT ON TABLE staging.job_execution_log IS 'Execution history for scheduled jobs';

-- Report subscriptions
CREATE TABLE staging.report_subscription (
    subscription_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID NOT NULL REFERENCES claims.organization(organization_id),

    -- Subscription details
    subscription_name VARCHAR(255) NOT NULL,
    report_type VARCHAR(100) NOT NULL, -- CODING_ACCURACY, DENIAL_ANALYSIS, PROVIDER_PRODUCTIVITY, etc.

    -- Schedule
    frequency VARCHAR(50) NOT NULL, -- DAILY, WEEKLY, MONTHLY, QUARTERLY
    delivery_day_of_week INTEGER, -- 0=Sunday, 6=Saturday
    delivery_day_of_month INTEGER, -- 1-31
    delivery_time TIME,

    -- Report parameters
    report_parameters JSONB, -- Filters, date ranges, etc.

    -- Recipients
    recipient_emails TEXT[] NOT NULL,
    recipient_names TEXT[],

    -- Format options
    output_format VARCHAR(20) DEFAULT 'PDF', -- PDF, EXCEL, CSV
    include_charts BOOLEAN DEFAULT true,
    include_raw_data BOOLEAN DEFAULT false,

    -- Delivery method
    delivery_method VARCHAR(50) DEFAULT 'EMAIL', -- EMAIL, SFTP, API
    delivery_config JSONB, -- Configuration for delivery method

    -- Status
    is_active BOOLEAN DEFAULT true,
    last_generated_at TIMESTAMPTZ,
    last_delivered_at TIMESTAMPTZ,
    next_delivery_at TIMESTAMPTZ,

    -- Audit trail
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100),
    updated_by VARCHAR(100),

    UNIQUE(organization_id, subscription_name)
);

CREATE INDEX idx_report_subscription_org ON staging.report_subscription(organization_id);
CREATE INDEX idx_report_subscription_type ON staging.report_subscription(report_type);
CREATE INDEX idx_report_subscription_active ON staging.report_subscription(is_active) WHERE is_active = true;
CREATE INDEX idx_report_subscription_next_delivery ON staging.report_subscription(next_delivery_at) WHERE is_active = true;

COMMENT ON TABLE staging.report_subscription IS 'Scheduled report subscriptions';

CREATE TRIGGER update_report_subscription_updated_at BEFORE UPDATE ON staging.report_subscription
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Report generation log
CREATE TABLE staging.report_generation_log (
    report_log_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    subscription_id UUID REFERENCES staging.report_subscription(subscription_id) ON DELETE SET NULL,
    organization_id UUID NOT NULL REFERENCES claims.organization(organization_id),

    -- Report details
    report_type VARCHAR(100) NOT NULL,
    report_name VARCHAR(255) NOT NULL,
    report_parameters JSONB,

    -- Generation
    generated_at TIMESTAMPTZ NOT NULL,
    generation_duration_seconds INTEGER,
    generation_status VARCHAR(50) NOT NULL, -- SUCCESS, FAILURE, PARTIAL

    -- Output
    output_format VARCHAR(20),
    file_path TEXT,
    file_size_bytes BIGINT,

    -- Delivery
    delivery_method VARCHAR(50),
    delivered_at TIMESTAMPTZ,
    delivery_status VARCHAR(50), -- PENDING, SENT, FAILED
    delivery_error TEXT,

    -- Recipients
    recipients TEXT[],

    -- Content summary
    record_count INTEGER,
    page_count INTEGER,

    -- Error tracking
    error_message TEXT,

    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_report_generation_subscription ON staging.report_generation_log(subscription_id);
CREATE INDEX idx_report_generation_org ON staging.report_generation_log(organization_id);
CREATE INDEX idx_report_generation_type ON staging.report_generation_log(report_type);
CREATE INDEX idx_report_generation_generated ON staging.report_generation_log(generated_at);
CREATE INDEX idx_report_generation_status ON staging.report_generation_log(generation_status);

COMMENT ON TABLE staging.report_generation_log IS 'Log of report generation and delivery';

-- Data refresh schedule (for materialized views and summary tables)
CREATE TABLE staging.data_refresh_schedule (
    refresh_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID REFERENCES claims.organization(organization_id),

    -- Target identification
    target_type VARCHAR(50) NOT NULL, -- MATERIALIZED_VIEW, SUMMARY_TABLE, CACHE
    target_name VARCHAR(255) NOT NULL,
    schema_name VARCHAR(50) NOT NULL,

    -- Refresh configuration
    refresh_type VARCHAR(50) NOT NULL, -- FULL, INCREMENTAL
    refresh_method VARCHAR(50), -- CONCURRENT, EXCLUSIVE

    -- Schedule
    schedule_type VARCHAR(50) NOT NULL, -- CRON, INTERVAL, TRIGGER
    cron_expression VARCHAR(100),
    interval_minutes INTEGER,

    -- Dependencies
    depends_on TEXT[], -- Array of other refresh targets that must complete first
    execution_order INTEGER DEFAULT 100,

    -- Status
    is_active BOOLEAN DEFAULT true,
    last_refreshed_at TIMESTAMPTZ,
    last_refresh_duration_seconds INTEGER,
    next_refresh_at TIMESTAMPTZ,

    -- Performance tracking
    average_duration_seconds INTEGER,
    total_refreshes INTEGER DEFAULT 0,

    -- Error tracking
    last_error_message TEXT,
    consecutive_failures INTEGER DEFAULT 0,

    -- Audit trail
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(schema_name, target_name)
);

CREATE INDEX idx_data_refresh_org ON staging.data_refresh_schedule(organization_id);
CREATE INDEX idx_data_refresh_target_type ON staging.data_refresh_schedule(target_type);
CREATE INDEX idx_data_refresh_active ON staging.data_refresh_schedule(is_active) WHERE is_active = true;
CREATE INDEX idx_data_refresh_next ON staging.data_refresh_schedule(next_refresh_at) WHERE is_active = true;
CREATE INDEX idx_data_refresh_order ON staging.data_refresh_schedule(execution_order);

COMMENT ON TABLE staging.data_refresh_schedule IS 'Schedule for refreshing materialized views and summary tables';

CREATE TRIGGER update_data_refresh_schedule_updated_at BEFORE UPDATE ON staging.data_refresh_schedule
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
