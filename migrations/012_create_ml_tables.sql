-- Migration: 012_create_ml_tables
-- Description: Create machine learning schema tables for predictive analytics
-- Date: 2025-10-14

-- ML model registry
CREATE TABLE ml.model_registry (
    model_id BIGINT GENERATED ALWAYS AS IDENTITY (CACHE 100) PRIMARY KEY,
    organization_id BIGINT REFERENCES claims.organization(organization_id),

    -- Model identification
    model_name VARCHAR(255) NOT NULL,
    model_type VARCHAR(50) NOT NULL, -- CLASSIFICATION, REGRESSION, CLUSTERING, ANOMALY_DETECTION
    model_purpose VARCHAR(100) NOT NULL, -- DENIAL_PREDICTION, CODING_SUGGESTION, AUDIT_RISK, etc.
    model_version VARCHAR(50) NOT NULL,

    -- Model details
    algorithm VARCHAR(100), -- RANDOM_FOREST, GRADIENT_BOOSTING, NEURAL_NETWORK, etc.
    framework VARCHAR(50), -- SCIKIT_LEARN, TENSORFLOW, PYTORCH, etc.
    model_description TEXT,

    -- Training information
    training_dataset_size INTEGER,
    training_start_date DATE,
    training_end_date DATE,
    trained_at TIMESTAMPTZ,
    trained_by VARCHAR(100),

    -- Performance metrics
    accuracy NUMERIC(5,4),
    precision_score NUMERIC(5,4),
    recall_score NUMERIC(5,4),
    f1_score NUMERIC(5,4),
    auc_roc NUMERIC(5,4),
    mean_absolute_error NUMERIC(15,4),
    root_mean_squared_error NUMERIC(15,4),

    -- Model artifacts
    model_file_path TEXT,
    model_file_size_bytes BIGINT,
    model_hash VARCHAR(64),
    feature_list TEXT[],
    feature_importance JSONB,

    -- Hyperparameters
    hyperparameters JSONB,

    -- Deployment
    deployment_status VARCHAR(50) DEFAULT 'DEVELOPMENT', -- DEVELOPMENT, STAGING, PRODUCTION, RETIRED
    deployed_at TIMESTAMPTZ,
    retirement_date DATE,

    -- Usage tracking
    prediction_count INTEGER DEFAULT 0,
    last_prediction_at TIMESTAMPTZ,

    -- Validation
    validation_score NUMERIC(5,4),
    cross_validation_scores NUMERIC(5,4)[],

    -- Status
    is_active BOOLEAN DEFAULT true,

    -- Audit trail
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100),

    UNIQUE(model_name, model_version)
);

CREATE INDEX idx_model_registry_org ON ml.model_registry(organization_id);
CREATE INDEX idx_model_registry_type ON ml.model_registry(model_type);
CREATE INDEX idx_model_registry_purpose ON ml.model_registry(model_purpose);
CREATE INDEX idx_model_registry_status ON ml.model_registry(deployment_status);
CREATE INDEX idx_model_registry_active ON ml.model_registry(is_active) WHERE is_active = true;

COMMENT ON TABLE ml.model_registry IS 'Registry of trained machine learning models';

CREATE TRIGGER update_model_registry_updated_at BEFORE UPDATE ON ml.model_registry
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Model predictions
CREATE TABLE ml.model_prediction (
    prediction_id BIGINT GENERATED ALWAYS AS IDENTITY (CACHE 100) PRIMARY KEY,
    model_id BIGINT NOT NULL REFERENCES ml.model_registry(model_id),
    encounter_id BIGINT REFERENCES claims.encounter(encounter_id) ON DELETE CASCADE,
    service_line_id BIGINT REFERENCES claims.service_line(service_line_id) ON DELETE CASCADE,

    -- Prediction details
    prediction_type VARCHAR(50) NOT NULL, -- DENIAL_RISK, CODING_ERROR, AUDIT_PRIORITY, etc.
    predicted_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,

    -- Input features
    input_features JSONB NOT NULL,

    -- Prediction output
    predicted_value VARCHAR(255),
    predicted_class VARCHAR(100),
    prediction_score NUMERIC(8,6), -- Confidence score 0-1
    prediction_probability NUMERIC(8,6),

    -- Classification predictions
    class_probabilities JSONB, -- For multi-class predictions

    -- Risk scoring
    risk_score NUMERIC(8,4),
    risk_level VARCHAR(20), -- LOW, MEDIUM, HIGH, CRITICAL

    -- Explanation
    feature_contributions JSONB, -- SHAP values or similar
    explanation_text TEXT,
    top_influencing_features TEXT[],

    -- Actual outcome (for validation)
    actual_value VARCHAR(255),
    actual_class VARCHAR(100),
    outcome_recorded_at TIMESTAMPTZ,

    -- Prediction accuracy
    was_correct BOOLEAN,
    prediction_error NUMERIC(15,4),

    -- Action taken
    action_taken VARCHAR(100),
    action_result VARCHAR(100),

    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_model_prediction_model ON ml.model_prediction(model_id);
CREATE INDEX idx_model_prediction_encounter ON ml.model_prediction(encounter_id);
CREATE INDEX idx_model_prediction_service_line ON ml.model_prediction(service_line_id);
CREATE INDEX idx_model_prediction_type ON ml.model_prediction(prediction_type);
CREATE INDEX idx_model_prediction_predicted_at ON ml.model_prediction(predicted_at);
CREATE INDEX idx_model_prediction_risk_level ON ml.model_prediction(risk_level);
CREATE INDEX idx_model_prediction_score ON ml.model_prediction(prediction_score);

COMMENT ON TABLE ml.model_prediction IS 'Predictions made by ML models';

-- Feature engineering definitions
CREATE TABLE ml.feature_definition (
    feature_id BIGINT GENERATED ALWAYS AS IDENTITY (CACHE 100) PRIMARY KEY,

    -- Feature identification
    feature_name VARCHAR(255) NOT NULL UNIQUE,
    feature_category VARCHAR(100), -- DEMOGRAPHIC, FINANCIAL, TEMPORAL, PROVIDER, CODING, etc.
    feature_type VARCHAR(50) NOT NULL, -- NUMERIC, CATEGORICAL, BOOLEAN, TEXT, DATETIME

    -- Feature calculation
    calculation_logic TEXT NOT NULL, -- SQL query or function to calculate feature
    calculation_type VARCHAR(50), -- SQL, FUNCTION, AGGREGATION, DERIVED

    -- Dependencies
    source_tables TEXT[],
    dependent_features TEXT[],

    -- Feature properties
    is_nullable BOOLEAN DEFAULT true,
    default_value VARCHAR(255),
    allowed_values TEXT[], -- For categorical features

    -- Statistics (updated periodically)
    mean_value NUMERIC(15,4),
    std_deviation NUMERIC(15,4),
    min_value NUMERIC(15,4),
    max_value NUMERIC(15,4),
    distinct_count INTEGER,
    null_percentage NUMERIC(5,2),

    -- Normalization
    normalization_method VARCHAR(50), -- STANDARDIZE, MIN_MAX, LOG, NONE
    normalization_params JSONB,

    -- Usage tracking
    used_in_models TEXT[], -- Array of model names using this feature
    importance_scores JSONB, -- Importance scores from different models

    -- Status
    is_active BOOLEAN DEFAULT true,

    -- Audit trail
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100)
);

CREATE INDEX idx_feature_definition_name ON ml.feature_definition(feature_name);
CREATE INDEX idx_feature_definition_category ON ml.feature_definition(feature_category);
CREATE INDEX idx_feature_definition_type ON ml.feature_definition(feature_type);
CREATE INDEX idx_feature_definition_active ON ml.feature_definition(is_active) WHERE is_active = true;

COMMENT ON TABLE ml.feature_definition IS 'Definitions of features for ML models';

CREATE TRIGGER update_feature_definition_updated_at BEFORE UPDATE ON ml.feature_definition
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Training datasets
CREATE TABLE ml.training_dataset (
    dataset_id BIGINT GENERATED ALWAYS AS IDENTITY (CACHE 100) PRIMARY KEY,
    organization_id BIGINT REFERENCES claims.organization(organization_id),

    -- Dataset identification
    dataset_name VARCHAR(255) NOT NULL,
    dataset_version VARCHAR(50) NOT NULL,
    dataset_purpose VARCHAR(100), -- Model purpose this dataset is for

    -- Dataset details
    dataset_description TEXT,
    record_count INTEGER NOT NULL,
    feature_count INTEGER NOT NULL,
    features_included TEXT[],

    -- Date range
    data_start_date DATE,
    data_end_date DATE,

    -- Target variable
    target_variable VARCHAR(100),
    target_distribution JSONB, -- Distribution of target classes/values

    -- Class balance (for classification)
    is_balanced BOOLEAN,
    class_weights JSONB,

    -- Data splits
    training_split_percentage NUMERIC(5,2) DEFAULT 70.00,
    validation_split_percentage NUMERIC(5,2) DEFAULT 15.00,
    test_split_percentage NUMERIC(5,2) DEFAULT 15.00,

    -- Quality metrics
    missing_data_percentage NUMERIC(5,2),
    duplicate_records INTEGER,
    outlier_records INTEGER,

    -- Storage
    dataset_file_path TEXT,
    dataset_file_size_bytes BIGINT,
    dataset_format VARCHAR(20), -- CSV, PARQUET, JSON

    -- Status
    dataset_status VARCHAR(50) DEFAULT 'ACTIVE', -- ACTIVE, ARCHIVED, DEPRECATED
    is_active BOOLEAN DEFAULT true,

    -- Audit trail
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100),

    UNIQUE(dataset_name, dataset_version)
);

CREATE INDEX idx_training_dataset_org ON ml.training_dataset(organization_id);
CREATE INDEX idx_training_dataset_purpose ON ml.training_dataset(dataset_purpose);
CREATE INDEX idx_training_dataset_status ON ml.training_dataset(dataset_status);
CREATE INDEX idx_training_dataset_active ON ml.training_dataset(is_active) WHERE is_active = true;

COMMENT ON TABLE ml.training_dataset IS 'Training datasets for ML models';

CREATE TRIGGER update_training_dataset_updated_at BEFORE UPDATE ON ml.training_dataset
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Model performance monitoring
CREATE TABLE ml.model_performance_log (
    performance_id BIGINT GENERATED ALWAYS AS IDENTITY (CACHE 100) PRIMARY KEY,
    model_id BIGINT NOT NULL REFERENCES ml.model_registry(model_id),

    -- Measurement period
    measurement_date DATE NOT NULL,
    period_start TIMESTAMPTZ NOT NULL,
    period_end TIMESTAMPTZ NOT NULL,

    -- Volume metrics
    prediction_count INTEGER DEFAULT 0,
    unique_encounters INTEGER DEFAULT 0,

    -- Performance metrics
    accuracy NUMERIC(5,4),
    precision_score NUMERIC(5,4),
    recall_score NUMERIC(5,4),
    f1_score NUMERIC(5,4),
    auc_roc NUMERIC(5,4),

    -- Error metrics
    mean_absolute_error NUMERIC(15,4),
    mean_squared_error NUMERIC(15,4),
    root_mean_squared_error NUMERIC(15,4),

    -- Drift detection
    feature_drift_detected BOOLEAN DEFAULT false,
    concept_drift_detected BOOLEAN DEFAULT false,
    drift_score NUMERIC(8,6),
    drifted_features TEXT[],

    -- Performance by segment
    performance_by_facility JSONB,
    performance_by_provider_type JSONB,
    performance_by_procedure_category JSONB,

    -- Confusion matrix (for classification)
    confusion_matrix JSONB,

    -- Resource usage
    average_prediction_time_ms NUMERIC(10,3),
    peak_memory_mb NUMERIC(15,2),

    -- Recommendations
    requires_retraining BOOLEAN DEFAULT false,
    performance_alert_level VARCHAR(20), -- GOOD, WARNING, CRITICAL

    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_model_performance_model ON ml.model_performance_log(model_id);
CREATE INDEX idx_model_performance_date ON ml.model_performance_log(measurement_date);
CREATE INDEX idx_model_performance_period ON ml.model_performance_log(period_start, period_end);
CREATE INDEX idx_model_performance_alert ON ml.model_performance_log(performance_alert_level)
    WHERE performance_alert_level IN ('WARNING', 'CRITICAL');

COMMENT ON TABLE ml.model_performance_log IS 'Performance monitoring for deployed ML models';

-- A/B testing experiments
CREATE TABLE ml.ab_test_experiment (
    experiment_id BIGINT GENERATED ALWAYS AS IDENTITY (CACHE 100) PRIMARY KEY,
    organization_id BIGINT REFERENCES claims.organization(organization_id),

    -- Experiment details
    experiment_name VARCHAR(255) NOT NULL,
    experiment_description TEXT,
    hypothesis TEXT,

    -- Models being compared
    control_model_id BIGINT REFERENCES ml.model_registry(model_id),
    treatment_model_id BIGINT REFERENCES ml.model_registry(model_id),

    -- Traffic split
    control_percentage NUMERIC(5,2) DEFAULT 50.00,
    treatment_percentage NUMERIC(5,2) DEFAULT 50.00,

    -- Period
    start_date DATE NOT NULL,
    end_date DATE,
    planned_duration_days INTEGER,

    -- Sample size
    target_sample_size INTEGER,
    current_sample_size INTEGER DEFAULT 0,

    -- Results
    control_metric_value NUMERIC(15,4),
    treatment_metric_value NUMERIC(15,4),
    metric_difference NUMERIC(15,4),
    statistical_significance NUMERIC(5,4), -- p-value
    is_significant BOOLEAN,

    -- Winner
    winner VARCHAR(20), -- CONTROL, TREATMENT, INCONCLUSIVE
    winner_declared_at TIMESTAMPTZ,

    -- Status
    experiment_status VARCHAR(50) DEFAULT 'DRAFT', -- DRAFT, RUNNING, COMPLETED, CANCELLED
    is_active BOOLEAN DEFAULT false,

    -- Audit trail
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100),

    UNIQUE(experiment_name)
);

CREATE INDEX idx_ab_test_org ON ml.ab_test_experiment(organization_id);
CREATE INDEX idx_ab_test_status ON ml.ab_test_experiment(experiment_status);
CREATE INDEX idx_ab_test_active ON ml.ab_test_experiment(is_active) WHERE is_active = true;

COMMENT ON TABLE ml.ab_test_experiment IS 'A/B testing experiments for model comparison';

CREATE TRIGGER update_ab_test_experiment_updated_at BEFORE UPDATE ON ml.ab_test_experiment
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
