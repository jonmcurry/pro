--
-- PostgreSQL database dump
--

\restrict 9eBTtefqbDVKRlAdhPhpyOZSjCFnruKLYzWN4BATBGLDOrR3yeXwpf0jvHH6eBC

-- Dumped from database version 17.6
-- Dumped by pg_dump version 17.6

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: claims; Type: SCHEMA; Schema: -; Owner: -
--

CREATE SCHEMA claims;


--
-- Name: SCHEMA claims; Type: COMMENT; Schema: -; Owner: -
--

COMMENT ON SCHEMA claims IS 'Main schema for claims data, flags, audits, and denials';


--
-- Name: ml; Type: SCHEMA; Schema: -; Owner: -
--

CREATE SCHEMA ml;


--
-- Name: SCHEMA ml; Type: COMMENT; Schema: -; Owner: -
--

COMMENT ON SCHEMA ml IS 'Machine learning schema for predictive models and features';


--
-- Name: staging; Type: SCHEMA; Schema: -; Owner: -
--

CREATE SCHEMA staging;


--
-- Name: SCHEMA staging; Type: COMMENT; Schema: -; Owner: -
--

COMMENT ON SCHEMA staging IS 'Staging schema for import, configuration, and processing';


--
-- Name: cleanup_old_queue_entries(integer); Type: FUNCTION; Schema: staging; Owner: -
--

CREATE FUNCTION staging.cleanup_old_queue_entries(retention_days integer DEFAULT 90) RETURNS integer
    LANGUAGE plpgsql
    AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM staging.file_processing_queue
    WHERE queue_status IN ('COMPLETED', 'FAILED')
      AND processing_completed_at < CURRENT_TIMESTAMP - (retention_days || ' days')::INTERVAL;

    GET DIAGNOSTICS deleted_count = ROW_COUNT;

    RETURN deleted_count;
END;
$$;


--
-- Name: FUNCTION cleanup_old_queue_entries(retention_days integer); Type: COMMENT; Schema: staging; Owner: -
--

COMMENT ON FUNCTION staging.cleanup_old_queue_entries(retention_days integer) IS 'Removes completed/failed queue entries older than specified days (default 90)';


--
-- Name: update_queue_updated_at(); Type: FUNCTION; Schema: staging; Owner: -
--

CREATE FUNCTION staging.update_queue_updated_at() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$;


SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: encounter; Type: TABLE; Schema: claims; Owner: -
--

CREATE TABLE claims.encounter (
    encounter_id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    facility_id uuid NOT NULL,
    organization_id uuid NOT NULL,
    region_id uuid,
    submitter_id character varying(80) NOT NULL,
    submitter_name character varying(255),
    patient_control_number character varying(38) NOT NULL,
    transaction_set_control_number character varying(9),
    subscriber_id character varying(80) NOT NULL,
    subscriber_last_name character varying(255) NOT NULL,
    subscriber_first_name character varying(255) NOT NULL,
    subscriber_middle_name character varying(255),
    subscriber_name_suffix character varying(50),
    subscriber_gender character(1),
    subscriber_birth_date date NOT NULL,
    subscriber_address_line1 character varying(255),
    subscriber_address_line2 character varying(255),
    subscriber_city character varying(100),
    subscriber_state character(2),
    subscriber_postal_code character varying(15),
    subscriber_country character(3) DEFAULT 'USA'::bpchar,
    payer_responsibility_code character(1) NOT NULL,
    payer_id character varying(80),
    payer_name character varying(255),
    claim_filing_indicator character varying(2) DEFAULT 'MB'::character varying,
    billing_provider_id uuid,
    billing_provider_npi character varying(10),
    billing_provider_tax_id character varying(20),
    billing_provider_name character varying(255),
    billing_provider_address_line1 character varying(255),
    billing_provider_address_line2 character varying(255),
    billing_provider_city character varying(100),
    billing_provider_state character(2),
    billing_provider_postal_code character varying(15),
    total_claim_charge_amount numeric(18,2) NOT NULL,
    place_of_service_code character varying(2),
    claim_frequency_code character(1) DEFAULT '1'::bpchar,
    signature_indicator character(1),
    assignment_indicator character(1),
    benefits_assignment_indicator character(1),
    release_of_information_code character(1),
    patient_signature_code character(1),
    date_of_service_from date NOT NULL,
    date_of_service_to date,
    onset_of_illness_date date,
    initial_treatment_date date,
    last_seen_date date,
    acute_manifestation_date date,
    accident_date date,
    last_menstrual_period_date date,
    last_xray_date date,
    prescription_date date,
    disability_from_date date,
    disability_to_date date,
    last_worked_date date,
    authorized_return_to_work_date date,
    admission_date date,
    discharge_date date,
    assumed_care_date date,
    relinquished_care_date date,
    delay_reason_code character varying(2),
    special_program_code character varying(3),
    patient_amount_paid numeric(18,2),
    service_authorization_code character varying(50),
    referring_provider_id uuid,
    referring_provider_npi character varying(10),
    referring_provider_name character varying(255),
    rendering_provider_id uuid,
    rendering_provider_npi character varying(10),
    rendering_provider_name character varying(255),
    service_facility_id uuid,
    service_facility_npi character varying(10),
    service_facility_name character varying(255),
    service_facility_address_line1 character varying(255),
    service_facility_address_line2 character varying(255),
    service_facility_city character varying(100),
    service_facility_state character(2),
    service_facility_postal_code character varying(15),
    supervising_provider_id uuid,
    supervising_provider_npi character varying(10),
    supervising_provider_name character varying(255),
    other_payer_paid_amount numeric(18,2),
    other_payer_id character varying(80),
    other_payer_name character varying(255),
    other_payer_claim_number character varying(50),
    other_payer_claim_filing_indicator character varying(2),
    ambulance_transport_reason_code character(1),
    ambulance_transport_distance numeric(15,4),
    ambulance_patient_weight numeric(10,2),
    ambulance_patient_count integer,
    coder_id uuid,
    coding_date date,
    claim_status character varying(50) DEFAULT 'NEW'::character varying,
    case_status character varying(50),
    financial_class character varying(50),
    import_batch_id uuid,
    import_date timestamp with time zone,
    import_configuration_id uuid,
    is_active boolean DEFAULT true,
    soft_deleted boolean DEFAULT false,
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    created_by character varying(100),
    updated_by character varying(100),
    CONSTRAINT chk_dos_range CHECK (((date_of_service_to IS NULL) OR (date_of_service_to >= date_of_service_from))),
    CONSTRAINT chk_payer_responsibility CHECK ((payer_responsibility_code = ANY (ARRAY['P'::bpchar, 'S'::bpchar]))),
    CONSTRAINT encounter_total_claim_charge_amount_check CHECK (((total_claim_charge_amount >= (0)::numeric) AND (total_claim_charge_amount <= 99999.99)))
);


--
-- Name: TABLE encounter; Type: COMMENT; Schema: claims; Owner: -
--

COMMENT ON TABLE claims.encounter IS 'Main encounter/claim table containing all 837p claim-level data elements';


--
-- Name: service_line; Type: TABLE; Schema: claims; Owner: -
--

CREATE TABLE claims.service_line (
    service_line_id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    encounter_id uuid NOT NULL,
    line_number smallint NOT NULL,
    product_service_id_qualifier character varying(2) DEFAULT 'HC'::character varying,
    procedure_code character varying(48) NOT NULL,
    procedure_modifier_1 character varying(2),
    procedure_modifier_2 character varying(2),
    procedure_modifier_3 character varying(2),
    procedure_modifier_4 character varying(2),
    procedure_description text,
    line_item_charge_amount numeric(18,2) NOT NULL,
    unit_basis_measurement_code character varying(2) DEFAULT 'UN'::character varying,
    service_unit_count numeric(15,1) NOT NULL,
    place_of_service_code character varying(2),
    service_date_from date NOT NULL,
    service_date_to date,
    emergency_indicator boolean DEFAULT false,
    epsdt_indicator boolean DEFAULT false,
    family_planning_indicator boolean DEFAULT false,
    rendering_provider_id uuid,
    rendering_provider_npi character varying(10),
    supervising_provider_id uuid,
    supervising_provider_npi character varying(10),
    ordering_provider_id uuid,
    ordering_provider_npi character varying(10),
    referring_provider_id uuid,
    referring_provider_npi character varying(10),
    service_facility_id uuid,
    service_facility_npi character varying(10),
    prior_authorization_number character varying(50),
    referral_number character varying(50),
    line_note text,
    revenue_code character varying(4),
    ndc_code character varying(11),
    ndc_unit_count numeric(15,3),
    ndc_measurement_unit character varying(2),
    dme_rental_price numeric(18,2),
    dme_purchase_price numeric(18,2),
    dme_frequency_code character varying(1),
    anesthesia_minutes integer,
    obstetric_additional_units integer,
    test_result_value numeric(20,1),
    test_result_measurement_code character varying(20),
    ambulance_patient_count integer,
    ambulance_transport_distance numeric(15,4),
    ambulance_patient_weight numeric(10,2),
    diagnosis_code_pointer_1 smallint,
    diagnosis_code_pointer_2 smallint,
    diagnosis_code_pointer_3 smallint,
    diagnosis_code_pointer_4 smallint,
    diagnosis_code_pointer_5 smallint,
    diagnosis_code_pointer_6 smallint,
    diagnosis_code_pointer_7 smallint,
    diagnosis_code_pointer_8 smallint,
    diagnosis_code_pointer_9 smallint,
    diagnosis_code_pointer_10 smallint,
    diagnosis_code_pointer_11 smallint,
    diagnosis_code_pointer_12 smallint,
    other_payer_line_paid_amount numeric(18,2),
    other_payer_line_service_id character varying(48),
    line_status character varying(50) DEFAULT 'ACTIVE'::character varying,
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    created_by character varying(100),
    updated_by character varying(100),
    CONSTRAINT chk_service_date_range CHECK (((service_date_to IS NULL) OR (service_date_to >= service_date_from))),
    CONSTRAINT service_line_line_item_charge_amount_check CHECK ((line_item_charge_amount >= (0)::numeric)),
    CONSTRAINT service_line_service_unit_count_check CHECK (((service_unit_count > (0)::numeric) AND (service_unit_count <= 9999.9)))
);


--
-- Name: TABLE service_line; Type: COMMENT; Schema: claims; Owner: -
--

COMMENT ON TABLE claims.service_line IS 'Service line items (procedures) for encounters - Loop 2400 data';


--
-- Name: model_prediction; Type: TABLE; Schema: ml; Owner: -
--

CREATE TABLE ml.model_prediction (
    prediction_id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    model_id uuid NOT NULL,
    encounter_id uuid,
    service_line_id uuid,
    prediction_type character varying(50) NOT NULL,
    predicted_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    input_features jsonb NOT NULL,
    predicted_value character varying(255),
    predicted_class character varying(100),
    prediction_score numeric(8,6),
    prediction_probability numeric(8,6),
    class_probabilities jsonb,
    risk_score numeric(8,4),
    risk_level character varying(20),
    feature_contributions jsonb,
    explanation_text text,
    top_influencing_features text[],
    actual_value character varying(255),
    actual_class character varying(100),
    outcome_recorded_at timestamp with time zone,
    was_correct boolean,
    prediction_error numeric(15,4),
    action_taken character varying(100),
    action_result character varying(100),
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP
);


--
-- Name: TABLE model_prediction; Type: COMMENT; Schema: ml; Owner: -
--

COMMENT ON TABLE ml.model_prediction IS 'Predictions made by ML models';


--
-- Name: model_registry; Type: TABLE; Schema: ml; Owner: -
--

CREATE TABLE ml.model_registry (
    model_id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    organization_id uuid,
    model_name character varying(255) NOT NULL,
    model_type character varying(50) NOT NULL,
    model_purpose character varying(100) NOT NULL,
    model_version character varying(50) NOT NULL,
    algorithm character varying(100),
    framework character varying(50),
    model_description text,
    training_dataset_size integer,
    training_start_date date,
    training_end_date date,
    trained_at timestamp with time zone,
    trained_by character varying(100),
    accuracy numeric(5,4),
    precision_score numeric(5,4),
    recall_score numeric(5,4),
    f1_score numeric(5,4),
    auc_roc numeric(5,4),
    mean_absolute_error numeric(15,4),
    root_mean_squared_error numeric(15,4),
    model_file_path text,
    model_file_size_bytes bigint,
    model_hash character varying(64),
    feature_list text[],
    feature_importance jsonb,
    hyperparameters jsonb,
    deployment_status character varying(50) DEFAULT 'DEVELOPMENT'::character varying,
    deployed_at timestamp with time zone,
    retirement_date date,
    prediction_count integer DEFAULT 0,
    last_prediction_at timestamp with time zone,
    validation_score numeric(5,4),
    cross_validation_scores numeric(5,4)[],
    is_active boolean DEFAULT true,
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    created_by character varying(100)
);


--
-- Name: TABLE model_registry; Type: COMMENT; Schema: ml; Owner: -
--

COMMENT ON TABLE ml.model_registry IS 'Registry of trained machine learning models';


--
-- Name: audit_assignment; Type: TABLE; Schema: claims; Owner: -
--

CREATE TABLE claims.audit_assignment (
    audit_id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    organization_id uuid NOT NULL,
    facility_id uuid,
    audit_name character varying(255) NOT NULL,
    audit_type character varying(50) NOT NULL,
    audit_scope character varying(50) NOT NULL,
    selection_criteria jsonb,
    total_population integer,
    sample_size integer,
    sampling_method character varying(50),
    dos_from date,
    dos_to date,
    reviewer_id uuid,
    assigned_at timestamp with time zone,
    due_date date,
    audit_status character varying(50) DEFAULT 'ASSIGNED'::character varying,
    completed_at timestamp with time zone,
    completion_percentage numeric(5,2) DEFAULT 0.00,
    encounters_reviewed integer DEFAULT 0,
    encounters_with_errors integer DEFAULT 0,
    total_flags_found integer DEFAULT 0,
    error_rate numeric(5,2),
    total_billed_amount numeric(18,2),
    total_overpayment_amount numeric(18,2),
    total_underpayment_amount numeric(18,2),
    net_financial_impact numeric(18,2),
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    created_by character varying(100),
    updated_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    updated_by character varying(100)
);


--
-- Name: TABLE audit_assignment; Type: COMMENT; Schema: claims; Owner: -
--

COMMENT ON TABLE claims.audit_assignment IS 'Audit assignments for retrospective claim reviews';


--
-- Name: audit_encounter; Type: TABLE; Schema: claims; Owner: -
--

CREATE TABLE claims.audit_encounter (
    audit_encounter_id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    audit_id uuid NOT NULL,
    encounter_id uuid NOT NULL,
    review_status character varying(50) DEFAULT 'PENDING'::character varying,
    reviewed_at timestamp with time zone,
    review_duration_minutes integer,
    has_errors boolean DEFAULT false,
    error_count integer DEFAULT 0,
    severity_high_count integer DEFAULT 0,
    severity_medium_count integer DEFAULT 0,
    severity_low_count integer DEFAULT 0,
    original_billed_amount numeric(18,2),
    corrected_billed_amount numeric(18,2),
    overpayment_amount numeric(18,2),
    underpayment_amount numeric(18,2),
    net_financial_impact numeric(18,2),
    reviewer_notes text,
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP
);


--
-- Name: TABLE audit_encounter; Type: COMMENT; Schema: claims; Owner: -
--

COMMENT ON TABLE claims.audit_encounter IS 'Encounters selected for audit review';


--
-- Name: coder; Type: TABLE; Schema: claims; Owner: -
--

CREATE TABLE claims.coder (
    coder_id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    coder_code character varying(50) NOT NULL,
    last_name character varying(255) NOT NULL,
    first_name character varying(255) NOT NULL,
    middle_name character varying(255),
    coder_group character varying(100),
    certifications text[],
    organization_id uuid,
    email public.citext,
    is_active boolean DEFAULT true,
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    created_by character varying(100),
    updated_by character varying(100)
);


--
-- Name: TABLE coder; Type: COMMENT; Schema: claims; Owner: -
--

COMMENT ON TABLE claims.coder IS 'Medical coders who code/bill claims';


--
-- Name: coder_accuracy; Type: TABLE; Schema: claims; Owner: -
--

CREATE TABLE claims.coder_accuracy (
    accuracy_id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    coder_id uuid NOT NULL,
    organization_id uuid NOT NULL,
    period_start_date date NOT NULL,
    period_end_date date NOT NULL,
    period_type character varying(20) NOT NULL,
    encounters_coded integer DEFAULT 0,
    service_lines_coded integer DEFAULT 0,
    encounters_audited integer DEFAULT 0,
    encounters_with_errors integer DEFAULT 0,
    service_lines_audited integer DEFAULT 0,
    service_lines_with_errors integer DEFAULT 0,
    encounter_accuracy_rate numeric(5,2),
    service_line_accuracy_rate numeric(5,2),
    overall_accuracy_rate numeric(5,2),
    high_severity_errors integer DEFAULT 0,
    medium_severity_errors integer DEFAULT 0,
    low_severity_errors integer DEFAULT 0,
    coding_errors integer DEFAULT 0,
    documentation_errors integer DEFAULT 0,
    em_errors integer DEFAULT 0,
    modifier_errors integer DEFAULT 0,
    diagnosis_errors integer DEFAULT 0,
    other_errors integer DEFAULT 0,
    total_financial_impact numeric(18,2),
    overpayment_total numeric(18,2),
    underpayment_total numeric(18,2),
    average_encounters_per_day numeric(8,2),
    average_service_lines_per_encounter numeric(8,2),
    calculated_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP
);


--
-- Name: TABLE coder_accuracy; Type: COMMENT; Schema: claims; Owner: -
--

COMMENT ON TABLE claims.coder_accuracy IS 'Coder accuracy metrics over time periods';


--
-- Name: conversion_factor; Type: TABLE; Schema: claims; Owner: -
--

CREATE TABLE claims.conversion_factor (
    conversion_factor_id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    factor_year integer NOT NULL,
    effective_date date NOT NULL,
    termination_date date,
    conversion_factor numeric(10,4) NOT NULL,
    budget_neutrality_adjustment numeric(8,6) DEFAULT 1.000000,
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    created_by character varying(100)
);


--
-- Name: TABLE conversion_factor; Type: COMMENT; Schema: claims; Owner: -
--

COMMENT ON TABLE claims.conversion_factor IS 'Annual Medicare conversion factors for RVU to dollar conversion';


--
-- Name: denial_appeal; Type: TABLE; Schema: claims; Owner: -
--

CREATE TABLE claims.denial_appeal (
    appeal_id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    denial_id uuid NOT NULL,
    appeal_level character varying(20) NOT NULL,
    appeal_type character varying(50),
    appeal_method character varying(50),
    filed_date date NOT NULL,
    due_date date,
    decision_date date,
    appeal_received_date date,
    appeal_reason text,
    supporting_documentation text[],
    clinical_rationale text,
    payer_decision character varying(50),
    payer_response text,
    payer_decision_reason text,
    additional_payment_amount numeric(18,2),
    final_allowed_amount numeric(18,2),
    final_paid_amount numeric(18,2),
    appeal_status character varying(50) DEFAULT 'FILED'::character varying,
    assigned_to character varying(100),
    internal_notes text,
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    created_by character varying(100),
    updated_by character varying(100)
);


--
-- Name: TABLE denial_appeal; Type: COMMENT; Schema: claims; Owner: -
--

COMMENT ON TABLE claims.denial_appeal IS 'Appeal actions and correspondence for denied claims';


--
-- Name: denial_event; Type: TABLE; Schema: claims; Owner: -
--

CREATE TABLE claims.denial_event (
    denial_id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    encounter_id uuid NOT NULL,
    service_line_id uuid,
    organization_id uuid NOT NULL,
    facility_id uuid,
    denial_type character varying(50) NOT NULL,
    denial_category character varying(50) NOT NULL,
    payer_id character varying(80),
    payer_name character varying(255),
    claim_filing_indicator character varying(2),
    claim_adjustment_group_code character varying(2),
    claim_adjustment_reason_code character varying(5) NOT NULL,
    remittance_advice_remark_code character varying(5),
    denial_reason_description text,
    payer_denial_reason text,
    denied_amount numeric(18,2) NOT NULL,
    billed_amount numeric(18,2),
    allowed_amount numeric(18,2),
    paid_amount numeric(18,2) DEFAULT 0,
    service_date date NOT NULL,
    initial_submission_date date,
    denial_date date NOT NULL,
    received_date date,
    remittance_advice_number character varying(50),
    check_eft_number character varying(50),
    root_cause_category character varying(50),
    root_cause_subcategory character varying(100),
    root_cause_details text,
    responsible_party character varying(50),
    coder_id uuid,
    provider_id uuid,
    is_preventable boolean,
    preventable_category character varying(100),
    prevention_recommendations text,
    denial_status character varying(50) DEFAULT 'NEW'::character varying,
    resolution_status character varying(50),
    resolution_date date,
    appeal_filed boolean DEFAULT false,
    appeal_level character varying(20),
    appeal_deadline date,
    internal_notes text,
    resolution_notes text,
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    created_by character varying(100),
    updated_by character varying(100),
    CONSTRAINT denial_event_denied_amount_check CHECK ((denied_amount >= (0)::numeric))
);


--
-- Name: TABLE denial_event; Type: COMMENT; Schema: claims; Owner: -
--

COMMENT ON TABLE claims.denial_event IS 'Denial events from payer remittances with root cause analysis';


--
-- Name: denial_reason_code; Type: TABLE; Schema: claims; Owner: -
--

CREATE TABLE claims.denial_reason_code (
    reason_code_id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    code_type character varying(10) NOT NULL,
    reason_code character varying(5) NOT NULL,
    short_description character varying(255),
    long_description text,
    category character varying(50),
    subcategory character varying(100),
    recommended_action text,
    is_appealable boolean DEFAULT true,
    prevention_tips text,
    is_active boolean DEFAULT true,
    effective_date date,
    termination_date date,
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP
);


--
-- Name: TABLE denial_reason_code; Type: COMMENT; Schema: claims; Owner: -
--

COMMENT ON TABLE claims.denial_reason_code IS 'Reference table for CARC and RARC codes';


--
-- Name: denial_statistics; Type: TABLE; Schema: claims; Owner: -
--

CREATE TABLE claims.denial_statistics (
    statistic_id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    organization_id uuid NOT NULL,
    facility_id uuid,
    statistic_dimension character varying(50) NOT NULL,
    dimension_value character varying(255) NOT NULL,
    period_start_date date NOT NULL,
    period_end_date date NOT NULL,
    period_type character varying(20) NOT NULL,
    total_denials integer DEFAULT 0,
    total_denied_amount numeric(18,2) DEFAULT 0,
    total_billed_amount numeric(18,2) DEFAULT 0,
    denial_rate numeric(5,2),
    appeals_filed integer DEFAULT 0,
    appeals_won integer DEFAULT 0,
    appeals_lost integer DEFAULT 0,
    appeal_success_rate numeric(5,2),
    amount_recovered numeric(18,2) DEFAULT 0,
    amount_written_off numeric(18,2) DEFAULT 0,
    recovery_rate numeric(5,2),
    preventable_denials integer DEFAULT 0,
    preventable_amount numeric(18,2) DEFAULT 0,
    calculated_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP
);


--
-- Name: TABLE denial_statistics; Type: COMMENT; Schema: claims; Owner: -
--

COMMENT ON TABLE claims.denial_statistics IS 'Aggregated denial statistics by various dimensions';


--
-- Name: diagnosis_evaluation; Type: TABLE; Schema: claims; Owner: -
--

CREATE TABLE claims.diagnosis_evaluation (
    diagnosis_eval_id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    audit_encounter_id uuid NOT NULL,
    encounter_diagnosis_id uuid,
    reviewer_id uuid NOT NULL,
    original_diagnosis_code character varying(30),
    original_sequence_number smallint,
    original_is_principal boolean,
    corrected_diagnosis_code character varying(30),
    corrected_sequence_number smallint,
    corrected_is_principal boolean,
    evaluation_result character varying(50) NOT NULL,
    has_error boolean DEFAULT false,
    issue_id uuid,
    issue_description text,
    issue_severity character varying(20),
    documentation_sufficient boolean,
    documentation_notes text,
    hcc_impact boolean DEFAULT false,
    hcc_category_affected character varying(10),
    evaluated_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_dx_evaluation_result CHECK (((evaluation_result)::text = ANY ((ARRAY['CORRECT'::character varying, 'INCORRECT'::character varying, 'UNSUPPORTED'::character varying, 'MISSING'::character varying, 'ADDITIONAL_NEEDED'::character varying, 'SPECIFICITY_ISSUE'::character varying])::text[])))
);


--
-- Name: TABLE diagnosis_evaluation; Type: COMMENT; Schema: claims; Owner: -
--

COMMENT ON TABLE claims.diagnosis_evaluation IS 'Audit findings for diagnosis codes';


--
-- Name: encounter_diagnosis; Type: TABLE; Schema: claims; Owner: -
--

CREATE TABLE claims.encounter_diagnosis (
    diagnosis_id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    encounter_id uuid NOT NULL,
    sequence_number smallint NOT NULL,
    diagnosis_code_qualifier character varying(3) DEFAULT 'ABK'::character varying,
    diagnosis_code character varying(30) NOT NULL,
    diagnosis_description text,
    is_principal boolean DEFAULT false,
    is_admitting boolean DEFAULT false,
    is_external_cause boolean DEFAULT false,
    is_patient_reason boolean DEFAULT false,
    present_on_admission_indicator character(1),
    hcc_indicator boolean DEFAULT false,
    hcc_category character varying(10),
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_sequence_range CHECK (((sequence_number >= 1) AND (sequence_number <= 12)))
);


--
-- Name: TABLE encounter_diagnosis; Type: COMMENT; Schema: claims; Owner: -
--

COMMENT ON TABLE claims.encounter_diagnosis IS 'Diagnosis codes associated with encounters (ICD-10-CM)';


--
-- Name: encounter_flag; Type: TABLE; Schema: claims; Owner: -
--

CREATE TABLE claims.encounter_flag (
    flag_id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    encounter_id uuid NOT NULL,
    issue_id uuid NOT NULL,
    flag_type character varying(20) DEFAULT 'POST_BILL'::character varying,
    severity character varying(20),
    flag_reason text,
    flagged_element character varying(255),
    proposed_code character varying(50),
    proposed_modifier character varying(10),
    proposed_quantity numeric(15,3),
    proposed_diagnosis_code character varying(30),
    flag_status character varying(20) DEFAULT 'OPEN'::character varying,
    resolution_note text,
    resolved_at timestamp with time zone,
    resolved_by character varying(100),
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    created_by character varying(100) DEFAULT 'SYSTEM'::character varying
);


--
-- Name: TABLE encounter_flag; Type: COMMENT; Schema: claims; Owner: -
--

COMMENT ON TABLE claims.encounter_flag IS 'Flags assigned to encounters by the rules engine';


--
-- Name: encounter_note; Type: TABLE; Schema: claims; Owner: -
--

CREATE TABLE claims.encounter_note (
    note_id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    encounter_id uuid NOT NULL,
    note_type character varying(50),
    note_text text NOT NULL,
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    created_by character varying(100)
);


--
-- Name: TABLE encounter_note; Type: COMMENT; Schema: claims; Owner: -
--

COMMENT ON TABLE claims.encounter_note IS 'Notes and comments associated with encounters';


--
-- Name: facility; Type: TABLE; Schema: claims; Owner: -
--

CREATE TABLE claims.facility (
    facility_id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    organization_id uuid NOT NULL,
    region_id uuid,
    facility_code character varying(50) NOT NULL,
    facility_name character varying(255) NOT NULL,
    npi character varying(10),
    tax_id character varying(20),
    facility_type character varying(50),
    address_line1 character varying(255),
    address_line2 character varying(255),
    city character varying(100),
    state_code character(2),
    postal_code character varying(15),
    country_code character(3) DEFAULT 'USA'::bpchar,
    phone character varying(20),
    email public.citext,
    ehr_system character varying(100),
    is_active boolean DEFAULT true,
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    created_by character varying(100),
    updated_by character varying(100)
);


--
-- Name: TABLE facility; Type: COMMENT; Schema: claims; Owner: -
--

COMMENT ON TABLE claims.facility IS 'Facility entities - must belong to organization, optionally to region';


--
-- Name: flag_category; Type: TABLE; Schema: claims; Owner: -
--

CREATE TABLE claims.flag_category (
    category_id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    category_code character varying(10) NOT NULL,
    category_name character varying(100) NOT NULL,
    category_description text,
    is_active boolean DEFAULT true,
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP
);


--
-- Name: TABLE flag_category; Type: COMMENT; Schema: claims; Owner: -
--

COMMENT ON TABLE claims.flag_category IS 'Flag category definitions (COD, DOC, EMO, EMU, etc.)';


--
-- Name: flag_issue; Type: TABLE; Schema: claims; Owner: -
--

CREATE TABLE claims.flag_issue (
    issue_id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    category_id uuid NOT NULL,
    issue_code character varying(20) NOT NULL,
    issue_description text NOT NULL,
    severity character varying(20) DEFAULT 'MEDIUM'::character varying,
    is_active boolean DEFAULT true,
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP
);


--
-- Name: TABLE flag_issue; Type: COMMENT; Schema: claims; Owner: -
--

COMMENT ON TABLE claims.flag_issue IS 'Specific flag issue definitions with descriptions';


--
-- Name: gpci_reference; Type: TABLE; Schema: claims; Owner: -
--

CREATE TABLE claims.gpci_reference (
    gpci_id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    locality_code character varying(5) NOT NULL,
    locality_name character varying(255) NOT NULL,
    state_code character(2) NOT NULL,
    effective_year integer NOT NULL,
    effective_date date NOT NULL,
    termination_date date,
    work_gpci numeric(6,3) DEFAULT 1.000 NOT NULL,
    pe_gpci numeric(6,3) DEFAULT 1.000 NOT NULL,
    mp_gpci numeric(6,3) DEFAULT 1.000 NOT NULL,
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP
);


--
-- Name: TABLE gpci_reference; Type: COMMENT; Schema: claims; Owner: -
--

COMMENT ON TABLE claims.gpci_reference IS 'Geographic Practice Cost Indexes by Medicare locality';


--
-- Name: modifier_adjustment; Type: TABLE; Schema: claims; Owner: -
--

CREATE TABLE claims.modifier_adjustment (
    adjustment_id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    modifier_code character varying(2) NOT NULL,
    modifier_description text,
    payment_percentage numeric(5,2),
    payment_multiplier numeric(5,3),
    applies_to_professional boolean DEFAULT true,
    applies_to_technical boolean DEFAULT true,
    affects_rvu boolean DEFAULT true,
    combining_rules text,
    sequencing_rules text,
    is_active boolean DEFAULT true,
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP
);


--
-- Name: TABLE modifier_adjustment; Type: COMMENT; Schema: claims; Owner: -
--

COMMENT ON TABLE claims.modifier_adjustment IS 'Modifier-based reimbursement adjustment rules';


--
-- Name: mv_denial_statistics; Type: MATERIALIZED VIEW; Schema: claims; Owner: -
--

CREATE MATERIALIZED VIEW claims.mv_denial_statistics AS
 SELECT organization_id,
    facility_id,
    date_trunc('day'::text, (denial_date)::timestamp with time zone) AS denial_date,
    payer_id,
    root_cause_category,
    count(DISTINCT denial_id) AS denial_count,
    sum(denied_amount) AS total_denied_amount,
    count(DISTINCT
        CASE
            WHEN is_preventable THEN denial_id
            ELSE NULL::uuid
        END) AS preventable_count,
    count(DISTINCT
        CASE
            WHEN appeal_filed THEN denial_id
            ELSE NULL::uuid
        END) AS appeal_count
   FROM claims.denial_event de
  GROUP BY organization_id, facility_id, (date_trunc('day'::text, (denial_date)::timestamp with time zone)), payer_id, root_cause_category
  WITH NO DATA;


--
-- Name: MATERIALIZED VIEW mv_denial_statistics; Type: COMMENT; Schema: claims; Owner: -
--

COMMENT ON MATERIALIZED VIEW claims.mv_denial_statistics IS 'Pre-aggregated denial statistics for dashboard performance';


--
-- Name: mv_flag_statistics; Type: MATERIALIZED VIEW; Schema: claims; Owner: -
--

CREATE MATERIALIZED VIEW claims.mv_flag_statistics AS
 SELECT e.organization_id,
    e.facility_id,
    date_trunc('day'::text, ef.created_at) AS flag_date,
    fc.category_code,
    ef.severity,
    count(DISTINCT ef.flag_id) AS flag_count,
    count(DISTINCT ef.encounter_id) AS affected_encounters
   FROM (((claims.encounter_flag ef
     JOIN claims.flag_issue fi ON ((ef.issue_id = fi.issue_id)))
     JOIN claims.flag_category fc ON ((fi.category_id = fc.category_id)))
     JOIN claims.encounter e ON ((ef.encounter_id = e.encounter_id)))
  WHERE (e.is_active = true)
  GROUP BY e.organization_id, e.facility_id, (date_trunc('day'::text, ef.created_at)), fc.category_code, ef.severity
  WITH NO DATA;


--
-- Name: MATERIALIZED VIEW mv_flag_statistics; Type: COMMENT; Schema: claims; Owner: -
--

COMMENT ON MATERIALIZED VIEW claims.mv_flag_statistics IS 'Pre-aggregated flag statistics for dashboard performance';


--
-- Name: organization; Type: TABLE; Schema: claims; Owner: -
--

CREATE TABLE claims.organization (
    organization_id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    organization_code character varying(50) NOT NULL,
    organization_name character varying(255) NOT NULL,
    tax_id character varying(20),
    npi character varying(10),
    address_line1 character varying(255),
    address_line2 character varying(255),
    city character varying(100),
    state_code character(2),
    postal_code character varying(15),
    country_code character(3) DEFAULT 'USA'::bpchar,
    phone character varying(20),
    email public.citext,
    is_active boolean DEFAULT true,
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    created_by character varying(100),
    updated_by character varying(100)
);


--
-- Name: TABLE organization; Type: COMMENT; Schema: claims; Owner: -
--

COMMENT ON TABLE claims.organization IS 'Top-level organization entities';


--
-- Name: provider; Type: TABLE; Schema: claims; Owner: -
--

CREATE TABLE claims.provider (
    provider_id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    npi character varying(10) NOT NULL,
    provider_type character varying(50) NOT NULL,
    last_name character varying(255) NOT NULL,
    first_name character varying(255) NOT NULL,
    middle_name character varying(255),
    name_suffix character varying(50),
    taxonomy_code character varying(10),
    license_number character varying(50),
    license_state character(2),
    specialty character varying(100),
    provider_group character varying(255),
    organization_id uuid,
    address_line1 character varying(255),
    address_line2 character varying(255),
    city character varying(100),
    state_code character(2),
    postal_code character varying(15),
    country_code character(3) DEFAULT 'USA'::bpchar,
    phone character varying(20),
    email public.citext,
    is_active boolean DEFAULT true,
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    created_by character varying(100),
    updated_by character varying(100)
);


--
-- Name: TABLE provider; Type: COMMENT; Schema: claims; Owner: -
--

COMMENT ON TABLE claims.provider IS 'Healthcare providers (physicians, practitioners)';


--
-- Name: provider_accuracy; Type: TABLE; Schema: claims; Owner: -
--

CREATE TABLE claims.provider_accuracy (
    accuracy_id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    provider_id uuid NOT NULL,
    organization_id uuid NOT NULL,
    period_start_date date NOT NULL,
    period_end_date date NOT NULL,
    period_type character varying(20) NOT NULL,
    encounters_billed integer DEFAULT 0,
    service_lines_billed integer DEFAULT 0,
    encounters_audited integer DEFAULT 0,
    encounters_with_errors integer DEFAULT 0,
    service_lines_audited integer DEFAULT 0,
    service_lines_with_errors integer DEFAULT 0,
    encounter_accuracy_rate numeric(5,2),
    service_line_accuracy_rate numeric(5,2),
    overall_accuracy_rate numeric(5,2),
    high_severity_errors integer DEFAULT 0,
    medium_severity_errors integer DEFAULT 0,
    low_severity_errors integer DEFAULT 0,
    documentation_issues integer DEFAULT 0,
    em_level_issues integer DEFAULT 0,
    modifier_issues integer DEFAULT 0,
    diagnosis_issues integer DEFAULT 0,
    total_financial_impact numeric(18,2),
    overpayment_total numeric(18,2),
    underpayment_total numeric(18,2),
    calculated_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP
);


--
-- Name: TABLE provider_accuracy; Type: COMMENT; Schema: claims; Owner: -
--

COMMENT ON TABLE claims.provider_accuracy IS 'Provider documentation and coding accuracy metrics';


--
-- Name: region; Type: TABLE; Schema: claims; Owner: -
--

CREATE TABLE claims.region (
    region_id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    organization_id uuid NOT NULL,
    region_code character varying(50) NOT NULL,
    region_name character varying(255) NOT NULL,
    description text,
    is_active boolean DEFAULT true,
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    created_by character varying(100),
    updated_by character varying(100)
);


--
-- Name: TABLE region; Type: COMMENT; Schema: claims; Owner: -
--

COMMENT ON TABLE claims.region IS 'Regional divisions within organizations (optional level)';


--
-- Name: reviewer; Type: TABLE; Schema: claims; Owner: -
--

CREATE TABLE claims.reviewer (
    reviewer_id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    reviewer_code character varying(50) NOT NULL,
    last_name character varying(255) NOT NULL,
    first_name character varying(255) NOT NULL,
    middle_name character varying(255),
    reviewer_group character varying(100),
    certifications text[],
    organization_id uuid,
    email public.citext,
    is_active boolean DEFAULT true,
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    created_by character varying(100),
    updated_by character varying(100)
);


--
-- Name: TABLE reviewer; Type: COMMENT; Schema: claims; Owner: -
--

COMMENT ON TABLE claims.reviewer IS 'Audit reviewers who perform retrospective reviews';


--
-- Name: rvu_reference; Type: TABLE; Schema: claims; Owner: -
--

CREATE TABLE claims.rvu_reference (
    rvu_id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    hcpcs_code character varying(5) NOT NULL,
    modifier character varying(2),
    effective_year integer NOT NULL,
    effective_date date NOT NULL,
    termination_date date,
    work_rvu numeric(10,3) DEFAULT 0.000,
    pe_rvu_nonfacility numeric(10,3) DEFAULT 0.000,
    pe_rvu_facility numeric(10,3) DEFAULT 0.000,
    mp_rvu numeric(10,3) DEFAULT 0.000,
    total_rvu_nonfacility numeric(10,3) DEFAULT 0.000,
    total_rvu_facility numeric(10,3) DEFAULT 0.000,
    status_code character varying(3),
    multiple_surgery_indicator character(1),
    bilateral_surgery_indicator character(1),
    assistant_surgery_indicator character(1),
    co_surgery_indicator character(1),
    team_surgery_indicator character(1),
    global_surgery_indicator character varying(3),
    pre_op_percentage numeric(5,2),
    intra_op_percentage numeric(5,2),
    post_op_percentage numeric(5,2),
    pc_tc_indicator character(1),
    short_description character varying(255),
    long_description text,
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP
);


--
-- Name: TABLE rvu_reference; Type: COMMENT; Schema: claims; Owner: -
--

COMMENT ON TABLE claims.rvu_reference IS 'RVU reference data from CMS Physician Fee Schedule';


--
-- Name: service_line_adjustment; Type: TABLE; Schema: claims; Owner: -
--

CREATE TABLE claims.service_line_adjustment (
    adjustment_id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    service_line_id uuid NOT NULL,
    claim_adjustment_group_code character varying(2) NOT NULL,
    adjustment_reason_code character varying(5) NOT NULL,
    adjustment_amount numeric(18,2) NOT NULL,
    adjustment_quantity numeric(15,3),
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP
);


--
-- Name: TABLE service_line_adjustment; Type: COMMENT; Schema: claims; Owner: -
--

COMMENT ON TABLE claims.service_line_adjustment IS 'Line-level claim adjustments from other payers';


--
-- Name: service_line_diagnosis_pointer; Type: TABLE; Schema: claims; Owner: -
--

CREATE TABLE claims.service_line_diagnosis_pointer (
    pointer_id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    service_line_id uuid NOT NULL,
    diagnosis_id uuid NOT NULL,
    pointer_sequence smallint NOT NULL,
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP
);


--
-- Name: TABLE service_line_diagnosis_pointer; Type: COMMENT; Schema: claims; Owner: -
--

COMMENT ON TABLE claims.service_line_diagnosis_pointer IS 'Explicit mapping between service lines and diagnoses';


--
-- Name: service_line_evaluation; Type: TABLE; Schema: claims; Owner: -
--

CREATE TABLE claims.service_line_evaluation (
    evaluation_id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    audit_encounter_id uuid NOT NULL,
    service_line_id uuid NOT NULL,
    reviewer_id uuid NOT NULL,
    original_procedure_code character varying(48),
    original_modifier_1 character varying(2),
    original_modifier_2 character varying(2),
    original_modifier_3 character varying(2),
    original_modifier_4 character varying(2),
    original_units numeric(15,1),
    original_charge_amount numeric(18,2),
    corrected_procedure_code character varying(48),
    corrected_modifier_1 character varying(2),
    corrected_modifier_2 character varying(2),
    corrected_modifier_3 character varying(2),
    corrected_modifier_4 character varying(2),
    corrected_units numeric(15,1),
    corrected_charge_amount numeric(18,2),
    evaluation_result character varying(50) NOT NULL,
    has_error boolean DEFAULT false,
    issue_id uuid,
    issue_description text,
    issue_severity character varying(20),
    reimbursement_impact numeric(18,2),
    impact_type character varying(20),
    documentation_sufficient boolean,
    documentation_notes text,
    confidence_level character varying(20),
    requires_second_review boolean DEFAULT false,
    evaluated_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_evaluation_result CHECK (((evaluation_result)::text = ANY ((ARRAY['CORRECT'::character varying, 'OVERCODED'::character varying, 'UNDERCODED'::character varying, 'INCORRECT'::character varying, 'UNSUPPORTED'::character varying, 'BUNDLED'::character varying])::text[])))
);


--
-- Name: TABLE service_line_evaluation; Type: COMMENT; Schema: claims; Owner: -
--

COMMENT ON TABLE claims.service_line_evaluation IS 'Detailed audit findings for individual service lines';


--
-- Name: service_line_flag; Type: TABLE; Schema: claims; Owner: -
--

CREATE TABLE claims.service_line_flag (
    flag_id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    service_line_id uuid NOT NULL,
    issue_id uuid NOT NULL,
    flag_type character varying(20) DEFAULT 'POST_BILL'::character varying,
    severity character varying(20),
    flag_reason text,
    flagged_element character varying(255),
    proposed_code character varying(50),
    proposed_modifier character varying(10),
    proposed_quantity numeric(15,3),
    flag_status character varying(20) DEFAULT 'OPEN'::character varying,
    resolution_note text,
    resolved_at timestamp with time zone,
    resolved_by character varying(100),
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    created_by character varying(100) DEFAULT 'SYSTEM'::character varying
);


--
-- Name: TABLE service_line_flag; Type: COMMENT; Schema: claims; Owner: -
--

COMMENT ON TABLE claims.service_line_flag IS 'Flags assigned to service lines by the rules engine';


--
-- Name: service_line_reimbursement; Type: TABLE; Schema: claims; Owner: -
--

CREATE TABLE claims.service_line_reimbursement (
    reimbursement_id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    service_line_id uuid NOT NULL,
    rvu_id uuid,
    conversion_factor_id uuid,
    gpci_id uuid,
    work_rvu numeric(10,3),
    pe_rvu numeric(10,3),
    mp_rvu numeric(10,3),
    total_rvu numeric(10,3),
    work_gpci numeric(6,3) DEFAULT 1.000,
    pe_gpci numeric(6,3) DEFAULT 1.000,
    mp_gpci numeric(6,3) DEFAULT 1.000,
    conversion_factor numeric(10,4),
    base_medicare_payment numeric(18,2),
    modifier_adjustment_percentage numeric(5,2) DEFAULT 100.00,
    adjusted_medicare_payment numeric(18,2),
    unit_count numeric(15,1),
    total_medicare_payment numeric(18,2),
    billed_amount numeric(18,2),
    payment_to_charge_ratio numeric(8,4),
    calculation_method character varying(50),
    calculation_notes text,
    is_estimated boolean DEFAULT true,
    calculated_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP
);


--
-- Name: TABLE service_line_reimbursement; Type: COMMENT; Schema: claims; Owner: -
--

COMMENT ON TABLE claims.service_line_reimbursement IS 'Estimated Medicare reimbursement for service lines based on RVU';


--
-- Name: v_audit_assignment_status; Type: VIEW; Schema: claims; Owner: -
--

CREATE VIEW claims.v_audit_assignment_status AS
 SELECT aa.audit_id,
    aa.audit_name,
    aa.audit_type,
    aa.organization_id,
    aa.facility_id,
    aa.reviewer_id,
    r.last_name AS reviewer_last_name,
    r.first_name AS reviewer_first_name,
    aa.audit_status,
    aa.due_date,
    aa.sample_size,
    aa.encounters_reviewed,
    aa.completion_percentage,
    round(((100.0 * (aa.encounters_reviewed)::numeric) / (NULLIF(aa.sample_size, 0))::numeric), 2) AS actual_completion_percentage,
    aa.encounters_with_errors,
    aa.total_flags_found,
    aa.error_rate,
    aa.total_billed_amount,
    aa.total_overpayment_amount,
    aa.total_underpayment_amount,
    aa.net_financial_impact,
    aa.assigned_at,
    aa.completed_at,
    (EXTRACT(epoch FROM (COALESCE(aa.completed_at, CURRENT_TIMESTAMP) - aa.assigned_at)) / (86400)::numeric) AS days_in_progress,
    (aa.due_date - CURRENT_DATE) AS days_until_due
   FROM (claims.audit_assignment aa
     LEFT JOIN claims.reviewer r ON ((aa.reviewer_id = r.reviewer_id)));


--
-- Name: VIEW v_audit_assignment_status; Type: COMMENT; Schema: claims; Owner: -
--

COMMENT ON VIEW claims.v_audit_assignment_status IS 'Audit assignment status and progress tracking';


--
-- Name: v_claim_status_summary; Type: VIEW; Schema: claims; Owner: -
--

CREATE VIEW claims.v_claim_status_summary AS
 SELECT organization_id,
    facility_id,
    claim_status,
    count(DISTINCT encounter_id) AS encounter_count,
    sum(total_claim_charge_amount) AS total_amount,
    avg(total_claim_charge_amount) AS avg_amount,
    min(date_of_service_from) AS earliest_dos,
    max(date_of_service_from) AS latest_dos
   FROM claims.encounter e
  WHERE ((is_active = true) AND (soft_deleted = false))
  GROUP BY organization_id, facility_id, claim_status;


--
-- Name: VIEW v_claim_status_summary; Type: COMMENT; Schema: claims; Owner: -
--

COMMENT ON VIEW claims.v_claim_status_summary IS 'Summary of claims by status';


--
-- Name: v_coder_performance; Type: VIEW; Schema: claims; Owner: -
--

CREATE VIEW claims.v_coder_performance AS
 SELECT c.coder_id,
    c.coder_code,
    c.last_name,
    c.first_name,
    c.organization_id,
    count(DISTINCT e.encounter_id) AS encounters_coded,
    count(DISTINCT sl.service_line_id) AS service_lines_coded,
    sum(e.total_claim_charge_amount) AS total_amount_coded,
    sum(slr.total_rvu) AS total_rvus,
    round(avg(slr.total_rvu), 3) AS avg_rvu_per_service,
    count(DISTINCT ae.audit_encounter_id) AS encounters_audited,
    count(DISTINCT
        CASE
            WHEN ae.has_errors THEN ae.audit_encounter_id
            ELSE NULL::uuid
        END) AS encounters_with_errors,
    round((100.0 * ((1)::numeric - ((count(DISTINCT
        CASE
            WHEN ae.has_errors THEN ae.audit_encounter_id
            ELSE NULL::uuid
        END))::numeric / (NULLIF(count(DISTINCT ae.audit_encounter_id), 0))::numeric))), 2) AS accuracy_rate,
    sum(ae.severity_high_count) AS high_severity_errors,
    sum(ae.severity_medium_count) AS medium_severity_errors,
    sum(ae.severity_low_count) AS low_severity_errors,
    sum(ae.overpayment_amount) AS total_overpayment,
    sum(ae.underpayment_amount) AS total_underpayment,
    sum(ae.net_financial_impact) AS net_financial_impact,
    count(DISTINCT ef.flag_id) AS total_flags_generated,
    count(DISTINCT
        CASE
            WHEN ((ef.severity)::text = 'HIGH'::text) THEN ef.flag_id
            ELSE NULL::uuid
        END) AS high_severity_flags,
    round(((count(DISTINCT e.encounter_id))::numeric / (NULLIF(count(DISTINCT e.coding_date), 0))::numeric), 2) AS avg_encounters_per_day
   FROM (((((claims.coder c
     LEFT JOIN claims.encounter e ON (((c.coder_id = e.coder_id) AND (e.coding_date >= (CURRENT_DATE - '30 days'::interval)))))
     LEFT JOIN claims.service_line sl ON ((e.encounter_id = sl.encounter_id)))
     LEFT JOIN claims.service_line_reimbursement slr ON ((sl.service_line_id = slr.service_line_id)))
     LEFT JOIN claims.audit_encounter ae ON ((e.encounter_id = ae.encounter_id)))
     LEFT JOIN claims.encounter_flag ef ON (((e.encounter_id = ef.encounter_id) AND ((ef.created_by)::text = 'SYSTEM'::text))))
  WHERE (c.is_active = true)
  GROUP BY c.coder_id, c.coder_code, c.last_name, c.first_name, c.organization_id;


--
-- Name: VIEW v_coder_performance; Type: COMMENT; Schema: claims; Owner: -
--

COMMENT ON VIEW claims.v_coder_performance IS 'Coder performance and accuracy metrics';


--
-- Name: v_denial_by_payer; Type: VIEW; Schema: claims; Owner: -
--

CREATE VIEW claims.v_denial_by_payer AS
 SELECT organization_id,
    facility_id,
    payer_id,
    payer_name,
    date_trunc('month'::text, (denial_date)::timestamp with time zone) AS month,
    count(DISTINCT denial_id) AS denial_count,
    count(DISTINCT encounter_id) AS affected_encounters,
    sum(denied_amount) AS total_denied_amount,
    sum(billed_amount) AS total_billed_amount,
    sum(paid_amount) AS total_paid_amount,
    round(((100.0 * sum(denied_amount)) / NULLIF(sum(billed_amount), (0)::numeric)), 2) AS denial_rate_percentage,
    count(DISTINCT
        CASE
            WHEN ((root_cause_category)::text = 'CODING'::text) THEN denial_id
            ELSE NULL::uuid
        END) AS coding_denials,
    count(DISTINCT
        CASE
            WHEN ((root_cause_category)::text = 'DOCUMENTATION'::text) THEN denial_id
            ELSE NULL::uuid
        END) AS documentation_denials,
    count(DISTINCT
        CASE
            WHEN ((root_cause_category)::text = 'AUTHORIZATION'::text) THEN denial_id
            ELSE NULL::uuid
        END) AS authorization_denials,
    count(DISTINCT
        CASE
            WHEN ((root_cause_category)::text = 'ELIGIBILITY'::text) THEN denial_id
            ELSE NULL::uuid
        END) AS eligibility_denials,
    count(DISTINCT
        CASE
            WHEN ((root_cause_category)::text = 'MEDICAL_NECESSITY'::text) THEN denial_id
            ELSE NULL::uuid
        END) AS medical_necessity_denials,
    count(DISTINCT
        CASE
            WHEN is_preventable THEN denial_id
            ELSE NULL::uuid
        END) AS preventable_denials,
    sum(
        CASE
            WHEN is_preventable THEN denied_amount
            ELSE (0)::numeric
        END) AS preventable_denied_amount,
    round(((100.0 * (count(DISTINCT
        CASE
            WHEN is_preventable THEN denial_id
            ELSE NULL::uuid
        END))::numeric) / (NULLIF(count(DISTINCT denial_id), 0))::numeric), 2) AS preventable_percentage,
    count(DISTINCT
        CASE
            WHEN appeal_filed THEN denial_id
            ELSE NULL::uuid
        END) AS appeals_filed,
    count(DISTINCT
        CASE
            WHEN ((resolution_status)::text = 'OVERTURNED'::text) THEN denial_id
            ELSE NULL::uuid
        END) AS appeals_won,
    round(((100.0 * (count(DISTINCT
        CASE
            WHEN ((resolution_status)::text = 'OVERTURNED'::text) THEN denial_id
            ELSE NULL::uuid
        END))::numeric) / (NULLIF(count(DISTINCT
        CASE
            WHEN appeal_filed THEN denial_id
            ELSE NULL::uuid
        END), 0))::numeric), 2) AS appeal_success_rate
   FROM claims.denial_event de
  GROUP BY organization_id, facility_id, payer_id, payer_name, (date_trunc('month'::text, (denial_date)::timestamp with time zone));


--
-- Name: VIEW v_denial_by_payer; Type: COMMENT; Schema: claims; Owner: -
--

COMMENT ON VIEW claims.v_denial_by_payer IS 'Denial statistics by payer';


--
-- Name: v_denial_by_reason; Type: VIEW; Schema: claims; Owner: -
--

CREATE VIEW claims.v_denial_by_reason AS
 SELECT de.organization_id,
    de.facility_id,
    de.claim_adjustment_reason_code AS carc,
    drc.short_description AS carc_description,
    drc.category AS denial_category,
    date_trunc('month'::text, (de.denial_date)::timestamp with time zone) AS month,
    count(DISTINCT de.denial_id) AS denial_count,
    count(DISTINCT de.encounter_id) AS affected_encounters,
    sum(de.denied_amount) AS total_denied_amount,
    avg(de.denied_amount) AS avg_denied_amount,
    count(DISTINCT
        CASE
            WHEN de.is_preventable THEN de.denial_id
            ELSE NULL::uuid
        END) AS preventable_count,
    sum(
        CASE
            WHEN de.is_preventable THEN de.denied_amount
            ELSE (0)::numeric
        END) AS preventable_amount,
    count(DISTINCT
        CASE
            WHEN ((de.resolution_status)::text = 'OVERTURNED'::text) THEN de.denial_id
            ELSE NULL::uuid
        END) AS overturned_count,
    count(DISTINCT
        CASE
            WHEN ((de.resolution_status)::text = 'WRITTEN_OFF'::text) THEN de.denial_id
            ELSE NULL::uuid
        END) AS written_off_count
   FROM (claims.denial_event de
     LEFT JOIN claims.denial_reason_code drc ON ((((de.claim_adjustment_reason_code)::text = (drc.reason_code)::text) AND ((drc.code_type)::text = 'CARC'::text))))
  GROUP BY de.organization_id, de.facility_id, de.claim_adjustment_reason_code, drc.short_description, drc.category, (date_trunc('month'::text, (de.denial_date)::timestamp with time zone));


--
-- Name: VIEW v_denial_by_reason; Type: COMMENT; Schema: claims; Owner: -
--

COMMENT ON VIEW claims.v_denial_by_reason IS 'Denial statistics by CARC reason code';


--
-- Name: v_fifo_violations; Type: VIEW; Schema: claims; Owner: -
--

CREATE VIEW claims.v_fifo_violations AS
 SELECT e1.encounter_id AS earlier_encounter_id,
    e1.patient_control_number AS earlier_pcn,
    e1.date_of_service_from AS earlier_service_date,
    e1.import_date AS earlier_import_date,
    e2.encounter_id AS later_encounter_id,
    e2.patient_control_number AS later_pcn,
    e2.date_of_service_from AS later_service_date,
    e2.import_date AS later_import_date,
    e1.facility_id,
    f.facility_code,
    f.facility_name,
    EXTRACT(epoch FROM (e1.import_date - e2.import_date)) AS import_gap_seconds,
    (e2.date_of_service_from - e1.date_of_service_from) AS service_date_gap_days
   FROM ((claims.encounter e1
     JOIN claims.encounter e2 ON ((e1.facility_id = e2.facility_id)))
     JOIN claims.facility f ON ((e1.facility_id = f.facility_id)))
  WHERE ((e1.date_of_service_from > e2.date_of_service_from) AND (e1.import_date < e2.import_date) AND (e1.is_active = true) AND (e2.is_active = true) AND (e1.encounter_id <> e2.encounter_id) AND (e1.import_date > (CURRENT_TIMESTAMP - '30 days'::interval)))
  ORDER BY e1.facility_id, e1.import_date DESC;


--
-- Name: VIEW v_fifo_violations; Type: COMMENT; Schema: claims; Owner: -
--

COMMENT ON VIEW claims.v_fifo_violations IS 'Detects cases where claims were processed out of service date order (FIFO violations)';


--
-- Name: v_flags_by_category; Type: VIEW; Schema: claims; Owner: -
--

CREATE VIEW claims.v_flags_by_category AS
 SELECT e.organization_id,
    e.facility_id,
    fc.category_code,
    fc.category_name,
    fi.issue_code,
    fi.issue_description,
    fi.severity,
    date_trunc('month'::text, ef.created_at) AS month,
    count(DISTINCT ef.flag_id) AS flag_count,
    count(DISTINCT ef.encounter_id) AS affected_encounters,
    count(DISTINCT
        CASE
            WHEN ((ef.flag_status)::text = 'OPEN'::text) THEN ef.flag_id
            ELSE NULL::uuid
        END) AS open_flags,
    count(DISTINCT
        CASE
            WHEN ((ef.flag_status)::text = 'RESOLVED'::text) THEN ef.flag_id
            ELSE NULL::uuid
        END) AS resolved_flags,
    count(DISTINCT
        CASE
            WHEN ((ef.flag_status)::text = 'ACCEPTED'::text) THEN ef.flag_id
            ELSE NULL::uuid
        END) AS accepted_flags,
    count(DISTINCT
        CASE
            WHEN ((ef.flag_status)::text = 'REJECTED'::text) THEN ef.flag_id
            ELSE NULL::uuid
        END) AS rejected_flags,
    round(((100.0 * (count(DISTINCT
        CASE
            WHEN ((ef.flag_status)::text = ANY ((ARRAY['RESOLVED'::character varying, 'ACCEPTED'::character varying])::text[])) THEN ef.flag_id
            ELSE NULL::uuid
        END))::numeric) / (NULLIF(count(DISTINCT ef.flag_id), 0))::numeric), 2) AS resolution_rate_percentage,
    avg((EXTRACT(epoch FROM (ef.resolved_at - ef.created_at)) / (3600)::numeric)) AS avg_resolution_hours
   FROM (((claims.encounter_flag ef
     JOIN claims.flag_issue fi ON ((ef.issue_id = fi.issue_id)))
     JOIN claims.flag_category fc ON ((fi.category_id = fc.category_id)))
     JOIN claims.encounter e ON ((ef.encounter_id = e.encounter_id)))
  WHERE (e.is_active = true)
  GROUP BY e.organization_id, e.facility_id, fc.category_code, fc.category_name, fi.issue_code, fi.issue_description, fi.severity, (date_trunc('month'::text, ef.created_at));


--
-- Name: VIEW v_flags_by_category; Type: COMMENT; Schema: claims; Owner: -
--

COMMENT ON VIEW claims.v_flags_by_category IS 'Flag statistics by category and issue type';


--
-- Name: v_management_overview; Type: VIEW; Schema: claims; Owner: -
--

CREATE VIEW claims.v_management_overview AS
 SELECT e.organization_id,
    e.facility_id,
    date_trunc('month'::text, (e.date_of_service_from)::timestamp with time zone) AS month,
    count(DISTINCT e.encounter_id) AS total_encounters,
    count(DISTINCT sl.service_line_id) AS total_service_lines,
    count(DISTINCT e.billing_provider_id) AS active_providers,
    count(DISTINCT e.coder_id) AS active_coders,
    sum(e.total_claim_charge_amount) AS total_billed_amount,
    avg(e.total_claim_charge_amount) AS avg_claim_amount,
    sum(sl.line_item_charge_amount) AS total_line_charges,
    sum(slr.total_rvu) AS total_rvus,
    sum(slr.total_medicare_payment) AS estimated_medicare_payment,
    count(DISTINCT
        CASE
            WHEN (ef.flag_id IS NOT NULL) THEN e.encounter_id
            ELSE NULL::uuid
        END) AS encounters_with_flags,
    count(DISTINCT ef.flag_id) AS total_flags,
    count(DISTINCT
        CASE
            WHEN ((ef.severity)::text = 'HIGH'::text) THEN ef.flag_id
            ELSE NULL::uuid
        END) AS high_severity_flags,
    count(DISTINCT
        CASE
            WHEN ((ef.severity)::text = 'MEDIUM'::text) THEN ef.flag_id
            ELSE NULL::uuid
        END) AS medium_severity_flags,
    count(DISTINCT
        CASE
            WHEN ((ef.severity)::text = 'LOW'::text) THEN ef.flag_id
            ELSE NULL::uuid
        END) AS low_severity_flags,
    round(((100.0 * (count(DISTINCT
        CASE
            WHEN (ef.flag_id IS NOT NULL) THEN e.encounter_id
            ELSE NULL::uuid
        END))::numeric) / (NULLIF(count(DISTINCT e.encounter_id), 0))::numeric), 2) AS flag_rate_percentage,
    count(DISTINCT de.denial_id) AS total_denials,
    sum(de.denied_amount) AS total_denied_amount,
    round(((100.0 * (count(DISTINCT de.denial_id))::numeric) / (NULLIF(count(DISTINCT e.encounter_id), 0))::numeric), 2) AS denial_rate_percentage
   FROM ((((claims.encounter e
     LEFT JOIN claims.service_line sl ON ((e.encounter_id = sl.encounter_id)))
     LEFT JOIN claims.service_line_reimbursement slr ON ((sl.service_line_id = slr.service_line_id)))
     LEFT JOIN claims.encounter_flag ef ON (((e.encounter_id = ef.encounter_id) AND ((ef.flag_status)::text = 'OPEN'::text))))
     LEFT JOIN claims.denial_event de ON ((e.encounter_id = de.encounter_id)))
  WHERE ((e.is_active = true) AND (e.soft_deleted = false))
  GROUP BY e.organization_id, e.facility_id, (date_trunc('month'::text, (e.date_of_service_from)::timestamp with time zone));


--
-- Name: VIEW v_management_overview; Type: COMMENT; Schema: claims; Owner: -
--

COMMENT ON VIEW claims.v_management_overview IS 'High-level metrics for management dashboard';


--
-- Name: v_procedure_volume; Type: VIEW; Schema: claims; Owner: -
--

CREATE VIEW claims.v_procedure_volume AS
 SELECT e.organization_id,
    e.facility_id,
    sl.procedure_code,
    sl.procedure_description,
    date_trunc('month'::text, (sl.service_date_from)::timestamp with time zone) AS month,
    count(DISTINCT sl.service_line_id) AS procedure_count,
    count(DISTINCT e.encounter_id) AS encounter_count,
    count(DISTINCT e.rendering_provider_id) AS provider_count,
    sum(sl.service_unit_count) AS total_units,
    sum(sl.line_item_charge_amount) AS total_charges,
    avg(sl.line_item_charge_amount) AS avg_charge,
    sum(slr.total_rvu) AS total_rvus,
    avg(slr.total_rvu) AS avg_rvu,
    sum(slr.total_medicare_payment) AS estimated_payment,
    count(DISTINCT
        CASE
            WHEN (slf.flag_id IS NOT NULL) THEN sl.service_line_id
            ELSE NULL::uuid
        END) AS flagged_lines,
    round(((100.0 * (count(DISTINCT
        CASE
            WHEN (slf.flag_id IS NOT NULL) THEN sl.service_line_id
            ELSE NULL::uuid
        END))::numeric) / (NULLIF(count(DISTINCT sl.service_line_id), 0))::numeric), 2) AS flag_rate_percentage
   FROM (((claims.service_line sl
     JOIN claims.encounter e ON ((sl.encounter_id = e.encounter_id)))
     LEFT JOIN claims.service_line_reimbursement slr ON ((sl.service_line_id = slr.service_line_id)))
     LEFT JOIN claims.service_line_flag slf ON (((sl.service_line_id = slf.service_line_id) AND ((slf.flag_status)::text = 'OPEN'::text))))
  WHERE ((e.is_active = true) AND (e.soft_deleted = false))
  GROUP BY e.organization_id, e.facility_id, sl.procedure_code, sl.procedure_description, (date_trunc('month'::text, (sl.service_date_from)::timestamp with time zone));


--
-- Name: VIEW v_procedure_volume; Type: COMMENT; Schema: claims; Owner: -
--

COMMENT ON VIEW claims.v_procedure_volume IS 'Procedure volume and performance analysis';


--
-- Name: v_provider_documentation_accuracy; Type: VIEW; Schema: claims; Owner: -
--

CREATE VIEW claims.v_provider_documentation_accuracy AS
 SELECT p.provider_id,
    p.npi,
    p.last_name,
    p.first_name,
    p.specialty,
    p.organization_id,
    count(DISTINCT e.encounter_id) AS encounters_billed,
    count(DISTINCT sl.service_line_id) AS service_lines_billed,
    count(DISTINCT ae.audit_encounter_id) AS encounters_audited,
    count(DISTINCT
        CASE
            WHEN ae.has_errors THEN ae.audit_encounter_id
            ELSE NULL::uuid
        END) AS encounters_with_errors,
    round((100.0 * ((1)::numeric - ((count(DISTINCT
        CASE
            WHEN ae.has_errors THEN ae.audit_encounter_id
            ELSE NULL::uuid
        END))::numeric / (NULLIF(count(DISTINCT ae.audit_encounter_id), 0))::numeric))), 2) AS documentation_accuracy_rate,
    sum(ae.severity_high_count) AS high_severity_errors,
    sum(ae.severity_medium_count) AS medium_severity_errors,
    count(DISTINCT
        CASE
            WHEN ((sle.evaluation_result)::text = 'OVERCODED'::text) THEN sle.evaluation_id
            ELSE NULL::uuid
        END) AS overcoding_instances,
    count(DISTINCT
        CASE
            WHEN ((sle.evaluation_result)::text = 'UNDERCODED'::text) THEN sle.evaluation_id
            ELSE NULL::uuid
        END) AS undercoding_instances,
    count(DISTINCT
        CASE
            WHEN ((sle.evaluation_result)::text = 'UNSUPPORTED'::text) THEN sle.evaluation_id
            ELSE NULL::uuid
        END) AS unsupported_instances,
    sum(ae.overpayment_amount) AS total_overpayment,
    sum(ae.underpayment_amount) AS total_underpayment,
    sum(ae.net_financial_impact) AS net_financial_impact
   FROM ((((claims.provider p
     LEFT JOIN claims.encounter e ON (((p.provider_id = e.rendering_provider_id) AND (e.date_of_service_from >= (CURRENT_DATE - '90 days'::interval)))))
     LEFT JOIN claims.service_line sl ON ((e.encounter_id = sl.encounter_id)))
     LEFT JOIN claims.audit_encounter ae ON ((e.encounter_id = ae.encounter_id)))
     LEFT JOIN claims.service_line_evaluation sle ON ((sl.service_line_id = sle.service_line_id)))
  WHERE (p.is_active = true)
  GROUP BY p.provider_id, p.npi, p.last_name, p.first_name, p.specialty, p.organization_id;


--
-- Name: VIEW v_provider_documentation_accuracy; Type: COMMENT; Schema: claims; Owner: -
--

COMMENT ON VIEW claims.v_provider_documentation_accuracy IS 'Provider documentation and coding accuracy metrics';


--
-- Name: v_provider_productivity; Type: VIEW; Schema: claims; Owner: -
--

CREATE VIEW claims.v_provider_productivity AS
 SELECT p.provider_id,
    p.npi,
    p.last_name,
    p.first_name,
    p.specialty,
    e.organization_id,
    e.facility_id,
    date_trunc('month'::text, (e.date_of_service_from)::timestamp with time zone) AS month,
    count(DISTINCT e.encounter_id) AS encounter_count,
    count(DISTINCT sl.service_line_id) AS service_line_count,
    avg(count(DISTINCT e.encounter_id)) OVER (PARTITION BY p.provider_id, (date_trunc('month'::text, (e.date_of_service_from)::timestamp with time zone))) AS avg_daily_encounters,
    sum(e.total_claim_charge_amount) AS total_charges,
    avg(e.total_claim_charge_amount) AS avg_charge_per_encounter,
    sum(sl.line_item_charge_amount) AS total_line_charges,
    sum(slr.total_rvu) AS total_work_rvus,
    avg(slr.total_rvu) AS avg_rvu_per_service,
    sum(slr.total_medicare_payment) AS estimated_collections,
    count(DISTINCT
        CASE
            WHEN ((sl.procedure_code)::text ~~ '99___'::text) THEN sl.service_line_id
            ELSE NULL::uuid
        END) AS em_visit_count,
    count(DISTINCT
        CASE
            WHEN ((sl.procedure_code)::text !~~ '99___'::text) THEN sl.service_line_id
            ELSE NULL::uuid
        END) AS non_em_procedure_count
   FROM (((claims.provider p
     LEFT JOIN claims.encounter e ON ((p.provider_id = e.rendering_provider_id)))
     LEFT JOIN claims.service_line sl ON ((e.encounter_id = sl.encounter_id)))
     LEFT JOIN claims.service_line_reimbursement slr ON ((sl.service_line_id = slr.service_line_id)))
  WHERE ((p.is_active = true) AND (e.is_active = true))
  GROUP BY p.provider_id, p.npi, p.last_name, p.first_name, p.specialty, e.organization_id, e.facility_id, (date_trunc('month'::text, (e.date_of_service_from)::timestamp with time zone));


--
-- Name: VIEW v_provider_productivity; Type: COMMENT; Schema: claims; Owner: -
--

COMMENT ON VIEW claims.v_provider_productivity IS 'Provider productivity and RVU analysis';


--
-- Name: v_reimbursement_analysis; Type: VIEW; Schema: claims; Owner: -
--

CREATE VIEW claims.v_reimbursement_analysis AS
 SELECT e.organization_id,
    e.facility_id,
    date_trunc('month'::text, (e.date_of_service_from)::timestamp with time zone) AS month,
    count(DISTINCT e.encounter_id) AS encounter_count,
    count(DISTINCT sl.service_line_id) AS service_line_count,
    sum(e.total_claim_charge_amount) AS total_billed,
    sum(sl.line_item_charge_amount) AS total_line_charges,
    sum(slr.total_rvu) AS total_rvus,
    sum(slr.total_medicare_payment) AS estimated_medicare_payment,
    round((sum(slr.total_medicare_payment) / NULLIF(sum(sl.line_item_charge_amount), (0)::numeric)), 4) AS payment_to_charge_ratio,
    sum(COALESCE(de.denied_amount, (0)::numeric)) AS total_denied,
    round(((100.0 * sum(COALESCE(de.denied_amount, (0)::numeric))) / NULLIF(sum(e.total_claim_charge_amount), (0)::numeric)), 2) AS denial_percentage,
    (sum(slr.total_medicare_payment) - sum(COALESCE(de.denied_amount, (0)::numeric))) AS net_expected_payment
   FROM (((claims.encounter e
     LEFT JOIN claims.service_line sl ON ((e.encounter_id = sl.encounter_id)))
     LEFT JOIN claims.service_line_reimbursement slr ON ((sl.service_line_id = slr.service_line_id)))
     LEFT JOIN claims.denial_event de ON ((e.encounter_id = de.encounter_id)))
  WHERE ((e.is_active = true) AND (e.soft_deleted = false))
  GROUP BY e.organization_id, e.facility_id, (date_trunc('month'::text, (e.date_of_service_from)::timestamp with time zone));


--
-- Name: VIEW v_reimbursement_analysis; Type: COMMENT; Schema: claims; Owner: -
--

COMMENT ON VIEW claims.v_reimbursement_analysis IS 'Comprehensive reimbursement and financial analysis';


--
-- Name: v_service_line_flags_detail; Type: VIEW; Schema: claims; Owner: -
--

CREATE VIEW claims.v_service_line_flags_detail AS
 SELECT slf.flag_id,
    e.organization_id,
    e.facility_id,
    e.encounter_id,
    e.patient_control_number,
    sl.service_line_id,
    sl.procedure_code,
    sl.procedure_description,
    sl.line_item_charge_amount,
    fi.issue_code,
    fi.issue_description,
    slf.severity,
    slf.flag_reason,
    slf.flagged_element,
    slf.proposed_code,
    slf.proposed_modifier,
    slf.proposed_quantity,
    slf.flag_status,
    slf.created_at AS flagged_at,
    slf.resolved_at,
    slf.resolution_note,
    e.coder_id,
    e.rendering_provider_id
   FROM (((claims.service_line_flag slf
     JOIN claims.service_line sl ON ((slf.service_line_id = sl.service_line_id)))
     JOIN claims.encounter e ON ((sl.encounter_id = e.encounter_id)))
     JOIN claims.flag_issue fi ON ((slf.issue_id = fi.issue_id)))
  WHERE ((e.is_active = true) AND (e.soft_deleted = false));


--
-- Name: VIEW v_service_line_flags_detail; Type: COMMENT; Schema: claims; Owner: -
--

COMMENT ON VIEW claims.v_service_line_flags_detail IS 'Detailed view of service line flags';


--
-- Name: ab_test_experiment; Type: TABLE; Schema: ml; Owner: -
--

CREATE TABLE ml.ab_test_experiment (
    experiment_id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    organization_id uuid,
    experiment_name character varying(255) NOT NULL,
    experiment_description text,
    hypothesis text,
    control_model_id uuid,
    treatment_model_id uuid,
    control_percentage numeric(5,2) DEFAULT 50.00,
    treatment_percentage numeric(5,2) DEFAULT 50.00,
    start_date date NOT NULL,
    end_date date,
    planned_duration_days integer,
    target_sample_size integer,
    current_sample_size integer DEFAULT 0,
    control_metric_value numeric(15,4),
    treatment_metric_value numeric(15,4),
    metric_difference numeric(15,4),
    statistical_significance numeric(5,4),
    is_significant boolean,
    winner character varying(20),
    winner_declared_at timestamp with time zone,
    experiment_status character varying(50) DEFAULT 'DRAFT'::character varying,
    is_active boolean DEFAULT false,
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    created_by character varying(100)
);


--
-- Name: TABLE ab_test_experiment; Type: COMMENT; Schema: ml; Owner: -
--

COMMENT ON TABLE ml.ab_test_experiment IS 'A/B testing experiments for model comparison';


--
-- Name: feature_definition; Type: TABLE; Schema: ml; Owner: -
--

CREATE TABLE ml.feature_definition (
    feature_id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    feature_name character varying(255) NOT NULL,
    feature_category character varying(100),
    feature_type character varying(50) NOT NULL,
    calculation_logic text NOT NULL,
    calculation_type character varying(50),
    source_tables text[],
    dependent_features text[],
    is_nullable boolean DEFAULT true,
    default_value character varying(255),
    allowed_values text[],
    mean_value numeric(15,4),
    std_deviation numeric(15,4),
    min_value numeric(15,4),
    max_value numeric(15,4),
    distinct_count integer,
    null_percentage numeric(5,2),
    normalization_method character varying(50),
    normalization_params jsonb,
    used_in_models text[],
    importance_scores jsonb,
    is_active boolean DEFAULT true,
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    created_by character varying(100)
);


--
-- Name: TABLE feature_definition; Type: COMMENT; Schema: ml; Owner: -
--

COMMENT ON TABLE ml.feature_definition IS 'Definitions of features for ML models';


--
-- Name: model_performance_log; Type: TABLE; Schema: ml; Owner: -
--

CREATE TABLE ml.model_performance_log (
    performance_id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    model_id uuid NOT NULL,
    measurement_date date NOT NULL,
    period_start timestamp with time zone NOT NULL,
    period_end timestamp with time zone NOT NULL,
    prediction_count integer DEFAULT 0,
    unique_encounters integer DEFAULT 0,
    accuracy numeric(5,4),
    precision_score numeric(5,4),
    recall_score numeric(5,4),
    f1_score numeric(5,4),
    auc_roc numeric(5,4),
    mean_absolute_error numeric(15,4),
    mean_squared_error numeric(15,4),
    root_mean_squared_error numeric(15,4),
    feature_drift_detected boolean DEFAULT false,
    concept_drift_detected boolean DEFAULT false,
    drift_score numeric(8,6),
    drifted_features text[],
    performance_by_facility jsonb,
    performance_by_provider_type jsonb,
    performance_by_procedure_category jsonb,
    confusion_matrix jsonb,
    average_prediction_time_ms numeric(10,3),
    peak_memory_mb numeric(15,2),
    requires_retraining boolean DEFAULT false,
    performance_alert_level character varying(20),
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP
);


--
-- Name: TABLE model_performance_log; Type: COMMENT; Schema: ml; Owner: -
--

COMMENT ON TABLE ml.model_performance_log IS 'Performance monitoring for deployed ML models';


--
-- Name: training_dataset; Type: TABLE; Schema: ml; Owner: -
--

CREATE TABLE ml.training_dataset (
    dataset_id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    organization_id uuid,
    dataset_name character varying(255) NOT NULL,
    dataset_version character varying(50) NOT NULL,
    dataset_purpose character varying(100),
    dataset_description text,
    record_count integer NOT NULL,
    feature_count integer NOT NULL,
    features_included text[],
    data_start_date date,
    data_end_date date,
    target_variable character varying(100),
    target_distribution jsonb,
    is_balanced boolean,
    class_weights jsonb,
    training_split_percentage numeric(5,2) DEFAULT 70.00,
    validation_split_percentage numeric(5,2) DEFAULT 15.00,
    test_split_percentage numeric(5,2) DEFAULT 15.00,
    missing_data_percentage numeric(5,2),
    duplicate_records integer,
    outlier_records integer,
    dataset_file_path text,
    dataset_file_size_bytes bigint,
    dataset_format character varying(20),
    dataset_status character varying(50) DEFAULT 'ACTIVE'::character varying,
    is_active boolean DEFAULT true,
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    created_by character varying(100)
);


--
-- Name: TABLE training_dataset; Type: COMMENT; Schema: ml; Owner: -
--

COMMENT ON TABLE ml.training_dataset IS 'Training datasets for ML models';


--
-- Name: application_version; Type: TABLE; Schema: staging; Owner: -
--

CREATE TABLE staging.application_version (
    version character varying(50) NOT NULL,
    installed_at timestamp without time zone DEFAULT now() NOT NULL,
    upgraded_from character varying(50),
    notes text
);


--
-- Name: TABLE application_version; Type: COMMENT; Schema: staging; Owner: -
--

COMMENT ON TABLE staging.application_version IS 'Tracks application version history for upgrade management';


--
-- Name: COLUMN application_version.version; Type: COMMENT; Schema: staging; Owner: -
--

COMMENT ON COLUMN staging.application_version.version IS 'Semantic version number (e.g., 1.0.0, 1.1.0)';


--
-- Name: COLUMN application_version.upgraded_from; Type: COMMENT; Schema: staging; Owner: -
--

COMMENT ON COLUMN staging.application_version.upgraded_from IS 'Previous version if this was an upgrade, NULL for fresh install';


--
-- Name: data_refresh_schedule; Type: TABLE; Schema: staging; Owner: -
--

CREATE TABLE staging.data_refresh_schedule (
    refresh_id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    organization_id uuid,
    target_type character varying(50) NOT NULL,
    target_name character varying(255) NOT NULL,
    schema_name character varying(50) NOT NULL,
    refresh_type character varying(50) NOT NULL,
    refresh_method character varying(50),
    schedule_type character varying(50) NOT NULL,
    cron_expression character varying(100),
    interval_minutes integer,
    depends_on text[],
    execution_order integer DEFAULT 100,
    is_active boolean DEFAULT true,
    last_refreshed_at timestamp with time zone,
    last_refresh_duration_seconds integer,
    next_refresh_at timestamp with time zone,
    average_duration_seconds integer,
    total_refreshes integer DEFAULT 0,
    last_error_message text,
    consecutive_failures integer DEFAULT 0,
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP
);


--
-- Name: TABLE data_refresh_schedule; Type: COMMENT; Schema: staging; Owner: -
--

COMMENT ON TABLE staging.data_refresh_schedule IS 'Schedule for refreshing materialized views and summary tables';


--
-- Name: file_processing_queue; Type: TABLE; Schema: staging; Owner: -
--

CREATE TABLE staging.file_processing_queue (
    queue_id uuid DEFAULT gen_random_uuid() NOT NULL,
    facility_id uuid NOT NULL,
    import_batch_id uuid NOT NULL,
    file_path text NOT NULL,
    file_hash text NOT NULL,
    file_format text NOT NULL,
    organization_id uuid NOT NULL,
    queued_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP NOT NULL,
    processing_started_at timestamp with time zone,
    processing_completed_at timestamp with time zone,
    queue_status text DEFAULT 'QUEUED'::text NOT NULL,
    priority integer DEFAULT 100 NOT NULL,
    retry_count integer DEFAULT 0 NOT NULL,
    max_retries integer DEFAULT 3 NOT NULL,
    last_error text,
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP NOT NULL,
    created_by text DEFAULT 'SYSTEM'::text,
    updated_by text DEFAULT 'SYSTEM'::text,
    CONSTRAINT valid_priority CHECK (((priority >= 0) AND (priority <= 1000))),
    CONSTRAINT valid_queue_status CHECK ((queue_status = ANY (ARRAY['QUEUED'::text, 'PROCESSING'::text, 'COMPLETED'::text, 'FAILED'::text, 'RETRY'::text]))),
    CONSTRAINT valid_retry_count CHECK (((retry_count >= 0) AND (retry_count <= max_retries)))
);


--
-- Name: TABLE file_processing_queue; Type: COMMENT; Schema: staging; Owner: -
--

COMMENT ON TABLE staging.file_processing_queue IS 'FIFO queue for file processing ensuring chronological order per facility';


--
-- Name: COLUMN file_processing_queue.queued_at; Type: COMMENT; Schema: staging; Owner: -
--

COMMENT ON COLUMN staging.file_processing_queue.queued_at IS 'Timestamp when file was added to queue (used for FIFO ordering)';


--
-- Name: COLUMN file_processing_queue.queue_status; Type: COMMENT; Schema: staging; Owner: -
--

COMMENT ON COLUMN staging.file_processing_queue.queue_status IS 'Current status: QUEUED, PROCESSING, COMPLETED, FAILED, RETRY';


--
-- Name: COLUMN file_processing_queue.priority; Type: COMMENT; Schema: staging; Owner: -
--

COMMENT ON COLUMN staging.file_processing_queue.priority IS 'Priority level (0-1000, lower = higher priority, default = 100)';


--
-- Name: COLUMN file_processing_queue.retry_count; Type: COMMENT; Schema: staging; Owner: -
--

COMMENT ON COLUMN staging.file_processing_queue.retry_count IS 'Number of times this file has been retried after failure';


--
-- Name: file_upload; Type: TABLE; Schema: staging; Owner: -
--

CREATE TABLE staging.file_upload (
    upload_id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    organization_id uuid NOT NULL,
    original_filename character varying(500) NOT NULL,
    file_type character varying(50) NOT NULL,
    total_size_bytes bigint NOT NULL,
    uploaded_size_bytes bigint DEFAULT 0,
    chunk_count integer DEFAULT 0,
    chunks_received integer DEFAULT 0,
    upload_status character varying(50) DEFAULT 'IN_PROGRESS'::character varying,
    storage_path text,
    file_hash character varying(64),
    started_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    completed_at timestamp with time zone,
    expires_at timestamp with time zone,
    error_message text,
    created_by character varying(100)
);


--
-- Name: TABLE file_upload; Type: COMMENT; Schema: staging; Owner: -
--

COMMENT ON TABLE staging.file_upload IS 'Tracks multi-part file uploads';


--
-- Name: import_batch; Type: TABLE; Schema: staging; Owner: -
--

CREATE TABLE staging.import_batch (
    batch_id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    organization_id uuid NOT NULL,
    facility_id uuid,
    batch_name character varying(255),
    batch_type character varying(50) NOT NULL,
    file_format character varying(50),
    original_filename character varying(500),
    file_path text,
    file_size_bytes bigint,
    file_hash character varying(64),
    import_status character varying(50) DEFAULT 'PENDING'::character varying,
    total_records integer DEFAULT 0,
    processed_records integer DEFAULT 0,
    successful_records integer DEFAULT 0,
    failed_records integer DEFAULT 0,
    skipped_records integer DEFAULT 0,
    duplicate_records integer DEFAULT 0,
    started_at timestamp with time zone,
    completed_at timestamp with time zone,
    processing_duration_seconds numeric(15,3),
    configuration_id uuid,
    rules_applied boolean DEFAULT false,
    error_message text,
    error_details jsonb,
    validation_passed boolean,
    validation_errors jsonb,
    validation_warnings jsonb,
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    created_by character varying(100),
    CONSTRAINT chk_batch_type CHECK (((batch_type)::text = ANY ((ARRAY['EDI_837P'::character varying, 'CSV'::character varying, 'MANUAL'::character varying])::text[])))
);


--
-- Name: TABLE import_batch; Type: COMMENT; Schema: staging; Owner: -
--

COMMENT ON TABLE staging.import_batch IS 'Tracks file import batches and processing metrics';


--
-- Name: import_configuration; Type: TABLE; Schema: staging; Owner: -
--

CREATE TABLE staging.import_configuration (
    configuration_id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    organization_id uuid NOT NULL,
    configuration_name character varying(255) NOT NULL,
    configuration_type character varying(50) NOT NULL,
    description text,
    csv_delimiter character varying(5),
    csv_quote_char character varying(1),
    csv_has_header boolean DEFAULT true,
    csv_encoding character varying(20) DEFAULT 'UTF-8'::character varying,
    header_mappings jsonb,
    field_transformations jsonb,
    validation_rules jsonb,
    required_fields text[],
    default_values jsonb,
    deduplication_enabled boolean DEFAULT true,
    deduplication_fields text[],
    deduplication_window_days integer DEFAULT 90,
    auto_apply_rules boolean DEFAULT true,
    rules_to_apply text[],
    is_active boolean DEFAULT true,
    is_default boolean DEFAULT false,
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    created_by character varying(100),
    updated_by character varying(100)
);


--
-- Name: TABLE import_configuration; Type: COMMENT; Schema: staging; Owner: -
--

COMMENT ON TABLE staging.import_configuration IS 'Import configuration profiles for different file types and sources';


--
-- Name: import_error_log; Type: TABLE; Schema: staging; Owner: -
--

CREATE TABLE staging.import_error_log (
    error_id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    batch_id uuid,
    record_number integer,
    line_number integer,
    field_name character varying(255),
    error_type character varying(50) NOT NULL,
    error_severity character varying(20) DEFAULT 'ERROR'::character varying,
    error_code character varying(50),
    error_message text NOT NULL,
    error_details jsonb,
    raw_data text,
    resolution_status character varying(50) DEFAULT 'UNRESOLVED'::character varying,
    resolution_note text,
    resolved_at timestamp with time zone,
    resolved_by character varying(100),
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP
);


--
-- Name: TABLE import_error_log; Type: COMMENT; Schema: staging; Owner: -
--

COMMENT ON TABLE staging.import_error_log IS 'Detailed error log for import failures';


--
-- Name: job_execution_log; Type: TABLE; Schema: staging; Owner: -
--

CREATE TABLE staging.job_execution_log (
    execution_id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    job_id uuid NOT NULL,
    started_at timestamp with time zone NOT NULL,
    completed_at timestamp with time zone,
    duration_seconds integer,
    execution_status character varying(50) NOT NULL,
    status_message text,
    records_processed integer DEFAULT 0,
    records_successful integer DEFAULT 0,
    records_failed integer DEFAULT 0,
    peak_memory_mb numeric(15,2),
    cpu_time_seconds numeric(15,3),
    error_message text,
    error_stack_trace text,
    error_details jsonb,
    is_retry boolean DEFAULT false,
    retry_attempt integer DEFAULT 0,
    original_execution_id uuid,
    execution_output jsonb,
    log_file_path text,
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP
);


--
-- Name: TABLE job_execution_log; Type: COMMENT; Schema: staging; Owner: -
--

COMMENT ON TABLE staging.job_execution_log IS 'Execution history for scheduled jobs';


--
-- Name: processing_metrics; Type: TABLE; Schema: staging; Owner: -
--

CREATE TABLE staging.processing_metrics (
    metric_id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    batch_id uuid,
    metric_type character varying(50) NOT NULL,
    metric_name character varying(255) NOT NULL,
    started_at timestamp with time zone NOT NULL,
    completed_at timestamp with time zone,
    duration_milliseconds numeric(15,3),
    records_processed integer DEFAULT 0,
    records_per_second numeric(15,3),
    memory_used_mb numeric(15,2),
    cpu_time_ms numeric(15,3),
    success_count integer DEFAULT 0,
    error_count integer DEFAULT 0,
    warning_count integer DEFAULT 0,
    details jsonb,
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP
);


--
-- Name: TABLE processing_metrics; Type: COMMENT; Schema: staging; Owner: -
--

COMMENT ON TABLE staging.processing_metrics IS 'Performance metrics for import processing stages';


--
-- Name: report_generation_log; Type: TABLE; Schema: staging; Owner: -
--

CREATE TABLE staging.report_generation_log (
    report_log_id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    subscription_id uuid,
    organization_id uuid NOT NULL,
    report_type character varying(100) NOT NULL,
    report_name character varying(255) NOT NULL,
    report_parameters jsonb,
    generated_at timestamp with time zone NOT NULL,
    generation_duration_seconds integer,
    generation_status character varying(50) NOT NULL,
    output_format character varying(20),
    file_path text,
    file_size_bytes bigint,
    delivery_method character varying(50),
    delivered_at timestamp with time zone,
    delivery_status character varying(50),
    delivery_error text,
    recipients text[],
    record_count integer,
    page_count integer,
    error_message text,
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP
);


--
-- Name: TABLE report_generation_log; Type: COMMENT; Schema: staging; Owner: -
--

COMMENT ON TABLE staging.report_generation_log IS 'Log of report generation and delivery';


--
-- Name: report_subscription; Type: TABLE; Schema: staging; Owner: -
--

CREATE TABLE staging.report_subscription (
    subscription_id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    organization_id uuid NOT NULL,
    subscription_name character varying(255) NOT NULL,
    report_type character varying(100) NOT NULL,
    frequency character varying(50) NOT NULL,
    delivery_day_of_week integer,
    delivery_day_of_month integer,
    delivery_time time without time zone,
    report_parameters jsonb,
    recipient_emails text[] NOT NULL,
    recipient_names text[],
    output_format character varying(20) DEFAULT 'PDF'::character varying,
    include_charts boolean DEFAULT true,
    include_raw_data boolean DEFAULT false,
    delivery_method character varying(50) DEFAULT 'EMAIL'::character varying,
    delivery_config jsonb,
    is_active boolean DEFAULT true,
    last_generated_at timestamp with time zone,
    last_delivered_at timestamp with time zone,
    next_delivery_at timestamp with time zone,
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    created_by character varying(100),
    updated_by character varying(100)
);


--
-- Name: TABLE report_subscription; Type: COMMENT; Schema: staging; Owner: -
--

COMMENT ON TABLE staging.report_subscription IS 'Scheduled report subscriptions';


--
-- Name: rules_configuration; Type: TABLE; Schema: staging; Owner: -
--

CREATE TABLE staging.rules_configuration (
    rule_config_id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    organization_id uuid NOT NULL,
    facility_id uuid,
    rule_code character varying(50) NOT NULL,
    rule_name character varying(255) NOT NULL,
    rule_category character varying(50),
    rule_type character varying(50) NOT NULL,
    rule_definition text NOT NULL,
    rule_parameters jsonb,
    description text,
    severity character varying(20) DEFAULT 'MEDIUM'::character varying,
    auto_flag boolean DEFAULT true,
    flag_issue_id uuid,
    applies_to_claim_types text[],
    applies_to_specialties text[],
    applies_to_place_of_service text[],
    effective_date_from date,
    effective_date_to date,
    execution_order integer DEFAULT 100,
    timeout_seconds integer DEFAULT 5,
    times_triggered integer DEFAULT 0,
    last_triggered_at timestamp with time zone,
    is_active boolean DEFAULT true,
    is_deleted boolean DEFAULT false,
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    created_by character varying(100),
    updated_by character varying(100)
);


--
-- Name: TABLE rules_configuration; Type: COMMENT; Schema: staging; Owner: -
--

COMMENT ON TABLE staging.rules_configuration IS 'Rules engine configuration for automated flagging';


--
-- Name: scheduled_job; Type: TABLE; Schema: staging; Owner: -
--

CREATE TABLE staging.scheduled_job (
    job_id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    organization_id uuid NOT NULL,
    job_name character varying(255) NOT NULL,
    job_type character varying(50) NOT NULL,
    job_description text,
    schedule_type character varying(50) NOT NULL,
    cron_expression character varying(100),
    interval_minutes integer,
    scheduled_time time without time zone,
    job_config jsonb,
    timeout_minutes integer DEFAULT 60,
    max_retries integer DEFAULT 3,
    retry_delay_minutes integer DEFAULT 5,
    allow_concurrent boolean DEFAULT false,
    max_concurrent_executions integer DEFAULT 1,
    is_active boolean DEFAULT true,
    is_running boolean DEFAULT false,
    last_run_at timestamp with time zone,
    last_run_status character varying(50),
    last_run_duration_seconds integer,
    next_run_at timestamp with time zone,
    total_executions integer DEFAULT 0,
    successful_executions integer DEFAULT 0,
    failed_executions integer DEFAULT 0,
    last_error_message text,
    consecutive_failures integer DEFAULT 0,
    notify_on_success boolean DEFAULT false,
    notify_on_failure boolean DEFAULT true,
    notification_emails text[],
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    created_by character varying(100),
    updated_by character varying(100)
);


--
-- Name: TABLE scheduled_job; Type: COMMENT; Schema: staging; Owner: -
--

COMMENT ON TABLE staging.scheduled_job IS 'Configuration for scheduled automated jobs';


--
-- Name: schema_migrations; Type: TABLE; Schema: staging; Owner: -
--

CREATE TABLE staging.schema_migrations (
    migration_name character varying(255) NOT NULL,
    applied_at timestamp without time zone DEFAULT now() NOT NULL,
    checksum text NOT NULL,
    execution_time_ms integer,
    description text
);


--
-- Name: TABLE schema_migrations; Type: COMMENT; Schema: staging; Owner: -
--

COMMENT ON TABLE staging.schema_migrations IS 'Tracks which database migrations have been applied';


--
-- Name: COLUMN schema_migrations.migration_name; Type: COMMENT; Schema: staging; Owner: -
--

COMMENT ON COLUMN staging.schema_migrations.migration_name IS 'Name of the migration file (e.g., 001_create_schemas.sql)';


--
-- Name: COLUMN schema_migrations.checksum; Type: COMMENT; Schema: staging; Owner: -
--

COMMENT ON COLUMN staging.schema_migrations.checksum IS 'SHA-256 checksum of the migration file for integrity verification';


--
-- Name: COLUMN schema_migrations.execution_time_ms; Type: COMMENT; Schema: staging; Owner: -
--

COMMENT ON COLUMN staging.schema_migrations.execution_time_ms IS 'Time taken to execute the migration in milliseconds';


--
-- Name: upgrade_test; Type: TABLE; Schema: staging; Owner: -
--

CREATE TABLE staging.upgrade_test (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    upgrade_version character varying(20) NOT NULL,
    upgraded_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    notes text
);


--
-- Name: v_queue_health; Type: VIEW; Schema: staging; Owner: -
--

CREATE VIEW staging.v_queue_health AS
 SELECT f.facility_id,
    f.facility_code,
    f.facility_name,
    o.organization_name,
    count(*) FILTER (WHERE (q.queue_status = 'QUEUED'::text)) AS queued_count,
    count(*) FILTER (WHERE (q.queue_status = 'PROCESSING'::text)) AS processing_count,
    count(*) FILTER (WHERE (q.queue_status = 'COMPLETED'::text)) AS completed_count,
    count(*) FILTER (WHERE (q.queue_status = 'FAILED'::text)) AS failed_count,
    count(*) FILTER (WHERE (q.queue_status = 'RETRY'::text)) AS retry_count,
    min(q.queued_at) FILTER (WHERE (q.queue_status = 'QUEUED'::text)) AS oldest_queued,
    max(q.queued_at) FILTER (WHERE (q.queue_status = 'QUEUED'::text)) AS newest_queued,
    avg(EXTRACT(epoch FROM (q.processing_completed_at - q.processing_started_at))) FILTER (WHERE (q.queue_status = 'COMPLETED'::text)) AS avg_processing_seconds,
    max(EXTRACT(epoch FROM (q.processing_completed_at - q.processing_started_at))) FILTER (WHERE (q.queue_status = 'COMPLETED'::text)) AS max_processing_seconds
   FROM ((claims.facility f
     JOIN claims.organization o ON ((f.organization_id = o.organization_id)))
     LEFT JOIN staging.file_processing_queue q ON (((f.facility_id = q.facility_id) AND (q.created_at > (CURRENT_TIMESTAMP - '24:00:00'::interval)))))
  WHERE (f.is_active = true)
  GROUP BY f.facility_id, f.facility_code, f.facility_name, o.organization_name
  ORDER BY (count(*) FILTER (WHERE (q.queue_status = 'QUEUED'::text))) DESC, f.facility_code;


--
-- Name: VIEW v_queue_health; Type: COMMENT; Schema: staging; Owner: -
--

COMMENT ON VIEW staging.v_queue_health IS 'Real-time view of file processing queue health by facility (last 24 hours)';


--
-- Name: v_queue_statistics; Type: VIEW; Schema: staging; Owner: -
--

CREATE VIEW staging.v_queue_statistics AS
 SELECT date_trunc('hour'::text, q.queued_at) AS hour,
    f.facility_code,
    f.facility_name,
    count(*) AS total_files,
    count(*) FILTER (WHERE (q.queue_status = 'COMPLETED'::text)) AS completed_files,
    count(*) FILTER (WHERE (q.queue_status = 'FAILED'::text)) AS failed_files,
    avg(EXTRACT(epoch FROM (q.processing_completed_at - q.queued_at))) FILTER (WHERE (q.queue_status = 'COMPLETED'::text)) AS avg_total_seconds,
    avg(EXTRACT(epoch FROM (q.processing_started_at - q.queued_at))) FILTER (WHERE (q.queue_status = 'COMPLETED'::text)) AS avg_queue_wait_seconds,
    avg(EXTRACT(epoch FROM (q.processing_completed_at - q.processing_started_at))) FILTER (WHERE (q.queue_status = 'COMPLETED'::text)) AS avg_processing_seconds
   FROM (staging.file_processing_queue q
     JOIN claims.facility f ON ((q.facility_id = f.facility_id)))
  WHERE (q.created_at > (CURRENT_TIMESTAMP - '7 days'::interval))
  GROUP BY (date_trunc('hour'::text, q.queued_at)), f.facility_code, f.facility_name
  ORDER BY (date_trunc('hour'::text, q.queued_at)) DESC, f.facility_code;


--
-- Name: VIEW v_queue_statistics; Type: COMMENT; Schema: staging; Owner: -
--

COMMENT ON VIEW staging.v_queue_statistics IS 'Hourly statistics on file processing queue performance';


--
-- Name: audit_assignment audit_assignment_pkey; Type: CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.audit_assignment
    ADD CONSTRAINT audit_assignment_pkey PRIMARY KEY (audit_id);


--
-- Name: audit_encounter audit_encounter_audit_id_encounter_id_key; Type: CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.audit_encounter
    ADD CONSTRAINT audit_encounter_audit_id_encounter_id_key UNIQUE (audit_id, encounter_id);


--
-- Name: audit_encounter audit_encounter_pkey; Type: CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.audit_encounter
    ADD CONSTRAINT audit_encounter_pkey PRIMARY KEY (audit_encounter_id);


--
-- Name: coder_accuracy coder_accuracy_coder_id_period_start_date_period_type_key; Type: CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.coder_accuracy
    ADD CONSTRAINT coder_accuracy_coder_id_period_start_date_period_type_key UNIQUE (coder_id, period_start_date, period_type);


--
-- Name: coder_accuracy coder_accuracy_pkey; Type: CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.coder_accuracy
    ADD CONSTRAINT coder_accuracy_pkey PRIMARY KEY (accuracy_id);


--
-- Name: coder coder_coder_code_key; Type: CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.coder
    ADD CONSTRAINT coder_coder_code_key UNIQUE (coder_code);


--
-- Name: coder coder_pkey; Type: CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.coder
    ADD CONSTRAINT coder_pkey PRIMARY KEY (coder_id);


--
-- Name: conversion_factor conversion_factor_factor_year_key; Type: CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.conversion_factor
    ADD CONSTRAINT conversion_factor_factor_year_key UNIQUE (factor_year);


--
-- Name: conversion_factor conversion_factor_pkey; Type: CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.conversion_factor
    ADD CONSTRAINT conversion_factor_pkey PRIMARY KEY (conversion_factor_id);


--
-- Name: denial_appeal denial_appeal_pkey; Type: CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.denial_appeal
    ADD CONSTRAINT denial_appeal_pkey PRIMARY KEY (appeal_id);


--
-- Name: denial_event denial_event_pkey; Type: CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.denial_event
    ADD CONSTRAINT denial_event_pkey PRIMARY KEY (denial_id);


--
-- Name: denial_reason_code denial_reason_code_code_type_reason_code_key; Type: CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.denial_reason_code
    ADD CONSTRAINT denial_reason_code_code_type_reason_code_key UNIQUE (code_type, reason_code);


--
-- Name: denial_reason_code denial_reason_code_pkey; Type: CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.denial_reason_code
    ADD CONSTRAINT denial_reason_code_pkey PRIMARY KEY (reason_code_id);


--
-- Name: denial_statistics denial_statistics_organization_id_facility_id_statistic_dim_key; Type: CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.denial_statistics
    ADD CONSTRAINT denial_statistics_organization_id_facility_id_statistic_dim_key UNIQUE (organization_id, facility_id, statistic_dimension, dimension_value, period_start_date, period_type);


--
-- Name: denial_statistics denial_statistics_pkey; Type: CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.denial_statistics
    ADD CONSTRAINT denial_statistics_pkey PRIMARY KEY (statistic_id);


--
-- Name: diagnosis_evaluation diagnosis_evaluation_pkey; Type: CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.diagnosis_evaluation
    ADD CONSTRAINT diagnosis_evaluation_pkey PRIMARY KEY (diagnosis_eval_id);


--
-- Name: encounter_diagnosis encounter_diagnosis_pkey; Type: CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.encounter_diagnosis
    ADD CONSTRAINT encounter_diagnosis_pkey PRIMARY KEY (diagnosis_id);


--
-- Name: encounter_flag encounter_flag_pkey; Type: CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.encounter_flag
    ADD CONSTRAINT encounter_flag_pkey PRIMARY KEY (flag_id);


--
-- Name: encounter_note encounter_note_pkey; Type: CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.encounter_note
    ADD CONSTRAINT encounter_note_pkey PRIMARY KEY (note_id);


--
-- Name: encounter encounter_pkey; Type: CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.encounter
    ADD CONSTRAINT encounter_pkey PRIMARY KEY (encounter_id);


--
-- Name: facility facility_organization_id_facility_code_key; Type: CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.facility
    ADD CONSTRAINT facility_organization_id_facility_code_key UNIQUE (organization_id, facility_code);


--
-- Name: facility facility_pkey; Type: CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.facility
    ADD CONSTRAINT facility_pkey PRIMARY KEY (facility_id);


--
-- Name: flag_category flag_category_category_code_key; Type: CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.flag_category
    ADD CONSTRAINT flag_category_category_code_key UNIQUE (category_code);


--
-- Name: flag_category flag_category_pkey; Type: CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.flag_category
    ADD CONSTRAINT flag_category_pkey PRIMARY KEY (category_id);


--
-- Name: flag_issue flag_issue_issue_code_key; Type: CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.flag_issue
    ADD CONSTRAINT flag_issue_issue_code_key UNIQUE (issue_code);


--
-- Name: flag_issue flag_issue_pkey; Type: CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.flag_issue
    ADD CONSTRAINT flag_issue_pkey PRIMARY KEY (issue_id);


--
-- Name: gpci_reference gpci_reference_pkey; Type: CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.gpci_reference
    ADD CONSTRAINT gpci_reference_pkey PRIMARY KEY (gpci_id);


--
-- Name: modifier_adjustment modifier_adjustment_modifier_code_key; Type: CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.modifier_adjustment
    ADD CONSTRAINT modifier_adjustment_modifier_code_key UNIQUE (modifier_code);


--
-- Name: modifier_adjustment modifier_adjustment_pkey; Type: CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.modifier_adjustment
    ADD CONSTRAINT modifier_adjustment_pkey PRIMARY KEY (adjustment_id);


--
-- Name: organization organization_organization_code_key; Type: CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.organization
    ADD CONSTRAINT organization_organization_code_key UNIQUE (organization_code);


--
-- Name: organization organization_pkey; Type: CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.organization
    ADD CONSTRAINT organization_pkey PRIMARY KEY (organization_id);


--
-- Name: provider_accuracy provider_accuracy_pkey; Type: CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.provider_accuracy
    ADD CONSTRAINT provider_accuracy_pkey PRIMARY KEY (accuracy_id);


--
-- Name: provider_accuracy provider_accuracy_provider_id_period_start_date_period_type_key; Type: CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.provider_accuracy
    ADD CONSTRAINT provider_accuracy_provider_id_period_start_date_period_type_key UNIQUE (provider_id, period_start_date, period_type);


--
-- Name: provider provider_npi_key; Type: CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.provider
    ADD CONSTRAINT provider_npi_key UNIQUE (npi);


--
-- Name: provider provider_pkey; Type: CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.provider
    ADD CONSTRAINT provider_pkey PRIMARY KEY (provider_id);


--
-- Name: region region_organization_id_region_code_key; Type: CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.region
    ADD CONSTRAINT region_organization_id_region_code_key UNIQUE (organization_id, region_code);


--
-- Name: region region_pkey; Type: CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.region
    ADD CONSTRAINT region_pkey PRIMARY KEY (region_id);


--
-- Name: reviewer reviewer_pkey; Type: CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.reviewer
    ADD CONSTRAINT reviewer_pkey PRIMARY KEY (reviewer_id);


--
-- Name: reviewer reviewer_reviewer_code_key; Type: CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.reviewer
    ADD CONSTRAINT reviewer_reviewer_code_key UNIQUE (reviewer_code);


--
-- Name: rvu_reference rvu_reference_pkey; Type: CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.rvu_reference
    ADD CONSTRAINT rvu_reference_pkey PRIMARY KEY (rvu_id);


--
-- Name: service_line_adjustment service_line_adjustment_pkey; Type: CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.service_line_adjustment
    ADD CONSTRAINT service_line_adjustment_pkey PRIMARY KEY (adjustment_id);


--
-- Name: service_line_diagnosis_pointer service_line_diagnosis_pointer_pkey; Type: CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.service_line_diagnosis_pointer
    ADD CONSTRAINT service_line_diagnosis_pointer_pkey PRIMARY KEY (pointer_id);


--
-- Name: service_line_evaluation service_line_evaluation_pkey; Type: CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.service_line_evaluation
    ADD CONSTRAINT service_line_evaluation_pkey PRIMARY KEY (evaluation_id);


--
-- Name: service_line_flag service_line_flag_pkey; Type: CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.service_line_flag
    ADD CONSTRAINT service_line_flag_pkey PRIMARY KEY (flag_id);


--
-- Name: service_line service_line_pkey; Type: CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.service_line
    ADD CONSTRAINT service_line_pkey PRIMARY KEY (service_line_id);


--
-- Name: service_line_reimbursement service_line_reimbursement_pkey; Type: CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.service_line_reimbursement
    ADD CONSTRAINT service_line_reimbursement_pkey PRIMARY KEY (reimbursement_id);


--
-- Name: service_line_reimbursement service_line_reimbursement_service_line_id_key; Type: CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.service_line_reimbursement
    ADD CONSTRAINT service_line_reimbursement_service_line_id_key UNIQUE (service_line_id);


--
-- Name: encounter_diagnosis uk_encounter_diagnosis_seq; Type: CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.encounter_diagnosis
    ADD CONSTRAINT uk_encounter_diagnosis_seq UNIQUE (encounter_id, sequence_number);


--
-- Name: service_line uk_encounter_line; Type: CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.service_line
    ADD CONSTRAINT uk_encounter_line UNIQUE (encounter_id, line_number);


--
-- Name: gpci_reference uk_gpci_locality_year; Type: CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.gpci_reference
    ADD CONSTRAINT uk_gpci_locality_year UNIQUE (locality_code, effective_year);


--
-- Name: service_line_diagnosis_pointer uk_line_diag_pointer; Type: CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.service_line_diagnosis_pointer
    ADD CONSTRAINT uk_line_diag_pointer UNIQUE (service_line_id, pointer_sequence);


--
-- Name: rvu_reference uk_rvu_code_year_modifier; Type: CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.rvu_reference
    ADD CONSTRAINT uk_rvu_code_year_modifier UNIQUE (hcpcs_code, effective_year, modifier);


--
-- Name: ab_test_experiment ab_test_experiment_experiment_name_key; Type: CONSTRAINT; Schema: ml; Owner: -
--

ALTER TABLE ONLY ml.ab_test_experiment
    ADD CONSTRAINT ab_test_experiment_experiment_name_key UNIQUE (experiment_name);


--
-- Name: ab_test_experiment ab_test_experiment_pkey; Type: CONSTRAINT; Schema: ml; Owner: -
--

ALTER TABLE ONLY ml.ab_test_experiment
    ADD CONSTRAINT ab_test_experiment_pkey PRIMARY KEY (experiment_id);


--
-- Name: feature_definition feature_definition_feature_name_key; Type: CONSTRAINT; Schema: ml; Owner: -
--

ALTER TABLE ONLY ml.feature_definition
    ADD CONSTRAINT feature_definition_feature_name_key UNIQUE (feature_name);


--
-- Name: feature_definition feature_definition_pkey; Type: CONSTRAINT; Schema: ml; Owner: -
--

ALTER TABLE ONLY ml.feature_definition
    ADD CONSTRAINT feature_definition_pkey PRIMARY KEY (feature_id);


--
-- Name: model_performance_log model_performance_log_pkey; Type: CONSTRAINT; Schema: ml; Owner: -
--

ALTER TABLE ONLY ml.model_performance_log
    ADD CONSTRAINT model_performance_log_pkey PRIMARY KEY (performance_id);


--
-- Name: model_prediction model_prediction_pkey; Type: CONSTRAINT; Schema: ml; Owner: -
--

ALTER TABLE ONLY ml.model_prediction
    ADD CONSTRAINT model_prediction_pkey PRIMARY KEY (prediction_id);


--
-- Name: model_registry model_registry_model_name_model_version_key; Type: CONSTRAINT; Schema: ml; Owner: -
--

ALTER TABLE ONLY ml.model_registry
    ADD CONSTRAINT model_registry_model_name_model_version_key UNIQUE (model_name, model_version);


--
-- Name: model_registry model_registry_pkey; Type: CONSTRAINT; Schema: ml; Owner: -
--

ALTER TABLE ONLY ml.model_registry
    ADD CONSTRAINT model_registry_pkey PRIMARY KEY (model_id);


--
-- Name: training_dataset training_dataset_dataset_name_dataset_version_key; Type: CONSTRAINT; Schema: ml; Owner: -
--

ALTER TABLE ONLY ml.training_dataset
    ADD CONSTRAINT training_dataset_dataset_name_dataset_version_key UNIQUE (dataset_name, dataset_version);


--
-- Name: training_dataset training_dataset_pkey; Type: CONSTRAINT; Schema: ml; Owner: -
--

ALTER TABLE ONLY ml.training_dataset
    ADD CONSTRAINT training_dataset_pkey PRIMARY KEY (dataset_id);


--
-- Name: application_version application_version_pkey; Type: CONSTRAINT; Schema: staging; Owner: -
--

ALTER TABLE ONLY staging.application_version
    ADD CONSTRAINT application_version_pkey PRIMARY KEY (version);


--
-- Name: data_refresh_schedule data_refresh_schedule_pkey; Type: CONSTRAINT; Schema: staging; Owner: -
--

ALTER TABLE ONLY staging.data_refresh_schedule
    ADD CONSTRAINT data_refresh_schedule_pkey PRIMARY KEY (refresh_id);


--
-- Name: data_refresh_schedule data_refresh_schedule_schema_name_target_name_key; Type: CONSTRAINT; Schema: staging; Owner: -
--

ALTER TABLE ONLY staging.data_refresh_schedule
    ADD CONSTRAINT data_refresh_schedule_schema_name_target_name_key UNIQUE (schema_name, target_name);


--
-- Name: file_processing_queue file_processing_queue_pkey; Type: CONSTRAINT; Schema: staging; Owner: -
--

ALTER TABLE ONLY staging.file_processing_queue
    ADD CONSTRAINT file_processing_queue_pkey PRIMARY KEY (queue_id);


--
-- Name: file_upload file_upload_pkey; Type: CONSTRAINT; Schema: staging; Owner: -
--

ALTER TABLE ONLY staging.file_upload
    ADD CONSTRAINT file_upload_pkey PRIMARY KEY (upload_id);


--
-- Name: import_batch import_batch_pkey; Type: CONSTRAINT; Schema: staging; Owner: -
--

ALTER TABLE ONLY staging.import_batch
    ADD CONSTRAINT import_batch_pkey PRIMARY KEY (batch_id);


--
-- Name: import_configuration import_configuration_organization_id_configuration_name_key; Type: CONSTRAINT; Schema: staging; Owner: -
--

ALTER TABLE ONLY staging.import_configuration
    ADD CONSTRAINT import_configuration_organization_id_configuration_name_key UNIQUE (organization_id, configuration_name);


--
-- Name: import_configuration import_configuration_pkey; Type: CONSTRAINT; Schema: staging; Owner: -
--

ALTER TABLE ONLY staging.import_configuration
    ADD CONSTRAINT import_configuration_pkey PRIMARY KEY (configuration_id);


--
-- Name: import_error_log import_error_log_pkey; Type: CONSTRAINT; Schema: staging; Owner: -
--

ALTER TABLE ONLY staging.import_error_log
    ADD CONSTRAINT import_error_log_pkey PRIMARY KEY (error_id);


--
-- Name: job_execution_log job_execution_log_pkey; Type: CONSTRAINT; Schema: staging; Owner: -
--

ALTER TABLE ONLY staging.job_execution_log
    ADD CONSTRAINT job_execution_log_pkey PRIMARY KEY (execution_id);


--
-- Name: processing_metrics processing_metrics_pkey; Type: CONSTRAINT; Schema: staging; Owner: -
--

ALTER TABLE ONLY staging.processing_metrics
    ADD CONSTRAINT processing_metrics_pkey PRIMARY KEY (metric_id);


--
-- Name: report_generation_log report_generation_log_pkey; Type: CONSTRAINT; Schema: staging; Owner: -
--

ALTER TABLE ONLY staging.report_generation_log
    ADD CONSTRAINT report_generation_log_pkey PRIMARY KEY (report_log_id);


--
-- Name: report_subscription report_subscription_organization_id_subscription_name_key; Type: CONSTRAINT; Schema: staging; Owner: -
--

ALTER TABLE ONLY staging.report_subscription
    ADD CONSTRAINT report_subscription_organization_id_subscription_name_key UNIQUE (organization_id, subscription_name);


--
-- Name: report_subscription report_subscription_pkey; Type: CONSTRAINT; Schema: staging; Owner: -
--

ALTER TABLE ONLY staging.report_subscription
    ADD CONSTRAINT report_subscription_pkey PRIMARY KEY (subscription_id);


--
-- Name: rules_configuration rules_configuration_organization_id_rule_code_key; Type: CONSTRAINT; Schema: staging; Owner: -
--

ALTER TABLE ONLY staging.rules_configuration
    ADD CONSTRAINT rules_configuration_organization_id_rule_code_key UNIQUE (organization_id, rule_code);


--
-- Name: rules_configuration rules_configuration_pkey; Type: CONSTRAINT; Schema: staging; Owner: -
--

ALTER TABLE ONLY staging.rules_configuration
    ADD CONSTRAINT rules_configuration_pkey PRIMARY KEY (rule_config_id);


--
-- Name: scheduled_job scheduled_job_organization_id_job_name_key; Type: CONSTRAINT; Schema: staging; Owner: -
--

ALTER TABLE ONLY staging.scheduled_job
    ADD CONSTRAINT scheduled_job_organization_id_job_name_key UNIQUE (organization_id, job_name);


--
-- Name: scheduled_job scheduled_job_pkey; Type: CONSTRAINT; Schema: staging; Owner: -
--

ALTER TABLE ONLY staging.scheduled_job
    ADD CONSTRAINT scheduled_job_pkey PRIMARY KEY (job_id);


--
-- Name: schema_migrations schema_migrations_pkey; Type: CONSTRAINT; Schema: staging; Owner: -
--

ALTER TABLE ONLY staging.schema_migrations
    ADD CONSTRAINT schema_migrations_pkey PRIMARY KEY (migration_name);


--
-- Name: upgrade_test upgrade_test_pkey; Type: CONSTRAINT; Schema: staging; Owner: -
--

ALTER TABLE ONLY staging.upgrade_test
    ADD CONSTRAINT upgrade_test_pkey PRIMARY KEY (id);


--
-- Name: idx_audit_assignment_dos_range; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_audit_assignment_dos_range ON claims.audit_assignment USING btree (dos_from, dos_to);


--
-- Name: idx_audit_assignment_due_date; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_audit_assignment_due_date ON claims.audit_assignment USING btree (due_date);


--
-- Name: idx_audit_assignment_facility; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_audit_assignment_facility ON claims.audit_assignment USING btree (facility_id);


--
-- Name: idx_audit_assignment_org; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_audit_assignment_org ON claims.audit_assignment USING btree (organization_id);


--
-- Name: idx_audit_assignment_reviewer; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_audit_assignment_reviewer ON claims.audit_assignment USING btree (reviewer_id);


--
-- Name: idx_audit_assignment_status; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_audit_assignment_status ON claims.audit_assignment USING btree (audit_status);


--
-- Name: idx_audit_encounter_audit; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_audit_encounter_audit ON claims.audit_encounter USING btree (audit_id);


--
-- Name: idx_audit_encounter_audit_status; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_audit_encounter_audit_status ON claims.audit_encounter USING btree (audit_id, review_status, has_errors);


--
-- Name: idx_audit_encounter_encounter; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_audit_encounter_encounter ON claims.audit_encounter USING btree (encounter_id);


--
-- Name: idx_audit_encounter_has_errors; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_audit_encounter_has_errors ON claims.audit_encounter USING btree (has_errors) WHERE (has_errors = true);


--
-- Name: idx_audit_encounter_status; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_audit_encounter_status ON claims.audit_encounter USING btree (review_status);


--
-- Name: idx_coder_accuracy_coder; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_coder_accuracy_coder ON claims.coder_accuracy USING btree (coder_id);


--
-- Name: idx_coder_accuracy_org; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_coder_accuracy_org ON claims.coder_accuracy USING btree (organization_id);


--
-- Name: idx_coder_accuracy_period; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_coder_accuracy_period ON claims.coder_accuracy USING btree (period_start_date, period_end_date);


--
-- Name: idx_coder_accuracy_rate; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_coder_accuracy_rate ON claims.coder_accuracy USING btree (overall_accuracy_rate);


--
-- Name: idx_coder_active; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_coder_active ON claims.coder USING btree (is_active) WHERE (is_active = true);


--
-- Name: idx_coder_code; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_coder_code ON claims.coder USING btree (coder_code);


--
-- Name: idx_coder_group; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_coder_group ON claims.coder USING btree (coder_group);


--
-- Name: idx_coder_name; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_coder_name ON claims.coder USING btree (last_name, first_name);


--
-- Name: idx_coder_org; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_coder_org ON claims.coder USING btree (organization_id);


--
-- Name: idx_conversion_factor_effective; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_conversion_factor_effective ON claims.conversion_factor USING btree (effective_date);


--
-- Name: idx_conversion_factor_year; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_conversion_factor_year ON claims.conversion_factor USING btree (factor_year);


--
-- Name: idx_denial_appeal_decision; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_denial_appeal_decision ON claims.denial_appeal USING btree (payer_decision);


--
-- Name: idx_denial_appeal_denial; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_denial_appeal_denial ON claims.denial_appeal USING btree (denial_id);


--
-- Name: idx_denial_appeal_due_date; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_denial_appeal_due_date ON claims.denial_appeal USING btree (due_date);


--
-- Name: idx_denial_appeal_filed_date; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_denial_appeal_filed_date ON claims.denial_appeal USING btree (filed_date);


--
-- Name: idx_denial_appeal_level; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_denial_appeal_level ON claims.denial_appeal USING btree (appeal_level);


--
-- Name: idx_denial_appeal_status; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_denial_appeal_status ON claims.denial_appeal USING btree (appeal_status);


--
-- Name: idx_denial_event_appeal_deadline; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_denial_event_appeal_deadline ON claims.denial_event USING btree (appeal_deadline) WHERE ((appeal_filed = false) AND ((denial_status)::text <> ALL ((ARRAY['CLOSED'::character varying, 'WRITTEN_OFF'::character varying])::text[])));


--
-- Name: idx_denial_event_carc; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_denial_event_carc ON claims.denial_event USING btree (claim_adjustment_reason_code);


--
-- Name: idx_denial_event_coder; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_denial_event_coder ON claims.denial_event USING btree (coder_id);


--
-- Name: idx_denial_event_denial_date; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_denial_event_denial_date ON claims.denial_event USING btree (denial_date);


--
-- Name: idx_denial_event_encounter; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_denial_event_encounter ON claims.denial_event USING btree (encounter_id);


--
-- Name: idx_denial_event_facility; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_denial_event_facility ON claims.denial_event USING btree (facility_id);


--
-- Name: idx_denial_event_facility_date; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_denial_event_facility_date ON claims.denial_event USING btree (facility_id, denial_date);


--
-- Name: idx_denial_event_org; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_denial_event_org ON claims.denial_event USING btree (organization_id);


--
-- Name: idx_denial_event_org_status_date; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_denial_event_org_status_date ON claims.denial_event USING btree (organization_id, denial_status, denial_date);


--
-- Name: idx_denial_event_payer; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_denial_event_payer ON claims.denial_event USING btree (payer_id);


--
-- Name: idx_denial_event_preventable; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_denial_event_preventable ON claims.denial_event USING btree (is_preventable) WHERE (is_preventable = true);


--
-- Name: idx_denial_event_provider; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_denial_event_provider ON claims.denial_event USING btree (provider_id);


--
-- Name: idx_denial_event_resolution; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_denial_event_resolution ON claims.denial_event USING btree (resolution_status);


--
-- Name: idx_denial_event_responsible; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_denial_event_responsible ON claims.denial_event USING btree (responsible_party);


--
-- Name: idx_denial_event_root_cause; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_denial_event_root_cause ON claims.denial_event USING btree (root_cause_category);


--
-- Name: idx_denial_event_service_date; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_denial_event_service_date ON claims.denial_event USING btree (service_date);


--
-- Name: idx_denial_event_service_line; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_denial_event_service_line ON claims.denial_event USING btree (service_line_id);


--
-- Name: idx_denial_event_status; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_denial_event_status ON claims.denial_event USING btree (denial_status);


--
-- Name: idx_denial_org_preventable_date; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_denial_org_preventable_date ON claims.denial_event USING btree (organization_id, is_preventable, denial_date) WHERE (is_preventable = true);


--
-- Name: idx_denial_reason_code_active; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_denial_reason_code_active ON claims.denial_reason_code USING btree (is_active) WHERE (is_active = true);


--
-- Name: idx_denial_reason_code_category; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_denial_reason_code_category ON claims.denial_reason_code USING btree (category);


--
-- Name: idx_denial_reason_code_code; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_denial_reason_code_code ON claims.denial_reason_code USING btree (reason_code);


--
-- Name: idx_denial_reason_code_type; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_denial_reason_code_type ON claims.denial_reason_code USING btree (code_type);


--
-- Name: idx_denial_stats_denial_rate; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_denial_stats_denial_rate ON claims.denial_statistics USING btree (denial_rate);


--
-- Name: idx_denial_stats_dimension; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_denial_stats_dimension ON claims.denial_statistics USING btree (statistic_dimension);


--
-- Name: idx_denial_stats_facility; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_denial_stats_facility ON claims.denial_statistics USING btree (facility_id);


--
-- Name: idx_denial_stats_org; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_denial_stats_org ON claims.denial_statistics USING btree (organization_id);


--
-- Name: idx_denial_stats_period; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_denial_stats_period ON claims.denial_statistics USING btree (period_start_date, period_end_date);


--
-- Name: idx_diagnosis_eval_audit_enc; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_diagnosis_eval_audit_enc ON claims.diagnosis_evaluation USING btree (audit_encounter_id);


--
-- Name: idx_diagnosis_eval_diagnosis; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_diagnosis_eval_diagnosis ON claims.diagnosis_evaluation USING btree (encounter_diagnosis_id);


--
-- Name: idx_diagnosis_eval_has_error; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_diagnosis_eval_has_error ON claims.diagnosis_evaluation USING btree (has_error) WHERE (has_error = true);


--
-- Name: idx_diagnosis_eval_hcc; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_diagnosis_eval_hcc ON claims.diagnosis_evaluation USING btree (hcc_impact) WHERE (hcc_impact = true);


--
-- Name: idx_diagnosis_eval_result; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_diagnosis_eval_result ON claims.diagnosis_evaluation USING btree (evaluation_result);


--
-- Name: idx_diagnosis_eval_reviewer; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_diagnosis_eval_reviewer ON claims.diagnosis_evaluation USING btree (reviewer_id);


--
-- Name: idx_enc_diag_code; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_enc_diag_code ON claims.encounter_diagnosis USING btree (diagnosis_code);


--
-- Name: idx_enc_diag_encounter; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_enc_diag_encounter ON claims.encounter_diagnosis USING btree (encounter_id);


--
-- Name: idx_enc_diag_hcc; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_enc_diag_hcc ON claims.encounter_diagnosis USING btree (hcc_indicator, hcc_category) WHERE (hcc_indicator = true);


--
-- Name: idx_enc_diag_principal; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_enc_diag_principal ON claims.encounter_diagnosis USING btree (encounter_id, is_principal) WHERE (is_principal = true);


--
-- Name: idx_encounter_active; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_encounter_active ON claims.encounter USING btree (is_active) WHERE (is_active = true);


--
-- Name: idx_encounter_active_created; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_encounter_active_created ON claims.encounter USING btree (created_at DESC) WHERE ((is_active = true) AND (soft_deleted = false));


--
-- Name: idx_encounter_billing_provider; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_encounter_billing_provider ON claims.encounter USING btree (billing_provider_id);


--
-- Name: idx_encounter_coder; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_encounter_coder ON claims.encounter USING btree (coder_id);


--
-- Name: idx_encounter_coding_date; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_encounter_coding_date ON claims.encounter USING btree (coding_date);


--
-- Name: idx_encounter_created_at; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_encounter_created_at ON claims.encounter USING btree (created_at);


--
-- Name: idx_encounter_diagnosis_code; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_encounter_diagnosis_code ON claims.encounter_diagnosis USING btree (diagnosis_code, encounter_id) WHERE (diagnosis_code IS NOT NULL);


--
-- Name: INDEX idx_encounter_diagnosis_code; Type: COMMENT; Schema: claims; Owner: -
--

COMMENT ON INDEX claims.idx_encounter_diagnosis_code IS 'Phase 5: Optimizes diagnosis code lookups for validation rules.';


--
-- Name: idx_encounter_diagnosis_principal; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_encounter_diagnosis_principal ON claims.encounter_diagnosis USING btree (encounter_id, sequence_number) WHERE (is_principal = true);


--
-- Name: idx_encounter_diagnosis_sequence; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_encounter_diagnosis_sequence ON claims.encounter_diagnosis USING btree (encounter_id, sequence_number, is_principal) WHERE (is_principal = true);


--
-- Name: INDEX idx_encounter_diagnosis_sequence; Type: COMMENT; Schema: claims; Owner: -
--

COMMENT ON INDEX claims.idx_encounter_diagnosis_sequence IS 'Phase 5: Optimizes principal diagnosis lookups.';


--
-- Name: idx_encounter_dos_from; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_encounter_dos_from ON claims.encounter USING btree (date_of_service_from);


--
-- Name: idx_encounter_dos_range; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_encounter_dos_range ON claims.encounter USING btree (date_of_service_from, date_of_service_to);


--
-- Name: idx_encounter_dos_to; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_encounter_dos_to ON claims.encounter USING btree (date_of_service_to);


--
-- Name: idx_encounter_facility; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_encounter_facility ON claims.encounter USING btree (facility_id);


--
-- Name: idx_encounter_facility_date; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_encounter_facility_date ON claims.encounter USING btree (facility_id, date_of_service_from DESC) WHERE ((is_active = true) AND (soft_deleted = false));


--
-- Name: INDEX idx_encounter_facility_date; Type: COMMENT; Schema: claims; Owner: -
--

COMMENT ON INDEX claims.idx_encounter_facility_date IS 'Phase 5: Optimizes facility-based encounter queries.';


--
-- Name: idx_encounter_facility_dos; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_encounter_facility_dos ON claims.encounter USING btree (facility_id, date_of_service_from);


--
-- Name: idx_encounter_flag_created; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_encounter_flag_created ON claims.encounter_flag USING btree (created_at);


--
-- Name: idx_encounter_flag_enc_status; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_encounter_flag_enc_status ON claims.encounter_flag USING btree (encounter_id, flag_status);


--
-- Name: idx_encounter_flag_encounter; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_encounter_flag_encounter ON claims.encounter_flag USING btree (encounter_id);


--
-- Name: idx_encounter_flag_issue; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_encounter_flag_issue ON claims.encounter_flag USING btree (issue_id);


--
-- Name: idx_encounter_flag_severity; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_encounter_flag_severity ON claims.encounter_flag USING btree (severity);


--
-- Name: idx_encounter_flag_status; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_encounter_flag_status ON claims.encounter_flag USING btree (flag_status);


--
-- Name: idx_encounter_flag_status_created; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_encounter_flag_status_created ON claims.encounter_flag USING btree (flag_status, created_at) WHERE ((flag_status)::text = 'OPEN'::text);


--
-- Name: idx_encounter_flag_type; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_encounter_flag_type ON claims.encounter_flag USING btree (flag_type);


--
-- Name: idx_encounter_import_batch; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_encounter_import_batch ON claims.encounter USING btree (import_batch_id);


--
-- Name: idx_encounter_import_date; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_encounter_import_date ON claims.encounter USING btree (import_date);


--
-- Name: idx_encounter_import_date_facility; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_encounter_import_date_facility ON claims.encounter USING btree (facility_id, import_date DESC, date_of_service_from DESC) WHERE (is_active = true);


--
-- Name: idx_encounter_needs_review; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_encounter_needs_review ON claims.encounter USING btree (encounter_id, claim_status) WHERE ((claim_status)::text = ANY ((ARRAY['PENDING'::character varying, 'FLAGGED'::character varying])::text[]));


--
-- Name: idx_encounter_not_deleted; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_encounter_not_deleted ON claims.encounter USING btree (soft_deleted) WHERE (soft_deleted = false);


--
-- Name: idx_encounter_note_created; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_encounter_note_created ON claims.encounter_note USING btree (created_at);


--
-- Name: idx_encounter_note_encounter; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_encounter_note_encounter ON claims.encounter_note USING btree (encounter_id);


--
-- Name: idx_encounter_note_type; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_encounter_note_type ON claims.encounter_note USING btree (note_type);


--
-- Name: idx_encounter_org_dos; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_encounter_org_dos ON claims.encounter USING btree (organization_id, date_of_service_from);


--
-- Name: idx_encounter_org_dos_status; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_encounter_org_dos_status ON claims.encounter USING btree (organization_id, date_of_service_from DESC, claim_status) WHERE ((is_active = true) AND (soft_deleted = false));


--
-- Name: idx_encounter_org_facility_dos; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_encounter_org_facility_dos ON claims.encounter USING btree (organization_id, facility_id, date_of_service_from);


--
-- Name: idx_encounter_organization; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_encounter_organization ON claims.encounter USING btree (organization_id);


--
-- Name: idx_encounter_patient_control; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_encounter_patient_control ON claims.encounter USING btree (patient_control_number);


--
-- Name: idx_encounter_patient_control_trgm; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_encounter_patient_control_trgm ON claims.encounter USING gin (patient_control_number public.gin_trgm_ops);


--
-- Name: idx_encounter_provider_dos; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_encounter_provider_dos ON claims.encounter USING btree (billing_provider_id, date_of_service_from);


--
-- Name: idx_encounter_provider_status_dos; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_encounter_provider_status_dos ON claims.encounter USING btree (rendering_provider_id, claim_status, date_of_service_from) WHERE ((claim_status)::text = ANY ((ARRAY['PENDING'::character varying, 'FLAGGED'::character varying, 'NEW'::character varying])::text[]));


--
-- Name: idx_encounter_referring_provider; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_encounter_referring_provider ON claims.encounter USING btree (referring_provider_id);


--
-- Name: idx_encounter_rendering_provider; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_encounter_rendering_provider ON claims.encounter USING btree (rendering_provider_id);


--
-- Name: idx_encounter_service_date_facility; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_encounter_service_date_facility ON claims.encounter USING btree (facility_id, date_of_service_from, import_date) WHERE (is_active = true);


--
-- Name: idx_encounter_status; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_encounter_status ON claims.encounter USING btree (claim_status);


--
-- Name: idx_encounter_status_date; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_encounter_status_date ON claims.encounter USING btree (claim_status, date_of_service_from DESC, organization_id) WHERE ((is_active = true) AND (soft_deleted = false));


--
-- Name: INDEX idx_encounter_status_date; Type: COMMENT; Schema: claims; Owner: -
--

COMMENT ON INDEX claims.idx_encounter_status_date IS 'Phase 5: Optimizes claim status queries for dashboards and reporting.';


--
-- Name: idx_encounter_status_dos; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_encounter_status_dos ON claims.encounter USING btree (claim_status, date_of_service_from);


--
-- Name: idx_encounter_subscriber; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_encounter_subscriber ON claims.encounter USING btree (subscriber_id);


--
-- Name: idx_encounter_subscriber_dos; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_encounter_subscriber_dos ON claims.encounter USING btree (subscriber_id, date_of_service_from DESC) WHERE ((is_active = true) AND (soft_deleted = false));


--
-- Name: idx_encounter_subscriber_history; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_encounter_subscriber_history ON claims.encounter USING btree (subscriber_id, date_of_service_from DESC) WHERE ((is_active = true) AND (soft_deleted = false));


--
-- Name: INDEX idx_encounter_subscriber_history; Type: COMMENT; Schema: claims; Owner: -
--

COMMENT ON INDEX claims.idx_encounter_subscriber_history IS 'Phase 5: Optimizes encounter history lookups by subscriber for temporal pattern detection.';


--
-- Name: idx_encounter_subscriber_last_name_trgm; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_encounter_subscriber_last_name_trgm ON claims.encounter USING gin (subscriber_last_name public.gin_trgm_ops);


--
-- Name: idx_facility_active; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_facility_active ON claims.facility USING btree (is_active) WHERE (is_active = true);


--
-- Name: idx_facility_code; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_facility_code ON claims.facility USING btree (facility_code);


--
-- Name: idx_facility_name_trgm; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_facility_name_trgm ON claims.facility USING gin (facility_name public.gin_trgm_ops);


--
-- Name: idx_facility_npi; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_facility_npi ON claims.facility USING btree (npi);


--
-- Name: idx_facility_org; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_facility_org ON claims.facility USING btree (organization_id);


--
-- Name: idx_facility_region; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_facility_region ON claims.facility USING btree (region_id);


--
-- Name: idx_facility_single_region; Type: INDEX; Schema: claims; Owner: -
--

CREATE UNIQUE INDEX idx_facility_single_region ON claims.facility USING btree (facility_id) WHERE (region_id IS NOT NULL);


--
-- Name: idx_flag_category_code; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_flag_category_code ON claims.flag_category USING btree (category_code);


--
-- Name: idx_flag_issue_category; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_flag_issue_category ON claims.flag_issue USING btree (category_id);


--
-- Name: idx_flag_issue_code; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_flag_issue_code ON claims.flag_issue USING btree (issue_code);


--
-- Name: idx_flag_org_severity_status; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_flag_org_severity_status ON claims.encounter_flag USING btree (severity, flag_status, created_at) WHERE ((flag_status)::text = 'OPEN'::text);


--
-- Name: idx_gpci_locality; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_gpci_locality ON claims.gpci_reference USING btree (locality_code);


--
-- Name: idx_gpci_state; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_gpci_state ON claims.gpci_reference USING btree (state_code);


--
-- Name: idx_gpci_year; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_gpci_year ON claims.gpci_reference USING btree (effective_year);


--
-- Name: idx_line_diag_ptr_diag; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_line_diag_ptr_diag ON claims.service_line_diagnosis_pointer USING btree (diagnosis_id);


--
-- Name: idx_line_diag_ptr_line; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_line_diag_ptr_line ON claims.service_line_diagnosis_pointer USING btree (service_line_id);


--
-- Name: idx_modifier_adjustment_active; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_modifier_adjustment_active ON claims.modifier_adjustment USING btree (is_active) WHERE (is_active = true);


--
-- Name: idx_modifier_adjustment_code; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_modifier_adjustment_code ON claims.modifier_adjustment USING btree (modifier_code);


--
-- Name: idx_mv_denial_stats_date; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_mv_denial_stats_date ON claims.mv_denial_statistics USING btree (denial_date);


--
-- Name: idx_mv_denial_stats_unique; Type: INDEX; Schema: claims; Owner: -
--

CREATE UNIQUE INDEX idx_mv_denial_stats_unique ON claims.mv_denial_statistics USING btree (organization_id, facility_id, denial_date, payer_id, root_cause_category);


--
-- Name: idx_mv_flag_stats_date; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_mv_flag_stats_date ON claims.mv_flag_statistics USING btree (flag_date);


--
-- Name: idx_mv_flag_stats_unique; Type: INDEX; Schema: claims; Owner: -
--

CREATE UNIQUE INDEX idx_mv_flag_stats_unique ON claims.mv_flag_statistics USING btree (organization_id, facility_id, flag_date, category_code, severity);


--
-- Name: idx_organization_active; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_organization_active ON claims.organization USING btree (is_active) WHERE (is_active = true);


--
-- Name: idx_organization_code; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_organization_code ON claims.organization USING btree (organization_code);


--
-- Name: idx_provider_accuracy_org; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_provider_accuracy_org ON claims.provider_accuracy USING btree (organization_id);


--
-- Name: idx_provider_accuracy_period; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_provider_accuracy_period ON claims.provider_accuracy USING btree (period_start_date, period_end_date);


--
-- Name: idx_provider_accuracy_provider; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_provider_accuracy_provider ON claims.provider_accuracy USING btree (provider_id);


--
-- Name: idx_provider_accuracy_rate; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_provider_accuracy_rate ON claims.provider_accuracy USING btree (overall_accuracy_rate);


--
-- Name: idx_provider_active; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_provider_active ON claims.provider USING btree (is_active) WHERE (is_active = true);


--
-- Name: idx_provider_last_name_trgm; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_provider_last_name_trgm ON claims.provider USING gin (last_name public.gin_trgm_ops);


--
-- Name: idx_provider_name; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_provider_name ON claims.provider USING btree (last_name, first_name);


--
-- Name: idx_provider_npi; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_provider_npi ON claims.provider USING btree (npi);


--
-- Name: idx_provider_npi_lookup; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_provider_npi_lookup ON claims.provider USING btree (npi) WHERE ((is_active = true) AND (npi IS NOT NULL));


--
-- Name: INDEX idx_provider_npi_lookup; Type: COMMENT; Schema: claims; Owner: -
--

COMMENT ON INDEX claims.idx_provider_npi_lookup IS 'Phase 5: Optimizes provider lookups by NPI in cache population.';


--
-- Name: idx_provider_org; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_provider_org ON claims.provider USING btree (organization_id);


--
-- Name: idx_provider_specialty; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_provider_specialty ON claims.provider USING btree (specialty);


--
-- Name: INDEX idx_provider_specialty; Type: COMMENT; Schema: claims; Owner: -
--

COMMENT ON INDEX claims.idx_provider_specialty IS 'Phase 5: Optimizes provider specialty lookups for analytics and ML features.';


--
-- Name: idx_provider_type; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_provider_type ON claims.provider USING btree (provider_type);


--
-- Name: idx_region_active; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_region_active ON claims.region USING btree (is_active) WHERE (is_active = true);


--
-- Name: idx_region_code; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_region_code ON claims.region USING btree (region_code);


--
-- Name: idx_region_org; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_region_org ON claims.region USING btree (organization_id);


--
-- Name: idx_reviewer_active; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_reviewer_active ON claims.reviewer USING btree (is_active) WHERE (is_active = true);


--
-- Name: idx_reviewer_code; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_reviewer_code ON claims.reviewer USING btree (reviewer_code);


--
-- Name: idx_reviewer_group; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_reviewer_group ON claims.reviewer USING btree (reviewer_group);


--
-- Name: idx_reviewer_name; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_reviewer_name ON claims.reviewer USING btree (last_name, first_name);


--
-- Name: idx_reviewer_org; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_reviewer_org ON claims.reviewer USING btree (organization_id);


--
-- Name: idx_rvu_code_year; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_rvu_code_year ON claims.rvu_reference USING btree (hcpcs_code, effective_year);


--
-- Name: idx_rvu_effective_date; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_rvu_effective_date ON claims.rvu_reference USING btree (effective_date);


--
-- Name: idx_rvu_hcpcs; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_rvu_hcpcs ON claims.rvu_reference USING btree (hcpcs_code);


--
-- Name: idx_rvu_status; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_rvu_status ON claims.rvu_reference USING btree (status_code);


--
-- Name: idx_rvu_year; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_rvu_year ON claims.rvu_reference USING btree (effective_year);


--
-- Name: idx_service_line_adj_line; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_service_line_adj_line ON claims.service_line_adjustment USING btree (service_line_id);


--
-- Name: idx_service_line_adj_reason; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_service_line_adj_reason ON claims.service_line_adjustment USING btree (adjustment_reason_code);


--
-- Name: idx_service_line_date_from; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_service_line_date_from ON claims.service_line USING btree (service_date_from);


--
-- Name: idx_service_line_date_to; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_service_line_date_to ON claims.service_line USING btree (service_date_to);


--
-- Name: idx_service_line_enc_line; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_service_line_enc_line ON claims.service_line USING btree (encounter_id, line_number);


--
-- Name: idx_service_line_encounter; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_service_line_encounter ON claims.service_line USING btree (encounter_id);


--
-- Name: idx_service_line_encounter_proc; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_service_line_encounter_proc ON claims.service_line USING btree (encounter_id, procedure_code) INCLUDE (service_unit_count, line_item_charge_amount, service_date_from);


--
-- Name: idx_service_line_eval_audit_enc; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_service_line_eval_audit_enc ON claims.service_line_evaluation USING btree (audit_encounter_id);


--
-- Name: idx_service_line_eval_has_error; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_service_line_eval_has_error ON claims.service_line_evaluation USING btree (has_error) WHERE (has_error = true);


--
-- Name: idx_service_line_eval_issue; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_service_line_eval_issue ON claims.service_line_evaluation USING btree (issue_id);


--
-- Name: idx_service_line_eval_line; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_service_line_eval_line ON claims.service_line_evaluation USING btree (service_line_id);


--
-- Name: idx_service_line_eval_result; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_service_line_eval_result ON claims.service_line_evaluation USING btree (evaluation_result);


--
-- Name: idx_service_line_eval_result_severity; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_service_line_eval_result_severity ON claims.service_line_evaluation USING btree (evaluation_result, issue_severity) WHERE (has_error = true);


--
-- Name: idx_service_line_eval_reviewer; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_service_line_eval_reviewer ON claims.service_line_evaluation USING btree (reviewer_id);


--
-- Name: idx_service_line_flag_created; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_service_line_flag_created ON claims.service_line_flag USING btree (created_at);


--
-- Name: idx_service_line_flag_issue; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_service_line_flag_issue ON claims.service_line_flag USING btree (issue_id);


--
-- Name: idx_service_line_flag_line; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_service_line_flag_line ON claims.service_line_flag USING btree (service_line_id);


--
-- Name: idx_service_line_flag_status; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_service_line_flag_status ON claims.service_line_flag USING btree (flag_status);


--
-- Name: idx_service_line_flag_type; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_service_line_flag_type ON claims.service_line_flag USING btree (flag_type);


--
-- Name: idx_service_line_ndc; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_service_line_ndc ON claims.service_line USING btree (ndc_code) WHERE (ndc_code IS NOT NULL);


--
-- Name: idx_service_line_proc_date; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_service_line_proc_date ON claims.service_line USING btree (procedure_code, service_date_from);


--
-- Name: idx_service_line_proc_date_facility; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_service_line_proc_date_facility ON claims.service_line USING btree (procedure_code, service_date_from, encounter_id);


--
-- Name: idx_service_line_procedure; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_service_line_procedure ON claims.service_line USING btree (procedure_code);


--
-- Name: idx_service_line_provider; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_service_line_provider ON claims.service_line USING btree (rendering_provider_id, service_date_from DESC) WHERE (rendering_provider_id IS NOT NULL);


--
-- Name: idx_service_line_reimb_calculated; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_service_line_reimb_calculated ON claims.service_line_reimbursement USING btree (calculated_at);


--
-- Name: idx_service_line_reimb_line; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_service_line_reimb_line ON claims.service_line_reimbursement USING btree (service_line_id);


--
-- Name: idx_service_line_reimb_rvu; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_service_line_reimb_rvu ON claims.service_line_reimbursement USING btree (rvu_id);


--
-- Name: idx_service_line_rendering_provider; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_service_line_rendering_provider ON claims.service_line USING btree (rendering_provider_id);


--
-- Name: idx_service_line_revenue; Type: INDEX; Schema: claims; Owner: -
--

CREATE INDEX idx_service_line_revenue ON claims.service_line USING btree (revenue_code) WHERE (revenue_code IS NOT NULL);


--
-- Name: idx_ab_test_active; Type: INDEX; Schema: ml; Owner: -
--

CREATE INDEX idx_ab_test_active ON ml.ab_test_experiment USING btree (is_active) WHERE (is_active = true);


--
-- Name: idx_ab_test_org; Type: INDEX; Schema: ml; Owner: -
--

CREATE INDEX idx_ab_test_org ON ml.ab_test_experiment USING btree (organization_id);


--
-- Name: idx_ab_test_status; Type: INDEX; Schema: ml; Owner: -
--

CREATE INDEX idx_ab_test_status ON ml.ab_test_experiment USING btree (experiment_status);


--
-- Name: idx_feature_definition_active; Type: INDEX; Schema: ml; Owner: -
--

CREATE INDEX idx_feature_definition_active ON ml.feature_definition USING btree (is_active) WHERE (is_active = true);


--
-- Name: idx_feature_definition_category; Type: INDEX; Schema: ml; Owner: -
--

CREATE INDEX idx_feature_definition_category ON ml.feature_definition USING btree (feature_category);


--
-- Name: idx_feature_definition_name; Type: INDEX; Schema: ml; Owner: -
--

CREATE INDEX idx_feature_definition_name ON ml.feature_definition USING btree (feature_name);


--
-- Name: idx_feature_definition_type; Type: INDEX; Schema: ml; Owner: -
--

CREATE INDEX idx_feature_definition_type ON ml.feature_definition USING btree (feature_type);


--
-- Name: idx_model_performance_alert; Type: INDEX; Schema: ml; Owner: -
--

CREATE INDEX idx_model_performance_alert ON ml.model_performance_log USING btree (performance_alert_level) WHERE ((performance_alert_level)::text = ANY ((ARRAY['WARNING'::character varying, 'CRITICAL'::character varying])::text[]));


--
-- Name: idx_model_performance_date; Type: INDEX; Schema: ml; Owner: -
--

CREATE INDEX idx_model_performance_date ON ml.model_performance_log USING btree (measurement_date);


--
-- Name: idx_model_performance_model; Type: INDEX; Schema: ml; Owner: -
--

CREATE INDEX idx_model_performance_model ON ml.model_performance_log USING btree (model_id);


--
-- Name: idx_model_performance_period; Type: INDEX; Schema: ml; Owner: -
--

CREATE INDEX idx_model_performance_period ON ml.model_performance_log USING btree (period_start, period_end);


--
-- Name: idx_model_prediction_encounter; Type: INDEX; Schema: ml; Owner: -
--

CREATE INDEX idx_model_prediction_encounter ON ml.model_prediction USING btree (encounter_id);


--
-- Name: idx_model_prediction_encounter_type; Type: INDEX; Schema: ml; Owner: -
--

CREATE INDEX idx_model_prediction_encounter_type ON ml.model_prediction USING btree (encounter_id, prediction_type, predicted_at DESC);


--
-- Name: idx_model_prediction_model; Type: INDEX; Schema: ml; Owner: -
--

CREATE INDEX idx_model_prediction_model ON ml.model_prediction USING btree (model_id);


--
-- Name: idx_model_prediction_predicted_at; Type: INDEX; Schema: ml; Owner: -
--

CREATE INDEX idx_model_prediction_predicted_at ON ml.model_prediction USING btree (predicted_at);


--
-- Name: idx_model_prediction_risk; Type: INDEX; Schema: ml; Owner: -
--

CREATE INDEX idx_model_prediction_risk ON ml.model_prediction USING btree (risk_level, predicted_at DESC) WHERE ((risk_level)::text = ANY ((ARRAY['HIGH'::character varying, 'CRITICAL'::character varying])::text[]));


--
-- Name: idx_model_prediction_risk_level; Type: INDEX; Schema: ml; Owner: -
--

CREATE INDEX idx_model_prediction_risk_level ON ml.model_prediction USING btree (risk_level);


--
-- Name: idx_model_prediction_score; Type: INDEX; Schema: ml; Owner: -
--

CREATE INDEX idx_model_prediction_score ON ml.model_prediction USING btree (prediction_score);


--
-- Name: idx_model_prediction_service_line; Type: INDEX; Schema: ml; Owner: -
--

CREATE INDEX idx_model_prediction_service_line ON ml.model_prediction USING btree (service_line_id);


--
-- Name: idx_model_prediction_type; Type: INDEX; Schema: ml; Owner: -
--

CREATE INDEX idx_model_prediction_type ON ml.model_prediction USING btree (prediction_type);


--
-- Name: idx_model_registry_active; Type: INDEX; Schema: ml; Owner: -
--

CREATE INDEX idx_model_registry_active ON ml.model_registry USING btree (is_active) WHERE (is_active = true);


--
-- Name: idx_model_registry_deployment; Type: INDEX; Schema: ml; Owner: -
--

CREATE INDEX idx_model_registry_deployment ON ml.model_registry USING btree (deployment_status, model_purpose) WHERE (is_active = true);


--
-- Name: idx_model_registry_org; Type: INDEX; Schema: ml; Owner: -
--

CREATE INDEX idx_model_registry_org ON ml.model_registry USING btree (organization_id);


--
-- Name: idx_model_registry_purpose; Type: INDEX; Schema: ml; Owner: -
--

CREATE INDEX idx_model_registry_purpose ON ml.model_registry USING btree (model_purpose);


--
-- Name: idx_model_registry_status; Type: INDEX; Schema: ml; Owner: -
--

CREATE INDEX idx_model_registry_status ON ml.model_registry USING btree (deployment_status);


--
-- Name: idx_model_registry_type; Type: INDEX; Schema: ml; Owner: -
--

CREATE INDEX idx_model_registry_type ON ml.model_registry USING btree (model_type);


--
-- Name: idx_training_dataset_active; Type: INDEX; Schema: ml; Owner: -
--

CREATE INDEX idx_training_dataset_active ON ml.training_dataset USING btree (is_active) WHERE (is_active = true);


--
-- Name: idx_training_dataset_org; Type: INDEX; Schema: ml; Owner: -
--

CREATE INDEX idx_training_dataset_org ON ml.training_dataset USING btree (organization_id);


--
-- Name: idx_training_dataset_purpose; Type: INDEX; Schema: ml; Owner: -
--

CREATE INDEX idx_training_dataset_purpose ON ml.training_dataset USING btree (dataset_purpose);


--
-- Name: idx_training_dataset_status; Type: INDEX; Schema: ml; Owner: -
--

CREATE INDEX idx_training_dataset_status ON ml.training_dataset USING btree (dataset_status);


--
-- Name: idx_application_version_installed_at; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX idx_application_version_installed_at ON staging.application_version USING btree (installed_at DESC);


--
-- Name: idx_data_refresh_active; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX idx_data_refresh_active ON staging.data_refresh_schedule USING btree (is_active) WHERE (is_active = true);


--
-- Name: idx_data_refresh_next; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX idx_data_refresh_next ON staging.data_refresh_schedule USING btree (next_refresh_at) WHERE (is_active = true);


--
-- Name: idx_data_refresh_order; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX idx_data_refresh_order ON staging.data_refresh_schedule USING btree (execution_order);


--
-- Name: idx_data_refresh_org; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX idx_data_refresh_org ON staging.data_refresh_schedule USING btree (organization_id);


--
-- Name: idx_data_refresh_target_type; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX idx_data_refresh_target_type ON staging.data_refresh_schedule USING btree (target_type);


--
-- Name: idx_file_upload_expires; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX idx_file_upload_expires ON staging.file_upload USING btree (expires_at) WHERE ((upload_status)::text = 'IN_PROGRESS'::text);


--
-- Name: idx_file_upload_org; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX idx_file_upload_org ON staging.file_upload USING btree (organization_id);


--
-- Name: idx_file_upload_status; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX idx_file_upload_status ON staging.file_upload USING btree (upload_status);


--
-- Name: idx_import_batch_created; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX idx_import_batch_created ON staging.import_batch USING btree (created_at);


--
-- Name: idx_import_batch_facility; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX idx_import_batch_facility ON staging.import_batch USING btree (facility_id);


--
-- Name: idx_import_batch_file_hash; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX idx_import_batch_file_hash ON staging.import_batch USING btree (file_hash);


--
-- Name: idx_import_batch_org; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX idx_import_batch_org ON staging.import_batch USING btree (organization_id);


--
-- Name: idx_import_batch_org_created; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX idx_import_batch_org_created ON staging.import_batch USING btree (organization_id, created_at DESC);


--
-- Name: idx_import_batch_org_date; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX idx_import_batch_org_date ON staging.import_batch USING btree (organization_id, created_at DESC);


--
-- Name: INDEX idx_import_batch_org_date; Type: COMMENT; Schema: staging; Owner: -
--

COMMENT ON INDEX staging.idx_import_batch_org_date IS 'Phase 5: Optimizes organization-based batch queries.';


--
-- Name: idx_import_batch_org_status_created; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX idx_import_batch_org_status_created ON staging.import_batch USING btree (organization_id, import_status, created_at);


--
-- Name: idx_import_batch_started; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX idx_import_batch_started ON staging.import_batch USING btree (started_at);


--
-- Name: idx_import_batch_status; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX idx_import_batch_status ON staging.import_batch USING btree (import_status);


--
-- Name: idx_import_batch_type; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX idx_import_batch_type ON staging.import_batch USING btree (batch_type);


--
-- Name: idx_import_config_active; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX idx_import_config_active ON staging.import_configuration USING btree (is_active) WHERE (is_active = true);


--
-- Name: idx_import_config_default; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX idx_import_config_default ON staging.import_configuration USING btree (organization_id, is_default) WHERE (is_default = true);


--
-- Name: idx_import_config_org; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX idx_import_config_org ON staging.import_configuration USING btree (organization_id);


--
-- Name: idx_import_config_type; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX idx_import_config_type ON staging.import_configuration USING btree (configuration_type);


--
-- Name: idx_import_error_batch; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX idx_import_error_batch ON staging.import_error_log USING btree (batch_id);


--
-- Name: idx_import_error_created; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX idx_import_error_created ON staging.import_error_log USING btree (created_at);


--
-- Name: idx_import_error_severity; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX idx_import_error_severity ON staging.import_error_log USING btree (error_severity);


--
-- Name: idx_import_error_status; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX idx_import_error_status ON staging.import_error_log USING btree (resolution_status);


--
-- Name: idx_import_error_type; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX idx_import_error_type ON staging.import_error_log USING btree (error_type);


--
-- Name: idx_job_execution_job; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX idx_job_execution_job ON staging.job_execution_log USING btree (job_id);


--
-- Name: idx_job_execution_retry; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX idx_job_execution_retry ON staging.job_execution_log USING btree (original_execution_id) WHERE (is_retry = true);


--
-- Name: idx_job_execution_started; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX idx_job_execution_started ON staging.job_execution_log USING btree (started_at);


--
-- Name: idx_job_execution_status; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX idx_job_execution_status ON staging.job_execution_log USING btree (execution_status);


--
-- Name: idx_processing_metrics_batch; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX idx_processing_metrics_batch ON staging.processing_metrics USING btree (batch_id);


--
-- Name: idx_processing_metrics_started; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX idx_processing_metrics_started ON staging.processing_metrics USING btree (started_at);


--
-- Name: idx_processing_metrics_type; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX idx_processing_metrics_type ON staging.processing_metrics USING btree (metric_type);


--
-- Name: idx_processing_queue_status_created; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX idx_processing_queue_status_created ON staging.file_processing_queue USING btree (queue_status, created_at DESC);


--
-- Name: idx_queue_facility_fifo; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX idx_queue_facility_fifo ON staging.file_processing_queue USING btree (facility_id, priority, queued_at) WHERE (queue_status = ANY (ARRAY['QUEUED'::text, 'RETRY'::text]));


--
-- Name: INDEX idx_queue_facility_fifo; Type: COMMENT; Schema: staging; Owner: -
--

COMMENT ON INDEX staging.idx_queue_facility_fifo IS 'Phase 5: Optimizes per-facility FIFO queue processing with priority support.';


--
-- Name: idx_queue_facility_stats; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX idx_queue_facility_stats ON staging.file_processing_queue USING btree (facility_id, queue_status, created_at DESC);


--
-- Name: idx_queue_failed; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX idx_queue_failed ON staging.file_processing_queue USING btree (queue_status, queued_at DESC) WHERE (queue_status = 'FAILED'::text);


--
-- Name: idx_queue_fifo_by_facility; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX idx_queue_fifo_by_facility ON staging.file_processing_queue USING btree (facility_id, priority, queued_at) WHERE (queue_status = 'QUEUED'::text);


--
-- Name: idx_queue_fifo_global; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX idx_queue_fifo_global ON staging.file_processing_queue USING btree (priority, queued_at) WHERE (queue_status = 'QUEUED'::text);


--
-- Name: idx_queue_global_fifo; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX idx_queue_global_fifo ON staging.file_processing_queue USING btree (priority, queued_at) WHERE (queue_status = ANY (ARRAY['QUEUED'::text, 'RETRY'::text]));


--
-- Name: INDEX idx_queue_global_fifo; Type: COMMENT; Schema: staging; Owner: -
--

COMMENT ON INDEX staging.idx_queue_global_fifo IS 'Phase 5: Optimizes global FIFO queue processing with priority support.';


--
-- Name: idx_queue_organization; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX idx_queue_organization ON staging.file_processing_queue USING btree (organization_id, queue_status, queued_at);


--
-- Name: idx_queue_processing; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX idx_queue_processing ON staging.file_processing_queue USING btree (queue_status, processing_started_at DESC) WHERE (queue_status = 'PROCESSING'::text);


--
-- Name: idx_queue_retry; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX idx_queue_retry ON staging.file_processing_queue USING btree (queue_status, retry_count, queued_at) WHERE ((queue_status = 'RETRY'::text) AND (retry_count < max_retries));


--
-- Name: idx_queue_status_monitoring; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX idx_queue_status_monitoring ON staging.file_processing_queue USING btree (queue_status, queued_at DESC) WHERE (queue_status = 'PROCESSING'::text);


--
-- Name: INDEX idx_queue_status_monitoring; Type: COMMENT; Schema: staging; Owner: -
--

COMMENT ON INDEX staging.idx_queue_status_monitoring IS 'Phase 5: Optimizes queue status monitoring queries for currently processing files.';


--
-- Name: idx_report_generation_generated; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX idx_report_generation_generated ON staging.report_generation_log USING btree (generated_at);


--
-- Name: idx_report_generation_org; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX idx_report_generation_org ON staging.report_generation_log USING btree (organization_id);


--
-- Name: idx_report_generation_status; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX idx_report_generation_status ON staging.report_generation_log USING btree (generation_status);


--
-- Name: idx_report_generation_subscription; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX idx_report_generation_subscription ON staging.report_generation_log USING btree (subscription_id);


--
-- Name: idx_report_generation_type; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX idx_report_generation_type ON staging.report_generation_log USING btree (report_type);


--
-- Name: idx_report_subscription_active; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX idx_report_subscription_active ON staging.report_subscription USING btree (is_active) WHERE (is_active = true);


--
-- Name: idx_report_subscription_next_delivery; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX idx_report_subscription_next_delivery ON staging.report_subscription USING btree (next_delivery_at) WHERE (is_active = true);


--
-- Name: idx_report_subscription_org; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX idx_report_subscription_org ON staging.report_subscription USING btree (organization_id);


--
-- Name: idx_report_subscription_type; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX idx_report_subscription_type ON staging.report_subscription USING btree (report_type);


--
-- Name: idx_rules_config_active; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX idx_rules_config_active ON staging.rules_configuration USING btree (is_active) WHERE (is_active = true);


--
-- Name: idx_rules_config_category; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX idx_rules_config_category ON staging.rules_configuration USING btree (rule_category);


--
-- Name: idx_rules_config_code; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX idx_rules_config_code ON staging.rules_configuration USING btree (rule_code);


--
-- Name: idx_rules_config_execution_order; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX idx_rules_config_execution_order ON staging.rules_configuration USING btree (execution_order) WHERE (is_active = true);


--
-- Name: idx_rules_config_facility; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX idx_rules_config_facility ON staging.rules_configuration USING btree (facility_id);


--
-- Name: idx_rules_config_org; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX idx_rules_config_org ON staging.rules_configuration USING btree (organization_id);


--
-- Name: idx_scheduled_job_active; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX idx_scheduled_job_active ON staging.scheduled_job USING btree (is_active) WHERE (is_active = true);


--
-- Name: idx_scheduled_job_next_run; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX idx_scheduled_job_next_run ON staging.scheduled_job USING btree (next_run_at) WHERE ((is_active = true) AND (is_running = false));


--
-- Name: idx_scheduled_job_org; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX idx_scheduled_job_org ON staging.scheduled_job USING btree (organization_id);


--
-- Name: idx_scheduled_job_running; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX idx_scheduled_job_running ON staging.scheduled_job USING btree (is_running) WHERE (is_running = true);


--
-- Name: idx_scheduled_job_type; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX idx_scheduled_job_type ON staging.scheduled_job USING btree (job_type);


--
-- Name: idx_schema_migrations_applied_at; Type: INDEX; Schema: staging; Owner: -
--

CREATE INDEX idx_schema_migrations_applied_at ON staging.schema_migrations USING btree (applied_at);


--
-- Name: service_line sync_encounter_totals_delete; Type: TRIGGER; Schema: claims; Owner: -
--

CREATE TRIGGER sync_encounter_totals_delete AFTER DELETE ON claims.service_line FOR EACH ROW EXECUTE FUNCTION public.update_encounter_totals();


--
-- Name: service_line sync_encounter_totals_insert; Type: TRIGGER; Schema: claims; Owner: -
--

CREATE TRIGGER sync_encounter_totals_insert AFTER INSERT ON claims.service_line FOR EACH ROW EXECUTE FUNCTION public.update_encounter_totals();


--
-- Name: service_line sync_encounter_totals_update; Type: TRIGGER; Schema: claims; Owner: -
--

CREATE TRIGGER sync_encounter_totals_update AFTER UPDATE ON claims.service_line FOR EACH ROW WHEN ((old.line_item_charge_amount IS DISTINCT FROM new.line_item_charge_amount)) EXECUTE FUNCTION public.update_encounter_totals();


--
-- Name: audit_assignment update_audit_assignment_updated_at; Type: TRIGGER; Schema: claims; Owner: -
--

CREATE TRIGGER update_audit_assignment_updated_at BEFORE UPDATE ON claims.audit_assignment FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();


--
-- Name: coder update_coder_updated_at; Type: TRIGGER; Schema: claims; Owner: -
--

CREATE TRIGGER update_coder_updated_at BEFORE UPDATE ON claims.coder FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();


--
-- Name: conversion_factor update_conversion_factor_updated_at; Type: TRIGGER; Schema: claims; Owner: -
--

CREATE TRIGGER update_conversion_factor_updated_at BEFORE UPDATE ON claims.conversion_factor FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();


--
-- Name: denial_appeal update_denial_appeal_updated_at; Type: TRIGGER; Schema: claims; Owner: -
--

CREATE TRIGGER update_denial_appeal_updated_at BEFORE UPDATE ON claims.denial_appeal FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();


--
-- Name: denial_event update_denial_event_updated_at; Type: TRIGGER; Schema: claims; Owner: -
--

CREATE TRIGGER update_denial_event_updated_at BEFORE UPDATE ON claims.denial_event FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();


--
-- Name: denial_reason_code update_denial_reason_code_updated_at; Type: TRIGGER; Schema: claims; Owner: -
--

CREATE TRIGGER update_denial_reason_code_updated_at BEFORE UPDATE ON claims.denial_reason_code FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();


--
-- Name: encounter update_encounter_updated_at; Type: TRIGGER; Schema: claims; Owner: -
--

CREATE TRIGGER update_encounter_updated_at BEFORE UPDATE ON claims.encounter FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();


--
-- Name: facility update_facility_updated_at; Type: TRIGGER; Schema: claims; Owner: -
--

CREATE TRIGGER update_facility_updated_at BEFORE UPDATE ON claims.facility FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();


--
-- Name: gpci_reference update_gpci_reference_updated_at; Type: TRIGGER; Schema: claims; Owner: -
--

CREATE TRIGGER update_gpci_reference_updated_at BEFORE UPDATE ON claims.gpci_reference FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();


--
-- Name: modifier_adjustment update_modifier_adjustment_updated_at; Type: TRIGGER; Schema: claims; Owner: -
--

CREATE TRIGGER update_modifier_adjustment_updated_at BEFORE UPDATE ON claims.modifier_adjustment FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();


--
-- Name: organization update_organization_updated_at; Type: TRIGGER; Schema: claims; Owner: -
--

CREATE TRIGGER update_organization_updated_at BEFORE UPDATE ON claims.organization FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();


--
-- Name: provider update_provider_updated_at; Type: TRIGGER; Schema: claims; Owner: -
--

CREATE TRIGGER update_provider_updated_at BEFORE UPDATE ON claims.provider FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();


--
-- Name: region update_region_updated_at; Type: TRIGGER; Schema: claims; Owner: -
--

CREATE TRIGGER update_region_updated_at BEFORE UPDATE ON claims.region FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();


--
-- Name: reviewer update_reviewer_updated_at; Type: TRIGGER; Schema: claims; Owner: -
--

CREATE TRIGGER update_reviewer_updated_at BEFORE UPDATE ON claims.reviewer FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();


--
-- Name: rvu_reference update_rvu_reference_updated_at; Type: TRIGGER; Schema: claims; Owner: -
--

CREATE TRIGGER update_rvu_reference_updated_at BEFORE UPDATE ON claims.rvu_reference FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();


--
-- Name: service_line update_service_line_updated_at; Type: TRIGGER; Schema: claims; Owner: -
--

CREATE TRIGGER update_service_line_updated_at BEFORE UPDATE ON claims.service_line FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();


--
-- Name: encounter validate_encounter_dos; Type: TRIGGER; Schema: claims; Owner: -
--

CREATE TRIGGER validate_encounter_dos BEFORE INSERT OR UPDATE ON claims.encounter FOR EACH ROW EXECUTE FUNCTION public.validate_dos();


--
-- Name: ab_test_experiment update_ab_test_experiment_updated_at; Type: TRIGGER; Schema: ml; Owner: -
--

CREATE TRIGGER update_ab_test_experiment_updated_at BEFORE UPDATE ON ml.ab_test_experiment FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();


--
-- Name: feature_definition update_feature_definition_updated_at; Type: TRIGGER; Schema: ml; Owner: -
--

CREATE TRIGGER update_feature_definition_updated_at BEFORE UPDATE ON ml.feature_definition FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();


--
-- Name: model_registry update_model_registry_updated_at; Type: TRIGGER; Schema: ml; Owner: -
--

CREATE TRIGGER update_model_registry_updated_at BEFORE UPDATE ON ml.model_registry FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();


--
-- Name: training_dataset update_training_dataset_updated_at; Type: TRIGGER; Schema: ml; Owner: -
--

CREATE TRIGGER update_training_dataset_updated_at BEFORE UPDATE ON ml.training_dataset FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();


--
-- Name: file_processing_queue trg_queue_updated_at; Type: TRIGGER; Schema: staging; Owner: -
--

CREATE TRIGGER trg_queue_updated_at BEFORE UPDATE ON staging.file_processing_queue FOR EACH ROW EXECUTE FUNCTION staging.update_queue_updated_at();


--
-- Name: data_refresh_schedule update_data_refresh_schedule_updated_at; Type: TRIGGER; Schema: staging; Owner: -
--

CREATE TRIGGER update_data_refresh_schedule_updated_at BEFORE UPDATE ON staging.data_refresh_schedule FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();


--
-- Name: import_configuration update_import_config_updated_at; Type: TRIGGER; Schema: staging; Owner: -
--

CREATE TRIGGER update_import_config_updated_at BEFORE UPDATE ON staging.import_configuration FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();


--
-- Name: report_subscription update_report_subscription_updated_at; Type: TRIGGER; Schema: staging; Owner: -
--

CREATE TRIGGER update_report_subscription_updated_at BEFORE UPDATE ON staging.report_subscription FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();


--
-- Name: rules_configuration update_rules_config_updated_at; Type: TRIGGER; Schema: staging; Owner: -
--

CREATE TRIGGER update_rules_config_updated_at BEFORE UPDATE ON staging.rules_configuration FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();


--
-- Name: scheduled_job update_scheduled_job_updated_at; Type: TRIGGER; Schema: staging; Owner: -
--

CREATE TRIGGER update_scheduled_job_updated_at BEFORE UPDATE ON staging.scheduled_job FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();


--
-- Name: audit_assignment audit_assignment_facility_id_fkey; Type: FK CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.audit_assignment
    ADD CONSTRAINT audit_assignment_facility_id_fkey FOREIGN KEY (facility_id) REFERENCES claims.facility(facility_id);


--
-- Name: audit_assignment audit_assignment_organization_id_fkey; Type: FK CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.audit_assignment
    ADD CONSTRAINT audit_assignment_organization_id_fkey FOREIGN KEY (organization_id) REFERENCES claims.organization(organization_id);


--
-- Name: audit_assignment audit_assignment_reviewer_id_fkey; Type: FK CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.audit_assignment
    ADD CONSTRAINT audit_assignment_reviewer_id_fkey FOREIGN KEY (reviewer_id) REFERENCES claims.reviewer(reviewer_id);


--
-- Name: audit_encounter audit_encounter_audit_id_fkey; Type: FK CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.audit_encounter
    ADD CONSTRAINT audit_encounter_audit_id_fkey FOREIGN KEY (audit_id) REFERENCES claims.audit_assignment(audit_id) ON DELETE CASCADE;


--
-- Name: audit_encounter audit_encounter_encounter_id_fkey; Type: FK CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.audit_encounter
    ADD CONSTRAINT audit_encounter_encounter_id_fkey FOREIGN KEY (encounter_id) REFERENCES claims.encounter(encounter_id) ON DELETE CASCADE;


--
-- Name: coder_accuracy coder_accuracy_coder_id_fkey; Type: FK CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.coder_accuracy
    ADD CONSTRAINT coder_accuracy_coder_id_fkey FOREIGN KEY (coder_id) REFERENCES claims.coder(coder_id);


--
-- Name: coder_accuracy coder_accuracy_organization_id_fkey; Type: FK CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.coder_accuracy
    ADD CONSTRAINT coder_accuracy_organization_id_fkey FOREIGN KEY (organization_id) REFERENCES claims.organization(organization_id);


--
-- Name: coder coder_organization_id_fkey; Type: FK CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.coder
    ADD CONSTRAINT coder_organization_id_fkey FOREIGN KEY (organization_id) REFERENCES claims.organization(organization_id);


--
-- Name: denial_appeal denial_appeal_denial_id_fkey; Type: FK CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.denial_appeal
    ADD CONSTRAINT denial_appeal_denial_id_fkey FOREIGN KEY (denial_id) REFERENCES claims.denial_event(denial_id) ON DELETE CASCADE;


--
-- Name: denial_event denial_event_coder_id_fkey; Type: FK CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.denial_event
    ADD CONSTRAINT denial_event_coder_id_fkey FOREIGN KEY (coder_id) REFERENCES claims.coder(coder_id);


--
-- Name: denial_event denial_event_encounter_id_fkey; Type: FK CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.denial_event
    ADD CONSTRAINT denial_event_encounter_id_fkey FOREIGN KEY (encounter_id) REFERENCES claims.encounter(encounter_id) ON DELETE CASCADE;


--
-- Name: denial_event denial_event_facility_id_fkey; Type: FK CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.denial_event
    ADD CONSTRAINT denial_event_facility_id_fkey FOREIGN KEY (facility_id) REFERENCES claims.facility(facility_id);


--
-- Name: denial_event denial_event_organization_id_fkey; Type: FK CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.denial_event
    ADD CONSTRAINT denial_event_organization_id_fkey FOREIGN KEY (organization_id) REFERENCES claims.organization(organization_id);


--
-- Name: denial_event denial_event_provider_id_fkey; Type: FK CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.denial_event
    ADD CONSTRAINT denial_event_provider_id_fkey FOREIGN KEY (provider_id) REFERENCES claims.provider(provider_id);


--
-- Name: denial_event denial_event_service_line_id_fkey; Type: FK CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.denial_event
    ADD CONSTRAINT denial_event_service_line_id_fkey FOREIGN KEY (service_line_id) REFERENCES claims.service_line(service_line_id) ON DELETE CASCADE;


--
-- Name: denial_statistics denial_statistics_facility_id_fkey; Type: FK CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.denial_statistics
    ADD CONSTRAINT denial_statistics_facility_id_fkey FOREIGN KEY (facility_id) REFERENCES claims.facility(facility_id);


--
-- Name: denial_statistics denial_statistics_organization_id_fkey; Type: FK CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.denial_statistics
    ADD CONSTRAINT denial_statistics_organization_id_fkey FOREIGN KEY (organization_id) REFERENCES claims.organization(organization_id);


--
-- Name: diagnosis_evaluation diagnosis_evaluation_audit_encounter_id_fkey; Type: FK CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.diagnosis_evaluation
    ADD CONSTRAINT diagnosis_evaluation_audit_encounter_id_fkey FOREIGN KEY (audit_encounter_id) REFERENCES claims.audit_encounter(audit_encounter_id) ON DELETE CASCADE;


--
-- Name: diagnosis_evaluation diagnosis_evaluation_encounter_diagnosis_id_fkey; Type: FK CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.diagnosis_evaluation
    ADD CONSTRAINT diagnosis_evaluation_encounter_diagnosis_id_fkey FOREIGN KEY (encounter_diagnosis_id) REFERENCES claims.encounter_diagnosis(diagnosis_id) ON DELETE CASCADE;


--
-- Name: diagnosis_evaluation diagnosis_evaluation_issue_id_fkey; Type: FK CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.diagnosis_evaluation
    ADD CONSTRAINT diagnosis_evaluation_issue_id_fkey FOREIGN KEY (issue_id) REFERENCES claims.flag_issue(issue_id);


--
-- Name: diagnosis_evaluation diagnosis_evaluation_reviewer_id_fkey; Type: FK CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.diagnosis_evaluation
    ADD CONSTRAINT diagnosis_evaluation_reviewer_id_fkey FOREIGN KEY (reviewer_id) REFERENCES claims.reviewer(reviewer_id);


--
-- Name: encounter encounter_billing_provider_id_fkey; Type: FK CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.encounter
    ADD CONSTRAINT encounter_billing_provider_id_fkey FOREIGN KEY (billing_provider_id) REFERENCES claims.provider(provider_id);


--
-- Name: encounter encounter_coder_id_fkey; Type: FK CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.encounter
    ADD CONSTRAINT encounter_coder_id_fkey FOREIGN KEY (coder_id) REFERENCES claims.coder(coder_id);


--
-- Name: encounter_diagnosis encounter_diagnosis_encounter_id_fkey; Type: FK CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.encounter_diagnosis
    ADD CONSTRAINT encounter_diagnosis_encounter_id_fkey FOREIGN KEY (encounter_id) REFERENCES claims.encounter(encounter_id) ON DELETE CASCADE;


--
-- Name: encounter encounter_facility_id_fkey; Type: FK CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.encounter
    ADD CONSTRAINT encounter_facility_id_fkey FOREIGN KEY (facility_id) REFERENCES claims.facility(facility_id);


--
-- Name: encounter_flag encounter_flag_encounter_id_fkey; Type: FK CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.encounter_flag
    ADD CONSTRAINT encounter_flag_encounter_id_fkey FOREIGN KEY (encounter_id) REFERENCES claims.encounter(encounter_id) ON DELETE CASCADE;


--
-- Name: encounter_flag encounter_flag_issue_id_fkey; Type: FK CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.encounter_flag
    ADD CONSTRAINT encounter_flag_issue_id_fkey FOREIGN KEY (issue_id) REFERENCES claims.flag_issue(issue_id);


--
-- Name: encounter_note encounter_note_encounter_id_fkey; Type: FK CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.encounter_note
    ADD CONSTRAINT encounter_note_encounter_id_fkey FOREIGN KEY (encounter_id) REFERENCES claims.encounter(encounter_id) ON DELETE CASCADE;


--
-- Name: encounter encounter_organization_id_fkey; Type: FK CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.encounter
    ADD CONSTRAINT encounter_organization_id_fkey FOREIGN KEY (organization_id) REFERENCES claims.organization(organization_id);


--
-- Name: encounter encounter_referring_provider_id_fkey; Type: FK CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.encounter
    ADD CONSTRAINT encounter_referring_provider_id_fkey FOREIGN KEY (referring_provider_id) REFERENCES claims.provider(provider_id);


--
-- Name: encounter encounter_region_id_fkey; Type: FK CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.encounter
    ADD CONSTRAINT encounter_region_id_fkey FOREIGN KEY (region_id) REFERENCES claims.region(region_id);


--
-- Name: encounter encounter_rendering_provider_id_fkey; Type: FK CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.encounter
    ADD CONSTRAINT encounter_rendering_provider_id_fkey FOREIGN KEY (rendering_provider_id) REFERENCES claims.provider(provider_id);


--
-- Name: encounter encounter_service_facility_id_fkey; Type: FK CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.encounter
    ADD CONSTRAINT encounter_service_facility_id_fkey FOREIGN KEY (service_facility_id) REFERENCES claims.facility(facility_id);


--
-- Name: encounter encounter_supervising_provider_id_fkey; Type: FK CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.encounter
    ADD CONSTRAINT encounter_supervising_provider_id_fkey FOREIGN KEY (supervising_provider_id) REFERENCES claims.provider(provider_id);


--
-- Name: facility facility_organization_id_fkey; Type: FK CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.facility
    ADD CONSTRAINT facility_organization_id_fkey FOREIGN KEY (organization_id) REFERENCES claims.organization(organization_id) ON DELETE CASCADE;


--
-- Name: facility facility_region_id_fkey; Type: FK CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.facility
    ADD CONSTRAINT facility_region_id_fkey FOREIGN KEY (region_id) REFERENCES claims.region(region_id) ON DELETE SET NULL;


--
-- Name: flag_issue flag_issue_category_id_fkey; Type: FK CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.flag_issue
    ADD CONSTRAINT flag_issue_category_id_fkey FOREIGN KEY (category_id) REFERENCES claims.flag_category(category_id);


--
-- Name: provider_accuracy provider_accuracy_organization_id_fkey; Type: FK CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.provider_accuracy
    ADD CONSTRAINT provider_accuracy_organization_id_fkey FOREIGN KEY (organization_id) REFERENCES claims.organization(organization_id);


--
-- Name: provider_accuracy provider_accuracy_provider_id_fkey; Type: FK CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.provider_accuracy
    ADD CONSTRAINT provider_accuracy_provider_id_fkey FOREIGN KEY (provider_id) REFERENCES claims.provider(provider_id);


--
-- Name: provider provider_organization_id_fkey; Type: FK CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.provider
    ADD CONSTRAINT provider_organization_id_fkey FOREIGN KEY (organization_id) REFERENCES claims.organization(organization_id);


--
-- Name: region region_organization_id_fkey; Type: FK CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.region
    ADD CONSTRAINT region_organization_id_fkey FOREIGN KEY (organization_id) REFERENCES claims.organization(organization_id) ON DELETE CASCADE;


--
-- Name: reviewer reviewer_organization_id_fkey; Type: FK CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.reviewer
    ADD CONSTRAINT reviewer_organization_id_fkey FOREIGN KEY (organization_id) REFERENCES claims.organization(organization_id);


--
-- Name: service_line_adjustment service_line_adjustment_service_line_id_fkey; Type: FK CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.service_line_adjustment
    ADD CONSTRAINT service_line_adjustment_service_line_id_fkey FOREIGN KEY (service_line_id) REFERENCES claims.service_line(service_line_id) ON DELETE CASCADE;


--
-- Name: service_line_diagnosis_pointer service_line_diagnosis_pointer_diagnosis_id_fkey; Type: FK CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.service_line_diagnosis_pointer
    ADD CONSTRAINT service_line_diagnosis_pointer_diagnosis_id_fkey FOREIGN KEY (diagnosis_id) REFERENCES claims.encounter_diagnosis(diagnosis_id) ON DELETE CASCADE;


--
-- Name: service_line_diagnosis_pointer service_line_diagnosis_pointer_service_line_id_fkey; Type: FK CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.service_line_diagnosis_pointer
    ADD CONSTRAINT service_line_diagnosis_pointer_service_line_id_fkey FOREIGN KEY (service_line_id) REFERENCES claims.service_line(service_line_id) ON DELETE CASCADE;


--
-- Name: service_line service_line_encounter_id_fkey; Type: FK CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.service_line
    ADD CONSTRAINT service_line_encounter_id_fkey FOREIGN KEY (encounter_id) REFERENCES claims.encounter(encounter_id) ON DELETE CASCADE;


--
-- Name: service_line_evaluation service_line_evaluation_audit_encounter_id_fkey; Type: FK CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.service_line_evaluation
    ADD CONSTRAINT service_line_evaluation_audit_encounter_id_fkey FOREIGN KEY (audit_encounter_id) REFERENCES claims.audit_encounter(audit_encounter_id) ON DELETE CASCADE;


--
-- Name: service_line_evaluation service_line_evaluation_issue_id_fkey; Type: FK CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.service_line_evaluation
    ADD CONSTRAINT service_line_evaluation_issue_id_fkey FOREIGN KEY (issue_id) REFERENCES claims.flag_issue(issue_id);


--
-- Name: service_line_evaluation service_line_evaluation_reviewer_id_fkey; Type: FK CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.service_line_evaluation
    ADD CONSTRAINT service_line_evaluation_reviewer_id_fkey FOREIGN KEY (reviewer_id) REFERENCES claims.reviewer(reviewer_id);


--
-- Name: service_line_evaluation service_line_evaluation_service_line_id_fkey; Type: FK CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.service_line_evaluation
    ADD CONSTRAINT service_line_evaluation_service_line_id_fkey FOREIGN KEY (service_line_id) REFERENCES claims.service_line(service_line_id) ON DELETE CASCADE;


--
-- Name: service_line_flag service_line_flag_issue_id_fkey; Type: FK CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.service_line_flag
    ADD CONSTRAINT service_line_flag_issue_id_fkey FOREIGN KEY (issue_id) REFERENCES claims.flag_issue(issue_id);


--
-- Name: service_line_flag service_line_flag_service_line_id_fkey; Type: FK CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.service_line_flag
    ADD CONSTRAINT service_line_flag_service_line_id_fkey FOREIGN KEY (service_line_id) REFERENCES claims.service_line(service_line_id) ON DELETE CASCADE;


--
-- Name: service_line service_line_ordering_provider_id_fkey; Type: FK CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.service_line
    ADD CONSTRAINT service_line_ordering_provider_id_fkey FOREIGN KEY (ordering_provider_id) REFERENCES claims.provider(provider_id);


--
-- Name: service_line service_line_referring_provider_id_fkey; Type: FK CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.service_line
    ADD CONSTRAINT service_line_referring_provider_id_fkey FOREIGN KEY (referring_provider_id) REFERENCES claims.provider(provider_id);


--
-- Name: service_line_reimbursement service_line_reimbursement_conversion_factor_id_fkey; Type: FK CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.service_line_reimbursement
    ADD CONSTRAINT service_line_reimbursement_conversion_factor_id_fkey FOREIGN KEY (conversion_factor_id) REFERENCES claims.conversion_factor(conversion_factor_id);


--
-- Name: service_line_reimbursement service_line_reimbursement_gpci_id_fkey; Type: FK CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.service_line_reimbursement
    ADD CONSTRAINT service_line_reimbursement_gpci_id_fkey FOREIGN KEY (gpci_id) REFERENCES claims.gpci_reference(gpci_id);


--
-- Name: service_line_reimbursement service_line_reimbursement_rvu_id_fkey; Type: FK CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.service_line_reimbursement
    ADD CONSTRAINT service_line_reimbursement_rvu_id_fkey FOREIGN KEY (rvu_id) REFERENCES claims.rvu_reference(rvu_id);


--
-- Name: service_line_reimbursement service_line_reimbursement_service_line_id_fkey; Type: FK CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.service_line_reimbursement
    ADD CONSTRAINT service_line_reimbursement_service_line_id_fkey FOREIGN KEY (service_line_id) REFERENCES claims.service_line(service_line_id) ON DELETE CASCADE;


--
-- Name: service_line service_line_rendering_provider_id_fkey; Type: FK CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.service_line
    ADD CONSTRAINT service_line_rendering_provider_id_fkey FOREIGN KEY (rendering_provider_id) REFERENCES claims.provider(provider_id);


--
-- Name: service_line service_line_service_facility_id_fkey; Type: FK CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.service_line
    ADD CONSTRAINT service_line_service_facility_id_fkey FOREIGN KEY (service_facility_id) REFERENCES claims.facility(facility_id);


--
-- Name: service_line service_line_supervising_provider_id_fkey; Type: FK CONSTRAINT; Schema: claims; Owner: -
--

ALTER TABLE ONLY claims.service_line
    ADD CONSTRAINT service_line_supervising_provider_id_fkey FOREIGN KEY (supervising_provider_id) REFERENCES claims.provider(provider_id);


--
-- Name: ab_test_experiment ab_test_experiment_control_model_id_fkey; Type: FK CONSTRAINT; Schema: ml; Owner: -
--

ALTER TABLE ONLY ml.ab_test_experiment
    ADD CONSTRAINT ab_test_experiment_control_model_id_fkey FOREIGN KEY (control_model_id) REFERENCES ml.model_registry(model_id);


--
-- Name: ab_test_experiment ab_test_experiment_organization_id_fkey; Type: FK CONSTRAINT; Schema: ml; Owner: -
--

ALTER TABLE ONLY ml.ab_test_experiment
    ADD CONSTRAINT ab_test_experiment_organization_id_fkey FOREIGN KEY (organization_id) REFERENCES claims.organization(organization_id);


--
-- Name: ab_test_experiment ab_test_experiment_treatment_model_id_fkey; Type: FK CONSTRAINT; Schema: ml; Owner: -
--

ALTER TABLE ONLY ml.ab_test_experiment
    ADD CONSTRAINT ab_test_experiment_treatment_model_id_fkey FOREIGN KEY (treatment_model_id) REFERENCES ml.model_registry(model_id);


--
-- Name: model_performance_log model_performance_log_model_id_fkey; Type: FK CONSTRAINT; Schema: ml; Owner: -
--

ALTER TABLE ONLY ml.model_performance_log
    ADD CONSTRAINT model_performance_log_model_id_fkey FOREIGN KEY (model_id) REFERENCES ml.model_registry(model_id);


--
-- Name: model_prediction model_prediction_encounter_id_fkey; Type: FK CONSTRAINT; Schema: ml; Owner: -
--

ALTER TABLE ONLY ml.model_prediction
    ADD CONSTRAINT model_prediction_encounter_id_fkey FOREIGN KEY (encounter_id) REFERENCES claims.encounter(encounter_id) ON DELETE CASCADE;


--
-- Name: model_prediction model_prediction_model_id_fkey; Type: FK CONSTRAINT; Schema: ml; Owner: -
--

ALTER TABLE ONLY ml.model_prediction
    ADD CONSTRAINT model_prediction_model_id_fkey FOREIGN KEY (model_id) REFERENCES ml.model_registry(model_id);


--
-- Name: model_prediction model_prediction_service_line_id_fkey; Type: FK CONSTRAINT; Schema: ml; Owner: -
--

ALTER TABLE ONLY ml.model_prediction
    ADD CONSTRAINT model_prediction_service_line_id_fkey FOREIGN KEY (service_line_id) REFERENCES claims.service_line(service_line_id) ON DELETE CASCADE;


--
-- Name: model_registry model_registry_organization_id_fkey; Type: FK CONSTRAINT; Schema: ml; Owner: -
--

ALTER TABLE ONLY ml.model_registry
    ADD CONSTRAINT model_registry_organization_id_fkey FOREIGN KEY (organization_id) REFERENCES claims.organization(organization_id);


--
-- Name: training_dataset training_dataset_organization_id_fkey; Type: FK CONSTRAINT; Schema: ml; Owner: -
--

ALTER TABLE ONLY ml.training_dataset
    ADD CONSTRAINT training_dataset_organization_id_fkey FOREIGN KEY (organization_id) REFERENCES claims.organization(organization_id);


--
-- Name: data_refresh_schedule data_refresh_schedule_organization_id_fkey; Type: FK CONSTRAINT; Schema: staging; Owner: -
--

ALTER TABLE ONLY staging.data_refresh_schedule
    ADD CONSTRAINT data_refresh_schedule_organization_id_fkey FOREIGN KEY (organization_id) REFERENCES claims.organization(organization_id);


--
-- Name: file_processing_queue file_processing_queue_facility_id_fkey; Type: FK CONSTRAINT; Schema: staging; Owner: -
--

ALTER TABLE ONLY staging.file_processing_queue
    ADD CONSTRAINT file_processing_queue_facility_id_fkey FOREIGN KEY (facility_id) REFERENCES claims.facility(facility_id);


--
-- Name: file_processing_queue file_processing_queue_import_batch_id_fkey; Type: FK CONSTRAINT; Schema: staging; Owner: -
--

ALTER TABLE ONLY staging.file_processing_queue
    ADD CONSTRAINT file_processing_queue_import_batch_id_fkey FOREIGN KEY (import_batch_id) REFERENCES staging.import_batch(batch_id);


--
-- Name: file_processing_queue file_processing_queue_organization_id_fkey; Type: FK CONSTRAINT; Schema: staging; Owner: -
--

ALTER TABLE ONLY staging.file_processing_queue
    ADD CONSTRAINT file_processing_queue_organization_id_fkey FOREIGN KEY (organization_id) REFERENCES claims.organization(organization_id);


--
-- Name: file_upload file_upload_organization_id_fkey; Type: FK CONSTRAINT; Schema: staging; Owner: -
--

ALTER TABLE ONLY staging.file_upload
    ADD CONSTRAINT file_upload_organization_id_fkey FOREIGN KEY (organization_id) REFERENCES claims.organization(organization_id);


--
-- Name: import_batch import_batch_facility_id_fkey; Type: FK CONSTRAINT; Schema: staging; Owner: -
--

ALTER TABLE ONLY staging.import_batch
    ADD CONSTRAINT import_batch_facility_id_fkey FOREIGN KEY (facility_id) REFERENCES claims.facility(facility_id);


--
-- Name: import_batch import_batch_organization_id_fkey; Type: FK CONSTRAINT; Schema: staging; Owner: -
--

ALTER TABLE ONLY staging.import_batch
    ADD CONSTRAINT import_batch_organization_id_fkey FOREIGN KEY (organization_id) REFERENCES claims.organization(organization_id);


--
-- Name: import_configuration import_configuration_organization_id_fkey; Type: FK CONSTRAINT; Schema: staging; Owner: -
--

ALTER TABLE ONLY staging.import_configuration
    ADD CONSTRAINT import_configuration_organization_id_fkey FOREIGN KEY (organization_id) REFERENCES claims.organization(organization_id);


--
-- Name: import_error_log import_error_log_batch_id_fkey; Type: FK CONSTRAINT; Schema: staging; Owner: -
--

ALTER TABLE ONLY staging.import_error_log
    ADD CONSTRAINT import_error_log_batch_id_fkey FOREIGN KEY (batch_id) REFERENCES staging.import_batch(batch_id) ON DELETE CASCADE;


--
-- Name: job_execution_log job_execution_log_job_id_fkey; Type: FK CONSTRAINT; Schema: staging; Owner: -
--

ALTER TABLE ONLY staging.job_execution_log
    ADD CONSTRAINT job_execution_log_job_id_fkey FOREIGN KEY (job_id) REFERENCES staging.scheduled_job(job_id) ON DELETE CASCADE;


--
-- Name: job_execution_log job_execution_log_original_execution_id_fkey; Type: FK CONSTRAINT; Schema: staging; Owner: -
--

ALTER TABLE ONLY staging.job_execution_log
    ADD CONSTRAINT job_execution_log_original_execution_id_fkey FOREIGN KEY (original_execution_id) REFERENCES staging.job_execution_log(execution_id);


--
-- Name: processing_metrics processing_metrics_batch_id_fkey; Type: FK CONSTRAINT; Schema: staging; Owner: -
--

ALTER TABLE ONLY staging.processing_metrics
    ADD CONSTRAINT processing_metrics_batch_id_fkey FOREIGN KEY (batch_id) REFERENCES staging.import_batch(batch_id) ON DELETE CASCADE;


--
-- Name: report_generation_log report_generation_log_organization_id_fkey; Type: FK CONSTRAINT; Schema: staging; Owner: -
--

ALTER TABLE ONLY staging.report_generation_log
    ADD CONSTRAINT report_generation_log_organization_id_fkey FOREIGN KEY (organization_id) REFERENCES claims.organization(organization_id);


--
-- Name: report_generation_log report_generation_log_subscription_id_fkey; Type: FK CONSTRAINT; Schema: staging; Owner: -
--

ALTER TABLE ONLY staging.report_generation_log
    ADD CONSTRAINT report_generation_log_subscription_id_fkey FOREIGN KEY (subscription_id) REFERENCES staging.report_subscription(subscription_id) ON DELETE SET NULL;


--
-- Name: report_subscription report_subscription_organization_id_fkey; Type: FK CONSTRAINT; Schema: staging; Owner: -
--

ALTER TABLE ONLY staging.report_subscription
    ADD CONSTRAINT report_subscription_organization_id_fkey FOREIGN KEY (organization_id) REFERENCES claims.organization(organization_id);


--
-- Name: rules_configuration rules_configuration_facility_id_fkey; Type: FK CONSTRAINT; Schema: staging; Owner: -
--

ALTER TABLE ONLY staging.rules_configuration
    ADD CONSTRAINT rules_configuration_facility_id_fkey FOREIGN KEY (facility_id) REFERENCES claims.facility(facility_id);


--
-- Name: rules_configuration rules_configuration_flag_issue_id_fkey; Type: FK CONSTRAINT; Schema: staging; Owner: -
--

ALTER TABLE ONLY staging.rules_configuration
    ADD CONSTRAINT rules_configuration_flag_issue_id_fkey FOREIGN KEY (flag_issue_id) REFERENCES claims.flag_issue(issue_id);


--
-- Name: rules_configuration rules_configuration_organization_id_fkey; Type: FK CONSTRAINT; Schema: staging; Owner: -
--

ALTER TABLE ONLY staging.rules_configuration
    ADD CONSTRAINT rules_configuration_organization_id_fkey FOREIGN KEY (organization_id) REFERENCES claims.organization(organization_id);


--
-- Name: scheduled_job scheduled_job_organization_id_fkey; Type: FK CONSTRAINT; Schema: staging; Owner: -
--

ALTER TABLE ONLY staging.scheduled_job
    ADD CONSTRAINT scheduled_job_organization_id_fkey FOREIGN KEY (organization_id) REFERENCES claims.organization(organization_id);


--
-- PostgreSQL database dump complete
--

\unrestrict 9eBTtefqbDVKRlAdhPhpyOZSjCFnruKLYzWN4BATBGLDOrR3yeXwpf0jvHH6eBC

