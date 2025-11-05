# DATABASE SCHEMA REFERENCE

**Professional Smart Claims Processing System**
**Database: PostgreSQL 14+**
**Version: 1.0 (45 Migrations Applied)**
**Generated: 2025-11-05**

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Schema Overview](#schema-overview)
3. [Complete Table Definitions](#complete-table-definitions)
4. [Foreign Key Relationships](#foreign-key-relationships)
5. [Index Strategy](#index-strategy)
6. [Data Types and Design Decisions](#data-types-and-design-decisions)
7. [Performance Considerations](#performance-considerations)
8. [Migration History](#migration-history)

---

## Executive Summary

### Database Statistics

| Metric | Count |
|--------|-------|
| **Total Schemas** | 4 (claims, staging, ml, analytics) |
| **Total Tables** | 60+ |
| **Total Columns** | 800+ |
| **Total Indexes** | 200+ |
| **Total Views** | 20+ |
| **Materialized Views** | 6 |
| **Functions/Procedures** | 15+ |

### Key Design Decisions

#### Identity Column Strategy
- **BIGINT GENERATED ALWAYS AS IDENTITY**: Used for all primary keys
- **CACHE Values**: 50-100 for high-volume tables
- **Rationale**: Sequential integer keys provide optimal B-tree index performance and minimize storage overhead compared to UUIDs

#### Data Type Choices
- **NUMERIC(18,2)**: Financial amounts (supports up to $999,999,999,999,999.99)
- **VARCHAR vs TEXT**: VARCHAR with explicit lengths for structured fields; TEXT for free-form content
- **TIMESTAMPTZ**: All timestamps stored with timezone awareness
- **JSONB**: Used for flexible/dynamic data (API responses, feature data, configurations)
- **citext**: Case-insensitive text for email addresses

#### Foreign Key Cascade Strategy
- **ON DELETE CASCADE**: Used for parent-child relationships where child records are meaningless without parent
- **ON DELETE SET NULL**: Used for optional references (supervising provider, region)
- **ON DELETE RESTRICT**: Used for reference data (taxonomy codes, denial reason codes)

### Performance Characteristics

#### Expected Query Performance (10,000 claims dataset)
- **Full file ingestion**: <15 seconds
- **Encounter lookup by ID**: <1ms (indexed)
- **Provider search by NPI**: <2ms (indexed + unique)
- **Flag queries by organization**: <50ms (composite index)
- **Dashboard view refresh**: <500ms (materialized views)
- **Audit assignment queries**: <100ms (optimized indexes)

#### Scalability Targets
- **Encounters**: 10M+ records
- **Service Lines**: 50M+ records
- **Concurrent Users**: 50+
- **Import Throughput**: 750 claims/second/worker
- **FIFO Processing**: Strict chronological order guaranteed

---

## Schema Overview

### Schema: `claims`
**Purpose**: Core claims processing data including encounters, providers, diagnoses, procedures, flags, audits, and denials

**Table Count**: 35 tables

**Key Tables**:
- Organization hierarchy (organization, region, facility)
- Provider management (provider, coder, reviewer, provider_taxonomy, provider_enrichment_queue)
- Encounter/claim data (encounter, encounter_note, encounter_diagnosis)
- Service line data (service_line, service_line_adjustment, service_line_diagnosis_pointer)
- Flag management (flag_category, flag_issue, encounter_flag, service_line_flag)
- Audit system (audit_assignment, audit_encounter, service_line_evaluation, diagnosis_evaluation, coder_accuracy, provider_accuracy)
- Denial tracking (denial_event, denial_appeal, denial_reason_code, denial_statistics)
- Reference data (rvu_reference, conversion_factor, gpci_reference, modifier_adjustment, field_definitions, import_headers)
- Reimbursement (service_line_reimbursement)

---

### Schema: `staging`
**Purpose**: File import pipeline, batch processing, configuration, and temporary storage

**Table Count**: 15 tables

**Key Tables**:
- File processing (import_batch, file_upload, file_processing_queue, file_processing_progress)
- Raw data staging (raw_claims, failed_claims)
- Configuration (import_configuration, rules_configuration, processing_configuration)
- Sequence tracking (batch_sequences)
- Metrics/monitoring (processing_metrics, import_error_log)
- Version control (schema_migrations, application_version)

**Purpose**: Decouples file ingestion from claim validation in a two-stage pipeline:
1. **Stage 1 (Ingestion)**: Files → raw_claims (fast, no validation)
2. **Stage 2 (Processing)**: raw_claims → encounters/errors (validated, with rules)

---

### Schema: `ml`
**Purpose**: Machine learning models, predictions, features, and training datasets

**Table Count**: 6 tables

**Key Tables**:
- Model management (model_registry, model_performance_log, ab_test_experiment)
- Predictions (model_prediction)
- Feature engineering (feature_definition, training_dataset)

**Integration**: Predictive analytics for denial risk, coding error detection, audit priority scoring

---

### Schema: `analytics`
**Purpose**: Pre-aggregated statistics and materialized views for dashboard performance

**Table Count**: 6 materialized views

**Key Views**:
- flag_statistics_daily
- encounter_statistics_daily
- procedure_statistics
- provider_performance
- payer_statistics
- ml_model_performance_summary

**Refresh Strategy**: CONCURRENTLY (no table locking), scheduled refresh via `analytics.refresh_all_views()`

---

## Complete Table Definitions

### Claims Schema Tables

#### claims.organization

**Purpose**: Top-level organization entities

```sql
CREATE TABLE claims.organization (
    organization_id BIGINT GENERATED ALWAYS AS IDENTITY (CACHE 50) PRIMARY KEY,
    organization_code VARCHAR(50) NOT NULL UNIQUE,
    organization_name VARCHAR(255) NOT NULL,
    tax_id VARCHAR(20),
    npi VARCHAR(10),
    address_line1 VARCHAR(255),
    address_line2 VARCHAR(255),
    city VARCHAR(100),
    state_code CHAR(2),
    postal_code VARCHAR(15),
    country_code CHAR(3) DEFAULT 'USA',
    phone VARCHAR(20),
    email citext,
    project_id BIGINT,  -- Added in migration 028
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100),
    updated_by VARCHAR(100)
);
```

**Indexes**:
- PRIMARY KEY: `organization_id`
- UNIQUE: `organization_code`
- `idx_organization_code` ON (organization_code)
- `idx_organization_active` ON (is_active) WHERE is_active = true

**Triggers**:
- `update_organization_updated_at`: Auto-update `updated_at` timestamp

---

#### claims.region

**Purpose**: Regional divisions within organizations (optional level)

```sql
CREATE TABLE claims.region (
    region_id BIGINT GENERATED ALWAYS AS IDENTITY (CACHE 50) PRIMARY KEY,
    organization_id BIGINT NOT NULL REFERENCES claims.organization(organization_id) ON DELETE CASCADE,
    region_code VARCHAR(50) NOT NULL,
    region_name VARCHAR(255) NOT NULL,
    description TEXT,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100),
    updated_by VARCHAR(100),
    UNIQUE(organization_id, region_code)
);
```

**Foreign Keys**:
- `organization_id` → `claims.organization(organization_id)` ON DELETE CASCADE

**Indexes**:
- PRIMARY KEY: `region_id`
- UNIQUE: `(organization_id, region_code)`
- `idx_region_org` ON (organization_id)
- `idx_region_code` ON (region_code)
- `idx_region_active` ON (is_active) WHERE is_active = true

**Triggers**:
- `update_region_updated_at`: Auto-update `updated_at` timestamp

---

#### claims.facility

**Purpose**: Facility entities - must belong to organization, optionally to region

```sql
CREATE TABLE claims.facility (
    facility_id BIGINT GENERATED ALWAYS AS IDENTITY (CACHE 50) PRIMARY KEY,
    organization_id BIGINT NOT NULL REFERENCES claims.organization(organization_id) ON DELETE CASCADE,
    region_id BIGINT REFERENCES claims.region(region_id) ON DELETE SET NULL,
    facility_code VARCHAR(50) NOT NULL,
    facility_name VARCHAR(255) NOT NULL,
    npi VARCHAR(10),
    tax_id VARCHAR(20),
    facility_type VARCHAR(50),
    address_line1 VARCHAR(255),
    address_line2 VARCHAR(255),
    city VARCHAR(100),
    state_code CHAR(2),
    postal_code VARCHAR(15),
    country_code CHAR(3) DEFAULT 'USA',
    phone VARCHAR(20),
    email citext,
    ehr_system VARCHAR(100),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100),
    updated_by VARCHAR(100),
    UNIQUE(organization_id, facility_code)
);
```

**Foreign Keys**:
- `organization_id` → `claims.organization(organization_id)` ON DELETE CASCADE
- `region_id` → `claims.region(region_id)` ON DELETE SET NULL

**Indexes**:
- PRIMARY KEY: `facility_id`
- UNIQUE: `(organization_id, facility_code)`
- `idx_facility_org` ON (organization_id)
- `idx_facility_region` ON (region_id)
- `idx_facility_code` ON (facility_code)
- `idx_facility_npi` ON (npi)
- `idx_facility_active` ON (is_active) WHERE is_active = true
- `idx_facility_single_region` UNIQUE ON (facility_id) WHERE region_id IS NOT NULL
- `idx_facility_name_trgm` ON (facility_name) USING gin (facility_name gin_trgm_ops)

**Triggers**:
- `update_facility_updated_at`: Auto-update `updated_at` timestamp

**Constraints**:
- Facility can only be in one region

---

#### claims.provider

**Purpose**: Healthcare providers (physicians, practitioners)

```sql
CREATE TABLE claims.provider (
    provider_id BIGINT GENERATED ALWAYS AS IDENTITY (CACHE 100) PRIMARY KEY,
    npi VARCHAR(10) NOT NULL UNIQUE,
    provider_type VARCHAR(50) NOT NULL,  -- Billing, Rendering, Referring, Supervising, Ordering
    last_name VARCHAR(255) NOT NULL,
    first_name VARCHAR(255) NOT NULL,
    middle_name VARCHAR(255),
    name_suffix VARCHAR(50),
    full_name VARCHAR(512),  -- Added in migration 034
    taxonomy_code VARCHAR(10) REFERENCES claims.provider_taxonomy(taxonomy_code) ON DELETE RESTRICT ON UPDATE CASCADE,  -- Added FK in migration 044
    license_number VARCHAR(50),
    license_state CHAR(2),
    specialty VARCHAR(100),
    provider_group VARCHAR(255),
    organization_id BIGINT REFERENCES claims.organization(organization_id),
    address_line1 VARCHAR(255),
    address_line2 VARCHAR(255),
    city VARCHAR(100),
    state_code CHAR(2),
    postal_code VARCHAR(15),
    country_code CHAR(3) DEFAULT 'USA',
    phone VARCHAR(20),
    email citext,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100),
    updated_by VARCHAR(100)
);
```

**Foreign Keys**:
- `organization_id` → `claims.organization(organization_id)`
- `taxonomy_code` → `claims.provider_taxonomy(taxonomy_code)` ON DELETE RESTRICT ON UPDATE CASCADE

**Indexes**:
- PRIMARY KEY: `provider_id`
- UNIQUE: `npi`
- `idx_provider_npi` ON (npi)
- `idx_provider_type` ON (provider_type)
- `idx_provider_specialty` ON (specialty)
- `idx_provider_org` ON (organization_id)
- `idx_provider_active` ON (is_active) WHERE is_active = true
- `idx_provider_name` ON (last_name, first_name)
- `idx_provider_last_name_trgm` ON (last_name) USING gin (last_name gin_trgm_ops)
- `idx_provider_npi_lookup` ON (npi) WHERE is_active = true AND npi IS NOT NULL (migration 016)
- `idx_provider_specialty_type` ON (specialty, provider_type) WHERE is_active = true AND specialty IS NOT NULL (migration 016)
- `idx_provider_taxonomy_code` ON (taxonomy_code) (migration 044)

**Triggers**:
- `update_provider_updated_at`: Auto-update `updated_at` timestamp

---

#### claims.provider_taxonomy

**Purpose**: NUCC Healthcare Provider Taxonomy code set - maps taxonomy codes to specialty display names

```sql
CREATE TABLE claims.provider_taxonomy (
    taxonomy_code VARCHAR(10) PRIMARY KEY,
    provider_type VARCHAR(100) NOT NULL,        -- Individual, Organization
    classification VARCHAR(200) NOT NULL,       -- e.g., "Allopathic & Osteopathic Physicians"
    specialization VARCHAR(200),                -- e.g., "Family Medicine"
    specialty_display VARCHAR(200) NOT NULL,    -- User-friendly display name
    definition TEXT,                            -- Official NUCC definition
    is_active BOOLEAN DEFAULT true,
    effective_date DATE DEFAULT '2024-01-01',
    notes TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
```

**Indexes**:
- PRIMARY KEY: `taxonomy_code`
- `idx_taxonomy_specialty` ON (specialty_display)
- `idx_taxonomy_classification` ON (classification)
- `idx_taxonomy_active` ON (is_active) WHERE is_active = true

**Data**: Pre-populated with 300+ NUCC taxonomy codes covering all major specialties (Family Medicine, Internal Medicine, Pediatrics, Surgery, Orthopedics, Emergency Medicine, Anesthesiology, Radiology, Pathology, Psychiatry, etc.)

---

#### claims.provider_enrichment_queue

**Purpose**: Queue for asynchronous provider data enrichment from CMS NPI Registry API

```sql
CREATE TABLE claims.provider_enrichment_queue (
    queue_id BIGINT GENERATED ALWAYS AS IDENTITY (CACHE 100) PRIMARY KEY,
    provider_id BIGINT NOT NULL REFERENCES claims.provider(provider_id) ON DELETE CASCADE,
    npi VARCHAR(10) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'PENDING',  -- PENDING, IN_PROGRESS, COMPLETED, FAILED
    priority INTEGER NOT NULL DEFAULT 5,  -- 1-10 (10=highest priority)
    retry_count INTEGER NOT NULL DEFAULT 0,
    max_retries INTEGER NOT NULL DEFAULT 3,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    next_retry_at TIMESTAMPTZ,
    last_error TEXT,
    last_error_at TIMESTAMPTZ,
    api_response JSONB,  -- Full NPI Registry API response for audit trail
    CONSTRAINT unique_provider_enrichment UNIQUE(provider_id),
    CONSTRAINT valid_status CHECK (status IN ('PENDING', 'IN_PROGRESS', 'COMPLETED', 'FAILED')),
    CONSTRAINT valid_priority CHECK (priority BETWEEN 1 AND 10)
);
```

**Foreign Keys**:
- `provider_id` → `claims.provider(provider_id)` ON DELETE CASCADE

**Indexes**:
- PRIMARY KEY: `queue_id`
- UNIQUE: `provider_id`
- `idx_enrichment_status_pending` ON (status, priority DESC, created_at ASC) WHERE status IN ('PENDING', 'FAILED')
- `idx_enrichment_retry` ON (next_retry_at, priority DESC) WHERE status = 'FAILED' AND next_retry_at IS NOT NULL
- `idx_enrichment_npi` ON (npi)
- `idx_enrichment_completed` ON (completed_at DESC) WHERE status = 'COMPLETED'
- `idx_enrichment_in_progress` ON (started_at DESC) WHERE status = 'IN_PROGRESS'

**Triggers**:
- `trigger_provider_enrichment_completed`: Auto-updates provider.updated_at when enrichment completes

**Purpose**: Ensures claims processing is never blocked by external API calls - providers are enriched asynchronously after initial import

---

#### claims.coder

**Purpose**: Medical coders who code/bill claims

```sql
CREATE TABLE claims.coder (
    coder_id BIGINT GENERATED ALWAYS AS IDENTITY (CACHE 100) PRIMARY KEY,
    coder_code VARCHAR(50) NOT NULL UNIQUE,
    last_name VARCHAR(255) NOT NULL,
    first_name VARCHAR(255) NOT NULL,
    middle_name VARCHAR(255),
    coder_group VARCHAR(100),
    certifications TEXT[],  -- Array of certifications (CPC, CCS, etc.)
    organization_id BIGINT REFERENCES claims.organization(organization_id),
    email citext,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100),
    updated_by VARCHAR(100)
);
```

**Foreign Keys**:
- `organization_id` → `claims.organization(organization_id)`

**Indexes**:
- PRIMARY KEY: `coder_id`
- UNIQUE: `coder_code`
- `idx_coder_code` ON (coder_code)
- `idx_coder_group` ON (coder_group)
- `idx_coder_org` ON (organization_id)
- `idx_coder_active` ON (is_active) WHERE is_active = true
- `idx_coder_name` ON (last_name, first_name)

**Triggers**:
- `update_coder_updated_at`: Auto-update `updated_at` timestamp

---

#### claims.reviewer

**Purpose**: Audit reviewers who perform retrospective reviews

```sql
CREATE TABLE claims.reviewer (
    reviewer_id BIGINT GENERATED ALWAYS AS IDENTITY (CACHE 100) PRIMARY KEY,
    reviewer_code VARCHAR(50) NOT NULL UNIQUE,
    last_name VARCHAR(255) NOT NULL,
    first_name VARCHAR(255) NOT NULL,
    middle_name VARCHAR(255),
    reviewer_group VARCHAR(100),
    certifications TEXT[],  -- Array of certifications
    organization_id BIGINT REFERENCES claims.organization(organization_id),
    email citext,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100),
    updated_by VARCHAR(100)
);
```

**Foreign Keys**:
- `organization_id` → `claims.organization(organization_id)`

**Indexes**:
- PRIMARY KEY: `reviewer_id`
- UNIQUE: `reviewer_code`
- `idx_reviewer_code` ON (reviewer_code)
- `idx_reviewer_group` ON (reviewer_group)
- `idx_reviewer_org` ON (organization_id)
- `idx_reviewer_active` ON (is_active) WHERE is_active = true
- `idx_reviewer_name` ON (last_name, first_name)

**Triggers**:
- `update_reviewer_updated_at`: Auto-update `updated_at` timestamp

---

#### claims.encounter

**Purpose**: Main encounter/claim table containing all 837p claim-level data elements

```sql
CREATE TABLE claims.encounter (
    encounter_id BIGINT GENERATED ALWAYS AS IDENTITY (CACHE 100) PRIMARY KEY,

    -- Organizational references
    facility_id BIGINT NOT NULL REFERENCES claims.facility(facility_id),
    organization_id BIGINT NOT NULL REFERENCES claims.organization(organization_id),
    region_id BIGINT REFERENCES claims.region(region_id),

    -- Submitter information (Loop 1000A)
    submitter_id VARCHAR(80) NOT NULL,
    submitter_name VARCHAR(255),

    -- Control numbers
    patient_control_number VARCHAR(38) NOT NULL,  -- CLM01 (only 20 chars stored/returned)
    transaction_set_control_number VARCHAR(9),

    -- Patient/Subscriber information (Loop 2000B/2010BA)
    subscriber_id VARCHAR(80) NOT NULL,  -- Medicare Beneficiary Identifier (MBI)
    subscriber_last_name VARCHAR(255) NOT NULL,
    subscriber_first_name VARCHAR(255) NOT NULL,
    subscriber_middle_name VARCHAR(255),
    subscriber_name_suffix VARCHAR(50),
    subscriber_gender CHAR(1),  -- M, F, U
    subscriber_birth_date DATE NOT NULL,
    subscriber_address_line1 VARCHAR(255),
    subscriber_address_line2 VARCHAR(255),
    subscriber_city VARCHAR(100),
    subscriber_state CHAR(2),
    subscriber_postal_code VARCHAR(15),
    subscriber_country CHAR(3) DEFAULT 'USA',
    medical_record_number VARCHAR(50),  -- Added in migration 035

    -- Payer information (Loop 2010BB)
    payer_responsibility_code CHAR(1) NOT NULL,  -- P (Primary) or S (Secondary)
    payer_id VARCHAR(80),
    payer_name VARCHAR(255),
    claim_filing_indicator VARCHAR(2) DEFAULT 'MB',  -- Medicare Part B

    -- Billing provider (Loop 2010AA)
    billing_provider_id BIGINT REFERENCES claims.provider(provider_id),
    billing_provider_npi VARCHAR(10),
    billing_provider_tax_id VARCHAR(20),
    billing_provider_name VARCHAR(255),
    billing_provider_address_line1 VARCHAR(255),
    billing_provider_address_line2 VARCHAR(255),
    billing_provider_city VARCHAR(100),
    billing_provider_state CHAR(2),
    billing_provider_postal_code VARCHAR(15),

    -- Claim information (Loop 2300 CLM)
    total_claim_charge_amount NUMERIC(18,2) NOT NULL,
    place_of_service_code VARCHAR(2),
    claim_frequency_code CHAR(1) DEFAULT '1',  -- 1 = Original
    signature_indicator CHAR(1),
    assignment_indicator CHAR(1),
    benefits_assignment_indicator CHAR(1),
    release_of_information_code CHAR(1),
    patient_signature_code CHAR(1),

    -- Dates (Loop 2300 DTP)
    date_of_service_from DATE NOT NULL,
    date_of_service_to DATE,
    onset_of_illness_date DATE,
    initial_treatment_date DATE,
    last_seen_date DATE,
    acute_manifestation_date DATE,
    accident_date DATE,
    last_menstrual_period_date DATE,
    last_xray_date DATE,
    prescription_date DATE,
    disability_from_date DATE,
    disability_to_date DATE,
    last_worked_date DATE,
    authorized_return_to_work_date DATE,
    admission_date DATE,
    discharge_date DATE,
    assumed_care_date DATE,
    relinquished_care_date DATE,

    -- Additional claim information
    delay_reason_code VARCHAR(2),
    special_program_code VARCHAR(3),
    patient_amount_paid NUMERIC(18,2),
    service_authorization_code VARCHAR(50),

    -- Referring provider (Loop 2310A)
    referring_provider_id BIGINT REFERENCES claims.provider(provider_id),
    referring_provider_npi VARCHAR(10),
    referring_provider_name VARCHAR(255),

    -- Rendering provider (Loop 2310B)
    rendering_provider_id BIGINT REFERENCES claims.provider(provider_id),
    rendering_provider_npi VARCHAR(10),
    rendering_provider_name VARCHAR(255),

    -- Service facility (Loop 2310C)
    service_facility_id BIGINT REFERENCES claims.facility(facility_id),
    service_facility_npi VARCHAR(10),
    service_facility_name VARCHAR(255),
    service_facility_address_line1 VARCHAR(255),
    service_facility_address_line2 VARCHAR(255),
    service_facility_city VARCHAR(100),
    service_facility_state CHAR(2),
    service_facility_postal_code VARCHAR(15),

    -- Supervising provider (Loop 2310D)
    supervising_provider_id BIGINT REFERENCES claims.provider(provider_id),
    supervising_provider_npi VARCHAR(10),
    supervising_provider_name VARCHAR(255),

    -- Other payer information (Loop 2320 for COB)
    other_payer_paid_amount NUMERIC(18,2),
    other_payer_id VARCHAR(80),
    other_payer_name VARCHAR(255),
    other_payer_claim_number VARCHAR(50),
    other_payer_claim_filing_indicator VARCHAR(2),

    -- Ambulance information (Loop 2300 CR1)
    ambulance_transport_reason_code CHAR(1),
    ambulance_transport_distance NUMERIC(15,4),
    ambulance_patient_weight NUMERIC(10,2),
    ambulance_patient_count INTEGER,

    -- Coder/billing information
    coder_id BIGINT REFERENCES claims.coder(coder_id),
    coding_date DATE,

    -- Status and workflow
    claim_status VARCHAR(50) DEFAULT 'NEW',  -- NEW, PENDING, FLAGGED, REVIEWED, ACCEPTED, REJECTED
    case_status VARCHAR(50),
    financial_class VARCHAR(50),

    -- Import tracking
    import_batch_id BIGINT REFERENCES staging.import_batch(batch_id) ON DELETE SET NULL ON UPDATE CASCADE,  -- FK added in migration 045
    import_date TIMESTAMPTZ,
    import_configuration_id BIGINT REFERENCES staging.import_configuration(configuration_id) ON DELETE SET NULL ON UPDATE CASCADE,  -- FK added in migration 045

    -- Audit trail
    is_active BOOLEAN DEFAULT true,
    soft_deleted BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100),
    updated_by VARCHAR(100),

    -- Constraints
    CONSTRAINT chk_dos_range CHECK (date_of_service_to IS NULL OR date_of_service_to >= date_of_service_from),
    CONSTRAINT chk_payer_responsibility CHECK (payer_responsibility_code IN ('P', 'S'))
);
```

**Foreign Keys**:
- `facility_id` → `claims.facility(facility_id)` (required)
- `organization_id` → `claims.organization(organization_id)` (required)
- `region_id` → `claims.region(region_id)` (optional)
- `billing_provider_id` → `claims.provider(provider_id)` (optional)
- `referring_provider_id` → `claims.provider(provider_id)` (optional)
- `rendering_provider_id` → `claims.provider(provider_id)` (optional)
- `service_facility_id` → `claims.facility(facility_id)` (optional)
- `supervising_provider_id` → `claims.provider(provider_id)` (optional, added index in migration 043)
- `coder_id` → `claims.coder(coder_id)` (optional)
- `import_batch_id` → `staging.import_batch(batch_id)` ON DELETE SET NULL (optional, FK added in migration 045)
- `import_configuration_id` → `staging.import_configuration(configuration_id)` ON DELETE SET NULL (optional, FK added in migration 045)

**Indexes** (optimized for 10,000 claims / 15 seconds requirement):
- PRIMARY KEY: `encounter_id`
- `idx_encounter_facility` ON (facility_id)
- `idx_encounter_organization` ON (organization_id)
- `idx_encounter_region` ON (region_id) WHERE region_id IS NOT NULL (migration 043)
- `idx_encounter_patient_control` ON (patient_control_number)
- `idx_encounter_subscriber` ON (subscriber_id)
- `idx_encounter_dos_from` ON (date_of_service_from)
- `idx_encounter_dos_to` ON (date_of_service_to)
- `idx_encounter_dos_range` ON (date_of_service_from, date_of_service_to)
- `idx_encounter_billing_provider` ON (billing_provider_id)
- `idx_encounter_rendering_provider` ON (rendering_provider_id)
- `idx_encounter_referring_provider` ON (referring_provider_id)
- `idx_encounter_supervising_provider` ON (supervising_provider_id) WHERE supervising_provider_id IS NOT NULL (migration 043)
- `idx_encounter_service_facility` ON (service_facility_id) WHERE service_facility_id IS NOT NULL (migration 043)
- `idx_encounter_coder` ON (coder_id)
- `idx_encounter_coding_date` ON (coding_date)
- `idx_encounter_status` ON (claim_status)
- `idx_encounter_import_batch` ON (import_batch_id)
- `idx_encounter_import_date` ON (import_date)
- `idx_encounter_active` ON (is_active) WHERE is_active = true
- `idx_encounter_not_deleted` ON (soft_deleted) WHERE soft_deleted = false
- `idx_encounter_created_at` ON (created_at)
- `idx_encounter_org_dos` ON (organization_id, date_of_service_from)
- `idx_encounter_facility_dos` ON (facility_id, date_of_service_from)
- `idx_encounter_provider_dos` ON (billing_provider_id, date_of_service_from)
- `idx_encounter_status_dos` ON (claim_status, date_of_service_from)
- `idx_encounter_needs_review` ON (encounter_id, claim_status) WHERE claim_status IN ('PENDING', 'FLAGGED')
- `idx_encounter_patient_control_trgm` ON (patient_control_number) USING gin (patient_control_number gin_trgm_ops)
- `idx_encounter_subscriber_last_name_trgm` ON (subscriber_last_name) USING gin (subscriber_last_name gin_trgm_ops)
- `idx_encounter_org_facility_dos` ON (organization_id, facility_id, date_of_service_from) (migration 014)
- `idx_encounter_provider_status_dos` ON (rendering_provider_id, claim_status, date_of_service_from) WHERE claim_status IN ('PENDING', 'FLAGGED', 'NEW') (migration 014)
- `idx_encounter_subscriber_history` ON (subscriber_id, date_of_service_from DESC) WHERE is_active = true AND soft_deleted = false (migration 016)
- `idx_encounter_status_date` ON (claim_status, date_of_service_from DESC, organization_id) WHERE is_active = true AND soft_deleted = false (migration 016)
- `idx_encounter_facility_date` ON (facility_id, date_of_service_from DESC) WHERE is_active = true AND soft_deleted = false (migration 016)
- `idx_encounter_service_date_facility` ON (facility_id, date_of_service_from ASC, import_date ASC) WHERE is_active = true (migration 015)
- `idx_encounter_import_date_facility` ON (facility_id, import_date DESC, date_of_service_from DESC) WHERE is_active = true (migration 015)
- `idx_encounter_org_dos_status` ON (organization_id, date_of_service_from DESC, claim_status) WHERE is_active = true AND soft_deleted = false (migration 018)
- `idx_encounter_active_created` ON (created_at DESC) WHERE is_active = true AND soft_deleted = false (migration 018)
- `idx_encounter_subscriber_dos` ON (subscriber_id, date_of_service_from DESC) WHERE is_active = true AND soft_deleted = false (migration 018)

**Triggers**:
- `update_encounter_updated_at`: Auto-update `updated_at` timestamp
- `validate_encounter_dos`: Validates date of service is not in future
- `sync_encounter_totals_insert/update/delete`: Auto-calculates total_claim_charge_amount from service lines (migration 014)

**Check Constraints**:
- Date of service to >= date of service from
- Payer responsibility code in ('P', 'S')

**Design Notes**:
- Comprehensive 837p claim data model covering all loops and segments
- Denormalized provider information (NPI + name) for performance
- Soft delete support (is_active, soft_deleted flags)
- Extensive indexing strategy optimized for common query patterns
- Check constraint removed from total_claim_charge_amount (migration 029) to allow flexible financial scenarios

---

#### claims.encounter_note

**Purpose**: Notes and comments associated with encounters

```sql
CREATE TABLE claims.encounter_note (
    note_id BIGINT GENERATED ALWAYS AS IDENTITY (CACHE 100) PRIMARY KEY,
    encounter_id BIGINT NOT NULL REFERENCES claims.encounter(encounter_id) ON DELETE CASCADE,
    note_type VARCHAR(50),  -- GENERAL, AUDIT, BILLING, etc.
    note_text TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100)
);
```

**Foreign Keys**:
- `encounter_id` → `claims.encounter(encounter_id)` ON DELETE CASCADE

**Indexes**:
- PRIMARY KEY: `note_id`
- `idx_encounter_note_encounter` ON (encounter_id)
- `idx_encounter_note_type` ON (note_type)
- `idx_encounter_note_created` ON (created_at)

---

#### claims.encounter_diagnosis

**Purpose**: Diagnosis codes associated with encounters (ICD-10-CM)

```sql
CREATE TABLE claims.encounter_diagnosis (
    diagnosis_id BIGINT GENERATED ALWAYS AS IDENTITY (CACHE 100) PRIMARY KEY,
    encounter_id BIGINT NOT NULL REFERENCES claims.encounter(encounter_id) ON DELETE CASCADE,
    sequence_number SMALLINT NOT NULL,  -- 1-12 for principal and secondary diagnoses
    diagnosis_code_qualifier VARCHAR(3) DEFAULT 'ABK',  -- ABK for ICD-10-CM
    diagnosis_code VARCHAR(30) NOT NULL,
    diagnosis_description TEXT,
    is_principal BOOLEAN DEFAULT false,  -- True for first/principal diagnosis
    is_admitting BOOLEAN DEFAULT false,
    is_external_cause BOOLEAN DEFAULT false,
    is_patient_reason BOOLEAN DEFAULT false,
    present_on_admission_indicator CHAR(1),  -- Y, N, U, W, or null
    hcc_indicator BOOLEAN DEFAULT false,
    hcc_category VARCHAR(10),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT chk_sequence_range CHECK (sequence_number BETWEEN 1 AND 12),
    CONSTRAINT uk_encounter_diagnosis_seq UNIQUE (encounter_id, sequence_number)
);
```

**Foreign Keys**:
- `encounter_id` → `claims.encounter(encounter_id)` ON DELETE CASCADE

**Indexes**:
- PRIMARY KEY: `diagnosis_id`
- UNIQUE: `(encounter_id, sequence_number)`
- `idx_enc_diag_encounter` ON (encounter_id)
- `idx_enc_diag_code` ON (diagnosis_code)
- `idx_enc_diag_principal` ON (encounter_id, is_principal) WHERE is_principal = true
- `idx_enc_diag_hcc` ON (hcc_indicator, hcc_category) WHERE hcc_indicator = true
- `idx_encounter_diagnosis_code` ON (diagnosis_code, encounter_id) (migration 016)
- `idx_encounter_diagnosis_sequence` ON (encounter_id, sequence_number, is_principal) WHERE is_principal = true (migration 016)
- `idx_encounter_diagnosis_principal` ON (encounter_id, sequence_number) WHERE is_principal = true (migration 018)

**Check Constraints**:
- Sequence number between 1 and 12

---

#### claims.service_line

**Purpose**: Service line items (procedures) for encounters - Loop 2400 data

```sql
CREATE TABLE claims.service_line (
    service_line_id BIGINT GENERATED ALWAYS AS IDENTITY (CACHE 100) PRIMARY KEY,
    encounter_id BIGINT NOT NULL REFERENCES claims.encounter(encounter_id) ON DELETE CASCADE,
    line_number SMALLINT NOT NULL,  -- Service line number

    -- Service information (Loop 2400 SV1)
    product_service_id_qualifier VARCHAR(2) DEFAULT 'HC',  -- HC for HCPCS
    procedure_code VARCHAR(48) NOT NULL,  -- CPT/HCPCS code
    procedure_modifier_1 VARCHAR(2),
    procedure_modifier_2 VARCHAR(2),
    procedure_modifier_3 VARCHAR(2),
    procedure_modifier_4 VARCHAR(2),
    procedure_description TEXT,
    line_item_charge_amount NUMERIC(18,2) NOT NULL,
    unit_basis_measurement_code VARCHAR(2) DEFAULT 'UN',  -- UN for units, MJ for minutes
    service_unit_count NUMERIC(15,1) NOT NULL CHECK (service_unit_count > 0 AND service_unit_count <= 9999.9),

    -- Place of service
    place_of_service_code VARCHAR(2),

    -- Dates
    service_date_from DATE NOT NULL,
    service_date_to DATE,

    -- Emergency indicator
    emergency_indicator BOOLEAN DEFAULT false,

    -- EPSDT indicator
    epsdt_indicator BOOLEAN DEFAULT false,

    -- Family planning indicator
    family_planning_indicator BOOLEAN DEFAULT false,

    -- Rendering provider at line level (Loop 2420A)
    rendering_provider_id BIGINT REFERENCES claims.provider(provider_id),
    rendering_provider_npi VARCHAR(10),

    -- Supervising provider at line level (Loop 2420D)
    supervising_provider_id BIGINT REFERENCES claims.provider(provider_id),
    supervising_provider_npi VARCHAR(10),

    -- Ordering provider at line level (Loop 2420E)
    ordering_provider_id BIGINT REFERENCES claims.provider(provider_id),
    ordering_provider_npi VARCHAR(10),

    -- Referring provider at line level (Loop 2420F)
    referring_provider_id BIGINT REFERENCES claims.provider(provider_id),
    referring_provider_npi VARCHAR(10),

    -- Service facility at line level (Loop 2420C)
    service_facility_id BIGINT REFERENCES claims.facility(facility_id),
    service_facility_npi VARCHAR(10),

    -- Prior authorization
    prior_authorization_number VARCHAR(50),

    -- Referral number
    referral_number VARCHAR(50),

    -- Line note/description
    line_note TEXT,

    -- Revenue code (for institutional claims, may be present on professional)
    revenue_code VARCHAR(4),

    -- NDC information (Loop 2410 for drugs)
    ndc_code VARCHAR(11),
    ndc_unit_count NUMERIC(15,3),
    ndc_measurement_unit VARCHAR(2),

    -- DME information
    dme_rental_price NUMERIC(18,2),
    dme_purchase_price NUMERIC(18,2),
    dme_frequency_code VARCHAR(1),

    -- Anesthesia information
    anesthesia_minutes INTEGER,
    obstetric_additional_units INTEGER,

    -- Test results (Loop 2400 MEA)
    test_result_value NUMERIC(20,1),
    test_result_measurement_code VARCHAR(20),

    -- Ambulance information (line level)
    ambulance_patient_count INTEGER,
    ambulance_transport_distance NUMERIC(15,4),
    ambulance_patient_weight NUMERIC(10,2),

    -- Diagnosis pointers (up to 12)
    diagnosis_code_pointer_1 SMALLINT,
    diagnosis_code_pointer_2 SMALLINT,
    diagnosis_code_pointer_3 SMALLINT,
    diagnosis_code_pointer_4 SMALLINT,
    diagnosis_code_pointer_5 SMALLINT,
    diagnosis_code_pointer_6 SMALLINT,
    diagnosis_code_pointer_7 SMALLINT,
    diagnosis_code_pointer_8 SMALLINT,
    diagnosis_code_pointer_9 SMALLINT,
    diagnosis_code_pointer_10 SMALLINT,
    diagnosis_code_pointer_11 SMALLINT,
    diagnosis_code_pointer_12 SMALLINT,

    -- Other payer information at line level (Loop 2430)
    other_payer_line_paid_amount NUMERIC(18,2),
    other_payer_line_service_id VARCHAR(48),

    -- Status
    line_status VARCHAR(50) DEFAULT 'ACTIVE',

    -- Audit trail
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100),
    updated_by VARCHAR(100),

    CONSTRAINT uk_encounter_line UNIQUE (encounter_id, line_number),
    CONSTRAINT chk_service_date_range CHECK (service_date_to IS NULL OR service_date_to >= service_date_from)
);
```

**Foreign Keys**:
- `encounter_id` → `claims.encounter(encounter_id)` ON DELETE CASCADE
- `rendering_provider_id` → `claims.provider(provider_id)` (optional)
- `supervising_provider_id` → `claims.provider(provider_id)` (optional, added index in migration 043)
- `ordering_provider_id` → `claims.provider(provider_id)` (optional, added index in migration 043)
- `referring_provider_id` → `claims.provider(provider_id)` (optional, added index in migration 043)
- `service_facility_id` → `claims.facility(facility_id)` (optional, added index in migration 043)

**Indexes**:
- PRIMARY KEY: `service_line_id`
- UNIQUE: `(encounter_id, line_number)`
- `idx_service_line_encounter` ON (encounter_id)
- `idx_service_line_procedure` ON (procedure_code)
- `idx_service_line_date_from` ON (service_date_from)
- `idx_service_line_date_to` ON (service_date_to)
- `idx_service_line_rendering_provider` ON (rendering_provider_id)
- `idx_service_line_supervising_provider` ON (supervising_provider_id) WHERE supervising_provider_id IS NOT NULL (migration 043)
- `idx_service_line_ordering_provider` ON (ordering_provider_id) WHERE ordering_provider_id IS NOT NULL (migration 043)
- `idx_service_line_referring_provider` ON (referring_provider_id) WHERE referring_provider_id IS NOT NULL (migration 043)
- `idx_service_line_service_facility` ON (service_facility_id) WHERE service_facility_id IS NOT NULL (migration 043)
- `idx_service_line_ndc` ON (ndc_code) WHERE ndc_code IS NOT NULL
- `idx_service_line_revenue` ON (revenue_code) WHERE revenue_code IS NOT NULL
- `idx_service_line_enc_line` ON (encounter_id, line_number)
- `idx_service_line_proc_date` ON (procedure_code, service_date_from)
- `idx_service_line_duplicate_lookup` ON (procedure_code, service_date_from, rendering_provider_id) WHERE service_date_from IS NOT NULL AND line_status = 'ACTIVE' (migration 016)
- `idx_service_line_date_range` ON (service_date_from DESC, service_date_to) WHERE service_date_from IS NOT NULL AND line_status = 'ACTIVE' (migration 016)
- `idx_service_line_proc_date_facility` ON (procedure_code, service_date_from, encounter_id) (migration 014)
- `idx_service_line_encounter_proc` ON (encounter_id, procedure_code) INCLUDE (service_unit_count, line_item_charge_amount, service_date_from) (migration 018)
- `idx_service_line_proc_date` ON (procedure_code, service_date_from DESC) (migration 018)
- `idx_service_line_provider` ON (rendering_provider_id, service_date_from DESC) WHERE rendering_provider_id IS NOT NULL (migration 018)

**Triggers**:
- `update_service_line_updated_at`: Auto-update `updated_at` timestamp
- `sync_encounter_totals_insert/update/delete`: Triggers encounter total recalculation (migration 014)

**Check Constraints**:
- Service unit count > 0 and <= 9999.9
- Service date to >= service date from

**Design Notes**:
- Complete 837p service line data model (Loop 2400)
- Supports up to 12 diagnosis pointers per line
- Denormalized provider information for performance
- Check constraint removed from line_item_charge_amount (migration 029) to support adjustments
- Comprehensive indexing for performance optimization

---

#### claims.service_line_adjustment

**Purpose**: Line-level claim adjustments from other payers

```sql
CREATE TABLE claims.service_line_adjustment (
    adjustment_id BIGINT GENERATED ALWAYS AS IDENTITY (CACHE 100) PRIMARY KEY,
    service_line_id BIGINT NOT NULL REFERENCES claims.service_line(service_line_id) ON DELETE CASCADE,
    claim_adjustment_group_code VARCHAR(2) NOT NULL,  -- CO, CR, OA, PI, PR
    adjustment_reason_code VARCHAR(5) NOT NULL,
    adjustment_amount NUMERIC(18,2) NOT NULL,
    adjustment_quantity NUMERIC(15,3),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
```

**Foreign Keys**:
- `service_line_id` → `claims.service_line(service_line_id)` ON DELETE CASCADE

**Indexes**:
- PRIMARY KEY: `adjustment_id`
- `idx_service_line_adj_line` ON (service_line_id)
- `idx_service_line_adj_reason` ON (adjustment_reason_code)

---

#### claims.service_line_diagnosis_pointer

**Purpose**: Explicit mapping between service lines and diagnoses

```sql
CREATE TABLE claims.service_line_diagnosis_pointer (
    pointer_id BIGINT GENERATED ALWAYS AS IDENTITY (CACHE 100) PRIMARY KEY,
    service_line_id BIGINT NOT NULL REFERENCES claims.service_line(service_line_id) ON DELETE CASCADE,
    diagnosis_id BIGINT NOT NULL REFERENCES claims.encounter_diagnosis(diagnosis_id) ON DELETE CASCADE,
    pointer_sequence SMALLINT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT uk_line_diag_pointer UNIQUE (service_line_id, pointer_sequence)
);
```

**Foreign Keys**:
- `service_line_id` → `claims.service_line(service_line_id)` ON DELETE CASCADE
- `diagnosis_id` → `claims.encounter_diagnosis(diagnosis_id)` ON DELETE CASCADE

**Indexes**:
- PRIMARY KEY: `pointer_id`
- UNIQUE: `(service_line_id, pointer_sequence)`
- `idx_line_diag_ptr_line` ON (service_line_id)
- `idx_line_diag_ptr_diag` ON (diagnosis_id)

---

#### claims.flag_category

**Purpose**: Flag category definitions (COD, DOC, EMO, EMU, etc.)

```sql
CREATE TABLE claims.flag_category (
    category_id BIGINT GENERATED ALWAYS AS IDENTITY (CACHE 100) PRIMARY KEY,
    category_code VARCHAR(10) NOT NULL UNIQUE,  -- COD, DOC, EMO, EMU, EMI, EMT, MOD, OTH, QTY, SUP, DX
    category_name VARCHAR(100) NOT NULL,
    category_description TEXT,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
```

**Indexes**:
- PRIMARY KEY: `category_id`
- UNIQUE: `category_code`
- `idx_flag_category_code` ON (category_code)

**Pre-populated Data**:
- COD: Coding Issues
- DOC: Documentation Issues
- EMO: E/M Over-coded
- EMU: E/M Under-coded
- EMI: E/M Incorrect Category
- EMT: E/M Time Not Documented
- MOD: Modifier Issues
- OTH: Other Issues
- QTY: Quantity Issues
- SUP: Supervision Requirements
- DX: Diagnosis Issues

---

#### claims.flag_issue

**Purpose**: Specific flag issue definitions with descriptions

```sql
CREATE TABLE claims.flag_issue (
    issue_id BIGINT GENERATED ALWAYS AS IDENTITY (CACHE 100) PRIMARY KEY,
    category_id BIGINT NOT NULL REFERENCES claims.flag_category(category_id),
    issue_code VARCHAR(20) NOT NULL UNIQUE,
    issue_description TEXT NOT NULL,
    severity VARCHAR(20) DEFAULT 'MEDIUM',  -- HIGH, MEDIUM, LOW
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
```

**Foreign Keys**:
- `category_id` → `claims.flag_category(category_id)`

**Indexes**:
- PRIMARY KEY: `issue_id`
- UNIQUE: `issue_code`
- `idx_flag_issue_category` ON (category_id)
- `idx_flag_issue_code` ON (issue_code)

**Pre-populated Data**: 30+ issue codes covering:
- Coding issues (bundled, incorrect, missed, time missing)
- Documentation issues (missing, limited)
- E/M over/under coding (by level)
- Modifier issues (incorrect, missing, unnecessary)
- Provider/date/signature issues
- Quantity discrepancies
- Supervision requirements
- Diagnosis issues (additional, unsupported, incorrect, specificity)

---

#### claims.encounter_flag

**Purpose**: Flags assigned to encounters by the rules engine

```sql
CREATE TABLE claims.encounter_flag (
    flag_id BIGINT GENERATED ALWAYS AS IDENTITY (CACHE 100) PRIMARY KEY,
    encounter_id BIGINT NOT NULL REFERENCES claims.encounter(encounter_id) ON DELETE CASCADE,
    issue_id BIGINT NOT NULL REFERENCES claims.flag_issue(issue_id),

    -- Flag details
    flag_type VARCHAR(20) DEFAULT 'POST_BILL',  -- POST_BILL, PRE_BILL
    severity VARCHAR(20),
    flag_reason TEXT,
    flagged_element VARCHAR(255),  -- What was flagged (e.g., "CPT 99214", "Modifier 25")

    -- Proposed changes (if applicable)
    proposed_code VARCHAR(50),
    proposed_modifier VARCHAR(10),
    proposed_quantity NUMERIC(15,3),
    proposed_diagnosis_code VARCHAR(30),

    -- Status
    flag_status VARCHAR(20) DEFAULT 'OPEN',  -- OPEN, REVIEWED, ACCEPTED, REJECTED, RESOLVED
    resolution_note TEXT,
    resolved_at TIMESTAMPTZ,
    resolved_by VARCHAR(100),

    -- Audit trail
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100) DEFAULT 'SYSTEM'
);
```

**Foreign Keys**:
- `encounter_id` → `claims.encounter(encounter_id)` ON DELETE CASCADE
- `issue_id` → `claims.flag_issue(issue_id)`

**Indexes**:
- PRIMARY KEY: `flag_id`
- `idx_encounter_flag_encounter` ON (encounter_id)
- `idx_encounter_flag_issue` ON (issue_id)
- `idx_encounter_flag_status` ON (flag_status)
- `idx_encounter_flag_type` ON (flag_type)
- `idx_encounter_flag_severity` ON (severity)
- `idx_encounter_flag_created` ON (created_at)
- `idx_encounter_flag_enc_status` ON (encounter_id, flag_status)
- `idx_encounter_flag_status_created` ON (flag_status, created_at) WHERE flag_status = 'OPEN'
- `idx_flag_org_severity_status` ON (severity, flag_status, created_at) WHERE flag_status = 'OPEN' (migration 014)
- `idx_encounter_flag_severity` ON (encounter_id, severity, flag_status) WHERE flag_status IN ('OPEN', 'CLOSED') (migration 016)
- `idx_encounter_flag_created` ON (created_at DESC, flag_status) WHERE flag_status = 'OPEN' (migration 016)

---

#### claims.service_line_flag

**Purpose**: Flags assigned to service lines by the rules engine

```sql
CREATE TABLE claims.service_line_flag (
    flag_id BIGINT GENERATED ALWAYS AS IDENTITY (CACHE 100) PRIMARY KEY,
    service_line_id BIGINT NOT NULL REFERENCES claims.service_line(service_line_id) ON DELETE CASCADE,
    issue_id BIGINT NOT NULL REFERENCES claims.flag_issue(issue_id),

    -- Flag details
    flag_type VARCHAR(20) DEFAULT 'POST_BILL',  -- POST_BILL, PRE_BILL
    severity VARCHAR(20),
    flag_reason TEXT,
    flagged_element VARCHAR(255),

    -- Proposed changes
    proposed_code VARCHAR(50),
    proposed_modifier VARCHAR(10),
    proposed_quantity NUMERIC(15,3),

    -- Status
    flag_status VARCHAR(20) DEFAULT 'OPEN',
    resolution_note TEXT,
    resolved_at TIMESTAMPTZ,
    resolved_by VARCHAR(100),

    -- Audit trail
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100) DEFAULT 'SYSTEM'
);
```

**Foreign Keys**:
- `service_line_id` → `claims.service_line(service_line_id)` ON DELETE CASCADE
- `issue_id` → `claims.flag_issue(issue_id)`

**Indexes**:
- PRIMARY KEY: `flag_id`
- `idx_service_line_flag_line` ON (service_line_id)
- `idx_service_line_flag_issue` ON (issue_id)
- `idx_service_line_flag_status` ON (flag_status)
- `idx_service_line_flag_type` ON (flag_type)
- `idx_service_line_flag_created` ON (created_at)
- `idx_service_line_flag_severity` ON (service_line_id, severity, flag_status) WHERE flag_status IN ('OPEN', 'CLOSED') (migration 016)

---

### Additional Claims Schema Tables

Due to length constraints, here's a summary of remaining tables in the claims schema:

#### Audit Tables
- **audit_assignment**: Retrospective audit assignments
- **audit_encounter**: Encounters selected for audit
- **service_line_evaluation**: Detailed audit findings for service lines
- **diagnosis_evaluation**: Audit findings for diagnosis codes
- **coder_accuracy**: Coder accuracy metrics over time
- **provider_accuracy**: Provider documentation accuracy metrics

#### Denial Tables
- **denial_event**: Denial events from payer remittances with root cause analysis
- **denial_appeal**: Appeal actions and correspondence
- **denial_reason_code**: CARC/RARC code reference
- **denial_statistics**: Aggregated denial statistics

#### RVU Tables
- **rvu_reference**: RVU data from CMS Physician Fee Schedule
- **conversion_factor**: Annual Medicare conversion factors
- **gpci_reference**: Geographic Practice Cost Indexes
- **modifier_adjustment**: Modifier-based reimbursement adjustments
- **service_line_reimbursement**: Medicare reimbursement estimates

#### Metadata Tables
- **field_definitions**: Column metadata for dynamic field mapping (migration 033)
- **import_headers**: Data element headers for imports (migration 030)

---

### Staging Schema Tables

#### staging.import_batch

**Purpose**: Tracks file import batches and processing metrics

```sql
CREATE TABLE staging.import_batch (
    batch_id BIGINT GENERATED ALWAYS AS IDENTITY (CACHE 100) PRIMARY KEY,
    organization_id BIGINT NOT NULL REFERENCES claims.organization(organization_id),
    facility_id BIGINT REFERENCES claims.facility(facility_id),

    -- Batch information
    batch_name VARCHAR(255),
    batch_type VARCHAR(50) NOT NULL,  -- EDI_837P, CSV, MANUAL
    file_format VARCHAR(50),  -- For CSV: EXCEL, ATHENA, EPIC, CERNER, etc.

    -- File details
    original_filename VARCHAR(500),
    file_path TEXT,
    file_size_bytes BIGINT,
    file_hash VARCHAR(64),  -- SHA-256 for deduplication

    -- Processing status
    import_status VARCHAR(50) DEFAULT 'PENDING',  -- PENDING, QUEUED, INGESTING, INGESTED, PROCESSING, COMPLETED, FAILED
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
    configuration_id BIGINT REFERENCES staging.import_configuration(configuration_id) ON DELETE SET NULL ON UPDATE CASCADE,  -- FK added in migration 045
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

    CONSTRAINT chk_batch_type CHECK (batch_type IN ('EDI_837P', 'CSV', 'MANUAL')),
    CONSTRAINT ck_import_batch_import_status CHECK (import_status IN ('PENDING', 'QUEUED', 'INGESTING', 'INGESTED', 'PROCESSING', 'COMPLETED', 'FAILED'))  -- Updated in migration 023
);
```

**Foreign Keys**:
- `organization_id` → `claims.organization(organization_id)` (required)
- `facility_id` → `claims.facility(facility_id)` (optional)
- `configuration_id` → `staging.import_configuration(configuration_id)` ON DELETE SET NULL (optional, FK added in migration 045)

**Indexes**:
- PRIMARY KEY: `batch_id`
- `idx_import_batch_org` ON (organization_id)
- `idx_import_batch_facility` ON (facility_id)
- `idx_import_batch_status` ON (import_status)
- `idx_import_batch_type` ON (batch_type)
- `idx_import_batch_created` ON (created_at)
- `idx_import_batch_file_hash` ON (file_hash)
- `idx_import_batch_started` ON (started_at)
- `idx_import_batch_org_status_created` ON (organization_id, import_status, created_at)
- `idx_import_batch_org_created` ON (organization_id, created_at DESC) (migration 018)
- `idx_import_batch_configuration_id` ON (configuration_id) (migration 045)

**Design Notes**:
- Two-stage processing pipeline support (INGESTING, INGESTED states added in migration 023)
- SHA-256 file hash for duplicate detection
- JSONB columns for flexible error/validation data

---

#### staging.raw_claims

**Purpose**: Two-stage processing pipeline: Stage 1 stores raw parsed claims before Stage 2 validates and inserts to encounters

```sql
CREATE TABLE staging.raw_claims (
    raw_claim_id BIGINT GENERATED ALWAYS AS IDENTITY (CACHE 100) PRIMARY KEY,
    batch_id BIGINT NOT NULL REFERENCES staging.import_batch(batch_id) ON DELETE CASCADE,
    queue_id BIGINT NOT NULL REFERENCES staging.file_processing_queue(queue_id) ON DELETE CASCADE,

    -- Original parsed data stored as JSONB for flexibility
    encounter_fields JSONB NOT NULL,
    service_line_fields JSONB,
    diagnosis_fields JSONB,

    -- Metadata from original file
    row_number INTEGER NOT NULL,
    facility_code TEXT,

    -- Processing status tracking
    processing_status TEXT NOT NULL DEFAULT 'PENDING',  -- PENDING, PROCESSING, COMPLETED, FAILED

    -- Timestamps for tracking pipeline latency
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMPTZ,

    -- Error tracking
    error_message TEXT,

    -- FIFO ordering field (extracted from encounter_fields for performance)
    date_of_service_from DATE,

    -- Batch sequence tracking (migration 024)
    batch_sequence_number INTEGER,

    CONSTRAINT ck_raw_claims_status CHECK (processing_status IN ('PENDING', 'PROCESSING', 'COMPLETED', 'FAILED'))
);
```

**Foreign Keys**:
- `batch_id` → `staging.import_batch(batch_id)` ON DELETE CASCADE
- `queue_id` → `staging.file_processing_queue(queue_id)` ON DELETE CASCADE

**Indexes**:
- PRIMARY KEY: `raw_claim_id`
- `idx_raw_claims_pending` ON (ingested_at ASC) WHERE processing_status IN ('PENDING', 'PROCESSING')
- `idx_raw_claims_batch` ON (batch_id, processing_status)
- `idx_raw_claims_queue` ON (queue_id, processing_status)
- `idx_raw_claims_fifo` ON (facility_code, date_of_service_from ASC, ingested_at ASC) WHERE processing_status = 'PENDING'
- `idx_raw_claims_stale` ON (processing_status, ingested_at) WHERE processing_status = 'PROCESSING'
- `idx_raw_claims_sequence_status` ON (batch_sequence_number, processing_status) WHERE processing_status IN ('PENDING', 'PROCESSING', 'COMPLETED') (migration 024)

**Purpose**: Decouples file ingestion from claim validation:
- **Stage 1 (Fast)**: Parse file → raw_claims (JSONB storage, no validation)
- **Stage 2 (Validated)**: raw_claims → encounters/service_lines/diagnoses (with validation and rules)

---

#### staging.batch_sequences

**Purpose**: Tracks batch sequence numbers for strict FIFO ordering with multi-worker processing

```sql
CREATE TABLE staging.batch_sequences (
    sequence_number INTEGER PRIMARY KEY,
    assigned_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMPTZ,

    -- Batch metadata
    batch_id BIGINT NOT NULL REFERENCES staging.import_batch(batch_id) ON DELETE CASCADE,
    claim_count INTEGER NOT NULL,
    processing_stage TEXT NOT NULL DEFAULT 'STAGE2',

    -- Worker tracking
    worker_id TEXT,

    -- Processing metrics
    processing_time_seconds REAL,
    success_count INTEGER DEFAULT 0,
    failure_count INTEGER DEFAULT 0,

    -- Error tracking
    errors JSONB,

    CONSTRAINT ck_batch_sequences_stage CHECK (processing_stage IN ('STAGE2', 'VALIDATION', 'RULES', 'COMPLETION'))
);
```

**Foreign Keys**:
- `batch_id` → `staging.import_batch(batch_id)` ON DELETE CASCADE

**Indexes**:
- PRIMARY KEY: `sequence_number`
- `idx_batch_sequences_incomplete` ON (sequence_number) WHERE completed_at IS NULL
- `idx_batch_sequences_performance` ON (assigned_at, completed_at, processing_time_seconds)

**Purpose**: Enables Sequential Completion Manager for multi-worker pipeline with strict FIFO ordering (Aegis-inspired architecture)

---

#### staging.file_processing_queue

**Purpose**: FIFO queue for file processing ensuring chronological order per facility

```sql
CREATE TABLE staging.file_processing_queue (
    queue_id BIGINT GENERATED ALWAYS AS IDENTITY (CACHE 100) PRIMARY KEY,
    facility_id BIGINT NOT NULL REFERENCES claims.facility(facility_id),
    import_batch_id BIGINT NOT NULL REFERENCES staging.import_batch(batch_id),
    file_path TEXT NOT NULL,
    file_hash TEXT NOT NULL,
    file_format TEXT NOT NULL,
    organization_id BIGINT NOT NULL REFERENCES claims.organization(organization_id),

    -- FIFO ordering
    queued_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    processing_started_at TIMESTAMPTZ,
    processing_completed_at TIMESTAMPTZ,

    -- Status tracking
    queue_status TEXT NOT NULL DEFAULT 'QUEUED',  -- QUEUED, PROCESSING, COMPLETED, FAILED, RETRY, STREAMING, PARTIAL_SUCCESS

    -- Priority (lower number = higher priority, default = 100)
    priority INTEGER NOT NULL DEFAULT 100,

    -- Retry handling
    retry_count INTEGER NOT NULL DEFAULT 0,
    max_retries INTEGER NOT NULL DEFAULT 3,
    last_error TEXT,

    -- Audit trail
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_by TEXT DEFAULT 'SYSTEM',
    updated_by TEXT DEFAULT 'SYSTEM',

    CONSTRAINT valid_queue_status CHECK (queue_status IN ('QUEUED', 'PROCESSING', 'COMPLETED', 'FAILED', 'RETRY', 'STREAMING', 'PARTIAL_SUCCESS')),  -- Updated in migration 017
    CONSTRAINT valid_priority CHECK (priority >= 0 AND priority <= 1000),
    CONSTRAINT valid_retry_count CHECK (retry_count >= 0 AND retry_count <= max_retries)
);
```

**Foreign Keys**:
- `facility_id` → `claims.facility(facility_id)` (required)
- `import_batch_id` → `staging.import_batch(batch_id)` (required)
- `organization_id` → `claims.organization(organization_id)` (required)

**Indexes**:
- PRIMARY KEY: `queue_id`
- `idx_queue_fifo_by_facility` ON (facility_id, priority ASC, queued_at ASC) WHERE queue_status = 'QUEUED'
- `idx_queue_fifo_global` ON (priority ASC, queued_at ASC) WHERE queue_status = 'QUEUED'
- `idx_queue_processing` ON (queue_status, processing_started_at DESC) WHERE queue_status = 'PROCESSING'
- `idx_queue_failed` ON (queue_status, queued_at DESC) WHERE queue_status = 'FAILED'
- `idx_queue_retry` ON (queue_status, retry_count ASC, queued_at ASC) WHERE queue_status = 'RETRY' AND retry_count < max_retries
- `idx_queue_facility_stats` ON (facility_id, queue_status, created_at DESC)
- `idx_queue_organization` ON (organization_id, queue_status, queued_at ASC)
- `idx_queue_global_fifo` ON (priority ASC, queued_at ASC) WHERE queue_status IN ('QUEUED', 'RETRY') (migration 016)
- `idx_queue_facility_fifo` ON (facility_id, priority ASC, queued_at ASC) WHERE queue_status IN ('QUEUED', 'RETRY') (migration 016)
- `idx_queue_status_monitoring` ON (queue_status, queued_at DESC) WHERE queue_status = 'PROCESSING' (migration 016)
- `idx_processing_queue_status_created` ON (queue_status, created_at DESC) (migration 018)

**Triggers**:
- `trg_queue_updated_at`: Auto-update `updated_at` timestamp (migration 015)

**Functions**:
- `staging.update_queue_updated_at()`: Trigger function for updated_at
- `staging.cleanup_old_queue_entries(retention_days)`: Removes old completed/failed entries (default 90 days)

---

#### staging.file_processing_progress

**Purpose**: Real-time progress tracking for streaming file processing

```sql
CREATE TABLE staging.file_processing_progress (
    id BIGINT GENERATED ALWAYS AS IDENTITY (CACHE 100) PRIMARY KEY,
    queue_id BIGINT NOT NULL REFERENCES staging.file_processing_queue(queue_id) ON DELETE CASCADE,

    -- Claim counts
    total_claims INTEGER NOT NULL DEFAULT 0,
    processed_claims INTEGER NOT NULL DEFAULT 0,
    failed_claims INTEGER NOT NULL DEFAULT 0,

    -- Flag statistics
    flags_created INTEGER NOT NULL DEFAULT 0,
    critical_flags INTEGER NOT NULL DEFAULT 0,

    -- Timing information
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ,
    estimated_completion_at TIMESTAMPTZ,

    -- Performance metrics
    claims_per_second DECIMAL(10, 2),
    average_processing_time_ms INTEGER,

    -- Metadata
    is_active BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

**Foreign Keys**:
- `queue_id` → `staging.file_processing_queue(queue_id)` ON DELETE CASCADE

**Indexes**:
- PRIMARY KEY: `id`
- `idx_progress_queue_id` ON (queue_id)
- `idx_progress_active` ON (is_active, updated_at DESC) WHERE is_active = true

**Triggers**:
- `update_file_processing_progress_updated_at`: Auto-update `updated_at` timestamp

---

#### staging.failed_claims

**Purpose**: Individual claim failures during streaming processing

```sql
CREATE TABLE staging.failed_claims (
    id BIGINT GENERATED ALWAYS AS IDENTITY (CACHE 100) PRIMARY KEY,
    queue_id BIGINT NOT NULL REFERENCES staging.file_processing_queue(queue_id) ON DELETE CASCADE,
    progress_id BIGINT REFERENCES staging.file_processing_progress(id) ON DELETE CASCADE,

    -- Claim identification
    claim_number VARCHAR(50),
    subscriber_id_from_file VARCHAR(50),
    provider_npi VARCHAR(20),

    -- Error information
    error_message TEXT NOT NULL,
    error_type VARCHAR(100),
    stack_trace TEXT,

    -- Claim data (JSON for debugging)
    claim_data JSONB,

    -- Retry information
    retry_count INTEGER NOT NULL DEFAULT 0,
    last_retry_at TIMESTAMPTZ,
    can_retry BOOLEAN NOT NULL DEFAULT true,

    -- Metadata
    failed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

**Foreign Keys**:
- `queue_id` → `staging.file_processing_queue(queue_id)` ON DELETE CASCADE
- `progress_id` → `staging.file_processing_progress(id)` ON DELETE CASCADE

**Indexes**:
- PRIMARY KEY: `id`
- `idx_failed_claims_queue_id` ON (queue_id)
- `idx_failed_claims_progress_id` ON (progress_id)
- `idx_failed_claims_can_retry` ON (can_retry, retry_count) WHERE can_retry = true
- `idx_failed_claims_error_type` ON (error_type)

---

#### Additional Staging Tables

- **file_upload**: Multi-part file upload tracking
- **import_configuration**: Import configuration profiles for different file types
- **rules_configuration**: Rules engine configuration for automated flagging
- **processing_metrics**: Performance metrics for import processing stages
- **import_error_log**: Detailed error log for import failures
- **processing_configuration**: Runtime configuration (FIFO mode, worker count, batch size) (migration 024)
- **schema_migrations**: Tracks applied database migrations (migration 020)
- **application_version**: Application version history (migration 020)

---

### ML Schema Tables

#### ml.model_registry

**Purpose**: Registry of trained machine learning models

```sql
CREATE TABLE ml.model_registry (
    model_id BIGINT GENERATED ALWAYS AS IDENTITY (CACHE 100) PRIMARY KEY,
    organization_id BIGINT REFERENCES claims.organization(organization_id),

    -- Model identification
    model_name VARCHAR(255) NOT NULL,
    model_type VARCHAR(50) NOT NULL,  -- CLASSIFICATION, REGRESSION, CLUSTERING, ANOMALY_DETECTION
    model_purpose VARCHAR(100) NOT NULL,  -- DENIAL_PREDICTION, CODING_SUGGESTION, AUDIT_RISK, etc.
    model_version VARCHAR(50) NOT NULL,

    -- Model details
    algorithm VARCHAR(100),  -- RANDOM_FOREST, GRADIENT_BOOSTING, NEURAL_NETWORK, etc.
    framework VARCHAR(50),  -- SCIKIT_LEARN, TENSORFLOW, PYTORCH, etc.
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
    deployment_status VARCHAR(50) DEFAULT 'DEVELOPMENT',  -- DEVELOPMENT, STAGING, PRODUCTION, RETIRED
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
```

**Indexes**:
- PRIMARY KEY: `model_id`
- UNIQUE: `(model_name, model_version)`
- `idx_model_registry_org` ON (organization_id)
- `idx_model_registry_type` ON (model_type)
- `idx_model_registry_purpose` ON (model_purpose)
- `idx_model_registry_status` ON (deployment_status)
- `idx_model_registry_active` ON (is_active) WHERE is_active = true
- `idx_model_registry_deployment` ON (deployment_status, model_purpose) WHERE is_active = true (migration 018)

**Triggers**:
- `update_model_registry_updated_at`: Auto-update `updated_at` timestamp

---

#### ml.model_prediction

**Purpose**: Predictions made by ML models

```sql
CREATE TABLE ml.model_prediction (
    prediction_id BIGINT GENERATED ALWAYS AS IDENTITY (CACHE 100) PRIMARY KEY,
    model_id BIGINT NOT NULL REFERENCES ml.model_registry(model_id),
    encounter_id BIGINT REFERENCES claims.encounter(encounter_id) ON DELETE CASCADE,
    service_line_id BIGINT REFERENCES claims.service_line(service_line_id) ON DELETE CASCADE,

    -- Prediction details
    prediction_type VARCHAR(50) NOT NULL,  -- DENIAL_RISK, CODING_ERROR, AUDIT_PRIORITY, etc.
    predicted_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,

    -- Input features
    input_features JSONB NOT NULL,

    -- Prediction output
    predicted_value VARCHAR(255),
    predicted_class VARCHAR(100),
    prediction_score NUMERIC(8,6),  -- Confidence score 0-1
    prediction_probability NUMERIC(8,6),

    -- Classification predictions
    class_probabilities JSONB,  -- For multi-class predictions

    -- Risk scoring
    risk_score NUMERIC(8,4),
    risk_level VARCHAR(20),  -- LOW, MEDIUM, HIGH, CRITICAL

    -- Explanation
    feature_contributions JSONB,  -- SHAP values or similar
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
```

**Indexes**:
- PRIMARY KEY: `prediction_id`
- `idx_model_prediction_model` ON (model_id)
- `idx_model_prediction_encounter` ON (encounter_id)
- `idx_model_prediction_service_line` ON (service_line_id)
- `idx_model_prediction_type` ON (prediction_type)
- `idx_model_prediction_predicted_at` ON (predicted_at)
- `idx_model_prediction_risk_level` ON (risk_level)
- `idx_model_prediction_score` ON (prediction_score)
- `idx_model_prediction_encounter_type` ON (encounter_id, prediction_type, predicted_at DESC) (migration 018)
- `idx_model_prediction_risk` ON (risk_level, predicted_at DESC) WHERE risk_level IN ('HIGH', 'CRITICAL') (migration 018)

---

#### Additional ML Tables

- **feature_definition**: Definitions of features for ML models
- **training_dataset**: Training datasets for ML models
- **model_performance_log**: Performance monitoring for deployed ML models
- **ab_test_experiment**: A/B testing experiments for model comparison

---

### Analytics Schema - Materialized Views

#### analytics.flag_statistics_daily

**Purpose**: Daily aggregated flag statistics for dashboards

```sql
CREATE MATERIALIZED VIEW analytics.flag_statistics_daily AS
SELECT
    f.organization_id,
    f.facility_id,
    DATE(f.created_at) as flag_date,
    f.flag_category,
    f.flag_severity,
    f.flag_status,
    COUNT(*) as flag_count,
    COUNT(DISTINCT f.encounter_id) as unique_encounters,
    COUNT(DISTINCT f.service_line_id) as unique_service_lines,
    AVG(f.financial_impact) as avg_financial_impact,
    SUM(f.financial_impact) as total_financial_impact,
    MIN(f.financial_impact) as min_financial_impact,
    MAX(f.financial_impact) as max_financial_impact,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY f.financial_impact) as median_financial_impact
FROM claims.flag f
WHERE f.created_at >= CURRENT_DATE - INTERVAL '90 days'
GROUP BY f.organization_id, f.facility_id, DATE(f.created_at), f.flag_category, f.flag_severity, f.flag_status;
```

**Indexes**:
- UNIQUE: `idx_flag_stats_daily_pk` ON (organization_id, COALESCE(facility_id, 0), flag_date, flag_category, flag_severity, flag_status)
- `idx_flag_stats_org_date` ON (organization_id, flag_date DESC)
- `idx_flag_stats_facility_date` ON (facility_id, flag_date DESC) WHERE facility_id IS NOT NULL

**Refresh**: Via `analytics.refresh_all_views()` function (CONCURRENTLY)

---

#### Additional Analytics Materialized Views

- **encounter_statistics_daily**: Daily encounter volume and financial statistics
- **procedure_statistics**: Procedure code usage and financial statistics
- **provider_performance**: Provider activity and quality metrics
- **payer_statistics**: Payer performance and denial rate tracking
- **ml_model_performance_summary**: ML model performance tracking

**Refresh Function**: `analytics.refresh_all_views()` - Refreshes all materialized views concurrently without table locking

---

## Foreign Key Relationships

### Organization Hierarchy

```
claims.organization (1)
    ├─→ claims.region (N) ON DELETE CASCADE
    │       └─→ claims.facility (N) ON DELETE SET NULL
    └─→ claims.facility (N) ON DELETE CASCADE
```

### Provider Relationships

```
claims.organization (1)
    ├─→ claims.provider (N)
    ├─→ claims.coder (N)
    └─→ claims.reviewer (N)

claims.provider_taxonomy (1)
    └─→ claims.provider (N) ON DELETE RESTRICT
```

### Encounter Relationships

```
claims.facility (1) ←─ claims.encounter (N)
claims.organization (1) ←─ claims.encounter (N)
claims.region (1) ←─ claims.encounter (N)
claims.provider (1) ←─ claims.encounter (N) [billing, rendering, referring, supervising]
claims.coder (1) ←─ claims.encounter (N)
staging.import_batch (1) ←─ claims.encounter (N) ON DELETE SET NULL
staging.import_configuration (1) ←─ claims.encounter (N) ON DELETE SET NULL

claims.encounter (1)
    ├─→ claims.encounter_note (N) ON DELETE CASCADE
    ├─→ claims.encounter_diagnosis (N) ON DELETE CASCADE
    ├─→ claims.service_line (N) ON DELETE CASCADE
    ├─→ claims.encounter_flag (N) ON DELETE CASCADE
    ├─→ claims.audit_encounter (N) ON DELETE CASCADE
    └─→ claims.denial_event (N) ON DELETE CASCADE
```

### Service Line Relationships

```
claims.encounter (1)
    └─→ claims.service_line (N) ON DELETE CASCADE
            ├─→ claims.service_line_adjustment (N) ON DELETE CASCADE
            ├─→ claims.service_line_diagnosis_pointer (N) ON DELETE CASCADE
            ├─→ claims.service_line_flag (N) ON DELETE CASCADE
            ├─→ claims.service_line_evaluation (N) ON DELETE CASCADE
            └─→ claims.service_line_reimbursement (1) ON DELETE CASCADE

claims.provider (1) ←─ claims.service_line (N) [rendering, supervising, ordering, referring]
claims.facility (1) ←─ claims.service_line (N) [service_facility]
```

### Flag System Relationships

```
claims.flag_category (1)
    └─→ claims.flag_issue (N)
            ├─→ claims.encounter_flag (N)
            └─→ claims.service_line_flag (N)

claims.encounter (1) ←─ claims.encounter_flag (N) ON DELETE CASCADE
claims.service_line (1) ←─ claims.service_line_flag (N) ON DELETE CASCADE
```

### Audit System Relationships

```
claims.organization (1)
    └─→ claims.audit_assignment (N)
            └─→ claims.audit_encounter (N) ON DELETE CASCADE

claims.reviewer (1) ←─ claims.audit_assignment (N)
claims.encounter (1) ←─ claims.audit_encounter (N) ON DELETE CASCADE
claims.service_line (1) ←─ claims.service_line_evaluation (N) ON DELETE CASCADE
claims.encounter_diagnosis (1) ←─ claims.diagnosis_evaluation (N) ON DELETE CASCADE
```

### Denial System Relationships

```
claims.encounter (1)
    └─→ claims.denial_event (N) ON DELETE CASCADE
            └─→ claims.denial_appeal (N) ON DELETE CASCADE

claims.service_line (1) ←─ claims.denial_event (N) ON DELETE CASCADE
claims.denial_reason_code (1) ←─ claims.denial_event (N) [CARC/RARC lookup]
```

### Staging Pipeline Relationships

```
claims.organization (1)
    └─→ staging.import_batch (N)
            ├─→ staging.file_processing_queue (N)
            │       ├─→ staging.file_processing_progress (1) ON DELETE CASCADE
            │       ├─→ staging.failed_claims (N) ON DELETE CASCADE
            │       └─→ staging.raw_claims (N) ON DELETE CASCADE
            ├─→ staging.processing_metrics (N) ON DELETE CASCADE
            ├─→ staging.import_error_log (N) ON DELETE CASCADE
            ├─→ staging.batch_sequences (N) ON DELETE CASCADE
            └─→ claims.encounter (N) ON DELETE SET NULL

staging.import_configuration (1)
    ├─→ staging.import_batch (N) ON DELETE SET NULL
    └─→ claims.encounter (N) ON DELETE SET NULL
```

### ML System Relationships

```
ml.model_registry (1)
    ├─→ ml.model_prediction (N)
    ├─→ ml.model_performance_log (N)
    └─→ ml.ab_test_experiment (N) [control_model, treatment_model]

claims.encounter (1) ←─ ml.model_prediction (N) ON DELETE CASCADE
claims.service_line (1) ←─ ml.model_prediction (N) ON DELETE CASCADE
```

---

## Index Strategy

### Primary Index Types

#### B-tree Indexes (Default)
- **Usage**: 95% of indexes
- **Purpose**: Equality and range queries
- **Performance**: O(log n) lookup time
- **Examples**: All primary keys, foreign keys, date ranges, numeric ranges

#### GIN Indexes (Full-text Search)
- **Usage**: Text search columns
- **Purpose**: Fuzzy string matching with pg_trgm
- **Performance**: Slower writes, fast text searches
- **Columns**:
  - `encounter.patient_control_number`
  - `encounter.subscriber_last_name`
  - `provider.last_name`
  - `facility.facility_name`

#### Partial Indexes (Filtered)
- **Usage**: ~30% of indexes
- **Purpose**: Index only relevant rows (e.g., WHERE is_active = true)
- **Benefit**: 40-60% smaller index size, faster maintenance
- **Examples**:
  - Active records: WHERE is_active = true
  - Open flags: WHERE flag_status = 'OPEN'
  - Non-NULL FKs: WHERE provider_id IS NOT NULL
  - FIFO processing: WHERE queue_status IN ('PENDING', 'RETRY')

#### Composite Indexes
- **Usage**: Query patterns with multiple filters
- **Purpose**: Optimize multi-column WHERE clauses and ORDER BY
- **Column Ordering**: Most selective column first (exception: FIFO uses time first)
- **Examples**:
  - `(organization_id, date_of_service_from, claim_status)`
  - `(facility_id, priority ASC, queued_at ASC)`
  - `(procedure_code, service_date_from, rendering_provider_id)`

#### Covering Indexes (INCLUDE clause)
- **Usage**: High-traffic queries
- **Purpose**: Avoid table lookups (index-only scans)
- **Example**: `idx_service_line_encounter_proc` includes (service_unit_count, line_item_charge_amount, service_date_from)

### Index Naming Convention

Format: `idx_<table>_<columns>_<condition>`

Examples:
- `idx_encounter_facility_dos` - Composite index on facility_id and date_of_service_from
- `idx_encounter_active` - Partial index WHERE is_active = true
- `idx_queue_fifo_by_facility` - FIFO ordering index for facility-based queuing

### Performance Indexes by Use Case

#### Dashboard Queries
- `idx_encounter_org_dos_status`: Organization dashboard filtering
- `idx_encounter_facility_dos`: Facility-level reporting
- `idx_flag_stats_org_date`: Daily flag statistics
- `idx_denial_stats_date`: Denial trend analysis

#### Import Pipeline
- `idx_queue_fifo_by_facility`: FIFO claim processing per facility
- `idx_queue_fifo_global`: Global FIFO across all facilities
- `idx_raw_claims_sequence_status`: Batch sequence ordering
- `idx_import_batch_file_hash`: Duplicate file detection

#### Rules Engine
- `idx_service_line_duplicate_lookup`: Duplicate service detection
- `idx_encounter_subscriber_history`: Patient history lookups
- `idx_provider_npi_lookup`: Provider credential validation
- `idx_encounter_diagnosis_code`: Diagnosis validation

#### Audit System
- `idx_audit_encounter_audit_status`: Audit progress tracking
- `idx_service_line_eval_result_severity`: Error finding analysis
- `idx_coder_accuracy_period`: Coder performance reports

#### Search Operations
- `idx_encounter_patient_control_trgm`: Fuzzy claim number search
- `idx_encounter_subscriber_last_name_trgm`: Fuzzy patient name search
- `idx_provider_last_name_trgm`: Fuzzy provider name search

### Index Maintenance Strategy

#### CONCURRENTLY Creation
- All indexes created with `CREATE INDEX CONCURRENTLY` (migrations 016, 018, 043)
- Zero downtime during index creation
- Safe for production deployments

#### REINDEX Strategy
- Recommended: Monthly REINDEX CONCURRENTLY for high-churn tables
- Target tables: encounter, service_line, raw_claims, import_batch
- Prevents index bloat

#### Statistics Updates
- `ANALYZE` run after all index creations (migrations 016, 018)
- Recommended: Daily ANALYZE on high-churn tables
- Ensures query planner uses optimal execution plans

---

## Data Types and Design Decisions

### Identity Columns (Primary Keys)

**Design**: BIGINT GENERATED ALWAYS AS IDENTITY

**Rationale**:
- **BIGINT**: Supports 9,223,372,036,854,775,807 records (sufficient for decades)
- **GENERATED ALWAYS**: Prevents accidental manual ID assignment
- **IDENTITY**: Modern PostgreSQL sequence management (vs SERIAL)
- **CACHE values**: 50-100 for high-volume tables to reduce sequence contention

**Alternative Rejected**: UUID
- Reason: ~3x larger storage (16 bytes vs 8 bytes), slower B-tree index performance, random insertion causes page splits

### Numeric Types

#### Financial Amounts
- **Type**: NUMERIC(18,2)
- **Range**: -999,999,999,999,999.99 to 999,999,999,999,999.99
- **Precision**: Exact decimal arithmetic (no rounding errors)
- **Usage**: All charge amounts, payments, adjustments, RVU calculations

#### Unit Counts
- **Type**: NUMERIC(15,1) or INTEGER
- **Precision**: One decimal place for fractional units (e.g., 1.5 units)
- **Range**: 0.1 to 9999.9 for service units

#### Percentages
- **Type**: NUMERIC(5,2) or NUMERIC(8,6)
- **Examples**:
  - NUMERIC(5,2): 0.00 to 100.00 (e.g., denial rate 15.75%)
  - NUMERIC(8,6): 0.000000 to 1.000000 (e.g., ML confidence 0.852341)

#### RVU Values
- **Type**: NUMERIC(10,3)
- **Precision**: Three decimal places per CMS specification
- **Examples**: 1.234, 10.567, 125.890

### String Types

#### Fixed-Length Codes
- **Type**: CHAR(n)
- **Usage**: State codes (CHAR(2)), country codes (CHAR(3)), gender (CHAR(1))
- **Rationale**: Fixed length, space-padded, slight performance benefit

#### Variable-Length Fields
- **Type**: VARCHAR(n)
- **Usage**: Names, codes, identifiers
- **Length Guidelines**:
  - Names: VARCHAR(255) - standard length for person/organization names
  - Codes: VARCHAR(10-50) - match industry standard lengths (NPI=10, taxonomy=10)
  - Identifiers: VARCHAR(38-80) - accommodate MBI, payer IDs
  - Addresses: VARCHAR(255) - support long street addresses

#### Unlimited Text
- **Type**: TEXT
- **Usage**: Descriptions, notes, error messages, definitions
- **Rationale**: No length limit, same performance as VARCHAR for PostgreSQL

#### Case-Insensitive Text
- **Type**: citext (extension)
- **Usage**: Email addresses
- **Rationale**: Case-insensitive comparisons, proper email handling

### Date/Time Types

#### Timestamps with Timezone
- **Type**: TIMESTAMPTZ
- **Usage**: All audit fields (created_at, updated_at), processing timestamps
- **Rationale**: Timezone-aware, handles daylight saving, UTC storage
- **Display**: Application converts to user's local timezone

#### Dates Only
- **Type**: DATE
- **Usage**: Date of service, birth dates, effective dates
- **Rationale**: No time component needed, smaller storage (4 bytes vs 8 bytes)

### Boolean Types

**Type**: BOOLEAN (true/false/null)

**Usage**: Flags (is_active, soft_deleted, is_principal, emergency_indicator)

**Rationale**: Clear intent, indexable, supports three-state logic with NULL

### Array Types

**Type**: TEXT[] or NUMERIC[]

**Usage**:
- Certifications: TEXT[] for coder/reviewer certifications
- Feature lists: TEXT[] for ML features
- Cross-validation scores: NUMERIC[] for model scores

**Rationale**: PostgreSQL native array support, no separate table needed for simple lists

### JSONB Type

**Type**: JSONB (binary JSON storage)

**Usage**:
- API responses: NPI Registry API data
- Dynamic configurations: Import/rules configurations
- Feature data: ML feature contributions
- Error details: Validation/processing errors
- Performance metrics: Processing statistics

**Rationale**:
- Schema flexibility for varying data structures
- Indexable with GIN indexes
- Binary format faster than JSON
- Supports partial updates

**Alternative Rejected**: JSON (text-based)
- Reason: JSONB is faster to process, supports indexing, enables efficient queries

### Constraint Strategy

#### CHECK Constraints
- **Usage**: Data validation at database level
- **Examples**:
  - Date ranges: `CHECK (date_to >= date_from)`
  - Enum values: `CHECK (status IN ('OPEN', 'CLOSED', 'PENDING'))`
  - Numeric ranges: `CHECK (service_unit_count > 0 AND service_unit_count <= 9999.9)`
  - Codes: `CHECK (payer_responsibility_code IN ('P', 'S'))`

#### UNIQUE Constraints
- **Usage**: Enforce business rules
- **Examples**:
  - Composite uniques: `UNIQUE(organization_id, facility_code)`
  - Sequence uniqueness: `UNIQUE(encounter_id, sequence_number)`
  - Code uniqueness: `UNIQUE(npi)`, `UNIQUE(taxonomy_code)`

#### NOT NULL Constraints
- **Philosophy**: Only required business fields
- **Examples**: encounter_id, date_of_service_from, total_claim_charge_amount
- **Optional FKs**: Allow NULL for optional relationships (supervising provider, region)

### Denormalization Strategy

#### Provider Information
- **Denormalized**: NPI + name stored in encounter/service_line
- **Rationale**:
  - Avoid JOIN for common queries (80% of queries need provider name)
  - Historical accuracy (name at time of service)
  - Performance: 2-3x faster than JOIN

#### Financial Totals
- **Denormalized**: total_claim_charge_amount in encounter
- **Maintained By**: Triggers on service_line INSERT/UPDATE/DELETE (migration 014)
- **Rationale**: Dashboard queries need claim totals without service line aggregation

#### Date of Service
- **Denormalized**: date_of_service_from in raw_claims (extracted from JSONB)
- **Rationale**: FIFO ordering without JSONB extraction on every query

---

## Performance Considerations

### Identity Generation

**CACHE Values**:
- High-volume tables (encounter, service_line): CACHE 100
- Medium-volume tables (organization, provider): CACHE 50
- Low-volume tables (flag_category): CACHE 10

**Impact**: Reduces sequence lock contention in multi-worker environments

### Batch Processing

**Optimal Batch Size**: 750 claims per batch (based on Aegis production data - migration 024)

**Rationale**:
- Balance between transaction size and throughput
- Prevents long-running transactions
- Optimal for FIFO Sequential Completion Manager

### FIFO Processing

**Strict Mode**: Enabled via `staging.processing_configuration` (migration 024)
- Guarantees claims processed in chronological order per facility
- Prevents out-of-order processing that could violate business rules
- Uses sequence-controlled batch completion

**Architecture**: Aegis-inspired Sequential Completion Manager
- Multi-worker processing with strict ordering
- Batches assigned sequence numbers
- Completion enforced in sequential order
- Stuck sequence detection (default threshold: 5 minutes)

### Query Optimization

#### Materialized Views
- **Refresh**: CONCURRENTLY (no table locking)
- **Schedule**: Off-peak hours (nightly recommended)
- **Coverage**: 90 days of data (configurable)
- **Impact**: 10-100x faster dashboard queries

#### Partial Indexes
- **Usage**: 30% of all indexes
- **Benefit**: 40-60% smaller index size
- **Examples**: is_active, flag_status, queue_status filters

#### Covering Indexes
- **Usage**: High-traffic queries
- **Benefit**: Index-only scans (no table access)
- **Example**: Service line queries can be satisfied entirely from index

### Connection Pooling

**Recommended**: PgBouncer or similar
- **Mode**: Transaction pooling
- **Pool Size**: 50-100 connections per database
- **Rationale**: PostgreSQL performs best with limited active connections

### Vacuum Strategy

**Autovacuum**: Enabled (PostgreSQL default)

**Manual Vacuum**: Recommended weekly for high-churn tables
- Tables: raw_claims, import_batch, file_processing_queue
- Command: `VACUUM ANALYZE table_name;`

**Full Vacuum**: Monthly during maintenance window
- Purpose: Reclaim disk space from deleted rows
- Command: `VACUUM FULL ANALYZE;` (requires table lock)

### Partition Strategy

**Current**: No partitioning implemented

**Recommended Future**: Partition encounter table by date_of_service_from
- Method: Range partitioning by month or quarter
- Trigger: When encounter table exceeds 10M records
- Benefit: Faster queries, easier archival, improved maintenance

---

## Migration History

### Complete Migration Timeline

| Migration | Description | Date | Key Changes |
|-----------|-------------|------|-------------|
| 001 | Create schemas | 2025-10-14 | Created claims, staging, ml schemas. Enabled extensions (citext, pg_trgm, pgcrypto) |
| 002 | Organization tables | 2025-10-14 | organization, region, facility hierarchy |
| 003 | Provider tables | 2025-10-14 | provider, coder, reviewer tables |
| 004 | Encounter tables | 2025-10-14 | Main encounter table with 837p data elements |
| 005 | Diagnosis/procedure | 2025-10-14 | encounter_diagnosis, service_line, adjustments |
| 006 | Flag tables | 2025-10-14 | Flag system (categories, issues, encounter/line flags) |
| 007 | Staging tables | 2025-10-14 | Import pipeline (batch, config, rules, metrics, errors) |
| 008 | Audit tables | 2025-10-14 | Audit system (assignments, evaluations, accuracy tracking) |
| 009 | RVU tables | 2025-10-14 | RVU reference, conversion factors, GPCI, modifiers |
| 010 | Denial tables | 2025-10-14 | Denial tracking, appeals, reason codes, statistics |
| 011 | Schedule tables | 2025-10-14 | Scheduling tables (removed in migration 027) |
| 012 | ML tables | 2025-10-14 | ML registry, predictions, features, datasets |
| 013 | Dashboard views | 2025-10-14 | Comprehensive views for reporting |
| 014 | Utility functions | 2025-10-14 | Functions, triggers, additional indexes, materialized views |
| 015 | FIFO queue | 2025-10-14 | File processing queue with FIFO ordering |
| 016 | Phase 5 indexes | 2025-10-14 | Performance indexes for cache population (CONCURRENTLY) |
| 017 | Streaming progress | 2025-10-14 | Real-time progress tracking for streaming |
| 018 | Phase 6 indexes | 2025-10-15 | Strategic indexes for query optimization (CONCURRENTLY) |
| 019 | Materialized views | 2025-10-15 | Analytics materialized views, created analytics schema |
| 020 | Version tracking | 2025-10-20 | schema_migrations, application_version tables |
| 021 | Initial version | 2025-10-20 | Insert initial version record |
| 022 | Test upgrade | 2025-10-20 | Test migration for upgrade process |
| 023 | Raw claims table | 2025-10-24 | Two-stage processing pipeline (raw_claims) |
| 024 | Batch sequences | 2025-10-24 | Strict FIFO ordering with sequence tracking |
| 025 | Rename duration | 2025-10-24 | Column rename for consistency |
| 026 | Fix timestamps | 2025-10-24 | Timestamp column corrections |
| 027 | Drop scheduling | 2025-10-24 | Remove unused scheduling tables |
| 028 | Add project_id | 2025-10-24 | Add project_id to organization table |
| 029 | Drop charge constraints | 2025-10-24 | Remove CHECK constraints from charge amounts |
| 030 | Import headers | 2025-10-24 | import_headers table for header tracking |
| 031 | Delete project proc | 2025-10-24 | Stored procedure for project deletion |
| 032 | Claims detail view | 2025-10-24 | View for claims detail reporting |
| 033 | Field definitions | 2025-10-24 | field_definitions table for dynamic field mapping |
| 034 | Provider full name | 2025-10-24 | Add full_name column to provider table |
| 035 | Medical record number | 2025-10-24 | Add medical_record_number to encounter |
| 036 | Phase 3 segments | 2025-10-24 | Advanced 837p segment support |
| 037 | Phase 4 COB | 2025-10-24 | Coordination of benefits enhancements |
| 038 | Phase 5 specialized | 2025-10-24 | Specialized claim type support |
| 039 | Phase 6 loops | 2025-10-24 | Additional 837p loop support |
| 041 | Provider taxonomy | 2025-11-05 | NUCC taxonomy code reference (300+ codes) |
| 042 | Provider enrichment | 2025-11-05 | NPI Registry API enrichment queue |
| 043 | FK indexes | 2025-11-05 | Missing foreign key indexes (CONCURRENTLY) |
| 044 | Taxonomy FK | 2025-11-05 | Foreign key from provider to provider_taxonomy |
| 045 | Staging FKs | 2025-11-05 | Foreign keys for import tracking columns |

**Migration Count**: 45 migrations applied

**Database Version**: 1.0

**Schema Evolution**: Stable (20+ migrations since October 2024)

---

## Appendix: Quick Reference

### Table Count by Schema

| Schema | Tables | Views | Materialized Views |
|--------|--------|-------|-------------------|
| claims | 35 | 15+ | 0 |
| staging | 15 | 5+ | 0 |
| ml | 6 | 0 | 0 |
| analytics | 0 | 0 | 6 |
| **Total** | **56** | **20+** | **6** |

### Primary Key Data Type

All primary keys: **BIGINT GENERATED ALWAYS AS IDENTITY**

### Foreign Key Cascade Summary

| Cascade Type | Count | Usage |
|-------------|-------|-------|
| ON DELETE CASCADE | 40+ | Parent-child relationships |
| ON DELETE SET NULL | 15+ | Optional references |
| ON DELETE RESTRICT | 5+ | Reference data |

### Index Summary

| Index Type | Count | Purpose |
|-----------|-------|---------|
| B-tree | 180+ | General purpose |
| GIN | 5+ | Full-text search |
| Partial | 60+ | Filtered indexes |
| Composite | 80+ | Multi-column queries |
| Covering | 5+ | Index-only scans |

### Extension Dependencies

- **citext**: Case-insensitive text (email addresses)
- **pg_trgm**: Trigram matching for fuzzy search
- **pgcrypto**: Cryptographic functions (file hashing)

### Key Limits

| Resource | Limit |
|----------|-------|
| Max BIGINT value | 9,223,372,036,854,775,807 |
| Max financial amount | $999,999,999,999,999.99 |
| Max service units | 9999.9 |
| Max diagnosis pointers | 12 per service line |
| Max diagnosis codes | 12 per encounter |
| Max sequence number | 12 (diagnosis) |

---

**Document Prepared For**: DBA Review and System Documentation
**Database Platform**: PostgreSQL 14+
**Application**: Professional Smart Claims Processing System
**Maintained By**: Development Team
**Last Updated**: 2025-11-05
