// Database models that map directly to PostgreSQL tables

use chrono::{DateTime, NaiveDate, Utc};
use serde::{Deserialize, Serialize};
use sqlx::FromRow;

use rust_decimal::Decimal;

// ============================================================================
// ORGANIZATION HIERARCHY
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Organization {
    pub organization_id: i64,
    pub organization_code: String,
    pub organization_name: String,
    pub tax_id: Option<String>,
    pub npi: Option<String>,
    pub address_line1: Option<String>,
    pub address_line2: Option<String>,
    pub city: Option<String>,
    pub state_code: Option<String>,
    pub postal_code: Option<String>,
    pub country_code: Option<String>,
    pub phone: Option<String>,
    pub email: Option<String>,
    pub is_active: bool,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Region {
    pub region_id: i64,
    pub organization_id: i64,
    pub region_code: String,
    pub region_name: String,
    pub description: Option<String>,
    pub is_active: bool,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Facility {
    pub facility_id: i64,
    pub organization_id: i64,
    pub region_id: Option<i64>,
    pub facility_code: String,
    pub facility_name: String,
    pub npi: Option<String>,
    pub tax_id: Option<String>,
    pub facility_type: Option<String>,
    pub address_line1: Option<String>,
    pub address_line2: Option<String>,
    pub city: Option<String>,
    pub state_code: Option<String>,
    pub postal_code: Option<String>,
    pub country_code: Option<String>,
    pub phone: Option<String>,
    pub email: Option<String>,
    pub ehr_system: Option<String>,
    pub is_active: bool,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

// ============================================================================
// PROVIDERS AND PERSONNEL
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Provider {
    pub provider_id: i64,
    pub npi: String,
    pub provider_type: String,
    pub last_name: String,
    pub first_name: String,
    pub middle_name: Option<String>,
    pub name_suffix: Option<String>,
    pub taxonomy_code: Option<String>,
    pub license_number: Option<String>,
    pub license_state: Option<String>,
    pub specialty: Option<String>,
    pub provider_group: Option<String>,
    pub organization_id: Option<i64>,
    pub address_line1: Option<String>,
    pub address_line2: Option<String>,
    pub city: Option<String>,
    pub state_code: Option<String>,
    pub postal_code: Option<String>,
    pub country_code: Option<String>,
    pub phone: Option<String>,
    pub email: Option<String>,
    pub is_active: bool,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Coder {
    pub coder_id: i64,
    pub coder_code: String,
    pub last_name: String,
    pub first_name: String,
    pub middle_name: Option<String>,
    pub coder_group: Option<String>,
    pub certifications: Vec<String>,
    pub organization_id: Option<i64>,
    pub email: Option<String>,
    pub is_active: bool,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Reviewer {
    pub reviewer_id: i64,
    pub reviewer_code: String,
    pub last_name: String,
    pub first_name: String,
    pub middle_name: Option<String>,
    pub reviewer_group: Option<String>,
    pub certifications: Vec<String>,
    pub organization_id: Option<i64>,
    pub email: Option<String>,
    pub is_active: bool,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

// ============================================================================
// ENCOUNTERS AND SERVICE LINES
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Encounter {
    pub encounter_id: i64,
    pub facility_id: i64,
    pub organization_id: i64,
    pub region_id: Option<i64>,

    // Submitter information
    pub submitter_id: String,
    pub submitter_name: Option<String>,

    // Control numbers
    pub patient_control_number: String,
    pub transaction_set_control_number: Option<String>,

    // Patient/Subscriber information
    pub subscriber_id: String,
    pub subscriber_last_name: String,
    pub subscriber_first_name: String,
    pub subscriber_middle_name: Option<String>,
    pub subscriber_name_suffix: Option<String>,
    pub subscriber_gender: Option<String>,
    pub subscriber_birth_date: NaiveDate,
    pub subscriber_address_line1: Option<String>,
    pub subscriber_address_line2: Option<String>,
    pub subscriber_city: Option<String>,
    pub subscriber_state: Option<String>,
    pub subscriber_postal_code: Option<String>,
    pub subscriber_country: Option<String>,

    // Payer information
    pub payer_responsibility_code: String,
    pub payer_id: Option<String>,
    pub payer_name: Option<String>,
    pub claim_filing_indicator: Option<String>,

    // Billing provider
    pub billing_provider_id: Option<i64>,
    pub billing_provider_npi: Option<String>,
    pub billing_provider_tax_id: Option<String>,
    pub billing_provider_name: Option<String>,

    // Claim information
    pub total_claim_charge_amount: Decimal,
    pub place_of_service_code: Option<String>,
    pub claim_frequency_code: Option<String>,

    // Dates
    pub date_of_service_from: NaiveDate,
    pub date_of_service_to: Option<NaiveDate>,

    // Providers
    pub referring_provider_id: Option<i64>,
    pub referring_provider_npi: Option<String>,
    pub rendering_provider_id: Option<i64>,
    pub rendering_provider_npi: Option<String>,
    pub supervising_provider_id: Option<i64>,
    pub supervising_provider_npi: Option<String>,

    // Service facility
    pub service_facility_id: Option<i64>,
    pub service_facility_npi: Option<String>,

    // Coder information
    pub coder_id: Option<i64>,
    pub coding_date: Option<NaiveDate>,

    // Status and workflow
    pub claim_status: String,
    pub case_status: Option<String>,
    pub financial_class: Option<String>,

    // Import tracking
    pub import_batch_id: Option<i64>,
    pub import_date: Option<DateTime<Utc>>,

    // Audit trail
    pub is_active: bool,
    pub soft_deleted: bool,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub created_by: Option<String>,
    pub updated_by: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct ServiceLine {
    pub service_line_id: i64,
    pub encounter_id: i64,
    pub line_number: i16,

    // Service information
    pub product_service_id_qualifier: Option<String>,
    pub procedure_code: String,
    pub procedure_modifier_1: Option<String>,
    pub procedure_modifier_2: Option<String>,
    pub procedure_modifier_3: Option<String>,
    pub procedure_modifier_4: Option<String>,
    pub procedure_description: Option<String>,
    pub line_item_charge_amount: Decimal,
    pub unit_basis_measurement_code: Option<String>,
    pub service_unit_count: Decimal,

    // Place of service
    pub place_of_service_code: Option<String>,

    // Dates
    pub service_date_from: NaiveDate,
    pub service_date_to: Option<NaiveDate>,

    // Providers at line level
    pub rendering_provider_id: Option<i64>,
    pub rendering_provider_npi: Option<String>,
    pub supervising_provider_id: Option<i64>,
    pub supervising_provider_npi: Option<String>,
    pub ordering_provider_id: Option<i64>,
    pub ordering_provider_npi: Option<String>,
    pub referring_provider_id: Option<i64>,
    pub referring_provider_npi: Option<String>,

    // Service facility at line level
    pub service_facility_id: Option<i64>,
    pub service_facility_npi: Option<String>,

    // Prior authorization and referral
    pub prior_authorization_number: Option<String>,
    pub referral_number: Option<String>,

    // Line note
    pub line_note: Option<String>,

    // Revenue code
    pub revenue_code: Option<String>,

    // NDC information
    pub ndc_code: Option<String>,
    pub ndc_unit_count: Option<Decimal>,
    pub ndc_measurement_unit: Option<String>,

    // Diagnosis pointers
    pub diagnosis_code_pointer_1: Option<i16>,
    pub diagnosis_code_pointer_2: Option<i16>,
    pub diagnosis_code_pointer_3: Option<i16>,
    pub diagnosis_code_pointer_4: Option<i16>,

    // Status
    pub line_status: String,

    // Audit trail
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub created_by: Option<String>,
    pub updated_by: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct EncounterDiagnosis {
    pub diagnosis_id: i64,
    pub encounter_id: i64,
    pub sequence_number: i16,
    pub diagnosis_code_qualifier: Option<String>,
    pub diagnosis_code: String,
    pub diagnosis_description: Option<String>,
    pub is_principal: bool,
    pub is_admitting: bool,
    pub is_external_cause: bool,
    pub is_patient_reason: bool,
    pub present_on_admission_indicator: Option<String>,
    pub hcc_indicator: bool,
    pub hcc_category: Option<String>,
    pub created_at: DateTime<Utc>,
}

// ============================================================================
// FLAGS
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct FlagCategory {
    pub category_id: i64,
    pub category_code: String,
    pub category_name: String,
    pub category_description: Option<String>,
    pub is_active: bool,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct FlagIssue {
    pub issue_id: i64,
    pub category_id: i64,
    pub issue_code: String,
    pub issue_description: String,
    pub severity: String,
    pub is_active: bool,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct EncounterFlag {
    pub flag_id: i64,
    pub encounter_id: i64,
    pub issue_id: i64,
    pub flag_type: String,
    pub severity: Option<String>,
    pub flag_reason: Option<String>,
    pub flagged_element: Option<String>,
    pub proposed_code: Option<String>,
    pub proposed_modifier: Option<String>,
    pub proposed_quantity: Option<Decimal>,
    pub proposed_diagnosis_code: Option<String>,
    pub flag_status: String,
    pub resolution_note: Option<String>,
    pub resolved_at: Option<DateTime<Utc>>,
    pub resolved_by: Option<String>,
    pub created_at: DateTime<Utc>,
    pub created_by: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct ServiceLineFlag {
    pub flag_id: i64,
    pub service_line_id: i64,
    pub issue_id: i64,
    pub flag_type: String,
    pub severity: Option<String>,
    pub flag_reason: Option<String>,
    pub flagged_element: Option<String>,
    pub proposed_code: Option<String>,
    pub proposed_modifier: Option<String>,
    pub proposed_quantity: Option<Decimal>,
    pub flag_status: String,
    pub resolution_note: Option<String>,
    pub resolved_at: Option<DateTime<Utc>>,
    pub resolved_by: Option<String>,
    pub created_at: DateTime<Utc>,
    pub created_by: String,
}

// ============================================================================
// IMPORT BATCH
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct ImportBatch {
    pub batch_id: i64,
    pub organization_id: i64,
    pub facility_id: Option<i64>,
    pub batch_name: Option<String>,
    pub batch_type: String,
    pub file_format: Option<String>,
    pub original_filename: Option<String>,
    pub file_path: Option<String>,
    pub file_size_bytes: Option<i64>,
    pub file_hash: Option<String>,
    pub import_status: String,
    pub total_records: i32,
    pub processed_records: i32,
    pub successful_records: i32,
    pub failed_records: i32,
    pub skipped_records: i32,
    pub duplicate_records: i32,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    pub processing_duration_seconds: Option<Decimal>,
    pub configuration_id: Option<i64>,
    pub rules_applied: bool,
    pub error_message: Option<String>,
    pub created_at: DateTime<Utc>,
    pub created_by: Option<String>,
}

// ============================================================================
// RVU REFERENCE
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct RvuReference {
    pub rvu_id: i64,
    pub hcpcs_code: String,
    pub modifier: Option<String>,
    pub effective_year: i32,
    pub effective_date: NaiveDate,
    pub termination_date: Option<NaiveDate>,
    pub work_rvu: Decimal,
    pub pe_rvu_nonfacility: Decimal,
    pub pe_rvu_facility: Decimal,
    pub mp_rvu: Decimal,
    pub total_rvu_nonfacility: Decimal,
    pub total_rvu_facility: Decimal,
    pub status_code: Option<String>,
    pub global_surgery_indicator: Option<String>,
    pub short_description: Option<String>,
    pub long_description: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct ConversionFactor {
    pub conversion_factor_id: i64,
    pub factor_year: i32,
    pub effective_date: NaiveDate,
    pub termination_date: Option<NaiveDate>,
    pub conversion_factor: Decimal,
    pub budget_neutrality_adjustment: Decimal,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub created_by: Option<String>,
}

// ============================================================================
// DENIAL EVENT
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct DenialEvent {
    pub denial_id: i64,
    pub encounter_id: i64,
    pub service_line_id: Option<i64>,
    pub organization_id: i64,
    pub facility_id: Option<i64>,
    pub denial_type: String,
    pub denial_category: String,
    pub payer_id: Option<String>,
    pub payer_name: Option<String>,
    pub claim_filing_indicator: Option<String>,
    pub claim_adjustment_group_code: Option<String>,
    pub claim_adjustment_reason_code: String,
    pub remittance_advice_remark_code: Option<String>,
    pub denial_reason_description: Option<String>,
    pub payer_denial_reason: Option<String>,
    pub denied_amount: Decimal,
    pub billed_amount: Option<Decimal>,
    pub allowed_amount: Option<Decimal>,
    pub paid_amount: Decimal,
    pub service_date: NaiveDate,
    pub initial_submission_date: Option<NaiveDate>,
    pub denial_date: NaiveDate,
    pub received_date: Option<NaiveDate>,
    pub root_cause_category: Option<String>,
    pub root_cause_subcategory: Option<String>,
    pub root_cause_details: Option<String>,
    pub responsible_party: Option<String>,
    pub coder_id: Option<i64>,
    pub provider_id: Option<i64>,
    pub is_preventable: Option<bool>,
    pub preventable_category: Option<String>,
    pub prevention_recommendations: Option<String>,
    pub denial_status: String,
    pub resolution_status: Option<String>,
    pub resolution_date: Option<NaiveDate>,
    pub appeal_filed: bool,
    pub appeal_level: Option<String>,
    pub appeal_deadline: Option<NaiveDate>,
    pub internal_notes: Option<String>,
    pub resolution_notes: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub created_by: Option<String>,
    pub updated_by: Option<String>,
}
