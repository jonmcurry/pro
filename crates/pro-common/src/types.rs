use chrono::{DateTime, NaiveDate, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Organization entity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Organization {
    pub organization_id: Uuid,
    pub organization_code: String,
    pub organization_name: String,
    pub tax_id: Option<String>,
    pub npi: Option<String>,
    pub is_active: bool,
}

/// Facility entity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Facility {
    pub facility_id: Uuid,
    pub organization_id: Uuid,
    pub region_id: Option<Uuid>,
    pub facility_code: String,
    pub facility_name: String,
    pub npi: Option<String>,
    pub is_active: bool,
}

/// Provider entity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Provider {
    pub provider_id: Uuid,
    pub npi: String,
    pub provider_type: String,
    pub last_name: String,
    pub first_name: String,
    pub middle_name: Option<String>,
    pub specialty: Option<String>,
    pub is_active: bool,
}

/// Coder entity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Coder {
    pub coder_id: Uuid,
    pub coder_code: String,
    pub last_name: String,
    pub first_name: String,
    pub certifications: Vec<String>,
    pub is_active: bool,
}

/// Encounter (claim) entity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Encounter {
    pub encounter_id: Uuid,
    pub facility_id: Uuid,
    pub organization_id: Uuid,
    pub patient_control_number: String,
    pub subscriber_id: String,
    pub subscriber_last_name: String,
    pub subscriber_first_name: String,
    pub subscriber_birth_date: NaiveDate,
    pub total_claim_charge_amount: rust_decimal::Decimal,
    pub date_of_service_from: NaiveDate,
    pub date_of_service_to: Option<NaiveDate>,
    pub claim_status: String,
    pub billing_provider_npi: Option<String>,
    pub rendering_provider_npi: Option<String>,
    pub coder_id: Option<Uuid>,
    pub coding_date: Option<NaiveDate>,
}

/// Service line entity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceLine {
    pub service_line_id: Uuid,
    pub encounter_id: Uuid,
    pub line_number: i16,
    pub procedure_code: String,
    pub procedure_modifier_1: Option<String>,
    pub procedure_modifier_2: Option<String>,
    pub procedure_modifier_3: Option<String>,
    pub procedure_modifier_4: Option<String>,
    pub procedure_description: Option<String>,
    pub line_item_charge_amount: rust_decimal::Decimal,
    pub service_unit_count: rust_decimal::Decimal,
    pub service_date_from: NaiveDate,
    pub service_date_to: Option<NaiveDate>,
    pub place_of_service_code: Option<String>,
}

/// Diagnosis entity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Diagnosis {
    pub diagnosis_id: Uuid,
    pub encounter_id: Uuid,
    pub sequence_number: i16,
    pub diagnosis_code: String,
    pub diagnosis_description: Option<String>,
    pub is_principal: bool,
}

/// Flag category
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlagCategory {
    pub category_id: Uuid,
    pub category_code: String,
    pub category_name: String,
    pub category_description: Option<String>,
}

/// Flag issue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlagIssue {
    pub issue_id: Uuid,
    pub category_id: Uuid,
    pub issue_code: String,
    pub issue_description: String,
    pub severity: String,
}

/// Encounter flag
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncounterFlag {
    pub flag_id: Uuid,
    pub encounter_id: Uuid,
    pub issue_id: Uuid,
    pub flag_type: String,
    pub severity: Option<String>,
    pub flag_reason: Option<String>,
    pub flagged_element: Option<String>,
    pub flag_status: String,
    pub created_at: DateTime<Utc>,
}

/// Service line flag
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceLineFlag {
    pub flag_id: Uuid,
    pub service_line_id: Uuid,
    pub issue_id: Uuid,
    pub flag_type: String,
    pub severity: Option<String>,
    pub flag_reason: Option<String>,
    pub flagged_element: Option<String>,
    pub proposed_code: Option<String>,
    pub proposed_modifier: Option<String>,
    pub flag_status: String,
    pub created_at: DateTime<Utc>,
}

/// Import batch
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportBatch {
    pub batch_id: Uuid,
    pub organization_id: Uuid,
    pub facility_id: Option<Uuid>,
    pub batch_name: Option<String>,
    pub batch_type: String,
    pub file_format: Option<String>,
    pub original_filename: Option<String>,
    pub import_status: String,
    pub total_records: i32,
    pub processed_records: i32,
    pub successful_records: i32,
    pub failed_records: i32,
    pub created_at: DateTime<Utc>,
}

/// Denial event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DenialEvent {
    pub denial_id: Uuid,
    pub encounter_id: Uuid,
    pub service_line_id: Option<Uuid>,
    pub organization_id: Uuid,
    pub denial_type: String,
    pub denial_category: String,
    pub claim_adjustment_reason_code: String,
    pub denied_amount: rust_decimal::Decimal,
    pub denial_date: NaiveDate,
    pub denial_status: String,
    pub is_preventable: Option<bool>,
    pub root_cause_category: Option<String>,
}

/// RVU reference data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RvuReference {
    pub rvu_id: Uuid,
    pub hcpcs_code: String,
    pub modifier: Option<String>,
    pub effective_year: i32,
    pub work_rvu: rust_decimal::Decimal,
    pub pe_rvu_nonfacility: rust_decimal::Decimal,
    pub pe_rvu_facility: rust_decimal::Decimal,
    pub mp_rvu: rust_decimal::Decimal,
    pub total_rvu_nonfacility: rust_decimal::Decimal,
    pub total_rvu_facility: rust_decimal::Decimal,
}

/// Conversion factor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversionFactor {
    pub conversion_factor_id: Uuid,
    pub factor_year: i32,
    pub effective_date: NaiveDate,
    pub conversion_factor: rust_decimal::Decimal,
}

/// Audit assignment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditAssignment {
    pub audit_id: Uuid,
    pub organization_id: Uuid,
    pub facility_id: Option<Uuid>,
    pub audit_name: String,
    pub audit_type: String,
    pub audit_scope: String,
    pub sample_size: Option<i32>,
    pub reviewer_id: Option<Uuid>,
    pub audit_status: String,
    pub due_date: Option<NaiveDate>,
}

/// Claim status enum
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClaimStatus {
    New,
    Pending,
    Flagged,
    Reviewed,
    Accepted,
    Rejected,
}

impl std::fmt::Display for ClaimStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ClaimStatus::New => write!(f, "NEW"),
            ClaimStatus::Pending => write!(f, "PENDING"),
            ClaimStatus::Flagged => write!(f, "FLAGGED"),
            ClaimStatus::Reviewed => write!(f, "REVIEWED"),
            ClaimStatus::Accepted => write!(f, "ACCEPTED"),
            ClaimStatus::Rejected => write!(f, "REJECTED"),
        }
    }
}

/// Flag status enum
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FlagStatus {
    Open,
    Reviewed,
    Accepted,
    Rejected,
    Resolved,
}

impl std::fmt::Display for FlagStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FlagStatus::Open => write!(f, "OPEN"),
            FlagStatus::Reviewed => write!(f, "REVIEWED"),
            FlagStatus::Accepted => write!(f, "ACCEPTED"),
            FlagStatus::Rejected => write!(f, "REJECTED"),
            FlagStatus::Resolved => write!(f, "RESOLVED"),
        }
    }
}

/// Import status enum
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImportStatus {
    Pending,
    Processing,
    Completed,
    Failed,
    Partial,
}

impl std::fmt::Display for ImportStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ImportStatus::Pending => write!(f, "PENDING"),
            ImportStatus::Processing => write!(f, "PROCESSING"),
            ImportStatus::Completed => write!(f, "COMPLETED"),
            ImportStatus::Failed => write!(f, "FAILED"),
            ImportStatus::Partial => write!(f, "PARTIAL"),
        }
    }
}

/// File type enum
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FileType {
    Edi837P,
    Csv,
}

impl std::fmt::Display for FileType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FileType::Edi837P => write!(f, "EDI_837P"),
            FileType::Csv => write!(f, "CSV"),
        }
    }
}
