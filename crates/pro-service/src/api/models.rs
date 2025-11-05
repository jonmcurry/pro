//! API response models for dashboard and analytics endpoints
//!
//! These structs map to the database views defined in migrations 013, 014, 015, and 019

use chrono::{NaiveDate, NaiveDateTime};
use serde::{Deserialize, Serialize};
use sqlx::{types::Decimal, FromRow};


// ============================================================================
// Dashboard View Models (Migration 013)
// ============================================================================

#[derive(Debug, Serialize, Deserialize, FromRow)]
pub struct ManagementOverview {
    pub organization_id: i64,
    pub organization_name: String,
    pub facility_id: Option<i64>,
    pub facility_name: Option<String>,
    pub period_month: NaiveDate,
    pub total_encounters: Option<i64>,
    pub total_service_lines: Option<i64>,
    pub active_providers: Option<i64>,
    pub active_coders: Option<i64>,
    pub total_billed_amount: Option<Decimal>,
    pub avg_claim_amount: Option<Decimal>,
    pub total_rvus: Option<Decimal>,
    pub estimated_medicare_payment: Option<Decimal>,
    pub encounters_with_flags: Option<i64>,
    pub total_flag_count: Option<i64>,
    pub high_severity_flags: Option<i64>,
    pub medium_severity_flags: Option<i64>,
    pub low_severity_flags: Option<i64>,
    pub flag_rate_percent: Option<Decimal>,
    pub total_denials: Option<i64>,
    pub denied_amount: Option<Decimal>,
    pub denial_rate_percent: Option<Decimal>,
}

#[derive(Debug, Serialize, Deserialize, FromRow)]
pub struct ClaimStatusSummary {
    pub organization_id: i64,
    pub facility_id: Option<i64>,
    pub claim_status: String,
    pub encounter_count: Option<i64>,
    pub total_billed_amount: Option<Decimal>,
    pub avg_billed_amount: Option<Decimal>,
}

#[derive(Debug, Serialize, Deserialize, FromRow)]
pub struct CoderPerformance {
    pub coder_id: i64,
    pub coder_name: String,
    pub organization_id: i64,
    pub encounters_coded: Option<i64>,
    pub service_lines_coded: Option<i64>,
    pub work_rvus: Option<Decimal>,
    pub total_rvus: Option<Decimal>,
    pub audits_conducted: Option<i64>,
    pub audits_passed: Option<i64>,
    pub audit_accuracy_rate: Option<Decimal>,
    pub critical_errors: Option<i64>,
    pub major_errors: Option<i64>,
    pub minor_errors: Option<i64>,
    pub total_overpayment: Option<Decimal>,
    pub total_underpayment: Option<Decimal>,
    pub flags_generated: Option<i64>,
    pub flags_accepted: Option<i64>,
    pub avg_encounters_per_day: Option<Decimal>,
}

#[derive(Debug, Serialize, Deserialize, FromRow)]
pub struct ProviderDocumentationAccuracy {
    pub provider_id: i64,
    pub provider_name: String,
    pub organization_id: i64,
    pub facility_id: Option<i64>,
    pub encounters_documented: Option<i64>,
    pub service_lines: Option<i64>,
    pub audits_reviewed: Option<i64>,
    pub documentation_accuracy_rate: Option<Decimal>,
    pub overcoding_instances: Option<i64>,
    pub undercoding_instances: Option<i64>,
    pub unsupported_instances: Option<i64>,
    pub total_overpayment_risk: Option<Decimal>,
    pub total_underpayment_risk: Option<Decimal>,
}

#[derive(Debug, Serialize, Deserialize, FromRow)]
pub struct FlagsByCategory {
    pub organization_id: i64,
    pub facility_id: Option<i64>,
    pub flag_category: String,
    pub issue_type: String,
    pub severity_level: String,
    pub flag_count: Option<i64>,
    pub open_flags: Option<i64>,
    pub resolved_flags: Option<i64>,
    pub accepted_flags: Option<i64>,
    pub rejected_flags: Option<i64>,
    pub resolution_rate_percent: Option<Decimal>,
    pub avg_resolution_time_hours: Option<Decimal>,
}

#[derive(Debug, Serialize, Deserialize, FromRow)]
pub struct ServiceLineFlagsDetail {
    pub flag_id: i64,
    pub encounter_id: i64,
    pub service_line_number: i32,
    pub flag_category: String,
    pub issue_type: String,
    pub severity_level: String,
    pub flag_description: String,
    pub proposed_correction: Option<String>,
    pub procedure_code: String,
    pub charged_amount: Decimal,
    pub flag_status: String,
    pub resolution_notes: Option<String>,
    pub coder_id: Option<i64>,
    pub coder_name: Option<String>,
    pub provider_id: i64,
    pub provider_name: String,
    pub facility_name: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, FromRow)]
pub struct DenialByPayer {
    pub organization_id: i64,
    pub facility_id: Option<i64>,
    pub payer_id: i64,
    pub payer_name: String,
    pub period_month: NaiveDate,
    pub denial_count: Option<i64>,
    pub denied_amount: Option<Decimal>,
    pub total_billed_amount: Option<Decimal>,
    pub denial_rate_percent: Option<Decimal>,
    pub coding_error_denials: Option<i64>,
    pub documentation_denials: Option<i64>,
    pub authorization_denials: Option<i64>,
    pub timely_filing_denials: Option<i64>,
    pub other_denials: Option<i64>,
    pub preventable_denials: Option<i64>,
    pub appeals_filed: Option<i64>,
    pub appeals_overturned: Option<i64>,
    pub appeal_success_rate: Option<Decimal>,
}

#[derive(Debug, Serialize, Deserialize, FromRow)]
pub struct DenialByReason {
    pub organization_id: i64,
    pub facility_id: Option<i64>,
    pub denial_reason_code: String,
    pub denial_reason_description: Option<String>,
    pub denial_count: Option<i64>,
    pub denied_amount: Option<Decimal>,
    pub preventable_count: Option<i64>,
    pub overturned_count: Option<i64>,
    pub written_off_count: Option<i64>,
}

#[derive(Debug, Serialize, Deserialize, FromRow)]
pub struct ProcedureVolume {
    pub organization_id: i64,
    pub facility_id: Option<i64>,
    pub procedure_code: String,
    pub procedure_description: Option<String>,
    pub procedure_count: Option<i64>,
    pub total_units: Option<Decimal>,
    pub total_charges: Option<Decimal>,
    pub total_rvus: Option<Decimal>,
    pub estimated_payment: Option<Decimal>,
    pub flag_count: Option<i64>,
    pub flag_rate_percent: Option<Decimal>,
}

#[derive(Debug, Serialize, Deserialize, FromRow)]
pub struct ProviderProductivity {
    pub provider_id: i64,
    pub provider_name: String,
    pub organization_id: i64,
    pub facility_id: Option<i64>,
    pub period_month: NaiveDate,
    pub encounter_count: Option<i64>,
    pub service_line_count: Option<i64>,
    pub total_charges: Option<Decimal>,
    pub total_work_rvus: Option<Decimal>,
    pub total_rvus: Option<Decimal>,
    pub estimated_collections: Option<Decimal>,
    pub avg_rvus_per_encounter: Option<Decimal>,
    pub em_visits: Option<i64>,
    pub non_em_procedures: Option<i64>,
}

#[derive(Debug, Serialize, Deserialize, FromRow)]
pub struct AuditAssignmentStatus {
    pub assignment_id: i64,
    pub organization_id: i64,
    pub facility_id: Option<i64>,
    pub audit_type: String,
    pub sample_size: i32,
    pub completed_count: Option<i64>,
    pub completion_percent: Option<Decimal>,
    pub errors_found: Option<i64>,
    pub error_rate: Option<Decimal>,
    pub flags_generated: Option<i64>,
    pub total_overpayment: Option<Decimal>,
    pub total_underpayment: Option<Decimal>,
    pub reviewer_id: Option<i64>,
    pub reviewer_name: Option<String>,
    pub assigned_date: NaiveDate,
    pub due_date: NaiveDate,
    pub days_in_progress: Option<i32>,
    pub days_until_due: Option<i32>,
}

#[derive(Debug, Serialize, Deserialize, FromRow)]
pub struct ReimbursementAnalysis {
    pub organization_id: i64,
    pub facility_id: Option<i64>,
    pub period_month: NaiveDate,
    pub encounter_count: Option<i64>,
    pub service_line_count: Option<i64>,
    pub total_charges: Option<Decimal>,
    pub total_rvus: Option<Decimal>,
    pub rvu_based_estimate: Option<Decimal>,
    pub charge_to_rvu_ratio: Option<Decimal>,
    pub denial_count: Option<i64>,
    pub denial_amount: Option<Decimal>,
    pub net_expected_payment: Option<Decimal>,
}

// ============================================================================
// Queue Monitoring Models (Migration 015)
// ============================================================================

#[derive(Debug, Serialize, Deserialize, FromRow)]
pub struct QueueHealth {
    pub facility_id: Option<i64>,
    pub facility_name: Option<String>,
    pub queued_count: Option<i64>,
    pub processing_count: Option<i64>,
    pub completed_count: Option<i64>,
    pub failed_count: Option<i64>,
    pub oldest_queued: Option<NaiveDateTime>,
    pub newest_queued: Option<NaiveDateTime>,
    pub avg_processing_time_seconds: Option<Decimal>,
    pub max_processing_time_seconds: Option<Decimal>,
}

#[derive(Debug, Serialize, Deserialize, FromRow)]
pub struct QueueStatistics {
    pub facility_id: Option<i64>,
    pub facility_name: Option<String>,
    pub hour: Option<NaiveDateTime>,
    pub files_queued: Option<i64>,
    pub files_completed: Option<i64>,
    pub files_failed: Option<i64>,
    pub completion_rate: Option<Decimal>,
    pub avg_queue_wait_seconds: Option<Decimal>,
    pub avg_processing_seconds: Option<Decimal>,
}

#[derive(Debug, Serialize, Deserialize, FromRow)]
pub struct FifoViolation {
    pub organization_id: i64,
    pub facility_id: Option<i64>,
    pub earlier_encounter_id: i64,
    pub earlier_service_date: NaiveDate,
    pub earlier_created_at: NaiveDateTime,
    pub later_encounter_id: i64,
    pub later_service_date: NaiveDate,
    pub later_created_at: NaiveDateTime,
    pub time_gap_hours: Option<Decimal>,
}

// ============================================================================
// Analytics Models (Migration 019 - Materialized Views)
// ============================================================================

#[derive(Debug, Serialize, Deserialize, FromRow)]
pub struct FlagStatisticsDaily {
    pub organization_id: i64,
    pub facility_id: Option<i64>,
    pub flag_date: NaiveDate,
    pub flag_category: String,
    pub severity_level: String,
    pub flag_count: Option<i64>,
    pub resolved_count: Option<i64>,
    pub accepted_count: Option<i64>,
    pub median_resolution_hours: Option<Decimal>,
    pub total_financial_impact: Option<Decimal>,
}

#[derive(Debug, Serialize, Deserialize, FromRow)]
pub struct EncounterStatisticsDaily {
    pub organization_id: i64,
    pub facility_id: Option<i64>,
    pub encounter_date: NaiveDate,
    pub claim_status: String,
    pub payer_id: i64,
    pub encounter_count: Option<i64>,
    pub total_charges: Option<Decimal>,
    pub total_rvus: Option<Decimal>,
    pub service_line_count: Option<i64>,
}

#[derive(Debug, Serialize, Deserialize, FromRow)]
pub struct ProcedureStatistics {
    pub organization_id: i64,
    pub facility_id: Option<i64>,
    pub procedure_code: String,
    pub procedure_count: Option<i64>,
    pub total_charges: Option<Decimal>,
    pub total_rvus: Option<Decimal>,
    pub flag_count: Option<i64>,
    pub flag_rate: Option<Decimal>,
    pub common_modifiers: Option<Vec<String>>,
}

#[derive(Debug, Serialize, Deserialize, FromRow)]
pub struct ProviderPerformance {
    pub organization_id: i64,
    pub facility_id: Option<i64>,
    pub provider_id: i64,
    pub encounter_count: Option<i64>,
    pub total_charges: Option<Decimal>,
    pub total_rvus: Option<Decimal>,
    pub flag_count: Option<i64>,
    pub flags_per_encounter: Option<Decimal>,
}

#[derive(Debug, Serialize, Deserialize, FromRow)]
pub struct PayerStatistics {
    pub organization_id: i64,
    pub facility_id: Option<i64>,
    pub payer_id: i64,
    pub encounter_count: Option<i64>,
    pub total_charges: Option<Decimal>,
    pub denial_count: Option<i64>,
    pub denial_rate: Option<Decimal>,
    pub top_procedures: Option<Vec<String>>,
}

#[derive(Debug, Serialize, Deserialize, FromRow)]
pub struct MlModelPerformance {
    pub model_name: String,
    pub prediction_count: Option<i64>,
    pub correct_predictions: Option<i64>,
    pub accuracy: Option<Decimal>,
    pub avg_confidence_score: Option<Decimal>,
}

// ============================================================================
// Common Query Parameters
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct DashboardQueryParams {
    pub organization_id: Option<i64>,
    pub facility_id: Option<i64>,
    pub start_date: Option<NaiveDate>,
    pub end_date: Option<NaiveDate>,
    pub limit: Option<i64>,
}
