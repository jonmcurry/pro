/// Flag category codes from SRD
pub mod flag_categories {
    pub const COD: &str = "COD"; // Coding Issues
    pub const DOC: &str = "DOC"; // Documentation Issues
    pub const EMO: &str = "EMO"; // E/M Over-coded
    pub const EMU: &str = "EMU"; // E/M Under-coded
    pub const EMI: &str = "EMI"; // E/M Incorrect Category
    pub const EMT: &str = "EMT"; // E/M Time Not Documented
    pub const MOD: &str = "MOD"; // Modifier Issues
    pub const OTH: &str = "OTH"; // Other Issues
    pub const QTY: &str = "QTY"; // Quantity Issues
    pub const SUP: &str = "SUP"; // Supervision Requirements
    pub const DX: &str = "DX"; // Diagnosis Issues
}

/// Flag severity levels
pub mod flag_severity {
    pub const HIGH: &str = "HIGH";
    pub const MEDIUM: &str = "MEDIUM";
    pub const LOW: &str = "LOW";
}

/// Claim statuses
pub mod claim_status {
    pub const NEW: &str = "NEW";
    pub const PENDING: &str = "PENDING";
    pub const FLAGGED: &str = "FLAGGED";
    pub const REVIEWED: &str = "REVIEWED";
    pub const ACCEPTED: &str = "ACCEPTED";
    pub const REJECTED: &str = "REJECTED";
}

/// Flag statuses
pub mod flag_status {
    pub const OPEN: &str = "OPEN";
    pub const REVIEWED: &str = "REVIEWED";
    pub const ACCEPTED: &str = "ACCEPTED";
    pub const REJECTED: &str = "REJECTED";
    pub const RESOLVED: &str = "RESOLVED";
}

/// Import batch types
pub mod batch_type {
    pub const EDI_837P: &str = "EDI_837P";
    pub const CSV: &str = "CSV";
    pub const MANUAL: &str = "MANUAL";
}

/// Import statuses
pub mod import_status {
    pub const PENDING: &str = "PENDING";
    pub const PROCESSING: &str = "PROCESSING";
    pub const COMPLETED: &str = "COMPLETED";
    pub const FAILED: &str = "FAILED";
    pub const PARTIAL: &str = "PARTIAL";
}

/// Audit types
pub mod audit_type {
    pub const RANDOM: &str = "RANDOM";
    pub const TARGETED: &str = "TARGETED";
    pub const PROVIDER_SPECIFIC: &str = "PROVIDER_SPECIFIC";
    pub const PROCEDURE_SPECIFIC: &str = "PROCEDURE_SPECIFIC";
}

/// Audit statuses
pub mod audit_status {
    pub const ASSIGNED: &str = "ASSIGNED";
    pub const IN_PROGRESS: &str = "IN_PROGRESS";
    pub const COMPLETED: &str = "COMPLETED";
    pub const CANCELLED: &str = "CANCELLED";
}

/// Denial categories
pub mod denial_category {
    pub const SOFT: &str = "SOFT";
    pub const HARD: &str = "HARD";
    pub const PREVENTABLE: &str = "PREVENTABLE";
    pub const NON_PREVENTABLE: &str = "NON_PREVENTABLE";
}

/// Root cause categories
pub mod root_cause {
    pub const CODING: &str = "CODING";
    pub const DOCUMENTATION: &str = "DOCUMENTATION";
    pub const AUTHORIZATION: &str = "AUTHORIZATION";
    pub const ELIGIBILITY: &str = "ELIGIBILITY";
    pub const TIMELY_FILING: &str = "TIMELY_FILING";
    pub const MEDICAL_NECESSITY: &str = "MEDICAL_NECESSITY";
    pub const DUPLICATE: &str = "DUPLICATE";
}

/// Place of Service codes (common ones)
pub mod place_of_service {
    pub const OFFICE: &str = "11";
    pub const HOME: &str = "12";
    pub const INPATIENT_HOSPITAL: &str = "21";
    pub const OUTPATIENT_HOSPITAL: &str = "22";
    pub const EMERGENCY_ROOM: &str = "23";
    pub const AMBULATORY_SURGICAL_CENTER: &str = "24";
    pub const NURSING_FACILITY: &str = "32";
}

/// Payer responsibility codes
pub mod payer_responsibility {
    pub const PRIMARY: &str = "P";
    pub const SECONDARY: &str = "S";
}

/// Claim filing indicator codes
pub mod claim_filing {
    pub const MEDICARE_PART_B: &str = "MB";
    pub const MEDICARE_PART_A: &str = "MA";
    pub const COMMERCIAL: &str = "CI";
    pub const MEDICAID: &str = "MC";
}

/// Product/Service ID qualifiers
pub mod product_qualifier {
    pub const HCPCS: &str = "HC";
    pub const CPT: &str = "HC";
    pub const NDC: &str = "N4";
}

/// Diagnosis code qualifiers
pub mod diagnosis_qualifier {
    pub const ICD_10_CM: &str = "ABK";
    pub const ICD_9_CM: &str = "ABF"; // Legacy
}

/// Gender codes
pub mod gender {
    pub const MALE: &str = "M";
    pub const FEMALE: &str = "F";
    pub const UNKNOWN: &str = "U";
}

/// Performance targets from SRD
pub mod performance {
    /// Target: Process 10,000 claims in 15 seconds
    pub const TARGET_CLAIMS_PER_SECOND: f64 = 666.67;

    /// Maximum claim charge amount per 837p spec
    pub const MAX_CLAIM_CHARGE: f64 = 99999.99;

    /// Maximum service units per line - bounded only by the
    /// NUMERIC(15,1) column type (X12 837P SV104 imposes no cap).
    pub const MAX_SERVICE_UNITS: f64 = 99_999_999_999_999.9;

    /// Maximum diagnoses per encounter
    pub const MAX_DIAGNOSES: usize = 12;

    /// Maximum modifiers per procedure
    pub const MAX_MODIFIERS: usize = 4;
}

/// Medicare conversion factor for 2024
pub const CONVERSION_FACTOR_2024: f64 = 33.2875;

/// Default RVU year
pub const DEFAULT_RVU_YEAR: i32 = 2024;

/// Default date used as a placeholder when no date is available (1900-01-01)
/// This is a sentinel value indicating "no date provided" in claim processing.
/// Using a constant instead of `from_ymd_opt(1900, 1, 1).unwrap()` because:
/// 1. It avoids repeated unwrap() calls in production code
/// 2. It makes the intent clear (this is a sentinel, not a real date)
/// 3. It's guaranteed to be valid at compile time via lazy_static
use chrono::NaiveDate;
use lazy_static::lazy_static;

lazy_static! {
    /// Default date sentinel (1900-01-01) - indicates "no date provided"
    pub static ref DEFAULT_DATE: NaiveDate =
        NaiveDate::from_ymd_opt(1900, 1, 1).expect("1900-01-01 is always a valid date");
}
