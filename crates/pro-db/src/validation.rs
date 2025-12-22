// Data validation and deduplication logic

use chrono::NaiveDate;
use pro_common::{Error, Result};
use rust_decimal::Decimal;
use sha2::{Digest, Sha256};
use sqlx::PgPool;
use std::collections::{HashMap, HashSet};


/// File hash for duplicate detection
#[derive(Debug, Clone)]
pub struct FileHash {
    pub hash: String,
    pub algorithm: String,
}

impl FileHash {
    /// Generate SHA-256 hash from file content
    pub fn from_content(content: &[u8]) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(content);
        let result = hasher.finalize();

        Self {
            hash: format!("{:x}", result),
            algorithm: "SHA-256".to_string(),
        }
    }

    /// Generate hash from string content
    pub fn from_string(content: &str) -> Self {
        Self::from_content(content.as_bytes())
    }
}

/// Duplicate detection result
#[derive(Debug, Clone)]
pub enum DuplicateStatus {
    Unique,
    Duplicate {
        existing_batch_id: i64,
        imported_at: chrono::NaiveDateTime
    },
}

/// Validator for checking if a file has been imported before
pub struct FileValidator {
    pool: PgPool,
}

impl FileValidator {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Check if a file hash already exists in the database
    pub async fn check_file_duplicate(&self, file_hash: &str) -> Result<DuplicateStatus> {
        let result = sqlx::query_as::<_, (i64, chrono::NaiveDateTime)>(
            r#"
            SELECT batch_id, created_at
            FROM staging.import_batch
            WHERE file_hash = $1
            AND is_deleted = false
            ORDER BY created_at DESC
            LIMIT 1
            "#
        )
        .bind(file_hash)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| Error::Database(e))?;

        match result {
            Some((batch_id, created_at)) => Ok(DuplicateStatus::Duplicate {
                existing_batch_id: batch_id,
                imported_at: created_at,
            }),
            None => Ok(DuplicateStatus::Unique),
        }
    }
}

/// Patient control number deduplication
#[derive(Debug, Clone)]
pub struct PatientControlNumberValidator {
    pool: PgPool,
}

impl PatientControlNumberValidator {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Check for duplicate patient control numbers within the same organization/facility
    pub async fn check_pcn_duplicate(
        &self,
        patient_control_number: &str,
        organization_id: i64,
        facility_id: Option<i64>,
        date_of_service_from: NaiveDate,
    ) -> Result<Vec<i64>> {
        let result = sqlx::query_as::<_, (i64,)>(
            r#"
            SELECT encounter_id
            FROM claims.encounter
            WHERE patient_control_number = $1
            AND organization_id = $2
            AND ($3::uuid IS NULL OR facility_id = $3)
            AND date_of_service_from = $4
            AND is_deleted = false
            ORDER BY created_at DESC
            "#
        )
        .bind(patient_control_number)
        .bind(organization_id)
        .bind(facility_id)
        .bind(date_of_service_from)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| Error::Database(e))?;

        Ok(result.into_iter().map(|r| r.0).collect())
    }

    /// Find similar patient control numbers (fuzzy matching)
    pub async fn find_similar_pcn(
        &self,
        patient_control_number: &str,
        organization_id: i64,
        threshold: f64,
    ) -> Result<Vec<(String, f64)>> {
        let result = sqlx::query_as::<_, (i64, String, f32)>(
            r#"
            SELECT
                encounter_id,
                patient_control_number,
                similarity(patient_control_number, $1) as sim
            FROM claims.encounter
            WHERE organization_id = $2
            AND is_deleted = false
            AND similarity(patient_control_number, $1) > $3
            ORDER BY sim DESC
            LIMIT 50
            "#
        )
        .bind(patient_control_number)
        .bind(organization_id)
        .bind(threshold as f32)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| Error::Database(e))?;

        Ok(result
            .into_iter()
            .map(|(_id, pcn, sim)| (pcn, sim as f64))
            .collect())
    }
}

/// Service line deduplication
#[derive(Debug, Clone)]
pub struct ServiceLineValidator {
    pool: PgPool,
}

impl ServiceLineValidator {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Check for duplicate service lines within an encounter
    pub async fn check_service_line_duplicate(
        &self,
        encounter_id: i64,
        procedure_code: &str,
        date_of_service: NaiveDate,
        service_unit_count: Decimal,
    ) -> Result<Vec<i64>> {
        let result = sqlx::query_as::<_, (i64,)>(
            r#"
            SELECT service_line_id
            FROM claims.service_line
            WHERE encounter_id = $1
            AND procedure_code = $2
            AND date_of_service = $3
            AND service_unit_count = $4
            AND is_deleted = false
            ORDER BY created_at DESC
            "#
        )
        .bind(encounter_id)
        .bind(procedure_code)
        .bind(date_of_service)
        .bind(service_unit_count)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| Error::Database(e))?;

        Ok(result.into_iter().map(|r| r.0).collect())
    }

    /// Detect service line duplicates within a batch of data before insertion
    pub fn detect_batch_duplicates(
        service_lines: &[(String, NaiveDate, Decimal)], // (procedure_code, date, units)
    ) -> Vec<(usize, usize)> {
        let mut seen = HashMap::new();
        let mut duplicates = Vec::new();

        for (idx, (code, date, units)) in service_lines.iter().enumerate() {
            let key = (code.clone(), *date, *units);

            if let Some(&first_idx) = seen.get(&key) {
                duplicates.push((first_idx, idx));
            } else {
                seen.insert(key, idx);
            }
        }

        duplicates
    }
}

/// Business rule validation
#[derive(Debug, Clone)]
pub struct BusinessRuleValidator {
    pool: PgPool,
}

impl BusinessRuleValidator {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Validate encounter business rules
    pub async fn validate_encounter(&self, encounter: &EncounterValidation) -> Result<ValidationResult> {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        // Rule 1: Date of service cannot be in the future
        let today = chrono::Utc::now().naive_utc().date();
        if encounter.date_of_service_from > today {
            errors.push("Date of service cannot be in the future".to_string());
        }

        // Rule 2: Date range validation
        if let Some(to_date) = encounter.date_of_service_to {
            if to_date < encounter.date_of_service_from {
                errors.push("Date of service 'to' cannot be before 'from'".to_string());
            }

            // Rule 3: Date span should not exceed 365 days
            let days_diff = (to_date - encounter.date_of_service_from).num_days();
            if days_diff > 365 {
                warnings.push("Date of service span exceeds 365 days".to_string());
            }
        }

        // Rule 4: Validate subscriber birth date (only if provided)
        if let Some(birth_date) = encounter.subscriber_birth_date {
            if birth_date > today {
                errors.push("Subscriber birth date cannot be in the future".to_string());
            }

            // Rule 5: Check age reasonability
            let age = (encounter.date_of_service_from - birth_date).num_days() / 365;
            if age < 0 {
                errors.push("Subscriber birth date is after date of service".to_string());
            } else if age > 120 {
                warnings.push("Subscriber age exceeds 120 years".to_string());
            }
        }

        // Rule 6: Validate total charge amount
        if encounter.total_claim_charge_amount <= Decimal::ZERO {
            errors.push("Total claim charge amount must be greater than zero".to_string());
        }

        if encounter.total_claim_charge_amount > Decimal::new(999999, 2) {
            warnings.push("Total claim charge amount exceeds $999,999.99".to_string());
        }

        // Rule 7: Validate place of service
        if let Some(ref pos) = encounter.place_of_service_code {
            if !Self::is_valid_place_of_service(pos) {
                errors.push(format!("Invalid place of service code: {}", pos));
            }
        }

        // Rule 8: Validate diagnosis count
        if encounter.diagnosis_count == 0 {
            errors.push("At least one diagnosis code is required".to_string());
        }

        if encounter.diagnosis_count > 12 {
            warnings.push("More than 12 diagnosis codes present".to_string());
        }

        // Rule 9: Check organization and facility exist
        let org_exists = self.check_organization_exists(encounter.organization_id).await?;
        if !org_exists {
            errors.push("Organization does not exist".to_string());
        }

        if let Some(facility_id) = encounter.facility_id {
            let facility_exists = self.check_facility_exists(facility_id).await?;
            if !facility_exists {
                errors.push("Facility does not exist".to_string());
            }
        }

        Ok(ValidationResult { errors, warnings })
    }

    /// Validate service line business rules
    pub async fn validate_service_line(&self, service_line: &ServiceLineValidation) -> Result<ValidationResult> {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        // Rule 1: Service unit count must be positive
        if service_line.service_unit_count <= Decimal::ZERO {
            errors.push("Service unit count must be greater than zero".to_string());
        }

        // Rule 2: Line item charge must be positive
        if service_line.line_item_charge_amount <= Decimal::ZERO {
            errors.push("Line item charge amount must be greater than zero".to_string());
        }

        // Rule 3: Units should typically be between 0.1 and 100
        if service_line.service_unit_count > Decimal::new(100, 0) {
            warnings.push("Service unit count exceeds 100 units".to_string());
        }

        // Rule 4: Check for valid procedure code format
        if service_line.procedure_code.len() < 4 || service_line.procedure_code.len() > 5 {
            warnings.push("Procedure code length should be 4-5 characters".to_string());
        }

        // Rule 5: Validate modifier combinations
        let modifiers: Vec<Option<String>> = vec![
            service_line.procedure_modifier_1.clone(),
            service_line.procedure_modifier_2.clone(),
            service_line.procedure_modifier_3.clone(),
            service_line.procedure_modifier_4.clone(),
        ];

        let unique_modifiers: HashSet<_> = modifiers.iter()
            .filter_map(|m| m.as_ref())
            .collect();

        let non_empty_count = modifiers.iter().filter(|m| m.is_some()).count();

        if unique_modifiers.len() != non_empty_count {
            errors.push("Duplicate modifiers detected on service line".to_string());
        }

        // Rule 6: Date of service should match encounter date range
        // (This requires encounter context, would be validated at higher level)

        Ok(ValidationResult { errors, warnings })
    }

    /// Check if place of service code is valid
    fn is_valid_place_of_service(code: &str) -> bool {
        // Common place of service codes (CMS-maintained list)
        matches!(
            code,
            "01" | "02" | "03" | "04" | "05" | "06" | "07" | "08" | "09" |
            "10" | "11" | "12" | "13" | "14" | "15" | "16" | "17" | "18" |
            "19" | "20" | "21" | "22" | "23" | "24" | "25" | "26" | "31" |
            "32" | "33" | "34" | "41" | "42" | "49" | "50" | "51" | "52" |
            "53" | "54" | "55" | "56" | "57" | "58" | "60" | "61" | "62" |
            "65" | "71" | "72" | "81"
        )
    }

    /// Check if organization exists
    async fn check_organization_exists(&self, organization_id: i64) -> Result<bool> {
        let result = sqlx::query_as::<_, (i64,)>(
            r#"
            SELECT COUNT(*) as count
            FROM claims.organization
            WHERE organization_id = $1
            AND is_active = true
            AND is_deleted = false
            "#
        )
        .bind(organization_id)
        .fetch_one(&self.pool)
        .await
        .map_err(|e| Error::Database(e))?;

        Ok(result.0 > 0)
    }

    /// Check if facility exists
    async fn check_facility_exists(&self, facility_id: i64) -> Result<bool> {
        let result = sqlx::query_as::<_, (i64,)>(
            r#"
            SELECT COUNT(*) as count
            FROM claims.facility
            WHERE facility_id = $1
            AND is_active = true
            AND is_deleted = false
            "#
        )
        .bind(facility_id)
        .fetch_one(&self.pool)
        .await
        .map_err(|e| Error::Database(e))?;

        Ok(result.0 > 0)
    }
}

/// Encounter data for validation
#[derive(Debug, Clone)]
pub struct EncounterValidation {
    pub organization_id: i64,
    pub facility_id: Option<i64>,
    pub patient_control_number: String,
    pub date_of_service_from: NaiveDate,
    pub date_of_service_to: Option<NaiveDate>,
    pub subscriber_birth_date: Option<NaiveDate>,
    pub total_claim_charge_amount: Decimal,
    pub place_of_service_code: Option<String>,
    pub diagnosis_count: usize,
}

/// Service line data for validation
#[derive(Debug, Clone)]
pub struct ServiceLineValidation {
    pub procedure_code: String,
    pub procedure_modifier_1: Option<String>,
    pub procedure_modifier_2: Option<String>,
    pub procedure_modifier_3: Option<String>,
    pub procedure_modifier_4: Option<String>,
    pub service_unit_count: Decimal,
    pub line_item_charge_amount: Decimal,
    pub date_of_service: NaiveDate,
}

/// Validation result with errors and warnings
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

impl ValidationResult {
    pub fn new() -> Self {
        Self {
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    pub fn is_valid(&self) -> bool {
        self.errors.is_empty()
    }

    pub fn merge(&mut self, other: ValidationResult) {
        self.errors.extend(other.errors);
        self.warnings.extend(other.warnings);
    }
}

impl Default for ValidationResult {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_file_hash_generation() {
        let content = "test content";
        let hash = FileHash::from_string(content);

        assert_eq!(hash.algorithm, "SHA-256");
        assert_eq!(hash.hash.len(), 64); // SHA-256 produces 64 hex characters

        // Same content should produce same hash
        let hash2 = FileHash::from_string(content);
        assert_eq!(hash.hash, hash2.hash);

        // Different content should produce different hash
        let hash3 = FileHash::from_string("different content");
        assert_ne!(hash.hash, hash3.hash);
    }

    #[test]
    fn test_service_line_duplicate_detection() {
        // detect_batch_duplicates is now a static function
        let service_lines = vec![
            ("99213".to_string(), NaiveDate::from_ymd_opt(2024, 1, 15).unwrap(), Decimal::new(1, 0)),
            ("99214".to_string(), NaiveDate::from_ymd_opt(2024, 1, 15).unwrap(), Decimal::new(1, 0)),
            ("99213".to_string(), NaiveDate::from_ymd_opt(2024, 1, 15).unwrap(), Decimal::new(1, 0)), // Duplicate of index 0
            ("99213".to_string(), NaiveDate::from_ymd_opt(2024, 1, 16).unwrap(), Decimal::new(1, 0)), // Different date
        ];

        let duplicates = ServiceLineValidator::detect_batch_duplicates(&service_lines);

        assert_eq!(duplicates.len(), 1);
        assert_eq!(duplicates[0], (0, 2)); // Index 2 is duplicate of index 0
    }

    #[test]
    fn test_validation_result() {
        let mut result = ValidationResult::new();

        assert!(result.is_valid());

        result.errors.push("Test error".to_string());
        assert!(!result.is_valid());

        result.warnings.push("Test warning".to_string());
        assert_eq!(result.warnings.len(), 1);
    }

    #[test]
    fn test_validation_result_merge() {
        let mut result1 = ValidationResult::new();
        result1.errors.push("Error 1".to_string());
        result1.warnings.push("Warning 1".to_string());

        let mut result2 = ValidationResult::new();
        result2.errors.push("Error 2".to_string());
        result2.warnings.push("Warning 2".to_string());

        result1.merge(result2);

        assert_eq!(result1.errors.len(), 2);
        assert_eq!(result1.warnings.len(), 2);
    }

    #[test]
    fn test_place_of_service_validation() {
        // is_valid_place_of_service is now a static function
        assert!(BusinessRuleValidator::is_valid_place_of_service("11")); // Office
        assert!(BusinessRuleValidator::is_valid_place_of_service("21")); // Inpatient Hospital
        assert!(BusinessRuleValidator::is_valid_place_of_service("23")); // Emergency Room
        assert!(!BusinessRuleValidator::is_valid_place_of_service("99")); // Invalid
        assert!(!BusinessRuleValidator::is_valid_place_of_service("ABC")); // Invalid
    }
}
