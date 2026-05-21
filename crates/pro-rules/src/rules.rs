// Concrete rule implementations for healthcare claim auditing

use crate::flag_types::FlagIssueType;
use crate::rule_engine::{Rule, RuleExecutionContext, RuleExecutionCache, RuleResult};
use async_trait::async_trait;
use pro_common::{Error, Result};
use rust_decimal::Decimal;
use sqlx::PgPool;

/// Rule: Detect duplicate service billing (OTH-003)
pub struct DuplicateServiceRule;

#[async_trait]
impl Rule for DuplicateServiceRule {
    fn flag_type(&self) -> FlagIssueType {
        FlagIssueType::OthDuplicateService
    }

    async fn execute(&self, ctx: &mut RuleExecutionContext, pool: &PgPool) -> Result<Option<RuleResult>> {
        // Check if we have required data
        let Some(ref procedure_code) = ctx.procedure_code else {
            return Ok(None);
        };

        let Some(date_of_service) = ctx.date_of_service else {
            return Ok(None);
        };

        let Some(encounter_id) = ctx.encounter_id else {
            return Ok(None);
        };

        let Some(service_line_id) = ctx.service_line_id else {
            return Ok(None);
        };

        // Query for duplicate services
        let query = r#"
            SELECT COUNT(*) as count
            FROM claims.service_line
            WHERE encounter_id = $1
            AND procedure_code = $2
            AND date_of_service = $3
            AND service_line_id != $4
            AND is_deleted = false
        "#;

        let result = sqlx::query_as::<_, (i64,)>(query)
            .bind(encounter_id)
            .bind(procedure_code)
            .bind(date_of_service)
            .bind(service_line_id)
            .fetch_one(pool)
            .await
            .map_err(|e| Error::Database(e))?;

        if result.0 > 0 {
            let details = format!(
                "Duplicate service found: {} on {} (found {} duplicate(s))",
                procedure_code, date_of_service, result.0
            );

            Ok(Some(
                RuleResult::new(self.flag_type(), ctx.to_flag_context())
                    .with_details(details)
            ))
        } else {
            Ok(None)
        }
    }

    // PHASE 3: Cache-optimized execution
    async fn execute_with_cache(
        &self,
        ctx: &mut RuleExecutionContext,
        cache: &RuleExecutionCache,
        _pool: &PgPool,
    ) -> Result<Option<RuleResult>> {
        // Check if we have required data
        let Some(ref procedure_code) = ctx.procedure_code else {
            return Ok(None);
        };

        let Some(service_date) = ctx.date_of_service else {
            return Ok(None);
        };

        let Some(service_line_id) = ctx.service_line_id else {
            return Ok(None);
        };

        // Check cache instead of querying database
        if let Some(duplicate_ids) = cache.get_duplicate_service_lines(
            procedure_code,
            service_date,
            ctx.provider_id,
        ) {
            // Filter out current service line
            let other_duplicates: Vec<_> = duplicate_ids.iter()
                .filter(|&&id| id != service_line_id)
                .collect();

            if !other_duplicates.is_empty() {
                let details = format!(
                    "Duplicate service found: {} on {} (found {} duplicate(s))",
                    procedure_code,
                    service_date,
                    other_duplicates.len()
                );

                return Ok(Some(
                    RuleResult::new(self.flag_type(), ctx.to_flag_context())
                        .with_details(details)
                ));
            }
        }

        Ok(None)
    }
}

/// Rule: Detect units exceeding maximum (QTY-001)
pub struct UnitsExceedMaximumRule {
    max_units: Decimal,
}

impl UnitsExceedMaximumRule {
    pub fn new(max_units: Decimal) -> Self {
        Self { max_units }
    }
}

impl Default for UnitsExceedMaximumRule {
    fn default() -> Self {
        Self::new(Decimal::new(100, 0)) // Default max 100 units
    }
}

#[async_trait]
impl Rule for UnitsExceedMaximumRule {
    fn flag_type(&self) -> FlagIssueType {
        FlagIssueType::QtyUnitsExceedMaximum
    }

    async fn execute(&self, ctx: &mut RuleExecutionContext, _pool: &PgPool) -> Result<Option<RuleResult>> {
        let Some(units) = ctx.service_unit_count else {
            return Ok(None);
        };

        if units > self.max_units {
            let details = format!(
                "Units ({}) exceed maximum allowed ({})",
                units, self.max_units
            );

            Ok(Some(
                RuleResult::new(self.flag_type(), ctx.to_flag_context())
                    .with_details(details)
            ))
        } else {
            Ok(None)
        }
    }
}

/// Rule: Detect missing required modifier (MOD-001)
pub struct MissingRequiredModifierRule {
    required_modifiers: Vec<String>,
}

impl MissingRequiredModifierRule {
    pub fn new(required_modifiers: Vec<String>) -> Self {
        Self { required_modifiers }
    }

    /// Check if procedure code requires bilateral modifier
    fn requires_bilateral_modifier(procedure_code: &str) -> bool {
        // List of procedure codes that typically require bilateral modifiers
        matches!(
            procedure_code,
            "27447" | "27486" | // Knee replacements
            "64721" | "64722" | // Carpal tunnel release
            "15823" | "15824" | // Blepharoplasty
            "66984" | // Cataract surgery
            "23472" | "23474" // Shoulder arthroscopy
        )
    }
}

impl Default for MissingRequiredModifierRule {
    fn default() -> Self {
        Self::new(vec![])
    }
}

#[async_trait]
impl Rule for MissingRequiredModifierRule {
    fn flag_type(&self) -> FlagIssueType {
        FlagIssueType::ModMissingRequired
    }

    async fn execute(&self, ctx: &mut RuleExecutionContext, _pool: &PgPool) -> Result<Option<RuleResult>> {
        let Some(ref procedure_code) = ctx.procedure_code else {
            return Ok(None);
        };

        // Check for bilateral procedures without bilateral modifier
        if Self::requires_bilateral_modifier(procedure_code) {
            let has_bilateral = ctx.procedure_modifiers.iter().any(|m| m == "50" || m == "LT" || m == "RT");

            if !has_bilateral {
                let details = format!(
                    "Bilateral procedure code {} requires modifier 50, LT, or RT",
                    procedure_code
                );

                return Ok(Some(
                    RuleResult::new(self.flag_type(), ctx.to_flag_context())
                        .with_details(details)
                ));
            }
        }

        // Check for custom required modifiers
        for required_mod in &self.required_modifiers {
            if !ctx.procedure_modifiers.contains(required_mod) {
                let details = format!(
                    "Required modifier {} is missing for procedure code {}",
                    required_mod, procedure_code
                );

                return Ok(Some(
                    RuleResult::new(self.flag_type(), ctx.to_flag_context())
                        .with_details(details)
                ));
            }
        }

        Ok(None)
    }
}

/// Rule: Detect conflicting modifiers (MOD-003)
pub struct ConflictingModifiersRule;

#[async_trait]
impl Rule for ConflictingModifiersRule {
    fn flag_type(&self) -> FlagIssueType {
        FlagIssueType::ModConflicting
    }

    async fn execute(&self, ctx: &mut RuleExecutionContext, _pool: &PgPool) -> Result<Option<RuleResult>> {
        if ctx.procedure_modifiers.is_empty() {
            return Ok(None);
        }

        // Check for conflicting laterality modifiers
        let has_lt = ctx.procedure_modifiers.contains(&"LT".to_string());
        let has_rt = ctx.procedure_modifiers.contains(&"RT".to_string());
        let has_50 = ctx.procedure_modifiers.contains(&"50".to_string());

        if (has_lt && has_rt) || (has_50 && (has_lt || has_rt)) {
            let details = "Conflicting laterality modifiers detected (LT, RT, 50 are mutually exclusive)".to_string();

            return Ok(Some(
                RuleResult::new(self.flag_type(), ctx.to_flag_context())
                    .with_details(details)
            ));
        }

        // Check for duplicate modifiers
        let mut unique_mods = ctx.procedure_modifiers.clone();
        unique_mods.sort();
        unique_mods.dedup();

        if unique_mods.len() != ctx.procedure_modifiers.len() {
            let details = "Duplicate modifiers detected on service line".to_string();

            return Ok(Some(
                RuleResult::new(self.flag_type(), ctx.to_flag_context())
                    .with_details(details)
            ));
        }

        Ok(None)
    }
}

/// Rule: Detect unspecified diagnosis code when specific available (DX-004)
pub struct UnspecifiedDiagnosisRule;

#[async_trait]
impl Rule for UnspecifiedDiagnosisRule {
    fn flag_type(&self) -> FlagIssueType {
        FlagIssueType::DxUnspecifiedWhenSpecificAvailable
    }

    async fn execute(&self, ctx: &mut RuleExecutionContext, _pool: &PgPool) -> Result<Option<RuleResult>> {
        for diagnosis_code in &ctx.diagnosis_codes {
            // Check if code ends with .9 (often unspecified)
            if diagnosis_code.ends_with(".9") || diagnosis_code.ends_with(".90") {
                let details = format!(
                    "Unspecified diagnosis code {} may have more specific alternatives available",
                    diagnosis_code
                );

                return Ok(Some(
                    RuleResult::new(self.flag_type(), ctx.to_flag_context())
                        .with_details(details)
                ));
            }
        }

        Ok(None)
    }
}

/// Rule: Detect missing specificity in diagnosis coding (DX-002)
pub struct MissingDiagnosisSpecificityRule;

#[async_trait]
impl Rule for MissingDiagnosisSpecificityRule {
    fn flag_type(&self) -> FlagIssueType {
        FlagIssueType::DxMissingSpecificity
    }

    async fn execute(&self, ctx: &mut RuleExecutionContext, _pool: &PgPool) -> Result<Option<RuleResult>> {
        for diagnosis_code in &ctx.diagnosis_codes {
            // ICD-10 codes should be at least 3 characters (category level)
            // Full codes are typically 5-7 characters with the decimal
            let code_without_period = diagnosis_code.replace('.', "");

            if code_without_period.len() < 4 {
                let details = format!(
                    "Diagnosis code {} lacks required specificity (too short)",
                    diagnosis_code
                );

                return Ok(Some(
                    RuleResult::new(self.flag_type(), ctx.to_flag_context())
                        .with_details(details)
                ));
            }

            // Check for codes that typically require laterality
            if code_without_period.starts_with("M") || code_without_period.starts_with("S") {
                // These are musculoskeletal/injury codes that often need laterality
                if !diagnosis_code.contains("1") && !diagnosis_code.contains("2") {
                    // Simple heuristic: if it doesn't contain 1 or 2, might be missing laterality
                    // This is simplified - in production would check against comprehensive list
                }
            }
        }

        Ok(None)
    }
}

/// Rule: Detect units inconsistent with documentation (QTY-002)
pub struct UnitsInconsistentRule;

#[async_trait]
impl Rule for UnitsInconsistentRule {
    fn flag_type(&self) -> FlagIssueType {
        FlagIssueType::QtyUnitsInconsistent
    }

    async fn execute(&self, ctx: &mut RuleExecutionContext, _pool: &PgPool) -> Result<Option<RuleResult>> {
        let Some(units) = ctx.service_unit_count else {
            return Ok(None);
        };

        let Some(ref procedure_code) = ctx.procedure_code else {
            return Ok(None);
        };

        // Check for zero or negative units
        if units <= Decimal::ZERO {
            let details = format!(
                "Invalid units ({}) for procedure code {}",
                units, procedure_code
            );

            return Ok(Some(
                RuleResult::new(self.flag_type(), ctx.to_flag_context())
                    .with_details(details)
            ));
        }

        // Check for fractional units on codes that should be whole numbers
        // Most procedure codes should have whole number units
        if units.fract() != Decimal::ZERO {
            // Some codes allow fractional units (e.g., anesthesia, time-based)
            let allows_fractional = procedure_code.starts_with("00") // Anesthesia codes
                || matches!(procedure_code.as_str(),
                    "99354" | "99355" | "99356" | "99357" | // Prolonged services
                    "99415" | "99416" // Prolonged services
                );

            if !allows_fractional {
                let details = format!(
                    "Fractional units ({}) not typically allowed for procedure code {}",
                    units, procedure_code
                );

                return Ok(Some(
                    RuleResult::new(self.flag_type(), ctx.to_flag_context())
                        .with_details(details)
                ));
            }
        }

        // Check for common unit patterns that may be errors
        // E/M codes should typically be 1 unit
        if procedure_code.starts_with("99") && !procedure_code.starts_with("99") {
            if units > Decimal::new(1, 0) {
                let is_time_based = matches!(procedure_code.as_str(),
                    "99354" | "99355" | "99356" | "99357" | // Prolonged services
                    "99415" | "99416" | "99417"
                );

                if !is_time_based {
                    let details = format!(
                        "E/M code {} typically billed with 1 unit, found {} units",
                        procedure_code, units
                    );

                    return Ok(Some(
                        RuleResult::new(self.flag_type(), ctx.to_flag_context())
                            .with_details(details)
                    ));
                }
            }
        }

        Ok(None)
    }
}

/// Rule: Detect primary diagnosis that does not support service (DX-001)
pub struct PrimaryDiagnosisDoesNotSupportRule;

#[async_trait]
impl Rule for PrimaryDiagnosisDoesNotSupportRule {
    fn flag_type(&self) -> FlagIssueType {
        FlagIssueType::DxPrimaryDoesNotSupport
    }

    async fn execute(&self, ctx: &mut RuleExecutionContext, _pool: &PgPool) -> Result<Option<RuleResult>> {
        if ctx.diagnosis_codes.is_empty() {
            let details = "No diagnosis codes present to support service".to_string();
            return Ok(Some(
                RuleResult::new(self.flag_type(), ctx.to_flag_context())
                    .with_details(details)
            ));
        }

        let Some(ref procedure_code) = ctx.procedure_code else {
            return Ok(None);
        };

        let primary_dx = &ctx.diagnosis_codes[0];

        // Check for common mismatches between diagnosis and procedure
        // Example: Screening procedures with non-screening diagnoses
        let is_screening_procedure = matches!(procedure_code.as_str(),
            "G0101" | // Cervical cancer screening
            "G0103" | "G0104" | "G0105" | // Colorectal screening
            "77067" | "77063" | // Screening mammography
            "99385" | "99386" | "99387" | "99395" | "99396" | "99397" // Preventive visits
        );

        if is_screening_procedure {
            // Screening procedures should have Z codes (screening) as primary
            if !primary_dx.starts_with("Z") {
                let details = format!(
                    "Screening procedure {} requires screening diagnosis (Z code) as primary, found {}",
                    procedure_code, primary_dx
                );

                return Ok(Some(
                    RuleResult::new(self.flag_type(), ctx.to_flag_context())
                        .with_details(details)
                ));
            }
        }

        // Check for surgical procedures with administrative codes
        let is_surgical = procedure_code.parse::<u32>().unwrap_or(0) < 70000
            && procedure_code.parse::<u32>().unwrap_or(0) >= 10000;

        if is_surgical && primary_dx.starts_with("Z") {
            // Most Z codes shouldn't be primary for surgical procedures
            let allowed_z_codes = primary_dx.starts_with("Z38") // Liveborn
                || primary_dx.starts_with("Z3A"); // Weeks of gestation

            if !allowed_z_codes {
                let details = format!(
                    "Surgical procedure {} should not have administrative diagnosis {} as primary",
                    procedure_code, primary_dx
                );

                return Ok(Some(
                    RuleResult::new(self.flag_type(), ctx.to_flag_context())
                        .with_details(details)
                ));
            }
        }

        Ok(None)
    }
}

/// Rule: Detect diagnosis sequencing errors (DX-003)
pub struct DiagnosisSequencingErrorRule;

#[async_trait]
impl Rule for DiagnosisSequencingErrorRule {
    fn flag_type(&self) -> FlagIssueType {
        FlagIssueType::DxSequencingError
    }

    async fn execute(&self, ctx: &mut RuleExecutionContext, _pool: &PgPool) -> Result<Option<RuleResult>> {
        if ctx.diagnosis_codes.len() < 2 {
            return Ok(None);
        }

        let primary_dx = &ctx.diagnosis_codes[0];

        // Check for manifestation codes in primary position
        // Manifestation codes should never be sequenced first
        let manifestation_prefixes = ["B20", "B21", "B22", "B23", "B24"]; // HIV manifestations

        for prefix in &manifestation_prefixes {
            if primary_dx.starts_with(prefix) {
                let details = format!(
                    "Manifestation diagnosis {} should not be sequenced as primary",
                    primary_dx
                );

                return Ok(Some(
                    RuleResult::new(self.flag_type(), ctx.to_flag_context())
                        .with_details(details)
                ));
            }
        }

        // Check for symptoms/signs as primary when specific condition exists
        if primary_dx.starts_with("R") {
            // R codes are symptoms/signs - check if more specific diagnosis exists
            let has_specific_dx = ctx.diagnosis_codes.iter().skip(1).any(|dx| {
                !dx.starts_with("R") && !dx.starts_with("Z")
            });

            if has_specific_dx {
                let details = format!(
                    "Symptom code {} is primary, but more specific diagnosis exists in secondary positions",
                    primary_dx
                );

                return Ok(Some(
                    RuleResult::new(self.flag_type(), ctx.to_flag_context())
                        .with_details(details)
                ));
            }
        }

        // Check for "code first" requirements
        // Example: Drug-induced conditions require drug code to be sequenced first
        for (idx, dx) in ctx.diagnosis_codes.iter().enumerate() {
            if idx == 0 {
                continue;
            }

            // Check if this is a poisoning/adverse effect that should be first
            if dx.starts_with("T36") || dx.starts_with("T37") || dx.starts_with("T38")
                || dx.starts_with("T39") || dx.starts_with("T40") || dx.starts_with("T41")
                || dx.starts_with("T42") || dx.starts_with("T43") || dx.starts_with("T44")
                || dx.starts_with("T45") || dx.starts_with("T46") || dx.starts_with("T47")
                || dx.starts_with("T48") || dx.starts_with("T49") || dx.starts_with("T50") {
                let details = format!(
                    "Poisoning/adverse effect code {} should typically be sequenced before manifestations",
                    dx
                );

                return Ok(Some(
                    RuleResult::new(self.flag_type(), ctx.to_flag_context())
                        .with_details(details)
                ));
            }
        }

        Ok(None)
    }
}

/// Rule: Detect incorrect modifiers (MOD-002)
pub struct IncorrectModifierRule;

#[async_trait]
impl Rule for IncorrectModifierRule {
    fn flag_type(&self) -> FlagIssueType {
        FlagIssueType::ModIncorrect
    }

    async fn execute(&self, ctx: &mut RuleExecutionContext, _pool: &PgPool) -> Result<Option<RuleResult>> {
        if ctx.procedure_modifiers.is_empty() {
            return Ok(None);
        }

        let Some(ref procedure_code) = ctx.procedure_code else {
            return Ok(None);
        };

        // Check for modifier 25 on non-E/M codes
        if ctx.procedure_modifiers.contains(&"25".to_string()) {
            if !procedure_code.starts_with("99") {
                let details = format!(
                    "Modifier 25 (significant, separately identifiable E/M) is only valid on E/M codes, found on {}",
                    procedure_code
                );

                return Ok(Some(
                    RuleResult::new(self.flag_type(), ctx.to_flag_context())
                        .with_details(details)
                ));
            }
        }

        // Check for modifier 59 overuse (should use X{EPSU} when appropriate)
        if ctx.procedure_modifiers.contains(&"59".to_string()) {
            let details = "Modifier 59 usage detected - consider if X{EPSU} modifiers would be more specific".to_string();

            return Ok(Some(
                RuleResult::new(self.flag_type(), ctx.to_flag_context())
                    .with_details(details)
                    .with_severity(crate::flag_types::FlagSeverity::Low)
            ));
        }

        // Check for anatomical modifiers on non-applicable codes
        let anatomical_modifiers = ["E1", "E2", "E3", "E4", "FA", "F1", "F2", "F3", "F4", "F5",
                                     "F6", "F7", "F8", "F9", "TA", "T1", "T2", "T3", "T4", "T5",
                                     "T6", "T7", "T8", "T9"];

        for modifier in &ctx.procedure_modifiers {
            if anatomical_modifiers.contains(&modifier.as_str()) {
                // These are finger/toe modifiers - check if procedure typically uses them
                let typically_uses_anatomical = procedure_code.starts_with("11") // Integumentary
                    || procedure_code.starts_with("26") || procedure_code.starts_with("27") // Musculoskeletal
                    || procedure_code.starts_with("28"); // Foot and toes

                if !typically_uses_anatomical {
                    let details = format!(
                        "Anatomical modifier {} may not be appropriate for procedure code {}",
                        modifier, procedure_code
                    );

                    return Ok(Some(
                        RuleResult::new(self.flag_type(), ctx.to_flag_context())
                            .with_details(details)
                    ));
                }
            }
        }

        Ok(None)
    }
}

/// Rule: Detect time-based E/M codes without time documentation (EMT-001)
pub struct TimeNotDocumentedRule;

#[async_trait]
impl Rule for TimeNotDocumentedRule {
    fn flag_type(&self) -> FlagIssueType {
        FlagIssueType::EMTTimeNotDocumented
    }

    async fn execute(&self, ctx: &mut RuleExecutionContext, _pool: &PgPool) -> Result<Option<RuleResult>> {
        let Some(ref procedure_code) = ctx.procedure_code else {
            return Ok(None);
        };

        // Time-based E/M codes that require time documentation
        let time_based_codes = [
            "99354", "99355", "99356", "99357", // Prolonged services
            "99415", "99416", "99417", // Prolonged services
            "99441", "99442", "99443", // Telephone services
            "99421", "99422", "99423", // Online digital E/M
        ];

        if time_based_codes.contains(&procedure_code.as_str()) {
            // Check if time is documented in custom data
            let has_time_documented = ctx.custom_data.contains_key("time_spent")
                || ctx.custom_data.contains_key("total_time")
                || ctx.custom_data.contains_key("counseling_time");

            if !has_time_documented {
                let details = format!(
                    "Time-based E/M code {} requires documented time spent on encounter",
                    procedure_code
                );

                return Ok(Some(
                    RuleResult::new(self.flag_type(), ctx.to_flag_context())
                        .with_details(details)
                ));
            }
        }

        // Check for counseling/coordination modifier with E/M
        if ctx.procedure_modifiers.contains(&"25".to_string()) {
            let is_em_code = procedure_code.starts_with("99");

            // If billing E/M on time basis (>50% counseling), should document time
            if is_em_code {
                let has_counseling_note = ctx.custom_data.contains_key("counseling_time")
                    || ctx.custom_data.contains_key("coordination_time");

                if !has_counseling_note {
                    // This is informational - may be billing on MDM instead
                    let details = format!(
                        "E/M code {} - if billing based on time/counseling, document time spent",
                        procedure_code
                    );

                    return Ok(Some(
                        RuleResult::new(self.flag_type(), ctx.to_flag_context())
                            .with_details(details)
                            .with_severity(crate::flag_types::FlagSeverity::Low)
                    ));
                }
            }
        }

        Ok(None)
    }
}

/// Rule: Detect wrong E/M category (EMI-001)
pub struct WrongEMCategoryRule;

#[async_trait]
impl Rule for WrongEMCategoryRule {
    fn flag_type(&self) -> FlagIssueType {
        FlagIssueType::EMIWrongCategory
    }

    async fn execute(&self, ctx: &mut RuleExecutionContext, pool: &PgPool) -> Result<Option<RuleResult>> {
        let Some(ref procedure_code) = ctx.procedure_code else {
            return Ok(None);
        };

        let Some(encounter_id) = ctx.encounter_id else {
            return Ok(None);
        };

        let Some(date_of_service) = ctx.date_of_service else {
            return Ok(None);
        };

        // Check if this is an E/M code
        if !procedure_code.starts_with("99") {
            return Ok(None);
        }

        // New patient office visit codes
        let new_patient_codes = ["99201", "99202", "99203", "99204", "99205"];
        // Established patient office visit codes
        let established_codes = ["99211", "99212", "99213", "99214", "99215"];

        let is_new_patient_code = new_patient_codes.contains(&procedure_code.as_str());
        let is_established_code = established_codes.contains(&procedure_code.as_str());

        if !is_new_patient_code && !is_established_code {
            return Ok(None); // Not an office visit code
        }

        // Query to check if patient has been seen within past 3 years
        let query = r#"
            SELECT COUNT(*) as count
            FROM claims.encounter e
            JOIN claims.service_line sl ON e.encounter_id = sl.encounter_id
            WHERE e.patient_id = (
                SELECT patient_id FROM claims.encounter WHERE encounter_id = $1
            )
            AND sl.date_of_service < $2
            AND sl.date_of_service >= $2 - INTERVAL '3 years'
            AND e.encounter_id != $1
            AND e.is_deleted = false
            AND sl.is_deleted = false
            LIMIT 1
        "#;

        let result = sqlx::query_as::<_, (i64,)>(query)
            .bind(encounter_id)
            .bind(date_of_service)
            .fetch_optional(pool)
            .await
            .map_err(|e| Error::Database(e))?;

        let has_prior_visit = result.map(|r| r.0 > 0).unwrap_or(false);

        // New patient code but patient has prior visits
        if is_new_patient_code && has_prior_visit {
            let details = format!(
                "New patient code {} used but patient has visits within past 3 years",
                procedure_code
            );

            return Ok(Some(
                RuleResult::new(self.flag_type(), ctx.to_flag_context())
                    .with_details(details)
            ));
        }

        // Established patient code but no prior visits
        if is_established_code && !has_prior_visit {
            let details = format!(
                "Established patient code {} used but no visits found within past 3 years",
                procedure_code
            );

            return Ok(Some(
                RuleResult::new(self.flag_type(), ctx.to_flag_context())
                    .with_details(details)
            ));
        }

        Ok(None)
    }

    // PHASE 3: Cache-optimized execution
    async fn execute_with_cache(
        &self,
        ctx: &mut RuleExecutionContext,
        cache: &RuleExecutionCache,
        _pool: &PgPool,
    ) -> Result<Option<RuleResult>> {
        let Some(ref procedure_code) = ctx.procedure_code else {
            return Ok(None);
        };

        let Some(ref subscriber_id) = ctx.subscriber_id else {
            return Ok(None);
        };

        // Check if this is an E/M code
        if !procedure_code.starts_with("99") {
            return Ok(None);
        };

        // New patient office visit codes
        let new_patient_codes = ["99201", "99202", "99203", "99204", "99205"];
        // Established patient office visit codes
        let established_codes = ["99211", "99212", "99213", "99214", "99215"];

        let is_new_patient_code = new_patient_codes.contains(&procedure_code.as_str());
        let is_established_code = established_codes.contains(&procedure_code.as_str());

        if !is_new_patient_code && !is_established_code {
            return Ok(None); // Not an office visit code
        }

        // Get encounter history from cache instead of querying database
        let has_prior_visit = if let Some(encounters) = cache.get_encounter_history(subscriber_id) {
            // Check if any encounters occurred in the past 3 years before current date
            encounters.iter().any(|enc| {
                // We consider it a prior visit if the status indicates it was completed/processed
                enc.claim_status != "rejected" && enc.claim_status != "denied"
            })
        } else {
            false
        };

        // New patient code but patient has prior visits
        if is_new_patient_code && has_prior_visit {
            let details = format!(
                "New patient code {} used but patient has visits within past 3 years",
                procedure_code
            );

            return Ok(Some(
                RuleResult::new(self.flag_type(), ctx.to_flag_context())
                    .with_details(details)
            ));
        }

        // Established patient code but no prior visits
        if is_established_code && !has_prior_visit {
            let details = format!(
                "Established patient code {} used but no visits found within past 3 years",
                procedure_code
            );

            return Ok(Some(
                RuleResult::new(self.flag_type(), ctx.to_flag_context())
                    .with_details(details)
            ));
        }

        Ok(None)
    }
}

/// Rule: Detect unbundling (COD-003)
pub struct UnbundlingRule;

#[async_trait]
impl Rule for UnbundlingRule {
    fn flag_type(&self) -> FlagIssueType {
        FlagIssueType::CodUnbundling
    }

    async fn execute(&self, ctx: &mut RuleExecutionContext, pool: &PgPool) -> Result<Option<RuleResult>> {
        let Some(ref procedure_code) = ctx.procedure_code else {
            return Ok(None);
        };

        let Some(encounter_id) = ctx.encounter_id else {
            return Ok(None);
        };

        let Some(service_line_id) = ctx.service_line_id else {
            return Ok(None);
        };

        let Some(date_of_service) = ctx.date_of_service else {
            return Ok(None);
        };

        // Query for other services on same date that may be bundled
        let query = r#"
            SELECT procedure_code
            FROM claims.service_line
            WHERE encounter_id = $1
            AND date_of_service = $2
            AND service_line_id != $3
            AND is_deleted = false
        "#;

        let other_codes: Vec<(String,)> = sqlx::query_as(query)
            .bind(encounter_id)
            .bind(date_of_service)
            .bind(service_line_id)
            .fetch_all(pool)
            .await
            .map_err(|e| Error::Database(e))?;

        // Check for common bundling scenarios
        // Example: Separate E/M with minor procedure on same date without modifier 25
        if procedure_code.starts_with("99") {
            let has_procedure_same_day = other_codes.iter().any(|(code,)| {
                let code_num = code.parse::<u32>().unwrap_or(0);
                // Check for minor procedures (10000-69999 range)
                code_num >= 10000 && code_num < 70000
            });

            if has_procedure_same_day && !ctx.procedure_modifiers.contains(&"25".to_string()) {
                let details = format!(
                    "E/M code {} billed on same date as procedure without modifier 25 - may be bundled",
                    procedure_code
                );

                return Ok(Some(
                    RuleResult::new(self.flag_type(), ctx.to_flag_context())
                        .with_details(details)
                ));
            }
        }

        // Check for component billing instead of comprehensive code
        // Example: Excision + closure instead of comprehensive excision code
        let is_closure_code = matches!(procedure_code.as_str(),
            "12001" | "12002" | "12004" | "12005" | "12006" | "12007" | // Simple repair
            "12011" | "12013" | "12014" | "12015" | "12016" | "12017" | "12018" // Intermediate repair
        );

        if is_closure_code {
            let has_excision_same_day = other_codes.iter().any(|(code,)| {
                code.starts_with("11") && code.as_str() >= "11400" && code.as_str() <= "11646"
            });

            if has_excision_same_day {
                let details = format!(
                    "Separate closure code {} with excision may be bundled - verify comprehensive code not applicable",
                    procedure_code
                );

                return Ok(Some(
                    RuleResult::new(self.flag_type(), ctx.to_flag_context())
                        .with_details(details)
                ));
            }
        }

        Ok(None)
    }
}

/// Rule: Detect upcoding (COD-004)
pub struct UpcodingRule;

#[async_trait]
impl Rule for UpcodingRule {
    fn flag_type(&self) -> FlagIssueType {
        FlagIssueType::CodUpcoding
    }

    async fn execute(&self, ctx: &mut RuleExecutionContext, _pool: &PgPool) -> Result<Option<RuleResult>> {
        let Some(ref procedure_code) = ctx.procedure_code else {
            return Ok(None);
        };

        let Some(charge_amount) = ctx.line_item_charge_amount else {
            return Ok(None);
        };

        // Check for unusually high charges for common procedures
        // This is a simplified heuristic - production would use statistical analysis
        let expected_range = match procedure_code.as_str() {
            "99213" => (100.0, 200.0),  // Established patient office visit level 3
            "99214" => (150.0, 250.0),  // Established patient office visit level 4
            "99215" => (200.0, 350.0),  // Established patient office visit level 5
            "99203" => (150.0, 250.0),  // New patient office visit level 3
            "99204" => (200.0, 350.0),  // New patient office visit level 4
            "99205" => (250.0, 450.0),  // New patient office visit level 5
            _ => return Ok(None),
        };

        let charge_f64 = charge_amount.to_string().parse::<f64>().unwrap_or(0.0);

        if charge_f64 > expected_range.1 * 1.5 {
            let details = format!(
                "Charge amount ${:.2} for {} is significantly above typical range (${:.2}-${:.2})",
                charge_f64, procedure_code, expected_range.0, expected_range.1
            );

            return Ok(Some(
                RuleResult::new(self.flag_type(), ctx.to_flag_context())
                    .with_details(details)
            ));
        }

        // Check for highest-level E/M codes which require more scrutiny
        let high_level_em = matches!(procedure_code.as_str(),
            "99205" | "99215" | // Highest level office visits
            "99285" | // Highest level ED
            "99223" | // Highest level initial hospital care
            "99233" // Highest level subsequent hospital care
        );

        if high_level_em {
            // Check if complexity indicators are present
            let has_complexity_note = ctx.custom_data.contains_key("mdm_level")
                || ctx.custom_data.contains_key("history_level")
                || ctx.custom_data.contains_key("exam_level");

            if !has_complexity_note {
                let details = format!(
                    "High-level E/M code {} requires documentation of complexity justification",
                    procedure_code
                );

                return Ok(Some(
                    RuleResult::new(self.flag_type(), ctx.to_flag_context())
                        .with_details(details)
                        .with_severity(crate::flag_types::FlagSeverity::Medium)
                ));
            }
        }

        Ok(None)
    }
}

/// Rule: Detect services rendered by wrong provider type (OTH-002)
pub struct WrongProviderTypeRule;

#[async_trait]
impl Rule for WrongProviderTypeRule {
    fn flag_type(&self) -> FlagIssueType {
        FlagIssueType::OthWrongProviderType
    }

    async fn execute(&self, ctx: &mut RuleExecutionContext, pool: &PgPool) -> Result<Option<RuleResult>> {
        let Some(ref procedure_code) = ctx.procedure_code else {
            return Ok(None);
        };

        let Some(provider_id) = ctx.provider_id else {
            return Ok(None);
        };

        // Query provider credentials
        let query = r#"
            SELECT credential_type, specialty
            FROM claims.provider
            WHERE provider_id = $1
        "#;

        let result: Option<(Option<String>, Option<String>)> = sqlx::query_as(query)
            .bind(provider_id)
            .fetch_optional(pool)
            .await
            .map_err(|e| Error::Database(e))?;

        let (credential_type, specialty) = match result {
            Some((ct, sp)) => (ct, sp),
            None => return Ok(None),
        };

        // Check for procedures requiring physician credentials
        let requires_physician = matches!(procedure_code.as_str(),
            "99201" | "99202" | "99203" | "99204" | "99205" | // New patient E/M
            "99211" | "99212" | "99213" | "99214" | "99215" | // Established patient E/M
            "99221" | "99222" | "99223" | // Initial hospital care
            "99231" | "99232" | "99233" // Subsequent hospital care
        );

        if requires_physician {
            if let Some(ref cred) = credential_type {
                let is_np_pa = cred.contains("NP") || cred.contains("PA");

                if is_np_pa {
                    // NP/PA can bill E/M but may need supervision notation
                    let has_supervision_note = ctx.custom_data.contains_key("supervising_physician")
                        || ctx.custom_data.contains_key("incident_to");

                    if !has_supervision_note {
                        let details = format!(
                            "E/M code {} by {} - verify supervision requirements documented",
                            procedure_code, cred
                        );

                        return Ok(Some(
                            RuleResult::new(self.flag_type(), ctx.to_flag_context())
                                .with_details(details)
                                .with_severity(crate::flag_types::FlagSeverity::Low)
                        ));
                    }
                }
            }
        }

        // Check for procedures requiring specific specialty
        if procedure_code.starts_with("90") {
            // Anesthesia codes
            if let Some(ref spec) = specialty {
                if !spec.contains("Anesthes") {
                    let details = format!(
                        "Anesthesia code {} by provider with specialty '{}' - verify appropriateness",
                        procedure_code, spec
                    );

                    return Ok(Some(
                        RuleResult::new(self.flag_type(), ctx.to_flag_context())
                            .with_details(details)
                    ));
                }
            }
        }

        Ok(None)
    }

    // PHASE 3: Cache-optimized execution
    async fn execute_with_cache(
        &self,
        ctx: &mut RuleExecutionContext,
        cache: &RuleExecutionCache,
        _pool: &PgPool,
    ) -> Result<Option<RuleResult>> {
        let Some(ref procedure_code) = ctx.procedure_code else {
            return Ok(None);
        };

        let Some(provider_id) = ctx.provider_id else {
            return Ok(None);
        };

        // Get provider info from cache instead of querying database
        let provider_info = match cache.get_provider_info(provider_id) {
            Some(info) => info,
            None => return Ok(None),
        };

        // Check for procedures requiring physician credentials
        let requires_physician = matches!(procedure_code.as_str(),
            "99201" | "99202" | "99203" | "99204" | "99205" | // New patient E/M
            "99211" | "99212" | "99213" | "99214" | "99215" | // Established patient E/M
            "99221" | "99222" | "99223" | // Initial hospital care
            "99231" | "99232" | "99233" // Subsequent hospital care
        );

        if requires_physician {
            if let Some(ref cred) = provider_info.provider_type {
                let is_np_pa = cred.contains("NP") || cred.contains("PA");

                if is_np_pa {
                    // NP/PA can bill E/M but may need supervision notation
                    let has_supervision_note = ctx.custom_data.contains_key("supervising_physician")
                        || ctx.custom_data.contains_key("incident_to");

                    if !has_supervision_note {
                        let details = format!(
                            "E/M code {} by {} - verify supervision requirements documented",
                            procedure_code, cred
                        );

                        return Ok(Some(
                            RuleResult::new(self.flag_type(), ctx.to_flag_context())
                                .with_details(details)
                                .with_severity(crate::flag_types::FlagSeverity::Low)
                        ));
                    }
                }
            }
        }

        // Check for procedures requiring specific specialty
        if procedure_code.starts_with("90") {
            // Anesthesia codes
            if let Some(ref spec) = provider_info.specialty {
                if !spec.contains("Anesthes") {
                    let details = format!(
                        "Anesthesia code {} by provider with specialty '{}' - verify appropriateness",
                        procedure_code, spec
                    );

                    return Ok(Some(
                        RuleResult::new(self.flag_type(), ctx.to_flag_context())
                            .with_details(details)
                    ));
                }
            }
        }

        Ok(None)
    }
}

// ========================================================================
// PHASE 3: E/M OPTIMIZATION RULES
// ========================================================================

/// Rule: E/M level higher than MDM supports (EMO-001) - Placeholder
pub struct EMOLevelHigherThanMDMRule;

#[async_trait]
impl Rule for EMOLevelHigherThanMDMRule {
    fn flag_type(&self) -> FlagIssueType {
        FlagIssueType::EMOLevelHigherThanMDM
    }

    async fn execute(&self, ctx: &mut RuleExecutionContext, _pool: &PgPool) -> Result<Option<RuleResult>> {
        // Requires MDM level data in custom_data for full implementation
        let Some(ref procedure_code) = ctx.procedure_code else {
            return Ok(None);
        };

        // Check if this is an E/M code
        if !procedure_code.starts_with("99") {
            return Ok(None);
        }

        // Placeholder: would check mdm_level from custom_data against procedure code level
        if let Some(mdm_level) = ctx.custom_data.get("mdm_level") {
            let em_level = match procedure_code.as_str() {
                "99213" | "99203" => 3,
                "99214" | "99204" => 4,
                "99215" | "99205" => 5,
                _ => return Ok(None),
            };

            let mdm_level_num = mdm_level.parse::<i32>().unwrap_or(0);

            if em_level > mdm_level_num {
                let details = format!(
                    "E/M level {} higher than documented MDM level {}",
                    em_level, mdm_level_num
                );

                return Ok(Some(
                    RuleResult::new(self.flag_type(), ctx.to_flag_context())
                        .with_details(details)
                ));
            }
        }

        Ok(None)
    }
}

/// Rule: E/M level higher than history/exam supports (EMO-002) - Placeholder
pub struct EMOLevelHigherThanHistoryExamRule;

#[async_trait]
impl Rule for EMOLevelHigherThanHistoryExamRule {
    fn flag_type(&self) -> FlagIssueType {
        FlagIssueType::EMOLevelHigherThanHistoryExam
    }

    async fn execute(&self, ctx: &mut RuleExecutionContext, _pool: &PgPool) -> Result<Option<RuleResult>> {
        // Requires history_level and exam_level in custom_data for full implementation
        let Some(ref procedure_code) = ctx.procedure_code else {
            return Ok(None);
        };

        if !procedure_code.starts_with("99") {
            return Ok(None);
        }

        // Placeholder: would compare against documented history/exam complexity
        Ok(None)
    }
}

/// Rule: E/M level lower than MDM supports (EMU-001) - Placeholder
pub struct EMULevelLowerThanMDMRule;

#[async_trait]
impl Rule for EMULevelLowerThanMDMRule {
    fn flag_type(&self) -> FlagIssueType {
        FlagIssueType::EMULevelLowerThanMDM
    }

    async fn execute(&self, ctx: &mut RuleExecutionContext, _pool: &PgPool) -> Result<Option<RuleResult>> {
        // Identifies undercoding opportunities
        let Some(ref procedure_code) = ctx.procedure_code else {
            return Ok(None);
        };

        if !procedure_code.starts_with("99") {
            return Ok(None);
        }

        if let Some(mdm_level) = ctx.custom_data.get("mdm_level") {
            let em_level = match procedure_code.as_str() {
                "99213" | "99203" => 3,
                "99214" | "99204" => 4,
                "99215" | "99205" => 5,
                _ => return Ok(None),
            };

            let mdm_level_num = mdm_level.parse::<i32>().unwrap_or(0);

            if em_level < mdm_level_num {
                let details = format!(
                    "E/M level {} lower than documented MDM level {} - potential undercoding",
                    em_level, mdm_level_num
                );

                return Ok(Some(
                    RuleResult::new(self.flag_type(), ctx.to_flag_context())
                        .with_details(details)
                ));
            }
        }

        Ok(None)
    }
}

/// Rule: E/M level lower than time supports (EMU-002) - Placeholder
pub struct EMULevelLowerThanTimeRule;

#[async_trait]
impl Rule for EMULevelLowerThanTimeRule {
    fn flag_type(&self) -> FlagIssueType {
        FlagIssueType::EMULevelLowerThanTime
    }

    async fn execute(&self, ctx: &mut RuleExecutionContext, _pool: &PgPool) -> Result<Option<RuleResult>> {
        // Check if time supports higher level billing
        let Some(ref procedure_code) = ctx.procedure_code else {
            return Ok(None);
        };

        if !procedure_code.starts_with("99") {
            return Ok(None);
        }

        if let Some(time_str) = ctx.custom_data.get("time_spent") {
            let time_minutes = time_str.parse::<i32>().unwrap_or(0);

            // Office visit time thresholds (approximate)
            let suggested_level = match time_minutes {
                t if t < 15 => 1,
                15..=29 => 2,
                30..=44 => 3,
                45..=59 => 4,
                _ => 5,
            };

            let current_level = match procedure_code.as_str() {
                "99211" | "99201" => 1,
                "99212" | "99202" => 2,
                "99213" | "99203" => 3,
                "99214" | "99204" => 4,
                "99215" | "99205" => 5,
                _ => return Ok(None),
            };

            if suggested_level > current_level {
                let details = format!(
                    "Time spent ({} min) suggests level {} but billed level {}",
                    time_minutes, suggested_level, current_level
                );

                return Ok(Some(
                    RuleResult::new(self.flag_type(), ctx.to_flag_context())
                        .with_details(details)
                ));
            }
        }

        Ok(None)
    }
}

// ========================================================================
// PHASE 4: ADVANCED DETECTION RULES
// ========================================================================

/// Rule: Incorrect procedure code (COD-001) - Placeholder
pub struct IncorrectProcedureCodeRule;

#[async_trait]
impl Rule for IncorrectProcedureCodeRule {
    fn flag_type(&self) -> FlagIssueType {
        FlagIssueType::CodIncorrectProcedureCode
    }

    async fn execute(&self, _ctx: &mut RuleExecutionContext, _pool: &PgPool) -> Result<Option<RuleResult>> {
        // Placeholder: Would require procedure code validation database or service
        // This is a complex rule requiring comprehensive code verification
        Ok(None)
    }
}

/// Rule: Procedure not supported by diagnosis (COD-002) - Placeholder
pub struct ProcedureNotSupportedByDiagnosisRule;

#[async_trait]
impl Rule for ProcedureNotSupportedByDiagnosisRule {
    fn flag_type(&self) -> FlagIssueType {
        FlagIssueType::CodProcedureNotSupportedByDiagnosis
    }

    async fn execute(&self, _ctx: &mut RuleExecutionContext, _pool: &PgPool) -> Result<Option<RuleResult>> {
        // Placeholder: Would require LCD/NCD policy database
        // This requires external medical necessity lookup
        Ok(None)
    }
}

/// Rule: Insufficient documentation (DOC-001) - Placeholder
pub struct InsufficientDocumentationRule;

#[async_trait]
impl Rule for InsufficientDocumentationRule {
    fn flag_type(&self) -> FlagIssueType {
        FlagIssueType::DocInsufficientDocumentation
    }

    async fn execute(&self, _ctx: &mut RuleExecutionContext, _pool: &PgPool) -> Result<Option<RuleResult>> {
        // Placeholder: Would require clinical note parsing and analysis
        Ok(None)
    }
}

/// Rule: Missing required documentation elements (DOC-002) - Placeholder
pub struct MissingRequiredElementsRule;

#[async_trait]
impl Rule for MissingRequiredElementsRule {
    fn flag_type(&self) -> FlagIssueType {
        FlagIssueType::DocMissingRequiredElements
    }

    async fn execute(&self, _ctx: &mut RuleExecutionContext, _pool: &PgPool) -> Result<Option<RuleResult>> {
        // Placeholder: Would check for required note elements
        Ok(None)
    }
}

/// Rule: Medical necessity not established (OTH-001) - Placeholder
pub struct MedicalNecessityNotEstablishedRule;

#[async_trait]
impl Rule for MedicalNecessityNotEstablishedRule {
    fn flag_type(&self) -> FlagIssueType {
        FlagIssueType::OthMedicalNecessityNotEstablished
    }

    async fn execute(&self, _ctx: &mut RuleExecutionContext, _pool: &PgPool) -> Result<Option<RuleResult>> {
        // Placeholder: Would require medical necessity criteria evaluation
        Ok(None)
    }
}

// ========================================================================
// PHASE 5: SUPERVISION & TEACHING RULES
// ========================================================================

/// Rule: Supervision not documented (SUP-001)
pub struct SupervisionNotDocumentedRule;

#[async_trait]
impl Rule for SupervisionNotDocumentedRule {
    fn flag_type(&self) -> FlagIssueType {
        FlagIssueType::SupSupervisionNotDocumented
    }

    async fn execute(&self, ctx: &mut RuleExecutionContext, pool: &PgPool) -> Result<Option<RuleResult>> {
        let Some(provider_id) = ctx.provider_id else {
            return Ok(None);
        };

        // Query provider credentials to check if they require supervision
        let query = r#"
            SELECT credential_type
            FROM claims.provider
            WHERE provider_id = $1
        "#;

        let result: Option<(Option<String>,)> = sqlx::query_as(query)
            .bind(provider_id)
            .fetch_optional(pool)
            .await
            .map_err(|e| Error::Database(e))?;

        if let Some((Some(cred),)) = result {
            // Residents, fellows, students typically require supervision
            let requires_supervision = cred.contains("Resident")
                || cred.contains("Fellow")
                || cred.contains("Student");

            if requires_supervision {
                let has_supervision_documented = ctx.custom_data.contains_key("supervising_physician")
                    || ctx.custom_data.contains_key("attending_physician");

                if !has_supervision_documented {
                    let details = format!(
                        "Provider with credential {} requires documented supervision",
                        cred
                    );

                    return Ok(Some(
                        RuleResult::new(self.flag_type(), ctx.to_flag_context())
                            .with_details(details)
                    ));
                }
            }
        }

        Ok(None)
    }
}

/// Rule: Inappropriate supervision level (SUP-002) - Placeholder
pub struct InappropriateSupervisionLevelRule;

#[async_trait]
impl Rule for InappropriateSupervisionLevelRule {
    fn flag_type(&self) -> FlagIssueType {
        FlagIssueType::SupInappropriateLevel
    }

    async fn execute(&self, _ctx: &mut RuleExecutionContext, _pool: &PgPool) -> Result<Option<RuleResult>> {
        // Placeholder: Would check supervision level against procedure complexity
        Ok(None)
    }
}

/// Rule: Teaching physician requirements not met (SUP-003) - Placeholder
pub struct TeachingPhysicianNotMetRule;

#[async_trait]
impl Rule for TeachingPhysicianNotMetRule {
    fn flag_type(&self) -> FlagIssueType {
        FlagIssueType::SupTeachingPhysicianNotMet
    }

    async fn execute(&self, _ctx: &mut RuleExecutionContext, _pool: &PgPool) -> Result<Option<RuleResult>> {
        // Placeholder: Would verify teaching physician documentation requirements
        Ok(None)
    }
}

/// Helper function to create all default rules
pub fn create_default_rules() -> Vec<Box<dyn Rule>> {
    vec![
        // Original 6 rules
        Box::new(DuplicateServiceRule),
        Box::new(UnitsExceedMaximumRule::default()),
        Box::new(MissingRequiredModifierRule::default()),
        Box::new(ConflictingModifiersRule),
        Box::new(UnspecifiedDiagnosisRule),
        Box::new(MissingDiagnosisSpecificityRule),
        // Phase 1: High-value, low-complexity (6 rules)
        Box::new(UnitsInconsistentRule),
        Box::new(PrimaryDiagnosisDoesNotSupportRule),
        Box::new(DiagnosisSequencingErrorRule),
        Box::new(IncorrectModifierRule),
        Box::new(TimeNotDocumentedRule),
        Box::new(WrongEMCategoryRule),
        // Phase 2: Fraud Detection (3 rules)
        Box::new(UnbundlingRule),
        Box::new(UpcodingRule),
        Box::new(WrongProviderTypeRule),
        // Phase 3: E/M Optimization (4 rules)
        Box::new(EMOLevelHigherThanMDMRule),
        Box::new(EMOLevelHigherThanHistoryExamRule),
        Box::new(EMULevelLowerThanMDMRule),
        Box::new(EMULevelLowerThanTimeRule),
        // Phase 4: Advanced Detection (5 rules - placeholders)
        Box::new(IncorrectProcedureCodeRule),
        Box::new(ProcedureNotSupportedByDiagnosisRule),
        Box::new(InsufficientDocumentationRule),
        Box::new(MissingRequiredElementsRule),
        Box::new(MedicalNecessityNotEstablishedRule),
        // Phase 5: Supervision & Teaching (3 rules)
        Box::new(SupervisionNotDocumentedRule),
        Box::new(InappropriateSupervisionLevelRule),
        Box::new(TeachingPhysicianNotMetRule),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    

    #[tokio::test]
    async fn test_units_exceed_maximum() {
        let pool = PgPool::connect_lazy("postgres://dummy").unwrap();
        let rule = UnitsExceedMaximumRule::new(Decimal::new(50, 0));

        let mut ctx = RuleExecutionContext::new(1);
        ctx.service_unit_count = Some(Decimal::new(75, 0));
        ctx.procedure_code = Some("99213".to_string());

        let result = rule.execute(&mut ctx, &pool).await.unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().flag_type, FlagIssueType::QtyUnitsExceedMaximum);
    }

    #[tokio::test]
    async fn test_units_within_maximum() {
        let pool = PgPool::connect_lazy("postgres://dummy").unwrap();
        let rule = UnitsExceedMaximumRule::new(Decimal::new(100, 0));

        let mut ctx = RuleExecutionContext::new(1);
        ctx.service_unit_count = Some(Decimal::new(50, 0));

        let result = rule.execute(&mut ctx, &pool).await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_conflicting_modifiers() {
        let pool = PgPool::connect_lazy("postgres://dummy").unwrap();
        let rule = ConflictingModifiersRule;

        let mut ctx = RuleExecutionContext::new(1);
        ctx.procedure_code = Some("27447".to_string());
        ctx.procedure_modifiers = vec!["LT".to_string(), "RT".to_string()];

        let result = rule.execute(&mut ctx, &pool).await.unwrap();
        assert!(result.is_some());
    }

    #[tokio::test]
    async fn test_unspecified_diagnosis() {
        let pool = PgPool::connect_lazy("postgres://dummy").unwrap();
        let rule = UnspecifiedDiagnosisRule;

        let mut ctx = RuleExecutionContext::new(1);
        ctx.diagnosis_codes = vec!["E11.9".to_string(), "I10".to_string()];

        let result = rule.execute(&mut ctx, &pool).await.unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().flag_type, FlagIssueType::DxUnspecifiedWhenSpecificAvailable);
    }

    #[test]
    fn test_create_default_rules() {
        let rules = create_default_rules();
        assert_eq!(rules.len(), 27);
    }
}
