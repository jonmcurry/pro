// Rule loader - loads rules from database and instantiates them

use crate::flag_types::FlagIssueType;
use crate::rule_engine::{Rule, RuleEngine};
use crate::rules::*;
use crate::template::RuleTemplate;
use crate::templates::*;
use pro_common::{Error, Result};
use rust_decimal::Decimal;
use serde_json::Value as JsonValue;
use sqlx::{PgPool, Row};
use std::sync::Arc;
use tracing::{debug, info, warn};

/// Loaded rule metadata from database
#[derive(Debug, Clone)]
pub struct LoadedRuleInfo {
    pub rule_id: i64,
    pub rule_code: String,
    pub rule_name: String,
    pub template_code: Option<String>,
    pub execution_order: i32,
    pub execution_level: String,
}

/// Load all active rules from database and instantiate them
///
/// This function:
/// 1. Reads RULE_ENCRYPTION_KEY from environment
/// 2. Queries active rules for the facility (or all if facility_id is None)
/// 3. Decrypts rule parameters
/// 4. Instantiates appropriate Rust rule implementations
/// 5. Returns configured RuleEngine ready for execution
///
/// # Arguments
/// * `pool` - Database connection pool
/// * `facility_id` - Optional facility ID to load rules for (None = all global rules)
///
/// # Returns
/// * `Ok(RuleEngine)` - Configured engine with all active rules loaded
/// * `Err(Error)` - If encryption key not set or database error
pub async fn load_rules_from_database(
    pool: &PgPool,
    facility_id: Option<i64>,
) -> Result<(RuleEngine, Vec<LoadedRuleInfo>)> {
    info!("Loading rules from database (facility_id: {:?})", facility_id);

    // Get encryption key from environment
    let encryption_key = std::env::var("RULE_ENCRYPTION_KEY").map_err(|_| {
        Error::Config("RULE_ENCRYPTION_KEY environment variable not set".into())
    })?;

    let mut engine = RuleEngine::new(pool.clone());
    let mut loaded_rules = Vec::new();

    // Query active rules for facility (or all if facility_id is None)
    let rows = if let Some(fid) = facility_id {
        query_facility_rules(pool, fid, &encryption_key).await?
    } else {
        query_global_rules(pool, &encryption_key).await?
    };

    info!("Found {} rule(s) to load", rows.len());

    for row in rows {
        let rule_id: i64 = row.get("rule_id");
        let rule_code: String = row.get("rule_code");
        let rule_name: String = row.get("rule_name");
        let template_code: Option<String> = row.get("template_code");
        let execution_order: i32 = row.get("execution_order");
        let execution_level: String = row.get("execution_level");

        debug!(
            "Loading rule: {} ({}) - template: {:?}",
            rule_code, rule_name, template_code
        );

        // Instantiate rule based on template
        match instantiate_rule(&rule_code, &template_code, &row) {
            Ok(rule) => {
                engine.add_rule_arc(rule);
                loaded_rules.push(LoadedRuleInfo {
                    rule_id,
                    rule_code: rule_code.clone(),
                    rule_name,
                    template_code,
                    execution_order,
                    execution_level,
                });
                debug!("✓ Successfully loaded rule: {}", rule_code);
            }
            Err(e) => {
                warn!("Failed to instantiate rule {}: {} - skipping", rule_code, e);
            }
        }
    }

    info!("Successfully loaded {} rule(s)", loaded_rules.len());

    Ok((engine, loaded_rules))
}

/// Query rules for a specific facility
async fn query_facility_rules(
    pool: &PgPool,
    facility_id: i64,
    encryption_key: &str,
) -> Result<Vec<sqlx::postgres::PgRow>> {
    let query = r#"
        SELECT
            rd.rule_id,
            rd.rule_code,
            rd.rule_name,
            rd.template_id,
            rt.template_code,
            rt.rust_struct_name,
            CASE
                WHEN rd.rule_parameters_encrypted IS NOT NULL
                THEN pgp_sym_decrypt(rd.rule_parameters_encrypted, $1)::text
                ELSE NULL
            END AS rule_parameters,
            CASE
                WHEN fra.parameter_overrides_encrypted IS NOT NULL
                THEN pgp_sym_decrypt(fra.parameter_overrides_encrypted, $1)::text
                WHEN ora.parameter_overrides_encrypted IS NOT NULL
                THEN pgp_sym_decrypt(ora.parameter_overrides_encrypted, $1)::text
                ELSE NULL
            END AS parameter_overrides,
            rd.execution_order,
            rd.execution_level,
            rd.flag_issue_id,
            fi.issue_code,
            claims.get_flag_issue_type_name(fi.issue_code) AS flag_issue_type_name,
            vafr.assignment_level
        FROM claims.v_active_facility_rules vafr
        INNER JOIN claims.rule_definition rd ON vafr.rule_id = rd.rule_id
        INNER JOIN claims.flag_issue fi ON rd.flag_issue_id = fi.issue_id
        LEFT JOIN claims.rule_template rt ON rd.template_id = rt.template_id
        LEFT JOIN claims.facility_rule_assignment fra ON (
            fra.facility_id = vafr.facility_id AND fra.rule_id = rd.rule_id
        )
        LEFT JOIN claims.organization_rule_assignment ora ON (
            ora.organization_id = vafr.organization_id
            AND ora.rule_id = rd.rule_id
            AND fra.assignment_id IS NULL
        )
        WHERE vafr.facility_id = $2
        AND vafr.is_enabled = true
        ORDER BY rd.execution_order
    "#;

    let rows = sqlx::query(query)
        .bind(encryption_key)
        .bind(facility_id)
        .fetch_all(pool)
        .await
        .map_err(|e| Error::Database(e))?;

    Ok(rows)
}

/// Query all globally active rules (no facility filter)
pub(crate) async fn query_global_rules(
    pool: &PgPool,
    encryption_key: &str,
) -> Result<Vec<sqlx::postgres::PgRow>> {
    let query = r#"
        SELECT
            rd.rule_id,
            rd.rule_code,
            rd.rule_name,
            rd.template_id,
            rt.template_code,
            rt.rust_struct_name,
            CASE
                WHEN rd.rule_parameters_encrypted IS NOT NULL
                THEN pgp_sym_decrypt(rd.rule_parameters_encrypted, $1)::text
                ELSE NULL
            END AS rule_parameters,
            NULL::text AS parameter_overrides,
            rd.execution_order,
            rd.execution_level,
            rd.flag_issue_id,
            fi.issue_code,
            claims.get_flag_issue_type_name(fi.issue_code) AS flag_issue_type_name,
            'GLOBAL' AS assignment_level
        FROM claims.rule_definition rd
        INNER JOIN claims.flag_issue fi ON rd.flag_issue_id = fi.issue_id
        LEFT JOIN claims.rule_template rt ON rd.template_id = rt.template_id
        WHERE rd.is_active = true
        AND (rt.is_active IS NULL OR rt.is_active = true)
        ORDER BY rd.execution_order
    "#;

    let rows = sqlx::query(query)
        .bind(encryption_key)
        .fetch_all(pool)
        .await
        .map_err(|e| Error::Database(e))?;

    Ok(rows)
}

/// Instantiate a rule from database row data
/// Convert flag_issue_type_name string to FlagIssueType enum
/// PHASE 4: Maps database enum name to Rust enum variant
/// NOTE: Only maps to variants that exist in the current FlagIssueType enum
fn parse_flag_issue_type(type_name: &str) -> Result<FlagIssueType> {
    match type_name {
        // Coding (COD) issues
        "CodIncorrectProcedureCode" => Ok(FlagIssueType::CodIncorrectProcedureCode),
        "CodProcedureNotSupportedByDiagnosis" => Ok(FlagIssueType::CodProcedureNotSupportedByDiagnosis),
        "CodUnbundling" => Ok(FlagIssueType::CodUnbundling),
        "CodUpcoding" => Ok(FlagIssueType::CodUpcoding),

        // Documentation (DOC) issues
        "DocInsufficientDocumentation" => Ok(FlagIssueType::DocInsufficientDocumentation),
        "DocMissingRequiredElements" => Ok(FlagIssueType::DocMissingRequiredElements),

        // E/M Overcoded (EMO) issues
        "EMOLevelHigherThanMDM" => Ok(FlagIssueType::EMOLevelHigherThanMDM),
        "EMOLevelHigherThanHistoryExam" => Ok(FlagIssueType::EMOLevelHigherThanHistoryExam),

        // E/M Undercoded (EMU) issues
        "EMULevelLowerThanMDM" => Ok(FlagIssueType::EMULevelLowerThanMDM),
        "EMULevelLowerThanTime" => Ok(FlagIssueType::EMULevelLowerThanTime),

        // E/M Incorrect Category (EMI)
        "EMIWrongCategory" => Ok(FlagIssueType::EMIWrongCategory),

        // E/M Time Not Documented (EMT)
        "EMTTimeNotDocumented" => Ok(FlagIssueType::EMTTimeNotDocumented),

        // Modifier (MOD) issues
        "ModMissingRequired" => Ok(FlagIssueType::ModMissingRequired),
        "ModIncorrect" => Ok(FlagIssueType::ModIncorrect),
        "ModConflicting" => Ok(FlagIssueType::ModConflicting),

        // Other (OTH) issues
        "OthMedicalNecessityNotEstablished" => Ok(FlagIssueType::OthMedicalNecessityNotEstablished),
        "OthWrongProviderType" => Ok(FlagIssueType::OthWrongProviderType),
        "OthDuplicateService" => Ok(FlagIssueType::OthDuplicateService),

        // Quantity/Units (QTY) issues
        "QtyUnitsExceedMaximum" => Ok(FlagIssueType::QtyUnitsExceedMaximum),
        "QtyUnitsInconsistent" => Ok(FlagIssueType::QtyUnitsInconsistent),

        // Supervision (SUP) issues
        "SupSupervisionNotDocumented" => Ok(FlagIssueType::SupSupervisionNotDocumented),
        "SupInappropriateLevel" => Ok(FlagIssueType::SupInappropriateLevel),
        "SupTeachingPhysicianNotMet" => Ok(FlagIssueType::SupTeachingPhysicianNotMet),

        // Diagnosis (DX) issues
        "DxPrimaryDoesNotSupport" => Ok(FlagIssueType::DxPrimaryDoesNotSupport),
        "DxMissingSpecificity" => Ok(FlagIssueType::DxMissingSpecificity),
        "DxSequencingError" => Ok(FlagIssueType::DxSequencingError),
        "DxUnspecifiedWhenSpecificAvailable" => Ok(FlagIssueType::DxUnspecifiedWhenSpecificAvailable),

        _ => {
            // Log warning and use a default fallback
            warn!("Unknown flag issue type '{}' - defaulting to OthDuplicateService", type_name);
            Ok(FlagIssueType::OthDuplicateService)
        }
    }
}

pub(crate) fn instantiate_rule(
    rule_code: &str,
    template_code: &Option<String>,
    row: &sqlx::postgres::PgRow,
) -> Result<Arc<dyn Rule>> {
    // Get rule metadata from row
    let rule_name: String = row.get("rule_name");
    let flag_issue_type_name: String = row.get("flag_issue_type_name");

    // PHASE 4: Parse flag_issue_type from database
    let flag_issue_type = parse_flag_issue_type(&flag_issue_type_name)?;

    // Get parameters (already decrypted in query)
    let rule_parameters: Option<String> = row.get("rule_parameters");
    let parameter_overrides: Option<String> = row.get("parameter_overrides");

    // Merge base parameters with overrides
    let params = merge_parameters(rule_parameters, parameter_overrides)?;

    match template_code.as_deref() {
        // Legacy hard-coded rules (Phase 1)
        Some("LEGACY") => match rule_code {
            "DUPLICATE_SERVICE" => Ok(Arc::new(DuplicateServiceRule)),
            "UNITS_EXCEED_MAX" => Ok(Arc::new(UnitsExceedMaximumRule::new(Decimal::from(999)))),
            "MISSING_REQUIRED_MODIFIER" => {
                Ok(Arc::new(MissingRequiredModifierRule::new(Vec::new())))
            }
            "CONFLICTING_MODIFIERS" => Ok(Arc::new(ConflictingModifiersRule)),
            "UNSPECIFIED_DIAGNOSIS" => Ok(Arc::new(UnspecifiedDiagnosisRule)),
            "MISSING_DIAGNOSIS_SPECIFICITY" => Ok(Arc::new(MissingDiagnosisSpecificityRule)),
            _ => Err(Error::Config(format!("Unknown legacy rule: {}", rule_code))),
        },

        // Template-based rules (Phase 3)
        Some("THRESHOLD") => {
            let template = ThresholdRuleTemplate;
            template.instantiate(rule_code.to_string(), rule_name, flag_issue_type, params)
        }
        Some("DUPLICATE") => {
            let template = DuplicateRuleTemplate;
            template.instantiate(rule_code.to_string(), rule_name, flag_issue_type, params)
        }
        Some("MISSING_FIELD") => {
            let template = MissingFieldRuleTemplate;
            template.instantiate(rule_code.to_string(), rule_name, flag_issue_type, params)
        }
        Some("FIELD_PATTERN") => {
            let template = FieldPatternRuleTemplate;
            template.instantiate(rule_code.to_string(), rule_name, flag_issue_type, params)
        }
        Some("CROSS_FIELD") => {
            let template = CrossFieldRuleTemplate;
            template.instantiate(rule_code.to_string(), rule_name, flag_issue_type, params)
        }

        // Unknown template
        Some(template) => Err(Error::Config(format!(
            "Unknown template '{}' for rule '{}'",
            template, rule_code
        ))),

        // No template defined
        None => Err(Error::Config(format!(
            "Rule '{}' has no template defined",
            rule_code
        ))),
    }
}

/// Merge base parameters with facility/org overrides
fn merge_parameters(base: Option<String>, overrides: Option<String>) -> Result<JsonValue> {
    let mut params: JsonValue = match base {
        Some(json_str) if !json_str.trim().is_empty() => {
            serde_json::from_str(&json_str).map_err(|e| {
                Error::Config(format!("Invalid rule parameters JSON: {}", e))
            })?
        }
        _ => serde_json::json!({}),
    };

    if let Some(override_str) = overrides {
        if !override_str.trim().is_empty() {
            let override_val: JsonValue = serde_json::from_str(&override_str).map_err(|e| {
                Error::Config(format!("Invalid parameter overrides JSON: {}", e))
            })?;
            merge_json(&mut params, override_val);
        }
    }

    Ok(params)
}

/// Recursively merge two JSON values (overlay takes precedence)
fn merge_json(base: &mut JsonValue, overlay: JsonValue) {
    if let (Some(base_obj), Some(overlay_obj)) = (base.as_object_mut(), overlay.as_object()) {
        for (key, value) in overlay_obj {
            base_obj.insert(key.clone(), value.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge_parameters_empty() {
        let result = merge_parameters(None, None).unwrap();
        assert_eq!(result, serde_json::json!({}));
    }

    #[test]
    fn test_merge_parameters_base_only() {
        let base = Some(r#"{"threshold": 100}"#.to_string());
        let result = merge_parameters(base, None).unwrap();
        assert_eq!(result, serde_json::json!({"threshold": 100}));
    }

    #[test]
    fn test_merge_parameters_with_override() {
        let base = Some(r#"{"threshold": 100, "field": "total_charge"}"#.to_string());
        let overrides = Some(r#"{"threshold": 200}"#.to_string());
        let result = merge_parameters(base, overrides).unwrap();
        assert_eq!(
            result,
            serde_json::json!({"threshold": 200, "field": "total_charge"})
        );
    }

    #[test]
    fn test_merge_json() {
        let mut base = serde_json::json!({"a": 1, "b": 2});
        let overlay = serde_json::json!({"b": 3, "c": 4});
        merge_json(&mut base, overlay);
        assert_eq!(base, serde_json::json!({"a": 1, "b": 3, "c": 4}));
    }
}
