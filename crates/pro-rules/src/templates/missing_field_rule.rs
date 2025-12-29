// MissingFieldRule Template - Required field validation
//
// Checks if required fields are present and non-empty
//
// Example use cases:
// - Flag service lines missing procedure codes
// - Flag service lines missing units
// - Flag encounters missing patient information

use crate::flag_types::FlagIssueType;
use crate::rule_engine::{Rule, RuleExecutionContext, RuleResult};
use crate::template::{get_string_array_param, ParameterSchema, RuleTemplate};
use async_trait::async_trait;
use pro_common::Result;
use serde_json::Value as JsonValue;
use sqlx::PgPool;
use std::sync::Arc;

/// MissingFieldRule template
pub struct MissingFieldRuleTemplate;

impl RuleTemplate for MissingFieldRuleTemplate {
    fn template_code(&self) -> &str {
        "MISSING_FIELD"
    }

    fn template_name(&self) -> &str {
        "Missing Required Field Rule"
    }

    fn parameter_schema(&self) -> Vec<ParameterSchema> {
        vec![
            ParameterSchema {
                name: "fields".to_string(),
                param_type: "array".to_string(),
                required: true,
                description: "List of required fields to check".to_string(),
                default: None,
                min: None,
                max: None,
                pattern: None,
                enum_values: None,
            },
            ParameterSchema {
                name: "check_empty".to_string(),
                param_type: "boolean".to_string(),
                required: false,
                description: "Also flag empty strings (not just NULL)".to_string(),
                default: Some(serde_json::json!(true)),
                min: None,
                max: None,
                pattern: None,
                enum_values: None,
            },
        ]
    }

    fn instantiate(
        &self,
        rule_code: String,
        rule_name: String,
        flag_issue_type: FlagIssueType,
        params: JsonValue,
    ) -> Result<Arc<dyn Rule>> {
        self.validate_parameters(&params)?;

        let fields = get_string_array_param(&params, "fields")?;

        let check_empty = params
            .get("check_empty")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        Ok(Arc::new(MissingFieldRule {
            rule_code,
            rule_name,
            flag_issue_type,
            fields,
            check_empty,
        }))
    }
}

/// Concrete MissingFieldRule instance
#[derive(Debug, Clone)]
pub struct MissingFieldRule {
    pub rule_code: String,
    pub rule_name: String,
    pub flag_issue_type: FlagIssueType,
    pub fields: Vec<String>,
    pub check_empty: bool,
}

#[async_trait]
impl Rule for MissingFieldRule {
    fn flag_type(&self) -> FlagIssueType {
        self.flag_issue_type
    }

    async fn execute(&self, ctx: &RuleExecutionContext, _pool: &PgPool) -> Result<Option<RuleResult>> {
        let mut missing_fields = Vec::new();

        for field in &self.fields {
            let is_missing = self.check_field_missing(ctx, field);
            if is_missing {
                missing_fields.push(field.clone());
            }
        }

        if !missing_fields.is_empty() {
            Ok(Some(
                RuleResult::new(self.flag_issue_type, ctx.to_flag_context())
                    .with_details(format!("Missing required field(s): {}", missing_fields.join(", ")))
            ))
        } else {
            Ok(None)
        }
    }
}

impl MissingFieldRule {
    fn check_field_missing(&self, ctx: &RuleExecutionContext, field: &str) -> bool {
        match field {
            // Service line fields
            "procedure_code" => self.is_string_missing(&ctx.procedure_code),
            "service_unit_count" => ctx.service_unit_count.is_none(),
            "line_item_charge_amount" => ctx.line_item_charge_amount.is_none(),
            "date_of_service" => ctx.date_of_service.is_none(),
            "place_of_service_code" => self.is_string_missing(&ctx.place_of_service_code),

            // Encounter fields
            "encounter_id" => ctx.encounter_id.is_none(),
            "facility_id" => ctx.facility_id.is_none(),
            "provider_id" => ctx.provider_id.is_none(),
            "total_claim_charge_amount" => ctx.total_claim_charge_amount.is_none(),
            "date_of_service_from" => ctx.date_of_service_from.is_none(),
            "date_of_service_to" => ctx.date_of_service_to.is_none(),

            // Provider fields
            "provider_type" => self.is_string_missing(&ctx.provider_type),
            "provider_specialty" => self.is_string_missing(&ctx.provider_specialty),

            // Unknown field
            _ => false,
        }
    }

    fn is_string_missing(&self, field: &Option<String>) -> bool {
        match field {
            None => true,
            Some(s) if self.check_empty && s.trim().is_empty() => true,
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_template_schema() {
        let template = MissingFieldRuleTemplate;
        let schema = template.parameter_schema();

        assert_eq!(schema.len(), 2);
        assert_eq!(schema[0].name, "fields");
        assert_eq!(schema[1].name, "check_empty");
        assert!(schema[0].required);
        assert!(!schema[1].required); // has default
    }

    #[test]
    fn test_instantiate() {
        let template = MissingFieldRuleTemplate;
        let params = serde_json::json!({
            "fields": ["procedure_code", "service_unit_count"]
        });

        let result = template.instantiate(
            "TEST_RULE".to_string(),
            "Test Rule".to_string(),
            FlagIssueType::ModMissingRequired,
            params,
        );

        assert!(result.is_ok());
    }

    #[test]
    fn test_is_string_missing() {
        let rule = MissingFieldRule {
            rule_code: "TEST".to_string(),
            rule_name: "Test".to_string(),
            flag_issue_type: FlagIssueType::ModMissingRequired,
            fields: vec!["test".to_string()],
            check_empty: true,
        };

        assert!(rule.is_string_missing(&None));
        assert!(rule.is_string_missing(&Some("".to_string())));
        assert!(rule.is_string_missing(&Some("   ".to_string())));
        assert!(!rule.is_string_missing(&Some("value".to_string())));

        let rule_no_empty_check = MissingFieldRule {
            check_empty: false,
            ..rule
        };

        assert!(rule_no_empty_check.is_string_missing(&None));
        assert!(!rule_no_empty_check.is_string_missing(&Some("".to_string())));
    }
}
