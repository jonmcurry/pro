// FieldPatternRule Template - Regex pattern matching
//
// Validates field values against regex patterns
//
// Example use cases:
// - Validate procedure codes match expected format (e.g., 5-digit CPT codes)
// - Validate diagnosis codes are properly formatted
// - Flag invalid modifiers

use crate::flag_types::FlagIssueType;
use crate::rule_engine::{Rule, RuleExecutionContext, RuleResult};
use crate::template::{get_string_param, ParameterSchema, RuleTemplate};
use async_trait::async_trait;
use pro_common::{Error, Result};
use regex::Regex;
use serde_json::Value as JsonValue;
use sqlx::PgPool;
use std::sync::Arc;

/// FieldPatternRule template
pub struct FieldPatternRuleTemplate;

impl RuleTemplate for FieldPatternRuleTemplate {
    fn template_code(&self) -> &str {
        "FIELD_PATTERN"
    }

    fn template_name(&self) -> &str {
        "Field Pattern Validation Rule"
    }

    fn parameter_schema(&self) -> Vec<ParameterSchema> {
        vec![
            ParameterSchema {
                name: "field".to_string(),
                param_type: "string".to_string(),
                required: true,
                description: "Field to validate".to_string(),
                default: None,
                min: None,
                max: None,
                pattern: None,
                enum_values: Some(vec![
                    "procedure_code".to_string(),
                    "place_of_service_code".to_string(),
                    "provider_type".to_string(),
                    "provider_specialty".to_string(),
                ]),
            },
            ParameterSchema {
                name: "pattern".to_string(),
                param_type: "string".to_string(),
                required: true,
                description: "Regex pattern to match (e.g., '^\\d{5}$' for 5-digit codes)"
                    .to_string(),
                default: None,
                min: None,
                max: None,
                pattern: None,
                enum_values: None,
            },
            ParameterSchema {
                name: "invert_match".to_string(),
                param_type: "boolean".to_string(),
                required: false,
                description: "If true, flag when pattern DOES match (for exclusion rules)"
                    .to_string(),
                default: Some(serde_json::json!(false)),
                min: None,
                max: None,
                pattern: None,
                enum_values: None,
            },
            ParameterSchema {
                name: "case_sensitive".to_string(),
                param_type: "boolean".to_string(),
                required: false,
                description: "Whether regex should be case-sensitive".to_string(),
                default: Some(serde_json::json!(true)),
                min: None,
                max: None,
                pattern: None,
                enum_values: None,
            },
            ParameterSchema {
                name: "allow_null".to_string(),
                param_type: "boolean".to_string(),
                required: false,
                description: "If true, NULL/empty values pass validation".to_string(),
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

        let field = get_string_param(&params, "field")?;
        let pattern_str = get_string_param(&params, "pattern")?;

        let invert_match = params
            .get("invert_match")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let case_sensitive = params
            .get("case_sensitive")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        let allow_null = params
            .get("allow_null")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        // Compile regex
        let regex = if case_sensitive {
            Regex::new(&pattern_str)
        } else {
            Regex::new(&format!("(?i){}", pattern_str))
        }
        .map_err(|e| Error::Config(format!("Invalid regex pattern: {}", e)))?;

        Ok(Arc::new(FieldPatternRule {
            rule_code,
            rule_name,
            flag_issue_type,
            field,
            pattern: pattern_str,
            regex,
            invert_match,
            case_sensitive,
            allow_null,
        }))
    }
}

/// Concrete FieldPatternRule instance
#[derive(Debug, Clone)]
pub struct FieldPatternRule {
    pub rule_code: String,
    pub rule_name: String,
    pub flag_issue_type: FlagIssueType,
    pub field: String,
    pub pattern: String,
    pub regex: Regex,
    pub invert_match: bool,
    pub case_sensitive: bool,
    pub allow_null: bool,
}

#[async_trait]
impl Rule for FieldPatternRule {
    fn flag_type(&self) -> FlagIssueType {
        self.flag_issue_type
    }

    async fn execute(&self, ctx: &RuleExecutionContext, _pool: &PgPool) -> Result<Option<RuleResult>> {
        let field_value = self.get_field_value(ctx);

        let is_invalid = self.check_pattern(&field_value);

        if is_invalid {
            let message = if self.invert_match {
                format!(
                    "Field '{}' matches excluded pattern: {}",
                    self.field, self.pattern
                )
            } else {
                format!(
                    "Field '{}' does not match required pattern: {}",
                    self.field, self.pattern
                )
            };

            Ok(Some(
                RuleResult::new(self.flag_issue_type, ctx.to_flag_context())
                    .with_details(message)
            ))
        } else {
            Ok(None)
        }
    }
}

impl FieldPatternRule {
    fn get_field_value(&self, ctx: &RuleExecutionContext) -> Option<String> {
        match self.field.as_str() {
            "procedure_code" => ctx.procedure_code.clone(),
            "place_of_service_code" => ctx.place_of_service_code.clone(),
            "provider_type" => ctx.provider_type.clone(),
            "provider_specialty" => ctx.provider_specialty.clone(),
            _ => None,
        }
    }

    fn check_pattern(&self, field_value: &Option<String>) -> bool {
        match field_value {
            None => !self.allow_null,
            Some(s) if s.trim().is_empty() => !self.allow_null,
            Some(s) => {
                let matches = self.regex.is_match(s);
                if self.invert_match {
                    matches // Flag if it DOES match (exclusion rule)
                } else {
                    !matches // Flag if it DOESN'T match (validation rule)
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_template_schema() {
        let template = FieldPatternRuleTemplate;
        let schema = template.parameter_schema();

        assert_eq!(schema.len(), 5);
        assert_eq!(schema[0].name, "field");
        assert_eq!(schema[1].name, "pattern");
        assert_eq!(schema[2].name, "invert_match");
        assert!(schema[0].required);
        assert!(schema[1].required);
        assert!(!schema[2].required); // has default
    }

    #[test]
    fn test_instantiate_valid_regex() {
        let template = FieldPatternRuleTemplate;
        let params = serde_json::json!({
            "field": "procedure_code",
            "pattern": "^\\d{5}$"
        });

        let result = template.instantiate(
            "TEST_RULE".to_string(),
            "Test Rule".to_string(),
            FlagIssueType::ModIncorrect,
            params,
        );

        assert!(result.is_ok());
    }

    #[test]
    fn test_instantiate_invalid_regex() {
        let template = FieldPatternRuleTemplate;
        let params = serde_json::json!({
            "field": "procedure_code",
            "pattern": "[invalid(regex"
        });

        let result = template.instantiate(
            "TEST_RULE".to_string(),
            "Test Rule".to_string(),
            FlagIssueType::ModIncorrect,
            params,
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_check_pattern() {
        let regex = Regex::new(r"^\d{5}$").unwrap();
        let rule = FieldPatternRule {
            rule_code: "TEST".to_string(),
            rule_name: "Test".to_string(),
            flag_issue_type: FlagIssueType::ModIncorrect,
            field: "procedure_code".to_string(),
            pattern: r"^\d{5}$".to_string(),
            regex,
            invert_match: false,
            case_sensitive: true,
            allow_null: true,
        };

        // Valid 5-digit code
        assert!(!rule.check_pattern(&Some("12345".to_string())));

        // Invalid 4-digit code
        assert!(rule.check_pattern(&Some("1234".to_string())));

        // NULL allowed
        assert!(!rule.check_pattern(&None));

        // Test invert_match
        let inverted_rule = FieldPatternRule {
            invert_match: true,
            ..rule.clone()
        };

        // Should flag when pattern DOES match
        assert!(inverted_rule.check_pattern(&Some("12345".to_string())));
        assert!(!inverted_rule.check_pattern(&Some("1234".to_string())));
    }
}
