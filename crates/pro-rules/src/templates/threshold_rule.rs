// ThresholdRule Template - Numeric comparison rules
//
// Checks if a numeric field meets a threshold condition (>, <, >=, <=, ==, !=)
//
// Example use cases:
// - Flag charges over $10,000
// - Flag claims with units > 100
// - Flag claims with total charge < $0

use crate::flag_types::FlagIssueType;
use crate::rule_engine::{Rule, RuleExecutionContext, RuleResult};
use crate::template::{
    get_decimal_param, get_optional_decimal_param, get_string_param, ParameterSchema,
    RuleTemplate,
};
use async_trait::async_trait;
use pro_common::{Error, Result};
use rust_decimal::Decimal;
use serde_json::Value as JsonValue;
use sqlx::PgPool;
use std::sync::Arc;

/// Comparison operator
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ComparisonOperator {
    GreaterThan,
    GreaterThanOrEqual,
    LessThan,
    LessThanOrEqual,
    Equal,
    NotEqual,
}

impl ComparisonOperator {
    pub fn from_string(s: &str) -> Result<Self> {
        match s {
            ">" | "gt" => Ok(Self::GreaterThan),
            ">=" | "gte" => Ok(Self::GreaterThanOrEqual),
            "<" | "lt" => Ok(Self::LessThan),
            "<=" | "lte" => Ok(Self::LessThanOrEqual),
            "==" | "eq" => Ok(Self::Equal),
            "!=" | "ne" => Ok(Self::NotEqual),
            _ => Err(Error::Config(format!("Invalid operator: {}", s))),
        }
    }

    pub fn compare(&self, value: Decimal, threshold: Decimal) -> bool {
        match self {
            Self::GreaterThan => value > threshold,
            Self::GreaterThanOrEqual => value >= threshold,
            Self::LessThan => value < threshold,
            Self::LessThanOrEqual => value <= threshold,
            Self::Equal => value == threshold,
            Self::NotEqual => value != threshold,
        }
    }
}

/// ThresholdRule template
pub struct ThresholdRuleTemplate;

impl RuleTemplate for ThresholdRuleTemplate {
    fn template_code(&self) -> &str {
        "THRESHOLD"
    }

    fn template_name(&self) -> &str {
        "Threshold Comparison Rule"
    }

    fn parameter_schema(&self) -> Vec<ParameterSchema> {
        vec![
            ParameterSchema {
                name: "field".to_string(),
                param_type: "string".to_string(),
                required: true,
                description: "Field to check (e.g., 'total_charge', 'units')".to_string(),
                default: None,
                min: None,
                max: None,
                pattern: None,
                enum_values: Some(vec![
                    "total_charge".to_string(),
                    "units".to_string(),
                    "line_item_charge_amount".to_string(),
                    "total_claim_charge_amount".to_string(),
                ]),
            },
            ParameterSchema {
                name: "operator".to_string(),
                param_type: "string".to_string(),
                required: true,
                description: "Comparison operator".to_string(),
                default: None,
                min: None,
                max: None,
                pattern: None,
                enum_values: Some(vec![
                    ">".to_string(),
                    ">=".to_string(),
                    "<".to_string(),
                    "<=".to_string(),
                    "==".to_string(),
                    "!=".to_string(),
                ]),
            },
            ParameterSchema {
                name: "threshold".to_string(),
                param_type: "number".to_string(),
                required: true,
                description: "Threshold value to compare against".to_string(),
                default: None,
                min: None,
                max: None,
                pattern: None,
                enum_values: None,
            },
            ParameterSchema {
                name: "min_threshold".to_string(),
                param_type: "number".to_string(),
                required: false,
                description: "Optional minimum threshold for range checks".to_string(),
                default: None,
                min: None,
                max: None,
                pattern: None,
                enum_values: None,
            },
            ParameterSchema {
                name: "max_threshold".to_string(),
                param_type: "number".to_string(),
                required: false,
                description: "Optional maximum threshold for range checks".to_string(),
                default: None,
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
        issue_code: String,
        params: JsonValue,
    ) -> Result<Arc<dyn Rule>> {
        self.validate_parameters(&params)?;

        let field = get_string_param(&params, "field")?;
        let operator_str = get_string_param(&params, "operator")?;
        let operator = ComparisonOperator::from_string(&operator_str)?;
        let threshold = get_decimal_param(&params, "threshold")?;
        let min_threshold = get_optional_decimal_param(&params, "min_threshold")?;
        let max_threshold = get_optional_decimal_param(&params, "max_threshold")?;

        Ok(Arc::new(ThresholdRule {
            rule_code,
            rule_name,
            flag_issue_type,
            issue_code,
            field,
            operator,
            threshold,
            min_threshold,
            max_threshold,
        }))
    }
}

/// Concrete ThresholdRule instance
#[derive(Debug, Clone)]
pub struct ThresholdRule {
    pub rule_code: String,
    pub rule_name: String,
    pub flag_issue_type: FlagIssueType,
    pub issue_code: String,
    pub field: String,
    pub operator: ComparisonOperator,
    pub threshold: Decimal,
    pub min_threshold: Option<Decimal>,
    pub max_threshold: Option<Decimal>,
}

#[async_trait]
impl Rule for ThresholdRule {
    fn flag_type(&self) -> FlagIssueType {
        self.flag_issue_type
    }

    /// THRESHOLD rules don't need database access - they're pure CPU evaluation
    fn requires_db_access(&self) -> bool {
        false
    }

    /// Synchronous execution - avoids async overhead for CPU-only rules
    fn execute_sync(&self, ctx: &mut RuleExecutionContext) -> Result<Option<RuleResult>> {
        self.evaluate(ctx)
    }

    async fn execute(&self, ctx: &mut RuleExecutionContext, _pool: &PgPool) -> Result<Option<RuleResult>> {
        self.evaluate(ctx)
    }
}

impl ThresholdRule {
    /// Core evaluation logic - shared between sync and async paths
    fn evaluate(&self, ctx: &RuleExecutionContext) -> Result<Option<RuleResult>> {
        let value = match self.get_field_value(ctx) {
            Some(v) => v,
            None => return Ok(None), // Field not available, skip
        };

        // Check range if specified
        if let Some(min) = self.min_threshold {
            if value < min {
                return Ok(Some(
                    RuleResult::new(self.flag_issue_type, ctx.to_flag_context())
                        .with_details(format!(
                            "{} ({}) is below minimum threshold ({})",
                            self.field, value, min
                        ))
                        .with_issue_code(self.issue_code.clone())
                ));
            }
        }

        if let Some(max) = self.max_threshold {
            if value > max {
                return Ok(Some(
                    RuleResult::new(self.flag_issue_type, ctx.to_flag_context())
                        .with_details(format!(
                            "{} ({}) exceeds maximum threshold ({})",
                            self.field, value, max
                        ))
                        .with_issue_code(self.issue_code.clone())
                ));
            }
        }

        // Check main threshold
        if self.operator.compare(value, self.threshold) {
            Ok(Some(
                RuleResult::new(self.flag_issue_type, ctx.to_flag_context())
                    .with_details(format!(
                        "{} ({}) {} threshold ({})",
                        self.field,
                        value,
                        match self.operator {
                            ComparisonOperator::GreaterThan => "exceeds",
                            ComparisonOperator::GreaterThanOrEqual => "exceeds or equals",
                            ComparisonOperator::LessThan => "is below",
                            ComparisonOperator::LessThanOrEqual => "is below or equals",
                            ComparisonOperator::Equal => "equals",
                            ComparisonOperator::NotEqual => "does not equal",
                        },
                        self.threshold
                    ))
                    .with_issue_code(self.issue_code.clone())
            ))
        } else {
            Ok(None)
        }
    }

    fn get_field_value(&self, ctx: &RuleExecutionContext) -> Option<Decimal> {
        match self.field.as_str() {
            "units" => ctx.service_unit_count,
            "line_item_charge_amount" => ctx.line_item_charge_amount,
            "total_claim_charge_amount" => ctx.total_claim_charge_amount,
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_comparison_operators() {
        let val = Decimal::from(100);
        let threshold = Decimal::from(50);

        assert!(ComparisonOperator::GreaterThan.compare(val, threshold));
        assert!(ComparisonOperator::GreaterThanOrEqual.compare(val, threshold));
        assert!(!ComparisonOperator::LessThan.compare(val, threshold));
        assert!(!ComparisonOperator::LessThanOrEqual.compare(val, threshold));
        assert!(!ComparisonOperator::Equal.compare(val, threshold));
        assert!(ComparisonOperator::NotEqual.compare(val, threshold));
    }

    #[test]
    fn test_operator_parsing() {
        assert_eq!(
            ComparisonOperator::from_string(">").unwrap(),
            ComparisonOperator::GreaterThan
        );
        assert_eq!(
            ComparisonOperator::from_string("gt").unwrap(),
            ComparisonOperator::GreaterThan
        );
        assert_eq!(
            ComparisonOperator::from_string(">=").unwrap(),
            ComparisonOperator::GreaterThanOrEqual
        );
        assert!(ComparisonOperator::from_string("invalid").is_err());
    }

    #[test]
    fn test_template_schema() {
        let template = ThresholdRuleTemplate;
        let schema = template.parameter_schema();

        assert_eq!(schema.len(), 5);
        assert_eq!(schema[0].name, "field");
        assert_eq!(schema[1].name, "operator");
        assert_eq!(schema[2].name, "threshold");
        assert!(schema[0].required);
        assert!(!schema[3].required); // min_threshold
    }
}
