// CrossFieldRule Template - Compare two fields
//
// Validates relationships between two fields (e.g., field1 > field2)
//
// Example use cases:
// - Flag when total_charge < line_item_charge (data error)
// - Flag when date_of_service_to < date_of_service_from (invalid date range)

use crate::flag_types::FlagIssueType;
use crate::rule_engine::{Rule, RuleExecutionContext, RuleResult};
use crate::template::{get_string_param, ParameterSchema, RuleTemplate};
use async_trait::async_trait;
use chrono::NaiveDate;
use pro_common::{Error, Result};
use rust_decimal::Decimal;
use serde_json::Value as JsonValue;
use sqlx::PgPool;
use std::sync::Arc;

/// Comparison operator for cross-field validation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CrossFieldOperator {
    GreaterThan,
    GreaterThanOrEqual,
    LessThan,
    LessThanOrEqual,
    Equal,
    NotEqual,
}

impl CrossFieldOperator {
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

    pub fn compare_decimal(&self, left: Decimal, right: Decimal) -> bool {
        match self {
            Self::GreaterThan => left > right,
            Self::GreaterThanOrEqual => left >= right,
            Self::LessThan => left < right,
            Self::LessThanOrEqual => left <= right,
            Self::Equal => left == right,
            Self::NotEqual => left != right,
        }
    }

    pub fn compare_date(&self, left: NaiveDate, right: NaiveDate) -> bool {
        match self {
            Self::GreaterThan => left > right,
            Self::GreaterThanOrEqual => left >= right,
            Self::LessThan => left < right,
            Self::LessThanOrEqual => left <= right,
            Self::Equal => left == right,
            Self::NotEqual => left != right,
        }
    }

    pub fn as_str(&self) -> &str {
        match self {
            Self::GreaterThan => ">",
            Self::GreaterThanOrEqual => ">=",
            Self::LessThan => "<",
            Self::LessThanOrEqual => "<=",
            Self::Equal => "==",
            Self::NotEqual => "!=",
        }
    }
}

/// CrossFieldRule template
pub struct CrossFieldRuleTemplate;

impl RuleTemplate for CrossFieldRuleTemplate {
    fn template_code(&self) -> &str {
        "CROSS_FIELD"
    }

    fn template_name(&self) -> &str {
        "Cross-Field Comparison Rule"
    }

    fn parameter_schema(&self) -> Vec<ParameterSchema> {
        vec![
            ParameterSchema {
                name: "field1".to_string(),
                param_type: "string".to_string(),
                required: true,
                description: "First field to compare".to_string(),
                default: None,
                min: None,
                max: None,
                pattern: None,
                enum_values: Some(vec![
                    "total_claim_charge_amount".to_string(),
                    "line_item_charge_amount".to_string(),
                    "service_unit_count".to_string(),
                    "date_of_service".to_string(),
                    "date_of_service_from".to_string(),
                    "date_of_service_to".to_string(),
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
                name: "field2".to_string(),
                param_type: "string".to_string(),
                required: true,
                description: "Second field to compare".to_string(),
                default: None,
                min: None,
                max: None,
                pattern: None,
                enum_values: Some(vec![
                    "total_claim_charge_amount".to_string(),
                    "line_item_charge_amount".to_string(),
                    "service_unit_count".to_string(),
                    "date_of_service".to_string(),
                    "date_of_service_from".to_string(),
                    "date_of_service_to".to_string(),
                ]),
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

        let field1 = get_string_param(&params, "field1")?;
        let operator_str = get_string_param(&params, "operator")?;
        let operator = CrossFieldOperator::from_string(&operator_str)?;
        let field2 = get_string_param(&params, "field2")?;

        Ok(Arc::new(CrossFieldRule {
            rule_code,
            rule_name,
            flag_issue_type,
            issue_code,
            field1,
            operator,
            field2,
        }))
    }
}

/// Concrete CrossFieldRule instance
#[derive(Debug, Clone)]
pub struct CrossFieldRule {
    pub rule_code: String,
    pub rule_name: String,
    pub flag_issue_type: FlagIssueType,
    pub issue_code: String,
    pub field1: String,
    pub operator: CrossFieldOperator,
    pub field2: String,
}

#[async_trait]
impl Rule for CrossFieldRule {
    fn flag_type(&self) -> FlagIssueType {
        self.flag_issue_type
    }

    /// CROSS_FIELD rules don't need database access - they're pure CPU evaluation
    fn requires_db_access(&self) -> bool {
        false
    }

    /// Synchronous execution - avoids async overhead for CPU-only rules
    fn execute_sync(&self, ctx: &RuleExecutionContext) -> Result<Option<RuleResult>> {
        self.evaluate(ctx)
    }

    async fn execute(&self, ctx: &RuleExecutionContext, _pool: &PgPool) -> Result<Option<RuleResult>> {
        self.evaluate(ctx)
    }
}

impl CrossFieldRule {
    /// Core evaluation logic - shared between sync and async paths
    fn evaluate(&self, ctx: &RuleExecutionContext) -> Result<Option<RuleResult>> {
        // Try to compare as decimals first
        if let (Some(val1), Some(val2)) = (
            self.get_decimal_field(ctx, &self.field1),
            self.get_decimal_field(ctx, &self.field2),
        ) {
            if self.operator.compare_decimal(val1, val2) {
                return Ok(Some(
                    RuleResult::new(self.flag_issue_type, ctx.to_flag_context())
                        .with_details(format!(
                            "Field comparison failed: {} ({}) {} {} ({})",
                            self.field1,
                            val1,
                            self.operator.as_str(),
                            self.field2,
                            val2
                        ))
                        .with_issue_code(self.issue_code.clone())
                ));
            }
        }
        // Try to compare as dates
        else if let (Some(date1), Some(date2)) = (
            self.get_date_field(ctx, &self.field1),
            self.get_date_field(ctx, &self.field2),
        ) {
            if self.operator.compare_date(date1, date2) {
                return Ok(Some(
                    RuleResult::new(self.flag_issue_type, ctx.to_flag_context())
                        .with_details(format!(
                            "Date comparison failed: {} ({}) {} {} ({})",
                            self.field1,
                            date1,
                            self.operator.as_str(),
                            self.field2,
                            date2
                        ))
                        .with_issue_code(self.issue_code.clone())
                ));
            }
        }
        // If neither worked, fields are either missing or incompatible types
        else {
            return Ok(None); // Skip if fields not available
        }

        Ok(None)
    }

    fn get_decimal_field(&self, ctx: &RuleExecutionContext, field: &str) -> Option<Decimal> {
        match field {
            "total_claim_charge_amount" => ctx.total_claim_charge_amount,
            "line_item_charge_amount" => ctx.line_item_charge_amount,
            "service_unit_count" => ctx.service_unit_count,
            _ => None,
        }
    }

    fn get_date_field(&self, ctx: &RuleExecutionContext, field: &str) -> Option<NaiveDate> {
        match field {
            "date_of_service" => ctx.date_of_service,
            "date_of_service_from" => ctx.date_of_service_from,
            "date_of_service_to" => ctx.date_of_service_to,
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_operator_parsing() {
        assert_eq!(
            CrossFieldOperator::from_string(">").unwrap(),
            CrossFieldOperator::GreaterThan
        );
        assert_eq!(
            CrossFieldOperator::from_string("lt").unwrap(),
            CrossFieldOperator::LessThan
        );
        assert!(CrossFieldOperator::from_string("invalid").is_err());
    }

    #[test]
    fn test_decimal_comparison() {
        let val1 = Decimal::from(100);
        let val2 = Decimal::from(50);

        assert!(CrossFieldOperator::GreaterThan.compare_decimal(val1, val2));
        assert!(!CrossFieldOperator::LessThan.compare_decimal(val1, val2));
        assert!(CrossFieldOperator::NotEqual.compare_decimal(val1, val2));
    }

    #[test]
    fn test_template_schema() {
        let template = CrossFieldRuleTemplate;
        let schema = template.parameter_schema();

        assert_eq!(schema.len(), 3);
        assert_eq!(schema[0].name, "field1");
        assert_eq!(schema[1].name, "operator");
        assert_eq!(schema[2].name, "field2");
        assert!(schema[0].required);
        assert!(schema[1].required);
        assert!(schema[2].required);
    }

    #[test]
    fn test_instantiate() {
        let template = CrossFieldRuleTemplate;
        let params = serde_json::json!({
            "field1": "total_claim_charge_amount",
            "operator": ">",
            "field2": "line_item_charge_amount"
        });

        let result = template.instantiate(
            "TEST_RULE".to_string(),
            "Test Rule".to_string(),
            FlagIssueType::QtyUnitsInconsistent,
            params,
        );

        assert!(result.is_ok());
    }
}
