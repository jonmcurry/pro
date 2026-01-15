//! Composite Rules
//!
//! PHASE 6: Rules that combine multiple sub-rules with logical operators

use crate::flag_types::{FlagIssueType, FlagSeverity};
use crate::rule_engine::{Rule, RuleExecutionContext, RuleResult};
use async_trait::async_trait;
use pro_common::Result;
use sqlx::PgPool;
use std::sync::Arc;

/// Composite rule that combines multiple sub-rules
pub struct CompositeRule {
    name: String,
    flag_type: FlagIssueType,
    condition: RuleCondition,
    subrules: Vec<Arc<dyn Rule>>,
}

/// Logical condition for combining sub-rule results
#[derive(Debug, Clone, Copy)]
pub enum RuleCondition {
    /// All sub-rules must trigger (AND)
    All,
    /// Any sub-rule triggers (OR)
    Any,
    /// Majority of sub-rules must trigger (>50%)
    Majority,
    /// At least N sub-rules must trigger
    AtLeast(usize),
}

impl CompositeRule {
    pub fn new(name: impl Into<String>, flag_type: FlagIssueType, condition: RuleCondition) -> Self {
        Self {
            name: name.into(),
            flag_type,
            condition,
            subrules: Vec::new(),
        }
    }

    /// Add a sub-rule to this composite
    pub fn add_subrule(mut self, rule: Arc<dyn Rule>) -> Self {
        self.subrules.push(rule);
        self
    }

    /// Check if condition is satisfied
    fn check_condition(&self, triggered_count: usize) -> bool {
        match self.condition {
            RuleCondition::All => triggered_count == self.subrules.len(),
            RuleCondition::Any => triggered_count > 0,
            RuleCondition::Majority => triggered_count > self.subrules.len() / 2,
            RuleCondition::AtLeast(n) => triggered_count >= n,
        }
    }

    /// Create a composite result from sub-rule results
    fn create_composite_result(&self, subrule_results: Vec<RuleResult>, ctx: &RuleExecutionContext) -> RuleResult {
        // Combine descriptions
        let descriptions: Vec<String> = subrule_results.iter()
            .map(|r| r.description.clone())
            .collect();

        let combined_description = format!("{}: {}", self.name, descriptions.join("; "));

        // Use highest severity from sub-rules
        let max_severity = subrule_results.iter()
            .map(|r| r.severity)
            .max()
            .unwrap_or(FlagSeverity::Medium);

        // Sum financial impacts
        let total_impact = subrule_results.iter()
            .filter_map(|r| r.financial_impact)
            .sum();

        RuleResult {
            flag_type: self.flag_type,
            severity: max_severity,
            description: combined_description,
            details: Some(format!("{} of {} sub-rules triggered", subrule_results.len(), self.subrules.len())),
            financial_impact: Some(total_impact),
            context: ctx.to_flag_context(),
            // Legacy composite rule doesn't have database issue_code
            issue_code: None,
        }
    }
}

#[async_trait]
impl Rule for CompositeRule {
    fn flag_type(&self) -> FlagIssueType {
        self.flag_type
    }

    fn name(&self) -> &str {
        &self.name
    }

    async fn execute(&self, ctx: &RuleExecutionContext, pool: &PgPool) -> Result<Option<RuleResult>> {
        let mut subrule_results = Vec::new();

        // Execute all sub-rules
        for subrule in &self.subrules {
            if let Some(result) = subrule.execute(ctx, pool).await? {
                subrule_results.push(result);
            }
        }

        // Check if condition is satisfied
        if self.check_condition(subrule_results.len()) {
            Ok(Some(self.create_composite_result(subrule_results, ctx)))
        } else {
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_condition_all() {
        let rule = CompositeRule::new("test", FlagIssueType::CodUpcoding, RuleCondition::All);
        assert!(!rule.check_condition(0));
        assert!(rule.check_condition(0)); // No subrules added, so 0 == len
    }

    #[test]
    fn test_condition_any() {
        let rule = CompositeRule::new("test", FlagIssueType::CodUpcoding, RuleCondition::Any);
        assert!(!rule.check_condition(0));
        assert!(rule.check_condition(1));
        assert!(rule.check_condition(5));
    }

    #[test]
    fn test_condition_majority() {
        let rule = CompositeRule::new("test", FlagIssueType::CodUpcoding, RuleCondition::Majority);
        // Assuming 5 subrules, majority is > 2
        assert!(!rule.check_condition(2));
        assert!(rule.check_condition(3));
    }

    #[test]
    fn test_condition_at_least() {
        let rule = CompositeRule::new("test", FlagIssueType::CodUpcoding, RuleCondition::AtLeast(3));
        assert!(!rule.check_condition(2));
        assert!(rule.check_condition(3));
        assert!(rule.check_condition(4));
    }
}
