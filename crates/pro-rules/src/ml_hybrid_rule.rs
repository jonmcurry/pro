//! ML-Hybrid Rules
//!
//! PHASE 6: Rules that use ML predictions to decide whether to execute expensive logic

use crate::flag_types::FlagIssueType;
use crate::rule_engine::{Rule, RuleExecutionContext, RuleResult};
use async_trait::async_trait;
use pro_common::Result;
use sqlx::PgPool;

/// Trait for rules that can be filtered by ML predictions
///
/// ML-hybrid rules check a denial prediction score before executing expensive logic.
/// This allows skipping rules that are unlikely to trigger, improving performance.
#[async_trait]
pub trait MLHybridRule: Rule {
    /// Minimum ML prediction score to execute this rule (0.0 - 1.0)
    ///
    /// If the ML model predicts a denial risk below this threshold,
    /// the rule will be skipped entirely.
    fn ml_threshold(&self) -> f64 {
        0.3 // Default: skip rule if prediction score < 30%
    }

    /// Execute rule with ML pre-filtering
    ///
    /// This checks if an ML prediction exists for the claim and compares
    /// the prediction score against the threshold before executing.
    async fn execute_with_ml_filter(
        &self,
        ctx: &RuleExecutionContext,
        pool: &PgPool,
    ) -> Result<Option<RuleResult>> {
        // Check if we have an ML prediction for this encounter
        if let Some(encounter_id) = ctx.encounter_id {
            if let Some(ml_score) = self.get_ml_prediction_score(encounter_id, pool).await? {
                // Skip rule if ML predicts low likelihood
                if ml_score < self.ml_threshold() {
                    return Ok(None);
                }
            }
        }

        // Execute rule normally if:
        // 1. No ML prediction available
        // 2. ML score is above threshold
        self.execute(ctx, pool).await
    }

    /// Get ML prediction score from database
    async fn get_ml_prediction_score(&self, encounter_id: i64, pool: &PgPool) -> Result<Option<f64>> {
        let query = r#"
            SELECT prediction_score
            FROM ml.model_prediction
            WHERE encounter_id = $1
            AND prediction_type = 'DENIAL_RISK'
            ORDER BY predicted_at DESC
            LIMIT 1
        "#;

        let score = sqlx::query_scalar::<_, f64>(query)
            .bind(encounter_id)
            .fetch_optional(pool)
            .await?;

        Ok(score)
    }
}

/// Example: Expensive compliance rule that benefits from ML filtering
pub struct ExpensiveComplianceRule;

#[async_trait]
impl Rule for ExpensiveComplianceRule {
    fn flag_type(&self) -> FlagIssueType {
        FlagIssueType::CodUpcoding // Placeholder - use existing flag type
    }

    async fn execute(&self, _ctx: &RuleExecutionContext, _pool: &PgPool) -> Result<Option<RuleResult>> {
        // Expensive compliance logic would go here
        // This is just a placeholder
        Ok(None)
    }
}

impl MLHybridRule for ExpensiveComplianceRule {
    fn ml_threshold(&self) -> f64 {
        0.4 // Only execute if ML predicts >= 40% denial risk
    }
}
