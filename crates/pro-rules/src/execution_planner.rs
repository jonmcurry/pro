//! Rule Execution Planner
//!
//! PHASE 6: Intelligent rule ordering for early termination and optimal performance

use crate::flag_types::FlagIssueType;
use crate::rule_engine::Rule;
use rustc_hash::FxHashMap;
use std::sync::Arc;

/// Helper to parse flag type from string (PHASE 8)
/// This matches the flag_type strings stored in the database
fn parse_flag_type(type_str: &str) -> Result<FlagIssueType, ()> {
    // Try to parse as the enum variant name (e.g., "CodUpcoding")
    // This is a simplified parser - in production, use serde or strum
    match type_str {
        // Coding (4 types)
        "CodIncorrectProcedureCode" => Ok(FlagIssueType::CodIncorrectProcedureCode),
        "CodProcedureNotSupportedByDiagnosis" => Ok(FlagIssueType::CodProcedureNotSupportedByDiagnosis),
        "CodUnbundling" => Ok(FlagIssueType::CodUnbundling),
        "CodUpcoding" => Ok(FlagIssueType::CodUpcoding),

        // Documentation (2 types)
        "DocInsufficientDocumentation" => Ok(FlagIssueType::DocInsufficientDocumentation),
        "DocMissingRequiredElements" => Ok(FlagIssueType::DocMissingRequiredElements),

        // E/M Overcoded (2 types)
        "EMOLevelHigherThanMDM" => Ok(FlagIssueType::EMOLevelHigherThanMDM),
        "EMOLevelHigherThanHistoryExam" => Ok(FlagIssueType::EMOLevelHigherThanHistoryExam),

        // E/M Undercoded (2 types)
        "EMULevelLowerThanMDM" => Ok(FlagIssueType::EMULevelLowerThanMDM),
        "EMULevelLowerThanTime" => Ok(FlagIssueType::EMULevelLowerThanTime),

        // E/M Incorrect Category (1 type)
        "EMIWrongCategory" => Ok(FlagIssueType::EMIWrongCategory),

        // E/M Time Not Documented (1 type)
        "EMTTimeNotDocumented" => Ok(FlagIssueType::EMTTimeNotDocumented),

        // Modifier (3 types)
        "ModMissingRequired" => Ok(FlagIssueType::ModMissingRequired),
        "ModIncorrect" => Ok(FlagIssueType::ModIncorrect),
        "ModConflicting" => Ok(FlagIssueType::ModConflicting),

        // Other (3 types)
        "OthMedicalNecessityNotEstablished" => Ok(FlagIssueType::OthMedicalNecessityNotEstablished),
        "OthWrongProviderType" => Ok(FlagIssueType::OthWrongProviderType),
        "OthDuplicateService" => Ok(FlagIssueType::OthDuplicateService),

        // Quantity (2 types)
        "QtyUnitsExceedMaximum" => Ok(FlagIssueType::QtyUnitsExceedMaximum),
        "QtyUnitsInconsistent" => Ok(FlagIssueType::QtyUnitsInconsistent),

        // Supervision (3 types)
        "SupSupervisionNotDocumented" => Ok(FlagIssueType::SupSupervisionNotDocumented),
        "SupInappropriateLevel" => Ok(FlagIssueType::SupInappropriateLevel),
        "SupTeachingPhysicianNotMet" => Ok(FlagIssueType::SupTeachingPhysicianNotMet),

        // Diagnosis (4 types)
        "DxPrimaryDoesNotSupport" => Ok(FlagIssueType::DxPrimaryDoesNotSupport),
        "DxMissingSpecificity" => Ok(FlagIssueType::DxMissingSpecificity),
        "DxSequencingError" => Ok(FlagIssueType::DxSequencingError),
        "DxUnspecifiedWhenSpecificAvailable" => Ok(FlagIssueType::DxUnspecifiedWhenSpecificAvailable),

        _ => Err(()),
    }
}

/// Statistics for a rule's execution
#[derive(Debug, Clone)]
pub struct RuleStats {
    /// Percentage of times rule creates a flag (0.0 - 1.0)
    pub trigger_rate: f64,
    /// Average financial impact when triggered
    pub avg_financial_impact: f64,
    /// Average execution time in milliseconds
    pub avg_execution_time_ms: f64,
    /// Total executions
    pub execution_count: u64,
}

impl Default for RuleStats {
    fn default() -> Self {
        Self {
            trigger_rate: 0.5,            // Assume 50% trigger rate initially
            avg_financial_impact: 100.0,   // $100 default impact
            avg_execution_time_ms: 10.0,   // 10ms default
            execution_count: 0,
        }
    }
}

/// Rule execution planner that optimizes rule ordering
pub struct RuleExecutionPlanner {
    rules: Vec<Arc<dyn Rule>>,
    stats: FxHashMap<FlagIssueType, RuleStats>,
}

impl RuleExecutionPlanner {
    pub fn new(rules: Vec<Arc<dyn Rule>>) -> Self {
        let mut stats = FxHashMap::default();

        // Initialize stats for all rules
        for rule in &rules {
            stats.insert(rule.flag_type(), RuleStats::default());
        }

        Self { rules, stats }
    }

    /// Update statistics for a rule
    pub fn update_stats(
        &mut self,
        flag_type: FlagIssueType,
        triggered: bool,
        financial_impact: Option<f64>,
        execution_time_ms: f64,
    ) {
        if let Some(stats) = self.stats.get_mut(&flag_type) {
            let count = stats.execution_count;

            // Update trigger rate (exponential moving average)
            let alpha = 0.1; // Weight for new observation
            if triggered {
                stats.trigger_rate = stats.trigger_rate * (1.0 - alpha) + alpha;
            } else {
                stats.trigger_rate = stats.trigger_rate * (1.0 - alpha);
            }

            // Update average financial impact
            if let Some(impact) = financial_impact {
                stats.avg_financial_impact =
                    (stats.avg_financial_impact * count as f64 + impact) / (count + 1) as f64;
            }

            // Update average execution time
            stats.avg_execution_time_ms =
                (stats.avg_execution_time_ms * count as f64 + execution_time_ms) / (count + 1) as f64;

            stats.execution_count += 1;
        }
    }

    /// Generate optimal execution order
    ///
    /// Rules are ordered by their "value score" which balances:
    /// - Probability of triggering (higher is better)
    /// - Financial impact (higher is better)
    /// - Execution speed (faster is better)
    pub fn plan_execution(&self) -> Vec<Arc<dyn Rule>> {
        let default_stats = RuleStats::default();

        let mut scored_rules: Vec<_> = self.rules.iter()
            .map(|rule| {
                let stats = self.stats.get(&rule.flag_type())
                    .unwrap_or(&default_stats);

                // Value score = (trigger_rate * avg_impact) / execution_time
                // This prioritizes rules that are likely to trigger,
                // have high impact, and execute quickly
                let score = (stats.trigger_rate * stats.avg_financial_impact)
                    / stats.avg_execution_time_ms.max(0.1);

                (Arc::clone(rule), score)
            })
            .collect();

        // Sort by score (highest first)
        scored_rules.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scored_rules.into_iter().map(|(rule, _)| rule).collect()
    }

    /// Get statistics for a rule
    pub fn get_stats(&self, flag_type: FlagIssueType) -> Option<&RuleStats> {
        self.stats.get(&flag_type)
    }

    /// Get all statistics
    pub fn get_all_stats(&self) -> &FxHashMap<FlagIssueType, RuleStats> {
        &self.stats
    }

    /// Load statistics from historical data (PHASE 8)
    /// Reads from materialized view for fast initialization
    pub async fn load_historical_stats(&mut self, pool: &sqlx::PgPool) -> pro_common::Result<()> {
        use sqlx::Row;

        let rows = sqlx::query(
            r#"
            SELECT
                flag_type,
                trigger_rate,
                avg_financial_impact,
                avg_execution_time_ms,
                execution_count
            FROM claims.rule_execution_stats_summary
            ORDER BY trigger_rate * COALESCE(avg_financial_impact, 0) DESC
            "#
        )
        .fetch_all(pool)
        .await?;

        let mut loaded_count = 0;

        // Update stats for each flag type found in historical data
        for row in rows {
            let flag_type_str: String = row.get("flag_type");

            // Parse flag type string to enum
            if let Ok(flag_type) = parse_flag_type(&flag_type_str) {
                // Only update if we have stats for this flag type
                if let Some(stats) = self.stats.get_mut(&flag_type) {
                    stats.trigger_rate = row.get("trigger_rate");
                    stats.avg_financial_impact = row.get::<Option<rust_decimal::Decimal>, _>("avg_financial_impact")
                        .map(|d| d.to_string().parse::<f64>().unwrap_or(100.0))
                        .unwrap_or(100.0);
                    stats.avg_execution_time_ms = row.get::<f32, _>("avg_execution_time_ms") as f64;
                    stats.execution_count = row.get::<i64, _>("execution_count") as u64;

                    loaded_count += 1;

                    tracing::debug!(
                        "Loaded historical stats for {:?}: trigger_rate={:.3}, avg_impact=${:.2}, avg_time={:.2}ms, count={}",
                        flag_type,
                        stats.trigger_rate,
                        stats.avg_financial_impact,
                        stats.avg_execution_time_ms,
                        stats.execution_count
                    );
                }
            }
        }

        if loaded_count > 0 {
            tracing::info!("Loaded historical statistics for {} rule types", loaded_count);
        } else {
            tracing::warn!("No historical statistics found - using default values");
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stats_update() {
        let mut planner = RuleExecutionPlanner::new(Vec::new());
        let flag_type = FlagIssueType::CodUpcoding;

        // Initialize
        planner.stats.insert(flag_type, RuleStats::default());

        // Update with a trigger
        planner.update_stats(flag_type, true, Some(500.0), 15.0);

        let stats = planner.get_stats(flag_type).unwrap();
        assert!(stats.trigger_rate > 0.5); // Should increase from 0.5
        assert_eq!(stats.execution_count, 1);
    }

    #[test]
    fn test_execution_planning() {
        let planner = RuleExecutionPlanner::new(Vec::new());
        let ordered = planner.plan_execution();
        assert_eq!(ordered.len(), 0);
    }
}
