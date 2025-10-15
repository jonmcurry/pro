//! Rule Execution Planner
//!
//! PHASE 6: Intelligent rule ordering for early termination and optimal performance

use crate::flag_types::FlagIssueType;
use crate::rule_engine::Rule;
use rustc_hash::FxHashMap;
use std::sync::Arc;

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

    /// Load statistics from historical data (placeholder for database integration)
    pub async fn load_historical_stats(&mut self, _pool: &sqlx::PgPool) -> pro_common::Result<()> {
        // TODO: Load stats from database
        // SELECT flag_type,
        //        AVG(CASE WHEN flag_created THEN 1.0 ELSE 0.0 END) as trigger_rate,
        //        AVG(financial_impact) as avg_impact,
        //        AVG(execution_time_ms) as avg_time
        // FROM rule_execution_log
        // GROUP BY flag_type

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
