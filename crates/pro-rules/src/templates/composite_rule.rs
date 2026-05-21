// CompositeRule Template - Compound conditions with AND/OR logic
//
// Allows combining multiple field checks in a single database-configured rule.
// No recompilation needed for new compound rules.
//
// Example use cases:
// - AHRQ quality indicators (CPT + Diagnosis + Date conditions)
// - Complex billing rules (multiple criteria must be met)
// - Exclusion rules (flag if condition A but NOT condition B)

use crate::flag_types::{FlagIssueType, FlagSeverity};
use crate::rule_engine::{Rule, RuleExecutionContext, RuleResult};
use crate::template::{ParameterSchema, RuleTemplate};
use async_trait::async_trait;
use chrono::NaiveDate;
use pro_common::{Error, Result};
use regex::Regex;
use rustc_hash::FxHashSet;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use sqlx::PgPool;
use std::sync::Arc;

/// CompositeRule template for compound conditions
pub struct CompositeRuleTemplate;

impl RuleTemplate for CompositeRuleTemplate {
    fn template_code(&self) -> &str {
        "COMPOSITE"
    }

    fn template_name(&self) -> &str {
        "Composite Rule (AND/OR Conditions)"
    }

    fn parameter_schema(&self) -> Vec<ParameterSchema> {
        vec![
            ParameterSchema {
                name: "operator".to_string(),
                param_type: "string".to_string(),
                required: true,
                description: "Logical operator: AND (all must match), OR (any must match)".to_string(),
                default: Some(serde_json::json!("AND")),
                min: None,
                max: None,
                pattern: None,
                enum_values: Some(vec!["AND".to_string(), "OR".to_string()]),
            },
            ParameterSchema {
                name: "conditions".to_string(),
                param_type: "array".to_string(),
                required: true,
                description: "Array of condition objects".to_string(),
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
        // Parse operator
        let operator = params
            .get("operator")
            .and_then(|v| v.as_str())
            .unwrap_or("AND");

        let logic_operator = match operator.to_uppercase().as_str() {
            "AND" => LogicOperator::And,
            "OR" => LogicOperator::Or,
            _ => return Err(Error::Config(format!("Invalid operator: {}", operator))),
        };

        // Parse conditions array
        let conditions_value = params
            .get("conditions")
            .ok_or_else(|| Error::Config("Missing 'conditions' parameter".to_string()))?;

        let conditions: Vec<Condition> = serde_json::from_value(conditions_value.clone())
            .map_err(|e| Error::Config(format!("Invalid conditions format: {}", e)))?;

        if conditions.is_empty() {
            return Err(Error::Config("Conditions array cannot be empty".to_string()));
        }

        // Extract CPT codes from the first CptIn condition for indexing
        // This allows the rule engine to skip this rule for non-matching CPT codes
        let applicable_cpts: Option<Vec<String>> = conditions.iter()
            .find_map(|c| {
                if let Condition::CptIn { codes } = c {
                    // Uppercase for consistent indexing
                    Some(codes.iter().map(|s| s.to_uppercase()).collect())
                } else {
                    None
                }
            });

        // Compile regex patterns
        let mut compiled_conditions: Vec<CompiledCondition> = conditions
            .into_iter()
            .map(|c| c.compile())
            .collect::<Result<Vec<_>>>()?;

        // PERFORMANCE OPTIMIZATION: Reorder conditions for optimal short-circuit evaluation
        // For AND logic: Put cheap checks first so we fail fast on non-matches
        // For OR logic: Put cheap checks first so we succeed fast on matches
        //
        // Cost ranking (low to high):
        // 1. DateGte, DateLte (single comparison)
        // 2. CptIn, PosIn (O(1) HashSet lookup)
        // 3. ModifierIn, ModifierNotIn (O(N) where N = modifiers, usually 0-4)
        // 4. CptPattern, PosPattern (regex match on single string)
        // 5. DxIn (O(N) where N = diagnosis codes, usually 1-12, then O(1) HashSet)
        // 6. DxPattern, DxPatternExclude (O(N) regex matches)
        //
        // Also: If rule has applicable_cpts (was selected via CPT index), we can skip CptIn
        // since it was already verified during rule selection. This saves 1 condition eval per rule.
        compiled_conditions.sort_by_key(|c| c.evaluation_cost());

        // If rule has applicable_cpts (selected via CPT index), find the CptIn condition
        // so we can skip it during evaluation (the index already confirmed the match)
        let cpt_verified_idx = if applicable_cpts.is_some() {
            compiled_conditions.iter().position(|c| matches!(c, CompiledCondition::CptIn { .. }))
        } else {
            None
        };

        Ok(Arc::new(CompositeRule {
            rule_code,
            rule_name,
            flag_issue_type,
            issue_code,
            operator: logic_operator,
            conditions: compiled_conditions,
            applicable_cpts,
            cpt_verified_idx,
        }))
    }
}

/// Logical operator for combining conditions
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LogicOperator {
    And, // All conditions must match
    Or,  // Any condition must match
}

/// Condition type for different kinds of checks
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Condition {
    /// Check if CPT/procedure code is in a list
    #[serde(rename = "cpt_in")]
    CptIn { codes: Vec<String> },

    /// Check if CPT/procedure code matches pattern
    #[serde(rename = "cpt_pattern")]
    CptPattern { pattern: String },

    /// Check if any diagnosis code is in a list
    #[serde(rename = "dx_in")]
    DxIn { codes: Vec<String> },

    /// Check if any diagnosis code matches pattern
    #[serde(rename = "dx_pattern")]
    DxPattern { pattern: String },

    /// Check if any diagnosis code matches pattern but exclude others
    #[serde(rename = "dx_pattern_exclude")]
    DxPatternExclude { include: String, exclude: String },

    /// Check if service date is >= min_date
    #[serde(rename = "date_gte")]
    DateGte { min_date: String },

    /// Check if service date is <= max_date
    #[serde(rename = "date_lte")]
    DateLte { max_date: String },

    /// Check if place of service code is in a list
    #[serde(rename = "pos_in")]
    PosIn { codes: Vec<String> },

    /// Check if place of service code matches pattern
    #[serde(rename = "pos_pattern")]
    PosPattern { pattern: String },

    /// Check if modifier is present
    #[serde(rename = "modifier_in")]
    ModifierIn { modifiers: Vec<String> },

    /// Check if modifier is NOT present
    #[serde(rename = "modifier_not_in")]
    ModifierNotIn { modifiers: Vec<String> },
}

/// Compiled condition with pre-compiled regex patterns and HashSets for O(1) lookup
/// PERFORMANCE: Using FxHashSet instead of Vec for code lookups gives O(1) vs O(N) performance
/// All codes are normalized to UPPERCASE at compile time for fast case-insensitive matching
#[derive(Debug, Clone)]
pub enum CompiledCondition {
    /// CPT codes stored in HashSet, all UPPERCASE for O(1) case-insensitive lookup
    CptIn { codes: FxHashSet<String> },
    CptPattern { regex: Regex },
    /// Diagnosis codes stored in HashSet, all UPPERCASE for O(1) case-insensitive lookup
    DxIn { codes: FxHashSet<String> },
    DxPattern { regex: Regex },
    DxPatternExclude { include_regex: Regex, exclude_regex: Regex },
    DateGte { min_date: NaiveDate },
    DateLte { max_date: NaiveDate },
    /// POS codes stored in HashSet, all UPPERCASE for O(1) case-insensitive lookup
    PosIn { codes: FxHashSet<String> },
    PosPattern { regex: Regex },
    /// Modifiers stored in HashSet, all UPPERCASE for O(1) case-insensitive lookup
    ModifierIn { modifiers: FxHashSet<String> },
    ModifierNotIn { modifiers: FxHashSet<String> },
}

impl Condition {
    /// Compile condition into optimized form with HashSets and pre-compiled regexes
    /// PERFORMANCE: All codes are normalized to UPPERCASE for fast case-insensitive matching
    fn compile(self) -> Result<CompiledCondition> {
        match self {
            Condition::CptIn { codes } => {
                // Convert to HashSet with uppercase for O(1) lookup
                let codes_set: FxHashSet<String> = codes.into_iter()
                    .map(|c| c.to_uppercase())
                    .collect();
                Ok(CompiledCondition::CptIn { codes: codes_set })
            }
            Condition::CptPattern { pattern } => {
                let regex = Regex::new(&pattern)
                    .map_err(|e| Error::Config(format!("Invalid CPT pattern: {}", e)))?;
                Ok(CompiledCondition::CptPattern { regex })
            }
            Condition::DxIn { codes } => {
                // Convert to HashSet with uppercase for O(1) lookup
                let codes_set: FxHashSet<String> = codes.into_iter()
                    .map(|c| c.to_uppercase())
                    .collect();
                Ok(CompiledCondition::DxIn { codes: codes_set })
            }
            Condition::DxPattern { pattern } => {
                let regex = Regex::new(&pattern)
                    .map_err(|e| Error::Config(format!("Invalid DX pattern: {}", e)))?;
                Ok(CompiledCondition::DxPattern { regex })
            }
            Condition::DxPatternExclude { include, exclude } => {
                let include_regex = Regex::new(&include)
                    .map_err(|e| Error::Config(format!("Invalid DX include pattern: {}", e)))?;
                let exclude_regex = Regex::new(&exclude)
                    .map_err(|e| Error::Config(format!("Invalid DX exclude pattern: {}", e)))?;
                Ok(CompiledCondition::DxPatternExclude { include_regex, exclude_regex })
            }
            Condition::DateGte { min_date } => {
                let date = NaiveDate::parse_from_str(&min_date, "%Y-%m-%d")
                    .map_err(|e| Error::Config(format!("Invalid min_date '{}': {}", min_date, e)))?;
                Ok(CompiledCondition::DateGte { min_date: date })
            }
            Condition::DateLte { max_date } => {
                let date = NaiveDate::parse_from_str(&max_date, "%Y-%m-%d")
                    .map_err(|e| Error::Config(format!("Invalid max_date '{}': {}", max_date, e)))?;
                Ok(CompiledCondition::DateLte { max_date: date })
            }
            Condition::PosIn { codes } => {
                // Convert to HashSet with uppercase for O(1) lookup
                let codes_set: FxHashSet<String> = codes.into_iter()
                    .map(|c| c.to_uppercase())
                    .collect();
                Ok(CompiledCondition::PosIn { codes: codes_set })
            }
            Condition::PosPattern { pattern } => {
                let regex = Regex::new(&pattern)
                    .map_err(|e| Error::Config(format!("Invalid POS pattern: {}", e)))?;
                Ok(CompiledCondition::PosPattern { regex })
            }
            Condition::ModifierIn { modifiers } => {
                // Convert to HashSet with uppercase for O(1) lookup
                let mods_set: FxHashSet<String> = modifiers.into_iter()
                    .map(|m| m.to_uppercase())
                    .collect();
                Ok(CompiledCondition::ModifierIn { modifiers: mods_set })
            }
            Condition::ModifierNotIn { modifiers } => {
                // Convert to HashSet with uppercase for O(1) lookup
                let mods_set: FxHashSet<String> = modifiers.into_iter()
                    .map(|m| m.to_uppercase())
                    .collect();
                Ok(CompiledCondition::ModifierNotIn { modifiers: mods_set })
            }
        }
    }
}

impl CompiledCondition {
    /// Evaluate condition against execution context
    /// PERFORMANCE: Uses pre-computed uppercase values from ctx (no allocations in hot loop)
    /// All code lookups use O(1) HashSet.contains()
    /// DxPattern/DxPatternExclude use mutable cache to deduplicate regex across rules
    #[inline]
    fn evaluate(&self, ctx: &mut RuleExecutionContext) -> bool {
        match self {
            CompiledCondition::CptIn { codes } => {
                ctx.procedure_code_upper
                    .as_ref()
                    .map(|cpt| codes.contains(cpt))
                    .unwrap_or(false)
            }
            CompiledCondition::CptPattern { regex } => {
                ctx.procedure_code
                    .as_ref()
                    .map(|cpt| regex.is_match(cpt))
                    .unwrap_or(false)
            }
            CompiledCondition::DxIn { codes } => {
                ctx.diagnosis_codes_upper
                    .iter()
                    .any(|dx| codes.contains(dx))
            }
            CompiledCondition::DxPattern { regex } => {
                let pattern_str = regex.as_str();
                if let Some(cache) = &ctx.dx_pattern_cache {
                    if let Some(&cached) = cache.get(pattern_str) {
                        return cached;
                    }
                }
                let result = ctx.diagnosis_codes.iter().any(|dx| regex.is_match(dx));
                if let Some(cache) = &mut ctx.dx_pattern_cache {
                    cache.insert(pattern_str.to_string(), result);
                }
                result
            }
            CompiledCondition::DxPatternExclude { include_regex, exclude_regex } => {
                let key = (include_regex.as_str().to_string(), exclude_regex.as_str().to_string());
                if let Some(cache) = &ctx.dx_pattern_exclude_cache {
                    if let Some(&cached) = cache.get(&key) {
                        return cached;
                    }
                }
                let has_include = ctx.diagnosis_codes.iter().any(|dx| include_regex.is_match(dx));
                let has_exclude = ctx.diagnosis_codes.iter().any(|dx| exclude_regex.is_match(dx));
                let result = has_include && !has_exclude;
                if let Some(cache) = &mut ctx.dx_pattern_exclude_cache {
                    cache.insert(key, result);
                }
                result
            }
            CompiledCondition::DateGte { min_date } => {
                ctx.date_of_service
                    .map(|dos| dos >= *min_date)
                    .unwrap_or(false)
            }
            CompiledCondition::DateLte { max_date } => {
                ctx.date_of_service
                    .map(|dos| dos <= *max_date)
                    .unwrap_or(false)
            }
            CompiledCondition::PosIn { codes } => {
                ctx.place_of_service_upper
                    .as_ref()
                    .map(|pos| codes.contains(pos))
                    .unwrap_or(false)
            }
            CompiledCondition::PosPattern { regex } => {
                ctx.place_of_service_code
                    .as_ref()
                    .map(|pos| regex.is_match(pos))
                    .unwrap_or(false)
            }
            CompiledCondition::ModifierIn { modifiers } => {
                ctx.modifiers_upper
                    .iter()
                    .any(|m| modifiers.contains(m))
            }
            CompiledCondition::ModifierNotIn { modifiers } => {
                !ctx.modifiers_upper
                    .iter()
                    .any(|m| modifiers.contains(m))
            }
        }
    }

    /// PERFORMANCE: Evaluation cost for condition ordering optimization
    /// Lower cost = evaluated first in short-circuit AND/OR
    /// This allows cheap checks to fail/succeed fast before expensive checks
    #[inline]
    fn evaluation_cost(&self) -> u8 {
        match self {
            // Cost 1: Single comparison (cheapest)
            CompiledCondition::DateGte { .. } | CompiledCondition::DateLte { .. } => 1,

            // Cost 2: O(1) HashSet lookup
            CompiledCondition::CptIn { .. } | CompiledCondition::PosIn { .. } => 2,

            // Cost 3: O(N) where N = modifiers (usually 0-4)
            CompiledCondition::ModifierIn { .. } | CompiledCondition::ModifierNotIn { .. } => 3,

            // Cost 4: Single regex match
            CompiledCondition::CptPattern { .. } | CompiledCondition::PosPattern { .. } => 4,

            // Cost 5: O(N) where N = diagnosis codes (usually 1-12), then O(1) HashSet
            // This is expensive because diagnosis_codes can have 1-12 codes
            // and many rules have 50-200 DX codes in their condition
            CompiledCondition::DxIn { .. } => 5,

            // Cost 6: O(N) regex matches over diagnosis codes (most expensive)
            CompiledCondition::DxPattern { .. } | CompiledCondition::DxPatternExclude { .. } => 6,
        }
    }

    /// Get description of condition for logging/debugging
    fn description(&self) -> String {
        match self {
            CompiledCondition::CptIn { codes } => {
                let codes_vec: Vec<_> = codes.iter().collect();
                format!("CPT in [{}]", codes_vec.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", "))
            }
            CompiledCondition::CptPattern { regex } => format!("CPT matches /{}/", regex.as_str()),
            CompiledCondition::DxIn { codes } => {
                let codes_vec: Vec<_> = codes.iter().collect();
                format!("DX in [{}]", codes_vec.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", "))
            }
            CompiledCondition::DxPattern { regex } => format!("DX matches /{}/", regex.as_str()),
            CompiledCondition::DxPatternExclude { include_regex, exclude_regex } => {
                format!("DX matches /{}/ except /{}/", include_regex.as_str(), exclude_regex.as_str())
            }
            CompiledCondition::DateGte { min_date } => format!("Date >= {}", min_date),
            CompiledCondition::DateLte { max_date } => format!("Date <= {}", max_date),
            CompiledCondition::PosIn { codes } => {
                let codes_vec: Vec<_> = codes.iter().collect();
                format!("POS in [{}]", codes_vec.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", "))
            }
            CompiledCondition::PosPattern { regex } => format!("POS matches /{}/", regex.as_str()),
            CompiledCondition::ModifierIn { modifiers } => {
                let mods_vec: Vec<_> = modifiers.iter().collect();
                format!("Modifier in [{}]", mods_vec.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", "))
            }
            CompiledCondition::ModifierNotIn { modifiers } => {
                let mods_vec: Vec<_> = modifiers.iter().collect();
                format!("Modifier not in [{}]", mods_vec.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", "))
            }
        }
    }
}

/// Concrete CompositeRule instance
#[derive(Debug)]
pub struct CompositeRule {
    pub rule_code: String,
    pub rule_name: String,
    pub flag_issue_type: FlagIssueType,
    /// Database issue_code for flag_issue JOIN (e.g., "TEST_99213_SA")
    pub issue_code: String,
    pub operator: LogicOperator,
    pub conditions: Vec<CompiledCondition>,
    /// Cached CPT codes from the first CptIn condition (for index optimization)
    pub applicable_cpts: Option<Vec<String>>,
    /// Index of CptIn condition that can be skipped when rule is selected via CPT index
    /// (the index already verified the CPT match, so re-checking is redundant)
    pub cpt_verified_idx: Option<usize>,
}

impl CompositeRule {
    /// Execute the rule logic (shared between sync and async paths)
    /// PERFORMANCE: Uses short-circuit evaluation for AND/OR operators
    /// - AND: Stops on first false (avoids evaluating remaining conditions)
    /// - OR: Stops on first true (avoids evaluating remaining conditions)
    fn evaluate(&self, ctx: &mut RuleExecutionContext) -> Result<Option<RuleResult>> {
        let triggered = match self.operator {
            LogicOperator::And => {
                for (idx, condition) in self.conditions.iter().enumerate() {
                    if self.cpt_verified_idx == Some(idx) {
                        continue;
                    }
                    if !condition.evaluate(ctx) {
                        return Ok(None);
                    }
                }
                true
            }
            LogicOperator::Or => {
                let mut any_matched = false;
                for (idx, condition) in self.conditions.iter().enumerate() {
                    if self.cpt_verified_idx == Some(idx) {
                        any_matched = true;
                        break;
                    }
                    if condition.evaluate(ctx) {
                        any_matched = true;
                        break;
                    }
                }
                any_matched
            }
        };

        if triggered {
            // PERFORMANCE: Simplified description to minimize allocations
            // With 50+ rules triggering per claim, string allocations add up significantly
            // Just use rule name - the detailed condition info is rarely needed
            let description = self.rule_name.clone();

            Ok(Some(
                RuleResult::new(self.flag_issue_type, ctx.to_flag_context())
                    .with_severity(FlagSeverity::Medium)
                    .with_details(description)
                    .with_issue_code(self.issue_code.clone()),
            ))
        } else {
            Ok(None)
        }
    }
}

#[async_trait]
impl Rule for CompositeRule {
    fn flag_type(&self) -> FlagIssueType {
        self.flag_issue_type
    }

    fn name(&self) -> &str {
        &self.rule_name
    }

    /// COMPOSITE rules don't need database access - they're pure CPU evaluation
    fn requires_db_access(&self) -> bool {
        false
    }

    /// Return the CPT codes this rule applies to (for index optimization)
    /// This allows the rule engine to skip this rule for non-matching CPT codes
    fn applicable_cpt_codes(&self) -> Option<&[String]> {
        self.applicable_cpts.as_deref()
    }

    /// Synchronous execution - avoids async overhead for CPU-only rules
    fn execute_sync(&self, ctx: &mut RuleExecutionContext) -> Result<Option<RuleResult>> {
        self.evaluate(ctx)
    }

    async fn execute(&self, ctx: &mut RuleExecutionContext, _pool: &PgPool) -> Result<Option<RuleResult>> {
        self.evaluate(ctx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_template_code() {
        let template = CompositeRuleTemplate;
        assert_eq!(template.template_code(), "COMPOSITE");
    }

    #[test]
    fn test_parse_conditions() {
        let json = serde_json::json!([
            {"type": "cpt_in", "codes": ["99281", "99282", "99283"]},
            {"type": "dx_pattern", "pattern": "^F11"},
            {"type": "date_gte", "min_date": "2012-07-01"}
        ]);

        let conditions: Vec<Condition> = serde_json::from_value(json).unwrap();
        assert_eq!(conditions.len(), 3);
    }

    #[test]
    fn test_compile_valid_conditions() {
        let condition = Condition::DxPattern {
            pattern: "^F11".to_string(),
        };
        let compiled = condition.compile();
        assert!(compiled.is_ok());
    }

    #[test]
    fn test_compile_invalid_regex() {
        let condition = Condition::DxPattern {
            pattern: "[invalid(".to_string(),
        };
        let compiled = condition.compile();
        assert!(compiled.is_err());
    }
}
