// Template implementations module

pub mod threshold_rule;
pub mod duplicate_rule;
pub mod missing_field_rule;
pub mod field_pattern_rule;
pub mod cross_field_rule;

// Re-export templates
pub use threshold_rule::{ThresholdRuleTemplate, ThresholdRule, ComparisonOperator};
pub use duplicate_rule::{DuplicateRuleTemplate, DuplicateRule};
pub use missing_field_rule::{MissingFieldRuleTemplate, MissingFieldRule};
pub use field_pattern_rule::{FieldPatternRuleTemplate, FieldPatternRule};
pub use cross_field_rule::{CrossFieldRuleTemplate, CrossFieldRule, CrossFieldOperator};
