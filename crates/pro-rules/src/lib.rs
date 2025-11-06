// Rules engine for Professional SMART - Healthcare claim auditing

pub mod flag_types;
pub mod rule_engine;
pub mod rules;
pub mod loader; // Data-driven rule loading from database
pub mod template; // PHASE 3: Rule template system
pub mod templates; // PHASE 3: Template implementations
pub mod hot_reload; // PHASE 4: Hot reload infrastructure
pub mod result_cache; // PHASE 5: Rule result caching
pub mod ml_hybrid_rule; // PHASE 6: ML-hybrid rules
pub mod composite_rule; // PHASE 6: Composite rules
pub mod execution_planner; // PHASE 6: Rule execution planning
pub mod interned_context; // PHASE 6: String interning support

// Re-export commonly used items
pub use flag_types::{FlagCategory, FlagIssueType, FlagSeverity, FlagContext};
pub use rule_engine::{RuleEngine, RuleResult, RuleExecutionContext, RuleExecutionCache};
pub use loader::{load_rules_from_database, LoadedRuleInfo};
pub use template::{RuleTemplate, ParameterSchema, TemplateRegistry}; // PHASE 3
pub use templates::*; // PHASE 3: Export all template implementations
pub use hot_reload::{ReloadCoordinator, setup_reload_signal}; // PHASE 4
pub use result_cache::{RuleResultCache, CacheStats}; // PHASE 5
pub use ml_hybrid_rule::MLHybridRule; // PHASE 6
pub use composite_rule::CompositeRule; // PHASE 6
pub use execution_planner::RuleExecutionPlanner; // PHASE 6
pub use pro_common::{Error, Result};
