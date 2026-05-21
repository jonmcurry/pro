// Rule execution engine for healthcare claim auditing

use crate::flag_types::{FlagContext, FlagIssueType, FlagSeverity};
use async_trait::async_trait;
use pro_common::{Error, Result};
use rust_decimal::Decimal;
use sqlx::{PgPool, Row};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::info;

use chrono::NaiveDate;
use rustc_hash::FxHashMap; // PHASE 4: Faster HashMap for performance

/// Result of rule execution
#[derive(Debug, Clone)]
pub struct RuleResult {
    pub flag_type: FlagIssueType,
    pub severity: FlagSeverity,
    pub description: String,
    pub details: Option<String>,
    pub financial_impact: Option<Decimal>,
    pub context: FlagContext,
    /// Database issue_code for flag_issue JOIN (e.g., "TEST_99213_SA", "QM_AHRQOP001A")
    /// This is the actual issue_code from claims.flag_issue table, NOT the FlagIssueType enum code
    pub issue_code: Option<String>,
}

impl RuleResult {
    pub fn new(flag_type: FlagIssueType, context: FlagContext) -> Self {
        Self {
            flag_type,
            severity: flag_type.default_severity(),
            description: flag_type.description().to_string(),
            details: None,
            financial_impact: None,
            context,
            issue_code: None,
        }
    }

    pub fn with_issue_code(mut self, issue_code: String) -> Self {
        self.issue_code = Some(issue_code);
        self
    }

    pub fn with_details(mut self, details: String) -> Self {
        self.details = Some(details);
        self
    }

    pub fn with_financial_impact(mut self, amount: Decimal) -> Self {
        self.financial_impact = Some(amount);
        self
    }

    pub fn with_severity(mut self, severity: FlagSeverity) -> Self {
        self.severity = severity;
        self
    }
}

/// Context for rule execution
#[derive(Debug, Clone)]
pub struct RuleExecutionContext {
    pub organization_id: i64,
    pub facility_id: Option<i64>,
    pub encounter_id: Option<i64>,
    pub service_line_id: Option<i64>,
    pub provider_id: Option<i64>,
    pub coder_id: Option<i64>,

    // Service line data
    pub procedure_code: Option<String>,
    pub procedure_modifiers: Vec<String>,
    pub service_unit_count: Option<Decimal>,
    pub line_item_charge_amount: Option<Decimal>,
    pub date_of_service: Option<chrono::NaiveDate>,

    // Encounter data
    pub diagnosis_codes: Vec<String>,
    pub place_of_service_code: Option<String>,
    pub total_claim_charge_amount: Option<Decimal>,
    pub date_of_service_from: Option<chrono::NaiveDate>,
    pub date_of_service_to: Option<chrono::NaiveDate>,

    // Provider data
    pub provider_type: Option<String>,
    pub provider_specialty: Option<String>,

    // Facility data
    pub facility_state_code: Option<String>,
    pub facility_type: Option<String>,

    // Patient/subscriber data (PHASE 3 - needed for cache lookups)
    pub subscriber_id: Option<String>,

    // Additional context
    pub custom_data: HashMap<String, String>,

    // PERFORMANCE: Pre-computed uppercase values for fast condition matching
    // These avoid repeated to_uppercase() allocations in hot loops (537 rules × 3 service lines)
    pub procedure_code_upper: Option<String>,
    pub diagnosis_codes_upper: Vec<String>,
    pub place_of_service_upper: Option<String>,
    pub modifiers_upper: Vec<String>,

    // PERFORMANCE: DxPattern regex result cache
    // Maps regex pattern string -> whether ANY diagnosis code matched
    // Shared across all rules evaluating the same encounter's diagnosis codes,
    // deduplicating regex evaluation when 20+ rules share the same DxPattern
    pub dx_pattern_cache: Option<FxHashMap<String, bool>>,
    pub dx_pattern_exclude_cache: Option<FxHashMap<(String, String), bool>>,
}

impl RuleExecutionContext {
    pub fn new(organization_id: i64) -> Self {
        Self {
            organization_id,
            facility_id: None,
            encounter_id: None,
            service_line_id: None,
            provider_id: None,
            coder_id: None,
            procedure_code: None,
            procedure_modifiers: Vec::new(),
            service_unit_count: None,
            line_item_charge_amount: None,
            date_of_service: None,
            diagnosis_codes: Vec::new(),
            place_of_service_code: None,
            total_claim_charge_amount: None,
            date_of_service_from: None,
            date_of_service_to: None,
            provider_type: None,
            provider_specialty: None,
            facility_state_code: None,
            facility_type: None,
            subscriber_id: None,
            custom_data: HashMap::new(),
            // PERFORMANCE: Initialize empty, will be populated via finalize()
            procedure_code_upper: None,
            diagnosis_codes_upper: Vec::new(),
            place_of_service_upper: None,
            modifiers_upper: Vec::new(),
            dx_pattern_cache: None,
            dx_pattern_exclude_cache: None,
        }
    }

    /// PERFORMANCE: Pre-compute uppercase values before rule execution
    /// Call this once after setting all context fields, before executing rules
    /// This avoids thousands of repeated to_uppercase() allocations
    #[inline]
    pub fn finalize(&mut self) {
        self.procedure_code_upper = self.procedure_code.as_ref().map(|s| s.to_uppercase());
        self.diagnosis_codes_upper = self.diagnosis_codes.iter().map(|s| s.to_uppercase()).collect();
        self.place_of_service_upper = self.place_of_service_code.as_ref().map(|s| s.to_uppercase());
        self.modifiers_upper = self.procedure_modifiers.iter().map(|s| s.to_uppercase()).collect();
        self.dx_pattern_cache = Some(FxHashMap::default());
        self.dx_pattern_exclude_cache = Some(FxHashMap::default());
    }

    /// PERFORMANCE: Finalize with pre-computed uppercase diagnosis codes
    /// Use this when diagnosis codes are shared across multiple service lines in an encounter
    /// This avoids computing uppercase N times for N service lines
    #[inline]
    pub fn finalize_with_shared_dx(&mut self, diagnosis_codes_upper: &[String]) {
        self.procedure_code_upper = self.procedure_code.as_ref().map(|s| s.to_uppercase());
        self.diagnosis_codes_upper = diagnosis_codes_upper.to_vec();
        self.place_of_service_upper = self.place_of_service_code.as_ref().map(|s| s.to_uppercase());
        self.modifiers_upper = self.procedure_modifiers.iter().map(|s| s.to_uppercase()).collect();
        self.dx_pattern_cache = Some(FxHashMap::default());
        self.dx_pattern_exclude_cache = Some(FxHashMap::default());
    }

    pub fn to_flag_context(&self) -> FlagContext {
        FlagContext {
            encounter_id: self.encounter_id,
            service_line_id: self.service_line_id,
            organization_id: self.organization_id,
            facility_id: self.facility_id,
            provider_id: self.provider_id,
            coder_id: self.coder_id,
            procedure_code: self.procedure_code.clone(),
            diagnosis_codes: self.diagnosis_codes.clone(),
            modifiers: self.procedure_modifiers.clone(),
            units: self.service_unit_count,
            charge_amount: self.line_item_charge_amount,
        }
    }
}

/// Provider information for cache
#[derive(Debug, Clone)]
pub struct ProviderInfo {
    pub provider_type: Option<String>,
    pub taxonomy_code: Option<String>,
    pub specialty: Option<String>,
}

/// Encounter history information for cache
#[derive(Debug, Clone)]
pub struct EncounterInfo {
    pub encounter_id: i64,
    pub date_of_service: NaiveDate,
    pub claim_status: String,
}

/// Rule execution cache for batch optimization (PHASE 3 OPTIMIZATION)
/// PHASE 4: Uses FxHashMap for ~30% faster lookups
#[derive(Debug, Clone)]
pub struct RuleExecutionCache {
    /// Duplicate service line lookup: (procedure_code, date, provider_id) -> Vec<service_line_id>
    duplicate_service_lines: FxHashMap<(String, NaiveDate, Option<i64>), Vec<i64>>,

    /// Provider credentials lookup: provider_id -> ProviderInfo
    provider_credentials: FxHashMap<i64, ProviderInfo>,

    /// Patient encounter history: subscriber_id -> Vec<EncounterInfo>
    encounter_history: FxHashMap<String, Vec<EncounterInfo>>,
}

impl RuleExecutionCache {
    pub fn new() -> Self {
        Self {
            duplicate_service_lines: FxHashMap::default(),
            provider_credentials: FxHashMap::default(),
            encounter_history: FxHashMap::default(),
        }
    }

    /// Pre-fetch all data needed for batch rule execution
    /// PHASE 4: Pre-allocate capacity for better memory performance
    pub async fn populate_for_batch(
        &mut self,
        service_lines: &[(&str, NaiveDate, Option<i64>)], // (id, procedure_code, date, provider_id)
        provider_ids: &[i64],
        subscriber_ids: &[String],
        pool: &PgPool,
    ) -> Result<()> {
        // PHASE 4: Reserve capacity upfront to avoid reallocations
        self.duplicate_service_lines.reserve(service_lines.len());
        self.provider_credentials.reserve(provider_ids.len());
        self.encounter_history.reserve(subscriber_ids.len());

        // Batch query 1: Check all service lines for duplicates
        if !service_lines.is_empty() {
            self.populate_duplicate_checks(service_lines, pool).await?;
        }

        // Batch query 2: Fetch provider credentials
        if !provider_ids.is_empty() {
            self.populate_provider_info(provider_ids, pool).await?;
        }

        // Batch query 3: Fetch encounter history
        if !subscriber_ids.is_empty() {
            self.populate_encounter_history(subscriber_ids, pool).await?;
        }

        Ok(())
    }

    // PHASE 4: Optimized with single UNNEST query instead of loop
    async fn populate_duplicate_checks(
        &mut self,
        service_lines: &[(&str, NaiveDate, Option<i64>)],
        pool: &PgPool,
    ) -> Result<()> {
        // Build unique keys to query
        let mut unique_keys: Vec<(String, NaiveDate, Option<i64>)> = service_lines.iter()
            .map(|(proc_code, date, provider_id)| {
                (proc_code.to_string(), *date, *provider_id)
            })
            .collect();
        unique_keys.sort();
        unique_keys.dedup();

        if unique_keys.is_empty() {
            return Ok(());
        }

        // PHASE 4: Use UNNEST for batch query instead of looping
        // Separate keys into with-provider and without-provider for efficient querying
        let mut proc_codes: Vec<String> = Vec::with_capacity(unique_keys.len());
        let mut dates: Vec<NaiveDate> = Vec::with_capacity(unique_keys.len());
        let mut provider_ids: Vec<Option<i64>> = Vec::with_capacity(unique_keys.len());

        for (proc_code, date, provider_id) in &unique_keys {
            proc_codes.push(proc_code.clone());
            dates.push(*date);
            provider_ids.push(*provider_id);
        }

        // Single batch query using UNNEST - handles both NULL and non-NULL provider_ids
        let query = r#"
            WITH lookup_keys AS (
                SELECT
                    unnest($1::text[]) as proc_code,
                    unnest($2::date[]) as svc_date,
                    unnest($3::uuid[]) as prov_id
            )
            SELECT
                lk.proc_code,
                lk.svc_date,
                lk.prov_id,
                sl.service_line_id
            FROM lookup_keys lk
            LEFT JOIN claims.service_line sl ON
                sl.procedure_code = lk.proc_code
                AND sl.service_date_from = lk.svc_date
                AND (
                    (lk.prov_id IS NULL AND sl.rendering_provider_id IS NULL)
                    OR sl.rendering_provider_id = lk.prov_id
                )
        "#;

        let rows = sqlx::query(query)
            .bind(&proc_codes)
            .bind(&dates)
            .bind(&provider_ids)
            .fetch_all(pool)
            .await
            .map_err(Error::Database)?;

        // Group results by key
        for row in rows {
            let proc_code: String = row.get("proc_code");
            let date: NaiveDate = row.get("svc_date");
            let provider_id: Option<i64> = row.get("prov_id");
            let service_line_id: Option<i64> = row.get("service_line_id");

            if let Some(sl_id) = service_line_id {
                self.duplicate_service_lines
                    .entry((proc_code, date, provider_id))
                    .or_insert_with(Vec::new)
                    .push(sl_id);
            }
        }

        Ok(())
    }

    async fn populate_provider_info(
        &mut self,
        provider_ids: &[i64],
        pool: &PgPool,
    ) -> Result<()> {
        let query = r#"
            SELECT provider_id, provider_type, taxonomy_code, specialty
            FROM claims.provider
            WHERE provider_id = ANY($1)
        "#;

        let rows = sqlx::query(query)
            .bind(provider_ids)
            .fetch_all(pool)
            .await
            .map_err(Error::Database)?;

        for row in rows {
            let provider_id: i64 = row.get("provider_id");
            self.provider_credentials.insert(provider_id, ProviderInfo {
                provider_type: row.get("provider_type"),
                taxonomy_code: row.get("taxonomy_code"),
                specialty: row.get("specialty"),
            });
        }

        Ok(())
    }

    async fn populate_encounter_history(
        &mut self,
        subscriber_ids: &[String],
        pool: &PgPool,
    ) -> Result<()> {
        let query = r#"
            SELECT subscriber_id, encounter_id, date_of_service_from, claim_status
            FROM claims.encounter
            WHERE subscriber_id = ANY($1)
            AND date_of_service_from >= CURRENT_DATE - INTERVAL '3 years'
            ORDER BY subscriber_id, date_of_service_from DESC
        "#;

        let rows = sqlx::query(query)
            .bind(subscriber_ids)
            .fetch_all(pool)
            .await
            .map_err(Error::Database)?;

        for row in rows {
            let subscriber_id: String = row.get("subscriber_id");
            let encounter_id: i64 = row.get("encounter_id");
            let date_of_service: NaiveDate = row.get("date_of_service_from");
            let claim_status: String = row.get("claim_status");

            self.encounter_history
                .entry(subscriber_id)
                .or_insert_with(Vec::new)
                .push(EncounterInfo {
                    encounter_id,
                    date_of_service,
                    claim_status,
                });
        }

        Ok(())
    }

    /// Lookup duplicate service lines from cache
    pub fn get_duplicate_service_lines(
        &self,
        procedure_code: &str,
        service_date: NaiveDate,
        provider_id: Option<i64>,
    ) -> Option<&Vec<i64>> {
        self.duplicate_service_lines.get(&(procedure_code.to_string(), service_date, provider_id))
    }

    /// Lookup provider info from cache
    pub fn get_provider_info(&self, provider_id: i64) -> Option<&ProviderInfo> {
        self.provider_credentials.get(&provider_id)
    }

    /// Lookup encounter history from cache
    pub fn get_encounter_history(&self, subscriber_id: &str) -> Option<&Vec<EncounterInfo>> {
        self.encounter_history.get(subscriber_id)
    }
}

/// Trait for rule implementations
#[async_trait]
pub trait Rule: Send + Sync {
    /// Get the flag issue type this rule detects
    fn flag_type(&self) -> FlagIssueType;

    /// Execute the rule and return results if flag conditions are met
    async fn execute(&self, ctx: &mut RuleExecutionContext, pool: &PgPool) -> Result<Option<RuleResult>>;

    /// Execute rule with cache (PHASE 3 OPTIMIZATION)
    /// Default implementation calls execute() - rules can override to use cache
    async fn execute_with_cache(
        &self,
        ctx: &mut RuleExecutionContext,
        _cache: &RuleExecutionCache,
        pool: &PgPool,
    ) -> Result<Option<RuleResult>> {
        self.execute(ctx, pool).await
    }

    /// Get rule name for logging/reporting
    fn name(&self) -> &str {
        self.flag_type().name()
    }

    /// Check if rule is enabled (can be overridden)
    fn is_enabled(&self) -> bool {
        true
    }

    /// Check if rule requires database access during execution
    /// Rules that don't require DB access can be executed synchronously for better performance
    fn requires_db_access(&self) -> bool {
        true // Default: assume rules need DB access (safe default)
    }

    /// Get the CPT codes this rule applies to (for indexing optimization)
    /// Returns None if rule applies to all CPT codes or uses pattern matching
    fn applicable_cpt_codes(&self) -> Option<&[String]> {
        None // Default: rule applies to all CPT codes
    }

    /// Synchronous execution for rules that don't need database access
    /// This avoids async overhead for CPU-only rules like COMPOSITE
    fn execute_sync(&self, _ctx: &mut RuleExecutionContext) -> Result<Option<RuleResult>> {
        Err(Error::Validation("Rule does not support synchronous execution".into()))
    }
}

/// Main rule engine
pub struct RuleEngine {
    pool: PgPool,
    rules: Vec<Arc<dyn Rule>>, // PHASE 3: Use Arc for parallel execution
    enabled_flag_types: Option<Vec<FlagIssueType>>,
    /// PHASE 7: Execution planner for intelligent rule ordering
    execution_planner: Option<crate::execution_planner::RuleExecutionPlanner>,
    /// PHASE 7: String interner for memory optimization
    string_interner: Option<std::sync::Arc<pro_common::StringInterner>>,
    /// CPT code index: maps CPT codes to rule indices for O(1) lookup
    /// Built when rules are loaded for fast filtering
    cpt_rule_index: FxHashMap<String, Vec<usize>>,
    /// Rules that apply to all CPT codes (no cpt_in filter or uses pattern matching)
    universal_rules: Vec<usize>,
    /// PERFORMANCE: Cached flag indicating all rules support sync execution
    /// Computed once when index is built, avoids per-call iteration
    all_sync_capable: bool,
}

impl RuleEngine {
    pub fn new(pool: PgPool) -> Self {
        Self {
            pool,
            rules: Vec::new(),
            enabled_flag_types: None,
            execution_planner: None,  // PHASE 7: Optional, enabled with enable_execution_planner()
            string_interner: None,    // PHASE 7: Optional, enabled with enable_string_interning()
            cpt_rule_index: FxHashMap::default(),
            universal_rules: Vec::new(),
            all_sync_capable: true,   // Default true, updated when rules are loaded
        }
    }

    /// Build the CPT code index for fast rule filtering
    /// This dramatically improves performance when many rules are loaded (e.g., 500+ rules)
    /// by allowing O(1) lookup of applicable rules instead of O(n) iteration
    pub fn build_cpt_index(&mut self) {
        self.cpt_rule_index.clear();
        self.universal_rules.clear();

        // PERFORMANCE: Compute sync capability once during index build
        self.all_sync_capable = self.rules.iter().all(|r| !r.requires_db_access());

        for (idx, rule) in self.rules.iter().enumerate() {
            if let Some(cpt_codes) = rule.applicable_cpt_codes() {
                // Rule has specific CPT codes - add to index
                for cpt in cpt_codes {
                    self.cpt_rule_index
                        .entry(cpt.to_uppercase())
                        .or_insert_with(Vec::new)
                        .push(idx);
                }
            } else {
                // Rule applies to all CPT codes
                self.universal_rules.push(idx);
            }
        }

        // Log index statistics
        let indexed_cpts = self.cpt_rule_index.len();
        let universal_count = self.universal_rules.len();
        let total_rules = self.rules.len();
        let indexed_rule_count = self.cpt_rule_index.values().map(|v| v.len()).sum::<usize>();
        if total_rules > 0 {
            info!(
                total_rules = total_rules,
                indexed_cpts = indexed_cpts,
                indexed_rules = indexed_rule_count,
                universal_rules = universal_count,
                all_sync = self.all_sync_capable,
                "CPT index built: {} CPT codes -> {} rules indexed, {} universal rules, sync_capable={}",
                indexed_cpts, indexed_rule_count, universal_count, self.all_sync_capable
            );

            // PERFORMANCE WARNING: Too many universal rules will cause slow processing
            // Universal rules run on EVERY service line regardless of CPT code
            if universal_count > 50 {
                tracing::warn!(
                    universal_count = universal_count,
                    total_rules = total_rules,
                    "PERFORMANCE WARNING: {} universal rules (no cpt_in filter) will execute on every service line. \
                     Consider adding cpt_in conditions to rules for better performance. \
                     Target: <50 universal rules for optimal throughput.",
                    universal_count
                );
            }
        }
    }

    /// PHASE 7: Enable execution planner for intelligent rule ordering
    /// This reorders rules based on historical performance statistics
    pub fn enable_execution_planner(&mut self) {
        self.execution_planner = Some(crate::execution_planner::RuleExecutionPlanner::new(self.rules.clone()));
    }

    /// PHASE 7: Enable string interning for memory optimization
    /// This reduces memory usage by deduplicating common strings
    pub fn enable_string_interning(&mut self) {
        self.string_interner = Some(std::sync::Arc::new(pro_common::StringInterner::new()));
    }

    /// PHASE 7: Get execution planner (if enabled)
    pub fn execution_planner(&self) -> Option<&crate::execution_planner::RuleExecutionPlanner> {
        self.execution_planner.as_ref()
    }

    /// PHASE 7: Get mutable execution planner (if enabled)
    pub fn execution_planner_mut(&mut self) -> Option<&mut crate::execution_planner::RuleExecutionPlanner> {
        self.execution_planner.as_mut()
    }

    /// Replace all rules atomically (PHASE 4: for hot reload)
    pub fn replace_rules(&mut self, new_rules: Vec<Arc<dyn Rule>>) {
        self.rules = new_rules;
    }

    /// Clear the execution cache (PHASE 4: for hot reload)
    pub fn clear_cache(&mut self) {
        // Cache is managed externally, but this method is here for future use
        // In Phase 5, result cache will be integrated into the engine
    }

    /// Add a rule to the engine
    pub fn add_rule<R: Rule + 'static>(&mut self, rule: R) {
        self.rules.push(Arc::new(rule)); // PHASE 3: Wrap in Arc
    }

    /// Add a rule that's already wrapped in Arc (for database-loaded rules)
    pub fn add_rule_arc(&mut self, rule: Arc<dyn Rule>) {
        self.rules.push(rule);
    }

    /// Enable only specific flag types (for testing or selective execution)
    pub fn enable_only(&mut self, flag_types: Vec<FlagIssueType>) {
        self.enabled_flag_types = Some(flag_types);
    }

    /// Enable all flag types
    pub fn enable_all(&mut self) {
        self.enabled_flag_types = None;
    }

    /// Execute all rules against a context
    /// PERFORMANCE: Uses fully synchronous path when all rules are sync-capable
    /// Falls back to mixed sync/async for rules needing DB access
    pub async fn execute_all(&self, ctx: &mut RuleExecutionContext) -> Result<Vec<RuleResult>> {
        if self.all_sync_capable {
            return self.execute_all_sync(ctx);
        }

        let mut results = Vec::new();

        for rule in &self.rules {
            if !rule.is_enabled() {
                continue;
            }

            if let Some(ref enabled) = self.enabled_flag_types {
                if !enabled.contains(&rule.flag_type()) {
                    continue;
                }
            }

            if !rule.requires_db_access() {
                match rule.execute_sync(ctx) {
                    Ok(Some(result)) => results.push(result),
                    Ok(None) => {}
                    Err(_) => {
                        match rule.execute(ctx, &self.pool).await {
                            Ok(Some(result)) => results.push(result),
                            Ok(None) => {}
                            Err(e) => {
                                eprintln!("Error executing rule {}: {}", rule.name(), e);
                            }
                        }
                    }
                }
            } else {
                match rule.execute(ctx, &self.pool).await {
                    Ok(Some(result)) => results.push(result),
                    Ok(None) => {}
                    Err(e) => {
                        eprintln!("Error executing rule {}: {}", rule.name(), e);
                    }
                }
            }
        }

        Ok(results)
    }

    /// PERFORMANCE-CRITICAL: Fully synchronous execution of ALL rules
    #[inline]
    pub fn execute_all_sync(&self, ctx: &mut RuleExecutionContext) -> Result<Vec<RuleResult>> {
        let mut results = Vec::with_capacity(16);
        let has_filter = self.enabled_flag_types.is_some();

        for rule in &self.rules {
            if !rule.is_enabled() {
                continue;
            }

            if has_filter {
                if let Some(ref enabled) = self.enabled_flag_types {
                    if !enabled.contains(&rule.flag_type()) {
                        continue;
                    }
                }
            }

            match rule.execute_sync(ctx) {
                Ok(Some(result)) => results.push(result),
                Ok(None) => {}
                Err(e) => {
                    eprintln!("Error executing rule {}: {}", rule.name(), e);
                }
            }
        }

        Ok(results)
    }

    /// Execute rules using CPT index for fast filtering
    /// Only executes rules that apply to the given procedure code + universal rules
    pub async fn execute_all_indexed(&self, ctx: &mut RuleExecutionContext) -> Result<Vec<RuleResult>> {
        if self.all_rules_sync_capable() {
            return self.execute_all_indexed_sync(ctx);
        }

        let mut results = Vec::new();
        let procedure_code = ctx.procedure_code_upper.as_ref().cloned();

        let mut rule_indices: Vec<usize> = Vec::new();

        if let Some(ref cpt) = procedure_code {
            if let Some(indices) = self.cpt_rule_index.get(cpt) {
                rule_indices.extend(indices.iter().copied());
            }
        }

        rule_indices.extend(self.universal_rules.iter().copied());

        for idx in rule_indices {
            let rule = &self.rules[idx];

            if !rule.is_enabled() {
                continue;
            }

            if let Some(ref enabled) = self.enabled_flag_types {
                if !enabled.contains(&rule.flag_type()) {
                    continue;
                }
            }

            if !rule.requires_db_access() {
                match rule.execute_sync(ctx) {
                    Ok(Some(result)) => results.push(result),
                    Ok(None) => {}
                    Err(_) => {
                        match rule.execute(ctx, &self.pool).await {
                            Ok(Some(result)) => results.push(result),
                            Ok(None) => {}
                            Err(e) => {
                                eprintln!("Error executing rule {}: {}", rule.name(), e);
                            }
                        }
                    }
                }
            } else {
                match rule.execute(ctx, &self.pool).await {
                    Ok(Some(result)) => results.push(result),
                    Ok(None) => {}
                    Err(e) => {
                        eprintln!("Error executing rule {}: {}", rule.name(), e);
                    }
                }
            }
        }

        Ok(results)
    }

    /// Check if all loaded rules support synchronous execution
    /// PERFORMANCE: Returns cached value computed during build_cpt_index()
    #[inline]
    fn all_rules_sync_capable(&self) -> bool {
        self.all_sync_capable
    }

    /// PERFORMANCE-CRITICAL: Fully synchronous rule execution with CPT indexing
    /// - Zero async overhead (no tokio task switching)
    /// - Pre-allocated result vector
    /// - Minimal branching in hot loop
    /// - CPT index filters ~80% of rules before evaluation
    /// - DxPattern cache deduplicates regex across rules sharing same patterns
    #[inline]
    pub fn execute_all_indexed_sync(&self, ctx: &mut RuleExecutionContext) -> Result<Vec<RuleResult>> {
        let mut results = Vec::with_capacity(16);

        let procedure_code = ctx.procedure_code_upper.as_ref().cloned();

        // Collect rule indices: CPT-specific + universal (pre-deduplicated at build time)
        let mut rule_indices: Vec<usize> = Vec::with_capacity(self.universal_rules.len() + 50);

        if let Some(ref cpt) = procedure_code {
            if let Some(indices) = self.cpt_rule_index.get(cpt) {
                rule_indices.extend(indices.iter().copied());
            }
        }

        rule_indices.extend(self.universal_rules.iter().copied());

        let has_filter = self.enabled_flag_types.is_some();

        for idx in rule_indices {
            let rule = &self.rules[idx];

            if !rule.is_enabled() {
                continue;
            }

            if has_filter {
                if let Some(ref enabled) = self.enabled_flag_types {
                    if !enabled.contains(&rule.flag_type()) {
                        continue;
                    }
                }
            }

            match rule.execute_sync(ctx) {
                Ok(Some(result)) => results.push(result),
                Ok(None) => {}
                Err(_) => {}
            }
        }

        Ok(results)
    }

    /// Execute specific rule by flag type
    pub async fn execute_rule(
        &self,
        flag_type: FlagIssueType,
        ctx: &mut RuleExecutionContext,
    ) -> Result<Option<RuleResult>> {
        for rule in &self.rules {
            if rule.flag_type() == flag_type {
                return rule.execute(ctx, &self.pool).await;
            }
        }

        Err(Error::Validation(format!(
            "Rule for flag type {:?} not found",
            flag_type
        )))
    }

    /// Execute all rules with pre-populated cache (PHASE 3 OPTIMIZATION)
    pub async fn execute_all_with_cache(
        &self,
        ctx: &mut RuleExecutionContext,
        cache: &RuleExecutionCache,
    ) -> Result<Vec<RuleResult>> {
        let mut results = Vec::with_capacity(self.rules.len() / 4);

        let rules_to_execute: Vec<_> = if let Some(planner) = &self.execution_planner {
            planner.plan_execution()
        } else {
            self.rules.iter().map(|r| Arc::clone(r)).collect()
        };

        for rule in &rules_to_execute {
            if !rule.is_enabled() {
                continue;
            }

            if let Some(ref enabled) = self.enabled_flag_types {
                if !enabled.contains(&rule.flag_type()) {
                    continue;
                }
            }

            match rule.execute_with_cache(ctx, cache, &self.pool).await {
                Ok(Some(result)) => results.push(result),
                Ok(None) => {}
                Err(e) => {
                    eprintln!("Error executing rule {}: {}", rule.name(), e);
                }
            }
        }

        Ok(results)
    }

    /// Execute all rules with both execution cache and result cache (PHASE 5 OPTIMIZATION)
    pub async fn execute_all_with_result_cache(
        &self,
        ctx: &mut RuleExecutionContext,
        exec_cache: &RuleExecutionCache,
        result_cache: &crate::result_cache::RuleResultCache,
    ) -> Result<Vec<RuleResult>> {
        if let Some(cached_results) = result_cache.get(ctx) {
            return Ok(cached_results);
        }

        let results = self.execute_all_with_cache(ctx, exec_cache).await?;

        result_cache.insert(ctx, results.clone());

        Ok(results)
    }

    /// Execute all rules with pre-populated cache (delegates to serial with cache)
    pub async fn execute_all_parallel(
        &self,
        ctx: &mut RuleExecutionContext,
        cache: &RuleExecutionCache,
    ) -> Result<Vec<RuleResult>> {
        self.execute_all_with_cache(ctx, cache).await
    }

    /// Persist flag results to database
    pub async fn persist_flags(&self, results: Vec<RuleResult>) -> Result<Vec<i64>> {
        let mut flag_ids = Vec::new();

        for result in results {
            let flag_id = self.create_flag(&result).await?;
            flag_ids.push(flag_id);
        }

        Ok(flag_ids)
    }

    /// Persist flag results within existing transaction (PHASE 2 OPTIMIZATION)
    pub async fn persist_flags_with_tx(
        &self,
        results: Vec<RuleResult>,
        tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    ) -> Result<Vec<i64>> {
        let mut flag_ids = Vec::new();

        for result in results {
            let flag_id = self.create_flag_with_tx(&result, tx).await?;
            flag_ids.push(flag_id);
        }

        Ok(flag_ids)
    }

    /// Create a single flag in the database
    /// Routes to encounter_flag or service_line_flag based on context
    async fn create_flag(&self, result: &RuleResult) -> Result<i64> {
        let flag_code = result.flag_type.code();
        let severity = result.severity.as_str();

        // Build flag reason from description + details
        let flag_reason = match &result.details {
            Some(details) => format!("{}: {}", result.description, details),
            None => result.description.clone(),
        };

        // Determine if this is a service line flag or encounter flag
        if let Some(service_line_id) = result.context.service_line_id {
            // Service line level flag
            let query = r#"
                INSERT INTO claims.service_line_flag (
                    service_line_id,
                    issue_id,
                    flag_type,
                    severity,
                    flag_reason,
                    flagged_element,
                    flag_status,
                    created_at,
                    created_by
                )
                SELECT $1, fi.issue_id, 'POST_BILL', $2, $3, $4, 'OPEN', CURRENT_TIMESTAMP, 'RULES_ENGINE'
                FROM claims.flag_issue fi
                WHERE fi.issue_code = $5
                RETURNING flag_id
            "#;

            let row = sqlx::query_as::<_, (i64,)>(query)
                .bind(service_line_id)
                .bind(severity)
                .bind(&flag_reason)
                .bind(flag_code)  // flagged_element = issue code
                .bind(flag_code)  // issue_code lookup
                .fetch_one(&self.pool)
                .await
                .map_err(|e| Error::Database(e))?;

            Ok(row.0)
        } else if let Some(encounter_id) = result.context.encounter_id {
            // Encounter level flag
            let query = r#"
                INSERT INTO claims.encounter_flag (
                    encounter_id,
                    issue_id,
                    flag_type,
                    severity,
                    flag_reason,
                    flagged_element,
                    flag_status,
                    created_at,
                    created_by
                )
                SELECT $1, fi.issue_id, 'POST_BILL', $2, $3, $4, 'OPEN', CURRENT_TIMESTAMP, 'RULES_ENGINE'
                FROM claims.flag_issue fi
                WHERE fi.issue_code = $5
                RETURNING flag_id
            "#;

            let row = sqlx::query_as::<_, (i64,)>(query)
                .bind(encounter_id)
                .bind(severity)
                .bind(&flag_reason)
                .bind(flag_code)  // flagged_element = issue code
                .bind(flag_code)  // issue_code lookup
                .fetch_one(&self.pool)
                .await
                .map_err(|e| Error::Database(e))?;

            Ok(row.0)
        } else {
            Err(Error::Validation("Flag must have either encounter_id or service_line_id".into()))
        }
    }

    /// Create a single flag within transaction (PHASE 2 OPTIMIZATION)
    /// Routes to encounter_flag or service_line_flag based on context
    async fn create_flag_with_tx(
        &self,
        result: &RuleResult,
        tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    ) -> Result<i64> {
        let flag_code = result.flag_type.code();
        let severity = result.severity.as_str();

        // Build flag reason from description + details
        let flag_reason = match &result.details {
            Some(details) => format!("{}: {}", result.description, details),
            None => result.description.clone(),
        };

        // Determine if this is a service line flag or encounter flag
        if let Some(service_line_id) = result.context.service_line_id {
            // Service line level flag
            let query = r#"
                INSERT INTO claims.service_line_flag (
                    service_line_id,
                    issue_id,
                    flag_type,
                    severity,
                    flag_reason,
                    flagged_element,
                    flag_status,
                    created_at,
                    created_by
                )
                SELECT $1, fi.issue_id, 'POST_BILL', $2, $3, $4, 'OPEN', CURRENT_TIMESTAMP, 'RULES_ENGINE'
                FROM claims.flag_issue fi
                WHERE fi.issue_code = $5
                RETURNING flag_id
            "#;

            let row = sqlx::query_as::<_, (i64,)>(query)
                .bind(service_line_id)
                .bind(severity)
                .bind(&flag_reason)
                .bind(flag_code)  // flagged_element = issue code
                .bind(flag_code)  // issue_code lookup
                .fetch_one(&mut **tx)
                .await
                .map_err(|e| Error::Database(e))?;

            Ok(row.0)
        } else if let Some(encounter_id) = result.context.encounter_id {
            // Encounter level flag
            let query = r#"
                INSERT INTO claims.encounter_flag (
                    encounter_id,
                    issue_id,
                    flag_type,
                    severity,
                    flag_reason,
                    flagged_element,
                    flag_status,
                    created_at,
                    created_by
                )
                SELECT $1, fi.issue_id, 'POST_BILL', $2, $3, $4, 'OPEN', CURRENT_TIMESTAMP, 'RULES_ENGINE'
                FROM claims.flag_issue fi
                WHERE fi.issue_code = $5
                RETURNING flag_id
            "#;

            let row = sqlx::query_as::<_, (i64,)>(query)
                .bind(encounter_id)
                .bind(severity)
                .bind(&flag_reason)
                .bind(flag_code)  // flagged_element = issue code
                .bind(flag_code)  // issue_code lookup
                .fetch_one(&mut **tx)
                .await
                .map_err(|e| Error::Database(e))?;

            Ok(row.0)
        } else {
            Err(Error::Validation("Flag must have either encounter_id or service_line_id".into()))
        }
    }

    /// Get rule count
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rule_result_creation() {
        let org_id = 1i64;
        let context = FlagContext::new(org_id);
        let result = RuleResult::new(FlagIssueType::CodUpcoding, context);

        assert_eq!(result.flag_type, FlagIssueType::CodUpcoding);
        assert_eq!(result.severity, FlagSeverity::High);
        assert!(!result.description.is_empty());
    }

    #[test]
    fn test_rule_result_builder() {
        let org_id = 1i64;
        let context = FlagContext::new(org_id);
        let result = RuleResult::new(FlagIssueType::ModMissingRequired, context)
            .with_details("Modifier 25 required for E/M with procedure".to_string())
            .with_financial_impact(Decimal::new(15000, 2))
            .with_severity(FlagSeverity::High);

        assert!(result.details.is_some());
        assert!(result.financial_impact.is_some());
        assert_eq!(result.severity, FlagSeverity::High);
    }

    #[test]
    fn test_rule_execution_context() {
        let org_id = 1i64;
        let mut ctx = RuleExecutionContext::new(org_id);

        ctx.procedure_code = Some("99213".to_string());
        ctx.procedure_modifiers = vec!["25".to_string()];
        ctx.diagnosis_codes = vec!["E11.9".to_string(), "I10".to_string()];

        let flag_ctx = ctx.to_flag_context();
        assert_eq!(flag_ctx.organization_id, org_id);
        assert_eq!(flag_ctx.procedure_code, Some("99213".to_string()));
        assert_eq!(flag_ctx.modifiers.len(), 1);
        assert_eq!(flag_ctx.diagnosis_codes.len(), 2);
    }

    #[tokio::test]
    async fn test_rule_engine_creation() {
        let pool = PgPool::connect_lazy("postgres://dummy").unwrap();
        let engine = RuleEngine::new(pool);

        assert_eq!(engine.rule_count(), 0);
    }
}
