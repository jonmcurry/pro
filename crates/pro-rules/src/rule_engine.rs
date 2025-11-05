// Rule execution engine for healthcare claim auditing

use crate::flag_types::{FlagContext, FlagIssueType, FlagSeverity};
use async_trait::async_trait;
use pro_common::{Error, Result};
use rust_decimal::Decimal;
use sqlx::{PgPool, Row};
use std::collections::HashMap;
use std::sync::Arc;

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
        }
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

    // Patient/subscriber data (PHASE 3 - needed for cache lookups)
    pub subscriber_id: Option<String>,

    // Additional context
    pub custom_data: HashMap<String, String>,
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
            subscriber_id: None,
            custom_data: HashMap::new(),
        }
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
    async fn execute(&self, ctx: &RuleExecutionContext, pool: &PgPool) -> Result<Option<RuleResult>>;

    /// Execute rule with cache (PHASE 3 OPTIMIZATION)
    /// Default implementation calls execute() - rules can override to use cache
    async fn execute_with_cache(
        &self,
        ctx: &RuleExecutionContext,
        _cache: &RuleExecutionCache,
        pool: &PgPool,
    ) -> Result<Option<RuleResult>> {
        // Default: ignore cache, call regular execute
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
}

/// Main rule engine
pub struct RuleEngine {
    pool: PgPool,
    rules: Vec<Arc<dyn Rule>>, // PHASE 3: Use Arc for parallel execution
    enabled_flag_types: Option<Vec<FlagIssueType>>,
}

impl RuleEngine {
    pub fn new(pool: PgPool) -> Self {
        Self {
            pool,
            rules: Vec::new(),
            enabled_flag_types: None,
        }
    }

    /// Add a rule to the engine
    pub fn add_rule<R: Rule + 'static>(&mut self, rule: R) {
        self.rules.push(Arc::new(rule)); // PHASE 3: Wrap in Arc
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
    pub async fn execute_all(&self, ctx: &RuleExecutionContext) -> Result<Vec<RuleResult>> {
        let mut results = Vec::new();

        for rule in &self.rules {
            // Skip if rule is not enabled
            if !rule.is_enabled() {
                continue;
            }

            // Skip if not in enabled flag types list
            if let Some(ref enabled) = self.enabled_flag_types {
                if !enabled.contains(&rule.flag_type()) {
                    continue;
                }
            }

            // Execute rule
            match rule.execute(ctx, &self.pool).await {
                Ok(Some(result)) => results.push(result),
                Ok(None) => {} // Rule didn't trigger
                Err(e) => {
                    // Log error but continue with other rules
                    eprintln!("Error executing rule {}: {}", rule.name(), e);
                }
            }
        }

        Ok(results)
    }

    /// Execute specific rule by flag type
    pub async fn execute_rule(
        &self,
        flag_type: FlagIssueType,
        ctx: &RuleExecutionContext,
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
    /// PHASE 4: Pre-allocate results capacity
    pub async fn execute_all_with_cache(
        &self,
        ctx: &RuleExecutionContext,
        cache: &RuleExecutionCache,
    ) -> Result<Vec<RuleResult>> {
        // PHASE 4: Pre-allocate capacity (most rules won't trigger, but this avoids reallocation)
        let mut results = Vec::with_capacity(self.rules.len() / 4);

        for rule in &self.rules {
            // Skip if rule is not enabled
            if !rule.is_enabled() {
                continue;
            }

            // Skip if not in enabled flag types list
            if let Some(ref enabled) = self.enabled_flag_types {
                if !enabled.contains(&rule.flag_type()) {
                    continue;
                }
            }

            // Execute rule with cache
            match rule.execute_with_cache(ctx, cache, &self.pool).await {
                Ok(Some(result)) => results.push(result),
                Ok(None) => {} // Rule didn't trigger
                Err(e) => {
                    // Log error but continue with other rules
                    eprintln!("Error executing rule {}: {}", rule.name(), e);
                }
            }
        }

        Ok(results)
    }

    /// Execute all rules with both execution cache and result cache (PHASE 5 OPTIMIZATION)
    /// Combines Phase 3 execution cache with Phase 5 result cache for maximum performance
    /// PHASE 6: Automatically uses parallel execution when >= 5 rules are enabled (3-5x speedup)
    pub async fn execute_all_with_result_cache(
        &self,
        ctx: &RuleExecutionContext,
        exec_cache: &RuleExecutionCache,
        result_cache: &crate::result_cache::RuleResultCache,
    ) -> Result<Vec<RuleResult>> {
        // Try to get cached results first
        if let Some(cached_results) = result_cache.get(ctx) {
            return Ok(cached_results);
        }

        // PHASE 6: Count enabled rules to decide serial vs parallel execution
        let enabled_rule_count = self.rules.iter()
            .filter(|rule| {
                if !rule.is_enabled() {
                    return false;
                }
                if let Some(ref enabled) = self.enabled_flag_types {
                    enabled.contains(&rule.flag_type())
                } else {
                    true
                }
            })
            .count();

        // PHASE 6: Use parallel execution for >= 5 rules (3-5x faster)
        let results = if enabled_rule_count >= 5 {
            self.execute_all_parallel(ctx, exec_cache).await?
        } else {
            // For fewer rules, serial execution has less overhead
            self.execute_all_with_cache(ctx, exec_cache).await?
        };

        // Store results in cache for future use
        result_cache.insert(ctx, results.clone());

        Ok(results)
    }

    /// Execute all rules in parallel with pre-populated cache (PHASE 3 OPTIMIZATION)
    /// This provides maximum performance by running independent rules concurrently
    /// PHASE 4: Optimized with pre-allocation
    pub async fn execute_all_parallel(
        &self,
        ctx: &RuleExecutionContext,
        cache: &RuleExecutionCache,
    ) -> Result<Vec<RuleResult>> {
        use tokio::task::JoinSet;

        let mut join_set = JoinSet::new();

        // Share context and cache across tasks
        let ctx = Arc::new(ctx.clone());
        let cache = Arc::new(cache.clone());

        // Spawn parallel task for each enabled rule
        for rule in &self.rules {
            // Skip if rule is not enabled
            if !rule.is_enabled() {
                continue;
            }

            // Skip if not in enabled flag types list
            if let Some(ref enabled) = self.enabled_flag_types {
                if !enabled.contains(&rule.flag_type()) {
                    continue;
                }
            }

            // Clone Arc references for the task
            let rule_arc = Arc::clone(rule);
            let ctx_arc = Arc::clone(&ctx);
            let cache_arc = Arc::clone(&cache);
            let pool = self.pool.clone();

            // Spawn task for this rule
            join_set.spawn(async move {
                rule_arc.execute_with_cache(&ctx_arc, &cache_arc, &pool).await
            });
        }

        // Collect results as tasks complete
        // PHASE 4: Pre-allocate based on typical flag rate (~10-20%)
        let mut results = Vec::with_capacity(self.rules.len() / 5);
        while let Some(join_result) = join_set.join_next().await {
            match join_result {
                Ok(Ok(Some(rule_result))) => results.push(rule_result),
                Ok(Ok(None)) => {} // Rule didn't trigger
                Ok(Err(e)) => {
                    eprintln!("Rule execution error: {}", e);
                }
                Err(e) => {
                    eprintln!("Task join error: {}", e);
                }
            }
        }

        Ok(results)
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
    async fn create_flag(&self, result: &RuleResult) -> Result<i64> {
        let flag_code = result.flag_type.code();
        let flag_category = result.flag_type.category().code();
        let severity = result.severity.as_str();

        let query = r#"
            INSERT INTO claims.flag (
                encounter_id,
                service_line_id,
                organization_id,
                facility_id,
                provider_id,
                coder_id,
                flag_category,
                flag_issue_code,
                flag_issue_type,
                flag_severity,
                flag_description,
                flag_details,
                financial_impact,
                flag_status,
                created_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, 'OPEN', CURRENT_TIMESTAMP)
            RETURNING flag_id
        "#;

        let flag_issue_type = result.flag_type.name();

        let row = sqlx::query_as::<_, (i64,)>(query)
            .bind(result.context.encounter_id)
            .bind(result.context.service_line_id)
            .bind(result.context.organization_id)
            .bind(result.context.facility_id)
            .bind(result.context.provider_id)
            .bind(result.context.coder_id)
            .bind(flag_category)
            .bind(flag_code)
            .bind(flag_issue_type)
            .bind(severity)
            .bind(&result.description)
            .bind(&result.details)
            .bind(result.financial_impact)
            .fetch_one(&self.pool)
            .await
            .map_err(|e| Error::Database(e))?;

        Ok(row.0)
    }

    /// Create a single flag within transaction (PHASE 2 OPTIMIZATION)
    async fn create_flag_with_tx(
        &self,
        result: &RuleResult,
        tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    ) -> Result<i64> {
        let flag_code = result.flag_type.code();
        let flag_category = result.flag_type.category().code();
        let severity = result.severity.as_str();

        let query = r#"
            INSERT INTO claims.flag (
                encounter_id,
                service_line_id,
                organization_id,
                facility_id,
                provider_id,
                coder_id,
                flag_category,
                flag_issue_code,
                flag_issue_type,
                flag_severity,
                flag_description,
                flag_details,
                financial_impact,
                flag_status,
                created_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, 'OPEN', CURRENT_TIMESTAMP)
            RETURNING flag_id
        "#;

        let flag_issue_type = result.flag_type.name();

        let row = sqlx::query_as::<_, (i64,)>(query)
            .bind(result.context.encounter_id)
            .bind(result.context.service_line_id)
            .bind(result.context.organization_id)
            .bind(result.context.facility_id)
            .bind(result.context.provider_id)
            .bind(result.context.coder_id)
            .bind(flag_category)
            .bind(flag_code)
            .bind(flag_issue_type)
            .bind(severity)
            .bind(&result.description)
            .bind(&result.details)
            .bind(result.financial_impact)
            .fetch_one(&mut **tx)
            .await
            .map_err(|e| Error::Database(e))?;

        Ok(row.0)
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
