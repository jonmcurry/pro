//! Claims Processor - Stage 2 of two-stage processing pipeline
//!
//! Processes claims from staging.raw_claims table and inserts them into either:
//! - claims.encounter (if valid)
//! - staging.import_error_log (if invalid)
//!
//! This is Stage 2 of the pipeline:
//! - Stage 1: File -> staging.raw_claims (fast ingestion) - ClaimsImporter::ingest_file_to_staging()
//! - Stage 2: staging.raw_claims -> encounters/errors (validated processing) - THIS MODULE

use anyhow::{Context, Result};
use futures::future::join_all;
use pro_rules::{RuleEngine, RuleExecutionContext, load_rules_from_database};
use serde_json::Value as JsonValue;
use sqlx::PgPool;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, Semaphore};
use tracing::{debug, error, info, warn};

/// Default maximum concurrent encounters to process in parallel within a batch.
/// Configurable via MAX_CONCURRENT_ENCOUNTERS env var.
///
/// Sized for the target hardware profile (8 vCPU box co-located with Postgres,
/// default STAGE2_WORKER_COUNT=4, DB_MAX_CONNECTIONS=24):
///     worst-case demand = workers * MAX_CONCURRENT_ENCOUNTERS = 16
///     leaves headroom in the 24-connection pool for prewarm + status updates
///     keeps active Postgres backends near 2x vCPU count (PG sweet spot)
///
/// Override via MAX_CONCURRENT_ENCOUNTERS env var on larger hardware.
const DEFAULT_MAX_CONCURRENT_ENCOUNTERS: usize = 4;

/// Get the configured max concurrent encounters from env or use default
fn get_max_concurrent_encounters() -> usize {
    std::env::var("MAX_CONCURRENT_ENCOUNTERS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(DEFAULT_MAX_CONCURRENT_ENCOUNTERS)
}

/// Helper function to extract a string value from a serde_json::Value
/// Handles both String values and converts other types to strings
fn get_string_value(value: &JsonValue) -> Option<String> {
    match value {
        JsonValue::String(s) => Some(s.clone()),
        JsonValue::Number(n) => Some(n.to_string()),
        JsonValue::Bool(b) => Some(b.to_string()),
        JsonValue::Null => None,
        // For arrays and objects, return None (caller should handle separately)
        _ => None,
    }
}

/// Helper function to get a string from a HashMap<String, JsonValue>
fn get_field_as_string(fields: &HashMap<String, JsonValue>, key: &str) -> Option<String> {
    fields.get(key).and_then(get_string_value)
}

/// PERFORMANCE: Extract a field directly from a JsonValue object without deserializing
/// This avoids cloning the entire JsonValue just to get one field
fn get_field_from_json(value: &JsonValue, key: &str) -> Option<String> {
    value.get(key).and_then(get_string_value)
}

/// Wrapper type for encounter fields that provides String-like access to JsonValue fields
/// This allows existing code patterns to work with minimal changes
struct EncounterFieldsWrapper {
    inner: HashMap<String, JsonValue>,
}

impl EncounterFieldsWrapper {
    fn new(value: JsonValue) -> Result<Self, serde_json::Error> {
        let inner: HashMap<String, JsonValue> = serde_json::from_value(value)?;
        Ok(Self { inner })
    }

    /// Get a field as Option<String>, converting from JsonValue
    fn get(&self, key: &str) -> Option<String> {
        self.inner.get(key).and_then(get_string_value)
    }

    /// Get a field as Option<&str> by returning a reference to a cached string
    /// For complex patterns, use get() and work with the owned String
    fn get_raw(&self, key: &str) -> Option<&JsonValue> {
        self.inner.get(key)
    }

    /// Get the inner HashMap for direct access (e.g., for other_insurance array)
    fn inner(&self) -> &HashMap<String, JsonValue> {
        &self.inner
    }
}


/// Provider metadata accumulated during prewarm collection.
/// Shared by the per-encounter and batch-level prewarm paths.
struct ProviderData {
    npi: String,
    provider_type: String,
    last_name: String,
    first_name: String,
    taxonomy_code: Option<String>,
}

/// Claims processor for Stage 2 of two-stage pipeline
#[derive(Clone)]
pub struct ClaimsProcessor {
    pool: PgPool,
    /// Cache of taxonomy_code -> specialty_display for fast lookups
    /// Loaded lazily on first provider insert
    taxonomy_cache: Arc<RwLock<HashMap<String, String>>>,
    /// Flag to track if cache has been loaded
    taxonomy_cache_loaded: Arc<RwLock<bool>>,
    /// Cache of NPI -> provider_id for avoiding redundant DB upserts
    /// PERFORMANCE: Reduces ~16 provider DB calls per encounter to only unique NPIs
    provider_cache: Arc<RwLock<HashMap<String, i64>>>,
    /// Rules engine for post-bill auditing
    /// Loaded from database based on ENABLE_DATABASE_RULES env var
    /// NOTE: Uses Arc without RwLock because rules are loaded once at startup
    /// and never modified during runtime. This eliminates lock contention.
    rule_engine: Arc<RuleEngine>,
    /// Whether to defer rules execution to background processing
    /// PERFORMANCE: When true, rules run asynchronously after import completes
    /// This dramatically improves import throughput (10K claims/30sec target)
    defer_rules: bool,
}

impl ClaimsProcessor {
    /// Create a new claims processor with rules engine
    ///
    /// Rules are loaded from database based on ENABLE_DATABASE_RULES env var:
    /// - If true: Load rules from database (requires RULE_ENCRYPTION_KEY)
    /// - If false: Use empty rule engine (no rules executed)
    ///
    /// Rules execution mode controlled by DEFER_RULES_EXECUTION env var:
    /// - If true: Rules queued for background processing (faster import throughput)
    /// - If false: Rules executed inline during import (default, slower but immediate)
    pub async fn new(pool: PgPool) -> Self {
        // Check if database-driven rules are enabled
        let use_database_rules = std::env::var("ENABLE_DATABASE_RULES")
            .unwrap_or_else(|_| "false".to_string())
            .parse::<bool>()
            .unwrap_or(false);

        // Check if rules should be deferred to background processing
        // Default: false (inline execution) - rules execute during claim processing
        // Set DEFER_RULES_EXECUTION=true to queue rules for background processing instead
        let defer_rules_setting = std::env::var("DEFER_RULES_EXECUTION")
            .unwrap_or_else(|_| "false".to_string())
            .parse::<bool>()
            .unwrap_or(false);

        let (rule_engine, defer_rules) = if use_database_rules {
            info!("Loading rules from database for Stage 2 processor...");
            match load_rules_from_database(&pool, None).await {
                Ok((engine, rules)) => {
                    let rule_count = rules.len();
                    info!("Stage 2: Loaded {} rule(s) from database", rule_count);

                    if defer_rules_setting {
                        info!("Stage 2: Rules execution DEFERRED (DEFER_RULES_EXECUTION=true)");
                    } else {
                        info!("Stage 2: Rules execution INLINE (DEFER_RULES_EXECUTION not set or false)");
                    }

                    (engine, defer_rules_setting)
                }
                Err(e) => {
                    error!("Failed to load rules from database: {}", e);
                    warn!("Stage 2 will run without rules engine");
                    (RuleEngine::new(pool.clone()), false)
                }
            }
        } else {
            info!("Stage 2: Rules engine disabled (ENABLE_DATABASE_RULES=false)");
            (RuleEngine::new(pool.clone()), false)
        };

        Self {
            pool,
            taxonomy_cache: Arc::new(RwLock::new(HashMap::new())),
            taxonomy_cache_loaded: Arc::new(RwLock::new(false)),
            provider_cache: Arc::new(RwLock::new(HashMap::new())),
            rule_engine: Arc::new(rule_engine),
            defer_rules,
        }
    }

    /// Load taxonomy cache from database (called lazily on first use)
    async fn ensure_taxonomy_cache_loaded(&self) -> Result<()> {
        // Quick check without write lock
        {
            let loaded = self.taxonomy_cache_loaded.read().await;
            if *loaded {
                return Ok(());
            }
        }

        // Need to load - acquire write lock
        let mut loaded = self.taxonomy_cache_loaded.write().await;

        // Double-check after acquiring write lock (another thread may have loaded)
        if *loaded {
            return Ok(());
        }

        info!("Loading taxonomy cache from database...");

        let taxonomies: Vec<(String, String)> = sqlx::query_as(
            r#"
            SELECT taxonomy_code, specialty_display
            FROM claims.provider_taxonomy
            WHERE is_active = true
            "#
        )
        .fetch_all(&self.pool)
        .await
        .context("Failed to load taxonomy cache")?;

        let mut cache = self.taxonomy_cache.write().await;
        for (code, specialty) in taxonomies {
            cache.insert(code, specialty);
        }

        info!("Loaded {} taxonomy codes into cache", cache.len());
        *loaded = true;
        Ok(())
    }

    /// Lookup specialty from taxonomy code using cache
    /// Returns (validated_taxonomy_code, specialty) or (None, None) if not found
    async fn lookup_taxonomy(&self, taxonomy_code: &str) -> (Option<String>, Option<String>) {
        if taxonomy_code.is_empty() {
            return (None, None);
        }

        // Ensure cache is loaded
        if let Err(e) = self.ensure_taxonomy_cache_loaded().await {
            warn!("Failed to load taxonomy cache: {:?}", e);
            return (None, None);
        }

        let cache = self.taxonomy_cache.read().await;
        if let Some(specialty) = cache.get(taxonomy_code) {
            (Some(taxonomy_code.to_string()), Some(specialty.clone()))
        } else {
            warn!("Taxonomy code '{}' not found in cache", taxonomy_code);
            (None, None)
        }
    }

    /// Process pending claims from staging.raw_claims (STAGE 2)
    /// This method processes claims that were ingested in Stage 1
    /// Performance target: 10,000 claims / 15 seconds = 666.67 claims/sec
    pub async fn process_pending_claims(&self, limit: Option<usize>) -> Result<ProcessResult> {
        let limit = limit.unwrap_or(10000); // Default batch of 10k claims

        info!("====== STAGE 2: Starting processing of pending raw claims (limit: {}) ======", limit);

        // PHASE 1 FIX: Recover stale PROCESSING claims (stuck for > 5 minutes)
        // This prevents claims from being permanently stuck if a previous run crashed
        let stale_recovered = sqlx::query_scalar::<_, i64>(
            r#"
            WITH stale_claims AS (
                SELECT raw_claim_id
                FROM staging.raw_claims
                WHERE processing_status = 'PROCESSING'
                AND processed_at IS NULL
                AND ingested_at < CURRENT_TIMESTAMP - INTERVAL '5 minutes'
                LIMIT 10000
            )
            UPDATE staging.raw_claims rc
            SET processing_status = 'PENDING'
            FROM stale_claims sc
            WHERE rc.raw_claim_id = sc.raw_claim_id
            RETURNING rc.raw_claim_id
            "#
        )
        .fetch_all(&self.pool)
        .await
        .map(|v| v.len() as i64)
        .unwrap_or(0);

        if stale_recovered > 0 {
            info!("Recovered {} stale PROCESSING claims back to PENDING", stale_recovered);
        }

        // Query pending raw claims (FIFO order)
        let raw_claims: Vec<RawClaim> = sqlx::query_as(
            r#"
            SELECT
                raw_claim_id,
                batch_id,
                queue_id,
                encounter_fields,
                service_line_fields,
                diagnosis_fields,
                row_number,
                facility_code,
                date_of_service_from
            FROM staging.raw_claims
            WHERE processing_status = 'PENDING'
            ORDER BY ingested_at ASC
            LIMIT $1
            FOR UPDATE SKIP LOCKED
            "#
        )
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await
        .context("Failed to query pending raw claims")?;

        if raw_claims.is_empty() {
            info!("No pending raw claims to process");
            return Ok(ProcessResult {
                total_processed: 0,
                successful: 0,
                failed: 0,
            });
        }

        info!("Found {} pending raw claims to process", raw_claims.len());

        // Mark claims as PROCESSING
        let raw_claim_ids: Vec<i64> = raw_claims.iter().map(|c| c.raw_claim_id).collect();
        sqlx::query(
            r#"
            UPDATE staging.raw_claims
            SET processing_status = 'PROCESSING'
            WHERE raw_claim_id = ANY($1)
            "#
        )
        .bind(&raw_claim_ids)
        .execute(&self.pool)
        .await
        .context("Failed to mark claims as PROCESSING")?;

        let process_start = chrono::Utc::now();

        // Group claims by batch for metrics tracking
        let batch_ids: Vec<i64> = raw_claims.iter().map(|c| c.batch_id).collect();
        let unique_batch_ids: std::collections::HashSet<i64> = batch_ids.into_iter().collect();

        // PHASE 6 OPTIMIZATION: Update batch status to PROCESSING (batch query instead of loop)
        let batch_ids_vec: Vec<i64> = unique_batch_ids.iter().copied().collect();
        if !batch_ids_vec.is_empty() {
            sqlx::query(
                r#"
                UPDATE staging.import_batch
                SET import_status = 'PROCESSING'
                WHERE batch_id = ANY($1) AND import_status = 'INGESTED'
                "#
            )
            .bind(&batch_ids_vec)
            .execute(&self.pool)
            .await
            .context("Failed to update batch status to PROCESSING")?;
        }

        let mut result = ProcessResult {
            total_processed: raw_claims.len(),
            successful: 0,
            failed: 0,
        };

        // Facility lookup cache for performance (shared across encounters)
        let mut facility_cache: HashMap<String, (Option<i64>, i64, Option<i64>)> = HashMap::new();

        info!("Processing {} raw claims...", raw_claims.len());

        // Group raw_claims by encounter (patient_control_number + date_of_service)
        use std::collections::HashMap as StdHashMap;
        let mut encounter_groups: StdHashMap<(String, String), Vec<RawClaim>> = StdHashMap::new();

        for raw_claim in raw_claims {
            // PERFORMANCE: Extract encounter key fields directly from JsonValue
            // This avoids cloning/deserializing the entire encounter_fields object

            let patient_control_number = match get_field_from_json(&raw_claim.encounter_fields, "patient_control_number") {
                Some(pcn) => pcn,
                None => {
                    error!("Missing patient_control_number for raw_claim_id {}", raw_claim.raw_claim_id);
                    result.failed += 1;
                    continue;
                }
            };

            let date_of_service = match get_field_from_json(&raw_claim.encounter_fields, "date_of_service_from") {
                Some(dos) => dos,
                None => {
                    error!("Missing date_of_service_from for raw_claim_id {}", raw_claim.raw_claim_id);
                    result.failed += 1;
                    continue;
                }
            };

            let encounter_key = (patient_control_number, date_of_service);
            encounter_groups.entry(encounter_key).or_insert_with(Vec::new).push(raw_claim);
        }

        info!("Grouped {} raw claims into {} encounters", result.total_processed, encounter_groups.len());

        // PHASE 2 FIX: Collect successful/failed claim IDs for batch status updates
        // This prevents cascading rollbacks - each encounter is independent
        let mut successful_claim_ids: Vec<i64> = Vec::with_capacity(result.total_processed);
        let mut failed_claims: Vec<(i64, i64, i32, String, String)> = Vec::new(); // (raw_claim_id, batch_id, row_number, error_message, raw_data)

        // Process each encounter group with per-encounter transactions
        for ((patient_control_number, date_of_service), service_lines) in encounter_groups {
            debug!("Processing encounter: {} on {} ({} service lines)",
                patient_control_number, date_of_service, service_lines.len());

            // PHASE 2 FIX: Per-encounter transaction - failures don't cascade
            let mut tx = self.pool.begin().await
                .context("Failed to begin encounter transaction")?;

            // Validate and insert encounter with all service lines
            match self.process_encounter_with_service_lines(&mut tx, &service_lines, &mut facility_cache).await {
                Ok(encounter_id) => {
                    // Commit this encounter immediately - no batching
                    tx.commit().await
                        .context("Failed to commit encounter transaction")?;

                    result.successful += service_lines.len();

                    // Collect claim IDs for batch status update later
                    for service_line in &service_lines {
                        successful_claim_ids.push(service_line.raw_claim_id);
                    }

                    debug!("Successfully processed encounter: {} on {} -> encounter_id {}",
                        patient_control_number, date_of_service, encounter_id);
                }
                Err(e) => {
                    // Rollback failed encounter (may already be rolled back by DB)
                    let _ = tx.rollback().await;

                    result.failed += service_lines.len();
                    error!("Failed to process encounter {} on {}: {:#}", patient_control_number, date_of_service, e);

                    // Collect failed claim info for batch error logging later
                    for service_line in &service_lines {
                        let error_message = format!("Row {}: {:#}", service_line.row_number, e);
                        let raw_data = serde_json::to_string(&service_line.encounter_fields).unwrap_or_default();
                        failed_claims.push((
                            service_line.raw_claim_id,
                            service_line.batch_id,
                            service_line.row_number,
                            error_message,
                            raw_data,
                        ));
                    }
                }
            }
        }

        // PHASE 2 & 3 FIX: Batch update successful claims status
        if !successful_claim_ids.is_empty() {
            sqlx::query(
                r#"
                UPDATE staging.raw_claims
                SET processing_status = 'COMPLETED',
                    processed_at = CURRENT_TIMESTAMP
                WHERE raw_claim_id = ANY($1)
                "#
            )
            .bind(&successful_claim_ids)
            .execute(&self.pool)
            .await
            .context("Failed to batch update successful claims")?;

            debug!("Batch updated {} claims to COMPLETED", successful_claim_ids.len());
        }

        // PHASE 3 FIX: Batch update failed claims and insert error logs
        if !failed_claims.is_empty() {
            // Extract claim IDs and error messages for batch update
            let failed_ids: Vec<i64> = failed_claims.iter().map(|(id, _, _, _, _)| *id).collect();

            // Update status in batch - use a single UPDATE with CASE for error messages
            // Since we need different error messages per claim, we use a loop but outside transaction
            for (raw_claim_id, batch_id, row_number, error_message, raw_data) in &failed_claims {
                // These are outside any transaction, so failures don't cascade
                let _ = sqlx::query(
                    r#"
                    UPDATE staging.raw_claims
                    SET processing_status = 'FAILED',
                        error_message = $2,
                        processed_at = CURRENT_TIMESTAMP
                    WHERE raw_claim_id = $1
                    "#
                )
                .bind(raw_claim_id)
                .bind(error_message)
                .execute(&self.pool)
                .await;

                // Log error to staging.import_error_log (fire-and-forget)
                let _ = sqlx::query(
                    r#"
                    INSERT INTO staging.import_error_log (
                        batch_id,
                        record_number,
                        error_type,
                        error_severity,
                        error_message,
                        raw_data
                    )
                    VALUES ($1, $2, 'VALIDATION', 'ERROR', $3, $4)
                    "#
                )
                .bind(batch_id)
                .bind(row_number)
                .bind(error_message)
                .bind(raw_data)
                .execute(&self.pool)
                .await;
            }

            debug!("Batch updated {} claims to FAILED with error logs", failed_ids.len());
        }

        let process_end = chrono::Utc::now();

        info!("Processing complete: {} total, {} successful, {} failed",
            result.total_processed, result.successful, result.failed);

        // PHASE 6 OPTIMIZATION: Update batch statuses with single query using batch counts
        // Pre-compute counts for all batches at once, then update in batch
        if !unique_batch_ids.is_empty() {
            let batch_ids_vec: Vec<i64> = unique_batch_ids.iter().copied().collect();

            // Single query to get counts for all batches
            sqlx::query(
                r#"
                UPDATE staging.import_batch ib
                SET import_status = CASE
                        WHEN counts.failed_count = 0 THEN 'COMPLETED'
                        WHEN counts.successful_count > 0 THEN 'PARTIAL'
                        ELSE 'FAILED'
                    END,
                    successful_records = counts.successful_count,
                    failed_records = counts.failed_count,
                    completed_at = CURRENT_TIMESTAMP
                FROM (
                    SELECT
                        batch_id,
                        COUNT(*) FILTER (WHERE processing_status = 'COMPLETED') as successful_count,
                        COUNT(*) FILTER (WHERE processing_status = 'FAILED') as failed_count
                    FROM staging.raw_claims
                    WHERE batch_id = ANY($1)
                    GROUP BY batch_id
                ) counts
                WHERE ib.batch_id = counts.batch_id
                "#
            )
            .bind(&batch_ids_vec)
            .execute(&self.pool)
            .await
            .context("Failed to update batch completion status")?;
        }

        // Log PROCESS metric for each batch (Stage 2 performance)
        for batch_id in &unique_batch_ids {
            let batch_claim_count = result.total_processed / unique_batch_ids.len();
            let batch_success = result.successful / unique_batch_ids.len();
            let batch_failed = result.failed / unique_batch_ids.len();

            if let Err(e) = self.log_processing_metric(
                *batch_id,
                "PROCESS",
                "Claims Processing",
                process_start,
                process_end,
                batch_claim_count as i32,
                batch_success as i32,
                batch_failed as i32,
                Some(serde_json::json!({
                    "stage": "PROCESS",
                    "batch_size": batch_claim_count,
                    "successful": batch_success,
                    "failed": batch_failed
                })),
                "PROCESS"
            ).await {
                warn!("Failed to log PROCESS metric: {}", e);
            }
        }

        info!("====== STAGE 2 COMPLETE: {} successful, {} failed ======",
            result.successful, result.failed);

        Ok(result)
    }

    /// Process an encounter with multiple service lines
    /// This creates ONE encounter and N service_line records
    /// PHASE 4 FIX: Changed from Vec<RawClaim> to &[RawClaim] to avoid cloning
    async fn process_encounter_with_service_lines(
        &self,
        tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
        service_lines: &[RawClaim],
        facility_cache: &mut HashMap<String, (Option<i64>, i64, Option<i64>)>,
    ) -> Result<i64> {
        if service_lines.is_empty() {
            return Err(anyhow::anyhow!("No service lines provided"));
        }

        // Use first service line for encounter-level data (all should have same encounter info)
        let first_line = &service_lines[0];

        // Deserialize encounter fields from first line
        // Use EncounterFieldsWrapper to handle mixed types (strings, arrays like other_insurance)
        let encounter_fields = EncounterFieldsWrapper::new(first_line.encounter_fields.clone())
            .context("Failed to deserialize encounter_fields")?;

        // PERFORMANCE OPTIMIZATION: Pre-warm provider cache with batch query
        // This queries all existing providers for this encounter in ONE query,
        // so subsequent ensure_provider_exists() calls are cache hits instead of DB round-trips
        self.prewarm_provider_cache(tx, &encounter_fields, service_lines).await?;

        // Extract facility_code
        let facility_code = encounter_fields.get("facility_code")
            .or_else(|| encounter_fields.get("facility_npi"))
            .context("Missing facility_code or facility_npi")?;

        // Check cache first
        let (facility_id, organization_id, region_id) = if let Some(cached) = facility_cache.get(&facility_code) {
            *cached
        } else {
            let facility = sqlx::query_as::<_, (Option<i64>, i64, Option<i64>)>(
                r#"
                SELECT facility_id, organization_id, region_id
                FROM claims.facility
                WHERE facility_code = $1 OR npi = $1
                "#
            )
            .bind(&facility_code)
            .fetch_optional(&mut **tx)
            .await?;

            let facility_result = facility
                .with_context(|| format!("Facility not found: {}", facility_code))?;

            facility_cache.insert(facility_code.clone(), facility_result);
            facility_result
        };

        // Extract required encounter fields
        let patient_control_number = encounter_fields.get("patient_control_number")
            .context("Missing patient_control_number")?;
        let subscriber_last_name = encounter_fields.get("subscriber_last_name")
            .context("Missing subscriber_last_name")?;
        let subscriber_first_name = encounter_fields.get("subscriber_first_name")
            .context("Missing subscriber_first_name")?;
        let date_of_service_from = encounter_fields.get("date_of_service_from")
            .context("Missing date_of_service_from")?;
        let subscriber_id = encounter_fields.get("subscriber_id")
            .context("Missing subscriber_id")?;
        let subscriber_birth_date_str = encounter_fields.get("subscriber_birth_date")
            .filter(|s| !s.is_empty());

        // Subscriber demographics - optional fields
        let subscriber_middle_name = encounter_fields.get("subscriber_middle_name").filter(|s| !s.is_empty());
        let subscriber_name_suffix = encounter_fields.get("subscriber_name_suffix").filter(|s| !s.is_empty());
        // Truncate subscriber_gender to 1 char (CHAR(1) in DB)
        let subscriber_gender_str = encounter_fields.get("subscriber_gender").filter(|s| !s.is_empty());
        let subscriber_gender = subscriber_gender_str.as_ref().map(|s| if s.len() > 1 { &s[..1] } else { s.as_str() });
        let subscriber_address_line1 = encounter_fields.get("subscriber_address_line1").filter(|s| !s.is_empty());
        let subscriber_address_line2 = encounter_fields.get("subscriber_address_line2").filter(|s| !s.is_empty());
        let subscriber_city = encounter_fields.get("subscriber_city").filter(|s| !s.is_empty());
        // Truncate subscriber_state to 2 chars (CHAR(2) in DB)
        let subscriber_state_str = encounter_fields.get("subscriber_state").filter(|s| !s.is_empty());
        let subscriber_state = subscriber_state_str.as_ref().map(|s| if s.len() > 2 { &s[..2] } else { s.as_str() });
        let subscriber_postal_code = encounter_fields.get("subscriber_postal_code").filter(|s| !s.is_empty());
        // Truncate subscriber_country to 3 chars (CHAR(3) in DB)
        let subscriber_country_str = encounter_fields.get("subscriber_country").filter(|s| !s.is_empty());
        let subscriber_country = subscriber_country_str.as_ref().map(|s| if s.len() > 3 { &s[..3] } else { s.as_str() });

        // Claim reference fields
        let claim_number = encounter_fields.get("claim_number").filter(|s| !s.is_empty());
        let paperwork_report_type = encounter_fields.get("paperwork_report_type").filter(|s| !s.is_empty());
        let paperwork_transmission_code = encounter_fields.get("paperwork_transmission_code").filter(|s| !s.is_empty());
        let paperwork_control_number = encounter_fields.get("paperwork_control_number").filter(|s| !s.is_empty());

        // Optional fields
        let submitter_id = encounter_fields.get("submitter_id").unwrap_or_else(|| facility_code.clone());
        // Truncate payer_responsibility_code to 1 char (CHAR(1) in DB, must be 'P' or 'S')
        let payer_responsibility_code_str = encounter_fields.get("payer_responsibility_code")
            .unwrap_or_else(|| "P".to_string());
        let payer_responsibility_code = if payer_responsibility_code_str.len() > 1 {
            &payer_responsibility_code_str[..1]
        } else {
            &payer_responsibility_code_str
        };

        // Use total_claim_charge_amount from CLM02 segment (authoritative value from 837 file)
        let total_claim_charge = encounter_fields.get("total_claim_charge_amount")
            .and_then(|s| s.parse::<rust_decimal::Decimal>().ok())
            .unwrap_or(rust_decimal::Decimal::ZERO);

        // Parse dates
        let dos_from = chrono::NaiveDate::parse_from_str(&date_of_service_from, "%Y-%m-%d")
            .context("Invalid date format for date_of_service_from")?;
        let subscriber_dob = subscriber_birth_date_str.as_ref()
            .and_then(|s| chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d").ok());

        // Optional fields - now returns Option<String>, use .as_deref() where needed
        let payer_id = encounter_fields.get("payer_id");
        let payer_name = encounter_fields.get("payer_name");
        // Claim filing indicator (SBR09) - filter empty strings to allow DB default
        let claim_filing_indicator = encounter_fields.get("claim_filing_indicator_code")
            .filter(|s| !s.is_empty());
        debug!("[ENCOUNTER] claim_filing_indicator_code: raw={:?}, filtered={:?}",
            encounter_fields.get("claim_filing_indicator_code"), claim_filing_indicator);
        // Truncate place_of_service to 2 chars (VARCHAR(2) in DB)
        let place_of_service_str = encounter_fields.get("place_of_service_code");
        let place_of_service = place_of_service_str.as_ref().map(|s| if s.len() > 2 { &s[..2] } else { s.as_str() });
        let medical_record_number = encounter_fields.get("medical_record_number");

        // Patient fields (when patient is different from subscriber)
        let patient_last_name = encounter_fields.get("patient_last_name").filter(|s| !s.is_empty());
        let patient_first_name = encounter_fields.get("patient_first_name").filter(|s| !s.is_empty());
        let patient_middle_name = encounter_fields.get("patient_middle_name").filter(|s| !s.is_empty());
        let patient_name_suffix = encounter_fields.get("patient_name_suffix").filter(|s| !s.is_empty());
        let patient_dob_str = encounter_fields.get("patient_date_of_birth").filter(|s| !s.is_empty());
        let patient_dob = patient_dob_str.as_ref().and_then(|s| chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d").ok());
        // Truncate patient_gender to 1 char (CHAR(1) in DB)
        let patient_gender_str = encounter_fields.get("patient_gender").filter(|s| !s.is_empty());
        let patient_gender = patient_gender_str.as_ref().map(|s| if s.len() > 1 { &s[..1] } else { s.as_str() });
        let patient_address_line1 = encounter_fields.get("patient_address_line1").filter(|s| !s.is_empty());
        let patient_address_line2 = encounter_fields.get("patient_address_line2").filter(|s| !s.is_empty());
        let patient_city = encounter_fields.get("patient_city").filter(|s| !s.is_empty());
        // Truncate patient_state to 2 chars (CHAR(2) in DB)
        let patient_state_str = encounter_fields.get("patient_state").filter(|s| !s.is_empty());
        let patient_state = patient_state_str.as_ref().map(|s| if s.len() > 2 { &s[..2] } else { s.as_str() });
        let patient_postal_code = encounter_fields.get("patient_postal_code").filter(|s| !s.is_empty());
        // Truncate patient_relationship_code to 3 chars (VARCHAR(3) in DB after migration 061)
        let patient_relationship_code_str = encounter_fields.get("patient_relationship_code").filter(|s| !s.is_empty());
        let patient_relationship_code = patient_relationship_code_str.as_ref().map(|s| if s.len() > 3 { &s[..3] } else { s.as_str() });

        // Extract provider NPIs and names (with empty string filtering)
        // NPI must be exactly 10 digits - reject invalid values
        let rendering_provider_npi = encounter_fields.get("rendering_provider_npi")
            .filter(|s| !s.is_empty())
            .filter(|s| s.len() == 10 && s.chars().all(|c| c.is_ascii_digit()));
        let rendering_provider_last_name = encounter_fields.get("rendering_provider_last_name").filter(|s| !s.is_empty());
        let rendering_provider_first_name = encounter_fields.get("rendering_provider_first_name").filter(|s| !s.is_empty());
        let rendering_provider_taxonomy = encounter_fields.get("rendering_provider_taxonomy").filter(|s| !s.is_empty());

        let referring_provider_npi = encounter_fields.get("referring_provider_npi")
            .filter(|s| !s.is_empty())
            .filter(|s| s.len() == 10 && s.chars().all(|c| c.is_ascii_digit()));
        let referring_provider_last_name = encounter_fields.get("referring_provider_last_name").filter(|s| !s.is_empty());
        let referring_provider_first_name = encounter_fields.get("referring_provider_first_name").filter(|s| !s.is_empty());

        let supervising_provider_npi = encounter_fields.get("supervising_provider_npi")
            .filter(|s| !s.is_empty())
            .filter(|s| s.len() == 10 && s.chars().all(|c| c.is_ascii_digit()));
        let supervising_provider_last_name = encounter_fields.get("supervising_provider_last_name").filter(|s| !s.is_empty());
        let supervising_provider_first_name = encounter_fields.get("supervising_provider_first_name").filter(|s| !s.is_empty());

        let billing_provider_npi = encounter_fields.get("billing_provider_npi")
            .filter(|s| !s.is_empty())
            .filter(|s| s.len() == 10 && s.chars().all(|c| c.is_ascii_digit()));
        let billing_provider_name = encounter_fields.get("billing_provider_name").filter(|s| !s.is_empty());
        let billing_provider_tax_id = encounter_fields.get("billing_provider_tax_id").filter(|s| !s.is_empty());
        let billing_provider_address_line1 = encounter_fields.get("billing_provider_address_line1").filter(|s| !s.is_empty());
        let billing_provider_city = encounter_fields.get("billing_provider_city").filter(|s| !s.is_empty());
        let billing_provider_state = encounter_fields.get("billing_provider_state").filter(|s| !s.is_empty());
        let billing_provider_postal_code = encounter_fields.get("billing_provider_postal_code").filter(|s| !s.is_empty());

        // Construct provider names from first/last
        let rendering_provider_name = match (rendering_provider_first_name.as_ref(), rendering_provider_last_name.as_ref()) {
            (Some(first), Some(last)) => Some(format!("{}, {}", last, first)),
            (None, Some(last)) => Some(last.to_string()),
            _ => None,
        };
        let referring_provider_name = match (referring_provider_first_name.as_ref(), referring_provider_last_name.as_ref()) {
            (Some(first), Some(last)) => Some(format!("{}, {}", last, first)),
            (None, Some(last)) => Some(last.to_string()),
            _ => None,
        };
        let supervising_provider_name = match (supervising_provider_first_name.as_ref(), supervising_provider_last_name.as_ref()) {
            (Some(first), Some(last)) => Some(format!("{}, {}", last, first)),
            (None, Some(last)) => Some(last.to_string()),
            _ => None,
        };

        // Service Facility information (Loop 2310C - NM1*77, N3, N4)
        // NPI must be exactly 10 digits - reject invalid values like facility codes
        let service_facility_npi = encounter_fields.get("service_facility_npi")
            .filter(|s| !s.is_empty())
            .filter(|s| s.len() == 10 && s.chars().all(|c| c.is_ascii_digit()));
        let service_facility_name = encounter_fields.get("service_facility_name").filter(|s| !s.is_empty());
        let service_facility_address_line1 = encounter_fields.get("service_facility_address_line1").filter(|s| !s.is_empty());
        let service_facility_address_line2 = encounter_fields.get("service_facility_address_line2").filter(|s| !s.is_empty());
        let service_facility_city = encounter_fields.get("service_facility_city").filter(|s| !s.is_empty());
        let service_facility_state = encounter_fields.get("service_facility_state").filter(|s| !s.is_empty());
        let service_facility_postal_code = encounter_fields.get("service_facility_postal_code").filter(|s| !s.is_empty());

        // Log service facility fields for debugging
        debug!("Service facility from encounter_fields: npi={:?}, name={:?}, addr1={:?}, city={:?}, state={:?}",
            service_facility_npi, service_facility_name, service_facility_address_line1,
            service_facility_city, service_facility_state);

        // Billing date from BHT segment (transaction creation date)
        let billing_date_str = encounter_fields.get("billing_date").filter(|s| !s.is_empty());
        let billing_date = billing_date_str.as_ref().and_then(|s| chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d").ok());

        // Additional provider taxonomy codes
        let referring_provider_taxonomy = encounter_fields.get("referring_provider_taxonomy").filter(|s| !s.is_empty());
        let supervising_provider_taxonomy = encounter_fields.get("supervising_provider_taxonomy").filter(|s| !s.is_empty());

        // Other payer information (COB)
        let other_payer_id = encounter_fields.get("other_payer_id").filter(|s| !s.is_empty());
        let other_payer_name = encounter_fields.get("other_payer_name").filter(|s| !s.is_empty());
        let other_payer_paid_amount = encounter_fields.get("other_payer_paid_amount")
            .filter(|s| !s.is_empty())
            .and_then(|s| s.parse::<rust_decimal::Decimal>().ok());

        // Condition codes - convert comma-separated string to JSON array
        let condition_codes: Option<serde_json::Value> = encounter_fields.get("condition_codes")
            .filter(|s| !s.is_empty())
            .map(|s| {
                let codes: Vec<&str> = s.split(',').map(|c| c.trim()).collect();
                serde_json::json!(codes)
            });

        // Patient responsibility amount
        let patient_responsibility_amount = encounter_fields.get("patient_responsibility_amount")
            .filter(|s| !s.is_empty())
            .and_then(|s| s.parse::<rust_decimal::Decimal>().ok());

        // Claim indicator fields from CLM segment
        // Truncate to 1 char (CHAR(1) in DB)
        let signature_indicator_str = encounter_fields.get("signature_indicator").filter(|s| !s.is_empty());
        let signature_indicator = signature_indicator_str.as_ref().map(|s| if s.len() > 1 { &s[..1] } else { s.as_str() });
        let assignment_indicator_str = encounter_fields.get("assignment_indicator").filter(|s| !s.is_empty());
        let assignment_indicator = assignment_indicator_str.as_ref().map(|s| if s.len() > 1 { &s[..1] } else { s.as_str() });
        let benefits_assignment_indicator_str = encounter_fields.get("benefits_assignment_indicator").filter(|s| !s.is_empty());
        let benefits_assignment_indicator = benefits_assignment_indicator_str.as_ref().map(|s| if s.len() > 1 { &s[..1] } else { s.as_str() });
        let release_of_information_code_str = encounter_fields.get("release_of_information_code").filter(|s| !s.is_empty());
        let release_of_information_code = release_of_information_code_str.as_ref().map(|s| if s.len() > 1 { &s[..1] } else { s.as_str() });
        let patient_signature_code_str = encounter_fields.get("patient_signature_code").filter(|s| !s.is_empty());
        let patient_signature_code = patient_signature_code_str.as_ref().map(|s| if s.len() > 1 { &s[..1] } else { s.as_str() });


        // Additional claim reference fields
        // Truncate delay_reason_code to 2 chars (VARCHAR(2) in DB)
        let delay_reason_code_str = encounter_fields.get("delay_reason_code").filter(|s| !s.is_empty());
        let delay_reason_code = delay_reason_code_str.as_ref().map(|s| if s.len() > 2 { &s[..2] } else { s.as_str() });
        // Truncate special_program_code to 3 chars (VARCHAR(3) in DB)
        let special_program_code_str = encounter_fields.get("special_program_code").filter(|s| !s.is_empty());
        let special_program_code = special_program_code_str.as_ref().map(|s| if s.len() > 3 { &s[..3] } else { s.as_str() });
        // Truncate service_authorization_code to 50 chars (VARCHAR(50) in DB)
        let service_authorization_code_str = encounter_fields.get("service_authorization_code").filter(|s| !s.is_empty());
        let service_authorization_code = service_authorization_code_str.as_ref().map(|s| if s.len() > 50 { &s[..50] } else { s.as_str() });

        // Patient amount paid (numeric)
        let patient_amount_paid = encounter_fields.get("patient_amount_paid")
            .filter(|s| !s.is_empty())
            .and_then(|s| s.parse::<rust_decimal::Decimal>().ok());

        // Submitter and transaction info
        // Truncate submitter_name to 255 chars (VARCHAR(255) in DB)
        let submitter_name_str = encounter_fields.get("submitter_name").filter(|s| !s.is_empty());
        let submitter_name = submitter_name_str.as_ref().map(|s| if s.len() > 255 { &s[..255] } else { s.as_str() });
        // Truncate transaction_set_control_number to 9 chars (VARCHAR(9) in DB)
        let transaction_set_control_number_str = encounter_fields.get("transaction_set_control_number").filter(|s| !s.is_empty());
        let transaction_set_control_number = transaction_set_control_number_str.as_ref().map(|s| if s.len() > 9 { &s[..9] } else { s.as_str() });

        // Onset of illness date
        let onset_of_illness_date = encounter_fields.get("onset_of_illness_date")
            .filter(|s| !s.is_empty())
            .and_then(|s| chrono::NaiveDate::parse_from_str(&s, "%Y-%m-%d").ok());

        // Date fields
        let initial_treatment_date = encounter_fields.get("initial_treatment_date")
            .filter(|s| !s.is_empty())
            .and_then(|s| chrono::NaiveDate::parse_from_str(&s, "%Y-%m-%d").ok());
        let last_seen_date = encounter_fields.get("last_seen_date")
            .filter(|s| !s.is_empty())
            .and_then(|s| chrono::NaiveDate::parse_from_str(&s, "%Y-%m-%d").ok());
        let acute_manifestation_date = encounter_fields.get("acute_manifestation_date")
            .filter(|s| !s.is_empty())
            .and_then(|s| chrono::NaiveDate::parse_from_str(&s, "%Y-%m-%d").ok());
        let accident_date = encounter_fields.get("accident_date")
            .filter(|s| !s.is_empty())
            .and_then(|s| chrono::NaiveDate::parse_from_str(&s, "%Y-%m-%d").ok());
        let last_xray_date = encounter_fields.get("last_xray_date")
            .filter(|s| !s.is_empty())
            .and_then(|s| chrono::NaiveDate::parse_from_str(&s, "%Y-%m-%d").ok());
        let disability_from_date = encounter_fields.get("disability_from_date")
            .filter(|s| !s.is_empty())
            .and_then(|s| chrono::NaiveDate::parse_from_str(&s, "%Y-%m-%d").ok());
        let disability_to_date = encounter_fields.get("disability_to_date")
            .filter(|s| !s.is_empty())
            .and_then(|s| chrono::NaiveDate::parse_from_str(&s, "%Y-%m-%d").ok());
        let last_worked_date = encounter_fields.get("last_worked_date")
            .filter(|s| !s.is_empty())
            .and_then(|s| chrono::NaiveDate::parse_from_str(&s, "%Y-%m-%d").ok());
        let authorized_return_to_work_date = encounter_fields.get("authorized_return_to_work_date")
            .filter(|s| !s.is_empty())
            .and_then(|s| chrono::NaiveDate::parse_from_str(&s, "%Y-%m-%d").ok());
        let admission_date = encounter_fields.get("admission_date")
            .filter(|s| !s.is_empty())
            .and_then(|s| chrono::NaiveDate::parse_from_str(&s, "%Y-%m-%d").ok());
        let discharge_date = encounter_fields.get("discharge_date")
            .filter(|s| !s.is_empty())
            .and_then(|s| chrono::NaiveDate::parse_from_str(&s, "%Y-%m-%d").ok());

        // Ensure providers exist in claims.provider table and get their provider_ids
        // Provider errors are handled internally using savepoints - claims proceed with NULL provider_id on error
        //
        // NOTE: Advisory locks were removed - they caused massive failures when multiple workers
        // tried to process claims with the same provider NPI. The ensure_provider_exists function
        // uses INSERT ON CONFLICT DO NOTHING which is safe for concurrent access.

        let rendering_provider_id = if let Some(ref npi) = rendering_provider_npi {
            self.ensure_provider_exists(
                tx,
                npi,
                "Rendering",
                rendering_provider_last_name.as_deref(),
                rendering_provider_first_name.as_deref(),
                None,
                rendering_provider_taxonomy.as_deref(),
                Some(organization_id),
            ).await.unwrap_or_else(|e| {
                warn!("Unexpected error ensuring rendering provider NPI={}: {:?}", npi, e);
                None
            })
        } else {
            None
        };

        let referring_provider_id = if let Some(ref npi) = referring_provider_npi {
            self.ensure_provider_exists(
                tx,
                npi,
                "Referring",
                referring_provider_last_name.as_deref(),
                referring_provider_first_name.as_deref(),
                None,
                None,
                Some(organization_id),
            ).await.unwrap_or_else(|e| {
                warn!("Unexpected error ensuring referring provider NPI={}: {:?}", npi, e);
                None
            })
        } else {
            None
        };

        let supervising_provider_id = if let Some(ref npi) = supervising_provider_npi {
            self.ensure_provider_exists(
                tx,
                npi,
                "Supervising",
                supervising_provider_last_name.as_deref(),
                supervising_provider_first_name.as_deref(),
                None,
                None,
                Some(organization_id),
            ).await.unwrap_or_else(|e| {
                warn!("Unexpected error ensuring supervising provider NPI={}: {:?}", npi, e);
                None
            })
        } else {
            None
        };

        let billing_provider_id = if let Some(ref npi) = billing_provider_npi {
            // For billing provider, we may only have organization name (not first/last)
            // Split billing_provider_name into last/first if it contains a comma
            let (last, first) = if let Some(ref name) = billing_provider_name {
                if name.contains(',') {
                    let parts: Vec<&str> = name.splitn(2, ',').collect();
                    (Some(parts[0].trim()), parts.get(1).map(|s| s.trim()))
                } else {
                    (Some(name.as_str()), None)
                }
            } else {
                (None, None)
            };

            self.ensure_provider_exists(
                tx,
                npi,
                "Billing",
                last,
                first,
                None,
                None,
                Some(organization_id),
            ).await.unwrap_or_else(|e| {
                warn!("Unexpected error ensuring billing provider NPI={}: {:?}", npi, e);
                None
            })
        } else {
            None
        };

        // Insert encounter and get generated ID (ONE record for all service lines)
        // Note: Every claim is imported as a new encounter - no duplicate checking
        let encounter_id: i64 = sqlx::query_scalar(
            r#"
            INSERT INTO claims.encounter (
                facility_id,
                organization_id,
                region_id,
                submitter_id,
                patient_control_number,
                subscriber_id,
                subscriber_last_name,
                subscriber_first_name,
                subscriber_middle_name,
                subscriber_name_suffix,
                subscriber_gender,
                subscriber_birth_date,
                subscriber_address_line1,
                subscriber_address_line2,
                subscriber_city,
                subscriber_state,
                subscriber_postal_code,
                subscriber_country,
                date_of_service_from,
                date_of_service_to,
                total_claim_charge_amount,
                payer_id,
                payer_name,
                payer_responsibility_code,
                claim_filing_indicator,
                place_of_service_code,
                medical_record_number,
                claim_number,
                paperwork_report_type,
                paperwork_transmission_code,
                paperwork_control_number,
                rendering_provider_id,
                rendering_provider_npi,
                rendering_provider_name,
                rendering_provider_taxonomy,
                referring_provider_id,
                referring_provider_npi,
                referring_provider_name,
                referring_provider_taxonomy,
                supervising_provider_id,
                supervising_provider_npi,
                supervising_provider_name,
                supervising_provider_taxonomy,
                billing_provider_id,
                billing_provider_npi,
                billing_provider_tax_id,
                billing_provider_name,
                billing_provider_address_line1,
                billing_provider_city,
                billing_provider_state,
                billing_provider_postal_code,
                claim_status,
                patient_last_name,
                patient_first_name,
                patient_middle_name,
                patient_name_suffix,
                patient_date_of_birth,
                patient_gender,
                patient_address_line1,
                patient_address_line2,
                patient_city,
                patient_state,
                patient_postal_code,
                patient_relationship_code,
                service_facility_npi,
                service_facility_name,
                service_facility_address_line1,
                service_facility_address_line2,
                service_facility_city,
                service_facility_state,
                service_facility_postal_code,
                billing_date,
                other_payer_id,
                other_payer_name,
                other_payer_paid_amount,
                condition_codes,
                patient_responsibility_amount,
                initial_treatment_date,
                last_seen_date,
                acute_manifestation_date,
                accident_date,
                last_xray_date,
                disability_from_date,
                disability_to_date,
                last_worked_date,
                authorized_return_to_work_date,
                admission_date,
                discharge_date,
                signature_indicator,
                assignment_indicator,
                benefits_assignment_indicator,
                release_of_information_code,
                patient_signature_code,
                delay_reason_code,
                special_program_code,
                service_authorization_code,
                patient_amount_paid,
                submitter_name,
                transaction_set_control_number,
                onset_of_illness_date
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20,
                    $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40,
                    $41, $42, $43, $44, $45, $46, $47, $48, $49, $50, $51, $52, $53, $54, $55, $56, $57, $58, $59, $60,
                    $61, $62, $63, $64, $65, $66, $67, $68, $69, $70, $71, $72, $73, $74, $75, $76, $77, $78, $79, $80,
                    $81, $82, $83, $84, $85, $86, $87, $88, $89, $90, $91, $92, $93, $94, $95, $96, $97, $98, $99, $100)
            RETURNING encounter_id
            "#
        )
        .bind(facility_id)                              // $1
        .bind(organization_id)                          // $2
        .bind(region_id)                                // $3
        .bind(&submitter_id)                            // $4
        .bind(&patient_control_number)                  // $5
        .bind(&subscriber_id)                           // $6
        .bind(&subscriber_last_name)                    // $7
        .bind(&subscriber_first_name)                   // $8
        .bind(subscriber_middle_name.as_deref())        // $9
        .bind(subscriber_name_suffix.as_deref())        // $10
        .bind(subscriber_gender)                        // $11
        .bind(subscriber_dob)                           // $12
        .bind(subscriber_address_line1.as_deref())      // $13
        .bind(subscriber_address_line2.as_deref())      // $14
        .bind(subscriber_city.as_deref())               // $15
        .bind(subscriber_state)                         // $16
        .bind(subscriber_postal_code.as_deref())        // $17
        .bind(subscriber_country)                       // $18
        .bind(dos_from)                                 // $19
        .bind(dos_from)                                 // $20 date_of_service_to same as from for now
        .bind(total_claim_charge)                       // $21
        .bind(payer_id.as_deref())                      // $22
        .bind(payer_name.as_deref())                    // $23
        .bind(payer_responsibility_code)                // $24
        .bind(claim_filing_indicator.as_deref())        // $25
        .bind(place_of_service)                         // $26
        .bind(medical_record_number.as_deref())         // $27
        .bind(claim_number.as_deref())                  // $28
        .bind(paperwork_report_type.as_deref())         // $29
        .bind(paperwork_transmission_code.as_deref())   // $30
        .bind(paperwork_control_number.as_deref())      // $31
        .bind(rendering_provider_id)                    // $32
        .bind(rendering_provider_npi.as_deref())        // $33
        .bind(rendering_provider_name.as_deref())       // $34
        .bind(rendering_provider_taxonomy.as_deref())   // $35
        .bind(referring_provider_id)                    // $36
        .bind(referring_provider_npi.as_deref())        // $37
        .bind(referring_provider_name.as_deref())       // $38
        .bind(referring_provider_taxonomy.as_deref())   // $39
        .bind(supervising_provider_id)                  // $40
        .bind(supervising_provider_npi.as_deref())      // $41
        .bind(supervising_provider_name.as_deref())     // $42
        .bind(supervising_provider_taxonomy.as_deref()) // $43
        .bind(billing_provider_id)                      // $44
        .bind(billing_provider_npi.as_deref())          // $45
        .bind(billing_provider_tax_id.as_deref())       // $46
        .bind(billing_provider_name.as_deref())         // $47
        .bind(billing_provider_address_line1.as_deref()) // $48
        .bind(billing_provider_city.as_deref())         // $49
        .bind(billing_provider_state.as_deref())        // $50
        .bind(billing_provider_postal_code.as_deref())  // $51
        .bind("NEW")                                    // $52 claim_status
        .bind(patient_last_name.as_deref())             // $53
        .bind(patient_first_name.as_deref())            // $54
        .bind(patient_middle_name.as_deref())           // $55
        .bind(patient_name_suffix.as_deref())           // $56
        .bind(patient_dob)                              // $57
        .bind(patient_gender)                           // $58
        .bind(patient_address_line1.as_deref())         // $59
        .bind(patient_address_line2.as_deref())         // $60
        .bind(patient_city.as_deref())                  // $61
        .bind(patient_state)                            // $62
        .bind(patient_postal_code.as_deref())           // $63
        .bind(patient_relationship_code)                // $64
        .bind(service_facility_npi.as_deref())          // $65
        .bind(service_facility_name.as_deref())         // $66
        .bind(service_facility_address_line1.as_deref()) // $67
        .bind(service_facility_address_line2.as_deref()) // $68
        .bind(service_facility_city.as_deref())         // $69
        .bind(service_facility_state.as_deref())        // $70
        .bind(service_facility_postal_code.as_deref())  // $71
        .bind(billing_date)                             // $72
        .bind(other_payer_id.as_deref())                // $73
        .bind(other_payer_name.as_deref())              // $74
        .bind(other_payer_paid_amount)                  // $75
        .bind(condition_codes.as_ref())                 // $76
        .bind(patient_responsibility_amount)            // $77
        .bind(initial_treatment_date)                   // $78
        .bind(last_seen_date)                           // $79
        .bind(acute_manifestation_date)                 // $80
        .bind(accident_date)                            // $81
        .bind(last_xray_date)                           // $82
        .bind(disability_from_date)                     // $83
        .bind(disability_to_date)                       // $84
        .bind(last_worked_date)                         // $85
        .bind(authorized_return_to_work_date)           // $86
        .bind(admission_date)                           // $87
        .bind(discharge_date)                           // $88
        .bind(signature_indicator)                      // $89
        .bind(assignment_indicator)                     // $90
        .bind(benefits_assignment_indicator)            // $91
        .bind(release_of_information_code)              // $92
        .bind(patient_signature_code)                   // $93
        .bind(delay_reason_code)                        // $94
        .bind(special_program_code)                     // $95
        .bind(service_authorization_code)               // $96
        .bind(patient_amount_paid)                      // $97
        .bind(submitter_name)                           // $98
        .bind(transaction_set_control_number)           // $99
        .bind(onset_of_illness_date)                    // $100
        .fetch_one(&mut **tx)
        .await
        .map_err(|e| {
            error!("Database error inserting encounter: {:?}", e);
            error!("  patient_control_number={}, facility_id={:?}, organization_id={}",
                patient_control_number, facility_id, organization_id);
            e
        })
        .context("Failed to insert encounter")?;

        // IMPORTANT: Import diagnoses FIRST so they exist when service line pointers reference them
        self.import_diagnoses(tx, encounter_id, first_line).await?;

        // Insert all service lines for this encounter
        // For EDI files: one raw_claim contains ALL service lines (service_line_1_*, service_line_2_*, etc.)
        // For CSV files: each raw_claim contains ONE service line (always service_line_1_*)
        // PERFORMANCE: Collect service line contexts for rule execution (avoids re-querying DB)
        let mut service_line_contexts: Vec<ServiceLineRuleContext> = Vec::new();
        let mut line_number = 1;
        for raw_claim in service_lines {
            // Check how many service lines are in this raw_claim's JSONB
            // PHASE 4 FIX: Use new function that works directly with JsonValue (no clone)
            let num_service_lines = raw_claim.service_line_fields.as_ref()
                .map(|v| Self::count_service_lines_in_json_value(v))
                .unwrap_or(0);

            if num_service_lines == 0 {
                // No service lines found, skip this raw_claim
                warn!("No service lines found in raw_claim {}", raw_claim.raw_claim_id);
                continue;
            }

            // Import each service line from this raw_claim
            for sl_idx in 1..=num_service_lines {
                let prefix = format!("service_line_{}_", sl_idx);
                let sl_ctx = self.import_service_line(tx, encounter_id, organization_id, raw_claim, line_number, &prefix).await?;
                service_line_contexts.push(sl_ctx);
                line_number += 1;
            }
        }

        // PERFORMANCE OPTIMIZATION: Collect modifiers from service_line_contexts (already in memory)
        // instead of expensive SELECT DISTINCT ... LATERAL query against service_line table
        let all_modifiers: std::collections::BTreeSet<String> = service_line_contexts.iter()
            .flat_map(|ctx| ctx.modifiers.iter().cloned())
            .filter(|m| !m.is_empty())
            .collect();

        // Insert aggregated modifiers (skips the expensive SELECT query)
        self.insert_encounter_procedure_modifiers_fast(tx, encounter_id, &all_modifiers).await?;

        // Extract diagnosis codes from first service line's raw_claim for rule execution
        // PERFORMANCE: Reuse parsed data instead of re-querying DB
        let diagnosis_codes: Vec<String> = first_line.diagnosis_fields.as_ref()
            .and_then(|df| serde_json::from_value::<HashMap<String, Vec<String>>>(df.clone()).ok())
            .map(|fields| {
                let mut codes: Vec<(usize, String)> = Vec::new();
                for (field_name, values) in &fields {
                    if field_name.starts_with("diagnosis_code_") {
                        if let Some(seq_str) = field_name.strip_prefix("diagnosis_code_") {
                            if let Ok(seq) = seq_str.parse::<usize>() {
                                for code in values {
                                    codes.push((seq, code.clone()));
                                }
                            }
                        }
                    } else if field_name == "diagnosis_code" {
                        for (idx, code) in values.iter().enumerate() {
                            codes.push((idx + 1, code.clone()));
                        }
                    }
                }
                codes.sort_by_key(|(seq, _)| *seq);
                codes.into_iter().map(|(_, code)| code).collect()
            })
            .unwrap_or_default();

        // Execute rules engine for this encounter (OPTIMIZED: no DB re-queries)
        // PERFORMANCE: When defer_rules is enabled, skip inline execution for maximum throughput
        // Deferred rules will be processed by a background worker using the rules_processing_queue
        let _flags_created = if self.defer_rules {
            // Enqueue encounter for background rule processing
            sqlx::query(
                "SELECT staging.enqueue_for_rules_processing($1, $2, NULL, 5)"
            )
            .bind(encounter_id)
            .bind(organization_id)
            .execute(&mut **tx)
            .await
            .ok(); // Fire-and-forget, don't fail import if queue fails
            debug!("Queued rules execution for encounter_id={}", encounter_id);
            0
        } else {
            self.execute_rules_for_service_lines(
                tx, encounter_id, organization_id, &service_line_contexts, &diagnosis_codes
            ).await?
        };

        // Insert billing payer into encounter_payer table (first SBR - the one being billed)
        self.insert_encounter_payer(
            tx,
            encounter_id,
            payer_responsibility_code,
            payer_id.as_deref(),
            payer_name.as_deref(),
            claim_filing_indicator.as_deref(),
            true, // is_billing_payer
            None, // paid_amount (billing payer hasn't paid yet)
            None, // claim_control_number
            billing_provider_id,
        ).await?;

        // Insert COB payers (from Loop 2320) into encounter_payer table
        self.import_encounter_payers_from_cob(tx, encounter_id, encounter_fields.inner(), billing_provider_id).await?;

        Ok(encounter_id)
    }

    /// Process an encounter with thread-safe facility cache (for parallel processing)
    /// This is a wrapper around process_encounter_with_service_lines that handles
    /// the Arc<RwLock<HashMap>> cache instead of a mutable HashMap reference.
    async fn process_encounter_with_service_lines_parallel(
        &self,
        tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
        service_lines: &[RawClaim],
        facility_cache: &Arc<tokio::sync::RwLock<HashMap<String, (Option<i64>, i64, Option<i64>)>>>,
    ) -> Result<i64> {
        if service_lines.is_empty() {
            return Err(anyhow::anyhow!("No service lines provided"));
        }

        // Use first service line for encounter-level data
        let first_line = &service_lines[0];

        // Deserialize encounter fields from first line
        let encounter_fields = EncounterFieldsWrapper::new(first_line.encounter_fields.clone())
            .context("Failed to deserialize encounter_fields")?;

        // Pre-warm provider cache
        self.prewarm_provider_cache(tx, &encounter_fields, service_lines).await?;

        // Extract facility_code
        let facility_code = encounter_fields.get("facility_code")
            .or_else(|| encounter_fields.get("facility_npi"))
            .context("Missing facility_code or facility_npi")?;

        // Thread-safe cache lookup/insert
        let (facility_id, organization_id, region_id) = {
            // First try read lock
            let cache_read = facility_cache.read().await;
            if let Some(cached) = cache_read.get(&facility_code) {
                *cached
            } else {
                // Drop read lock before acquiring write lock
                drop(cache_read);

                // Query database
                let facility = sqlx::query_as::<_, (Option<i64>, i64, Option<i64>)>(
                    r#"
                    SELECT facility_id, organization_id, region_id
                    FROM claims.facility
                    WHERE facility_code = $1 OR npi = $1
                    "#
                )
                .bind(&facility_code)
                .fetch_optional(&mut **tx)
                .await?;

                let facility_result = facility
                    .with_context(|| format!("Facility not found: {}", facility_code))?;

                // Insert into cache with write lock
                let mut cache_write = facility_cache.write().await;
                cache_write.insert(facility_code.clone(), facility_result);
                facility_result
            }
        };

        // Create a local mutable HashMap with the single facility we need
        // This allows us to reuse the original method's logic without duplicating code
        let mut local_cache: HashMap<String, (Option<i64>, i64, Option<i64>)> = HashMap::new();
        local_cache.insert(facility_code.clone(), (facility_id, organization_id, region_id));

        // Call the original method with the local cache (it will find the facility in cache)
        self.process_encounter_with_service_lines(tx, service_lines, &mut local_cache).await
    }

    /// Insert a payer record into encounter_payer table
    #[allow(clippy::too_many_arguments)]
    async fn insert_encounter_payer(
        &self,
        tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
        encounter_id: i64,
        payer_responsibility_code: &str,
        payer_id: Option<&str>,
        payer_name: Option<&str>,
        claim_filing_indicator: Option<&str>,
        is_billing_payer: bool,
        paid_amount: Option<rust_decimal::Decimal>,
        claim_control_number: Option<&str>,
        billing_provider_id: Option<i64>,
    ) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO claims.encounter_payer (
                encounter_id,
                payer_responsibility_code,
                payer_id,
                payer_name,
                claim_filing_indicator,
                is_billing_payer,
                paid_amount,
                claim_control_number,
                billing_provider_id,
                submitted_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, CURRENT_TIMESTAMP)
            "#
        )
        .bind(encounter_id)
        .bind(payer_responsibility_code)
        .bind(payer_id)
        .bind(payer_name)
        .bind(claim_filing_indicator)
        .bind(is_billing_payer)
        .bind(paid_amount)
        .bind(claim_control_number)
        .bind(billing_provider_id)
        .execute(&mut **tx)
        .await
        .context("Failed to insert encounter_payer record")?;

        debug!("[PAYER] Inserted encounter_payer: encounter_id={}, payer_resp={}, payer_id={:?}, is_billing={}, paid_amount={:?}",
            encounter_id, payer_responsibility_code, payer_id, is_billing_payer, paid_amount);

        Ok(())
    }

    /// Import COB payers from other_insurance JSON into encounter_payer table
    /// Optimized: Uses batch INSERT instead of individual inserts per payer (N+1 fix)
    async fn import_encounter_payers_from_cob(
        &self,
        tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
        encounter_id: i64,
        encounter_fields: &HashMap<String, JsonValue>,
        billing_provider_id: Option<i64>,
    ) -> Result<()> {
        // Check if other_insurance data exists in encounter_fields
        let other_insurance_value = match encounter_fields.get("other_insurance") {
            Some(value) => value,
            None => return Ok(()), // No COB data
        };

        // The other_insurance field is already a JsonValue array (not a string that needs parsing)
        let other_insurance_array: Vec<serde_json::Value> = match other_insurance_value {
            JsonValue::Array(arr) => arr.clone(),
            JsonValue::String(json_str) => {
                // Fallback: if it's a string, try parsing it
                match serde_json::from_str(json_str) {
                    Ok(arr) => arr,
                    Err(e) => {
                        warn!("[COB] Failed to parse other_insurance JSON string: {:?}", e);
                        return Ok(());
                    }
                }
            },
            _ => {
                warn!("[COB] other_insurance is not an array or string: {:?}", other_insurance_value);
                return Ok(());
            }
        };

        if other_insurance_array.is_empty() {
            return Ok(());
        }

        // Collect valid payer records for batch insert
        struct PayerRecord {
            payer_resp: String,
            payer_id: Option<String>,
            payer_name: Option<String>,
            claim_filing_indicator: Option<String>,
            paid_amount: Option<rust_decimal::Decimal>,
            claim_control_number: Option<String>,
        }

        let mut payers: Vec<PayerRecord> = Vec::with_capacity(other_insurance_array.len());

        for oi in &other_insurance_array {
            let payer_resp_seq = oi.get("payer_responsibility_sequence")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty());

            // Skip if no payer_responsibility_sequence (required field)
            let payer_resp = match payer_resp_seq {
                Some(p) => p.to_string(),
                None => {
                    warn!("[COB] Skipping COB payer with missing payer_responsibility_sequence");
                    continue;
                }
            };

            payers.push(PayerRecord {
                payer_resp,
                payer_id: oi.get("payer_id")
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty())
                    .map(|s| s.to_string()),
                payer_name: oi.get("payer_name")
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty())
                    .map(|s| s.to_string()),
                claim_filing_indicator: oi.get("claim_filing_indicator")
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty())
                    .map(|s| s.to_string()),
                paid_amount: oi.get("paid_amount")
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty())
                    .and_then(|s| s.parse().ok()),
                claim_control_number: oi.get("claim_control_number")
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty())
                    .map(|s| s.to_string()),
            });
        }

        if payers.is_empty() {
            return Ok(());
        }

        debug!("[COB] Batch inserting {} COB payers into encounter_payer for encounter_id={}",
            payers.len(), encounter_id);

        // Build batch INSERT with multiple VALUES tuples
        // Each payer needs 9 params: encounter_id, payer_resp, payer_id, payer_name,
        // claim_filing_indicator, is_billing_payer, paid_amount, claim_control_number, billing_provider_id
        let values_per_row = 9;
        let value_placeholders: Vec<String> = (0..payers.len())
            .map(|i| {
                let base = i * values_per_row;
                format!(
                    "(${}, ${}, ${}, ${}, ${}, ${}, ${}, ${}, ${})",
                    base + 1, base + 2, base + 3, base + 4, base + 5,
                    base + 6, base + 7, base + 8, base + 9
                )
            })
            .collect();

        let query = format!(
            r#"
            INSERT INTO claims.encounter_payer (
                encounter_id,
                payer_responsibility_code,
                payer_id,
                payer_name,
                claim_filing_indicator,
                is_billing_payer,
                paid_amount,
                claim_control_number,
                billing_provider_id
            )
            VALUES {}
            "#,
            value_placeholders.join(", ")
        );

        let mut query_builder = sqlx::query(&query);

        for payer in &payers {
            query_builder = query_builder
                .bind(encounter_id)
                .bind(&payer.payer_resp)
                .bind(&payer.payer_id)
                .bind(&payer.payer_name)
                .bind(&payer.claim_filing_indicator)
                .bind(false) // is_billing_payer = false for COB payers
                .bind(payer.paid_amount)
                .bind(&payer.claim_control_number)
                .bind(billing_provider_id);
        }

        query_builder.execute(&mut **tx)
            .await
            .context("Failed to batch insert encounter_payer records")?;

        debug!("[COB] Batch inserted {} COB payers for encounter_id={}", payers.len(), encounter_id);

        Ok(())
    }

    /// Import other insurance records from encounter_fields JSON
    /// Optimized: Uses batch INSERT instead of individual inserts per record (N+1 fix)
    async fn import_other_insurance(
        &self,
        tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
        encounter_id: i64,
        encounter_fields: &HashMap<String, String>,
    ) -> Result<()> {
        // Check if other_insurance data exists in encounter_fields
        let other_insurance_json = match encounter_fields.get("other_insurance") {
            Some(json_str) => json_str,
            None => return Ok(()), // No COB data
        };

        // Parse the JSON array
        let other_insurance_array: Vec<serde_json::Value> = match serde_json::from_str(other_insurance_json) {
            Ok(arr) => arr,
            Err(e) => {
                warn!("[COB] Failed to parse other_insurance JSON: {:?}", e);
                return Ok(());
            }
        };

        if other_insurance_array.is_empty() {
            return Ok(());
        }

        // Collect valid records for batch insert
        struct OtherInsuranceRecord {
            payer_resp: String,
            individual_rel_code: Option<String>,
            group_policy_number: Option<String>,
            group_name: Option<String>,
            insurance_type_code: Option<String>,
            coordination_benefits_code: Option<String>,
            claim_filing_indicator: Option<String>,
            payer_id: Option<String>,
            payer_name: Option<String>,
            payer_address_line1: Option<String>,
            payer_address_line2: Option<String>,
            payer_city: Option<String>,
            payer_state: Option<String>,
            payer_postal_code: Option<String>,
            paid_amount: Option<rust_decimal::Decimal>,
            claim_control_number: Option<String>,
            benefits_assignment: Option<String>,
            release_of_info: Option<String>,
        }

        let mut records: Vec<OtherInsuranceRecord> = Vec::with_capacity(other_insurance_array.len());

        for oi in &other_insurance_array {
            let payer_resp_seq = oi.get("payer_responsibility_sequence")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty());

            // Skip if no payer_responsibility_sequence (required field)
            let payer_resp = match payer_resp_seq {
                Some(p) => p.to_string(),
                None => {
                    warn!("[COB] Skipping other_insurance record with missing payer_responsibility_sequence");
                    continue;
                }
            };

            records.push(OtherInsuranceRecord {
                payer_resp,
                individual_rel_code: oi.get("individual_relationship_code")
                    .and_then(|v| v.as_str()).filter(|s| !s.is_empty()).map(|s| s.to_string()),
                group_policy_number: oi.get("group_policy_number")
                    .and_then(|v| v.as_str()).filter(|s| !s.is_empty()).map(|s| s.to_string()),
                group_name: oi.get("group_name")
                    .and_then(|v| v.as_str()).filter(|s| !s.is_empty()).map(|s| s.to_string()),
                insurance_type_code: oi.get("insurance_type_code")
                    .and_then(|v| v.as_str()).filter(|s| !s.is_empty()).map(|s| s.to_string()),
                coordination_benefits_code: oi.get("coordination_benefits_code")
                    .and_then(|v| v.as_str()).filter(|s| !s.is_empty()).map(|s| s.to_string()),
                claim_filing_indicator: oi.get("claim_filing_indicator")
                    .and_then(|v| v.as_str()).filter(|s| !s.is_empty()).map(|s| s.to_string()),
                payer_id: oi.get("payer_id")
                    .and_then(|v| v.as_str()).filter(|s| !s.is_empty()).map(|s| s.to_string()),
                payer_name: oi.get("payer_name")
                    .and_then(|v| v.as_str()).filter(|s| !s.is_empty()).map(|s| s.to_string()),
                payer_address_line1: oi.get("payer_address_line1")
                    .and_then(|v| v.as_str()).filter(|s| !s.is_empty()).map(|s| s.to_string()),
                payer_address_line2: oi.get("payer_address_line2")
                    .and_then(|v| v.as_str()).filter(|s| !s.is_empty()).map(|s| s.to_string()),
                payer_city: oi.get("payer_city")
                    .and_then(|v| v.as_str()).filter(|s| !s.is_empty()).map(|s| s.to_string()),
                payer_state: oi.get("payer_state")
                    .and_then(|v| v.as_str()).filter(|s| !s.is_empty()).map(|s| s.to_string()),
                payer_postal_code: oi.get("payer_postal_code")
                    .and_then(|v| v.as_str()).filter(|s| !s.is_empty()).map(|s| s.to_string()),
                paid_amount: oi.get("paid_amount")
                    .and_then(|v| v.as_str()).filter(|s| !s.is_empty()).and_then(|s| s.parse().ok()),
                claim_control_number: oi.get("claim_control_number")
                    .and_then(|v| v.as_str()).filter(|s| !s.is_empty()).map(|s| s.to_string()),
                benefits_assignment: oi.get("benefits_assignment_certification")
                    .and_then(|v| v.as_str()).filter(|s| !s.is_empty()).map(|s| s.to_string()),
                release_of_info: oi.get("release_of_information_code")
                    .and_then(|v| v.as_str()).filter(|s| !s.is_empty()).map(|s| s.to_string()),
            });
        }

        if records.is_empty() {
            return Ok(());
        }

        debug!("[COB] Batch inserting {} other_insurance records for encounter_id={}",
            records.len(), encounter_id);

        // Build batch INSERT with multiple VALUES tuples (19 columns per row)
        let values_per_row = 19;
        let value_placeholders: Vec<String> = (0..records.len())
            .map(|i| {
                let base = i * values_per_row;
                format!(
                    "(${}, ${}, ${}, ${}, ${}, ${}, ${}, ${}, ${}, ${}, ${}, ${}, ${}, ${}, ${}, ${}, ${}, ${}, ${})",
                    base + 1, base + 2, base + 3, base + 4, base + 5, base + 6, base + 7,
                    base + 8, base + 9, base + 10, base + 11, base + 12, base + 13, base + 14,
                    base + 15, base + 16, base + 17, base + 18, base + 19
                )
            })
            .collect();

        let query = format!(
            r#"
            INSERT INTO claims.other_insurance (
                encounter_id,
                payer_responsibility_sequence,
                individual_relationship_code,
                group_policy_number,
                group_name,
                insurance_type_code,
                coordination_benefits_code,
                claim_filing_indicator,
                payer_id,
                payer_name,
                payer_address_line1,
                payer_address_line2,
                payer_city,
                payer_state,
                payer_postal_code,
                paid_amount,
                claim_control_number,
                benefits_assignment_certification,
                release_of_information_code
            )
            VALUES {}
            "#,
            value_placeholders.join(", ")
        );

        let mut query_builder = sqlx::query(&query);

        for rec in &records {
            query_builder = query_builder
                .bind(encounter_id)
                .bind(&rec.payer_resp)
                .bind(&rec.individual_rel_code)
                .bind(&rec.group_policy_number)
                .bind(&rec.group_name)
                .bind(&rec.insurance_type_code)
                .bind(&rec.coordination_benefits_code)
                .bind(&rec.claim_filing_indicator)
                .bind(&rec.payer_id)
                .bind(&rec.payer_name)
                .bind(&rec.payer_address_line1)
                .bind(&rec.payer_address_line2)
                .bind(&rec.payer_city)
                .bind(&rec.payer_state)
                .bind(&rec.payer_postal_code)
                .bind(rec.paid_amount)
                .bind(&rec.claim_control_number)
                .bind(&rec.benefits_assignment)
                .bind(&rec.release_of_info);
        }

        query_builder.execute(&mut **tx)
            .await
            .context("Failed to batch insert other_insurance records")?;

        debug!("[COB] Batch inserted {} other_insurance records for encounter_id={}",
            records.len(), encounter_id);

        Ok(())
    }

    /// Insert aggregated procedure modifiers from all service lines into encounter_procedure_modifier table
    /// Collects all unique modifiers, sorts them, and stores as comma-separated string
    async fn insert_encounter_procedure_modifiers(
        &self,
        tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
        encounter_id: i64,
    ) -> Result<()> {
        // Query all unique modifiers from service lines for this encounter
        let modifiers: Vec<String> = sqlx::query_scalar(
            r#"
            SELECT DISTINCT modifier
            FROM claims.service_line,
                 LATERAL (
                     VALUES
                         (procedure_modifier_1),
                         (procedure_modifier_2),
                         (procedure_modifier_3),
                         (procedure_modifier_4)
                 ) AS m(modifier)
            WHERE encounter_id = $1
              AND modifier IS NOT NULL
              AND modifier != ''
            ORDER BY modifier
            "#
        )
        .bind(encounter_id)
        .fetch_all(&mut **tx)
        .await
        .context("Failed to query service line modifiers")?;

        // Only insert if there are modifiers
        if modifiers.is_empty() {
            return Ok(());
        }

        // Join modifiers with commas (already sorted by query)
        let modifiers_csv = modifiers.join(",");

        // Truncate to 20 chars if necessary (VARCHAR(20))
        let modifiers_csv = if modifiers_csv.len() > 20 {
            &modifiers_csv[..20]
        } else {
            &modifiers_csv
        };

        // Insert into encounter_procedure_modifier table
        sqlx::query(
            r#"
            INSERT INTO claims.encounter_procedure_modifier (encounter_id, modifiers)
            VALUES ($1, $2)
            ON CONFLICT (encounter_id) DO UPDATE SET
                modifiers = EXCLUDED.modifiers,
                updated_at = CURRENT_TIMESTAMP
            "#
        )
        .bind(encounter_id)
        .bind(modifiers_csv)
        .execute(&mut **tx)
        .await
        .context("Failed to insert encounter_procedure_modifier")?;

        debug!("[MODIFIERS] Inserted encounter modifiers: encounter_id={}, modifiers={}",
            encounter_id, modifiers_csv);

        Ok(())
    }

    /// OPTIMIZED: Insert procedure modifiers from pre-collected data
    /// Avoids the expensive SELECT DISTINCT ... LATERAL query by using modifiers
    /// already collected during service line import.
    ///
    /// PERFORMANCE: Saves ~40-50ms per encounter by eliminating the SELECT query
    async fn insert_encounter_procedure_modifiers_fast(
        &self,
        tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
        encounter_id: i64,
        modifiers: &std::collections::BTreeSet<String>,
    ) -> Result<()> {
        // Only insert if there are modifiers
        if modifiers.is_empty() {
            return Ok(());
        }

        // BTreeSet is already sorted, so just join with commas
        let modifiers_csv: String = modifiers.iter().cloned().collect::<Vec<_>>().join(",");

        // Truncate to 20 chars if necessary (VARCHAR(20))
        let modifiers_csv = if modifiers_csv.len() > 20 {
            &modifiers_csv[..20]
        } else {
            &modifiers_csv
        };

        // Single INSERT (no SELECT needed - we already have the data!)
        sqlx::query(
            r#"
            INSERT INTO claims.encounter_procedure_modifier (encounter_id, modifiers)
            VALUES ($1, $2)
            ON CONFLICT (encounter_id) DO UPDATE SET
                modifiers = EXCLUDED.modifiers,
                updated_at = CURRENT_TIMESTAMP
            "#
        )
        .bind(encounter_id)
        .bind(modifiers_csv)
        .execute(&mut **tx)
        .await
        .context("Failed to insert encounter_procedure_modifier")?;

        debug!("[MODIFIERS] Inserted encounter modifiers (fast): encounter_id={}, modifiers={}",
            encounter_id, modifiers_csv);

        Ok(())
    }

    /// Process a single raw claim from staging.raw_claims
    /// @deprecated - Use process_encounter_with_service_lines instead
    async fn process_raw_claim(
        &self,
        tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
        raw_claim: &RawClaim,
        facility_cache: &mut HashMap<String, (Option<i64>, i64, Option<i64>)>,
    ) -> Result<i64> {
        // Deserialize JSONB fields
        let encounter_fields: HashMap<String, String> = serde_json::from_value(raw_claim.encounter_fields.clone())
            .context("Failed to deserialize encounter_fields")?;

        // Extract facility_code
        let facility_code = encounter_fields.get("facility_code")
            .or_else(|| encounter_fields.get("facility_npi"))
            .context("Missing facility_code or facility_npi")?;

        // Check cache first before querying database
        let (facility_id, organization_id, region_id) = if let Some(cached) = facility_cache.get(facility_code) {
            // Cache hit
            *cached
        } else {
            // Cache miss - query database and cache result
            let facility = sqlx::query_as::<_, (Option<i64>, i64, Option<i64>)>(
                r#"
                SELECT facility_id, organization_id, region_id
                FROM claims.facility
                WHERE facility_code = $1 OR npi = $1
                "#
            )
            .bind(facility_code)
            .fetch_optional(&mut **tx)
            .await?;

            let facility_result = facility
                .with_context(|| format!("Facility not found: {}", facility_code))?;

            // Store in cache
            facility_cache.insert(facility_code.clone(), facility_result);
            facility_result
        };

        // Generate encounter ID
        let encounter_id = 0i64; // TODO: Use RETURNING

        // Extract required encounter fields
        let patient_control_number = encounter_fields.get("patient_control_number")
            .context("Missing patient_control_number")?;
        let subscriber_last_name = encounter_fields.get("subscriber_last_name")
            .context("Missing subscriber_last_name")?;
        let subscriber_first_name = encounter_fields.get("subscriber_first_name")
            .context("Missing subscriber_first_name")?;
        let date_of_service_from = encounter_fields.get("date_of_service_from")
            .context("Missing date_of_service_from")?;
        let subscriber_id = encounter_fields.get("subscriber_id")
            .context("Missing subscriber_id")?;
        let subscriber_birth_date_str = encounter_fields.get("subscriber_birth_date")
            .filter(|s| !s.is_empty())
            .map(|s| s.as_str());

        // Optional fields with defaults
        let submitter_id = encounter_fields.get("submitter_id")
            .unwrap_or(facility_code);
        // Truncate payer_responsibility_code to 1 char (CHAR(1) in DB, must be 'P' or 'S')
        let payer_responsibility_code = encounter_fields.get("payer_responsibility_code")
            .map(|s| s.as_str())
            .map(|s| if s.len() > 1 { &s[..1] } else { s })
            .unwrap_or("P");

        // Parse service line fields if present
        let service_line_fields: Option<HashMap<String, String>> = raw_claim.service_line_fields.as_ref()
            .and_then(|v| serde_json::from_value(v.clone()).ok());

        // Use total_claim_charge_amount from CLM02 segment (authoritative value from 837 file)
        let total_claim_charge = encounter_fields.get("total_claim_charge_amount")
            .and_then(|s| s.parse::<rust_decimal::Decimal>().ok())
            .unwrap_or(rust_decimal::Decimal::ZERO);

        // Parse dates
        let dos_from = chrono::NaiveDate::parse_from_str(date_of_service_from, "%Y-%m-%d")
            .context("Invalid date format for date_of_service_from")?;
        let subscriber_dob = subscriber_birth_date_str
            .and_then(|s| chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d").ok());

        // Optional fields - subscriber demographics
        let payer_id = encounter_fields.get("payer_id").map(|s| s.as_str());
        let payer_name = encounter_fields.get("payer_name").map(|s| s.as_str());
        let place_of_service = encounter_fields.get("place_of_service_code").map(|s| s.as_str());
        let medical_record_number = encounter_fields.get("medical_record_number").map(|s| s.as_str());

        let subscriber_middle_name = encounter_fields.get("subscriber_middle_name").map(|s| s.as_str());
        let subscriber_name_suffix = encounter_fields.get("subscriber_name_suffix").map(|s| s.as_str());
        let subscriber_gender = encounter_fields.get("subscriber_gender").map(|s| s.as_str());
        let subscriber_address_line1 = encounter_fields.get("subscriber_address_line1").map(|s| s.as_str());
        let subscriber_address_line2 = encounter_fields.get("subscriber_address_line2").map(|s| s.as_str());
        let subscriber_city = encounter_fields.get("subscriber_city").map(|s| s.as_str());
        let subscriber_state = encounter_fields.get("subscriber_state").map(|s| s.as_str());
        let subscriber_postal_code = encounter_fields.get("subscriber_postal_code").map(|s| s.as_str());
        let subscriber_country = encounter_fields.get("subscriber_country").map(|s| s.as_str());

        // Provider information
        let rendering_provider_npi = encounter_fields.get("rendering_provider_npi")
            .map(|s| s.as_str())
            .filter(|s| !s.is_empty())
            .filter(|s| s.len() == 10 && s.chars().all(|c| c.is_ascii_digit()));
        let rendering_provider_name = encounter_fields.get("rendering_provider_name").map(|s| s.as_str()).filter(|s| !s.is_empty());
        let referring_provider_npi = encounter_fields.get("referring_provider_npi")
            .map(|s| s.as_str())
            .filter(|s| !s.is_empty())
            .filter(|s| s.len() == 10 && s.chars().all(|c| c.is_ascii_digit()));
        let referring_provider_name = encounter_fields.get("referring_provider_name").map(|s| s.as_str()).filter(|s| !s.is_empty());
        // NPI must be exactly 10 digits - reject invalid values like facility codes
        let service_facility_npi = encounter_fields.get("service_facility_npi")
            .map(|s| s.as_str())
            .filter(|s| !s.is_empty())
            .filter(|s| s.len() == 10 && s.chars().all(|c| c.is_ascii_digit()));
        let service_facility_name = encounter_fields.get("service_facility_name").map(|s| s.as_str()).filter(|s| !s.is_empty());
        let service_facility_address_line1 = encounter_fields.get("service_facility_address_line1").map(|s| s.as_str()).filter(|s| !s.is_empty());
        let service_facility_address_line2 = encounter_fields.get("service_facility_address_line2").map(|s| s.as_str()).filter(|s| !s.is_empty());
        let service_facility_city = encounter_fields.get("service_facility_city").map(|s| s.as_str()).filter(|s| !s.is_empty());
        let service_facility_state = encounter_fields.get("service_facility_state").map(|s| s.as_str()).filter(|s| !s.is_empty());
        let service_facility_postal_code = encounter_fields.get("service_facility_postal_code").map(|s| s.as_str()).filter(|s| !s.is_empty());

        // Debug: Log service facility fields from raw_claims
        debug!("Service facility from raw_claims: npi={:?}, name={:?}, addr1={:?}, city={:?}, state={:?}",
            service_facility_npi, service_facility_name, service_facility_address_line1,
            service_facility_city, service_facility_state);
        let supervising_provider_npi = encounter_fields.get("supervising_provider_npi")
            .map(|s| s.as_str())
            .filter(|s| !s.is_empty())
            .filter(|s| s.len() == 10 && s.chars().all(|c| c.is_ascii_digit()));
        let supervising_provider_name = encounter_fields.get("supervising_provider_name").map(|s| s.as_str()).filter(|s| !s.is_empty());
        let billing_provider_npi = encounter_fields.get("billing_provider_npi")
            .map(|s| s.as_str())
            .filter(|s| !s.is_empty())
            .filter(|s| s.len() == 10 && s.chars().all(|c| c.is_ascii_digit()));
        let billing_provider_name = encounter_fields.get("billing_provider_name").map(|s| s.as_str()).filter(|s| !s.is_empty());
        let billing_provider_tax_id = encounter_fields.get("billing_provider_tax_id").map(|s| s.as_str());
        let billing_provider_address_line1 = encounter_fields.get("billing_provider_address_line1").map(|s| s.as_str());
        let billing_provider_city = encounter_fields.get("billing_provider_city").map(|s| s.as_str());
        let billing_provider_state = encounter_fields.get("billing_provider_state").map(|s| s.as_str());
        let billing_provider_postal_code = encounter_fields.get("billing_provider_postal_code").map(|s| s.as_str());

        // Phase 2: Payer address information
        let payer_address_line1 = encounter_fields.get("payer_address_line1").map(|s| s.as_str());
        let payer_address_line2 = encounter_fields.get("payer_address_line2").map(|s| s.as_str());
        let payer_city = encounter_fields.get("payer_city").map(|s| s.as_str());
        let payer_state = encounter_fields.get("payer_state").map(|s| s.as_str());
        let payer_postal_code = encounter_fields.get("payer_postal_code").map(|s| s.as_str());

        // Phase 2: Claim supplemental information
        let transaction_set_control_number = encounter_fields.get("transaction_set_control_number").map(|s| s.as_str());
        let submitter_name = encounter_fields.get("submitter_name").map(|s| s.as_str());
        let claim_filing_indicator = encounter_fields.get("claim_filing_indicator_code").map(|s| s.as_str()).filter(|s| !s.is_empty());
        debug!("[ENCOUNTER] claim_filing_indicator_code from encounter_fields: raw={:?}, filtered={:?}",
            encounter_fields.get("claim_filing_indicator_code"),
            claim_filing_indicator);
        let claim_frequency_code = encounter_fields.get("claim_frequency_code").map(|s| s.as_str());
        let signature_indicator = encounter_fields.get("signature_indicator").map(|s| s.as_str());
        let assignment_indicator = encounter_fields.get("assignment_indicator").map(|s| s.as_str());
        let benefits_assignment_indicator = encounter_fields.get("benefits_assignment_indicator").map(|s| s.as_str());
        let release_of_information_code = encounter_fields.get("release_of_information_code").map(|s| s.as_str());
        let patient_signature_code = encounter_fields.get("patient_signature_code").map(|s| s.as_str());
        let delay_reason_code = encounter_fields.get("delay_reason_code").map(|s| s.as_str());
        let special_program_code = encounter_fields.get("special_program_code").map(|s| s.as_str());

        // Parse patient_amount_paid if present
        let patient_amount_paid = encounter_fields.get("patient_amount_paid")
            .and_then(|s| s.parse::<rust_decimal::Decimal>().ok());

        let service_authorization_code = encounter_fields.get("service_authorization_code").map(|s| s.as_str());

        // Phase 2: Basic COB fields
        let other_payer_paid_amount = encounter_fields.get("other_payer_paid_amount")
            .and_then(|s| s.parse::<rust_decimal::Decimal>().ok());
        let other_payer_id = encounter_fields.get("other_payer_id").map(|s| s.as_str());
        let other_payer_name = encounter_fields.get("other_payer_name").map(|s| s.as_str());
        let other_payer_claim_number = encounter_fields.get("other_payer_claim_number").map(|s| s.as_str());
        let other_payer_claim_filing_indicator = encounter_fields.get("other_payer_claim_filing_indicator").map(|s| s.as_str());

        // Phase 3: Advanced segments
        // Phase 3.1: Reference numbers (REF segments) - stored as JSONB
        let reference_numbers = encounter_fields.get("reference_numbers").map(|s| s.as_str());

        // Phase 3.2: Provider taxonomy codes (PRV segments)
        let rendering_provider_taxonomy = encounter_fields.get("rendering_provider_taxonomy").map(|s| s.as_str());
        let referring_provider_taxonomy = encounter_fields.get("referring_provider_taxonomy").map(|s| s.as_str());
        let supervising_provider_taxonomy = encounter_fields.get("supervising_provider_taxonomy").map(|s| s.as_str());

        // Extract provider first/last names (from 837P NM1 segments)
        let rendering_provider_last_name = encounter_fields.get("rendering_provider_last_name").map(|s| s.as_str());
        let rendering_provider_first_name = encounter_fields.get("rendering_provider_first_name").map(|s| s.as_str());
        let referring_provider_last_name = encounter_fields.get("referring_provider_last_name").map(|s| s.as_str());
        let referring_provider_first_name = encounter_fields.get("referring_provider_first_name").map(|s| s.as_str());
        let supervising_provider_last_name = encounter_fields.get("supervising_provider_last_name").map(|s| s.as_str());
        let supervising_provider_first_name = encounter_fields.get("supervising_provider_first_name").map(|s| s.as_str());

        // Ensure providers exist in claims.provider table and get their provider_ids
        // Provider errors are handled internally using savepoints - claims proceed with NULL provider_id on error
        let rendering_provider_id = if let Some(npi) = rendering_provider_npi {
            self.ensure_provider_exists(
                tx,
                npi,
                "Rendering",
                rendering_provider_last_name,
                rendering_provider_first_name,
                None,
                rendering_provider_taxonomy,
                Some(organization_id),
            ).await.unwrap_or_else(|e| {
                warn!("Unexpected error ensuring rendering provider NPI={}: {:?}", npi, e);
                None
            })
        } else {
            None
        };

        let referring_provider_id = if let Some(npi) = referring_provider_npi {
            self.ensure_provider_exists(
                tx,
                npi,
                "Referring",
                referring_provider_last_name,
                referring_provider_first_name,
                None,
                referring_provider_taxonomy,
                Some(organization_id),
            ).await.unwrap_or_else(|e| {
                warn!("Unexpected error ensuring referring provider NPI={}: {:?}", npi, e);
                None
            })
        } else {
            None
        };

        let supervising_provider_id = if let Some(npi) = supervising_provider_npi {
            self.ensure_provider_exists(
                tx,
                npi,
                "Supervising",
                supervising_provider_last_name,
                supervising_provider_first_name,
                None,
                supervising_provider_taxonomy,
                Some(organization_id),
            ).await.unwrap_or_else(|e| {
                warn!("Unexpected error ensuring supervising provider NPI={}: {:?}", npi, e);
                None
            })
        } else {
            None
        };

        let billing_provider_id = if let Some(npi) = billing_provider_npi {
            // For billing provider, we may only have organization name (not first/last)
            // Split billing_provider_name into last/first if it contains a comma
            let (last, first) = if let Some(name) = billing_provider_name {
                if name.contains(',') {
                    let parts: Vec<&str> = name.splitn(2, ',').collect();
                    (Some(parts[0].trim()), parts.get(1).map(|s| s.trim()))
                } else {
                    (Some(name), None)
                }
            } else {
                (None, None)
            };

            self.ensure_provider_exists(
                tx,
                npi,
                "Billing",
                last,
                first,
                None,
                None,
                Some(organization_id),
            ).await.unwrap_or_else(|e| {
                warn!("Unexpected error ensuring billing provider NPI={}: {:?}", npi, e);
                None
            })
        } else {
            None
        };

        // Phase 3.4: Condition codes (CRC segments) - stored as JSONB
        let condition_codes = encounter_fields.get("condition_codes").map(|s| s.as_str());

        // Phase 3.5: Supplemental amounts (AMT segments)
        let non_covered_charges = encounter_fields.get("non_covered_charges")
            .and_then(|s| s.parse::<rust_decimal::Decimal>().ok());
        let patient_responsibility_amount = encounter_fields.get("patient_responsibility_amount")
            .and_then(|s| s.parse::<rust_decimal::Decimal>().ok());

        // Parse date_of_service_to if available
        let dos_to = encounter_fields.get("date_of_service_to")
            .and_then(|s| chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d").ok());

        // Parse clinical dates (16 date types from DTP segments)
        let onset_of_illness_date = encounter_fields.get("onset_of_illness_date")
            .and_then(|s| chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d").ok());
        let initial_treatment_date = encounter_fields.get("initial_treatment_date")
            .and_then(|s| chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d").ok());
        let last_seen_date = encounter_fields.get("last_seen_date")
            .and_then(|s| chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d").ok());
        let acute_manifestation_date = encounter_fields.get("acute_manifestation_date")
            .and_then(|s| chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d").ok());
        let accident_date = encounter_fields.get("accident_date")
            .and_then(|s| chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d").ok());
        let last_menstrual_period_date = encounter_fields.get("last_menstrual_period_date")
            .and_then(|s| chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d").ok());
        let last_xray_date = encounter_fields.get("last_xray_date")
            .and_then(|s| chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d").ok());
        let disability_from_date = encounter_fields.get("disability_from_date")
            .and_then(|s| chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d").ok());
        let disability_to_date = encounter_fields.get("disability_to_date")
            .and_then(|s| chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d").ok());
        let last_worked_date = encounter_fields.get("last_worked_date")
            .and_then(|s| chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d").ok());
        let authorized_return_to_work_date = encounter_fields.get("authorized_return_to_work_date")
            .and_then(|s| chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d").ok());
        let admission_date = encounter_fields.get("admission_date")
            .and_then(|s| chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d").ok());
        let discharge_date = encounter_fields.get("discharge_date")
            .and_then(|s| chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d").ok());
        let assumed_care_date = encounter_fields.get("assumed_care_date")
            .and_then(|s| chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d").ok());
        let relinquished_care_date = encounter_fields.get("relinquished_care_date")
            .and_then(|s| chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d").ok());

        // Insert encounter with all available fields
        sqlx::query(
            r#"
            INSERT INTO claims.encounter (
                encounter_id,
                facility_id,
                organization_id,
                region_id,
                submitter_id,
                patient_control_number,
                subscriber_id,
                subscriber_last_name,
                subscriber_first_name,
                subscriber_middle_name,
                subscriber_name_suffix,
                subscriber_gender,
                subscriber_birth_date,
                subscriber_address_line1,
                subscriber_address_line2,
                subscriber_city,
                subscriber_state,
                subscriber_postal_code,
                subscriber_country,
                payer_responsibility_code,
                payer_id,
                payer_name,
                payer_address_line1,
                payer_address_line2,
                payer_city,
                payer_state,
                payer_postal_code,
                billing_provider_id,
                billing_provider_npi,
                billing_provider_tax_id,
                billing_provider_name,
                billing_provider_address_line1,
                billing_provider_city,
                billing_provider_state,
                billing_provider_postal_code,
                rendering_provider_id,
                rendering_provider_npi,
                rendering_provider_name,
                referring_provider_id,
                referring_provider_npi,
                referring_provider_name,
                service_facility_npi,
                service_facility_name,
                service_facility_address_line1,
                service_facility_address_line2,
                service_facility_city,
                service_facility_state,
                service_facility_postal_code,
                supervising_provider_id,
                supervising_provider_npi,
                supervising_provider_name,
                total_claim_charge_amount,
                place_of_service_code,
                date_of_service_from,
                date_of_service_to,
                onset_of_illness_date,
                initial_treatment_date,
                last_seen_date,
                acute_manifestation_date,
                accident_date,
                last_menstrual_period_date,
                last_xray_date,
                disability_from_date,
                disability_to_date,
                last_worked_date,
                authorized_return_to_work_date,
                admission_date,
                discharge_date,
                assumed_care_date,
                relinquished_care_date,
                transaction_set_control_number,
                submitter_name,
                claim_filing_indicator,
                claim_frequency_code,
                signature_indicator,
                assignment_indicator,
                benefits_assignment_indicator,
                release_of_information_code,
                patient_signature_code,
                delay_reason_code,
                special_program_code,
                patient_amount_paid,
                service_authorization_code,
                other_payer_paid_amount,
                other_payer_id,
                other_payer_name,
                other_payer_claim_number,
                other_payer_claim_filing_indicator,
                reference_numbers,
                rendering_provider_taxonomy,
                referring_provider_taxonomy,
                supervising_provider_taxonomy,
                condition_codes,
                non_covered_charges,
                patient_responsibility_amount,
                medical_record_number,
                claim_status
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41, $42, $43, $44, $45, $46, $47, $48, $49, $50, $51, $52, $53, $54, $55, $56, $57, $58, $59, $60, $61, $62, $63, $64, $65, $66, $67, $68, $69, $70, $71, $72, $73, $74, $75, $76, $77, $78, $79, $80, $81, $82, $83, $84, $85, $86, $87, $88, $89, $90, $91, $92, $93, $94, $95, $96, $97, $98, $99, $100)
            "#
        )
        .bind(encounter_id)
        .bind(facility_id)
        .bind(organization_id)
        .bind(region_id)
        .bind(submitter_id)
        .bind(patient_control_number)
        .bind(subscriber_id)
        .bind(subscriber_last_name)
        .bind(subscriber_first_name)
        .bind(subscriber_middle_name)
        .bind(subscriber_name_suffix)
        .bind(subscriber_gender)
        .bind(subscriber_dob)
        .bind(subscriber_address_line1)
        .bind(subscriber_address_line2)
        .bind(subscriber_city)
        .bind(subscriber_state)
        .bind(subscriber_postal_code)
        .bind(subscriber_country)
        .bind(payer_responsibility_code)
        .bind(payer_id)
        .bind(payer_name)
        .bind(payer_address_line1)
        .bind(payer_address_line2)
        .bind(payer_city)
        .bind(payer_state)
        .bind(payer_postal_code)
        .bind(billing_provider_id)
        .bind(billing_provider_npi)
        .bind(billing_provider_tax_id)
        .bind(billing_provider_name)
        .bind(billing_provider_address_line1)
        .bind(billing_provider_city)
        .bind(billing_provider_state)
        .bind(billing_provider_postal_code)
        .bind(rendering_provider_id)
        .bind(rendering_provider_npi)
        .bind(rendering_provider_name)
        .bind(referring_provider_id)
        .bind(referring_provider_npi)
        .bind(referring_provider_name)
        .bind(service_facility_npi)
        .bind(service_facility_name)
        .bind(service_facility_address_line1)
        .bind(service_facility_address_line2)
        .bind(service_facility_city)
        .bind(service_facility_state)
        .bind(service_facility_postal_code)
        .bind(supervising_provider_id)
        .bind(supervising_provider_npi)
        .bind(supervising_provider_name)
        .bind(total_claim_charge)
        .bind(place_of_service)
        .bind(dos_from)
        .bind(dos_to)
        .bind(onset_of_illness_date)
        .bind(initial_treatment_date)
        .bind(last_seen_date)
        .bind(acute_manifestation_date)
        .bind(accident_date)
        .bind(last_menstrual_period_date)
        .bind(last_xray_date)
        .bind(disability_from_date)
        .bind(disability_to_date)
        .bind(last_worked_date)
        .bind(authorized_return_to_work_date)
        .bind(admission_date)
        .bind(discharge_date)
        .bind(assumed_care_date)
        .bind(relinquished_care_date)
        .bind(transaction_set_control_number)
        .bind(submitter_name)
        .bind(claim_filing_indicator)
        .bind(claim_frequency_code)
        .bind(signature_indicator)
        .bind(assignment_indicator)
        .bind(benefits_assignment_indicator)
        .bind(release_of_information_code)
        .bind(patient_signature_code)
        .bind(delay_reason_code)
        .bind(special_program_code)
        .bind(patient_amount_paid)
        .bind(service_authorization_code)
        .bind(other_payer_paid_amount)
        .bind(other_payer_id)
        .bind(other_payer_name)
        .bind(other_payer_claim_number)
        .bind(other_payer_claim_filing_indicator)
        .bind(reference_numbers)
        .bind(rendering_provider_taxonomy)
        .bind(referring_provider_taxonomy)
        .bind(supervising_provider_taxonomy)
        .bind(condition_codes)
        .bind(non_covered_charges)
        .bind(patient_responsibility_amount)
        .bind(medical_record_number)
        .bind("IMPORTED")
        .execute(&mut **tx)
        .await
        .context("Failed to insert encounter")?;

        debug!("Inserted encounter {} for patient {}", encounter_id, patient_control_number);

        // Insert claim note if present
        if let Some(claim_note) = encounter_fields.get("claim_note") {
            if !claim_note.is_empty() {
                sqlx::query(
                    r#"
                    INSERT INTO claims.encounter_note (
                        encounter_id,
                        note_type,
                        note_text,
                        created_by
                    )
                    VALUES ($1, $2, $3, $4)
                    "#
                )
                .bind(encounter_id)
                .bind("CLAIM")
                .bind(claim_note.as_str())
                .bind("SYSTEM")
                .execute(&mut **tx)
                .await
                .context("Failed to insert claim note")?;

                debug!("Inserted claim note for encounter {}", encounter_id);
            }
        }

        // Import diagnoses FIRST so they exist when service line pointers reference them
        self.import_diagnoses(tx, encounter_id, raw_claim).await?;

        // Import all service lines from this raw_claim
        // For EDI: may have multiple service lines (service_line_1_*, service_line_2_*, etc.)
        // For CSV: typically one service line (service_line_1_*)
        // PERFORMANCE: Collect service line contexts for rule execution
        let service_line_fields: HashMap<String, String> = raw_claim.service_line_fields.as_ref()
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or_default();

        let num_service_lines = Self::count_service_lines_in_jsonb(&service_line_fields);
        let num_service_lines = if num_service_lines == 0 { 1 } else { num_service_lines }; // Default to 1

        let mut service_line_contexts: Vec<ServiceLineRuleContext> = Vec::with_capacity(num_service_lines);
        for sl_idx in 1..=num_service_lines {
            let prefix = format!("service_line_{}_", sl_idx);
            let sl_ctx = self.import_service_line(tx, encounter_id, organization_id, raw_claim, sl_idx as i32, &prefix).await?;
            service_line_contexts.push(sl_ctx);
        }

        // Extract diagnosis codes for rule execution (reuse parsed data)
        let diagnosis_codes: Vec<String> = raw_claim.diagnosis_fields.as_ref()
            .and_then(|df| serde_json::from_value::<HashMap<String, Vec<String>>>(df.clone()).ok())
            .map(|fields| {
                let mut codes: Vec<(usize, String)> = Vec::new();
                for (field_name, values) in &fields {
                    if field_name.starts_with("diagnosis_code_") {
                        if let Some(seq_str) = field_name.strip_prefix("diagnosis_code_") {
                            if let Ok(seq) = seq_str.parse::<usize>() {
                                for code in values {
                                    codes.push((seq, code.clone()));
                                }
                            }
                        }
                    } else if field_name == "diagnosis_code" {
                        for (idx, code) in values.iter().enumerate() {
                            codes.push((idx + 1, code.clone()));
                        }
                    }
                }
                codes.sort_by_key(|(seq, _)| *seq);
                codes.into_iter().map(|(_, code)| code).collect()
            })
            .unwrap_or_default();

        // Execute rules engine (OPTIMIZED: no DB re-queries)
        let _flags_created = self.execute_rules_for_service_lines(
            tx, encounter_id, organization_id, &service_line_contexts, &diagnosis_codes
        ).await?;

        Ok(encounter_id)
    }

    /// Count the number of service lines in the service_line_fields JSONB
    /// Returns the highest service line number found (e.g., if keys include service_line_3_*, returns 3)
    fn count_service_lines_in_jsonb(service_line_fields: &HashMap<String, String>) -> usize {
        let mut max_line = 0;
        for key in service_line_fields.keys() {
            // Keys are like "service_line_1_procedure_code", "service_line_2_charge_amount", etc.
            if let Some(rest) = key.strip_prefix("service_line_") {
                if let Some(num_str) = rest.split('_').next() {
                    if let Ok(num) = num_str.parse::<usize>() {
                        max_line = max_line.max(num);
                    }
                }
            }
        }
        max_line
    }

    /// PHASE 4 FIX: Count service lines directly from JsonValue without cloning/deserializing
    /// Returns the highest service line number found
    fn count_service_lines_in_json_value(service_line_fields: &JsonValue) -> usize {
        let mut max_line = 0;
        if let Some(obj) = service_line_fields.as_object() {
            for key in obj.keys() {
                // Keys are like "service_line_1_procedure_code", "service_line_2_charge_amount", etc.
                if let Some(rest) = key.strip_prefix("service_line_") {
                    if let Some(num_str) = rest.split('_').next() {
                        if let Ok(num) = num_str.parse::<usize>() {
                            max_line = max_line.max(num);
                        }
                    }
                }
            }
        }
        max_line
    }

    /// Import service line for an encounter
    /// The `service_line_prefix` parameter specifies which service line to read from the JSONB
    /// (e.g., "service_line_1_" or "service_line_2_")
    /// Import a service line and return context for rule execution
    /// Returns ServiceLineRuleContext to avoid re-querying DB for rules
    async fn import_service_line(
        &self,
        tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
        encounter_id: i64,
        organization_id: i64,
        raw_claim: &RawClaim,
        line_number: i32,
        service_line_prefix: &str,
    ) -> Result<ServiceLineRuleContext> {
        // Deserialize encounter fields - use JsonValue to handle mixed types (strings, arrays like other_insurance)
        let encounter_fields: HashMap<String, JsonValue> = serde_json::from_value(raw_claim.encounter_fields.clone())
            .context("Failed to deserialize encounter_fields")?;

        let service_line_fields: HashMap<String, String> = raw_claim.service_line_fields.as_ref()
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or_default();

        // Use the provided prefix to read the correct service line from JSONB
        // For CSV: always "service_line_1_" (one service line per raw_claim)
        // For EDI: "service_line_1_", "service_line_2_", etc. (all service lines in one raw_claim)
        let prefix = service_line_prefix;

        // Extract required service line fields using line number prefix
        let procedure_code = service_line_fields.get(&format!("{}procedure_code", prefix))
            .context("Missing procedure_code")?;
        let line_item_charge_amount = service_line_fields.get(&format!("{}charge_amount", prefix))
            .context("Missing line_item_charge_amount")?;
        let default_unit_count = "1".to_string();
        let service_unit_count = service_line_fields.get(&format!("{}units", prefix))
            .unwrap_or(&default_unit_count);

        // Get service date - use service_date_from if available, otherwise fall back to encounter DOS
        let service_date_str = service_line_fields.get(&format!("{}date_from", prefix))
            .cloned()
            .or_else(|| get_field_as_string(&encounter_fields, "date_of_service_from"))
            .context("Missing service_date_from and date_of_service_from")?;

        // Parse decimal values
        let charge_amount = line_item_charge_amount.parse::<rust_decimal::Decimal>()
            .context("Invalid charge amount")?;
        // DB constraint: service_unit_count > 0 AND service_unit_count <= 9999.9
        // Clamp out-of-range / unparseable values to keep the row insertable.
        let unit_count = {
            let parsed = service_unit_count.parse::<rust_decimal::Decimal>()
                .unwrap_or(rust_decimal::Decimal::ONE);
            // 9999.9 as Decimal (mantissa 99999, scale 1)
            let max_units = rust_decimal::Decimal::new(99_999, 1);
            if parsed <= rust_decimal::Decimal::ZERO {
                warn!("service_unit_count {} <= 0, defaulting to 1 (procedure {})", parsed, procedure_code);
                rust_decimal::Decimal::ONE
            } else if parsed > max_units {
                warn!("service_unit_count {} > 9999.9, clamping (procedure {})", parsed, procedure_code);
                max_units
            } else {
                parsed
            }
        };

        // Parse service date
        let service_date = chrono::NaiveDate::parse_from_str(&service_date_str, "%Y-%m-%d")
            .context("Invalid date format for service_date_from")?;

        // Extract diagnosis pointers (CRITICAL for medical necessity validation)
        // NOTE: diagnosis_pointers field contains comma-separated values like "1,2,3"
        let diagnosis_pointers = service_line_fields.get(&format!("{}diagnosis_pointers", prefix))
            .map(|s| s.as_str())
            .unwrap_or("");
        let pointers: Vec<&str> = diagnosis_pointers.split(',').filter(|s| !s.is_empty()).collect();
        let pointer_1 = pointers.get(0).and_then(|s| s.trim().parse::<i16>().ok());
        let pointer_2 = pointers.get(1).and_then(|s| s.trim().parse::<i16>().ok());
        let pointer_3 = pointers.get(2).and_then(|s| s.trim().parse::<i16>().ok());
        let pointer_4 = pointers.get(3).and_then(|s| s.trim().parse::<i16>().ok());

        // Helper function to truncate to 2 chars for VARCHAR(2) columns
        fn truncate_2(s: &str) -> &str {
            if s.len() > 2 { &s[..2] } else { s }
        }

        // Extract procedure modifiers (filter out empty strings) - truncate to 2 chars (VARCHAR(2))
        let modifier_1 = service_line_fields.get(&format!("{}modifier_1", prefix))
            .filter(|s| !s.is_empty())
            .map(|s| truncate_2(s.as_str()));
        let modifier_2 = service_line_fields.get(&format!("{}modifier_2", prefix))
            .filter(|s| !s.is_empty())
            .map(|s| truncate_2(s.as_str()));
        let modifier_3 = service_line_fields.get(&format!("{}modifier_3", prefix))
            .filter(|s| !s.is_empty())
            .map(|s| truncate_2(s.as_str()));
        let modifier_4 = service_line_fields.get(&format!("{}modifier_4", prefix))
            .filter(|s| !s.is_empty())
            .map(|s| truncate_2(s.as_str()));

        // Extract additional service line fields with prefix - truncate VARCHAR(2) fields
        let product_service_id_qualifier = service_line_fields.get(&format!("{}product_service_id_qualifier", prefix))
            .filter(|s| !s.is_empty())
            .map(|s| truncate_2(s.as_str()))
            .unwrap_or("HC"); // Default to HCPCS
        let unit_basis_measurement_code = service_line_fields.get(&format!("{}unit_basis_measurement_code", prefix))
            .filter(|s| !s.is_empty())
            .map(|s| truncate_2(s.as_str()));
        let service_date_to = service_line_fields.get(&format!("{}date_to", prefix))
            .and_then(|s| chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d").ok());
        let place_of_service_code = service_line_fields.get(&format!("{}place_of_service_code", prefix))
            .filter(|s| !s.is_empty())
            .map(|s| truncate_2(s.as_str()));

        // Convert EDI Y/N indicators to boolean values
        let emergency_indicator = service_line_fields.get(&format!("{}emergency_indicator", prefix))
            .map(|s| s.trim().eq_ignore_ascii_case("Y"))
            .unwrap_or(false);
        let epsdt_indicator = service_line_fields.get(&format!("{}epsdt_indicator", prefix))
            .map(|s| s.trim().eq_ignore_ascii_case("Y"))
            .unwrap_or(false);
        let family_planning_indicator = service_line_fields.get(&format!("{}family_planning_indicator", prefix))
            .map(|s| s.trim().eq_ignore_ascii_case("Y"))
            .unwrap_or(false);

        // Phase 3: Service line level fields
        // Phase 3.1: Provider NPIs at service line level (Loop 2420)
        let sl_rendering_provider_npi = service_line_fields.get(&format!("{}rendering_provider_npi", prefix))
            .filter(|s| !s.is_empty())
            .map(|s| s.as_str());
        let sl_ordering_provider_npi = service_line_fields.get(&format!("{}ordering_provider_npi", prefix))
            .filter(|s| !s.is_empty())
            .map(|s| s.as_str());
        let sl_supervising_provider_npi = service_line_fields.get(&format!("{}supervising_provider_npi", prefix))
            .filter(|s| !s.is_empty())
            .map(|s| s.as_str());
        let sl_referring_provider_npi = service_line_fields.get(&format!("{}referring_provider_npi", prefix))
            .filter(|s| !s.is_empty())
            .map(|s| s.as_str());

        // Phase 3.2: Provider names at service line level
        let sl_rendering_provider_last_name = service_line_fields.get(&format!("{}rendering_provider_last_name", prefix))
            .filter(|s| !s.is_empty())
            .map(|s| s.as_str());
        let sl_rendering_provider_first_name = service_line_fields.get(&format!("{}rendering_provider_first_name", prefix))
            .filter(|s| !s.is_empty())
            .map(|s| s.as_str());
        let sl_ordering_provider_last_name = service_line_fields.get(&format!("{}ordering_provider_last_name", prefix))
            .filter(|s| !s.is_empty())
            .map(|s| s.as_str());
        let sl_ordering_provider_first_name = service_line_fields.get(&format!("{}ordering_provider_first_name", prefix))
            .filter(|s| !s.is_empty())
            .map(|s| s.as_str());
        let sl_supervising_provider_last_name = service_line_fields.get(&format!("{}supervising_provider_last_name", prefix))
            .filter(|s| !s.is_empty())
            .map(|s| s.as_str());
        let sl_supervising_provider_first_name = service_line_fields.get(&format!("{}supervising_provider_first_name", prefix))
            .filter(|s| !s.is_empty())
            .map(|s| s.as_str());
        let sl_referring_provider_last_name = service_line_fields.get(&format!("{}referring_provider_last_name", prefix))
            .filter(|s| !s.is_empty())
            .map(|s| s.as_str());
        let sl_referring_provider_first_name = service_line_fields.get(&format!("{}referring_provider_first_name", prefix))
            .filter(|s| !s.is_empty())
            .map(|s| s.as_str());

        // Phase 3.3: Rendering provider taxonomy (from encounter level, not service line level)
        let sl_rendering_provider_taxonomy_owned = get_field_as_string(&encounter_fields, "rendering_provider_taxonomy");
        let sl_rendering_provider_taxonomy = sl_rendering_provider_taxonomy_owned.as_deref()
            .filter(|s| !s.is_empty());

        // Phase 3.4: Ensure service line providers exist and get their IDs
        // Provider errors are logged but do NOT fail the claim - claims proceed with NULL provider_id
        let sl_rendering_provider_id = if let Some(npi) = sl_rendering_provider_npi {
            match self.ensure_provider_exists(
                tx,
                npi,
                "Rendering",
                sl_rendering_provider_last_name,
                sl_rendering_provider_first_name,
                None,
                sl_rendering_provider_taxonomy,
                Some(organization_id),
            ).await {
                Ok(provider_id) => {
                    if provider_id.is_none() {
                        warn!("Failed to create/find service line rendering provider: NPI={}, Name={} {}",
                            npi,
                            sl_rendering_provider_first_name.unwrap_or(""),
                            sl_rendering_provider_last_name.unwrap_or(""));
                    }
                    provider_id
                },
                Err(e) => {
                    error!("Error ensuring service line rendering provider exists: NPI={}, Error={:?}", npi, e);
                    None
                }
            }
        } else {
            None
        };

        let sl_ordering_provider_id = if let Some(npi) = sl_ordering_provider_npi {
            match self.ensure_provider_exists(
                tx,
                npi,
                "Ordering",
                sl_ordering_provider_last_name,
                sl_ordering_provider_first_name,
                None,
                None,
                Some(organization_id),
            ).await {
                Ok(provider_id) => {
                    if provider_id.is_none() {
                        warn!("Failed to create/find service line ordering provider: NPI={}, Name={} {}",
                            npi,
                            sl_ordering_provider_first_name.unwrap_or(""),
                            sl_ordering_provider_last_name.unwrap_or(""));
                    }
                    provider_id
                },
                Err(e) => {
                    error!("Error ensuring service line ordering provider exists: NPI={}, Error={:?}", npi, e);
                    None
                }
            }
        } else {
            None
        };

        let sl_supervising_provider_id = if let Some(npi) = sl_supervising_provider_npi {
            match self.ensure_provider_exists(
                tx,
                npi,
                "Supervising",
                sl_supervising_provider_last_name,
                sl_supervising_provider_first_name,
                None,
                None,
                Some(organization_id),
            ).await {
                Ok(provider_id) => {
                    if provider_id.is_none() {
                        warn!("Failed to create/find service line supervising provider: NPI={}, Name={} {}",
                            npi,
                            sl_supervising_provider_first_name.unwrap_or(""),
                            sl_supervising_provider_last_name.unwrap_or(""));
                    }
                    provider_id
                },
                Err(e) => {
                    error!("Error ensuring service line supervising provider exists: NPI={}, Error={:?}", npi, e);
                    None
                }
            }
        } else {
            None
        };

        let sl_referring_provider_id = if let Some(npi) = sl_referring_provider_npi {
            match self.ensure_provider_exists(
                tx,
                npi,
                "Referring",
                sl_referring_provider_last_name,
                sl_referring_provider_first_name,
                None,
                None,
                Some(organization_id),
            ).await {
                Ok(provider_id) => {
                    if provider_id.is_none() {
                        warn!("Failed to create/find service line referring provider: NPI={}, Name={} {}",
                            npi,
                            sl_referring_provider_first_name.unwrap_or(""),
                            sl_referring_provider_last_name.unwrap_or(""));
                    }
                    provider_id
                },
                Err(e) => {
                    error!("Error ensuring service line referring provider exists: NPI={}, Error={:?}", npi, e);
                    None
                }
            }
        } else {
            None
        };

        // Phase 3.5: Supplemental amounts at service line level
        let sl_approved_amount = service_line_fields.get(&format!("{}approved_amount", prefix))
            .and_then(|s| s.parse::<rust_decimal::Decimal>().ok());
        let sl_non_covered_charges = service_line_fields.get(&format!("{}non_covered_charges", prefix))
            .and_then(|s| s.parse::<rust_decimal::Decimal>().ok());

        // Phase 3.6: NDC information (drug codes)
        let sl_ndc_code = service_line_fields.get(&format!("{}ndc_code", prefix))
            .filter(|s| !s.is_empty())
            .map(|s| s.as_str());
        let sl_ndc_unit_count = service_line_fields.get(&format!("{}ndc_unit_count", prefix))
            .and_then(|s| s.parse::<rust_decimal::Decimal>().ok());
        // Truncate ndc_measurement_unit to 2 chars (VARCHAR(2) in DB)
        let sl_ndc_measurement_unit = service_line_fields.get(&format!("{}ndc_measurement_unit", prefix))
            .filter(|s| !s.is_empty())
            .map(|s| truncate_2(s.as_str()));

        // Phase 3.7: Authorization and referral information
        let sl_prior_authorization = service_line_fields.get(&format!("{}prior_authorization_number", prefix))
            .filter(|s| !s.is_empty())
            .map(|s| s.as_str());
        let sl_referral_number = service_line_fields.get(&format!("{}referral_number", prefix))
            .filter(|s| !s.is_empty())
            .map(|s| s.as_str());

        // Phase 3.8: Line note and revenue code
        let sl_line_note = service_line_fields.get(&format!("{}line_note", prefix))
            .filter(|s| !s.is_empty())
            .map(|s| s.as_str());
        let sl_revenue_code = service_line_fields.get(&format!("{}revenue_code", prefix))
            .filter(|s| !s.is_empty())
            .map(|s| s.as_str());

        // Phase 3.9: Other payer line amount (COB)
        let sl_other_payer_paid = service_line_fields.get(&format!("{}other_payer_line_paid_amount", prefix))
            .and_then(|s| s.parse::<rust_decimal::Decimal>().ok());

        // Phase 3.10: Pricing information (from HCP segment)
        let sl_allowed_amount = service_line_fields.get(&format!("{}allowed_amount", prefix))
            .and_then(|s| s.parse::<rust_decimal::Decimal>().ok());
        let sl_saving_amount = service_line_fields.get(&format!("{}saving_amount", prefix))
            .and_then(|s| s.parse::<rust_decimal::Decimal>().ok());

        // Insert service line and get generated ID
        let service_line_id: i64 = sqlx::query_scalar(
            r#"
            INSERT INTO claims.service_line (
                encounter_id,
                line_number,
                product_service_id_qualifier,
                procedure_code,
                procedure_modifier_1,
                procedure_modifier_2,
                procedure_modifier_3,
                procedure_modifier_4,
                line_item_charge_amount,
                unit_basis_measurement_code,
                service_unit_count,
                service_date_from,
                service_date_to,
                place_of_service_code,
                emergency_indicator,
                epsdt_indicator,
                family_planning_indicator,
                diagnosis_code_pointer_1,
                diagnosis_code_pointer_2,
                diagnosis_code_pointer_3,
                diagnosis_code_pointer_4,
                rendering_provider_id,
                rendering_provider_npi,
                ordering_provider_id,
                ordering_provider_npi,
                supervising_provider_id,
                supervising_provider_npi,
                referring_provider_id,
                referring_provider_npi,
                rendering_provider_taxonomy,
                approved_amount,
                non_covered_charges,
                ndc_code,
                ndc_unit_count,
                ndc_measurement_unit,
                prior_authorization_number,
                referral_number,
                line_note,
                revenue_code,
                other_payer_line_paid_amount,
                line_status,
                allowed_amount,
                saving_amount
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41, $42, $43)
            RETURNING service_line_id
            "#
        )
        .bind(encounter_id)
        .bind(line_number)
        .bind(product_service_id_qualifier)
        .bind(procedure_code)
        .bind(modifier_1)
        .bind(modifier_2)
        .bind(modifier_3)
        .bind(modifier_4)
        .bind(charge_amount)
        .bind(unit_basis_measurement_code)
        .bind(unit_count)
        .bind(service_date)
        .bind(service_date_to)
        .bind(place_of_service_code)
        .bind(emergency_indicator)
        .bind(epsdt_indicator)
        .bind(family_planning_indicator)
        .bind(pointer_1)
        .bind(pointer_2)
        .bind(pointer_3)
        .bind(pointer_4)
        .bind(sl_rendering_provider_id)
        .bind(sl_rendering_provider_npi)
        .bind(sl_ordering_provider_id)
        .bind(sl_ordering_provider_npi)
        .bind(sl_supervising_provider_id)
        .bind(sl_supervising_provider_npi)
        .bind(sl_referring_provider_id)
        .bind(sl_referring_provider_npi)
        .bind(sl_rendering_provider_taxonomy)
        .bind(sl_approved_amount)
        .bind(sl_non_covered_charges)
        .bind(sl_ndc_code)
        .bind(sl_ndc_unit_count)
        .bind(sl_ndc_measurement_unit)
        .bind(sl_prior_authorization)
        .bind(sl_referral_number)
        .bind(sl_line_note)
        .bind(sl_revenue_code)
        .bind(sl_other_payer_paid)
        .bind("IMPORTED")
        .bind(sl_allowed_amount)
        .bind(sl_saving_amount)
        .fetch_one(&mut **tx)
        .await
        .map_err(|e| {
            error!("DATABASE ERROR inserting service line: {:?}", e);
            error!("  procedure_code: {}", procedure_code);
            error!("  charge_amount: {}", charge_amount);
            error!("  unit_count: {}", unit_count);
            error!("  service_date: {}", service_date);
            error!("  diagnosis_pointers: {:?},{:?},{:?},{:?}", pointer_1, pointer_2, pointer_3, pointer_4);
            anyhow::anyhow!("Failed to insert service line: {}", e)
        })?;

        debug!("Inserted service line {} for encounter {}", service_line_id, encounter_id);

        // Now populate the service_line_diagnosis_pointer junction table
        // This links the service line to its diagnosis codes via the diagnosis pointers
        self.import_service_line_diagnosis_pointers(
            tx,
            service_line_id,
            encounter_id,
            pointer_1,
            pointer_2,
            pointer_3,
            pointer_4
        ).await?;

        // Build modifiers list for rule context
        let mut modifiers = Vec::with_capacity(4);
        if let Some(m) = modifier_1 { modifiers.push(m.to_string()); }
        if let Some(m) = modifier_2 { modifiers.push(m.to_string()); }
        if let Some(m) = modifier_3 { modifiers.push(m.to_string()); }
        if let Some(m) = modifier_4 { modifiers.push(m.to_string()); }

        // Return context for rule execution (avoids re-querying DB)
        Ok(ServiceLineRuleContext {
            service_line_id,
            procedure_code: procedure_code.to_string(),
            modifiers,
            units: unit_count,
            charge: charge_amount,
            service_date,
            place_of_service: place_of_service_code.map(|s| s.to_string()),
        })
    }

    /// Import service line diagnosis pointers (junction table)
    /// Links service lines to encounter diagnoses based on diagnosis code pointers
    /// Optimized: Single INSERT with subquery instead of N SELECT + N INSERT
    async fn import_service_line_diagnosis_pointers(
        &self,
        tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
        service_line_id: i64,
        encounter_id: i64,
        pointer_1: Option<i16>,
        pointer_2: Option<i16>,
        pointer_3: Option<i16>,
        pointer_4: Option<i16>,
    ) -> Result<()> {
        // Collect all non-null pointers with their sequence positions
        let pointers: Vec<(i16, i16)> = vec![
            (1, pointer_1),
            (2, pointer_2),
            (3, pointer_3),
            (4, pointer_4),
        ]
        .into_iter()
        .filter_map(|(seq, ptr)| ptr.map(|p| (seq, p)))
        .collect();

        if pointers.is_empty() {
            debug!("No diagnosis pointers for service_line {} (all None)", service_line_id);
            return Ok(());
        }

        debug!("Processing {} diagnosis pointers for service_line {} encounter {}",
            pointers.len(), service_line_id, encounter_id);

        // Build batch INSERT with subquery to resolve diagnosis_id in one query
        // This eliminates N SELECT + N INSERT, replacing with 1 INSERT ... SELECT
        let mut query_parts: Vec<String> = Vec::with_capacity(pointers.len());
        let mut param_idx = 3; // $1 = service_line_id, $2 = encounter_id, $3+ = pointer params

        for _ in &pointers {
            // Each row: SELECT diagnosis_id for this pointer, with pointer_sequence
            query_parts.push(format!(
                "SELECT $1::bigint, ed.diagnosis_id, ${}::smallint FROM claims.encounter_diagnosis ed WHERE ed.encounter_id = $2 AND ed.sequence_number = ${}",
                param_idx, param_idx + 1
            ));
            param_idx += 2;
        }

        let query = format!(
            r#"
            INSERT INTO claims.service_line_diagnosis_pointer (
                service_line_id,
                diagnosis_id,
                pointer_sequence
            )
            {}
            ON CONFLICT (service_line_id, pointer_sequence) DO NOTHING
            "#,
            query_parts.join(" UNION ALL ")
        );

        let mut query_builder = sqlx::query(&query)
            .bind(service_line_id)
            .bind(encounter_id);

        for (pointer_sequence, diagnosis_pointer) in &pointers {
            query_builder = query_builder
                .bind(*pointer_sequence)
                .bind(*diagnosis_pointer);
        }

        let result = query_builder
            .execute(&mut **tx)
            .await
            .context("Failed to batch insert service line diagnosis pointers")?;

        debug!(
            "Inserted {} diagnosis pointer rows for service_line {} encounter {}",
            result.rows_affected(), service_line_id, encounter_id);

        debug!(
            "Batch linked service line {} to {} diagnosis pointers for encounter {}",
            service_line_id,
            result.rows_affected(),
            encounter_id
        );

        Ok(())
    }

    /// Import diagnoses for an encounter
    async fn import_diagnoses(
        &self,
        tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
        encounter_id: i64,
        raw_claim: &RawClaim,
    ) -> Result<()> {
        // Parse diagnosis fields from raw_claim
        let diagnosis_fields: HashMap<String, Vec<String>> = match &raw_claim.diagnosis_fields {
            Some(df) => serde_json::from_value(df.clone()).unwrap_or_default(),
            None => return Ok(()), // No diagnoses to import
        };

        // Collect all diagnosis codes from diagnosis_code_1, diagnosis_code_2, etc.
        let mut all_diagnoses: Vec<(usize, String)> = Vec::new();

        for (field_name, codes) in &diagnosis_fields {
            // Match field names like "diagnosis_code_1", "diagnosis_code_2", etc.
            if field_name.starts_with("diagnosis_code_") {
                if let Some(seq_str) = field_name.strip_prefix("diagnosis_code_") {
                    if let Ok(sequence) = seq_str.parse::<usize>() {
                        for code in codes {
                            all_diagnoses.push((sequence, code.clone()));
                        }
                    }
                }
            }
            // Also support legacy format "diagnosis_code" (single field with array)
            else if field_name == "diagnosis_code" {
                for (idx, code) in codes.iter().enumerate() {
                    all_diagnoses.push((idx + 1, code.clone()));
                }
            }
        }

        // Sort by sequence number to maintain proper order
        all_diagnoses.sort_by_key(|(seq, _)| *seq);

        if all_diagnoses.is_empty() {
            return Ok(());
        }

        // Build batch INSERT with multiple rows for better performance
        // Instead of N individual INSERTs, we do 1 INSERT with N rows
        let mut query_parts: Vec<String> = Vec::with_capacity(all_diagnoses.len());
        let mut param_idx = 1;

        for _ in &all_diagnoses {
            query_parts.push(format!(
                "(${}, ${}, ${}, ${})",
                param_idx, param_idx + 1, param_idx + 2, param_idx + 3
            ));
            param_idx += 4;
        }

        let query = format!(
            r#"
            INSERT INTO claims.encounter_diagnosis (
                encounter_id,
                sequence_number,
                diagnosis_code,
                is_principal
            )
            VALUES {}
            RETURNING diagnosis_id
            "#,
            query_parts.join(", ")
        );

        let mut query_builder = sqlx::query_scalar::<_, i64>(&query);

        for (idx, (sequence, code)) in all_diagnoses.iter().enumerate() {
            let sequence_number = *sequence as i16;
            let is_principal = idx == 0;
            query_builder = query_builder
                .bind(encounter_id)
                .bind(sequence_number)
                .bind(code)
                .bind(is_principal);
        }

        let diagnosis_ids: Vec<i64> = query_builder
            .fetch_all(&mut **tx)
            .await
            .context("Failed to batch insert diagnoses")?;

        debug!(
            "Batch inserted {} diagnoses for encounter {}, ids={:?}",
            diagnosis_ids.len(),
            encounter_id,
            diagnosis_ids
        );

        Ok(())
    }

    /// Log a processing metric to staging.processing_metrics
    async fn log_processing_metric(
        &self,
        batch_id: i64,
        metric_type: &str,
        metric_name: &str,
        started_at: chrono::DateTime<chrono::Utc>,
        completed_at: chrono::DateTime<chrono::Utc>,
        records_processed: i32,
        success_count: i32,
        error_count: i32,
        details: Option<serde_json::Value>,
        processing_stage: &str,
    ) -> Result<()> {
        let duration_ms = (completed_at - started_at).num_milliseconds();
        let duration_sec = duration_ms as f64 / 1000.0;
        let records_per_second = if duration_sec > 0.0 {
            records_processed as f64 / duration_sec
        } else {
            0.0
        };

        sqlx::query(
            r#"
            INSERT INTO staging.processing_metrics (
                batch_id,
                metric_type,
                metric_name,
                started_at,
                completed_at,
                duration_seconds,
                records_processed,
                records_per_second,
                success_count,
                error_count,
                details,
                processing_stage
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            "#
        )
        .bind(batch_id)
        .bind(metric_type)
        .bind(metric_name)
        .bind(started_at)
        .bind(completed_at)
        .bind(duration_sec)
        .bind(records_processed)
        .bind(rust_decimal::Decimal::from_f64_retain(records_per_second).unwrap_or(rust_decimal::Decimal::ZERO))
        .bind(success_count)
        .bind(error_count)
        .bind(details)
        .bind(processing_stage)
        .execute(&self.pool)
        .await
        .context("Failed to insert processing metric")?;

        Ok(())
    }

    /// Process a specific sequenced batch (for multi-worker FIFO processing)
    /// This method is called by workers in the batch sequencing system
    pub async fn process_sequenced_batch(
        &self,
        claim_ids: &[i64],
        sequence_number: i32,
        worker_id: String,
    ) -> Result<crate::batch_sequencer::BatchResult> {
        info!(
            "Worker {} processing sequenced batch {} ({} claims)",
            worker_id, sequence_number, claim_ids.len()
        );

        let batch_start = chrono::Utc::now();
        let mut success_count = 0;
        let mut failure_count = 0;
        let mut errors = Vec::new();

        // Query claims in this batch
        let raw_claims: Vec<RawClaim> = sqlx::query_as(
            r#"
            SELECT
                raw_claim_id,
                batch_id,
                queue_id,
                encounter_fields,
                service_line_fields,
                diagnosis_fields,
                row_number,
                facility_code,
                date_of_service_from
            FROM staging.raw_claims
            WHERE raw_claim_id = ANY($1)
            AND batch_sequence_number = $2
            AND processing_status = 'PROCESSING'
            "#
        )
        .bind(claim_ids)
        .bind(sequence_number)
        .fetch_all(&self.pool)
        .await
        .context("Failed to fetch sequenced batch claims")?;

        if raw_claims.is_empty() {
            warn!("No claims found for batch sequence {}", sequence_number);
            return Ok(crate::batch_sequencer::BatchResult {
                sequence_number,
                batch_id: 0,
                success_count: 0,
                failure_count: 0,
                processing_time_seconds: 0.0,
                worker_id,
                errors: vec![],
            });
        }

        let batch_id = raw_claims[0].batch_id;

        // PROVIDER CACHE PREWARM (batch level)
        // Commit all providers referenced anywhere in this batch in a single
        // dedicated transaction BEFORE spawning per-encounter parallel tasks.
        // This is what prevents the `service_line_*_provider_id_fkey` FK race
        // between concurrent encounter transactions.
        if let Err(e) = self.prewarm_provider_cache_for_batch(&raw_claims).await {
            warn!("Batch provider prewarm failed for sequence {}: {:#}", sequence_number, e);
            // Don't abort the batch - per-encounter prewarm will retry on demand.
        }

        // Thread-safe facility lookup cache (shared across parallel encounters)
        let facility_cache: Arc<tokio::sync::RwLock<HashMap<String, (Option<i64>, i64, Option<i64>)>>> =
            Arc::new(tokio::sync::RwLock::new(HashMap::new()));

        // PHASE 2 FIX: Collect successful/failed claim IDs for batch status updates
        // These will be collected from parallel task results
        let mut successful_claim_ids: Vec<i64> = Vec::with_capacity(claim_ids.len());
        let mut failed_claims: Vec<(i64, i64, i32, String, String)> = Vec::new(); // (raw_claim_id, batch_id, row_number, error_message, raw_data)

        // Group raw_claims by encounter (patient_control_number + date_of_service)
        use std::collections::HashMap as StdHashMap;
        let mut encounter_groups: StdHashMap<(String, String), Vec<RawClaim>> = StdHashMap::new();

        for raw_claim in raw_claims {
            // PERFORMANCE: Extract encounter key fields directly from JsonValue
            // This avoids cloning/deserializing the entire encounter_fields object
            // which was ~50KB per claim causing significant memory churn

            let patient_control_number = match get_field_from_json(&raw_claim.encounter_fields, "patient_control_number") {
                Some(pcn) => pcn,
                None => {
                    failure_count += 1;
                    let error_message = format!("Row {}: Missing patient_control_number", raw_claim.row_number);
                    errors.push(error_message.clone());
                    error!("Missing patient_control_number for raw_claim_id {}", raw_claim.raw_claim_id);
                    continue;
                }
            };

            let date_of_service = match get_field_from_json(&raw_claim.encounter_fields, "date_of_service_from") {
                Some(dos) => dos,
                None => {
                    failure_count += 1;
                    let error_message = format!("Row {}: Missing date_of_service_from", raw_claim.row_number);
                    errors.push(error_message.clone());
                    error!("Missing date_of_service_from for raw_claim_id {}", raw_claim.raw_claim_id);
                    continue;
                }
            };

            let encounter_key = (patient_control_number, date_of_service);
            encounter_groups.entry(encounter_key).or_insert_with(Vec::new).push(raw_claim);
        }

        let encounter_count = encounter_groups.len();
        debug!("Worker {} grouped {} raw claims into {} encounters",
            worker_id, claim_ids.len(), encounter_count);

        // PERFORMANCE OPTIMIZATION: Process encounters in PARALLEL within a batch
        // This is FIFO-safe because:
        // 1. FIFO is enforced at BATCH level by SequentialCompletionManager
        // 2. Within a batch, encounters are already in arbitrary order (HashMap)
        // 3. The BatchResult is reported AFTER all encounters complete
        // 4. CompletionManager commits batches in strict sequence order

        // Semaphore limits concurrent encounters to avoid DB connection exhaustion
        let max_concurrent = get_max_concurrent_encounters();
        let semaphore = Arc::new(Semaphore::new(max_concurrent));

        // Spawn parallel tasks for each encounter
        let mut handles = Vec::with_capacity(encounter_count);

        for ((patient_control_number, date_of_service), service_lines) in encounter_groups {
            // Clone Arcs for the spawned task
            let pool = self.pool.clone();
            let facility_cache = Arc::clone(&facility_cache);
            let semaphore = Arc::clone(&semaphore);
            let processor = self.clone();
            let pcn = patient_control_number.clone();
            let dos = date_of_service.clone();

            let handle = tokio::spawn(async move {
                // Acquire semaphore permit (limits concurrent encounters)
                let _permit = semaphore.acquire().await.expect("Semaphore closed");

                debug!("Processing encounter: {} on {} ({} service lines)",
                    pcn, dos, service_lines.len());

                // Per-encounter transaction - failures don't cascade
                let tx_result = pool.begin().await;
                let mut tx = match tx_result {
                    Ok(tx) => tx,
                    Err(e) => {
                        let error_str = format!("Failed to begin transaction: {}", e);
                        error!("Failed to begin encounter transaction for {} on {}: {}", pcn, dos, e);
                        // Return failure info for all service lines
                        let failed: Vec<(i64, i64, i32, String, String)> = service_lines.iter().map(|sl| {
                            let error_message = format!("Row {}: {}", sl.row_number, error_str);
                            let raw_data = serde_json::to_string(&sl.encounter_fields).unwrap_or_default();
                            (sl.raw_claim_id, sl.batch_id, sl.row_number, error_message, raw_data)
                        }).collect();
                        return (vec![], failed, vec![error_str]);
                    }
                };

                // Process the encounter with thread-safe facility cache
                match processor.process_encounter_with_service_lines_parallel(&mut tx, &service_lines, &facility_cache).await {
                    Ok(encounter_id) => {
                        // Commit this encounter immediately
                        if let Err(e) = tx.commit().await {
                            let error_str = format!("Failed to commit transaction: {}", e);
                            error!("Failed to commit encounter {} on {}: {}", pcn, dos, e);
                            let failed: Vec<(i64, i64, i32, String, String)> = service_lines.iter().map(|sl| {
                                let error_message = format!("Row {}: {}", sl.row_number, error_str);
                                let raw_data = serde_json::to_string(&sl.encounter_fields).unwrap_or_default();
                                (sl.raw_claim_id, sl.batch_id, sl.row_number, error_message, raw_data)
                            }).collect();
                            return (vec![], failed, vec![error_str]);
                        }

                        // Collect successful claim IDs
                        let successful: Vec<i64> = service_lines.iter().map(|sl| sl.raw_claim_id).collect();

                        debug!("Successfully processed encounter: {} on {} -> encounter_id {} ({} service lines)",
                            pcn, dos, encounter_id, service_lines.len());

                        (successful, vec![], vec![])
                    }
                    Err(e) => {
                        // Rollback failed encounter (may already be rolled back by DB)
                        let _ = tx.rollback().await;

                        // Use {:#} to render the full anyhow chain (top context + underlying DB error)
                        let error_str = format!("{:#}", e);
                        error!("Failed to process encounter {} on {}: {}", pcn, dos, error_str);

                        // Collect failed claim info
                        let failed: Vec<(i64, i64, i32, String, String)> = service_lines.iter().map(|sl| {
                            let error_message = format!("Row {}: {}", sl.row_number, error_str);
                            let raw_data = serde_json::to_string(&sl.encounter_fields).unwrap_or_default();
                            (sl.raw_claim_id, sl.batch_id, sl.row_number, error_message, raw_data)
                        }).collect();

                        (vec![], failed, vec![error_str])
                    }
                }
            });

            handles.push(handle);
        }

        // Wait for ALL encounters in this batch to complete
        // This maintains FIFO ordering at the batch level
        let results = join_all(handles).await;

        // Aggregate results from all parallel tasks
        for result in results {
            match result {
                Ok((successful, failed, errs)) => {
                    success_count += successful.len();
                    successful_claim_ids.extend(successful);
                    failure_count += failed.len();
                    failed_claims.extend(failed);
                    errors.extend(errs);
                }
                Err(e) => {
                    // Task panicked - this shouldn't happen but handle it gracefully
                    error!("Encounter processing task panicked: {}", e);
                    errors.push(format!("Task panic: {}", e));
                }
            }
        }

        // PHASE 2 & 3 FIX: Batch update successful claims status (outside transactions)
        if !successful_claim_ids.is_empty() {
            sqlx::query(
                r#"
                UPDATE staging.raw_claims
                SET processing_status = 'COMPLETED',
                    processed_at = CURRENT_TIMESTAMP
                WHERE raw_claim_id = ANY($1)
                "#
            )
            .bind(&successful_claim_ids)
            .execute(&self.pool)
            .await
            .context("Failed to batch update successful claims")?;

            debug!("Batch updated {} claims to COMPLETED", successful_claim_ids.len());
        }

        // PHASE 3 FIX: Batch update failed claims and insert error logs (outside transactions)
        if !failed_claims.is_empty() {
            for (raw_claim_id, err_batch_id, row_number, error_message, raw_data) in &failed_claims {
                // Update status (fire-and-forget)
                let _ = sqlx::query(
                    r#"
                    UPDATE staging.raw_claims
                    SET processing_status = 'FAILED',
                        error_message = $2,
                        processed_at = CURRENT_TIMESTAMP
                    WHERE raw_claim_id = $1
                    "#
                )
                .bind(raw_claim_id)
                .bind(error_message)
                .execute(&self.pool)
                .await;

                // Log error (fire-and-forget)
                let _ = sqlx::query(
                    r#"
                    INSERT INTO staging.import_error_log (
                        batch_id,
                        record_number,
                        error_type,
                        error_severity,
                        error_message,
                        raw_data
                    )
                    VALUES ($1, $2, 'VALIDATION', 'ERROR', $3, $4)
                    "#
                )
                .bind(err_batch_id)
                .bind(row_number)
                .bind(error_message)
                .bind(raw_data)
                .execute(&self.pool)
                .await;
            }

            debug!("Batch updated {} claims to FAILED with error logs", failed_claims.len());
        }

        let batch_end = chrono::Utc::now();
        let processing_time = (batch_end - batch_start).num_milliseconds() as f64 / 1000.0;

        info!(
            "Worker {} completed batch {} in {:.2}s ({} success, {} failed)",
            worker_id, sequence_number, processing_time, success_count, failure_count
        );

        // Log processing metrics
        let total_records = (success_count + failure_count) as i32;
        let details = serde_json::json!({
            "worker_id": worker_id,
            "sequence_number": sequence_number,
            "batch_size": claim_ids.len(),
            "processing_time_seconds": processing_time
        });

        if let Err(e) = self.log_processing_metric(
            batch_id,
            "batch_processing",
            "sequenced_batch_stage2",
            batch_start,
            batch_end,
            total_records,
            success_count as i32,
            failure_count as i32,
            Some(details),
            "STAGE2"
        ).await {
            warn!("Failed to log processing metric for batch {}: {}", sequence_number, e);
        }

        Ok(crate::batch_sequencer::BatchResult {
            sequence_number,
            batch_id,
            success_count,
            failure_count,
            processing_time_seconds: processing_time,
            worker_id,
            errors,
        })
    }

    /// PERFORMANCE OPTIMIZATION: Batch pre-warm provider cache for an encounter
    ///
    /// Collects all unique NPIs from encounter + service lines and queries existing
    /// providers in a single batch query. This pre-populates the cache so that
    /// Pre-warm provider cache and CREATE all providers for this encounter
    ///
    /// PERFORMANCE OPTIMIZATION v2.12.73.86:
    /// This function now handles ALL provider creation for an encounter in batch.
    /// - Collects provider data (NPI, type, names, taxonomy) from encounter and service lines
    /// - Batch queries existing providers (1 query)
    /// - Batch inserts new providers with metadata (1 query)
    /// - Batch queues providers for enrichment (1 query)
    ///
    /// After this call, ensure_provider_exists() becomes a pure cache lookup.
    /// This eliminates 16+ sequential DB operations per encounter.
    async fn prewarm_provider_cache(
        &self,
        _tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
        encounter_fields: &EncounterFieldsWrapper,
        service_lines: &[RawClaim],
    ) -> Result<()> {
        // PROVIDER CACHE RACE FIX:
        // Provider upserts run in a DEDICATED short-lived transaction on the pool,
        // NOT in the encounter transaction passed via `_tx`. This guarantees that any
        // provider_id we insert into the shared `provider_cache` is already committed
        // and visible to other concurrent encounter transactions when they reference
        // it as a foreign key. Using the encounter tx here previously caused FK
        // violations like `service_line_ordering_provider_id_fkey` because parallel
        // encounter tasks would read cached IDs from each other's uncommitted txs.

        let mut providers: HashMap<String, ProviderData> = HashMap::new();
        Self::collect_providers_from_encounter(&mut providers, encounter_fields);
        for raw_claim in service_lines {
            Self::collect_providers_from_service_lines(&mut providers, raw_claim);
        }

        self.upsert_providers_in_own_tx(providers).await
    }

    /// Batch-level provider prewarm: collect ALL provider NPIs from every raw_claim
    /// in the batch and upsert them in a single dedicated transaction BEFORE any
    /// per-encounter task starts. Combined with `upsert_providers_in_own_tx` this
    /// guarantees the shared provider_cache only ever holds committed provider_ids,
    /// eliminating the cross-transaction FK race for parallel encounter processing.
    async fn prewarm_provider_cache_for_batch(&self, raw_claims: &[RawClaim]) -> Result<()> {
        let mut providers: HashMap<String, ProviderData> = HashMap::new();
        for raw_claim in raw_claims {
            let encounter_fields = EncounterFieldsWrapper::new(raw_claim.encounter_fields.clone())
                .context("Failed to deserialize encounter_fields for batch prewarm")?;
            Self::collect_providers_from_encounter(&mut providers, &encounter_fields);
            Self::collect_providers_from_service_lines(&mut providers, raw_claim);
        }

        if providers.is_empty() {
            return Ok(());
        }

        debug!("Batch provider prewarm: {} distinct NPIs across {} raw_claims",
            providers.len(), raw_claims.len());
        self.upsert_providers_in_own_tx(providers).await
    }

    /// Collect encounter-level provider NPIs into `providers`.
    fn collect_providers_from_encounter(
        providers: &mut HashMap<String, ProviderData>,
        encounter_fields: &EncounterFieldsWrapper,
    ) {
        let mut add = |npi: Option<String>, ptype: &str, last: Option<String>, first: Option<String>, taxonomy: Option<String>| {
            if let Some(npi) = npi {
                if npi.len() == 10 && npi.chars().all(|c| c.is_ascii_digit()) {
                    providers.entry(npi.clone()).or_insert(ProviderData {
                        npi,
                        provider_type: ptype.to_string(),
                        last_name: last.unwrap_or_else(|| "Unknown".to_string()),
                        first_name: first.unwrap_or_default(),
                        taxonomy_code: taxonomy,
                    });
                }
            }
        };

        add(
            encounter_fields.get("rendering_provider_npi"),
            "Rendering",
            encounter_fields.get("rendering_provider_last_name"),
            encounter_fields.get("rendering_provider_first_name"),
            encounter_fields.get("rendering_provider_taxonomy"),
        );
        add(
            encounter_fields.get("referring_provider_npi"),
            "Referring",
            encounter_fields.get("referring_provider_last_name"),
            encounter_fields.get("referring_provider_first_name"),
            encounter_fields.get("referring_provider_taxonomy"),
        );
        add(
            encounter_fields.get("supervising_provider_npi"),
            "Supervising",
            encounter_fields.get("supervising_provider_last_name"),
            encounter_fields.get("supervising_provider_first_name"),
            encounter_fields.get("supervising_provider_taxonomy"),
        );
        let billing_name = encounter_fields.get("billing_provider_name");
        let (billing_last, billing_first) = if let Some(ref name) = billing_name {
            if name.contains(',') {
                let parts: Vec<&str> = name.splitn(2, ',').collect();
                (Some(parts[0].trim().to_string()), parts.get(1).map(|s| s.trim().to_string()))
            } else {
                (Some(name.clone()), None)
            }
        } else {
            (None, None)
        };
        add(
            encounter_fields.get("billing_provider_npi"),
            "Billing",
            billing_last,
            billing_first,
            None,
        );
    }

    /// Collect service-line-level provider NPIs (rendering / ordering / supervising / referring,
    /// across the 12 possible service_line_N_ prefixes).
    fn collect_providers_from_service_lines(
        providers: &mut HashMap<String, ProviderData>,
        raw_claim: &RawClaim,
    ) {
        let Some(slf) = &raw_claim.service_line_fields else { return; };
        let Ok(fields) = serde_json::from_value::<HashMap<String, String>>(slf.clone()) else { return; };

        let mut add = |npi: Option<String>, ptype: &str, last: Option<String>, first: Option<String>| {
            if let Some(npi) = npi {
                if npi.len() == 10 && npi.chars().all(|c| c.is_ascii_digit()) {
                    providers.entry(npi.clone()).or_insert(ProviderData {
                        npi,
                        provider_type: ptype.to_string(),
                        last_name: last.unwrap_or_else(|| "Unknown".to_string()),
                        first_name: first.unwrap_or_default(),
                        taxonomy_code: None,
                    });
                }
            }
        };

        for prefix_num in 1..=12 {
            let prefix = format!("service_line_{}_", prefix_num);
            add(
                fields.get(&format!("{}rendering_provider_npi", prefix)).cloned(),
                "Rendering",
                fields.get(&format!("{}rendering_provider_last_name", prefix)).cloned(),
                fields.get(&format!("{}rendering_provider_first_name", prefix)).cloned(),
            );
            add(
                fields.get(&format!("{}ordering_provider_npi", prefix)).cloned(),
                "Ordering",
                fields.get(&format!("{}ordering_provider_last_name", prefix)).cloned(),
                fields.get(&format!("{}ordering_provider_first_name", prefix)).cloned(),
            );
            add(
                fields.get(&format!("{}supervising_provider_npi", prefix)).cloned(),
                "Supervising",
                fields.get(&format!("{}supervising_provider_last_name", prefix)).cloned(),
                fields.get(&format!("{}supervising_provider_first_name", prefix)).cloned(),
            );
            add(
                fields.get(&format!("{}referring_provider_npi", prefix)).cloned(),
                "Referring",
                fields.get(&format!("{}referring_provider_last_name", prefix)).cloned(),
                fields.get(&format!("{}referring_provider_first_name", prefix)).cloned(),
            );
        }
    }

    /// Upsert providers in their own short-lived transaction on `self.pool` and only
    /// after that transaction commits, write the resulting provider_ids into the
    /// shared provider_cache. Failures are propagated (no `unwrap_or_default`) so a
    /// broken upsert doesn't leave a poisoned/aborted transaction behind.
    async fn upsert_providers_in_own_tx(
        &self,
        providers: HashMap<String, ProviderData>,
    ) -> Result<()> {
        if providers.is_empty() {
            return Ok(());
        }

        // Skip NPIs we already know about (already cached = already committed).
        let providers_to_process: Vec<ProviderData> = {
            let cache = self.provider_cache.read().await;
            providers.into_iter()
                .filter(|(npi, _)| !cache.contains_key(npi))
                .map(|(_, data)| data)
                .collect()
        };
        if providers_to_process.is_empty() {
            return Ok(());
        }

        // Dedicated short-lived transaction - independent of any encounter tx.
        let mut tx = self.pool.begin().await
            .context("Failed to begin provider upsert transaction")?;

        let npis_to_query: Vec<&str> = providers_to_process.iter()
            .map(|p| p.npi.as_str())
            .collect();

        let existing_providers: Vec<(String, i64)> = sqlx::query_as(
            r#"
            SELECT npi, provider_id
            FROM claims.provider
            WHERE npi = ANY($1)
            "#
        )
        .bind(&npis_to_query)
        .fetch_all(&mut *tx)
        .await
        .context("Failed to query existing providers during prewarm")?;

        let existing_npis: std::collections::HashSet<String> = existing_providers.iter()
            .map(|(npi, _)| npi.clone())
            .collect();

        let new_providers: Vec<&ProviderData> = providers_to_process.iter()
            .filter(|p| !existing_npis.contains(&p.npi))
            .collect();

        let mut inserted_providers: Vec<(String, i64)> = Vec::new();
        if !new_providers.is_empty() {
            let mut specialties: Vec<Option<String>> = Vec::with_capacity(new_providers.len());
            for provider in &new_providers {
                if let Some(ref tax_code) = provider.taxonomy_code {
                    let (_, spec) = self.lookup_taxonomy(tax_code).await;
                    specialties.push(spec);
                } else {
                    specialties.push(None);
                }
            }

            let npis: Vec<&str> = new_providers.iter().map(|p| p.npi.as_str()).collect();
            let types: Vec<&str> = new_providers.iter().map(|p| p.provider_type.as_str()).collect();
            let last_names: Vec<&str> = new_providers.iter().map(|p| p.last_name.as_str()).collect();
            let first_names: Vec<&str> = new_providers.iter().map(|p| p.first_name.as_str()).collect();
            let taxonomies: Vec<Option<&str>> = new_providers.iter()
                .map(|p| p.taxonomy_code.as_deref())
                .collect();
            let specialty_refs: Vec<Option<&str>> = specialties.iter()
                .map(|s| s.as_deref())
                .collect();

            inserted_providers = sqlx::query_as(
                r#"
                INSERT INTO claims.provider (
                    npi, provider_type, last_name, first_name, taxonomy_code, specialty,
                    is_active, created_at, updated_at
                )
                SELECT
                    unnest($1::text[]),
                    unnest($2::text[]),
                    unnest($3::text[]),
                    unnest($4::text[]),
                    unnest($5::text[]),
                    unnest($6::text[]),
                    true,
                    CURRENT_TIMESTAMP,
                    CURRENT_TIMESTAMP
                ON CONFLICT (npi) DO UPDATE SET updated_at = CURRENT_TIMESTAMP
                RETURNING npi, provider_id
                "#
            )
            .bind(&npis)
            .bind(&types)
            .bind(&last_names)
            .bind(&first_names)
            .bind(&taxonomies)
            .bind(&specialty_refs)
            .fetch_all(&mut *tx)
            .await
            .context("Failed to batch insert providers during prewarm")?;

            if !inserted_providers.is_empty() {
                let provider_ids: Vec<i64> = inserted_providers.iter().map(|(_, id)| *id).collect();
                let provider_npis: Vec<&str> = inserted_providers.iter().map(|(npi, _)| npi.as_str()).collect();
                sqlx::query(
                    r#"
                    INSERT INTO claims.provider_enrichment_queue (provider_id, npi, priority)
                    SELECT unnest($1::bigint[]), unnest($2::text[]), 5
                    ON CONFLICT (provider_id) DO NOTHING
                    "#
                )
                .bind(&provider_ids)
                .bind(&provider_npis)
                .execute(&mut *tx)
                .await
                .context("Failed to enqueue providers for enrichment during prewarm")?;
            }
        }

        // Commit BEFORE writing to the shared cache so cache entries always reference
        // committed provider rows visible to other transactions.
        tx.commit().await
            .context("Failed to commit provider upsert transaction")?;

        if !existing_providers.is_empty() || !inserted_providers.is_empty() {
            let mut cache = self.provider_cache.write().await;
            for (npi, provider_id) in &existing_providers {
                cache.insert(npi.clone(), *provider_id);
            }
            for (npi, provider_id) in &inserted_providers {
                cache.insert(npi.clone(), *provider_id);
            }
            debug!(
                "Provider prewarm: cached {} existing + {} new providers",
                existing_providers.len(), inserted_providers.len()
            );
        }

        Ok(())
    }

    /// Get provider_id from cache (CACHE-ONLY - no DB operations)
    /// Returns the provider_id if found in cache, None otherwise
    ///
    /// PERFORMANCE OPTIMIZATION v2.12.73.86:
    /// This function is now CACHE-ONLY. All provider creation happens in
    /// prewarm_provider_cache() which batch-inserts all providers for an encounter
    /// in a single query. This eliminates 16+ sequential DB operations per encounter.
    ///
    /// If a provider NPI is not in cache, it means prewarm_provider_cache() didn't
    /// find/create it, and the encounter proceeds with NULL provider_id (supported).
    #[allow(unused_variables)]
    async fn ensure_provider_exists(
        &self,
        tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
        npi: &str,
        provider_type: &str,
        last_name: Option<&str>,
        first_name: Option<&str>,
        middle_name: Option<&str>,
        taxonomy_code: Option<&str>,
        organization_id: Option<i64>,
    ) -> Result<Option<i64>> {
        // Skip if NPI is empty
        if npi.is_empty() {
            return Ok(None);
        }

        // Validate NPI format (10 digits)
        if npi.len() != 10 || !npi.chars().all(|c| c.is_ascii_digit()) {
            debug!("Invalid NPI format: {} (expected 10 digits)", npi);
            return Ok(None);
        }

        // PERFORMANCE: Cache-only lookup - no DB operations
        // All providers were pre-created by prewarm_provider_cache()
        let cache = self.provider_cache.read().await;
        if let Some(&provider_id) = cache.get(npi) {
            return Ok(Some(provider_id));
        }

        // Provider not in cache - this shouldn't happen if prewarm_provider_cache worked
        // Return None and log at debug level (not warn - this is expected for invalid NPIs)
        debug!("Provider NPI {} not found in cache after prewarm", npi);
        Ok(None)
    }

    /// Execute rules for service lines and persist flags (OPTIMIZED)
    ///
    /// PERFORMANCE OPTIMIZATIONS:
    /// - Takes pre-collected service line data (no extra DB queries)
    /// - Shares diagnosis codes reference (no cloning per service line)
    /// - Batch inserts flags (single query for all flags)
    /// - Pre-caches issue_id lookups
    /// - Only acquires rule_engine read lock once
    async fn execute_rules_for_service_lines(
        &self,
        tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
        encounter_id: i64,
        organization_id: i64,
        service_line_contexts: &[ServiceLineRuleContext],
        diagnosis_codes: &[String],
    ) -> Result<usize> {
        if service_line_contexts.is_empty() {
            return Ok(0);
        }

        // PERFORMANCE OPTIMIZATION: Direct Arc access - no lock acquisition needed
        // Rules are loaded once at startup and never modified during runtime
        let rule_engine = &self.rule_engine;

        // Skip if no rules loaded
        let rule_count = rule_engine.rule_count();
        if rule_count == 0 {
            debug!("RULES: No rules loaded, skipping rule execution for encounter {}", encounter_id);
            return Ok(0);
        }

        // PERFORMANCE: Use debug! - this runs for every encounter (thousands per batch)
        debug!(
            "RULES: Executing {} rules for encounter {} with {} service lines",
            rule_count, encounter_id, service_line_contexts.len()
        );

        // Collect all flags to batch insert
        // Tuple: (service_line_id, issue_code, flag_reason, severity)
        let mut flags_to_insert: Vec<(i64, String, String, String)> = Vec::new();

        // Execute rules for each service line
        for sl_ctx in service_line_contexts {
            // Build rule execution context (stack allocated, no heap for small vecs)
            let mut ctx = RuleExecutionContext::new(organization_id);
            ctx.encounter_id = Some(encounter_id);
            ctx.service_line_id = Some(sl_ctx.service_line_id);
            ctx.procedure_code = Some(sl_ctx.procedure_code.clone());
            ctx.procedure_modifiers = sl_ctx.modifiers.clone();
            ctx.service_unit_count = Some(sl_ctx.units);
            ctx.line_item_charge_amount = Some(sl_ctx.charge);
            ctx.date_of_service = Some(sl_ctx.service_date);
            ctx.place_of_service_code = sl_ctx.place_of_service.clone();
            // Share reference to diagnosis codes (no clone)
            ctx.diagnosis_codes = diagnosis_codes.to_vec();

            // PERFORMANCE: Pre-compute uppercase values ONCE before executing 537 rules
            // This avoids thousands of to_uppercase() allocations in the hot loop
            ctx.finalize();

            // Execute rules for this service line using CPT indexing
            // CPT indexing skips rules that don't match this service line's CPT code
            // Analysis: 492/537 rules (91.6%) have cpt_in conditions and are indexable
            // Only 45 universal rules (no cpt_in) run on every service line
            // Expected: ~80% reduction in rule evaluations (1,611 -> ~315 per claim)
            //
            // PERFORMANCE: Use direct sync execution to eliminate async overhead
            // With 543 composite rules (all sync-capable), calling the async wrapper
            // for 30,000 service lines adds 9-30 seconds of pure scheduler overhead.
            // Direct sync call eliminates this entirely.
            let results = rule_engine.execute_all_indexed_sync(&ctx);
            match results {
                Ok(results) => {
                    // PERFORMANCE: Use debug! instead of info! - this runs millions of times with 500+ rules
                    debug!(
                        "RULES: service_line {} (CPT={}) triggered {} rules",
                        sl_ctx.service_line_id, sl_ctx.procedure_code, results.len()
                    );
                    for result in results {
                        // PERFORMANCE: Minimize allocations - with 50+ flags per claim, this adds up
                        // Use issue_code directly if available, fall back to flag_type.code()
                        let flag_code = result.issue_code
                            .unwrap_or_else(|| result.flag_type.code().to_string());
                        let severity = result.severity.as_str().to_string();
                        // Simplified flag_reason - just use description (details already included)
                        let flag_reason = result.details.unwrap_or(result.description);

                        flags_to_insert.push((sl_ctx.service_line_id, flag_code, flag_reason, severity));
                    }
                }
                Err(e) => {
                    warn!("Error executing rules for service_line {}: {}", sl_ctx.service_line_id, e);
                }
            }
        }

        // Early return if no flags
        if flags_to_insert.is_empty() {
            // PERFORMANCE: Use debug! instead of info! - runs for every encounter without flags
            debug!("RULES: No flags collected for encounter {}", encounter_id);
            return Ok(0);
        }

        // Debug: Log flags being inserted
        debug!(
            "Attempting to insert {} flags for encounter {}: {:?}",
            flags_to_insert.len(),
            encounter_id,
            flags_to_insert.iter().map(|(sl, code, _, _)| (sl, code.as_str())).collect::<Vec<_>>()
        );

        // BATCH INSERT: Single query for all flags using UNNEST
        // This is ~10x faster than individual inserts
        let service_line_ids: Vec<i64> = flags_to_insert.iter().map(|(id, _, _, _)| *id).collect();
        let issue_codes: Vec<&str> = flags_to_insert.iter().map(|(_, code, _, _)| code.as_str()).collect();
        let flag_reasons: Vec<&str> = flags_to_insert.iter().map(|(_, _, reason, _)| reason.as_str()).collect();
        let severities: Vec<&str> = flags_to_insert.iter().map(|(_, _, _, sev)| sev.as_str()).collect();

        let rows_inserted = sqlx::query_scalar::<_, i64>(
            r#"
            WITH flag_data AS (
                SELECT
                    unnest($1::bigint[]) as service_line_id,
                    unnest($2::text[]) as issue_code,
                    unnest($3::text[]) as flag_reason,
                    unnest($4::text[]) as severity
            )
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
            SELECT
                fd.service_line_id,
                fi.issue_id,
                'POST_BILL',
                fd.severity,
                fd.flag_reason,
                fd.issue_code,
                'OPEN',
                CURRENT_TIMESTAMP,
                'RULES_ENGINE'
            FROM flag_data fd
            JOIN claims.flag_issue fi ON fi.issue_code = fd.issue_code
            ON CONFLICT (service_line_id, issue_id) WHERE flag_status = 'OPEN'
            DO NOTHING
            RETURNING flag_id
            "#
        )
        .bind(&service_line_ids)
        .bind(&issue_codes)
        .bind(&flag_reasons)
        .bind(&severities)
        .fetch_all(&mut **tx)
        .await
        .map(|rows| rows.len())
        .unwrap_or(0);

        if rows_inserted > 0 {
            info!("RULES: Inserted {} flag(s) for encounter {}", rows_inserted, encounter_id);
        } else if !flags_to_insert.is_empty() {
            // Flags were collected but none inserted - likely issue_code mismatch with flag_issue table
            warn!(
                "RULES: {} flags collected but 0 inserted for encounter {}. Issue codes may not exist in flag_issue table: {:?}",
                flags_to_insert.len(),
                encounter_id,
                issue_codes
            );
        }

        Ok(rows_inserted)
    }
}

/// Service line context for rule execution (avoids re-querying DB)
#[derive(Debug, Clone)]
pub struct ServiceLineRuleContext {
    pub service_line_id: i64,
    pub procedure_code: String,
    pub modifiers: Vec<String>,
    pub units: rust_decimal::Decimal,
    pub charge: rust_decimal::Decimal,
    pub service_date: chrono::NaiveDate,
    pub place_of_service: Option<String>,
}

/// Raw claim from staging.raw_claims table
#[derive(Debug, Clone, sqlx::FromRow)]
struct RawClaim {
    raw_claim_id: i64,
    batch_id: i64,
    queue_id: i64,
    encounter_fields: JsonValue,
    service_line_fields: Option<JsonValue>,
    diagnosis_fields: Option<JsonValue>,
    row_number: i32,
    facility_code: Option<String>,
    date_of_service_from: Option<chrono::NaiveDate>,
}

/// Result of Stage 2 processing
#[derive(Debug, Clone)]
pub struct ProcessResult {
    pub total_processed: usize,
    pub successful: usize,
    pub failed: usize,
}

impl ProcessResult {
    /// Check if processing was completely successful
    pub fn is_success(&self) -> bool {
        self.failed == 0 && self.successful > 0
    }

    /// Get a summary string
    pub fn summary(&self) -> String {
        format!(
            "Processing complete: {} total, {} successful, {} failed",
            self.total_processed, self.successful, self.failed
        )
    }
}
