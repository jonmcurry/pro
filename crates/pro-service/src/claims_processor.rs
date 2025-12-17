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
use pro_common::DEFAULT_DATE;
use serde_json::Value as JsonValue;
use sqlx::PgPool;
use std::collections::HashMap;
use tracing::{debug, error, info, warn};

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


/// Claims processor for Stage 2 of two-stage pipeline
#[derive(Clone)]
pub struct ClaimsProcessor {
    pool: PgPool,
}

impl ClaimsProcessor {
    /// Create a new claims processor
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Process pending claims from staging.raw_claims (STAGE 2)
    /// This method processes claims that were ingested in Stage 1
    /// Performance target: 10,000 claims / 15 seconds = 666.67 claims/sec
    pub async fn process_pending_claims(&self, limit: Option<usize>) -> Result<ProcessResult> {
        let limit = limit.unwrap_or(10000); // Default batch of 10k claims

        info!("====== STAGE 2: Starting processing of pending raw claims (limit: {}) ======", limit);

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

        // Begin transaction for batch processing
        let mut tx = self.pool.begin().await
            .context("Failed to begin database transaction")?;

        // Batch commit configuration
        const BATCH_SIZE: usize = 1000;
        let mut batch_count = 0;

        // Facility lookup cache for performance
        let mut facility_cache: HashMap<String, (Option<i64>, i64, Option<i64>)> = HashMap::new();

        info!("Processing {} raw claims...", raw_claims.len());

        // Group raw_claims by encounter (patient_control_number + date_of_service)
        use std::collections::HashMap as StdHashMap;
        let mut encounter_groups: StdHashMap<(String, String), Vec<RawClaim>> = StdHashMap::new();

        for raw_claim in raw_claims {
            // Extract encounter key from encounter_fields
            // Use JsonValue to handle mixed types (strings, arrays like other_insurance)
            let encounter_fields: StdHashMap<String, JsonValue> = match serde_json::from_value(raw_claim.encounter_fields.clone()) {
                Ok(fields) => fields,
                Err(e) => {
                    error!("Failed to deserialize encounter_fields for raw_claim_id {}: {}", raw_claim.raw_claim_id, e);
                    result.failed += 1;
                    continue;
                }
            };

            let patient_control_number = match get_field_as_string(&encounter_fields, "patient_control_number") {
                Some(pcn) => pcn,
                None => {
                    error!("Missing patient_control_number for raw_claim_id {}", raw_claim.raw_claim_id);
                    result.failed += 1;
                    continue;
                }
            };

            let date_of_service = match get_field_as_string(&encounter_fields, "date_of_service_from") {
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

        // Process each encounter group
        for ((patient_control_number, date_of_service), service_lines) in encounter_groups {
            debug!("Processing encounter: {} on {} ({} service lines)",
                patient_control_number, date_of_service, service_lines.len());

            // Validate and insert encounter with all service lines
            match self.process_encounter_with_service_lines(&mut tx, service_lines.clone(), &mut facility_cache).await {
                Ok(encounter_id) => {
                    result.successful += service_lines.len();
                    batch_count += service_lines.len();

                    // Mark all raw_claims in this encounter as COMPLETED
                    for service_line in &service_lines {
                        sqlx::query(
                            r#"
                            UPDATE staging.raw_claims
                            SET processing_status = 'COMPLETED',
                                processed_at = CURRENT_TIMESTAMP
                            WHERE raw_claim_id = $1
                            "#
                        )
                        .bind(service_line.raw_claim_id)
                        .execute(&mut *tx)
                        .await?;
                    }

                    debug!("Successfully processed encounter: {} on {} -> encounter_id {}",
                        patient_control_number, date_of_service, encounter_id);

                    // Commit batch every BATCH_SIZE rows
                    if batch_count >= BATCH_SIZE {
                        debug!("Committing batch of {} claims", batch_count);
                        tx.commit().await
                            .context("Failed to commit batch transaction")?;
                        info!("Committed batch: {} claims processed so far", result.successful);

                        // Start new transaction
                        tx = self.pool.begin().await
                            .context("Failed to begin new batch transaction")?;
                        batch_count = 0;
                    }
                }
                Err(e) => {
                    result.failed += service_lines.len();
                    error!("Failed to process encounter {} on {}: {}", patient_control_number, date_of_service, e);

                    // Mark all raw_claims in this encounter as FAILED
                    for service_line in &service_lines {
                        let error_message = format!("Row {}: {}", service_line.row_number, e);

                        sqlx::query(
                            r#"
                            UPDATE staging.raw_claims
                            SET processing_status = 'FAILED',
                                error_message = $2
                            WHERE raw_claim_id = $1
                            "#
                        )
                        .bind(service_line.raw_claim_id)
                        .bind(&error_message)
                        .execute(&mut *tx)
                        .await?;

                        // Log error to staging.import_error_log
                        let error_log_id: i64 = sqlx::query_scalar(
                            r#"
                            INSERT INTO staging.import_error_log (
                                batch_id,
                                record_number,
                                error_type,
                                error_severity,
                                error_message,
                                raw_data
                            )
                            VALUES ($1, $2, $3, $4, $5, $6)
                            RETURNING error_id
                            "#
                        )
                        .bind(service_line.batch_id)
                        .bind(service_line.row_number)
                        .bind("VALIDATION")
                        .bind("ERROR")
                        .bind(&error_message)
                        .bind(serde_json::to_string(&service_line.encounter_fields).ok())
                        .fetch_one(&mut *tx)
                        .await?;
                    }

                    batch_count += service_lines.len();

                    // Commit batch every BATCH_SIZE rows
                    if batch_count >= BATCH_SIZE {
                        debug!("Committing batch of {} claims (with errors)", batch_count);
                        tx.commit().await
                            .context("Failed to commit batch transaction")?;
                        info!("Committed batch: {} claims processed, {} failed so far",
                            result.successful, result.failed);

                        // Start new transaction
                        tx = self.pool.begin().await
                            .context("Failed to begin new batch transaction")?;
                        batch_count = 0;
                    }
                }
            }
        }

        // Commit final batch
        tx.commit().await
            .context("Failed to commit final batch transaction")?;

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
    async fn process_encounter_with_service_lines(
        &self,
        tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
        service_lines: Vec<RawClaim>,
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
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "1900-01-01".to_string()); // Default to 1900-01-01 if missing or empty

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

        // Calculate total claim charge from ALL service lines
        let mut total_claim_charge = rust_decimal::Decimal::ZERO;
        for service_line in &service_lines {
            if let Some(slf_value) = &service_line.service_line_fields {
                if let Ok(slf) = serde_json::from_value::<HashMap<String, String>>(slf_value.clone()) {
                    if let Some(charge_str) = slf.get("line_item_charge_amount") {
                        if let Ok(charge) = charge_str.parse::<rust_decimal::Decimal>() {
                            total_claim_charge += charge;
                        }
                    }
                }
            }
        }

        // Parse dates
        let dos_from = chrono::NaiveDate::parse_from_str(&date_of_service_from, "%Y-%m-%d")
            .context("Invalid date format for date_of_service_from")?;
        let subscriber_dob = chrono::NaiveDate::parse_from_str(&subscriber_birth_date_str, "%Y-%m-%d")
            .unwrap_or(*DEFAULT_DATE); // Fallback to 1900-01-01

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
        let rendering_provider_npi = encounter_fields.get("rendering_provider_npi").filter(|s| !s.is_empty());
        let rendering_provider_last_name = encounter_fields.get("rendering_provider_last_name").filter(|s| !s.is_empty());
        let rendering_provider_first_name = encounter_fields.get("rendering_provider_first_name").filter(|s| !s.is_empty());
        let rendering_provider_taxonomy = encounter_fields.get("rendering_provider_taxonomy").filter(|s| !s.is_empty());

        let referring_provider_npi = encounter_fields.get("referring_provider_npi").filter(|s| !s.is_empty());
        let referring_provider_last_name = encounter_fields.get("referring_provider_last_name").filter(|s| !s.is_empty());
        let referring_provider_first_name = encounter_fields.get("referring_provider_first_name").filter(|s| !s.is_empty());

        let supervising_provider_npi = encounter_fields.get("supervising_provider_npi").filter(|s| !s.is_empty());
        let supervising_provider_last_name = encounter_fields.get("supervising_provider_last_name").filter(|s| !s.is_empty());
        let supervising_provider_first_name = encounter_fields.get("supervising_provider_first_name").filter(|s| !s.is_empty());

        let billing_provider_npi = encounter_fields.get("billing_provider_npi").filter(|s| !s.is_empty());
        let billing_provider_name = encounter_fields.get("billing_provider_name").filter(|s| !s.is_empty());

        // Service Facility information (Loop 2310C - NM1*77, N3, N4)
        let service_facility_npi = encounter_fields.get("service_facility_npi").filter(|s| !s.is_empty());
        let service_facility_name = encounter_fields.get("service_facility_name").filter(|s| !s.is_empty());
        let service_facility_address_line1 = encounter_fields.get("service_facility_address_line1").filter(|s| !s.is_empty());
        let service_facility_address_line2 = encounter_fields.get("service_facility_address_line2").filter(|s| !s.is_empty());
        let service_facility_city = encounter_fields.get("service_facility_city").filter(|s| !s.is_empty());
        let service_facility_state = encounter_fields.get("service_facility_state").filter(|s| !s.is_empty());
        let service_facility_postal_code = encounter_fields.get("service_facility_postal_code").filter(|s| !s.is_empty());

        // Log service facility fields for debugging
        info!("Service facility from encounter_fields: npi={:?}, name={:?}, addr1={:?}, city={:?}, state={:?}",
            service_facility_npi, service_facility_name, service_facility_address_line1,
            service_facility_city, service_facility_state);

        // Ensure providers exist in claims.provider table and get their provider_ids
        let rendering_provider_id = if let Some(ref npi) = rendering_provider_npi {
            match self.ensure_provider_exists(
                tx,
                npi,
                "Rendering",
                rendering_provider_last_name.as_deref(),
                rendering_provider_first_name.as_deref(),
                None,
                rendering_provider_taxonomy.as_deref(),
                Some(organization_id),
            ).await {
                Ok(provider_id) => {
                    if provider_id.is_none() {
                        warn!("Failed to create/find rendering provider: NPI={}, Name={} {}",
                            npi,
                            rendering_provider_first_name.as_deref().unwrap_or(""),
                            rendering_provider_last_name.as_deref().unwrap_or(""));
                    }
                    provider_id
                },
                Err(e) => {
                    error!("Error ensuring rendering provider exists: NPI={}, Error={:?}", npi, e);
                    None
                }
            }
        } else {
            None
        };

        let referring_provider_id = if let Some(ref npi) = referring_provider_npi {
            match self.ensure_provider_exists(
                tx,
                npi,
                "Referring",
                referring_provider_last_name.as_deref(),
                referring_provider_first_name.as_deref(),
                None,
                None,
                Some(organization_id),
            ).await {
                Ok(provider_id) => {
                    if provider_id.is_none() {
                        warn!("Failed to create/find referring provider: NPI={}, Name={} {}",
                            npi,
                            referring_provider_first_name.as_deref().unwrap_or(""),
                            referring_provider_last_name.as_deref().unwrap_or(""));
                    }
                    provider_id
                },
                Err(e) => {
                    error!("Error ensuring referring provider exists: NPI={}, Error={:?}", npi, e);
                    None
                }
            }
        } else {
            None
        };

        let supervising_provider_id = if let Some(ref npi) = supervising_provider_npi {
            match self.ensure_provider_exists(
                tx,
                npi,
                "Supervising",
                supervising_provider_last_name.as_deref(),
                supervising_provider_first_name.as_deref(),
                None,
                None,
                Some(organization_id),
            ).await {
                Ok(provider_id) => {
                    if provider_id.is_none() {
                        warn!("Failed to create/find supervising provider: NPI={}, Name={} {}",
                            npi,
                            supervising_provider_first_name.as_deref().unwrap_or(""),
                            supervising_provider_last_name.as_deref().unwrap_or(""));
                    }
                    provider_id
                },
                Err(e) => {
                    error!("Error ensuring supervising provider exists: NPI={}, Error={:?}", npi, e);
                    None
                }
            }
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

            match self.ensure_provider_exists(
                tx,
                npi,
                "Billing",
                last,
                first,
                None,
                None,
                Some(organization_id),
            ).await {
                Ok(provider_id) => {
                    if provider_id.is_none() {
                        warn!("Failed to create/find billing provider: NPI={}, Name={:?}",
                            npi, billing_provider_name);
                    }
                    provider_id
                },
                Err(e) => {
                    error!("Error ensuring billing provider exists: NPI={}, Error={:?}", npi, e);
                    None
                }
            }
        } else {
            None
        };

        // Check for existing encounter (same claim resubmitted)
        // Match on: patient_control_number + facility_id + date_of_service_from
        // If exists, UPDATE payer fields and add to encounter_payer table
        let existing_encounter: Option<i64> = sqlx::query_scalar(
            r#"
            SELECT encounter_id
            FROM claims.encounter
            WHERE patient_control_number = $1
              AND facility_id = $2
              AND date_of_service_from = $3
              AND soft_deleted = false
            LIMIT 1
            "#
        )
        .bind(&patient_control_number)
        .bind(facility_id)
        .bind(dos_from)
        .fetch_optional(&mut **tx)
        .await?;

        if let Some(existing_id) = existing_encounter {
            info!("[RESUBMIT] Encounter exists: encounter_id={}, updating payer info. patient_control_number={}, payer_id={:?}, payer_name={:?}",
                existing_id, &patient_control_number, &payer_id, &payer_name);

            // Update encounter with new billing payer info
            sqlx::query(
                r#"
                UPDATE claims.encounter
                SET payer_responsibility_code = $1,
                    payer_id = $2,
                    payer_name = $3,
                    claim_filing_indicator = $4,
                    updated_at = CURRENT_TIMESTAMP
                WHERE encounter_id = $5
                "#
            )
            .bind(payer_responsibility_code)
            .bind(payer_id.as_deref())
            .bind(payer_name.as_deref())
            .bind(claim_filing_indicator.as_deref())
            .bind(existing_id)
            .execute(&mut **tx)
            .await
            .context("Failed to update encounter payer info")?;

            // Insert billing payer into encounter_payer table
            self.insert_encounter_payer(
                tx,
                existing_id,
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
            self.import_encounter_payers_from_cob(tx, existing_id, encounter_fields.inner(), billing_provider_id).await?;

            return Ok(existing_id);
        }

        // Insert encounter and get generated ID (ONE record for all service lines)
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
                subscriber_birth_date,
                date_of_service_from,
                date_of_service_to,
                total_claim_charge_amount,
                payer_id,
                payer_name,
                payer_responsibility_code,
                claim_filing_indicator,
                place_of_service_code,
                medical_record_number,
                rendering_provider_id,
                referring_provider_id,
                supervising_provider_id,
                billing_provider_id,
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
                service_facility_postal_code
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22,
                    $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41, $42)
            RETURNING encounter_id
            "#
        )
        .bind(facility_id)
        .bind(organization_id)
        .bind(region_id)
        .bind(&submitter_id)
        .bind(&patient_control_number)
        .bind(&subscriber_id)
        .bind(&subscriber_last_name)
        .bind(&subscriber_first_name)
        .bind(subscriber_dob)
        .bind(dos_from)
        .bind(dos_from) // date_of_service_to same as from for now
        .bind(total_claim_charge)
        .bind(payer_id.as_deref())
        .bind(payer_name.as_deref())
        .bind(payer_responsibility_code)
        .bind(claim_filing_indicator.as_deref())
        .bind(place_of_service)
        .bind(medical_record_number.as_deref())
        .bind(rendering_provider_id)
        .bind(referring_provider_id)
        .bind(supervising_provider_id)
        .bind(billing_provider_id)
        .bind("NEW")
        .bind(patient_last_name.as_deref())
        .bind(patient_first_name.as_deref())
        .bind(patient_middle_name.as_deref())
        .bind(patient_name_suffix.as_deref())
        .bind(patient_dob)
        .bind(patient_gender)
        .bind(patient_address_line1.as_deref())
        .bind(patient_address_line2.as_deref())
        .bind(patient_city.as_deref())
        .bind(patient_state)
        .bind(patient_postal_code.as_deref())
        .bind(patient_relationship_code)
        .bind(service_facility_npi.as_deref())
        .bind(service_facility_name.as_deref())
        .bind(service_facility_address_line1.as_deref())
        .bind(service_facility_address_line2.as_deref())
        .bind(service_facility_city.as_deref())
        .bind(service_facility_state.as_deref())
        .bind(service_facility_postal_code.as_deref())
        .fetch_one(&mut **tx)
        .await
        .map_err(|e| {
            error!("Database error inserting encounter: {:?}", e);
            error!("  patient_control_number={}, facility_id={:?}, organization_id={}",
                patient_control_number, facility_id, organization_id);
            // Log all VARCHAR(2) and CHAR fields to identify which is too long
            error!("  DEBUG field lengths: place_of_service={:?}(len={}), payer_responsibility={:?}(len={})",
                place_of_service, place_of_service.map(|s| s.len()).unwrap_or(0),
                payer_responsibility_code, payer_responsibility_code.len());
            error!("  DEBUG patient fields: gender={:?}(len={}), state={:?}(len={}), relationship={:?}(len={})",
                patient_gender, patient_gender.map(|s| s.len()).unwrap_or(0),
                patient_state, patient_state.map(|s| s.len()).unwrap_or(0),
                patient_relationship_code, patient_relationship_code.map(|s| s.len()).unwrap_or(0));
            error!("  DEBUG service_facility: npi={:?}, name={:?}", service_facility_npi, service_facility_name);
            e
        })
        .context("Failed to insert encounter")?;

        // Insert all service lines for this encounter
        let mut line_number = 1;
        for service_line in &service_lines {
            self.import_service_line(tx, encounter_id, organization_id, service_line, line_number).await?;
            line_number += 1;
        }

        // Import diagnoses from first service line (all should have same diagnoses)
        self.import_diagnoses(tx, encounter_id, first_line).await?;

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

        debug!("[COB] Inserting {} COB payers into encounter_payer for encounter_id={}",
            other_insurance_array.len(), encounter_id);

        for oi in &other_insurance_array {
            let payer_resp_seq = oi.get("payer_responsibility_sequence")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty());
            let payer_id = oi.get("payer_id")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty());
            let payer_name = oi.get("payer_name")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty());
            let claim_filing_indicator = oi.get("claim_filing_indicator")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty());
            let paid_amount: Option<rust_decimal::Decimal> = oi.get("paid_amount")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
                .and_then(|s| s.parse().ok());
            let claim_control_number = oi.get("claim_control_number")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty());

            // Skip if no payer_responsibility_sequence (required field)
            let payer_resp = match payer_resp_seq {
                Some(p) => p,
                None => {
                    warn!("[COB] Skipping COB payer with missing payer_responsibility_sequence");
                    continue;
                }
            };

            self.insert_encounter_payer(
                tx,
                encounter_id,
                payer_resp,
                payer_id,
                payer_name,
                claim_filing_indicator,
                false, // is_billing_payer = false for COB payers
                paid_amount,
                claim_control_number,
                billing_provider_id,
            ).await?;
        }

        Ok(())
    }

    /// Import other insurance records from encounter_fields JSON
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

        debug!("[COB] Inserting {} other_insurance records for encounter_id={}",
            other_insurance_array.len(), encounter_id);

        for oi in &other_insurance_array {
            let payer_resp_seq = oi.get("payer_responsibility_sequence")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty());
            let individual_rel_code = oi.get("individual_relationship_code")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty());
            let group_policy_number = oi.get("group_policy_number")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty());
            let group_name = oi.get("group_name")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty());
            let insurance_type_code = oi.get("insurance_type_code")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty());
            let coordination_benefits_code = oi.get("coordination_benefits_code")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty());
            let claim_filing_indicator = oi.get("claim_filing_indicator")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty());
            let payer_id = oi.get("payer_id")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty());
            let payer_name = oi.get("payer_name")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty());
            let payer_address_line1 = oi.get("payer_address_line1")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty());
            let payer_address_line2 = oi.get("payer_address_line2")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty());
            let payer_city = oi.get("payer_city")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty());
            let payer_state = oi.get("payer_state")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty());
            let payer_postal_code = oi.get("payer_postal_code")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty());
            let paid_amount: Option<rust_decimal::Decimal> = oi.get("paid_amount")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
                .and_then(|s| s.parse().ok());
            let claim_control_number = oi.get("claim_control_number")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty());
            let benefits_assignment = oi.get("benefits_assignment_certification")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty());
            let release_of_info = oi.get("release_of_information_code")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty());

            // Skip if no payer_responsibility_sequence (required field)
            let payer_resp = match payer_resp_seq {
                Some(p) => p,
                None => {
                    warn!("[COB] Skipping other_insurance record with missing payer_responsibility_sequence");
                    continue;
                }
            };

            sqlx::query(
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
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19)
                "#
            )
            .bind(encounter_id)
            .bind(payer_resp)
            .bind(individual_rel_code)
            .bind(group_policy_number)
            .bind(group_name)
            .bind(insurance_type_code)
            .bind(coordination_benefits_code)
            .bind(claim_filing_indicator)
            .bind(payer_id)
            .bind(payer_name)
            .bind(payer_address_line1)
            .bind(payer_address_line2)
            .bind(payer_city)
            .bind(payer_state)
            .bind(payer_postal_code)
            .bind(paid_amount)
            .bind(claim_control_number)
            .bind(benefits_assignment)
            .bind(release_of_info)
            .execute(&mut **tx)
            .await
            .context("Failed to insert other_insurance record")?;

            debug!("[COB] Inserted other_insurance: payer_resp={}, payer_id={:?}, payer_name={:?}, paid_amount={:?}",
                payer_resp, payer_id, payer_name, paid_amount);
        }

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
            .map(|s| s.as_str())
            .unwrap_or("1900-01-01"); // Default to 1900-01-01 if missing or empty

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

        // Calculate total claim charge from service lines
        let total_claim_charge = service_line_fields.as_ref()
            .and_then(|slf| slf.get("line_item_charge_amount"))
            .and_then(|s| s.parse::<rust_decimal::Decimal>().ok())
            .unwrap_or(rust_decimal::Decimal::ZERO);

        // Parse dates
        let dos_from = chrono::NaiveDate::parse_from_str(date_of_service_from, "%Y-%m-%d")
            .context("Invalid date format for date_of_service_from")?;
        let subscriber_dob = chrono::NaiveDate::parse_from_str(subscriber_birth_date_str, "%Y-%m-%d")
            .unwrap_or(*DEFAULT_DATE); // Fallback to 1900-01-01

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
        let rendering_provider_npi = encounter_fields.get("rendering_provider_npi").map(|s| s.as_str()).filter(|s| !s.is_empty());
        let rendering_provider_name = encounter_fields.get("rendering_provider_name").map(|s| s.as_str()).filter(|s| !s.is_empty());
        let referring_provider_npi = encounter_fields.get("referring_provider_npi").map(|s| s.as_str()).filter(|s| !s.is_empty());
        let referring_provider_name = encounter_fields.get("referring_provider_name").map(|s| s.as_str()).filter(|s| !s.is_empty());
        let service_facility_npi = encounter_fields.get("service_facility_npi").map(|s| s.as_str()).filter(|s| !s.is_empty());
        let service_facility_name = encounter_fields.get("service_facility_name").map(|s| s.as_str()).filter(|s| !s.is_empty());
        let service_facility_address_line1 = encounter_fields.get("service_facility_address_line1").map(|s| s.as_str()).filter(|s| !s.is_empty());
        let service_facility_address_line2 = encounter_fields.get("service_facility_address_line2").map(|s| s.as_str()).filter(|s| !s.is_empty());
        let service_facility_city = encounter_fields.get("service_facility_city").map(|s| s.as_str()).filter(|s| !s.is_empty());
        let service_facility_state = encounter_fields.get("service_facility_state").map(|s| s.as_str()).filter(|s| !s.is_empty());
        let service_facility_postal_code = encounter_fields.get("service_facility_postal_code").map(|s| s.as_str()).filter(|s| !s.is_empty());

        // Debug: Log service facility fields from raw_claims
        info!("Service facility from raw_claims: npi={:?}, name={:?}, addr1={:?}, city={:?}, state={:?}",
            service_facility_npi, service_facility_name, service_facility_address_line1,
            service_facility_city, service_facility_state);
        let supervising_provider_npi = encounter_fields.get("supervising_provider_npi").map(|s| s.as_str()).filter(|s| !s.is_empty());
        let supervising_provider_name = encounter_fields.get("supervising_provider_name").map(|s| s.as_str()).filter(|s| !s.is_empty());
        let billing_provider_npi = encounter_fields.get("billing_provider_npi").map(|s| s.as_str()).filter(|s| !s.is_empty());
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
        let rendering_provider_id = if let Some(npi) = rendering_provider_npi {
            match self.ensure_provider_exists(
                tx,
                npi,
                "Rendering",
                rendering_provider_last_name,
                rendering_provider_first_name,
                None,
                rendering_provider_taxonomy,
                Some(organization_id),
            ).await {
                Ok(provider_id) => {
                    if provider_id.is_none() {
                        warn!("Failed to create/find rendering provider: NPI={}, Name={} {}",
                            npi,
                            rendering_provider_first_name.unwrap_or(""),
                            rendering_provider_last_name.unwrap_or(""));
                    }
                    provider_id
                },
                Err(e) => {
                    error!("Error ensuring rendering provider exists: NPI={}, Error={:?}", npi, e);
                    None
                }
            }
        } else {
            None
        };

        let referring_provider_id = if let Some(npi) = referring_provider_npi {
            match self.ensure_provider_exists(
                tx,
                npi,
                "Referring",
                referring_provider_last_name,
                referring_provider_first_name,
                None,
                referring_provider_taxonomy,
                Some(organization_id),
            ).await {
                Ok(provider_id) => {
                    if provider_id.is_none() {
                        warn!("Failed to create/find referring provider: NPI={}, Name={} {}",
                            npi,
                            referring_provider_first_name.unwrap_or(""),
                            referring_provider_last_name.unwrap_or(""));
                    }
                    provider_id
                },
                Err(e) => {
                    error!("Error ensuring referring provider exists: NPI={}, Error={:?}", npi, e);
                    None
                }
            }
        } else {
            None
        };

        let supervising_provider_id = if let Some(npi) = supervising_provider_npi {
            match self.ensure_provider_exists(
                tx,
                npi,
                "Supervising",
                supervising_provider_last_name,
                supervising_provider_first_name,
                None,
                supervising_provider_taxonomy,
                Some(organization_id),
            ).await {
                Ok(provider_id) => {
                    if provider_id.is_none() {
                        warn!("Failed to create/find supervising provider: NPI={}, Name={} {}",
                            npi,
                            supervising_provider_first_name.unwrap_or(""),
                            supervising_provider_last_name.unwrap_or(""));
                    }
                    provider_id
                },
                Err(e) => {
                    error!("Error ensuring supervising provider exists: NPI={}, Error={:?}", npi, e);
                    None
                }
            }
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

            match self.ensure_provider_exists(
                tx,
                npi,
                "Billing",
                last,
                first,
                None,
                None,
                Some(organization_id),
            ).await {
                Ok(provider_id) => {
                    if provider_id.is_none() {
                        warn!("Failed to create/find billing provider: NPI={}, Name={:?}",
                            npi, billing_provider_name);
                    }
                    provider_id
                },
                Err(e) => {
                    error!("Error ensuring billing provider exists: NPI={}, Error={:?}", npi, e);
                    None
                }
            }
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

        // Import service line (use line number 1 for single-line processing)
        self.import_service_line(tx, encounter_id, organization_id, raw_claim, 1).await?;

        // Import diagnoses
        self.import_diagnoses(tx, encounter_id, raw_claim).await?;

        Ok(encounter_id)
    }

    /// Import service line for an encounter
    async fn import_service_line(
        &self,
        tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
        encounter_id: i64,
        organization_id: i64,
        raw_claim: &RawClaim,
        line_number: i32,
    ) -> Result<()> {
        // Deserialize encounter fields - use JsonValue to handle mixed types (strings, arrays like other_insurance)
        let encounter_fields: HashMap<String, JsonValue> = serde_json::from_value(raw_claim.encounter_fields.clone())
            .context("Failed to deserialize encounter_fields")?;

        let service_line_fields: HashMap<String, String> = raw_claim.service_line_fields.as_ref()
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or_default();

        // IMPORTANT: Each RawClaim from staging represents ONE service line, so service_line_fields
        // always uses "service_line_1_" prefix. The line_number parameter is only used for the
        // INSERT statement to set the correct line_number column value.
        let prefix = "service_line_1_";

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
        let unit_count = service_unit_count.parse::<rust_decimal::Decimal>()
            .unwrap_or(rust_decimal::Decimal::ONE);

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
                line_status
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41)
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

        Ok(())
    }

    /// Import service line diagnosis pointers (junction table)
    /// Links service lines to encounter diagnoses based on diagnosis code pointers
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
        let pointers = vec![
            (1, pointer_1),
            (2, pointer_2),
            (3, pointer_3),
            (4, pointer_4),
        ];

        for (pointer_sequence, pointer_value) in pointers {
            if let Some(diagnosis_pointer) = pointer_value {
                // Find the diagnosis_id for this encounter and sequence number
                let diagnosis_id: Option<i64> = sqlx::query_scalar(
                    r#"
                    SELECT diagnosis_id
                    FROM claims.encounter_diagnosis
                    WHERE encounter_id = $1
                    AND sequence_number = $2
                    "#
                )
                .bind(encounter_id)
                .bind(diagnosis_pointer)
                .fetch_optional(&mut **tx)
                .await?;

                if let Some(diag_id) = diagnosis_id {
                    // Insert the junction record
                    sqlx::query(
                        r#"
                        INSERT INTO claims.service_line_diagnosis_pointer (
                            service_line_id,
                            diagnosis_id,
                            pointer_sequence
                        )
                        VALUES ($1, $2, $3)
                        ON CONFLICT (service_line_id, diagnosis_id, pointer_sequence) DO NOTHING
                        "#
                    )
                    .bind(service_line_id)
                    .bind(diag_id)
                    .bind(pointer_sequence as i16)
                    .execute(&mut **tx)
                    .await
                    .context("Failed to insert service line diagnosis pointer")?;

                    debug!(
                        "Linked service line {} to diagnosis {} (pointer {} -> sequence {})",
                        service_line_id, diag_id, pointer_sequence, diagnosis_pointer
                    );
                } else {
                    warn!(
                        "Service line {} references diagnosis pointer {} for encounter {}, but no diagnosis with that sequence exists",
                        service_line_id, diagnosis_pointer, encounter_id
                    );
                }
            }
        }

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

        // Insert diagnoses in order
        for (idx, (sequence, code)) in all_diagnoses.iter().enumerate() {
            let sequence_number = *sequence as i16;

            let diagnosis_id: i64 = sqlx::query_scalar(
                r#"
                INSERT INTO claims.encounter_diagnosis (
                    encounter_id,
                    sequence_number,
                    diagnosis_code,
                    is_principal
                )
                VALUES ($1, $2, $3, $4)
                RETURNING diagnosis_id
                "#
            )
            .bind(encounter_id)
            .bind(sequence_number)
            .bind(code)
            .bind(idx == 0)  // First diagnosis is principal
            .fetch_one(&mut **tx)
            .await
            .context("Failed to insert diagnosis")?;

            debug!("Inserted diagnosis {} ({}) for encounter {}, diagnosis_id={}",
                sequence_number, code, encounter_id, diagnosis_id);
        }

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
                metric_id,
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
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            "#
        )
        .bind(0i64) // TODO: Refactor to use RETURNING
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

        // Begin transaction for batch processing
        let mut tx = self.pool.begin().await
            .context("Failed to begin transaction")?;

        // Facility lookup cache
        let mut facility_cache: HashMap<String, (Option<i64>, i64, Option<i64>)> = HashMap::new();

        // Group raw_claims by encounter (patient_control_number + date_of_service)
        use std::collections::HashMap as StdHashMap;
        let mut encounter_groups: StdHashMap<(String, String), Vec<RawClaim>> = StdHashMap::new();

        for raw_claim in raw_claims {
            // Extract encounter key from encounter_fields
            // Use JsonValue to handle mixed types (strings, arrays like other_insurance)
            let encounter_fields: StdHashMap<String, JsonValue> = match serde_json::from_value(raw_claim.encounter_fields.clone()) {
                Ok(fields) => fields,
                Err(e) => {
                    failure_count += 1;
                    let error_message = format!("Row {}: Failed to deserialize encounter_fields: {}", raw_claim.row_number, e);
                    errors.push(error_message.clone());
                    error!("Failed to deserialize encounter_fields for raw_claim_id {}: {}", raw_claim.raw_claim_id, e);
                    continue;
                }
            };

            let patient_control_number = match get_field_as_string(&encounter_fields, "patient_control_number") {
                Some(pcn) => pcn,
                None => {
                    failure_count += 1;
                    let error_message = format!("Row {}: Missing patient_control_number", raw_claim.row_number);
                    errors.push(error_message.clone());
                    error!("Missing patient_control_number for raw_claim_id {}", raw_claim.raw_claim_id);
                    continue;
                }
            };

            let date_of_service = match get_field_as_string(&encounter_fields, "date_of_service_from") {
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

        info!("Worker {} grouped {} raw claims into {} encounters",
            worker_id, claim_ids.len(), encounter_groups.len());

        // Process each encounter group
        for ((patient_control_number, date_of_service), service_lines) in encounter_groups {
            debug!("Processing encounter: {} on {} ({} service lines)",
                patient_control_number, date_of_service, service_lines.len());

            // Validate and insert encounter with all service lines
            match self.process_encounter_with_service_lines(&mut tx, service_lines.clone(), &mut facility_cache).await {
                Ok(encounter_id) => {
                    success_count += service_lines.len();

                    // Mark all raw_claims in this encounter as COMPLETED (bulk update)
                    let claim_ids: Vec<i64> = service_lines.iter().map(|sl| sl.raw_claim_id).collect();
                    sqlx::query(
                        r#"
                        UPDATE staging.raw_claims
                        SET processing_status = 'COMPLETED',
                            processed_at = CURRENT_TIMESTAMP
                        WHERE raw_claim_id = ANY($1)
                        "#
                    )
                    .bind(&claim_ids)
                    .execute(&mut *tx)
                    .await?;

                    debug!("Successfully processed encounter: {} on {} -> encounter_id {} ({} service lines)",
                        patient_control_number, date_of_service, encounter_id, service_lines.len());
                }
                Err(e) => {
                    failure_count += service_lines.len();
                    let error_str = e.to_string();
                    error!("Failed to process encounter {} on {}: {}", patient_control_number, date_of_service, error_str);

                    // Collect error logs and claim IDs for bulk operations
                    let mut error_log_inserts = Vec::new();
                    let claim_ids: Vec<i64> = service_lines.iter().map(|sl| sl.raw_claim_id).collect();

                    for service_line in &service_lines {
                        let error_message = format!("Row {}: {}", service_line.row_number, error_str);
                        errors.push(error_message.clone());

                        // Prepare error log insert
                        error_log_inserts.push((
                            0i64, // error_log_id - database will generate
                            service_line.batch_id,
                            service_line.row_number,
                            error_message,
                            serde_json::to_string(&service_line.encounter_fields).ok(),
                        ));
                    }

                    // Bulk insert error logs
                    for (_error_log_id, batch_id, record_number, error_message, raw_data) in error_log_inserts {
                        sqlx::query(
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
                        .bind(record_number)
                        .bind(&error_message)
                        .bind(raw_data)
                        .execute(&mut *tx)
                        .await?;
                    }

                    // Bulk update claims as FAILED
                    sqlx::query(
                        r#"
                        UPDATE staging.raw_claims
                        SET processing_status = 'FAILED',
                            processed_at = CURRENT_TIMESTAMP,
                            error_message = $1
                        WHERE raw_claim_id = ANY($2)
                        "#
                    )
                    .bind(&error_str)
                    .bind(&claim_ids)
                    .execute(&mut *tx)
                    .await?;
                }
            }
        }

        // Commit transaction
        tx.commit().await
            .context("Failed to commit sequenced batch transaction")?;

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

    /// Ensure a provider exists in claims.provider table, creating if necessary
    /// Returns the provider_id (either existing or newly created)
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
            warn!("Invalid NPI format: {} (expected 10 digits)", npi);
            return Ok(None);
        }

        // Retry logic to handle deadlocks from concurrent provider creation
        const MAX_RETRIES: u32 = 5;
        let mut retry_count = 0;

        loop {
            // Check if provider already exists
            let existing_provider: Option<i64> = sqlx::query_scalar(
                r#"
                SELECT provider_id
                FROM claims.provider
                WHERE npi = $1
                "#
            )
            .bind(npi)
            .fetch_optional(&mut **tx)
            .await
            .context("Failed to query existing provider")?;

            if let Some(provider_id) = existing_provider {
                // Provider exists, return the ID
                return Ok(Some(provider_id));
            }

            // Provider doesn't exist, create it
            // Use "Unknown" for last_name if not provided, and empty string for first_name
            let last_name_value = last_name.unwrap_or("Unknown");
            let first_name_value = first_name.unwrap_or("");

            // Lookup specialty from taxonomy code if provided AND validate taxonomy exists in reference table
            // If taxonomy code doesn't exist in provider_taxonomy, we must NOT insert it (FK constraint)
            let (validated_taxonomy_code, specialty) = if let Some(tax_code) = taxonomy_code {
                if tax_code.is_empty() {
                    (None, None)
                } else {
                    // Check if taxonomy exists and get specialty in one query
                    let result: Option<String> = sqlx::query_scalar(
                        r#"
                        SELECT specialty_display
                        FROM claims.provider_taxonomy
                        WHERE taxonomy_code = $1 AND is_active = true
                        "#
                    )
                    .bind(tax_code)
                    .fetch_optional(&mut **tx)
                    .await
                    .unwrap_or(None);

                    if result.is_some() {
                        // Taxonomy exists, use it
                        (Some(tax_code), result)
                    } else {
                        // Taxonomy doesn't exist in reference table - set to NULL to avoid FK violation
                        warn!("Taxonomy code '{}' not found in provider_taxonomy table, setting to NULL for provider NPI={}",
                            tax_code, npi);
                        (None, None)
                    }
                }
            } else {
                (None, None)
            };

            // Log if taxonomy lookup succeeded
            if let Some(ref spec) = specialty {
                debug!("Mapped taxonomy {} to specialty: {}", validated_taxonomy_code.unwrap_or(""), spec);
            }

            // Try to insert the provider. Use DO NOTHING to reduce lock contention.
            // Note: When ON CONFLICT DO NOTHING is triggered, RETURNING returns nothing,
            // so fetch_optional will return Ok(None)
            let insert_result = sqlx::query_scalar::<_, i64>(
                r#"
                INSERT INTO claims.provider (
                    npi,
                    provider_type,
                    last_name,
                    first_name,
                    middle_name,
                    taxonomy_code,
                    specialty,
                    organization_id,
                    is_active,
                    created_at,
                    updated_at
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, true, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT (npi) DO NOTHING
                RETURNING provider_id
                "#
            )
            .bind(npi)
            .bind(provider_type)
            .bind(last_name_value)
            .bind(first_name_value)
            .bind(middle_name)
            .bind(validated_taxonomy_code) // Use validated taxonomy (NULL if not in reference table)
            .bind(specialty.as_deref())
            .bind(organization_id)
            .fetch_optional(&mut **tx)
            .await;

            match insert_result {
                Ok(Some(provider_id)) => {
                    // Successfully inserted, return the new provider_id
                    debug!("Created new provider: NPI={}, Type={}, Name={} {}, Specialty={:?}",
                        npi, provider_type, first_name_value, last_name_value, specialty);

                    // Enqueue provider for background NPI enrichment (fire-and-forget)
                    // This does not block claim processing if it fails
                    let _ = sqlx::query(
                        r#"
                        INSERT INTO claims.provider_enrichment_queue (provider_id, npi, priority)
                        VALUES ($1, $2, $3)
                        ON CONFLICT (provider_id) DO NOTHING
                        "#
                    )
                    .bind(provider_id)
                    .bind(npi)
                    .bind(5) // Default priority
                    .execute(&mut **tx)
                    .await;

                    return Ok(Some(provider_id));
                }
                Ok(None) => {
                    // ON CONFLICT DO NOTHING was triggered - provider was created by another transaction
                    // Query again to get the provider_id
                    let existing_id: Option<i64> = sqlx::query_scalar(
                        r#"
                        SELECT provider_id
                        FROM claims.provider
                        WHERE npi = $1
                        "#
                    )
                    .bind(npi)
                    .fetch_optional(&mut **tx)
                    .await
                    .context("Failed to query provider after conflict")?;

                    if let Some(id) = existing_id {
                        debug!("Provider already exists (concurrent creation): NPI={}, provider_id={}",
                            npi, id);
                        return Ok(Some(id));
                    } else {
                        // Very rare: provider was deleted between INSERT and SELECT
                        // Retry the whole operation
                        if retry_count < MAX_RETRIES {
                            retry_count += 1;
                            let backoff_ms = 10u64 * 2u64.pow(retry_count - 1); // 10ms, 20ms, 40ms, 80ms, 160ms
                            warn!("Provider disappeared after conflict, retrying ({}/{}): NPI={}, backoff={}ms",
                                retry_count, MAX_RETRIES, npi, backoff_ms);
                            tokio::time::sleep(tokio::time::Duration::from_millis(backoff_ms)).await;
                            continue;
                        } else {
                            return Err(anyhow::anyhow!(
                                "Provider creation failed after {} retries: NPI={}",
                                MAX_RETRIES, npi
                            ));
                        }
                    }
                }
                Err(e) => {
                    // Check if this is a deadlock error (PostgreSQL error code 40P01)
                    let error_string = e.to_string();
                    let is_deadlock = error_string.contains("deadlock detected")
                        || error_string.contains("40P01");

                    if is_deadlock && retry_count < MAX_RETRIES {
                        retry_count += 1;
                        // Exponential backoff with jitter: 10ms, 20ms, 40ms, 80ms, 160ms
                        let backoff_ms = 10u64 * 2u64.pow(retry_count - 1);
                        let jitter = (rand::random::<u64>() % 10) as u64; // 0-9ms jitter
                        let total_backoff = backoff_ms + jitter;

                        warn!("Deadlock detected creating provider, retrying ({}/{}): NPI={}, backoff={}ms",
                            retry_count, MAX_RETRIES, npi, total_backoff);

                        tokio::time::sleep(tokio::time::Duration::from_millis(total_backoff)).await;
                        continue;
                    } else {
                        // Not a deadlock or max retries exceeded
                        return Err(e).context("Failed to insert provider");
                    }
                }
            }
        }
    }
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
