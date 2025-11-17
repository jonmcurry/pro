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
use serde_json::Value as JsonValue;
use sqlx::PgPool;
use std::collections::HashMap;
use tracing::{debug, error, info, warn};


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
            let encounter_fields: StdHashMap<String, String> = match serde_json::from_value(raw_claim.encounter_fields.clone()) {
                Ok(fields) => fields,
                Err(e) => {
                    error!("Failed to deserialize encounter_fields for raw_claim_id {}: {}", raw_claim.raw_claim_id, e);
                    result.failed += 1;
                    continue;
                }
            };

            let patient_control_number = match encounter_fields.get("patient_control_number") {
                Some(pcn) => pcn.clone(),
                None => {
                    error!("Missing patient_control_number for raw_claim_id {}", raw_claim.raw_claim_id);
                    result.failed += 1;
                    continue;
                }
            };

            let date_of_service = match encounter_fields.get("date_of_service_from") {
                Some(dos) => dos.clone(),
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
        let encounter_fields: HashMap<String, String> = serde_json::from_value(first_line.encounter_fields.clone())
            .context("Failed to deserialize encounter_fields")?;

        // Extract facility_code
        let facility_code = encounter_fields.get("facility_code")
            .or_else(|| encounter_fields.get("facility_npi"))
            .context("Missing facility_code or facility_npi")?;

        // Check cache first
        let (facility_id, organization_id, region_id) = if let Some(cached) = facility_cache.get(facility_code) {
            *cached
        } else {
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
            .map(|s| s.as_str())
            .unwrap_or("1900-01-01"); // Default to 1900-01-01 if missing or empty

        // Optional fields
        let submitter_id = encounter_fields.get("submitter_id").unwrap_or(facility_code);
        let payer_responsibility_code = encounter_fields.get("payer_responsibility_code")
            .map(|s| s.as_str()).unwrap_or("P");

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
        let dos_from = chrono::NaiveDate::parse_from_str(date_of_service_from, "%Y-%m-%d")
            .context("Invalid date format for date_of_service_from")?;
        let subscriber_dob = chrono::NaiveDate::parse_from_str(subscriber_birth_date_str, "%Y-%m-%d")
            .unwrap_or_else(|_| chrono::NaiveDate::from_ymd_opt(1900, 1, 1).unwrap()); // Fallback to 1900-01-01

        // Optional fields
        let payer_id = encounter_fields.get("payer_id").map(|s| s.as_str());
        let payer_name = encounter_fields.get("payer_name").map(|s| s.as_str());
        let place_of_service = encounter_fields.get("place_of_service_code").map(|s| s.as_str());
        let medical_record_number = encounter_fields.get("medical_record_number").map(|s| s.as_str());

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
                place_of_service_code,
                medical_record_number,
                claim_status
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18)
            RETURNING encounter_id
            "#
        )
        .bind(facility_id)
        .bind(organization_id)
        .bind(region_id)
        .bind(submitter_id)
        .bind(patient_control_number)
        .bind(subscriber_id)
        .bind(subscriber_last_name)
        .bind(subscriber_first_name)
        .bind(subscriber_dob)
        .bind(dos_from)
        .bind(dos_from) // date_of_service_to same as from for now
        .bind(total_claim_charge)
        .bind(payer_id)
        .bind(payer_name)
        .bind(payer_responsibility_code)
        .bind(place_of_service)
        .bind(medical_record_number)
        .bind("NEW")
        .fetch_one(&mut **tx)
        .await
        .context("Failed to insert encounter")?;

        // Insert all service lines for this encounter
        let mut line_number = 1;
        for service_line in &service_lines {
            self.import_service_line(tx, encounter_id, service_line, line_number).await?;
            line_number += 1;
        }

        // Import diagnoses from first service line (all should have same diagnoses)
        self.import_diagnoses(tx, encounter_id, first_line).await?;

        Ok(encounter_id)
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
        let payer_responsibility_code = encounter_fields.get("payer_responsibility_code")
            .map(|s| s.as_str())
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
            .unwrap_or_else(|_| chrono::NaiveDate::from_ymd_opt(1900, 1, 1).unwrap()); // Fallback to 1900-01-01

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
        let rendering_provider_npi = encounter_fields.get("rendering_provider_npi").map(|s| s.as_str());
        let rendering_provider_name = encounter_fields.get("rendering_provider_name").map(|s| s.as_str());
        let referring_provider_npi = encounter_fields.get("referring_provider_npi").map(|s| s.as_str());
        let referring_provider_name = encounter_fields.get("referring_provider_name").map(|s| s.as_str());
        let service_facility_npi = encounter_fields.get("service_facility_npi").map(|s| s.as_str());
        let service_facility_name = encounter_fields.get("service_facility_name").map(|s| s.as_str());
        let service_facility_address_line1 = encounter_fields.get("service_facility_address_line1").map(|s| s.as_str());
        let service_facility_address_line2 = encounter_fields.get("service_facility_address_line2").map(|s| s.as_str());
        let service_facility_city = encounter_fields.get("service_facility_city").map(|s| s.as_str());
        let service_facility_state = encounter_fields.get("service_facility_state").map(|s| s.as_str());
        let service_facility_postal_code = encounter_fields.get("service_facility_postal_code").map(|s| s.as_str());
        let supervising_provider_npi = encounter_fields.get("supervising_provider_npi").map(|s| s.as_str());
        let supervising_provider_name = encounter_fields.get("supervising_provider_name").map(|s| s.as_str());
        let billing_provider_npi = encounter_fields.get("billing_provider_npi").map(|s| s.as_str());
        let billing_provider_name = encounter_fields.get("billing_provider_name").map(|s| s.as_str());
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
        let claim_filing_indicator = encounter_fields.get("claim_filing_indicator_code").map(|s| s.as_str());
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
            self.ensure_provider_exists(
                tx,
                npi,
                "Rendering",
                rendering_provider_last_name,
                rendering_provider_first_name,
                None,
                rendering_provider_taxonomy,
                Some(organization_id),
            ).await.unwrap_or(None)
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
            ).await.unwrap_or(None)
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
            ).await.unwrap_or(None)
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
            ).await.unwrap_or(None)
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
        self.import_service_line(tx, encounter_id, raw_claim, 1).await?;

        // Import diagnoses
        self.import_diagnoses(tx, encounter_id, raw_claim).await?;

        Ok(encounter_id)
    }

    /// Import service line for an encounter
    async fn import_service_line(
        &self,
        tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
        encounter_id: i64,
        raw_claim: &RawClaim,
        line_number: i32,
    ) -> Result<()> {
        // Deserialize encounter and service line fields
        let encounter_fields: HashMap<String, String> = serde_json::from_value(raw_claim.encounter_fields.clone())
            .context("Failed to deserialize encounter_fields")?;

        let service_line_fields: HashMap<String, String> = raw_claim.service_line_fields.as_ref()
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or_default();

        // Build field name prefix using line_number (e.g., "service_line_1_")
        let prefix = format!("service_line_{}_", line_number);

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
            .or_else(|| encounter_fields.get("date_of_service_from"))
            .context("Missing service_date_from and date_of_service_from")?;

        // Parse decimal values
        let charge_amount = line_item_charge_amount.parse::<rust_decimal::Decimal>()
            .context("Invalid charge amount")?;
        let unit_count = service_unit_count.parse::<rust_decimal::Decimal>()
            .unwrap_or(rust_decimal::Decimal::ONE);

        // Parse service date
        let service_date = chrono::NaiveDate::parse_from_str(service_date_str, "%Y-%m-%d")
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

        // Extract procedure modifiers (filter out empty strings)
        let modifier_1 = service_line_fields.get(&format!("{}modifier_1", prefix))
            .filter(|s| !s.is_empty())
            .map(|s| s.as_str());
        let modifier_2 = service_line_fields.get(&format!("{}modifier_2", prefix))
            .filter(|s| !s.is_empty())
            .map(|s| s.as_str());
        let modifier_3 = service_line_fields.get(&format!("{}modifier_3", prefix))
            .filter(|s| !s.is_empty())
            .map(|s| s.as_str());
        let modifier_4 = service_line_fields.get(&format!("{}modifier_4", prefix))
            .filter(|s| !s.is_empty())
            .map(|s| s.as_str());

        // Extract additional service line fields
        let product_service_id_qualifier = service_line_fields.get("product_service_id_qualifier")
            .map(|s| s.as_str())
            .unwrap_or("HC"); // Default to HCPCS
        let unit_basis_measurement_code = service_line_fields.get("unit_basis_measurement_code")
            .map(|s| s.as_str());
        let service_date_to = service_line_fields.get("service_date_to")
            .and_then(|s| chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d").ok());
        let place_of_service_code = service_line_fields.get("place_of_service_code")
            .map(|s| s.as_str());

        // Convert EDI Y/N indicators to boolean values
        let emergency_indicator = service_line_fields.get("emergency_indicator")
            .map(|s| s.trim().eq_ignore_ascii_case("Y"))
            .unwrap_or(false);
        let epsdt_indicator = service_line_fields.get("epsdt_indicator")
            .map(|s| s.trim().eq_ignore_ascii_case("Y"))
            .unwrap_or(false);
        let family_planning_indicator = service_line_fields.get("family_planning_indicator")
            .map(|s| s.trim().eq_ignore_ascii_case("Y"))
            .unwrap_or(false);

        // Phase 3: Service line level fields
        // Phase 3.1: Provider NPIs at service line level (Loop 2420)
        let sl_rendering_provider_npi = service_line_fields.get("rendering_provider_npi")
            .map(|s| s.as_str());
        let sl_ordering_provider_npi = service_line_fields.get("ordering_provider_npi")
            .map(|s| s.as_str());
        let sl_supervising_provider_npi = service_line_fields.get("supervising_provider_npi")
            .map(|s| s.as_str());
        let sl_referring_provider_npi = service_line_fields.get("referring_provider_npi")
            .map(|s| s.as_str());

        // Phase 3.2: Rendering provider taxonomy at service line level
        let sl_rendering_provider_taxonomy = service_line_fields.get("rendering_provider_taxonomy")
            .map(|s| s.as_str());

        // Phase 3.5: Supplemental amounts at service line level
        let sl_approved_amount = service_line_fields.get("approved_amount")
            .and_then(|s| s.parse::<rust_decimal::Decimal>().ok());
        let sl_non_covered_charges = service_line_fields.get("non_covered_charges")
            .and_then(|s| s.parse::<rust_decimal::Decimal>().ok());

        // Phase 3.6: NDC information (drug codes)
        let sl_ndc_code = service_line_fields.get("ndc_code")
            .map(|s| s.as_str());
        let sl_ndc_unit_count = service_line_fields.get("ndc_unit_count")
            .and_then(|s| s.parse::<rust_decimal::Decimal>().ok());
        let sl_ndc_measurement_unit = service_line_fields.get("ndc_measurement_unit")
            .map(|s| s.as_str());

        // Phase 3.7: Authorization and referral information
        let sl_prior_authorization = service_line_fields.get("prior_authorization_number")
            .map(|s| s.as_str());
        let sl_referral_number = service_line_fields.get("referral_number")
            .map(|s| s.as_str());

        // Phase 3.8: Line note and revenue code
        let sl_line_note = service_line_fields.get("line_note")
            .map(|s| s.as_str());
        let sl_revenue_code = service_line_fields.get("revenue_code")
            .map(|s| s.as_str());

        // Phase 3.9: Other payer line amount (COB)
        let sl_other_payer_paid = service_line_fields.get("other_payer_line_paid_amount")
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
                rendering_provider_npi,
                ordering_provider_npi,
                supervising_provider_npi,
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
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37)
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
        .bind(sl_rendering_provider_npi)
        .bind(sl_ordering_provider_npi)
        .bind(sl_supervising_provider_npi)
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

        // Get all diagnosis codes
        for (field_name, codes) in &diagnosis_fields {
            if field_name == "diagnosis_code" {
                for (idx, code) in codes.iter().enumerate() {
                    let sequence_number = (idx + 1) as i16;

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
                    .bind(idx == 0)
                    .fetch_one(&mut **tx)
                    .await
                    .context("Failed to insert diagnosis")?;

                    debug!("Inserted diagnosis {} ({}) for encounter {}",
                        sequence_number, code, encounter_id);
                }
            }
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
            let encounter_fields: StdHashMap<String, String> = match serde_json::from_value(raw_claim.encounter_fields.clone()) {
                Ok(fields) => fields,
                Err(e) => {
                    failure_count += 1;
                    let error_message = format!("Row {}: Failed to deserialize encounter_fields: {}", raw_claim.row_number, e);
                    errors.push(error_message.clone());
                    error!("Failed to deserialize encounter_fields for raw_claim_id {}: {}", raw_claim.raw_claim_id, e);
                    continue;
                }
            };

            let patient_control_number = match encounter_fields.get("patient_control_number") {
                Some(pcn) => pcn.clone(),
                None => {
                    failure_count += 1;
                    let error_message = format!("Row {}: Missing patient_control_number", raw_claim.row_number);
                    errors.push(error_message.clone());
                    error!("Missing patient_control_number for raw_claim_id {}", raw_claim.raw_claim_id);
                    continue;
                }
            };

            let date_of_service = match encounter_fields.get("date_of_service_from") {
                Some(dos) => dos.clone(),
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
                    for (error_log_id, batch_id, record_number, error_message, raw_data) in error_log_inserts {
                        sqlx::query(
                            r#"
                            INSERT INTO staging.import_error_log (
                                error_id,
                                batch_id,
                                record_number,
                                error_type,
                                error_severity,
                                error_message,
                                raw_data
                            )
                            VALUES ($1, $2, $3, 'VALIDATION', 'ERROR', $4, $5)
                            "#
                        )
                        .bind(error_log_id)
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

        // Lookup specialty from taxonomy code if provided
        let specialty = if let Some(tax_code) = taxonomy_code {
            sqlx::query_scalar::<_, String>(
                r#"
                SELECT specialty_display
                FROM claims.provider_taxonomy
                WHERE taxonomy_code = $1 AND is_active = true
                "#
            )
            .bind(tax_code)
            .fetch_optional(&mut **tx)
            .await
            .unwrap_or(None)
        } else {
            None
        };

        // Log if taxonomy lookup succeeded
        if let Some(ref spec) = specialty {
            debug!("Mapped taxonomy {} to specialty: {}", taxonomy_code.unwrap_or(""), spec);
        } else if taxonomy_code.is_some() && !taxonomy_code.unwrap().is_empty() {
            warn!("No specialty mapping found for taxonomy code: {}", taxonomy_code.unwrap());
        }

        let provider_id: i64 = sqlx::query_scalar(
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
            ON CONFLICT (npi) DO UPDATE
            SET updated_at = CURRENT_TIMESTAMP
            RETURNING provider_id
            "#
        )
        .bind(npi)
        .bind(provider_type)
        .bind(last_name_value)
        .bind(first_name_value)
        .bind(middle_name)
        .bind(taxonomy_code)
        .bind(specialty.as_deref())
        .bind(organization_id)
        .fetch_one(&mut **tx)
        .await
        .context("Failed to insert provider")?;

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

        Ok(Some(provider_id))
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
