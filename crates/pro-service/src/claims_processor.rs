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
use uuid::Uuid;

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
        let raw_claim_ids: Vec<Uuid> = raw_claims.iter().map(|c| c.raw_claim_id).collect();
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
        let batch_ids: Vec<Uuid> = raw_claims.iter().map(|c| c.batch_id).collect();
        let unique_batch_ids: std::collections::HashSet<Uuid> = batch_ids.into_iter().collect();

        // Update batch status to PROCESSING
        for batch_id in &unique_batch_ids {
            sqlx::query(
                r#"
                UPDATE staging.import_batch
                SET import_status = 'PROCESSING'
                WHERE batch_id = $1 AND import_status = 'INGESTED'
                "#
            )
            .bind(batch_id)
            .execute(&self.pool)
            .await?;
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
        let mut facility_cache: HashMap<String, (Uuid, Uuid, Option<Uuid>)> = HashMap::new();

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
                        let error_log_id = Uuid::new_v4();
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
                            VALUES ($1, $2, $3, $4, $5, $6, $7)
                            "#
                        )
                        .bind(error_log_id)
                        .bind(service_line.batch_id)
                        .bind(service_line.row_number)
                        .bind("VALIDATION")
                        .bind("ERROR")
                        .bind(&error_message)
                        .bind(serde_json::to_string(&service_line.encounter_fields).ok())
                        .execute(&mut *tx)
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

        // Update batch statuses to COMPLETED
        for batch_id in &unique_batch_ids {
            let batch_status = if result.failed == 0 {
                "COMPLETED"
            } else if result.successful > 0 {
                "PARTIAL"
            } else {
                "FAILED"
            };

            sqlx::query(
                r#"
                UPDATE staging.import_batch
                SET import_status = $1,
                    successful_records = (
                        SELECT COUNT(*) FROM staging.raw_claims
                        WHERE batch_id = $2 AND processing_status = 'COMPLETED'
                    ),
                    failed_records = (
                        SELECT COUNT(*) FROM staging.raw_claims
                        WHERE batch_id = $2 AND processing_status = 'FAILED'
                    ),
                    completed_at = CURRENT_TIMESTAMP
                WHERE batch_id = $2
                "#
            )
            .bind(batch_status)
            .bind(batch_id)
            .execute(&self.pool)
            .await?;
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
        facility_cache: &mut HashMap<String, (Uuid, Uuid, Option<Uuid>)>,
    ) -> Result<Uuid> {
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
            let facility = sqlx::query_as::<_, (Uuid, Uuid, Option<Uuid>)>(
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

        // Generate encounter ID (ONE for all service lines)
        let encounter_id = Uuid::new_v4();

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
            .context("Missing subscriber_birth_date")?;

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
            .context("Invalid date format for subscriber_birth_date")?;

        // Optional fields
        let payer_id = encounter_fields.get("payer_id").map(|s| s.as_str());
        let payer_name = encounter_fields.get("payer_name").map(|s| s.as_str());
        let place_of_service = encounter_fields.get("place_of_service_code").map(|s| s.as_str());
        let medical_record_number = encounter_fields.get("medical_record_number").map(|s| s.as_str());

        // Insert encounter (ONE record)
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
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19)
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
        .execute(&mut **tx)
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
        facility_cache: &mut HashMap<String, (Uuid, Uuid, Option<Uuid>)>,
    ) -> Result<Uuid> {
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
            let facility = sqlx::query_as::<_, (Uuid, Uuid, Option<Uuid>)>(
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
        let encounter_id = Uuid::new_v4();

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
            .context("Missing subscriber_birth_date")?;

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
            .context("Invalid date format for subscriber_birth_date")?;

        // Optional fields
        let payer_id = encounter_fields.get("payer_id").map(|s| s.as_str());
        let payer_name = encounter_fields.get("payer_name").map(|s| s.as_str());
        let place_of_service = encounter_fields.get("place_of_service_code").map(|s| s.as_str());
        let medical_record_number = encounter_fields.get("medical_record_number").map(|s| s.as_str());

        // Insert encounter
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
                subscriber_birth_date,
                payer_responsibility_code,
                payer_id,
                payer_name,
                total_claim_charge_amount,
                place_of_service_code,
                date_of_service_from,
                medical_record_number,
                claim_status
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18)
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
        .bind(subscriber_dob)
        .bind(payer_responsibility_code)
        .bind(payer_id)
        .bind(payer_name)
        .bind(total_claim_charge)
        .bind(place_of_service)
        .bind(dos_from)
        .bind(medical_record_number)
        .bind("IMPORTED")
        .execute(&mut **tx)
        .await
        .context("Failed to insert encounter")?;

        debug!("Inserted encounter {} for patient {}", encounter_id, patient_control_number);

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
        encounter_id: Uuid,
        raw_claim: &RawClaim,
        line_number: i32,
    ) -> Result<()> {
        let service_line_id = Uuid::new_v4();

        // Deserialize encounter and service line fields
        let encounter_fields: HashMap<String, String> = serde_json::from_value(raw_claim.encounter_fields.clone())
            .context("Failed to deserialize encounter_fields")?;

        let service_line_fields: HashMap<String, String> = raw_claim.service_line_fields.as_ref()
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or_default();

        // Extract required service line fields
        let procedure_code = service_line_fields.get("procedure_code")
            .context("Missing procedure_code")?;
        let line_item_charge_amount = service_line_fields.get("line_item_charge_amount")
            .context("Missing line_item_charge_amount")?;
        let default_unit_count = "1".to_string();
        let service_unit_count = service_line_fields.get("service_unit_count")
            .unwrap_or(&default_unit_count);

        // Get service date - use service_date_from if available, otherwise fall back to encounter DOS
        let service_date_str = service_line_fields.get("service_date_from")
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

        // Insert service line
        sqlx::query(
            r#"
            INSERT INTO claims.service_line (
                service_line_id,
                encounter_id,
                line_number,
                procedure_code,
                line_item_charge_amount,
                service_unit_count,
                service_date_from,
                line_status
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            "#
        )
        .bind(service_line_id)
        .bind(encounter_id)
        .bind(line_number)
        .bind(procedure_code)
        .bind(charge_amount)
        .bind(unit_count)
        .bind(service_date)
        .bind("IMPORTED")
        .execute(&mut **tx)
        .await
        .context("Failed to insert service line")?;

        debug!("Inserted service line {} for encounter {}", service_line_id, encounter_id);

        Ok(())
    }

    /// Import diagnoses for an encounter
    async fn import_diagnoses(
        &self,
        tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
        encounter_id: Uuid,
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
                    let diagnosis_id = Uuid::new_v4();
                    let sequence_number = (idx + 1) as i16;

                    sqlx::query(
                        r#"
                        INSERT INTO claims.encounter_diagnosis (
                            diagnosis_id,
                            encounter_id,
                            sequence_number,
                            diagnosis_code,
                            is_principal
                        )
                        VALUES ($1, $2, $3, $4, $5)
                        "#
                    )
                    .bind(diagnosis_id)
                    .bind(encounter_id)
                    .bind(sequence_number)
                    .bind(code)
                    .bind(idx == 0)
                    .execute(&mut **tx)
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
        batch_id: Uuid,
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
        .bind(Uuid::new_v4())
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
        claim_ids: &[Uuid],
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
                batch_id: Uuid::nil(),
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
        let mut facility_cache: HashMap<String, (Uuid, Uuid, Option<Uuid>)> = HashMap::new();

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
                    let claim_ids: Vec<Uuid> = service_lines.iter().map(|sl| sl.raw_claim_id).collect();
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
                    let claim_ids: Vec<Uuid> = service_lines.iter().map(|sl| sl.raw_claim_id).collect();

                    for service_line in &service_lines {
                        let error_message = format!("Row {}: {}", service_line.row_number, error_str);
                        errors.push(error_message.clone());

                        // Prepare error log insert
                        error_log_inserts.push((
                            Uuid::new_v4(),
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

        let processing_time = (chrono::Utc::now() - batch_start).num_milliseconds() as f64 / 1000.0;

        info!(
            "Worker {} completed batch {} in {:.2}s ({} success, {} failed)",
            worker_id, sequence_number, processing_time, success_count, failure_count
        );

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
}

/// Raw claim from staging.raw_claims table
#[derive(Debug, Clone, sqlx::FromRow)]
struct RawClaim {
    raw_claim_id: Uuid,
    batch_id: Uuid,
    queue_id: Uuid,
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
