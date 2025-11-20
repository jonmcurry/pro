//! Claims importer for CSV and EDI files
//!
//! Imports claims data from CSV and EDI 837p files into the database.
//! Integrates with the FIFO queue system and progress tracking.

use anyhow::{Context, Result};
use pro_parser_csv::CsvParser;
use pro_parser_edi::EdiParser;
use pro_worker::progress::ProgressTracker;
use pro_worker::queue_manager::QueueManager;
use sqlx::PgPool;
use std::collections::HashMap;
use std::path::Path;
use tokio::sync::broadcast;
use tracing::{debug, error, info, warn};


/// Claims importer that processes CSV and EDI files
#[derive(Clone)]
pub struct ClaimsImporter {
    pool: PgPool,
    queue_manager: QueueManager,
    broadcaster: broadcast::Sender<pro_worker::progress::ProgressEvent>,
}

impl ClaimsImporter {
    /// Create a new claims importer with queue and progress tracking
    pub fn new(pool: PgPool) -> Self {
        let queue_manager = QueueManager::new(pool.clone());
        let (broadcaster, _) = broadcast::channel(1000);

        Self {
            pool,
            queue_manager,
            broadcaster,
        }
    }

    /// Import a CSV file containing claims with full queue integration and progress tracking
    pub async fn import_file(&self, file_path: &Path) -> Result<ImportResult> {
        self.import_file_with_queue(file_path, None).await
    }

    /// Enqueue a file for processing (adds to staging.file_processing_queue)
    pub async fn enqueue_file(&self, file_path: &Path) -> Result<i64> {
        let file_path_str = file_path.display().to_string();
        let filename = file_path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();

        info!("Enqueuing file for processing: {}", filename);

        // Detect file type based on extension
        let file_extension = file_path.extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("");
        let file_ext_lower = file_extension.to_lowercase();

        let (batch_type, file_format, queue_format) = match file_ext_lower.as_str() {
            "edi" | "837p" => {
                info!("Detected EDI file format: .{}", file_ext_lower);
                ("EDI_837P", "837P", pro_worker::types::FileFormat::Edi837p)
            },
            "csv" => {
                info!("Detected CSV file format");
                ("CSV", "ATHENA", pro_worker::types::FileFormat::Csv)
            },
            _ => {
                warn!("Unknown file extension: {}, defaulting to CSV", file_extension);
                ("CSV", "ATHENA", pro_worker::types::FileFormat::Csv)
            }
        };

        // Calculate file hash for deduplication
        let file_hash = self.calculate_file_hash(file_path)?;

        // Get facility info from file
        let (facility_id, org_id) = self.extract_facility_info(file_path, &file_ext_lower).await?;

        // Create import batch and get generated ID
        let batch_id: i64 = sqlx::query_scalar(
            r#"
            INSERT INTO staging.import_batch (
                organization_id,
                facility_id,
                batch_name,
                batch_type,
                file_format,
                original_filename,
                file_path,
                file_hash,
                import_status,
                total_records
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            RETURNING batch_id
            "#
        )
        .bind(org_id)
        .bind(facility_id)
        .bind(&filename)
        .bind(batch_type)
        .bind(file_format)
        .bind(&filename)
        .bind(&file_path_str)
        .bind(&file_hash)
        .bind("QUEUED")
        .bind(0) // Will be updated when processing starts
        .fetch_one(&self.pool)
        .await
        .context("Failed to create import batch record")?;

        // Enqueue in file_processing_queue
        // facility_id is guaranteed to be Some() at this point (either from lookup or fallback)
        let queue_id = self.queue_manager.enqueue_file(
            facility_id.expect("facility_id should always be present after extract_facility_info"),
            batch_id,
            file_path_str,
            file_hash,
            queue_format,
            org_id,
            None // Default priority
        ).await?;

        info!("File enqueued successfully: queue_id={}, batch_id={}", queue_id, batch_id);

        Ok(queue_id)
    }

    /// Extract facility information from file (supports both CSV and EDI formats)
    async fn extract_facility_info(&self, file_path: &Path, file_ext: &str) -> Result<(Option<i64>, i64)> {
        let file_path_str = file_path.display().to_string();

        // Route to appropriate parser based on file extension
        let facility_identifier = match file_ext {
            "edi" | "837p" => {
                // Parse EDI file to extract facility NPI
                info!("Extracting facility info from EDI file");
                let file_content = std::fs::read_to_string(file_path)
                    .context("Failed to read EDI file")?;

                let mut edi_parser = EdiParser::new();
                match edi_parser.parse(&file_content) {
                    Ok(transaction) => {
                        // Extract facility NPI from the first claim's service facility
                        if let Some(first_claim) = transaction.claims.first() {
                            if let Some(npi) = &first_claim.service_facility_npi {
                                info!("Extracted facility NPI from EDI: {}", npi);
                                first_claim.service_facility_npi.clone()
                            } else {
                                warn!("No facility NPI in first claim");
                                None
                            }
                        } else {
                            warn!("No claims found in EDI file");
                            None
                        }
                    }
                    Err(e) => {
                        warn!("Failed to parse EDI file for facility extraction: {}", e);
                        None
                    }
                }
            },
            "csv" | _ => {
                // Parse CSV file to extract facility code
                info!("Extracting facility info from CSV file");
                let mut parser = CsvParser::with_auto_detection();
                match parser.parse_file(&file_path_str) {
                    Ok(parsed_rows) => {
                        if let Some(first_row) = parsed_rows.first() {
                            first_row.encounter_fields.get("facility_code").cloned()
                        } else {
                            None
                        }
                    }
                    Err(e) => {
                        warn!("Failed to parse CSV file for facility extraction: {}", e);
                        None
                    }
                }
            }
        };

        // Look up facility by code/NPI if we extracted one
        if let Some(identifier) = facility_identifier {
            let result = sqlx::query_as::<_, (i64, i64)>(
                r#"
                SELECT facility_id, organization_id
                FROM claims.facility
                WHERE facility_code = $1 OR npi = $1
                LIMIT 1
                "#
            )
            .bind(&identifier)
            .fetch_optional(&self.pool)
            .await?;

            if let Some((fac_id, org_id)) = result {
                info!("Found facility in database: facility_id={}", fac_id);
                return Ok((Some(fac_id), org_id));
            } else {
                warn!("Facility not found in database: {}", identifier);
            }
        }

        // Fallback to first organization and its first facility
        let result = sqlx::query_as::<_, (i64, i64)>(
            r#"
            SELECT f.facility_id, f.organization_id
            FROM claims.facility f
            JOIN claims.organization o ON f.organization_id = o.organization_id
            ORDER BY o.organization_id, f.facility_id
            LIMIT 1
            "#
        )
        .fetch_one(&self.pool)
        .await
        .context("No facility found in database")?;

        let (fac_id, org_id) = result;
        info!("Using fallback facility: facility_id={}, organization_id={}", fac_id, org_id);
        Ok((Some(fac_id), org_id))
    }

    /// Calculate SHA-256 hash of file for deduplication
    fn calculate_file_hash(&self, file_path: &Path) -> Result<String> {
        use sha2::{Sha256, Digest};

        // Retry logic for file access (handles cases where file is still being written or locked)
        let mut last_error = None;
        for attempt in 1..=5 {
            match std::fs::File::open(file_path) {
                Ok(mut file) => {
                    let mut hasher = Sha256::new();
                    std::io::copy(&mut file, &mut hasher)
                        .with_context(|| format!("Failed to read file for hashing: {}", file_path.display()))?;
                    let hash = hasher.finalize();
                    if attempt > 1 {
                        info!("Successfully opened file on attempt {}: {}", attempt, file_path.display());
                    }
                    return Ok(format!("{:x}", hash));
                }
                Err(e) => {
                    warn!("Failed to open file (attempt {}): {} - Error: {}", attempt, file_path.display(), e);
                    last_error = Some(e);
                    if attempt < 5 {
                        std::thread::sleep(std::time::Duration::from_millis(1000 * attempt));
                    }
                }
            }
        }

        Err(anyhow::anyhow!(
            "Failed to open file for hashing after 5 attempts: {} - Last error: {:?}",
            file_path.display(),
            last_error
        ))
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
                details
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
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
        .execute(&self.pool)
        .await
        .context("Failed to insert processing metric")?;

        Ok(())
    }

    /// STAGE 1: Ingest CSV file to staging.raw_claims (two-stage pipeline)
    /// This is the new two-stage processing approach:
    /// - Stage 1: Fast ingestion (file -> raw_claims) - THIS METHOD
    /// - Stage 2: Validated processing (raw_claims -> encounters/errors) - ClaimsProcessor
    pub async fn ingest_file_to_staging(&self, file_path: &Path, queue_id: Option<i64>) -> Result<IngestResult> {
        let file_path_str = file_path.display().to_string();
        let filename = file_path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();

        info!("====== STAGE 1: Starting file ingestion to staging: {} ======", file_path_str);
        info!("File name: {}", filename);

        // Parse CSV file using auto-detection
        info!("Parsing CSV file...");
        let parse_start = chrono::Utc::now();
        let mut parser = CsvParser::with_auto_detection();

        let parsed_rows = match parser.parse_file(&file_path_str) {
            Ok(rows) => {
                info!("Successfully parsed CSV file");
                rows
            }
            Err(e) => {
                error!("Failed to parse CSV file: {}", e);
                return Err(e).context("Failed to parse CSV file");
            }
        };
        let parse_end = chrono::Utc::now();

        info!("Parsed {} rows from CSV file", parsed_rows.len());

        if parsed_rows.is_empty() {
            warn!("CSV file is empty, no rows to ingest");
        }

        // Get facility info from the first row to set up the batch
        info!("Looking up facility information...");
        let (facility_id, org_id) = if let Some(first_row) = parsed_rows.first() {
            if let Some(facility_code) = first_row.encounter_fields.get("facility_code") {
                info!("Found facility_code in first row: {}", facility_code);

                let result = sqlx::query_as::<_, (i64, i64)>(
                    r#"
                    SELECT facility_id, organization_id
                    FROM claims.facility
                    WHERE facility_code = $1 OR npi = $1
                    LIMIT 1
                    "#
                )
                .bind(facility_code)
                .fetch_optional(&self.pool)
                .await?;

                if let Some((fac_id, org_id)) = result {
                    info!("Found facility in database: facility_id={}, organization_id={}", fac_id, org_id);
                    (Some(fac_id), org_id)
                } else {
                    warn!("Facility not found in database: {}", facility_code);
                    let org: Option<i64> = sqlx::query_scalar(
                        "SELECT organization_id FROM claims.organization LIMIT 1"
                    )
                    .fetch_optional(&self.pool)
                    .await?;

                    let org = org.context("No organization found in database")?;
                    info!("Using fallback organization: {}", org);
                    (None, org)
                }
            } else {
                warn!("No facility_code found in first row");
                let org: Option<i64> = sqlx::query_scalar(
                    "SELECT organization_id FROM claims.organization LIMIT 1"
                )
                .fetch_optional(&self.pool)
                .await?;

                let org = org.context("No organization found in database")?;
                info!("Using fallback organization: {}", org);
                (None, org)
            }
        } else {
            warn!("CSV file has no rows");
            let org: Option<i64> = sqlx::query_scalar(
                "SELECT organization_id FROM claims.organization LIMIT 1"
            )
            .fetch_optional(&self.pool)
            .await?;

            let org = org.context("No organization found in database")?;
            info!("Using fallback organization: {}", org);
            (None, org)
        };

        // Get queue_id - must be provided for two-stage pipeline
        let queue_id = queue_id.context("queue_id required for two-stage pipeline ingestion")?;

        // Create import batch record with INGESTING status
        let started_at = chrono::Utc::now();

        info!("Creating import batch record");

        let batch_id: i64 = sqlx::query_scalar(
            r#"
            INSERT INTO staging.import_batch (
                organization_id,
                facility_id,
                batch_name,
                batch_type,
                file_format,
                original_filename,
                file_path,
                import_status,
                total_records,
                started_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            RETURNING batch_id
            "#
        )
        .bind(org_id)
        .bind(facility_id)
        .bind(&filename)
        .bind("CSV")
        .bind("ATHENA")
        .bind(&filename)
        .bind(&file_path_str)
        .bind("INGESTING")  // New status for Stage 1
        .bind(parsed_rows.len() as i32)
        .bind(started_at)
        .fetch_one(&self.pool)
        .await
        .context("Failed to create import batch record")?;

        info!("Import batch record created successfully: batch_id={}", batch_id);

        // Log PARSE metric
        if let Err(e) = self.log_processing_metric(
            batch_id,
            "PARSE",
            "CSV Parsing",
            parse_start,
            parse_end,
            parsed_rows.len() as i32,
            parsed_rows.len() as i32,
            0,
            Some(serde_json::json!({
                "filename": filename,
                "format": "CSV",
                "stage": "INGEST"
            }))
        ).await {
            warn!("Failed to log PARSE metric: {}", e);
        }

        // Begin transaction for batch insertion to staging.raw_claims
        let mut tx = self.pool.begin().await
            .context("Failed to begin database transaction")?;

        let ingest_start = chrono::Utc::now();

        // Batch commit configuration
        const BATCH_SIZE: usize = 1000;
        let mut batch_count = 0;
        let mut ingested_count = 0;

        info!("Ingesting {} rows to staging.raw_claims...", parsed_rows.len());

        // Insert each parsed row to staging.raw_claims
        for parsed_row in parsed_rows {
            // Serialize parsed data to JSONB
            let encounter_fields_json = serde_json::to_value(&parsed_row.encounter_fields)
                .context("Failed to serialize encounter_fields to JSON")?;
            let service_line_fields_json = if !parsed_row.service_line_fields.is_empty() {
                Some(serde_json::to_value(&parsed_row.service_line_fields)
                    .context("Failed to serialize service_line_fields to JSON")?)
            } else {
                None
            };
            let diagnosis_fields_json = if !parsed_row.diagnosis_fields.is_empty() {
                Some(serde_json::to_value(&parsed_row.diagnosis_fields)
                    .context("Failed to serialize diagnosis_fields to JSON")?)
            } else {
                None
            };

            // Extract facility_code and date_of_service_from for FIFO ordering
            let facility_code = parsed_row.encounter_fields.get("facility_code").map(|s| s.as_str());
            let date_of_service_from = parsed_row.encounter_fields.get("date_of_service_from")
                .and_then(|s| chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d").ok());

            // Insert to staging.raw_claims
            sqlx::query(
                r#"
                INSERT INTO staging.raw_claims (
                    raw_claim_id,
                    batch_id,
                    queue_id,
                    encounter_fields,
                    service_line_fields,
                    diagnosis_fields,
                    row_number,
                    facility_code,
                    processing_status,
                    date_of_service_from
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                "#
            )
            .bind(0i64) // TODO: Refactor to use RETURNING
            .bind(batch_id)
            .bind(queue_id)
            .bind(encounter_fields_json)
            .bind(service_line_fields_json)
            .bind(diagnosis_fields_json)
            .bind(parsed_row.row_number as i32)
            .bind(facility_code)
            .bind("PENDING")  // Stage 2 will process these
            .bind(date_of_service_from)
            .execute(&mut *tx)
            .await
            .context("Failed to insert raw claim")?;

            ingested_count += 1;
            batch_count += 1;

            // Commit batch every BATCH_SIZE rows for better performance
            if batch_count >= BATCH_SIZE {
                debug!("Committing batch of {} rows to staging.raw_claims", batch_count);
                tx.commit().await
                    .context("Failed to commit batch transaction")?;
                info!("Committed batch: {} claims ingested so far", ingested_count);

                // Start new transaction
                tx = self.pool.begin().await
                    .context("Failed to begin new batch transaction")?;
                batch_count = 0;
            }
        }

        // Commit final batch
        tx.commit().await
            .context("Failed to commit final batch transaction")?;
        info!("Successfully ingested all {} rows to staging.raw_claims", ingested_count);

        // Update batch status to INGESTED
        let completed_at = chrono::Utc::now();
        let duration = (completed_at - started_at).num_milliseconds() as f64 / 1000.0;

        sqlx::query(
            r#"
            UPDATE staging.import_batch
            SET import_status = $1,
                processed_records = $2,
                completed_at = $3,
                processing_duration_seconds = $4
            WHERE batch_id = $5
            "#
        )
        .bind("INGESTED")  // Stage 1 complete, Stage 2 pending
        .bind(ingested_count as i32)
        .bind(completed_at)
        .bind(duration)
        .bind(batch_id)
        .execute(&self.pool)
        .await
        .context("Failed to update import batch record")?;

        // Log INGEST metric (Stage 1 performance)
        let ingest_end = chrono::Utc::now();
        if let Err(e) = self.log_processing_metric_with_stage(
            batch_id,
            "INGEST",  // New metric type for Stage 1
            "Staging Ingestion",
            ingest_start,
            ingest_end,
            ingested_count as i32,
            ingested_count as i32,
            0,
            Some(serde_json::json!({
                "total_rows": ingested_count,
                "batch_id": batch_id,
                "queue_id": queue_id
            })),
            "INGEST"  // Processing stage
        ).await {
            warn!("Failed to log INGEST metric: {}", e);
        }

        info!("====== STAGE 1 COMPLETE: {} rows ingested to staging.raw_claims ======", ingested_count);

        Ok(IngestResult {
            batch_id,
            total_rows: ingested_count,
            ingested_at: ingest_end,
        })
    }

    /// STAGE 1: Ingest EDI 837p file to staging.raw_claims (two-stage pipeline)
    /// Similar to CSV ingestion but uses EDI parser
    pub async fn ingest_edi_to_staging(&self, file_path: &Path, queue_id: Option<i64>) -> Result<IngestResult> {
        let file_path_str = file_path.display().to_string();
        let filename = file_path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();

        info!("====== STAGE 1: Starting EDI file ingestion to staging: {} ======", file_path_str);

        // Get batch_id from queue (it was created during enqueue_file)
        let batch_id: i64 = sqlx::query_scalar(
            "SELECT import_batch_id FROM staging.file_processing_queue WHERE queue_id = $1"
        )
        .bind(queue_id.ok_or_else(|| anyhow::anyhow!("queue_id required for EDI ingestion"))?)
        .fetch_one(&self.pool)
        .await
        .context("Failed to get batch_id from queue")?;

        info!("Parsing EDI 837p file...");
        let parse_start = chrono::Utc::now();

        use pro_parser_edi::EdiParser;
        let mut parser = EdiParser::new();

        let transaction = match parser.parse_file(&file_path_str) {
            Ok(txn) => {
                info!("Successfully parsed EDI 837p file: {} claims", txn.claims.len());
                txn
            }
            Err(e) => {
                error!("Failed to parse EDI 837p file: {}", e);
                return Err(anyhow::anyhow!("Failed to parse EDI 837p file: {}", e));
            }
        };
        let parse_end = chrono::Utc::now();

        info!("Parsed {} claims from EDI file", transaction.claims.len());
        info!("Submitter info: org_name={:?}, id_qualifier={:?}, id_code={:?}",
            transaction.submitter.submitter_organization_name,
            transaction.submitter.identification_code_qualifier,
            transaction.submitter.identification_code);

        // Log PARSE metric
        if let Err(e) = self.log_processing_metric_with_stage(
            queue_id.unwrap(),
            "FILE_PROCESSING",
            "PARSE",
            parse_start,
            parse_end,
            transaction.claims.len() as i32,
            transaction.claims.len() as i32,
            0,
            Some(serde_json::json!({
                "filename": filename,
                "format": "EDI_837P"
            })),
            "INGEST"
        ).await {
            warn!("Failed to log PARSE metric: {}", e);
        }

        // Begin transaction for batch insertion to staging.raw_claims
        let mut tx = self.pool.begin().await
            .context("Failed to begin database transaction")?;

        let ingest_start = chrono::Utc::now();

        const BATCH_SIZE: usize = 1000;
        let mut batch_count = 0;
        let mut ingested_count = 0;

        // Insert each claim into staging.raw_claims as JSONB
        for (idx, claim) in transaction.claims.iter().enumerate() {
            let row_number = (idx + 1) as i32;

            // DEBUG: Log the parsed claim to see what we got from the parser
            info!("Claim {}: subscriber_name='{}  {}', payer_name='{}', patient_control='{}', date_of_service={}",
                idx + 1,
                claim.subscriber_first_name,
                claim.subscriber_last_name,
                claim.payer_name,
                claim.patient_control_number,
                claim.date_of_service_from
            );

            // Transform EDI ParsedClaim to match CSV database structure:
            // - encounter_fields: Main claim/subscriber/payer data (JSONB)
            // - service_line_fields: Service lines (JSONB HashMap)
            // - diagnosis_fields: Diagnosis codes (JSONB HashMap)
            use serde_json::Map;

            // ENCOUNTER FIELDS - Main claim data
            let mut encounter_fields = Map::new();

            // Subscriber information
            encounter_fields.insert("subscriber_last_name".to_string(), serde_json::json!(claim.subscriber_last_name));
            encounter_fields.insert("subscriber_first_name".to_string(), serde_json::json!(claim.subscriber_first_name));
            encounter_fields.insert("subscriber_middle_name".to_string(), serde_json::json!(claim.subscriber_middle_name.clone().unwrap_or_default()));
            encounter_fields.insert("subscriber_name_suffix".to_string(), serde_json::json!(claim.subscriber_name_suffix.clone().unwrap_or_default()));
            encounter_fields.insert("subscriber_id".to_string(), serde_json::json!(claim.subscriber_id));
            // Use subscriber_birth_date to match claims_processor expectation
            encounter_fields.insert("subscriber_birth_date".to_string(), serde_json::json!(claim.subscriber_date_of_birth.map(|d| d.format("%Y-%m-%d").to_string()).unwrap_or_default()));
            encounter_fields.insert("subscriber_gender".to_string(), serde_json::json!(claim.subscriber_gender.clone().unwrap_or_default()));
            encounter_fields.insert("subscriber_address_line1".to_string(), serde_json::json!(claim.subscriber_address_line1.clone().unwrap_or_default()));
            encounter_fields.insert("subscriber_address_line2".to_string(), serde_json::json!(claim.subscriber_address_line2.clone().unwrap_or_default()));
            encounter_fields.insert("subscriber_city".to_string(), serde_json::json!(claim.subscriber_city.clone().unwrap_or_default()));
            encounter_fields.insert("subscriber_state".to_string(), serde_json::json!(claim.subscriber_state.clone().unwrap_or_default()));
            encounter_fields.insert("subscriber_postal_code".to_string(), serde_json::json!(claim.subscriber_postal_code.clone().unwrap_or_default()));
            encounter_fields.insert("subscriber_country".to_string(), serde_json::json!(claim.subscriber_country.clone().unwrap_or_default()));
            encounter_fields.insert("medical_record_number".to_string(), serde_json::json!(claim.medical_record_number.clone().unwrap_or_default()));

            // Payer information
            encounter_fields.insert("payer_name".to_string(), serde_json::json!(claim.payer_name));
            encounter_fields.insert("payer_id".to_string(), serde_json::json!(claim.payer_id));
            encounter_fields.insert("payer_address_line1".to_string(), serde_json::json!(claim.payer_address_line1.clone().unwrap_or_default()));
            encounter_fields.insert("payer_address_line2".to_string(), serde_json::json!(claim.payer_address_line2.clone().unwrap_or_default()));
            encounter_fields.insert("payer_city".to_string(), serde_json::json!(claim.payer_city.clone().unwrap_or_default()));
            encounter_fields.insert("payer_state".to_string(), serde_json::json!(claim.payer_state.clone().unwrap_or_default()));
            encounter_fields.insert("payer_postal_code".to_string(), serde_json::json!(claim.payer_postal_code.clone().unwrap_or_default()));

            // Claim information
            encounter_fields.insert("patient_control_number".to_string(), serde_json::json!(claim.patient_control_number));
            encounter_fields.insert("total_claim_charge_amount".to_string(), serde_json::json!(claim.total_claim_charge_amount.to_string()));
            encounter_fields.insert("place_of_service_code".to_string(), serde_json::json!(claim.place_of_service_code.clone().unwrap_or_default()));
            encounter_fields.insert("claim_frequency_code".to_string(), serde_json::json!(claim.claim_frequency_code.clone().unwrap_or_default()));
            encounter_fields.insert("signature_indicator".to_string(), serde_json::json!(claim.provider_signature_indicator.clone().unwrap_or_default()));
            encounter_fields.insert("assignment_indicator".to_string(), serde_json::json!(claim.assignment_indicator.clone().unwrap_or_default()));
            encounter_fields.insert("benefits_assignment_indicator".to_string(), serde_json::json!(claim.benefits_assignment_indicator.clone().unwrap_or_default()));
            encounter_fields.insert("release_of_information_code".to_string(), serde_json::json!(claim.release_of_information_code.clone().unwrap_or_default()));
            encounter_fields.insert("patient_signature_code".to_string(), serde_json::json!(claim.patient_signature_code.clone().unwrap_or_default()));
            encounter_fields.insert("date_of_service_from".to_string(), serde_json::json!(claim.date_of_service_from.format("%Y-%m-%d").to_string()));
            encounter_fields.insert("date_of_service_to".to_string(), serde_json::json!(claim.date_of_service_to.map(|d| d.format("%Y-%m-%d").to_string()).unwrap_or_default()));

            // Claim dates (Loop 2300 DTP segments)
            encounter_fields.insert("onset_of_illness_date".to_string(), serde_json::json!(claim.onset_of_illness_date.map(|d| d.format("%Y-%m-%d").to_string()).unwrap_or_default()));
            encounter_fields.insert("initial_treatment_date".to_string(), serde_json::json!(claim.initial_treatment_date.map(|d| d.format("%Y-%m-%d").to_string()).unwrap_or_default()));
            encounter_fields.insert("last_seen_date".to_string(), serde_json::json!(claim.last_seen_date.map(|d| d.format("%Y-%m-%d").to_string()).unwrap_or_default()));
            encounter_fields.insert("acute_manifestation_date".to_string(), serde_json::json!(claim.acute_manifestation_date.map(|d| d.format("%Y-%m-%d").to_string()).unwrap_or_default()));
            encounter_fields.insert("accident_date".to_string(), serde_json::json!(claim.accident_date.map(|d| d.format("%Y-%m-%d").to_string()).unwrap_or_default()));
            encounter_fields.insert("last_menstrual_period_date".to_string(), serde_json::json!(claim.last_menstrual_period_date.map(|d| d.format("%Y-%m-%d").to_string()).unwrap_or_default()));
            encounter_fields.insert("last_xray_date".to_string(), serde_json::json!(claim.last_xray_date.map(|d| d.format("%Y-%m-%d").to_string()).unwrap_or_default()));
            encounter_fields.insert("disability_from_date".to_string(), serde_json::json!(claim.disability_from_date.map(|d| d.format("%Y-%m-%d").to_string()).unwrap_or_default()));
            encounter_fields.insert("disability_to_date".to_string(), serde_json::json!(claim.disability_to_date.map(|d| d.format("%Y-%m-%d").to_string()).unwrap_or_default()));
            encounter_fields.insert("last_worked_date".to_string(), serde_json::json!(claim.last_worked_date.map(|d| d.format("%Y-%m-%d").to_string()).unwrap_or_default()));
            encounter_fields.insert("authorized_return_to_work_date".to_string(), serde_json::json!(claim.authorized_return_to_work_date.map(|d| d.format("%Y-%m-%d").to_string()).unwrap_or_default()));
            encounter_fields.insert("admission_date".to_string(), serde_json::json!(claim.admission_date.map(|d| d.format("%Y-%m-%d").to_string()).unwrap_or_default()));
            encounter_fields.insert("discharge_date".to_string(), serde_json::json!(claim.discharge_date.map(|d| d.format("%Y-%m-%d").to_string()).unwrap_or_default()));

            // Claim supplemental information
            encounter_fields.insert("delay_reason_code".to_string(), serde_json::json!(claim.delay_reason_code.clone().unwrap_or_default()));
            encounter_fields.insert("special_program_code".to_string(), serde_json::json!(claim.special_program_code.clone().unwrap_or_default()));
            encounter_fields.insert("patient_amount_paid".to_string(), serde_json::json!(claim.patient_amount_paid.map(|d| d.to_string()).unwrap_or_default()));
            encounter_fields.insert("patient_responsibility_amount".to_string(), serde_json::json!(claim.patient_responsibility_amount.map(|d| d.to_string()).unwrap_or_default()));
            encounter_fields.insert("service_authorization_code".to_string(), serde_json::json!(claim.service_authorization_code.clone().unwrap_or_default()));

            // Claim identifiers
            encounter_fields.insert("claim_number".to_string(), serde_json::json!(claim.claim_number.clone().unwrap_or_default()));

            // Claim note (will be inserted into encounter_note table separately)
            encounter_fields.insert("claim_note".to_string(), serde_json::json!(claim.claim_note.clone().unwrap_or_default()));

            // Related causes and accident information
            encounter_fields.insert("related_causes_code_1".to_string(), serde_json::json!(claim.related_causes_code_1.clone().unwrap_or_default()));
            encounter_fields.insert("related_causes_code_2".to_string(), serde_json::json!(claim.related_causes_code_2.clone().unwrap_or_default()));
            encounter_fields.insert("related_causes_code_3".to_string(), serde_json::json!(claim.related_causes_code_3.clone().unwrap_or_default()));
            encounter_fields.insert("auto_accident_state".to_string(), serde_json::json!(claim.auto_accident_state.clone().unwrap_or_default()));
            encounter_fields.insert("auto_accident_country".to_string(), serde_json::json!(claim.auto_accident_country.clone().unwrap_or_default()));

            // COB (Coordination of Benefits) information
            encounter_fields.insert("other_payer_paid_amount".to_string(), serde_json::json!(claim.other_payer_paid_amount.map(|d| d.to_string()).unwrap_or_default()));
            encounter_fields.insert("other_payer_id".to_string(), serde_json::json!(claim.other_payer_id.clone().unwrap_or_default()));
            encounter_fields.insert("other_payer_name".to_string(), serde_json::json!(claim.other_payer_name.clone().unwrap_or_default()));
            encounter_fields.insert("other_payer_claim_number".to_string(), serde_json::json!(claim.other_payer_claim_number.clone().unwrap_or_default()));

            // Ambulance Information (CR1 segment)
            encounter_fields.insert("ambulance_transport_reason_code".to_string(), serde_json::json!(claim.ambulance_transport_reason_code.clone().unwrap_or_default()));
            encounter_fields.insert("ambulance_transport_distance".to_string(), serde_json::json!(claim.ambulance_transport_distance.map(|d| d.to_string()).unwrap_or_default()));
            encounter_fields.insert("ambulance_patient_weight".to_string(), serde_json::json!(claim.ambulance_patient_weight.map(|d| d.to_string()).unwrap_or_default()));
            encounter_fields.insert("ambulance_patient_count".to_string(), serde_json::json!(claim.ambulance_patient_count.map(|i| i.to_string()).unwrap_or_default()));

            // Paperwork/Attachments (PWK segment)
            encounter_fields.insert("paperwork_report_type".to_string(), serde_json::json!(claim.paperwork_report_type.clone().unwrap_or_default()));
            encounter_fields.insert("paperwork_transmission_code".to_string(), serde_json::json!(claim.paperwork_transmission_code.clone().unwrap_or_default()));
            encounter_fields.insert("paperwork_control_number".to_string(), serde_json::json!(claim.paperwork_control_number.clone().unwrap_or_default()));

            // Condition Indicators (CRC segments)
            encounter_fields.insert("condition_codes".to_string(), serde_json::json!(claim.condition_codes.join(",")));

            // Provider NPIs and Names
            encounter_fields.insert("rendering_provider_npi".to_string(), serde_json::json!(claim.rendering_provider_npi.clone().unwrap_or_default()));
            encounter_fields.insert("rendering_provider_last_name".to_string(), serde_json::json!(claim.rendering_provider_last_name.clone().unwrap_or_default()));
            encounter_fields.insert("rendering_provider_first_name".to_string(), serde_json::json!(claim.rendering_provider_first_name.clone().unwrap_or_default()));
            encounter_fields.insert("rendering_provider_taxonomy".to_string(), serde_json::json!(claim.rendering_provider_taxonomy.clone().unwrap_or_default()));

            encounter_fields.insert("referring_provider_npi".to_string(), serde_json::json!(claim.referring_provider_npi.clone().unwrap_or_default()));
            encounter_fields.insert("referring_provider_last_name".to_string(), serde_json::json!(claim.referring_provider_last_name.clone().unwrap_or_default()));
            encounter_fields.insert("referring_provider_first_name".to_string(), serde_json::json!(claim.referring_provider_first_name.clone().unwrap_or_default()));

            encounter_fields.insert("supervising_provider_npi".to_string(), serde_json::json!(claim.supervising_provider_npi.clone().unwrap_or_default()));
            encounter_fields.insert("supervising_provider_last_name".to_string(), serde_json::json!(claim.supervising_provider_last_name.clone().unwrap_or_default()));
            encounter_fields.insert("supervising_provider_first_name".to_string(), serde_json::json!(claim.supervising_provider_first_name.clone().unwrap_or_default()));

            encounter_fields.insert("billing_provider_npi".to_string(), serde_json::json!(transaction.billing_provider.npi.clone()));
            encounter_fields.insert("billing_provider_name".to_string(), serde_json::json!(transaction.billing_provider.organization_name.clone().unwrap_or_default()));
            encounter_fields.insert("billing_provider_tax_id".to_string(), serde_json::json!(transaction.billing_provider.tax_id.clone().unwrap_or_default()));

            encounter_fields.insert("service_facility_npi".to_string(), serde_json::json!(claim.service_facility_npi.clone().unwrap_or_default()));
            encounter_fields.insert("service_facility_name".to_string(), serde_json::json!(claim.service_facility_name.clone().unwrap_or_default()));

            // CRITICAL: facility_npi for Stage 2 facility resolution
            encounter_fields.insert("facility_npi".to_string(), serde_json::json!(claim.service_facility_npi.clone().unwrap_or_default()));

            // CRITICAL: facility_code extraction - ALWAYS from NM1*85 (Billing Provider) ONLY
            // Check qualifier 46 first, then REF segments (G2/1C/1J), then default to NPI from NM1*85
            info!("Claim {}: BillingProvider values - npi='{}' (len={}), facility_id={:?}, provider_number={:?}, org_name={:?}",
                idx + 1,
                transaction.billing_provider.npi,
                transaction.billing_provider.npi.len(),
                transaction.billing_provider.facility_id,
                transaction.billing_provider.provider_number,
                transaction.billing_provider.organization_name
            );

            let facility_code = if let Some(facility_id) = &transaction.billing_provider.facility_id {
                info!("Claim {}: Using facility_code from NM1*85 qualifier 46: {}", idx + 1, facility_id);
                facility_id.clone()
            } else if let Some(provider_number) = &transaction.billing_provider.provider_number {
                info!("Claim {}: Using facility_code from NM1*85 REF (G2/1C/1J): {}", idx + 1, provider_number);
                provider_number.clone()
            } else if !transaction.billing_provider.npi.is_empty() {
                info!("Claim {}: Using facility_code from NM1*85 NPI: {}", idx + 1, transaction.billing_provider.npi);
                transaction.billing_provider.npi.clone()
            } else {
                error!("Claim {}: NM1*85 billing provider has no identifier - npi='{}', facility_id={:?}, provider_number={:?}",
                    idx + 1,
                    transaction.billing_provider.npi,
                    transaction.billing_provider.facility_id,
                    transaction.billing_provider.provider_number
                );
                return Err(anyhow::anyhow!("Claim {}: NM1*85 billing provider has no identifier (no qualifier 46, REF, or NPI)", idx + 1));
            };

            encounter_fields.insert("facility_code".to_string(), serde_json::json!(facility_code));

            let encounter_fields_json = serde_json::Value::Object(encounter_fields);

            // DIAGNOSIS FIELDS - Separate JSONB column as HashMap<String, Vec<String>>
            let mut diagnosis_map: std::collections::HashMap<String, Vec<String>> = std::collections::HashMap::new();
            for (i, diagnosis) in claim.diagnoses.iter().enumerate() {
                let field_name = format!("diagnosis_code_{}", i + 1);
                diagnosis_map.insert(field_name, vec![diagnosis.diagnosis_code.clone()]);
            }
            let diagnosis_fields_json = serde_json::to_value(&diagnosis_map)
                .context("Failed to serialize diagnosis fields")?;

            // SERVICE LINE FIELDS - Separate JSONB column as HashMap<String, String>
            // Process ALL service lines (not just first 3)
            let mut service_line_map: std::collections::HashMap<String, String> = std::collections::HashMap::new();
            for (i, line) in claim.service_lines.iter().enumerate() {
                let prefix = format!("service_line_{}", i + 1);

                // Basic service line information
                service_line_map.insert(format!("{}_date_from", prefix), line.service_date_from.format("%Y-%m-%d").to_string());
                service_line_map.insert(format!("{}_date_to", prefix), line.service_date_to.map(|d| d.format("%Y-%m-%d").to_string()).unwrap_or_default());
                service_line_map.insert(format!("{}_procedure_code", prefix), line.procedure_code.clone());
                service_line_map.insert(format!("{}_product_service_id_qualifier", prefix), line.product_service_id_qualifier.clone());
                service_line_map.insert(format!("{}_modifier_1", prefix), line.procedure_modifier_1.clone().unwrap_or_default());
                service_line_map.insert(format!("{}_modifier_2", prefix), line.procedure_modifier_2.clone().unwrap_or_default());
                service_line_map.insert(format!("{}_modifier_3", prefix), line.procedure_modifier_3.clone().unwrap_or_default());
                service_line_map.insert(format!("{}_modifier_4", prefix), line.procedure_modifier_4.clone().unwrap_or_default());
                service_line_map.insert(format!("{}_charge_amount", prefix), line.line_item_charge_amount.to_string());
                service_line_map.insert(format!("{}_units", prefix), line.service_unit_count.to_string());
                service_line_map.insert(format!("{}_unit_basis_measurement_code", prefix), line.unit_basis_measurement_code.clone());
                service_line_map.insert(format!("{}_place_of_service_code", prefix), line.place_of_service_code.clone().unwrap_or_default());

                // Service line indicators
                service_line_map.insert(format!("{}_emergency_indicator", prefix), line.emergency_indicator.map(|b| if b { "Y" } else { "N" }).unwrap_or("N").to_string());
                service_line_map.insert(format!("{}_epsdt_indicator", prefix), line.epsdt_indicator.map(|b| if b { "Y" } else { "N" }).unwrap_or("N").to_string());
                service_line_map.insert(format!("{}_family_planning_indicator", prefix), line.family_planning_indicator.map(|b| if b { "Y" } else { "N" }).unwrap_or("N").to_string());

                // NDC (drug) information
                service_line_map.insert(format!("{}_ndc_code", prefix), line.ndc_code.clone().unwrap_or_default());
                service_line_map.insert(format!("{}_ndc_unit_count", prefix), line.ndc_unit_count.map(|d| d.to_string()).unwrap_or_default());
                service_line_map.insert(format!("{}_ndc_measurement_unit", prefix), line.ndc_measurement_unit.clone().unwrap_or_default());

                // Authorization and referral
                service_line_map.insert(format!("{}_prior_authorization_number", prefix), line.prior_authorization_number.clone().unwrap_or_default());
                service_line_map.insert(format!("{}_referral_number", prefix), line.referral_number.clone().unwrap_or_default());

                // Line note and revenue code
                service_line_map.insert(format!("{}_line_note", prefix), line.line_note.clone().unwrap_or_default());
                service_line_map.insert(format!("{}_revenue_code", prefix), line.revenue_code.clone().unwrap_or_default());

                // COB (Other payer information)
                service_line_map.insert(format!("{}_other_payer_line_paid_amount", prefix), line.other_payer_line_paid_amount.map(|d| d.to_string()).unwrap_or_default());

                // HCP - Health Care Pricing (adjudication)
                service_line_map.insert(format!("{}_allowed_amount", prefix), line.allowed_amount.map(|d| d.to_string()).unwrap_or_default());
                service_line_map.insert(format!("{}_saving_amount", prefix), line.saving_amount.map(|d| d.to_string()).unwrap_or_default());

                // Diagnosis pointers
                let mut pointers = Vec::new();
                if let Some(p1) = line.diagnosis_code_pointer_1 {
                    pointers.push(p1.to_string());
                }
                if let Some(p2) = line.diagnosis_code_pointer_2 {
                    pointers.push(p2.to_string());
                }
                if let Some(p3) = line.diagnosis_code_pointer_3 {
                    pointers.push(p3.to_string());
                }
                if let Some(p4) = line.diagnosis_code_pointer_4 {
                    pointers.push(p4.to_string());
                }
                service_line_map.insert(format!("{}_diagnosis_pointers", prefix), pointers.join(","));

                // Provider NPIs and names at service line level (Loop 2420)
                service_line_map.insert(format!("{}_rendering_provider_npi", prefix), line.rendering_provider_npi.clone().unwrap_or_default());
                service_line_map.insert(format!("{}_rendering_provider_last_name", prefix), line.rendering_provider_last_name.clone().unwrap_or_default());
                service_line_map.insert(format!("{}_rendering_provider_first_name", prefix), line.rendering_provider_first_name.clone().unwrap_or_default());
                service_line_map.insert(format!("{}_ordering_provider_npi", prefix), line.ordering_provider_npi.clone().unwrap_or_default());
                service_line_map.insert(format!("{}_ordering_provider_last_name", prefix), line.ordering_provider_last_name.clone().unwrap_or_default());
                service_line_map.insert(format!("{}_ordering_provider_first_name", prefix), line.ordering_provider_first_name.clone().unwrap_or_default());
                service_line_map.insert(format!("{}_supervising_provider_npi", prefix), line.supervising_provider_npi.clone().unwrap_or_default());
                service_line_map.insert(format!("{}_referring_provider_npi", prefix), line.referring_provider_npi.clone().unwrap_or_default());
            }
            let service_line_fields_json = serde_json::to_value(&service_line_map)
                .context("Failed to serialize service line fields")?;

            // Insert into staging.raw_claims with all 3 JSONB columns
            sqlx::query(
                r#"
                INSERT INTO staging.raw_claims (
                    batch_id,
                    queue_id,
                    encounter_fields,
                    service_line_fields,
                    diagnosis_fields,
                    row_number,
                    processing_status,
                    date_of_service_from
                )
                VALUES ($1, $2, $3, $4, $5, $6, 'PENDING', $7)
                "#
            )
            .bind(batch_id)
            .bind(queue_id.unwrap())
            .bind(&encounter_fields_json)
            .bind(&service_line_fields_json)
            .bind(&diagnosis_fields_json)
            .bind(row_number)
            .bind(claim.date_of_service_from)
            .execute(&mut *tx)
            .await
            .context("Failed to insert claim into staging.raw_claims")?;

            ingested_count += 1;
            batch_count += 1;

            // Commit in batches
            if batch_count >= BATCH_SIZE {
                tx.commit().await
                    .context("Failed to commit batch transaction")?;
                info!("Committed batch of {} claims to staging.raw_claims", batch_count);

                // Start new transaction
                tx = self.pool.begin().await
                    .context("Failed to begin new transaction")?;
                batch_count = 0;
            }
        }

        // Commit remaining rows
        if batch_count > 0 {
            tx.commit().await
                .context("Failed to commit final transaction")?;
            info!("Committed final batch of {} claims to staging.raw_claims", batch_count);
        }

        let ingest_end = chrono::Utc::now();

        // Log INGEST metric
        if let Err(e) = self.log_processing_metric_with_stage(
            queue_id.unwrap(),
            "FILE_PROCESSING",
            "INGEST",
            ingest_start,
            ingest_end,
            ingested_count as i32,
            ingested_count as i32,
            0,
            Some(serde_json::json!({
                "filename": filename,
                "format": "EDI_837P"
            })),
            "INGEST"
        ).await {
            warn!("Failed to log INGEST metric: {}", e);
        }

        info!("====== STAGE 1 COMPLETE: {} EDI claims ingested to staging ======", ingested_count);

        Ok(IngestResult {
            batch_id,
            total_rows: ingested_count,
            ingested_at: ingest_end,
        })
    }

    /// Log a processing metric with processing_stage column (for two-stage pipeline)
    async fn log_processing_metric_with_stage(
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

    /// Import a CSV file with optional queue_id for tracking
    /// LEGACY METHOD - For backward compatibility with single-stage processing
    /// New code should use ingest_file_to_staging() for Stage 1 instead
    pub async fn import_file_with_queue(&self, file_path: &Path, queue_id: Option<i64>) -> Result<ImportResult> {
        let file_path_str = file_path.display().to_string();
        let filename = file_path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();

        info!("====== Starting import of file: {} ======", file_path_str);
        info!("File name: {}", filename);

        // Parse CSV file using auto-detection
        info!("Parsing CSV file...");
        let parse_start = chrono::Utc::now();
        let mut parser = CsvParser::with_auto_detection();

        let parsed_rows = match parser.parse_file(&file_path_str) {
            Ok(rows) => {
                info!("Successfully parsed CSV file");
                rows
            }
            Err(e) => {
                error!("Failed to parse CSV file: {}", e);
                return Err(e).context("Failed to parse CSV file");
            }
        };
        let parse_end = chrono::Utc::now();

        info!("Parsed {} rows from CSV file", parsed_rows.len());

        if parsed_rows.is_empty() {
            warn!("CSV file is empty, no rows to import");
        }

        // Get facility info from the first row to set up the batch
        // This identifies which facility/organization this file belongs to
        info!("Looking up facility information...");
        let (facility_id, org_id) = if let Some(first_row) = parsed_rows.first() {
            if let Some(facility_code) = first_row.encounter_fields.get("facility_code") {
                info!("Found facility_code in first row: {}", facility_code);

                let result = sqlx::query_as::<_, (i64, i64)>(
                    r#"
                    SELECT facility_id, organization_id
                    FROM claims.facility
                    WHERE facility_code = $1 OR npi = $1
                    LIMIT 1
                    "#
                )
                .bind(facility_code)
                .fetch_optional(&self.pool)
                .await?;

                if let Some((fac_id, org_id)) = result {
                    info!("Found facility in database: facility_id={}, organization_id={}", fac_id, org_id);
                    (Some(fac_id), org_id)
                } else {
                    warn!("Facility not found in database: {}", facility_code);
                    // Facility not found - get first org as fallback
                    let org: Option<i64> = sqlx::query_scalar(
                        "SELECT organization_id FROM claims.organization LIMIT 1"
                    )
                    .fetch_optional(&self.pool)
                    .await?;

                    let org = org.context("No organization found in database")?;
                    info!("Using fallback organization: {}", org);
                    (None, org)
                }
            } else {
                warn!("No facility_code found in first row");
                // No facility code in data - get first org
                let org: Option<i64> = sqlx::query_scalar(
                    "SELECT organization_id FROM claims.organization LIMIT 1"
                )
                .fetch_optional(&self.pool)
                .await?;

                let org = org.context("No organization found in database")?;
                info!("Using fallback organization: {}", org);
                (None, org)
            }
        } else {
            warn!("CSV file has no rows");
            // Empty file - get first org
            let org: Option<i64> = sqlx::query_scalar(
                "SELECT organization_id FROM claims.organization LIMIT 1"
            )
            .fetch_optional(&self.pool)
            .await?;

            let org = org.context("No organization found in database")?;
            info!("Using fallback organization: {}", org);
            (None, org)
        };

        // Create import batch record and get generated ID
        let started_at = chrono::Utc::now();

        info!("Creating import batch record");

        let batch_id: i64 = sqlx::query_scalar(
            r#"
            INSERT INTO staging.import_batch (
                organization_id,
                facility_id,
                batch_name,
                batch_type,
                file_format,
                original_filename,
                file_path,
                import_status,
                total_records,
                started_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            RETURNING batch_id
            "#
        )
        .bind(org_id)
        .bind(facility_id)
        .bind(&filename)
        .bind("CSV")
        .bind("ATHENA")
        .bind(&filename)
        .bind(&file_path_str)
        .bind("PROCESSING")
        .bind(parsed_rows.len() as i32)
        .bind(started_at)
        .fetch_one(&self.pool)
        .await
        .context("Failed to create import batch record")?;

        info!("Import batch record created successfully: batch_id={}", batch_id);

        // Log PARSE metric
        if let Err(e) = self.log_processing_metric(
            batch_id,
            "PARSE",
            "CSV Parsing",
            parse_start,
            parse_end,
            parsed_rows.len() as i32,
            parsed_rows.len() as i32,
            0,
            Some(serde_json::json!({
                "filename": filename,
                "format": "CSV"
            }))
        ).await {
            warn!("Failed to log PARSE metric: {}", e);
        }

        let mut result = ImportResult {
            total_rows: parsed_rows.len(),
            successful: 0,
            failed: 0,
            errors: Vec::new(),
        };

        // Create progress tracker for real-time metrics
        let progress_tracker = if let Some(qid) = queue_id {
            Some(
                ProgressTracker::new(qid, parsed_rows.len(), self.pool.clone(), self.broadcaster.clone())
                    .await
                    .context("Failed to create progress tracker")?
            )
        } else {
            None
        };

        // Begin transaction
        let mut tx = self.pool.begin().await
            .context("Failed to begin database transaction")?;

        let insert_start = chrono::Utc::now();

        // Batch commit configuration - commit every N rows for better performance
        const BATCH_SIZE: usize = 1000;
        let mut batch_count = 0;

        // Facility lookup cache for performance optimization
        // Key: facility_code, Value: (facility_id, organization_id, region_id)
        let mut facility_cache: HashMap<String, (Option<i64>, i64, Option<i64>)> = HashMap::new();

        // Import each row with progress tracking
        for parsed_row in parsed_rows {
            debug!("Processing row {}", parsed_row.row_number);

            // Check for parse errors
            if !parsed_row.errors.is_empty() {
                warn!("Row {} has parse errors: {:?}", parsed_row.row_number, parsed_row.errors);
                result.failed += 1;

                // Log each error to staging.import_error_log
                for error in &parsed_row.errors {
                    let error_message = format!("Row {}: {}", parsed_row.row_number, error);
                    result.errors.push(error_message.clone());

                    // Log to database
                    if let Err(e) = self.log_import_error(
                        batch_id,
                        parsed_row.row_number,
                        None,
                        "VALIDATION",
                        &error_message,
                        None
                    ).await {
                        warn!("Failed to log error to database: {}", e);
                    }

                    // Record failed claim in progress tracker
                    if let Some(ref tracker) = progress_tracker {
                        let claim_number = parsed_row.encounter_fields.get("patient_control_number")
                            .map(|s| s.to_string());
                        let claim_data = serde_json::to_value(&parsed_row.encounter_fields).ok();

                        if let Err(e) = tracker.record_claim_failed(
                            claim_number,
                            error_message,
                            claim_data,
                            Some("VALIDATION".to_string()),
                            false // Validation errors can't be retried
                        ).await {
                            warn!("Failed to record claim failure: {}", e);
                        }
                    }
                }
                continue;
            }

            // Log warnings
            for warning in &parsed_row.warnings {
                debug!("Row {} warning: {}", parsed_row.row_number, warning);
            }

            // Import the row
            let start_time = std::time::Instant::now();
            match self.import_encounter(&mut tx, &parsed_row, &mut facility_cache).await {
                Ok(_) => {
                    result.successful += 1;
                    batch_count += 1;
                    let processing_time_ms = start_time.elapsed().as_millis() as u64;
                    debug!("Successfully imported row {} in {}ms", parsed_row.row_number, processing_time_ms);

                    // Record successful claim processing
                    if let Some(ref tracker) = progress_tracker {
                        let claim_number = parsed_row.encounter_fields.get("patient_control_number")
                            .map(|s| s.to_string());

                        if let Err(e) = tracker.record_claim_processed(
                            claim_number,
                            0, // flags_count - TODO: count actual flags from rules engine
                            0, // critical_flags_count
                            processing_time_ms
                        ).await {
                            warn!("Failed to record claim success: {}", e);
                        }
                    }

                    // Commit batch every BATCH_SIZE rows for better performance
                    if batch_count >= BATCH_SIZE {
                        debug!("Committing batch of {} rows", batch_count);
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
                    let error_message = format!("Row {}: {}", parsed_row.row_number, e);
                    error!("Failed to import row {}: {}", parsed_row.row_number, e);
                    result.failed += 1;
                    result.errors.push(error_message.clone());

                    // Log to database
                    if let Err(log_err) = self.log_import_error(
                        batch_id,
                        parsed_row.row_number,
                        None,
                        "IMPORT",
                        &error_message,
                        None
                    ).await {
                        warn!("Failed to log error to database: {}", log_err);
                    }

                    // Record failed claim in progress tracker
                    if let Some(ref tracker) = progress_tracker {
                        let claim_number = parsed_row.encounter_fields.get("patient_control_number")
                            .map(|s| s.to_string());
                        let claim_data = serde_json::to_value(&parsed_row.encounter_fields).ok();

                        if let Err(e) = tracker.record_claim_failed(
                            claim_number,
                            error_message,
                            claim_data,
                            Some("IMPORT".to_string()),
                            true // Import errors can be retried
                        ).await {
                            warn!("Failed to record claim failure: {}", e);
                        }
                    }
                }
            }
        }

        // Commit or rollback based on results
        let import_status = if result.failed == 0 {
            tx.commit().await
                .context("Failed to commit transaction")?;
            info!("Successfully imported all {} rows", result.successful);

            // Mark progress as completed
            if let Some(ref tracker) = progress_tracker {
                tracker.complete().await
                    .context("Failed to mark progress as completed")?;
            }

            "COMPLETED"
        } else if result.successful > 0 {
            // Partial success - commit what we can
            tx.commit().await
                .context("Failed to commit transaction")?;
            warn!("Imported {} rows, {} failed", result.successful, result.failed);

            // Mark progress as completed (partial success)
            if let Some(ref tracker) = progress_tracker {
                tracker.complete().await
                    .context("Failed to mark progress as completed")?;
            }

            "PARTIAL"
        } else {
            // Total failure - rollback
            tx.rollback().await
                .context("Failed to rollback transaction")?;
            error!("All rows failed to import");

            // Mark progress as failed
            if let Some(ref tracker) = progress_tracker {
                tracker.fail("All claims failed to import".to_string()).await
                    .context("Failed to mark progress as failed")?;
            }

            "FAILED"
        };

        // Update queue status if queue_id was provided
        if let Some(qid) = queue_id {
            if import_status == "COMPLETED" || import_status == "PARTIAL" {
                self.queue_manager.mark_completed(qid).await
                    .context("Failed to mark queue entry as completed")?;
            } else {
                self.queue_manager.mark_failed(qid, "Import failed").await
                    .context("Failed to mark queue entry as failed")?;
            }
        }

        // Update import batch record with final status
        let completed_at = chrono::Utc::now();
        let duration = (completed_at - started_at).num_milliseconds() as f64 / 1000.0;

        sqlx::query(
            r#"
            UPDATE staging.import_batch
            SET import_status = $1,
                processed_records = $2,
                successful_records = $3,
                failed_records = $4,
                completed_at = $5,
                processing_duration_seconds = $6
            WHERE batch_id = $7
            "#
        )
        .bind(import_status)
        .bind(result.total_rows as i32)
        .bind(result.successful as i32)
        .bind(result.failed as i32)
        .bind(completed_at)
        .bind(duration)
        .bind(batch_id)
        .execute(&self.pool)
        .await
        .context("Failed to update import batch record")?;

        // Log INSERT metric
        let insert_end = chrono::Utc::now();
        if let Err(e) = self.log_processing_metric(
            batch_id,
            "INSERT",
            "Database Insert",
            insert_start,
            insert_end,
            result.total_rows as i32,
            result.successful as i32,
            result.failed as i32,
            Some(serde_json::json!({
                "total_rows": result.total_rows,
                "successful": result.successful,
                "failed": result.failed,
                "import_status": import_status
            }))
        ).await {
            warn!("Failed to log INSERT metric: {}", e);
        }

        Ok(result)
    }

    /// Log an import error to the staging.import_error_log table
    async fn log_import_error(
        &self,
        batch_id: i64,
        record_number: usize,
        field_name: Option<String>,
        error_type: &str,
        error_message: &str,
        raw_data: Option<String>,
    ) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO staging.import_error_log (
                batch_id,
                record_number,
                field_name,
                error_type,
                error_severity,
                error_message,
                raw_data
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            "#
        )
        .bind(batch_id)
        .bind(record_number as i32)
        .bind(field_name)
        .bind(error_type)
        .bind("ERROR")
        .bind(error_message)
        .bind(raw_data)
        .execute(&self.pool)
        .await
        .context("Failed to insert import error log")?;

        Ok(())
    }

    /// Import a single encounter (claim) from parsed row
    async fn import_encounter(
        &self,
        tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
        parsed_row: &pro_parser_csv::parser::ParsedRow,
        facility_cache: &mut HashMap<String, (Option<i64>, i64, Option<i64>)>,
    ) -> Result<i64> {
        // Extract facility_code from encounter fields
        let facility_code = parsed_row.encounter_fields.get("facility_code")
            .or_else(|| parsed_row.encounter_fields.get("facility_npi"))
            .context("Missing facility_code or facility_npi")?;

        // Check cache first before querying database
        let (facility_id, organization_id, region_id) = if let Some(cached) = facility_cache.get(facility_code) {
            // Cache hit - use cached values
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

            let result = facility
                .with_context(|| format!("Facility not found: {}", facility_code))?;

            // Store in cache for future lookups
            facility_cache.insert(facility_code.clone(), result);
            result
        };

        // Generate encounter ID
        let encounter_id = 0i64; // TODO: Use RETURNING

        // Extract required encounter fields
        let patient_control_number = parsed_row.encounter_fields.get("patient_control_number")
            .context("Missing patient_control_number")?;
        let subscriber_last_name = parsed_row.encounter_fields.get("subscriber_last_name")
            .context("Missing subscriber_last_name")?;
        let subscriber_first_name = parsed_row.encounter_fields.get("subscriber_first_name")
            .context("Missing subscriber_first_name")?;
        let date_of_service_from = parsed_row.encounter_fields.get("date_of_service_from")
            .context("Missing date_of_service_from")?;

        // Required fields
        let subscriber_id = parsed_row.encounter_fields.get("subscriber_id")
            .context("Missing subscriber_id")?;
        let subscriber_birth_date_str = parsed_row.encounter_fields.get("subscriber_birth_date")
            .context("Missing subscriber_birth_date")?;

        // Optional fields with defaults
        let submitter_id = parsed_row.encounter_fields.get("submitter_id")
            .unwrap_or(facility_code); // Use facility_code as default submitter
        let payer_responsibility_code = parsed_row.encounter_fields.get("payer_responsibility_code")
            .map(|s| s.as_str())
            .unwrap_or("P"); // Default to Primary

        // Calculate total claim charge from service lines
        let total_claim_charge = parsed_row.service_line_fields.get("line_item_charge_amount")
            .and_then(|s| s.parse::<rust_decimal::Decimal>().ok())
            .unwrap_or(rust_decimal::Decimal::ZERO);

        // Parse dates
        let dos_from = chrono::NaiveDate::parse_from_str(date_of_service_from, "%Y-%m-%d")
            .context("Invalid date format for date_of_service_from")?;
        let subscriber_dob = chrono::NaiveDate::parse_from_str(subscriber_birth_date_str, "%Y-%m-%d")
            .context("Invalid date format for subscriber_birth_date")?;

        // Optional fields
        let payer_id = parsed_row.encounter_fields.get("payer_id").map(|s| s.as_str());
        let payer_name = parsed_row.encounter_fields.get("payer_name").map(|s| s.as_str());
        let place_of_service = parsed_row.encounter_fields.get("place_of_service_code").map(|s| s.as_str());
        let medical_record_number = parsed_row.encounter_fields.get("medical_record_number").map(|s| s.as_str());

        // Insert encounter with all required fields
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

        // Import service lines if present
        if !parsed_row.service_line_fields.is_empty() {
            self.import_service_line(tx, encounter_id, parsed_row).await?;
        }

        // Import diagnoses if present
        if !parsed_row.diagnosis_fields.is_empty() {
            self.import_diagnoses(tx, encounter_id, parsed_row).await?;
        }

        Ok(encounter_id)
    }

    /// Import service line for an encounter
    async fn import_service_line(
        &self,
        tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
        encounter_id: i64,
        parsed_row: &pro_parser_csv::parser::ParsedRow,
    ) -> Result<()> {
        let service_line_id = 0i64; // TODO: Use RETURNING

        // Extract required service line fields
        let procedure_code = parsed_row.service_line_fields.get("procedure_code")
            .context("Missing procedure_code")?;
        let line_item_charge_amount = parsed_row.service_line_fields.get("line_item_charge_amount")
            .context("Missing line_item_charge_amount")?;
        let default_unit_count = "1".to_string();
        let service_unit_count = parsed_row.service_line_fields.get("service_unit_count")
            .unwrap_or(&default_unit_count);

        // Get service date - use service_date_from if available, otherwise fall back to encounter DOS
        let service_date_str = parsed_row.service_line_fields.get("service_date_from")
            .or_else(|| parsed_row.encounter_fields.get("date_of_service_from"))
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
        .bind(1) // Line number
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
        encounter_id: i64,
        parsed_row: &pro_parser_csv::parser::ParsedRow,
    ) -> Result<()> {
        // Get all diagnosis codes
        for (field_name, codes) in &parsed_row.diagnosis_fields {
            if field_name == "diagnosis_code" {
                for (idx, code) in codes.iter().enumerate() {
                    let diagnosis_id = 0i64; // TODO: Use RETURNING
                    let sequence_number = (idx + 1) as i16;  // Changed to i16 for SMALLINT

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

                    debug!("Inserted diagnosis {} ({}) for encounter {}", sequence_number, code, encounter_id);
                }
            }
        }

        Ok(())
    }
}

/// Result of a Stage 1 file ingestion operation
#[derive(Debug, Clone)]
pub struct IngestResult {
    pub batch_id: i64,
    pub total_rows: usize,
    pub ingested_at: chrono::DateTime<chrono::Utc>,
}

/// Result of a file import operation (legacy single-stage)
#[derive(Debug, Clone)]
pub struct ImportResult {
    pub total_rows: usize,
    pub successful: usize,
    pub failed: usize,
    pub errors: Vec<String>,
}

impl ImportResult {
    /// Check if import was completely successful
    pub fn is_success(&self) -> bool {
        self.failed == 0 && self.successful > 0
    }

    /// Get a summary string
    pub fn summary(&self) -> String {
        format!(
            "Import complete: {} total, {} successful, {} failed",
            self.total_rows, self.successful, self.failed
        )
    }
}
