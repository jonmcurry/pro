//! Claims importer for CSV files
//!
//! Imports claims data from CSV files into the database using the pro-parser-csv crate.
//! Integrates with the FIFO queue system and progress tracking.

use anyhow::{Context, Result};
use pro_parser_csv::CsvParser;
use pro_worker::progress::ProgressTracker;
use pro_worker::queue_manager::QueueManager;
use sqlx::PgPool;
use std::collections::HashMap;
use std::path::Path;
use tokio::sync::broadcast;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// Claims importer that processes CSV files
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
    pub async fn enqueue_file(&self, file_path: &Path) -> Result<Uuid> {
        let file_path_str = file_path.display().to_string();
        let filename = file_path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();

        info!("Enqueuing file for processing: {}", filename);

        // Calculate file hash for deduplication
        let file_hash = self.calculate_file_hash(file_path)?;

        // Get facility info from file
        let (facility_id, org_id) = self.extract_facility_info(file_path).await?;

        // Create import batch first
        let batch_id = Uuid::new_v4();
        sqlx::query(
            r#"
            INSERT INTO staging.import_batch (
                batch_id,
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
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            "#
        )
        .bind(batch_id)
        .bind(org_id)
        .bind(facility_id)
        .bind(&filename)
        .bind("CSV")
        .bind("ATHENA")
        .bind(&filename)
        .bind(&file_path_str)
        .bind(&file_hash)
        .bind("QUEUED")
        .bind(0) // Will be updated when processing starts
        .execute(&self.pool)
        .await
        .context("Failed to create import batch record")?;

        // Enqueue in file_processing_queue
        let queue_id = self.queue_manager.enqueue_file(
            facility_id.unwrap_or_else(|| Uuid::new_v4()),
            batch_id,
            file_path_str,
            file_hash,
            pro_worker::types::FileFormat::Csv,
            org_id,
            None // Default priority
        ).await?;

        info!("File enqueued successfully: queue_id={}, batch_id={}", queue_id, batch_id);

        Ok(queue_id)
    }

    /// Extract facility information from file
    async fn extract_facility_info(&self, file_path: &Path) -> Result<(Option<Uuid>, Uuid)> {
        // Parse just the first row to get facility info
        let file_path_str = file_path.display().to_string();
        let mut parser = CsvParser::with_auto_detection();
        let parsed_rows = parser.parse_file(&file_path_str)?;

        if let Some(first_row) = parsed_rows.first() {
            if let Some(facility_code) = first_row.encounter_fields.get("facility_code") {
                let result = sqlx::query_as::<_, (Uuid, Uuid)>(
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
                    return Ok((Some(fac_id), org_id));
                }
            }
        }

        // Fallback to first organization
        let org: Uuid = sqlx::query_scalar(
            "SELECT organization_id FROM claims.organization LIMIT 1"
        )
        .fetch_one(&self.pool)
        .await
        .context("No organization found in database")?;

        Ok((None, org))
    }

    /// Calculate SHA-256 hash of file for deduplication
    fn calculate_file_hash(&self, file_path: &Path) -> Result<String> {
        use sha2::{Sha256, Digest};

        let mut file = std::fs::File::open(file_path)
            .context("Failed to open file for hashing")?;
        let mut hasher = Sha256::new();
        std::io::copy(&mut file, &mut hasher)
            .context("Failed to read file for hashing")?;

        let hash = hasher.finalize();
        Ok(format!("{:x}", hash))
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
        .execute(&self.pool)
        .await
        .context("Failed to insert processing metric")?;

        Ok(())
    }

    /// STAGE 1: Ingest CSV file to staging.raw_claims (two-stage pipeline)
    /// This is the new two-stage processing approach:
    /// - Stage 1: Fast ingestion (file -> raw_claims) - THIS METHOD
    /// - Stage 2: Validated processing (raw_claims -> encounters/errors) - ClaimsProcessor
    pub async fn ingest_file_to_staging(&self, file_path: &Path, queue_id: Option<Uuid>) -> Result<IngestResult> {
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

                let result = sqlx::query_as::<_, (Uuid, Uuid)>(
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
                    let org: Option<Uuid> = sqlx::query_scalar(
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
                let org: Option<Uuid> = sqlx::query_scalar(
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
            let org: Option<Uuid> = sqlx::query_scalar(
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
        let batch_id = Uuid::new_v4();
        let started_at = chrono::Utc::now();

        info!("Creating import batch record: batch_id={}", batch_id);

        sqlx::query(
            r#"
            INSERT INTO staging.import_batch (
                batch_id,
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
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            "#
        )
        .bind(batch_id)
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
        .execute(&self.pool)
        .await
        .context("Failed to create import batch record")?;

        info!("Import batch record created successfully");

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
            .bind(Uuid::new_v4())
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

    /// Log a processing metric with processing_stage column (for two-stage pipeline)
    async fn log_processing_metric_with_stage(
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

    /// Import a CSV file with optional queue_id for tracking
    /// LEGACY METHOD - For backward compatibility with single-stage processing
    /// New code should use ingest_file_to_staging() for Stage 1 instead
    pub async fn import_file_with_queue(&self, file_path: &Path, queue_id: Option<Uuid>) -> Result<ImportResult> {
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

                let result = sqlx::query_as::<_, (Uuid, Uuid)>(
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
                    let org: Option<Uuid> = sqlx::query_scalar(
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
                let org: Option<Uuid> = sqlx::query_scalar(
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
            let org: Option<Uuid> = sqlx::query_scalar(
                "SELECT organization_id FROM claims.organization LIMIT 1"
            )
            .fetch_optional(&self.pool)
            .await?;

            let org = org.context("No organization found in database")?;
            info!("Using fallback organization: {}", org);
            (None, org)
        };

        // Create import batch record
        let batch_id = Uuid::new_v4();
        let started_at = chrono::Utc::now();

        info!("Creating import batch record: batch_id={}", batch_id);

        sqlx::query(
            r#"
            INSERT INTO staging.import_batch (
                batch_id,
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
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            "#
        )
        .bind(batch_id)
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
        .execute(&self.pool)
        .await
        .context("Failed to create import batch record")?;

        info!("Import batch record created successfully");

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
        let mut facility_cache: HashMap<String, (Uuid, Uuid, Option<Uuid>)> = HashMap::new();

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
        batch_id: Uuid,
        record_number: usize,
        field_name: Option<String>,
        error_type: &str,
        error_message: &str,
        raw_data: Option<String>,
    ) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO staging.import_error_log (
                error_id,
                batch_id,
                record_number,
                field_name,
                error_type,
                error_severity,
                error_message,
                raw_data
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            "#
        )
        .bind(Uuid::new_v4())
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
        facility_cache: &mut HashMap<String, (Uuid, Uuid, Option<Uuid>)>,
    ) -> Result<Uuid> {
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

            let result = facility
                .with_context(|| format!("Facility not found: {}", facility_code))?;

            // Store in cache for future lookups
            facility_cache.insert(facility_code.clone(), result);
            result
        };

        // Generate encounter ID
        let encounter_id = Uuid::new_v4();

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
                claim_status
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
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
        encounter_id: Uuid,
        parsed_row: &pro_parser_csv::parser::ParsedRow,
    ) -> Result<()> {
        let service_line_id = Uuid::new_v4();

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
        encounter_id: Uuid,
        parsed_row: &pro_parser_csv::parser::ParsedRow,
    ) -> Result<()> {
        // Get all diagnosis codes
        for (field_name, codes) in &parsed_row.diagnosis_fields {
            if field_name == "diagnosis_code" {
                for (idx, code) in codes.iter().enumerate() {
                    let diagnosis_id = Uuid::new_v4();
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
    pub batch_id: Uuid,
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
