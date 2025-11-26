//! File Processor Module
//!
//! Extracted from IngestionPipeline as part of god object refactoring.
//! Handles file-level processing for EDI and CSV files.

use crate::converters;
use crate::progress::ProgressTracker;
use crate::types::{ClaimProcessingResult, IngestionJob, ProcessingStats};
use futures::StreamExt;
use pro_common::{Error, Result};
use pro_db::{
    BusinessRuleValidator, FileValidator, PatientControlNumberValidator, ServiceLineValidator,
};
use pro_parser_csv::CsvParser;
use pro_parser_edi::parser::EdiParser;
use pro_rules::RuleEngine;
use pro_rvu::PaymentCalculator;
use sqlx::PgPool;
use tokio::sync::broadcast;
use tracing::{error, info, warn};

/// File processor for handling EDI and CSV file processing
pub struct FileProcessor {
    pool: PgPool,
    rule_engine: RuleEngine,
    payment_calculator: PaymentCalculator,
}

impl FileProcessor {
    /// Create a new file processor
    pub fn new(pool: PgPool, rule_engine: RuleEngine, payment_calculator: PaymentCalculator) -> Self {
        Self {
            pool,
            rule_engine,
            payment_calculator,
        }
    }

    /// Get reference to the database pool
    pub fn pool(&self) -> &PgPool {
        &self.pool
    }

    /// Get reference to the rule engine
    pub fn rule_engine(&self) -> &RuleEngine {
        &self.rule_engine
    }

    /// Get reference to the payment calculator
    pub fn payment_calculator(&self) -> &PaymentCalculator {
        &self.payment_calculator
    }

    /// Process an EDI 837p file
    pub async fn process_edi_file(
        &self,
        job: &IngestionJob,
        claim_processor: &crate::claim_processor::ClaimProcessor,
    ) -> Result<ProcessingStats> {
        info!("Processing EDI file: {}", job.file_path);

        // Read file content
        let content = tokio::fs::read_to_string(&job.file_path)
            .await
            .map_err(Error::Io)?;

        // Parse EDI file
        let mut edi_parser = EdiParser::new();
        let parsed_result = edi_parser.parse(&content)?;

        let mut stats = ProcessingStats::default();
        stats.total_records = parsed_result.claims.len();

        info!(
            "Parsed {} claims from EDI file",
            parsed_result.claims.len()
        );

        // Validate file hash for duplicates
        let file_validator = FileValidator::new(self.pool.clone());
        let dup_status = file_validator
            .check_file_duplicate(&job.file_hash)
            .await?;

        if matches!(dup_status, pro_db::DuplicateStatus::Duplicate { .. }) {
            warn!(
                "File {} is a duplicate (hash: {})",
                job.file_path, job.file_hash
            );
            stats.duplicate_records = stats.total_records;
            return Ok(stats);
        }

        // Sort claims by service date for FIFO processing
        let mut claims = parsed_result.claims;
        claims.sort_by(|a, b| {
            a.date_of_service_from
                .cmp(&b.date_of_service_from)
                .then_with(|| a.patient_control_number.cmp(&b.patient_control_number))
        });

        info!("Sorted {} claims by service date (FIFO order)", claims.len());

        // Initialize validators
        let pcn_validator = PatientControlNumberValidator::new(self.pool.clone());
        let service_line_validator = ServiceLineValidator::new(self.pool.clone());
        let business_validator = BusinessRuleValidator::new(self.pool.clone());

        // Group claims by service date for transaction batching
        use std::collections::BTreeMap;
        let mut claim_batches: BTreeMap<chrono::NaiveDate, Vec<pro_parser_edi::types::ParsedClaim>> =
            BTreeMap::new();

        for claim in claims {
            claim_batches
                .entry(claim.date_of_service_from)
                .or_insert_with(Vec::new)
                .push(claim);
        }

        info!(
            "Grouped {} claims into {} date batches for FIFO-compliant transaction processing",
            stats.total_records,
            claim_batches.len()
        );

        // Process each date batch sequentially
        for (service_date, batch) in claim_batches {
            info!(
                "Processing batch of {} claims for service date {}",
                batch.len(),
                service_date
            );

            // Pre-populate cache for batch rule execution
            let mut exec_cache = pro_rules::RuleExecutionCache::new();

            // Create result cache for this batch
            let result_cache = pro_rules::RuleResultCache::with_config(
                chrono::Duration::seconds(60),
                batch.len() * 10,
            );

            // Pre-allocate Vec capacities
            let estimated_lines = batch.len() * 5;
            let mut service_line_data = Vec::with_capacity(estimated_lines);
            let mut provider_ids = Vec::with_capacity(batch.len() * 2);
            let mut subscriber_ids = Vec::with_capacity(batch.len());

            for claim in &batch {
                for service_line in &claim.service_lines {
                    service_line_data.push((
                        service_line.procedure_code.as_str(),
                        service_line.service_date_from,
                        None,
                    ));
                }
                subscriber_ids.push(claim.subscriber_id.clone());
            }

            provider_ids.sort();
            provider_ids.dedup();
            subscriber_ids.sort();
            subscriber_ids.dedup();

            // Populate execution cache
            if let Err(e) = exec_cache
                .populate_for_batch(&service_line_data, &provider_ids, &subscriber_ids, &self.pool)
                .await
            {
                warn!(
                    "Failed to populate execution cache for batch {}: {}",
                    service_date, e
                );
            }

            info!(
                "Populated execution cache for batch {}: {} service lines, {} providers, {} subscribers",
                service_date,
                service_line_data.len(),
                provider_ids.len(),
                subscriber_ids.len()
            );

            // Start transaction for this date batch
            let mut tx = self.pool.begin().await.map_err(Error::Database)?;

            let mut batch_stats = (0, 0, 0, 0);

            for claim in batch {
                let claim_result = claim_processor
                    .process_claim_in_transaction_with_caches(
                        &claim,
                        &mut tx,
                        job.organization_id,
                        &exec_cache,
                        &result_cache,
                        &pcn_validator,
                        &service_line_validator,
                        &business_validator,
                    )
                    .await;

                match claim_result {
                    Ok(result) => {
                        batch_stats.0 += 1;
                        if result.success {
                            batch_stats.1 += 1;
                        } else {
                            batch_stats.2 += 1;
                        }
                        batch_stats.3 += result.warnings.len();
                        stats.total_flags += result.flag_count;
                    }
                    Err(e) => {
                        error!("Failed to process claim: {}", e);
                        batch_stats.2 += 1;
                    }
                }
            }

            // Commit entire batch
            tx.commit().await.map_err(Error::Database)?;

            stats.parsed_records += batch_stats.0;
            stats.inserted_records += batch_stats.1;
            stats.validation_errors += batch_stats.2;
            stats.validation_warnings += batch_stats.3;

            info!(
                "Committed batch for {}: {} inserted, {} errors, {} warnings",
                service_date, batch_stats.1, batch_stats.2, batch_stats.3
            );
        }

        Ok(stats)
    }

    /// Process an EDI 837p file using streaming architecture
    pub async fn process_edi_file_stream(
        &self,
        job: &IngestionJob,
        progress_broadcaster: Option<broadcast::Sender<crate::progress::ProgressEvent>>,
        claim_processor: &crate::claim_processor::ClaimProcessor,
    ) -> Result<ProcessingStats> {
        info!("Processing EDI file with streaming: {}", job.file_path);

        // Validate file hash for duplicates
        let file_validator = FileValidator::new(self.pool.clone());
        let dup_status = file_validator
            .check_file_duplicate(&job.file_hash)
            .await?;

        if matches!(dup_status, pro_db::DuplicateStatus::Duplicate { .. }) {
            warn!(
                "File {} is a duplicate (hash: {})",
                job.file_path, job.file_hash
            );
            return Ok(ProcessingStats {
                total_records: 0,
                duplicate_records: 0,
                ..Default::default()
            });
        }

        // Create parser and stream
        let mut edi_parser = EdiParser::new();
        let claim_stream = edi_parser.parse_file_stream(&job.file_path).await?;

        // Initialize validators
        let pcn_validator = PatientControlNumberValidator::new(self.pool.clone());
        let service_line_validator = ServiceLineValidator::new(self.pool.clone());
        let business_validator = BusinessRuleValidator::new(self.pool.clone());

        // Initialize caches
        let exec_cache = pro_rules::RuleExecutionCache::new();
        let result_cache = pro_rules::RuleResultCache::with_config(chrono::Duration::seconds(60), 1000);

        // Create progress tracker if broadcaster provided
        let progress_tracker = if let Some(broadcaster) = progress_broadcaster {
            Some(
                ProgressTracker::new(job.queue_id, 0, self.pool.clone(), broadcaster).await?,
            )
        } else {
            None
        };

        let mut stats = ProcessingStats::default();
        let mut claim_count = 0;

        tokio::pin!(claim_stream);

        // Process claims one-at-a-time
        while let Some(claim_result) = claim_stream.next().await {
            match claim_result {
                Ok(claim) => {
                    claim_count += 1;
                    stats.total_records = claim_count;

                    let claim_start = std::time::Instant::now();

                    // Process claim within transaction
                    let mut tx = self.pool.begin().await.map_err(Error::Database)?;

                    let result = claim_processor
                        .process_claim_in_transaction_with_caches(
                            &claim,
                            &mut tx,
                            job.organization_id,
                            &exec_cache,
                            &result_cache,
                            &pcn_validator,
                            &service_line_validator,
                            &business_validator,
                        )
                        .await;

                    match result {
                        Ok(claim_result) => {
                            tx.commit().await.map_err(Error::Database)?;

                            stats.parsed_records += 1;
                            if claim_result.success {
                                stats.inserted_records += 1;
                            } else {
                                stats.validation_errors += 1;
                            }
                            stats.validation_warnings += claim_result.warnings.len();
                            stats.total_flags += claim_result.flag_count;

                            if let Some(ref tracker) = progress_tracker {
                                let processing_time_ms = claim_start.elapsed().as_millis() as u64;
                                let _ = tracker
                                    .record_claim_processed(
                                        Some(claim.patient_control_number.clone()),
                                        claim_result.flag_count,
                                        0,
                                        processing_time_ms,
                                    )
                                    .await;
                            }
                        }
                        Err(e) => {
                            let _ = tx.rollback().await;

                            error!("Failed to process claim: {}", e);
                            stats.validation_errors += 1;

                            if let Some(ref tracker) = progress_tracker {
                                let _ = tracker
                                    .record_claim_failed(
                                        Some(claim.patient_control_number.clone()),
                                        format!("{}", e),
                                        Some(serde_json::to_value(&claim).unwrap_or_default()),
                                        Some("ProcessingError".to_string()),
                                        true,
                                    )
                                    .await;
                            }
                        }
                    }
                }
                Err(e) => {
                    error!("Failed to parse claim: {}", e);
                    stats.validation_errors += 1;

                    if let Some(ref tracker) = progress_tracker {
                        let _ = tracker
                            .record_claim_failed(
                                None,
                                format!("Parse error: {}", e),
                                None,
                                Some("ParseError".to_string()),
                                false,
                            )
                            .await;
                    }
                }
            }
        }

        if let Some(tracker) = progress_tracker {
            let _ = tracker.complete().await;
        }

        info!(
            "Completed streaming processing of {} claims: {} inserted, {} errors",
            stats.total_records, stats.inserted_records, stats.validation_errors
        );

        Ok(stats)
    }

    /// Process a CSV file
    pub async fn process_csv_file(
        &self,
        job: &IngestionJob,
        claim_processor: &crate::claim_processor::ClaimProcessor,
    ) -> Result<ProcessingStats> {
        info!("Processing CSV file: {}", job.file_path);

        // Read file content
        let content = tokio::fs::read_to_string(&job.file_path)
            .await
            .map_err(Error::Io)?;

        // Parse CSV file with auto-detection
        let mut csv_parser = CsvParser::with_auto_detection();
        let parsed_result = csv_parser.parse_reader(content.as_bytes())?;

        let mut stats = ProcessingStats::default();
        stats.total_records = parsed_result.len();

        info!("Parsed {} records from CSV file", parsed_result.len());

        // Validate file hash for duplicates
        let file_validator = FileValidator::new(self.pool.clone());
        let dup_status = file_validator
            .check_file_duplicate(&job.file_hash)
            .await?;

        if matches!(dup_status, pro_db::DuplicateStatus::Duplicate { .. }) {
            warn!(
                "File {} is a duplicate (hash: {})",
                job.file_path, job.file_hash
            );
            stats.duplicate_records = stats.total_records;
            return Ok(stats);
        }

        // Initialize validators
        let pcn_validator = PatientControlNumberValidator::new(self.pool.clone());
        let service_line_validator = ServiceLineValidator::new(self.pool.clone());
        let business_validator = BusinessRuleValidator::new(self.pool.clone());

        // Sort CSV rows by service date for FIFO processing
        let mut csv_rows = parsed_result;
        csv_rows.sort_by(|a, b| {
            let date_a = a
                .encounter_fields
                .get("date_of_service_from")
                .and_then(|s| chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d").ok());
            let date_b = b
                .encounter_fields
                .get("date_of_service_from")
                .and_then(|s| chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d").ok());

            match (date_a, date_b) {
                (Some(date_a_val), Some(date_b_val)) => date_a_val.cmp(&date_b_val).then_with(|| {
                    a.encounter_fields
                        .get("patient_control_number")
                        .cmp(&b.encounter_fields.get("patient_control_number"))
                }),
                (Some(_), None) => std::cmp::Ordering::Less,
                (None, Some(_)) => std::cmp::Ordering::Greater,
                (None, None) => a.row_number.cmp(&b.row_number),
            }
        });

        info!(
            "Sorted {} CSV rows by service date (FIFO order)",
            csv_rows.len()
        );

        // Process each CSV record
        for csv_row in csv_rows {
            if !csv_row.errors.is_empty() {
                warn!(
                    "CSV row {} has {} errors: {:?}",
                    csv_row.row_number,
                    csv_row.errors.len(),
                    csv_row.errors
                );
                stats.validation_errors += csv_row.errors.len();
                continue;
            }

            // Convert CSV row to ParsedClaim
            let claim = match converters::convert_csv_to_claim(&csv_row) {
                Ok(c) => c,
                Err(e) => {
                    error!(
                        "Failed to convert CSV row {} to claim: {}",
                        csv_row.row_number, e
                    );
                    stats.validation_errors += 1;
                    continue;
                }
            };

            // Process the claim using existing EDI processing logic
            let claim_result = claim_processor
                .process_claim(
                    &claim,
                    job.organization_id,
                    &pcn_validator,
                    &service_line_validator,
                    &business_validator,
                )
                .await;

            match claim_result {
                Ok(result) => {
                    stats.parsed_records += 1;
                    if result.success {
                        stats.inserted_records += 1;
                    } else {
                        stats.validation_errors += 1;
                    }
                    stats.validation_warnings += result.warnings.len() + csv_row.warnings.len();
                    stats.total_flags += result.flag_count;
                }
                Err(e) => {
                    error!(
                        "Failed to process claim from CSV row {}: {}",
                        csv_row.row_number, e
                    );
                    stats.validation_errors += 1;
                }
            }
        }

        Ok(stats)
    }
}
