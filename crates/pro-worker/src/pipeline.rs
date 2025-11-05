// File processing pipeline

use crate::types::{ClaimProcessingResult, FileFormat, IngestionJob, ProcessingStats, ProcessingStatus};
use crate::progress::ProgressTracker; // PHASE 5
use chrono::Utc;
use pro_common::{Error, Result};
use pro_db::{
    models::{Encounter, EncounterDiagnosis, ServiceLine},
    repositories::{EncounterRepository, ServiceLineRepository},
    BusinessRuleValidator, FileValidator, PatientControlNumberValidator,
    ServiceLineValidator,
};
use pro_parser_csv::CsvParser;
use pro_parser_edi::parser::EdiParser;
use pro_rules::RuleEngine;
use pro_rvu::PaymentCalculator;
use sqlx::PgPool;
use tracing::{info, warn, error};

use futures::StreamExt; // PHASE 5
use tokio::sync::broadcast; // PHASE 5

/// Main processing pipeline for ingestion
pub struct IngestionPipeline {
    pool: PgPool,
    rule_engine: RuleEngine,
    payment_calculator: PaymentCalculator,
}

impl IngestionPipeline {
    /// Create a new ingestion pipeline
    pub fn new(pool: PgPool) -> Self {
        // Initialize rule engine with database pool
        let mut rule_engine = RuleEngine::new(pool.clone());

        // Add default rules directly
        rule_engine.add_rule(pro_rules::rules::DuplicateServiceRule);
        rule_engine.add_rule(pro_rules::rules::UnitsExceedMaximumRule::default());
        rule_engine.add_rule(pro_rules::rules::MissingRequiredModifierRule::default());
        rule_engine.add_rule(pro_rules::rules::ConflictingModifiersRule);
        rule_engine.add_rule(pro_rules::rules::UnspecifiedDiagnosisRule);
        rule_engine.add_rule(pro_rules::rules::MissingDiagnosisSpecificityRule);

        // Initialize payment calculator with sample data
        // In production, this would load from database
        let payment_calculator = PaymentCalculator::with_sample_data();

        Self {
            pool,
            rule_engine,
            payment_calculator,
        }
    }

    /// Process a file ingestion job
    pub async fn process_job(&self, mut job: IngestionJob) -> Result<(IngestionJob, ProcessingStats)> {
        info!(
            "Starting processing job {} for file: {}",
            job.job_id, job.file_path
        );

        job.start();

        // Update job status in database
        self.update_job_status(&job).await?;

        // Process file based on format
        let result = match job.file_format {
            FileFormat::Edi837p => {
                // PHASE 5: Check if streaming is enabled
                let enable_streaming = std::env::var("ENABLE_STREAMING")
                    .unwrap_or_else(|_| "false".to_string())
                    .parse::<bool>()
                    .unwrap_or(false);

                if enable_streaming {
                    // TODO: Get broadcaster from application state
                    self.process_edi_file_stream(&job, None).await
                } else {
                    self.process_edi_file(&job).await
                }
            },
            FileFormat::Csv => self.process_csv_file(&job).await,
        };

        // Update final status
        let (final_status, stats) = match result {
            Ok(stats) => {
                if stats.validation_errors > 0 || stats.duplicate_records > 0 {
                    (ProcessingStatus::Partial, stats)
                } else {
                    (ProcessingStatus::Completed, stats)
                }
            }
            Err(e) => {
                error!("Job {} failed: {}", job.job_id, e);
                job.complete(ProcessingStatus::Failed);
                self.update_job_status(&job).await?;
                return Err(e);
            }
        };

        job.complete(final_status);
        self.update_job_status(&job).await?;

        info!(
            "Completed job {} in {} ms. Stats: {:?}",
            job.job_id,
            job.duration_ms().unwrap_or(0),
            stats
        );

        Ok((job, stats))
    }

    /// Process an EDI 837p file
    async fn process_edi_file(&self, job: &IngestionJob) -> Result<ProcessingStats> {
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
            warn!("File {} is a duplicate (hash: {})", job.file_path, job.file_hash);
            stats.duplicate_records = stats.total_records;
            return Ok(stats);
        }

        // *** CRITICAL: Sort claims by service date for FIFO processing ***
        let mut claims = parsed_result.claims;
        claims.sort_by(|a, b| {
            // Primary sort: date_of_service_from (oldest first)
            a.date_of_service_from.cmp(&b.date_of_service_from)
                // Secondary sort: patient_control_number (for stable ordering)
                .then_with(|| a.patient_control_number.cmp(&b.patient_control_number))
        });

        info!(
            "Sorted {} claims by service date (FIFO order)",
            claims.len()
        );

        // Initialize validators
        let pcn_validator = PatientControlNumberValidator::new(self.pool.clone());
        let service_line_validator = ServiceLineValidator::new(self.pool.clone());
        let business_validator = BusinessRuleValidator::new(self.pool.clone());

        // *** PHASE 2: Group claims by service date for transaction batching ***
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

        // Process each date batch sequentially (maintains FIFO)
        for (service_date, batch) in claim_batches {
            info!(
                "Processing batch of {} claims for service date {}",
                batch.len(),
                service_date
            );

            // *** PHASE 3: Pre-populate cache for batch rule execution ***
            let mut exec_cache = pro_rules::RuleExecutionCache::new();

            // *** PHASE 5: Create result cache for this batch ***
            // TTL: 60 seconds (conservative), Size: batch size * 10 (estimate 10 service lines per claim)
            let result_cache = pro_rules::RuleResultCache::with_config(
                chrono::Duration::seconds(60),
                batch.len() * 10,
            );

            // PHASE 4: Pre-allocate Vec capacities to avoid reallocations
            // Estimate: average 5 service lines per claim
            let estimated_lines = batch.len() * 5;
            let mut service_line_data = Vec::with_capacity(estimated_lines);
            let mut provider_ids = Vec::with_capacity(batch.len() * 2); // Rendering + referring providers
            let mut subscriber_ids = Vec::with_capacity(batch.len());

            for claim in &batch {
                // Collect service line information
                for service_line in &claim.service_lines {
                    service_line_data.push((
                        service_line.procedure_code.as_str(),
                        service_line.service_date_from,
                        None, // TODO: Provider ID lookup by NPI
                    ));

                    // TODO: Collect provider IDs from database lookup by NPI
                    // Currently disabled - needs provider_id lookup
                }

                // Collect subscriber IDs
                subscriber_ids.push(claim.subscriber_id.clone());
            }

            // Deduplicate provider IDs and subscriber IDs
            provider_ids.sort();
            provider_ids.dedup();
            subscriber_ids.sort();
            subscriber_ids.dedup();

            // Populate execution cache (outside transaction, uses pool directly)
            if let Err(e) = exec_cache.populate_for_batch(
                &service_line_data,
                &provider_ids,
                &subscriber_ids,
                &self.pool,
            ).await {
                warn!("Failed to populate execution cache for batch {}: {}", service_date, e);
                // Continue anyway - rules will fall back to direct DB queries
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

            let mut batch_stats = (0, 0, 0, 0); // (parsed, inserted, errors, warnings)

            for claim in batch {
                let claim_result = self
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
                        batch_stats.0 += 1; // parsed
                        if result.success {
                            batch_stats.1 += 1; // inserted
                        } else {
                            batch_stats.2 += 1; // errors
                        }
                        batch_stats.3 += result.warnings.len(); // warnings
                        stats.total_flags += result.flag_count;
                    }
                    Err(e) => {
                        error!("Failed to process claim: {}", e);
                        batch_stats.2 += 1; // errors
                        // Continue processing other claims in batch
                    }
                }
            }

            // Commit entire batch
            tx.commit().await.map_err(Error::Database)?;

            // Update overall stats
            stats.parsed_records += batch_stats.0;
            stats.inserted_records += batch_stats.1;
            stats.validation_errors += batch_stats.2;
            stats.validation_warnings += batch_stats.3;

            info!(
                "Committed batch for {}: {} inserted, {} errors, {} warnings",
                service_date,
                batch_stats.1,
                batch_stats.2,
                batch_stats.3
            );
        }

        Ok(stats)
    }

    /// PHASE 5: Process an EDI 837p file using streaming architecture
    ///
    /// This method processes claims one-at-a-time as they're parsed,
    /// providing real-time progress updates and lower memory usage.
    async fn process_edi_file_stream(
        &self,
        job: &IngestionJob,
        progress_broadcaster: Option<broadcast::Sender<crate::progress::ProgressEvent>>,
    ) -> Result<ProcessingStats> {
        info!("Processing EDI file with streaming: {}", job.file_path);

        // Validate file hash for duplicates
        let file_validator = FileValidator::new(self.pool.clone());
        let dup_status = file_validator
            .check_file_duplicate(&job.file_hash)
            .await?;

        if matches!(dup_status, pro_db::DuplicateStatus::Duplicate { .. }) {
            warn!("File {} is a duplicate (hash: {})", job.file_path, job.file_hash);
            return Ok(ProcessingStats {
                total_records: 0,
                duplicate_records: 0,
                ..Default::default()
            });
        }

        // Create parser and stream
        let mut edi_parser = EdiParser::new();
        let claim_stream = edi_parser
            .parse_file_stream(&job.file_path)
            .await?;

        // Initialize validators
        let pcn_validator = PatientControlNumberValidator::new(self.pool.clone());
        let service_line_validator = ServiceLineValidator::new(self.pool.clone());
        let business_validator = BusinessRuleValidator::new(self.pool.clone());

        // Initialize caches
        let exec_cache = pro_rules::RuleExecutionCache::new();
        let result_cache = pro_rules::RuleResultCache::with_config(
            chrono::Duration::seconds(60),
            1000, // Smaller cache since we're streaming
        );

        // Create progress tracker if broadcaster provided
        let progress_tracker = if let Some(broadcaster) = progress_broadcaster {
            Some(ProgressTracker::new(
                job.queue_id,
                0, // Total will be updated as we parse
                self.pool.clone(),
                broadcaster,
            ).await?)
        } else {
            None
        };

        let mut stats = ProcessingStats::default();
        let mut claim_count = 0;

        // Pin the stream for iteration
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

                    let result = self.process_claim_in_transaction_with_caches(
                        &claim,
                        &mut tx,
                        job.organization_id,
                        &exec_cache,
                        &result_cache,
                        &pcn_validator,
                        &service_line_validator,
                        &business_validator,
                    ).await;

                    match result {
                        Ok(claim_result) => {
                            // Commit transaction
                            tx.commit().await.map_err(Error::Database)?;

                            stats.parsed_records += 1;
                            if claim_result.success {
                                stats.inserted_records += 1;
                            } else {
                                stats.validation_errors += 1;
                            }
                            stats.validation_warnings += claim_result.warnings.len();
                            stats.total_flags += claim_result.flag_count;

                            // Update progress tracker
                            if let Some(ref tracker) = progress_tracker {
                                let processing_time_ms = claim_start.elapsed().as_millis() as u64;
                                let _ = tracker.record_claim_processed(
                                    Some(claim.patient_control_number.clone()),
                                    claim_result.flag_count,
                                    0, // TODO: Track critical flags
                                    processing_time_ms,
                                ).await;
                            }
                        }
                        Err(e) => {
                            // Rollback transaction
                            let _ = tx.rollback().await;

                            error!("Failed to process claim: {}", e);
                            stats.validation_errors += 1;

                            // Record failed claim
                            if let Some(ref tracker) = progress_tracker {
                                let _ = tracker.record_claim_failed(
                                    Some(claim.patient_control_number.clone()),
                                    format!("{}", e),
                                    Some(serde_json::to_value(&claim).unwrap_or_default()),
                                    Some("ProcessingError".to_string()),
                                    true, // can_retry
                                ).await;
                            }
                        }
                    }
                }
                Err(e) => {
                    error!("Failed to parse claim: {}", e);
                    stats.validation_errors += 1;

                    // Record parse error
                    if let Some(ref tracker) = progress_tracker {
                        let _ = tracker.record_claim_failed(
                            None,
                            format!("Parse error: {}", e),
                            None,
                            Some("ParseError".to_string()),
                            false, // Cannot retry parse errors
                        ).await;
                    }
                }
            }
        }

        // Complete progress tracking
        if let Some(tracker) = progress_tracker {
            let _ = tracker.complete().await;
        }

        info!(
            "Completed streaming processing of {} claims: {} inserted, {} errors",
            stats.total_records,
            stats.inserted_records,
            stats.validation_errors
        );

        Ok(stats)
    }

    /// Process a CSV file
    async fn process_csv_file(&self, job: &IngestionJob) -> Result<ProcessingStats> {
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

        info!(
            "Parsed {} records from CSV file",
            parsed_result.len()
        );

        // Validate file hash for duplicates
        let file_validator = FileValidator::new(self.pool.clone());
        let dup_status = file_validator
            .check_file_duplicate(&job.file_hash)
            .await?;

        if matches!(dup_status, pro_db::DuplicateStatus::Duplicate { .. }) {
            warn!("File {} is a duplicate (hash: {})", job.file_path, job.file_hash);
            stats.duplicate_records = stats.total_records;
            return Ok(stats);
        }

        // Initialize validators
        let pcn_validator = PatientControlNumberValidator::new(self.pool.clone());
        let service_line_validator = ServiceLineValidator::new(self.pool.clone());
        let business_validator = BusinessRuleValidator::new(self.pool.clone());

        // *** CRITICAL: Sort CSV rows by service date for FIFO processing ***
        let mut csv_rows = parsed_result;
        csv_rows.sort_by(|a, b| {
            // Try to extract service dates from both rows
            let date_a = a.encounter_fields.get("date_of_service_from")
                .and_then(|s| chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d").ok());
            let date_b = b.encounter_fields.get("date_of_service_from")
                .and_then(|s| chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d").ok());

            // Sort by date (oldest first), then by PCN for stability
            match (date_a, date_b) {
                (Some(date_a_val), Some(date_b_val)) => date_a_val.cmp(&date_b_val)
                    .then_with(|| {
                        a.encounter_fields.get("patient_control_number")
                            .cmp(&b.encounter_fields.get("patient_control_number"))
                    }),
                (Some(_), None) => std::cmp::Ordering::Less,  // Records with dates come first
                (None, Some(_)) => std::cmp::Ordering::Greater,
                (None, None) => a.row_number.cmp(&b.row_number), // Fall back to row order
            }
        });

        info!(
            "Sorted {} CSV rows by service date (FIFO order)",
            csv_rows.len()
        );

        // Process each CSV record
        for csv_row in csv_rows {
            // Check for parsing errors in CSV row
            if !csv_row.errors.is_empty() {
                warn!("CSV row {} has {} errors: {:?}", csv_row.row_number, csv_row.errors.len(), csv_row.errors);
                stats.validation_errors += csv_row.errors.len();
                continue;
            }

            // Convert CSV row to ParsedClaim
            let claim = match self.convert_csv_to_claim(&csv_row, job.organization_id) {
                Ok(c) => c,
                Err(e) => {
                    error!("Failed to convert CSV row {} to claim: {}", csv_row.row_number, e);
                    stats.validation_errors += 1;
                    continue;
                }
            };

            // Process the claim using existing EDI processing logic
            let claim_result = self
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
                    error!("Failed to process claim from CSV row {}: {}", csv_row.row_number, e);
                    stats.validation_errors += 1;
                }
            }
        }

        Ok(stats)
    }

    /// Convert CSV ParsedRow to ParsedClaim format
    fn convert_csv_to_claim(
        &self,
        csv_row: &pro_parser_csv::parser::ParsedRow,
        _organization_id: i64,
    ) -> Result<pro_parser_edi::types::ParsedClaim> {
        use pro_parser_edi::types::ParsedClaim;
        use chrono::NaiveDate;
        use rust_decimal::Decimal;
        use std::str::FromStr;

        // Helper function to get required field from encounter_fields
        let get_required = |field: &str| -> Result<String> {
            csv_row.encounter_fields
                .get(field)
                .cloned()
                .ok_or_else(|| Error::Parse(format!("Required field '{}' not found in CSV", field)))
        };

        // Helper function to get optional field
        let get_optional = |field: &str| -> Option<String> {
            csv_row.encounter_fields.get(field).cloned()
        };

        // Helper function to parse date
        let parse_date = |field: &str| -> Result<Option<NaiveDate>> {
            if let Some(date_str) = csv_row.encounter_fields.get(field) {
                NaiveDate::parse_from_str(date_str, "%Y-%m-%d")
                    .map(Some)
                    .map_err(|_| Error::Parse(format!("Invalid date format for field '{}'", field)))
            } else {
                Ok(None)
            }
        };

        // Helper function to parse decimal
        let parse_decimal = |field: &str| -> Result<Decimal> {
            let value_str = get_required(field)?;
            Decimal::from_str(&value_str)
                .map_err(|_| Error::Parse(format!("Invalid decimal format for field '{}'", field)))
        };

        // Build ParsedClaim structure
        let claim = ParsedClaim {
            // Temporary ID for processing
            temp_id: 0,

            // Subscriber hierarchical level (defaults for CSV)
            subscriber_hl_number: "1".to_string(),
            subscriber_relationship_code: "18".to_string(), // Self

            // Subscriber information
            subscriber_entity_identifier: "IL".to_string(), // Insured/Subscriber
            subscriber_entity_type: "1".to_string(), // Person
            subscriber_last_name: get_required("subscriber_last_name")?,
            subscriber_first_name: get_required("subscriber_first_name")?,
            subscriber_middle_name: get_optional("subscriber_middle_name"),
            subscriber_name_suffix: get_optional("subscriber_name_suffix"),
            subscriber_id_code_qualifier: "MI".to_string(), // Member Identification Number
            subscriber_id: get_required("subscriber_id")?,

            // Subscriber demographic
            subscriber_date_of_birth: parse_date("subscriber_date_of_birth")?,
            subscriber_gender: get_optional("subscriber_gender"),

            // Subscriber address
            subscriber_address_line1: get_optional("subscriber_address_line1"),
            subscriber_address_line2: get_optional("subscriber_address_line2"),
            subscriber_city: get_optional("subscriber_city"),
            subscriber_state: get_optional("subscriber_state"),
            subscriber_postal_code: get_optional("subscriber_postal_code"),
            subscriber_country: get_optional("subscriber_country"),

            // Medical Record Number
            medical_record_number: get_optional("medical_record_number"),

            // Payer information
            payer_entity_identifier: "PR".to_string(), // Payer
            payer_entity_type: "2".to_string(), // Non-Person Entity
            payer_name: get_required("payer_name")?,
            payer_id_qualifier: "PI".to_string(), // Payer Identification
            payer_id: get_required("payer_id")?,
            payer_address_line1: get_optional("payer_address_line1"),
            payer_address_line2: get_optional("payer_address_line2"),
            payer_city: get_optional("payer_city"),
            payer_state: get_optional("payer_state"),
            payer_postal_code: get_optional("payer_postal_code"),

            // Claim information
            patient_control_number: get_required("patient_control_number")?,
            total_claim_charge_amount: parse_decimal("total_claim_charge_amount")?,
            place_of_service_code: get_optional("place_of_service_code"),
            claim_frequency_code: Some("1".to_string()), // Original claim
            provider_signature_indicator: Some("Y".to_string()),
            assignment_indicator: Some("Y".to_string()),
            benefits_assignment_indicator: Some("Y".to_string()),
            release_of_information_code: Some("Y".to_string()),

            // Dates
            date_of_service_from: parse_date("date_of_service_from")?
                .ok_or_else(|| Error::Parse("Required field 'date_of_service_from' not found".to_string()))?,
            date_of_service_to: parse_date("date_of_service_to")?,

            // Diagnosis codes - extract from diagnosis_fields
            diagnoses: self.extract_diagnoses_from_csv(csv_row)?,

            // Service lines - extract from service_line_fields
            service_lines: self.extract_service_lines_from_csv(csv_row)?,

            // Provider information (NPIs)
            rendering_provider_npi: get_optional("rendering_provider_npi"),
            referring_provider_npi: get_optional("referring_provider_npi"),
            supervising_provider_npi: get_optional("supervising_provider_npi"),
            service_facility_npi: get_optional("service_facility_npi"),

            // Fields not typically in CSV (use defaults)
            onset_of_illness_date: None,
            initial_treatment_date: None,
            last_seen_date: None,
            acute_manifestation_date: None,
            accident_date: None,
            last_menstrual_period_date: None,
            last_xray_date: None,
            disability_from_date: None,
            disability_to_date: None,
            last_worked_date: None,
            authorized_return_to_work_date: None,
            admission_date: None,
            discharge_date: None,
            delay_reason_code: None,
            special_program_code: None,
            patient_amount_paid: None,
            service_authorization_code: None,
            claim_note: None,
            referring_provider_qualifier: None,
            referring_provider_last_name: None,
            referring_provider_first_name: None,
            rendering_provider_qualifier: None,
            rendering_provider_last_name: None,
            rendering_provider_first_name: None,
            rendering_provider_taxonomy: None,
            service_facility_qualifier: None,
            service_facility_name: None,
            service_facility_address_line1: None,
            service_facility_address_line2: None,
            service_facility_city: None,
            service_facility_state: None,
            service_facility_postal_code: None,
            supervising_provider_qualifier: None,
            supervising_provider_last_name: None,
            supervising_provider_first_name: None,
            other_payer_paid_amount: None,
            other_payer_id: None,
            other_payer_name: None,
            other_payer_claim_number: None,
            patient_signature_code: None,
            related_causes_code_1: None,
            related_causes_code_2: None,
            related_causes_code_3: None,
            auto_accident_state: None,
            auto_accident_country: None,
        };

        Ok(claim)
    }

    /// Extract diagnosis codes from CSV row
    fn extract_diagnoses_from_csv(
        &self,
        csv_row: &pro_parser_csv::parser::ParsedRow,
    ) -> Result<Vec<pro_parser_edi::types::DiagnosisCode>> {
        use pro_parser_edi::types::DiagnosisCode;

        let mut diagnoses = Vec::new();
        let mut sequence_number: i16 = 1;

        // Look for diagnosis fields (diagnosis_code_1, diagnosis_code_2, etc.)
        for i in 1..=12 {
            let field_name = format!("diagnosis_code_{}", i);
            if let Some(codes) = csv_row.diagnosis_fields.get(&field_name) {
                for code in codes {
                    if !code.trim().is_empty() {
                        diagnoses.push(DiagnosisCode {
                            sequence_number,
                            diagnosis_code_qualifier: "ABK".to_string(), // ICD-10-CM
                            diagnosis_code: code.clone(),
                            is_principal: sequence_number == 1, // First diagnosis is principal
                        });
                        sequence_number += 1;
                    }
                }
            }
        }

        // If no individual diagnosis codes, check for a single diagnosis_code field
        if diagnoses.is_empty() {
            if let Some(codes) = csv_row.diagnosis_fields.get("diagnosis_code") {
                for code in codes {
                    if !code.trim().is_empty() {
                        diagnoses.push(DiagnosisCode {
                            sequence_number,
                            diagnosis_code_qualifier: "ABK".to_string(),
                            diagnosis_code: code.clone(),
                            is_principal: sequence_number == 1,
                        });
                        sequence_number += 1;
                    }
                }
            }
        }

        if diagnoses.is_empty() {
            return Err(Error::Parse("No diagnosis codes found in CSV row".to_string()));
        }

        Ok(diagnoses)
    }

    /// Extract service lines from CSV row
    fn extract_service_lines_from_csv(
        &self,
        csv_row: &pro_parser_csv::parser::ParsedRow,
    ) -> Result<Vec<pro_parser_edi::types::ServiceLine>> {
        use pro_parser_edi::types::ServiceLine;
        use chrono::NaiveDate;
        use rust_decimal::Decimal;
        use std::str::FromStr;

        // CSV typically has one service line per row
        // Service line fields are in service_line_fields HashMap

        let get_required = |field: &str| -> Result<String> {
            csv_row.service_line_fields
                .get(field)
                .cloned()
                .ok_or_else(|| Error::Parse(format!("Required service line field '{}' not found", field)))
        };

        let get_optional = |field: &str| -> Option<String> {
            csv_row.service_line_fields.get(field).cloned()
        };

        let parse_decimal = |field: &str| -> Result<Decimal> {
            let value_str = get_required(field)?;
            Decimal::from_str(&value_str)
                .map_err(|_| Error::Parse(format!("Invalid decimal for service line field '{}'", field)))
        };

        let parse_date = |field: &str| -> Result<Option<NaiveDate>> {
            if let Some(date_str) = csv_row.service_line_fields.get(field) {
                NaiveDate::parse_from_str(date_str, "%Y-%m-%d")
                    .map(Some)
                    .map_err(|_| Error::Parse(format!("Invalid date for service line field '{}'", field)))
            } else {
                Ok(None)
            }
        };

        let service_line = ServiceLine {
            line_number: 1, // CSV typically has one service line per row

            // Service information
            product_service_id_qualifier: "HC".to_string(), // HCPCS
            procedure_code: get_required("procedure_code")?,
            procedure_modifier_1: get_optional("procedure_modifier_1"),
            procedure_modifier_2: get_optional("procedure_modifier_2"),
            procedure_modifier_3: get_optional("procedure_modifier_3"),
            procedure_modifier_4: get_optional("procedure_modifier_4"),
            line_item_charge_amount: parse_decimal("line_item_charge_amount")?,
            unit_basis_measurement_code: "UN".to_string(), // Units
            service_unit_count: parse_decimal("service_unit_count")
                .unwrap_or(Decimal::from(1)), // Default to 1 unit

            // Dates
            service_date_from: parse_date("service_date_from")?
                .or_else(|| csv_row.encounter_fields.get("date_of_service_from")
                    .and_then(|s| NaiveDate::parse_from_str(s, "%Y-%m-%d").ok()))
                .ok_or_else(|| Error::Parse("Service date required".to_string()))?,
            service_date_to: parse_date("service_date_to")?,

            // Place of service
            place_of_service_code: get_optional("place_of_service_code"),

            // Indicators (not typically in CSV)
            emergency_indicator: None,
            epsdt_indicator: None,
            family_planning_indicator: None,

            // Diagnosis pointers (1-based indices)
            diagnosis_code_pointer_1: get_optional("diagnosis_code_pointer_1")
                .and_then(|s| s.parse::<i16>().ok()),
            diagnosis_code_pointer_2: get_optional("diagnosis_code_pointer_2")
                .and_then(|s| s.parse::<i16>().ok()),
            diagnosis_code_pointer_3: get_optional("diagnosis_code_pointer_3")
                .and_then(|s| s.parse::<i16>().ok()),
            diagnosis_code_pointer_4: get_optional("diagnosis_code_pointer_4")
                .and_then(|s| s.parse::<i16>().ok()),

            // Provider NPIs at line level
            rendering_provider_npi: get_optional("rendering_provider_npi"),
            rendering_provider_last_name: None,
            rendering_provider_first_name: None,
            supervising_provider_npi: get_optional("supervising_provider_npi"),
            ordering_provider_npi: get_optional("ordering_provider_npi"),
            ordering_provider_last_name: None,
            ordering_provider_first_name: None,
            referring_provider_npi: get_optional("referring_provider_npi"),

            // NDC information
            ndc_code: get_optional("ndc_code"),
            ndc_unit_count: get_optional("ndc_unit_count")
                .and_then(|s| Decimal::from_str(&s).ok()),
            ndc_measurement_unit: get_optional("ndc_measurement_unit"),

            // Prior authorization
            prior_authorization_number: get_optional("prior_authorization_number"),

            // Referral number
            referral_number: get_optional("referral_number"),

            // Line note
            line_note: get_optional("line_note"),

            // Revenue code
            revenue_code: get_optional("revenue_code"),

            // Other payer line adjudication
            other_payer_line_paid_amount: None,
        };

        Ok(vec![service_line])
    }

    /// Process a single claim
    async fn process_claim(
        &self,
        claim: &pro_parser_edi::types::ParsedClaim,
        organization_id: i64,
        _pcn_validator: &PatientControlNumberValidator,
        _service_line_validator: &ServiceLineValidator,
        _business_validator: &BusinessRuleValidator,
    ) -> Result<ClaimProcessingResult> {
        let patient_control_number = claim.patient_control_number.clone();

        let mut result = ClaimProcessingResult {
            patient_control_number: patient_control_number.clone(),
            encounter_id: None,
            success: false,
            errors: Vec::new(),
            warnings: Vec::new(),
            service_line_count: claim.service_lines.len(),
            flag_count: 0,
        };

        // Convert claim to encounter model
        let encounter = match self.convert_claim_to_encounter(claim, organization_id) {
            Ok(enc) => enc,
            Err(e) => {
                error!("Failed to convert claim to encounter: {}", e);
                result.errors.push(format!("Conversion error: {}", e));
                return Ok(result);
            }
        };

        // Create repositories
        let encounter_repo = EncounterRepository::new(&self.pool);
        let service_line_repo = ServiceLineRepository::new(&self.pool);

        // Insert encounter into database
        let encounter_id = match encounter_repo.create(&encounter).await {
            Ok(id) => id,
            Err(e) => {
                error!("Failed to insert encounter: {}", e);
                result.errors.push(format!("Database error: {}", e));
                return Ok(result);
            }
        };

        result.encounter_id = Some(encounter_id);

        // *** PERFORMANCE: Batch insert diagnosis codes (10x faster) ***
        let diagnoses: Vec<EncounterDiagnosis> = claim.diagnoses.iter().enumerate()
            .map(|(idx, parsed_dx)| EncounterDiagnosis {
                diagnosis_id: 0,
                encounter_id,
                sequence_number: (idx + 1) as i16,
                diagnosis_code_qualifier: Some(parsed_dx.diagnosis_code_qualifier.clone()),
                diagnosis_code: parsed_dx.diagnosis_code.clone(),
                diagnosis_description: None, // Would be looked up from ICD-10 reference
                is_principal: parsed_dx.is_principal,
                is_admitting: false, // Not provided in Loop 2300 HI segment
                is_external_cause: false, // Would need to analyze code prefix
                is_patient_reason: false, // Would need additional EDI segments
                present_on_admission_indicator: None, // Not provided in professional 837p
                hcc_indicator: false, // Would be computed with HCC engine
                hcc_category: None, // Would be computed with HCC engine
                created_at: Utc::now(),
            })
            .collect();

        match encounter_repo.create_diagnoses_batch(&diagnoses).await {
            Ok(dx_ids) => {
                // Aggregate logging: only log summary (not per-diagnosis)
                if cfg!(debug_assertions) {
                    info!("Inserted {} diagnoses for encounter {}", dx_ids.len(), encounter_id);
                }
            }
            Err(e) => {
                warn!("Failed to batch insert diagnoses for encounter {}: {}", encounter_id, e);
                result.warnings.push(format!("Diagnosis batch insert error: {}", e));
            }
        }

        // *** PERFORMANCE: Batch insert service lines (10x faster) ***
        let service_lines: Vec<ServiceLine> = claim.service_lines.iter().enumerate()
            .map(|(idx, parsed_line)| self.convert_service_line(parsed_line, encounter_id, (idx + 1) as i16))
            .collect();

        let service_line_ids = match service_line_repo.create_batch(&service_lines).await {
            Ok(ids) => {
                // Aggregate logging: only log summary (not per-service line)
                if cfg!(debug_assertions) {
                    info!("Inserted {} service lines for encounter {}", ids.len(), encounter_id);
                }
                ids
            }
            Err(e) => {
                error!("Failed to batch insert service lines for encounter {}: {}", encounter_id, e);
                result.warnings.push(format!("Service line batch insert error: {}", e));
                Vec::new()
            }
        };

        // *** PERFORMANCE: Reduced logging - only log in debug mode or on errors ***
        let mut total_flags = 0;

        // Build rule execution context for encounter-level rules
        let mut encounter_ctx = pro_rules::RuleExecutionContext::new(organization_id);
        encounter_ctx.encounter_id = Some(encounter_id);
        encounter_ctx.facility_id = Some(encounter.facility_id);
        encounter_ctx.total_claim_charge_amount = Some(claim.total_claim_charge_amount);
        encounter_ctx.place_of_service_code = claim.place_of_service_code.clone();
        encounter_ctx.date_of_service_from = Some(claim.date_of_service_from);
        encounter_ctx.date_of_service_to = claim.date_of_service_to;

        // Add diagnosis codes
        encounter_ctx.diagnosis_codes = claim.diagnoses.iter()
            .map(|d| d.diagnosis_code.clone())
            .collect();

        // PHASE 6: Execute encounter-level rules (no cache available in this method)
        match self.rule_engine.execute_all(&encounter_ctx).await {
            Ok(rule_results) => {
                if !rule_results.is_empty() {
                    total_flags += rule_results.len();

                    // Persist flags to database
                    match self.rule_engine.persist_flags(rule_results).await {
                        Ok(_flag_ids) => {
                            // Only log in debug mode
                            if cfg!(debug_assertions) {
                                info!("Persisted {} encounter-level flags", total_flags);
                            }
                        }
                        Err(e) => {
                            warn!("Failed to persist encounter flags: {}", e);
                            result.warnings.push(format!("Failed to persist encounter flags: {}", e));
                        }
                    }
                }
            }
            Err(e) => {
                warn!("Error running encounter-level rules: {}", e);
                result.warnings.push(format!("Encounter rules error: {}", e));
            }
        }

        // PHASE 6: Run service line-level rules (no cache in this legacy method)
        for (idx, (service_line_id, parsed_line)) in service_line_ids.iter().zip(claim.service_lines.iter()).enumerate() {
            let mut line_ctx = pro_rules::RuleExecutionContext::new(organization_id);
            line_ctx.encounter_id = Some(encounter_id);
            line_ctx.service_line_id = Some(*service_line_id);
            line_ctx.facility_id = Some(encounter.facility_id);
            line_ctx.procedure_code = Some(parsed_line.procedure_code.clone());
            line_ctx.service_unit_count = Some(parsed_line.service_unit_count);
            line_ctx.line_item_charge_amount = Some(parsed_line.line_item_charge_amount);
            line_ctx.date_of_service = Some(parsed_line.service_date_from);
            line_ctx.place_of_service_code = parsed_line.place_of_service_code.clone();

            // Add modifiers
            let mut modifiers = Vec::new();
            if let Some(ref m) = parsed_line.procedure_modifier_1 { modifiers.push(m.clone()); }
            if let Some(ref m) = parsed_line.procedure_modifier_2 { modifiers.push(m.clone()); }
            if let Some(ref m) = parsed_line.procedure_modifier_3 { modifiers.push(m.clone()); }
            if let Some(ref m) = parsed_line.procedure_modifier_4 { modifiers.push(m.clone()); }
            line_ctx.procedure_modifiers = modifiers;

            // Add diagnosis codes from encounter
            line_ctx.diagnosis_codes = claim.diagnoses.iter()
                .map(|d| d.diagnosis_code.clone())
                .collect();

            // Execute service line rules
            match self.rule_engine.execute_all(&line_ctx).await {
                Ok(rule_results) => {
                    if !rule_results.is_empty() {
                        total_flags += rule_results.len();

                        // Persist flags
                        match self.rule_engine.persist_flags(rule_results).await {
                            Ok(_flag_ids) => {
                                // Only log in debug mode
                            }
                            Err(e) => {
                                warn!("Failed to persist service line flags: {}", e);
                                result.warnings.push(format!("Failed to persist line {} flags: {}", idx + 1, e));
                            }
                        }
                    }
                }
                Err(e) => {
                    warn!("Error running service line rules: {}", e);
                    result.warnings.push(format!("Line {} rules error: {}", idx + 1, e));
                }
            }
        }

        result.flag_count = total_flags;

        // *** PERFORMANCE: Calculate RVU payments with reduced logging ***
        let mut total_expected_payment = rust_decimal::Decimal::ZERO;

        // Default locality code (would be looked up from facility in production)
        let locality_code = "99"; // National average

        // Current year for RVU calculation
        let current_year = chrono::Utc::now().format("%Y").to_string().parse::<i32>().unwrap_or(2024);

        for parsed_line in claim.service_lines.iter() {
            // Build modifiers list
            let mut modifiers = Vec::new();
            if let Some(ref m) = parsed_line.procedure_modifier_1 { modifiers.push(m.clone()); }
            if let Some(ref m) = parsed_line.procedure_modifier_2 { modifiers.push(m.clone()); }
            if let Some(ref m) = parsed_line.procedure_modifier_3 { modifiers.push(m.clone()); }
            if let Some(ref m) = parsed_line.procedure_modifier_4 { modifiers.push(m.clone()); }

            // Get place of service (default to encounter level if not at line level)
            let pos_code = parsed_line.place_of_service_code.as_ref()
                .or(claim.place_of_service_code.as_ref())
                .map(|s| s.as_str())
                .unwrap_or("11"); // Default to office

            // Calculate payment
            if let Ok(payment_calc) = self.payment_calculator.calculate(
                &parsed_line.procedure_code,
                current_year,
                locality_code,
                pos_code,
                modifiers,
                parsed_line.service_unit_count,
            ) {
                total_expected_payment += payment_calc.total_payment;
                // Only log per-line details in debug mode
                // TODO: Persist payment calculation to database
                // This would require a service_line_payment table
            }
            // Errors are expected for non-RVU codes (supplies, drugs), so don't log
        }

        result.success = result.errors.is_empty();

        // *** PERFORMANCE: Single summary log per claim instead of 50+ individual logs ***
        info!(
            "Processed claim {} (enc: {}): {} dx, {} lines, {} flags, ${:.2} RVU",
            patient_control_number,
            encounter_id,
            diagnoses.len(),
            service_line_ids.len(),
            total_flags,
            total_expected_payment
        );

        Ok(result)
    }

    /// Process a single claim within an existing transaction (PHASE 2 OPTIMIZATION)
    async fn process_claim_in_transaction(
        &self,
        claim: &pro_parser_edi::types::ParsedClaim,
        tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
        organization_id: i64,
        _pcn_validator: &PatientControlNumberValidator,
        _service_line_validator: &ServiceLineValidator,
        _business_validator: &BusinessRuleValidator,
    ) -> Result<ClaimProcessingResult> {
        let patient_control_number = claim.patient_control_number.clone();

        let mut result = ClaimProcessingResult {
            patient_control_number: patient_control_number.clone(),
            encounter_id: None,
            success: false,
            errors: Vec::new(),
            warnings: Vec::new(),
            service_line_count: claim.service_lines.len(),
            flag_count: 0,
        };

        // Convert claim to encounter model
        let encounter = match self.convert_claim_to_encounter(claim, organization_id) {
            Ok(enc) => enc,
            Err(e) => {
                error!("Failed to convert claim to encounter: {}", e);
                result.errors.push(format!("Conversion error: {}", e));
                return Ok(result);
            }
        };

        // Create repositories
        let encounter_repo = EncounterRepository::new(&self.pool);
        let service_line_repo = ServiceLineRepository::new(&self.pool);

        // Insert encounter into database (within transaction)
        let encounter_id = match encounter_repo.create_with_tx(&encounter, tx).await {
            Ok(id) => id,
            Err(e) => {
                error!("Failed to insert encounter: {}", e);
                result.errors.push(format!("Database error: {}", e));
                return Ok(result);
            }
        };

        result.encounter_id = Some(encounter_id);

        // *** PERFORMANCE: Batch insert diagnosis codes (within transaction) ***
        let diagnoses: Vec<EncounterDiagnosis> = claim.diagnoses.iter().enumerate()
            .map(|(idx, parsed_dx)| EncounterDiagnosis {
                diagnosis_id: 0,
                encounter_id,
                sequence_number: (idx + 1) as i16,
                diagnosis_code_qualifier: Some(parsed_dx.diagnosis_code_qualifier.clone()),
                diagnosis_code: parsed_dx.diagnosis_code.clone(),
                diagnosis_description: None,
                is_principal: parsed_dx.is_principal,
                is_admitting: false,
                is_external_cause: false,
                is_patient_reason: false,
                present_on_admission_indicator: None,
                hcc_indicator: false,
                hcc_category: None,
                created_at: Utc::now(),
            })
            .collect();

        match encounter_repo.create_diagnoses_batch_with_tx(&diagnoses, tx).await {
            Ok(dx_ids) => {
                if cfg!(debug_assertions) {
                    info!("Inserted {} diagnoses for encounter {}", dx_ids.len(), encounter_id);
                }
            }
            Err(e) => {
                warn!("Failed to batch insert diagnoses for encounter {}: {}", encounter_id, e);
                result.warnings.push(format!("Diagnosis batch insert error: {}", e));
            }
        }

        // *** PERFORMANCE: Batch insert service lines (within transaction) ***
        let service_lines: Vec<ServiceLine> = claim.service_lines.iter().enumerate()
            .map(|(idx, parsed_line)| self.convert_service_line(parsed_line, encounter_id, (idx + 1) as i16))
            .collect();

        let service_line_ids = match service_line_repo.create_batch_with_tx(&service_lines, tx).await {
            Ok(ids) => {
                if cfg!(debug_assertions) {
                    info!("Inserted {} service lines for encounter {}", ids.len(), encounter_id);
                }
                ids
            }
            Err(e) => {
                error!("Failed to batch insert service lines for encounter {}: {}", encounter_id, e);
                result.warnings.push(format!("Service line batch insert error: {}", e));
                Vec::new()
            }
        };

        let mut total_flags = 0;

        // Build rule execution context for encounter-level rules
        let mut encounter_ctx = pro_rules::RuleExecutionContext::new(organization_id);
        encounter_ctx.encounter_id = Some(encounter_id);
        encounter_ctx.facility_id = Some(encounter.facility_id);
        encounter_ctx.total_claim_charge_amount = Some(claim.total_claim_charge_amount);
        encounter_ctx.place_of_service_code = claim.place_of_service_code.clone();
        encounter_ctx.date_of_service_from = Some(claim.date_of_service_from);
        encounter_ctx.date_of_service_to = claim.date_of_service_to;

        // Add diagnosis codes
        encounter_ctx.diagnosis_codes = claim.diagnoses.iter()
            .map(|d| d.diagnosis_code.clone())
            .collect();

        // Execute encounter-level rules
        match self.rule_engine.execute_all(&encounter_ctx).await {
            Ok(rule_results) => {
                if !rule_results.is_empty() {
                    total_flags += rule_results.len();

                    // Persist flags to database (within transaction)
                    match self.rule_engine.persist_flags_with_tx(rule_results, tx).await {
                        Ok(_flag_ids) => {
                            if cfg!(debug_assertions) {
                                info!("Persisted {} encounter-level flags", total_flags);
                            }
                        }
                        Err(e) => {
                            warn!("Failed to persist encounter flags: {}", e);
                            result.warnings.push(format!("Failed to persist encounter flags: {}", e));
                        }
                    }
                }
            }
            Err(e) => {
                warn!("Error running encounter-level rules: {}", e);
                result.warnings.push(format!("Encounter rules error: {}", e));
            }
        }

        // Run service line-level rules
        for (idx, (service_line_id, parsed_line)) in service_line_ids.iter().zip(claim.service_lines.iter()).enumerate() {
            let mut line_ctx = pro_rules::RuleExecutionContext::new(organization_id);
            line_ctx.encounter_id = Some(encounter_id);
            line_ctx.service_line_id = Some(*service_line_id);
            line_ctx.facility_id = Some(encounter.facility_id);
            line_ctx.procedure_code = Some(parsed_line.procedure_code.clone());
            line_ctx.service_unit_count = Some(parsed_line.service_unit_count);
            line_ctx.line_item_charge_amount = Some(parsed_line.line_item_charge_amount);
            line_ctx.date_of_service = Some(parsed_line.service_date_from);
            line_ctx.place_of_service_code = parsed_line.place_of_service_code.clone();

            // Add modifiers
            let mut modifiers = Vec::new();
            if let Some(ref m) = parsed_line.procedure_modifier_1 { modifiers.push(m.clone()); }
            if let Some(ref m) = parsed_line.procedure_modifier_2 { modifiers.push(m.clone()); }
            if let Some(ref m) = parsed_line.procedure_modifier_3 { modifiers.push(m.clone()); }
            if let Some(ref m) = parsed_line.procedure_modifier_4 { modifiers.push(m.clone()); }
            line_ctx.procedure_modifiers = modifiers;

            // Add diagnosis codes from encounter
            line_ctx.diagnosis_codes = claim.diagnoses.iter()
                .map(|d| d.diagnosis_code.clone())
                .collect();

            // Execute service line rules
            match self.rule_engine.execute_all(&line_ctx).await {
                Ok(rule_results) => {
                    if !rule_results.is_empty() {
                        total_flags += rule_results.len();

                        // Persist flags (within transaction)
                        match self.rule_engine.persist_flags_with_tx(rule_results, tx).await {
                            Ok(_flag_ids) => {
                                // Only log in debug mode
                            }
                            Err(e) => {
                                warn!("Failed to persist service line flags: {}", e);
                                result.warnings.push(format!("Failed to persist line {} flags: {}", idx + 1, e));
                            }
                        }
                    }
                }
                Err(e) => {
                    warn!("Error running service line rules: {}", e);
                    result.warnings.push(format!("Line {} rules error: {}", idx + 1, e));
                }
            }
        }

        result.flag_count = total_flags;

        // *** PERFORMANCE: Calculate RVU payments with reduced logging ***
        let mut total_expected_payment = rust_decimal::Decimal::ZERO;

        // Default locality code (would be looked up from facility in production)
        let locality_code = "99"; // National average

        // Current year for RVU calculation
        let current_year = chrono::Utc::now().format("%Y").to_string().parse::<i32>().unwrap_or(2024);

        for parsed_line in claim.service_lines.iter() {
            // Build modifiers list
            let mut modifiers = Vec::new();
            if let Some(ref m) = parsed_line.procedure_modifier_1 { modifiers.push(m.clone()); }
            if let Some(ref m) = parsed_line.procedure_modifier_2 { modifiers.push(m.clone()); }
            if let Some(ref m) = parsed_line.procedure_modifier_3 { modifiers.push(m.clone()); }
            if let Some(ref m) = parsed_line.procedure_modifier_4 { modifiers.push(m.clone()); }

            // Get place of service (default to encounter level if not at line level)
            let pos_code = parsed_line.place_of_service_code.as_ref()
                .or(claim.place_of_service_code.as_ref())
                .map(|s| s.as_str())
                .unwrap_or("11"); // Default to office

            // Calculate payment
            if let Ok(payment_calc) = self.payment_calculator.calculate(
                &parsed_line.procedure_code,
                current_year,
                locality_code,
                pos_code,
                modifiers,
                parsed_line.service_unit_count,
            ) {
                total_expected_payment += payment_calc.total_payment;
            }
        }

        result.success = result.errors.is_empty();

        // *** PERFORMANCE: Single summary log per claim ***
        info!(
            "Processed claim {} (enc: {}): {} dx, {} lines, {} flags, ${:.2} RVU",
            patient_control_number,
            encounter_id,
            diagnoses.len(),
            service_line_ids.len(),
            total_flags,
            total_expected_payment
        );

        Ok(result)
    }

    /// Process a single claim within an existing transaction with both caches (PHASE 3 & 5 OPTIMIZATION)
    async fn process_claim_in_transaction_with_caches(
        &self,
        claim: &pro_parser_edi::types::ParsedClaim,
        tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
        organization_id: i64,
        exec_cache: &pro_rules::RuleExecutionCache,
        result_cache: &pro_rules::RuleResultCache,
        _pcn_validator: &PatientControlNumberValidator,
        _service_line_validator: &ServiceLineValidator,
        _business_validator: &BusinessRuleValidator,
    ) -> Result<ClaimProcessingResult> {
        let patient_control_number = claim.patient_control_number.clone();

        let mut result = ClaimProcessingResult {
            patient_control_number: patient_control_number.clone(),
            encounter_id: None,
            success: false,
            errors: Vec::new(),
            warnings: Vec::new(),
            service_line_count: claim.service_lines.len(),
            flag_count: 0,
        };

        // Convert claim to encounter model
        let encounter = match self.convert_claim_to_encounter(claim, organization_id) {
            Ok(enc) => enc,
            Err(e) => {
                error!("Failed to convert claim to encounter: {}", e);
                result.errors.push(format!("Conversion error: {}", e));
                return Ok(result);
            }
        };

        // Create repositories
        let encounter_repo = EncounterRepository::new(&self.pool);
        let service_line_repo = ServiceLineRepository::new(&self.pool);

        // Insert encounter into database (within transaction)
        let encounter_id = match encounter_repo.create_with_tx(&encounter, tx).await {
            Ok(id) => id,
            Err(e) => {
                error!("Failed to insert encounter: {}", e);
                result.errors.push(format!("Database error: {}", e));
                return Ok(result);
            }
        };

        result.encounter_id = Some(encounter_id);

        // *** PERFORMANCE: Batch insert diagnosis codes (within transaction) ***
        let diagnoses: Vec<EncounterDiagnosis> = claim.diagnoses.iter().enumerate()
            .map(|(idx, parsed_dx)| EncounterDiagnosis {
                diagnosis_id: 0,
                encounter_id,
                sequence_number: (idx + 1) as i16,
                diagnosis_code_qualifier: Some(parsed_dx.diagnosis_code_qualifier.clone()),
                diagnosis_code: parsed_dx.diagnosis_code.clone(),
                diagnosis_description: None,
                is_principal: parsed_dx.is_principal,
                is_admitting: false,
                is_external_cause: false,
                is_patient_reason: false,
                present_on_admission_indicator: None,
                hcc_indicator: false,
                hcc_category: None,
                created_at: Utc::now(),
            })
            .collect();

        match encounter_repo.create_diagnoses_batch_with_tx(&diagnoses, tx).await {
            Ok(dx_ids) => {
                if cfg!(debug_assertions) {
                    info!("Inserted {} diagnoses for encounter {}", dx_ids.len(), encounter_id);
                }
            }
            Err(e) => {
                warn!("Failed to batch insert diagnoses for encounter {}: {}", encounter_id, e);
                result.warnings.push(format!("Diagnosis batch insert error: {}", e));
            }
        }

        // *** PERFORMANCE: Batch insert service lines (within transaction) ***
        let service_lines: Vec<ServiceLine> = claim.service_lines.iter().enumerate()
            .map(|(idx, parsed_line)| self.convert_service_line(parsed_line, encounter_id, (idx + 1) as i16))
            .collect();

        let service_line_ids = match service_line_repo.create_batch_with_tx(&service_lines, tx).await {
            Ok(ids) => {
                if cfg!(debug_assertions) {
                    info!("Inserted {} service lines for encounter {}", ids.len(), encounter_id);
                }
                ids
            }
            Err(e) => {
                error!("Failed to batch insert service lines for encounter {}: {}", encounter_id, e);
                result.warnings.push(format!("Service line batch insert error: {}", e));
                Vec::new()
            }
        };

        let mut total_flags = 0;

        // Build rule execution context for encounter-level rules
        let mut encounter_ctx = pro_rules::RuleExecutionContext::new(organization_id);
        encounter_ctx.encounter_id = Some(encounter_id);
        encounter_ctx.facility_id = Some(encounter.facility_id);
        encounter_ctx.total_claim_charge_amount = Some(claim.total_claim_charge_amount);
        encounter_ctx.place_of_service_code = claim.place_of_service_code.clone();
        encounter_ctx.date_of_service_from = Some(claim.date_of_service_from);
        encounter_ctx.date_of_service_to = claim.date_of_service_to;
        encounter_ctx.subscriber_id = Some(claim.subscriber_id.clone()); // PHASE 3

        // Add diagnosis codes
        encounter_ctx.diagnosis_codes = claim.diagnoses.iter()
            .map(|d| d.diagnosis_code.clone())
            .collect();

        // *** PHASE 5: Execute encounter-level rules with both caches ***
        match self.rule_engine.execute_all_with_result_cache(&encounter_ctx, exec_cache, result_cache).await {
            Ok(rule_results) => {
                if !rule_results.is_empty() {
                    total_flags += rule_results.len();

                    // Persist flags to database (within transaction)
                    match self.rule_engine.persist_flags_with_tx(rule_results, tx).await {
                        Ok(_flag_ids) => {
                            if cfg!(debug_assertions) {
                                info!("Persisted {} encounter-level flags", total_flags);
                            }
                        }
                        Err(e) => {
                            warn!("Failed to persist encounter flags: {}", e);
                            result.warnings.push(format!("Failed to persist encounter flags: {}", e));
                        }
                    }
                }
            }
            Err(e) => {
                warn!("Error running encounter-level rules: {}", e);
                result.warnings.push(format!("Encounter rules error: {}", e));
            }
        }

        // *** PHASE 3: Run service line-level rules with cache ***
        for (idx, (service_line_id, parsed_line)) in service_line_ids.iter().zip(claim.service_lines.iter()).enumerate() {
            let mut line_ctx = pro_rules::RuleExecutionContext::new(organization_id);
            line_ctx.encounter_id = Some(encounter_id);
            line_ctx.service_line_id = Some(*service_line_id);
            line_ctx.facility_id = Some(encounter.facility_id);
            line_ctx.procedure_code = Some(parsed_line.procedure_code.clone());
            line_ctx.service_unit_count = Some(parsed_line.service_unit_count);
            line_ctx.line_item_charge_amount = Some(parsed_line.line_item_charge_amount);
            line_ctx.date_of_service = Some(parsed_line.service_date_from);
            line_ctx.place_of_service_code = parsed_line.place_of_service_code.clone();
            line_ctx.subscriber_id = Some(claim.subscriber_id.clone()); // PHASE 3

            // Add modifiers
            let mut modifiers = Vec::new();
            if let Some(ref m) = parsed_line.procedure_modifier_1 { modifiers.push(m.clone()); }
            if let Some(ref m) = parsed_line.procedure_modifier_2 { modifiers.push(m.clone()); }
            if let Some(ref m) = parsed_line.procedure_modifier_3 { modifiers.push(m.clone()); }
            if let Some(ref m) = parsed_line.procedure_modifier_4 { modifiers.push(m.clone()); }
            line_ctx.procedure_modifiers = modifiers;

            // Add diagnosis codes from encounter
            line_ctx.diagnosis_codes = claim.diagnoses.iter()
                .map(|d| d.diagnosis_code.clone())
                .collect();

            // *** PHASE 5: Execute service line rules with both caches ***
            match self.rule_engine.execute_all_with_result_cache(&line_ctx, exec_cache, result_cache).await {
                Ok(rule_results) => {
                    if !rule_results.is_empty() {
                        total_flags += rule_results.len();

                        // Persist flags (within transaction)
                        match self.rule_engine.persist_flags_with_tx(rule_results, tx).await {
                            Ok(_flag_ids) => {
                                // Only log in debug mode
                            }
                            Err(e) => {
                                warn!("Failed to persist service line flags: {}", e);
                                result.warnings.push(format!("Failed to persist line {} flags: {}", idx + 1, e));
                            }
                        }
                    }
                }
                Err(e) => {
                    warn!("Error running service line rules: {}", e);
                    result.warnings.push(format!("Line {} rules error: {}", idx + 1, e));
                }
            }
        }

        result.flag_count = total_flags;

        // *** PERFORMANCE: Calculate RVU payments with reduced logging ***
        let mut total_expected_payment = rust_decimal::Decimal::ZERO;

        // Default locality code (would be looked up from facility in production)
        let locality_code = "99"; // National average

        // Current year for RVU calculation
        let current_year = chrono::Utc::now().format("%Y").to_string().parse::<i32>().unwrap_or(2024);

        for parsed_line in claim.service_lines.iter() {
            // Build modifiers list
            let mut modifiers = Vec::new();
            if let Some(ref m) = parsed_line.procedure_modifier_1 { modifiers.push(m.clone()); }
            if let Some(ref m) = parsed_line.procedure_modifier_2 { modifiers.push(m.clone()); }
            if let Some(ref m) = parsed_line.procedure_modifier_3 { modifiers.push(m.clone()); }
            if let Some(ref m) = parsed_line.procedure_modifier_4 { modifiers.push(m.clone()); }

            // Get place of service (default to encounter level if not at line level)
            let pos_code = parsed_line.place_of_service_code.as_ref()
                .or(claim.place_of_service_code.as_ref())
                .map(|s| s.as_str())
                .unwrap_or("11"); // Default to office

            // Calculate payment
            if let Ok(payment_calc) = self.payment_calculator.calculate(
                &parsed_line.procedure_code,
                current_year,
                locality_code,
                pos_code,
                modifiers,
                parsed_line.service_unit_count,
            ) {
                total_expected_payment += payment_calc.total_payment;
            }
        }

        result.success = result.errors.is_empty();

        // *** PERFORMANCE: Single summary log per claim ***
        info!(
            "Processed claim {} (enc: {}): {} dx, {} lines, {} flags, ${:.2} RVU",
            patient_control_number,
            encounter_id,
            diagnoses.len(),
            service_line_ids.len(),
            total_flags,
            total_expected_payment
        );

        Ok(result)
    }

    /// Convert parsed EDI claim to Encounter model
    fn convert_claim_to_encounter(
        &self,
        claim: &pro_parser_edi::types::ParsedClaim,
        organization_id: i64,
    ) -> Result<Encounter> {
        // For now, use hardcoded UUIDs for facility and region
        // In production, these would be looked up from the database
        let facility_id = 0i64; // TODO: Look up from database
        let region_id = None; // TODO: Look up from database

        // Extract submitter info (would come from transaction header in production)
        let submitter_id = "SYSTEM".to_string();
        let submitter_name = Some("Automated Import".to_string());

        // Build the encounter
        let encounter = Encounter {
            encounter_id: 0,
            facility_id,
            organization_id,
            region_id,

            // Submitter information
            submitter_id,
            submitter_name,

            // Control numbers
            patient_control_number: claim.patient_control_number.clone(),
            transaction_set_control_number: None,

            // Patient/Subscriber information
            subscriber_id: claim.subscriber_id.clone(),
            subscriber_last_name: claim.subscriber_last_name.clone(),
            subscriber_first_name: claim.subscriber_first_name.clone(),
            subscriber_middle_name: claim.subscriber_middle_name.clone(),
            subscriber_name_suffix: claim.subscriber_name_suffix.clone(),
            subscriber_gender: claim.subscriber_gender.clone(),
            subscriber_birth_date: claim.subscriber_date_of_birth.unwrap_or_else(|| {
                // Default to a reasonable date if missing
                chrono::NaiveDate::from_ymd_opt(1900, 1, 1).unwrap()
            }),
            subscriber_address_line1: claim.subscriber_address_line1.clone(),
            subscriber_address_line2: claim.subscriber_address_line2.clone(),
            subscriber_city: claim.subscriber_city.clone(),
            subscriber_state: claim.subscriber_state.clone(),
            subscriber_postal_code: claim.subscriber_postal_code.clone(),
            subscriber_country: claim.subscriber_country.clone(),

            // Payer information
            payer_responsibility_code: "P".to_string(), // Primary
            payer_id: Some(claim.payer_id.clone()),
            payer_name: Some(claim.payer_name.clone()),
            claim_filing_indicator: None,

            // Billing provider - would be looked up in production
            billing_provider_id: None,
            billing_provider_npi: None,
            billing_provider_tax_id: None,
            billing_provider_name: None,

            // Claim information
            total_claim_charge_amount: claim.total_claim_charge_amount,
            place_of_service_code: claim.place_of_service_code.clone(),
            claim_frequency_code: claim.claim_frequency_code.clone(),

            // Dates
            date_of_service_from: claim.date_of_service_from,
            date_of_service_to: claim.date_of_service_to,

            // Providers - would be looked up by NPI in production
            referring_provider_id: None,
            referring_provider_npi: claim.referring_provider_npi.clone(),
            rendering_provider_id: None,
            rendering_provider_npi: claim.rendering_provider_npi.clone(),
            supervising_provider_id: None,
            supervising_provider_npi: claim.supervising_provider_npi.clone(),

            // Service facility
            service_facility_id: None,
            service_facility_npi: claim.service_facility_npi.clone(),

            // Coder information
            coder_id: None,
            coding_date: None,

            // Status and workflow
            claim_status: "NEW".to_string(),
            case_status: Some("PENDING".to_string()),
            financial_class: None,

            // Import tracking
            import_batch_id: None, // Would be set from job context
            import_date: Some(Utc::now()),

            // Audit trail
            is_active: true,
            soft_deleted: false,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            created_by: Some("WORKER".to_string()),
            updated_by: Some("WORKER".to_string()),
        };

        Ok(encounter)
    }

    /// Convert parsed EDI service line to ServiceLine model
    fn convert_service_line(
        &self,
        parsed_line: &pro_parser_edi::types::ServiceLine,
        encounter_id: i64,
        line_number: i16,
    ) -> ServiceLine {
        ServiceLine {
            service_line_id: 0,
            encounter_id,
            line_number,

            // Service information
            product_service_id_qualifier: Some(parsed_line.product_service_id_qualifier.clone()),
            procedure_code: parsed_line.procedure_code.clone(),
            procedure_modifier_1: parsed_line.procedure_modifier_1.clone(),
            procedure_modifier_2: parsed_line.procedure_modifier_2.clone(),
            procedure_modifier_3: parsed_line.procedure_modifier_3.clone(),
            procedure_modifier_4: parsed_line.procedure_modifier_4.clone(),
            procedure_description: None, // Would be looked up from CPT reference
            line_item_charge_amount: parsed_line.line_item_charge_amount,
            unit_basis_measurement_code: Some(parsed_line.unit_basis_measurement_code.clone()),
            service_unit_count: parsed_line.service_unit_count,

            // Place of service
            place_of_service_code: parsed_line.place_of_service_code.clone(),

            // Dates
            service_date_from: parsed_line.service_date_from,
            service_date_to: parsed_line.service_date_to,

            // Providers at line level - would be looked up by NPI
            rendering_provider_id: None,
            rendering_provider_npi: parsed_line.rendering_provider_npi.clone(),
            supervising_provider_id: None,
            supervising_provider_npi: parsed_line.supervising_provider_npi.clone(),
            ordering_provider_id: None,
            ordering_provider_npi: parsed_line.ordering_provider_npi.clone(),
            referring_provider_id: None,
            referring_provider_npi: parsed_line.referring_provider_npi.clone(),

            // Service facility at line level
            service_facility_id: None,
            service_facility_npi: None,

            // Prior authorization and referral
            prior_authorization_number: parsed_line.prior_authorization_number.clone(),
            referral_number: parsed_line.referral_number.clone(),

            // Line note
            line_note: parsed_line.line_note.clone(),

            // Revenue code
            revenue_code: parsed_line.revenue_code.clone(),

            // NDC information
            ndc_code: parsed_line.ndc_code.clone(),
            ndc_unit_count: parsed_line.ndc_unit_count,
            ndc_measurement_unit: parsed_line.ndc_measurement_unit.clone(),

            // Diagnosis pointers
            diagnosis_code_pointer_1: parsed_line.diagnosis_code_pointer_1,
            diagnosis_code_pointer_2: parsed_line.diagnosis_code_pointer_2,
            diagnosis_code_pointer_3: parsed_line.diagnosis_code_pointer_3,
            diagnosis_code_pointer_4: parsed_line.diagnosis_code_pointer_4,

            // Status
            line_status: "NEW".to_string(),

            // Audit trail
            created_at: Utc::now(),
            updated_at: Utc::now(),
            created_by: Some("WORKER".to_string()),
            updated_by: Some("WORKER".to_string()),
        }
    }

    /// Update job status in database
    async fn update_job_status(&self, job: &IngestionJob) -> Result<()> {
        let query = r#"
            UPDATE staging.import_batch
            SET
                status = $1,
                started_at = $2,
                completed_at = $3,
                updated_at = CURRENT_TIMESTAMP
            WHERE import_batch_id = $4
        "#;

        sqlx::query(query)
            .bind(job.status.as_str())
            .bind(job.started_at)
            .bind(job.completed_at)
            .bind(job.import_batch_id)
            .execute(&self.pool)
            .await
            .map_err(|e| Error::Database(e))?;

        Ok(())
    }

    /// Get database pool
    pub fn pool(&self) -> &PgPool {
        &self.pool
    }

    /// Get rule engine
    pub fn rule_engine(&self) -> &RuleEngine {
        &self.rule_engine
    }

    /// Get payment calculator
    pub fn payment_calculator(&self) -> &PaymentCalculator {
        &self.payment_calculator
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_pipeline_creation() {
        let pool = PgPool::connect_lazy("postgres://dummy").unwrap();
        let pipeline = IngestionPipeline::new(pool);

        assert_eq!(pipeline.rule_engine().rule_count(), 6);
    }

    #[test]
    fn test_file_format_detection() {
        let edi_format = FileFormat::from_extension("edi");
        assert_eq!(edi_format, Some(FileFormat::Edi837p));

        let csv_format = FileFormat::from_extension("csv");
        assert_eq!(csv_format, Some(FileFormat::Csv));
    }
}
