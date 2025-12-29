// File processing pipeline

use crate::converters;
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
use pro_rules::{RuleEngine, load_rules_from_database};
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
    /// Reserved for future facility-specific rule loading feature
    #[allow(dead_code)]
    facility_id: Option<i64>,
}

impl IngestionPipeline {
    /// Create a new ingestion pipeline with database-driven rules
    ///
    /// Rules are loaded from database based on ENABLE_DATABASE_RULES env var:
    /// - If true: Load rules from database (requires RULE_ENCRYPTION_KEY)
    /// - If false: Use legacy hard-coded rules (backward compatibility)
    pub async fn new(pool: PgPool, facility_id: Option<i64>) -> Result<Self> {
        // Check if database-driven rules are enabled
        let use_database_rules = std::env::var("ENABLE_DATABASE_RULES")
            .unwrap_or_else(|_| "false".to_string())
            .parse::<bool>()
            .unwrap_or(false);

        let rule_engine = if use_database_rules {
            info!("Loading rules from database (facility_id: {:?})", facility_id);
            match load_rules_from_database(&pool, facility_id).await {
                Ok((engine, rules)) => {
                    info!("Loaded {} rule(s) from database", rules.len());
                    for rule in rules {
                        info!("  - {} ({}): {}", rule.rule_code, rule.execution_level, rule.rule_name);
                    }
                    engine
                }
                Err(e) => {
                    error!("Failed to load rules from database: {}", e);
                    warn!("Falling back to legacy hard-coded rules");
                    Self::create_legacy_rule_engine(&pool)
                }
            }
        } else {
            info!("Using legacy hard-coded rules (ENABLE_DATABASE_RULES=false)");
            Self::create_legacy_rule_engine(&pool)
        };

        // Initialize payment calculator with sample data
        // In production, this would load from database
        let payment_calculator = PaymentCalculator::with_sample_data();

        Ok(Self {
            pool,
            rule_engine,
            payment_calculator,
            facility_id,
        })
    }

    /// Create legacy rule engine with hard-coded rules (backward compatibility)
    fn create_legacy_rule_engine(pool: &PgPool) -> RuleEngine {
        let mut rule_engine = RuleEngine::new(pool.clone());

        // Add default rules directly (legacy approach)
        rule_engine.add_rule(pro_rules::rules::DuplicateServiceRule);
        rule_engine.add_rule(pro_rules::rules::UnitsExceedMaximumRule::default());
        rule_engine.add_rule(pro_rules::rules::MissingRequiredModifierRule::default());
        rule_engine.add_rule(pro_rules::rules::ConflictingModifiersRule);
        rule_engine.add_rule(pro_rules::rules::UnspecifiedDiagnosisRule);
        rule_engine.add_rule(pro_rules::rules::MissingDiagnosisSpecificityRule);

        rule_engine
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

    /// Convert CSV row to ParsedClaim - delegates to converters module
    fn convert_csv_to_claim(
        &self,
        csv_row: &pro_parser_csv::parser::ParsedRow,
        _organization_id: i64,
    ) -> Result<pro_parser_edi::types::ParsedClaim> {
        converters::convert_csv_to_claim(csv_row)
    }

    /// Extract diagnosis codes from CSV row - delegates to converters module
    /// Reserved for future direct CSV diagnosis extraction
    #[allow(dead_code)]
    fn extract_diagnoses_from_csv(
        &self,
        csv_row: &pro_parser_csv::parser::ParsedRow,
    ) -> Result<Vec<pro_parser_edi::types::DiagnosisCode>> {
        converters::extract_diagnoses_from_csv(csv_row)
    }

    /// Extract service lines from CSV row - delegates to converters module
    /// Reserved for future direct CSV service line extraction
    #[allow(dead_code)]
    fn extract_service_lines_from_csv(
        &self,
        csv_row: &pro_parser_csv::parser::ParsedRow,
    ) -> Result<Vec<pro_parser_edi::types::ServiceLine>> {
        converters::extract_service_lines_from_csv(csv_row)
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

        // Load facility data for rules
        if let Ok(Some((state_code, facility_type))) = sqlx::query_as::<_, (Option<String>, Option<String>)>(
            "SELECT state_code, facility_type FROM core.facility WHERE facility_id = $1"
        )
        .bind(encounter.facility_id)
        .fetch_optional(&self.pool)
        .await {
            encounter_ctx.facility_state_code = state_code;
            encounter_ctx.facility_type = facility_type;
        }

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

            // Load facility data for rules (reuse from encounter_ctx to avoid duplicate query)
            line_ctx.facility_state_code = encounter_ctx.facility_state_code.clone();
            line_ctx.facility_type = encounter_ctx.facility_type.clone();

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
    /// Superseded by process_claim_in_transaction_with_caches for better performance
    #[allow(dead_code)]
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

        // Load facility data for rules
        if let Ok(Some((state_code, facility_type))) = sqlx::query_as::<_, (Option<String>, Option<String>)>(
            "SELECT state_code, facility_type FROM core.facility WHERE facility_id = $1"
        )
        .bind(encounter.facility_id)
        .fetch_optional(&self.pool)
        .await {
            encounter_ctx.facility_state_code = state_code;
            encounter_ctx.facility_type = facility_type;
        }

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

            // Load facility data for rules (reuse from encounter_ctx)
            line_ctx.facility_state_code = encounter_ctx.facility_state_code.clone();
            line_ctx.facility_type = encounter_ctx.facility_type.clone();

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

        // Load facility data for rules
        if let Ok(Some((state_code, facility_type))) = sqlx::query_as::<_, (Option<String>, Option<String>)>(
            "SELECT state_code, facility_type FROM core.facility WHERE facility_id = $1"
        )
        .bind(encounter.facility_id)
        .fetch_optional(&self.pool)
        .await {
            encounter_ctx.facility_state_code = state_code;
            encounter_ctx.facility_type = facility_type;
        }

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

            // Load facility data for rules (reuse from encounter_ctx)
            line_ctx.facility_state_code = encounter_ctx.facility_state_code.clone();
            line_ctx.facility_type = encounter_ctx.facility_type.clone();

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

    /// Convert parsed EDI claim to Encounter model - delegates to converters module
    fn convert_claim_to_encounter(
        &self,
        claim: &pro_parser_edi::types::ParsedClaim,
        organization_id: i64,
    ) -> Result<Encounter> {
        converters::convert_claim_to_encounter(claim, organization_id)
    }

    /// Convert parsed EDI service line to ServiceLine model - delegates to converters module
    fn convert_service_line(
        &self,
        parsed_line: &pro_parser_edi::types::ServiceLine,
        encounter_id: i64,
        line_number: i16,
    ) -> ServiceLine {
        converters::convert_service_line(parsed_line, encounter_id, line_number)
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
        let pipeline = IngestionPipeline::new(pool, None).await.unwrap();

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
