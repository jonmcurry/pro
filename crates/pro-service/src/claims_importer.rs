//! Claims importer for CSV files
//!
//! Imports claims data from CSV files into the database using the pro-parser-csv crate.

use anyhow::{Context, Result};
use pro_parser_csv::CsvParser;
use sqlx::PgPool;
use std::path::Path;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// Claims importer that processes CSV files
#[derive(Clone)]
pub struct ClaimsImporter {
    pool: PgPool,
}

impl ClaimsImporter {
    /// Create a new claims importer
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Import a CSV file containing claims
    pub async fn import_file(&self, file_path: &Path) -> Result<ImportResult> {
        let file_path_str = file_path.display().to_string();
        let filename = file_path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();

        info!("====== Starting import of file: {} ======", file_path_str);
        info!("File name: {}", filename);

        // Parse CSV file using auto-detection
        info!("Parsing CSV file...");
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

        let mut result = ImportResult {
            total_rows: parsed_rows.len(),
            successful: 0,
            failed: 0,
            errors: Vec::new(),
        };

        // Begin transaction
        let mut tx = self.pool.begin().await
            .context("Failed to begin database transaction")?;

        // Import each row
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
                }
                continue;
            }

            // Log warnings
            for warning in &parsed_row.warnings {
                debug!("Row {} warning: {}", parsed_row.row_number, warning);
            }

            // Import the row
            match self.import_encounter(&mut tx, &parsed_row).await {
                Ok(_) => {
                    result.successful += 1;
                    debug!("Successfully imported row {}", parsed_row.row_number);
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
                }
            }
        }

        // Commit or rollback based on results
        let import_status = if result.failed == 0 {
            tx.commit().await
                .context("Failed to commit transaction")?;
            info!("Successfully imported all {} rows", result.successful);
            "COMPLETED"
        } else if result.successful > 0 {
            // Partial success - commit what we can
            tx.commit().await
                .context("Failed to commit transaction")?;
            warn!("Imported {} rows, {} failed", result.successful, result.failed);
            "PARTIAL"
        } else {
            // Total failure - rollback
            tx.rollback().await
                .context("Failed to rollback transaction")?;
            error!("All rows failed to import");
            "FAILED"
        };

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
    ) -> Result<Uuid> {
        // Extract facility_code from encounter fields
        let facility_code = parsed_row.encounter_fields.get("facility_code")
            .or_else(|| parsed_row.encounter_fields.get("facility_npi"))
            .context("Missing facility_code or facility_npi")?;

        // Look up facility to get facility_id, organization_id, and region_id
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

        let (facility_id, organization_id, region_id) = facility
            .with_context(|| format!("Facility not found: {}", facility_code))?;

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

/// Result of a file import operation
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
