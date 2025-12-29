//! Claim Processor Module
//!
//! Extracted from IngestionPipeline as part of god object refactoring.
//! Handles individual claim processing logic.

use crate::converters;
use crate::types::ClaimProcessingResult;
use chrono::Utc;
use pro_common::Result;
use pro_db::{
    models::{EncounterDiagnosis, ServiceLine},
    repositories::{EncounterRepository, ServiceLineRepository},
    BusinessRuleValidator, PatientControlNumberValidator, ServiceLineValidator,
};
use pro_rules::RuleEngine;
use pro_rvu::PaymentCalculator;
use sqlx::PgPool;
use tracing::{error, info, warn};

/// Claim processor for handling individual claim processing
pub struct ClaimProcessor {
    pool: PgPool,
    rule_engine: RuleEngine,
    payment_calculator: PaymentCalculator,
}

impl ClaimProcessor {
    /// Create a new claim processor
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

    /// Process a single claim
    pub async fn process_claim(
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
        let encounter = match converters::convert_claim_to_encounter(claim, organization_id) {
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

        // Batch insert diagnosis codes
        let diagnoses: Vec<EncounterDiagnosis> = claim
            .diagnoses
            .iter()
            .enumerate()
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

        match encounter_repo.create_diagnoses_batch(&diagnoses).await {
            Ok(dx_ids) => {
                if cfg!(debug_assertions) {
                    info!(
                        "Inserted {} diagnoses for encounter {}",
                        dx_ids.len(),
                        encounter_id
                    );
                }
            }
            Err(e) => {
                warn!(
                    "Failed to batch insert diagnoses for encounter {}: {}",
                    encounter_id, e
                );
                result
                    .warnings
                    .push(format!("Diagnosis batch insert error: {}", e));
            }
        }

        // Batch insert service lines
        let service_lines: Vec<ServiceLine> = claim
            .service_lines
            .iter()
            .enumerate()
            .map(|(idx, parsed_line)| {
                converters::convert_service_line(parsed_line, encounter_id, (idx + 1) as i16)
            })
            .collect();

        let service_line_ids = match service_line_repo.create_batch(&service_lines).await {
            Ok(ids) => {
                if cfg!(debug_assertions) {
                    info!(
                        "Inserted {} service lines for encounter {}",
                        ids.len(),
                        encounter_id
                    );
                }
                ids
            }
            Err(e) => {
                error!(
                    "Failed to batch insert service lines for encounter {}: {}",
                    encounter_id, e
                );
                result
                    .warnings
                    .push(format!("Service line batch insert error: {}", e));
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
        if let Ok(Some((state_code, facility_type))) =
            sqlx::query_as::<_, (Option<String>, Option<String>)>(
                "SELECT state_code, facility_type FROM core.facility WHERE facility_id = $1",
            )
            .bind(encounter.facility_id)
            .fetch_optional(&self.pool)
            .await
        {
            encounter_ctx.facility_state_code = state_code;
            encounter_ctx.facility_type = facility_type;
        }

        // Add diagnosis codes
        encounter_ctx.diagnosis_codes = claim
            .diagnoses
            .iter()
            .map(|d| d.diagnosis_code.clone())
            .collect();

        // Execute encounter-level rules
        match self.rule_engine.execute_all(&encounter_ctx).await {
            Ok(rule_results) => {
                if !rule_results.is_empty() {
                    total_flags += rule_results.len();

                    match self.rule_engine.persist_flags(rule_results).await {
                        Ok(_flag_ids) => {
                            if cfg!(debug_assertions) {
                                info!("Persisted {} encounter-level flags", total_flags);
                            }
                        }
                        Err(e) => {
                            warn!("Failed to persist encounter flags: {}", e);
                            result
                                .warnings
                                .push(format!("Failed to persist encounter flags: {}", e));
                        }
                    }
                }
            }
            Err(e) => {
                warn!("Error running encounter-level rules: {}", e);
                result
                    .warnings
                    .push(format!("Encounter rules error: {}", e));
            }
        }

        // Run service line-level rules
        for (idx, (service_line_id, parsed_line)) in service_line_ids
            .iter()
            .zip(claim.service_lines.iter())
            .enumerate()
        {
            let mut line_ctx = pro_rules::RuleExecutionContext::new(organization_id);
            line_ctx.encounter_id = Some(encounter_id);
            line_ctx.service_line_id = Some(*service_line_id);
            line_ctx.facility_id = Some(encounter.facility_id);
            line_ctx.procedure_code = Some(parsed_line.procedure_code.clone());
            line_ctx.service_unit_count = Some(parsed_line.service_unit_count);
            line_ctx.line_item_charge_amount = Some(parsed_line.line_item_charge_amount);
            line_ctx.date_of_service = Some(parsed_line.service_date_from);
            line_ctx.place_of_service_code = parsed_line.place_of_service_code.clone();

            line_ctx.facility_state_code = encounter_ctx.facility_state_code.clone();
            line_ctx.facility_type = encounter_ctx.facility_type.clone();

            // Add modifiers
            let mut modifiers = Vec::new();
            if let Some(ref m) = parsed_line.procedure_modifier_1 {
                modifiers.push(m.clone());
            }
            if let Some(ref m) = parsed_line.procedure_modifier_2 {
                modifiers.push(m.clone());
            }
            if let Some(ref m) = parsed_line.procedure_modifier_3 {
                modifiers.push(m.clone());
            }
            if let Some(ref m) = parsed_line.procedure_modifier_4 {
                modifiers.push(m.clone());
            }
            line_ctx.procedure_modifiers = modifiers;

            line_ctx.diagnosis_codes = claim
                .diagnoses
                .iter()
                .map(|d| d.diagnosis_code.clone())
                .collect();

            match self.rule_engine.execute_all(&line_ctx).await {
                Ok(rule_results) => {
                    if !rule_results.is_empty() {
                        total_flags += rule_results.len();

                        match self.rule_engine.persist_flags(rule_results).await {
                            Ok(_flag_ids) => {}
                            Err(e) => {
                                warn!("Failed to persist service line flags: {}", e);
                                result.warnings.push(format!(
                                    "Failed to persist line {} flags: {}",
                                    idx + 1,
                                    e
                                ));
                            }
                        }
                    }
                }
                Err(e) => {
                    warn!("Error running service line rules: {}", e);
                    result
                        .warnings
                        .push(format!("Line {} rules error: {}", idx + 1, e));
                }
            }
        }

        result.flag_count = total_flags;

        // Calculate RVU payments
        let mut total_expected_payment = rust_decimal::Decimal::ZERO;
        let locality_code = "99";
        let current_year = chrono::Utc::now()
            .format("%Y")
            .to_string()
            .parse::<i32>()
            .unwrap_or(2024);

        for parsed_line in claim.service_lines.iter() {
            let mut modifiers = Vec::new();
            if let Some(ref m) = parsed_line.procedure_modifier_1 {
                modifiers.push(m.clone());
            }
            if let Some(ref m) = parsed_line.procedure_modifier_2 {
                modifiers.push(m.clone());
            }
            if let Some(ref m) = parsed_line.procedure_modifier_3 {
                modifiers.push(m.clone());
            }
            if let Some(ref m) = parsed_line.procedure_modifier_4 {
                modifiers.push(m.clone());
            }

            let pos_code = parsed_line
                .place_of_service_code
                .as_ref()
                .or(claim.place_of_service_code.as_ref())
                .map(|s| s.as_str())
                .unwrap_or("11");

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

    /// Process a single claim within an existing transaction
    pub async fn process_claim_in_transaction(
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

        let encounter = match converters::convert_claim_to_encounter(claim, organization_id) {
            Ok(enc) => enc,
            Err(e) => {
                error!("Failed to convert claim to encounter: {}", e);
                result.errors.push(format!("Conversion error: {}", e));
                return Ok(result);
            }
        };

        let encounter_repo = EncounterRepository::new(&self.pool);
        let service_line_repo = ServiceLineRepository::new(&self.pool);

        let encounter_id = match encounter_repo.create_with_tx(&encounter, tx).await {
            Ok(id) => id,
            Err(e) => {
                error!("Failed to insert encounter: {}", e);
                result.errors.push(format!("Database error: {}", e));
                return Ok(result);
            }
        };

        result.encounter_id = Some(encounter_id);

        let diagnoses: Vec<EncounterDiagnosis> = claim
            .diagnoses
            .iter()
            .enumerate()
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

        match encounter_repo
            .create_diagnoses_batch_with_tx(&diagnoses, tx)
            .await
        {
            Ok(dx_ids) => {
                if cfg!(debug_assertions) {
                    info!(
                        "Inserted {} diagnoses for encounter {}",
                        dx_ids.len(),
                        encounter_id
                    );
                }
            }
            Err(e) => {
                warn!(
                    "Failed to batch insert diagnoses for encounter {}: {}",
                    encounter_id, e
                );
                result
                    .warnings
                    .push(format!("Diagnosis batch insert error: {}", e));
            }
        }

        let service_lines: Vec<ServiceLine> = claim
            .service_lines
            .iter()
            .enumerate()
            .map(|(idx, parsed_line)| {
                converters::convert_service_line(parsed_line, encounter_id, (idx + 1) as i16)
            })
            .collect();

        let service_line_ids = match service_line_repo
            .create_batch_with_tx(&service_lines, tx)
            .await
        {
            Ok(ids) => {
                if cfg!(debug_assertions) {
                    info!(
                        "Inserted {} service lines for encounter {}",
                        ids.len(),
                        encounter_id
                    );
                }
                ids
            }
            Err(e) => {
                error!(
                    "Failed to batch insert service lines for encounter {}: {}",
                    encounter_id, e
                );
                result
                    .warnings
                    .push(format!("Service line batch insert error: {}", e));
                Vec::new()
            }
        };

        let mut total_flags = 0;

        let mut encounter_ctx = pro_rules::RuleExecutionContext::new(organization_id);
        encounter_ctx.encounter_id = Some(encounter_id);
        encounter_ctx.facility_id = Some(encounter.facility_id);
        encounter_ctx.total_claim_charge_amount = Some(claim.total_claim_charge_amount);
        encounter_ctx.place_of_service_code = claim.place_of_service_code.clone();
        encounter_ctx.date_of_service_from = Some(claim.date_of_service_from);
        encounter_ctx.date_of_service_to = claim.date_of_service_to;

        if let Ok(Some((state_code, facility_type))) =
            sqlx::query_as::<_, (Option<String>, Option<String>)>(
                "SELECT state_code, facility_type FROM core.facility WHERE facility_id = $1",
            )
            .bind(encounter.facility_id)
            .fetch_optional(&self.pool)
            .await
        {
            encounter_ctx.facility_state_code = state_code;
            encounter_ctx.facility_type = facility_type;
        }

        encounter_ctx.diagnosis_codes = claim
            .diagnoses
            .iter()
            .map(|d| d.diagnosis_code.clone())
            .collect();

        match self.rule_engine.execute_all(&encounter_ctx).await {
            Ok(rule_results) => {
                if !rule_results.is_empty() {
                    total_flags += rule_results.len();

                    match self
                        .rule_engine
                        .persist_flags_with_tx(rule_results, tx)
                        .await
                    {
                        Ok(_flag_ids) => {
                            if cfg!(debug_assertions) {
                                info!("Persisted {} encounter-level flags", total_flags);
                            }
                        }
                        Err(e) => {
                            warn!("Failed to persist encounter flags: {}", e);
                            result
                                .warnings
                                .push(format!("Failed to persist encounter flags: {}", e));
                        }
                    }
                }
            }
            Err(e) => {
                warn!("Error running encounter-level rules: {}", e);
                result
                    .warnings
                    .push(format!("Encounter rules error: {}", e));
            }
        }

        for (idx, (service_line_id, parsed_line)) in service_line_ids
            .iter()
            .zip(claim.service_lines.iter())
            .enumerate()
        {
            let mut line_ctx = pro_rules::RuleExecutionContext::new(organization_id);
            line_ctx.encounter_id = Some(encounter_id);
            line_ctx.service_line_id = Some(*service_line_id);
            line_ctx.facility_id = Some(encounter.facility_id);
            line_ctx.procedure_code = Some(parsed_line.procedure_code.clone());
            line_ctx.service_unit_count = Some(parsed_line.service_unit_count);
            line_ctx.line_item_charge_amount = Some(parsed_line.line_item_charge_amount);
            line_ctx.date_of_service = Some(parsed_line.service_date_from);
            line_ctx.place_of_service_code = parsed_line.place_of_service_code.clone();

            line_ctx.facility_state_code = encounter_ctx.facility_state_code.clone();
            line_ctx.facility_type = encounter_ctx.facility_type.clone();

            let mut modifiers = Vec::new();
            if let Some(ref m) = parsed_line.procedure_modifier_1 {
                modifiers.push(m.clone());
            }
            if let Some(ref m) = parsed_line.procedure_modifier_2 {
                modifiers.push(m.clone());
            }
            if let Some(ref m) = parsed_line.procedure_modifier_3 {
                modifiers.push(m.clone());
            }
            if let Some(ref m) = parsed_line.procedure_modifier_4 {
                modifiers.push(m.clone());
            }
            line_ctx.procedure_modifiers = modifiers;

            line_ctx.diagnosis_codes = claim
                .diagnoses
                .iter()
                .map(|d| d.diagnosis_code.clone())
                .collect();

            match self.rule_engine.execute_all(&line_ctx).await {
                Ok(rule_results) => {
                    if !rule_results.is_empty() {
                        total_flags += rule_results.len();

                        match self
                            .rule_engine
                            .persist_flags_with_tx(rule_results, tx)
                            .await
                        {
                            Ok(_flag_ids) => {}
                            Err(e) => {
                                warn!("Failed to persist service line flags: {}", e);
                                result.warnings.push(format!(
                                    "Failed to persist line {} flags: {}",
                                    idx + 1,
                                    e
                                ));
                            }
                        }
                    }
                }
                Err(e) => {
                    warn!("Error running service line rules: {}", e);
                    result
                        .warnings
                        .push(format!("Line {} rules error: {}", idx + 1, e));
                }
            }
        }

        result.flag_count = total_flags;

        let mut total_expected_payment = rust_decimal::Decimal::ZERO;
        let locality_code = "99";
        let current_year = chrono::Utc::now()
            .format("%Y")
            .to_string()
            .parse::<i32>()
            .unwrap_or(2024);

        for parsed_line in claim.service_lines.iter() {
            let mut modifiers = Vec::new();
            if let Some(ref m) = parsed_line.procedure_modifier_1 {
                modifiers.push(m.clone());
            }
            if let Some(ref m) = parsed_line.procedure_modifier_2 {
                modifiers.push(m.clone());
            }
            if let Some(ref m) = parsed_line.procedure_modifier_3 {
                modifiers.push(m.clone());
            }
            if let Some(ref m) = parsed_line.procedure_modifier_4 {
                modifiers.push(m.clone());
            }

            let pos_code = parsed_line
                .place_of_service_code
                .as_ref()
                .or(claim.place_of_service_code.as_ref())
                .map(|s| s.as_str())
                .unwrap_or("11");

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

    /// Process a single claim within an existing transaction with both caches
    pub async fn process_claim_in_transaction_with_caches(
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

        let encounter = match converters::convert_claim_to_encounter(claim, organization_id) {
            Ok(enc) => enc,
            Err(e) => {
                error!("Failed to convert claim to encounter: {}", e);
                result.errors.push(format!("Conversion error: {}", e));
                return Ok(result);
            }
        };

        let encounter_repo = EncounterRepository::new(&self.pool);
        let service_line_repo = ServiceLineRepository::new(&self.pool);

        let encounter_id = match encounter_repo.create_with_tx(&encounter, tx).await {
            Ok(id) => id,
            Err(e) => {
                error!("Failed to insert encounter: {}", e);
                result.errors.push(format!("Database error: {}", e));
                return Ok(result);
            }
        };

        result.encounter_id = Some(encounter_id);

        let diagnoses: Vec<EncounterDiagnosis> = claim
            .diagnoses
            .iter()
            .enumerate()
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

        match encounter_repo
            .create_diagnoses_batch_with_tx(&diagnoses, tx)
            .await
        {
            Ok(dx_ids) => {
                if cfg!(debug_assertions) {
                    info!(
                        "Inserted {} diagnoses for encounter {}",
                        dx_ids.len(),
                        encounter_id
                    );
                }
            }
            Err(e) => {
                warn!(
                    "Failed to batch insert diagnoses for encounter {}: {}",
                    encounter_id, e
                );
                result
                    .warnings
                    .push(format!("Diagnosis batch insert error: {}", e));
            }
        }

        let service_lines: Vec<ServiceLine> = claim
            .service_lines
            .iter()
            .enumerate()
            .map(|(idx, parsed_line)| {
                converters::convert_service_line(parsed_line, encounter_id, (idx + 1) as i16)
            })
            .collect();

        let service_line_ids = match service_line_repo
            .create_batch_with_tx(&service_lines, tx)
            .await
        {
            Ok(ids) => {
                if cfg!(debug_assertions) {
                    info!(
                        "Inserted {} service lines for encounter {}",
                        ids.len(),
                        encounter_id
                    );
                }
                ids
            }
            Err(e) => {
                error!(
                    "Failed to batch insert service lines for encounter {}: {}",
                    encounter_id, e
                );
                result
                    .warnings
                    .push(format!("Service line batch insert error: {}", e));
                Vec::new()
            }
        };

        let mut total_flags = 0;

        let mut encounter_ctx = pro_rules::RuleExecutionContext::new(organization_id);
        encounter_ctx.encounter_id = Some(encounter_id);
        encounter_ctx.facility_id = Some(encounter.facility_id);
        encounter_ctx.total_claim_charge_amount = Some(claim.total_claim_charge_amount);
        encounter_ctx.place_of_service_code = claim.place_of_service_code.clone();
        encounter_ctx.date_of_service_from = Some(claim.date_of_service_from);
        encounter_ctx.date_of_service_to = claim.date_of_service_to;
        encounter_ctx.subscriber_id = Some(claim.subscriber_id.clone());

        if let Ok(Some((state_code, facility_type))) =
            sqlx::query_as::<_, (Option<String>, Option<String>)>(
                "SELECT state_code, facility_type FROM core.facility WHERE facility_id = $1",
            )
            .bind(encounter.facility_id)
            .fetch_optional(&self.pool)
            .await
        {
            encounter_ctx.facility_state_code = state_code;
            encounter_ctx.facility_type = facility_type;
        }

        encounter_ctx.diagnosis_codes = claim
            .diagnoses
            .iter()
            .map(|d| d.diagnosis_code.clone())
            .collect();

        match self
            .rule_engine
            .execute_all_with_result_cache(&encounter_ctx, exec_cache, result_cache)
            .await
        {
            Ok(rule_results) => {
                if !rule_results.is_empty() {
                    total_flags += rule_results.len();

                    match self
                        .rule_engine
                        .persist_flags_with_tx(rule_results, tx)
                        .await
                    {
                        Ok(_flag_ids) => {
                            if cfg!(debug_assertions) {
                                info!("Persisted {} encounter-level flags", total_flags);
                            }
                        }
                        Err(e) => {
                            warn!("Failed to persist encounter flags: {}", e);
                            result
                                .warnings
                                .push(format!("Failed to persist encounter flags: {}", e));
                        }
                    }
                }
            }
            Err(e) => {
                warn!("Error running encounter-level rules: {}", e);
                result
                    .warnings
                    .push(format!("Encounter rules error: {}", e));
            }
        }

        for (idx, (service_line_id, parsed_line)) in service_line_ids
            .iter()
            .zip(claim.service_lines.iter())
            .enumerate()
        {
            let mut line_ctx = pro_rules::RuleExecutionContext::new(organization_id);
            line_ctx.encounter_id = Some(encounter_id);
            line_ctx.service_line_id = Some(*service_line_id);
            line_ctx.facility_id = Some(encounter.facility_id);
            line_ctx.procedure_code = Some(parsed_line.procedure_code.clone());
            line_ctx.service_unit_count = Some(parsed_line.service_unit_count);
            line_ctx.line_item_charge_amount = Some(parsed_line.line_item_charge_amount);
            line_ctx.date_of_service = Some(parsed_line.service_date_from);
            line_ctx.place_of_service_code = parsed_line.place_of_service_code.clone();
            line_ctx.subscriber_id = Some(claim.subscriber_id.clone());

            line_ctx.facility_state_code = encounter_ctx.facility_state_code.clone();
            line_ctx.facility_type = encounter_ctx.facility_type.clone();

            let mut modifiers = Vec::new();
            if let Some(ref m) = parsed_line.procedure_modifier_1 {
                modifiers.push(m.clone());
            }
            if let Some(ref m) = parsed_line.procedure_modifier_2 {
                modifiers.push(m.clone());
            }
            if let Some(ref m) = parsed_line.procedure_modifier_3 {
                modifiers.push(m.clone());
            }
            if let Some(ref m) = parsed_line.procedure_modifier_4 {
                modifiers.push(m.clone());
            }
            line_ctx.procedure_modifiers = modifiers;

            line_ctx.diagnosis_codes = claim
                .diagnoses
                .iter()
                .map(|d| d.diagnosis_code.clone())
                .collect();

            match self
                .rule_engine
                .execute_all_with_result_cache(&line_ctx, exec_cache, result_cache)
                .await
            {
                Ok(rule_results) => {
                    if !rule_results.is_empty() {
                        total_flags += rule_results.len();

                        match self
                            .rule_engine
                            .persist_flags_with_tx(rule_results, tx)
                            .await
                        {
                            Ok(_flag_ids) => {}
                            Err(e) => {
                                warn!("Failed to persist service line flags: {}", e);
                                result.warnings.push(format!(
                                    "Failed to persist line {} flags: {}",
                                    idx + 1,
                                    e
                                ));
                            }
                        }
                    }
                }
                Err(e) => {
                    warn!("Error running service line rules: {}", e);
                    result
                        .warnings
                        .push(format!("Line {} rules error: {}", idx + 1, e));
                }
            }
        }

        result.flag_count = total_flags;

        let mut total_expected_payment = rust_decimal::Decimal::ZERO;
        let locality_code = "99";
        let current_year = chrono::Utc::now()
            .format("%Y")
            .to_string()
            .parse::<i32>()
            .unwrap_or(2024);

        for parsed_line in claim.service_lines.iter() {
            let mut modifiers = Vec::new();
            if let Some(ref m) = parsed_line.procedure_modifier_1 {
                modifiers.push(m.clone());
            }
            if let Some(ref m) = parsed_line.procedure_modifier_2 {
                modifiers.push(m.clone());
            }
            if let Some(ref m) = parsed_line.procedure_modifier_3 {
                modifiers.push(m.clone());
            }
            if let Some(ref m) = parsed_line.procedure_modifier_4 {
                modifiers.push(m.clone());
            }

            let pos_code = parsed_line
                .place_of_service_code
                .as_ref()
                .or(claim.place_of_service_code.as_ref())
                .map(|s| s.as_str())
                .unwrap_or("11");

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
}
