//! Service Line Builder Module
//!
//! Builds service line records from raw claim data.
//! Extracted from ClaimsProcessor as part of god object refactoring.
//!
//! NOTE: This module is scaffolding for future refactoring. Currently unused
//! but retained for planned integration with the claims processing pipeline.

#![allow(dead_code)]

use anyhow::{Context, Result};
use serde_json::Value as JsonValue;
use std::collections::HashMap;

/// Builder for constructing service line records
pub struct ServiceLineBuilder;

impl ServiceLineBuilder {
    /// Extract service line data from raw claim fields
    pub fn extract_service_line_data(
        service_line_fields: &Option<JsonValue>,
        line_number: i16,
    ) -> Result<ServiceLineData> {
        let fields: HashMap<String, String> = match service_line_fields {
            Some(v) => serde_json::from_value(v.clone())?,
            None => return Err(anyhow::anyhow!("No service line fields provided")),
        };

        // Determine the prefix (may be empty or "service_line_")
        let prefix = if fields.contains_key("procedure_code") {
            ""
        } else {
            "service_line_"
        };

        // Extract procedure code (required)
        let procedure_code = fields
            .get(&format!("{}procedure_code", prefix))
            .context("Missing procedure_code")?
            .clone();

        // Extract charge amount
        let charge_amount = fields
            .get(&format!("{}line_item_charge_amount", prefix))
            .and_then(|s| s.parse::<rust_decimal::Decimal>().ok())
            .unwrap_or(rust_decimal::Decimal::ZERO);

        // Extract unit count
        let unit_count = fields
            .get(&format!("{}service_unit_count", prefix))
            .and_then(|s| s.parse::<rust_decimal::Decimal>().ok())
            .unwrap_or(rust_decimal::Decimal::ONE);

        // Extract modifiers
        let modifier_1 = fields.get(&format!("{}procedure_modifier_1", prefix)).cloned();
        let modifier_2 = fields.get(&format!("{}procedure_modifier_2", prefix)).cloned();
        let modifier_3 = fields.get(&format!("{}procedure_modifier_3", prefix)).cloned();
        let modifier_4 = fields.get(&format!("{}procedure_modifier_4", prefix)).cloned();

        // Extract service date
        let service_date_str = fields
            .get(&format!("{}service_date_from", prefix))
            .or_else(|| fields.get("date_of_service_from"))
            .context("Missing service_date_from")?;

        let service_date = chrono::NaiveDate::parse_from_str(service_date_str, "%Y-%m-%d")
            .context("Invalid service date format")?;

        let service_date_to = fields
            .get(&format!("{}service_date_to", prefix))
            .and_then(|s| chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d").ok());

        // Extract place of service
        let place_of_service_code = fields
            .get(&format!("{}place_of_service_code", prefix))
            .cloned();

        // Extract diagnosis pointers
        let pointer_1 = fields
            .get(&format!("{}diagnosis_code_pointer_1", prefix))
            .and_then(|s| s.parse::<i16>().ok());
        let pointer_2 = fields
            .get(&format!("{}diagnosis_code_pointer_2", prefix))
            .and_then(|s| s.parse::<i16>().ok());
        let pointer_3 = fields
            .get(&format!("{}diagnosis_code_pointer_3", prefix))
            .and_then(|s| s.parse::<i16>().ok());
        let pointer_4 = fields
            .get(&format!("{}diagnosis_code_pointer_4", prefix))
            .and_then(|s| s.parse::<i16>().ok());

        // Extract indicators
        let emergency_indicator = fields
            .get(&format!("{}emergency_indicator", prefix))
            .map(|s| s == "Y" || s == "1" || s.to_lowercase() == "true")
            .unwrap_or(false);

        let epsdt_indicator = fields
            .get(&format!("{}epsdt_indicator", prefix))
            .map(|s| s == "Y" || s == "1" || s.to_lowercase() == "true")
            .unwrap_or(false);

        let family_planning_indicator = fields
            .get(&format!("{}family_planning_indicator", prefix))
            .map(|s| s == "Y" || s == "1" || s.to_lowercase() == "true")
            .unwrap_or(false);

        // Extract NDC information
        let ndc_code = fields.get(&format!("{}ndc_code", prefix)).cloned();
        let ndc_unit_count = fields
            .get(&format!("{}ndc_unit_count", prefix))
            .and_then(|s| s.parse::<rust_decimal::Decimal>().ok());
        let ndc_measurement_unit = fields
            .get(&format!("{}ndc_measurement_unit", prefix))
            .cloned();

        // Extract authorization/referral
        let prior_authorization_number = fields
            .get(&format!("{}prior_authorization_number", prefix))
            .cloned();
        let referral_number = fields
            .get(&format!("{}referral_number", prefix))
            .cloned();

        Ok(ServiceLineData {
            line_number,
            procedure_code,
            modifier_1,
            modifier_2,
            modifier_3,
            modifier_4,
            charge_amount,
            unit_count,
            service_date,
            service_date_to,
            place_of_service_code,
            diagnosis_pointer_1: pointer_1,
            diagnosis_pointer_2: pointer_2,
            diagnosis_pointer_3: pointer_3,
            diagnosis_pointer_4: pointer_4,
            emergency_indicator,
            epsdt_indicator,
            family_planning_indicator,
            ndc_code,
            ndc_unit_count,
            ndc_measurement_unit,
            prior_authorization_number,
            referral_number,
        })
    }

    /// Calculate total charge from multiple service lines
    pub fn calculate_total_charge(service_lines: &[ServiceLineData]) -> rust_decimal::Decimal {
        service_lines
            .iter()
            .fold(rust_decimal::Decimal::ZERO, |acc, line| {
                acc + line.charge_amount
            })
    }
}

/// Extracted service line data
#[derive(Debug, Clone)]
pub struct ServiceLineData {
    pub line_number: i16,
    pub procedure_code: String,
    pub modifier_1: Option<String>,
    pub modifier_2: Option<String>,
    pub modifier_3: Option<String>,
    pub modifier_4: Option<String>,
    pub charge_amount: rust_decimal::Decimal,
    pub unit_count: rust_decimal::Decimal,
    pub service_date: chrono::NaiveDate,
    pub service_date_to: Option<chrono::NaiveDate>,
    pub place_of_service_code: Option<String>,
    pub diagnosis_pointer_1: Option<i16>,
    pub diagnosis_pointer_2: Option<i16>,
    pub diagnosis_pointer_3: Option<i16>,
    pub diagnosis_pointer_4: Option<i16>,
    pub emergency_indicator: bool,
    pub epsdt_indicator: bool,
    pub family_planning_indicator: bool,
    pub ndc_code: Option<String>,
    pub ndc_unit_count: Option<rust_decimal::Decimal>,
    pub ndc_measurement_unit: Option<String>,
    pub prior_authorization_number: Option<String>,
    pub referral_number: Option<String>,
}
