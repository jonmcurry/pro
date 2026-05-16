//! Encounter Builder Module
//!
//! Builds encounter records from raw claim data.
//! Extracted from ClaimsProcessor as part of god object refactoring.
//!
//! NOTE: This module is scaffolding for future refactoring. Currently unused
//! but retained for planned integration with the claims processing pipeline.

#![allow(dead_code)]

use anyhow::{Context, Result};
use std::collections::HashMap;

/// Builder for constructing encounter records from raw claim fields
pub struct EncounterBuilder;

impl EncounterBuilder {
    /// Extract and validate encounter-level data from raw claim fields
    pub fn extract_encounter_data(
        encounter_fields: &HashMap<String, String>,
    ) -> Result<EncounterData> {
        // Extract required fields
        let patient_control_number = encounter_fields
            .get("patient_control_number")
            .context("Missing patient_control_number")?
            .clone();

        let subscriber_last_name = encounter_fields
            .get("subscriber_last_name")
            .context("Missing subscriber_last_name")?
            .clone();

        let subscriber_first_name = encounter_fields
            .get("subscriber_first_name")
            .context("Missing subscriber_first_name")?
            .clone();

        let date_of_service_from_str = encounter_fields
            .get("date_of_service_from")
            .context("Missing date_of_service_from")?;

        let subscriber_id = encounter_fields
            .get("subscriber_id")
            .context("Missing subscriber_id")?
            .clone();

        // Parse dates
        let date_of_service_from =
            chrono::NaiveDate::parse_from_str(date_of_service_from_str, "%Y-%m-%d")
                .context("Invalid date format for date_of_service_from")?;

        let date_of_service_to = encounter_fields
            .get("date_of_service_to")
            .and_then(|s| chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d").ok());

        let subscriber_birth_date = encounter_fields
            .get("subscriber_birth_date")
            .filter(|s| !s.is_empty())
            .and_then(|s| chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d").ok());

        // Extract optional fields
        let facility_code = encounter_fields
            .get("facility_code")
            .or_else(|| encounter_fields.get("facility_npi"))
            .context("Missing facility_code or facility_npi")?
            .clone();

        let submitter_id = encounter_fields
            .get("submitter_id")
            .cloned()
            .unwrap_or_else(|| facility_code.clone());

        // Coerce SBR01 payer responsibility to P/S (encounter table constraint).
        // See builders::normalize_payer_responsibility_code for the mapping.
        let payer_responsibility_code = super::normalize_payer_responsibility_code(
            encounter_fields.get("payer_responsibility_code").map(|s| s.as_str()).unwrap_or("")
        ).to_string();

        let payer_id = encounter_fields.get("payer_id").cloned();
        let payer_name = encounter_fields.get("payer_name").cloned();
        let place_of_service = encounter_fields.get("place_of_service_code").cloned();
        let medical_record_number = encounter_fields.get("medical_record_number").cloned();

        Ok(EncounterData {
            patient_control_number,
            subscriber_id,
            subscriber_last_name,
            subscriber_first_name,
            subscriber_birth_date,
            date_of_service_from,
            date_of_service_to,
            facility_code,
            submitter_id,
            payer_id,
            payer_name,
            payer_responsibility_code,
            place_of_service,
            medical_record_number,
        })
    }

    /// Extract provider data from encounter fields
    pub fn extract_provider_data(
        encounter_fields: &HashMap<String, String>,
    ) -> ProviderExtractData {
        // Helper closure to get field and filter empty strings
        let get_field = |key: &str| -> Option<String> {
            encounter_fields
                .get(key)
                .filter(|s| !s.is_empty())
                .cloned()
        };

        ProviderExtractData {
            rendering_provider_npi: get_field("rendering_provider_npi"),
            rendering_provider_last_name: get_field("rendering_provider_last_name"),
            rendering_provider_first_name: get_field("rendering_provider_first_name"),
            rendering_provider_taxonomy: get_field("rendering_provider_taxonomy"),

            referring_provider_npi: get_field("referring_provider_npi"),
            referring_provider_last_name: get_field("referring_provider_last_name"),
            referring_provider_first_name: get_field("referring_provider_first_name"),

            supervising_provider_npi: get_field("supervising_provider_npi"),
            supervising_provider_last_name: get_field("supervising_provider_last_name"),
            supervising_provider_first_name: get_field("supervising_provider_first_name"),

            billing_provider_npi: get_field("billing_provider_npi"),
            billing_provider_name: get_field("billing_provider_name"),
        }
    }
}

/// Extracted encounter-level data
#[derive(Debug, Clone)]
pub struct EncounterData {
    pub patient_control_number: String,
    pub subscriber_id: String,
    pub subscriber_last_name: String,
    pub subscriber_first_name: String,
    pub subscriber_birth_date: Option<chrono::NaiveDate>,
    pub date_of_service_from: chrono::NaiveDate,
    pub date_of_service_to: Option<chrono::NaiveDate>,
    pub facility_code: String,
    pub submitter_id: String,
    pub payer_id: Option<String>,
    pub payer_name: Option<String>,
    pub payer_responsibility_code: String,
    pub place_of_service: Option<String>,
    pub medical_record_number: Option<String>,
}

/// Extracted provider data from encounter fields
#[derive(Debug, Clone, Default)]
pub struct ProviderExtractData {
    pub rendering_provider_npi: Option<String>,
    pub rendering_provider_last_name: Option<String>,
    pub rendering_provider_first_name: Option<String>,
    pub rendering_provider_taxonomy: Option<String>,

    pub referring_provider_npi: Option<String>,
    pub referring_provider_last_name: Option<String>,
    pub referring_provider_first_name: Option<String>,

    pub supervising_provider_npi: Option<String>,
    pub supervising_provider_last_name: Option<String>,
    pub supervising_provider_first_name: Option<String>,

    pub billing_provider_npi: Option<String>,
    pub billing_provider_name: Option<String>,
}
