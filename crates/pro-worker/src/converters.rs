//! Converters module - Data conversion functions extracted from IngestionPipeline
//!
//! This module contains all data conversion logic for transforming:
//! - CSV rows to ParsedClaim
//! - ParsedClaim to Encounter
//! - Parsed service lines to ServiceLine models
//! - Diagnosis extraction from CSV

use chrono::{NaiveDate, Utc};
use pro_common::{Error, Result};
use pro_db::models::{Encounter, ServiceLine};
use pro_parser_csv::parser::ParsedRow;
use pro_parser_edi::types::{DiagnosisCode, ParsedClaim, ServiceLine as ParsedServiceLine};
use rust_decimal::Decimal;
use std::str::FromStr;

/// Convert a CSV row to a ParsedClaim structure
pub fn convert_csv_to_claim(csv_row: &ParsedRow) -> Result<ParsedClaim> {
    // Helper function to get required field from encounter_fields
    let get_required = |field: &str| -> Result<String> {
        csv_row
            .encounter_fields
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
        subscriber_entity_type: "1".to_string(),        // Person
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
        payer_entity_type: "2".to_string(),        // Non-Person Entity
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
        diagnoses: extract_diagnoses_from_csv(csv_row)?,

        // Service lines - extract from service_line_fields
        service_lines: extract_service_lines_from_csv(csv_row)?,

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
        patient_responsibility_amount: None,
        service_authorization_code: None,
        claim_number: None,
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
        other_insurance: Vec::new(), // Full COB support
        patient_signature_code: None,

        // Patient fields (when different from subscriber)
        patient_entity_identifier: None,
        patient_entity_type: None,
        patient_last_name: None,
        patient_first_name: None,
        patient_middle_name: None,
        patient_name_suffix: None,
        patient_date_of_birth: None,
        patient_gender: None,
        patient_address_line1: None,
        patient_address_line2: None,
        patient_city: None,
        patient_state: None,
        patient_postal_code: None,
        patient_country: None,
        patient_relationship_code: None,

        related_causes_code_1: None,
        related_causes_code_2: None,
        related_causes_code_3: None,
        auto_accident_state: None,
        auto_accident_country: None,
        ambulance_transport_reason_code: None,
        ambulance_transport_distance: None,
        ambulance_patient_weight: None,
        ambulance_patient_count: None,
        paperwork_report_type: None,
        paperwork_transmission_code: None,
        paperwork_control_number: None,
        condition_codes: Vec::new(),
    };

    Ok(claim)
}

/// Extract diagnosis codes from CSV row
pub fn extract_diagnoses_from_csv(csv_row: &ParsedRow) -> Result<Vec<DiagnosisCode>> {
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
pub fn extract_service_lines_from_csv(csv_row: &ParsedRow) -> Result<Vec<ParsedServiceLine>> {
    // CSV typically has one service line per row
    // Service line fields are in service_line_fields HashMap

    let get_required = |field: &str| -> Result<String> {
        csv_row
            .service_line_fields
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

    let service_line = ParsedServiceLine {
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
        service_unit_count: parse_decimal("service_unit_count").unwrap_or(Decimal::from(1)), // Default to 1 unit

        // Dates
        service_date_from: parse_date("service_date_from")?
            .or_else(|| {
                csv_row
                    .encounter_fields
                    .get("date_of_service_from")
                    .and_then(|s| NaiveDate::parse_from_str(s, "%Y-%m-%d").ok())
            })
            .ok_or_else(|| Error::Parse("Service date required".to_string()))?,
        service_date_to: parse_date("service_date_to")?,

        // Place of service
        place_of_service_code: get_optional("place_of_service_code"),

        // Indicators (not typically in CSV)
        emergency_indicator: None,
        epsdt_indicator: None,
        family_planning_indicator: None,

        // Diagnosis pointers (1-based indices)
        diagnosis_code_pointer_1: get_optional("diagnosis_code_pointer_1").and_then(|s| s.parse::<i16>().ok()),
        diagnosis_code_pointer_2: get_optional("diagnosis_code_pointer_2").and_then(|s| s.parse::<i16>().ok()),
        diagnosis_code_pointer_3: get_optional("diagnosis_code_pointer_3").and_then(|s| s.parse::<i16>().ok()),
        diagnosis_code_pointer_4: get_optional("diagnosis_code_pointer_4").and_then(|s| s.parse::<i16>().ok()),

        // Provider NPIs at line level
        rendering_provider_npi: get_optional("rendering_provider_npi"),
        rendering_provider_last_name: None,
        rendering_provider_first_name: None,
        supervising_provider_npi: get_optional("supervising_provider_npi"),
        ordering_provider_npi: get_optional("ordering_provider_npi"),
        ordering_provider_last_name: None,
        ordering_provider_first_name: None,
        referring_provider_npi: get_optional("referring_provider_npi"),
        referring_provider_last_name: None,
        referring_provider_first_name: None,

        // NDC information
        ndc_code: get_optional("ndc_code"),
        ndc_unit_count: get_optional("ndc_unit_count").and_then(|s| Decimal::from_str(&s).ok()),
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
        line_adjudications: Vec::new(), // Full line adjudication support

        // HCP - Health Care Pricing
        allowed_amount: None,
        saving_amount: None,
    };

    Ok(vec![service_line])
}

/// Convert a ParsedClaim to an Encounter domain model
pub fn convert_claim_to_encounter(claim: &ParsedClaim, organization_id: i64) -> Result<Encounter> {
    // For now, use hardcoded values for facility and region
    // In production, these would be looked up from the database
    let facility_id = 0i64;
    let region_id = None;

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
            chrono::NaiveDate::from_ymd_opt(1900, 1, 1)
                .expect("Default date 1900-01-01 is always valid")
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
pub fn convert_service_line(
    parsed_line: &ParsedServiceLine,
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_extract_diagnoses_empty_returns_error() {
        let row = ParsedRow {
            row_number: 1,
            encounter_fields: HashMap::new(),
            diagnosis_fields: HashMap::new(),
            service_line_fields: HashMap::new(),
        };

        let result = extract_diagnoses_from_csv(&row);
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_diagnoses_with_codes() {
        let mut diagnosis_fields = HashMap::new();
        diagnosis_fields.insert("diagnosis_code_1".to_string(), vec!["J06.9".to_string()]);
        diagnosis_fields.insert("diagnosis_code_2".to_string(), vec!["R05.9".to_string()]);

        let row = ParsedRow {
            row_number: 1,
            encounter_fields: HashMap::new(),
            diagnosis_fields,
            service_line_fields: HashMap::new(),
        };

        let result = extract_diagnoses_from_csv(&row).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].diagnosis_code, "J06.9");
        assert!(result[0].is_principal);
        assert_eq!(result[1].diagnosis_code, "R05.9");
        assert!(!result[1].is_principal);
    }
}
