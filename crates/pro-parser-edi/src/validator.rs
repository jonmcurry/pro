// EDI validation rules for 837P transactions

use crate::types::{EdiSegment, ParsedClaim, Transaction837p};
use pro_common::{Error, Result, DEFAULT_DATE};
use pro_common::validation::*;

/// Validate complete 837P transaction structure
pub fn validate_transaction_structure(segments: &[EdiSegment]) -> Result<()> {
    // Check for required envelope segments
    validate_required_segment(segments, "ISA", "Interchange Control Header")?;
    validate_required_segment(segments, "GS", "Functional Group Header")?;
    validate_required_segment(segments, "ST", "Transaction Set Header")?;
    validate_required_segment(segments, "SE", "Transaction Set Trailer")?;
    validate_required_segment(segments, "GE", "Functional Group Trailer")?;
    validate_required_segment(segments, "IEA", "Interchange Control Trailer")?;

    // Validate segment order
    let isa_pos = find_segment_position(segments, "ISA")?;
    let gs_pos = find_segment_position(segments, "GS")?;
    let st_pos = find_segment_position(segments, "ST")?;

    if isa_pos > gs_pos || gs_pos > st_pos {
        return Err(Error::EdiParse(
            "Segments are not in correct order (ISA -> GS -> ST)".to_string()
        ));
    }

    // Validate ST identifier is 837
    let st = segments.iter().find(|s| s.segment_id == "ST")
        .ok_or_else(|| Error::EdiParse("ST segment not found".to_string()))?;

    if st.get_or_empty(0) != "837" {
        return Err(Error::EdiParse(format!(
            "Invalid transaction set identifier. Expected 837, got {}",
            st.get_or_empty(0)
        )));
    }

    Ok(())
}

/// Validate required segment exists
fn validate_required_segment(segments: &[EdiSegment], segment_id: &str, description: &str) -> Result<()> {
    if !segments.iter().any(|s| s.segment_id == segment_id) {
        return Err(Error::EdiParse(format!("Missing required segment: {} ({})", segment_id, description)));
    }
    Ok(())
}

/// Find segment position in list
fn find_segment_position(segments: &[EdiSegment], segment_id: &str) -> Result<usize> {
    segments.iter()
        .position(|s| s.segment_id == segment_id)
        .ok_or_else(|| Error::EdiParse(format!("Segment {} not found", segment_id)))
}

/// Validate complete parsed transaction
pub fn validate_transaction(transaction: &Transaction837p) -> Result<()> {
    // Validate interchange control number format
    if transaction.interchange_control_header.interchange_control_number.is_empty() {
        return Err(Error::Validation("Interchange control number is required".to_string()));
    }

    // Validate functional group
    if transaction.functional_group_header.functional_identifier_code != "HC" {
        return Err(Error::Validation(format!(
            "Invalid functional identifier code. Expected HC, got {}",
            transaction.functional_group_header.functional_identifier_code
        )));
    }

    // Validate transaction set
    if transaction.transaction_set_header.transaction_set_identifier_code != "837" {
        return Err(Error::Validation(format!(
            "Invalid transaction set identifier. Expected 837, got {}",
            transaction.transaction_set_header.transaction_set_identifier_code
        )));
    }

    // Validate submitter
    validate_submitter(transaction)?;

    // Validate receiver
    validate_receiver(transaction)?;

    // Validate billing provider
    validate_billing_provider(transaction)?;

    // Validate all claims
    for (idx, claim) in transaction.claims.iter().enumerate() {
        validate_claim(claim).map_err(|e| {
            Error::Validation(format!("Claim {} validation failed: {}", idx + 1, e))
        })?;
    }

    Ok(())
}

/// Validate submitter information
fn validate_submitter(transaction: &Transaction837p) -> Result<()> {
    let submitter = &transaction.submitter;

    if submitter.entity_identifier_code != "41" {
        return Err(Error::Validation(format!(
            "Invalid submitter entity identifier. Expected 41, got {}",
            submitter.entity_identifier_code
        )));
    }

    if submitter.entity_type_qualifier == "1" {
        // Person
        if submitter.submitter_last_name.is_none() || submitter.submitter_first_name.is_none() {
            return Err(Error::Validation("Submitter name is required for person entity type".to_string()));
        }
    } else if submitter.entity_type_qualifier == "2" {
        // Organization
        if submitter.submitter_organization_name.is_none() {
            return Err(Error::Validation("Submitter organization name is required".to_string()));
        }
    } else {
        return Err(Error::Validation(format!(
            "Invalid submitter entity type qualifier: {}",
            submitter.entity_type_qualifier
        )));
    }

    Ok(())
}

/// Validate receiver information
fn validate_receiver(transaction: &Transaction837p) -> Result<()> {
    let receiver = &transaction.receiver;

    if receiver.entity_identifier_code != "40" {
        return Err(Error::Validation(format!(
            "Invalid receiver entity identifier. Expected 40, got {}",
            receiver.entity_identifier_code
        )));
    }

    if receiver.receiver_name.is_empty() {
        return Err(Error::Validation("Receiver name is required".to_string()));
    }

    if receiver.identification_code.is_empty() {
        return Err(Error::Validation("Receiver identification code is required".to_string()));
    }

    Ok(())
}

/// Validate billing provider information
fn validate_billing_provider(transaction: &Transaction837p) -> Result<()> {
    let provider = &transaction.billing_provider;

    if provider.npi.is_empty() {
        return Err(Error::Validation("Billing provider NPI is required".to_string()));
    }

    // Validate NPI format
    validate_npi(&provider.npi)?;

    if provider.entity_type_qualifier == "1" {
        // Person
        if provider.last_name.is_none() || provider.first_name.is_none() {
            return Err(Error::Validation("Provider name is required for person entity type".to_string()));
        }
    } else if provider.entity_type_qualifier == "2" {
        // Organization
        if provider.organization_name.is_none() {
            return Err(Error::Validation("Provider organization name is required".to_string()));
        }
    }

    Ok(())
}

/// Validate individual claim
pub fn validate_claim(claim: &ParsedClaim) -> Result<()> {
    // Validate patient control number
    if claim.patient_control_number.is_empty() {
        return Err(Error::Validation("Patient control number is required".to_string()));
    }

    validate_length(&claim.patient_control_number, 1, 38, "Patient control number")?;

    // Validate subscriber information
    if claim.subscriber_last_name.is_empty() {
        return Err(Error::Validation("Subscriber last name is required".to_string()));
    }

    if claim.subscriber_first_name.is_empty() {
        return Err(Error::Validation("Subscriber first name is required".to_string()));
    }

    if claim.subscriber_id.is_empty() {
        return Err(Error::Validation("Subscriber ID is required".to_string()));
    }

    // Validate MBI format if subscriber ID qualifier indicates MBI
    if claim.subscriber_id_code_qualifier == "MI" {
        validate_mbi(&claim.subscriber_id)?;
    }

    // Validate gender code if present
    if let Some(gender) = &claim.subscriber_gender {
        if !["M", "F", "U"].contains(&gender.as_str()) {
            return Err(Error::Validation(format!("Invalid gender code: {}", gender)));
        }
    }

    // Validate payer information
    if claim.payer_name.is_empty() {
        return Err(Error::Validation("Payer name is required".to_string()));
    }

    if claim.payer_id.is_empty() {
        return Err(Error::Validation("Payer ID is required".to_string()));
    }

    // Validate claim amounts
    validate_non_negative(claim.total_claim_charge_amount, "Total claim charge amount")?;

    if claim.total_claim_charge_amount > rust_decimal::Decimal::new(9999999, 2) {
        return Err(Error::Validation(format!(
            "Total claim charge amount exceeds maximum: {}",
            claim.total_claim_charge_amount
        )));
    }

    // Validate place of service if present
    if let Some(pos) = &claim.place_of_service_code {
        validate_pos(pos)?;
    }

    // Validate date of service exists (not the default 1900-01-01)
    if claim.date_of_service_from == *DEFAULT_DATE {
        return Err(Error::Validation("Date of service is required (DTP*472 segment missing or empty)".to_string()));
    }

    // Validate date of service
    if claim.date_of_service_from > chrono::Utc::now().naive_utc().date() {
        return Err(Error::Validation("Date of service cannot be in the future".to_string()));
    }

    // Validate date range
    validate_date_range(claim.date_of_service_from, claim.date_of_service_to)?;

    // Validate diagnosis codes
    if claim.diagnoses.is_empty() {
        return Err(Error::Validation("At least one diagnosis code is required".to_string()));
    }

    if claim.diagnoses.len() > 12 {
        return Err(Error::Validation(format!(
            "Too many diagnosis codes. Maximum 12, got {}",
            claim.diagnoses.len()
        )));
    }

    for (idx, diagnosis) in claim.diagnoses.iter().enumerate() {
        validate_icd10(&diagnosis.diagnosis_code).map_err(|e| {
            Error::Validation(format!("Diagnosis {} validation failed: {}", idx + 1, e))
        })?;
    }

    // Validate service lines
    if claim.service_lines.is_empty() {
        return Err(Error::Validation("At least one service line is required".to_string()));
    }

    for (idx, service_line) in claim.service_lines.iter().enumerate() {
        validate_service_line(service_line).map_err(|e| {
            Error::Validation(format!("Service line {} validation failed: {}", idx + 1, e))
        })?;
    }

    // Validate rendering provider NPI if present
    if let Some(npi) = &claim.rendering_provider_npi {
        validate_npi(npi)?;
    }

    // Validate referring provider NPI if present
    if let Some(npi) = &claim.referring_provider_npi {
        validate_npi(npi)?;
    }

    // Validate service facility NPI if present
    if let Some(npi) = &claim.service_facility_npi {
        validate_npi(npi)?;
    }

    Ok(())
}

/// Validate individual service line
fn validate_service_line(service_line: &crate::types::ServiceLine) -> Result<()> {
    // Validate procedure code
    if service_line.procedure_code.is_empty() {
        return Err(Error::Validation("Procedure code is required".to_string()));
    }

    validate_procedure_code(&service_line.procedure_code)?;

    // Validate modifiers if present
    if let Some(modifier) = &service_line.procedure_modifier_1 {
        validate_modifier(modifier)?;
    }
    if let Some(modifier) = &service_line.procedure_modifier_2 {
        validate_modifier(modifier)?;
    }
    if let Some(modifier) = &service_line.procedure_modifier_3 {
        validate_modifier(modifier)?;
    }
    if let Some(modifier) = &service_line.procedure_modifier_4 {
        validate_modifier(modifier)?;
    }

    // Validate line item charge
    validate_non_negative(service_line.line_item_charge_amount, "Line item charge amount")?;

    // Validate service units
    validate_positive(service_line.service_unit_count, "Service unit count")?;

    if service_line.service_unit_count > rust_decimal::Decimal::new(99999, 1) {
        return Err(Error::Validation(format!(
            "Service unit count exceeds maximum: {}",
            service_line.service_unit_count
        )));
    }

    // Validate service date exists (not the default 1900-01-01)
    if service_line.service_date_from == *DEFAULT_DATE {
        return Err(Error::Validation("Service date is required (DTP*472 segment missing or empty at service line level)".to_string()));
    }

    // Validate service dates
    if service_line.service_date_from > chrono::Utc::now().naive_utc().date() {
        return Err(Error::Validation("Service date cannot be in the future".to_string()));
    }

    validate_date_range(service_line.service_date_from, service_line.service_date_to)?;

    // Validate place of service if present
    if let Some(pos) = &service_line.place_of_service_code {
        validate_pos(pos)?;
    }

    // Validate diagnosis pointers are within valid range (1-12)
    if let Some(ptr) = service_line.diagnosis_code_pointer_1 {
        validate_diagnosis_pointer(ptr)?;
    }
    if let Some(ptr) = service_line.diagnosis_code_pointer_2 {
        validate_diagnosis_pointer(ptr)?;
    }
    if let Some(ptr) = service_line.diagnosis_code_pointer_3 {
        validate_diagnosis_pointer(ptr)?;
    }
    if let Some(ptr) = service_line.diagnosis_code_pointer_4 {
        validate_diagnosis_pointer(ptr)?;
    }

    // Validate rendering provider NPI if present
    if let Some(npi) = &service_line.rendering_provider_npi {
        validate_npi(npi)?;
    }

    // Validate ordering provider NPI if present
    if let Some(npi) = &service_line.ordering_provider_npi {
        validate_npi(npi)?;
    }

    Ok(())
}

/// Validate diagnosis pointer is in range 1-12
fn validate_diagnosis_pointer(pointer: i16) -> Result<()> {
    if !(1..=12).contains(&pointer) {
        return Err(Error::Validation(format!(
            "Diagnosis pointer must be between 1 and 12, got {}",
            pointer
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_diagnosis_pointer() {
        assert!(validate_diagnosis_pointer(1).is_ok());
        assert!(validate_diagnosis_pointer(12).is_ok());
        assert!(validate_diagnosis_pointer(0).is_err());
        assert!(validate_diagnosis_pointer(13).is_err());
    }

    #[test]
    fn test_validate_required_segment() {
        let segments = vec![
            EdiSegment {
                segment_id: "ISA".to_string(),
                elements: vec![],
            },
        ];

        assert!(validate_required_segment(&segments, "ISA", "Test").is_ok());
        assert!(validate_required_segment(&segments, "GS", "Test").is_err());
    }
}
