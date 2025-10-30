// EDI loop parsers for hierarchical structures (1000A/B, 2000A/B, 2300, 2400, etc.)

use crate::segments::*;
use crate::types::*;
use pro_common::{Error, Result};
use uuid::Uuid;

/// Parse Loop 1000A - Submitter Name
pub fn parse_submitter(segments: &[EdiSegment]) -> Result<Submitter> {
    let mut submitter = Submitter {
        entity_identifier_code: String::new(),
        entity_type_qualifier: String::new(),
        submitter_last_name: None,
        submitter_first_name: None,
        submitter_organization_name: None,
        identification_code_qualifier: None,
        identification_code: None,
        contact_name: None,
        contact_phone: None,
        contact_phone_extension: None,
        contact_fax: None,
        contact_email: None,
    };

    for segment in segments {
        match segment.segment_id.as_str() {
            "NM1" => {
                let nm1 = Nm1Segment::parse(segment)?;
                submitter.entity_identifier_code = nm1.entity_identifier_code;
                let entity_type = nm1.entity_type_qualifier.clone();
                submitter.entity_type_qualifier = nm1.entity_type_qualifier;

                if entity_type == "1" {
                    // Person
                    submitter.submitter_last_name = nm1.last_name_or_org;
                    submitter.submitter_first_name = nm1.first_name;
                } else {
                    // Organization
                    submitter.submitter_organization_name = nm1.last_name_or_org;
                }

                submitter.identification_code_qualifier = nm1.identification_code_qualifier;
                submitter.identification_code = nm1.identification_code;
            }
            "PER" => {
                let per = PerSegment::parse(segment)?;
                submitter.contact_name = per.contact_name;

                // Parse communication numbers based on qualifiers
                if let Some(qual) = &per.communication_number_qualifier_1 {
                    match qual.as_str() {
                        "TE" => submitter.contact_phone = per.communication_number_1.clone(),
                        "FX" => submitter.contact_fax = per.communication_number_1.clone(),
                        "EM" => submitter.contact_email = per.communication_number_1.clone(),
                        _ => {}
                    }
                }

                if let Some(qual) = &per.communication_number_qualifier_2 {
                    match qual.as_str() {
                        "TE" => submitter.contact_phone = per.communication_number_2.clone(),
                        "FX" => submitter.contact_fax = per.communication_number_2.clone(),
                        "EM" => submitter.contact_email = per.communication_number_2.clone(),
                        "EX" => submitter.contact_phone_extension = per.communication_number_2.clone(),
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }

    Ok(submitter)
}

/// Parse Loop 1000B - Receiver Name
pub fn parse_receiver(segments: &[EdiSegment]) -> Result<Receiver> {
    let mut receiver = Receiver {
        entity_identifier_code: String::new(),
        entity_type_qualifier: String::new(),
        receiver_name: String::new(),
        identification_code_qualifier: String::new(),
        identification_code: String::new(),
    };

    for segment in segments {
        if segment.segment_id == "NM1" {
            let nm1 = Nm1Segment::parse(segment)?;
            receiver.entity_identifier_code = nm1.entity_identifier_code;
            receiver.entity_type_qualifier = nm1.entity_type_qualifier;
            receiver.receiver_name = nm1.last_name_or_org.unwrap_or_default();
            receiver.identification_code_qualifier = nm1.identification_code_qualifier.unwrap_or_default();
            receiver.identification_code = nm1.identification_code.unwrap_or_default();
        }
    }

    Ok(receiver)
}

/// Parse Loop 2000A - Billing Provider Hierarchical Level
pub fn parse_billing_provider(segments: &[EdiSegment]) -> Result<BillingProvider> {
    let mut provider = BillingProvider {
        hierarchical_id_number: String::new(),
        entity_identifier_code: String::new(),
        entity_type_qualifier: String::new(),
        organization_name: None,
        last_name: None,
        first_name: None,
        npi: String::new(),
        tax_id_type: None,
        tax_id: None,
        address_line1: None,
        address_line2: None,
        city: None,
        state: None,
        postal_code: None,
        country_code: None,
        contact_name: None,
        contact_phone: None,
    };

    for segment in segments {
        match segment.segment_id.as_str() {
            "HL" => {
                let hl = HlSegment::parse(segment)?;
                provider.hierarchical_id_number = hl.hierarchical_id_number;
            }
            "NM1" => {
                let nm1 = Nm1Segment::parse(segment)?;
                provider.entity_identifier_code = nm1.entity_identifier_code;
                let entity_type = nm1.entity_type_qualifier.clone();
                provider.entity_type_qualifier = nm1.entity_type_qualifier;

                if entity_type == "1" {
                    provider.last_name = nm1.last_name_or_org;
                    provider.first_name = nm1.first_name;
                } else {
                    provider.organization_name = nm1.last_name_or_org;
                }

                if nm1.identification_code_qualifier.as_deref() == Some("XX") {
                    provider.npi = nm1.identification_code.unwrap_or_default();
                }
            }
            "N3" => {
                let n3 = N3Segment::parse(segment)?;
                provider.address_line1 = Some(n3.address_line1);
                provider.address_line2 = n3.address_line2;
            }
            "N4" => {
                let n4 = N4Segment::parse(segment)?;
                provider.city = Some(n4.city);
                provider.state = n4.state_code;
                provider.postal_code = n4.postal_code;
                provider.country_code = n4.country_code;
            }
            "REF" => {
                let ref_seg = RefSegment::parse(segment)?;
                if ref_seg.reference_identification_qualifier == "EI" {
                    // Tax ID
                    provider.tax_id_type = Some("EI".to_string());
                    provider.tax_id = ref_seg.reference_identification;
                }
            }
            "PER" => {
                let per = PerSegment::parse(segment)?;
                provider.contact_name = per.contact_name;
                if per.communication_number_qualifier_1.as_deref() == Some("TE") {
                    provider.contact_phone = per.communication_number_1;
                }
            }
            _ => {}
        }
    }

    Ok(provider)
}

/// Parse Loop 2300 - Claim Information
pub fn parse_claim_info(segments: &[EdiSegment]) -> Result<ParsedClaim> {
    let mut claim = ParsedClaim {
        temp_id: Uuid::new_v4(),

        // Initialize all required fields with defaults
        subscriber_hl_number: String::new(),
        subscriber_relationship_code: String::new(),
        subscriber_entity_identifier: String::new(),
        subscriber_entity_type: String::new(),
        subscriber_last_name: String::new(),
        subscriber_first_name: String::new(),
        subscriber_middle_name: None,
        subscriber_name_suffix: None,
        subscriber_id_code_qualifier: String::new(),
        subscriber_id: String::new(),
        subscriber_date_of_birth: None,
        subscriber_gender: None,
        subscriber_address_line1: None,
        subscriber_address_line2: None,
        subscriber_city: None,
        subscriber_state: None,
        subscriber_postal_code: None,
        subscriber_country: None,
        medical_record_number: None,

        payer_entity_identifier: String::new(),
        payer_entity_type: String::new(),
        payer_name: String::new(),
        payer_id_qualifier: String::new(),
        payer_id: String::new(),
        payer_address_line1: None,
        payer_address_line2: None,
        payer_city: None,
        payer_state: None,
        payer_postal_code: None,

        patient_control_number: String::new(),
        total_claim_charge_amount: rust_decimal::Decimal::ZERO,
        place_of_service_code: None,
        claim_frequency_code: None,
        provider_signature_indicator: None,
        assignment_indicator: None,
        benefits_assignment_indicator: None,
        release_of_information_code: None,
        patient_signature_code: None,

        related_causes_code_1: None,
        related_causes_code_2: None,
        related_causes_code_3: None,
        auto_accident_state: None,
        auto_accident_country: None,

        date_of_service_from: chrono::NaiveDate::from_ymd_opt(1900, 1, 1).unwrap(),
        date_of_service_to: None,
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
        referring_provider_npi: None,
        referring_provider_last_name: None,
        referring_provider_first_name: None,

        rendering_provider_qualifier: None,
        rendering_provider_npi: None,
        rendering_provider_last_name: None,
        rendering_provider_first_name: None,
        rendering_provider_taxonomy: None,

        service_facility_qualifier: None,
        service_facility_npi: None,
        service_facility_name: None,
        service_facility_address_line1: None,
        service_facility_address_line2: None,
        service_facility_city: None,
        service_facility_state: None,
        service_facility_postal_code: None,

        supervising_provider_qualifier: None,
        supervising_provider_npi: None,
        supervising_provider_last_name: None,
        supervising_provider_first_name: None,

        other_payer_paid_amount: None,
        other_payer_id: None,
        other_payer_name: None,
        other_payer_claim_number: None,

        service_lines: Vec::new(),
        diagnoses: Vec::new(),
    };

    for segment in segments {
        match segment.segment_id.as_str() {
            "CLM" => {
                let clm = ClmSegment::parse(segment)?;
                claim.patient_control_number = clm.patient_control_number;
                claim.total_claim_charge_amount = clm.total_claim_charge_amount;
                claim.place_of_service_code = clm.place_of_service_code;
                claim.claim_frequency_code = clm.claim_frequency_code;
                claim.provider_signature_indicator = clm.provider_signature_indicator;
                claim.assignment_indicator = clm.assignment_indicator;
                claim.benefits_assignment_indicator = clm.benefits_assignment_indicator;
                claim.release_of_information_code = clm.release_information_code;
            }
            "DTP" => {
                let dtp = DtpSegment::parse(segment)?;
                match dtp.date_time_qualifier.as_str() {
                    "472" => claim.date_of_service_from = dtp.parse_date()?,
                    "433" => claim.onset_of_illness_date = Some(dtp.parse_date()?),
                    "454" => claim.initial_treatment_date = Some(dtp.parse_date()?),
                    "304" => claim.last_seen_date = Some(dtp.parse_date()?),
                    "453" => claim.acute_manifestation_date = Some(dtp.parse_date()?),
                    "439" => claim.accident_date = Some(dtp.parse_date()?),
                    "484" => claim.last_menstrual_period_date = Some(dtp.parse_date()?),
                    "455" => claim.last_xray_date = Some(dtp.parse_date()?),
                    "360" => claim.disability_from_date = Some(dtp.parse_date()?),
                    "361" => claim.disability_to_date = Some(dtp.parse_date()?),
                    "297" => claim.last_worked_date = Some(dtp.parse_date()?),
                    "296" => claim.authorized_return_to_work_date = Some(dtp.parse_date()?),
                    "435" => claim.admission_date = Some(dtp.parse_date()?),
                    "096" => claim.discharge_date = Some(dtp.parse_date()?),
                    _ => {}
                }
            }
            "HI" => {
                let hi = HiSegment::parse(segment)?;
                for (idx, (qualifier, code)) in hi.diagnoses.iter().enumerate() {
                    claim.diagnoses.push(DiagnosisCode {
                        sequence_number: (idx + 1) as i16,
                        diagnosis_code_qualifier: qualifier.clone(),
                        diagnosis_code: code.clone(),
                        is_principal: idx == 0,
                    });
                }
            }
            "REF" => {
                let ref_seg = RefSegment::parse(segment)?;
                match ref_seg.reference_identification_qualifier.as_str() {
                    "EA" => claim.medical_record_number = ref_seg.reference_identification,
                    _ => {}
                }
            }
            _ => {}
        }
    }

    Ok(claim)
}

/// Parse Loop 2400 - Service Line
pub fn parse_service_line(segments: &[EdiSegment], line_number: i16) -> Result<ServiceLine> {
    let mut service_line = ServiceLine {
        line_number,
        product_service_id_qualifier: String::new(),
        procedure_code: String::new(),
        procedure_modifier_1: None,
        procedure_modifier_2: None,
        procedure_modifier_3: None,
        procedure_modifier_4: None,
        line_item_charge_amount: rust_decimal::Decimal::ZERO,
        unit_basis_measurement_code: String::new(),
        service_unit_count: rust_decimal::Decimal::ZERO,
        service_date_from: chrono::NaiveDate::from_ymd_opt(1900, 1, 1).unwrap(),
        service_date_to: None,
        place_of_service_code: None,
        emergency_indicator: None,
        epsdt_indicator: None,
        family_planning_indicator: None,
        diagnosis_code_pointer_1: None,
        diagnosis_code_pointer_2: None,
        diagnosis_code_pointer_3: None,
        diagnosis_code_pointer_4: None,
        rendering_provider_npi: None,
        rendering_provider_last_name: None,
        rendering_provider_first_name: None,
        supervising_provider_npi: None,
        ordering_provider_npi: None,
        ordering_provider_last_name: None,
        ordering_provider_first_name: None,
        referring_provider_npi: None,
        ndc_code: None,
        ndc_unit_count: None,
        ndc_measurement_unit: None,
        prior_authorization_number: None,
        referral_number: None,
        line_note: None,
        revenue_code: None,
        other_payer_line_paid_amount: None,
    };

    for segment in segments {
        match segment.segment_id.as_str() {
            "SV1" => {
                let sv1 = Sv1Segment::parse(segment)?;
                service_line.product_service_id_qualifier = sv1.product_service_id_qualifier;
                service_line.procedure_code = sv1.procedure_code;
                service_line.procedure_modifier_1 = sv1.procedure_modifier_1;
                service_line.procedure_modifier_2 = sv1.procedure_modifier_2;
                service_line.procedure_modifier_3 = sv1.procedure_modifier_3;
                service_line.procedure_modifier_4 = sv1.procedure_modifier_4;
                service_line.line_item_charge_amount = sv1.line_item_charge_amount;
                service_line.unit_basis_measurement_code = sv1.unit_basis_measurement_code;
                service_line.service_unit_count = sv1.service_unit_count;
                service_line.place_of_service_code = sv1.place_of_service_code;

                // Map diagnosis pointers
                if let Some(ptr) = sv1.diagnosis_code_pointer.get(0) {
                    service_line.diagnosis_code_pointer_1 = Some(*ptr);
                }
                if let Some(ptr) = sv1.diagnosis_code_pointer.get(1) {
                    service_line.diagnosis_code_pointer_2 = Some(*ptr);
                }
                if let Some(ptr) = sv1.diagnosis_code_pointer.get(2) {
                    service_line.diagnosis_code_pointer_3 = Some(*ptr);
                }
                if let Some(ptr) = sv1.diagnosis_code_pointer.get(3) {
                    service_line.diagnosis_code_pointer_4 = Some(*ptr);
                }
            }
            "DTP" => {
                let dtp = DtpSegment::parse(segment)?;
                if dtp.date_time_qualifier == "472" {
                    if dtp.date_time_period_format_qualifier == "D8" {
                        service_line.service_date_from = dtp.parse_date()?;
                    } else if dtp.date_time_period_format_qualifier == "RD8" {
                        let (from, to) = dtp.parse_date_range()?;
                        service_line.service_date_from = from;
                        service_line.service_date_to = Some(to);
                    }
                }
            }
            "REF" => {
                let ref_seg = RefSegment::parse(segment)?;
                match ref_seg.reference_identification_qualifier.as_str() {
                    "G1" => service_line.prior_authorization_number = ref_seg.reference_identification,
                    "9F" => service_line.referral_number = ref_seg.reference_identification,
                    _ => {}
                }
            }
            _ => {}
        }
    }

    Ok(service_line)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_submitter() {
        let segments = vec![
            EdiSegment {
                segment_id: "NM1".to_string(),
                elements: vec![
                    "41".to_string(),
                    "2".to_string(),
                    "SUBMITTER ORG".to_string(),
                    "".to_string(),
                    "".to_string(),
                    "".to_string(),
                    "".to_string(),
                    "46".to_string(),
                    "12345".to_string(),
                ],
            },
        ];

        let submitter = parse_submitter(&segments).unwrap();
        assert_eq!(submitter.entity_identifier_code, "41");
        assert_eq!(submitter.submitter_organization_name, Some("SUBMITTER ORG".to_string()));
    }
}
