// EDI loop parsers for hierarchical structures (1000A/B, 2000A/B, 2300, 2400, etc.)

use crate::segments::*;
use crate::types::*;
use pro_common::{Error, Result, DEFAULT_DATE};

/// Helper function to write debug output to file
/// Falls back to using tracing::info! if file write fails
/// DISABLED: Debug logging removed for production performance
#[allow(unused_variables)]
fn debug_log(message: &str) {
    // Debug logging disabled for production performance
    // This function is a no-op but kept for code compatibility
}

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
        facility_id: None,
        provider_number: None,
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
                // Log RAW segment elements BEFORE parsing
                debug_log(&format!("RAW NM1 segment - elements count: {}, elements: {:?}",
                    segment.elements.len(),
                    segment.elements
                ));

                let nm1 = Nm1Segment::parse(segment)?;

                // Log PARSED NM1 segment
                debug_log(&format!("PARSED entity_id='{}', entity_type='{}', org/name={:?}, qualifier={:?}, code={:?}",
                    nm1.entity_identifier_code,
                    nm1.entity_type_qualifier,
                    nm1.last_name_or_org,
                    nm1.identification_code_qualifier,
                    nm1.identification_code
                ));

                // ONLY process NM1*85 (Billing Provider) - ignore other NM1 segments in this loop
                if nm1.entity_identifier_code == "85" {
                    debug_log("[NM1_DEBUG] Found NM1*85! Setting billing provider values");
                    provider.entity_identifier_code = nm1.entity_identifier_code.clone();
                    let entity_type = nm1.entity_type_qualifier.clone();
                    provider.entity_type_qualifier = nm1.entity_type_qualifier;

                    if entity_type == "1" {
                        provider.last_name = nm1.last_name_or_org;
                        provider.first_name = nm1.first_name;
                    } else {
                        provider.organization_name = nm1.last_name_or_org;
                    }

                    // Extract NPI (qualifier XX) or Facility ID (qualifier 46)
                    match nm1.identification_code_qualifier.as_deref() {
                        Some("XX") => {
                            let npi_value = nm1.identification_code.clone().unwrap_or_default();
                            debug_log(&format!("[NM1_DEBUG] Setting provider.npi = '{}'", npi_value));
                            provider.npi = npi_value;
                        }
                        Some("46") => {
                            debug_log(&format!("[NM1_DEBUG] Setting provider.facility_id = {:?}", nm1.identification_code));
                            provider.facility_id = nm1.identification_code;
                        }
                        _ => {
                            debug_log("[NM1_DEBUG] No XX or 46 qualifier found");
                        }
                    }
                } else {
                    debug_log(&format!("[NM1_DEBUG] Skipping NM1*{} (not 85)", nm1.entity_identifier_code));
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
                match ref_seg.reference_identification_qualifier.as_str() {
                    "EI" => {
                        // Tax ID
                        provider.tax_id_type = Some("EI".to_string());
                        provider.tax_id = ref_seg.reference_identification;
                    }
                    "G2" | "1C" | "1J" => {
                        // Provider Commercial Number / Facility ID
                        // G2 = Provider Commercial Number
                        // 1C = Medicare Provider Number
                        // 1J = Facility ID Number
                        provider.provider_number = ref_seg.reference_identification;
                    }
                    _ => {}
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
    use crate::types::{OtherInsurance, ClaimAdjustment, LineAdjudication};

    let mut claim = ParsedClaim {
        temp_id: 0, // Temporary ID, database will generate actual ID

        // Initialize all required fields with defaults
        subscriber_hl_number: String::new(),
        subscriber_relationship_code: String::new(),
        payer_responsibility_code: String::new(), // SBR01 - P=Primary, S=Secondary
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

        // Patient fields (when different from subscriber - Loop 2000C/2010CA)
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

        // SBR segment (Loop 2000B)
        claim_filing_indicator_code: None, // SBR09 - MB=Medicare B, MC=Medicaid, etc.

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

        date_of_service_from: *DEFAULT_DATE,
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

        billing_date: None, // Set by parser from BHT04 segment

        delay_reason_code: None,
        special_program_code: None,
        patient_amount_paid: None,
        patient_responsibility_amount: None,
        service_authorization_code: None,
        claim_number: None,
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

        // Legacy COB fields (for backwards compatibility)
        other_payer_paid_amount: None,
        other_payer_id: None,
        other_payer_name: None,
        other_payer_claim_number: None,

        // Full COB support
        other_insurance: Vec::new(),

        ambulance_transport_reason_code: None,
        ambulance_transport_distance: None,
        ambulance_patient_weight: None,
        ambulance_patient_count: None,

        paperwork_report_type: None,
        paperwork_transmission_code: None,
        paperwork_control_number: None,

        condition_codes: Vec::new(),

        service_lines: Vec::new(),
        diagnoses: Vec::new(),
    };

    let mut last_nm1_entity: Option<String> = None;
    let mut current_service_line: Option<ServiceLine> = None;

    // COB tracking: true when we're in Loop 2320 (Other Subscriber Information)
    let mut in_cob_loop = false;
    let mut current_other_insurance: Option<OtherInsurance> = None;

    // Patient tracking: true when patient is different from subscriber (Loop 2000C)
    let mut in_patient_loop = false;
    let mut primary_payer_captured = false; // Track if we've captured the primary payer

    // Line adjudication tracking (Loop 2430)
    let mut current_line_adjudication: Option<LineAdjudication> = None;

    for segment in segments {
        match segment.segment_id.as_str() {
            "CLM" => {
                debug_log(&format!("[CLM_PARSE] segment has {} elements", segment.elements.len()));
                debug_log(&format!("[CLM_PARSE] element[9]={:?}, element[10]={:?}, element[19]={:?}",
                    segment.get(9), segment.get(10), segment.get(19)));
                let clm = ClmSegment::parse(segment)?;
                debug_log(&format!("[CLM_PARSE] parsed: delay_reason={:?}, special_program={:?}, accident_state={:?}",
                    clm.delay_reason_code, clm.special_program_code, clm.accident_state));
                claim.patient_control_number = clm.patient_control_number;
                claim.total_claim_charge_amount = clm.total_claim_charge_amount;
                claim.place_of_service_code = clm.place_of_service_code;
                claim.claim_frequency_code = clm.claim_frequency_code;
                claim.provider_signature_indicator = clm.provider_signature_indicator;
                claim.assignment_indicator = clm.assignment_indicator;
                claim.benefits_assignment_indicator = clm.benefits_assignment_indicator;
                claim.release_of_information_code = clm.release_information_code;
                claim.delay_reason_code = clm.delay_reason_code;
                claim.special_program_code = clm.special_program_code;
                claim.auto_accident_state = clm.accident_state;
            }
            "NM1" => {
                let nm1 = Nm1Segment::parse(segment)?;

                // If we're inside a service line (Loop 2400), handle at service line level
                // Otherwise, handle at claim level (Loop 2300/2310)
                if let Some(ref mut line) = current_service_line {
                    // Service line level NM1 segments (Loop 2420)
                    debug_log(&format!(
                        "[SERVICE_LINE_NM1] entity_id='{}', entity_type='{}', name={:?}, qualifier={:?}, npi={:?}",
                        nm1.entity_identifier_code,
                        nm1.entity_type_qualifier,
                        nm1.last_name_or_org,
                        nm1.identification_code_qualifier,
                        nm1.identification_code
                    ));

                    match nm1.entity_identifier_code.as_str() {
                        "82" => {
                            // Loop 2420A - Rendering Provider at service line level
                            if nm1.identification_code_qualifier.as_deref() == Some("XX") {
                                line.rendering_provider_npi = nm1.identification_code.clone();
                                debug_log(&format!("[SERVICE_LINE] Set rendering_provider_npi = {:?}", line.rendering_provider_npi));
                            }
                            line.rendering_provider_last_name = nm1.last_name_or_org.clone();
                            line.rendering_provider_first_name = nm1.first_name.clone();
                        }
                        "DK" => {
                            // Loop 2420E - Ordering Provider at service line level
                            if nm1.identification_code_qualifier.as_deref() == Some("XX") {
                                line.ordering_provider_npi = nm1.identification_code.clone();
                                debug_log(&format!("[SERVICE_LINE] Set ordering_provider_npi = {:?}", line.ordering_provider_npi));
                            }
                            line.ordering_provider_last_name = nm1.last_name_or_org.clone();
                            line.ordering_provider_first_name = nm1.first_name.clone();
                        }
                        "DQ" => {
                            // Loop 2420D - Supervising Provider at service line level
                            if nm1.identification_code_qualifier.as_deref() == Some("XX") {
                                line.supervising_provider_npi = nm1.identification_code.clone();
                                debug_log(&format!("[SERVICE_LINE] Set supervising_provider_npi = {:?}", line.supervising_provider_npi));
                            }
                        }
                        "DN" => {
                            // Loop 2420F - Referring Provider at service line level
                            if nm1.identification_code_qualifier.as_deref() == Some("XX") {
                                line.referring_provider_npi = nm1.identification_code.clone();
                                debug_log(&format!("[SERVICE_LINE] Set referring_provider_npi = {:?}", line.referring_provider_npi));
                            }
                            line.referring_provider_last_name = nm1.last_name_or_org.clone();
                            line.referring_provider_first_name = nm1.first_name.clone();
                        }
                        _ => {
                            // Ignore other entity identifiers at service line level
                        }
                    }
                } else if in_cob_loop {
                    // COB Loop NM1 segments (Loop 2330A/2330B)
                    // These go to current_other_insurance, NOT to claim primary fields
                    debug_log(&format!(
                        "[COB_NM1] entity_id='{}', entity_type='{}', name={:?}, qualifier={:?}, id={:?}",
                        nm1.entity_identifier_code,
                        nm1.entity_type_qualifier,
                        nm1.last_name_or_org,
                        nm1.identification_code_qualifier,
                        nm1.identification_code
                    ));

                    last_nm1_entity = Some(format!("COB_{}", nm1.entity_identifier_code));

                    if let Some(ref mut other_ins) = current_other_insurance {
                        match nm1.entity_identifier_code.as_str() {
                            "IL" => {
                                // Loop 2330A - Other Subscriber Name
                                other_ins.other_subscriber_last_name = nm1.last_name_or_org.clone();
                                other_ins.other_subscriber_first_name = nm1.first_name.clone();
                                other_ins.other_subscriber_middle_name = nm1.middle_name.clone();
                                other_ins.other_subscriber_name_suffix = nm1.name_suffix.clone();
                                other_ins.other_subscriber_id_qualifier = nm1.identification_code_qualifier.clone();
                                other_ins.other_subscriber_id = nm1.identification_code.clone();
                            }
                            "PR" => {
                                // Loop 2330B - Other Payer Name
                                other_ins.payer_name = nm1.last_name_or_org.clone();
                                other_ins.payer_id = nm1.identification_code.clone();

                                // Also populate legacy fields for backwards compatibility
                                // (only populate if primary payer not set - first PR after CLM is primary)
                                if !claim.other_payer_id.is_some() {
                                    claim.other_payer_id = nm1.identification_code.clone();
                                    claim.other_payer_name = nm1.last_name_or_org.clone();
                                }
                            }
                            _ => {}
                        }
                    }
                } else {
                    // Claim level NM1 segments (Loop 2010BA/2010BB/2010CA/2310)
                    debug_log(&format!(
                        "[CLAIM_NM1] entity_id='{}', entity_type='{}', name={:?}, qualifier={:?}, npi={:?}, in_patient_loop={}",
                        nm1.entity_identifier_code,
                        nm1.entity_type_qualifier,
                        nm1.last_name_or_org,
                        nm1.identification_code_qualifier,
                        nm1.identification_code,
                        in_patient_loop
                    ));

                    last_nm1_entity = Some(nm1.entity_identifier_code.clone());

                    match nm1.entity_identifier_code.as_str() {
                        "IL" => {
                            // Loop 2010BA - Subscriber Name (only if NOT in COB)
                            claim.subscriber_entity_identifier = nm1.entity_identifier_code;
                            claim.subscriber_entity_type = nm1.entity_type_qualifier;
                            claim.subscriber_last_name = nm1.last_name_or_org.clone().unwrap_or_default();
                            claim.subscriber_first_name = nm1.first_name.clone().unwrap_or_default();
                            claim.subscriber_middle_name = nm1.middle_name.clone();
                            claim.subscriber_name_suffix = nm1.name_suffix.clone();
                            if nm1.identification_code_qualifier.as_deref() == Some("MI") {
                                claim.subscriber_id_code_qualifier = nm1.identification_code_qualifier.clone().unwrap_or_default();
                                claim.subscriber_id = nm1.identification_code.clone().unwrap_or_default();
                            }
                        }
                        "QC" => {
                            // Loop 2010CA - Patient Name (when patient != subscriber)
                            claim.patient_entity_identifier = Some(nm1.entity_identifier_code);
                            claim.patient_entity_type = Some(nm1.entity_type_qualifier);
                            claim.patient_last_name = nm1.last_name_or_org.clone();
                            claim.patient_first_name = nm1.first_name.clone();
                            claim.patient_middle_name = nm1.middle_name.clone();
                            claim.patient_name_suffix = nm1.name_suffix.clone();
                            in_patient_loop = true;
                        }
                        "PR" => {
                            // Loop 2010BB - Payer Name (Primary)
                            // Only capture if primary payer not yet captured
                            if !primary_payer_captured {
                                claim.payer_entity_identifier = nm1.entity_identifier_code;
                                claim.payer_entity_type = nm1.entity_type_qualifier;
                                claim.payer_name = nm1.last_name_or_org.clone().unwrap_or_default();
                                claim.payer_id_qualifier = nm1.identification_code_qualifier.clone().unwrap_or_default();
                                claim.payer_id = nm1.identification_code.clone().unwrap_or_default();
                                primary_payer_captured = true;
                            }
                        }
                        "82" => {
                            claim.rendering_provider_qualifier = nm1.identification_code_qualifier.clone();
                            if nm1.identification_code_qualifier.as_deref() == Some("XX") {
                                claim.rendering_provider_npi = nm1.identification_code.clone();
                                debug_log(&format!("[CLAIM] Set rendering_provider_npi = {:?}", claim.rendering_provider_npi));
                            }
                            claim.rendering_provider_last_name = nm1.last_name_or_org.clone();
                            claim.rendering_provider_first_name = nm1.first_name.clone();
                            debug_log(&format!("[CLAIM] Set rendering_provider_last_name = {:?}, first_name = {:?}",
                                claim.rendering_provider_last_name, claim.rendering_provider_first_name));
                        }
                        "77" => {
                            // Loop 2310C - Service Facility Location
                            claim.service_facility_qualifier = nm1.identification_code_qualifier.clone();
                            if nm1.identification_code_qualifier.as_deref() == Some("XX") {
                                claim.service_facility_npi = nm1.identification_code.clone();
                            }
                            claim.service_facility_name = nm1.last_name_or_org.clone();
                            debug_log(&format!("[CLAIM] Set service_facility_name = {:?}, npi = {:?}, qualifier = {:?}",
                                claim.service_facility_name, claim.service_facility_npi, claim.service_facility_qualifier));
                        }
                        "DN" => {
                            claim.referring_provider_qualifier = nm1.identification_code_qualifier.clone();
                            if nm1.identification_code_qualifier.as_deref() == Some("XX") {
                                claim.referring_provider_npi = nm1.identification_code.clone();
                            }
                            claim.referring_provider_last_name = nm1.last_name_or_org.clone();
                            claim.referring_provider_first_name = nm1.first_name.clone();
                        }
                        "DQ" => {
                            claim.supervising_provider_qualifier = nm1.identification_code_qualifier.clone();
                            if nm1.identification_code_qualifier.as_deref() == Some("XX") {
                                claim.supervising_provider_npi = nm1.identification_code.clone();
                            }
                            claim.supervising_provider_last_name = nm1.last_name_or_org.clone();
                            claim.supervising_provider_first_name = nm1.first_name.clone();
                        }
                        _ => {}
                    }
                }
            }
            "N3" => {
                let n3 = N3Segment::parse(segment)?;
                if let Some(entity) = &last_nm1_entity {
                    match entity.as_str() {
                        "IL" => {
                            claim.subscriber_address_line1 = Some(n3.address_line1.clone());
                            claim.subscriber_address_line2 = n3.address_line2.clone();
                        }
                        "QC" => {
                            // Patient address (when patient != subscriber)
                            claim.patient_address_line1 = Some(n3.address_line1.clone());
                            claim.patient_address_line2 = n3.address_line2.clone();
                        }
                        "PR" => {
                            claim.payer_address_line1 = Some(n3.address_line1.clone());
                            claim.payer_address_line2 = n3.address_line2.clone();
                        }
                        "77" => {
                            claim.service_facility_address_line1 = Some(n3.address_line1.clone());
                            claim.service_facility_address_line2 = n3.address_line2.clone();
                        }
                        "COB_IL" => {
                            // Other subscriber address (COB)
                            if let Some(ref mut other_ins) = current_other_insurance {
                                other_ins.other_subscriber_address_line1 = Some(n3.address_line1.clone());
                                other_ins.other_subscriber_address_line2 = n3.address_line2.clone();
                            }
                        }
                        "COB_PR" => {
                            // Other payer address (COB)
                            if let Some(ref mut other_ins) = current_other_insurance {
                                other_ins.payer_address_line1 = Some(n3.address_line1.clone());
                                other_ins.payer_address_line2 = n3.address_line2.clone();
                            }
                        }
                        _ => {}
                    }
                }
            }
            "N4" => {
                let n4 = N4Segment::parse(segment)?;
                if let Some(entity) = &last_nm1_entity {
                    match entity.as_str() {
                        "IL" => {
                            claim.subscriber_city = Some(n4.city.clone());
                            claim.subscriber_state = n4.state_code.clone();
                            claim.subscriber_postal_code = n4.postal_code.clone();
                            claim.subscriber_country = n4.country_code.clone();
                        }
                        "QC" => {
                            // Patient city/state (when patient != subscriber)
                            claim.patient_city = Some(n4.city.clone());
                            claim.patient_state = n4.state_code.clone();
                            claim.patient_postal_code = n4.postal_code.clone();
                            claim.patient_country = n4.country_code.clone();
                        }
                        "PR" => {
                            claim.payer_city = Some(n4.city.clone());
                            claim.payer_state = n4.state_code.clone();
                            claim.payer_postal_code = n4.postal_code.clone();
                        }
                        "77" => {
                            claim.service_facility_city = Some(n4.city.clone());
                            claim.service_facility_state = n4.state_code.clone();
                            claim.service_facility_postal_code = n4.postal_code.clone();
                        }
                        "COB_IL" => {
                            // Other subscriber city/state (COB)
                            if let Some(ref mut other_ins) = current_other_insurance {
                                other_ins.other_subscriber_city = Some(n4.city.clone());
                                other_ins.other_subscriber_state = n4.state_code.clone();
                                other_ins.other_subscriber_postal_code = n4.postal_code.clone();
                            }
                        }
                        "COB_PR" => {
                            // Other payer city/state (COB)
                            if let Some(ref mut other_ins) = current_other_insurance {
                                other_ins.payer_city = Some(n4.city.clone());
                                other_ins.payer_state = n4.state_code.clone();
                                other_ins.payer_postal_code = n4.postal_code.clone();
                            }
                        }
                        _ => {}
                    }
                }
            }
            "SBR" => {
                let sbr = SbrSegment::parse(segment)?;

                // Check if this is a second SBR (indicates COB Loop 2320)
                // First SBR is the subscriber loop (Loop 2000B) - the payer being BILLED
                // SBR01 indicates P=Primary, S=Secondary, T=Tertiary
                if claim.subscriber_relationship_code.is_empty() {
                    // First SBR - subscriber loop (payer being billed)
                    claim.payer_responsibility_code = sbr.payer_responsibility_sequence.clone();
                    claim.subscriber_relationship_code = sbr.individual_relationship_code;
                    claim.claim_filing_indicator_code = sbr.claim_filing_indicator_code.clone();
                    debug_log(&format!("[CLAIM] SBR payer_responsibility={}, claim_filing_indicator_code={:?}",
                        claim.payer_responsibility_code, claim.claim_filing_indicator_code));
                } else {
                    // Second or subsequent SBR - COB (Other Subscriber)
                    // Finish any existing COB record
                    if let Some(other_ins) = current_other_insurance.take() {
                        claim.other_insurance.push(other_ins);
                    }

                    // Start new COB record
                    in_cob_loop = true;
                    let mut other_ins = OtherInsurance::default();
                    other_ins.payer_responsibility_sequence = Some(sbr.payer_responsibility_sequence.clone());
                    other_ins.individual_relationship_code = Some(sbr.individual_relationship_code.clone());
                    other_ins.group_policy_number = sbr.group_policy_number.clone();
                    other_ins.group_name = sbr.group_name.clone();
                    other_ins.insurance_type_code = sbr.insurance_type_code.clone();
                    other_ins.coordination_benefits_code = sbr.coordination_of_benefits_code.clone();
                    other_ins.yes_no_condition_response = sbr.yes_no_condition_response_code.clone();
                    other_ins.employment_status_code = sbr.employment_status_code.clone();
                    other_ins.claim_filing_indicator = sbr.claim_filing_indicator_code.clone();
                    current_other_insurance = Some(other_ins);

                    debug_log(&format!(
                        "[COB] Entered COB loop - payer_seq={:?}, relationship={:?}",
                        sbr.payer_responsibility_sequence,
                        sbr.individual_relationship_code
                    ));
                }
            }
            "PAT" => {
                // PAT segment in Loop 2000C indicates patient is different from subscriber
                use crate::segments::PatSegment;
                if let Ok(pat) = PatSegment::parse(segment) {
                    claim.patient_relationship_code = pat.individual_relationship_code;
                }
            }
            "DMG" => {
                let dmg = DmgSegment::parse(segment)?;
                // DMG can appear for subscriber (Loop 2010BA) or patient (Loop 2010CA)
                // If we're in the patient loop (after NM1*QC), it's patient demographics
                if in_patient_loop {
                    claim.patient_date_of_birth = dmg.date_of_birth;
                    claim.patient_gender = dmg.gender_code;
                } else {
                    claim.subscriber_date_of_birth = dmg.date_of_birth;
                    claim.subscriber_gender = dmg.gender_code;
                }
            }
            "OI" => {
                // OI segment - Other Insurance Coverage Information (Loop 2320)
                use crate::segments::OiSegment;
                if let Ok(oi) = OiSegment::parse(segment) {
                    if let Some(ref mut other_ins) = current_other_insurance {
                        other_ins.benefits_assignment_certification = oi.benefits_assignment_certification;
                        other_ins.patient_signature_source_code = oi.patient_signature_source_code;
                        other_ins.release_of_information_code = oi.release_of_information_code;
                    }
                }
            }
            "DTP" => {
                let dtp = DtpSegment::parse(segment)?;

                // DTP*472 can appear at both claim level and service line level
                // If we're currently parsing a service line, it's the service date
                // Otherwise, it's the claim date
                if dtp.date_time_qualifier == "472" {
                    // Check date format: D8 (single date) or RD8 (date range)
                    if dtp.date_time_period_format_qualifier == "D8" {
                        if let Some(ref mut line) = current_service_line {
                            // Service line date
                            line.service_date_from = dtp.parse_date()?;
                        } else {
                            // Claim date (no active service line)
                            claim.date_of_service_from = dtp.parse_date()?;
                        }
                    } else if dtp.date_time_period_format_qualifier == "RD8" {
                        // Date range format: CCYYMMDD-CCYYMMDD
                        let (from, to) = dtp.parse_date_range()?;
                        if let Some(ref mut line) = current_service_line {
                            // Service line date range
                            line.service_date_from = from;
                            line.service_date_to = Some(to);
                        } else {
                            // Claim date range
                            claim.date_of_service_from = from;
                            claim.date_of_service_to = Some(to);
                        }
                    }
                } else {
                    // Other date types are claim-level only
                    match dtp.date_time_qualifier.as_str() {
                        "431" => claim.onset_of_illness_date = Some(dtp.parse_date()?),
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
            }
            "HI" => {
                let hi = HiSegment::parse(segment)?;
                for (qualifier, code) in hi.diagnoses.iter() {
                    // Only include diagnosis codes (ABK=Principal, ABF=Additional)
                    // Skip BP (procedure codes), BG (condition codes), etc.
                    if qualifier == "ABK" || qualifier == "ABF" {
                        let next_seq = claim.diagnoses.len() + 1;
                        claim.diagnoses.push(DiagnosisCode {
                            sequence_number: next_seq as i16,
                            diagnosis_code_qualifier: qualifier.clone(),
                            diagnosis_code: code.clone(),
                            is_principal: qualifier == "ABK",
                        });
                    }
                }
            }
            "REF" => {
                let ref_seg = RefSegment::parse(segment)?;

                if let Some(ref mut line) = current_service_line {
                    // Service line level REF segments (Loop 2420)
                    match ref_seg.reference_identification_qualifier.as_str() {
                        "G1" => line.prior_authorization_number = ref_seg.reference_identification,
                        "9F" => line.referral_number = ref_seg.reference_identification,
                        "6R" => {
                            // Line item control number - could be used for tracking
                            // Store in line_note if not already populated
                            if line.line_note.is_none() {
                                line.line_note = ref_seg.reference_identification.map(|v| format!("Line Control: {}", v));
                            }
                        }
                        _ => {}
                    }
                } else {
                    // Claim level REF segments (Loop 2300)
                    match ref_seg.reference_identification_qualifier.as_str() {
                        "EA" => claim.medical_record_number = ref_seg.reference_identification,
                        "D9" => claim.claim_number = ref_seg.reference_identification,
                        _ => {}
                    }
                }
            }
            "PRV" => {
                let prv = PrvSegment::parse(segment)?;
                // PRV*PE*PXC - Provider specialty at Loop 2310B (Rendering Provider)
                if prv.provider_code == "PE"
                    && prv.reference_identification_qualifier.as_deref() == Some("PXC") {
                    claim.rendering_provider_taxonomy = prv.reference_identification.clone();
                }
            }
            "NTE" => {
                let nte = NteSegment::parse(segment)?;
                // NTE*ADD - Additional claim or line information
                if nte.note_reference_code == "ADD" {
                    if let Some(ref mut line) = current_service_line {
                        // Service line note
                        line.line_note = Some(nte.description);
                    } else {
                        // Claim note
                        claim.claim_note = Some(nte.description);
                    }
                }
            }
            "CRC" => {
                let crc = CrcSegment::parse(segment)?;
                // CRC segments contain condition indicators
                // Store all condition codes from this segment
                claim.condition_codes.extend(crc.condition_codes);
            }
            "AMT" => {
                let amt = AmtSegment::parse(segment)?;

                // Check if we're in COB context
                if in_cob_loop {
                    if let Some(ref mut other_ins) = current_other_insurance {
                        match amt.amount_qualifier_code.as_str() {
                            "D" => {
                                // Other payer paid amount
                                other_ins.paid_amount = amt.monetary_amount;
                                // Also update legacy field
                                claim.other_payer_paid_amount = amt.monetary_amount;
                            }
                            "EAF" => {
                                // Remaining patient liability
                                other_ins.remaining_patient_liability = amt.monetary_amount;
                            }
                            _ => {}
                        }
                    }
                } else {
                    match amt.amount_qualifier_code.as_str() {
                        "D" => {
                            // Patient amount paid
                            claim.patient_amount_paid = amt.monetary_amount;
                        }
                        "F5" => {
                            // Patient responsibility amount
                            claim.patient_responsibility_amount = amt.monetary_amount;
                        }
                        "A8" => {
                            // COB - other payer paid amount (at claim level)
                            claim.other_payer_paid_amount = amt.monetary_amount;
                        }
                        "B6" => {
                            // Allowed amount (at service line level if in service line context)
                            if let Some(ref mut line) = current_service_line {
                                line.allowed_amount = amt.monetary_amount;
                            }
                        }
                        _ => {}
                    }
                }
            }
            "CAS" => {
                // CAS - Claim Adjustment Segment (Loop 2320 or 2430)
                use crate::segments::CasSegment;
                if let Ok(cas) = CasSegment::parse(segment) {
                    let adjustments: Vec<ClaimAdjustment> = cas.adjustments.iter()
                        .map(|(reason, amount, qty)| ClaimAdjustment {
                            adjustment_group_code: cas.adjustment_group_code.clone(),
                            adjustment_reason_code: reason.clone(),
                            adjustment_amount: *amount,
                            adjustment_quantity: *qty,
                        })
                        .collect();

                    // Determine context: service line adjudication or COB claim-level
                    if let Some(ref mut line_adj) = current_line_adjudication {
                        // CAS in Loop 2430 (service line level)
                        line_adj.adjustments.extend(adjustments);
                    } else if let Some(ref mut other_ins) = current_other_insurance {
                        // CAS in Loop 2320 (COB claim level)
                        other_ins.adjustments.extend(adjustments);
                    }
                }
            }
            "SVD" => {
                // SVD - Line Adjudication Information (Loop 2430)
                use crate::segments::SvdSegment;
                if let Ok(svd) = SvdSegment::parse(segment) {
                    // Save any previous line adjudication to current service line
                    if let Some(line_adj) = current_line_adjudication.take() {
                        if let Some(ref mut line) = current_service_line {
                            line.line_adjudications.push(line_adj);
                            // Also set legacy field for backwards compatibility
                            if line.other_payer_line_paid_amount.is_none() {
                                line.other_payer_line_paid_amount = svd.service_line_paid_amount;
                            }
                        }
                    }

                    // Start new line adjudication
                    current_line_adjudication = Some(LineAdjudication {
                        payer_id: svd.other_payer_primary_identifier,
                        paid_amount: svd.service_line_paid_amount,
                        procedure_code: svd.procedure_code,
                        procedure_modifier: svd.procedure_modifier_1,
                        paid_service_unit_count: svd.paid_service_unit_count,
                        bundled_line_number: svd.bundled_unbundled_line_number,
                        adjudication_date: None,
                        adjustments: Vec::new(),
                    });
                }
            }
            "CR1" => {
                let cr1 = Cr1Segment::parse(segment)?;
                // CR1 - Ambulance Transport Information
                claim.ambulance_transport_reason_code = cr1.ambulance_transport_reason_code;
                claim.ambulance_transport_distance = cr1.weight; // Weight field used for distance
                // Additional CR1 fields could be mapped if needed
                // cr1.unit_of_measurement_code, cr1.ambulance_transport_code
            }
            "PWK" => {
                let pwk = PwkSegment::parse(segment)?;
                // PWK - Paperwork/Attachment Information
                claim.paperwork_report_type = Some(pwk.report_type_code);
                claim.paperwork_transmission_code = pwk.report_transmission_code;
                claim.paperwork_control_number = pwk.identification_code;
            }
            "LX" => {
                // Save any pending line adjudication to current service line
                if let Some(line_adj) = current_line_adjudication.take() {
                    if let Some(ref mut line) = current_service_line {
                        line.line_adjudications.push(line_adj);
                    }
                }

                // Save previous service line if any
                if let Some(line) = current_service_line.take() {
                    claim.service_lines.push(line);
                }

                // Exit COB loop when we see LX (service lines follow after COB data)
                if in_cob_loop {
                    if let Some(other_ins) = current_other_insurance.take() {
                        claim.other_insurance.push(other_ins);
                    }
                    in_cob_loop = false;
                }

                // Start new service line
                let lx = LxSegment::parse(segment)?;
                current_service_line = Some(ServiceLine {
                    line_number: lx.line_number,
                    product_service_id_qualifier: String::new(),
                    procedure_code: String::new(),
                    procedure_modifier_1: None,
                    procedure_modifier_2: None,
                    procedure_modifier_3: None,
                    procedure_modifier_4: None,
                    line_item_charge_amount: rust_decimal::Decimal::ZERO,
                    unit_basis_measurement_code: String::new(),
                    service_unit_count: rust_decimal::Decimal::ZERO,
                    service_date_from: *DEFAULT_DATE,
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
                    referring_provider_last_name: None,
                    referring_provider_first_name: None,
                    ndc_code: None,
                    ndc_unit_count: None,
                    ndc_measurement_unit: None,
                    prior_authorization_number: None,
                    referral_number: None,
                    line_note: None,
                    revenue_code: None,
                    other_payer_line_paid_amount: None,
                    line_adjudications: Vec::new(),
                    allowed_amount: None,
                    saving_amount: None,
                });
            }
            "SV1" => {
                if let Some(ref mut line) = current_service_line {
                    let sv1 = Sv1Segment::parse(segment)?;
                    line.product_service_id_qualifier = sv1.product_service_id_qualifier;
                    line.procedure_code = sv1.procedure_code;
                    line.procedure_modifier_1 = sv1.procedure_modifier_1;
                    line.procedure_modifier_2 = sv1.procedure_modifier_2;
                    line.procedure_modifier_3 = sv1.procedure_modifier_3;
                    line.procedure_modifier_4 = sv1.procedure_modifier_4;
                    line.line_item_charge_amount = sv1.line_item_charge_amount;
                    line.unit_basis_measurement_code = sv1.unit_basis_measurement_code;
                    line.service_unit_count = sv1.service_unit_count;
                    line.place_of_service_code = sv1.place_of_service_code;

                    // Diagnosis pointers - SV1 returns Vec<i16>, need to map to individual fields
                    if let Some(&p1) = sv1.diagnosis_code_pointer.get(0) {
                        line.diagnosis_code_pointer_1 = Some(p1);
                    }
                    if let Some(&p2) = sv1.diagnosis_code_pointer.get(1) {
                        line.diagnosis_code_pointer_2 = Some(p2);
                    }
                    if let Some(&p3) = sv1.diagnosis_code_pointer.get(2) {
                        line.diagnosis_code_pointer_3 = Some(p3);
                    }
                    if let Some(&p4) = sv1.diagnosis_code_pointer.get(3) {
                        line.diagnosis_code_pointer_4 = Some(p4);
                    }
                }
            }
            "LIN" => {
                if let Some(ref mut line) = current_service_line {
                    let lin = LinSegment::parse(segment)?;
                    // LIN segment - Drug Identification (NDC)
                    // LIN*N4 is typically NDC code qualifier
                    if lin.product_service_id_qualifier == "N4" {
                        line.ndc_code = Some(lin.product_service_id);
                    }
                }
            }
            "CTP" => {
                if let Some(ref mut line) = current_service_line {
                    let ctp = CtpSegment::parse(segment)?;
                    // CTP - Pricing Information for drug/supply lines
                    // Store NDC quantity and unit information
                    if let Some(qty) = ctp.quantity {
                        line.ndc_unit_count = Some(qty);
                    }
                    if let Some(unit) = ctp.unit_of_measurement_code {
                        line.ndc_measurement_unit = Some(unit);
                    }
                }
            }
            "HCP" => {
                if let Some(ref mut line) = current_service_line {
                    let hcp = HcpSegment::parse(segment)?;
                    // HCP - Health Care Pricing (adjudication from other payers)
                    line.allowed_amount = hcp.allowed_amount.or(line.allowed_amount);
                    line.saving_amount = hcp.saving_amount;
                }
            }
            _ => {}
        }
    }

    // Save any pending line adjudication to current service line
    if let Some(line_adj) = current_line_adjudication.take() {
        if let Some(ref mut line) = current_service_line {
            line.line_adjudications.push(line_adj);
        }
    }

    // Don't forget to push the last service line
    if let Some(line) = current_service_line {
        claim.service_lines.push(line);
    }

    // Save any pending COB record
    if let Some(other_ins) = current_other_insurance.take() {
        claim.other_insurance.push(other_ins);
    }

    // If claim-level dates are still default (1900-01-01), copy from first service line
    // This handles cases where DTP*472 appears only at service line level
    if claim.date_of_service_from == *DEFAULT_DATE && !claim.service_lines.is_empty() {
        claim.date_of_service_from = claim.service_lines[0].service_date_from;
        claim.date_of_service_to = claim.service_lines[0].service_date_to;
    }

    // If claim-level rendering provider is not set but first service line has one, copy it up
    // This handles cases where NM1*82 appears only at service line level (Loop 2420A)
    if claim.rendering_provider_npi.is_none() && !claim.service_lines.is_empty() {
        if let Some(ref npi) = claim.service_lines[0].rendering_provider_npi {
            claim.rendering_provider_npi = Some(npi.clone());
            claim.rendering_provider_last_name = claim.service_lines[0].rendering_provider_last_name.clone();
            claim.rendering_provider_first_name = claim.service_lines[0].rendering_provider_first_name.clone();
            debug_log(&format!(
                "[CLAIM] Copied rendering_provider from first service line: npi={:?}, name={:?} {:?}",
                claim.rendering_provider_npi,
                claim.rendering_provider_first_name,
                claim.rendering_provider_last_name
            ));
        }
    }

    // If claim-level referring provider is not set but first service line has one, copy it up
    // This handles cases where NM1*DN appears only at service line level (Loop 2420F)
    if claim.referring_provider_npi.is_none() && !claim.service_lines.is_empty() {
        if let Some(ref npi) = claim.service_lines[0].referring_provider_npi {
            claim.referring_provider_npi = Some(npi.clone());
            claim.referring_provider_last_name = claim.service_lines[0].referring_provider_last_name.clone();
            claim.referring_provider_first_name = claim.service_lines[0].referring_provider_first_name.clone();
            debug_log(&format!(
                "[CLAIM] Copied referring_provider from first service line: npi={:?}, name={:?} {:?}",
                claim.referring_provider_npi,
                claim.referring_provider_first_name,
                claim.referring_provider_last_name
            ));
        }
    }

    // If patient fields are empty, copy from subscriber (patient IS the subscriber)
    // This handles the common case where NM1*IL exists but NM1*QC does not
    if claim.patient_last_name.is_none() && !claim.subscriber_last_name.is_empty() {
        claim.patient_last_name = Some(claim.subscriber_last_name.clone());
        claim.patient_first_name = Some(claim.subscriber_first_name.clone());
        claim.patient_middle_name = claim.subscriber_middle_name.clone();
        claim.patient_name_suffix = claim.subscriber_name_suffix.clone();
        claim.patient_date_of_birth = claim.subscriber_date_of_birth;
        claim.patient_gender = claim.subscriber_gender.clone();
        claim.patient_address_line1 = claim.subscriber_address_line1.clone();
        claim.patient_address_line2 = claim.subscriber_address_line2.clone();
        claim.patient_city = claim.subscriber_city.clone();
        claim.patient_state = claim.subscriber_state.clone();
        claim.patient_postal_code = claim.subscriber_postal_code.clone();
        debug_log("[CLAIM] Copied subscriber info to patient fields (patient = subscriber)");
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
        service_date_from: *DEFAULT_DATE,
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
        referring_provider_last_name: None,
        referring_provider_first_name: None,
        ndc_code: None,
        ndc_unit_count: None,
        ndc_measurement_unit: None,
        prior_authorization_number: None,
        referral_number: None,
        line_note: None,
        revenue_code: None,
        other_payer_line_paid_amount: None,
        line_adjudications: Vec::new(),
        allowed_amount: None,
        saving_amount: None,
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
