// Core types for 837P EDI parsing

use chrono::NaiveDate;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};


/// Complete 837P transaction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transaction837p {
    pub interchange_control_header: InterchangeControlHeader,
    pub functional_group_header: FunctionalGroupHeader,
    pub transaction_set_header: TransactionSetHeader,
    pub submitter: Submitter,
    pub receiver: Receiver,
    pub billing_provider: BillingProvider,
    pub claims: Vec<ParsedClaim>,
}

/// ISA - Interchange Control Header
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterchangeControlHeader {
    pub authorization_info_qualifier: String,
    pub authorization_information: String,
    pub security_info_qualifier: String,
    pub security_information: String,
    pub interchange_id_qualifier: String,
    pub interchange_sender_id: String,
    pub interchange_id_qualifier_2: String,
    pub interchange_receiver_id: String,
    pub interchange_date: String,
    pub interchange_time: String,
    pub repetition_separator: char,
    pub interchange_control_version: String,
    pub interchange_control_number: String,
    pub acknowledgment_requested: String,
    pub usage_indicator: String,
    pub component_element_separator: char,
}

/// GS - Functional Group Header
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionalGroupHeader {
    pub functional_identifier_code: String,
    pub application_sender_code: String,
    pub application_receiver_code: String,
    pub date: String,
    pub time: String,
    pub group_control_number: String,
    pub responsible_agency_code: String,
    pub version_release: String,
}

/// ST - Transaction Set Header
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionSetHeader {
    pub transaction_set_identifier_code: String,
    pub transaction_set_control_number: String,
    pub implementation_convention_reference: String,
}

/// Loop 1000A - Submitter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Submitter {
    pub entity_identifier_code: String,
    pub entity_type_qualifier: String,
    pub submitter_last_name: Option<String>,
    pub submitter_first_name: Option<String>,
    pub submitter_organization_name: Option<String>,
    pub identification_code_qualifier: Option<String>,
    pub identification_code: Option<String>,
    pub contact_name: Option<String>,
    pub contact_phone: Option<String>,
    pub contact_phone_extension: Option<String>,
    pub contact_fax: Option<String>,
    pub contact_email: Option<String>,
}

/// Loop 1000B - Receiver
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Receiver {
    pub entity_identifier_code: String,
    pub entity_type_qualifier: String,
    pub receiver_name: String,
    pub identification_code_qualifier: String,
    pub identification_code: String,
}

/// Loop 2000A - Billing Provider
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BillingProvider {
    pub hierarchical_id_number: String,
    pub entity_identifier_code: String,
    pub entity_type_qualifier: String,
    pub organization_name: Option<String>,
    pub last_name: Option<String>,
    pub first_name: Option<String>,
    pub npi: String,
    pub facility_id: Option<String>, // NM1 Qualifier 46 - Electronic Transmitter Identification Number
    pub provider_number: Option<String>, // REF*G2 - Provider Commercial Number / Facility ID
    pub tax_id_type: Option<String>,
    pub tax_id: Option<String>,
    pub address_line1: Option<String>,
    pub address_line2: Option<String>,
    pub city: Option<String>,
    pub state: Option<String>,
    pub postal_code: Option<String>,
    pub country_code: Option<String>,
    pub contact_name: Option<String>,
    pub contact_phone: Option<String>,
}

/// Complete parsed claim
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedClaim {
    // Temporary ID for processing
    pub temp_id: i64,

    // Loop 2000B - Subscriber Hierarchical Level
    pub subscriber_hl_number: String,
    pub subscriber_relationship_code: String,

    // Loop 2010BA - Subscriber Name
    pub subscriber_entity_identifier: String,
    pub subscriber_entity_type: String,
    pub subscriber_last_name: String,
    pub subscriber_first_name: String,
    pub subscriber_middle_name: Option<String>,
    pub subscriber_name_suffix: Option<String>,
    pub subscriber_id_code_qualifier: String,
    pub subscriber_id: String,

    // Loop 2010BA - Subscriber Demographic
    pub subscriber_date_of_birth: Option<NaiveDate>,
    pub subscriber_gender: Option<String>,

    // Loop 2010BA - Subscriber Address
    pub subscriber_address_line1: Option<String>,
    pub subscriber_address_line2: Option<String>,
    pub subscriber_city: Option<String>,
    pub subscriber_state: Option<String>,
    pub subscriber_postal_code: Option<String>,
    pub subscriber_country: Option<String>,

    // Loop 2010BA - Medical Record Number (REF*EA)
    pub medical_record_number: Option<String>,

    // Loop 2010BB - Payer
    pub payer_entity_identifier: String,
    pub payer_entity_type: String,
    pub payer_name: String,
    pub payer_id_qualifier: String,
    pub payer_id: String,
    pub payer_address_line1: Option<String>,
    pub payer_address_line2: Option<String>,
    pub payer_city: Option<String>,
    pub payer_state: Option<String>,
    pub payer_postal_code: Option<String>,

    // Loop 2300 - Claim Information
    pub patient_control_number: String,
    pub total_claim_charge_amount: Decimal,
    pub place_of_service_code: Option<String>,
    pub claim_frequency_code: Option<String>,
    pub provider_signature_indicator: Option<String>,
    pub assignment_indicator: Option<String>,
    pub benefits_assignment_indicator: Option<String>,
    pub release_of_information_code: Option<String>,
    pub patient_signature_code: Option<String>,

    // Loop 2300 - Related Causes (if any)
    pub related_causes_code_1: Option<String>,
    pub related_causes_code_2: Option<String>,
    pub related_causes_code_3: Option<String>,
    pub auto_accident_state: Option<String>,
    pub auto_accident_country: Option<String>,

    // Loop 2300 - Claim Dates
    pub date_of_service_from: NaiveDate,
    pub date_of_service_to: Option<NaiveDate>,
    pub onset_of_illness_date: Option<NaiveDate>,
    pub initial_treatment_date: Option<NaiveDate>,
    pub last_seen_date: Option<NaiveDate>,
    pub acute_manifestation_date: Option<NaiveDate>,
    pub accident_date: Option<NaiveDate>,
    pub last_menstrual_period_date: Option<NaiveDate>,
    pub last_xray_date: Option<NaiveDate>,
    pub disability_from_date: Option<NaiveDate>,
    pub disability_to_date: Option<NaiveDate>,
    pub last_worked_date: Option<NaiveDate>,
    pub authorized_return_to_work_date: Option<NaiveDate>,
    pub admission_date: Option<NaiveDate>,
    pub discharge_date: Option<NaiveDate>,

    // Loop 2300 - Claim Supplemental Information
    pub delay_reason_code: Option<String>,
    pub special_program_code: Option<String>,
    pub patient_amount_paid: Option<Decimal>,
    pub service_authorization_code: Option<String>,

    // Loop 2300 - Claim Note
    pub claim_note: Option<String>,

    // Loop 2310A - Referring Provider
    pub referring_provider_qualifier: Option<String>,
    pub referring_provider_npi: Option<String>,
    pub referring_provider_last_name: Option<String>,
    pub referring_provider_first_name: Option<String>,

    // Loop 2310B - Rendering Provider
    pub rendering_provider_qualifier: Option<String>,
    pub rendering_provider_npi: Option<String>,
    pub rendering_provider_last_name: Option<String>,
    pub rendering_provider_first_name: Option<String>,
    pub rendering_provider_taxonomy: Option<String>,

    // Loop 2310C - Service Facility
    pub service_facility_qualifier: Option<String>,
    pub service_facility_npi: Option<String>,
    pub service_facility_name: Option<String>,
    pub service_facility_address_line1: Option<String>,
    pub service_facility_address_line2: Option<String>,
    pub service_facility_city: Option<String>,
    pub service_facility_state: Option<String>,
    pub service_facility_postal_code: Option<String>,

    // Loop 2310D - Supervising Provider
    pub supervising_provider_qualifier: Option<String>,
    pub supervising_provider_npi: Option<String>,
    pub supervising_provider_last_name: Option<String>,
    pub supervising_provider_first_name: Option<String>,

    // Loop 2320 - Other Subscriber (COB)
    pub other_payer_paid_amount: Option<Decimal>,
    pub other_payer_id: Option<String>,
    pub other_payer_name: Option<String>,
    pub other_payer_claim_number: Option<String>,

    // Loop 2400 - Service Lines
    pub service_lines: Vec<ServiceLine>,

    // Loop 2300 HI - Diagnosis Codes (up to 12)
    pub diagnoses: Vec<DiagnosisCode>,
}

/// Service Line (Loop 2400)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceLine {
    pub line_number: i16,

    // SV1 segment
    pub product_service_id_qualifier: String,
    pub procedure_code: String,
    pub procedure_modifier_1: Option<String>,
    pub procedure_modifier_2: Option<String>,
    pub procedure_modifier_3: Option<String>,
    pub procedure_modifier_4: Option<String>,
    pub line_item_charge_amount: Decimal,
    pub unit_basis_measurement_code: String,
    pub service_unit_count: Decimal,

    // DTP segments - Service Dates
    pub service_date_from: NaiveDate,
    pub service_date_to: Option<NaiveDate>,

    // Place of Service
    pub place_of_service_code: Option<String>,

    // Emergency indicator
    pub emergency_indicator: Option<bool>,

    // EPSDT indicator
    pub epsdt_indicator: Option<bool>,

    // Family planning indicator
    pub family_planning_indicator: Option<bool>,

    // Diagnosis pointers (up to 12)
    pub diagnosis_code_pointer_1: Option<i16>,
    pub diagnosis_code_pointer_2: Option<i16>,
    pub diagnosis_code_pointer_3: Option<i16>,
    pub diagnosis_code_pointer_4: Option<i16>,

    // Loop 2420A - Rendering Provider (line level)
    pub rendering_provider_npi: Option<String>,
    pub rendering_provider_last_name: Option<String>,
    pub rendering_provider_first_name: Option<String>,

    // Loop 2420D - Supervising Provider (line level)
    pub supervising_provider_npi: Option<String>,

    // Loop 2420E - Ordering Provider (line level)
    pub ordering_provider_npi: Option<String>,
    pub ordering_provider_last_name: Option<String>,
    pub ordering_provider_first_name: Option<String>,

    // Loop 2420F - Referring Provider (line level)
    pub referring_provider_npi: Option<String>,

    // Loop 2410 - Drug Identification (if applicable)
    pub ndc_code: Option<String>,
    pub ndc_unit_count: Option<Decimal>,
    pub ndc_measurement_unit: Option<String>,

    // Prior authorization
    pub prior_authorization_number: Option<String>,

    // Referral number
    pub referral_number: Option<String>,

    // Line note
    pub line_note: Option<String>,

    // Revenue code (for institutional, may appear on professional)
    pub revenue_code: Option<String>,

    // Loop 2430 - Line Adjudication (from other payers)
    pub other_payer_line_paid_amount: Option<Decimal>,
}

/// Diagnosis Code from HI segment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosisCode {
    pub sequence_number: i16,
    pub diagnosis_code_qualifier: String,
    pub diagnosis_code: String,
    pub is_principal: bool,
}

/// EDI segment with parsed elements
#[derive(Debug, Clone)]
pub struct EdiSegment {
    pub segment_id: String,
    pub elements: Vec<String>,
}

impl EdiSegment {
    /// Get element at index (0-based, skipping segment ID)
    pub fn get(&self, index: usize) -> Option<&str> {
        self.elements.get(index).map(|s| s.as_str())
    }

    /// Get element at index or return empty string
    pub fn get_or_empty(&self, index: usize) -> &str {
        self.get(index).unwrap_or("")
    }

    /// Get element as optional string
    pub fn get_optional(&self, index: usize) -> Option<String> {
        self.get(index)
            .and_then(|s| if s.is_empty() { None } else { Some(s.to_string()) })
    }
}
