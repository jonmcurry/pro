// EDI segment parsers for individual segments

use crate::types::EdiSegment;
use chrono::NaiveDate;
use pro_common::{Error, Result};
use rust_decimal::Decimal;
use std::str::FromStr;

/// Parse EDI date in format CCYYMMDD or YYYYMMDD
pub fn parse_edi_date(date_str: &str) -> Result<NaiveDate> {
    if date_str.is_empty() {
        return Err(Error::Parse("Empty date string".to_string()));
    }

    // Handle both 8-digit (CCYYMMDD/YYYYMMDD) formats
    if date_str.len() == 8 {
        let year = date_str[0..4].parse::<i32>()
            .map_err(|_| Error::Parse(format!("Invalid year in date: {}", date_str)))?;
        let month = date_str[4..6].parse::<u32>()
            .map_err(|_| Error::Parse(format!("Invalid month in date: {}", date_str)))?;
        let day = date_str[6..8].parse::<u32>()
            .map_err(|_| Error::Parse(format!("Invalid day in date: {}", date_str)))?;

        NaiveDate::from_ymd_opt(year, month, day)
            .ok_or_else(|| Error::Parse(format!("Invalid date: {}", date_str)))
    } else {
        Err(Error::Parse(format!("Invalid date format (expected 8 digits): {}", date_str)))
    }
}

/// Parse EDI time in format HHMM or HHMMSS
pub fn parse_edi_time(time_str: &str) -> Result<(u32, u32, u32)> {
    if time_str.len() == 4 {
        let hour = time_str[0..2].parse::<u32>()
            .map_err(|_| Error::Parse(format!("Invalid hour in time: {}", time_str)))?;
        let minute = time_str[2..4].parse::<u32>()
            .map_err(|_| Error::Parse(format!("Invalid minute in time: {}", time_str)))?;
        Ok((hour, minute, 0))
    } else if time_str.len() == 6 {
        let hour = time_str[0..2].parse::<u32>()
            .map_err(|_| Error::Parse(format!("Invalid hour in time: {}", time_str)))?;
        let minute = time_str[2..4].parse::<u32>()
            .map_err(|_| Error::Parse(format!("Invalid minute in time: {}", time_str)))?;
        let second = time_str[4..6].parse::<u32>()
            .map_err(|_| Error::Parse(format!("Invalid second in time: {}", time_str)))?;
        Ok((hour, minute, second))
    } else {
        Err(Error::Parse(format!("Invalid time format: {}", time_str)))
    }
}

/// Parse EDI decimal amount
pub fn parse_edi_decimal(amount_str: &str) -> Result<Decimal> {
    if amount_str.is_empty() {
        return Err(Error::Parse("Empty amount string".to_string()));
    }

    Decimal::from_str(amount_str)
        .map_err(|_| Error::Parse(format!("Invalid decimal amount: {}", amount_str)))
}

/// Parse NM1 segment (Name)
/// Format: NM1*qualifier*entity_type*last*first*middle*prefix*suffix*id_qualifier*id
pub struct Nm1Segment {
    pub entity_identifier_code: String,
    pub entity_type_qualifier: String,
    pub last_name_or_org: Option<String>,
    pub first_name: Option<String>,
    pub middle_name: Option<String>,
    pub name_prefix: Option<String>,
    pub name_suffix: Option<String>,
    pub identification_code_qualifier: Option<String>,
    pub identification_code: Option<String>,
}

impl Nm1Segment {
    pub fn parse(segment: &EdiSegment) -> Result<Self> {
        if segment.segment_id != "NM1" {
            return Err(Error::Parse(format!("Expected NM1 segment, got {}", segment.segment_id)));
        }

        Ok(Self {
            entity_identifier_code: segment.get_or_empty(0).to_string(),
            entity_type_qualifier: segment.get_or_empty(1).to_string(),
            last_name_or_org: segment.get_optional(2),
            first_name: segment.get_optional(3),
            middle_name: segment.get_optional(4),
            name_prefix: segment.get_optional(5),
            name_suffix: segment.get_optional(6),
            identification_code_qualifier: segment.get_optional(6),
            identification_code: segment.get_optional(7),
        })
    }
}

/// Parse N3 segment (Address)
/// Format: N3*address_line1*address_line2
pub struct N3Segment {
    pub address_line1: String,
    pub address_line2: Option<String>,
}

impl N3Segment {
    pub fn parse(segment: &EdiSegment) -> Result<Self> {
        if segment.segment_id != "N3" {
            return Err(Error::Parse(format!("Expected N3 segment, got {}", segment.segment_id)));
        }

        Ok(Self {
            address_line1: segment.get_or_empty(0).to_string(),
            address_line2: segment.get_optional(1),
        })
    }
}

/// Parse N4 segment (City, State, Postal Code)
/// Format: N4*city*state*postal_code*country*location_qualifier*location_id
pub struct N4Segment {
    pub city: String,
    pub state_code: Option<String>,
    pub postal_code: Option<String>,
    pub country_code: Option<String>,
}

impl N4Segment {
    pub fn parse(segment: &EdiSegment) -> Result<Self> {
        if segment.segment_id != "N4" {
            return Err(Error::Parse(format!("Expected N4 segment, got {}", segment.segment_id)));
        }

        Ok(Self {
            city: segment.get_or_empty(0).to_string(),
            state_code: segment.get_optional(1),
            postal_code: segment.get_optional(2),
            country_code: segment.get_optional(3),
        })
    }
}

/// Parse REF segment (Reference Identification)
/// Format: REF*qualifier*reference_id*description
pub struct RefSegment {
    pub reference_identification_qualifier: String,
    pub reference_identification: Option<String>,
    pub description: Option<String>,
}

impl RefSegment {
    pub fn parse(segment: &EdiSegment) -> Result<Self> {
        if segment.segment_id != "REF" {
            return Err(Error::Parse(format!("Expected REF segment, got {}", segment.segment_id)));
        }

        Ok(Self {
            reference_identification_qualifier: segment.get_or_empty(0).to_string(),
            reference_identification: segment.get_optional(1),
            description: segment.get_optional(2),
        })
    }
}

/// Parse PER segment (Contact Information)
/// Format: PER*function_code*name*comm_qualifier1*comm_number1*comm_qualifier2*comm_number2
pub struct PerSegment {
    pub contact_function_code: String,
    pub contact_name: Option<String>,
    pub communication_number_qualifier_1: Option<String>,
    pub communication_number_1: Option<String>,
    pub communication_number_qualifier_2: Option<String>,
    pub communication_number_2: Option<String>,
    pub communication_number_qualifier_3: Option<String>,
    pub communication_number_3: Option<String>,
}

impl PerSegment {
    pub fn parse(segment: &EdiSegment) -> Result<Self> {
        if segment.segment_id != "PER" {
            return Err(Error::Parse(format!("Expected PER segment, got {}", segment.segment_id)));
        }

        Ok(Self {
            contact_function_code: segment.get_or_empty(0).to_string(),
            contact_name: segment.get_optional(1),
            communication_number_qualifier_1: segment.get_optional(2),
            communication_number_1: segment.get_optional(3),
            communication_number_qualifier_2: segment.get_optional(4),
            communication_number_2: segment.get_optional(5),
            communication_number_qualifier_3: segment.get_optional(6),
            communication_number_3: segment.get_optional(7),
        })
    }
}

/// Parse DMG segment (Demographic Information)
/// Format: DMG*date_qualifier*birth_date*gender
pub struct DmgSegment {
    pub date_time_period_format_qualifier: String,
    pub date_of_birth: Option<NaiveDate>,
    pub gender_code: Option<String>,
}

impl DmgSegment {
    pub fn parse(segment: &EdiSegment) -> Result<Self> {
        if segment.segment_id != "DMG" {
            return Err(Error::Parse(format!("Expected DMG segment, got {}", segment.segment_id)));
        }

        let birth_date = if let Some(date_str) = segment.get_optional(1) {
            if date_str.is_empty() {
                None  // Empty date field - treat as missing
            } else {
                Some(parse_edi_date(&date_str)?)
            }
        } else {
            None
        };

        Ok(Self {
            date_time_period_format_qualifier: segment.get_or_empty(0).to_string(),
            date_of_birth: birth_date,
            gender_code: segment.get_optional(2),
        })
    }
}

/// Parse SBR segment (Subscriber Information)
/// Format: SBR*payer_responsibility*individual_relationship*group_policy_number*group_name*insurance_type*...
pub struct SbrSegment {
    pub payer_responsibility_sequence: String,
    pub individual_relationship_code: String,
    pub group_policy_number: Option<String>,
    pub group_name: Option<String>,
    pub insurance_type_code: Option<String>,
    pub coordination_of_benefits_code: Option<String>,
    pub yes_no_condition_response_code: Option<String>,
    pub employment_status_code: Option<String>,
    pub claim_filing_indicator_code: Option<String>,
}

impl SbrSegment {
    pub fn parse(segment: &EdiSegment) -> Result<Self> {
        if segment.segment_id != "SBR" {
            return Err(Error::Parse(format!("Expected SBR segment, got {}", segment.segment_id)));
        }

        Ok(Self {
            payer_responsibility_sequence: segment.get_or_empty(0).to_string(),
            individual_relationship_code: segment.get_or_empty(1).to_string(),
            group_policy_number: segment.get_optional(2),
            group_name: segment.get_optional(3),
            insurance_type_code: segment.get_optional(4),
            coordination_of_benefits_code: segment.get_optional(5),
            yes_no_condition_response_code: segment.get_optional(6),
            employment_status_code: segment.get_optional(7),
            claim_filing_indicator_code: segment.get_optional(8),
        })
    }
}

/// Parse CLM segment (Claim Information)
/// Format: CLM*patient_control_number*charge_amount*claim_filing_indicator*place_of_service*...
pub struct ClmSegment {
    pub patient_control_number: String,
    pub total_claim_charge_amount: Decimal,
    pub claim_filing_indicator_code: Option<String>,
    pub place_of_service_code: Option<String>,
    pub claim_frequency_code: Option<String>,
    pub provider_signature_indicator: Option<String>,
    pub assignment_indicator: Option<String>,
    pub benefits_assignment_indicator: Option<String>,
    pub release_information_code: Option<String>,
}

impl ClmSegment {
    pub fn parse(segment: &EdiSegment) -> Result<Self> {
        if segment.segment_id != "CLM" {
            return Err(Error::Parse(format!("Expected CLM segment, got {}", segment.segment_id)));
        }

        let charge_amount = parse_edi_decimal(segment.get_or_empty(1))?;

        Ok(Self {
            patient_control_number: segment.get_or_empty(0).to_string(),
            total_claim_charge_amount: charge_amount,
            claim_filing_indicator_code: segment.get_optional(4),
            place_of_service_code: segment.get_optional(5),
            claim_frequency_code: segment.get_optional(6),
            provider_signature_indicator: segment.get_optional(7),
            assignment_indicator: segment.get_optional(8),
            benefits_assignment_indicator: segment.get_optional(9),
            release_information_code: segment.get_optional(10),
        })
    }
}

/// Parse DTP segment (Date/Time Period)
/// Format: DTP*qualifier*format*date
pub struct DtpSegment {
    pub date_time_qualifier: String,
    pub date_time_period_format_qualifier: String,
    pub date_time_period: String,
}

impl DtpSegment {
    pub fn parse(segment: &EdiSegment) -> Result<Self> {
        if segment.segment_id != "DTP" {
            return Err(Error::Parse(format!("Expected DTP segment, got {}", segment.segment_id)));
        }

        Ok(Self {
            date_time_qualifier: segment.get_or_empty(0).to_string(),
            date_time_period_format_qualifier: segment.get_or_empty(1).to_string(),
            date_time_period: segment.get_or_empty(2).to_string(),
        })
    }

    /// Parse the date from DTP segment
    pub fn parse_date(&self) -> Result<NaiveDate> {
        parse_edi_date(&self.date_time_period)
    }

    /// Parse date range from DTP segment (format: CCYYMMDD-CCYYMMDD)
    pub fn parse_date_range(&self) -> Result<(NaiveDate, NaiveDate)> {
        if let Some(pos) = self.date_time_period.find('-') {
            let from_str = &self.date_time_period[..pos];
            let to_str = &self.date_time_period[pos + 1..];
            let from_date = parse_edi_date(from_str)?;
            let to_date = parse_edi_date(to_str)?;
            Ok((from_date, to_date))
        } else {
            Err(Error::Parse(format!("Invalid date range format: {}", self.date_time_period)))
        }
    }
}

/// Parse SV1 segment (Professional Service)
/// Format: SV1*composite_medical_procedure*charge_amount*unit*quantity*facility_code*...
pub struct Sv1Segment {
    pub product_service_id_qualifier: String,
    pub procedure_code: String,
    pub procedure_modifier_1: Option<String>,
    pub procedure_modifier_2: Option<String>,
    pub procedure_modifier_3: Option<String>,
    pub procedure_modifier_4: Option<String>,
    pub line_item_charge_amount: Decimal,
    pub unit_basis_measurement_code: String,
    pub service_unit_count: Decimal,
    pub place_of_service_code: Option<String>,
    pub diagnosis_code_pointer: Vec<i16>,
}

impl Sv1Segment {
    pub fn parse(segment: &EdiSegment) -> Result<Self> {
        if segment.segment_id != "SV1" {
            return Err(Error::Parse(format!("Expected SV1 segment, got {}", segment.segment_id)));
        }

        // Element 01 is composite: qualifier:code:modifier1:modifier2:modifier3:modifier4
        let composite = segment.get_or_empty(0);
        let parts: Vec<&str> = composite.split(':').collect();

        let product_service_id_qualifier = parts.get(0).unwrap_or(&"").to_string();
        let procedure_code = parts.get(1).unwrap_or(&"").to_string();
        let procedure_modifier_1 = parts.get(2).and_then(|s| if s.is_empty() { None } else { Some(s.to_string()) });
        let procedure_modifier_2 = parts.get(3).and_then(|s| if s.is_empty() { None } else { Some(s.to_string()) });
        let procedure_modifier_3 = parts.get(4).and_then(|s| if s.is_empty() { None } else { Some(s.to_string()) });
        let procedure_modifier_4 = parts.get(5).and_then(|s| if s.is_empty() { None } else { Some(s.to_string()) });

        let charge_amount = parse_edi_decimal(segment.get_or_empty(1))?;
        let unit_basis_measurement_code = segment.get_or_empty(2).to_string();
        let service_unit_count = parse_edi_decimal(segment.get_or_empty(3))?;
        let place_of_service_code = segment.get_optional(4);

        // Element 07 is composite diagnosis code pointers
        let diagnosis_code_pointer = if let Some(pointers) = segment.get_optional(6) {
            pointers.split(':')
                .filter_map(|p| p.parse::<i16>().ok())
                .collect()
        } else {
            Vec::new()
        };

        Ok(Self {
            product_service_id_qualifier,
            procedure_code,
            procedure_modifier_1,
            procedure_modifier_2,
            procedure_modifier_3,
            procedure_modifier_4,
            line_item_charge_amount: charge_amount,
            unit_basis_measurement_code,
            service_unit_count,
            place_of_service_code,
            diagnosis_code_pointer,
        })
    }
}

/// Parse HI segment (Health Care Diagnosis Code)
/// Format: HI*qualifier:code*qualifier:code*...
pub struct HiSegment {
    pub diagnoses: Vec<(String, String)>, // (qualifier, code) pairs
}

impl HiSegment {
    pub fn parse(segment: &EdiSegment) -> Result<Self> {
        if segment.segment_id != "HI" {
            return Err(Error::Parse(format!("Expected HI segment, got {}", segment.segment_id)));
        }

        let mut diagnoses = Vec::new();

        // HI segment can have up to 12 diagnosis codes
        for i in 0..12 {
            if let Some(composite) = segment.get_optional(i) {
                let parts: Vec<&str> = composite.split(':').collect();
                if parts.len() >= 2 {
                    let qualifier = parts[0].to_string();
                    let code = parts[1].to_string();
                    diagnoses.push((qualifier, code));
                }
            } else {
                break;
            }
        }

        Ok(Self { diagnoses })
    }
}

/// Parse LX segment (Line Number)
/// Format: LX*line_number
pub struct LxSegment {
    pub line_number: i16,
}

impl LxSegment {
    pub fn parse(segment: &EdiSegment) -> Result<Self> {
        if segment.segment_id != "LX" {
            return Err(Error::Parse(format!("Expected LX segment, got {}", segment.segment_id)));
        }

        let line_number = segment.get_or_empty(0).parse::<i16>()
            .map_err(|_| Error::Parse(format!("Invalid line number: {}", segment.get_or_empty(0))))?;

        Ok(Self { line_number })
    }
}

/// Parse HL segment (Hierarchical Level)
/// Format: HL*id*parent_id*level_code*child_code
pub struct HlSegment {
    pub hierarchical_id_number: String,
    pub hierarchical_parent_id_number: Option<String>,
    pub hierarchical_level_code: String,
    pub hierarchical_child_code: Option<String>,
}

impl HlSegment {
    pub fn parse(segment: &EdiSegment) -> Result<Self> {
        if segment.segment_id != "HL" {
            return Err(Error::Parse(format!("Expected HL segment, got {}", segment.segment_id)));
        }

        Ok(Self {
            hierarchical_id_number: segment.get_or_empty(0).to_string(),
            hierarchical_parent_id_number: segment.get_optional(1),
            hierarchical_level_code: segment.get_or_empty(2).to_string(),
            hierarchical_child_code: segment.get_optional(3),
        })
    }
}

/// Parse PRV segment (Provider Information)
/// Format: PRV*provider_code*reference_id_qualifier*reference_id
pub struct PrvSegment {
    pub provider_code: String,
    pub reference_identification_qualifier: Option<String>,
    pub reference_identification: Option<String>,
}

impl PrvSegment {
    pub fn parse(segment: &EdiSegment) -> Result<Self> {
        if segment.segment_id != "PRV" {
            return Err(Error::Parse(format!("Expected PRV segment, got {}", segment.segment_id)));
        }

        Ok(Self {
            provider_code: segment.get_or_empty(0).to_string(),
            reference_identification_qualifier: segment.get_optional(1),
            reference_identification: segment.get_optional(2),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_edi_date() {
        assert!(parse_edi_date("20240115").is_ok());
        assert_eq!(parse_edi_date("20240115").unwrap(), NaiveDate::from_ymd_opt(2024, 1, 15).unwrap());
        assert!(parse_edi_date("").is_err());
        assert!(parse_edi_date("2024011").is_err()); // Too short
        assert!(parse_edi_date("20241301").is_err()); // Invalid month
    }

    #[test]
    fn test_parse_edi_decimal() {
        assert_eq!(parse_edi_decimal("12345.67").unwrap(), Decimal::new(1234567, 2));
        assert_eq!(parse_edi_decimal("100").unwrap(), Decimal::new(100, 0));
        assert!(parse_edi_decimal("").is_err());
        assert!(parse_edi_decimal("invalid").is_err());
    }

    #[test]
    fn test_parse_edi_time() {
        assert_eq!(parse_edi_time("1430").unwrap(), (14, 30, 0));
        assert_eq!(parse_edi_time("143015").unwrap(), (14, 30, 15));
        assert!(parse_edi_time("14301").is_err()); // Invalid length
    }

    #[test]
    fn test_nm1_segment() {
        let segment = EdiSegment {
            segment_id: "NM1".to_string(),
            elements: vec![
                "IL".to_string(),
                "1".to_string(),
                "DOE".to_string(),
                "JOHN".to_string(),
                "A".to_string(),
                "".to_string(),
                "JR".to_string(),
                "MI".to_string(),
                "123456789".to_string(),
            ],
        };

        let nm1 = Nm1Segment::parse(&segment).unwrap();
        assert_eq!(nm1.entity_identifier_code, "IL");
        assert_eq!(nm1.entity_type_qualifier, "1");
        assert_eq!(nm1.last_name_or_org, Some("DOE".to_string()));
        assert_eq!(nm1.first_name, Some("JOHN".to_string()));
        assert_eq!(nm1.middle_name, Some("A".to_string()));
        assert_eq!(nm1.name_suffix, Some("JR".to_string()));
        assert_eq!(nm1.identification_code, Some("123456789".to_string()));
    }

    #[test]
    fn test_clm_segment() {
        let segment = EdiSegment {
            segment_id: "CLM".to_string(),
            elements: vec![
                "PATIENT123".to_string(),
                "250.00".to_string(),
                "".to_string(),
                "".to_string(),
                "12".to_string(),
                "11".to_string(),
                "1".to_string(),
                "Y".to_string(),
                "Y".to_string(),
                "Y".to_string(),
                "I".to_string(),
            ],
        };

        let clm = ClmSegment::parse(&segment).unwrap();
        assert_eq!(clm.patient_control_number, "PATIENT123");
        assert_eq!(clm.total_claim_charge_amount, Decimal::new(25000, 2));
        assert_eq!(clm.place_of_service_code, Some("11".to_string()));
        assert_eq!(clm.claim_frequency_code, Some("1".to_string()));
    }
}
