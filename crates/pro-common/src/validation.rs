use lazy_static::lazy_static;
use regex::Regex;
use crate::error::{Error, Result};

lazy_static! {
    // NPI: exactly 10 digits
    static ref NPI_REGEX: Regex = Regex::new(r"^\d{10}$").unwrap();

    // ICD-10-CM: Letter + 2 digits, optionally followed by decimal and up to 4 alphanumeric characters
    // Supports 7th character extensions (A, D, S, etc.) for initial/subsequent/sequela encounters
    static ref ICD10_REGEX: Regex = Regex::new(r"^[A-TV-Z]\d{2}(\.[A-Z0-9]{1,4})?$").unwrap();

    // CPT: exactly 5 digits
    static ref CPT_REGEX: Regex = Regex::new(r"^\d{5}$").unwrap();

    // HCPCS: Letter followed by 4 digits
    static ref HCPCS_REGEX: Regex = Regex::new(r"^[A-Z]\d{4}$").unwrap();

    // Modifier: 2 alphanumeric characters
    static ref MODIFIER_REGEX: Regex = Regex::new(r"^[A-Z0-9]{2}$").unwrap();

    // MBI (Medicare Beneficiary Identifier): 11 characters
    // Format: 1A2N-3A4N-5A6N7 (A = letter excluding S,L,O,I,B,Z; N = number)
    static ref MBI_REGEX: Regex = Regex::new(r"^[1-9AC-HJKMNP-RT-Y][AC-HJKMNP-RT-Y]\d{2}[AC-HJKMNP-RT-Y][AC-HJKMNP-RT-Y]\d{2}[AC-HJKMNP-RT-Y][AC-HJKMNP-RT-Y]\d{2}$").unwrap();

    // Place of Service: 2 digits
    static ref POS_REGEX: Regex = Regex::new(r"^\d{2}$").unwrap();

    // Email validation
    static ref EMAIL_REGEX: Regex = Regex::new(
        r"^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$"
    ).unwrap();
}

/// Validate NPI format (10 digits)
pub fn validate_npi(npi: &str) -> Result<()> {
    if NPI_REGEX.is_match(npi) {
        Ok(())
    } else {
        Err(Error::Validation(format!("Invalid NPI format: {}", npi)))
    }
}

/// Validate ICD-10-CM diagnosis code format
pub fn validate_icd10(code: &str) -> Result<()> {
    if ICD10_REGEX.is_match(code) {
        Ok(())
    } else {
        Err(Error::Validation(format!("Invalid ICD-10-CM code format: {}", code)))
    }
}

/// Validate CPT or HCPCS procedure code
pub fn validate_procedure_code(code: &str) -> Result<()> {
    if CPT_REGEX.is_match(code) || HCPCS_REGEX.is_match(code) {
        Ok(())
    } else {
        Err(Error::Validation(format!("Invalid procedure code format: {}", code)))
    }
}

/// Validate modifier format
pub fn validate_modifier(modifier: &str) -> Result<()> {
    if MODIFIER_REGEX.is_match(modifier) {
        Ok(())
    } else {
        Err(Error::Validation(format!("Invalid modifier format: {}", modifier)))
    }
}

/// Validate Medicare Beneficiary Identifier (MBI)
pub fn validate_mbi(mbi: &str) -> Result<()> {
    // Remove hyphens if present
    let clean_mbi = mbi.replace('-', "");

    if clean_mbi.len() != 11 {
        return Err(Error::Validation(format!("MBI must be 11 characters: {}", mbi)));
    }

    if MBI_REGEX.is_match(&clean_mbi) {
        Ok(())
    } else {
        Err(Error::Validation(format!("Invalid MBI format: {}", mbi)))
    }
}

/// Validate Place of Service code
pub fn validate_pos(pos: &str) -> Result<()> {
    if POS_REGEX.is_match(pos) {
        Ok(())
    } else {
        Err(Error::Validation(format!("Invalid Place of Service code: {}", pos)))
    }
}

/// Validate email address
pub fn validate_email(email: &str) -> Result<()> {
    if EMAIL_REGEX.is_match(email) {
        Ok(())
    } else {
        Err(Error::Validation(format!("Invalid email address: {}", email)))
    }
}

/// Validate date range (from <= to)
pub fn validate_date_range(from: chrono::NaiveDate, to: Option<chrono::NaiveDate>) -> Result<()> {
    if let Some(to_date) = to {
        if from > to_date {
            return Err(Error::Validation(
                "Date from must be before or equal to date to".to_string()
            ));
        }
    }
    Ok(())
}

/// Validate amount is non-negative
pub fn validate_non_negative(amount: rust_decimal::Decimal, field_name: &str) -> Result<()> {
    if amount < rust_decimal::Decimal::ZERO {
        return Err(Error::Validation(format!("{} cannot be negative", field_name)));
    }
    Ok(())
}

/// Validate amount is positive
pub fn validate_positive(amount: rust_decimal::Decimal, field_name: &str) -> Result<()> {
    if amount <= rust_decimal::Decimal::ZERO {
        return Err(Error::Validation(format!("{} must be positive", field_name)));
    }
    Ok(())
}

/// Validate string length
pub fn validate_length(value: &str, min: usize, max: usize, field_name: &str) -> Result<()> {
    let len = value.len();
    if len < min || len > max {
        return Err(Error::Validation(
            format!("{} length must be between {} and {} characters, got {}", field_name, min, max, len)
        ));
    }
    Ok(())
}

/// Validate required field is not empty
pub fn validate_required(value: &str, field_name: &str) -> Result<()> {
    if value.trim().is_empty() {
        return Err(Error::Validation(format!("{} is required", field_name)));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_npi() {
        assert!(validate_npi("1234567890").is_ok());
        assert!(validate_npi("123456789").is_err()); // Too short
        assert!(validate_npi("12345678901").is_err()); // Too long
        assert!(validate_npi("12345ABCDE").is_err()); // Contains letters
    }

    #[test]
    fn test_validate_icd10() {
        assert!(validate_icd10("J20").is_ok());
        assert!(validate_icd10("J20.9").is_ok());
        assert!(validate_icd10("Z12.31").is_ok());
        assert!(validate_icd10("S06.0X0A").is_ok());
        assert!(validate_icd10("S93.401A").is_ok()); // 7th character extension
        assert!(validate_icd10("T14.90XA").is_ok()); // 7th character extension
        assert!(validate_icd10("S83.201A").is_ok()); // 7th character extension
        assert!(validate_icd10("20").is_err()); // Missing letter
        assert!(validate_icd10("J2").is_err()); // Too short
    }

    #[test]
    fn test_validate_procedure_code() {
        assert!(validate_procedure_code("99213").is_ok()); // CPT
        assert!(validate_procedure_code("J3490").is_ok()); // HCPCS
        assert!(validate_procedure_code("9921").is_err()); // Too short
        assert!(validate_procedure_code("992133").is_err()); // Too long
    }

    #[test]
    fn test_validate_modifier() {
        assert!(validate_modifier("25").is_ok());
        assert!(validate_modifier("GT").is_ok());
        assert!(validate_modifier("2").is_err()); // Too short
        assert!(validate_modifier("ABC").is_err()); // Too long
    }

    #[test]
    fn test_validate_mbi() {
        assert!(validate_mbi("1EG4TE5MK73").is_ok());
        assert!(validate_mbi("1EG4-TE5-MK73").is_ok()); // With hyphens
        assert!(validate_mbi("1234567890A").is_err()); // Wrong format
        assert!(validate_mbi("1EG4TE5MK7").is_err()); // Too short
    }

    #[test]
    fn test_validate_pos() {
        assert!(validate_pos("11").is_ok());
        assert!(validate_pos("21").is_ok());
        assert!(validate_pos("1").is_err()); // Too short
        assert!(validate_pos("111").is_err()); // Too long
    }

    #[test]
    fn test_validate_email() {
        assert!(validate_email("user@example.com").is_ok());
        assert!(validate_email("user.name+tag@example.co.uk").is_ok());
        assert!(validate_email("invalid.email").is_err());
        assert!(validate_email("@example.com").is_err());
    }
}
