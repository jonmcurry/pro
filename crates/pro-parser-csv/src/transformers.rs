// Field transformation functions for CSV parsing

use crate::mapping::TransformationType;
use chrono::NaiveDate;
use pro_common::{Error, Result};
use regex::Regex;

/// Apply transformation to a field value
pub fn apply_transformation(value: &str, transformation: &TransformationType) -> Result<String> {
    match transformation {
        TransformationType::Uppercase => Ok(value.to_uppercase()),

        TransformationType::Lowercase => Ok(value.to_lowercase()),

        TransformationType::Trim => Ok(value.trim().to_string()),

        TransformationType::RemoveSpaces => Ok(value.replace(' ', "")),

        TransformationType::RemoveNonAlphanumeric => {
            Ok(value.chars().filter(|c| c.is_alphanumeric()).collect())
        }

        TransformationType::PadLeft { length, pad_char } => {
            let mut result = value.to_string();
            while result.len() < *length {
                result.insert(0, *pad_char);
            }
            Ok(result)
        }

        TransformationType::PadRight { length, pad_char } => {
            let mut result = value.to_string();
            while result.len() < *length {
                result.push(*pad_char);
            }
            Ok(result)
        }

        TransformationType::Replace { pattern, replacement } => {
            Ok(value.replace(pattern, replacement))
        }

        TransformationType::Split { delimiter, take_index } => {
            let parts: Vec<&str> = value.split(delimiter.as_str()).collect();
            parts.get(*take_index)
                .map(|s| s.to_string())
                .ok_or_else(|| Error::Parse(format!("Split index {} out of bounds", take_index)))
        }

        TransformationType::DateFormat { from_format, to_format } => {
            transform_date_format(value, from_format, to_format)
        }

        TransformationType::Concat { fields: _, separator: _ } => {
            // This transformation requires multiple fields, handled separately
            Ok(value.to_string())
        }

        TransformationType::Custom { function_name } => {
            apply_custom_transformation(value, function_name)
        }
    }
}

/// Transform date from one format to another
fn transform_date_format(value: &str, from_format: &str, to_format: &str) -> Result<String> {
    // Parse common date formats
    let date = parse_date_flexible(value, from_format)?;

    // Format according to target format
    let formatted = match to_format {
        "YYYY-MM-DD" | "ISO" => date.format("%Y-%m-%d").to_string(),
        "MM/DD/YYYY" | "US" => date.format("%m/%d/%Y").to_string(),
        "DD/MM/YYYY" | "EU" => date.format("%d/%m/%Y").to_string(),
        "YYYYMMDD" | "COMPACT" => date.format("%Y%m%d").to_string(),
        "MMDDYYYY" => date.format("%m%d%Y").to_string(),
        _ => return Err(Error::Parse(format!("Unknown date format: {}", to_format))),
    };

    Ok(formatted)
}

/// Parse date from various formats
fn parse_date_flexible(value: &str, format_hint: &str) -> Result<NaiveDate> {
    // Try parsing based on format hint
    let date = match format_hint {
        "YYYY-MM-DD" | "ISO" => {
            NaiveDate::parse_from_str(value, "%Y-%m-%d")
        }
        "MM/DD/YYYY" | "US" => {
            NaiveDate::parse_from_str(value, "%m/%d/%Y")
        }
        "DD/MM/YYYY" | "EU" => {
            NaiveDate::parse_from_str(value, "%d/%m/%Y")
        }
        "YYYYMMDD" | "COMPACT" => {
            NaiveDate::parse_from_str(value, "%Y%m%d")
        }
        "MMDDYYYY" => {
            NaiveDate::parse_from_str(value, "%m%d%Y")
        }
        "M/D/YYYY" => {
            NaiveDate::parse_from_str(value, "%-m/%-d/%Y")
        }
        _ => {
            // Try multiple formats automatically
            parse_date_auto(value)
        }
    };

    date.or_else(|_| parse_date_auto(value))
        .map_err(|_| Error::Parse(format!("Unable to parse date: {}", value)))
}

/// Automatically detect and parse date format
fn parse_date_auto(value: &str) -> chrono::format::ParseResult<NaiveDate> {
    // Try common formats in order of likelihood
    let formats = vec![
        "%Y-%m-%d",      // 2024-01-15
        "%m/%d/%Y",      // 01/15/2024
        "%d/%m/%Y",      // 15/01/2024
        "%Y%m%d",        // 20240115
        "%m-%d-%Y",      // 01-15-2024
        "%d-%m-%Y",      // 15-01-2024
        "%-m/%-d/%Y",    // 1/15/2024 (no leading zero)
        "%B %d, %Y",     // January 15, 2024
        "%b %d, %Y",     // Jan 15, 2024
    ];

    for format in formats {
        if let Ok(date) = NaiveDate::parse_from_str(value, format) {
            return Ok(date);
        }
    }

    // Return a parse error using NaiveDate::parse_from_str with an invalid format
    NaiveDate::parse_from_str(value, "%Y-%m-%d") // This will fail and return proper ParseError
}

/// Apply custom transformation functions
fn apply_custom_transformation(value: &str, function_name: &str) -> Result<String> {
    match function_name {
        "normalize_npi" => normalize_npi(value),
        "normalize_phone" => normalize_phone(value),
        "extract_digits" => Ok(extract_digits(value)),
        "capitalize_name" => Ok(capitalize_name(value)),
        "clean_icd10" => clean_icd10(value),
        "standardize_gender" => standardize_gender(value),
        "format_mbi" => format_mbi(value),
        _ => Err(Error::Parse(format!("Unknown custom function: {}", function_name))),
    }
}

/// Normalize NPI to 10 digits
fn normalize_npi(value: &str) -> Result<String> {
    let digits = extract_digits(value);

    if digits.len() == 10 {
        Ok(digits)
    } else if digits.len() < 10 {
        // Pad with leading zeros
        Ok(format!("{:0>10}", digits))
    } else {
        Err(Error::Parse(format!("Invalid NPI: {}", value)))
    }
}

/// Normalize phone number
fn normalize_phone(value: &str) -> Result<String> {
    let digits = extract_digits(value);

    if digits.len() == 10 {
        Ok(format!("({}) {}-{}", &digits[0..3], &digits[3..6], &digits[6..10]))
    } else if digits.len() == 11 && digits.starts_with('1') {
        // Remove leading 1
        let without_one = &digits[1..];
        Ok(format!("({}) {}-{}", &without_one[0..3], &without_one[3..6], &without_one[6..10]))
    } else {
        Ok(digits) // Return as-is if not standard format
    }
}

/// Extract only digits from string
fn extract_digits(value: &str) -> String {
    value.chars().filter(|c| c.is_ascii_digit()).collect()
}

/// Capitalize name properly (first letter uppercase, rest lowercase)
fn capitalize_name(value: &str) -> String {
    value.split_whitespace()
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                None => String::new(),
                Some(first) => {
                    first.to_uppercase().collect::<String>() + &chars.as_str().to_lowercase()
                }
            }
        })
        .collect::<Vec<String>>()
        .join(" ")
}

/// Clean ICD-10 code (remove periods, standardize)
fn clean_icd10(value: &str) -> Result<String> {
    // Remove all non-alphanumeric except period
    let cleaned: String = value.chars()
        .filter(|c| c.is_alphanumeric() || *c == '.')
        .collect();

    // Ensure proper format: Letter + 2-6 characters, with optional period
    if cleaned.is_empty() {
        return Err(Error::Parse("Empty ICD-10 code".to_string()));
    }

    // Remove existing periods
    let no_period = cleaned.replace('.', "");

    // Add period after 3rd character if code is longer than 3 characters
    if no_period.len() > 3 {
        Ok(format!("{}.{}", &no_period[..3], &no_period[3..]))
    } else {
        Ok(no_period)
    }
}

/// Standardize gender codes
fn standardize_gender(value: &str) -> Result<String> {
    let normalized = value.trim().to_uppercase();

    match normalized.as_str() {
        "M" | "MALE" | "MAN" => Ok("M".to_string()),
        "F" | "FEMALE" | "WOMAN" => Ok("F".to_string()),
        "U" | "UNKNOWN" | "UNK" | "OTHER" => Ok("U".to_string()),
        _ => Err(Error::Parse(format!("Invalid gender code: {}", value))),
    }
}

/// Format MBI with hyphens
fn format_mbi(value: &str) -> Result<String> {
    // Remove existing hyphens and spaces
    let clean = value.replace('-', "").replace(' ', "");

    if clean.len() != 11 {
        return Err(Error::Parse(format!("MBI must be 11 characters, got {}", clean.len())));
    }

    // Format as 1234-567-8901
    Ok(format!("{}-{}-{}", &clean[0..4], &clean[4..7], &clean[7..11]))
}

/// Apply multiple transformations in sequence
pub fn apply_transformations(value: &str, transformations: &[TransformationType]) -> Result<String> {
    let mut result = value.to_string();

    for transformation in transformations {
        result = apply_transformation(&result, transformation)?;
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uppercase() {
        let result = apply_transformation("test", &TransformationType::Uppercase).unwrap();
        assert_eq!(result, "TEST");
    }

    #[test]
    fn test_trim() {
        let result = apply_transformation("  test  ", &TransformationType::Trim).unwrap();
        assert_eq!(result, "test");
    }

    #[test]
    fn test_remove_spaces() {
        let result = apply_transformation("hello world", &TransformationType::RemoveSpaces).unwrap();
        assert_eq!(result, "helloworld");
    }

    #[test]
    fn test_pad_left() {
        let result = apply_transformation("123", &TransformationType::PadLeft { length: 5, pad_char: '0' }).unwrap();
        assert_eq!(result, "00123");
    }

    #[test]
    fn test_extract_digits() {
        assert_eq!(extract_digits("ABC-123-XYZ-456"), "123456");
        assert_eq!(extract_digits("(555) 123-4567"), "5551234567");
    }

    #[test]
    fn test_capitalize_name() {
        assert_eq!(capitalize_name("john doe"), "John Doe");
        assert_eq!(capitalize_name("JANE SMITH"), "Jane Smith");
        assert_eq!(capitalize_name("mary-kate olsen"), "Mary-kate Olsen");
    }

    #[test]
    fn test_clean_icd10() {
        assert_eq!(clean_icd10("J20.9").unwrap(), "J20.9");
        assert_eq!(clean_icd10("J209").unwrap(), "J20.9");
        assert_eq!(clean_icd10("E11.9").unwrap(), "E11.9");
        assert_eq!(clean_icd10("Z12.31").unwrap(), "Z12.31");
    }

    #[test]
    fn test_standardize_gender() {
        assert_eq!(standardize_gender("M").unwrap(), "M");
        assert_eq!(standardize_gender("Male").unwrap(), "M");
        assert_eq!(standardize_gender("F").unwrap(), "F");
        assert_eq!(standardize_gender("Female").unwrap(), "F");
        assert_eq!(standardize_gender("Unknown").unwrap(), "U");
        assert!(standardize_gender("Invalid").is_err());
    }

    #[test]
    fn test_normalize_npi() {
        assert_eq!(normalize_npi("1234567890").unwrap(), "1234567890");
        assert_eq!(normalize_npi("123456789").unwrap(), "0123456789");
        assert!(normalize_npi("12345678901").is_err());
    }

    #[test]
    fn test_parse_date_auto() {
        assert!(parse_date_auto("2024-01-15").is_ok());
        assert!(parse_date_auto("01/15/2024").is_ok());
        assert!(parse_date_auto("20240115").is_ok());
        assert!(parse_date_auto("1/15/2024").is_ok());
    }

    #[test]
    fn test_date_format_transformation() {
        let result = transform_date_format("01/15/2024", "MM/DD/YYYY", "YYYY-MM-DD").unwrap();
        assert_eq!(result, "2024-01-15");

        let result = transform_date_format("2024-01-15", "YYYY-MM-DD", "MM/DD/YYYY").unwrap();
        assert_eq!(result, "01/15/2024");
    }
}
