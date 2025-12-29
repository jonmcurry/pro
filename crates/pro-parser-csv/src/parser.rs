// Main CSV parser with dynamic header mapping

use crate::detector::{FormatDetector, DetectionResult};
use crate::mapping::{HeaderMapping, FieldMapping, DataType, ValidationRule};
use crate::transformers::apply_transformation;
use chrono::NaiveDate;
use csv::{Reader, StringRecord};
use pro_common::{Error, Result};
use pro_common::validation::*;
use rust_decimal::Decimal;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::str::FromStr;


/// Parsed CSV row with mapped fields
#[derive(Debug, Clone)]
pub struct ParsedRow {
    pub row_number: usize,
    pub encounter_fields: HashMap<String, String>,
    pub service_line_fields: HashMap<String, String>,
    pub diagnosis_fields: HashMap<String, Vec<String>>,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

/// CSV parser with dynamic header mapping
pub struct CsvParser {
    mapping: HeaderMapping,
    detector: FormatDetector,
    auto_detect: bool,
}

impl CsvParser {
    /// Create new parser with specific mapping
    pub fn new(mapping: HeaderMapping) -> Self {
        Self {
            mapping,
            detector: FormatDetector::new(),
            auto_detect: false,
        }
    }

    /// Create new parser with auto-detection
    pub fn with_auto_detection() -> Self {
        Self {
            mapping: HeaderMapping::new("Auto".to_string(), "AUTO".to_string()),
            detector: FormatDetector::new(),
            auto_detect: true,
        }
    }

    /// Parse CSV file
    pub fn parse_file(&mut self, file_path: &str) -> Result<Vec<ParsedRow>> {
        let file = File::open(file_path)
            .map_err(|e| Error::Io(e))?;
        let reader = BufReader::new(file);
        self.parse_reader(reader)
    }

    /// Parse CSV from reader
    pub fn parse_reader<R: std::io::Read>(&mut self, reader: R) -> Result<Vec<ParsedRow>> {
        let mut csv_reader = Reader::from_reader(reader);

        // Get headers
        let headers = csv_reader.headers()
            .map_err(|e| Error::CsvParse(e.to_string()))?
            .clone();

        let header_strings: Vec<String> = headers.iter().map(|s| s.to_string()).collect();

        // Auto-detect format if enabled
        if self.auto_detect {
            let detection = self.detector.detect(&header_strings)?;
            self.mapping = detection.suggested_mapping;
        }

        // Create header lookup map
        let header_lookup = self.mapping.to_lookup_map();

        // Parse all rows
        let mut parsed_rows = Vec::new();
        let mut row_number = 0;

        for result in csv_reader.records() {
            row_number += 1;

            let record = result.map_err(|e| Error::CsvParse(e.to_string()))?;
            let parsed_row = self.parse_row(&record, &header_strings, &header_lookup, row_number)?;
            parsed_rows.push(parsed_row);
        }

        Ok(parsed_rows)
    }

    /// Parse single CSV row
    fn parse_row(
        &self,
        record: &StringRecord,
        headers: &[String],
        header_lookup: &HashMap<String, FieldMapping>,
        row_number: usize,
    ) -> Result<ParsedRow> {
        let mut parsed = ParsedRow {
            row_number,
            encounter_fields: HashMap::new(),
            service_line_fields: HashMap::new(),
            diagnosis_fields: HashMap::new(),
            errors: Vec::new(),
            warnings: Vec::new(),
        };

        // Process each field
        for (idx, header) in headers.iter().enumerate() {
            let value = record.get(idx).unwrap_or("");

            // Skip empty values unless required
            if value.trim().is_empty() {
                if let Some(mapping) = header_lookup.get(&header.to_lowercase()) {
                    if mapping.is_required {
                        parsed.errors.push(format!("Required field '{}' is empty", header));
                    }
                }
                continue;
            }

            // Get field mapping
            if let Some(mapping) = header_lookup.get(&header.to_lowercase()) {
                match self.process_field(value, mapping, &mut parsed) {
                    Ok(_) => {},
                    Err(e) => {
                        parsed.errors.push(format!("Field '{}': {}", header, e));
                    }
                }
            } else {
                parsed.warnings.push(format!("Unrecognized header: '{}'", header));
            }
        }

        // Apply transformations
        self.apply_row_transformations(&mut parsed);

        Ok(parsed)
    }

    /// Process individual field
    fn process_field(
        &self,
        value: &str,
        mapping: &FieldMapping,
        parsed: &mut ParsedRow,
    ) -> Result<()> {
        // Validate field
        self.validate_field(value, mapping)?;

        // Parse and convert based on data type
        let converted_value = self.convert_value(value, &mapping.data_type)?;

        // Store in appropriate bucket
        match mapping.target_table.as_str() {
            "encounter" => {
                parsed.encounter_fields.insert(mapping.target_field.clone(), converted_value);
            }
            "service_line" => {
                parsed.service_line_fields.insert(mapping.target_field.clone(), converted_value);
            }
            "encounter_diagnosis" => {
                // Handle diagnosis fields specially (can have multiple)
                parsed.diagnosis_fields
                    .entry(mapping.target_field.clone())
                    .or_insert_with(Vec::new)
                    .push(converted_value);
            }
            _ => {
                return Err(Error::Parse(format!("Unknown target table: {}", mapping.target_table)));
            }
        }

        Ok(())
    }

    /// Validate field value
    fn validate_field(&self, value: &str, mapping: &FieldMapping) -> Result<()> {
        for rule in &mapping.validation_rules {
            match rule {
                ValidationRule::NotEmpty => {
                    if value.trim().is_empty() {
                        return Err(Error::Validation("Field cannot be empty".to_string()));
                    }
                }
                ValidationRule::MinLength(min) => {
                    if value.len() < *min {
                        return Err(Error::Validation(format!("Minimum length is {}", min)));
                    }
                }
                ValidationRule::MaxLength(max) => {
                    if value.len() > *max {
                        return Err(Error::Validation(format!("Maximum length is {}", max)));
                    }
                }
                ValidationRule::Regex(pattern) => {
                    let re = regex::Regex::new(pattern)
                        .map_err(|_| Error::Validation("Invalid regex pattern".to_string()))?;
                    if !re.is_match(value) {
                        return Err(Error::Validation("Value does not match required pattern".to_string()));
                    }
                }
                ValidationRule::Range { min, max } => {
                    let num = value.parse::<f64>()
                        .map_err(|_| Error::Validation("Invalid number".to_string()))?;
                    if num < *min || num > *max {
                        return Err(Error::Validation(format!("Value must be between {} and {}", min, max)));
                    }
                }
                ValidationRule::OneOf(allowed) => {
                    if !allowed.contains(&value.to_string()) {
                        return Err(Error::Validation(format!("Value must be one of: {:?}", allowed)));
                    }
                }
                ValidationRule::Npi => {
                    validate_npi(value)?;
                }
                ValidationRule::Icd10 => {
                    validate_icd10(value)?;
                }
                ValidationRule::CptHcpcs => {
                    validate_procedure_code(value)?;
                }
                ValidationRule::Mbi => {
                    validate_mbi(value)?;
                }
            }
        }

        Ok(())
    }

    /// Convert value to appropriate type
    fn convert_value(&self, value: &str, data_type: &DataType) -> Result<String> {
        match data_type {
            DataType::String => Ok(value.trim().to_string()),

            DataType::Integer => {
                value.trim().parse::<i64>()
                    .map(|v| v.to_string())
                    .map_err(|_| Error::Parse(format!("Invalid integer: {}", value)))
            }

            DataType::Decimal => {
                // Remove currency symbols and commas
                let cleaned = value.trim().replace('$', "").replace(',', "");
                Decimal::from_str(&cleaned)
                    .map(|v| v.to_string())
                    .map_err(|_| Error::Parse(format!("Invalid decimal: {}", value)))
            }

            DataType::Date => {
                // Try to parse date in various formats
                self.parse_date(value)
                    .map(|d| d.format("%Y-%m-%d").to_string())
            }

            DataType::DateTime => {
                // Parse datetime
                self.parse_datetime(value)
            }

            DataType::Boolean => {
                let normalized = value.trim().to_lowercase();
                match normalized.as_str() {
                    "true" | "yes" | "y" | "1" | "t" => Ok("true".to_string()),
                    "false" | "no" | "n" | "0" | "f" => Ok("false".to_string()),
                    _ => Err(Error::Parse(format!("Invalid boolean: {}", value)))
                }
            }

            DataType::Uuid => {
                // Note: UUID type now represents BIGINT IDs in database
                value.trim().parse::<i64>()
                    .map(|v| v.to_string())
                    .map_err(|_| Error::Parse(format!("Invalid ID (expected integer): {}", value)))
            }
        }
    }

    /// Parse date from string
    fn parse_date(&self, value: &str) -> Result<NaiveDate> {
        crate::transformers::apply_transformation(value, &crate::mapping::TransformationType::DateFormat {
            from_format: "AUTO".to_string(),
            to_format: "YYYY-MM-DD".to_string(),
        }).and_then(|s| {
            NaiveDate::parse_from_str(&s, "%Y-%m-%d")
                .map_err(|_| Error::Parse(format!("Invalid date: {}", value)))
        })
    }

    /// Parse datetime from string
    fn parse_datetime(&self, value: &str) -> Result<String> {
        // Simplified datetime parsing
        Ok(value.trim().to_string())
    }

    /// Apply transformations to parsed row
    fn apply_row_transformations(&self, parsed: &mut ParsedRow) {
        for transformation in &self.mapping.transformations {
            let field_name = &transformation.target_field;

            // Check in encounter fields
            if let Some(value) = parsed.encounter_fields.get(field_name) {
                match apply_transformation(value, &transformation.transformation_type) {
                    Ok(transformed) => {
                        parsed.encounter_fields.insert(field_name.clone(), transformed);
                    }
                    Err(e) => {
                        parsed.warnings.push(format!("Transformation failed for '{}': {}", field_name, e));
                    }
                }
            }

            // Check in service line fields
            if let Some(value) = parsed.service_line_fields.get(field_name) {
                match apply_transformation(value, &transformation.transformation_type) {
                    Ok(transformed) => {
                        parsed.service_line_fields.insert(field_name.clone(), transformed);
                    }
                    Err(e) => {
                        parsed.warnings.push(format!("Transformation failed for '{}': {}", field_name, e));
                    }
                }
            }
        }
    }

    /// Get detection result for the current mapping
    pub fn detect_format(&self, file_path: &str) -> Result<DetectionResult> {
        let file = File::open(file_path)
            .map_err(|e| Error::Io(e))?;
        let reader = BufReader::new(file);
        let mut csv_reader = Reader::from_reader(reader);

        let headers = csv_reader.headers()
            .map_err(|e| Error::CsvParse(e.to_string()))?;

        let header_strings: Vec<String> = headers.iter().map(|s| s.to_string()).collect();
        self.detector.detect(&header_strings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mapping::PredefinedMappings;
    use std::io::Cursor;

    #[test]
    fn test_csv_parser_creation() {
        let mapping = PredefinedMappings::generic();
        let parser = CsvParser::new(mapping);
        assert_eq!(parser.mapping.source_system, "GENERIC");
    }

    #[test]
    fn test_parse_simple_csv() {
        let csv_data = "Patient Control Number,Date of Service,Procedure Code,Charge Amount\nPT123,2024-01-15,99213,150.00\n";
        let cursor = Cursor::new(csv_data);

        let mapping = PredefinedMappings::generic();
        let mut parser = CsvParser::new(mapping);

        let result = parser.parse_reader(cursor);
        assert!(result.is_ok());

        let rows = result.unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].row_number, 1);
        assert!(!rows[0].encounter_fields.is_empty());
    }

    #[test]
    fn test_convert_value_integer() {
        let mapping = PredefinedMappings::generic();
        let parser = CsvParser::new(mapping);

        let result = parser.convert_value("123", &DataType::Integer);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "123");
    }

    #[test]
    fn test_convert_value_decimal() {
        let mapping = PredefinedMappings::generic();
        let parser = CsvParser::new(mapping);

        let result = parser.convert_value("$1,234.56", &DataType::Decimal);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "1234.56");
    }

    #[test]
    fn test_convert_value_boolean() {
        let mapping = PredefinedMappings::generic();
        let parser = CsvParser::new(mapping);

        assert_eq!(parser.convert_value("true", &DataType::Boolean).unwrap(), "true");
        assert_eq!(parser.convert_value("yes", &DataType::Boolean).unwrap(), "true");
        assert_eq!(parser.convert_value("false", &DataType::Boolean).unwrap(), "false");
        assert_eq!(parser.convert_value("no", &DataType::Boolean).unwrap(), "false");
    }

    #[test]
    fn test_auto_detection() {
        let csv_data = "Patient ID,DOS,CPT,Units,Charges\nPT123,01/15/2024,99213,1,150.00\n";
        let cursor = Cursor::new(csv_data);

        let mut parser = CsvParser::with_auto_detection();
        let result = parser.parse_reader(cursor);

        assert!(result.is_ok());
        let rows = result.unwrap();
        assert_eq!(rows.len(), 1);
    }
}
