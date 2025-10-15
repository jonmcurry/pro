// Header mapping configuration for dynamic CSV parsing

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Complete header mapping configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeaderMapping {
    pub mapping_id: Option<uuid::Uuid>,
    pub mapping_name: String,
    pub source_system: String, // ATHENA, EPIC, CERNER, EXCEL, etc.
    pub field_mappings: Vec<FieldMapping>,
    pub transformations: Vec<Transformation>,
}

impl HeaderMapping {
    pub fn new(name: String, source: String) -> Self {
        Self {
            mapping_id: None,
            mapping_name: name,
            source_system: source,
            field_mappings: Vec::new(),
            transformations: Vec::new(),
        }
    }

    /// Add a field mapping
    pub fn add_mapping(&mut self, mapping: FieldMapping) {
        self.field_mappings.push(mapping);
    }

    /// Add a transformation
    pub fn add_transformation(&mut self, transformation: Transformation) {
        self.transformations.push(transformation);
    }

    /// Get mapping for a CSV header
    pub fn get_mapping(&self, csv_header: &str) -> Option<&FieldMapping> {
        self.field_mappings.iter().find(|m| {
            m.csv_header.eq_ignore_ascii_case(csv_header) ||
            m.alternate_headers.iter().any(|h| h.eq_ignore_ascii_case(csv_header))
        })
    }

    /// Convert to lookup map for fast access
    pub fn to_lookup_map(&self) -> HashMap<String, FieldMapping> {
        let mut map = HashMap::new();
        for mapping in &self.field_mappings {
            map.insert(mapping.csv_header.to_lowercase(), mapping.clone());
            for alt in &mapping.alternate_headers {
                map.insert(alt.to_lowercase(), mapping.clone());
            }
        }
        map
    }
}

/// Individual field mapping from CSV header to database field
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldMapping {
    pub csv_header: String,
    pub alternate_headers: Vec<String>,
    pub target_field: String,
    pub target_table: String,
    pub data_type: DataType,
    pub is_required: bool,
    pub default_value: Option<String>,
    pub validation_rules: Vec<ValidationRule>,
}

impl FieldMapping {
    pub fn new(csv_header: String, target_field: String, target_table: String, data_type: DataType) -> Self {
        Self {
            csv_header,
            alternate_headers: Vec::new(),
            target_field,
            target_table,
            data_type,
            is_required: false,
            default_value: None,
            validation_rules: Vec::new(),
        }
    }

    pub fn required(mut self) -> Self {
        self.is_required = true;
        self
    }

    pub fn with_default(mut self, default: String) -> Self {
        self.default_value = Some(default);
        self
    }

    pub fn with_alternate(mut self, alternate: String) -> Self {
        self.alternate_headers.push(alternate);
        self
    }

    pub fn with_validation(mut self, rule: ValidationRule) -> Self {
        self.validation_rules.push(rule);
        self
    }
}

/// Data types for field mapping
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DataType {
    String,
    Integer,
    Decimal,
    Date,
    DateTime,
    Boolean,
    Uuid,
}

/// Validation rules for fields
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationRule {
    MinLength(usize),
    MaxLength(usize),
    Regex(String),
    Range { min: f64, max: f64 },
    OneOf(Vec<String>),
    NotEmpty,
    Npi,
    Icd10,
    CptHcpcs,
    Mbi,
}

/// Transformation to apply to field values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transformation {
    pub target_field: String,
    pub transformation_type: TransformationType,
}

/// Types of transformations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransformationType {
    Uppercase,
    Lowercase,
    Trim,
    RemoveSpaces,
    RemoveNonAlphanumeric,
    PadLeft { length: usize, pad_char: char },
    PadRight { length: usize, pad_char: char },
    Replace { pattern: String, replacement: String },
    Split { delimiter: String, take_index: usize },
    DateFormat { from_format: String, to_format: String },
    Concat { fields: Vec<String>, separator: String },
    Custom { function_name: String },
}

/// Predefined mapping configurations for common EHR systems
pub struct PredefinedMappings;

impl PredefinedMappings {
    /// Athena Health CSV mapping
    pub fn athena() -> HeaderMapping {
        let mut mapping = HeaderMapping::new("Athena Health".to_string(), "ATHENA".to_string());

        mapping.add_mapping(
            FieldMapping::new("Patient ID".to_string(), "patient_control_number".to_string(), "encounter".to_string(), DataType::String)
                .required()
                .with_alternate("PatientID".to_string())
                .with_alternate("Patient #".to_string())
                .with_validation(ValidationRule::MaxLength(38))
        );

        mapping.add_mapping(
            FieldMapping::new("DOS".to_string(), "date_of_service_from".to_string(), "encounter".to_string(), DataType::Date)
                .required()
                .with_alternate("Date of Service".to_string())
                .with_alternate("Service Date".to_string())
        );

        mapping.add_mapping(
            FieldMapping::new("Provider NPI".to_string(), "rendering_provider_npi".to_string(), "encounter".to_string(), DataType::String)
                .with_alternate("Rendering NPI".to_string())
                .with_validation(ValidationRule::Npi)
        );

        mapping.add_mapping(
            FieldMapping::new("CPT".to_string(), "procedure_code".to_string(), "service_line".to_string(), DataType::String)
                .required()
                .with_alternate("CPT Code".to_string())
                .with_alternate("Procedure Code".to_string())
                .with_validation(ValidationRule::CptHcpcs)
        );

        mapping.add_mapping(
            FieldMapping::new("Modifier 1".to_string(), "procedure_modifier_1".to_string(), "service_line".to_string(), DataType::String)
                .with_alternate("Mod 1".to_string())
                .with_validation(ValidationRule::MaxLength(2))
        );

        mapping.add_mapping(
            FieldMapping::new("Units".to_string(), "service_unit_count".to_string(), "service_line".to_string(), DataType::Decimal)
                .required()
                .with_default("1".to_string())
                .with_validation(ValidationRule::Range { min: 0.0, max: 9999.9 })
        );

        mapping.add_mapping(
            FieldMapping::new("Charges".to_string(), "line_item_charge_amount".to_string(), "service_line".to_string(), DataType::Decimal)
                .required()
                .with_alternate("Charge Amount".to_string())
                .with_alternate("Billed Amount".to_string())
                .with_validation(ValidationRule::Range { min: 0.0, max: 999999.99 })
        );

        mapping.add_mapping(
            FieldMapping::new("Diagnosis 1".to_string(), "diagnosis_code".to_string(), "encounter_diagnosis".to_string(), DataType::String)
                .required()
                .with_alternate("DX1".to_string())
                .with_alternate("Primary Diagnosis".to_string())
                .with_validation(ValidationRule::Icd10)
        );

        mapping.add_mapping(
            FieldMapping::new("Patient Last Name".to_string(), "subscriber_last_name".to_string(), "encounter".to_string(), DataType::String)
                .required()
                .with_alternate("Last Name".to_string())
                .with_validation(ValidationRule::NotEmpty)
        );

        mapping.add_mapping(
            FieldMapping::new("Patient First Name".to_string(), "subscriber_first_name".to_string(), "encounter".to_string(), DataType::String)
                .required()
                .with_alternate("First Name".to_string())
                .with_validation(ValidationRule::NotEmpty)
        );

        mapping.add_mapping(
            FieldMapping::new("DOB".to_string(), "subscriber_birth_date".to_string(), "encounter".to_string(), DataType::Date)
                .required()
                .with_alternate("Date of Birth".to_string())
                .with_alternate("Birth Date".to_string())
        );

        mapping.add_mapping(
            FieldMapping::new("Gender".to_string(), "subscriber_gender".to_string(), "encounter".to_string(), DataType::String)
                .with_alternate("Sex".to_string())
                .with_validation(ValidationRule::OneOf(vec!["M".to_string(), "F".to_string(), "U".to_string()]))
        );

        mapping.add_mapping(
            FieldMapping::new("POS".to_string(), "place_of_service_code".to_string(), "encounter".to_string(), DataType::String)
                .with_alternate("Place of Service".to_string())
                .with_validation(ValidationRule::MaxLength(2))
        );

        mapping
    }

    /// Epic CSV mapping
    pub fn epic() -> HeaderMapping {
        let mut mapping = HeaderMapping::new("Epic".to_string(), "EPIC".to_string());

        mapping.add_mapping(
            FieldMapping::new("ACCOUNT NUMBER".to_string(), "patient_control_number".to_string(), "encounter".to_string(), DataType::String)
                .required()
                .with_alternate("Account".to_string())
        );

        mapping.add_mapping(
            FieldMapping::new("SERV DT".to_string(), "date_of_service_from".to_string(), "encounter".to_string(), DataType::Date)
                .required()
                .with_alternate("SERVICE DATE".to_string())
        );

        mapping.add_mapping(
            FieldMapping::new("PROC CD".to_string(), "procedure_code".to_string(), "service_line".to_string(), DataType::String)
                .required()
                .with_alternate("PROCEDURE".to_string())
        );

        mapping.add_mapping(
            FieldMapping::new("CHG".to_string(), "line_item_charge_amount".to_string(), "service_line".to_string(), DataType::Decimal)
                .required()
                .with_alternate("CHARGE".to_string())
        );

        mapping
    }

    /// Cerner CSV mapping
    pub fn cerner() -> HeaderMapping {
        let mut mapping = HeaderMapping::new("Cerner".to_string(), "CERNER".to_string());

        mapping.add_mapping(
            FieldMapping::new("Encounter_ID".to_string(), "patient_control_number".to_string(), "encounter".to_string(), DataType::String)
                .required()
                .with_alternate("EncounterID".to_string())
        );

        mapping.add_mapping(
            FieldMapping::new("Service_Date".to_string(), "date_of_service_from".to_string(), "encounter".to_string(), DataType::Date)
                .required()
        );

        mapping.add_mapping(
            FieldMapping::new("CPT_Code".to_string(), "procedure_code".to_string(), "service_line".to_string(), DataType::String)
                .required()
        );

        mapping
    }

    /// Generic/Excel CSV mapping (most flexible)
    pub fn generic() -> HeaderMapping {
        let mut mapping = HeaderMapping::new("Generic".to_string(), "GENERIC".to_string());

        // Add common field names with many alternates
        mapping.add_mapping(
            FieldMapping::new("Patient Control Number".to_string(), "patient_control_number".to_string(), "encounter".to_string(), DataType::String)
                .required()
                .with_alternate("Patient ID".to_string())
                .with_alternate("Account".to_string())
                .with_alternate("Encounter".to_string())
                .with_alternate("Claim".to_string())
        );

        mapping.add_mapping(
            FieldMapping::new("Date of Service".to_string(), "date_of_service_from".to_string(), "encounter".to_string(), DataType::Date)
                .required()
                .with_alternate("DOS".to_string())
                .with_alternate("Service Date".to_string())
                .with_alternate("Serv Date".to_string())
        );

        mapping.add_mapping(
            FieldMapping::new("Procedure Code".to_string(), "procedure_code".to_string(), "service_line".to_string(), DataType::String)
                .required()
                .with_alternate("CPT".to_string())
                .with_alternate("CPT Code".to_string())
                .with_alternate("HCPCS".to_string())
                .with_alternate("Procedure".to_string())
        );

        mapping.add_mapping(
            FieldMapping::new("Charge Amount".to_string(), "line_item_charge_amount".to_string(), "service_line".to_string(), DataType::Decimal)
                .required()
                .with_alternate("Charges".to_string())
                .with_alternate("Billed".to_string())
                .with_alternate("Amount".to_string())
        );

        mapping
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_mapping_creation() {
        let mapping = HeaderMapping::new("Test".to_string(), "TEST".to_string());
        assert_eq!(mapping.mapping_name, "Test");
        assert_eq!(mapping.source_system, "TEST");
        assert_eq!(mapping.field_mappings.len(), 0);
    }

    #[test]
    fn test_field_mapping_builder() {
        let mapping = FieldMapping::new(
            "Test Field".to_string(),
            "test_field".to_string(),
            "test_table".to_string(),
            DataType::String,
        )
        .required()
        .with_default("default".to_string())
        .with_alternate("Alt Field".to_string());

        assert_eq!(mapping.csv_header, "Test Field");
        assert!(mapping.is_required);
        assert_eq!(mapping.default_value, Some("default".to_string()));
        assert_eq!(mapping.alternate_headers.len(), 1);
    }

    #[test]
    fn test_get_mapping() {
        let mut header_mapping = HeaderMapping::new("Test".to_string(), "TEST".to_string());
        header_mapping.add_mapping(
            FieldMapping::new("Test".to_string(), "test".to_string(), "table".to_string(), DataType::String)
                .with_alternate("Alternative".to_string())
        );

        assert!(header_mapping.get_mapping("Test").is_some());
        assert!(header_mapping.get_mapping("test").is_some()); // Case insensitive
        assert!(header_mapping.get_mapping("Alternative").is_some());
        assert!(header_mapping.get_mapping("Unknown").is_none());
    }

    #[test]
    fn test_predefined_athena() {
        let mapping = PredefinedMappings::athena();
        assert_eq!(mapping.source_system, "ATHENA");
        assert!(!mapping.field_mappings.is_empty());
        assert!(mapping.get_mapping("Patient ID").is_some());
        assert!(mapping.get_mapping("DOS").is_some());
        assert!(mapping.get_mapping("CPT").is_some());
    }

    #[test]
    fn test_to_lookup_map() {
        let mut header_mapping = HeaderMapping::new("Test".to_string(), "TEST".to_string());
        header_mapping.add_mapping(
            FieldMapping::new("Field1".to_string(), "field1".to_string(), "table".to_string(), DataType::String)
                .with_alternate("Alt1".to_string())
        );

        let map = header_mapping.to_lookup_map();
        assert!(map.contains_key("field1"));
        assert!(map.contains_key("alt1"));
    }
}
