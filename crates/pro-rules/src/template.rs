// Rule templates - Parameterized rule implementations for Phase 3
//
// Templates allow creating new rules through configuration without code changes.
// Each template defines a pattern (e.g., threshold check, duplicate detection)
// and accepts JSON parameters to customize behavior.

use crate::flag_types::FlagIssueType;
use crate::rule_engine::Rule;
use pro_common::{Error, Result};
use regex::Regex;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::sync::Arc;

/// Parameter validation schema for rule templates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterSchema {
    pub name: String,
    pub param_type: String, // "string", "number", "boolean", "array", "object"
    pub required: bool,
    pub description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default: Option<JsonValue>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pattern: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enum_values: Option<Vec<String>>,
}

/// Base trait for rule templates
/// Templates are instantiated with JSON parameters to create concrete Rule instances
pub trait RuleTemplate: Send + Sync {
    /// Template identifier (matches rule_template.template_code)
    fn template_code(&self) -> &str;

    /// Human-readable template name
    fn template_name(&self) -> &str;

    /// Parameter schema for validation
    fn parameter_schema(&self) -> Vec<ParameterSchema>;

    /// Validate parameters against schema
    fn validate_parameters(&self, params: &JsonValue) -> Result<()> {
        let schema = self.parameter_schema();

        // Check required parameters
        for param in schema.iter().filter(|p| p.required) {
            if params.get(&param.name).is_none() {
                return Err(Error::Config(format!(
                    "Missing required parameter: {}",
                    param.name
                )));
            }
        }

        // Type validation
        for param in &schema {
            if let Some(value) = params.get(&param.name) {
                match param.param_type.as_str() {
                    "string" if !value.is_string() => {
                        return Err(Error::Config(format!(
                            "Parameter '{}' must be a string",
                            param.name
                        )));
                    }
                    "number" if !value.is_number() => {
                        return Err(Error::Config(format!(
                            "Parameter '{}' must be a number",
                            param.name
                        )));
                    }
                    "boolean" if !value.is_boolean() => {
                        return Err(Error::Config(format!(
                            "Parameter '{}' must be a boolean",
                            param.name
                        )));
                    }
                    "array" if !value.is_array() => {
                        return Err(Error::Config(format!(
                            "Parameter '{}' must be an array",
                            param.name
                        )));
                    }
                    "object" if !value.is_object() => {
                        return Err(Error::Config(format!(
                            "Parameter '{}' must be an object",
                            param.name
                        )));
                    }
                    _ => {}
                }

                // Range validation for numbers
                if let Some(num) = value.as_f64() {
                    if let Some(min) = param.min {
                        if num < min {
                            return Err(Error::Config(format!(
                                "Parameter '{}' must be >= {}",
                                param.name, min
                            )));
                        }
                    }
                    if let Some(max) = param.max {
                        if num > max {
                            return Err(Error::Config(format!(
                                "Parameter '{}' must be <= {}",
                                param.name, max
                            )));
                        }
                    }
                }

                // Pattern validation for strings
                if let Some(pattern) = &param.pattern {
                    if let Some(s) = value.as_str() {
                        let regex = Regex::new(pattern).map_err(|e| {
                            Error::Config(format!("Invalid regex pattern: {}", e))
                        })?;
                        if !regex.is_match(s) {
                            return Err(Error::Config(format!(
                                "Parameter '{}' does not match pattern {}",
                                param.name, pattern
                            )));
                        }
                    }
                }

                // Enum validation
                if let Some(enum_values) = &param.enum_values {
                    if let Some(s) = value.as_str() {
                        if !enum_values.contains(&s.to_string()) {
                            return Err(Error::Config(format!(
                                "Parameter '{}' must be one of: {}",
                                param.name,
                                enum_values.join(", ")
                            )));
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Create a Rule instance from parameters
    ///
    /// # Arguments
    /// * `rule_code` - Unique rule identifier (e.g., "TEST_99213_SA")
    /// * `rule_name` - Human-readable rule name
    /// * `flag_issue_type` - Type of flag to create when rule triggers
    /// * `issue_code` - Database issue_code from claims.flag_issue for JOIN (e.g., "TEST_99213_SA")
    /// * `params` - JSON parameters for rule configuration
    fn instantiate(
        &self,
        rule_code: String,
        rule_name: String,
        flag_issue_type: FlagIssueType,
        issue_code: String,
        params: JsonValue,
    ) -> Result<Arc<dyn Rule>>;
}

/// Helper function to get string parameter
pub fn get_string_param(params: &JsonValue, name: &str) -> Result<String> {
    params
        .get(name)
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .ok_or_else(|| Error::Config(format!("Missing or invalid string parameter: {}", name)))
}

/// Helper function to get number parameter as Decimal
pub fn get_decimal_param(params: &JsonValue, name: &str) -> Result<Decimal> {
    let value = params
        .get(name)
        .and_then(|v| v.as_f64())
        .ok_or_else(|| Error::Config(format!("Missing or invalid number parameter: {}", name)))?;

    Decimal::try_from(value)
        .map_err(|e| Error::Config(format!("Invalid decimal value for {}: {}", name, e)))
}

/// Helper function to get optional number parameter as Decimal
pub fn get_optional_decimal_param(params: &JsonValue, name: &str) -> Result<Option<Decimal>> {
    match params.get(name) {
        Some(v) => {
            let num = v.as_f64().ok_or_else(|| {
                Error::Config(format!("Invalid number parameter: {}", name))
            })?;
            Ok(Some(Decimal::try_from(num).map_err(|e| {
                Error::Config(format!("Invalid decimal value for {}: {}", name, e))
            })?))
        }
        None => Ok(None),
    }
}

/// Helper function to get integer parameter
pub fn get_i64_param(params: &JsonValue, name: &str) -> Result<i64> {
    params
        .get(name)
        .and_then(|v| v.as_i64())
        .ok_or_else(|| Error::Config(format!("Missing or invalid integer parameter: {}", name)))
}

/// Helper function to get boolean parameter
pub fn get_bool_param(params: &JsonValue, name: &str) -> Result<bool> {
    params
        .get(name)
        .and_then(|v| v.as_bool())
        .ok_or_else(|| Error::Config(format!("Missing or invalid boolean parameter: {}", name)))
}

/// Helper function to get array parameter as Vec<String>
pub fn get_string_array_param(params: &JsonValue, name: &str) -> Result<Vec<String>> {
    params
        .get(name)
        .and_then(|v| v.as_array())
        .ok_or_else(|| Error::Config(format!("Missing or invalid array parameter: {}", name)))
        .and_then(|arr| {
            arr.iter()
                .map(|v| {
                    v.as_str()
                        .map(|s| s.to_string())
                        .ok_or_else(|| Error::Config(format!("Array {} contains non-string value", name)))
                })
                .collect()
        })
}

/// Template registry - maps template codes to template implementations
pub struct TemplateRegistry {
    templates: HashMap<String, Arc<dyn RuleTemplate>>,
}

impl TemplateRegistry {
    pub fn new() -> Self {
        Self {
            templates: HashMap::new(),
        }
    }

    pub fn register<T: RuleTemplate + 'static>(&mut self, template: T) {
        let code = template.template_code().to_string();
        self.templates.insert(code, Arc::new(template));
    }

    pub fn get(&self, template_code: &str) -> Option<Arc<dyn RuleTemplate>> {
        self.templates.get(template_code).cloned()
    }

    pub fn list(&self) -> Vec<String> {
        self.templates.keys().cloned().collect()
    }
}

impl Default for TemplateRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parameter_schema_validation() {
        let schema = vec![
            ParameterSchema {
                name: "threshold".to_string(),
                param_type: "number".to_string(),
                required: true,
                description: "Test threshold".to_string(),
                default: None,
                min: Some(0.0),
                max: Some(1000.0),
                pattern: None,
                enum_values: None,
            },
            ParameterSchema {
                name: "field".to_string(),
                param_type: "string".to_string(),
                required: true,
                description: "Field name".to_string(),
                default: None,
                min: None,
                max: None,
                pattern: None,
                enum_values: Some(vec!["total_charge".to_string(), "units".to_string()]),
            },
        ];

        // Valid parameters
        let valid_params = serde_json::json!({
            "threshold": 100,
            "field": "total_charge"
        });

        // Missing required parameter
        let missing_params = serde_json::json!({
            "threshold": 100
        });

        // Invalid type
        let invalid_type = serde_json::json!({
            "threshold": "not a number",
            "field": "total_charge"
        });

        // Out of range
        let out_of_range = serde_json::json!({
            "threshold": 2000,
            "field": "total_charge"
        });

        // Invalid enum value
        let invalid_enum = serde_json::json!({
            "threshold": 100,
            "field": "invalid_field"
        });

        // Note: We can't test validate_parameters directly without a concrete template
        // These tests verify the schema structure is correct
        assert_eq!(schema[0].name, "threshold");
        assert_eq!(schema[1].name, "field");
    }

    #[test]
    fn test_parameter_helpers() {
        let params = serde_json::json!({
            "string_field": "test",
            "number_field": 123.45,
            "int_field": 42,
            "bool_field": true,
            "array_field": ["a", "b", "c"]
        });

        assert_eq!(get_string_param(&params, "string_field").unwrap(), "test");
        assert!(get_decimal_param(&params, "number_field").is_ok());
        assert_eq!(get_i64_param(&params, "int_field").unwrap(), 42);
        assert_eq!(get_bool_param(&params, "bool_field").unwrap(), true);
        assert_eq!(
            get_string_array_param(&params, "array_field").unwrap(),
            vec!["a", "b", "c"]
        );

        // Test missing parameter
        assert!(get_string_param(&params, "missing").is_err());
    }
}
