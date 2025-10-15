// Auto-detection of CSV format and best matching header mapping

use crate::mapping::{HeaderMapping, PredefinedMappings};
use pro_common::{Error, Result};
use std::collections::HashMap;

/// Detection result with confidence score
#[derive(Debug, Clone)]
pub struct DetectionResult {
    pub source_system: String,
    pub confidence: f64,
    pub matched_headers: usize,
    pub total_headers: usize,
    pub suggested_mapping: HeaderMapping,
}

/// CSV format detector
pub struct FormatDetector {
    predefined_mappings: Vec<HeaderMapping>,
}

impl Default for FormatDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl FormatDetector {
    pub fn new() -> Self {
        Self {
            predefined_mappings: vec![
                PredefinedMappings::athena(),
                PredefinedMappings::epic(),
                PredefinedMappings::cerner(),
                PredefinedMappings::generic(),
            ],
        }
    }

    /// Detect the most likely format based on CSV headers
    pub fn detect(&self, csv_headers: &[String]) -> Result<DetectionResult> {
        let mut best_match: Option<DetectionResult> = None;
        let mut best_confidence = 0.0;

        for mapping in &self.predefined_mappings {
            let result = self.score_mapping(csv_headers, mapping);

            if result.confidence > best_confidence {
                best_confidence = result.confidence;
                best_match = Some(result);
            }
        }

        best_match.ok_or_else(|| Error::Parse("Unable to detect CSV format".to_string()))
    }

    /// Score a mapping against CSV headers
    fn score_mapping(&self, csv_headers: &[String], mapping: &HeaderMapping) -> DetectionResult {
        let mut matched_headers = 0;
        let total_headers = csv_headers.len();

        // Check how many CSV headers match this mapping
        for csv_header in csv_headers {
            if mapping.get_mapping(csv_header).is_some() {
                matched_headers += 1;
            }
        }

        // Calculate confidence based on match percentage
        let match_percentage = if total_headers > 0 {
            matched_headers as f64 / total_headers as f64
        } else {
            0.0
        };

        // Check for key required fields
        let has_patient_id = csv_headers.iter().any(|h| {
            h.to_lowercase().contains("patient") || h.to_lowercase().contains("account")
        });

        let has_date = csv_headers.iter().any(|h| {
            h.to_lowercase().contains("date") || h.to_lowercase().contains("dos")
        });

        let has_procedure = csv_headers.iter().any(|h| {
            h.to_lowercase().contains("cpt") || h.to_lowercase().contains("procedure")
        });

        // Boost confidence if key fields are present
        let mut confidence = match_percentage;
        if has_patient_id {
            confidence += 0.1;
        }
        if has_date {
            confidence += 0.1;
        }
        if has_procedure {
            confidence += 0.1;
        }

        // Cap at 1.0
        confidence = confidence.min(1.0);

        DetectionResult {
            source_system: mapping.source_system.clone(),
            confidence,
            matched_headers,
            total_headers,
            suggested_mapping: mapping.clone(),
        }
    }

    /// Analyze CSV headers and provide recommendations
    pub fn analyze_headers(&self, csv_headers: &[String]) -> HeaderAnalysis {
        let mut analysis = HeaderAnalysis {
            total_headers: csv_headers.len(),
            recognized_headers: Vec::new(),
            unrecognized_headers: Vec::new(),
            potential_mappings: HashMap::new(),
            recommendations: Vec::new(),
        };

        // Try to match each header against all mappings
        for csv_header in csv_headers {
            let mut matched = false;

            for mapping in &self.predefined_mappings {
                if let Some(field_mapping) = mapping.get_mapping(csv_header) {
                    analysis.recognized_headers.push(csv_header.clone());
                    analysis.potential_mappings.insert(
                        csv_header.clone(),
                        field_mapping.target_field.clone(),
                    );
                    matched = true;
                    break;
                }
            }

            if !matched {
                analysis.unrecognized_headers.push(csv_header.clone());
            }
        }

        // Generate recommendations
        if analysis.unrecognized_headers.is_empty() {
            analysis.recommendations.push(
                "All headers recognized. CSV format detected successfully.".to_string()
            );
        } else if analysis.recognized_headers.len() > analysis.unrecognized_headers.len() {
            analysis.recommendations.push(format!(
                "Most headers recognized ({}/{}). Consider creating custom mapping for remaining fields.",
                analysis.recognized_headers.len(),
                analysis.total_headers
            ));
        } else {
            analysis.recommendations.push(
                "Many unrecognized headers. Custom mapping configuration recommended.".to_string()
            );
        }

        // Check for required fields
        let has_required_fields = analysis.potential_mappings.values().any(|v| {
            v == "patient_control_number" || v == "date_of_service_from" || v == "procedure_code"
        });

        if !has_required_fields {
            analysis.recommendations.push(
                "WARNING: Missing required fields (patient ID, date of service, or procedure code).".to_string()
            );
        }

        analysis
    }
}

/// Analysis result for CSV headers
#[derive(Debug, Clone)]
pub struct HeaderAnalysis {
    pub total_headers: usize,
    pub recognized_headers: Vec<String>,
    pub unrecognized_headers: Vec<String>,
    pub potential_mappings: HashMap<String, String>,
    pub recommendations: Vec<String>,
}

impl HeaderAnalysis {
    /// Get recognition rate as percentage
    pub fn recognition_rate(&self) -> f64 {
        if self.total_headers == 0 {
            0.0
        } else {
            (self.recognized_headers.len() as f64 / self.total_headers as f64) * 100.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_detector_creation() {
        let detector = FormatDetector::new();
        assert!(!detector.predefined_mappings.is_empty());
    }

    #[test]
    fn test_detect_athena_format() {
        let detector = FormatDetector::new();
        let headers = vec![
            "Patient ID".to_string(),
            "DOS".to_string(),
            "CPT".to_string(),
            "Units".to_string(),
            "Charges".to_string(),
        ];

        let result = detector.detect(&headers).unwrap();
        assert!(result.confidence > 0.0);
        assert!(result.matched_headers > 0);
    }

    #[test]
    fn test_analyze_headers() {
        let detector = FormatDetector::new();
        let headers = vec![
            "Patient ID".to_string(),
            "DOS".to_string(),
            "Unknown Field".to_string(),
        ];

        let analysis = detector.analyze_headers(&headers);
        assert_eq!(analysis.total_headers, 3);
        assert!(!analysis.recognized_headers.is_empty());
        assert!(!analysis.unrecognized_headers.is_empty());
        assert!(!analysis.recommendations.is_empty());
    }

    #[test]
    fn test_recognition_rate() {
        let analysis = HeaderAnalysis {
            total_headers: 10,
            recognized_headers: vec!["Field1".to_string(), "Field2".to_string()],
            unrecognized_headers: vec![],
            potential_mappings: HashMap::new(),
            recommendations: vec![],
        };

        assert_eq!(analysis.recognition_rate(), 20.0);
    }

    #[test]
    fn test_detect_best_match() {
        let detector = FormatDetector::new();
        let athena_headers = vec![
            "Patient ID".to_string(),
            "DOS".to_string(),
            "Provider NPI".to_string(),
            "CPT".to_string(),
            "Modifier 1".to_string(),
            "Units".to_string(),
            "Charges".to_string(),
        ];

        let result = detector.detect(&athena_headers).unwrap();
        assert_eq!(result.source_system, "ATHENA");
        assert!(result.confidence > 0.7); // Should have high confidence
    }
}
