//! Diagnosis Builder Module
//!
//! Builds diagnosis records from raw claim data.
//! Extracted from ClaimsProcessor as part of god object refactoring.
//!
//! NOTE: This module is scaffolding for future refactoring. Currently unused
//! but retained for planned integration with the claims processing pipeline.

#![allow(dead_code)]

use anyhow::Result;
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use tracing::debug;

/// Builder for constructing diagnosis records
pub struct DiagnosisBuilder;

impl DiagnosisBuilder {
    /// Extract diagnosis codes from raw claim fields
    pub fn extract_diagnoses(diagnosis_fields: &Option<JsonValue>) -> Result<Vec<DiagnosisData>> {
        let fields: HashMap<String, Vec<String>> = match diagnosis_fields {
            Some(df) => serde_json::from_value(df.clone()).unwrap_or_default(),
            None => return Ok(Vec::new()),
        };

        let mut all_diagnoses: Vec<(usize, String)> = Vec::new();

        for (field_name, codes) in &fields {
            // Match field names like "diagnosis_code_1", "diagnosis_code_2", etc.
            if field_name.starts_with("diagnosis_code_") {
                if let Some(seq_str) = field_name.strip_prefix("diagnosis_code_") {
                    if let Ok(sequence) = seq_str.parse::<usize>() {
                        for code in codes {
                            if !code.is_empty() {
                                all_diagnoses.push((sequence, code.clone()));
                            }
                        }
                    }
                }
            }
            // Also support legacy format "diagnosis_code" (single field with array)
            else if field_name == "diagnosis_code" {
                for (idx, code) in codes.iter().enumerate() {
                    if !code.is_empty() {
                        all_diagnoses.push((idx + 1, code.clone()));
                    }
                }
            }
        }

        // Sort by sequence number to maintain proper order
        all_diagnoses.sort_by_key(|(seq, _)| *seq);

        // Convert to DiagnosisData
        let diagnoses: Vec<DiagnosisData> = all_diagnoses
            .iter()
            .enumerate()
            .map(|(idx, (sequence, code))| DiagnosisData {
                sequence_number: *sequence as i16,
                diagnosis_code: code.clone(),
                is_principal: idx == 0, // First diagnosis is principal
            })
            .collect();

        debug!("Extracted {} diagnoses from raw claim", diagnoses.len());

        Ok(diagnoses)
    }
}

/// Extracted diagnosis data
#[derive(Debug, Clone)]
pub struct DiagnosisData {
    pub sequence_number: i16,
    pub diagnosis_code: String,
    pub is_principal: bool,
}
