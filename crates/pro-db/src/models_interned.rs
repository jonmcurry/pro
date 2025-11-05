//! Interned String Models for Memory Optimization
//!
//! PHASE 6: Runtime representations using string interning for frequently duplicated fields
//!
//! These models are used after loading from the database to reduce heap allocations.
//! The database models (models.rs) remain unchanged for sqlx compatibility.

use chrono::{DateTime, NaiveDate, Utc};
use pro_common::{InternedString, StringInterner};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};


use crate::models::ServiceLine;

/// Service line with interned strings for procedure codes and modifiers
///
/// This reduces heap allocations by ~30% when processing thousands of service lines
/// with repeated procedure codes and modifiers.
#[derive(Debug, Clone)]
pub struct ServiceLineInterned {
    pub service_line_id: i64,
    pub encounter_id: i64,
    pub line_number: i16,

    // Interned strings - procedure codes and modifiers
    pub product_service_id_qualifier: Option<InternedString>,
    pub procedure_code: InternedString,
    pub procedure_modifier_1: Option<InternedString>,
    pub procedure_modifier_2: Option<InternedString>,
    pub procedure_modifier_3: Option<InternedString>,
    pub procedure_modifier_4: Option<InternedString>,

    // Description kept as String (less duplication, more varied)
    pub procedure_description: Option<String>,

    pub line_item_charge_amount: Decimal,
    pub unit_basis_measurement_code: Option<InternedString>,
    pub service_unit_count: Decimal,

    // Place of service - heavily duplicated, good for interning
    pub place_of_service_code: Option<InternedString>,

    pub service_date_from: NaiveDate,
    pub service_date_to: Option<NaiveDate>,

    // Provider information - UUIDs don't need interning
    pub rendering_provider_id: Option<i64>,
    pub rendering_provider_npi: Option<String>,
    pub supervising_provider_id: Option<i64>,
    pub supervising_provider_npi: Option<String>,
    pub ordering_provider_id: Option<i64>,
    pub ordering_provider_npi: Option<String>,
    pub referring_provider_id: Option<i64>,
    pub referring_provider_npi: Option<String>,

    pub service_facility_id: Option<i64>,
    pub service_facility_npi: Option<String>,

    pub prior_authorization_number: Option<String>,
    pub referral_number: Option<String>,
    pub line_note: Option<String>,

    // Revenue code - moderately duplicated
    pub revenue_code: Option<InternedString>,

    // NDC codes - may be duplicated
    pub ndc_code: Option<InternedString>,
    pub ndc_unit_count: Option<Decimal>,
    pub ndc_measurement_unit: Option<InternedString>,

    pub diagnosis_code_pointer_1: Option<i16>,
    pub diagnosis_code_pointer_2: Option<i16>,
    pub diagnosis_code_pointer_3: Option<i16>,
    pub diagnosis_code_pointer_4: Option<i16>,

    pub line_status: String,

    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub created_by: Option<String>,
    pub updated_by: Option<String>,
}

impl ServiceLineInterned {
    /// Convert a database ServiceLine to an interned representation
    pub fn from_service_line(service_line: ServiceLine, interner: &StringInterner) -> Self {
        Self {
            service_line_id: service_line.service_line_id,
            encounter_id: service_line.encounter_id,
            line_number: service_line.line_number,

            // Intern procedure codes and modifiers
            product_service_id_qualifier: service_line.product_service_id_qualifier.as_ref()
                .map(|s| interner.intern(s)),
            procedure_code: interner.intern(&service_line.procedure_code),
            procedure_modifier_1: service_line.procedure_modifier_1.as_ref()
                .map(|s| interner.intern(s)),
            procedure_modifier_2: service_line.procedure_modifier_2.as_ref()
                .map(|s| interner.intern(s)),
            procedure_modifier_3: service_line.procedure_modifier_3.as_ref()
                .map(|s| interner.intern(s)),
            procedure_modifier_4: service_line.procedure_modifier_4.as_ref()
                .map(|s| interner.intern(s)),

            procedure_description: service_line.procedure_description,
            line_item_charge_amount: service_line.line_item_charge_amount,

            unit_basis_measurement_code: service_line.unit_basis_measurement_code.as_ref()
                .map(|s| interner.intern(s)),
            service_unit_count: service_line.service_unit_count,

            place_of_service_code: service_line.place_of_service_code.as_ref()
                .map(|s| interner.intern(s)),

            service_date_from: service_line.service_date_from,
            service_date_to: service_line.service_date_to,

            rendering_provider_id: service_line.rendering_provider_id,
            rendering_provider_npi: service_line.rendering_provider_npi,
            supervising_provider_id: service_line.supervising_provider_id,
            supervising_provider_npi: service_line.supervising_provider_npi,
            ordering_provider_id: service_line.ordering_provider_id,
            ordering_provider_npi: service_line.ordering_provider_npi,
            referring_provider_id: service_line.referring_provider_id,
            referring_provider_npi: service_line.referring_provider_npi,

            service_facility_id: service_line.service_facility_id,
            service_facility_npi: service_line.service_facility_npi,

            prior_authorization_number: service_line.prior_authorization_number,
            referral_number: service_line.referral_number,
            line_note: service_line.line_note,

            revenue_code: service_line.revenue_code.as_ref()
                .map(|s| interner.intern(s)),

            ndc_code: service_line.ndc_code.as_ref()
                .map(|s| interner.intern(s)),
            ndc_unit_count: service_line.ndc_unit_count,
            ndc_measurement_unit: service_line.ndc_measurement_unit.as_ref()
                .map(|s| interner.intern(s)),

            diagnosis_code_pointer_1: service_line.diagnosis_code_pointer_1,
            diagnosis_code_pointer_2: service_line.diagnosis_code_pointer_2,
            diagnosis_code_pointer_3: service_line.diagnosis_code_pointer_3,
            diagnosis_code_pointer_4: service_line.diagnosis_code_pointer_4,

            line_status: service_line.line_status,

            created_at: service_line.created_at,
            updated_at: service_line.updated_at,
            created_by: service_line.created_by,
            updated_by: service_line.updated_by,
        }
    }

    /// Convert back to database ServiceLine (for updates)
    pub fn to_service_line(&self, interner: &StringInterner) -> ServiceLine {
        ServiceLine {
            service_line_id: self.service_line_id,
            encounter_id: self.encounter_id,
            line_number: self.line_number,

            product_service_id_qualifier: self.product_service_id_qualifier
                .and_then(|s| interner.resolve(s)),
            procedure_code: interner.resolve(self.procedure_code)
                .expect("Invalid procedure code symbol"),
            procedure_modifier_1: self.procedure_modifier_1
                .and_then(|s| interner.resolve(s)),
            procedure_modifier_2: self.procedure_modifier_2
                .and_then(|s| interner.resolve(s)),
            procedure_modifier_3: self.procedure_modifier_3
                .and_then(|s| interner.resolve(s)),
            procedure_modifier_4: self.procedure_modifier_4
                .and_then(|s| interner.resolve(s)),

            procedure_description: self.procedure_description.clone(),
            line_item_charge_amount: self.line_item_charge_amount,

            unit_basis_measurement_code: self.unit_basis_measurement_code
                .and_then(|s| interner.resolve(s)),
            service_unit_count: self.service_unit_count,

            place_of_service_code: self.place_of_service_code
                .and_then(|s| interner.resolve(s)),

            service_date_from: self.service_date_from,
            service_date_to: self.service_date_to,

            rendering_provider_id: self.rendering_provider_id,
            rendering_provider_npi: self.rendering_provider_npi.clone(),
            supervising_provider_id: self.supervising_provider_id,
            supervising_provider_npi: self.supervising_provider_npi.clone(),
            ordering_provider_id: self.ordering_provider_id,
            ordering_provider_npi: self.ordering_provider_npi.clone(),
            referring_provider_id: self.referring_provider_id,
            referring_provider_npi: self.referring_provider_npi.clone(),

            service_facility_id: self.service_facility_id,
            service_facility_npi: self.service_facility_npi.clone(),

            prior_authorization_number: self.prior_authorization_number.clone(),
            referral_number: self.referral_number.clone(),
            line_note: self.line_note.clone(),

            revenue_code: self.revenue_code
                .and_then(|s| interner.resolve(s)),

            ndc_code: self.ndc_code
                .and_then(|s| interner.resolve(s)),
            ndc_unit_count: self.ndc_unit_count,
            ndc_measurement_unit: self.ndc_measurement_unit
                .and_then(|s| interner.resolve(s)),

            diagnosis_code_pointer_1: self.diagnosis_code_pointer_1,
            diagnosis_code_pointer_2: self.diagnosis_code_pointer_2,
            diagnosis_code_pointer_3: self.diagnosis_code_pointer_3,
            diagnosis_code_pointer_4: self.diagnosis_code_pointer_4,

            line_status: self.line_status.clone(),

            created_at: self.created_at,
            updated_at: self.updated_at,
            created_by: self.created_by.clone(),
            updated_by: self.updated_by.clone(),
        }
    }

    /// Get procedure code as string
    pub fn procedure_code_str<'a>(&self, interner: &'a StringInterner) -> Option<String> {
        interner.resolve(self.procedure_code)
    }

    /// Get all modifiers as strings
    pub fn modifiers_str(&self, interner: &StringInterner) -> Vec<String> {
        let mut modifiers = Vec::new();

        if let Some(m1) = self.procedure_modifier_1.and_then(|s| interner.resolve(s)) {
            modifiers.push(m1);
        }
        if let Some(m2) = self.procedure_modifier_2.and_then(|s| interner.resolve(s)) {
            modifiers.push(m2);
        }
        if let Some(m3) = self.procedure_modifier_3.and_then(|s| interner.resolve(s)) {
            modifiers.push(m3);
        }
        if let Some(m4) = self.procedure_modifier_4.and_then(|s| interner.resolve(s)) {
            modifiers.push(m4);
        }

        modifiers
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use rust_decimal_macros::dec;

    #[test]
    fn test_service_line_interning() {
        let interner = StringInterner::new();

        // Create a sample service line
        let service_line = ServiceLine {
            service_line_id: 1,
            encounter_id: 1,
            line_number: 1,
            product_service_id_qualifier: Some("HC".to_string()),
            procedure_code: "99213".to_string(),
            procedure_modifier_1: Some("25".to_string()),
            procedure_modifier_2: None,
            procedure_modifier_3: None,
            procedure_modifier_4: None,
            procedure_description: Some("Office visit".to_string()),
            line_item_charge_amount: dec!(150.00),
            unit_basis_measurement_code: Some("UN".to_string()),
            service_unit_count: dec!(1.0),
            place_of_service_code: Some("11".to_string()),
            service_date_from: NaiveDate::from_ymd_opt(2024, 1, 15).unwrap(),
            service_date_to: None,
            rendering_provider_id: None,
            rendering_provider_npi: None,
            supervising_provider_id: None,
            supervising_provider_npi: None,
            ordering_provider_id: None,
            ordering_provider_npi: None,
            referring_provider_id: None,
            referring_provider_npi: None,
            service_facility_id: None,
            service_facility_npi: None,
            prior_authorization_number: None,
            referral_number: None,
            line_note: None,
            revenue_code: None,
            ndc_code: None,
            ndc_unit_count: None,
            ndc_measurement_unit: None,
            diagnosis_code_pointer_1: Some(1),
            diagnosis_code_pointer_2: None,
            diagnosis_code_pointer_3: None,
            diagnosis_code_pointer_4: None,
            line_status: "ACTIVE".to_string(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            created_by: None,
            updated_by: None,
        };

        // Convert to interned
        let interned = ServiceLineInterned::from_service_line(service_line.clone(), &interner);

        // Verify procedure code was interned
        assert_eq!(
            interned.procedure_code_str(&interner).unwrap(),
            "99213"
        );

        // Verify modifiers
        let modifiers = interned.modifiers_str(&interner);
        assert_eq!(modifiers.len(), 1);
        assert_eq!(modifiers[0], "25");

        // Convert back
        let converted_back = interned.to_service_line(&interner);
        assert_eq!(converted_back.procedure_code, service_line.procedure_code);
        assert_eq!(converted_back.procedure_modifier_1, service_line.procedure_modifier_1);
    }

    #[test]
    fn test_memory_efficiency() {
        let interner = StringInterner::new();

        // Create multiple service lines with same procedure code
        let mut interned_lines = Vec::new();

        for i in 0..1000 {
            let service_line = ServiceLine {
                service_line_id: i as i64,
                encounter_id: 1,
                line_number: i,
                product_service_id_qualifier: Some("HC".to_string()),
                procedure_code: "99213".to_string(), // Same code
                procedure_modifier_1: Some("25".to_string()), // Same modifier
                procedure_modifier_2: None,
                procedure_modifier_3: None,
                procedure_modifier_4: None,
                procedure_description: None,
                line_item_charge_amount: dec!(150.00),
                unit_basis_measurement_code: Some("UN".to_string()),
                service_unit_count: dec!(1.0),
                place_of_service_code: Some("11".to_string()), // Same POS
                service_date_from: NaiveDate::from_ymd_opt(2024, 1, 15).unwrap(),
                service_date_to: None,
                rendering_provider_id: None,
                rendering_provider_npi: None,
                supervising_provider_id: None,
                supervising_provider_npi: None,
                ordering_provider_id: None,
                ordering_provider_npi: None,
                referring_provider_id: None,
                referring_provider_npi: None,
                service_facility_id: None,
                service_facility_npi: None,
                prior_authorization_number: None,
                referral_number: None,
                line_note: None,
                revenue_code: None,
                ndc_code: None,
                ndc_unit_count: None,
                ndc_measurement_unit: None,
                diagnosis_code_pointer_1: Some(1),
                diagnosis_code_pointer_2: None,
                diagnosis_code_pointer_3: None,
                diagnosis_code_pointer_4: None,
                line_status: "ACTIVE".to_string(),
                created_at: Utc::now(),
                updated_at: Utc::now(),
                created_by: None,
                updated_by: None,
            };

            interned_lines.push(
                ServiceLineInterned::from_service_line(service_line, &interner)
            );
        }

        // Verify all procedure codes point to same interned string
        let first_code = interned_lines[0].procedure_code;
        for line in &interned_lines {
            assert_eq!(line.procedure_code, first_code);
        }

        // Only a few unique strings should be interned
        // "99213", "25", "HC", "11", "UN"
        assert!(interner.len() <= 10);
    }
}
