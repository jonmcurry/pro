//! Interned String Support for Rule Execution
//!
//! PHASE 6: Helper utilities for using string interning with the rule engine
//!
//! This module provides conversion utilities and helpers to use interned strings
//! without breaking existing rule implementations.

use crate::rule_engine::RuleExecutionContext;
use pro_common::{InternedString, StringInterner};
use pro_db::models_interned::ServiceLineInterned;


/// Extended context with interned string support
///
/// This allows rules to access procedure codes as interned symbols
/// for memory efficiency while maintaining backward compatibility.
pub struct InternedExecutionContext<'a> {
    pub ctx: &'a RuleExecutionContext,
    pub interner: &'a StringInterner,
    pub procedure_code_sym: Option<InternedString>,
    pub modifier_syms: Vec<InternedString>,
    pub pos_code_sym: Option<InternedString>,
}

impl<'a> InternedExecutionContext<'a> {
    /// Create from a regular context by interning all strings
    pub fn from_context(ctx: &'a RuleExecutionContext, interner: &'a StringInterner) -> Self {
        let procedure_code_sym = ctx.procedure_code.as_ref()
            .map(|s| interner.intern(s));

        let modifier_syms = ctx.procedure_modifiers.iter()
            .map(|m| interner.intern(m))
            .collect();

        let pos_code_sym = ctx.place_of_service_code.as_ref()
            .map(|s| interner.intern(s));

        Self {
            ctx,
            interner,
            procedure_code_sym,
            modifier_syms,
            pos_code_sym,
        }
    }


    /// Get procedure code symbol
    pub fn procedure_code_symbol(&self) -> Option<InternedString> {
        self.procedure_code_sym
    }

    /// Get procedure code as string
    pub fn procedure_code_str(&self) -> Option<&str> {
        self.ctx.procedure_code.as_deref()
    }

    /// Get modifier symbols
    pub fn modifier_symbols(&self) -> &[InternedString] {
        &self.modifier_syms
    }

    /// Check if specific modifier exists (by symbol)
    pub fn has_modifier_sym(&self, modifier_sym: InternedString) -> bool {
        self.modifier_syms.contains(&modifier_sym)
    }

    /// Check if specific modifier exists (by string)
    pub fn has_modifier(&self, modifier: &str) -> bool {
        self.ctx.procedure_modifiers.iter().any(|m| m == modifier)
    }
}

/// Helper to batch-convert service lines to interned format
pub fn batch_intern_service_lines(
    service_lines: Vec<pro_db::models::ServiceLine>,
    interner: &StringInterner,
) -> Vec<ServiceLineInterned> {
    service_lines.into_iter()
        .map(|sl| ServiceLineInterned::from_service_line(sl, interner))
        .collect()
}

/// Helper to create execution contexts from interned service lines
pub fn create_contexts_from_interned(
    service_lines: &[ServiceLineInterned],
    organization_id: i64,
    diagnosis_map: &std::collections::HashMap<i64, Vec<String>>,
    interner: &StringInterner,
) -> Vec<RuleExecutionContext> {
    service_lines.iter()
        .map(|sl| {
            let diagnosis_codes = diagnosis_map.get(&sl.encounter_id)
                .cloned()
                .unwrap_or_default();

            let mut ctx = RuleExecutionContext::new(organization_id);
            ctx.service_line_id = Some(sl.service_line_id);
            ctx.encounter_id = Some(sl.encounter_id);
            ctx.procedure_code = sl.procedure_code_str(interner);
            ctx.procedure_modifiers = sl.modifiers_str(interner);
            ctx.service_unit_count = Some(sl.service_unit_count);
            ctx.line_item_charge_amount = Some(sl.line_item_charge_amount);
            ctx.date_of_service = Some(sl.service_date_from);
            ctx.diagnosis_codes = diagnosis_codes;
            ctx.place_of_service_code = sl.place_of_service_code
                .and_then(|s| interner.resolve(s));
            ctx.provider_id = sl.rendering_provider_id;

            ctx
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use rust_decimal_macros::dec;

    #[test]
    fn test_interned_context_conversion() {
        let interner = StringInterner::new();
        let org_id = 1i64;

        let mut ctx = RuleExecutionContext::new(org_id);
        ctx.procedure_code = Some("99213".to_string());
        ctx.procedure_modifiers = vec!["25".to_string(), "59".to_string()];
        ctx.place_of_service_code = Some("11".to_string());

        let interned_ctx = InternedExecutionContext::from_context(&ctx, &interner);

        // Verify procedure code was interned
        assert!(interned_ctx.procedure_code_symbol().is_some());
        assert_eq!(interned_ctx.procedure_code_str().unwrap(), "99213");

        // Verify modifiers
        assert_eq!(interned_ctx.modifier_symbols().len(), 2);
        assert!(interned_ctx.has_modifier("25"));
        assert!(interned_ctx.has_modifier("59"));
        assert!(!interned_ctx.has_modifier("99"));
    }

    #[test]
    fn test_batch_intern() {
        use pro_db::models::ServiceLine;
        use chrono::NaiveDate;

        let interner = StringInterner::new();

        // Create test service lines
        let service_lines = vec![
            ServiceLine {
                service_line_id: // TODO: Remove - database generates IDs now,
                encounter_id: // TODO: Remove - database generates IDs now,
                line_number: 1,
                product_service_id_qualifier: Some("HC".to_string()),
                procedure_code: "99213".to_string(),
                procedure_modifier_1: Some("25".to_string()),
                procedure_modifier_2: None,
                procedure_modifier_3: None,
                procedure_modifier_4: None,
                procedure_description: None,
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
            },
            ServiceLine {
                service_line_id: // TODO: Remove - database generates IDs now,
                encounter_id: // TODO: Remove - database generates IDs now,
                line_number: 2,
                product_service_id_qualifier: Some("HC".to_string()),
                procedure_code: "99213".to_string(), // Same code - should reuse interned string
                procedure_modifier_1: Some("25".to_string()),
                procedure_modifier_2: None,
                procedure_modifier_3: None,
                procedure_modifier_4: None,
                procedure_description: None,
                line_item_charge_amount: dec!(150.00),
                unit_basis_measurement_code: Some("UN".to_string()),
                service_unit_count: dec!(1.0),
                place_of_service_code: Some("11".to_string()),
                service_date_from: NaiveDate::from_ymd_opt(2024, 1, 16).unwrap(),
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
            },
        ];

        let interned = batch_intern_service_lines(service_lines, &interner);

        // Both should have same procedure code symbol
        assert_eq!(interned[0].procedure_code, interned[1].procedure_code);

        // Verify interner has expected strings
        assert!(interner.len() <= 10); // HC, 99213, 25, 11, UN = 5 unique strings
    }
}
