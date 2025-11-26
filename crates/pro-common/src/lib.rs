// Common types, errors, and utilities shared across the Professional SMART application

pub mod error;
pub mod types;
pub mod validation;
pub mod constants;
pub mod string_interner; // PHASE 6: String interning for memory optimization
pub mod newtype_ids;     // Strongly-typed ID and code newtypes

// Re-export commonly used items
pub use error::{Error, Result};
pub use types::*;
pub use string_interner::{StringInterner, InternedString, InternedProcedureCode};
pub use constants::DEFAULT_DATE;

// Re-export newtype IDs and codes
pub use newtype_ids::{
    // IDs
    EncounterId, ServiceLineId, DiagnosisId, ProviderId, FacilityId,
    OrganizationId, RegionId, BatchId, QueueId, FlagId, RuleId,
    // Codes
    Npi, NpiError, ProcedureCode, DiagnosisCode, TaxonomyCode, Modifier, PlaceOfServiceCode,
};
