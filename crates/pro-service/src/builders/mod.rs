//! Builder modules for constructing database entities from raw claim data.
//!
//! These builders encapsulate the logic for creating and populating:
//! - Encounters
//! - Service lines
//! - Diagnoses
//! - Providers
//!
//! Extracted from ClaimsProcessor as part of god object refactoring.
//!
//! NOTE: These modules are scaffolding for future refactoring. Currently unused
//! but retained for planned integration with the claims processing pipeline.

#![allow(unused_imports)]

pub mod encounter_builder;
pub mod service_line_builder;
pub mod diagnosis_builder;
pub mod provider_builder;

pub use encounter_builder::EncounterBuilder;
pub use service_line_builder::ServiceLineBuilder;
pub use diagnosis_builder::DiagnosisBuilder;
pub use provider_builder::ProviderBuilder;
