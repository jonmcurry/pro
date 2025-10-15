// Database access layer for Professional SMART

pub mod connection;
pub mod repositories;
pub mod models;
pub mod models_interned; // PHASE 6: String interning optimization
pub mod validation;

// Re-export commonly used items
pub use connection::{DbPool, create_pool};
pub use pro_common::{Error, Result};
pub use validation::{
    FileHash, FileValidator, PatientControlNumberValidator, ServiceLineValidator,
    BusinessRuleValidator, ValidationResult, DuplicateStatus,
    EncounterValidation, ServiceLineValidation,
};
