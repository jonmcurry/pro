// Common types, errors, and utilities shared across the Professional SMART application

pub mod error;
pub mod types;
pub mod validation;
pub mod constants;
pub mod string_interner; // PHASE 6: String interning for memory optimization

// Re-export commonly used items
pub use error::{Error, Result};
pub use types::*;
pub use string_interner::{StringInterner, InternedString, InternedProcedureCode};
