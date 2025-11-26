//! Strongly-typed ID and Code newtypes
//!
//! This module provides type-safe wrappers for database IDs and domain codes
//! to prevent accidentally mixing up different ID types (e.g., passing a
//! ProviderId where an EncounterId was expected).
//!
//! ## Usage
//! ```ignore
//! use pro_common::{EncounterId, ProviderId, Npi};
//!
//! fn process_encounter(id: EncounterId, provider: ProviderId) {
//!     // Type system prevents mixing up IDs
//! }
//!
//! // This would fail to compile:
//! // process_encounter(provider_id, encounter_id);
//! ```

use serde::{Deserialize, Serialize};
use std::fmt;
use std::hash::Hash;

// ============================================================================
// ID NEWTYPES
// ============================================================================

/// Macro to generate ID newtypes with common implementations
macro_rules! define_id {
    ($name:ident, $doc:expr) => {
        #[doc = $doc]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(pub i64);

        impl $name {
            /// Create a new ID
            pub const fn new(id: i64) -> Self {
                Self(id)
            }

            /// Get the raw ID value
            pub const fn value(&self) -> i64 {
                self.0
            }

            /// Check if this is a valid (non-zero) ID
            pub const fn is_valid(&self) -> bool {
                self.0 > 0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}", self.0)
            }
        }

        impl From<i64> for $name {
            fn from(id: i64) -> Self {
                Self(id)
            }
        }

        impl From<$name> for i64 {
            fn from(id: $name) -> Self {
                id.0
            }
        }

        // sqlx support
        impl<'r> sqlx::Decode<'r, sqlx::Postgres> for $name {
            fn decode(
                value: sqlx::postgres::PgValueRef<'r>,
            ) -> Result<Self, sqlx::error::BoxDynError> {
                let id = <i64 as sqlx::Decode<sqlx::Postgres>>::decode(value)?;
                Ok(Self(id))
            }
        }

        impl sqlx::Type<sqlx::Postgres> for $name {
            fn type_info() -> sqlx::postgres::PgTypeInfo {
                <i64 as sqlx::Type<sqlx::Postgres>>::type_info()
            }
        }

        impl<'q> sqlx::Encode<'q, sqlx::Postgres> for $name {
            fn encode_by_ref(
                &self,
                buf: &mut sqlx::postgres::PgArgumentBuffer,
            ) -> Result<sqlx::encode::IsNull, sqlx::error::BoxDynError> {
                <i64 as sqlx::Encode<sqlx::Postgres>>::encode_by_ref(&self.0, buf)
            }
        }
    };
}

// Generate ID types
define_id!(EncounterId, "Strongly-typed encounter ID");
define_id!(ServiceLineId, "Strongly-typed service line ID");
define_id!(DiagnosisId, "Strongly-typed diagnosis ID");
define_id!(ProviderId, "Strongly-typed provider ID");
define_id!(FacilityId, "Strongly-typed facility ID");
define_id!(OrganizationId, "Strongly-typed organization ID");
define_id!(RegionId, "Strongly-typed region ID");
define_id!(BatchId, "Strongly-typed import batch ID");
define_id!(QueueId, "Strongly-typed file processing queue ID");
define_id!(FlagId, "Strongly-typed flag ID");
define_id!(RuleId, "Strongly-typed rule ID");

// ============================================================================
// CODE NEWTYPES
// ============================================================================

/// National Provider Identifier (NPI)
///
/// A 10-digit identifier assigned to healthcare providers in the US.
/// Validates format on construction.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct Npi(String);

impl Npi {
    /// Create a new NPI, validating the format
    pub fn new(npi: &str) -> Result<Self, NpiError> {
        if npi.len() != 10 {
            return Err(NpiError::InvalidLength(npi.len()));
        }
        if !npi.chars().all(|c| c.is_ascii_digit()) {
            return Err(NpiError::InvalidCharacters);
        }
        // Luhn check (NPI uses Luhn algorithm with prefix 80840)
        if !Self::luhn_check(npi) {
            return Err(NpiError::InvalidChecksum);
        }
        Ok(Self(npi.to_string()))
    }

    /// Create an NPI without validation (use carefully)
    pub fn new_unchecked(npi: impl Into<String>) -> Self {
        Self(npi.into())
    }

    /// Get the NPI value
    pub fn value(&self) -> &str {
        &self.0
    }

    /// Validate NPI using Luhn algorithm with 80840 prefix
    fn luhn_check(npi: &str) -> bool {
        // Prepend 80840 to NPI for Luhn calculation
        let prefixed = format!("80840{}", npi);
        let digits: Vec<u32> = prefixed
            .chars()
            .filter_map(|c| c.to_digit(10))
            .collect();

        if digits.len() != 15 {
            return false;
        }

        let mut sum = 0;
        for (i, &digit) in digits.iter().rev().enumerate() {
            if i % 2 == 1 {
                let doubled = digit * 2;
                sum += if doubled > 9 { doubled - 9 } else { doubled };
            } else {
                sum += digit;
            }
        }

        sum % 10 == 0
    }
}

impl fmt::Display for Npi {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl TryFrom<String> for Npi {
    type Error = NpiError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::new(&value)
    }
}

impl From<Npi> for String {
    fn from(npi: Npi) -> Self {
        npi.0
    }
}

/// NPI validation error
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NpiError {
    InvalidLength(usize),
    InvalidCharacters,
    InvalidChecksum,
}

impl fmt::Display for NpiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NpiError::InvalidLength(len) => {
                write!(f, "NPI must be 10 digits, got {}", len)
            }
            NpiError::InvalidCharacters => {
                write!(f, "NPI must contain only digits")
            }
            NpiError::InvalidChecksum => {
                write!(f, "NPI failed Luhn checksum validation")
            }
        }
    }
}

impl std::error::Error for NpiError {}

/// CPT/HCPCS Procedure Code
///
/// A 5-character alphanumeric code identifying medical procedures.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ProcedureCode(String);

impl ProcedureCode {
    /// Create a new procedure code
    pub fn new(code: impl Into<String>) -> Self {
        Self(code.into().to_uppercase())
    }

    /// Get the code value
    pub fn value(&self) -> &str {
        &self.0
    }

    /// Check if this is a CPT code (5 digits)
    pub fn is_cpt(&self) -> bool {
        self.0.len() == 5 && self.0.chars().all(|c| c.is_ascii_digit())
    }

    /// Check if this is a HCPCS Level II code (letter + 4 digits)
    pub fn is_hcpcs(&self) -> bool {
        self.0.len() == 5
            && self.0.chars().next().map(|c| c.is_ascii_uppercase()).unwrap_or(false)
            && self.0.chars().skip(1).all(|c| c.is_ascii_digit())
    }
}

impl fmt::Display for ProcedureCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&str> for ProcedureCode {
    fn from(s: &str) -> Self {
        Self::new(s)
    }
}

impl From<String> for ProcedureCode {
    fn from(s: String) -> Self {
        Self::new(s)
    }
}

/// ICD-10 Diagnosis Code
///
/// ICD-10-CM codes range from 3-7 characters.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct DiagnosisCode(String);

impl DiagnosisCode {
    /// Create a new diagnosis code
    pub fn new(code: impl Into<String>) -> Self {
        // Remove dots and uppercase
        let code = code.into().replace('.', "").to_uppercase();
        Self(code)
    }

    /// Get the code value
    pub fn value(&self) -> &str {
        &self.0
    }

    /// Get the formatted code (with dot after 3rd character)
    pub fn formatted(&self) -> String {
        if self.0.len() > 3 {
            format!("{}.{}", &self.0[..3], &self.0[3..])
        } else {
            self.0.clone()
        }
    }

    /// Get the category (first 3 characters)
    pub fn category(&self) -> &str {
        if self.0.len() >= 3 {
            &self.0[..3]
        } else {
            &self.0
        }
    }

    /// Check if this code is billable (typically 4-7 characters)
    pub fn is_billable(&self) -> bool {
        self.0.len() >= 4 && self.0.len() <= 7
    }
}

impl fmt::Display for DiagnosisCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.formatted())
    }
}

impl From<&str> for DiagnosisCode {
    fn from(s: &str) -> Self {
        Self::new(s)
    }
}

impl From<String> for DiagnosisCode {
    fn from(s: String) -> Self {
        Self::new(s)
    }
}

/// Taxonomy Code
///
/// Provider specialty/taxonomy codes from the NUCC Health Care Provider Taxonomy.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TaxonomyCode(String);

impl TaxonomyCode {
    /// Create a new taxonomy code
    pub fn new(code: impl Into<String>) -> Self {
        Self(code.into().to_uppercase())
    }

    /// Get the code value
    pub fn value(&self) -> &str {
        &self.0
    }

    /// Validate taxonomy code format (10 alphanumeric characters)
    pub fn is_valid(&self) -> bool {
        self.0.len() == 10 && self.0.chars().all(|c| c.is_ascii_alphanumeric())
    }
}

impl fmt::Display for TaxonomyCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&str> for TaxonomyCode {
    fn from(s: &str) -> Self {
        Self::new(s)
    }
}

/// Procedure Modifier
///
/// 2-character modifier code for procedure codes.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Modifier(String);

impl Modifier {
    /// Create a new modifier
    pub fn new(code: impl Into<String>) -> Self {
        Self(code.into().to_uppercase())
    }

    /// Get the modifier value
    pub fn value(&self) -> &str {
        &self.0
    }

    /// Check if this is a valid modifier format
    pub fn is_valid(&self) -> bool {
        self.0.len() == 2 && self.0.chars().all(|c| c.is_ascii_alphanumeric())
    }
}

impl fmt::Display for Modifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&str> for Modifier {
    fn from(s: &str) -> Self {
        Self::new(s)
    }
}

/// Place of Service Code
///
/// 2-digit code indicating where services were performed.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PlaceOfServiceCode(String);

impl PlaceOfServiceCode {
    /// Create a new place of service code
    pub fn new(code: impl Into<String>) -> Self {
        let code = code.into();
        // Pad to 2 digits if needed
        let padded = if code.len() == 1 {
            format!("0{}", code)
        } else {
            code
        };
        Self(padded)
    }

    /// Get the code value
    pub fn value(&self) -> &str {
        &self.0
    }

    /// Check if this is a facility place of service
    pub fn is_facility(&self) -> bool {
        // Facility POS codes: 21-23 (inpatient), 24 (ASC), 41-42 (ambulance)
        matches!(
            self.0.as_str(),
            "21" | "22" | "23" | "24" | "41" | "42" | "51" | "52" | "53" | "54" | "55" | "56"
        )
    }

    /// Check if this is a non-facility place of service
    pub fn is_non_facility(&self) -> bool {
        !self.is_facility()
    }
}

impl fmt::Display for PlaceOfServiceCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&str> for PlaceOfServiceCode {
    fn from(s: &str) -> Self {
        Self::new(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encounter_id() {
        let id = EncounterId::new(42);
        assert_eq!(id.value(), 42);
        assert!(id.is_valid());
        assert_eq!(format!("{}", id), "42");

        let invalid = EncounterId::new(0);
        assert!(!invalid.is_valid());
    }

    #[test]
    fn test_npi_validation() {
        // Valid NPI (passes Luhn check)
        let npi = Npi::new("1234567893");
        assert!(npi.is_ok());

        // Invalid length
        assert!(matches!(
            Npi::new("123456789"),
            Err(NpiError::InvalidLength(9))
        ));

        // Invalid characters
        assert!(matches!(
            Npi::new("123456789A"),
            Err(NpiError::InvalidCharacters)
        ));
    }

    #[test]
    fn test_procedure_code() {
        let cpt = ProcedureCode::new("99213");
        assert!(cpt.is_cpt());
        assert!(!cpt.is_hcpcs());

        let hcpcs = ProcedureCode::new("J0585");
        assert!(!hcpcs.is_cpt());
        assert!(hcpcs.is_hcpcs());
    }

    #[test]
    fn test_diagnosis_code() {
        let code = DiagnosisCode::new("E11.65");
        assert_eq!(code.value(), "E1165");
        assert_eq!(code.formatted(), "E11.65");
        assert_eq!(code.category(), "E11");
        assert!(code.is_billable());
    }

    #[test]
    fn test_place_of_service() {
        let office = PlaceOfServiceCode::new("11");
        assert!(office.is_non_facility());

        let inpatient = PlaceOfServiceCode::new("21");
        assert!(inpatient.is_facility());

        // Single digit padding
        let padded = PlaceOfServiceCode::new("1");
        assert_eq!(padded.value(), "01");
    }
}
