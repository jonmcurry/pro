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

/// Coerce the raw SBR01 payer responsibility code to the two values the
/// `claims.encounter.payer_responsibility_code` column accepts: `'P'` or `'S'`.
///
/// X12 SBR01 may legitimately carry `P`, `S`, `T`, and other codes, but the
/// `chk_payer_responsibility` constraint on `claims.encounter` only allows
/// `P`/`S`. The full COB record - including tertiary - lives in
/// `claims.encounter_payer`, whose own check constraint allows `P`/`S`/`T`.
///
/// Mapping:
///   * `P` -> `P`
///   * `S` -> `S`
///   * `T` -> `S` (tertiary maps down to secondary on the main encounter row)
///   * empty / anything else -> `P`, with a warning naming the raw code
///
/// Not a silent fallback (CLAUDE.md Rule 3): the warning makes data-quality
/// drift visible in logs.
pub fn normalize_payer_responsibility_code(raw: &str) -> &'static str {
    // First non-whitespace character only - source data sometimes pads or
    // accidentally double-fills the field (e.g. "PP", " S"). Char-aware to
    // avoid the byte-slice panic the prior `&s[..1]` form risked on
    // multi-byte input.
    let first = raw.trim().chars().next();
    match first {
        Some('P') | Some('p') => "P",
        Some('S') | Some('s') => "S",
        Some('T') | Some('t') => {
            tracing::warn!(
                "payer_responsibility_code='{}' (tertiary) coerced to 'S' for encounter row \
                (full record preserved in encounter_payer)",
                raw
            );
            "S"
        }
        Some(other) => {
            tracing::warn!(
                "payer_responsibility_code='{}' (first char '{}') is not P/S/T; defaulting to 'P'",
                raw, other
            );
            "P"
        }
        None => "P",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_known_values() {
        assert_eq!(normalize_payer_responsibility_code("P"), "P");
        assert_eq!(normalize_payer_responsibility_code("S"), "S");
        assert_eq!(normalize_payer_responsibility_code("T"), "S");
    }

    #[test]
    fn handles_padding_and_case() {
        assert_eq!(normalize_payer_responsibility_code("p"), "P");
        assert_eq!(normalize_payer_responsibility_code(" S "), "S");
        assert_eq!(normalize_payer_responsibility_code("PP"), "P");
    }

    #[test]
    fn defaults_unknown_to_primary() {
        assert_eq!(normalize_payer_responsibility_code(""), "P");
        assert_eq!(normalize_payer_responsibility_code("A"), "P");
        assert_eq!(normalize_payer_responsibility_code("01"), "P");
    }

    #[test]
    fn does_not_panic_on_multibyte() {
        // The prior `&s[..1]` form would have panicked here.
        assert_eq!(normalize_payer_responsibility_code("\u{e9}"), "P");
    }
}
