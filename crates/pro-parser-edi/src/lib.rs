// 837P Professional Claims EDI Parser
// Implements ASC X12N Version 005010X222A1 specification

pub mod parser;
pub mod segments;
pub mod loops;
pub mod types;
pub mod validator;

pub use parser::EdiParser;
pub use types::{Transaction837p, ParsedClaim};
pub use pro_common::{Error, Result};
