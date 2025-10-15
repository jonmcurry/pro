use thiserror::Error;

/// Common error types for the Professional SMART application
#[derive(Error, Debug)]
pub enum Error {
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),

    #[error("Validation error: {0}")]
    Validation(String),

    #[error("Parse error: {0}")]
    Parse(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Already exists: {0}")]
    AlreadyExists(String),

    #[error("Invalid format: {0}")]
    InvalidFormat(String),

    #[error("EDI parsing error: {0}")]
    EdiParse(String),

    #[error("CSV parsing error: {0}")]
    CsvParse(String),

    #[error("Rules engine error: {0}")]
    RulesEngine(String),

    #[error("RVU calculation error: {0}")]
    RvuCalculation(String),

    #[error("Invalid state: {0}")]
    InvalidState(String),

    #[error("Unauthorized: {0}")]
    Unauthorized(String),

    #[error("Internal error: {0}")]
    Internal(String),

    #[error("External service error: {0}")]
    ExternalService(String),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

/// Result type alias for the application
pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    /// Check if the error is a not found error
    pub fn is_not_found(&self) -> bool {
        matches!(self, Error::NotFound(_))
    }

    /// Check if the error is a validation error
    pub fn is_validation(&self) -> bool {
        matches!(self, Error::Validation(_))
    }

    /// Check if the error is a database error
    pub fn is_database(&self) -> bool {
        matches!(self, Error::Database(_))
    }
}
