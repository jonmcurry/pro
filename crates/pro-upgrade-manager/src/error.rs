use thiserror::Error;

#[derive(Error, Debug)]
pub enum UpgradeError {
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Version not found")]
    VersionNotFound,

    #[error("Migration error: {0}")]
    Migration(String),

    #[error("Backup error: {0}")]
    Backup(String),

    #[error("Restore error: {0}")]
    Restore(String),

    #[error("Checksum mismatch for migration {migration}: expected {expected}, got {actual}")]
    ChecksumMismatch {
        migration: String,
        expected: String,
        actual: String,
    },

    #[error("PostgreSQL tools not found: {0}")]
    PgToolsNotFound(String),

    #[error("Command execution failed: {0}")]
    CommandFailed(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Parse error: {0}")]
    Parse(String),
}

pub type Result<T> = std::result::Result<T, UpgradeError>;
