use anyhow::{Context, Result};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use chrono::Local;

/// Service for managing .env configuration files
pub struct ConfigService {
    config_path: PathBuf,
}

impl ConfigService {
    /// Default configuration path on Windows
    pub fn default_path() -> PathBuf {
        PathBuf::from(r"C:\ProgramData\Professional SMART\config\.env")
    }

    pub fn new(config_path: PathBuf) -> Self {
        Self { config_path }
    }

    pub fn with_default_path() -> Self {
        Self::new(Self::default_path())
    }

    /// Check if configuration file exists
    /// Reserved for future configuration validation
    #[allow(dead_code)]
    pub fn exists(&self) -> bool {
        self.config_path.exists()
    }

    /// Read the current configuration
    pub fn read(&self) -> Result<HashMap<String, String>> {
        let content = fs::read_to_string(&self.config_path)
            .with_context(|| format!("Failed to read config file: {:?}", self.config_path))?;

        let mut config = HashMap::new();
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if let Some((key, value)) = line.split_once('=') {
                config.insert(key.trim().to_string(), value.trim().to_string());
            }
        }
        Ok(config)
    }

    /// Get a specific configuration value
    pub fn get(&self, key: &str) -> Result<Option<String>> {
        let config = self.read()?;
        Ok(config.get(key).cloned())
    }

    /// Get the current database name from configuration
    pub fn get_current_database(&self) -> Result<Option<String>> {
        self.get("DB_NAME")
    }

    /// Get database connection parameters
    pub fn get_db_params(&self) -> Result<DbParams> {
        let config = self.read()?;
        Ok(DbParams {
            host: config.get("DB_HOST").cloned().unwrap_or_else(|| "localhost".to_string()),
            port: config.get("DB_PORT").and_then(|p| p.parse().ok()).unwrap_or(5432),
            name: config.get("DB_NAME").cloned().unwrap_or_default(),
            user: config.get("DB_USER").cloned().unwrap_or_else(|| "postgres".to_string()),
            password: config.get("DB_PASSWORD").cloned().unwrap_or_default(),
        })
    }

    /// Create a backup of the current configuration
    pub fn backup(&self) -> Result<PathBuf> {
        let timestamp = Local::now().format("%Y%m%d_%H%M%S");
        let backup_path = self.config_path.with_extension(format!("{}.bak", timestamp));

        fs::copy(&self.config_path, &backup_path)
            .with_context(|| format!("Failed to backup config to {:?}", backup_path))?;

        Ok(backup_path)
    }

    /// Update a configuration value atomically
    pub fn update(&self, key: &str, value: &str) -> Result<()> {
        let content = fs::read_to_string(&self.config_path)
            .with_context(|| format!("Failed to read config file: {:?}", self.config_path))?;

        let mut lines: Vec<String> = Vec::new();
        let mut found = false;

        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with(&format!("{}=", key)) || trimmed.starts_with(&format!("{} =", key)) {
                lines.push(format!("{}={}", key, value));
                found = true;
            } else {
                lines.push(line.to_string());
            }
        }

        if !found {
            lines.push(format!("{}={}", key, value));
        }

        // Atomic write: write to temp file, then rename
        let temp_path = self.config_path.with_extension("tmp");
        fs::write(&temp_path, lines.join("\n"))
            .with_context(|| format!("Failed to write temp config: {:?}", temp_path))?;

        fs::rename(&temp_path, &self.config_path)
            .with_context(|| "Failed to rename temp config to final location")?;

        Ok(())
    }

    /// Update DATABASE_URL based on component values
    pub fn update_database_url(&self, db_params: &DbParams) -> Result<()> {
        let url = format!(
            "postgres://{}:{}@{}:{}/{}",
            db_params.user, db_params.password, db_params.host, db_params.port, db_params.name
        );
        self.update("DATABASE_URL", &url)
    }

    /// Switch to a new database (updates DB_NAME and DATABASE_URL)
    pub fn switch_database(&self, new_db_name: &str) -> Result<PathBuf> {
        // Create backup first
        let backup_path = self.backup()?;

        // Update DB_NAME
        self.update("DB_NAME", new_db_name)?;

        // Update DATABASE_URL if it exists
        let config = self.read()?;
        if config.contains_key("DATABASE_URL") {
            let mut params = self.get_db_params()?;
            params.name = new_db_name.to_string();
            self.update_database_url(&params)?;
        }

        Ok(backup_path)
    }
}

#[derive(Debug, Clone)]
pub struct DbParams {
    pub host: String,
    pub port: u16,
    pub name: String,
    pub user: String,
    pub password: String,
}

impl DbParams {
    /// Build connection string for the configured database
    /// Reserved for future direct connection scenarios
    #[allow(dead_code)]
    pub fn connection_string(&self) -> String {
        format!(
            "postgres://{}:{}@{}:{}/{}",
            self.user, self.password, self.host, self.port, self.name
        )
    }

    /// Build connection string for a specific database
    /// Reserved for future multi-database operations
    #[allow(dead_code)]
    pub fn connection_string_for_db(&self, db_name: &str) -> String {
        format!(
            "postgres://{}:{}@{}:{}/{}",
            self.user, self.password, self.host, self.port, db_name
        )
    }

    /// Build admin connection string (connects to postgres database)
    /// Reserved for future admin operations
    #[allow(dead_code)]
    pub fn admin_connection_string(&self) -> String {
        // Connect to postgres database for admin operations
        format!(
            "postgres://{}:{}@{}:{}/postgres",
            self.user, self.password, self.host, self.port
        )
    }
}
