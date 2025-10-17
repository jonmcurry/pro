//! Configuration loading from .env file

use anyhow::{anyhow, Context, Result};
use std::fs;
use std::path::PathBuf;

pub struct Config {
    pub database_url: String,
}

impl Config {
    /// Find and load the .env file from standard locations
    pub fn load() -> Result<Self> {
        let env_paths = vec![
            PathBuf::from(r"C:\ProgramData\Professional SMART\config\.env"),
            PathBuf::from(r"C:\Program Files\Professional SMART\config\.env"),
            PathBuf::from(".env"),
            PathBuf::from(r"..\..\.env"), // For development
        ];

        let env_file = env_paths
            .iter()
            .find(|p| p.exists())
            .context(format!(
                "Could not find .env file in any of these locations:\n{}",
                env_paths
                    .iter()
                    .map(|p| format!("  - {}", p.display()))
                    .collect::<Vec<_>>()
                    .join("\n")
            ))?;

        println!("Loading configuration from: {}", env_file.display());

        // Try to read the file directly first to check if we can access it
        let contents = fs::read_to_string(env_file).with_context(|| {
            format!(
                "Cannot read .env file: {}\n\
                Possible causes:\n\
                  - File permissions (run as Administrator if needed)\n\
                  - File is locked by another process\n\
                  - File encoding issues",
                env_file.display()
            )
        })?;

        // Check if file is empty
        if contents.trim().is_empty() {
            return Err(anyhow!(
                "The .env file exists but is empty: {}\n\
                Please run the Configuration Wizard to set up the database connection.",
                env_file.display()
            ));
        }

        // Parse the .env file manually to provide better error messages
        let mut database_url: Option<String> = None;

        for (line_num, line) in contents.lines().enumerate() {
            let line = line.trim();

            // Skip empty lines and comments
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // Parse KEY=VALUE
            if let Some(eq_pos) = line.find('=') {
                let key = line[..eq_pos].trim();
                let value = line[eq_pos + 1..].trim();

                if key == "DATABASE_URL" {
                    if value.is_empty() {
                        return Err(anyhow!(
                            "DATABASE_URL is empty in .env file at line {}: {}\n\
                            Please run the Configuration Wizard to set up the database connection.",
                            line_num + 1,
                            env_file.display()
                        ));
                    }
                    database_url = Some(value.to_string());
                    break;
                }
            }
        }

        let database_url = database_url.ok_or_else(|| {
            anyhow!(
                "DATABASE_URL not found in .env file: {}\n\
                The file exists but DATABASE_URL variable is missing.\n\
                File contents preview (first 200 chars):\n{}\n\
                Please run the Configuration Wizard to set up the database connection.",
                env_file.display(),
                if contents.len() > 200 {
                    &contents[..200]
                } else {
                    &contents
                }
            )
        })?;

        Ok(Config { database_url })
    }

    /// Mask password in database URL for display
    pub fn masked_url(&self) -> String {
        if let Some(at_pos) = self.database_url.rfind('@') {
            if let Some(colon_pos) = self.database_url[..at_pos].rfind(':') {
                let mut masked = self.database_url.clone();
                masked.replace_range(colon_pos + 1..at_pos, "****");
                return masked;
            }
        }
        self.database_url.clone()
    }
}
