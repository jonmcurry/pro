use anyhow::{Context, Result, bail};
use chrono::Local;
use std::path::PathBuf;
use std::process::Command;

/// Default backup directory on Windows
pub fn default_backup_dir() -> PathBuf {
    PathBuf::from(r"C:\ProgramData\Professional SMART\backups")
}

/// Service for database backup operations using pg_dump
pub struct BackupService {
    host: String,
    port: u16,
    user: String,
    password: String,
    backup_dir: PathBuf,
}

impl BackupService {
    pub fn new(host: &str, port: u16, user: &str, password: &str, backup_dir: PathBuf) -> Self {
        Self {
            host: host.to_string(),
            port,
            user: user.to_string(),
            password: password.to_string(),
            backup_dir,
        }
    }

    /// Create a backup of a database
    pub fn backup(&self, database_name: &str, output_path: Option<&str>) -> Result<BackupResult> {
        // Ensure backup directory exists
        if !self.backup_dir.exists() {
            std::fs::create_dir_all(&self.backup_dir)
                .context("Failed to create backup directory")?;
        }

        // Generate output filename if not specified
        let timestamp = Local::now().format("%Y%m%d_%H%M%S");
        let output_file = match output_path {
            Some(path) => PathBuf::from(path),
            None => self.backup_dir.join(format!("{}_{}.backup", database_name, timestamp)),
        };

        let start_time = std::time::Instant::now();

        // Run pg_dump
        let output = Command::new("pg_dump")
            .env("PGPASSWORD", &self.password)
            .args([
                "-h", &self.host,
                "-p", &self.port.to_string(),
                "-U", &self.user,
                "-Fc", // Custom format (compressed)
                "-f", output_file.to_str().unwrap(),
                database_name,
            ])
            .output()
            .context("Failed to execute pg_dump. Is PostgreSQL installed and in PATH?")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            bail!("pg_dump failed: {}", stderr);
        }

        // Get file size
        let metadata = std::fs::metadata(&output_file)
            .context("Failed to get backup file metadata")?;

        let duration = start_time.elapsed();

        Ok(BackupResult {
            path: output_file,
            size_bytes: metadata.len(),
            duration_secs: duration.as_secs(),
        })
    }

    /// Verify a backup file integrity
    pub fn verify(&self, backup_path: &str) -> Result<bool> {
        // Use pg_restore --list to verify backup integrity
        let output = Command::new("pg_restore")
            .env("PGPASSWORD", &self.password)
            .args(["--list", backup_path])
            .output()
            .context("Failed to execute pg_restore")?;

        Ok(output.status.success())
    }

    /// List all backups in the backup directory
    pub fn list_backups(&self) -> Result<Vec<BackupInfo>> {
        let mut backups = Vec::new();

        if !self.backup_dir.exists() {
            return Ok(backups);
        }

        for entry in std::fs::read_dir(&self.backup_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().map(|e| e == "backup").unwrap_or(false) {
                let metadata = std::fs::metadata(&path)?;
                let filename = path.file_name().unwrap().to_string_lossy().to_string();

                // Parse database name from filename (format: dbname_YYYYMMDD_HHMMSS.backup)
                let db_name = filename
                    .rsplit('_')
                    .skip(2)
                    .collect::<Vec<_>>()
                    .into_iter()
                    .rev()
                    .collect::<Vec<_>>()
                    .join("_");

                backups.push(BackupInfo {
                    path: path.clone(),
                    database_name: if db_name.is_empty() { filename.clone() } else { db_name },
                    size_bytes: metadata.len(),
                    created_at: metadata.created().ok(),
                });
            }
        }

        // Sort by creation time, newest first
        backups.sort_by(|a, b| b.created_at.cmp(&a.created_at));

        Ok(backups)
    }
}

#[derive(Debug)]
pub struct BackupResult {
    pub path: PathBuf,
    pub size_bytes: u64,
    pub duration_secs: u64,
}

#[derive(Debug)]
pub struct BackupInfo {
    pub path: PathBuf,
    pub database_name: String,
    pub size_bytes: u64,
    pub created_at: Option<std::time::SystemTime>,
}
