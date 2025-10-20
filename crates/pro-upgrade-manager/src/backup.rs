use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::process::Command;
use tracing::{info, warn, error};

use crate::error::{Result, UpgradeError};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupInfo {
    pub file_path: PathBuf,
    pub created_at: DateTime<Utc>,
    pub database_name: String,
    pub size_bytes: u64,
    pub compressed: bool,
}

pub struct BackupManager {
    backup_dir: PathBuf,
}

impl BackupManager {
    pub fn new<P: AsRef<Path>>(backup_dir: P) -> Result<Self> {
        let backup_dir = backup_dir.as_ref().to_path_buf();

        // Create backup directory if it doesn't exist
        if !backup_dir.exists() {
            std::fs::create_dir_all(&backup_dir)?;
            info!("Created backup directory: {}", backup_dir.display());
        }

        Ok(Self { backup_dir })
    }

    /// Find pg_dump executable
    fn find_pg_dump() -> Result<PathBuf> {
        // Try common PostgreSQL installation paths on Windows
        let common_paths = vec![
            r"C:\Program Files\PostgreSQL\16\bin\pg_dump.exe",
            r"C:\Program Files\PostgreSQL\15\bin\pg_dump.exe",
            r"C:\Program Files\PostgreSQL\14\bin\pg_dump.exe",
            r"C:\Program Files\PostgreSQL\13\bin\pg_dump.exe",
            r"C:\Program Files (x86)\PostgreSQL\16\bin\pg_dump.exe",
            r"C:\Program Files (x86)\PostgreSQL\15\bin\pg_dump.exe",
            r"C:\PostgreSQL\bin\pg_dump.exe",
        ];

        // Check if pg_dump is in PATH first
        if let Ok(output) = Command::new("pg_dump").arg("--version").output() {
            if output.status.success() {
                info!("Found pg_dump in system PATH");
                return Ok(PathBuf::from("pg_dump"));
            }
        }

        // Check common installation paths
        for path in common_paths {
            let path_buf = PathBuf::from(path);
            if path_buf.exists() {
                info!("Found pg_dump at: {}", path);
                return Ok(path_buf);
            }
        }

        Err(UpgradeError::PgToolsNotFound(
            "Could not find pg_dump.exe. Please ensure PostgreSQL client tools are installed.".to_string()
        ))
    }

    /// Find pg_restore executable
    fn find_pg_restore() -> Result<PathBuf> {
        // Try common PostgreSQL installation paths on Windows
        let common_paths = vec![
            r"C:\Program Files\PostgreSQL\16\bin\pg_restore.exe",
            r"C:\Program Files\PostgreSQL\15\bin\pg_restore.exe",
            r"C:\Program Files\PostgreSQL\14\bin\pg_restore.exe",
            r"C:\Program Files\PostgreSQL\13\bin\pg_restore.exe",
            r"C:\Program Files (x86)\PostgreSQL\16\bin\pg_restore.exe",
            r"C:\Program Files (x86)\PostgreSQL\15\bin\pg_restore.exe",
            r"C:\PostgreSQL\bin\pg_restore.exe",
        ];

        // Check if pg_restore is in PATH first
        if let Ok(output) = Command::new("pg_restore").arg("--version").output() {
            if output.status.success() {
                info!("Found pg_restore in system PATH");
                return Ok(PathBuf::from("pg_restore"));
            }
        }

        // Check common installation paths
        for path in common_paths {
            let path_buf = PathBuf::from(path);
            if path_buf.exists() {
                info!("Found pg_restore at: {}", path);
                return Ok(path_buf);
            }
        }

        Err(UpgradeError::PgToolsNotFound(
            "Could not find pg_restore.exe. Please ensure PostgreSQL client tools are installed.".to_string()
        ))
    }

    /// Create a backup of the database
    pub fn create_backup(
        &self,
        host: &str,
        port: u16,
        database: &str,
        username: &str,
        password: &str,
    ) -> Result<BackupInfo> {
        info!("Creating backup of database: {}", database);

        let pg_dump = Self::find_pg_dump()?;

        // Generate backup filename with timestamp
        let timestamp = Utc::now().format("%Y%m%d_%H%M%S");
        let filename = format!("professional_smart_backup_{}.sql.gz", timestamp);
        let backup_path = self.backup_dir.join(&filename);

        info!("Backup will be saved to: {}", backup_path.display());

        // Build pg_dump command
        // Using custom format (-Fc) which is compressed and can be used with pg_restore
        let output = Command::new(&pg_dump)
            .arg("-h").arg(host)
            .arg("-p").arg(port.to_string())
            .arg("-U").arg(username)
            .arg("-d").arg(database)
            .arg("-Fc")  // Custom format (compressed)
            .arg("-f").arg(&backup_path)
            .arg("--verbose")
            .env("PGPASSWORD", password)
            .output()
            .map_err(|e| {
                UpgradeError::Backup(format!("Failed to execute pg_dump: {}", e))
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            error!("pg_dump failed: {}", stderr);
            return Err(UpgradeError::Backup(format!(
                "pg_dump failed with exit code {:?}: {}",
                output.status.code(),
                stderr
            )));
        }

        // Verify backup was created
        if !backup_path.exists() {
            return Err(UpgradeError::Backup(
                "Backup file was not created".to_string()
            ));
        }

        // Get file size
        let metadata = std::fs::metadata(&backup_path)?;
        let size_bytes = metadata.len();

        info!(
            "Backup created successfully: {} ({} bytes)",
            backup_path.display(),
            size_bytes
        );

        Ok(BackupInfo {
            file_path: backup_path,
            created_at: Utc::now(),
            database_name: database.to_string(),
            size_bytes,
            compressed: true,
        })
    }

    /// Restore a database from a backup
    pub fn restore_backup(
        &self,
        backup_path: &Path,
        host: &str,
        port: u16,
        database: &str,
        username: &str,
        password: &str,
    ) -> Result<()> {
        info!("Restoring database from backup: {}", backup_path.display());

        if !backup_path.exists() {
            return Err(UpgradeError::Restore(
                format!("Backup file not found: {}", backup_path.display())
            ));
        }

        let pg_restore = Self::find_pg_restore()?;

        // Build pg_restore command
        let output = Command::new(&pg_restore)
            .arg("-h").arg(host)
            .arg("-p").arg(port.to_string())
            .arg("-U").arg(username)
            .arg("-d").arg(database)
            .arg("--clean")  // Drop objects before recreating
            .arg("--if-exists")  // Don't error if objects don't exist
            .arg("--verbose")
            .arg(backup_path)
            .env("PGPASSWORD", password)
            .output()
            .map_err(|e| {
                UpgradeError::Restore(format!("Failed to execute pg_restore: {}", e))
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            error!("pg_restore failed: {}", stderr);
            return Err(UpgradeError::Restore(format!(
                "pg_restore failed with exit code {:?}: {}",
                output.status.code(),
                stderr
            )));
        }

        info!("Database restored successfully from: {}", backup_path.display());
        Ok(())
    }

    /// List all backups in the backup directory
    pub fn list_backups(&self) -> Result<Vec<BackupInfo>> {
        let mut backups = Vec::new();

        for entry in std::fs::read_dir(&self.backup_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_file() {
                let filename = path.file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("")
                    .to_string();

                // Only include files that match our backup naming pattern
                if filename.starts_with("professional_smart_backup_") {
                    let metadata = std::fs::metadata(&path)?;
                    let size_bytes = metadata.len();

                    // Try to parse timestamp from filename
                    // Format: professional_smart_backup_YYYYMMDD_HHMMSS.sql.gz
                    let created_at = if let Some(timestamp_str) = filename
                        .strip_prefix("professional_smart_backup_")
                        .and_then(|s| s.split('.').next())
                    {
                        // Parse YYYYMMDD_HHMMSS format
                        chrono::NaiveDateTime::parse_from_str(timestamp_str, "%Y%m%d_%H%M%S")
                            .ok()
                            .map(|dt| DateTime::<Utc>::from_naive_utc_and_offset(dt, Utc))
                            .unwrap_or_else(Utc::now)
                    } else {
                        Utc::now()
                    };

                    let compressed = filename.ends_with(".gz");

                    backups.push(BackupInfo {
                        file_path: path,
                        created_at,
                        database_name: "professional_smart".to_string(),
                        size_bytes,
                        compressed,
                    });
                }
            }
        }

        // Sort by creation time, newest first
        backups.sort_by(|a, b| b.created_at.cmp(&a.created_at));

        Ok(backups)
    }

    /// Clean up old backups, keeping only the most recent N backups
    pub fn cleanup_old_backups(&self, keep_count: usize) -> Result<usize> {
        let backups = self.list_backups()?;
        let mut deleted_count = 0;

        if backups.len() > keep_count {
            info!(
                "Cleaning up old backups. Keeping {} most recent, deleting {}",
                keep_count,
                backups.len() - keep_count
            );

            for backup in backups.iter().skip(keep_count) {
                match std::fs::remove_file(&backup.file_path) {
                    Ok(_) => {
                        info!("Deleted old backup: {}", backup.file_path.display());
                        deleted_count += 1;
                    }
                    Err(e) => {
                        warn!("Failed to delete backup {}: {}", backup.file_path.display(), e);
                    }
                }
            }
        }

        Ok(deleted_count)
    }

    /// Get the backup directory path
    pub fn backup_dir(&self) -> &Path {
        &self.backup_dir
    }
}
