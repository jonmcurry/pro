use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Sha256, Digest};
use sqlx::{PgPool, Row};
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn, error};

use crate::error::{Result, UpgradeError};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationInfo {
    pub migration_name: String,
    pub applied_at: DateTime<Utc>,
    pub checksum: String,
    pub execution_time_ms: Option<i32>,
    pub description: Option<String>,
}

#[derive(Debug, Clone)]
pub struct PendingMigration {
    pub file_name: String,
    pub file_path: PathBuf,
    pub content: String,
    pub checksum: String,
}

pub struct MigrationManager {
    pool: PgPool,
    migrations_dir: PathBuf,
}

impl MigrationManager {
    pub fn new(pool: PgPool, migrations_dir: PathBuf) -> Self {
        Self {
            pool,
            migrations_dir,
        }
    }

    /// Calculate SHA-256 checksum of a string
    fn calculate_checksum(content: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    /// Get all applied migrations from the database
    pub async fn get_applied_migrations(&self) -> Result<Vec<MigrationInfo>> {
        // Check if migration tracking table exists
        let table_exists: bool = sqlx::query_scalar(
            r#"
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = 'staging'
                AND table_name = 'schema_migrations'
            )
            "#,
        )
        .fetch_one(&self.pool)
        .await?;

        if !table_exists {
            info!("Migration tracking table does not exist yet");
            return Ok(Vec::new());
        }

        let rows = sqlx::query(
            r#"
            SELECT
                migration_name,
                applied_at,
                checksum,
                execution_time_ms,
                description
            FROM staging.schema_migrations
            ORDER BY migration_name
            "#
        )
        .fetch_all(&self.pool)
        .await?;

        let migrations: Vec<MigrationInfo> = rows.into_iter().map(|r| MigrationInfo {
            migration_name: r.get("migration_name"),
            applied_at: r.get("applied_at"),
            checksum: r.get("checksum"),
            execution_time_ms: r.get("execution_time_ms"),
            description: r.get("description"),
        }).collect();

        info!("Found {} applied migrations", migrations.len());
        Ok(migrations)
    }

    /// Get all migration files from the migrations directory
    pub fn get_migration_files(&self) -> Result<Vec<PendingMigration>> {
        if !self.migrations_dir.exists() {
            return Err(UpgradeError::Migration(format!(
                "Migrations directory not found: {}",
                self.migrations_dir.display()
            )));
        }

        let mut migrations = Vec::new();

        for entry in std::fs::read_dir(&self.migrations_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_file() {
                if let Some(extension) = path.extension() {
                    if extension == "sql" {
                        let file_name = path
                            .file_name()
                            .and_then(|n| n.to_str())
                            .ok_or_else(|| {
                                UpgradeError::Migration(format!(
                                    "Invalid filename: {}",
                                    path.display()
                                ))
                            })?
                            .to_string();

                        let content = std::fs::read_to_string(&path)?;
                        let checksum = Self::calculate_checksum(&content);

                        migrations.push(PendingMigration {
                            file_name,
                            file_path: path,
                            content,
                            checksum,
                        });
                    }
                }
            }
        }

        // Sort migrations by filename (which should be numbered)
        migrations.sort_by(|a, b| a.file_name.cmp(&b.file_name));

        info!("Found {} migration files", migrations.len());
        Ok(migrations)
    }

    /// Get pending migrations that haven't been applied yet
    pub async fn get_pending_migrations(&self) -> Result<Vec<PendingMigration>> {
        let applied = self.get_applied_migrations().await?;
        let all_migrations = self.get_migration_files()?;

        // Create a set of applied migration names for quick lookup
        let applied_names: std::collections::HashSet<String> = applied
            .iter()
            .map(|m| m.migration_name.clone())
            .collect();

        // Filter out already applied migrations
        let pending: Vec<PendingMigration> = all_migrations
            .into_iter()
            .filter(|m| !applied_names.contains(&m.file_name))
            .collect();

        info!("Found {} pending migrations", pending.len());
        Ok(pending)
    }

    /// Verify checksums of applied migrations against current files
    pub async fn verify_checksums(&self) -> Result<Vec<String>> {
        let applied = self.get_applied_migrations().await?;
        let all_migrations = self.get_migration_files()?;

        let mut mismatches = Vec::new();

        // Create a map of migration files by name
        let migration_map: std::collections::HashMap<String, PendingMigration> = all_migrations
            .into_iter()
            .map(|m| (m.file_name.clone(), m))
            .collect();

        for applied_migration in applied {
            // Skip legacy checksums (from backfill)
            if applied_migration.checksum == "legacy" {
                continue;
            }

            if let Some(file_migration) = migration_map.get(&applied_migration.migration_name) {
                if applied_migration.checksum != file_migration.checksum {
                    let error_msg = format!(
                        "Checksum mismatch for {}: expected {}, got {}",
                        applied_migration.migration_name,
                        applied_migration.checksum,
                        file_migration.checksum
                    );
                    warn!("{}", error_msg);
                    mismatches.push(error_msg);
                }
            }
        }

        if mismatches.is_empty() {
            info!("All migration checksums verified successfully");
        } else {
            warn!("Found {} checksum mismatches", mismatches.len());
        }

        Ok(mismatches)
    }

    /// Apply a single migration
    pub async fn apply_migration(&self, migration: &PendingMigration) -> Result<i32> {
        info!("Applying migration: {}", migration.file_name);

        let start = std::time::Instant::now();

        // Split migration into individual statements
        // sqlx doesn't support multiple statements in a single query
        let statements = self.split_sql_statements(&migration.content);

        info!("Migration {} contains {} SQL statements", migration.file_name, statements.len());

        // Execute each statement separately
        for (idx, statement) in statements.iter().enumerate() {
            if statement.trim().is_empty() {
                continue;
            }

            debug!("Executing statement {}/{}", idx + 1, statements.len());

            match sqlx::raw_sql(statement).execute(&self.pool).await {
                Ok(_) => {
                    debug!("Statement {} executed successfully", idx + 1);
                }
                Err(e) => {
                    error!("Migration {} failed at statement {}: {}", migration.file_name, idx + 1, e);
                    return Err(UpgradeError::Migration(format!(
                        "Failed to apply migration {} at statement {}: {}",
                        migration.file_name, idx + 1, e
                    )));
                }
            }
        }

        let execution_time = start.elapsed().as_millis() as i32;
        info!(
            "Migration {} completed in {}ms",
            migration.file_name, execution_time
        );

        // Record the migration in the tracking table
        self.record_migration(migration, execution_time).await?;

        Ok(execution_time)
    }

    /// Split SQL content into individual statements
    /// Handles PostgreSQL dollar-quoted strings ($$) and regular semicolons
    fn split_sql_statements(&self, content: &str) -> Vec<String> {
        let mut statements = Vec::new();
        let mut current_statement = String::new();
        let mut in_dollar_quote = false;
        let mut dollar_quote_tag = String::new();

        for line in content.lines() {
            let trimmed = line.trim();

            // Skip empty lines and comment-only lines (but only if not inside dollar quotes)
            if !in_dollar_quote && (trimmed.is_empty() || trimmed.starts_with("--")) {
                continue;
            }

            // Check for dollar-quoted strings
            if trimmed.contains("$$") {
                if !in_dollar_quote {
                    // Entering dollar quote
                    in_dollar_quote = true;
                    // For now, assume simple $$ without tags (could be enhanced)
                    dollar_quote_tag = "$$".to_string();
                } else {
                    // Check if this closes the dollar quote
                    if trimmed.contains(&dollar_quote_tag) {
                        in_dollar_quote = false;
                        dollar_quote_tag.clear();
                    }
                }
            }

            // Add line to current statement
            current_statement.push_str(line);
            current_statement.push('\n');

            // Only split on semicolon if we're not inside a dollar-quoted block
            if !in_dollar_quote && trimmed.ends_with(';') {
                let stmt = current_statement.trim().to_string();
                if !stmt.is_empty() {
                    statements.push(stmt);
                }
                current_statement.clear();
            }
        }

        // Add any remaining statement
        let stmt = current_statement.trim().to_string();
        if !stmt.is_empty() {
            statements.push(stmt);
        }

        statements
    }

    /// Record a migration in the tracking table
    async fn record_migration(
        &self,
        migration: &PendingMigration,
        execution_time_ms: i32,
    ) -> Result<()> {
        // Ensure staging schema exists
        sqlx::raw_sql("CREATE SCHEMA IF NOT EXISTS staging")
            .execute(&self.pool)
            .await?;

        // Ensure migration tracking table exists
        sqlx::raw_sql(
            r#"
            CREATE TABLE IF NOT EXISTS staging.schema_migrations (
                migration_name VARCHAR(255) PRIMARY KEY,
                applied_at TIMESTAMP WITH TIME ZONE NOT NULL,
                checksum VARCHAR(64) NOT NULL,
                execution_time_ms INTEGER,
                description TEXT
            )
            "#
        )
        .execute(&self.pool)
        .await?;

        // Extract description from migration file (look for Description comment)
        let description = Self::extract_description(&migration.content);

        sqlx::query(
            r#"
            INSERT INTO staging.schema_migrations
                (migration_name, applied_at, checksum, execution_time_ms, description)
            VALUES ($1, NOW(), $2, $3, $4)
            ON CONFLICT (migration_name) DO NOTHING
            "#
        )
        .bind(&migration.file_name)
        .bind(&migration.checksum)
        .bind(execution_time_ms)
        .bind(description)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Extract description from migration file comments
    fn extract_description(content: &str) -> Option<String> {
        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("-- Description:") {
                return Some(
                    trimmed
                        .strip_prefix("-- Description:")
                        .unwrap()
                        .trim()
                        .to_string(),
                );
            }
        }
        None
    }

    /// Apply all pending migrations
    pub async fn apply_pending_migrations(&self) -> Result<Vec<String>> {
        let pending = self.get_pending_migrations().await?;

        if pending.is_empty() {
            info!("No pending migrations to apply");
            return Ok(Vec::new());
        }

        info!("Applying {} pending migrations", pending.len());

        let mut applied = Vec::new();

        for migration in pending {
            match self.apply_migration(&migration).await {
                Ok(execution_time) => {
                    info!(
                        "Successfully applied {} in {}ms",
                        migration.file_name, execution_time
                    );
                    applied.push(migration.file_name);
                }
                Err(e) => {
                    error!("Failed to apply migration {}: {}", migration.file_name, e);
                    return Err(e);
                }
            }
        }

        info!("Successfully applied {} migrations", applied.len());
        Ok(applied)
    }

    /// Check if migration tracking is set up
    pub async fn is_migration_tracking_setup(&self) -> Result<bool> {
        let table_exists: bool = sqlx::query_scalar(
            r#"
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = 'staging'
                AND table_name = 'schema_migrations'
            )
            "#,
        )
        .fetch_one(&self.pool)
        .await?;

        Ok(table_exists)
    }

    /// Get migrations directory
    pub fn migrations_dir(&self) -> &Path {
        &self.migrations_dir
    }
}
