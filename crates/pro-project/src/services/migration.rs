use anyhow::{Context, Result};
use sqlx::postgres::PgPool;
use sqlx::Row;
use pro_upgrade_manager::embedded_migrations::{self, EmbeddedMigration};

/// Service for managing database migrations
pub struct MigrationService {
    host: String,
    port: u16,
    user: String,
    password: String,
}

impl MigrationService {
    pub fn new(host: &str, port: u16, user: &str, password: &str) -> Self {
        Self {
            host: host.to_string(),
            port,
            user: user.to_string(),
            password: password.to_string(),
        }
    }

    fn connection_string(&self, database: &str) -> String {
        format!(
            "postgres://{}:{}@{}:{}/{}",
            self.user, self.password, self.host, self.port, database
        )
    }

    /// Get all embedded migrations
    pub fn get_all_migrations() -> Vec<EmbeddedMigration> {
        embedded_migrations::get_all_migrations()
    }

    /// Get the baseline migration
    pub fn get_baseline() -> &'static EmbeddedMigration {
        embedded_migrations::get_baseline()
    }

    /// Get the current application version (from embedded migrations)
    pub fn get_current_version() -> String {
        let migrations = Self::get_all_migrations();
        if !migrations.is_empty() {
            format!("2.12.35.0 ({} migrations)", migrations.len())
        } else {
            "2.12.35.0".to_string()
        }
    }

    /// Get applied migrations for a database
    pub async fn get_applied_migrations(&self, database_name: &str) -> Result<Vec<AppliedMigration>> {
        let pool = PgPool::connect(&self.connection_string(database_name))
            .await
            .context("Failed to connect to database")?;

        let rows = sqlx::query(
            r#"
            SELECT migration_name, applied_at, checksum
            FROM staging.schema_migrations
            ORDER BY migration_name
            "#,
        )
        .fetch_all(&pool)
        .await
        .context("Failed to query migrations")?;

        let mut migrations = Vec::new();
        for row in rows {
            let migration_name: String = row.get("migration_name");
            // Extract version from migration_name (e.g., "069" from "069_setup_smartproaudit_fdw.sql")
            let version = migration_name
                .split('_')
                .next()
                .unwrap_or(&migration_name)
                .to_string();
            migrations.push(AppliedMigration {
                version,
                migration_name,
                applied_at: row.get("applied_at"),
                checksum: row.get("checksum"),
            });
        }

        Ok(migrations)
    }

    /// Get pending migrations for a database
    pub async fn get_pending_migrations(&self, database_name: &str) -> Result<Vec<PendingMigration>> {
        let applied = self.get_applied_migrations(database_name).await?;
        let applied_versions: std::collections::HashSet<String> =
            applied.iter().map(|m| m.version.clone()).collect();

        let all_migrations = Self::get_all_migrations();
        let mut pending = Vec::new();

        for migration in all_migrations {
            if !applied_versions.contains(migration.version) {
                pending.push(PendingMigration {
                    version: migration.version.to_string(),
                    name: migration.name.to_string(),
                    file_name: migration.file_name(),
                });
            }
        }

        Ok(pending)
    }

    /// Apply a single migration to a database
    pub async fn apply_migration(
        &self,
        database_name: &str,
        migration: &EmbeddedMigration,
    ) -> Result<()> {
        let pool = PgPool::connect(&self.connection_string(database_name))
            .await
            .context("Failed to connect to database")?;

        // Split and execute migration SQL statements
        let statements = Self::split_sql_statements(migration.sql);
        for statement in statements {
            if statement.trim().is_empty() {
                continue;
            }
            sqlx::raw_sql(&statement)
                .execute(&pool)
                .await
                .with_context(|| format!("Failed to apply migration {}", migration.file_name()))?;
        }

        // Record migration in schema_migrations
        sqlx::query(
            r#"
            INSERT INTO staging.schema_migrations (migration_name, checksum)
            VALUES ($1, $2)
            ON CONFLICT (migration_name) DO NOTHING
            "#,
        )
        .bind(migration.file_name())
        .bind(migration.checksum())
        .execute(&pool)
        .await
        .context("Failed to record migration")?;

        Ok(())
    }

    /// Apply all pending migrations to a database
    pub async fn apply_all_pending(
        &self,
        database_name: &str,
        on_progress: impl Fn(&str, &str),
    ) -> Result<MigrationResult> {
        let pending = self.get_pending_migrations(database_name).await?;

        if pending.is_empty() {
            return Ok(MigrationResult {
                applied_count: 0,
                errors: Vec::new(),
            });
        }

        let all_migrations = Self::get_all_migrations();
        let mut applied_count = 0;
        let mut errors = Vec::new();

        for pending_migration in &pending {
            // Find the full migration
            if let Some(migration) = all_migrations.iter().find(|m| m.version == pending_migration.version) {
                on_progress(&pending_migration.file_name, "Applying");

                match self.apply_migration(database_name, migration).await {
                    Ok(()) => {
                        on_progress(&pending_migration.file_name, "OK");
                        applied_count += 1;
                    }
                    Err(e) => {
                        let error_msg = e.to_string();
                        on_progress(&pending_migration.file_name, &format!("FAILED: {}", error_msg));
                        errors.push((pending_migration.file_name.clone(), error_msg));
                        break; // Stop on first error
                    }
                }
            }
        }

        Ok(MigrationResult {
            applied_count,
            errors,
        })
    }

    /// Update the application_version table after upgrade
    pub async fn update_application_version(
        &self,
        database_name: &str,
        version: &str,
    ) -> Result<()> {
        let pool = PgPool::connect(&self.connection_string(database_name))
            .await
            .context("Failed to connect to database")?;

        sqlx::query(
            r#"
            INSERT INTO staging.application_version (version, description)
            VALUES ($1, 'Upgraded via pro-project')
            "#,
        )
        .bind(version)
        .execute(&pool)
        .await
        .context("Failed to update application version")?;

        Ok(())
    }

    /// Split SQL content into individual statements
    /// Handles dollar-quoted strings ($$) properly
    fn split_sql_statements(content: &str) -> Vec<String> {
        let mut statements = Vec::new();
        let mut current_statement = String::new();
        let mut in_dollar_quote = false;

        for line in content.lines() {
            let trimmed = line.trim();

            // Skip empty lines and comment-only lines (but only if not inside dollar quotes)
            if !in_dollar_quote && (trimmed.is_empty() || trimmed.starts_with("--")) {
                continue;
            }

            // Check for dollar-quoted strings
            if trimmed.contains("$$") {
                if !in_dollar_quote {
                    in_dollar_quote = true;
                } else {
                    in_dollar_quote = false;
                }
            }

            // Add line to current statement
            if !current_statement.is_empty() {
                current_statement.push('\n');
            }
            current_statement.push_str(line);

            // Check for statement terminator (semicolon at end, not in dollar quote)
            if !in_dollar_quote && trimmed.ends_with(';') {
                statements.push(current_statement.clone());
                current_statement.clear();
            }
        }

        // Don't forget any remaining statement
        if !current_statement.trim().is_empty() {
            statements.push(current_statement);
        }

        statements
    }
}

#[derive(Debug, Clone)]
pub struct AppliedMigration {
    pub version: String,
    pub migration_name: String,
    pub applied_at: chrono::DateTime<chrono::Utc>,
    pub checksum: String,
}

#[derive(Debug, Clone)]
pub struct PendingMigration {
    pub version: String,
    pub name: String,
    pub file_name: String,
}

#[derive(Debug)]
pub struct MigrationResult {
    pub applied_count: usize,
    pub errors: Vec<(String, String)>,
}
