use anyhow::{Context, Result, bail};
use sqlx::postgres::PgPool;
use sqlx::Row;
use pro_upgrade_manager::embedded_migrations;

/// Service for PostgreSQL database operations
pub struct DatabaseService {
    host: String,
    port: u16,
    user: String,
    password: String,
}

impl DatabaseService {
    pub fn new(host: &str, port: u16, user: &str, password: &str) -> Self {
        Self {
            host: host.to_string(),
            port,
            user: user.to_string(),
            password: password.to_string(),
        }
    }

    /// Get connection string for the postgres admin database
    fn admin_connection_string(&self) -> String {
        format!(
            "postgres://{}:{}@{}:{}/postgres",
            self.user, self.password, self.host, self.port
        )
    }

    /// Get connection string for a specific database
    fn connection_string(&self, database: &str) -> String {
        format!(
            "postgres://{}:{}@{}:{}/{}",
            self.user, self.password, self.host, self.port, database
        )
    }

    /// Check if a database exists
    pub async fn database_exists(&self, database_name: &str) -> Result<bool> {
        let pool = PgPool::connect(&self.admin_connection_string())
            .await
            .context("Failed to connect to PostgreSQL")?;

        let row = sqlx::query(
            "SELECT EXISTS(SELECT 1 FROM pg_database WHERE datname = $1) as exists",
        )
        .bind(database_name)
        .fetch_one(&pool)
        .await
        .context("Failed to check database existence")?;

        Ok(row.get::<bool, _>("exists"))
    }

    /// Create a new database
    pub async fn create_database(&self, database_name: &str) -> Result<()> {
        // Validate database name
        if !Self::is_valid_database_name(database_name) {
            bail!("Invalid database name '{}'. Must be alphanumeric with underscores, 1-63 characters.", database_name);
        }

        let pool = PgPool::connect(&self.admin_connection_string())
            .await
            .context("Failed to connect to PostgreSQL")?;

        // Check if already exists
        if self.database_exists(database_name).await? {
            bail!("Database '{}' already exists", database_name);
        }

        // Create database (use quoted identifier to preserve case)
        let sql = format!("CREATE DATABASE \"{}\"", database_name);
        sqlx::query(&sql)
            .execute(&pool)
            .await
            .with_context(|| format!("Failed to create database '{}'", database_name))?;

        Ok(())
    }

    /// Apply the baseline schema to a database
    pub async fn apply_baseline(&self, database_name: &str) -> Result<()> {
        let pool = PgPool::connect(&self.connection_string(database_name))
            .await
            .context("Failed to connect to new database")?;

        let baseline = embedded_migrations::get_baseline();

        // Execute baseline SQL
        sqlx::query(baseline.sql)
            .execute(&pool)
            .await
            .context("Failed to apply baseline schema")?;

        Ok(())
    }

    /// Drop a database
    pub async fn drop_database(&self, database_name: &str) -> Result<()> {
        let pool = PgPool::connect(&self.admin_connection_string())
            .await
            .context("Failed to connect to PostgreSQL")?;

        // Terminate existing connections
        let terminate_sql = format!(
            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = '{}'",
            database_name
        );
        sqlx::query(&terminate_sql)
            .execute(&pool)
            .await
            .context("Failed to terminate connections")?;

        // Drop database
        let drop_sql = format!("DROP DATABASE IF EXISTS \"{}\"", database_name);
        sqlx::query(&drop_sql)
            .execute(&pool)
            .await
            .with_context(|| format!("Failed to drop database '{}'", database_name))?;

        Ok(())
    }

    /// Check if a database has the Professional SMART schema
    pub async fn has_ps_schema(&self, database_name: &str) -> Result<bool> {
        let pool = PgPool::connect(&self.connection_string(database_name))
            .await
            .context("Failed to connect to database")?;

        // Check for staging.application_version table
        let row = sqlx::query(
            r#"
            SELECT EXISTS(
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = 'staging' AND table_name = 'application_version'
            ) as exists
            "#,
        )
        .fetch_one(&pool)
        .await
        .context("Failed to check schema")?;

        Ok(row.get::<bool, _>("exists"))
    }

    /// Get the schema version from a database based on applied migrations
    /// Schema version is derived from the highest migration number: 2.12.{max_migration}.0
    pub async fn get_schema_version(&self, database_name: &str) -> Result<Option<String>> {
        let pool = PgPool::connect(&self.connection_string(database_name))
            .await
            .context("Failed to connect to database")?;

        // Get the highest migration number from schema_migrations
        let row = sqlx::query(
            r#"
            SELECT migration_name FROM staging.schema_migrations
            ORDER BY migration_name DESC LIMIT 1
            "#,
        )
        .fetch_optional(&pool)
        .await
        .context("Failed to get schema version")?;

        // Extract migration number and format as 2.12.{migration}.0
        Ok(row.map(|r| {
            let migration_name: String = r.get("migration_name");
            // Extract the numeric prefix (e.g., "069" from "069_setup_smartproaudit_fdw.sql")
            let migration_num = migration_name
                .split('_')
                .next()
                .and_then(|s| s.parse::<u32>().ok())
                .unwrap_or(0);
            format!("2.12.{}.0", migration_num)
        }))
    }

    /// Get the count of applied migrations
    pub async fn get_migration_count(&self, database_name: &str) -> Result<i64> {
        let pool = PgPool::connect(&self.connection_string(database_name))
            .await
            .context("Failed to connect to database")?;

        let row = sqlx::query("SELECT COUNT(*) as count FROM staging.schema_migrations")
            .fetch_one(&pool)
            .await
            .context("Failed to count migrations")?;

        Ok(row.get::<i64, _>("count"))
    }

    /// Get database size in bytes
    pub async fn get_database_size(&self, database_name: &str) -> Result<i64> {
        let pool = PgPool::connect(&self.connection_string(database_name))
            .await
            .context("Failed to connect to database")?;

        let row = sqlx::query("SELECT pg_database_size(current_database()) as size")
            .fetch_one(&pool)
            .await
            .context("Failed to get database size")?;

        Ok(row.get::<i64, _>("size"))
    }

    /// Get table counts for key entities
    pub async fn get_entity_counts(&self, database_name: &str) -> Result<EntityCounts> {
        let pool = PgPool::connect(&self.connection_string(database_name))
            .await
            .context("Failed to connect to database")?;

        let counts = EntityCounts {
            organizations: Self::count_table(&pool, "claims.organization").await?,
            facilities: Self::count_table(&pool, "claims.facility").await?,
            providers: Self::count_table(&pool, "claims.provider").await?,
            encounters: Self::count_table(&pool, "claims.encounter").await?,
            service_lines: Self::count_table(&pool, "claims.service_line").await?,
            raw_claims_pending: Self::count_pending_claims(&pool).await?,
        };

        Ok(counts)
    }

    async fn count_table(pool: &PgPool, table: &str) -> Result<i64> {
        let sql = format!("SELECT COUNT(*) as count FROM {}", table);
        let row = sqlx::query(&sql)
            .fetch_one(pool)
            .await
            .unwrap_or_else(|_| {
                // Return a row-like struct that returns 0 if table doesn't exist
                panic!("Table {} doesn't exist", table)
            });

        Ok(row.try_get::<i64, _>("count").unwrap_or(0))
    }

    async fn count_pending_claims(pool: &PgPool) -> Result<i64> {
        let row = sqlx::query(
            "SELECT COUNT(*) as count FROM staging.raw_claims WHERE processing_status = 'PENDING'",
        )
        .fetch_optional(pool)
        .await
        .context("Failed to count pending claims")?;

        Ok(row.map(|r| r.get::<i64, _>("count")).unwrap_or(0))
    }

    /// Validate a database name
    fn is_valid_database_name(name: &str) -> bool {
        if name.is_empty() || name.len() > 63 {
            return false;
        }

        // Must start with a letter or underscore
        let first = name.chars().next().unwrap();
        if !first.is_ascii_alphabetic() && first != '_' {
            return false;
        }

        // Must contain only alphanumeric and underscore
        name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
    }

    /// Connect to a specific database and return a pool
    pub async fn connect(&self, database_name: &str) -> Result<PgPool> {
        PgPool::connect(&self.connection_string(database_name))
            .await
            .with_context(|| format!("Failed to connect to database '{}'", database_name))
    }
}

#[derive(Debug, Clone)]
pub struct EntityCounts {
    pub organizations: i64,
    pub facilities: i64,
    pub providers: i64,
    pub encounters: i64,
    pub service_lines: i64,
    pub raw_claims_pending: i64,
}
