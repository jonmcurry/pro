use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use sqlx::postgres::PgPool;
use sqlx::Row;

/// Information about a project database
#[derive(Debug, Clone)]
pub struct ProjectInfo {
    pub id: i32,
    pub project_name: String,
    pub database_name: String,
    pub organization: Option<String>,
    pub application_version: Option<String>,
    pub backend_version: Option<String>,
    pub database_version: Option<String>,
    pub connection_information: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub last_used_at: Option<DateTime<Utc>>,
    pub is_active: bool,
    pub notes: Option<String>,
}

/// Service for querying the SmartProAudit registry database
pub struct RegistryService {
    pool: PgPool,
}

impl RegistryService {
    /// Create a new registry service by connecting to SmartProAudit database
    pub async fn connect(host: &str, port: u16, user: &str, password: &str) -> Result<Self> {
        let connection_string = format!(
            "postgres://{}:{}@{}:{}/smartproaudit",
            user, password, host, port
        );

        let pool = PgPool::connect(&connection_string)
            .await
            .context("Failed to connect to SmartProAudit database")?;

        Ok(Self { pool })
    }

    /// Get all registered projects
    pub async fn list_projects(&self) -> Result<Vec<ProjectInfo>> {
        let rows = sqlx::query(
            r#"
            SELECT
                id, project_name, database_name, organization,
                application_version, backend_version, database_version,
                connection_information, created_at, updated_at, last_used_at,
                is_active, notes
            FROM projects.project
            ORDER BY last_used_at DESC NULLS LAST, created_at DESC
            "#,
        )
        .fetch_all(&self.pool)
        .await
        .context("Failed to query projects")?;

        let mut projects = Vec::new();
        for row in rows {
            projects.push(ProjectInfo {
                id: row.get("id"),
                project_name: row.get("project_name"),
                database_name: row.get("database_name"),
                organization: row.get("organization"),
                application_version: row.get("application_version"),
                backend_version: row.get("backend_version"),
                database_version: row.get("database_version"),
                connection_information: row.get("connection_information"),
                created_at: row.get("created_at"),
                updated_at: row.get("updated_at"),
                last_used_at: row.get("last_used_at"),
                is_active: row.get("is_active"),
                notes: row.get("notes"),
            });
        }

        Ok(projects)
    }

    /// Get a specific project by database name
    pub async fn get_project(&self, database_name: &str) -> Result<Option<ProjectInfo>> {
        let row = sqlx::query(
            r#"
            SELECT
                id, project_name, database_name, organization,
                application_version, backend_version, database_version,
                connection_information, created_at, updated_at, last_used_at,
                is_active, notes
            FROM projects.project
            WHERE database_name = $1
            "#,
        )
        .bind(database_name)
        .fetch_optional(&self.pool)
        .await
        .context("Failed to query project")?;

        Ok(row.map(|row| ProjectInfo {
            id: row.get("id"),
            project_name: row.get("project_name"),
            database_name: row.get("database_name"),
            organization: row.get("organization"),
            application_version: row.get("application_version"),
            backend_version: row.get("backend_version"),
            database_version: row.get("database_version"),
            connection_information: row.get("connection_information"),
            created_at: row.get("created_at"),
            updated_at: row.get("updated_at"),
            last_used_at: row.get("last_used_at"),
            is_active: row.get("is_active"),
            notes: row.get("notes"),
        }))
    }

    /// Get the currently active project
    /// Reserved for future project switching UI
    #[allow(dead_code)]
    pub async fn get_active_project(&self) -> Result<Option<ProjectInfo>> {
        let row = sqlx::query(
            r#"
            SELECT
                id, project_name, database_name, organization,
                application_version, backend_version, database_version,
                connection_information, created_at, updated_at, last_used_at,
                is_active, notes
            FROM projects.project
            WHERE is_active = true
            LIMIT 1
            "#,
        )
        .fetch_optional(&self.pool)
        .await
        .context("Failed to query active project")?;

        Ok(row.map(|row| ProjectInfo {
            id: row.get("id"),
            project_name: row.get("project_name"),
            database_name: row.get("database_name"),
            organization: row.get("organization"),
            application_version: row.get("application_version"),
            backend_version: row.get("backend_version"),
            database_version: row.get("database_version"),
            connection_information: row.get("connection_information"),
            created_at: row.get("created_at"),
            updated_at: row.get("updated_at"),
            last_used_at: row.get("last_used_at"),
            is_active: row.get("is_active"),
            notes: row.get("notes"),
        }))
    }

    /// Register a new project in the registry
    pub async fn register_project(
        &self,
        project_name: &str,
        database_name: &str,
        organization: Option<&str>,
        application_version: &str,
        database_version: &str,
        connection_info: &str,
    ) -> Result<i32> {
        let row = sqlx::query(
            r#"
            INSERT INTO projects.project (
                project_name, database_name, organization,
                application_version, database_version, connection_information,
                is_active
            ) VALUES ($1, $2, $3, $4, $5, $6, false)
            RETURNING id
            "#,
        )
        .bind(project_name)
        .bind(database_name)
        .bind(organization)
        .bind(application_version)
        .bind(database_version)
        .bind(connection_info)
        .fetch_one(&self.pool)
        .await
        .context("Failed to register project")?;

        Ok(row.get("id"))
    }

    /// Set a project as active (and deactivate all others)
    pub async fn set_active_project(&self, database_name: &str) -> Result<()> {
        // Deactivate all projects
        sqlx::query("UPDATE projects.project SET is_active = false")
            .execute(&self.pool)
            .await
            .context("Failed to deactivate projects")?;

        // Activate the specified project and update last_used_at
        sqlx::query(
            r#"
            UPDATE projects.project
            SET is_active = true, last_used_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
            WHERE database_name = $1
            "#,
        )
        .bind(database_name)
        .execute(&self.pool)
        .await
        .context("Failed to activate project")?;

        Ok(())
    }

    /// Update the database version for a project
    pub async fn update_database_version(&self, database_name: &str, version: &str) -> Result<()> {
        sqlx::query(
            r#"
            UPDATE projects.project
            SET database_version = $1, updated_at = CURRENT_TIMESTAMP
            WHERE database_name = $2
            "#,
        )
        .bind(version)
        .bind(database_name)
        .execute(&self.pool)
        .await
        .context("Failed to update database version")?;

        Ok(())
    }

    /// Delete a project from the registry
    pub async fn delete_project(&self, database_name: &str) -> Result<()> {
        sqlx::query("DELETE FROM projects.project WHERE database_name = $1")
            .bind(database_name)
            .execute(&self.pool)
            .await
            .context("Failed to delete project from registry")?;

        Ok(())
    }

    /// Check if a project exists in the registry
    /// Reserved for future project validation feature
    #[allow(dead_code)]
    pub async fn project_exists(&self, database_name: &str) -> Result<bool> {
        let row = sqlx::query(
            "SELECT EXISTS(SELECT 1 FROM projects.project WHERE database_name = $1) as exists",
        )
        .bind(database_name)
        .fetch_one(&self.pool)
        .await
        .context("Failed to check project existence")?;

        Ok(row.get::<bool, _>("exists"))
    }
}
