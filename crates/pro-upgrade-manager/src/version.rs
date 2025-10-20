use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::{PgPool, Row};
use tracing::{info, warn};

use crate::error::Result;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionInfo {
    pub version: String,
    pub installed_at: DateTime<Utc>,
    pub upgraded_from: Option<String>,
    pub notes: Option<String>,
}

pub struct VersionManager {
    pool: PgPool,
}

impl VersionManager {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Get the current application version from the database
    pub async fn get_current_version(&self) -> Result<Option<VersionInfo>> {
        info!("Checking current application version");

        // First check if the version tracking table exists
        let table_exists: bool = sqlx::query_scalar(
            r#"
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = 'staging'
                AND table_name = 'application_version'
            )
            "#,
        )
        .fetch_one(&self.pool)
        .await?;

        if !table_exists {
            warn!("Version tracking table does not exist - this is a legacy installation");
            return Ok(None);
        }

        // Get the most recent version
        let row = sqlx::query(
            r#"
            SELECT version, installed_at, upgraded_from, notes
            FROM staging.application_version
            ORDER BY installed_at DESC
            LIMIT 1
            "#
        )
        .fetch_optional(&self.pool)
        .await?;

        let version = row.map(|r| VersionInfo {
            version: r.get("version"),
            installed_at: r.get("installed_at"),
            upgraded_from: r.get("upgraded_from"),
            notes: r.get("notes"),
        });

        if let Some(ref v) = version {
            info!("Current version: {}", v.version);
        } else {
            warn!("Version tracking table exists but is empty");
        }

        Ok(version)
    }

    /// Check if the database exists
    pub async fn database_exists(&self, db_name: &str) -> Result<bool> {
        let exists: bool = sqlx::query_scalar(
            r#"
            SELECT EXISTS(
                SELECT FROM pg_database WHERE datname = $1
            )
            "#,
        )
        .bind(db_name)
        .fetch_one(&self.pool)
        .await?;

        Ok(exists)
    }

    /// Check if the database has the claims schema (indicates it's a Professional SMART database)
    pub async fn is_professional_smart_database(&self) -> Result<bool> {
        let has_claims_schema: bool = sqlx::query_scalar(
            r#"
            SELECT EXISTS (
                SELECT FROM information_schema.schemata
                WHERE schema_name = 'claims'
            )
            "#,
        )
        .fetch_one(&self.pool)
        .await?;

        Ok(has_claims_schema)
    }

    /// Detect the installation type: Fresh, Legacy (pre-version-tracking), or Upgrade
    pub async fn detect_installation_type(&self) -> Result<InstallationType> {
        // Check if version tracking exists
        let current_version = self.get_current_version().await?;

        if let Some(version) = current_version {
            // Has version tracking - this is an upgrade
            info!("Detected upgrade installation from version {}", version.version);
            return Ok(InstallationType::Upgrade(version));
        }

        // No version tracking - check if it's a legacy install or fresh
        let is_pro_smart = self.is_professional_smart_database().await?;

        if is_pro_smart {
            // Has claims schema but no version tracking - legacy installation
            info!("Detected legacy installation (pre-version-tracking)");
            Ok(InstallationType::Legacy)
        } else {
            // No claims schema - fresh installation
            info!("Detected fresh installation");
            Ok(InstallationType::Fresh)
        }
    }

    /// Record a new version installation
    pub async fn record_version(
        &self,
        version: &str,
        upgraded_from: Option<&str>,
        notes: Option<&str>,
    ) -> Result<()> {
        info!(
            "Recording version {} (upgraded from: {:?})",
            version, upgraded_from
        );

        sqlx::query(
            r#"
            INSERT INTO staging.application_version (version, installed_at, upgraded_from, notes)
            VALUES ($1, NOW(), $2, $3)
            ON CONFLICT (version) DO UPDATE
            SET upgraded_from = EXCLUDED.upgraded_from,
                notes = EXCLUDED.notes
            "#
        )
        .bind(version)
        .bind(upgraded_from)
        .bind(notes)
        .execute(&self.pool)
        .await?;

        info!("Version {} recorded successfully", version);
        Ok(())
    }

    /// Get version history
    pub async fn get_version_history(&self) -> Result<Vec<VersionInfo>> {
        let rows = sqlx::query(
            r#"
            SELECT version, installed_at, upgraded_from, notes
            FROM staging.application_version
            ORDER BY installed_at DESC
            "#
        )
        .fetch_all(&self.pool)
        .await?;

        let versions = rows.into_iter().map(|r| VersionInfo {
            version: r.get("version"),
            installed_at: r.get("installed_at"),
            upgraded_from: r.get("upgraded_from"),
            notes: r.get("notes"),
        }).collect();

        Ok(versions)
    }
}

#[derive(Debug, Clone)]
pub enum InstallationType {
    Fresh,
    Legacy,
    Upgrade(VersionInfo),
}

impl InstallationType {
    pub fn is_fresh(&self) -> bool {
        matches!(self, InstallationType::Fresh)
    }

    pub fn is_legacy(&self) -> bool {
        matches!(self, InstallationType::Legacy)
    }

    pub fn is_upgrade(&self) -> bool {
        matches!(self, InstallationType::Upgrade(_))
    }

    pub fn version(&self) -> Option<&str> {
        match self {
            InstallationType::Upgrade(v) => Some(&v.version),
            _ => None,
        }
    }
}
