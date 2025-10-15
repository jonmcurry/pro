use crate::models::{EncounterFlag, FlagIssue, ServiceLineFlag};
use crate::DbPool;
use pro_common::{Error, Result};
use sqlx::{query, query_as};
use uuid::Uuid;

pub struct FlagRepository<'a> {
    pool: &'a DbPool,
}

impl<'a> FlagRepository<'a> {
    pub fn new(pool: &'a DbPool) -> Self {
        Self { pool }
    }

    // ========================================================================
    // ENCOUNTER FLAGS
    // ========================================================================

    /// Get encounter flag by ID
    pub async fn get_encounter_flag_by_id(&self, flag_id: Uuid) -> Result<EncounterFlag> {
        query_as::<_, EncounterFlag>(
            r#"
            SELECT * FROM claims.encounter_flag
            WHERE flag_id = $1
            "#,
        )
        .bind(flag_id)
        .fetch_one(self.pool)
        .await
        .map_err(|e| match e {
            sqlx::Error::RowNotFound => Error::NotFound(format!("EncounterFlag {} not found", flag_id)),
            _ => Error::Database(e),
        })
    }

    /// Get all flags for an encounter
    pub async fn get_encounter_flags(&self, encounter_id: Uuid) -> Result<Vec<EncounterFlag>> {
        query_as::<_, EncounterFlag>(
            r#"
            SELECT * FROM claims.encounter_flag
            WHERE encounter_id = $1
            ORDER BY created_at DESC
            "#,
        )
        .bind(encounter_id)
        .fetch_all(self.pool)
        .await
        .map_err(Error::Database)
    }

    /// Get encounter flags by status
    pub async fn get_encounter_flags_by_status(
        &self,
        encounter_id: Uuid,
        status: &str,
    ) -> Result<Vec<EncounterFlag>> {
        query_as::<_, EncounterFlag>(
            r#"
            SELECT * FROM claims.encounter_flag
            WHERE encounter_id = $1
            AND flag_status = $2
            ORDER BY created_at DESC
            "#,
        )
        .bind(encounter_id)
        .bind(status)
        .fetch_all(self.pool)
        .await
        .map_err(Error::Database)
    }

    /// Get encounter flags by severity
    pub async fn get_encounter_flags_by_severity(
        &self,
        encounter_id: Uuid,
        severity: &str,
    ) -> Result<Vec<EncounterFlag>> {
        query_as::<_, EncounterFlag>(
            r#"
            SELECT * FROM claims.encounter_flag
            WHERE encounter_id = $1
            AND severity = $2
            ORDER BY created_at DESC
            "#,
        )
        .bind(encounter_id)
        .bind(severity)
        .fetch_all(self.pool)
        .await
        .map_err(Error::Database)
    }

    /// Create an encounter flag
    pub async fn create_encounter_flag(&self, flag: &EncounterFlag) -> Result<Uuid> {
        let id = query_as::<_, (Uuid,)>(
            r#"
            INSERT INTO claims.encounter_flag (
                encounter_id, issue_id,
                flag_type, severity, flag_reason, flagged_element,
                proposed_code, proposed_modifier, proposed_quantity, proposed_diagnosis_code,
                flag_status, created_by
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            RETURNING flag_id
            "#,
        )
        .bind(flag.encounter_id)
        .bind(flag.issue_id)
        .bind(&flag.flag_type)
        .bind(&flag.severity)
        .bind(&flag.flag_reason)
        .bind(&flag.flagged_element)
        .bind(&flag.proposed_code)
        .bind(&flag.proposed_modifier)
        .bind(flag.proposed_quantity)
        .bind(&flag.proposed_diagnosis_code)
        .bind(&flag.flag_status)
        .bind(&flag.created_by)
        .fetch_one(self.pool)
        .await
        .map_err(Error::Database)?;

        Ok(id.0)
    }

    /// Update encounter flag status
    pub async fn update_encounter_flag_status(
        &self,
        flag_id: Uuid,
        status: &str,
        resolution_note: Option<&str>,
        resolved_by: Option<&str>,
    ) -> Result<()> {
        let rows_affected = query(
            r#"
            UPDATE claims.encounter_flag
            SET
                flag_status = $2,
                resolution_note = $3,
                resolved_at = CASE WHEN $2 = 'RESOLVED' THEN CURRENT_TIMESTAMP ELSE resolved_at END,
                resolved_by = $4
            WHERE flag_id = $1
            "#,
        )
        .bind(flag_id)
        .bind(status)
        .bind(resolution_note)
        .bind(resolved_by)
        .execute(self.pool)
        .await
        .map_err(Error::Database)?
        .rows_affected();

        if rows_affected == 0 {
            return Err(Error::NotFound(format!("EncounterFlag {} not found", flag_id)));
        }

        Ok(())
    }

    /// Count encounter flags by status
    pub async fn count_encounter_flags_by_status(
        &self,
        encounter_id: Uuid,
        status: &str,
    ) -> Result<i64> {
        let count: (i64,) = query_as(
            r#"
            SELECT COUNT(*) FROM claims.encounter_flag
            WHERE encounter_id = $1
            AND flag_status = $2
            "#,
        )
        .bind(encounter_id)
        .bind(status)
        .fetch_one(self.pool)
        .await
        .map_err(Error::Database)?;

        Ok(count.0)
    }

    // ========================================================================
    // SERVICE LINE FLAGS
    // ========================================================================

    /// Get service line flag by ID
    pub async fn get_service_line_flag_by_id(&self, flag_id: Uuid) -> Result<ServiceLineFlag> {
        query_as::<_, ServiceLineFlag>(
            r#"
            SELECT * FROM claims.service_line_flag
            WHERE flag_id = $1
            "#,
        )
        .bind(flag_id)
        .fetch_one(self.pool)
        .await
        .map_err(|e| match e {
            sqlx::Error::RowNotFound => Error::NotFound(format!("ServiceLineFlag {} not found", flag_id)),
            _ => Error::Database(e),
        })
    }

    /// Get all flags for a service line
    pub async fn get_service_line_flags(&self, service_line_id: Uuid) -> Result<Vec<ServiceLineFlag>> {
        query_as::<_, ServiceLineFlag>(
            r#"
            SELECT * FROM claims.service_line_flag
            WHERE service_line_id = $1
            ORDER BY created_at DESC
            "#,
        )
        .bind(service_line_id)
        .fetch_all(self.pool)
        .await
        .map_err(Error::Database)
    }

    /// Get service line flags by status
    pub async fn get_service_line_flags_by_status(
        &self,
        service_line_id: Uuid,
        status: &str,
    ) -> Result<Vec<ServiceLineFlag>> {
        query_as::<_, ServiceLineFlag>(
            r#"
            SELECT * FROM claims.service_line_flag
            WHERE service_line_id = $1
            AND flag_status = $2
            ORDER BY created_at DESC
            "#,
        )
        .bind(service_line_id)
        .bind(status)
        .fetch_all(self.pool)
        .await
        .map_err(Error::Database)
    }

    /// Get service line flags by severity
    pub async fn get_service_line_flags_by_severity(
        &self,
        service_line_id: Uuid,
        severity: &str,
    ) -> Result<Vec<ServiceLineFlag>> {
        query_as::<_, ServiceLineFlag>(
            r#"
            SELECT * FROM claims.service_line_flag
            WHERE service_line_id = $1
            AND severity = $2
            ORDER BY created_at DESC
            "#,
        )
        .bind(service_line_id)
        .bind(severity)
        .fetch_all(self.pool)
        .await
        .map_err(Error::Database)
    }

    /// Create a service line flag
    pub async fn create_service_line_flag(&self, flag: &ServiceLineFlag) -> Result<Uuid> {
        let id = query_as::<_, (Uuid,)>(
            r#"
            INSERT INTO claims.service_line_flag (
                service_line_id, issue_id,
                flag_type, severity, flag_reason, flagged_element,
                proposed_code, proposed_modifier, proposed_quantity,
                flag_status, created_by
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            RETURNING flag_id
            "#,
        )
        .bind(flag.service_line_id)
        .bind(flag.issue_id)
        .bind(&flag.flag_type)
        .bind(&flag.severity)
        .bind(&flag.flag_reason)
        .bind(&flag.flagged_element)
        .bind(&flag.proposed_code)
        .bind(&flag.proposed_modifier)
        .bind(flag.proposed_quantity)
        .bind(&flag.flag_status)
        .bind(&flag.created_by)
        .fetch_one(self.pool)
        .await
        .map_err(Error::Database)?;

        Ok(id.0)
    }

    /// Update service line flag status
    pub async fn update_service_line_flag_status(
        &self,
        flag_id: Uuid,
        status: &str,
        resolution_note: Option<&str>,
        resolved_by: Option<&str>,
    ) -> Result<()> {
        let rows_affected = query(
            r#"
            UPDATE claims.service_line_flag
            SET
                flag_status = $2,
                resolution_note = $3,
                resolved_at = CASE WHEN $2 = 'RESOLVED' THEN CURRENT_TIMESTAMP ELSE resolved_at END,
                resolved_by = $4
            WHERE flag_id = $1
            "#,
        )
        .bind(flag_id)
        .bind(status)
        .bind(resolution_note)
        .bind(resolved_by)
        .execute(self.pool)
        .await
        .map_err(Error::Database)?
        .rows_affected();

        if rows_affected == 0 {
            return Err(Error::NotFound(format!("ServiceLineFlag {} not found", flag_id)));
        }

        Ok(())
    }

    /// Count service line flags by status
    pub async fn count_service_line_flags_by_status(
        &self,
        service_line_id: Uuid,
        status: &str,
    ) -> Result<i64> {
        let count: (i64,) = query_as(
            r#"
            SELECT COUNT(*) FROM claims.service_line_flag
            WHERE service_line_id = $1
            AND flag_status = $2
            "#,
        )
        .bind(service_line_id)
        .bind(status)
        .fetch_one(self.pool)
        .await
        .map_err(Error::Database)?;

        Ok(count.0)
    }

    // ========================================================================
    // FLAG ISSUES (REFERENCE DATA)
    // ========================================================================

    /// Get flag issue by ID
    pub async fn get_flag_issue_by_id(&self, issue_id: Uuid) -> Result<FlagIssue> {
        query_as::<_, FlagIssue>(
            r#"
            SELECT * FROM claims.flag_issue
            WHERE issue_id = $1
            "#,
        )
        .bind(issue_id)
        .fetch_one(self.pool)
        .await
        .map_err(|e| match e {
            sqlx::Error::RowNotFound => Error::NotFound(format!("FlagIssue {} not found", issue_id)),
            _ => Error::Database(e),
        })
    }

    /// Get flag issue by code
    pub async fn get_flag_issue_by_code(&self, issue_code: &str) -> Result<FlagIssue> {
        query_as::<_, FlagIssue>(
            r#"
            SELECT * FROM claims.flag_issue
            WHERE issue_code = $1
            AND is_active = true
            "#,
        )
        .bind(issue_code)
        .fetch_one(self.pool)
        .await
        .map_err(|e| match e {
            sqlx::Error::RowNotFound => Error::NotFound(format!("FlagIssue {} not found", issue_code)),
            _ => Error::Database(e),
        })
    }

    /// List all active flag issues
    pub async fn list_active_flag_issues(&self) -> Result<Vec<FlagIssue>> {
        query_as::<_, FlagIssue>(
            r#"
            SELECT * FROM claims.flag_issue
            WHERE is_active = true
            ORDER BY issue_code ASC
            "#,
        )
        .fetch_all(self.pool)
        .await
        .map_err(Error::Database)
    }

    /// List flag issues by category
    pub async fn list_flag_issues_by_category(&self, category_id: Uuid) -> Result<Vec<FlagIssue>> {
        query_as::<_, FlagIssue>(
            r#"
            SELECT * FROM claims.flag_issue
            WHERE category_id = $1
            AND is_active = true
            ORDER BY issue_code ASC
            "#,
        )
        .bind(category_id)
        .fetch_all(self.pool)
        .await
        .map_err(Error::Database)
    }

    /// List flag issues by severity
    pub async fn list_flag_issues_by_severity(&self, severity: &str) -> Result<Vec<FlagIssue>> {
        query_as::<_, FlagIssue>(
            r#"
            SELECT * FROM claims.flag_issue
            WHERE severity = $1
            AND is_active = true
            ORDER BY issue_code ASC
            "#,
        )
        .bind(severity)
        .fetch_all(self.pool)
        .await
        .map_err(Error::Database)
    }

    // ========================================================================
    // BATCH OPERATIONS
    // ========================================================================

    /// Create multiple encounter flags
    pub async fn create_encounter_flags_batch(&self, flags: &[EncounterFlag]) -> Result<Vec<Uuid>> {
        let mut ids = Vec::new();

        for flag in flags {
            let id = self.create_encounter_flag(flag).await?;
            ids.push(id);
        }

        Ok(ids)
    }

    /// Create multiple service line flags
    pub async fn create_service_line_flags_batch(&self, flags: &[ServiceLineFlag]) -> Result<Vec<Uuid>> {
        let mut ids = Vec::new();

        for flag in flags {
            let id = self.create_service_line_flag(flag).await?;
            ids.push(id);
        }

        Ok(ids)
    }

    /// Get all flags for an encounter (both encounter and service line flags)
    pub async fn get_all_flags_for_encounter(
        &self,
        encounter_id: Uuid,
    ) -> Result<(Vec<EncounterFlag>, Vec<ServiceLineFlag>)> {
        let encounter_flags = self.get_encounter_flags(encounter_id).await?;

        // Get all service line IDs for this encounter
        let service_line_ids: Vec<(Uuid,)> = query_as(
            r#"
            SELECT service_line_id FROM claims.service_line
            WHERE encounter_id = $1
            "#,
        )
        .bind(encounter_id)
        .fetch_all(self.pool)
        .await
        .map_err(Error::Database)?;

        let mut service_line_flags = Vec::new();
        for (service_line_id,) in service_line_ids {
            let flags = self.get_service_line_flags(service_line_id).await?;
            service_line_flags.extend(flags);
        }

        Ok((encounter_flags, service_line_flags))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connection::create_pool_default;

    #[tokio::test]
    #[ignore] // Requires database
    async fn test_flag_repository() {
        let pool = create_pool_default().await.unwrap();
        let repo = FlagRepository::new(&pool);

        // Test list_active_flag_issues
        let issues = repo.list_active_flag_issues().await;
        assert!(issues.is_ok());
    }
}
