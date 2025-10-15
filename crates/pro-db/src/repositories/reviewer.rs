use crate::models::Reviewer;
use crate::DbPool;
use pro_common::{Error, Result};
use sqlx::{query, query_as};
use uuid::Uuid;

pub struct ReviewerRepository<'a> {
    pool: &'a DbPool,
}

impl<'a> ReviewerRepository<'a> {
    pub fn new(pool: &'a DbPool) -> Self {
        Self { pool }
    }

    /// Get reviewer by ID
    pub async fn get_by_id(&self, reviewer_id: Uuid) -> Result<Reviewer> {
        query_as::<_, Reviewer>(
            r#"
            SELECT * FROM core.reviewer
            WHERE reviewer_id = $1
            "#,
        )
        .bind(reviewer_id)
        .fetch_one(self.pool)
        .await
        .map_err(|e| match e {
            sqlx::Error::RowNotFound => Error::NotFound(format!("Reviewer {} not found", reviewer_id)),
            _ => Error::Database(e),
        })
    }

    /// Get reviewer by code
    pub async fn get_by_reviewer_code(&self, reviewer_code: &str) -> Result<Reviewer> {
        query_as::<_, Reviewer>(
            r#"
            SELECT * FROM core.reviewer
            WHERE reviewer_code = $1
            AND is_active = true
            "#,
        )
        .bind(reviewer_code)
        .fetch_one(self.pool)
        .await
        .map_err(|e| match e {
            sqlx::Error::RowNotFound => Error::NotFound(format!("Reviewer with code {} not found", reviewer_code)),
            _ => Error::Database(e),
        })
    }

    /// List reviewers by organization
    pub async fn list_by_organization(
        &self,
        organization_id: Uuid,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<Reviewer>> {
        query_as::<_, Reviewer>(
            r#"
            SELECT * FROM core.reviewer
            WHERE organization_id = $1
            ORDER BY last_name ASC, first_name ASC
            LIMIT $2 OFFSET $3
            "#,
        )
        .bind(organization_id)
        .bind(limit)
        .bind(offset)
        .fetch_all(self.pool)
        .await
        .map_err(Error::Database)
    }

    /// List active reviewers by organization
    pub async fn list_active_by_organization(
        &self,
        organization_id: Uuid,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<Reviewer>> {
        query_as::<_, Reviewer>(
            r#"
            SELECT * FROM core.reviewer
            WHERE organization_id = $1
            AND is_active = true
            ORDER BY last_name ASC, first_name ASC
            LIMIT $2 OFFSET $3
            "#,
        )
        .bind(organization_id)
        .bind(limit)
        .bind(offset)
        .fetch_all(self.pool)
        .await
        .map_err(Error::Database)
    }

    /// List reviewers by group
    pub async fn list_by_group(
        &self,
        reviewer_group: &str,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<Reviewer>> {
        query_as::<_, Reviewer>(
            r#"
            SELECT * FROM core.reviewer
            WHERE reviewer_group = $1
            AND is_active = true
            ORDER BY last_name ASC, first_name ASC
            LIMIT $2 OFFSET $3
            "#,
        )
        .bind(reviewer_group)
        .bind(limit)
        .bind(offset)
        .fetch_all(self.pool)
        .await
        .map_err(Error::Database)
    }

    /// List reviewers by certification
    pub async fn list_by_certification(
        &self,
        certification: &str,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<Reviewer>> {
        query_as::<_, Reviewer>(
            r#"
            SELECT * FROM core.reviewer
            WHERE $1 = ANY(certifications)
            AND is_active = true
            ORDER BY last_name ASC, first_name ASC
            LIMIT $2 OFFSET $3
            "#,
        )
        .bind(certification)
        .bind(limit)
        .bind(offset)
        .fetch_all(self.pool)
        .await
        .map_err(Error::Database)
    }

    /// Create a new reviewer
    pub async fn create(&self, reviewer: &Reviewer) -> Result<Uuid> {
        let id = query_as::<_, (Uuid,)>(
            r#"
            INSERT INTO core.reviewer (
                reviewer_id,
                reviewer_code,
                last_name,
                first_name,
                middle_name,
                reviewer_group,
                certifications,
                organization_id,
                email,
                is_active
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            RETURNING reviewer_id
            "#,
        )
        .bind(reviewer.reviewer_id)
        .bind(&reviewer.reviewer_code)
        .bind(&reviewer.last_name)
        .bind(&reviewer.first_name)
        .bind(&reviewer.middle_name)
        .bind(&reviewer.reviewer_group)
        .bind(&reviewer.certifications)
        .bind(reviewer.organization_id)
        .bind(&reviewer.email)
        .bind(reviewer.is_active)
        .fetch_one(self.pool)
        .await
        .map_err(Error::Database)?;

        Ok(id.0)
    }

    /// Update reviewer information
    pub async fn update(&self, reviewer: &Reviewer) -> Result<()> {
        let rows_affected = query(
            r#"
            UPDATE core.reviewer
            SET
                reviewer_code = $2,
                last_name = $3,
                first_name = $4,
                middle_name = $5,
                reviewer_group = $6,
                certifications = $7,
                organization_id = $8,
                email = $9,
                is_active = $10,
                updated_at = CURRENT_TIMESTAMP
            WHERE reviewer_id = $1
            "#,
        )
        .bind(reviewer.reviewer_id)
        .bind(&reviewer.reviewer_code)
        .bind(&reviewer.last_name)
        .bind(&reviewer.first_name)
        .bind(&reviewer.middle_name)
        .bind(&reviewer.reviewer_group)
        .bind(&reviewer.certifications)
        .bind(reviewer.organization_id)
        .bind(&reviewer.email)
        .bind(reviewer.is_active)
        .execute(self.pool)
        .await
        .map_err(Error::Database)?
        .rows_affected();

        if rows_affected == 0 {
            return Err(Error::NotFound(format!("Reviewer {} not found", reviewer.reviewer_id)));
        }

        Ok(())
    }

    /// Update reviewer status (activate/deactivate)
    pub async fn update_status(&self, reviewer_id: Uuid, is_active: bool) -> Result<()> {
        let rows_affected = query(
            r#"
            UPDATE core.reviewer
            SET
                is_active = $2,
                updated_at = CURRENT_TIMESTAMP
            WHERE reviewer_id = $1
            "#,
        )
        .bind(reviewer_id)
        .bind(is_active)
        .execute(self.pool)
        .await
        .map_err(Error::Database)?
        .rows_affected();

        if rows_affected == 0 {
            return Err(Error::NotFound(format!("Reviewer {} not found", reviewer_id)));
        }

        Ok(())
    }

    /// Update reviewer organization
    pub async fn update_organization(
        &self,
        reviewer_id: Uuid,
        organization_id: Option<Uuid>,
    ) -> Result<()> {
        let rows_affected = query(
            r#"
            UPDATE core.reviewer
            SET
                organization_id = $2,
                updated_at = CURRENT_TIMESTAMP
            WHERE reviewer_id = $1
            "#,
        )
        .bind(reviewer_id)
        .bind(organization_id)
        .execute(self.pool)
        .await
        .map_err(Error::Database)?
        .rows_affected();

        if rows_affected == 0 {
            return Err(Error::NotFound(format!("Reviewer {} not found", reviewer_id)));
        }

        Ok(())
    }

    /// Add certification to reviewer
    pub async fn add_certification(&self, reviewer_id: Uuid, certification: &str) -> Result<()> {
        let rows_affected = query(
            r#"
            UPDATE core.reviewer
            SET
                certifications = array_append(certifications, $2),
                updated_at = CURRENT_TIMESTAMP
            WHERE reviewer_id = $1
            AND NOT ($2 = ANY(certifications))
            "#,
        )
        .bind(reviewer_id)
        .bind(certification)
        .execute(self.pool)
        .await
        .map_err(Error::Database)?
        .rows_affected();

        if rows_affected == 0 {
            return Err(Error::NotFound(format!("Reviewer {} not found or certification already exists", reviewer_id)));
        }

        Ok(())
    }

    /// Remove certification from reviewer
    pub async fn remove_certification(&self, reviewer_id: Uuid, certification: &str) -> Result<()> {
        let rows_affected = query(
            r#"
            UPDATE core.reviewer
            SET
                certifications = array_remove(certifications, $2),
                updated_at = CURRENT_TIMESTAMP
            WHERE reviewer_id = $1
            "#,
        )
        .bind(reviewer_id)
        .bind(certification)
        .execute(self.pool)
        .await
        .map_err(Error::Database)?
        .rows_affected();

        if rows_affected == 0 {
            return Err(Error::NotFound(format!("Reviewer {} not found", reviewer_id)));
        }

        Ok(())
    }

    /// Soft delete reviewer (deactivate)
    pub async fn soft_delete(&self, reviewer_id: Uuid) -> Result<()> {
        self.update_status(reviewer_id, false).await
    }

    /// Check if reviewer code exists
    pub async fn exists_by_reviewer_code(&self, reviewer_code: &str) -> Result<bool> {
        let count: (i64,) = query_as(
            r#"
            SELECT COUNT(*) FROM core.reviewer
            WHERE reviewer_code = $1
            "#,
        )
        .bind(reviewer_code)
        .fetch_one(self.pool)
        .await
        .map_err(Error::Database)?;

        Ok(count.0 > 0)
    }

    /// Count reviewers by organization
    pub async fn count_by_organization(&self, organization_id: Uuid) -> Result<i64> {
        let count: (i64,) = query_as(
            r#"
            SELECT COUNT(*) FROM core.reviewer
            WHERE organization_id = $1
            "#,
        )
        .bind(organization_id)
        .fetch_one(self.pool)
        .await
        .map_err(Error::Database)?;

        Ok(count.0)
    }

    /// Count active reviewers by organization
    pub async fn count_active_by_organization(&self, organization_id: Uuid) -> Result<i64> {
        let count: (i64,) = query_as(
            r#"
            SELECT COUNT(*) FROM core.reviewer
            WHERE organization_id = $1
            AND is_active = true
            "#,
        )
        .bind(organization_id)
        .fetch_one(self.pool)
        .await
        .map_err(Error::Database)?;

        Ok(count.0)
    }

    /// Count reviewers by group
    pub async fn count_by_group(&self, reviewer_group: &str) -> Result<i64> {
        let count: (i64,) = query_as(
            r#"
            SELECT COUNT(*) FROM core.reviewer
            WHERE reviewer_group = $1
            AND is_active = true
            "#,
        )
        .bind(reviewer_group)
        .fetch_one(self.pool)
        .await
        .map_err(Error::Database)?;

        Ok(count.0)
    }

    /// Get all reviewers (no pagination, use with caution)
    pub async fn get_all(&self) -> Result<Vec<Reviewer>> {
        query_as::<_, Reviewer>(
            r#"
            SELECT * FROM core.reviewer
            ORDER BY last_name ASC, first_name ASC
            "#,
        )
        .fetch_all(self.pool)
        .await
        .map_err(Error::Database)
    }

    /// Get all active reviewers (no pagination)
    pub async fn get_all_active(&self) -> Result<Vec<Reviewer>> {
        query_as::<_, Reviewer>(
            r#"
            SELECT * FROM core.reviewer
            WHERE is_active = true
            ORDER BY last_name ASC, first_name ASC
            "#,
        )
        .fetch_all(self.pool)
        .await
        .map_err(Error::Database)
    }

    /// Search reviewers by name (partial match)
    pub async fn search_by_name(
        &self,
        name_pattern: &str,
        limit: i64,
    ) -> Result<Vec<Reviewer>> {
        query_as::<_, Reviewer>(
            r#"
            SELECT * FROM core.reviewer
            WHERE (
                last_name ILIKE $1
                OR first_name ILIKE $1
                OR CONCAT(first_name, ' ', last_name) ILIKE $1
            )
            AND is_active = true
            ORDER BY last_name ASC, first_name ASC
            LIMIT $2
            "#,
        )
        .bind(format!("%{}%", name_pattern))
        .bind(limit)
        .fetch_all(self.pool)
        .await
        .map_err(Error::Database)
    }

    /// Batch lookup reviewers by codes
    pub async fn get_by_reviewer_codes(&self, reviewer_codes: &[String]) -> Result<Vec<Reviewer>> {
        query_as::<_, Reviewer>(
            r#"
            SELECT * FROM core.reviewer
            WHERE reviewer_code = ANY($1)
            ORDER BY last_name ASC, first_name ASC
            "#,
        )
        .bind(reviewer_codes)
        .fetch_all(self.pool)
        .await
        .map_err(Error::Database)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connection::create_pool_default;

    #[tokio::test]
    #[ignore] // Requires database
    async fn test_reviewer_repository() {
        let pool = create_pool_default().await.unwrap();
        let repo = ReviewerRepository::new(&pool);

        // Test get_all_active
        let reviewers = repo.get_all_active().await;
        assert!(reviewers.is_ok());
    }
}
