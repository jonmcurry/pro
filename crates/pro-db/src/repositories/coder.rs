use crate::models::Coder;
use crate::DbPool;
use pro_common::{Error, Result};
use sqlx::{query, query_as};


pub struct CoderRepository<'a> {
    pool: &'a DbPool,
}

impl<'a> CoderRepository<'a> {
    pub fn new(pool: &'a DbPool) -> Self {
        Self { pool }
    }

    /// Get coder by ID
    pub async fn get_by_id(&self, coder_id: i64) -> Result<Coder> {
        query_as::<_, Coder>(
            r#"
            SELECT * FROM core.coder
            WHERE coder_id = $1
            "#,
        )
        .bind(coder_id)
        .fetch_one(self.pool)
        .await
        .map_err(|e| match e {
            sqlx::Error::RowNotFound => Error::NotFound(format!("Coder {} not found", coder_id)),
            _ => Error::Database(e),
        })
    }

    /// Get coder by code
    pub async fn get_by_coder_code(&self, coder_code: &str) -> Result<Coder> {
        query_as::<_, Coder>(
            r#"
            SELECT * FROM core.coder
            WHERE coder_code = $1
            AND is_active = true
            "#,
        )
        .bind(coder_code)
        .fetch_one(self.pool)
        .await
        .map_err(|e| match e {
            sqlx::Error::RowNotFound => Error::NotFound(format!("Coder with code {} not found", coder_code)),
            _ => Error::Database(e),
        })
    }

    /// List coders by organization
    pub async fn list_by_organization(
        &self,
        organization_id: i64,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<Coder>> {
        query_as::<_, Coder>(
            r#"
            SELECT * FROM core.coder
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

    /// List active coders by organization
    pub async fn list_active_by_organization(
        &self,
        organization_id: i64,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<Coder>> {
        query_as::<_, Coder>(
            r#"
            SELECT * FROM core.coder
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

    /// List coders by group
    pub async fn list_by_group(
        &self,
        coder_group: &str,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<Coder>> {
        query_as::<_, Coder>(
            r#"
            SELECT * FROM core.coder
            WHERE coder_group = $1
            AND is_active = true
            ORDER BY last_name ASC, first_name ASC
            LIMIT $2 OFFSET $3
            "#,
        )
        .bind(coder_group)
        .bind(limit)
        .bind(offset)
        .fetch_all(self.pool)
        .await
        .map_err(Error::Database)
    }

    /// List coders by certification
    pub async fn list_by_certification(
        &self,
        certification: &str,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<Coder>> {
        query_as::<_, Coder>(
            r#"
            SELECT * FROM core.coder
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

    /// Create a new coder
    pub async fn create(&self, coder: &Coder) -> Result<i64> {
        let id = query_as::<_, (i64,)>(
            r#"
            INSERT INTO core.coder (
                coder_id,
                coder_code,
                last_name,
                first_name,
                middle_name,
                coder_group,
                certifications,
                organization_id,
                email,
                is_active
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            RETURNING coder_id
            "#,
        )
        .bind(coder.coder_id)
        .bind(&coder.coder_code)
        .bind(&coder.last_name)
        .bind(&coder.first_name)
        .bind(&coder.middle_name)
        .bind(&coder.coder_group)
        .bind(&coder.certifications)
        .bind(coder.organization_id)
        .bind(&coder.email)
        .bind(coder.is_active)
        .fetch_one(self.pool)
        .await
        .map_err(Error::Database)?;

        Ok(id.0)
    }

    /// Update coder information
    pub async fn update(&self, coder: &Coder) -> Result<()> {
        let rows_affected = query(
            r#"
            UPDATE core.coder
            SET
                coder_code = $2,
                last_name = $3,
                first_name = $4,
                middle_name = $5,
                coder_group = $6,
                certifications = $7,
                organization_id = $8,
                email = $9,
                is_active = $10,
                updated_at = CURRENT_TIMESTAMP
            WHERE coder_id = $1
            "#,
        )
        .bind(coder.coder_id)
        .bind(&coder.coder_code)
        .bind(&coder.last_name)
        .bind(&coder.first_name)
        .bind(&coder.middle_name)
        .bind(&coder.coder_group)
        .bind(&coder.certifications)
        .bind(coder.organization_id)
        .bind(&coder.email)
        .bind(coder.is_active)
        .execute(self.pool)
        .await
        .map_err(Error::Database)?
        .rows_affected();

        if rows_affected == 0 {
            return Err(Error::NotFound(format!("Coder {} not found", coder.coder_id)));
        }

        Ok(())
    }

    /// Update coder status (activate/deactivate)
    pub async fn update_status(&self, coder_id: i64, is_active: bool) -> Result<()> {
        let rows_affected = query(
            r#"
            UPDATE core.coder
            SET
                is_active = $2,
                updated_at = CURRENT_TIMESTAMP
            WHERE coder_id = $1
            "#,
        )
        .bind(coder_id)
        .bind(is_active)
        .execute(self.pool)
        .await
        .map_err(Error::Database)?
        .rows_affected();

        if rows_affected == 0 {
            return Err(Error::NotFound(format!("Coder {} not found", coder_id)));
        }

        Ok(())
    }

    /// Update coder organization
    pub async fn update_organization(
        &self,
        coder_id: i64,
        organization_id: Option<i64>,
    ) -> Result<()> {
        let rows_affected = query(
            r#"
            UPDATE core.coder
            SET
                organization_id = $2,
                updated_at = CURRENT_TIMESTAMP
            WHERE coder_id = $1
            "#,
        )
        .bind(coder_id)
        .bind(organization_id)
        .execute(self.pool)
        .await
        .map_err(Error::Database)?
        .rows_affected();

        if rows_affected == 0 {
            return Err(Error::NotFound(format!("Coder {} not found", coder_id)));
        }

        Ok(())
    }

    /// Add certification to coder
    pub async fn add_certification(&self, coder_id: i64, certification: &str) -> Result<()> {
        let rows_affected = query(
            r#"
            UPDATE core.coder
            SET
                certifications = array_append(certifications, $2),
                updated_at = CURRENT_TIMESTAMP
            WHERE coder_id = $1
            AND NOT ($2 = ANY(certifications))
            "#,
        )
        .bind(coder_id)
        .bind(certification)
        .execute(self.pool)
        .await
        .map_err(Error::Database)?
        .rows_affected();

        if rows_affected == 0 {
            return Err(Error::NotFound(format!("Coder {} not found or certification already exists", coder_id)));
        }

        Ok(())
    }

    /// Remove certification from coder
    pub async fn remove_certification(&self, coder_id: i64, certification: &str) -> Result<()> {
        let rows_affected = query(
            r#"
            UPDATE core.coder
            SET
                certifications = array_remove(certifications, $2),
                updated_at = CURRENT_TIMESTAMP
            WHERE coder_id = $1
            "#,
        )
        .bind(coder_id)
        .bind(certification)
        .execute(self.pool)
        .await
        .map_err(Error::Database)?
        .rows_affected();

        if rows_affected == 0 {
            return Err(Error::NotFound(format!("Coder {} not found", coder_id)));
        }

        Ok(())
    }

    /// Soft delete coder (deactivate)
    pub async fn soft_delete(&self, coder_id: i64) -> Result<()> {
        self.update_status(coder_id, false).await
    }

    /// Check if coder code exists
    pub async fn exists_by_coder_code(&self, coder_code: &str) -> Result<bool> {
        let count: (i64,) = query_as(
            r#"
            SELECT COUNT(*) FROM core.coder
            WHERE coder_code = $1
            "#,
        )
        .bind(coder_code)
        .fetch_one(self.pool)
        .await
        .map_err(Error::Database)?;

        Ok(count.0 > 0)
    }

    /// Count coders by organization
    pub async fn count_by_organization(&self, organization_id: i64) -> Result<i64> {
        let count: (i64,) = query_as(
            r#"
            SELECT COUNT(*) FROM core.coder
            WHERE organization_id = $1
            "#,
        )
        .bind(organization_id)
        .fetch_one(self.pool)
        .await
        .map_err(Error::Database)?;

        Ok(count.0)
    }

    /// Count active coders by organization
    pub async fn count_active_by_organization(&self, organization_id: i64) -> Result<i64> {
        let count: (i64,) = query_as(
            r#"
            SELECT COUNT(*) FROM core.coder
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

    /// Count coders by group
    pub async fn count_by_group(&self, coder_group: &str) -> Result<i64> {
        let count: (i64,) = query_as(
            r#"
            SELECT COUNT(*) FROM core.coder
            WHERE coder_group = $1
            AND is_active = true
            "#,
        )
        .bind(coder_group)
        .fetch_one(self.pool)
        .await
        .map_err(Error::Database)?;

        Ok(count.0)
    }

    /// Get all coders (no pagination, use with caution)
    pub async fn get_all(&self) -> Result<Vec<Coder>> {
        query_as::<_, Coder>(
            r#"
            SELECT * FROM core.coder
            ORDER BY last_name ASC, first_name ASC
            "#,
        )
        .fetch_all(self.pool)
        .await
        .map_err(Error::Database)
    }

    /// Get all active coders (no pagination)
    pub async fn get_all_active(&self) -> Result<Vec<Coder>> {
        query_as::<_, Coder>(
            r#"
            SELECT * FROM core.coder
            WHERE is_active = true
            ORDER BY last_name ASC, first_name ASC
            "#,
        )
        .fetch_all(self.pool)
        .await
        .map_err(Error::Database)
    }

    /// Search coders by name (partial match)
    pub async fn search_by_name(
        &self,
        name_pattern: &str,
        limit: i64,
    ) -> Result<Vec<Coder>> {
        query_as::<_, Coder>(
            r#"
            SELECT * FROM core.coder
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

    /// Batch lookup coders by codes
    pub async fn get_by_coder_codes(&self, coder_codes: &[String]) -> Result<Vec<Coder>> {
        query_as::<_, Coder>(
            r#"
            SELECT * FROM core.coder
            WHERE coder_code = ANY($1)
            ORDER BY last_name ASC, first_name ASC
            "#,
        )
        .bind(coder_codes)
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
    async fn test_coder_repository() {
        let pool = create_pool_default().await.unwrap();
        let repo = CoderRepository::new(&pool);

        // Test get_all_active
        let coders = repo.get_all_active().await;
        assert!(coders.is_ok());
    }
}
