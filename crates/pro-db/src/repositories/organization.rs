use crate::models::Organization;
use crate::DbPool;
use pro_common::{Error, Result};
use sqlx::query_as;


pub struct OrganizationRepository<'a> {
    pool: &'a DbPool,
}

impl<'a> OrganizationRepository<'a> {
    pub fn new(pool: &'a DbPool) -> Self {
        Self { pool }
    }

    /// Get organization by ID
    pub async fn get_by_id(&self, id: i64) -> Result<Organization> {
        query_as::<_, Organization>(
            r#"
            SELECT * FROM claims.organization
            WHERE organization_id = $1
            "#,
        )
        .bind(id)
        .fetch_one(self.pool)
        .await
        .map_err(|e| match e {
            sqlx::Error::RowNotFound => Error::NotFound(format!("Organization {} not found", id)),
            _ => Error::Database(e),
        })
    }

    /// Get organization by code
    pub async fn get_by_code(&self, code: &str) -> Result<Organization> {
        query_as::<_, Organization>(
            r#"
            SELECT * FROM claims.organization
            WHERE organization_code = $1
            "#,
        )
        .bind(code)
        .fetch_one(self.pool)
        .await
        .map_err(|e| match e {
            sqlx::Error::RowNotFound => Error::NotFound(format!("Organization {} not found", code)),
            _ => Error::Database(e),
        })
    }

    /// List all active organizations
    pub async fn list_active(&self) -> Result<Vec<Organization>> {
        query_as::<_, Organization>(
            r#"
            SELECT * FROM claims.organization
            WHERE is_active = true
            ORDER BY organization_name
            "#,
        )
        .fetch_all(self.pool)
        .await
        .map_err(Error::Database)
    }

    /// Create a new organization
    pub async fn create(&self, org: &Organization) -> Result<i64> {
        let id = query_as::<_, (i64,)>(
            r#"
            INSERT INTO claims.organization (
                organization_code, organization_name, tax_id, npi,
                address_line1, address_line2, city, state_code, postal_code,
                country_code, phone, email, is_active, created_by
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
            RETURNING organization_id
            "#,
        )
        .bind(&org.organization_code)
        .bind(&org.organization_name)
        .bind(&org.tax_id)
        .bind(&org.npi)
        .bind(&org.address_line1)
        .bind(&org.address_line2)
        .bind(&org.city)
        .bind(&org.state_code)
        .bind(&org.postal_code)
        .bind(&org.country_code)
        .bind(&org.phone)
        .bind(&org.email)
        .bind(org.is_active)
        .bind("SYSTEM")
        .fetch_one(self.pool)
        .await
        .map_err(Error::Database)?;

        Ok(id.0)
    }

    /// Update an existing organization
    pub async fn update(&self, org: &Organization) -> Result<()> {
        let rows_affected = sqlx::query(
            r#"
            UPDATE claims.organization
            SET organization_name = $2,
                tax_id = $3,
                npi = $4,
                address_line1 = $5,
                address_line2 = $6,
                city = $7,
                state_code = $8,
                postal_code = $9,
                country_code = $10,
                phone = $11,
                email = $12,
                is_active = $13,
                updated_by = $14
            WHERE organization_id = $1
            "#,
        )
        .bind(org.organization_id)
        .bind(&org.organization_name)
        .bind(&org.tax_id)
        .bind(&org.npi)
        .bind(&org.address_line1)
        .bind(&org.address_line2)
        .bind(&org.city)
        .bind(&org.state_code)
        .bind(&org.postal_code)
        .bind(&org.country_code)
        .bind(&org.phone)
        .bind(&org.email)
        .bind(org.is_active)
        .bind("SYSTEM")
        .execute(self.pool)
        .await
        .map_err(Error::Database)?
        .rows_affected();

        if rows_affected == 0 {
            return Err(Error::NotFound(format!(
                "Organization {} not found",
                org.organization_id
            )));
        }

        Ok(())
    }

    /// Deactivate an organization (soft delete)
    pub async fn deactivate(&self, id: i64) -> Result<()> {
        let rows_affected = sqlx::query(
            r#"
            UPDATE claims.organization
            SET is_active = false, updated_by = 'SYSTEM'
            WHERE organization_id = $1
            "#,
        )
        .bind(id)
        .execute(self.pool)
        .await
        .map_err(Error::Database)?
        .rows_affected();

        if rows_affected == 0 {
            return Err(Error::NotFound(format!("Organization {} not found", id)));
        }

        Ok(())
    }

    /// Check if organization code exists
    pub async fn exists_by_code(&self, code: &str) -> Result<bool> {
        let count: (i64,) = sqlx::query_as(
            r#"
            SELECT COUNT(*) FROM claims.organization
            WHERE organization_code = $1
            "#,
        )
        .bind(code)
        .fetch_one(self.pool)
        .await
        .map_err(Error::Database)?;

        Ok(count.0 > 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connection::create_pool_default;

    #[tokio::test]
    #[ignore] // Requires database
    async fn test_organization_crud() {
        let pool = create_pool_default().await.unwrap();
        let repo = OrganizationRepository::new(&pool);

        // Test list_active
        let orgs = repo.list_active().await;
        assert!(orgs.is_ok());
    }
}
