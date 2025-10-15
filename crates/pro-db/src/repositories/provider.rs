use crate::models::Provider;
use crate::DbPool;
use pro_common::{Error, Result};
use sqlx::{query, query_as};
use uuid::Uuid;

pub struct ProviderRepository<'a> {
    pool: &'a DbPool,
}

impl<'a> ProviderRepository<'a> {
    pub fn new(pool: &'a DbPool) -> Self {
        Self { pool }
    }

    /// Get provider by ID
    pub async fn get_by_id(&self, provider_id: Uuid) -> Result<Provider> {
        query_as::<_, Provider>(
            r#"
            SELECT * FROM core.provider
            WHERE provider_id = $1
            "#,
        )
        .bind(provider_id)
        .fetch_one(self.pool)
        .await
        .map_err(|e| match e {
            sqlx::Error::RowNotFound => Error::NotFound(format!("Provider {} not found", provider_id)),
            _ => Error::Database(e),
        })
    }

    /// Get provider by NPI (National Provider Identifier)
    pub async fn get_by_npi(&self, npi: &str) -> Result<Provider> {
        query_as::<_, Provider>(
            r#"
            SELECT * FROM core.provider
            WHERE npi = $1
            "#,
        )
        .bind(npi)
        .fetch_one(self.pool)
        .await
        .map_err(|e| match e {
            sqlx::Error::RowNotFound => Error::NotFound(format!("Provider with NPI {} not found", npi)),
            _ => Error::Database(e),
        })
    }

    /// Get active provider by NPI
    pub async fn get_active_by_npi(&self, npi: &str) -> Result<Provider> {
        query_as::<_, Provider>(
            r#"
            SELECT * FROM core.provider
            WHERE npi = $1
            AND is_active = true
            "#,
        )
        .bind(npi)
        .fetch_one(self.pool)
        .await
        .map_err(|e| match e {
            sqlx::Error::RowNotFound => Error::NotFound(format!("Active provider with NPI {} not found", npi)),
            _ => Error::Database(e),
        })
    }

    /// List providers by organization
    pub async fn list_by_organization(
        &self,
        organization_id: Uuid,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<Provider>> {
        query_as::<_, Provider>(
            r#"
            SELECT * FROM core.provider
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

    /// List active providers by organization
    pub async fn list_active_by_organization(
        &self,
        organization_id: Uuid,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<Provider>> {
        query_as::<_, Provider>(
            r#"
            SELECT * FROM core.provider
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

    /// List providers by type
    pub async fn list_by_provider_type(
        &self,
        provider_type: &str,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<Provider>> {
        query_as::<_, Provider>(
            r#"
            SELECT * FROM core.provider
            WHERE provider_type = $1
            AND is_active = true
            ORDER BY last_name ASC, first_name ASC
            LIMIT $2 OFFSET $3
            "#,
        )
        .bind(provider_type)
        .bind(limit)
        .bind(offset)
        .fetch_all(self.pool)
        .await
        .map_err(Error::Database)
    }

    /// List providers by specialty
    pub async fn list_by_specialty(
        &self,
        specialty: &str,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<Provider>> {
        query_as::<_, Provider>(
            r#"
            SELECT * FROM core.provider
            WHERE specialty = $1
            AND is_active = true
            ORDER BY last_name ASC, first_name ASC
            LIMIT $2 OFFSET $3
            "#,
        )
        .bind(specialty)
        .bind(limit)
        .bind(offset)
        .fetch_all(self.pool)
        .await
        .map_err(Error::Database)
    }

    /// List providers by taxonomy code
    pub async fn list_by_taxonomy(
        &self,
        taxonomy_code: &str,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<Provider>> {
        query_as::<_, Provider>(
            r#"
            SELECT * FROM core.provider
            WHERE taxonomy_code = $1
            AND is_active = true
            ORDER BY last_name ASC, first_name ASC
            LIMIT $2 OFFSET $3
            "#,
        )
        .bind(taxonomy_code)
        .bind(limit)
        .bind(offset)
        .fetch_all(self.pool)
        .await
        .map_err(Error::Database)
    }

    /// List providers by license state
    pub async fn list_by_license_state(
        &self,
        license_state: &str,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<Provider>> {
        query_as::<_, Provider>(
            r#"
            SELECT * FROM core.provider
            WHERE license_state = $1
            AND is_active = true
            ORDER BY last_name ASC, first_name ASC
            LIMIT $2 OFFSET $3
            "#,
        )
        .bind(license_state)
        .bind(limit)
        .bind(offset)
        .fetch_all(self.pool)
        .await
        .map_err(Error::Database)
    }

    /// Create a new provider
    pub async fn create(&self, provider: &Provider) -> Result<Uuid> {
        let id = query_as::<_, (Uuid,)>(
            r#"
            INSERT INTO core.provider (
                provider_id,
                npi,
                provider_type,
                last_name,
                first_name,
                middle_name,
                name_suffix,
                taxonomy_code,
                license_number,
                license_state,
                specialty,
                provider_group,
                organization_id,
                address_line1,
                address_line2,
                city,
                state_code,
                postal_code,
                country_code,
                phone,
                email,
                is_active
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22)
            RETURNING provider_id
            "#,
        )
        .bind(provider.provider_id)
        .bind(&provider.npi)
        .bind(&provider.provider_type)
        .bind(&provider.last_name)
        .bind(&provider.first_name)
        .bind(&provider.middle_name)
        .bind(&provider.name_suffix)
        .bind(&provider.taxonomy_code)
        .bind(&provider.license_number)
        .bind(&provider.license_state)
        .bind(&provider.specialty)
        .bind(&provider.provider_group)
        .bind(provider.organization_id)
        .bind(&provider.address_line1)
        .bind(&provider.address_line2)
        .bind(&provider.city)
        .bind(&provider.state_code)
        .bind(&provider.postal_code)
        .bind(&provider.country_code)
        .bind(&provider.phone)
        .bind(&provider.email)
        .bind(provider.is_active)
        .fetch_one(self.pool)
        .await
        .map_err(Error::Database)?;

        Ok(id.0)
    }

    /// Update provider information
    pub async fn update(&self, provider: &Provider) -> Result<()> {
        let rows_affected = query(
            r#"
            UPDATE core.provider
            SET
                npi = $2,
                provider_type = $3,
                last_name = $4,
                first_name = $5,
                middle_name = $6,
                name_suffix = $7,
                taxonomy_code = $8,
                license_number = $9,
                license_state = $10,
                specialty = $11,
                provider_group = $12,
                organization_id = $13,
                address_line1 = $14,
                address_line2 = $15,
                city = $16,
                state_code = $17,
                postal_code = $18,
                country_code = $19,
                phone = $20,
                email = $21,
                is_active = $22,
                updated_at = CURRENT_TIMESTAMP
            WHERE provider_id = $1
            "#,
        )
        .bind(provider.provider_id)
        .bind(&provider.npi)
        .bind(&provider.provider_type)
        .bind(&provider.last_name)
        .bind(&provider.first_name)
        .bind(&provider.middle_name)
        .bind(&provider.name_suffix)
        .bind(&provider.taxonomy_code)
        .bind(&provider.license_number)
        .bind(&provider.license_state)
        .bind(&provider.specialty)
        .bind(&provider.provider_group)
        .bind(provider.organization_id)
        .bind(&provider.address_line1)
        .bind(&provider.address_line2)
        .bind(&provider.city)
        .bind(&provider.state_code)
        .bind(&provider.postal_code)
        .bind(&provider.country_code)
        .bind(&provider.phone)
        .bind(&provider.email)
        .bind(provider.is_active)
        .execute(self.pool)
        .await
        .map_err(Error::Database)?
        .rows_affected();

        if rows_affected == 0 {
            return Err(Error::NotFound(format!("Provider {} not found", provider.provider_id)));
        }

        Ok(())
    }

    /// Update provider status (activate/deactivate)
    pub async fn update_status(&self, provider_id: Uuid, is_active: bool) -> Result<()> {
        let rows_affected = query(
            r#"
            UPDATE core.provider
            SET
                is_active = $2,
                updated_at = CURRENT_TIMESTAMP
            WHERE provider_id = $1
            "#,
        )
        .bind(provider_id)
        .bind(is_active)
        .execute(self.pool)
        .await
        .map_err(Error::Database)?
        .rows_affected();

        if rows_affected == 0 {
            return Err(Error::NotFound(format!("Provider {} not found", provider_id)));
        }

        Ok(())
    }

    /// Update provider organization
    pub async fn update_organization(
        &self,
        provider_id: Uuid,
        organization_id: Option<Uuid>,
    ) -> Result<()> {
        let rows_affected = query(
            r#"
            UPDATE core.provider
            SET
                organization_id = $2,
                updated_at = CURRENT_TIMESTAMP
            WHERE provider_id = $1
            "#,
        )
        .bind(provider_id)
        .bind(organization_id)
        .execute(self.pool)
        .await
        .map_err(Error::Database)?
        .rows_affected();

        if rows_affected == 0 {
            return Err(Error::NotFound(format!("Provider {} not found", provider_id)));
        }

        Ok(())
    }

    /// Soft delete provider (deactivate)
    pub async fn soft_delete(&self, provider_id: Uuid) -> Result<()> {
        self.update_status(provider_id, false).await
    }

    /// Check if NPI exists
    pub async fn exists_by_npi(&self, npi: &str) -> Result<bool> {
        let count: (i64,) = query_as(
            r#"
            SELECT COUNT(*) FROM core.provider
            WHERE npi = $1
            "#,
        )
        .bind(npi)
        .fetch_one(self.pool)
        .await
        .map_err(Error::Database)?;

        Ok(count.0 > 0)
    }

    /// Count providers by organization
    pub async fn count_by_organization(&self, organization_id: Uuid) -> Result<i64> {
        let count: (i64,) = query_as(
            r#"
            SELECT COUNT(*) FROM core.provider
            WHERE organization_id = $1
            "#,
        )
        .bind(organization_id)
        .fetch_one(self.pool)
        .await
        .map_err(Error::Database)?;

        Ok(count.0)
    }

    /// Count active providers by organization
    pub async fn count_active_by_organization(&self, organization_id: Uuid) -> Result<i64> {
        let count: (i64,) = query_as(
            r#"
            SELECT COUNT(*) FROM core.provider
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

    /// Count providers by specialty
    pub async fn count_by_specialty(&self, specialty: &str) -> Result<i64> {
        let count: (i64,) = query_as(
            r#"
            SELECT COUNT(*) FROM core.provider
            WHERE specialty = $1
            AND is_active = true
            "#,
        )
        .bind(specialty)
        .fetch_one(self.pool)
        .await
        .map_err(Error::Database)?;

        Ok(count.0)
    }

    /// Get all providers (no pagination, use with caution)
    pub async fn get_all(&self) -> Result<Vec<Provider>> {
        query_as::<_, Provider>(
            r#"
            SELECT * FROM core.provider
            ORDER BY last_name ASC, first_name ASC
            "#,
        )
        .fetch_all(self.pool)
        .await
        .map_err(Error::Database)
    }

    /// Get all active providers (no pagination)
    pub async fn get_all_active(&self) -> Result<Vec<Provider>> {
        query_as::<_, Provider>(
            r#"
            SELECT * FROM core.provider
            WHERE is_active = true
            ORDER BY last_name ASC, first_name ASC
            "#,
        )
        .fetch_all(self.pool)
        .await
        .map_err(Error::Database)
    }

    /// Search providers by name (partial match)
    pub async fn search_by_name(
        &self,
        name_pattern: &str,
        limit: i64,
    ) -> Result<Vec<Provider>> {
        query_as::<_, Provider>(
            r#"
            SELECT * FROM core.provider
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

    /// Batch lookup providers by NPIs
    pub async fn get_by_npis(&self, npis: &[String]) -> Result<Vec<Provider>> {
        query_as::<_, Provider>(
            r#"
            SELECT * FROM core.provider
            WHERE npi = ANY($1)
            ORDER BY last_name ASC, first_name ASC
            "#,
        )
        .bind(npis)
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
    async fn test_provider_repository() {
        let pool = create_pool_default().await.unwrap();
        let repo = ProviderRepository::new(&pool);

        // Test get_all_active
        let providers = repo.get_all_active().await;
        assert!(providers.is_ok());
    }
}
