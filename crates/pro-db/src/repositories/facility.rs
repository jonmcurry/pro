use crate::models::Facility;
use crate::DbPool;
use pro_common::{Error, Result};
use sqlx::{query, query_as};


pub struct FacilityRepository<'a> {
    pool: &'a DbPool,
}

impl<'a> FacilityRepository<'a> {
    pub fn new(pool: &'a DbPool) -> Self {
        Self { pool }
    }

    /// Get facility by ID
    pub async fn get_by_id(&self, facility_id: i64) -> Result<Facility> {
        query_as::<_, Facility>(
            r#"
            SELECT * FROM core.facility
            WHERE facility_id = $1
            "#,
        )
        .bind(facility_id)
        .fetch_one(self.pool)
        .await
        .map_err(|e| match e {
            sqlx::Error::RowNotFound => Error::NotFound(format!("Facility {} not found", facility_id)),
            _ => Error::Database(e),
        })
    }

    /// Get facility by facility code
    pub async fn get_by_facility_code(&self, facility_code: &str) -> Result<Facility> {
        query_as::<_, Facility>(
            r#"
            SELECT * FROM core.facility
            WHERE facility_code = $1
            AND is_active = true
            "#,
        )
        .bind(facility_code)
        .fetch_one(self.pool)
        .await
        .map_err(|e| match e {
            sqlx::Error::RowNotFound => Error::NotFound(format!("Facility with code {} not found", facility_code)),
            _ => Error::Database(e),
        })
    }

    /// Get facility by NPI
    pub async fn get_by_npi(&self, npi: &str) -> Result<Facility> {
        query_as::<_, Facility>(
            r#"
            SELECT * FROM core.facility
            WHERE npi = $1
            AND is_active = true
            "#,
        )
        .bind(npi)
        .fetch_one(self.pool)
        .await
        .map_err(|e| match e {
            sqlx::Error::RowNotFound => Error::NotFound(format!("Facility with NPI {} not found", npi)),
            _ => Error::Database(e),
        })
    }

    /// List facilities by organization
    pub async fn list_by_organization(
        &self,
        organization_id: i64,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<Facility>> {
        query_as::<_, Facility>(
            r#"
            SELECT * FROM core.facility
            WHERE organization_id = $1
            ORDER BY facility_name ASC
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

    /// List active facilities by organization
    pub async fn list_active_by_organization(
        &self,
        organization_id: i64,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<Facility>> {
        query_as::<_, Facility>(
            r#"
            SELECT * FROM core.facility
            WHERE organization_id = $1
            AND is_active = true
            ORDER BY facility_name ASC
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

    /// List facilities by region
    pub async fn list_by_region(
        &self,
        region_id: i64,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<Facility>> {
        query_as::<_, Facility>(
            r#"
            SELECT * FROM core.facility
            WHERE region_id = $1
            ORDER BY facility_name ASC
            LIMIT $2 OFFSET $3
            "#,
        )
        .bind(region_id)
        .bind(limit)
        .bind(offset)
        .fetch_all(self.pool)
        .await
        .map_err(Error::Database)
    }

    /// List facilities by state
    pub async fn list_by_state(
        &self,
        state_code: &str,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<Facility>> {
        query_as::<_, Facility>(
            r#"
            SELECT * FROM core.facility
            WHERE state_code = $1
            AND is_active = true
            ORDER BY facility_name ASC
            LIMIT $2 OFFSET $3
            "#,
        )
        .bind(state_code)
        .bind(limit)
        .bind(offset)
        .fetch_all(self.pool)
        .await
        .map_err(Error::Database)
    }

    /// List facilities by type
    pub async fn list_by_type(
        &self,
        facility_type: &str,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<Facility>> {
        query_as::<_, Facility>(
            r#"
            SELECT * FROM core.facility
            WHERE facility_type = $1
            AND is_active = true
            ORDER BY facility_name ASC
            LIMIT $2 OFFSET $3
            "#,
        )
        .bind(facility_type)
        .bind(limit)
        .bind(offset)
        .fetch_all(self.pool)
        .await
        .map_err(Error::Database)
    }

    /// Create a new facility
    pub async fn create(&self, facility: &Facility) -> Result<i64> {
        let id = query_as::<_, (i64,)>(
            r#"
            INSERT INTO core.facility (
                facility_id,
                organization_id,
                region_id,
                facility_code,
                facility_name,
                npi,
                tax_id,
                facility_type,
                address_line1,
                address_line2,
                city,
                state_code,
                postal_code,
                country_code,
                phone,
                email,
                ehr_system,
                is_active
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18)
            RETURNING facility_id
            "#,
        )
        .bind(facility.facility_id)
        .bind(facility.organization_id)
        .bind(facility.region_id)
        .bind(&facility.facility_code)
        .bind(&facility.facility_name)
        .bind(&facility.npi)
        .bind(&facility.tax_id)
        .bind(&facility.facility_type)
        .bind(&facility.address_line1)
        .bind(&facility.address_line2)
        .bind(&facility.city)
        .bind(&facility.state_code)
        .bind(&facility.postal_code)
        .bind(&facility.country_code)
        .bind(&facility.phone)
        .bind(&facility.email)
        .bind(&facility.ehr_system)
        .bind(facility.is_active)
        .fetch_one(self.pool)
        .await
        .map_err(Error::Database)?;

        Ok(id.0)
    }

    /// Update facility information
    pub async fn update(&self, facility: &Facility) -> Result<()> {
        let rows_affected = query(
            r#"
            UPDATE core.facility
            SET
                region_id = $2,
                facility_code = $3,
                facility_name = $4,
                npi = $5,
                tax_id = $6,
                facility_type = $7,
                address_line1 = $8,
                address_line2 = $9,
                city = $10,
                state_code = $11,
                postal_code = $12,
                country_code = $13,
                phone = $14,
                email = $15,
                ehr_system = $16,
                is_active = $17,
                updated_at = CURRENT_TIMESTAMP
            WHERE facility_id = $1
            "#,
        )
        .bind(facility.facility_id)
        .bind(facility.region_id)
        .bind(&facility.facility_code)
        .bind(&facility.facility_name)
        .bind(&facility.npi)
        .bind(&facility.tax_id)
        .bind(&facility.facility_type)
        .bind(&facility.address_line1)
        .bind(&facility.address_line2)
        .bind(&facility.city)
        .bind(&facility.state_code)
        .bind(&facility.postal_code)
        .bind(&facility.country_code)
        .bind(&facility.phone)
        .bind(&facility.email)
        .bind(&facility.ehr_system)
        .bind(facility.is_active)
        .execute(self.pool)
        .await
        .map_err(Error::Database)?
        .rows_affected();

        if rows_affected == 0 {
            return Err(Error::NotFound(format!("Facility {} not found", facility.facility_id)));
        }

        Ok(())
    }

    /// Update facility status (activate/deactivate)
    pub async fn update_status(&self, facility_id: i64, is_active: bool) -> Result<()> {
        let rows_affected = query(
            r#"
            UPDATE core.facility
            SET
                is_active = $2,
                updated_at = CURRENT_TIMESTAMP
            WHERE facility_id = $1
            "#,
        )
        .bind(facility_id)
        .bind(is_active)
        .execute(self.pool)
        .await
        .map_err(Error::Database)?
        .rows_affected();

        if rows_affected == 0 {
            return Err(Error::NotFound(format!("Facility {} not found", facility_id)));
        }

        Ok(())
    }

    /// Update facility region
    pub async fn update_region(&self, facility_id: i64, region_id: Option<i64>) -> Result<()> {
        let rows_affected = query(
            r#"
            UPDATE core.facility
            SET
                region_id = $2,
                updated_at = CURRENT_TIMESTAMP
            WHERE facility_id = $1
            "#,
        )
        .bind(facility_id)
        .bind(region_id)
        .execute(self.pool)
        .await
        .map_err(Error::Database)?
        .rows_affected();

        if rows_affected == 0 {
            return Err(Error::NotFound(format!("Facility {} not found", facility_id)));
        }

        Ok(())
    }

    /// Soft delete facility (deactivate)
    pub async fn soft_delete(&self, facility_id: i64) -> Result<()> {
        self.update_status(facility_id, false).await
    }

    /// Check if facility code exists for organization
    pub async fn exists_by_facility_code(
        &self,
        organization_id: i64,
        facility_code: &str,
    ) -> Result<bool> {
        let count: (i64,) = query_as(
            r#"
            SELECT COUNT(*) FROM core.facility
            WHERE organization_id = $1
            AND facility_code = $2
            "#,
        )
        .bind(organization_id)
        .bind(facility_code)
        .fetch_one(self.pool)
        .await
        .map_err(Error::Database)?;

        Ok(count.0 > 0)
    }

    /// Check if NPI exists
    pub async fn exists_by_npi(&self, npi: &str) -> Result<bool> {
        let count: (i64,) = query_as(
            r#"
            SELECT COUNT(*) FROM core.facility
            WHERE npi = $1
            "#,
        )
        .bind(npi)
        .fetch_one(self.pool)
        .await
        .map_err(Error::Database)?;

        Ok(count.0 > 0)
    }

    /// Count facilities by organization
    pub async fn count_by_organization(&self, organization_id: i64) -> Result<i64> {
        let count: (i64,) = query_as(
            r#"
            SELECT COUNT(*) FROM core.facility
            WHERE organization_id = $1
            "#,
        )
        .bind(organization_id)
        .fetch_one(self.pool)
        .await
        .map_err(Error::Database)?;

        Ok(count.0)
    }

    /// Count active facilities by organization
    pub async fn count_active_by_organization(&self, organization_id: i64) -> Result<i64> {
        let count: (i64,) = query_as(
            r#"
            SELECT COUNT(*) FROM core.facility
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

    /// Count facilities by region
    pub async fn count_by_region(&self, region_id: i64) -> Result<i64> {
        let count: (i64,) = query_as(
            r#"
            SELECT COUNT(*) FROM core.facility
            WHERE region_id = $1
            "#,
        )
        .bind(region_id)
        .fetch_one(self.pool)
        .await
        .map_err(Error::Database)?;

        Ok(count.0)
    }

    /// Get all facilities for an organization (no pagination)
    pub async fn get_all_by_organization(&self, organization_id: i64) -> Result<Vec<Facility>> {
        query_as::<_, Facility>(
            r#"
            SELECT * FROM core.facility
            WHERE organization_id = $1
            ORDER BY facility_name ASC
            "#,
        )
        .bind(organization_id)
        .fetch_all(self.pool)
        .await
        .map_err(Error::Database)
    }

    /// Search facilities by name (partial match)
    pub async fn search_by_name(
        &self,
        organization_id: i64,
        name_pattern: &str,
        limit: i64,
    ) -> Result<Vec<Facility>> {
        query_as::<_, Facility>(
            r#"
            SELECT * FROM core.facility
            WHERE organization_id = $1
            AND facility_name ILIKE $2
            AND is_active = true
            ORDER BY facility_name ASC
            LIMIT $3
            "#,
        )
        .bind(organization_id)
        .bind(format!("%{}%", name_pattern))
        .bind(limit)
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
    async fn test_facility_repository() {
        let pool = create_pool_default().await.unwrap();
        let repo = FacilityRepository::new(&pool);

        // Test count_by_organization with a sample organization ID
        let sample_org_id = 1i64;
        let count = repo.count_by_organization(sample_org_id).await;
        assert!(count.is_ok());
    }
}
