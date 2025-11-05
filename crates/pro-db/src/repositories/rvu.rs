use crate::models::{ConversionFactor, RvuReference};
use crate::DbPool;
use chrono::NaiveDate;
use pro_common::{Error, Result};
use sqlx::{query, query_as};


pub struct RvuRepository<'a> {
    pool: &'a DbPool,
}

impl<'a> RvuRepository<'a> {
    pub fn new(pool: &'a DbPool) -> Self {
        Self { pool }
    }

    // ========================================================================
    // RVU REFERENCE QUERIES
    // ========================================================================

    /// Get RVU reference by ID
    pub async fn get_rvu_by_id(&self, rvu_id: i64) -> Result<RvuReference> {
        query_as::<_, RvuReference>(
            r#"
            SELECT * FROM claims.rvu_reference
            WHERE rvu_id = $1
            "#,
        )
        .bind(rvu_id)
        .fetch_one(self.pool)
        .await
        .map_err(|e| match e {
            sqlx::Error::RowNotFound => Error::NotFound(format!("RvuReference {} not found", rvu_id)),
            _ => Error::Database(e),
        })
    }

    /// Get RVU by HCPCS code and year
    pub async fn get_rvu_by_code_and_year(
        &self,
        hcpcs_code: &str,
        year: i32,
    ) -> Result<RvuReference> {
        query_as::<_, RvuReference>(
            r#"
            SELECT * FROM claims.rvu_reference
            WHERE hcpcs_code = $1
            AND effective_year = $2
            ORDER BY effective_date DESC
            LIMIT 1
            "#,
        )
        .bind(hcpcs_code)
        .bind(year)
        .fetch_one(self.pool)
        .await
        .map_err(|e| match e {
            sqlx::Error::RowNotFound => Error::NotFound(format!(
                "RVU data for code {} and year {} not found",
                hcpcs_code, year
            )),
            _ => Error::Database(e),
        })
    }

    /// Get RVU by HCPCS code, modifier, and year
    pub async fn get_rvu_by_code_modifier_and_year(
        &self,
        hcpcs_code: &str,
        modifier: Option<&str>,
        year: i32,
    ) -> Result<RvuReference> {
        query_as::<_, RvuReference>(
            r#"
            SELECT * FROM claims.rvu_reference
            WHERE hcpcs_code = $1
            AND (modifier = $2 OR (modifier IS NULL AND $2 IS NULL))
            AND effective_year = $3
            ORDER BY effective_date DESC
            LIMIT 1
            "#,
        )
        .bind(hcpcs_code)
        .bind(modifier)
        .bind(year)
        .fetch_one(self.pool)
        .await
        .map_err(|e| match e {
            sqlx::Error::RowNotFound => Error::NotFound(format!(
                "RVU data for code {} with modifier {:?} and year {} not found",
                hcpcs_code, modifier, year
            )),
            _ => Error::Database(e),
        })
    }

    /// Get RVU by HCPCS code and date
    pub async fn get_rvu_by_code_and_date(
        &self,
        hcpcs_code: &str,
        service_date: NaiveDate,
    ) -> Result<RvuReference> {
        query_as::<_, RvuReference>(
            r#"
            SELECT * FROM claims.rvu_reference
            WHERE hcpcs_code = $1
            AND effective_date <= $2
            AND (termination_date IS NULL OR termination_date >= $2)
            ORDER BY effective_date DESC
            LIMIT 1
            "#,
        )
        .bind(hcpcs_code)
        .bind(service_date)
        .fetch_one(self.pool)
        .await
        .map_err(|e| match e {
            sqlx::Error::RowNotFound => Error::NotFound(format!(
                "RVU data for code {} on date {} not found",
                hcpcs_code, service_date
            )),
            _ => Error::Database(e),
        })
    }

    /// List RVU references by year
    pub async fn list_by_year(
        &self,
        year: i32,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<RvuReference>> {
        query_as::<_, RvuReference>(
            r#"
            SELECT * FROM claims.rvu_reference
            WHERE effective_year = $1
            ORDER BY hcpcs_code ASC
            LIMIT $2 OFFSET $3
            "#,
        )
        .bind(year)
        .bind(limit)
        .bind(offset)
        .fetch_all(self.pool)
        .await
        .map_err(Error::Database)
    }

    /// List RVU references by HCPCS code (all years)
    pub async fn list_by_hcpcs_code(&self, hcpcs_code: &str) -> Result<Vec<RvuReference>> {
        query_as::<_, RvuReference>(
            r#"
            SELECT * FROM claims.rvu_reference
            WHERE hcpcs_code = $1
            ORDER BY effective_year DESC, effective_date DESC
            "#,
        )
        .bind(hcpcs_code)
        .fetch_all(self.pool)
        .await
        .map_err(Error::Database)
    }

    /// Create RVU reference
    pub async fn create_rvu(&self, rvu: &RvuReference) -> Result<i64> {
        let id = query_as::<_, (i64,)>(
            r#"
            INSERT INTO claims.rvu_reference (
                rvu_id,
                hcpcs_code,
                modifier,
                effective_year,
                effective_date,
                termination_date,
                work_rvu,
                pe_rvu_nonfacility,
                pe_rvu_facility,
                mp_rvu,
                total_rvu_nonfacility,
                total_rvu_facility,
                status_code,
                global_surgery_indicator,
                short_description,
                long_description
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
            RETURNING rvu_id
            "#,
        )
        .bind(rvu.rvu_id)
        .bind(&rvu.hcpcs_code)
        .bind(&rvu.modifier)
        .bind(rvu.effective_year)
        .bind(rvu.effective_date)
        .bind(rvu.termination_date)
        .bind(rvu.work_rvu)
        .bind(rvu.pe_rvu_nonfacility)
        .bind(rvu.pe_rvu_facility)
        .bind(rvu.mp_rvu)
        .bind(rvu.total_rvu_nonfacility)
        .bind(rvu.total_rvu_facility)
        .bind(&rvu.status_code)
        .bind(&rvu.global_surgery_indicator)
        .bind(&rvu.short_description)
        .bind(&rvu.long_description)
        .fetch_one(self.pool)
        .await
        .map_err(Error::Database)?;

        Ok(id.0)
    }

    /// Batch insert RVU references
    pub async fn create_rvu_batch(&self, rvus: &[RvuReference]) -> Result<Vec<i64>> {
        let mut ids = Vec::new();

        for rvu in rvus {
            let id = self.create_rvu(rvu).await?;
            ids.push(id);
        }

        Ok(ids)
    }

    /// Update RVU reference
    pub async fn update_rvu(&self, rvu: &RvuReference) -> Result<()> {
        let rows_affected = query(
            r#"
            UPDATE claims.rvu_reference
            SET
                hcpcs_code = $2,
                modifier = $3,
                effective_year = $4,
                effective_date = $5,
                termination_date = $6,
                work_rvu = $7,
                pe_rvu_nonfacility = $8,
                pe_rvu_facility = $9,
                mp_rvu = $10,
                total_rvu_nonfacility = $11,
                total_rvu_facility = $12,
                status_code = $13,
                global_surgery_indicator = $14,
                short_description = $15,
                long_description = $16,
                updated_at = CURRENT_TIMESTAMP
            WHERE rvu_id = $1
            "#,
        )
        .bind(rvu.rvu_id)
        .bind(&rvu.hcpcs_code)
        .bind(&rvu.modifier)
        .bind(rvu.effective_year)
        .bind(rvu.effective_date)
        .bind(rvu.termination_date)
        .bind(rvu.work_rvu)
        .bind(rvu.pe_rvu_nonfacility)
        .bind(rvu.pe_rvu_facility)
        .bind(rvu.mp_rvu)
        .bind(rvu.total_rvu_nonfacility)
        .bind(rvu.total_rvu_facility)
        .bind(&rvu.status_code)
        .bind(&rvu.global_surgery_indicator)
        .bind(&rvu.short_description)
        .bind(&rvu.long_description)
        .execute(self.pool)
        .await
        .map_err(Error::Database)?
        .rows_affected();

        if rows_affected == 0 {
            return Err(Error::NotFound(format!("RvuReference {} not found", rvu.rvu_id)));
        }

        Ok(())
    }

    /// Delete RVU reference
    pub async fn delete_rvu(&self, rvu_id: i64) -> Result<()> {
        let rows_affected = query(
            r#"
            DELETE FROM claims.rvu_reference
            WHERE rvu_id = $1
            "#,
        )
        .bind(rvu_id)
        .execute(self.pool)
        .await
        .map_err(Error::Database)?
        .rows_affected();

        if rows_affected == 0 {
            return Err(Error::NotFound(format!("RvuReference {} not found", rvu_id)));
        }

        Ok(())
    }

    /// Count RVU references by year
    pub async fn count_by_year(&self, year: i32) -> Result<i64> {
        let count: (i64,) = query_as(
            r#"
            SELECT COUNT(*) FROM claims.rvu_reference
            WHERE effective_year = $1
            "#,
        )
        .bind(year)
        .fetch_one(self.pool)
        .await
        .map_err(Error::Database)?;

        Ok(count.0)
    }

    // ========================================================================
    // CONVERSION FACTOR QUERIES
    // ========================================================================

    /// Get conversion factor by ID
    pub async fn get_conversion_factor_by_id(&self, id: i64) -> Result<ConversionFactor> {
        query_as::<_, ConversionFactor>(
            r#"
            SELECT * FROM claims.conversion_factor
            WHERE conversion_factor_id = $1
            "#,
        )
        .bind(id)
        .fetch_one(self.pool)
        .await
        .map_err(|e| match e {
            sqlx::Error::RowNotFound => Error::NotFound(format!("ConversionFactor {} not found", id)),
            _ => Error::Database(e),
        })
    }

    /// Get conversion factor by year
    pub async fn get_conversion_factor_by_year(&self, year: i32) -> Result<ConversionFactor> {
        query_as::<_, ConversionFactor>(
            r#"
            SELECT * FROM claims.conversion_factor
            WHERE factor_year = $1
            ORDER BY effective_date DESC
            LIMIT 1
            "#,
        )
        .bind(year)
        .fetch_one(self.pool)
        .await
        .map_err(|e| match e {
            sqlx::Error::RowNotFound => Error::NotFound(format!("ConversionFactor for year {} not found", year)),
            _ => Error::Database(e),
        })
    }

    /// Get conversion factor by date
    pub async fn get_conversion_factor_by_date(&self, date: NaiveDate) -> Result<ConversionFactor> {
        query_as::<_, ConversionFactor>(
            r#"
            SELECT * FROM claims.conversion_factor
            WHERE effective_date <= $1
            AND (termination_date IS NULL OR termination_date >= $1)
            ORDER BY effective_date DESC
            LIMIT 1
            "#,
        )
        .bind(date)
        .fetch_one(self.pool)
        .await
        .map_err(|e| match e {
            sqlx::Error::RowNotFound => Error::NotFound(format!("ConversionFactor for date {} not found", date)),
            _ => Error::Database(e),
        })
    }

    /// List all conversion factors
    pub async fn list_conversion_factors(&self) -> Result<Vec<ConversionFactor>> {
        query_as::<_, ConversionFactor>(
            r#"
            SELECT * FROM claims.conversion_factor
            ORDER BY factor_year DESC, effective_date DESC
            "#,
        )
        .fetch_all(self.pool)
        .await
        .map_err(Error::Database)
    }

    /// Create conversion factor
    pub async fn create_conversion_factor(&self, cf: &ConversionFactor) -> Result<i64> {
        let id = query_as::<_, (i64,)>(
            r#"
            INSERT INTO claims.conversion_factor (
                conversion_factor_id,
                factor_year,
                effective_date,
                termination_date,
                conversion_factor,
                budget_neutrality_adjustment,
                created_by
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            RETURNING conversion_factor_id
            "#,
        )
        .bind(cf.conversion_factor_id)
        .bind(cf.factor_year)
        .bind(cf.effective_date)
        .bind(cf.termination_date)
        .bind(cf.conversion_factor)
        .bind(cf.budget_neutrality_adjustment)
        .bind(&cf.created_by)
        .fetch_one(self.pool)
        .await
        .map_err(Error::Database)?;

        Ok(id.0)
    }

    /// Update conversion factor
    pub async fn update_conversion_factor(&self, cf: &ConversionFactor) -> Result<()> {
        let rows_affected = query(
            r#"
            UPDATE claims.conversion_factor
            SET
                factor_year = $2,
                effective_date = $3,
                termination_date = $4,
                conversion_factor = $5,
                budget_neutrality_adjustment = $6,
                created_by = $7,
                updated_at = CURRENT_TIMESTAMP
            WHERE conversion_factor_id = $1
            "#,
        )
        .bind(cf.conversion_factor_id)
        .bind(cf.factor_year)
        .bind(cf.effective_date)
        .bind(cf.termination_date)
        .bind(cf.conversion_factor)
        .bind(cf.budget_neutrality_adjustment)
        .bind(&cf.created_by)
        .execute(self.pool)
        .await
        .map_err(Error::Database)?
        .rows_affected();

        if rows_affected == 0 {
            return Err(Error::NotFound(format!(
                "ConversionFactor {} not found",
                cf.conversion_factor_id
            )));
        }

        Ok(())
    }

    /// Delete conversion factor
    pub async fn delete_conversion_factor(&self, id: i64) -> Result<()> {
        let rows_affected = query(
            r#"
            DELETE FROM claims.conversion_factor
            WHERE conversion_factor_id = $1
            "#,
        )
        .bind(id)
        .execute(self.pool)
        .await
        .map_err(Error::Database)?
        .rows_affected();

        if rows_affected == 0 {
            return Err(Error::NotFound(format!("ConversionFactor {} not found", id)));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connection::create_pool_default;

    #[tokio::test]
    #[ignore] // Requires database
    async fn test_rvu_repository() {
        let pool = create_pool_default().await.unwrap();
        let repo = RvuRepository::new(&pool);

        // Test list_conversion_factors
        let factors = repo.list_conversion_factors().await;
        assert!(factors.is_ok());
    }
}
