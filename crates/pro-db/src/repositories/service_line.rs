use crate::models::ServiceLine;
use crate::DbPool;
use chrono::NaiveDate;
use pro_common::{Error, Result};
use sqlx::{query, query_as};


pub struct ServiceLineRepository<'a> {
    pool: &'a DbPool,
}

impl<'a> ServiceLineRepository<'a> {
    pub fn new(pool: &'a DbPool) -> Self {
        Self { pool }
    }

    /// Get service line by ID
    pub async fn get_by_id(&self, id: i64) -> Result<ServiceLine> {
        query_as::<_, ServiceLine>(
            r#"
            SELECT * FROM claims.service_line
            WHERE service_line_id = $1
            "#,
        )
        .bind(id)
        .fetch_one(self.pool)
        .await
        .map_err(|e| match e {
            sqlx::Error::RowNotFound => Error::NotFound(format!("ServiceLine {} not found", id)),
            _ => Error::Database(e),
        })
    }

    /// Get all service lines for an encounter
    pub async fn get_by_encounter(&self, encounter_id: i64) -> Result<Vec<ServiceLine>> {
        query_as::<_, ServiceLine>(
            r#"
            SELECT * FROM claims.service_line
            WHERE encounter_id = $1
            ORDER BY line_number ASC
            "#,
        )
        .bind(encounter_id)
        .fetch_all(self.pool)
        .await
        .map_err(Error::Database)
    }

    /// Get service lines by procedure code
    pub async fn get_by_procedure_code(
        &self,
        procedure_code: &str,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<ServiceLine>> {
        query_as::<_, ServiceLine>(
            r#"
            SELECT * FROM claims.service_line
            WHERE procedure_code = $1
            ORDER BY service_date_from DESC
            LIMIT $2 OFFSET $3
            "#,
        )
        .bind(procedure_code)
        .bind(limit)
        .bind(offset)
        .fetch_all(self.pool)
        .await
        .map_err(Error::Database)
    }

    /// Get service lines by date range
    pub async fn get_by_date_range(
        &self,
        from_date: NaiveDate,
        to_date: NaiveDate,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<ServiceLine>> {
        query_as::<_, ServiceLine>(
            r#"
            SELECT * FROM claims.service_line
            WHERE service_date_from >= $1
            AND service_date_from <= $2
            ORDER BY service_date_from DESC
            LIMIT $3 OFFSET $4
            "#,
        )
        .bind(from_date)
        .bind(to_date)
        .bind(limit)
        .bind(offset)
        .fetch_all(self.pool)
        .await
        .map_err(Error::Database)
    }

    /// Create a new service line
    pub async fn create(&self, service_line: &ServiceLine) -> Result<i64> {
        let id = query_as::<_, (i64,)>(
            r#"
            INSERT INTO claims.service_line (
                encounter_id, line_number,
                product_service_id_qualifier, procedure_code,
                procedure_modifier_1, procedure_modifier_2, procedure_modifier_3, procedure_modifier_4,
                procedure_description,
                line_item_charge_amount, unit_basis_measurement_code, service_unit_count,
                place_of_service_code,
                service_date_from, service_date_to,
                rendering_provider_id, rendering_provider_npi,
                supervising_provider_id, supervising_provider_npi,
                ordering_provider_id, ordering_provider_npi,
                referring_provider_id, referring_provider_npi,
                service_facility_id, service_facility_npi,
                prior_authorization_number, referral_number,
                line_note, revenue_code,
                ndc_code, ndc_unit_count, ndc_measurement_unit,
                diagnosis_code_pointer_1, diagnosis_code_pointer_2,
                diagnosis_code_pointer_3, diagnosis_code_pointer_4,
                line_status, created_by, updated_by
            )
            VALUES (
                $1, $2,
                $3, $4,
                $5, $6, $7, $8,
                $9,
                $10, $11, $12,
                $13,
                $14, $15,
                $16, $17,
                $18, $19,
                $20, $21,
                $22, $23,
                $24, $25,
                $26, $27,
                $28, $29,
                $30, $31, $32,
                $33, $34,
                $35, $36,
                $37, $38, $39
            )
            RETURNING service_line_id
            "#,
        )
        .bind(service_line.encounter_id)
        .bind(service_line.line_number)
        .bind(&service_line.product_service_id_qualifier)
        .bind(&service_line.procedure_code)
        .bind(&service_line.procedure_modifier_1)
        .bind(&service_line.procedure_modifier_2)
        .bind(&service_line.procedure_modifier_3)
        .bind(&service_line.procedure_modifier_4)
        .bind(&service_line.procedure_description)
        .bind(service_line.line_item_charge_amount)
        .bind(&service_line.unit_basis_measurement_code)
        .bind(service_line.service_unit_count)
        .bind(&service_line.place_of_service_code)
        .bind(service_line.service_date_from)
        .bind(service_line.service_date_to)
        .bind(service_line.rendering_provider_id)
        .bind(&service_line.rendering_provider_npi)
        .bind(service_line.supervising_provider_id)
        .bind(&service_line.supervising_provider_npi)
        .bind(service_line.ordering_provider_id)
        .bind(&service_line.ordering_provider_npi)
        .bind(service_line.referring_provider_id)
        .bind(&service_line.referring_provider_npi)
        .bind(service_line.service_facility_id)
        .bind(&service_line.service_facility_npi)
        .bind(&service_line.prior_authorization_number)
        .bind(&service_line.referral_number)
        .bind(&service_line.line_note)
        .bind(&service_line.revenue_code)
        .bind(&service_line.ndc_code)
        .bind(service_line.ndc_unit_count)
        .bind(&service_line.ndc_measurement_unit)
        .bind(service_line.diagnosis_code_pointer_1)
        .bind(service_line.diagnosis_code_pointer_2)
        .bind(service_line.diagnosis_code_pointer_3)
        .bind(service_line.diagnosis_code_pointer_4)
        .bind(&service_line.line_status)
        .bind(&service_line.created_by)
        .bind(&service_line.updated_by)
        .fetch_one(self.pool)
        .await
        .map_err(Error::Database)?;

        Ok(id.0)
    }

    /// Batch insert service lines for an encounter
    pub async fn create_batch(&self, service_lines: &[ServiceLine]) -> Result<Vec<i64>> {
        let mut ids = Vec::new();

        for service_line in service_lines {
            let id = self.create(service_line).await?;
            ids.push(id);
        }

        Ok(ids)
    }

    /// Create service line within existing transaction (PHASE 2 OPTIMIZATION)
    pub async fn create_with_tx(
        &self,
        service_line: &ServiceLine,
        tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    ) -> Result<i64> {
        let id = query_as::<_, (i64,)>(
            r#"
            INSERT INTO claims.service_line (
                encounter_id, line_number,
                product_service_id_qualifier, procedure_code,
                procedure_modifier_1, procedure_modifier_2, procedure_modifier_3, procedure_modifier_4,
                procedure_description,
                line_item_charge_amount, unit_basis_measurement_code, service_unit_count,
                place_of_service_code,
                service_date_from, service_date_to,
                rendering_provider_id, rendering_provider_npi,
                supervising_provider_id, supervising_provider_npi,
                ordering_provider_id, ordering_provider_npi,
                referring_provider_id, referring_provider_npi,
                service_facility_id, service_facility_npi,
                prior_authorization_number, referral_number,
                line_note, revenue_code,
                ndc_code, ndc_unit_count, ndc_measurement_unit,
                diagnosis_code_pointer_1, diagnosis_code_pointer_2,
                diagnosis_code_pointer_3, diagnosis_code_pointer_4,
                line_status, created_by, updated_by
            )
            VALUES (
                $1, $2,
                $3, $4,
                $5, $6, $7, $8,
                $9,
                $10, $11, $12,
                $13,
                $14, $15,
                $16, $17,
                $18, $19,
                $20, $21,
                $22, $23,
                $24, $25,
                $26, $27,
                $28, $29,
                $30, $31, $32,
                $33, $34,
                $35, $36,
                $37, $38, $39
            )
            RETURNING service_line_id
            "#,
        )
        .bind(service_line.encounter_id)
        .bind(service_line.line_number)
        .bind(&service_line.product_service_id_qualifier)
        .bind(&service_line.procedure_code)
        .bind(&service_line.procedure_modifier_1)
        .bind(&service_line.procedure_modifier_2)
        .bind(&service_line.procedure_modifier_3)
        .bind(&service_line.procedure_modifier_4)
        .bind(&service_line.procedure_description)
        .bind(service_line.line_item_charge_amount)
        .bind(&service_line.unit_basis_measurement_code)
        .bind(service_line.service_unit_count)
        .bind(&service_line.place_of_service_code)
        .bind(service_line.service_date_from)
        .bind(service_line.service_date_to)
        .bind(service_line.rendering_provider_id)
        .bind(&service_line.rendering_provider_npi)
        .bind(service_line.supervising_provider_id)
        .bind(&service_line.supervising_provider_npi)
        .bind(service_line.ordering_provider_id)
        .bind(&service_line.ordering_provider_npi)
        .bind(service_line.referring_provider_id)
        .bind(&service_line.referring_provider_npi)
        .bind(service_line.service_facility_id)
        .bind(&service_line.service_facility_npi)
        .bind(&service_line.prior_authorization_number)
        .bind(&service_line.referral_number)
        .bind(&service_line.line_note)
        .bind(&service_line.revenue_code)
        .bind(&service_line.ndc_code)
        .bind(service_line.ndc_unit_count)
        .bind(&service_line.ndc_measurement_unit)
        .bind(service_line.diagnosis_code_pointer_1)
        .bind(service_line.diagnosis_code_pointer_2)
        .bind(service_line.diagnosis_code_pointer_3)
        .bind(service_line.diagnosis_code_pointer_4)
        .bind(&service_line.line_status)
        .bind(&service_line.created_by)
        .bind(&service_line.updated_by)
        .fetch_one(&mut **tx)
        .await
        .map_err(Error::Database)?;

        Ok(id.0)
    }

    /// Batch insert service lines within transaction (PHASE 2 OPTIMIZATION)
    pub async fn create_batch_with_tx(
        &self,
        service_lines: &[ServiceLine],
        tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    ) -> Result<Vec<i64>> {
        let mut ids = Vec::new();

        for service_line in service_lines {
            let id = self.create_with_tx(service_line, tx).await?;
            ids.push(id);
        }

        Ok(ids)
    }

    /// Update a service line
    pub async fn update(&self, service_line: &ServiceLine) -> Result<()> {
        let rows_affected = query(
            r#"
            UPDATE claims.service_line
            SET
                procedure_code = $2,
                procedure_modifier_1 = $3,
                procedure_modifier_2 = $4,
                procedure_modifier_3 = $5,
                procedure_modifier_4 = $6,
                line_item_charge_amount = $7,
                service_unit_count = $8,
                line_status = $9,
                updated_by = $10
            WHERE service_line_id = $1
            "#,
        )
        .bind(service_line.service_line_id)
        .bind(&service_line.procedure_code)
        .bind(&service_line.procedure_modifier_1)
        .bind(&service_line.procedure_modifier_2)
        .bind(&service_line.procedure_modifier_3)
        .bind(&service_line.procedure_modifier_4)
        .bind(service_line.line_item_charge_amount)
        .bind(service_line.service_unit_count)
        .bind(&service_line.line_status)
        .bind(&service_line.updated_by)
        .execute(self.pool)
        .await
        .map_err(Error::Database)?
        .rows_affected();

        if rows_affected == 0 {
            return Err(Error::NotFound(format!(
                "ServiceLine {} not found",
                service_line.service_line_id
            )));
        }

        Ok(())
    }

    /// Delete a service line
    pub async fn delete(&self, id: i64) -> Result<()> {
        let rows_affected = query(
            r#"
            DELETE FROM claims.service_line
            WHERE service_line_id = $1
            "#,
        )
        .bind(id)
        .execute(self.pool)
        .await
        .map_err(Error::Database)?
        .rows_affected();

        if rows_affected == 0 {
            return Err(Error::NotFound(format!("ServiceLine {} not found", id)));
        }

        Ok(())
    }

    /// Count service lines for an encounter
    pub async fn count_by_encounter(&self, encounter_id: i64) -> Result<i64> {
        let count: (i64,) = query_as(
            r#"
            SELECT COUNT(*) FROM claims.service_line
            WHERE encounter_id = $1
            "#,
        )
        .bind(encounter_id)
        .fetch_one(self.pool)
        .await
        .map_err(Error::Database)?;

        Ok(count.0)
    }

    /// Get service lines with specific modifiers
    pub async fn get_by_modifier(
        &self,
        modifier: &str,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<ServiceLine>> {
        query_as::<_, ServiceLine>(
            r#"
            SELECT * FROM claims.service_line
            WHERE procedure_modifier_1 = $1
            OR procedure_modifier_2 = $1
            OR procedure_modifier_3 = $1
            OR procedure_modifier_4 = $1
            ORDER BY service_date_from DESC
            LIMIT $2 OFFSET $3
            "#,
        )
        .bind(modifier)
        .bind(limit)
        .bind(offset)
        .fetch_all(self.pool)
        .await
        .map_err(Error::Database)
    }

    /// Get service lines by provider
    pub async fn get_by_rendering_provider(
        &self,
        provider_id: i64,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<ServiceLine>> {
        query_as::<_, ServiceLine>(
            r#"
            SELECT * FROM claims.service_line
            WHERE rendering_provider_id = $1
            ORDER BY service_date_from DESC
            LIMIT $2 OFFSET $3
            "#,
        )
        .bind(provider_id)
        .bind(limit)
        .bind(offset)
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
    async fn test_service_line_crud() {
        let pool = create_pool_default().await.unwrap();
        let repo = ServiceLineRepository::new(&pool);

        // Test count_by_encounter with a sample encounter ID
        let sample_encounter_id = 1i64;
        let count = repo.count_by_encounter(sample_encounter_id).await;
        assert!(count.is_ok());
    }
}
