use crate::models::{Encounter, EncounterDiagnosis};
use crate::DbPool;
use chrono::NaiveDate;
use pro_common::{Error, Result};
use sqlx::{query, query_as, Row};


pub struct EncounterRepository<'a> {
    pool: &'a DbPool,
}

impl<'a> EncounterRepository<'a> {
    pub fn new(pool: &'a DbPool) -> Self {
        Self { pool }
    }

    /// Get encounter by ID
    pub async fn get_by_id(&self, id: i64) -> Result<Encounter> {
        query_as::<_, Encounter>(
            r#"
            SELECT * FROM claims.encounter
            WHERE encounter_id = $1
            "#,
        )
        .bind(id)
        .fetch_one(self.pool)
        .await
        .map_err(|e| match e {
            sqlx::Error::RowNotFound => Error::NotFound(format!("Encounter {} not found", id)),
            _ => Error::Database(e),
        })
    }

    /// Get encounter by patient control number
    pub async fn get_by_patient_control_number(&self, pcn: &str) -> Result<Encounter> {
        query_as::<_, Encounter>(
            r#"
            SELECT * FROM claims.encounter
            WHERE patient_control_number = $1
            "#,
        )
        .bind(pcn)
        .fetch_one(self.pool)
        .await
        .map_err(|e| match e {
            sqlx::Error::RowNotFound => Error::NotFound(format!("Encounter with PCN {} not found", pcn)),
            _ => Error::Database(e),
        })
    }

    /// List encounters by organization
    pub async fn list_by_organization(&self, organization_id: i64, limit: i64, offset: i64) -> Result<Vec<Encounter>> {
        query_as::<_, Encounter>(
            r#"
            SELECT * FROM claims.encounter
            WHERE organization_id = $1
            AND is_active = true
            AND soft_deleted = false
            ORDER BY created_at DESC
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

    /// List encounters by facility
    pub async fn list_by_facility(&self, facility_id: i64, limit: i64, offset: i64) -> Result<Vec<Encounter>> {
        query_as::<_, Encounter>(
            r#"
            SELECT * FROM claims.encounter
            WHERE facility_id = $1
            AND is_active = true
            AND soft_deleted = false
            ORDER BY created_at DESC
            LIMIT $2 OFFSET $3
            "#,
        )
        .bind(facility_id)
        .bind(limit)
        .bind(offset)
        .fetch_all(self.pool)
        .await
        .map_err(Error::Database)
    }

    /// List encounters by date range
    pub async fn list_by_date_range(
        &self,
        organization_id: i64,
        from_date: NaiveDate,
        to_date: NaiveDate,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<Encounter>> {
        query_as::<_, Encounter>(
            r#"
            SELECT * FROM claims.encounter
            WHERE organization_id = $1
            AND date_of_service_from >= $2
            AND date_of_service_from <= $3
            AND is_active = true
            AND soft_deleted = false
            ORDER BY date_of_service_from DESC, created_at DESC
            LIMIT $4 OFFSET $5
            "#,
        )
        .bind(organization_id)
        .bind(from_date)
        .bind(to_date)
        .bind(limit)
        .bind(offset)
        .fetch_all(self.pool)
        .await
        .map_err(Error::Database)
    }

    /// Create a new encounter
    pub async fn create(&self, encounter: &Encounter) -> Result<i64> {
        let id = query_as::<_, (i64,)>(
            r#"
            INSERT INTO claims.encounter (
                facility_id, organization_id, region_id,
                submitter_id, submitter_name,
                patient_control_number, transaction_set_control_number,
                subscriber_id, subscriber_last_name, subscriber_first_name,
                subscriber_middle_name, subscriber_name_suffix,
                subscriber_gender, subscriber_birth_date,
                subscriber_address_line1, subscriber_address_line2,
                subscriber_city, subscriber_state, subscriber_postal_code, subscriber_country,
                payer_responsibility_code, payer_id, payer_name, claim_filing_indicator,
                billing_provider_id, billing_provider_npi, billing_provider_tax_id, billing_provider_name,
                total_claim_charge_amount, place_of_service_code, claim_frequency_code,
                date_of_service_from, date_of_service_to,
                referring_provider_id, referring_provider_npi,
                rendering_provider_id, rendering_provider_npi,
                supervising_provider_id, supervising_provider_npi,
                service_facility_id, service_facility_npi,
                coder_id, coding_date,
                claim_status, case_status, financial_class,
                import_batch_id, import_date,
                is_active, soft_deleted, created_by, updated_by
            )
            VALUES (
                $1, $2, $3,
                $4, $5,
                $6, $7,
                $8, $9, $10,
                $11, $12,
                $13, $14,
                $15, $16,
                $17, $18, $19, $20,
                $21, $22, $23, $24,
                $25, $26, $27, $28,
                $29, $30, $31,
                $32, $33,
                $34, $35,
                $36, $37,
                $38, $39,
                $40, $41,
                $42, $43,
                $44, $45, $46,
                $47, $48,
                $49, $50, $51, $52
            )
            RETURNING encounter_id
            "#,
        )
        .bind(encounter.facility_id)
        .bind(encounter.organization_id)
        .bind(encounter.region_id)
        .bind(&encounter.submitter_id)
        .bind(&encounter.submitter_name)
        .bind(&encounter.patient_control_number)
        .bind(&encounter.transaction_set_control_number)
        .bind(&encounter.subscriber_id)
        .bind(&encounter.subscriber_last_name)
        .bind(&encounter.subscriber_first_name)
        .bind(&encounter.subscriber_middle_name)
        .bind(&encounter.subscriber_name_suffix)
        .bind(&encounter.subscriber_gender)
        .bind(encounter.subscriber_birth_date)
        .bind(&encounter.subscriber_address_line1)
        .bind(&encounter.subscriber_address_line2)
        .bind(&encounter.subscriber_city)
        .bind(&encounter.subscriber_state)
        .bind(&encounter.subscriber_postal_code)
        .bind(&encounter.subscriber_country)
        .bind(&encounter.payer_responsibility_code)
        .bind(&encounter.payer_id)
        .bind(&encounter.payer_name)
        .bind(&encounter.claim_filing_indicator)
        .bind(encounter.billing_provider_id)
        .bind(&encounter.billing_provider_npi)
        .bind(&encounter.billing_provider_tax_id)
        .bind(&encounter.billing_provider_name)
        .bind(encounter.total_claim_charge_amount)
        .bind(&encounter.place_of_service_code)
        .bind(&encounter.claim_frequency_code)
        .bind(encounter.date_of_service_from)
        .bind(encounter.date_of_service_to)
        .bind(encounter.referring_provider_id)
        .bind(&encounter.referring_provider_npi)
        .bind(encounter.rendering_provider_id)
        .bind(&encounter.rendering_provider_npi)
        .bind(encounter.supervising_provider_id)
        .bind(&encounter.supervising_provider_npi)
        .bind(encounter.service_facility_id)
        .bind(&encounter.service_facility_npi)
        .bind(encounter.coder_id)
        .bind(encounter.coding_date)
        .bind(&encounter.claim_status)
        .bind(&encounter.case_status)
        .bind(&encounter.financial_class)
        .bind(encounter.import_batch_id)
        .bind(encounter.import_date)
        .bind(encounter.is_active)
        .bind(encounter.soft_deleted)
        .bind(&encounter.created_by)
        .bind(&encounter.updated_by)
        .fetch_one(self.pool)
        .await
        .map_err(Error::Database)?;

        Ok(id.0)
    }

    /// Update an existing encounter
    pub async fn update(&self, encounter: &Encounter) -> Result<()> {
        let rows_affected = query(
            r#"
            UPDATE claims.encounter
            SET
                claim_status = $2,
                case_status = $3,
                coder_id = $4,
                coding_date = $5,
                updated_by = $6
            WHERE encounter_id = $1
            "#,
        )
        .bind(encounter.encounter_id)
        .bind(&encounter.claim_status)
        .bind(&encounter.case_status)
        .bind(encounter.coder_id)
        .bind(encounter.coding_date)
        .bind(&encounter.updated_by)
        .execute(self.pool)
        .await
        .map_err(Error::Database)?
        .rows_affected();

        if rows_affected == 0 {
            return Err(Error::NotFound(format!(
                "Encounter {} not found",
                encounter.encounter_id
            )));
        }

        Ok(())
    }

    /// Soft delete an encounter
    pub async fn soft_delete(&self, id: i64) -> Result<()> {
        let rows_affected = query(
            r#"
            UPDATE claims.encounter
            SET soft_deleted = true, updated_by = 'SYSTEM'
            WHERE encounter_id = $1
            "#,
        )
        .bind(id)
        .execute(self.pool)
        .await
        .map_err(Error::Database)?
        .rows_affected();

        if rows_affected == 0 {
            return Err(Error::NotFound(format!("Encounter {} not found", id)));
        }

        Ok(())
    }

    /// Get diagnoses for an encounter
    pub async fn get_diagnoses(&self, encounter_id: i64) -> Result<Vec<EncounterDiagnosis>> {
        query_as::<_, EncounterDiagnosis>(
            r#"
            SELECT * FROM claims.encounter_diagnosis
            WHERE encounter_id = $1
            ORDER BY sequence_number ASC
            "#,
        )
        .bind(encounter_id)
        .fetch_all(self.pool)
        .await
        .map_err(Error::Database)
    }

    /// Create a diagnosis for an encounter
    pub async fn create_diagnosis(&self, diagnosis: &EncounterDiagnosis) -> Result<i64> {
        let id = query_as::<_, (i64,)>(
            r#"
            INSERT INTO claims.encounter_diagnosis (
                encounter_id, sequence_number,
                diagnosis_code_qualifier, diagnosis_code, diagnosis_description,
                is_principal, is_admitting, is_external_cause, is_patient_reason,
                present_on_admission_indicator,
                hcc_indicator, hcc_category
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            RETURNING diagnosis_id
            "#,
        )
        .bind(diagnosis.encounter_id)
        .bind(diagnosis.sequence_number)
        .bind(&diagnosis.diagnosis_code_qualifier)
        .bind(&diagnosis.diagnosis_code)
        .bind(&diagnosis.diagnosis_description)
        .bind(diagnosis.is_principal)
        .bind(diagnosis.is_admitting)
        .bind(diagnosis.is_external_cause)
        .bind(diagnosis.is_patient_reason)
        .bind(&diagnosis.present_on_admission_indicator)
        .bind(diagnosis.hcc_indicator)
        .bind(&diagnosis.hcc_category)
        .fetch_one(self.pool)
        .await
        .map_err(Error::Database)?;

        Ok(id.0)
    }

    /// Create multiple diagnoses in a single batch operation (PERFORMANCE OPTIMIZATION)
    pub async fn create_diagnoses_batch(&self, diagnoses: &[EncounterDiagnosis]) -> Result<Vec<i64>> {
        if diagnoses.is_empty() {
            return Ok(Vec::new());
        }

        // Build multi-row INSERT statement
        let mut query_str = String::from(
            r#"
            INSERT INTO claims.encounter_diagnosis (
                encounter_id, sequence_number,
                diagnosis_code_qualifier, diagnosis_code, diagnosis_description,
                is_principal, is_admitting, is_external_cause, is_patient_reason,
                present_on_admission_indicator,
                hcc_indicator, hcc_category
            )
            VALUES
            "#,
        );

        // Add placeholders for each diagnosis (12 fields per diagnosis)
        for (idx, _) in diagnoses.iter().enumerate() {
            if idx > 0 {
                query_str.push_str(", ");
            }
            let base = idx * 12;
            query_str.push_str(&format!(
                "(${}, ${}, ${}, ${}, ${}, ${}, ${}, ${}, ${}, ${}, ${}, ${})",
                base + 1, base + 2, base + 3, base + 4, base + 5, base + 6,
                base + 7, base + 8, base + 9, base + 10, base + 11, base + 12
            ));
        }

        query_str.push_str(" RETURNING diagnosis_id");

        // Build and execute query
        let mut query = sqlx::query(&query_str);
        for diagnosis in diagnoses {
            query = query
                .bind(diagnosis.encounter_id)
                .bind(diagnosis.sequence_number)
                .bind(&diagnosis.diagnosis_code_qualifier)
                .bind(&diagnosis.diagnosis_code)
                .bind(&diagnosis.diagnosis_description)
                .bind(diagnosis.is_principal)
                .bind(diagnosis.is_admitting)
                .bind(diagnosis.is_external_cause)
                .bind(diagnosis.is_patient_reason)
                .bind(&diagnosis.present_on_admission_indicator)
                .bind(diagnosis.hcc_indicator)
                .bind(&diagnosis.hcc_category);
        }

        let rows = query
            .fetch_all(self.pool)
            .await
            .map_err(Error::Database)?;

        // Extract IDs from result rows
        let ids: Result<Vec<i64>> = rows
            .iter()
            .map(|row| row.try_get::<i64, _>(0).map_err(Error::Database))
            .collect();

        ids
    }

    /// Create encounter within existing transaction (PHASE 2 OPTIMIZATION)
    pub async fn create_with_tx(
        &self,
        encounter: &Encounter,
        tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    ) -> Result<i64> {
        let id = query_as::<_, (i64,)>(
            r#"
            INSERT INTO claims.encounter (
                facility_id, organization_id, region_id,
                submitter_id, submitter_name,
                patient_control_number, transaction_set_control_number,
                subscriber_id, subscriber_last_name, subscriber_first_name,
                subscriber_middle_name, subscriber_name_suffix,
                subscriber_gender, subscriber_birth_date,
                subscriber_address_line1, subscriber_address_line2,
                subscriber_city, subscriber_state, subscriber_postal_code, subscriber_country,
                payer_responsibility_code, payer_id, payer_name, claim_filing_indicator,
                billing_provider_id, billing_provider_npi, billing_provider_tax_id, billing_provider_name,
                total_claim_charge_amount, place_of_service_code, claim_frequency_code,
                date_of_service_from, date_of_service_to,
                referring_provider_id, referring_provider_npi,
                rendering_provider_id, rendering_provider_npi,
                supervising_provider_id, supervising_provider_npi,
                service_facility_id, service_facility_npi,
                coder_id, coding_date,
                claim_status, case_status, financial_class,
                import_batch_id, import_date,
                is_active, soft_deleted, created_by, updated_by
            )
            VALUES (
                $1, $2, $3,
                $4, $5,
                $6, $7,
                $8, $9, $10,
                $11, $12,
                $13, $14,
                $15, $16,
                $17, $18, $19, $20,
                $21, $22, $23, $24,
                $25, $26, $27, $28,
                $29, $30, $31,
                $32, $33,
                $34, $35,
                $36, $37,
                $38, $39,
                $40, $41,
                $42, $43,
                $44, $45, $46,
                $47, $48,
                $49, $50, $51, $52
            )
            RETURNING encounter_id
            "#,
        )
        .bind(encounter.facility_id)
        .bind(encounter.organization_id)
        .bind(encounter.region_id)
        .bind(&encounter.submitter_id)
        .bind(&encounter.submitter_name)
        .bind(&encounter.patient_control_number)
        .bind(&encounter.transaction_set_control_number)
        .bind(&encounter.subscriber_id)
        .bind(&encounter.subscriber_last_name)
        .bind(&encounter.subscriber_first_name)
        .bind(&encounter.subscriber_middle_name)
        .bind(&encounter.subscriber_name_suffix)
        .bind(&encounter.subscriber_gender)
        .bind(encounter.subscriber_birth_date)
        .bind(&encounter.subscriber_address_line1)
        .bind(&encounter.subscriber_address_line2)
        .bind(&encounter.subscriber_city)
        .bind(&encounter.subscriber_state)
        .bind(&encounter.subscriber_postal_code)
        .bind(&encounter.subscriber_country)
        .bind(&encounter.payer_responsibility_code)
        .bind(&encounter.payer_id)
        .bind(&encounter.payer_name)
        .bind(&encounter.claim_filing_indicator)
        .bind(encounter.billing_provider_id)
        .bind(&encounter.billing_provider_npi)
        .bind(&encounter.billing_provider_tax_id)
        .bind(&encounter.billing_provider_name)
        .bind(encounter.total_claim_charge_amount)
        .bind(&encounter.place_of_service_code)
        .bind(&encounter.claim_frequency_code)
        .bind(encounter.date_of_service_from)
        .bind(encounter.date_of_service_to)
        .bind(encounter.referring_provider_id)
        .bind(&encounter.referring_provider_npi)
        .bind(encounter.rendering_provider_id)
        .bind(&encounter.rendering_provider_npi)
        .bind(encounter.supervising_provider_id)
        .bind(&encounter.supervising_provider_npi)
        .bind(encounter.service_facility_id)
        .bind(&encounter.service_facility_npi)
        .bind(encounter.coder_id)
        .bind(encounter.coding_date)
        .bind(&encounter.claim_status)
        .bind(&encounter.case_status)
        .bind(&encounter.financial_class)
        .bind(encounter.import_batch_id)
        .bind(encounter.import_date)
        .bind(encounter.is_active)
        .bind(encounter.soft_deleted)
        .bind(&encounter.created_by)
        .bind(&encounter.updated_by)
        .fetch_one(&mut **tx)
        .await
        .map_err(Error::Database)?;

        Ok(id.0)
    }

    /// Create diagnoses batch within transaction (PHASE 2 OPTIMIZATION)
    pub async fn create_diagnoses_batch_with_tx(
        &self,
        diagnoses: &[EncounterDiagnosis],
        tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    ) -> Result<Vec<i64>> {
        if diagnoses.is_empty() {
            return Ok(Vec::new());
        }

        // Build multi-row INSERT statement
        let mut query_str = String::from(
            r#"
            INSERT INTO claims.encounter_diagnosis (
                encounter_id, sequence_number,
                diagnosis_code_qualifier, diagnosis_code, diagnosis_description,
                is_principal, is_admitting, is_external_cause, is_patient_reason,
                present_on_admission_indicator,
                hcc_indicator, hcc_category
            )
            VALUES
            "#,
        );

        // Add placeholders for each diagnosis (12 fields per diagnosis)
        for (idx, _) in diagnoses.iter().enumerate() {
            if idx > 0 {
                query_str.push_str(", ");
            }
            let base = idx * 12;
            query_str.push_str(&format!(
                "(${}, ${}, ${}, ${}, ${}, ${}, ${}, ${}, ${}, ${}, ${}, ${})",
                base + 1, base + 2, base + 3, base + 4, base + 5, base + 6,
                base + 7, base + 8, base + 9, base + 10, base + 11, base + 12
            ));
        }

        query_str.push_str(" RETURNING diagnosis_id");

        // Build and execute query
        let mut query = sqlx::query(&query_str);
        for diagnosis in diagnoses {
            query = query
                .bind(diagnosis.encounter_id)
                .bind(diagnosis.sequence_number)
                .bind(&diagnosis.diagnosis_code_qualifier)
                .bind(&diagnosis.diagnosis_code)
                .bind(&diagnosis.diagnosis_description)
                .bind(diagnosis.is_principal)
                .bind(diagnosis.is_admitting)
                .bind(diagnosis.is_external_cause)
                .bind(diagnosis.is_patient_reason)
                .bind(&diagnosis.present_on_admission_indicator)
                .bind(diagnosis.hcc_indicator)
                .bind(&diagnosis.hcc_category);
        }

        let rows = query
            .fetch_all(&mut **tx)
            .await
            .map_err(Error::Database)?;

        // Extract IDs from result rows
        let ids: Result<Vec<i64>> = rows
            .iter()
            .map(|row| row.try_get::<i64, _>(0).map_err(Error::Database))
            .collect();

        ids
    }

    /// Check if patient control number exists
    pub async fn exists_by_pcn(&self, pcn: &str) -> Result<bool> {
        let count: (i64,) = query_as(
            r#"
            SELECT COUNT(*) FROM claims.encounter
            WHERE patient_control_number = $1
            "#,
        )
        .bind(pcn)
        .fetch_one(self.pool)
        .await
        .map_err(Error::Database)?;

        Ok(count.0 > 0)
    }

    /// Count encounters by organization
    pub async fn count_by_organization(&self, organization_id: i64) -> Result<i64> {
        let count: (i64,) = query_as(
            r#"
            SELECT COUNT(*) FROM claims.encounter
            WHERE organization_id = $1
            AND is_active = true
            AND soft_deleted = false
            "#,
        )
        .bind(organization_id)
        .fetch_one(self.pool)
        .await
        .map_err(Error::Database)?;

        Ok(count.0)
    }

    // PHASE 6: Repository optimization methods removed temporarily
    // These require unified Flag model which will be added in future migration
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connection::create_pool_default;

    #[tokio::test]
    #[ignore] // Requires database
    async fn test_encounter_crud() {
        let pool = create_pool_default().await.unwrap();
        let repo = EncounterRepository::new(&pool);

        // Test count_by_organization with a sample organization ID
        let sample_org_id = 1i64;
        let count = repo.count_by_organization(sample_org_id).await;
        assert!(count.is_ok());
    }
}
