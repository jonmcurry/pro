use crate::models::DenialEvent;
use crate::DbPool;
use chrono::NaiveDate;
use pro_common::{Error, Result};
use rust_decimal::Decimal;
use sqlx::{query, query_as};
use uuid::Uuid;

pub struct DenialRepository<'a> {
    pool: &'a DbPool,
}

impl<'a> DenialRepository<'a> {
    pub fn new(pool: &'a DbPool) -> Self {
        Self { pool }
    }

    /// Get denial event by ID
    pub async fn get_by_id(&self, denial_id: Uuid) -> Result<DenialEvent> {
        query_as::<_, DenialEvent>(
            r#"
            SELECT * FROM claims.denial_event
            WHERE denial_id = $1
            "#,
        )
        .bind(denial_id)
        .fetch_one(self.pool)
        .await
        .map_err(|e| match e {
            sqlx::Error::RowNotFound => Error::NotFound(format!("DenialEvent {} not found", denial_id)),
            _ => Error::Database(e),
        })
    }

    /// Get denials by encounter
    pub async fn get_by_encounter(&self, encounter_id: Uuid) -> Result<Vec<DenialEvent>> {
        query_as::<_, DenialEvent>(
            r#"
            SELECT * FROM claims.denial_event
            WHERE encounter_id = $1
            ORDER BY denial_date DESC
            "#,
        )
        .bind(encounter_id)
        .fetch_all(self.pool)
        .await
        .map_err(Error::Database)
    }

    /// Get denials by service line
    pub async fn get_by_service_line(&self, service_line_id: Uuid) -> Result<Vec<DenialEvent>> {
        query_as::<_, DenialEvent>(
            r#"
            SELECT * FROM claims.denial_event
            WHERE service_line_id = $1
            ORDER BY denial_date DESC
            "#,
        )
        .bind(service_line_id)
        .fetch_all(self.pool)
        .await
        .map_err(Error::Database)
    }

    /// List denials by organization
    pub async fn list_by_organization(
        &self,
        organization_id: Uuid,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<DenialEvent>> {
        query_as::<_, DenialEvent>(
            r#"
            SELECT * FROM claims.denial_event
            WHERE organization_id = $1
            ORDER BY denial_date DESC
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

    /// List denials by facility
    pub async fn list_by_facility(
        &self,
        facility_id: Uuid,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<DenialEvent>> {
        query_as::<_, DenialEvent>(
            r#"
            SELECT * FROM claims.denial_event
            WHERE facility_id = $1
            ORDER BY denial_date DESC
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

    /// List denials by date range
    pub async fn list_by_date_range(
        &self,
        organization_id: Uuid,
        from_date: NaiveDate,
        to_date: NaiveDate,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<DenialEvent>> {
        query_as::<_, DenialEvent>(
            r#"
            SELECT * FROM claims.denial_event
            WHERE organization_id = $1
            AND denial_date >= $2
            AND denial_date <= $3
            ORDER BY denial_date DESC
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

    /// List denials by denial type
    pub async fn list_by_denial_type(
        &self,
        organization_id: Uuid,
        denial_type: &str,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<DenialEvent>> {
        query_as::<_, DenialEvent>(
            r#"
            SELECT * FROM claims.denial_event
            WHERE organization_id = $1
            AND denial_type = $2
            ORDER BY denial_date DESC
            LIMIT $3 OFFSET $4
            "#,
        )
        .bind(organization_id)
        .bind(denial_type)
        .bind(limit)
        .bind(offset)
        .fetch_all(self.pool)
        .await
        .map_err(Error::Database)
    }

    /// List denials by category
    pub async fn list_by_category(
        &self,
        organization_id: Uuid,
        denial_category: &str,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<DenialEvent>> {
        query_as::<_, DenialEvent>(
            r#"
            SELECT * FROM claims.denial_event
            WHERE organization_id = $1
            AND denial_category = $2
            ORDER BY denial_date DESC
            LIMIT $3 OFFSET $4
            "#,
        )
        .bind(organization_id)
        .bind(denial_category)
        .bind(limit)
        .bind(offset)
        .fetch_all(self.pool)
        .await
        .map_err(Error::Database)
    }

    /// List denials by payer
    pub async fn list_by_payer(
        &self,
        organization_id: Uuid,
        payer_name: &str,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<DenialEvent>> {
        query_as::<_, DenialEvent>(
            r#"
            SELECT * FROM claims.denial_event
            WHERE organization_id = $1
            AND payer_name = $2
            ORDER BY denial_date DESC
            LIMIT $3 OFFSET $4
            "#,
        )
        .bind(organization_id)
        .bind(payer_name)
        .bind(limit)
        .bind(offset)
        .fetch_all(self.pool)
        .await
        .map_err(Error::Database)
    }

    /// List denials by reason code
    pub async fn list_by_reason_code(
        &self,
        organization_id: Uuid,
        reason_code: &str,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<DenialEvent>> {
        query_as::<_, DenialEvent>(
            r#"
            SELECT * FROM claims.denial_event
            WHERE organization_id = $1
            AND claim_adjustment_reason_code = $2
            ORDER BY denial_date DESC
            LIMIT $3 OFFSET $4
            "#,
        )
        .bind(organization_id)
        .bind(reason_code)
        .bind(limit)
        .bind(offset)
        .fetch_all(self.pool)
        .await
        .map_err(Error::Database)
    }

    /// List denials by status
    pub async fn list_by_status(
        &self,
        organization_id: Uuid,
        denial_status: &str,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<DenialEvent>> {
        query_as::<_, DenialEvent>(
            r#"
            SELECT * FROM claims.denial_event
            WHERE organization_id = $1
            AND denial_status = $2
            ORDER BY denial_date DESC
            LIMIT $3 OFFSET $4
            "#,
        )
        .bind(organization_id)
        .bind(denial_status)
        .bind(limit)
        .bind(offset)
        .fetch_all(self.pool)
        .await
        .map_err(Error::Database)
    }

    /// List preventable denials
    pub async fn list_preventable(
        &self,
        organization_id: Uuid,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<DenialEvent>> {
        query_as::<_, DenialEvent>(
            r#"
            SELECT * FROM claims.denial_event
            WHERE organization_id = $1
            AND is_preventable = true
            ORDER BY denial_date DESC
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

    /// List denials pending appeal
    pub async fn list_pending_appeal(
        &self,
        organization_id: Uuid,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<DenialEvent>> {
        query_as::<_, DenialEvent>(
            r#"
            SELECT * FROM claims.denial_event
            WHERE organization_id = $1
            AND appeal_filed = false
            AND appeal_deadline >= CURRENT_DATE
            AND denial_status = 'OPEN'
            ORDER BY appeal_deadline ASC
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

    /// List denials by coder
    pub async fn list_by_coder(
        &self,
        coder_id: Uuid,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<DenialEvent>> {
        query_as::<_, DenialEvent>(
            r#"
            SELECT * FROM claims.denial_event
            WHERE coder_id = $1
            ORDER BY denial_date DESC
            LIMIT $2 OFFSET $3
            "#,
        )
        .bind(coder_id)
        .bind(limit)
        .bind(offset)
        .fetch_all(self.pool)
        .await
        .map_err(Error::Database)
    }

    /// List denials by provider
    pub async fn list_by_provider(
        &self,
        provider_id: Uuid,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<DenialEvent>> {
        query_as::<_, DenialEvent>(
            r#"
            SELECT * FROM claims.denial_event
            WHERE provider_id = $1
            ORDER BY denial_date DESC
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

    /// Create denial event
    pub async fn create(&self, denial: &DenialEvent) -> Result<Uuid> {
        let id = query_as::<_, (Uuid,)>(
            r#"
            INSERT INTO claims.denial_event (
                denial_id, encounter_id, service_line_id, organization_id, facility_id,
                denial_type, denial_category, payer_id, payer_name, claim_filing_indicator,
                claim_adjustment_group_code, claim_adjustment_reason_code, remittance_advice_remark_code,
                denial_reason_description, payer_denial_reason, denied_amount, billed_amount,
                allowed_amount, paid_amount, service_date, initial_submission_date, denial_date,
                received_date, root_cause_category, root_cause_subcategory, root_cause_details,
                responsible_party, coder_id, provider_id, is_preventable, preventable_category,
                prevention_recommendations, denial_status, resolution_status, resolution_date,
                appeal_filed, appeal_level, appeal_deadline, internal_notes, resolution_notes,
                created_by, updated_by
            )
            VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                $11, $12, $13, $14, $15, $16, $17, $18, $19, $20,
                $21, $22, $23, $24, $25, $26, $27, $28, $29, $30,
                $31, $32, $33, $34, $35, $36, $37, $38, $39, $40,
                $41, $42
            )
            RETURNING denial_id
            "#,
        )
        .bind(denial.denial_id)
        .bind(denial.encounter_id)
        .bind(denial.service_line_id)
        .bind(denial.organization_id)
        .bind(denial.facility_id)
        .bind(&denial.denial_type)
        .bind(&denial.denial_category)
        .bind(&denial.payer_id)
        .bind(&denial.payer_name)
        .bind(&denial.claim_filing_indicator)
        .bind(&denial.claim_adjustment_group_code)
        .bind(&denial.claim_adjustment_reason_code)
        .bind(&denial.remittance_advice_remark_code)
        .bind(&denial.denial_reason_description)
        .bind(&denial.payer_denial_reason)
        .bind(denial.denied_amount)
        .bind(denial.billed_amount)
        .bind(denial.allowed_amount)
        .bind(denial.paid_amount)
        .bind(denial.service_date)
        .bind(denial.initial_submission_date)
        .bind(denial.denial_date)
        .bind(denial.received_date)
        .bind(&denial.root_cause_category)
        .bind(&denial.root_cause_subcategory)
        .bind(&denial.root_cause_details)
        .bind(&denial.responsible_party)
        .bind(denial.coder_id)
        .bind(denial.provider_id)
        .bind(denial.is_preventable)
        .bind(&denial.preventable_category)
        .bind(&denial.prevention_recommendations)
        .bind(&denial.denial_status)
        .bind(&denial.resolution_status)
        .bind(denial.resolution_date)
        .bind(denial.appeal_filed)
        .bind(&denial.appeal_level)
        .bind(denial.appeal_deadline)
        .bind(&denial.internal_notes)
        .bind(&denial.resolution_notes)
        .bind(&denial.created_by)
        .bind(&denial.updated_by)
        .fetch_one(self.pool)
        .await
        .map_err(Error::Database)?;

        Ok(id.0)
    }

    /// Update denial event
    pub async fn update(&self, denial: &DenialEvent) -> Result<()> {
        let rows_affected = query(
            r#"
            UPDATE claims.denial_event
            SET
                root_cause_category = $2,
                root_cause_subcategory = $3,
                root_cause_details = $4,
                responsible_party = $5,
                coder_id = $6,
                provider_id = $7,
                is_preventable = $8,
                preventable_category = $9,
                prevention_recommendations = $10,
                denial_status = $11,
                resolution_status = $12,
                resolution_date = $13,
                appeal_filed = $14,
                appeal_level = $15,
                appeal_deadline = $16,
                internal_notes = $17,
                resolution_notes = $18,
                updated_by = $19,
                updated_at = CURRENT_TIMESTAMP
            WHERE denial_id = $1
            "#,
        )
        .bind(denial.denial_id)
        .bind(&denial.root_cause_category)
        .bind(&denial.root_cause_subcategory)
        .bind(&denial.root_cause_details)
        .bind(&denial.responsible_party)
        .bind(denial.coder_id)
        .bind(denial.provider_id)
        .bind(denial.is_preventable)
        .bind(&denial.preventable_category)
        .bind(&denial.prevention_recommendations)
        .bind(&denial.denial_status)
        .bind(&denial.resolution_status)
        .bind(denial.resolution_date)
        .bind(denial.appeal_filed)
        .bind(&denial.appeal_level)
        .bind(denial.appeal_deadline)
        .bind(&denial.internal_notes)
        .bind(&denial.resolution_notes)
        .bind(&denial.updated_by)
        .execute(self.pool)
        .await
        .map_err(Error::Database)?
        .rows_affected();

        if rows_affected == 0 {
            return Err(Error::NotFound(format!("DenialEvent {} not found", denial.denial_id)));
        }

        Ok(())
    }

    /// Update denial status
    pub async fn update_status(
        &self,
        denial_id: Uuid,
        denial_status: &str,
        resolution_status: Option<&str>,
        resolution_notes: Option<&str>,
        updated_by: Option<&str>,
    ) -> Result<()> {
        let rows_affected = query(
            r#"
            UPDATE claims.denial_event
            SET
                denial_status = $2,
                resolution_status = $3,
                resolution_date = CASE WHEN $2 = 'RESOLVED' THEN CURRENT_DATE ELSE resolution_date END,
                resolution_notes = COALESCE($4, resolution_notes),
                updated_by = $5,
                updated_at = CURRENT_TIMESTAMP
            WHERE denial_id = $1
            "#,
        )
        .bind(denial_id)
        .bind(denial_status)
        .bind(resolution_status)
        .bind(resolution_notes)
        .bind(updated_by)
        .execute(self.pool)
        .await
        .map_err(Error::Database)?
        .rows_affected();

        if rows_affected == 0 {
            return Err(Error::NotFound(format!("DenialEvent {} not found", denial_id)));
        }

        Ok(())
    }

    /// Update appeal information
    pub async fn update_appeal(
        &self,
        denial_id: Uuid,
        appeal_filed: bool,
        appeal_level: Option<&str>,
        appeal_deadline: Option<NaiveDate>,
        updated_by: Option<&str>,
    ) -> Result<()> {
        let rows_affected = query(
            r#"
            UPDATE claims.denial_event
            SET
                appeal_filed = $2,
                appeal_level = $3,
                appeal_deadline = $4,
                updated_by = $5,
                updated_at = CURRENT_TIMESTAMP
            WHERE denial_id = $1
            "#,
        )
        .bind(denial_id)
        .bind(appeal_filed)
        .bind(appeal_level)
        .bind(appeal_deadline)
        .bind(updated_by)
        .execute(self.pool)
        .await
        .map_err(Error::Database)?
        .rows_affected();

        if rows_affected == 0 {
            return Err(Error::NotFound(format!("DenialEvent {} not found", denial_id)));
        }

        Ok(())
    }

    /// Count denials by organization
    pub async fn count_by_organization(&self, organization_id: Uuid) -> Result<i64> {
        let count: (i64,) = query_as(
            r#"
            SELECT COUNT(*) FROM claims.denial_event
            WHERE organization_id = $1
            "#,
        )
        .bind(organization_id)
        .fetch_one(self.pool)
        .await
        .map_err(Error::Database)?;

        Ok(count.0)
    }

    /// Count denials by status
    pub async fn count_by_status(
        &self,
        organization_id: Uuid,
        denial_status: &str,
    ) -> Result<i64> {
        let count: (i64,) = query_as(
            r#"
            SELECT COUNT(*) FROM claims.denial_event
            WHERE organization_id = $1
            AND denial_status = $2
            "#,
        )
        .bind(organization_id)
        .bind(denial_status)
        .fetch_one(self.pool)
        .await
        .map_err(Error::Database)?;

        Ok(count.0)
    }

    /// Sum denied amount by organization
    pub async fn sum_denied_amount_by_organization(&self, organization_id: Uuid) -> Result<Decimal> {
        let sum: (Option<Decimal>,) = query_as(
            r#"
            SELECT SUM(denied_amount) FROM claims.denial_event
            WHERE organization_id = $1
            "#,
        )
        .bind(organization_id)
        .fetch_one(self.pool)
        .await
        .map_err(Error::Database)?;

        Ok(sum.0.unwrap_or_else(|| Decimal::ZERO))
    }

    /// Get denial rate by organization (percentage)
    pub async fn get_denial_rate_by_organization(
        &self,
        organization_id: Uuid,
    ) -> Result<Decimal> {
        let rate: (Option<Decimal>,) = query_as(
            r#"
            SELECT
                CASE
                    WHEN COUNT(DISTINCT e.encounter_id) > 0
                    THEN (COUNT(DISTINCT d.encounter_id)::decimal / COUNT(DISTINCT e.encounter_id)::decimal * 100)
                    ELSE 0
                END as denial_rate
            FROM claims.encounter e
            LEFT JOIN claims.denial_event d ON e.encounter_id = d.encounter_id
            WHERE e.organization_id = $1
            "#,
        )
        .bind(organization_id)
        .fetch_one(self.pool)
        .await
        .map_err(Error::Database)?;

        Ok(rate.0.unwrap_or_else(|| Decimal::ZERO))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connection::create_pool_default;

    #[tokio::test]
    #[ignore] // Requires database
    async fn test_denial_repository() {
        let pool = create_pool_default().await.unwrap();
        let repo = DenialRepository::new(&pool);

        // Test count_by_organization
        let sample_org_id = Uuid::new_v4();
        let count = repo.count_by_organization(sample_org_id).await;
        assert!(count.is_ok());
    }
}
