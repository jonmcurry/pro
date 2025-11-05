use crate::models::ImportBatch;
use crate::DbPool;
use chrono::{DateTime, Utc};
use pro_common::{Error, Result};
use rust_decimal::Decimal;
use sqlx::{query, query_as};


pub struct ImportBatchRepository<'a> {
    pool: &'a DbPool,
}

impl<'a> ImportBatchRepository<'a> {
    pub fn new(pool: &'a DbPool) -> Self {
        Self { pool }
    }

    /// Get import batch by ID
    pub async fn get_by_id(&self, batch_id: i64) -> Result<ImportBatch> {
        query_as::<_, ImportBatch>(
            r#"
            SELECT * FROM staging.import_batch
            WHERE batch_id = $1
            "#,
        )
        .bind(batch_id)
        .fetch_one(self.pool)
        .await
        .map_err(|e| match e {
            sqlx::Error::RowNotFound => Error::NotFound(format!("ImportBatch {} not found", batch_id)),
            _ => Error::Database(e),
        })
    }

    /// List import batches by organization
    pub async fn list_by_organization(
        &self,
        organization_id: i64,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<ImportBatch>> {
        query_as::<_, ImportBatch>(
            r#"
            SELECT * FROM staging.import_batch
            WHERE organization_id = $1
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

    /// List import batches by status
    pub async fn list_by_status(
        &self,
        organization_id: i64,
        import_status: &str,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<ImportBatch>> {
        query_as::<_, ImportBatch>(
            r#"
            SELECT * FROM staging.import_batch
            WHERE organization_id = $1
            AND import_status = $2
            ORDER BY created_at DESC
            LIMIT $3 OFFSET $4
            "#,
        )
        .bind(organization_id)
        .bind(import_status)
        .bind(limit)
        .bind(offset)
        .fetch_all(self.pool)
        .await
        .map_err(Error::Database)
    }

    /// List import batches by facility
    pub async fn list_by_facility(
        &self,
        facility_id: i64,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<ImportBatch>> {
        query_as::<_, ImportBatch>(
            r#"
            SELECT * FROM staging.import_batch
            WHERE facility_id = $1
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

    /// List import batches by date range
    pub async fn list_by_date_range(
        &self,
        organization_id: i64,
        from_date: DateTime<Utc>,
        to_date: DateTime<Utc>,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<ImportBatch>> {
        query_as::<_, ImportBatch>(
            r#"
            SELECT * FROM staging.import_batch
            WHERE organization_id = $1
            AND created_at >= $2
            AND created_at <= $3
            ORDER BY created_at DESC
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

    /// Create a new import batch
    pub async fn create(&self, batch: &ImportBatch) -> Result<i64> {
        let id = query_as::<_, (i64,)>(
            r#"
            INSERT INTO staging.import_batch (
                batch_id,
                organization_id,
                facility_id,
                batch_name,
                batch_type,
                file_format,
                original_filename,
                file_path,
                file_size_bytes,
                file_hash,
                import_status,
                total_records,
                processed_records,
                successful_records,
                failed_records,
                skipped_records,
                duplicate_records,
                started_at,
                completed_at,
                processing_duration_seconds,
                configuration_id,
                rules_applied,
                error_message,
                created_by
            )
            VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                $11, $12, $13, $14, $15, $16, $17, $18, $19, $20,
                $21, $22, $23, $24
            )
            RETURNING batch_id
            "#,
        )
        .bind(batch.batch_id)
        .bind(batch.organization_id)
        .bind(batch.facility_id)
        .bind(&batch.batch_name)
        .bind(&batch.batch_type)
        .bind(&batch.file_format)
        .bind(&batch.original_filename)
        .bind(&batch.file_path)
        .bind(batch.file_size_bytes)
        .bind(&batch.file_hash)
        .bind(&batch.import_status)
        .bind(batch.total_records)
        .bind(batch.processed_records)
        .bind(batch.successful_records)
        .bind(batch.failed_records)
        .bind(batch.skipped_records)
        .bind(batch.duplicate_records)
        .bind(batch.started_at)
        .bind(batch.completed_at)
        .bind(batch.processing_duration_seconds)
        .bind(batch.configuration_id)
        .bind(batch.rules_applied)
        .bind(&batch.error_message)
        .bind(&batch.created_by)
        .fetch_one(self.pool)
        .await
        .map_err(Error::Database)?;

        Ok(id.0)
    }

    /// Update import batch status and statistics
    pub async fn update_status(
        &self,
        batch_id: i64,
        import_status: &str,
        started_at: Option<DateTime<Utc>>,
        completed_at: Option<DateTime<Utc>>,
        processing_duration_seconds: Option<Decimal>,
    ) -> Result<()> {
        let rows_affected = query(
            r#"
            UPDATE staging.import_batch
            SET
                import_status = $2,
                started_at = COALESCE($3, started_at),
                completed_at = $4,
                processing_duration_seconds = $5
            WHERE batch_id = $1
            "#,
        )
        .bind(batch_id)
        .bind(import_status)
        .bind(started_at)
        .bind(completed_at)
        .bind(processing_duration_seconds)
        .execute(self.pool)
        .await
        .map_err(Error::Database)?
        .rows_affected();

        if rows_affected == 0 {
            return Err(Error::NotFound(format!("ImportBatch {} not found", batch_id)));
        }

        Ok(())
    }

    /// Update import batch statistics
    pub async fn update_statistics(
        &self,
        batch_id: i64,
        total_records: i32,
        processed_records: i32,
        successful_records: i32,
        failed_records: i32,
        skipped_records: i32,
        duplicate_records: i32,
    ) -> Result<()> {
        let rows_affected = query(
            r#"
            UPDATE staging.import_batch
            SET
                total_records = $2,
                processed_records = $3,
                successful_records = $4,
                failed_records = $5,
                skipped_records = $6,
                duplicate_records = $7
            WHERE batch_id = $1
            "#,
        )
        .bind(batch_id)
        .bind(total_records)
        .bind(processed_records)
        .bind(successful_records)
        .bind(failed_records)
        .bind(skipped_records)
        .bind(duplicate_records)
        .execute(self.pool)
        .await
        .map_err(Error::Database)?
        .rows_affected();

        if rows_affected == 0 {
            return Err(Error::NotFound(format!("ImportBatch {} not found", batch_id)));
        }

        Ok(())
    }

    /// Update import batch with error message
    pub async fn update_error(
        &self,
        batch_id: i64,
        error_message: &str,
    ) -> Result<()> {
        let rows_affected = query(
            r#"
            UPDATE staging.import_batch
            SET
                import_status = 'FAILED',
                error_message = $2,
                completed_at = CURRENT_TIMESTAMP
            WHERE batch_id = $1
            "#,
        )
        .bind(batch_id)
        .bind(error_message)
        .execute(self.pool)
        .await
        .map_err(Error::Database)?
        .rows_affected();

        if rows_affected == 0 {
            return Err(Error::NotFound(format!("ImportBatch {} not found", batch_id)));
        }

        Ok(())
    }

    /// Complete import batch with final statistics
    pub async fn complete(
        &self,
        batch_id: i64,
        import_status: &str,
        total_records: i32,
        processed_records: i32,
        successful_records: i32,
        failed_records: i32,
        skipped_records: i32,
        duplicate_records: i32,
        processing_duration_seconds: Option<Decimal>,
    ) -> Result<()> {
        let rows_affected = query(
            r#"
            UPDATE staging.import_batch
            SET
                import_status = $2,
                total_records = $3,
                processed_records = $4,
                successful_records = $5,
                failed_records = $6,
                skipped_records = $7,
                duplicate_records = $8,
                processing_duration_seconds = $9,
                completed_at = CURRENT_TIMESTAMP
            WHERE batch_id = $1
            "#,
        )
        .bind(batch_id)
        .bind(import_status)
        .bind(total_records)
        .bind(processed_records)
        .bind(successful_records)
        .bind(failed_records)
        .bind(skipped_records)
        .bind(duplicate_records)
        .bind(processing_duration_seconds)
        .execute(self.pool)
        .await
        .map_err(Error::Database)?
        .rows_affected();

        if rows_affected == 0 {
            return Err(Error::NotFound(format!("ImportBatch {} not found", batch_id)));
        }

        Ok(())
    }

    /// Check if file hash already exists (duplicate detection)
    pub async fn exists_by_file_hash(&self, file_hash: &str) -> Result<bool> {
        let count: (i64,) = query_as(
            r#"
            SELECT COUNT(*) FROM staging.import_batch
            WHERE file_hash = $1
            "#,
        )
        .bind(file_hash)
        .fetch_one(self.pool)
        .await
        .map_err(Error::Database)?;

        Ok(count.0 > 0)
    }

    /// Get existing batch by file hash
    pub async fn get_by_file_hash(&self, file_hash: &str) -> Result<Option<ImportBatch>> {
        let batch = query_as::<_, ImportBatch>(
            r#"
            SELECT * FROM staging.import_batch
            WHERE file_hash = $1
            ORDER BY created_at DESC
            LIMIT 1
            "#,
        )
        .bind(file_hash)
        .fetch_optional(self.pool)
        .await
        .map_err(Error::Database)?;

        Ok(batch)
    }

    /// Count batches by organization
    pub async fn count_by_organization(&self, organization_id: i64) -> Result<i64> {
        let count: (i64,) = query_as(
            r#"
            SELECT COUNT(*) FROM staging.import_batch
            WHERE organization_id = $1
            "#,
        )
        .bind(organization_id)
        .fetch_one(self.pool)
        .await
        .map_err(Error::Database)?;

        Ok(count.0)
    }

    /// Count batches by status
    pub async fn count_by_status(
        &self,
        organization_id: i64,
        import_status: &str,
    ) -> Result<i64> {
        let count: (i64,) = query_as(
            r#"
            SELECT COUNT(*) FROM staging.import_batch
            WHERE organization_id = $1
            AND import_status = $2
            "#,
        )
        .bind(organization_id)
        .bind(import_status)
        .fetch_one(self.pool)
        .await
        .map_err(Error::Database)?;

        Ok(count.0)
    }

    /// Get recent batches summary
    pub async fn get_recent_summary(
        &self,
        organization_id: i64,
        limit: i64,
    ) -> Result<Vec<ImportBatch>> {
        query_as::<_, ImportBatch>(
            r#"
            SELECT * FROM staging.import_batch
            WHERE organization_id = $1
            ORDER BY created_at DESC
            LIMIT $2
            "#,
        )
        .bind(organization_id)
        .bind(limit)
        .fetch_all(self.pool)
        .await
        .map_err(Error::Database)
    }

    /// Delete old completed batches (soft cleanup)
    pub async fn delete_old_batches(
        &self,
        organization_id: i64,
        older_than_days: i32,
    ) -> Result<u64> {
        let rows_affected = query(
            r#"
            DELETE FROM staging.import_batch
            WHERE organization_id = $1
            AND import_status IN ('COMPLETED', 'FAILED')
            AND created_at < CURRENT_TIMESTAMP - INTERVAL '1 day' * $2
            "#,
        )
        .bind(organization_id)
        .bind(older_than_days)
        .execute(self.pool)
        .await
        .map_err(Error::Database)?
        .rows_affected();

        Ok(rows_affected)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connection::create_pool_default;

    #[tokio::test]
    #[ignore] // Requires database
    async fn test_import_batch_repository() {
        let pool = create_pool_default().await.unwrap();
        let repo = ImportBatchRepository::new(&pool);

        // Test count_by_organization with a sample organization ID
        let sample_org_id = 1i64;
        let count = repo.count_by_organization(sample_org_id).await;
        assert!(count.is_ok());
    }
}
