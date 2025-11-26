//! Raw Claim Reader Module
//!
//! Handles reading and status management for raw claims in staging.
//! Extracted from ClaimsProcessor as part of god object refactoring.
//!
//! NOTE: This module is scaffolding for future refactoring. Currently unused
//! but retained for planned integration with the claims processing pipeline.

#![allow(dead_code)]

use anyhow::{Context, Result};
use serde_json::Value as JsonValue;
use sqlx::PgPool;
use std::collections::HashMap;
use tracing::info;

/// Reader for raw claims from staging.raw_claims table
pub struct RawClaimReader {
    pool: PgPool,
}

impl RawClaimReader {
    /// Create a new raw claim reader
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Query pending raw claims for processing
    pub async fn fetch_pending_claims(&self, limit: usize) -> Result<Vec<RawClaim>> {
        let raw_claims: Vec<RawClaim> = sqlx::query_as(
            r#"
            SELECT
                raw_claim_id,
                batch_id,
                queue_id,
                encounter_fields,
                service_line_fields,
                diagnosis_fields,
                row_number,
                facility_code,
                date_of_service_from
            FROM staging.raw_claims
            WHERE processing_status = 'PENDING'
            ORDER BY ingested_at ASC
            LIMIT $1
            FOR UPDATE SKIP LOCKED
            "#,
        )
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await
        .context("Failed to query pending raw claims")?;

        info!("Found {} pending raw claims to process", raw_claims.len());
        Ok(raw_claims)
    }

    /// Mark claims as PROCESSING
    pub async fn mark_claims_processing(&self, claim_ids: &[i64]) -> Result<()> {
        sqlx::query(
            r#"
            UPDATE staging.raw_claims
            SET processing_status = 'PROCESSING'
            WHERE raw_claim_id = ANY($1)
            "#,
        )
        .bind(claim_ids)
        .execute(&self.pool)
        .await
        .context("Failed to mark claims as PROCESSING")?;

        Ok(())
    }

    /// Mark a claim as COMPLETED
    pub async fn mark_claim_completed(
        &self,
        tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
        raw_claim_id: i64,
    ) -> Result<()> {
        sqlx::query(
            r#"
            UPDATE staging.raw_claims
            SET processing_status = 'COMPLETED',
                processed_at = CURRENT_TIMESTAMP
            WHERE raw_claim_id = $1
            "#,
        )
        .bind(raw_claim_id)
        .execute(&mut **tx)
        .await?;

        Ok(())
    }

    /// Mark claims as COMPLETED in bulk
    pub async fn mark_claims_completed(
        &self,
        tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
        claim_ids: &[i64],
    ) -> Result<()> {
        sqlx::query(
            r#"
            UPDATE staging.raw_claims
            SET processing_status = 'COMPLETED',
                processed_at = CURRENT_TIMESTAMP
            WHERE raw_claim_id = ANY($1)
            "#,
        )
        .bind(claim_ids)
        .execute(&mut **tx)
        .await?;

        Ok(())
    }

    /// Mark a claim as FAILED
    pub async fn mark_claim_failed(
        &self,
        tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
        raw_claim_id: i64,
        error_message: &str,
    ) -> Result<()> {
        sqlx::query(
            r#"
            UPDATE staging.raw_claims
            SET processing_status = 'FAILED',
                error_message = $2
            WHERE raw_claim_id = $1
            "#,
        )
        .bind(raw_claim_id)
        .bind(error_message)
        .execute(&mut **tx)
        .await?;

        Ok(())
    }

    /// Mark claims as FAILED in bulk
    pub async fn mark_claims_failed(
        &self,
        tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
        claim_ids: &[i64],
        error_message: &str,
    ) -> Result<()> {
        sqlx::query(
            r#"
            UPDATE staging.raw_claims
            SET processing_status = 'FAILED',
                processed_at = CURRENT_TIMESTAMP,
                error_message = $1
            WHERE raw_claim_id = ANY($2)
            "#,
        )
        .bind(error_message)
        .bind(claim_ids)
        .execute(&mut **tx)
        .await?;

        Ok(())
    }

    /// Group raw claims by encounter key (patient_control_number + date_of_service)
    pub fn group_claims_by_encounter(
        &self,
        raw_claims: Vec<RawClaim>,
    ) -> Result<HashMap<(String, String), Vec<RawClaim>>> {
        let mut encounter_groups: HashMap<(String, String), Vec<RawClaim>> = HashMap::new();

        for raw_claim in raw_claims {
            let encounter_fields: HashMap<String, String> =
                serde_json::from_value(raw_claim.encounter_fields.clone())?;

            let patient_control_number = encounter_fields
                .get("patient_control_number")
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "Missing patient_control_number for raw_claim_id {}",
                        raw_claim.raw_claim_id
                    )
                })?
                .clone();

            let date_of_service = encounter_fields
                .get("date_of_service_from")
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "Missing date_of_service_from for raw_claim_id {}",
                        raw_claim.raw_claim_id
                    )
                })?
                .clone();

            let encounter_key = (patient_control_number, date_of_service);
            encounter_groups
                .entry(encounter_key)
                .or_insert_with(Vec::new)
                .push(raw_claim);
        }

        Ok(encounter_groups)
    }

    /// Log an error to staging.import_error_log
    pub async fn log_import_error(
        &self,
        tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
        batch_id: i64,
        record_number: i32,
        error_type: &str,
        error_severity: &str,
        error_message: &str,
        raw_data: Option<String>,
    ) -> Result<i64> {
        let error_id: i64 = sqlx::query_scalar(
            r#"
            INSERT INTO staging.import_error_log (
                batch_id,
                record_number,
                error_type,
                error_severity,
                error_message,
                raw_data
            )
            VALUES ($1, $2, $3, $4, $5, $6)
            RETURNING error_id
            "#,
        )
        .bind(batch_id)
        .bind(record_number)
        .bind(error_type)
        .bind(error_severity)
        .bind(error_message)
        .bind(raw_data)
        .fetch_one(&mut **tx)
        .await?;

        Ok(error_id)
    }
}

/// Raw claim from staging.raw_claims table
#[derive(Debug, Clone, sqlx::FromRow)]
pub struct RawClaim {
    pub raw_claim_id: i64,
    pub batch_id: i64,
    pub queue_id: i64,
    pub encounter_fields: JsonValue,
    pub service_line_fields: Option<JsonValue>,
    pub diagnosis_fields: Option<JsonValue>,
    pub row_number: i32,
    pub facility_code: Option<String>,
    pub date_of_service_from: Option<chrono::NaiveDate>,
}
