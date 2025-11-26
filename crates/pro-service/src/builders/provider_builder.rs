//! Provider Builder Module
//!
//! Builds and ensures provider records exist in the database.
//! Extracted from ClaimsProcessor as part of god object refactoring.
//!
//! NOTE: This module is scaffolding for future refactoring. Currently unused
//! but retained for planned integration with the claims processing pipeline.

#![allow(dead_code)]

use anyhow::{Context, Result};
use sqlx::PgPool;
use tracing::{debug, warn};

/// Builder for creating and looking up provider records
pub struct ProviderBuilder {
    pool: PgPool,
}

impl ProviderBuilder {
    /// Create a new provider builder
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Ensure a provider exists in claims.provider table, creating if necessary
    /// Returns the provider_id (either existing or newly created)
    pub async fn ensure_provider_exists(
        &self,
        tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
        npi: &str,
        provider_type: &str,
        last_name: Option<&str>,
        first_name: Option<&str>,
        middle_name: Option<&str>,
        taxonomy_code: Option<&str>,
        organization_id: Option<i64>,
    ) -> Result<Option<i64>> {
        // Skip if NPI is empty
        if npi.is_empty() {
            return Ok(None);
        }

        // Validate NPI format (10 digits)
        if npi.len() != 10 || !npi.chars().all(|c| c.is_ascii_digit()) {
            warn!("Invalid NPI format: {} (expected 10 digits)", npi);
            return Ok(None);
        }

        // Retry logic to handle deadlocks from concurrent provider creation
        const MAX_RETRIES: u32 = 5;
        let mut retry_count = 0;

        loop {
            // Check if provider already exists
            let existing_provider: Option<i64> = sqlx::query_scalar(
                r#"
                SELECT provider_id
                FROM claims.provider
                WHERE npi = $1
                "#,
            )
            .bind(npi)
            .fetch_optional(&mut **tx)
            .await
            .context("Failed to query existing provider")?;

            if let Some(provider_id) = existing_provider {
                return Ok(Some(provider_id));
            }

            // Provider doesn't exist, create it
            let last_name_value = last_name.unwrap_or("Unknown");
            let first_name_value = first_name.unwrap_or("");

            // Lookup specialty from taxonomy code if provided
            let specialty = if let Some(tax_code) = taxonomy_code {
                sqlx::query_scalar::<_, String>(
                    r#"
                    SELECT specialty_display
                    FROM claims.provider_taxonomy
                    WHERE taxonomy_code = $1 AND is_active = true
                    "#,
                )
                .bind(tax_code)
                .fetch_optional(&mut **tx)
                .await
                .unwrap_or(None)
            } else {
                None
            };

            if let Some(ref spec) = specialty {
                debug!(
                    "Mapped taxonomy {} to specialty: {}",
                    taxonomy_code.unwrap_or(""),
                    spec
                );
            } else if taxonomy_code.is_some() && !taxonomy_code.unwrap().is_empty() {
                warn!(
                    "No specialty mapping found for taxonomy code: {}",
                    taxonomy_code.unwrap()
                );
            }

            // Try to insert the provider
            let insert_result = sqlx::query_scalar::<_, i64>(
                r#"
                INSERT INTO claims.provider (
                    npi,
                    provider_type,
                    last_name,
                    first_name,
                    middle_name,
                    taxonomy_code,
                    specialty,
                    organization_id,
                    is_active,
                    created_at,
                    updated_at
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, true, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT (npi) DO NOTHING
                RETURNING provider_id
                "#,
            )
            .bind(npi)
            .bind(provider_type)
            .bind(last_name_value)
            .bind(first_name_value)
            .bind(middle_name)
            .bind(taxonomy_code)
            .bind(specialty.as_deref())
            .bind(organization_id)
            .fetch_optional(&mut **tx)
            .await;

            match insert_result {
                Ok(Some(provider_id)) => {
                    debug!(
                        "Created new provider: NPI={}, Type={}, Name={} {}, Specialty={:?}",
                        npi, provider_type, first_name_value, last_name_value, specialty
                    );

                    // Enqueue provider for background NPI enrichment (fire-and-forget)
                    let _ = sqlx::query(
                        r#"
                        INSERT INTO claims.provider_enrichment_queue (provider_id, npi, priority)
                        VALUES ($1, $2, $3)
                        ON CONFLICT (provider_id) DO NOTHING
                        "#,
                    )
                    .bind(provider_id)
                    .bind(npi)
                    .bind(5) // Default priority
                    .execute(&mut **tx)
                    .await;

                    return Ok(Some(provider_id));
                }
                Ok(None) => {
                    // ON CONFLICT DO NOTHING was triggered - provider was created by another transaction
                    let existing_id: Option<i64> = sqlx::query_scalar(
                        r#"
                        SELECT provider_id
                        FROM claims.provider
                        WHERE npi = $1
                        "#,
                    )
                    .bind(npi)
                    .fetch_optional(&mut **tx)
                    .await
                    .context("Failed to query provider after conflict")?;

                    if let Some(id) = existing_id {
                        debug!(
                            "Provider already exists (concurrent creation): NPI={}, provider_id={}",
                            npi, id
                        );
                        return Ok(Some(id));
                    } else {
                        // Very rare: provider was deleted between INSERT and SELECT
                        if retry_count < MAX_RETRIES {
                            retry_count += 1;
                            let backoff_ms = 10u64 * 2u64.pow(retry_count - 1);
                            warn!(
                                "Provider disappeared after conflict, retrying ({}/{}): NPI={}, backoff={}ms",
                                retry_count, MAX_RETRIES, npi, backoff_ms
                            );
                            tokio::time::sleep(tokio::time::Duration::from_millis(backoff_ms))
                                .await;
                            continue;
                        } else {
                            return Err(anyhow::anyhow!(
                                "Provider creation failed after {} retries: NPI={}",
                                MAX_RETRIES,
                                npi
                            ));
                        }
                    }
                }
                Err(e) => {
                    let error_string = e.to_string();
                    let is_deadlock = error_string.contains("deadlock detected")
                        || error_string.contains("40P01");

                    if is_deadlock && retry_count < MAX_RETRIES {
                        retry_count += 1;
                        let backoff_ms = 10u64 * 2u64.pow(retry_count - 1);
                        let jitter = (rand::random::<u64>() % 10) as u64;
                        let total_backoff = backoff_ms + jitter;

                        warn!(
                            "Deadlock detected creating provider, retrying ({}/{}): NPI={}, backoff={}ms",
                            retry_count, MAX_RETRIES, npi, total_backoff
                        );

                        tokio::time::sleep(tokio::time::Duration::from_millis(total_backoff))
                            .await;
                        continue;
                    } else {
                        return Err(e).context("Failed to insert provider");
                    }
                }
            }
        }
    }

    /// Parse billing provider name into first/last components
    pub fn parse_billing_provider_name(name: Option<&str>) -> (Option<&str>, Option<&str>) {
        match name {
            Some(n) if n.contains(',') => {
                let parts: Vec<&str> = n.splitn(2, ',').collect();
                (Some(parts[0].trim()), parts.get(1).map(|s| s.trim()))
            }
            Some(n) => (Some(n), None),
            None => (None, None),
        }
    }
}
