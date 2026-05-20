use anyhow::{Context, Result};
use sqlx::PgPool;
use tokio::time::{sleep, Duration};
use tracing::{debug, error, info, warn};

use crate::client::NpiRegistryClient;

/// Background worker for enriching provider data from NPI Registry
///
/// This worker runs asynchronously and does not block claims processing.
/// It polls the enrichment queue, calls the NPI Registry API, and updates
/// provider records with detailed information.
pub struct EnrichmentWorker {
    pool: PgPool,
    client: NpiRegistryClient,
    config: WorkerConfig,
}

/// Worker configuration
#[derive(Debug, Clone)]
pub struct WorkerConfig {
    /// Number of providers to process per batch
    pub batch_size: usize,
    /// How long to wait between polls when queue is empty
    pub poll_interval: Duration,
    /// Delay between individual API requests (for rate limiting)
    pub rate_limit_delay: Duration,
    /// Whether the worker is enabled
    pub enabled: bool,
}

impl Default for WorkerConfig {
    fn default() -> Self {
        Self {
            batch_size: 10,
            poll_interval: Duration::from_secs(30),
            rate_limit_delay: Duration::from_millis(200), // 5 req/sec max
            enabled: true,
        }
    }
}

impl EnrichmentWorker {
    /// Create a new enrichment worker with default configuration
    pub fn new(pool: PgPool) -> Result<Self> {
        Self::with_config(pool, WorkerConfig::default())
    }

    /// Create a new enrichment worker with custom configuration
    pub fn with_config(pool: PgPool, config: WorkerConfig) -> Result<Self> {
        let client = NpiRegistryClient::new()
            .context("Failed to create NPI Registry client")?;

        Ok(Self {
            pool,
            client,
            config,
        })
    }

    /// Run the enrichment worker (infinite loop)
    ///
    /// This method runs forever, continuously processing the enrichment queue.
    /// It should be spawned in a separate async task.
    pub async fn run(&self) -> Result<()> {
        if !self.config.enabled {
            info!("NPI enrichment worker is disabled by configuration");
            return Ok(());
        }

        info!("Starting NPI enrichment worker (batch_size: {}, poll_interval: {:?}, rate_limit: {:?})",
            self.config.batch_size, self.config.poll_interval, self.config.rate_limit_delay);

        loop {
            match self.process_batch().await {
                Ok(processed) => {
                    if processed == 0 {
                        debug!("No pending enrichments, sleeping for {:?}", self.config.poll_interval);
                        sleep(self.config.poll_interval).await;
                    } else {
                        info!("Processed {} provider enrichments", processed);
                        // Continue immediately if we processed a full batch
                        if processed < self.config.batch_size {
                            sleep(Duration::from_secs(1)).await;
                        }
                    }
                }
                Err(e) => {
                    error!("Error processing enrichment batch: {}", e);
                    sleep(Duration::from_secs(10)).await;
                }
            }
        }
    }

    /// Process a single batch of pending enrichments
    ///
    /// Returns the number of providers successfully processed.
    async fn process_batch(&self) -> Result<usize> {
        // Fetch pending enrichments with FOR UPDATE SKIP LOCKED to avoid contention
        let pending: Vec<(i64, i64, String)> = sqlx::query_as(
            r#"
            SELECT queue_id, provider_id, npi
            FROM claims.provider_enrichment_queue
            WHERE (status = 'PENDING' OR (status = 'FAILED' AND next_retry_at <= CURRENT_TIMESTAMP))
              AND retry_count < max_retries
            ORDER BY priority DESC, created_at ASC
            LIMIT $1
            FOR UPDATE SKIP LOCKED
            "#
        )
        .bind(self.config.batch_size as i32)
        .fetch_all(&self.pool)
        .await
        .context("Failed to fetch pending enrichments from queue")?;

        if pending.is_empty() {
            return Ok(0);
        }

        debug!("Found {} pending enrichments to process", pending.len());

        let mut processed = 0;

        for (queue_id, provider_id, npi) in pending {
            // Mark as IN_PROGRESS
            if let Err(e) = self.mark_in_progress(queue_id).await {
                warn!("Failed to mark queue item as IN_PROGRESS: {}", e);
                continue;
            }

            // Enrich provider
            match self.enrich_provider(provider_id, &npi).await {
                Ok(_) => {
                    // Mark as COMPLETED
                    if let Err(e) = self.mark_completed(queue_id).await {
                        error!("Failed to mark queue item as COMPLETED: {}", e);
                    } else {
                        processed += 1;
                        info!("Successfully enriched provider NPI: {}", npi);
                    }
                }
                Err(e) => {
                    warn!("Failed to enrich provider NPI {}: {}", npi, e);

                    // Mark as FAILED and schedule retry
                    if let Err(retry_err) = self.mark_failed(queue_id, &e.to_string()).await {
                        error!("Failed to mark queue item as FAILED: {}", retry_err);
                    }
                }
            }

            // Rate limiting: wait between requests to avoid overwhelming the API
            sleep(self.config.rate_limit_delay).await;
        }

        Ok(processed)
    }

    /// Enrich a single provider from NPI Registry API
    async fn enrich_provider(&self, provider_id: i64, npi: &str) -> Result<()> {
        // Call NPI Registry API
        let response = self.client.lookup_npi(npi).await
            .context("NPI Registry API call failed")?;

        if response.results.is_empty() {
            anyhow::bail!("No results returned for NPI: {}", npi);
        }

        let provider_data = &response.results[0];

        // Start transaction
        let mut tx = self.pool.begin().await
            .context("Failed to begin transaction")?;

        // Get primary taxonomy (or first available)
        let primary_taxonomy = provider_data.taxonomies.iter()
            .find(|t| t.primary)
            .or_else(|| provider_data.taxonomies.first());

        // Lookup specialty from taxonomy code; auto-insert if missing
        let specialty = if let Some(taxonomy) = primary_taxonomy {
            let existing = sqlx::query_scalar::<_, String>(
                r#"
                SELECT specialty_display
                FROM claims.provider_taxonomy
                WHERE taxonomy_code = $1 AND is_active = true
                "#
            )
            .bind(&taxonomy.code)
            .fetch_optional(&mut *tx)
            .await
            .context("Failed to lookup specialty from taxonomy")?;

            if existing.is_none() {
                let provider_type = match provider_data.enumeration_type.as_str() {
                    "NPI-1" => "Individual",
                    "NPI-2" => "Organization",
                    _ => "Unknown",
                };
                let specialty_display = &taxonomy.desc;

                sqlx::query(
                    r#"
                    INSERT INTO claims.provider_taxonomy
                        (taxonomy_code, provider_type, classification, specialization, specialty_display, is_active)
                    VALUES ($1, $2, $3, NULL, $4, true)
                    ON CONFLICT (taxonomy_code) DO NOTHING
                    "#
                )
                .bind(&taxonomy.code)
                .bind(provider_type)
                .bind(specialty_display)
                .bind(specialty_display)
                .execute(&mut *tx)
                .await
                .context("Failed to auto-insert unknown taxonomy code")?;

                warn!(
                    "Auto-inserted taxonomy code '{}' ({}) into provider_taxonomy from NPI Registry",
                    taxonomy.code, taxonomy.desc
                );

                Some(taxonomy.desc.clone())
            } else {
                existing
            }
        } else {
            None
        };

        if let Some(ref spec) = specialty {
            debug!("Mapped taxonomy {} to specialty: {}", primary_taxonomy.unwrap().code, spec);
        }

        // Get location address (or first available)
        let location_address = provider_data.addresses.iter()
            .find(|a| a.address_purpose == "LOCATION")
            .or_else(|| provider_data.addresses.first());

        // Format postal code (remove hyphen for consistency)
        let postal_code = location_address
            .and_then(|a| a.postal_code.as_ref())
            .map(|pc| pc.replace("-", "").chars().take(9).collect::<String>());

        // Generate NPI Registry link
        let npi_registry_link = format!("https://npiregistry.cms.hhs.gov/api/?version=2.1&number={}", npi);

        // Update provider record with COALESCE to preserve existing non-null values
        sqlx::query(
            r#"
            UPDATE claims.provider
            SET
                first_name = COALESCE($2, first_name),
                last_name = COALESCE($3, last_name),
                middle_name = COALESCE($4, middle_name),
                name_suffix = COALESCE($5, name_suffix),
                taxonomy_code = COALESCE($6, taxonomy_code),
                specialty = COALESCE($7, specialty),
                license_number = COALESCE($8, license_number),
                license_state = COALESCE($9, license_state),
                address_line1 = COALESCE($10, address_line1),
                address_line2 = COALESCE($11, address_line2),
                city = COALESCE($12, city),
                state_code = COALESCE($13, state_code),
                postal_code = COALESCE($14, postal_code),
                phone = COALESCE($15, phone),
                npi_registry_link = $16,
                updated_at = CURRENT_TIMESTAMP,
                updated_by = 'NPI_ENRICHMENT'
            WHERE provider_id = $1
            "#
        )
        .bind(provider_id)
        .bind(provider_data.basic.first_name.as_deref())
        .bind(provider_data.basic.last_name.as_deref())
        .bind(provider_data.basic.middle_name.as_deref())
        .bind(provider_data.basic.credential.as_deref())
        .bind(primary_taxonomy.map(|t| t.code.as_str()))
        .bind(specialty.as_deref())
        .bind(primary_taxonomy.and_then(|t| t.license.as_deref()))
        .bind(primary_taxonomy.and_then(|t| t.state.as_deref()))
        .bind(location_address.and_then(|a| a.address_1.as_deref()))
        .bind(location_address.and_then(|a| a.address_2.as_deref()))
        .bind(location_address.and_then(|a| a.city.as_deref()))
        .bind(location_address.and_then(|a| a.state.as_deref()))
        .bind(postal_code.as_deref())
        .bind(location_address.and_then(|a| a.telephone_number.as_deref()))
        .bind(&npi_registry_link)
        .execute(&mut *tx)
        .await
        .context("Failed to update provider record")?;

        // Store full API response in queue for audit trail
        let api_response_json = serde_json::to_value(&response)
            .context("Failed to serialize API response")?;

        sqlx::query(
            r#"
            UPDATE claims.provider_enrichment_queue
            SET api_response = $2
            WHERE provider_id = $1
            "#
        )
        .bind(provider_id)
        .bind(api_response_json)
        .execute(&mut *tx)
        .await
        .context("Failed to store API response in queue")?;

        // Commit transaction
        tx.commit().await
            .context("Failed to commit enrichment transaction")?;

        debug!("Successfully enriched provider {} with NPI {}", provider_id, npi);

        Ok(())
    }

    /// Mark queue item as IN_PROGRESS
    async fn mark_in_progress(&self, queue_id: i64) -> Result<()> {
        sqlx::query(
            r#"
            UPDATE claims.provider_enrichment_queue
            SET status = 'IN_PROGRESS',
                started_at = CURRENT_TIMESTAMP
            WHERE queue_id = $1
            "#
        )
        .bind(queue_id)
        .execute(&self.pool)
        .await
        .context("Failed to mark queue item as IN_PROGRESS")?;

        Ok(())
    }

    /// Mark queue item as COMPLETED
    async fn mark_completed(&self, queue_id: i64) -> Result<()> {
        sqlx::query(
            r#"
            UPDATE claims.provider_enrichment_queue
            SET status = 'COMPLETED',
                completed_at = CURRENT_TIMESTAMP
            WHERE queue_id = $1
            "#
        )
        .bind(queue_id)
        .execute(&self.pool)
        .await
        .context("Failed to mark queue item as COMPLETED")?;

        Ok(())
    }

    /// Mark queue item as FAILED and schedule retry with exponential backoff
    async fn mark_failed(&self, queue_id: i64, error_message: &str) -> Result<()> {
        sqlx::query(
            r#"
            UPDATE claims.provider_enrichment_queue
            SET status = 'FAILED',
                retry_count = retry_count + 1,
                last_error = $2,
                last_error_at = CURRENT_TIMESTAMP,
                next_retry_at = CURRENT_TIMESTAMP + INTERVAL '1 hour' * POWER(2, retry_count)
            WHERE queue_id = $1
            "#
        )
        .bind(queue_id)
        .bind(error_message)
        .execute(&self.pool)
        .await
        .context("Failed to mark queue item as FAILED")?;

        Ok(())
    }

    /// Get queue statistics (for monitoring)
    pub async fn get_queue_stats(&self) -> Result<QueueStats> {
        let stats: (i64, i64, i64, i64) = sqlx::query_as(
            r#"
            SELECT
                COUNT(*) FILTER (WHERE status = 'PENDING') as pending,
                COUNT(*) FILTER (WHERE status = 'IN_PROGRESS') as in_progress,
                COUNT(*) FILTER (WHERE status = 'COMPLETED') as completed,
                COUNT(*) FILTER (WHERE status = 'FAILED' AND retry_count >= max_retries) as permanently_failed
            FROM claims.provider_enrichment_queue
            "#
        )
        .fetch_one(&self.pool)
        .await
        .context("Failed to fetch queue statistics")?;

        Ok(QueueStats {
            pending: stats.0 as usize,
            in_progress: stats.1 as usize,
            completed: stats.2 as usize,
            permanently_failed: stats.3 as usize,
        })
    }
}

/// Queue statistics for monitoring
#[derive(Debug, Clone)]
pub struct QueueStats {
    pub pending: usize,
    pub in_progress: usize,
    pub completed: usize,
    pub permanently_failed: usize,
}

impl QueueStats {
    pub fn total(&self) -> usize {
        self.pending + self.in_progress + self.completed + self.permanently_failed
    }

    pub fn completion_rate(&self) -> f64 {
        let total = self.total();
        if total == 0 {
            return 0.0;
        }
        (self.completed as f64 / total as f64) * 100.0
    }
}
