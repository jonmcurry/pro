//! Provider repository trait
//!
//! Defines provider-specific repository operations beyond basic CRUD.

use async_trait::async_trait;
use pro_common::Result;

use crate::models::Provider;

/// Provider-specific repository operations
#[async_trait]
pub trait ProviderRepositoryTrait: Send + Sync {
    // =========================================================================
    // QUERY OPERATIONS
    // =========================================================================

    /// Find provider by ID
    async fn find_by_id(&self, id: i64) -> Result<Option<Provider>>;

    /// Find provider by NPI
    async fn find_by_npi(&self, npi: &str) -> Result<Option<Provider>>;

    /// Find providers by organization
    async fn find_by_organization(
        &self,
        organization_id: i64,
        limit: Option<i64>,
        offset: Option<i64>,
    ) -> Result<Vec<Provider>>;

    /// Find providers by facility
    async fn find_by_facility(
        &self,
        facility_id: i64,
        limit: Option<i64>,
        offset: Option<i64>,
    ) -> Result<Vec<Provider>>;

    /// Find providers by taxonomy code
    async fn find_by_taxonomy(
        &self,
        organization_id: i64,
        taxonomy_code: &str,
        limit: Option<i64>,
        offset: Option<i64>,
    ) -> Result<Vec<Provider>>;

    /// Find providers by name (partial match)
    async fn find_by_name(
        &self,
        organization_id: i64,
        name_pattern: &str,
        limit: Option<i64>,
        offset: Option<i64>,
    ) -> Result<Vec<Provider>>;

    /// Search providers by multiple criteria
    async fn search(
        &self,
        organization_id: i64,
        npi: Option<&str>,
        name: Option<&str>,
        taxonomy: Option<&str>,
        limit: Option<i64>,
        offset: Option<i64>,
    ) -> Result<Vec<Provider>>;

    // =========================================================================
    // MUTATION OPERATIONS
    // =========================================================================

    /// Create a new provider
    async fn create(&self, provider: &Provider) -> Result<i64>;

    /// Create provider within transaction
    async fn create_with_tx(
        &self,
        provider: &Provider,
        tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    ) -> Result<i64>;

    /// Update a provider
    async fn update(&self, provider: &Provider) -> Result<()>;

    /// Update provider's primary taxonomy
    async fn update_taxonomy(&self, provider_id: i64, taxonomy_code: &str) -> Result<()>;

    /// Soft delete a provider
    async fn soft_delete(&self, provider_id: i64) -> Result<bool>;

    /// Link provider to facility
    async fn link_to_facility(&self, provider_id: i64, facility_id: i64) -> Result<()>;

    /// Unlink provider from facility
    async fn unlink_from_facility(&self, provider_id: i64, facility_id: i64) -> Result<()>;

    // =========================================================================
    // STATISTICS
    // =========================================================================

    /// Count providers by organization
    async fn count_by_organization(&self, organization_id: i64) -> Result<i64>;

    /// Count providers by facility
    async fn count_by_facility(&self, facility_id: i64) -> Result<i64>;

    /// Count providers by taxonomy
    async fn count_by_taxonomy(&self, organization_id: i64, taxonomy_code: &str) -> Result<i64>;

    /// Check if NPI exists for organization
    async fn npi_exists(&self, organization_id: i64, npi: &str) -> Result<bool>;

    /// Get provider's facility IDs
    async fn get_facility_ids(&self, provider_id: i64) -> Result<Vec<i64>>;
}

/// Mock implementation for testing
#[cfg(test)]
pub mod mock {
    use super::*;
    use std::collections::HashMap;
    use std::sync::{Arc, RwLock};

    /// Mock provider repository for unit testing
    pub struct MockProviderRepository {
        providers: Arc<RwLock<HashMap<i64, Provider>>>,
        next_id: Arc<RwLock<i64>>,
    }

    impl MockProviderRepository {
        pub fn new() -> Self {
            Self {
                providers: Arc::new(RwLock::new(HashMap::new())),
                next_id: Arc::new(RwLock::new(1)),
            }
        }

        pub fn with_providers(providers: Vec<Provider>) -> Self {
            let repo = Self::new();
            {
                let mut map = repo.providers.write().unwrap();
                let mut next_id = repo.next_id.write().unwrap();
                for prov in providers {
                    map.insert(prov.provider_id, prov.clone());
                    if prov.provider_id >= *next_id {
                        *next_id = prov.provider_id + 1;
                    }
                }
            }
            repo
        }
    }

    impl Default for MockProviderRepository {
        fn default() -> Self {
            Self::new()
        }
    }
}
