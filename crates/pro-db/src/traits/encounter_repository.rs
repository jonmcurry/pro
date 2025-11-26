//! Encounter repository trait
//!
//! Defines encounter-specific repository operations beyond basic CRUD.

use async_trait::async_trait;
use chrono::NaiveDate;
use pro_common::Result;

use crate::models::{Encounter, EncounterDiagnosis, ServiceLine};

/// Encounter-specific repository operations
#[async_trait]
pub trait EncounterRepositoryTrait: Send + Sync {
    // =========================================================================
    // QUERY OPERATIONS
    // =========================================================================

    /// Find encounter by ID
    async fn find_by_id(&self, id: i64) -> Result<Option<Encounter>>;

    /// Find encounter by patient control number
    async fn find_by_patient_control_number(&self, pcn: &str) -> Result<Option<Encounter>>;

    /// Find encounters by patient control number (may return multiple)
    async fn find_all_by_patient_control_number(&self, pcn: &str) -> Result<Vec<Encounter>>;

    /// Find encounters by facility
    async fn find_by_facility(
        &self,
        facility_id: i64,
        limit: Option<i64>,
        offset: Option<i64>,
    ) -> Result<Vec<Encounter>>;

    /// Find encounters by organization
    async fn find_by_organization(
        &self,
        organization_id: i64,
        limit: Option<i64>,
        offset: Option<i64>,
    ) -> Result<Vec<Encounter>>;

    /// Find encounters by date range
    async fn find_by_date_range(
        &self,
        organization_id: i64,
        start_date: NaiveDate,
        end_date: NaiveDate,
        limit: Option<i64>,
        offset: Option<i64>,
    ) -> Result<Vec<Encounter>>;

    /// Find encounters by claim status
    async fn find_by_status(
        &self,
        organization_id: i64,
        status: &str,
        limit: Option<i64>,
        offset: Option<i64>,
    ) -> Result<Vec<Encounter>>;

    /// Find encounters with flags
    async fn find_with_flags(
        &self,
        organization_id: i64,
        limit: Option<i64>,
        offset: Option<i64>,
    ) -> Result<Vec<Encounter>>;

    // =========================================================================
    // MUTATION OPERATIONS
    // =========================================================================

    /// Create a new encounter
    async fn create(&self, encounter: &Encounter) -> Result<i64>;

    /// Create encounter within transaction
    async fn create_with_tx(
        &self,
        encounter: &Encounter,
        tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    ) -> Result<i64>;

    /// Update an encounter
    async fn update(&self, encounter: &Encounter) -> Result<()>;

    /// Update encounter status
    async fn update_status(&self, encounter_id: i64, status: &str) -> Result<()>;

    /// Soft delete an encounter
    async fn soft_delete(&self, encounter_id: i64) -> Result<bool>;

    // =========================================================================
    // DIAGNOSIS OPERATIONS
    // =========================================================================

    /// Get diagnoses for an encounter
    async fn get_diagnoses(&self, encounter_id: i64) -> Result<Vec<EncounterDiagnosis>>;

    /// Create diagnoses for an encounter
    async fn create_diagnoses(&self, diagnoses: &[EncounterDiagnosis]) -> Result<Vec<i64>>;

    /// Create diagnoses within transaction
    async fn create_diagnoses_batch_with_tx(
        &self,
        diagnoses: &[EncounterDiagnosis],
        tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    ) -> Result<Vec<i64>>;

    // =========================================================================
    // SERVICE LINE OPERATIONS
    // =========================================================================

    /// Get service lines for an encounter
    async fn get_service_lines(&self, encounter_id: i64) -> Result<Vec<ServiceLine>>;

    // =========================================================================
    // STATISTICS
    // =========================================================================

    /// Count encounters by organization
    async fn count_by_organization(&self, organization_id: i64) -> Result<i64>;

    /// Count encounters by facility
    async fn count_by_facility(&self, facility_id: i64) -> Result<i64>;

    /// Count encounters by status
    async fn count_by_status(&self, organization_id: i64, status: &str) -> Result<i64>;

    /// Count flagged encounters
    async fn count_flagged(&self, organization_id: i64) -> Result<i64>;

    /// Check if patient control number exists for organization
    async fn pcn_exists(&self, organization_id: i64, pcn: &str) -> Result<bool>;
}

/// Mock implementation for testing
#[cfg(test)]
pub mod mock {
    use super::*;
    use std::collections::HashMap;
    use std::sync::{Arc, RwLock};

    /// Mock encounter repository for unit testing
    pub struct MockEncounterRepository {
        encounters: Arc<RwLock<HashMap<i64, Encounter>>>,
        next_id: Arc<RwLock<i64>>,
    }

    impl MockEncounterRepository {
        pub fn new() -> Self {
            Self {
                encounters: Arc::new(RwLock::new(HashMap::new())),
                next_id: Arc::new(RwLock::new(1)),
            }
        }

        pub fn with_encounters(encounters: Vec<Encounter>) -> Self {
            let repo = Self::new();
            {
                let mut map = repo.encounters.write().unwrap();
                let mut next_id = repo.next_id.write().unwrap();
                for enc in encounters {
                    map.insert(enc.encounter_id, enc.clone());
                    if enc.encounter_id >= *next_id {
                        *next_id = enc.encounter_id + 1;
                    }
                }
            }
            repo
        }
    }

    impl Default for MockEncounterRepository {
        fn default() -> Self {
            Self::new()
        }
    }
}
