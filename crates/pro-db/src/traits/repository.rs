//! Base repository trait
//!
//! Defines the fundamental CRUD operations that all repositories should support.

use async_trait::async_trait;
use pro_common::Result;

/// Base repository trait for CRUD operations
///
/// Generic over:
/// - `T`: The entity type
/// - `Id`: The ID type (can be a newtype like EncounterId)
///
/// # Example
/// ```ignore
/// use pro_db::traits::Repository;
/// use pro_common::EncounterId;
///
/// struct MyRepository;
///
/// #[async_trait]
/// impl Repository<Encounter, EncounterId> for MyRepository {
///     async fn find_by_id(&self, id: EncounterId) -> Result<Option<Encounter>> {
///         // Implementation
///     }
///     // ... other methods
/// }
/// ```
#[async_trait]
pub trait Repository<T, Id: Send + 'static>: Send + Sync {
    /// Find an entity by its ID
    async fn find_by_id(&self, id: Id) -> Result<Option<T>>;

    /// Find all entities with pagination
    async fn find_all(&self, limit: Option<i64>, offset: Option<i64>) -> Result<Vec<T>>;

    /// Create a new entity and return its ID
    async fn create(&self, entity: &T) -> Result<Id>;

    /// Update an existing entity
    async fn update(&self, entity: &T) -> Result<()>;

    /// Delete an entity by ID, returns true if deleted
    async fn delete(&self, id: Id) -> Result<bool>;

    /// Check if an entity exists by ID
    async fn exists(&self, id: Id) -> Result<bool> {
        Ok(self.find_by_id(id).await?.is_some())
    }

    /// Count total entities
    async fn count(&self) -> Result<i64>;
}

/// Extension trait for soft-deletable entities
#[async_trait]
pub trait SoftDeletable<Id: Send + 'static>: Send + Sync {
    /// Soft delete an entity by setting soft_deleted = true
    async fn soft_delete(&self, id: Id) -> Result<bool>;

    /// Restore a soft-deleted entity
    async fn restore(&self, id: Id) -> Result<bool>;

    /// Permanently delete an entity
    async fn hard_delete(&self, id: Id) -> Result<bool>;
}

/// Extension trait for batch operations
#[async_trait]
pub trait BatchRepository<T, Id: Send + 'static>: Repository<T, Id> {
    /// Create multiple entities in a single transaction
    async fn create_batch(&self, entities: &[T]) -> Result<Vec<Id>>;

    /// Update multiple entities in a single transaction
    async fn update_batch(&self, entities: &[T]) -> Result<()>;

    /// Delete multiple entities by IDs
    async fn delete_batch(&self, ids: &[Id]) -> Result<usize>;
}

/// Extension trait for transactional operations
#[async_trait]
pub trait TransactionalRepository<T, Id: Send + 'static>: Repository<T, Id> {
    /// Create an entity within an existing transaction
    async fn create_with_tx(
        &self,
        entity: &T,
        tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    ) -> Result<Id>;

    /// Update an entity within an existing transaction
    async fn update_with_tx(
        &self,
        entity: &T,
        tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    ) -> Result<()>;

    /// Delete an entity within an existing transaction
    async fn delete_with_tx(
        &self,
        id: Id,
        tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    ) -> Result<bool>;
}
