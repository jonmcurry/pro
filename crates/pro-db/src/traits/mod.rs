//! Repository trait abstractions
//!
//! This module provides trait-based abstractions for repository operations,
//! enabling better testability through dependency injection and mocking.

pub mod repository;
pub mod encounter_repository;
pub mod provider_repository;

pub use repository::Repository;
pub use encounter_repository::EncounterRepositoryTrait;
pub use provider_repository::ProviderRepositoryTrait;
