pub mod backup;
pub mod config;
pub mod database;
pub mod migration;
pub mod registry;
pub mod windows;

pub use backup::BackupService;
pub use config::ConfigService;
pub use database::DatabaseService;
pub use migration::MigrationService;
pub use registry::{ProjectInfo, RegistryService};
pub use windows::WindowsServiceManager;
