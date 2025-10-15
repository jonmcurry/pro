// Repository pattern for database access

pub mod organization;
pub mod facility;
pub mod provider;
pub mod coder;
pub mod reviewer;
pub mod encounter;
pub mod service_line;
pub mod diagnosis;
pub mod flag;
pub mod import_batch;
pub mod rvu;
pub mod denial;

// Re-export repositories
pub use organization::OrganizationRepository;
pub use facility::FacilityRepository;
pub use provider::ProviderRepository;
pub use coder::CoderRepository;
pub use reviewer::ReviewerRepository;
pub use encounter::EncounterRepository;
pub use service_line::ServiceLineRepository;
pub use diagnosis::DiagnosisRepository;
pub use flag::FlagRepository;
pub use import_batch::ImportBatchRepository;
pub use rvu::RvuRepository;
pub use denial::DenialRepository;
