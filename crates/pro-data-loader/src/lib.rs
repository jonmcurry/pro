//! Professional SMART Master Data Loader Library
//!
//! This library provides functionality for loading master data (organizations, regions,
//! facilities, providers) into the Professional SMART database from CSV files.
//!
//! It includes validation, parsing, and import capabilities that can be used by both
//! the CLI application and the GUI application.

pub mod config;
pub mod csv_parser;
pub mod importer;
pub mod models;
pub mod templates;
pub mod validator;

// Re-export commonly used types
pub use config::Config;
pub use models::{Facility, ImportResults, Organization, Provider, Region};

/// Load and validate all CSV files from the given paths
pub async fn load_and_validate_files(
    org_path: &str,
    region_path: &str,
    facility_path: &str,
    provider_path: &str,
) -> anyhow::Result<(
    Vec<Organization>,
    Vec<Region>,
    Vec<Facility>,
    Vec<Provider>,
)> {
    use anyhow::Context;

    // Parse CSV files
    let organizations =
        csv_parser::parse_organizations(org_path).context("Failed to parse organizations")?;
    let regions = csv_parser::parse_regions(region_path).context("Failed to parse regions")?;
    let facilities =
        csv_parser::parse_facilities(facility_path).context("Failed to parse facilities")?;
    let providers =
        csv_parser::parse_providers(provider_path).context("Failed to parse providers")?;

    // Validate data
    validator::validate_organizations(&organizations)?;
    validator::validate_regions(&regions, &organizations)?;
    validator::validate_facilities(&facilities, &regions, &organizations)?;
    validator::validate_providers(&providers, &facilities)?;

    Ok((organizations, regions, facilities, providers))
}

/// Import all validated data into the database
pub async fn import_all_data(
    database_url: &str,
    organizations: Vec<Organization>,
    regions: Vec<Region>,
    facilities: Vec<Facility>,
    providers: Vec<Provider>,
) -> anyhow::Result<ImportResults> {
    importer::import_all(database_url, organizations, regions, facilities, providers).await
}
