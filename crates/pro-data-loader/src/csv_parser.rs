//! CSV file parsing

use anyhow::{Context, Result};
use std::path::Path;

use crate::models::{Organization, Region, Facility, Provider};

pub fn parse_organizations<P: AsRef<Path>>(path: P) -> Result<Vec<Organization>> {
    let mut reader = csv::Reader::from_path(&path)
        .with_context(|| format!("Failed to open organizations CSV file: {}", path.as_ref().display()))?;

    let mut organizations = Vec::new();
    for (idx, result) in reader.deserialize().enumerate() {
        let org: Organization = result
            .map_err(|e| anyhow::anyhow!("Failed to parse organization at row {}: {}", idx + 2, e))?;
        organizations.push(org);
    }

    Ok(organizations)
}

pub fn parse_regions<P: AsRef<Path>>(path: P) -> Result<Vec<Region>> {
    let mut reader = csv::Reader::from_path(&path)
        .with_context(|| format!("Failed to open regions CSV file: {}", path.as_ref().display()))?;

    let mut regions = Vec::new();
    for (idx, result) in reader.deserialize().enumerate() {
        let region: Region = result
            .map_err(|e| anyhow::anyhow!("Failed to parse region at row {}: {}", idx + 2, e))?;
        regions.push(region);
    }

    Ok(regions)
}

pub fn parse_facilities<P: AsRef<Path>>(path: P) -> Result<Vec<Facility>> {
    let mut reader = csv::Reader::from_path(&path)
        .with_context(|| format!("Failed to open facilities CSV file: {}", path.as_ref().display()))?;

    let mut facilities = Vec::new();
    for (idx, result) in reader.deserialize().enumerate() {
        let facility: Facility = result
            .map_err(|e| anyhow::anyhow!("Failed to parse facility at row {}: {}", idx + 2, e))?;
        facilities.push(facility);
    }

    Ok(facilities)
}

pub fn parse_providers<P: AsRef<Path>>(path: P) -> Result<Vec<Provider>> {
    let mut reader = csv::Reader::from_path(&path)
        .with_context(|| format!("Failed to open providers CSV file: {}", path.as_ref().display()))?;

    let mut providers = Vec::new();
    for (idx, result) in reader.deserialize().enumerate() {
        let provider: Provider = result
            .map_err(|e| anyhow::anyhow!("Failed to parse provider at row {}: {}", idx + 2, e))?;
        providers.push(provider);
    }

    Ok(providers)
}
