//! Data validation

use anyhow::{anyhow, Result};
use std::collections::HashSet;

use crate::models::{Organization, Region, Facility, Provider};

pub fn validate_organizations(organizations: &[Organization]) -> Result<()> {
    if organizations.is_empty() {
        return Err(anyhow!("No organizations found in CSV file"));
    }

    let mut codes = HashSet::new();
    for (idx, org) in organizations.iter().enumerate() {
        // Check required fields
        if org.organization_code.trim().is_empty() {
            return Err(anyhow!("Organization code is empty at row {}", idx + 2));
        }
        if org.organization_name.trim().is_empty() {
            return Err(anyhow!("Organization name is empty at row {}", idx + 2));
        }

        // Check for duplicates
        if !codes.insert(org.organization_code.clone()) {
            return Err(anyhow!(
                "Duplicate organization code '{}' at row {}",
                org.organization_code,
                idx + 2
            ));
        }
    }

    Ok(())
}

pub fn validate_regions(regions: &[Region], organizations: &[Organization]) -> Result<()> {
    if regions.is_empty() {
        return Err(anyhow!("No regions found in CSV file"));
    }

    let org_codes: HashSet<String> = organizations
        .iter()
        .map(|o| o.organization_code.clone())
        .collect();

    let mut region_keys = HashSet::new();

    for (idx, region) in regions.iter().enumerate() {
        // Check required fields
        if region.organization_code.trim().is_empty() {
            return Err(anyhow!("Organization code is empty at row {}", idx + 2));
        }
        if region.region_code.trim().is_empty() {
            return Err(anyhow!("Region code is empty at row {}", idx + 2));
        }
        if region.region_name.trim().is_empty() {
            return Err(anyhow!("Region name is empty at row {}", idx + 2));
        }

        // Check referential integrity
        if !org_codes.contains(&region.organization_code) {
            return Err(anyhow!(
                "Region at row {} references unknown organization code '{}'",
                idx + 2,
                region.organization_code
            ));
        }

        // Check for duplicates (org_code + region_code must be unique)
        let key = format!("{}-{}", region.organization_code, region.region_code);
        if !region_keys.insert(key.clone()) {
            return Err(anyhow!(
                "Duplicate region '{}' at row {}",
                key,
                idx + 2
            ));
        }
    }

    Ok(())
}

pub fn validate_facilities(
    facilities: &[Facility],
    regions: &[Region],
    organizations: &[Organization],
) -> Result<()> {
    if facilities.is_empty() {
        return Err(anyhow!("No facilities found in CSV file"));
    }

    // Build map of valid organization codes
    let org_codes: HashSet<String> = organizations
        .iter()
        .map(|o| o.organization_code.clone())
        .collect();

    // Build map of valid region keys (only if regions are provided)
    let mut region_keys = HashSet::new();
    let validate_regions = !regions.is_empty();

    for region in regions {
        if org_codes.contains(&region.organization_code) {
            let key = format!("{}-{}", region.organization_code, region.region_code);
            region_keys.insert(key);
        }
    }

    let mut facility_codes = HashSet::new();

    for (idx, facility) in facilities.iter().enumerate() {
        // Check required fields
        if facility.organization_code.trim().is_empty() {
            return Err(anyhow!("Organization code is empty at row {}", idx + 2));
        }
        if facility.facility_code.trim().is_empty() {
            return Err(anyhow!("Facility code is empty at row {}", idx + 2));
        }
        if facility.facility_name.trim().is_empty() {
            return Err(anyhow!("Facility name is empty at row {}", idx + 2));
        }

        // Check organization exists
        if !org_codes.contains(&facility.organization_code) {
            return Err(anyhow!(
                "Facility at row {} references unknown organization code '{}'",
                idx + 2,
                facility.organization_code
            ));
        }

        // Check region referential integrity only if regions are being validated
        if validate_regions {
            if facility.region_code.trim().is_empty() {
                return Err(anyhow!("Region code is empty at row {} (regions are being imported, so region_code is required)", idx + 2));
            }

            let region_key = format!("{}-{}", facility.organization_code, facility.region_code);
            if !region_keys.contains(&region_key) {
                return Err(anyhow!(
                    "Facility at row {} references unknown region '{}' for organization '{}'",
                    idx + 2,
                    facility.region_code,
                    facility.organization_code
                ));
            }
        }
        // If regions are not being validated, region_code can be empty or ignored

        // Check for duplicate facility codes
        if !facility_codes.insert(facility.facility_code.clone()) {
            return Err(anyhow!(
                "Duplicate facility code '{}' at row {}",
                facility.facility_code,
                idx + 2
            ));
        }
    }

    Ok(())
}

pub fn validate_providers(providers: &[Provider], facilities: &[Facility]) -> Result<()> {
    if providers.is_empty() {
        return Err(anyhow!("No providers found in CSV file"));
    }

    let facility_codes: HashSet<String> = facilities
        .iter()
        .map(|f| f.facility_code.clone())
        .collect();

    let mut npi_codes = HashSet::new();

    for (idx, provider) in providers.iter().enumerate() {
        // Check required fields
        if provider.facility_code.trim().is_empty() {
            return Err(anyhow!("Facility code is empty at row {}", idx + 2));
        }
        if provider.provider_npi.trim().is_empty() {
            return Err(anyhow!("Provider NPI is empty at row {}", idx + 2));
        }
        if provider.first_name.trim().is_empty() {
            return Err(anyhow!("Provider first name is empty at row {}", idx + 2));
        }
        if provider.last_name.trim().is_empty() {
            return Err(anyhow!("Provider last name is empty at row {}", idx + 2));
        }

        // Validate NPI format (should be 10 digits)
        if provider.provider_npi.len() != 10 || !provider.provider_npi.chars().all(|c| c.is_numeric()) {
            return Err(anyhow!(
                "Invalid NPI '{}' at row {} (must be 10 digits)",
                provider.provider_npi,
                idx + 2
            ));
        }

        // Check referential integrity
        if !facility_codes.contains(&provider.facility_code) {
            return Err(anyhow!(
                "Provider at row {} references unknown facility code '{}'",
                idx + 2,
                provider.facility_code
            ));
        }

        // Check for duplicate NPIs
        if !npi_codes.insert(provider.provider_npi.clone()) {
            return Err(anyhow!(
                "Duplicate provider NPI '{}' at row {}",
                provider.provider_npi,
                idx + 2
            ));
        }
    }

    Ok(())
}
