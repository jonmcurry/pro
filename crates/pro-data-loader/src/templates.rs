//! CSV template generation

use anyhow::{Context, Result};
use std::fs;
use std::path::Path;

pub fn generate_templates<P: AsRef<Path>>(output_dir: P) -> Result<()> {
    let dir = output_dir.as_ref();

    // Create output directory
    fs::create_dir_all(dir)
        .with_context(|| format!("Failed to create directory: {}", dir.display()))?;

    // Organizations template
    let orgs_path = dir.join("organizations.csv");
    fs::write(
        &orgs_path,
        "organization_code,organization_name,tax_id,contact_email,address_line1,address_line2,city,state,zip_code,active\n\
         ORG001,Example Health System,12-3456789,admin@example.health,123 Main St,,Springfield,IL,62701,true\n"
    )
    .with_context(|| format!("Failed to write: {}", orgs_path.display()))?;

    println!("  Created: {}", orgs_path.display());

    // Regions template
    let regions_path = dir.join("regions.csv");
    fs::write(
        &regions_path,
        "organization_code,region_code,region_name,manager_name,manager_email,active\n\
         ORG001,R1,North Region,John Smith,john.smith@example.health,true\n\
         ORG001,R2,South Region,Jane Doe,jane.doe@example.health,true\n"
    )
    .with_context(|| format!("Failed to write: {}", regions_path.display()))?;

    println!("  Created: {}", regions_path.display());

    // Facilities template
    let facilities_path = dir.join("facilities.csv");
    fs::write(
        &facilities_path,
        "organization_code,region_code,facility_code,facility_name,facility_npi,tax_id,address_line1,address_line2,city,state,zip_code,phone,ehr_system,active\n\
         ORG001,R1,F1,North Medical Center,1234567890,12-3456789,100 Hospital Dr,,Chicago,IL,60601,555-1234,Athena,true\n\
         ORG001,R1,F2,North Clinic,1234567891,12-3456789,200 Clinic Ave,,Chicago,IL,60602,555-1235,Athena,true\n"
    )
    .with_context(|| format!("Failed to write: {}", facilities_path.display()))?;

    println!("  Created: {}", facilities_path.display());

    // Providers template
    let providers_path = dir.join("providers.csv");
    fs::write(
        &providers_path,
        "facility_code,provider_npi,first_name,last_name,middle_name,credentials,specialty,taxonomy_code,email,phone,active\n\
         F1,1234567890,John,Smith,A,MD,Family Medicine,207Q00000X,jsmith@example.health,555-1000,true\n\
         F1,1234567891,Jane,Doe,,DO,Internal Medicine,207R00000X,jdoe@example.health,555-1001,true\n"
    )
    .with_context(|| format!("Failed to write: {}", providers_path.display()))?;

    println!("  Created: {}", providers_path.display());

    Ok(())
}
