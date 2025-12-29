//! Database import logic

use anyhow::{Context, Result};
use sqlx::{PgPool, Postgres, Row, Transaction};
use std::collections::HashMap;

use crate::models::*;

pub async fn import_all(
    database_url: &str,
    organizations: Vec<Organization>,
    regions: Vec<Region>,
    facilities: Vec<Facility>,
    providers: Vec<Provider>,
) -> Result<ImportResults> {
    import_all_with_progress(database_url, organizations, regions, facilities, providers, |msg| {
        println!("{}", msg);
    })
    .await
}

pub async fn import_all_with_progress<F>(
    database_url: &str,
    organizations: Vec<Organization>,
    regions: Vec<Region>,
    facilities: Vec<Facility>,
    providers: Vec<Provider>,
    mut progress_callback: F,
) -> Result<ImportResults>
where
    F: FnMut(String),
{
    // Connect to database
    progress_callback("Connecting to database...".to_string());
    let pool = PgPool::connect(database_url)
        .await
        .context("Failed to connect to database")?;

    // Begin transaction
    progress_callback("Starting transaction...".to_string());
    let mut tx = pool.begin().await.context("Failed to begin transaction")?;

    let mut results = ImportResults::default();

    // Import in dependency order
    progress_callback(format!("Importing {} organizations...", organizations.len()));
    let org_map = match import_organizations(&mut tx, &organizations).await {
        Ok(map) => {
            results.organizations_inserted = map.len();
            progress_callback(format!("Inserted {} organizations", map.len()));
            map
        }
        Err(e) => {
            results.errors.push(format!("Failed to import organizations: {}", e));
            tx.rollback().await.ok();
            return Ok(results);
        }
    };

    if !regions.is_empty() {
        progress_callback(format!("Importing {} regions...", regions.len()));
        let region_map = match import_regions(&mut tx, &regions, &org_map).await {
            Ok(map) => {
                results.regions_inserted = map.len();
                progress_callback(format!("Inserted {} regions", map.len()));
                map
            }
            Err(e) => {
                results.errors.push(format!("Failed to import regions: {}", e));
                tx.rollback().await.ok();
                return Ok(results);
            }
        };

        progress_callback(format!("Importing {} facilities...", facilities.len()));
        let facility_map = match import_facilities(&mut tx, &facilities, &org_map, &region_map).await {
            Ok(map) => {
                results.facilities_inserted = map.len();
                progress_callback(format!("Inserted {} facilities", map.len()));
                map
            }
            Err(e) => {
                results.errors.push(format!("Failed to import facilities: {}", e));
                tx.rollback().await.ok();
                return Ok(results);
            }
        };

        if !providers.is_empty() {
            progress_callback(format!("Importing {} providers...", providers.len()));
            match import_providers(&mut tx, &providers, &facility_map).await {
                Ok(count) => {
                    results.providers_inserted = count;
                    progress_callback(format!("Inserted {} providers", count));
                }
                Err(e) => {
                    results.errors.push(format!("Failed to import providers: {}", e));
                    tx.rollback().await.ok();
                    return Ok(results);
                }
            }
        }
    } else {
        // No regions were provided, import facilities without region_map
        progress_callback(format!("Importing {} facilities (no regions)...", facilities.len()));
        let facility_map = match import_facilities(&mut tx, &facilities, &org_map, &std::collections::HashMap::new()).await {
            Ok(map) => {
                results.facilities_inserted = map.len();
                progress_callback(format!("Inserted {} facilities", map.len()));
                map
            }
            Err(e) => {
                results.errors.push(format!("Failed to import facilities: {}", e));
                tx.rollback().await.ok();
                return Ok(results);
            }
        };

        if !providers.is_empty() {
            progress_callback(format!("Importing {} providers...", providers.len()));
            match import_providers(&mut tx, &providers, &facility_map).await {
                Ok(count) => {
                    results.providers_inserted = count;
                    progress_callback(format!("Inserted {} providers", count));
                }
                Err(e) => {
                    results.errors.push(format!("Failed to import providers: {}", e));
                    tx.rollback().await.ok();
                    return Ok(results);
                }
            }
        }
    }

    // Commit transaction
    progress_callback("Committing transaction...".to_string());
    tx.commit()
        .await
        .context("Failed to commit transaction")?;

    progress_callback("Import completed successfully!".to_string());

    Ok(results)
}

async fn import_organizations(
    tx: &mut Transaction<'_, Postgres>,
    organizations: &[Organization],
) -> Result<HashMap<String, OrganizationDb>> {
    let mut map = HashMap::new();

    for org in organizations {
        let row = sqlx::query(
            r#"
            INSERT INTO claims.organization (
                organization_code, organization_name, tax_id, email,
                address_line1, address_line2, city, state_code, postal_code, is_active
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            ON CONFLICT (organization_code) DO UPDATE
            SET organization_name = EXCLUDED.organization_name,
                tax_id = EXCLUDED.tax_id,
                email = EXCLUDED.email,
                address_line1 = EXCLUDED.address_line1,
                address_line2 = EXCLUDED.address_line2,
                city = EXCLUDED.city,
                state_code = EXCLUDED.state_code,
                postal_code = EXCLUDED.postal_code,
                is_active = EXCLUDED.is_active,
                updated_at = CURRENT_TIMESTAMP
            RETURNING organization_id, organization_code
            "#,
        )
        .bind(&org.organization_code)
        .bind(&org.organization_name)
        .bind(&org.tax_id)
        .bind(&org.contact_email)
        .bind(&org.address_line1)
        .bind(&org.address_line2)
        .bind(&org.city)
        .bind(&org.state)
        .bind(&org.zip_code)
        .bind(org.active)
        .fetch_one(&mut **tx)
        .await?;

        map.insert(
            org.organization_code.clone(),
            OrganizationDb {
                organization_id: row.get("organization_id"),
                organization_code: row.get("organization_code"),
            },
        );
    }

    Ok(map)
}

async fn import_regions(
    tx: &mut Transaction<'_, Postgres>,
    regions: &[Region],
    org_map: &HashMap<String, OrganizationDb>,
) -> Result<HashMap<String, RegionDb>> {
    let mut map = HashMap::new();

    for region in regions {
        let org = org_map
            .get(&region.organization_code)
            .context(format!("Organization '{}' not found in map", region.organization_code))?;

        let row = sqlx::query(
            r#"
            INSERT INTO claims.region (
                organization_id, region_code, region_name, is_active
            )
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (organization_id, region_code) DO UPDATE
            SET region_name = EXCLUDED.region_name,
                is_active = EXCLUDED.is_active,
                updated_at = CURRENT_TIMESTAMP
            RETURNING region_id, organization_id, region_code
            "#,
        )
        .bind(org.organization_id)
        .bind(&region.region_code)
        .bind(&region.region_name)
        .bind(region.active)
        .fetch_one(&mut **tx)
        .await?;

        let key = format!("{}-{}", region.organization_code, region.region_code);
        map.insert(
            key,
            RegionDb {
                region_id: row.get("region_id"),
                organization_id: row.get("organization_id"),
                region_code: row.get("region_code"),
            },
        );
    }

    Ok(map)
}

async fn import_facilities(
    tx: &mut Transaction<'_, Postgres>,
    facilities: &[Facility],
    org_map: &HashMap<String, OrganizationDb>,
    region_map: &HashMap<String, RegionDb>,
) -> Result<HashMap<String, FacilityDb>> {
    let mut map = HashMap::new();

    for facility in facilities {
        // Get organization_id (required)
        let org = org_map
            .get(&facility.organization_code)
            .context(format!("Organization '{}' not found in map", facility.organization_code))?;

        // Try to find region if region_map is not empty
        let region_id = if !region_map.is_empty() {
            let region_key = format!("{}-{}", facility.organization_code, facility.region_code);
            let region = region_map
                .get(&region_key)
                .context(format!("Region '{}' not found in map", region_key))?;
            Some(region.region_id)
        } else {
            // If no regions were imported, region_id will be NULL
            None
        };

        let row = sqlx::query(
            r#"
            INSERT INTO claims.facility (
                organization_id, region_id, facility_code, facility_name, npi, tax_id,
                address_line1, address_line2, city, state_code, postal_code,
                phone, ehr_system, is_active
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
            ON CONFLICT (organization_id, facility_code) DO UPDATE
            SET region_id = EXCLUDED.region_id,
                facility_name = EXCLUDED.facility_name,
                npi = EXCLUDED.npi,
                tax_id = EXCLUDED.tax_id,
                address_line1 = EXCLUDED.address_line1,
                address_line2 = EXCLUDED.address_line2,
                city = EXCLUDED.city,
                state_code = EXCLUDED.state_code,
                postal_code = EXCLUDED.postal_code,
                phone = EXCLUDED.phone,
                ehr_system = EXCLUDED.ehr_system,
                is_active = EXCLUDED.is_active,
                updated_at = CURRENT_TIMESTAMP
            RETURNING facility_id, region_id, facility_code
            "#,
        )
        .bind(org.organization_id)
        .bind(region_id)
        .bind(&facility.facility_code)
        .bind(&facility.facility_name)
        .bind(&facility.facility_npi)
        .bind(&facility.tax_id)
        .bind(&facility.address_line1)
        .bind(&facility.address_line2)
        .bind(&facility.city)
        .bind(&facility.state)
        .bind(&facility.zip_code)
        .bind(&facility.phone)
        .bind(&facility.ehr_system)
        .bind(facility.active)
        .fetch_one(&mut **tx)
        .await?;

        map.insert(
            facility.facility_code.clone(),
            FacilityDb {
                facility_id: row.get("facility_id"),
                region_id: row.get("region_id"),
                facility_code: row.get("facility_code"),
            },
        );
    }

    Ok(map)
}

async fn import_providers(
    tx: &mut Transaction<'_, Postgres>,
    providers: &[Provider],
    facility_map: &HashMap<String, FacilityDb>,
) -> Result<usize> {
    let mut count = 0;

    for provider in providers {
        // Validate facility exists (required for provider import)
        let _facility = facility_map
            .get(&provider.facility_code)
            .context(format!("Facility '{}' not found in map", provider.facility_code))?;

        // Parse full_name if provided and first_name/last_name are empty
        let (first_name, last_name) = if provider.first_name.is_empty() || provider.last_name.is_empty() {
            if let Some(full_name) = &provider.full_name {
                parse_full_name(full_name)
            } else {
                (provider.first_name.clone(), provider.last_name.clone())
            }
        } else {
            (provider.first_name.clone(), provider.last_name.clone())
        };

        sqlx::query(
            r#"
            INSERT INTO claims.provider (
                npi, provider_type, first_name, last_name, middle_name, full_name,
                specialty, taxonomy_code, email, is_active
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            ON CONFLICT (npi) DO UPDATE
            SET provider_type = EXCLUDED.provider_type,
                first_name = EXCLUDED.first_name,
                last_name = EXCLUDED.last_name,
                middle_name = EXCLUDED.middle_name,
                full_name = EXCLUDED.full_name,
                specialty = EXCLUDED.specialty,
                taxonomy_code = EXCLUDED.taxonomy_code,
                email = EXCLUDED.email,
                is_active = EXCLUDED.is_active,
                updated_at = CURRENT_TIMESTAMP
            "#,
        )
        .bind(&provider.provider_npi)
        .bind("Rendering") // Default provider_type since it's required
        .bind(&first_name)
        .bind(&last_name)
        .bind(&provider.middle_name)
        .bind(&provider.full_name)
        .bind(&provider.specialty)
        .bind(&provider.taxonomy_code)
        .bind(&provider.email)
        .bind(provider.active)
        .execute(&mut **tx)
        .await?;

        count += 1;
    }

    Ok(count)
}

/// Parse a full name into first and last name components
/// Simple parser that handles common formats:
/// - "First Last"
/// - "Last, First"
/// - "First Middle Last" (treats everything before last word as first name)
fn parse_full_name(full_name: &str) -> (String, String) {
    let trimmed = full_name.trim();

    // Handle "Last, First" format
    if let Some(comma_pos) = trimmed.find(',') {
        let last = trimmed[..comma_pos].trim();
        let first = trimmed[comma_pos + 1..].trim();
        return (first.to_string(), last.to_string());
    }

    // Handle "First Last" or "First Middle Last" format
    let parts: Vec<&str> = trimmed.split_whitespace().collect();
    match parts.len() {
        0 => (String::new(), String::new()),
        1 => (parts[0].to_string(), String::new()),
        _ => {
            // Everything except last word is first name
            let first = parts[..parts.len() - 1].join(" ");
            let last = parts[parts.len() - 1].to_string();
            (first, last)
        }
    }
}
