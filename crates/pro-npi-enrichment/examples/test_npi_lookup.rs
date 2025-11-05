//! Example: Test NPI Registry API lookup
//!
//! This example demonstrates how to lookup provider information from the CMS NPI Registry API.
//!
//! Usage:
//!   cargo run --example test_npi_lookup <NPI>
//!
//! Example:
//!   cargo run --example test_npi_lookup 1234567890

use pro_npi_enrichment::NpiRegistryClient;
use std::env;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    // Get NPI from command line argument
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <NPI>", args[0]);
        eprintln!("\nExample: {} 1234567890", args[0]);
        eprintln!("\nTo find real NPIs to test:");
        eprintln!("  1. Visit https://npiregistry.cms.hhs.gov");
        eprintln!("  2. Search for a provider (e.g., 'John Smith' in your state)");
        eprintln!("  3. Copy the 10-digit NPI number");
        std::process::exit(1);
    }

    let npi = &args[1];

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║         CMS NPI Registry API Lookup Test                ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();
    println!("Looking up NPI: {}", npi);
    println!("API: https://npiregistry.cms.hhs.gov/api/");
    println!();

    // Create client
    let client = NpiRegistryClient::new()?;

    // Lookup NPI
    match client.lookup_npi(npi).await {
        Ok(response) => {
            println!("✓ SUCCESS - Found {} result(s)", response.result_count);
            println!();

            if let Some(provider) = response.results.first() {
                println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                println!("PROVIDER INFORMATION");
                println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                println!("  NPI: {}", provider.number);
                println!("  Type: {}", provider.enumeration_type);

                if provider.enumeration_type.contains("NPI-1") {
                    // Individual provider
                    if let Some(first) = &provider.basic.first_name {
                        print!("  Name: {}", first);
                        if let Some(middle) = &provider.basic.middle_name {
                            print!(" {}", middle);
                        }
                        if let Some(last) = &provider.basic.last_name {
                            print!(" {}", last);
                        }
                        if let Some(cred) = &provider.basic.credential {
                            print!(", {}", cred);
                        }
                        println!();
                    }
                    if let Some(gender) = &provider.basic.gender {
                        println!("  Gender: {}", gender);
                    }
                } else {
                    // Organization
                    if let Some(org_name) = &provider.basic.organization_name {
                        println!("  Organization: {}", org_name);
                    }
                }

                if let Some(status) = &provider.basic.status {
                    println!("  Status: {}", if status == "A" { "Active" } else { status });
                }
                if let Some(enum_date) = &provider.basic.enumeration_date {
                    println!("  Enumeration Date: {}", enum_date);
                }
                if let Some(updated) = &provider.basic.last_updated {
                    println!("  Last Updated: {}", updated);
                }

                println!();
                println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                println!("TAXONOMIES (SPECIALTIES)");
                println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

                if provider.taxonomies.is_empty() {
                    println!("  No taxonomies found");
                } else {
                    for (i, tax) in provider.taxonomies.iter().enumerate() {
                        println!("  {}. Code: {}", i + 1, tax.code);
                        println!("     Description: {}", tax.desc);
                        if let Some(group) = &tax.taxonomy_group {
                            println!("     Group: {}", group);
                        }
                        println!("     Primary: {}", if tax.primary { "Yes ⭐" } else { "No" });
                        if let Some(state) = &tax.state {
                            println!("     State: {}", state);
                        }
                        if let Some(license) = &tax.license {
                            println!("     License: {}", license);
                        }
                        println!();
                    }
                }

                println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                println!("ADDRESSES");
                println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

                if provider.addresses.is_empty() {
                    println!("  No addresses found");
                } else {
                    for (i, addr) in provider.addresses.iter().enumerate() {
                        println!("  {}. {} Address:", i + 1, addr.address_purpose);
                        if let Some(addr1) = &addr.address_1 {
                            println!("     {}", addr1);
                        }
                        if let Some(addr2) = &addr.address_2 {
                            println!("     {}", addr2);
                        }
                        print!("     ");
                        if let Some(city) = &addr.city {
                            print!("{}, ", city);
                        }
                        if let Some(state) = &addr.state {
                            print!("{} ", state);
                        }
                        if let Some(postal) = &addr.postal_code {
                            print!("{}", postal);
                        }
                        println!();
                        if let Some(country) = &addr.country_name {
                            println!("     {}", country);
                        }
                        if let Some(phone) = &addr.telephone_number {
                            println!("     Phone: {}", phone);
                        }
                        if let Some(fax) = &addr.fax_number {
                            println!("     Fax: {}", fax);
                        }
                        println!();
                    }
                }

                // Show taxonomy mapping instructions
                if let Some(primary_tax) = provider.taxonomies.iter().find(|t| t.primary) {
                    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                    println!("TAXONOMY LOOKUP");
                    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                    println!("  Primary Taxonomy Code: {}", primary_tax.code);
                    println!();
                    println!("  To lookup specialty display name in database:");
                    println!("    SELECT specialty_display");
                    println!("    FROM claims.provider_taxonomy");
                    println!("    WHERE taxonomy_code = '{}';", primary_tax.code);
                    println!();
                }

                println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                println!("NEXT STEPS");
                println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                println!("  1. Insert this provider into your database:");
                println!("     INSERT INTO claims.provider (npi, provider_type, last_name, first_name)");
                println!("     VALUES ('{}', 'Billing', '{}', '{}');",
                    provider.number,
                    provider.basic.last_name.as_deref().unwrap_or("Unknown"),
                    provider.basic.first_name.as_deref().unwrap_or("Unknown")
                );
                println!();
                println!("  2. Queue for enrichment:");
                println!("     INSERT INTO claims.provider_enrichment_queue");
                println!("     (provider_id, npi, priority)");
                println!("     VALUES (LAST_PROVIDER_ID, '{}', 10);", provider.number);
                println!();
                println!("  3. Wait 30-60 seconds for the enrichment worker to process");
                println!();
                println!("  4. Check results:");
                println!("     SELECT p.*, pt.specialty_display");
                println!("     FROM claims.provider p");
                println!("     LEFT JOIN claims.provider_taxonomy pt ON p.taxonomy_code = pt.taxonomy_code");
                println!("     WHERE p.npi = '{}';", provider.number);
                println!();
            }
        }
        Err(e) => {
            eprintln!("✗ ERROR: {}", e);
            eprintln!();
            eprintln!("Common issues:");
            eprintln!("  • NPI must be exactly 10 digits");
            eprintln!("  • NPI must exist in the CMS registry");
            eprintln!("  • Network connectivity issues");
            eprintln!("  • CMS API may be temporarily unavailable");
            std::process::exit(1);
        }
    }

    Ok(())
}
