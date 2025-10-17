//! Professional SMART Master Data Loader
//!
//! Standalone utility for loading master data (organizations, regions, facilities, providers)
//! into the Professional SMART database from CSV files.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use colored::*;
use std::path::PathBuf;

use pro_data_loader::{config, csv_parser, importer, templates, validator, Config};

#[derive(Parser)]
#[command(name = "pro-data-loader")]
#[command(about = "Professional SMART Master Data Loader", long_about = None)]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Path to organizations CSV file
    #[arg(long)]
    organizations: Option<PathBuf>,

    /// Path to regions CSV file
    #[arg(long)]
    regions: Option<PathBuf>,

    /// Path to facilities CSV file
    #[arg(long)]
    facilities: Option<PathBuf>,

    /// Path to providers CSV file
    #[arg(long)]
    providers: Option<PathBuf>,

    /// Directory containing all CSV files (looks for organizations.csv, regions.csv, etc.)
    #[arg(long)]
    csv_dir: Option<PathBuf>,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate CSV template files
    GenerateTemplates {
        /// Output directory for templates
        #[arg(default_value = ".")]
        output_dir: PathBuf,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Print banner
    print_banner();

    // Handle subcommands
    if let Some(command) = cli.command {
        match command {
            Commands::GenerateTemplates { output_dir } => {
                println!("\n{}", "Generating CSV templates...".cyan().bold());
                templates::generate_templates(&output_dir)?;
                println!("\n{}", "Templates generated successfully!".green().bold());
                println!("\nEdit the generated CSV files and run:");
                println!("  {} --csv-dir {}", "pro-data-loader".yellow(), output_dir.display());
                return Ok(());
            }
        }
    }

    // Load configuration
    println!("{}", "Loading configuration...".cyan());
    let config = Config::load()?;
    println!("  Database: {}\n", config.masked_url());

    // Determine CSV file paths
    let (orgs_path, regions_path, facilities_path, providers_path) =
        if let Some(csv_dir) = cli.csv_dir {
            // Load from directory
            (
                csv_dir.join("organizations.csv"),
                csv_dir.join("regions.csv"),
                csv_dir.join("facilities.csv"),
                csv_dir.join("providers.csv"),
            )
        } else if cli.organizations.is_some()
            || cli.regions.is_some()
            || cli.facilities.is_some()
            || cli.providers.is_some()
        {
            // Load from individual files
            (
                cli.organizations
                    .context("--organizations is required when not using --csv-dir")?,
                cli.regions
                    .context("--regions is required when not using --csv-dir")?,
                cli.facilities
                    .context("--facilities is required when not using --csv-dir")?,
                cli.providers
                    .context("--providers is required when not using --csv-dir")?,
            )
        } else {
            // Default: look in current directory
            println!("{}", "No CSV files specified, looking in current directory...".yellow());
            (
                PathBuf::from("organizations.csv"),
                PathBuf::from("regions.csv"),
                PathBuf::from("facilities.csv"),
                PathBuf::from("providers.csv"),
            )
        };

    // Verify files exist
    println!("{}", "Checking CSV files...".cyan());
    verify_file_exists(&orgs_path)?;
    verify_file_exists(&regions_path)?;
    verify_file_exists(&facilities_path)?;
    verify_file_exists(&providers_path)?;
    println!("  All CSV files found\n");

    // Parse CSV files
    println!("{}", "Parsing CSV files...".cyan());
    let organizations = csv_parser::parse_organizations(&orgs_path)?;
    println!("  Organizations: {} records", organizations.len());

    let regions = csv_parser::parse_regions(&regions_path)?;
    println!("  Regions:       {} records", regions.len());

    let facilities = csv_parser::parse_facilities(&facilities_path)?;
    println!("  Facilities:    {} records", facilities.len());

    let providers = csv_parser::parse_providers(&providers_path)?;
    println!("  Providers:     {} records\n", providers.len());

    // Validate data
    println!("{}", "Validating data...".cyan());
    validator::validate_organizations(&organizations)?;
    println!("  Organizations: {}", "OK".green());

    validator::validate_regions(&regions, &organizations)?;
    println!("  Regions:       {}", "OK".green());

    validator::validate_facilities(&facilities, &regions, &organizations)?;
    println!("  Facilities:    {}", "OK".green());

    validator::validate_providers(&providers, &facilities)?;
    println!("  Providers:     {}", "OK".green());

    // Confirm import
    println!("\n{}", "Ready to import data".yellow().bold());
    println!("  Organizations: {}", organizations.len());
    println!("  Regions:       {}", regions.len());
    println!("  Facilities:    {}", facilities.len());
    println!("  Providers:     {}", providers.len());
    println!("  Total:         {}", organizations.len() + regions.len() + facilities.len() + providers.len());
    println!("\n{}", "NOTE: Existing records will be updated (upsert)".yellow());

    print!("\n{} ", "Proceed with import? (yes/no):".cyan().bold());
    use std::io::{self, Write};
    io::stdout().flush()?;

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;

    if !input.trim().eq_ignore_ascii_case("yes") {
        println!("\n{}", "Import cancelled".yellow());
        return Ok(());
    }

    // Import data
    println!("\n{}", "Importing data to database...".cyan().bold());
    let results = importer::import_all(
        &config.database_url,
        organizations,
        regions,
        facilities,
        providers,
    )
    .await?;

    // Print results
    println!();
    if results.has_errors() {
        println!("{}", "Import completed with errors:".red().bold());
        for error in &results.errors {
            println!("  {} {}", "ERROR:".red(), error);
        }
    } else {
        println!("{}", "Import completed successfully!".green().bold());
        println!("\n{}", "Summary:".cyan().bold());
        println!("  Organizations: {}", results.organizations_inserted);
        println!("  Regions:       {}", results.regions_inserted);
        println!("  Facilities:    {}", results.facilities_inserted);
        println!("  Providers:     {}", results.providers_inserted);
        println!("  {}: {}", "Total".bold(), results.total_inserted());
    }

    Ok(())
}

fn print_banner() {
    println!("{}", "=".repeat(70).cyan());
    println!("{}", "Professional SMART - Master Data Loader".cyan().bold());
    println!("{}", "=".repeat(70).cyan());
}

fn verify_file_exists(path: &PathBuf) -> Result<()> {
    if !path.exists() {
        anyhow::bail!("File not found: {}", path.display());
    }
    println!("  Found: {}", path.display());
    Ok(())
}
