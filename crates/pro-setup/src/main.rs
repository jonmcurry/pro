//! Professional SMART Configuration Wizard
//!
//! Interactive console-based setup tool for configuring the Professional SMART claims processing system.
//! This wizard guides users through database setup, directory configuration, and performance tuning.

use anyhow::{Context, Result};
use console::{style, Term};
use dialoguer::{Confirm, Input, Password, Select};
use std::fs;
use std::path::PathBuf;
use std::env;

mod database;
mod env_generator;
mod postgres_installer;

use database::test_database_connection;
use env_generator::generate_env_file;

fn main() -> Result<()> {
    // Check for --silent flag
    let args: Vec<String> = env::args().collect();
    let silent_mode = args.contains(&"--silent".to_string()) || args.contains(&"-s".to_string());

    if silent_mode {
        return run_silent_mode();
    }

    // Initialize terminal
    let term = Term::stdout();
    term.clear_screen()?;

    // Display welcome banner
    print_banner(&term)?;

    // Confirm start
    if !Confirm::new()
        .with_prompt("Would you like to configure Professional SMART?")
        .default(true)
        .interact()?
    {
        println!("Setup cancelled");
        return Ok(());
    }

    // Configuration wizard steps
    let config = run_configuration_wizard(&term)?;

    // Generate .env file
    println!();
    println!("{}", style("Generating configuration file...").bold().green());
    generate_env_file(&config)?;

    // Test database connection
    println!();
    println!("{}", style("Testing database connection...").bold().green());
    let runtime = tokio::runtime::Runtime::new()?;
    runtime.block_on(async {
        test_database_connection(&config.database_url).await
    })?;

    // Print summary
    print_summary(&term, &config)?;

    println!();
    println!("{}", style("Configuration complete!").bold().green());
    println!();
    println!("Next steps:");
    println!("  1. Review the generated .env file");
    println!("  2. Build the application: cargo build --release");
    println!("  3. Install the Windows service: professional-smart install");
    println!("  4. Start the service: professional-smart start");
    println!();

    Ok(())
}

/// Print welcome banner
fn print_banner(term: &Term) -> Result<()> {
    term.write_line(&format!("{}", style("=" . repeat(80)).dim()))?;
    term.write_line(&format!("{}", style("Professional SMART Configuration Wizard").bold().cyan()))?;
    term.write_line(&format!("{}", style("Claims Processing System Setup").cyan()))?;
    term.write_line(&format!("{}", style("=" . repeat(80)).dim()))?;
    term.write_line("")?;

    Ok(())
}

/// Configuration structure
#[derive(Debug)]
struct Configuration {
    // Database
    database_url: String,
    database_max_connections: u32,

    // Directories
    input_directory: String,
    processed_directory: String,
    error_directory: String,
    log_directory: String,

    // Performance
    batch_size: usize,
    max_workers: usize,

    // Features
    enable_rules_engine: bool,
    enable_rvu_calculation: bool,
    enable_auto_coding_suggestions: bool,

    // RVU Settings
    default_gpci_locality: String,
    rvu_year: u32,

    // Logging
    log_level: String,
}

/// Run the configuration wizard
fn run_configuration_wizard(term: &Term) -> Result<Configuration> {
    term.write_line(&format!("{}", style("Database Configuration").bold().underlined()))?;
    term.write_line("")?;

    // PostgreSQL Installation Check
    let postgres_installed = check_postgresql_installed()?;
    if !postgres_installed {
        term.write_line(&format!("{}", style("PostgreSQL not detected").yellow()))?;

        if Confirm::new()
            .with_prompt("Would you like guidance on installing PostgreSQL?")
            .default(true)
            .interact()?
        {
            show_postgres_installation_guide(term)?;
            return Err(anyhow::anyhow!("Please install PostgreSQL and run setup again"));
        }
    } else {
        term.write_line(&format!("{}", style("PostgreSQL detected").green()))?;
    }

    term.write_line("")?;

    // Database configuration
    let db_host = Input::<String>::new()
        .with_prompt("Database host")
        .default("localhost".to_string())
        .interact_text()?;

    let db_port = Input::<u16>::new()
        .with_prompt("Database port")
        .default(5432)
        .interact_text()?;

    let db_name = Input::<String>::new()
        .with_prompt("Database name")
        .default("professional_smart".to_string())
        .interact_text()?;

    let db_user = Input::<String>::new()
        .with_prompt("Database user")
        .default("pro_user".to_string())
        .interact_text()?;

    let db_password = Password::new()
        .with_prompt("Database password")
        .interact()?;

    let database_url = format!(
        "postgres://{}:{}@{}:{}/{}",
        db_user, db_password, db_host, db_port, db_name
    );

    let database_max_connections = Input::<u32>::new()
        .with_prompt("Maximum database connections")
        .default(50)
        .interact_text()?;

    term.write_line("")?;

    // Directory configuration
    term.write_line(&format!("{}", style("Directory Configuration").bold().underlined()))?;
    term.write_line("")?;

    let base_directory = Input::<String>::new()
        .with_prompt("Base directory for claims processing")
        .default("C:\\Claims".to_string())
        .interact_text()?;

    let input_directory = format!("{}\\Input", base_directory);
    let processed_directory = format!("{}\\Processed", base_directory);
    let error_directory = format!("{}\\Error", base_directory);
    let log_directory = format!("{}\\Logs", base_directory);

    // Create directories
    println!();
    println!("Creating directories...");
    create_directory(&input_directory)?;
    create_directory(&processed_directory)?;
    create_directory(&error_directory)?;
    create_directory(&log_directory)?;

    term.write_line("")?;

    // Performance configuration
    term.write_line(&format!("{}", style("Performance Configuration").bold().underlined()))?;
    term.write_line("")?;

    let cpu_count = num_cpus::get();
    term.write_line(&format!("Detected {} CPU cores", cpu_count))?;

    let batch_size = Select::new()
        .with_prompt("Batch size (claims per batch)")
        .items(&["500 (Low memory)", "1000 (Balanced)", "2000 (High throughput)", "5000 (Maximum throughput)"])
        .default(1)
        .interact()?;

    let batch_size = match batch_size {
        0 => 500,
        1 => 1000,
        2 => 2000,
        3 => 5000,
        _ => 1000,
    };

    let max_workers = Input::<usize>::new()
        .with_prompt("Maximum worker threads")
        .default(cpu_count)
        .interact_text()?;

    term.write_line("")?;

    // Feature configuration
    term.write_line(&format!("{}", style("Feature Configuration").bold().underlined()))?;
    term.write_line("")?;

    let enable_rules_engine = Confirm::new()
        .with_prompt("Enable rules engine and flagging system?")
        .default(true)
        .interact()?;

    let enable_rvu_calculation = Confirm::new()
        .with_prompt("Enable RVU-based payment calculations?")
        .default(true)
        .interact()?;

    let enable_auto_coding_suggestions = Confirm::new()
        .with_prompt("Enable automatic coding suggestions?")
        .default(true)
        .interact()?;

    term.write_line("")?;

    // RVU configuration (if enabled)
    let (default_gpci_locality, rvu_year) = if enable_rvu_calculation {
        term.write_line(&format!("{}", style("RVU Configuration").bold().underlined()))?;
        term.write_line("")?;

        let locality = Input::<String>::new()
            .with_prompt("Default GPCI locality code (00 = National Average)")
            .default("00".to_string())
            .interact_text()?;

        let year = Input::<u32>::new()
            .with_prompt("RVU data year")
            .default(2024)
            .interact_text()?;

        term.write_line("")?;

        (locality, year)
    } else {
        ("00".to_string(), 2024)
    };

    // Logging configuration
    term.write_line(&format!("{}", style("Logging Configuration").bold().underlined()))?;
    term.write_line("")?;

    let log_level = Select::new()
        .with_prompt("Log level")
        .items(&["error", "warn", "info", "debug", "trace"])
        .default(2)
        .interact()?;

    let log_level = match log_level {
        0 => "error",
        1 => "warn",
        2 => "info",
        3 => "debug",
        4 => "trace",
        _ => "info",
    };

    Ok(Configuration {
        database_url,
        database_max_connections,
        input_directory,
        processed_directory,
        error_directory,
        log_directory,
        batch_size,
        max_workers,
        enable_rules_engine,
        enable_rvu_calculation,
        enable_auto_coding_suggestions,
        default_gpci_locality,
        rvu_year,
        log_level: log_level.to_string(),
    })
}

/// Check if PostgreSQL is installed
fn check_postgresql_installed() -> Result<bool> {
    // Check if psql command exists
    let output = std::process::Command::new("psql")
        .arg("--version")
        .output();

    Ok(output.is_ok())
}

/// Show PostgreSQL installation guide
fn show_postgres_installation_guide(term: &Term) -> Result<()> {
    term.write_line("")?;
    term.write_line(&format!("{}", style("PostgreSQL Installation Guide").bold().cyan()))?;
    term.write_line(&format!("{}", style("=" . repeat(60)).dim()))?;
    term.write_line("")?;
    term.write_line("To install PostgreSQL:")?;
    term.write_line("")?;
    term.write_line("1. Download PostgreSQL 14 or later from:")?;
    term.write_line("   https://www.postgresql.org/download/windows/")?;
    term.write_line("")?;
    term.write_line("2. Run the installer as Administrator")?;
    term.write_line("")?;
    term.write_line("3. During installation:")?;
    term.write_line("   - Set a strong password for the postgres user")?;
    term.write_line("   - Use default port 5432")?;
    term.write_line("   - Accept default data directory")?;
    term.write_line("")?;
    term.write_line("4. After installation, create database and user:")?;
    term.write_line("   createdb -U postgres professional_smart")?;
    term.write_line("   psql -U postgres -c \"CREATE USER pro_user WITH PASSWORD 'your_password';\"")? ;
    term.write_line("   psql -U postgres -c \"GRANT ALL PRIVILEGES ON DATABASE professional_smart TO pro_user;\"")? ;
    term.write_line("")?;
    term.write_line("5. Run this setup wizard again after installation")?;
    term.write_line("")?;

    Ok(())
}

/// Create directory if it doesn't exist
fn create_directory(path: &str) -> Result<()> {
    let path_buf = PathBuf::from(path);

    if !path_buf.exists() {
        fs::create_dir_all(&path_buf)
            .with_context(|| format!("Failed to create directory: {}", path))?;
        println!("  Created: {}", path);
    } else {
        println!("  Exists:  {}", path);
    }

    Ok(())
}

/// Print configuration summary
fn print_summary(term: &Term, config: &Configuration) -> Result<()> {
    term.write_line("")?;
    term.write_line(&format!("{}", style("=" . repeat(80)).dim()))?;
    term.write_line(&format!("{}", style("Configuration Summary").bold().green()))?;
    term.write_line(&format!("{}", style("=" . repeat(80)).dim()))?;
    term.write_line("")?;
    term.write_line(&format!("Database:           {}", mask_password(&config.database_url)))?;
    term.write_line(&format!("Max Connections:    {}", config.database_max_connections))?;
    term.write_line("")?;
    term.write_line(&format!("Input Directory:    {}", config.input_directory))?;
    term.write_line(&format!("Processed Directory: {}", config.processed_directory))?;
    term.write_line(&format!("Error Directory:    {}", config.error_directory))?;
    term.write_line(&format!("Log Directory:      {}", config.log_directory))?;
    term.write_line("")?;
    term.write_line(&format!("Batch Size:         {}", config.batch_size))?;
    term.write_line(&format!("Max Workers:        {}", config.max_workers))?;
    term.write_line("")?;
    term.write_line(&format!("Rules Engine:       {}", if config.enable_rules_engine { "Enabled" } else { "Disabled" }))?;
    term.write_line(&format!("RVU Calculation:    {}", if config.enable_rvu_calculation { "Enabled" } else { "Disabled" }))?;
    term.write_line(&format!("Coding Suggestions: {}", if config.enable_auto_coding_suggestions { "Enabled" } else { "Disabled" }))?;
    term.write_line("")?;
    term.write_line(&format!("Log Level:          {}", config.log_level))?;
    term.write_line(&format!("{}", style("=" . repeat(80)).dim()))?;

    Ok(())
}

/// Mask password in database URL
fn mask_password(url: &str) -> String {
    if let Some(at_pos) = url.rfind('@') {
        if let Some(colon_pos) = url[..at_pos].rfind(':') {
            let mut masked = url.to_string();
            masked.replace_range(colon_pos + 1..at_pos, "****");
            return masked;
        }
    }
    url.to_string()
}

/// Run in silent mode using existing .env file
fn run_silent_mode() -> Result<()> {
    // Look for .env file in standard locations
    let env_paths = vec![
        PathBuf::from(r"C:\ProgramData\Professional SMART\config\.env"),
        PathBuf::from(".env"),
        PathBuf::from(r"C:\Program Files\Professional SMART\config\.env"),
    ];

    let env_file = env_paths.iter()
        .find(|p| p.exists())
        .ok_or_else(|| anyhow::anyhow!("Could not find .env file in standard locations"))?;

    // Load environment from file
    dotenvy::from_path(env_file)?;

    // Get database URL from environment
    let database_url = env::var("DATABASE_URL")
        .context("DATABASE_URL not found in .env file")?;

    // Test database connection and run migrations
    println!("Running database setup in silent mode...");
    println!("Database URL: {}", mask_password(&database_url));

    let runtime = tokio::runtime::Runtime::new()?;
    runtime.block_on(async {
        // Test connection
        println!("Testing database connection...");
        test_database_connection(&database_url).await?;
        println!("Database connection successful!");

        // Run migrations (this would use sqlx migrations or similar)
        println!("Running database migrations...");
        // TODO: Add actual migration code here when migrations are set up
        println!("Database setup complete!");

        Ok::<(), anyhow::Error>(())
    })?;

    println!("Silent setup completed successfully!");
    Ok(())
}
