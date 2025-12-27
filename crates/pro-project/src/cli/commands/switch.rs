use crate::cli::Cli;
use crate::services::{ConfigService, DatabaseService, RegistryService, WindowsServiceManager};
use anyhow::{bail, Result};

pub async fn execute(cli: &Cli, name: &str, no_restart: bool) -> Result<()> {
    let password = cli.db_password.as_deref().unwrap_or("");

    println!("Switching project database...");
    println!();

    // Check if target database exists
    let db_service = DatabaseService::new(&cli.db_host, cli.db_port, &cli.db_user, password);
    if !db_service.database_exists(name).await? {
        // Try to list available projects
        let registry = RegistryService::connect(&cli.db_host, cli.db_port, &cli.db_user, password).await;
        if let Ok(reg) = registry {
            let projects = reg.list_projects().await.unwrap_or_default();
            if !projects.is_empty() {
                println!("Error: Database '{}' does not exist or is not accessible.", name);
                println!();
                println!("Available projects:");
                for p in &projects {
                    let marker = if p.is_active { " (current)" } else { "" };
                    println!("  - {}{}", p.database_name, marker);
                }
                println!();
                println!("Use 'pro-project create --name {}' to create a new project.", name);
            }
        }
        bail!("Database '{}' does not exist.", name);
    }

    // Verify database has PS schema
    if !db_service.has_ps_schema(name).await? {
        bail!("Database '{}' exists but does not have Professional SMART schema.", name);
    }

    // Get current database for logging
    let config = ConfigService::with_default_path();
    let previous_db = config.get_current_database()?.unwrap_or_else(|| "None".to_string());

    // Stop service if not --no-restart
    if !no_restart {
        print!("  Stopping service... ");
        match WindowsServiceManager::stop() {
            Ok(()) => println!("Done"),
            Err(e) => {
                println!("Warning: {}", e);
                println!("  Continuing without service control...");
            }
        }
    }

    // Backup and update configuration
    print!("  Updating configuration... ");
    let backup_path = config.switch_database(name)?;
    println!("Done");

    // Update registry
    print!("  Updating registry... ");
    let registry = RegistryService::connect(&cli.db_host, cli.db_port, &cli.db_user, password).await?;
    registry.set_active_project(name).await?;
    println!("Done");

    // Start service if not --no-restart
    if !no_restart {
        print!("  Starting service... ");
        match WindowsServiceManager::start() {
            Ok(()) => println!("Done"),
            Err(e) => {
                println!("Warning: {}", e);
                println!("  Service may need to be started manually.");
            }
        }
    }

    println!();
    println!("  Previous:    {}", previous_db);
    println!("  New:         {}", name);
    println!("  Config:      {}", ConfigService::default_path().display());
    println!("  Backup:      {}", backup_path.display());
    println!();

    if !no_restart {
        let status = WindowsServiceManager::get_status()?;
        println!("Service status: {}", status);
    }

    println!();
    println!("Successfully switched to project '{}'.", name);

    Ok(())
}
