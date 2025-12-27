use crate::cli::Cli;
use crate::services::{backup::default_backup_dir, BackupService, ConfigService, DatabaseService, RegistryService};
use anyhow::{bail, Result};
use std::io::{self, Write};

pub async fn execute(cli: &Cli, name: &str, force: bool, backup: bool) -> Result<()> {
    let password = cli.db_password.as_deref().unwrap_or("");

    // Check if this is the active database
    let config = ConfigService::with_default_path();
    let current_db = config.get_current_database()?.unwrap_or_default();
    if current_db == name {
        bail!("Cannot delete the currently active database '{}'.\nSwitch to a different database first with 'pro-project switch --name <OTHER_DB>'.", name);
    }

    // Verify database exists
    let db_service = DatabaseService::new(&cli.db_host, cli.db_port, &cli.db_user, password);
    if !db_service.database_exists(name).await? {
        bail!("Database '{}' does not exist.", name);
    }

    // Get entity counts for warning
    let counts = db_service.get_entity_counts(name).await.ok();

    // Create backup if requested
    let backup_path = if backup {
        print!("Creating backup before deletion... ");
        let backup_service = BackupService::new(
            &cli.db_host,
            cli.db_port,
            &cli.db_user,
            password,
            default_backup_dir(),
        );
        let result = backup_service.backup(name, None)?;
        println!("Done");
        Some(result.path)
    } else {
        None
    };

    // Show warning and confirmation
    if !force {
        println!();
        println!("WARNING: You are about to delete project database '{}'", name);
        println!();

        if let Some(c) = counts {
            println!("This database contains:");
            println!("  - {} organizations", c.organizations);
            println!("  - {} facilities", c.facilities);
            println!("  - {} encounters", c.encounters);
            println!("  - {} service lines", c.service_lines);
            println!();
        }

        println!("This action is IRREVERSIBLE.");
        println!();

        // Require typing database name to confirm
        print!("To confirm deletion, type the database name: ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input != name {
            println!();
            println!("Confirmation failed. Database name does not match.");
            println!("Deletion cancelled.");
            return Ok(());
        }
    }

    println!();
    println!("Deleting database '{}'...", name);

    // Drop the database
    print!("  Dropping database... ");
    db_service.drop_database(name).await?;
    println!("Done");

    // Remove from registry
    print!("  Updating registry... ");
    let registry = RegistryService::connect(&cli.db_host, cli.db_port, &cli.db_user, password).await?;
    registry.delete_project(name).await?;

    let remaining = registry.list_projects().await?.len();
    println!("Done");

    println!();
    println!("Project database '{}' has been deleted.", name);
    println!();

    if let Some(path) = backup_path {
        println!("Backup saved to:");
        println!("  {}", path.display());
        println!();
    }

    println!("Registry updated: {} projects remaining.", remaining);

    Ok(())
}
