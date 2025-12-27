use crate::cli::Cli;
use crate::services::{DatabaseService, RegistryService};
use anyhow::Result;
use chrono::Local;

pub async fn execute(cli: &Cli, name: &str, switch: bool) -> Result<()> {
    let password = cli.db_password.as_deref().unwrap_or("");

    println!("Creating project database '{}'...", name);
    println!();

    // Connect to database service
    let db_service = DatabaseService::new(&cli.db_host, cli.db_port, &cli.db_user, password);

    // Create the database
    println!("  Creating database... ");
    db_service.create_database(name).await?;
    println!("    Done");

    // Apply baseline schema
    println!("  Applying baseline schema...");
    db_service.apply_baseline(name).await?;
    println!("    Done");

    // Get schema version
    let version = db_service.get_schema_version(name).await?.unwrap_or_else(|| "2.12.32.0".to_string());
    let migration_count = db_service.get_migration_count(name).await.unwrap_or(69);

    // Register in SmartProAudit registry
    println!("  Registering in SmartProAudit...");
    let registry = RegistryService::connect(&cli.db_host, cli.db_port, &cli.db_user, password).await?;
    let connection_info = format!("{}:{}", cli.db_host, cli.db_port);
    registry
        .register_project(
            name,                   // project_name
            name,                   // database_name
            None,                   // organization
            "2.12.32.0",           // application_version
            &version,              // database_version
            &connection_info,      // connection_information
        )
        .await?;
    println!("    Done");

    println!();
    println!("Project database created successfully.");
    println!();
    println!("  Database:    {}", name);
    println!("  Host:        {}", cli.db_host);
    println!("  Port:        {}", cli.db_port);
    println!("  Created:     {}", Local::now().format("%Y-%m-%d %H:%M:%S"));
    println!("  Schema:      {} ({} migrations)", version, migration_count);
    println!();

    if switch {
        println!("Switching to new database...");
        crate::cli::commands::switch::execute(cli, name, false).await?;
    } else {
        println!("To switch to this project:");
        println!("  pro-project switch --name {}", name);
    }

    Ok(())
}
