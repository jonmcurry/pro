use crate::cli::Cli;
use crate::services::{ConfigService, DatabaseService, MigrationService, RegistryService};
use anyhow::{bail, Result};

pub async fn execute(cli: &Cli, name: Option<&str>) -> Result<()> {
    let password = cli.db_password.as_deref().unwrap_or("");

    // Determine which database to show info for
    let db_name = match name {
        Some(n) => n.to_string(),
        None => {
            // Use current active database from config
            let config = ConfigService::with_default_path();
            match config.get_current_database()? {
                Some(db) => db,
                None => bail!("No database specified and no active database configured."),
            }
        }
    };

    // Connect to registry to get project info
    let registry = RegistryService::connect(&cli.db_host, cli.db_port, &cli.db_user, password).await?;
    let db_service = DatabaseService::new(&cli.db_host, cli.db_port, &cli.db_user, password);
    let migration_service = MigrationService::new(&cli.db_host, cli.db_port, &cli.db_user, password);

    // Check if database exists
    if !db_service.database_exists(&db_name).await? {
        bail!("Database '{}' does not exist.", db_name);
    }

    // Get registry info
    let project = registry.get_project(&db_name).await?;

    // Get entity counts
    let counts = db_service.get_entity_counts(&db_name).await.ok();

    // Get database size
    let size = db_service.get_database_size(&db_name).await.ok();

    // Get migration info
    let migration_count = db_service.get_migration_count(&db_name).await.unwrap_or(0);
    let pending = migration_service.get_pending_migrations(&db_name).await.unwrap_or_default();

    // Get schema version
    let schema_version = db_service.get_schema_version(&db_name).await?.unwrap_or_else(|| "Unknown".to_string());

    println!("PROJECT INFORMATION: {}", db_name);
    println!("{}", "=".repeat(35 + db_name.len()));
    println!();

    println!("Connection:");
    println!("  Host:              {}", cli.db_host);
    println!("  Port:              {}", cli.db_port);
    println!("  Database:          {}", db_name);
    println!("  User:              {}", cli.db_user);
    println!();

    println!("Schema:");
    println!("  Current Version:   {}", schema_version);
    if let Some(ref p) = project {
        println!("  Installed:         {}", p.created_at.format("%Y-%m-%d %H:%M:%S"));
    }
    println!("  Migrations:        {} applied", migration_count);
    if !pending.is_empty() {
        println!("  Pending:           {} migrations", pending.len());
        for m in &pending {
            println!("                     - {}", m.file_name);
        }
    }
    println!();

    if let Some(c) = counts {
        println!("Statistics:");
        println!("  Organizations:     {}", c.organizations);
        println!("  Facilities:        {}", c.facilities);
        println!("  Providers:         {}", c.providers);
        println!("  Encounters:        {:>12}", format_number(c.encounters));
        println!("  Service Lines:     {:>12}", format_number(c.service_lines));
        if c.raw_claims_pending > 0 {
            println!("  Raw Claims:        {} (pending)", c.raw_claims_pending);
        }
        println!();
    }

    if let Some(s) = size {
        println!("Storage:");
        println!("  Database Size:     {}", format_size(s));
        println!();
    }

    if let Some(p) = project {
        println!("Registry Info:");
        if let Some(ref org) = p.organization {
            println!("  Organization:      {}", org);
        }
        println!("  Project Name:      {}", p.project_name);
        println!("  Created:           {}", p.created_at.format("%Y-%m-%d %H:%M:%S"));
        if let Some(last_used) = p.last_used_at {
            println!("  Last Used:         {}", last_used.format("%Y-%m-%d %H:%M:%S"));
        }
        println!("  Active:            {}", if p.is_active { "Yes" } else { "No" });
        if let Some(ref notes) = p.notes {
            if !notes.is_empty() {
                println!("  Notes:             {}", notes);
            }
        }
    }

    Ok(())
}

fn format_number(n: i64) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}

fn format_size(bytes: i64) -> String {
    const KB: i64 = 1024;
    const MB: i64 = KB * 1024;
    const GB: i64 = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.0} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.0} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}
