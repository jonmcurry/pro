use crate::cli::Cli;
use crate::services::{ConfigService, MigrationService, RegistryService};
use anyhow::Result;

pub async fn execute(cli: &Cli) -> Result<()> {
    let password = cli.db_password.as_deref().unwrap_or("");

    // Get current version from embedded migrations
    let current_version = MigrationService::get_current_version();

    // Connect to registry
    let registry = RegistryService::connect(&cli.db_host, cli.db_port, &cli.db_user, password).await?;
    let migration_service = MigrationService::new(&cli.db_host, cli.db_port, &cli.db_user, password);

    // Get current active database from config
    let config = ConfigService::with_default_path();
    let current_db = config.get_current_database().ok().flatten();

    // Get all projects
    let projects = registry.list_projects().await?;

    if projects.is_empty() {
        println!("No project databases registered.");
        return Ok(());
    }

    println!("PROJECT STATUS");
    println!("==============");
    println!();
    println!("Installed Version: {}", current_version);
    println!();
    println!(
        "  {:1} {:24} {:16} {}",
        "", "DATABASE", "SCHEMA VERSION", "STATUS"
    );
    println!(
        "  {:1} {:24} {:16} {}",
        "", "------------------------", "----------------", "------------------------"
    );

    let mut needs_upgrade = 0;

    for project in &projects {
        let is_current = current_db.as_ref().map(|c| c == &project.database_name).unwrap_or(false);
        let marker = if is_current { "*" } else { " " };

        // Get pending migrations for this database
        let pending = migration_service
            .get_pending_migrations(&project.database_name)
            .await
            .unwrap_or_default();

        // Use database_version from SmartProAudit registry as source of truth
        let version = project.database_version.as_deref().unwrap_or("Unknown");

        let status = if pending.is_empty() {
            "Up to date".to_string()
        } else {
            needs_upgrade += 1;
            format!("Needs upgrade ({} pending)", pending.len())
        };

        println!(
            "{} {:24} {:16} {}",
            marker,
            truncate(&project.database_name, 24),
            version,
            status
        );
    }

    println!();

    if needs_upgrade > 0 {
        println!(
            "{} of {} projects need schema upgrades.",
            needs_upgrade,
            projects.len()
        );
        println!();
        println!("Run 'pro-project upgrade --all --dry-run' to see pending migrations.");
        println!("Run 'pro-project upgrade --all --backup' to apply upgrades.");
    } else {
        println!("All {} projects are up to date.", projects.len());
    }

    Ok(())
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() > max_len {
        format!("{}...", &s[..max_len - 3])
    } else {
        s.to_string()
    }
}
