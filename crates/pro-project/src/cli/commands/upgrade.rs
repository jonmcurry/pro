use crate::cli::Cli;
use crate::services::{
    backup::default_backup_dir, BackupService, MigrationService, RegistryService,
    WindowsServiceManager,
};
use anyhow::{bail, Result};

pub async fn execute(
    cli: &Cli,
    name: Option<&str>,
    all: bool,
    backup: bool,
    dry_run: bool,
    continue_on_error: bool,
) -> Result<()> {
    let password = cli.db_password.as_deref().unwrap_or("");

    // Must specify either --name or --all
    if name.is_none() && !all {
        bail!("Must specify either --name <DATABASE> or --all");
    }

    let current_version = MigrationService::get_current_version();
    let migration_service = MigrationService::new(&cli.db_host, cli.db_port, &cli.db_user, password);
    let registry = RegistryService::connect(&cli.db_host, cli.db_port, &cli.db_user, password).await?;

    // Get projects to upgrade
    let projects = if all {
        registry.list_projects().await?
    } else {
        let db_name = name.unwrap();
        match registry.get_project(db_name).await? {
            Some(p) => vec![p],
            None => bail!("Project '{}' not found in registry.", db_name),
        }
    };

    if projects.is_empty() {
        println!("No projects found to upgrade.");
        return Ok(());
    }

    // Dry run mode
    if dry_run {
        return execute_dry_run(&migration_service, &projects, &current_version).await;
    }

    // Actual upgrade
    execute_upgrade(
        cli,
        &migration_service,
        &registry,
        &projects,
        &current_version,
        backup,
        continue_on_error,
    )
    .await
}

async fn execute_dry_run(
    migration_service: &MigrationService,
    projects: &[crate::services::ProjectInfo],
    current_version: &str,
) -> Result<()> {
    println!("Schema Upgrade Analysis (Dry Run)");
    println!("=================================");
    println!();
    println!("Installed Version:  {}", current_version);
    println!();

    let mut needs_upgrade = 0;

    for project in projects {
        let pending = migration_service
            .get_pending_migrations(&project.database_name)
            .await
            .unwrap_or_default();

        let version = project.database_version.as_deref().unwrap_or("Unknown");

        println!("Project: {}", project.database_name);
        println!("  Current:     {}", version);
        println!("  Pending:     {} migrations", pending.len());

        if pending.is_empty() {
            println!("  Status:      UP TO DATE");
        } else {
            needs_upgrade += 1;
            for m in &pending {
                println!("    - {}", m.file_name);
            }
            println!("  Status:      NEEDS UPGRADE");
        }
        println!();
    }

    println!("Summary: {} of {} projects need upgrade", needs_upgrade, projects.len());
    println!();

    if needs_upgrade > 0 {
        println!("To apply upgrades, run:");
        println!("  pro-project upgrade --all");
        println!("  pro-project upgrade --all --backup  (recommended)");
    }

    Ok(())
}

async fn execute_upgrade(
    cli: &Cli,
    migration_service: &MigrationService,
    registry: &RegistryService,
    projects: &[crate::services::ProjectInfo],
    current_version: &str,
    backup: bool,
    continue_on_error: bool,
) -> Result<()> {
    let password = cli.db_password.as_deref().unwrap_or("");

    println!("Schema Upgrade");
    println!("==============");
    println!();
    println!("Installed Version:  {}", current_version);
    println!();

    // Stop service before upgrades
    print!("Stopping service... ");
    match WindowsServiceManager::stop() {
        Ok(()) => println!("Done"),
        Err(e) => {
            println!("Warning: {}", e);
            println!("Continuing without service control...");
        }
    }
    println!();

    let mut succeeded = 0;
    let mut failed = 0;
    let mut skipped = 0;
    let mut failures: Vec<(String, String)> = Vec::new();

    let backup_service = BackupService::new(
        &cli.db_host,
        cli.db_port,
        &cli.db_user,
        password,
        default_backup_dir(),
    );

    for (i, project) in projects.iter().enumerate() {
        let pending = migration_service
            .get_pending_migrations(&project.database_name)
            .await
            .unwrap_or_default();

        if pending.is_empty() {
            skipped += 1;
            continue;
        }

        println!(
            "[{}/{}] Upgrading '{}'...",
            i + 1,
            projects.len(),
            project.database_name
        );

        // Create backup if requested
        if backup {
            print!("  Backup:      ");
            match backup_service.backup(&project.database_name, None) {
                Ok(result) => println!("{}", result.path.display()),
                Err(e) => {
                    println!("FAILED - {}", e);
                    failures.push((project.database_name.clone(), format!("Backup failed: {}", e)));
                    failed += 1;
                    if !continue_on_error {
                        break;
                    }
                    continue;
                }
            }
        }

        // Apply migrations
        print!("  Migrations:  ");
        let mut migration_failed = false;

        for m in &pending {
            print!("Applying {}... ", m.file_name);

            // Find the full migration
            let all_migrations = MigrationService::get_all_migrations();
            if let Some(migration) = all_migrations.iter().find(|em| em.version == m.version) {
                match migration_service
                    .apply_migration(&project.database_name, migration)
                    .await
                {
                    Ok(()) => println!("OK"),
                    Err(e) => {
                        println!("FAILED");
                        println!("               ERROR: {}", e);
                        failures.push((project.database_name.clone(), e.to_string()));
                        migration_failed = true;
                        break;
                    }
                }
            }
            print!("               ");
        }

        if migration_failed {
            failed += 1;
            let old_version = project.database_version.as_deref().unwrap_or("Unknown");
            println!("  Result:      FAILED (rolled back to {})", old_version);

            if !continue_on_error {
                println!();
                println!("Upgrade stopped due to error. Use --continue-on-error to proceed with remaining databases.");
                break;
            }
            println!("  Continuing due to --continue-on-error flag...");
        } else {
            succeeded += 1;
            let old_version = project.database_version.as_deref().unwrap_or("Unknown");

            // Compute actual new version from applied migrations
            let applied = migration_service
                .get_applied_migrations(&project.database_name)
                .await
                .unwrap_or_default();
            let new_version = if applied.is_empty() {
                "No migrations".to_string()
            } else {
                let max_migration = applied.iter()
                    .filter_map(|m| m.version.parse::<u32>().ok())
                    .max()
                    .unwrap_or(0);
                format!("2.12.{}.0", max_migration)
            };

            // Update version in SmartProAudit registry
            registry
                .update_database_version(&project.database_name, &new_version)
                .await
                .ok();

            println!();
            println!("  Result:      SUCCESS ({} -> {})", old_version, new_version);
        }

        println!();
    }

    // Restart service
    print!("Starting service... ");
    match WindowsServiceManager::start() {
        Ok(()) => println!("Done"),
        Err(e) => println!("Warning: {}", e),
    }

    println!();
    println!("UPGRADE SUMMARY");
    println!("===============");
    println!("  Total:       {} projects", projects.len());
    println!("  Succeeded:   {}", succeeded);
    println!("  Failed:      {}", failed);
    println!("  Skipped:     {} (already up to date)", skipped);

    if !failures.is_empty() {
        println!();
        println!("FAILED DATABASES:");
        for (db, error) in &failures {
            println!("  - {}: {}", db, error);
        }
        println!();
        println!("WARNING: Not all databases were upgraded. Review errors above.");
    } else if succeeded > 0 {
        println!();
        // Get the current max migration to show the version
        let all_migrations = MigrationService::get_all_migrations();
        let max_version = all_migrations.iter()
            .filter_map(|m| m.version.parse::<u32>().ok())
            .max()
            .unwrap_or(0);
        println!("All upgraded databases are now at 2.12.{}.0", max_version);
    }

    Ok(())
}
