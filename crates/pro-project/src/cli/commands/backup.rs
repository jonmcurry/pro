use crate::cli::Cli;
use crate::services::{backup::default_backup_dir, BackupService, ConfigService, DatabaseService};
use anyhow::{bail, Result};

pub async fn execute(cli: &Cli, name: Option<&str>, output: Option<&str>) -> Result<()> {
    let password = cli.db_password.as_deref().unwrap_or("");

    // Determine which database to backup
    let db_name = match name {
        Some(n) => n.to_string(),
        None => {
            let config = ConfigService::with_default_path();
            match config.get_current_database()? {
                Some(db) => db,
                None => bail!("No database specified and no active database configured."),
            }
        }
    };

    // Verify database exists
    let db_service = DatabaseService::new(&cli.db_host, cli.db_port, &cli.db_user, password);
    if !db_service.database_exists(&db_name).await? {
        bail!("Database '{}' does not exist.", db_name);
    }

    // Get database size for estimation
    let db_size = db_service.get_database_size(&db_name).await?;

    println!("Creating backup of '{}'...", db_name);
    println!();
    println!("  Source:      {} ({})", db_name, format_size(db_size));

    // Create backup
    let backup_service = BackupService::new(
        &cli.db_host,
        cli.db_port,
        &cli.db_user,
        password,
        default_backup_dir(),
    );

    let result = backup_service.backup(&db_name, output)?;

    let compression_ratio = if db_size > 0 {
        100 - (result.size_bytes as i64 * 100 / db_size)
    } else {
        0
    };

    println!("  Output:      {}", result.path.display());
    println!("  Format:      PostgreSQL custom (compressed)");
    println!("  Duration:    {} seconds", result.duration_secs);
    println!(
        "  Size:        {} ({}% compression)",
        format_size(result.size_bytes as i64),
        compression_ratio
    );
    println!();
    println!("Backup completed successfully.");

    Ok(())
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
