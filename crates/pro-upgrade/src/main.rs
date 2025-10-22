use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use pro_upgrade_manager::{BackupManager, MigrationManager, VersionManager};
use sqlx::postgres::PgPoolOptions;
use std::path::PathBuf;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

// Custom value parsers to trim whitespace from environment variables
fn trim_string(s: &str) -> Result<String, String> {
    Ok(s.trim().to_string())
}

fn trim_parse_u16(s: &str) -> Result<u16, String> {
    s.trim().parse::<u16>().map_err(|e| e.to_string())
}

fn trim_optional_string(s: &str) -> Result<String, String> {
    Ok(s.trim().to_string())
}

#[derive(Parser)]
#[command(name = "pro-upgrade")]
#[command(about = "Professional SMART Database Upgrade Tool", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    #[arg(long, env = "DATABASE_URL")]
    database_url: Option<String>,

    #[arg(long, env = "DB_HOST", default_value = "localhost", value_parser = trim_string)]
    db_host: String,

    #[arg(long, env = "DB_PORT", default_value = "5432", value_parser = trim_parse_u16)]
    db_port: u16,

    #[arg(long, env = "DB_NAME", default_value = "professional_smart", value_parser = trim_string)]
    db_name: String,

    #[arg(long, env = "DB_USER", default_value = "postgres", value_parser = trim_string)]
    db_user: String,

    #[arg(long, env = "DB_PASSWORD", value_parser = trim_optional_string)]
    db_password: Option<String>,
}

#[derive(Subcommand)]
enum Commands {
    CheckVersion,

    BackupDatabase {
        #[arg(long, default_value = "C:\\ProgramData\\Professional SMART\\backups")]
        backup_dir: PathBuf,
    },

    RestoreDatabase {
        backup_file: PathBuf,
    },

    ListPendingMigrations {
        #[arg(long, default_value = "C:\\Program Files\\Professional SMART\\migrations")]
        migrations_dir: PathBuf,
    },

    ApplyMigrations {
        #[arg(long, default_value = "C:\\Program Files\\Professional SMART\\migrations")]
        migrations_dir: PathBuf,
    },

    ListBackups {
        #[arg(long, default_value = "C:\\ProgramData\\Professional SMART\\backups")]
        backup_dir: PathBuf,
    },

    VerifyChecksums {
        #[arg(long, default_value = "C:\\Program Files\\Professional SMART\\migrations")]
        migrations_dir: PathBuf,
    },

    DetectInstallationType,

    ReconfigureDatabase {
        #[arg(long, default_value = "C:\\ProgramData\\Professional SMART\\config\\.env")]
        config_path: PathBuf,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)
        .context("Failed to set tracing subscriber")?;

    dotenvy::dotenv().ok();

    let cli = Cli::parse();

    let database_url = if let Some(ref url) = cli.database_url {
        url.clone()
    } else {
        let password = cli.db_password.as_ref()
            .context("Database password required (use --db-password or DB_PASSWORD env var)")?;

        format!(
            "postgres://{}:{}@{}:{}/{}",
            cli.db_user, password, cli.db_host, cli.db_port, cli.db_name
        )
    };

    let pool = PgPoolOptions::new()
        .max_connections(5)
        .connect(&database_url)
        .await
        .context("Failed to connect to database")?;

    match &cli.command {
        Commands::CheckVersion => {
            check_version(&pool).await?;
        }
        Commands::BackupDatabase { backup_dir } => {
            backup_database(&pool, backup_dir, &cli).await?;
        }
        Commands::RestoreDatabase { backup_file } => {
            restore_database(backup_file, &cli).await?;
        }
        Commands::ListPendingMigrations { migrations_dir } => {
            list_pending_migrations(&pool, migrations_dir).await?;
        }
        Commands::ApplyMigrations { migrations_dir } => {
            apply_migrations(&pool, migrations_dir).await?;
        }
        Commands::ListBackups { backup_dir } => {
            list_backups(backup_dir).await?;
        }
        Commands::VerifyChecksums { migrations_dir } => {
            verify_checksums(&pool, migrations_dir).await?;
        }
        Commands::DetectInstallationType => {
            detect_installation_type(&pool).await?;
        }
        Commands::ReconfigureDatabase { config_path } => {
            reconfigure_database(config_path).await?;
        }
    }

    Ok(())
}

async fn check_version(pool: &sqlx::PgPool) -> Result<()> {
    let version_manager = VersionManager::new(pool.clone());

    match version_manager.get_current_version().await? {
        Some(version) => {
            println!("Current Version: {}", version.version);
            println!("Installed At: {}", version.installed_at);
            if let Some(upgraded_from) = version.upgraded_from {
                println!("Upgraded From: {}", upgraded_from);
            }
            if let Some(notes) = version.notes {
                println!("Notes: {}", notes);
            }
        }
        None => {
            println!("No version tracking found - this appears to be a legacy installation");
        }
    }

    Ok(())
}

async fn backup_database(
    _pool: &sqlx::PgPool,
    backup_dir: &PathBuf,
    cli: &Cli,
) -> Result<()> {
    let backup_manager = BackupManager::new(backup_dir)
        .context("Failed to initialize backup manager")?;

    let password = cli.db_password.as_ref()
        .context("Database password required")?;

    info!("Creating backup...");
    let backup_info = backup_manager.create_backup(
        &cli.db_host,
        cli.db_port,
        &cli.db_name,
        &cli.db_user,
        password,
    )?;

    println!("Backup created successfully:");
    println!("  File: {}", backup_info.file_path.display());
    println!("  Size: {} bytes", backup_info.size_bytes);
    println!("  Created: {}", backup_info.created_at);
    println!("  Compressed: {}", backup_info.compressed);

    Ok(())
}

async fn restore_database(backup_file: &PathBuf, cli: &Cli) -> Result<()> {
    let backup_dir = backup_file.parent()
        .context("Invalid backup file path")?;

    let backup_manager = BackupManager::new(backup_dir)
        .context("Failed to initialize backup manager")?;

    let password = cli.db_password.as_ref()
        .context("Database password required")?;

    info!("Restoring from backup...");
    backup_manager.restore_backup(
        backup_file,
        &cli.db_host,
        cli.db_port,
        &cli.db_name,
        &cli.db_user,
        password,
    )?;

    println!("Database restored successfully from: {}", backup_file.display());

    Ok(())
}

async fn list_pending_migrations(pool: &sqlx::PgPool, migrations_dir: &PathBuf) -> Result<()> {
    let migration_manager = MigrationManager::new(pool.clone(), migrations_dir.clone());

    let pending = migration_manager.get_pending_migrations().await?;

    if pending.is_empty() {
        println!("No pending migrations");
    } else {
        println!("Pending migrations ({}): ", pending.len());
        for migration in pending {
            println!("  - {}", migration.file_name);
        }
    }

    Ok(())
}

async fn apply_migrations(pool: &sqlx::PgPool, migrations_dir: &PathBuf) -> Result<()> {
    let migration_manager = MigrationManager::new(pool.clone(), migrations_dir.clone());

    info!("Applying pending migrations...");
    let applied = migration_manager.apply_pending_migrations().await?;

    if applied.is_empty() {
        println!("No migrations to apply");
    } else {
        println!("Successfully applied {} migrations:", applied.len());
        for migration in applied {
            println!("  - {}", migration);
        }
    }

    Ok(())
}

async fn list_backups(backup_dir: &PathBuf) -> Result<()> {
    let backup_manager = BackupManager::new(backup_dir)
        .context("Failed to initialize backup manager")?;

    let backups = backup_manager.list_backups()?;

    if backups.is_empty() {
        println!("No backups found in: {}", backup_dir.display());
    } else {
        println!("Available backups ({}):", backups.len());
        for backup in backups {
            println!("  File: {}", backup.file_path.display());
            println!("    Created: {}", backup.created_at);
            println!("    Size: {} bytes", backup.size_bytes);
            println!("    Compressed: {}", backup.compressed);
            println!();
        }
    }

    Ok(())
}

async fn verify_checksums(pool: &sqlx::PgPool, migrations_dir: &PathBuf) -> Result<()> {
    let migration_manager = MigrationManager::new(pool.clone(), migrations_dir.clone());

    info!("Verifying migration checksums...");
    let mismatches = migration_manager.verify_checksums().await?;

    if mismatches.is_empty() {
        println!("All migration checksums verified successfully");
    } else {
        println!("WARNING: Found {} checksum mismatches:", mismatches.len());
        for mismatch in mismatches {
            println!("  - {}", mismatch);
        }
    }

    Ok(())
}

async fn detect_installation_type(pool: &sqlx::PgPool) -> Result<()> {
    let version_manager = VersionManager::new(pool.clone());

    let installation_type = version_manager.detect_installation_type().await?;

    match installation_type {
        pro_upgrade_manager::version::InstallationType::Fresh => {
            println!("Installation Type: Fresh");
            println!("This is a new installation with no existing database");
        }
        pro_upgrade_manager::version::InstallationType::Legacy => {
            println!("Installation Type: Legacy");
            println!("This is an existing installation without version tracking");
            println!("Upgrade path available - version tracking will be added");
        }
        pro_upgrade_manager::version::InstallationType::Upgrade(version) => {
            println!("Installation Type: Upgrade");
            println!("Current Version: {}", version.version);
            println!("Installed At: {}", version.installed_at);
        }
    }

    Ok(())
}

async fn reconfigure_database(config_path: &PathBuf) -> Result<()> {
    use std::io::{self, Write};

    println!("Database Reconfiguration Wizard");
    println!("================================");
    println!();
    println!("This will update the database credentials in: {}", config_path.display());
    println!();

    // Read current config if it exists
    let mut current_host = String::from("localhost");
    let mut current_port = String::from("5432");
    let mut current_name = String::from("professional_smart");
    let mut current_user = String::from("postgres");

    if config_path.exists() {
        println!("Reading current configuration...");
        let content = std::fs::read_to_string(config_path)
            .context("Failed to read current config file")?;

        for line in content.lines() {
            let line = line.trim();
            if line.starts_with("DB_HOST=") {
                current_host = line.strip_prefix("DB_HOST=").unwrap_or("localhost").trim_matches('"').to_string();
            } else if line.starts_with("DB_PORT=") {
                current_port = line.strip_prefix("DB_PORT=").unwrap_or("5432").trim_matches('"').to_string();
            } else if line.starts_with("DB_NAME=") {
                current_name = line.strip_prefix("DB_NAME=").unwrap_or("professional_smart").trim_matches('"').to_string();
            } else if line.starts_with("DB_USER=") {
                current_user = line.strip_prefix("DB_USER=").unwrap_or("postgres").trim_matches('"').to_string();
            }
        }
        println!();
    }

    // Prompt for new values
    print!("Database Host [{}]: ", current_host);
    io::stdout().flush()?;
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    let db_host = if input.trim().is_empty() { current_host } else { input.trim().to_string() };

    print!("Database Port [{}]: ", current_port);
    io::stdout().flush()?;
    input.clear();
    io::stdin().read_line(&mut input)?;
    let db_port = if input.trim().is_empty() { current_port } else { input.trim().to_string() };

    print!("Database Name [{}]: ", current_name);
    io::stdout().flush()?;
    input.clear();
    io::stdin().read_line(&mut input)?;
    let db_name = if input.trim().is_empty() { current_name } else { input.trim().to_string() };

    print!("Database User [{}]: ", current_user);
    io::stdout().flush()?;
    input.clear();
    io::stdin().read_line(&mut input)?;
    let db_user = if input.trim().is_empty() { current_user } else { input.trim().to_string() };

    print!("Database Password: ");
    io::stdout().flush()?;
    input.clear();
    io::stdin().read_line(&mut input)?;
    let db_password = input.trim().to_string();

    if db_password.is_empty() {
        anyhow::bail!("Password cannot be empty");
    }

    // Build DATABASE_URL
    let database_url = format!(
        "postgres://{}:{}@{}:{}/{}",
        db_user, db_password, db_host, db_port, db_name
    );

    // Test connection
    println!();
    println!("Testing database connection...");
    let pool = PgPoolOptions::new()
        .max_connections(1)
        .connect(&database_url)
        .await
        .context("Failed to connect to database with provided credentials")?;

    pool.close().await;
    println!("✓ Connection successful");

    // Read existing config to preserve other settings
    let mut config_lines = Vec::new();

    if config_path.exists() {
        let content = std::fs::read_to_string(config_path)?;
        for line in content.lines() {
            let trimmed = line.trim();
            // Skip old database credential lines
            if trimmed.starts_with("DATABASE_URL=")
                || trimmed.starts_with("DB_HOST=")
                || trimmed.starts_with("DB_PORT=")
                || trimmed.starts_with("DB_NAME=")
                || trimmed.starts_with("DB_USER=")
                || trimmed.starts_with("DB_PASSWORD=") {
                continue;
            }
            config_lines.push(line.to_string());
        }
    }

    // Create backup
    if config_path.exists() {
        let backup_path = format!("{}.backup_{}",
            config_path.display(),
            chrono::Utc::now().format("%Y%m%d_%H%M%S")
        );
        std::fs::copy(config_path, &backup_path)
            .context("Failed to create backup")?;
        println!("✓ Backup created: {}", backup_path);
    }

    // Build new config content
    let mut new_content = String::new();

    // Add header if new file
    if config_lines.is_empty() {
        new_content.push_str("# Professional SMART Configuration\n");
        new_content.push_str(&format!("# Updated by reconfigure-database on {}\n", chrono::Utc::now()));
        new_content.push_str("\n");
    }

    // Add database credentials at the top
    new_content.push_str("# Database Configuration\n");
    new_content.push_str(&format!("DATABASE_URL={}\n", database_url));
    new_content.push_str(&format!("DB_HOST={}\n", db_host));
    new_content.push_str(&format!("DB_PORT={}\n", db_port));
    new_content.push_str(&format!("DB_NAME={}\n", db_name));
    new_content.push_str(&format!("DB_USER={}\n", db_user));
    new_content.push_str(&format!("DB_PASSWORD={}\n", db_password));
    new_content.push_str("\n");

    // Add rest of the config (skip old database section header)
    let mut skip_next_blank = false;
    for line in config_lines {
        if line.trim() == "# Database Configuration" {
            skip_next_blank = true;
            continue;
        }
        if skip_next_blank && line.trim().is_empty() {
            skip_next_blank = false;
            continue;
        }
        new_content.push_str(&line);
        new_content.push_str("\n");
    }

    // Write new config
    std::fs::write(config_path, new_content)
        .context("Failed to write config file")?;

    println!("✓ Configuration updated successfully");
    println!();
    println!("IMPORTANT: You must restart the Professional SMART service for changes to take effect:");
    println!("  net stop ProfessionalSMART");
    println!("  net start ProfessionalSMART");

    Ok(())
}
