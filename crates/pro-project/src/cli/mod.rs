pub mod commands;

use clap::{Parser, Subcommand};
use commands::*;

#[derive(Parser)]
#[command(name = "pro-project")]
#[command(author = "Professional SMART Team")]
#[command(version)]
#[command(about = "Professional SMART Project Database Manager", long_about = None)]
pub struct Cli {
    /// Launch graphical user interface
    #[arg(long, global = true)]
    pub gui: bool,

    /// Database host
    #[arg(long, default_value = "localhost", global = true)]
    pub db_host: String,

    /// Database port
    #[arg(long, default_value = "5432", global = true)]
    pub db_port: u16,

    /// Database user
    #[arg(long, default_value = "postgres", global = true)]
    pub db_user: String,

    /// Database password (or use DB_PASSWORD environment variable)
    #[arg(long, env = "DB_PASSWORD", global = true)]
    pub db_password: Option<String>,

    #[command(subcommand)]
    pub command: Option<Commands>,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Create a new project database
    Create {
        /// Project database name (alphanumeric + underscores, max 63 chars)
        #[arg(long)]
        name: String,

        /// Automatically switch to the new database after creation
        #[arg(long, default_value = "false")]
        switch: bool,
    },

    /// Switch to a different project database
    Switch {
        /// Target project database name
        #[arg(long)]
        name: String,

        /// Update config only, do not restart service
        #[arg(long, default_value = "false")]
        no_restart: bool,
    },

    /// List all project databases
    List {
        /// Output format: table, json, or csv
        #[arg(long, default_value = "table")]
        format: String,
    },

    /// Display detailed information about a project
    Info {
        /// Project database name (defaults to current active)
        #[arg(long)]
        name: Option<String>,
    },

    /// Delete a project database
    Delete {
        /// Project database to delete
        #[arg(long)]
        name: String,

        /// Skip confirmation prompt
        #[arg(long, default_value = "false")]
        force: bool,

        /// Create backup before deletion
        #[arg(long, default_value = "false")]
        backup: bool,
    },

    /// Create a backup of a project database
    Backup {
        /// Project database to backup (defaults to current active)
        #[arg(long)]
        name: Option<String>,

        /// Output file path (auto-generated if not specified)
        #[arg(long)]
        output: Option<String>,
    },

    /// Show upgrade status of all project databases
    Status,

    /// Apply pending schema migrations
    Upgrade {
        /// Specific project database to upgrade
        #[arg(long, conflicts_with = "all")]
        name: Option<String>,

        /// Upgrade all registered project databases
        #[arg(long, conflicts_with = "name")]
        all: bool,

        /// Create backup before each upgrade
        #[arg(long, default_value = "false")]
        backup: bool,

        /// Show what would be upgraded without applying
        #[arg(long, default_value = "false")]
        dry_run: bool,

        /// Continue upgrading other databases if one fails
        #[arg(long, default_value = "false")]
        continue_on_error: bool,
    },

    /// Launch graphical user interface
    Gui,
}

pub async fn run(cli: Cli) -> anyhow::Result<()> {
    match &cli.command {
        Some(Commands::Create { name, switch }) => {
            create::execute(&cli, name, *switch).await
        }
        Some(Commands::Switch { name, no_restart }) => {
            switch::execute(&cli, name, *no_restart).await
        }
        Some(Commands::List { format }) => {
            list::execute(&cli, format).await
        }
        Some(Commands::Info { name }) => {
            info::execute(&cli, name.as_deref()).await
        }
        Some(Commands::Delete { name, force, backup }) => {
            delete::execute(&cli, name, *force, *backup).await
        }
        Some(Commands::Backup { name, output }) => {
            backup::execute(&cli, name.as_deref(), output.as_deref()).await
        }
        Some(Commands::Status) => {
            status::execute(&cli).await
        }
        Some(Commands::Upgrade { name, all, backup, dry_run, continue_on_error }) => {
            upgrade::execute(&cli, name.as_deref(), *all, *backup, *dry_run, *continue_on_error).await
        }
        Some(Commands::Gui) => {
            // Already handled in main.rs
            Ok(())
        }
        None => {
            // No command specified, show help
            println!("Professional SMART Project Database Manager");
            println!();
            println!("Use 'pro-project --help' for usage information.");
            println!("Use 'pro-project gui' to launch the graphical interface.");
            Ok(())
        }
    }
}
