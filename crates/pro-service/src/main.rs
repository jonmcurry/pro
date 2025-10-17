//! Professional SMART Windows Service Entry Point
//!
//! This crate provides the Windows service wrapper for the Professional SMART claims processing system.
//! It handles service lifecycle management and delegates to the pro-worker crate for actual processing.

use anyhow::Result;
use clap::{Parser, Subcommand};
use tracing::{error, info};

#[cfg(windows)]
use std::ffi::OsString;

#[cfg(windows)]
mod service;

#[cfg(windows)]
use service::{run_service, SERVICE_NAME};

// PHASE 5: WebSocket API support
mod websocket;
mod api;

// File watcher and claims importer
mod file_watcher;
mod claims_importer;

// Constants for service metadata
#[cfg(windows)]
const SERVICE_DISPLAY_NAME: &str = "Professional SMART Claims Processing Service";
#[cfg(windows)]
const SERVICE_DESCRIPTION: &str = "Automated claims processing, validation, and flagging system for healthcare providers";

/// Professional SMART Claims Processing Service
#[derive(Parser)]
#[command(name = "professional-smart")]
#[command(about = "Professional SMART Claims Processing Service", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run as Windows service (called by Service Control Manager)
    Service,

    /// Install Windows service
    Install {
        /// Service display name
        #[arg(long, default_value = "Professional SMART Claims Processing Service")]
        display_name: String,

        /// Service description
        #[arg(long, default_value = "Automated claims processing, validation, and flagging system for healthcare providers")]
        description: String,
    },

    /// Uninstall Windows service
    Uninstall,

    /// Start Windows service
    Start,

    /// Stop Windows service
    Stop,

    /// Run in console mode (for testing/debugging)
    Console,
}

fn main() -> Result<()> {
    // Parse command line arguments
    let cli = Cli::parse();

    match cli.command {
        Commands::Service => {
            // This is called by Windows Service Control Manager
            #[cfg(windows)]
            {
                if let Err(e) = windows_service::service_dispatcher::start(SERVICE_NAME, ffi_service_main) {
                    error!("Service dispatcher failed: {}", e);
                    return Err(e.into());
                }
            }

            #[cfg(not(windows))]
            {
                error!("Service mode is only supported on Windows");
                return Err(anyhow::anyhow!("Service mode requires Windows"));
            }
        }

        Commands::Install { display_name: _, description: _ } => {
            #[cfg(windows)]
            {
                info!("Installing Windows service...");
                service::install_service(SERVICE_DISPLAY_NAME, SERVICE_DESCRIPTION)?;
                info!("Service installed successfully");
                println!("Professional SMART service installed successfully.");
                println!("Use 'professional-smart start' to start the service.");
            }

            #[cfg(not(windows))]
            {
                error!("Service installation is only supported on Windows");
                return Err(anyhow::anyhow!("Service installation requires Windows"));
            }
        }

        Commands::Uninstall => {
            #[cfg(windows)]
            {
                info!("Uninstalling Windows service...");
                service::uninstall_service()?;
                info!("Service uninstalled successfully");
                println!("Professional SMART service uninstalled successfully.");
            }

            #[cfg(not(windows))]
            {
                error!("Service uninstallation is only supported on Windows");
                return Err(anyhow::anyhow!("Service uninstallation requires Windows"));
            }
        }

        Commands::Start => {
            #[cfg(windows)]
            {
                info!("Starting Windows service...");
                service::start_service()?;
                info!("Service start command sent");
                println!("Professional SMART service start command sent.");
                println!("Check Windows Event Log or service logs for status.");
            }

            #[cfg(not(windows))]
            {
                error!("Service management is only supported on Windows");
                return Err(anyhow::anyhow!("Service management requires Windows"));
            }
        }

        Commands::Stop => {
            #[cfg(windows)]
            {
                info!("Stopping Windows service...");
                service::stop_service()?;
                info!("Service stop command sent");
                println!("Professional SMART service stop command sent.");
            }

            #[cfg(not(windows))]
            {
                error!("Service management is only supported on Windows");
                return Err(anyhow::anyhow!("Service management requires Windows"));
            }
        }

        Commands::Console => {
            println!("Starting Professional SMART in console mode...");
            println!("Press Ctrl+C to stop");

            // Initialize logging for console mode
            init_console_logging()?;

            info!("Professional SMART starting in console mode");

            // Run the worker in console mode
            let runtime = tokio::runtime::Runtime::new()?;
            runtime.block_on(async {
                run_console_mode().await
            })?;
        }
    }

    Ok(())
}

/// FFI entry point for Windows service
#[cfg(windows)]
windows_service::define_windows_service!(ffi_service_main, service_main);

#[cfg(windows)]
fn service_main(_arguments: Vec<OsString>) {
    if let Err(e) = run_service() {
        error!("Service failed: {}", e);
    }
}

/// Initialize logging for console mode
fn init_console_logging() -> Result<()> {
    use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    Ok(())
}

/// Run in console mode for testing/debugging
async fn run_console_mode() -> Result<()> {
    // Load configuration
    dotenvy::dotenv().ok();

    info!("Loading configuration...");

    // Log configuration details (sanitized)
    let db_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "postgresql://localhost/professional_smart".to_string())
        .trim()
        .to_string();

    // Sanitize password from URL for logging
    let sanitized_url = if let Some(at_pos) = db_url.find('@') {
        if let Some(colon_pos) = db_url[..at_pos].rfind(':') {
            format!("{}:****{}", &db_url[..colon_pos], &db_url[at_pos..])
        } else {
            db_url.clone()
        }
    } else {
        db_url.clone()
    };

    info!("DATABASE_URL: '{}'", sanitized_url);

    // Initialize database connection pool
    info!("Connecting to database...");
    let db_pool = pro_db::connection::create_pool_default().await
        .map_err(|e| anyhow::anyhow!("Failed to create database pool: {}", e))?;

    // Test database connection
    pro_db::connection::test_connection(&db_pool).await
        .map_err(|e| anyhow::anyhow!("Failed to connect to database: {}", e))?;

    info!("Database connection pool initialized successfully");

    // PHASE 5: Check if API server is enabled
    let enable_api = std::env::var("STREAMING_ENABLE_WEBSOCKET")
        .unwrap_or_else(|_| "true".to_string())
        .parse::<bool>()
        .unwrap_or(true);

    // PHASE 5: Start API server (WebSocket + REST endpoints) if enabled
    let api_handle = if enable_api {
        let api_addr = std::env::var("WEBSOCKET_HOST")
            .unwrap_or_else(|_| "127.0.0.1:8080".to_string());

        info!("Starting API server (WebSocket + REST) on {}", api_addr);

        let pool_clone = db_pool.clone();
        Some(tokio::spawn(async move {
            if let Err(e) = api::serve(&api_addr, pool_clone).await {
                error!("API server error: {}", e);
            }
        }))
    } else {
        info!("API server disabled");
        None
    };

    // Initialize file watcher and claims importer
    let input_dir = std::env::var("INPUT_DIR")
        .unwrap_or_else(|_| "C:\\Program Files\\Professional SMART\\data\\input".to_string())
        .trim()
        .to_string();

    info!("Initializing claims processing...");
    info!("Input directory: '{}'", input_dir);

    let mut file_watcher = file_watcher::FileWatcher::new(&input_dir)
        .map_err(|e| anyhow::anyhow!("Failed to create file watcher: {}", e))?;

    let importer = claims_importer::ClaimsImporter::new(db_pool.clone());

    info!("Professional SMART console mode started");
    info!("Configuration loaded");
    info!("File watcher is monitoring: {}", input_dir);

    // Spawn file watcher in background task
    let watcher_handle = tokio::spawn(async move {
        let result = file_watcher.run(move |file_path| {
            let importer = importer.clone();
            async move {
                info!("Processing file: {}", file_path.display());

                let import_result = importer.import_file(&file_path).await?;

                info!("{}", import_result.summary());

                if !import_result.is_success() {
                    for error in &import_result.errors {
                        error!("Import error: {}", error);
                    }
                }

                Ok(())
            }
        }).await;

        if let Err(e) = result {
            error!("File watcher error: {}", e);
        }
    });

    // Keep running until Ctrl+C
    tokio::signal::ctrl_c().await?;

    info!("Shutting down...");

    // Stop file watcher
    info!("Stopping file watcher...");
    watcher_handle.abort();

    // PHASE 5: Gracefully shutdown API server
    if let Some(handle) = api_handle {
        info!("Stopping API server...");
        handle.abort();
    }

    // Close database pool
    info!("Closing database connections...");
    db_pool.close().await;

    info!("Professional SMART service shutdown complete");

    Ok(())
}
