//! Professional SMART Windows Service Entry Point
//!
//! This crate provides the Windows service wrapper for the Professional SMART claims processing system.
//! It handles service lifecycle management and delegates to the pro-worker crate for actual processing.

use anyhow::Result;
use clap::{Parser, Subcommand};
use tracing::{error, info, warn};

#[cfg(windows)]
use std::ffi::OsString;

#[cfg(windows)]
mod service;

#[cfg(windows)]
use service::{run_service, SERVICE_NAME};

// PHASE 5: WebSocket API support
mod websocket;
mod api;

// File watcher and claims processing (two-stage pipeline)
mod file_watcher;
mod claims_importer;  // Stage 1: File -> staging.raw_claims
mod claims_processor; // Stage 2: staging.raw_claims -> encounters/errors
mod batch_sequencer;  // Sequential completion for strict FIFO ordering

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

    // Set up processed and error directories
    let processed_dir = std::path::PathBuf::from(&input_dir).parent().unwrap().join("processed");
    let error_dir = std::path::PathBuf::from(&input_dir).parent().unwrap().join("error");

    // Create directories if they don't exist
    std::fs::create_dir_all(&processed_dir)?;
    std::fs::create_dir_all(&error_dir)?;

    info!("Initializing claims processing...");
    info!("Input directory: '{}'", input_dir);
    info!("Processed directory: '{}'", processed_dir.display());
    info!("Error directory: '{}'", error_dir.display());

    let mut file_watcher = file_watcher::FileWatcher::new(&input_dir)
        .map_err(|e| anyhow::anyhow!("Failed to create file watcher: {}", e))?;

    let importer = claims_importer::ClaimsImporter::new(db_pool.clone());

    info!("Professional SMART console mode started");
    info!("Configuration loaded");
    info!("File watcher is monitoring: {}", input_dir);

    // Spawn file watcher in background task to enqueue files
    let importer_for_watcher = importer.clone();
    let watcher_handle = tokio::spawn(async move {
        let result = file_watcher.run(move |file_path| {
            let importer = importer_for_watcher.clone();
            async move {
                info!("Enqueuing file for FIFO processing: {}", file_path.display());

                // Enqueue file instead of processing directly
                let queue_id = importer.enqueue_file(&file_path).await?;
                info!("File enqueued successfully: queue_id={}", queue_id);

                Ok(())
            }
        }).await;

        if let Err(e) = result {
            error!("File watcher error: {}", e);
        }
    });

    // Spawn queue processor to process enqueued files (STAGE 1: File Ingestion)
    let importer_for_processor = importer.clone();
    let db_pool_for_processor = db_pool.clone();
    let processed_dir_for_processor = processed_dir.clone();
    let error_dir_for_processor = error_dir.clone();
    let processor_handle = tokio::spawn(async move {
        use pro_worker::queue_manager::QueueManager;

        info!("Starting STAGE 1 queue processor (file ingestion to staging)...");
        let queue_manager = QueueManager::new(db_pool_for_processor);

        loop {
            // Dequeue next file (FIFO order)
            match queue_manager.dequeue_next_global().await {
                Ok(Some(queued_file)) => {
                    info!("STAGE 1: Processing queued file: {} (queue_id={})",
                        queued_file.file_path, queued_file.queue_id);

                    // Mark as processing
                    if let Err(e) = queue_manager.mark_processing(queued_file.queue_id).await {
                        error!("Failed to mark queue entry as processing: {}", e);
                        continue;
                    }

                    // Route based on file format
                    let file_path = std::path::PathBuf::from(&queued_file.file_path);
                    match queued_file.file_format {
                        pro_worker::types::FileFormat::Csv => {
                            // STAGE 1: Fast ingestion to staging.raw_claims (two-stage pipeline for CSV)
                            match importer_for_processor.ingest_file_to_staging(&file_path, Some(queued_file.queue_id)).await {
                                Ok(ingest_result) => {
                                    info!("STAGE 1 COMPLETE: batch_id={}, ingested {} rows to staging.raw_claims",
                                        ingest_result.batch_id, ingest_result.total_rows);

                                    // Mark queue entry as completed (Stage 1 done, Stage 2 will process asynchronously)
                                    if let Err(e) = queue_manager.mark_completed(queued_file.queue_id).await {
                                        error!("Failed to mark queue entry as completed: {}", e);
                                    }

                                    // Move file to processed directory
                                    if let Err(e) = move_file_to_processed(&file_path, &processed_dir_for_processor) {
                                        error!("Failed to move file to processed directory: {}", e);
                                    }
                                }
                                Err(e) => {
                                    error!("STAGE 1 FAILED: {}", e);
                                    if let Err(mark_err) = queue_manager.mark_failed(queued_file.queue_id, &e.to_string()).await {
                                        error!("Failed to mark queue entry as failed: {}", mark_err);
                                    }

                                    // Move file to error directory
                                    if let Err(e) = move_file_to_error(&file_path, &error_dir_for_processor, &e.to_string()) {
                                        error!("Failed to move file to error directory: {}", e);
                                    }
                                }
                            }
                        }
                        pro_worker::types::FileFormat::Edi837p => {
                            // STAGE 1: Process EDI 837p file through two-stage pipeline
                            match importer_for_processor.ingest_edi_to_staging(&file_path, Some(queued_file.queue_id)).await {
                                Ok(ingest_result) => {
                                    info!("STAGE 1 COMPLETE (EDI): batch_id={}, ingested {} claims to staging.raw_claims",
                                        ingest_result.batch_id, ingest_result.total_rows);

                                    // Mark queue entry as completed (Stage 1 done, Stage 2 will process asynchronously)
                                    if let Err(e) = queue_manager.mark_completed(queued_file.queue_id).await {
                                        error!("Failed to mark queue entry as completed: {}", e);
                                    }

                                    // Move file to processed directory
                                    if let Err(e) = move_file_to_processed(&file_path, &processed_dir_for_processor) {
                                        error!("Failed to move EDI file to processed directory: {}", e);
                                    }
                                }
                                Err(e) => {
                                    error!("STAGE 1 FAILED (EDI): {}", e);
                                    if let Err(mark_err) = queue_manager.mark_failed(queued_file.queue_id, &e.to_string()).await {
                                        error!("Failed to mark queue entry as failed: {}", mark_err);
                                    }

                                    // Move file to error directory
                                    if let Err(e) = move_file_to_error(&file_path, &error_dir_for_processor, &e.to_string()) {
                                        error!("Failed to move EDI file to error directory: {}", e);
                                    }
                                }
                            }
                        }
                    }
                }
                Ok(None) => {
                    // No files in queue, wait briefly
                    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
                }
                Err(e) => {
                    error!("Failed to dequeue file: {}", e);
                    tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
                }
            }
        }
    });

    // ========================================================================
    // STAGE 2: Multi-Worker Sequential Completion (Strict FIFO Ordering)
    // ========================================================================
    use tokio::sync::mpsc;
    use batch_sequencer::{SequencedBatchAcquirer, SequentialCompletionManager};

    // Configuration
    let worker_count = std::env::var("STAGE2_WORKER_COUNT")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(8); // Default: 8 workers

    let batch_size = std::env::var("BATCH_SIZE")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(750); // Default: 750 (Aegis proven optimal)

    info!("Starting STAGE 2 with {} workers (batch_size: {})", worker_count, batch_size);

    // Create channels for communication
    // NOTE: Using mpsc (multi-producer single-consumer) for batch distribution
    // Each batch should go to ONLY ONE worker to avoid duplicate processing
    let (batch_tx, batch_rx) = mpsc::channel::<batch_sequencer::SequencedBatch>(100);
    let (result_tx, result_rx) = mpsc::channel::<batch_sequencer::BatchResult>(100);
    let (shutdown_tx_acquirer, shutdown_rx_acquirer) = mpsc::channel::<()>(1);
    let (shutdown_tx_completion, shutdown_rx_completion) = mpsc::channel::<()>(1);

    // Spawn SequencedBatchAcquirer
    let acquirer = SequencedBatchAcquirer::new(db_pool.clone(), batch_size);
    let batch_tx_for_acquirer = batch_tx.clone();
    let acquirer_handle = tokio::spawn(async move {
        if let Err(e) = acquirer.start(batch_tx_for_acquirer, shutdown_rx_acquirer).await {
            error!("SequencedBatchAcquirer error: {}", e);
        }
    });

    // Spawn Worker Pool
    // Workers share a single batch_rx using Arc<Mutex<>>
    let batch_rx = std::sync::Arc::new(tokio::sync::Mutex::new(batch_rx));
    let mut worker_handles = Vec::new();
    for worker_id in 0..worker_count {
        let worker_id_str = format!("worker-{}", worker_id);
        let processor = claims_processor::ClaimsProcessor::new(db_pool.clone());
        let batch_rx_clone = batch_rx.clone();
        let result_tx_clone = result_tx.clone();

        let worker_handle = tokio::spawn(async move {
            info!("Stage 2 {} starting", worker_id_str);

            loop {
                // Lock mutex to receive next batch (only one worker gets it)
                let sequenced_batch = {
                    let mut rx = batch_rx_clone.lock().await;
                    match rx.recv().await {
                        Some(batch) => batch,
                        None => {
                            warn!("{} batch receiver closed", worker_id_str);
                            break;
                        }
                    }
                };

                info!(
                    "{} processing batch {} ({} claims)",
                    worker_id_str,
                    sequenced_batch.sequence_number,
                    sequenced_batch.claim_ids.len()
                );

                // Process the batch
                match processor.process_sequenced_batch(
                    &sequenced_batch.claim_ids,
                    sequenced_batch.sequence_number,
                    worker_id_str.clone(),
                ).await {
                    Ok(batch_result) => {
                        // Send result to completion manager
                        if let Err(e) = result_tx_clone.send(batch_result).await {
                            error!("{} failed to send result: {}", worker_id_str, e);
                        }
                    }
                    Err(e) => {
                        error!("{} failed to process batch: {}", worker_id_str, e);
                    }
                }
            }

            info!("{} shutting down", worker_id_str);
        });

        worker_handles.push(worker_handle);
    }

    // Spawn SequentialCompletionManager
    let completion_manager = SequentialCompletionManager::new(db_pool.clone());
    let completion_handle = tokio::spawn(async move {
        if let Err(e) = completion_manager.start(result_rx, shutdown_rx_completion).await {
            error!("SequentialCompletionManager error: {}", e);
        }
    });

    // Keep running until Ctrl+C
    tokio::signal::ctrl_c().await?;

    info!("Shutting down...");

    // Stop file watcher
    info!("Stopping file watcher...");
    watcher_handle.abort();

    // Stop Stage 1 queue processor
    info!("Stopping STAGE 1 queue processor...");
    processor_handle.abort();

    // Stop Stage 2 components
    info!("Stopping STAGE 2 batch acquirer...");
    shutdown_tx_acquirer.send(()).await.ok();
    acquirer_handle.abort();

    info!("Stopping STAGE 2 workers...");
    for worker_handle in worker_handles {
        worker_handle.abort();
    }

    info!("Stopping STAGE 2 completion manager...");
    shutdown_tx_completion.send(()).await.ok();
    completion_handle.abort();

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

/// Move a successfully processed file to the processed directory
fn move_file_to_processed(file_path: &std::path::Path, processed_dir: &std::path::Path) -> Result<()> {
    use anyhow::Context;

    let filename = file_path.file_name()
        .context("Failed to get filename")?;

    let dest = processed_dir.join(filename);

    info!("Moving file to processed: {} -> {}", file_path.display(), dest.display());

    std::fs::rename(file_path, &dest)
        .context("Failed to move file to processed directory")?;

    info!("File moved to processed directory: {}", dest.display());

    Ok(())
}

/// Move a failed file to the error directory with error details
fn move_file_to_error(file_path: &std::path::Path, error_dir: &std::path::Path, error_message: &str) -> Result<()> {
    use anyhow::Context;

    let filename = file_path.file_name()
        .context("Failed to get filename")?;

    let dest = error_dir.join(filename);

    warn!("Moving failed file to error: {} -> {}", file_path.display(), dest.display());

    std::fs::rename(file_path, &dest)
        .context("Failed to move file to error directory")?;

    // Write error message to companion .error file
    let error_file = dest.with_extension("error");
    std::fs::write(&error_file, error_message)
        .context("Failed to write error file")?;

    error!("File moved to error directory: {}", dest.display());
    error!("Error details written to: {}", error_file.display());

    Ok(())
}
