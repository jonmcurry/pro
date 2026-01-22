//! Windows Service Implementation
//!
//! This module handles all Windows service-specific functionality including:
//! - Service registration and lifecycle management
//! - Service Control Manager (SCM) integration
//! - Service installation/uninstallation
//! - Service start/stop/control operations

use anyhow::{Context, Result};
use std::ffi::OsString;
use std::time::Duration;
use tracing::{debug, error, info, warn};
use windows_service::{
    service::{
        ServiceAccess, ServiceControl, ServiceControlAccept, ServiceErrorControl, ServiceExitCode,
        ServiceInfo, ServiceStartType, ServiceState, ServiceStatus, ServiceType,
    },
    service_control_handler::{self, ServiceControlHandlerResult},
    service_manager::{ServiceManager, ServiceManagerAccess},
};

/// Service name (short name used by SCM)
pub const SERVICE_NAME: &str = "ProfessionalSMART";

/// Default service display name (for reference - install_service takes this as a parameter)
#[allow(dead_code)]
const SERVICE_DISPLAY_NAME: &str = "Professional SMART Claims Processing Service";

/// Default service description (for reference - install_service takes this as a parameter)
#[allow(dead_code)]
const SERVICE_DESCRIPTION: &str = "Automated claims processing, validation, and flagging system for healthcare providers. Processes EDI 837p and CSV claim files with comprehensive rules engine and RVU-based payment calculations.";

/// Main service run loop
pub fn run_service() -> Result<()> {
    // Load .env file FIRST before initializing logging
    // This allows LOG_LEVEL to be read from the config file
    load_env_config();

    // Initialize file logging for service mode (now reads LOG_LEVEL from env)
    let _guard = init_service_logging()?;

    // Log the effective log level
    let log_level = std::env::var("LOG_LEVEL")
        .or_else(|_| std::env::var("RUST_LOG"))
        .unwrap_or_else(|_| "info".to_string());
    info!("Professional SMART service starting (log level: {})", log_level);

    // Create shutdown channel
    let (shutdown_tx, shutdown_rx) = tokio::sync::mpsc::channel::<()>(1);

    // Clone the sender for the event handler
    let shutdown_tx_clone = shutdown_tx.clone();

    // Create event handler for service control events
    let event_handler = move |control_event| -> ServiceControlHandlerResult {
        match control_event {
            ServiceControl::Stop => {
                info!("Service stop requested");
                // Signal worker to shutdown gracefully
                if let Err(e) = shutdown_tx_clone.try_send(()) {
                    error!("Failed to send shutdown signal: {}", e);
                }
                ServiceControlHandlerResult::NoError
            }
            ServiceControl::Shutdown => {
                info!("System shutdown - stopping service");
                // Signal worker to shutdown immediately
                if let Err(e) = shutdown_tx_clone.try_send(()) {
                    error!("Failed to send shutdown signal: {}", e);
                }
                ServiceControlHandlerResult::NoError
            }
            ServiceControl::Interrogate => ServiceControlHandlerResult::NoError,
            _ => ServiceControlHandlerResult::NotImplemented,
        }
    };

    // Register service control handler
    let status_handle = service_control_handler::register(SERVICE_NAME, event_handler)
        .context("Failed to register service control handler")?;

    // Tell Windows the service is starting
    status_handle
        .set_service_status(ServiceStatus {
            service_type: ServiceType::OWN_PROCESS,
            current_state: ServiceState::StartPending,
            controls_accepted: ServiceControlAccept::empty(),
            exit_code: ServiceExitCode::Win32(0),
            checkpoint: 0,
            wait_hint: Duration::from_secs(5),
            process_id: None,
        })
        .context("Failed to set service status to StartPending")?;

    info!("Service registered with SCM, initializing...");

    // Log DATABASE_URL status (config was already loaded in load_env_config)
    if let Ok(db_url) = std::env::var("DATABASE_URL") {
        // Mask password in log
        let masked = if db_url.contains('@') {
            let parts: Vec<&str> = db_url.split('@').collect();
            if parts.len() == 2 {
                let before_at = parts[0];
                if let Some(colon_pos) = before_at.rfind(':') {
                    format!("{}:****@{}", &before_at[..colon_pos], parts[1])
                } else {
                    "****".to_string()
                }
            } else {
                "****".to_string()
            }
        } else {
            db_url.clone()
        };
        info!("DATABASE_URL is set: {}", masked);
    } else {
        warn!("DATABASE_URL environment variable is NOT set");
    }

    // Initialize worker
    info!("Initializing worker pipeline...");

    // Create Tokio runtime for async operations
    let runtime = tokio::runtime::Runtime::new()
        .context("Failed to create Tokio runtime")?;

    // Spawn worker task
    let worker_handle = runtime.spawn(async move {
        let mut shutdown_rx = shutdown_rx;
        info!("Worker task started");

        // Initialize database connection pool
        info!("Connecting to database...");

        // Log the DATABASE_URL being used (with password masked)
        match std::env::var("DATABASE_URL") {
            Ok(db_url) => {
                // Mask password in log
                let masked = if db_url.contains('@') {
                    let parts: Vec<&str> = db_url.split('@').collect();
                    if parts.len() == 2 {
                        let before_at = parts[0];
                        if let Some(colon_pos) = before_at.rfind(':') {
                            format!("{}:****@{}", &before_at[..colon_pos], parts[1])
                        } else {
                            "****".to_string()
                        }
                    } else {
                        "****".to_string()
                    }
                } else {
                    db_url.clone()
                };
                info!("Using DATABASE_URL: {}", masked);
            }
            Err(_) => {
                warn!("DATABASE_URL environment variable is NOT set - using default connection");
                warn!("Default will be: postgresql://localhost/professional_smart");
                warn!("This will attempt to authenticate as the current OS user");
            }
        }

        // MIGRATION CHECK: Apply pending database migrations before connecting
        info!("Checking for pending database migrations...");

        // Create a temporary connection to check/apply migrations
        match pro_db::connection::create_pool_default().await {
            Ok(temp_pool) => {
                let migration_manager = pro_upgrade_manager::MigrationManager::new_embedded(temp_pool.clone());

                match migration_manager.get_pending_migrations().await {
                    Ok(pending) => {
                        if pending.is_empty() {
                            info!("No pending migrations - database is up to date");
                        } else {
                            info!("Found {} pending migrations - applying...", pending.len());
                            for migration in &pending {
                                info!("  - {}", migration.file_name);
                            }

                            match migration_manager.apply_pending_migrations().await {
                                Ok(applied_names) => {
                                    info!("Successfully applied {} migrations", applied_names.len());
                                    for name in &applied_names {
                                        info!("  Applied: {}", name);
                                    }
                                }
                                Err(e) => {
                                    error!("Failed to apply migrations: {}", e);
                                    error!("Service will continue but may encounter database errors");
                                }
                            }
                        }
                    }
                    Err(e) => {
                        warn!("Could not check pending migrations: {}", e);
                        warn!("Service will continue but migrations may be needed");
                    }
                }

                // Close temporary pool
                temp_pool.close().await;
            }
            Err(e) => {
                warn!("Could not connect to database for migration check: {}", e);
                warn!("Migrations will not be applied automatically");
            }
        }

        info!("Creating main database connection pool...");
        let db_pool = match pro_db::connection::create_pool_default().await {
            Ok(pool) => pool,
            Err(e) => {
                error!("Failed to create database pool: {}", e);
                error!("Possible causes:");
                error!("  1. DATABASE_URL not set or incorrect");
                error!("  2. PostgreSQL server not running");
                error!("  3. Authentication failed (check username/password)");
                error!("  4. Database does not exist");
                error!("  5. Network/firewall issue");
                return;
            }
        };

        // Test database connection
        if let Err(e) = pro_db::connection::test_connection(&db_pool).await {
            error!("Failed to connect to database: {}", e);
            return;
        }

        info!("Database connection pool initialized successfully");

        // Start NPI enrichment worker (background provider enrichment)
        let enable_npi_enrichment = std::env::var("NPI_ENRICHMENT_ENABLED")
            .unwrap_or_else(|_| "true".to_string())
            .parse::<bool>()
            .unwrap_or(true);

        let npi_enrichment_handle = if enable_npi_enrichment {
            info!("Starting NPI enrichment worker (background provider enrichment)");

            use pro_npi_enrichment::{EnrichmentWorker, WorkerConfig};
            use tokio::time::Duration;

            let config = WorkerConfig {
                batch_size: std::env::var("NPI_BATCH_SIZE")
                    .ok()
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(10),
                poll_interval: Duration::from_secs(
                    std::env::var("NPI_POLL_INTERVAL_SECS")
                        .ok()
                        .and_then(|s| s.parse::<u64>().ok())
                        .unwrap_or(30)
                ),
                rate_limit_delay: Duration::from_millis(
                    std::env::var("NPI_RATE_LIMIT_MS")
                        .ok()
                        .and_then(|s| s.parse::<u64>().ok())
                        .unwrap_or(200)
                ),
                enabled: true,
            };

            let pool_clone = db_pool.clone();
            Some(tokio::spawn(async move {
                match EnrichmentWorker::with_config(pool_clone, config) {
                    Ok(worker) => {
                        if let Err(e) = worker.run().await {
                            error!("NPI enrichment worker error: {}", e);
                        }
                    }
                    Err(e) => {
                        error!("Failed to create NPI enrichment worker: {}", e);
                    }
                }
            }))
        } else {
            info!("NPI enrichment worker disabled");
            None
        };

        // Initialize file watcher and claims importer
        let input_dir = std::env::var("INPUT_DIR")
            .unwrap_or_else(|_| "C:\\Program Files\\Professional SMART\\data\\input".to_string());
        // Handle case where input_dir might not have a parent (e.g., root directory)
        let input_path = std::path::PathBuf::from(&input_dir);
        let parent_dir = match input_path.parent() {
            Some(p) => p,
            None => {
                error!("Input directory '{}' has no parent directory - cannot create processed/error directories", input_dir);
                return;
            }
        };
        let processed_dir = parent_dir.join("processed");
        let error_dir = parent_dir.join("error");

        info!("Initializing claims processing...");
        info!("Input directory: {}", input_dir);

        let mut file_watcher = match crate::file_watcher::FileWatcher::new(&input_dir) {
            Ok(w) => w,
            Err(e) => {
                error!("Failed to create file watcher: {}", e);
                return;
            }
        };

        let importer = crate::claims_importer::ClaimsImporter::new(db_pool.clone());

        info!("File watcher is monitoring: {}", input_dir);

        // Spawn file watcher in background task (enqueues files for two-stage processing)
        let importer_for_watcher = importer.clone();
        let watcher_handle = tokio::spawn(async move {
            let result = file_watcher.run(move |file_path| {
                let importer = importer_for_watcher.clone();
                async move {
                    info!("File detected: {} - enqueueing for processing", file_path.display());

                    // Enqueue file for two-stage processing
                    // NOTE: Return Err("SKIP_MOVE") to prevent file_watcher from moving the file.
                    // The file must stay in place for Stage 1 processor to read it.
                    // Stage 1 will handle moving the file after successful ingestion.
                    match importer.enqueue_file(&file_path).await {
                        Ok(queue_id) => {
                            info!("File enqueued successfully: queue_id={}", queue_id);
                            // Return special error to prevent moving the file
                            Err(anyhow::anyhow!("SKIP_MOVE"))
                        }
                        Err(e) => {
                            error!("Failed to enqueue file: {}", e);
                            Err(e)
                        }
                    }
                }
            }).await;

            if let Err(e) = result {
                error!("File watcher error: {}", e);
            }
        });

        // Spawn Stage 1 queue processor
        let db_pool_for_processor = db_pool.clone();
        let importer_for_processor = importer.clone();
        let stage1_handle = tokio::spawn(async move {
            use pro_worker::queue_manager::QueueManager;

            info!("Starting STAGE 1 queue processor (file ingestion to staging)...");
            let queue_manager = QueueManager::new(db_pool_for_processor);

            // Track consecutive empty polls for exponential backoff
            let mut consecutive_empty = 0u32;

            loop {
                match queue_manager.dequeue_next_global().await {
                    Ok(Some(queued_file)) => {
                        // Reset backoff when file found
                        consecutive_empty = 0;

                        info!("STAGE 1: Processing queued file: {} (queue_id={})",
                            queued_file.file_path, queued_file.queue_id);

                        if let Err(e) = queue_manager.mark_processing(queued_file.queue_id).await {
                            error!("Failed to mark queue entry as processing: {}", e);
                            continue;
                        }

                        let file_path = std::path::PathBuf::from(&queued_file.file_path);

                        // Route based on file format
                        match queued_file.file_format {
                            pro_worker::types::FileFormat::Csv => {
                                // CSV files: Process through two-stage pipeline
                                match importer_for_processor.ingest_file_to_staging(&file_path, Some(queued_file.queue_id)).await {
                                    Ok(ingest_result) => {
                                        info!("STAGE 1 COMPLETE: batch_id={}, ingested {} rows",
                                            ingest_result.batch_id, ingest_result.total_rows);

                                        if let Err(e) = queue_manager.mark_completed(queued_file.queue_id).await {
                                            error!("Failed to mark queue entry as completed: {}", e);
                                        }

                                        // Move file to processed directory
                                        if let Err(e) = move_file_to_processed(&file_path, &processed_dir) {
                                            error!("Failed to move file to processed directory: {}", e);
                                        }
                                    }
                                    Err(e) => {
                                        error!("STAGE 1 FAILED: {}", e);
                                        if let Err(mark_err) = queue_manager.mark_failed(queued_file.queue_id, &e.to_string()).await {
                                            error!("Failed to mark queue entry as failed: {}", mark_err);
                                        }

                                        // Move file to error directory
                                        if let Err(e) = move_file_to_error(&file_path, &error_dir, &e.to_string()) {
                                            error!("Failed to move file to error directory: {}", e);
                                        }
                                    }
                                }
                            }
                            pro_worker::types::FileFormat::Edi837p => {
                                // EDI 837p files: Process through two-stage pipeline
                                match importer_for_processor.ingest_edi_to_staging(&file_path, Some(queued_file.queue_id)).await {
                                    Ok(ingest_result) => {
                                        info!("STAGE 1 COMPLETE (EDI): batch_id={}, ingested {} claims",
                                            ingest_result.batch_id, ingest_result.total_rows);

                                        if let Err(e) = queue_manager.mark_completed(queued_file.queue_id).await {
                                            error!("Failed to mark queue entry as completed: {}", e);
                                        }

                                        // Move file to processed directory
                                        if let Err(e) = move_file_to_processed(&file_path, &processed_dir) {
                                            error!("Failed to move EDI file to processed directory: {}", e);
                                        }
                                    }
                                    Err(e) => {
                                        error!("STAGE 1 FAILED (EDI): {}", e);
                                        if let Err(mark_err) = queue_manager.mark_failed(queued_file.queue_id, &e.to_string()).await {
                                            error!("Failed to mark queue entry as failed: {}", mark_err);
                                        }

                                        // Move file to error directory
                                        if let Err(e) = move_file_to_error(&file_path, &error_dir, &e.to_string()) {
                                            error!("Failed to move EDI file to error directory: {}", e);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    Ok(None) => {
                        // No files in queue - use exponential backoff to reduce DB polling
                        // Start at 2s, double each time, max 30s
                        consecutive_empty = consecutive_empty.saturating_add(1);
                        let backoff_secs = std::cmp::min(1 << consecutive_empty.min(4), 30);
                        debug!("STAGE 1: No files in queue, backing off for {}s", backoff_secs);
                        tokio::time::sleep(tokio::time::Duration::from_secs(backoff_secs)).await;
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
        use crate::batch_sequencer::{SequencedBatchAcquirer, SequentialCompletionManager};

        // Configuration
        let worker_count = std::env::var("STAGE2_WORKER_COUNT")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(12); // Default: 12 workers (optimal based on testing)

        let batch_size = std::env::var("BATCH_SIZE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(250); // Default: 250 (smaller batches for consistent throughput)

        info!("Starting STAGE 2 with {} workers (batch_size: {})", worker_count, batch_size);

        // Create channels for communication
        // NOTE: Using mpsc (multi-producer single-consumer) for batch distribution
        // Each batch should go to ONLY ONE worker to avoid duplicate processing
        let (batch_tx, batch_rx) = mpsc::channel::<crate::batch_sequencer::SequencedBatch>(100);
        let (result_tx, result_rx) = mpsc::channel::<crate::batch_sequencer::BatchResult>(100);
        let (shutdown_tx_acquirer, shutdown_rx_acquirer) = mpsc::channel::<()>(1);
        let (shutdown_tx_completion, shutdown_rx_completion) = mpsc::channel::<()>(1);

        // Spawn SequencedBatchAcquirer
        let acquirer = SequencedBatchAcquirer::new(db_pool.clone(), batch_size).await.expect("Failed to create SequencedBatchAcquirer");
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
            let processor = crate::claims_processor::ClaimsProcessor::new(db_pool.clone()).await;
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

                    match processor.process_sequenced_batch(
                        &sequenced_batch.claim_ids,
                        sequenced_batch.sequence_number,
                        worker_id_str.clone(),
                    ).await {
                        Ok(batch_result) => {
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
        let completion_manager = SequentialCompletionManager::new(db_pool.clone()).await.expect("Failed to create SequentialCompletionManager");
        let completion_handle = tokio::spawn(async move {
            if let Err(e) = completion_manager.start(result_rx, shutdown_rx_completion).await {
                error!("SequentialCompletionManager error: {}", e);
            }
        });

        // Keep running until shutdown signal received
        loop {
            tokio::select! {
                _ = shutdown_rx.recv() => {
                    info!("Shutdown signal received, stopping worker");
                    break;
                }
                _ = tokio::time::sleep(Duration::from_secs(60)) => {
                    // Worker heartbeat
                    // info!("Worker heartbeat");
                }
            }
        }

        info!("Stopping file watcher...");
        watcher_handle.abort();

        info!("Stopping STAGE 1 queue processor...");
        stage1_handle.abort();

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

        // Stop NPI enrichment worker
        if let Some(handle) = npi_enrichment_handle {
            info!("Stopping NPI enrichment worker...");
            handle.abort();
        }

        info!("Closing database connections...");
        db_pool.close().await;

        info!("Worker task shutting down gracefully");
    });

    // Tell Windows the service is now running
    status_handle
        .set_service_status(ServiceStatus {
            service_type: ServiceType::OWN_PROCESS,
            current_state: ServiceState::Running,
            controls_accepted: ServiceControlAccept::STOP | ServiceControlAccept::SHUTDOWN,
            exit_code: ServiceExitCode::Win32(0),
            checkpoint: 0,
            wait_hint: Duration::default(),
            process_id: None,
        })
        .context("Failed to set service status to Running")?;

    info!("Service is running");

    // Wait for worker to complete (will run until stop is requested)
    runtime.block_on(worker_handle).ok();

    info!("Worker task completed, service stopping");

    // Tell Windows the service is stopping
    status_handle
        .set_service_status(ServiceStatus {
            service_type: ServiceType::OWN_PROCESS,
            current_state: ServiceState::StopPending,
            controls_accepted: ServiceControlAccept::empty(),
            exit_code: ServiceExitCode::Win32(0),
            checkpoint: 0,
            wait_hint: Duration::from_secs(5),
            process_id: None,
        })
        .context("Failed to set service status to StopPending")?;

    // Clean up
    info!("Service cleanup complete");

    // Tell Windows the service has stopped
    status_handle
        .set_service_status(ServiceStatus {
            service_type: ServiceType::OWN_PROCESS,
            current_state: ServiceState::Stopped,
            controls_accepted: ServiceControlAccept::empty(),
            exit_code: ServiceExitCode::Win32(0),
            checkpoint: 0,
            wait_hint: Duration::default(),
            process_id: None,
        })
        .context("Failed to set service status to Stopped")?;

    info!("Service stopped");

    Ok(())
}

/// Install Windows service
pub fn install_service(display_name: &str, description: &str) -> Result<()> {
    // Get path to current executable
    let exe_path = std::env::current_exe()
        .context("Failed to get current executable path")?;

    // Open service manager
    let manager = ServiceManager::local_computer(
        None::<&str>,
        ServiceManagerAccess::CONNECT | ServiceManagerAccess::CREATE_SERVICE,
    )
    .context("Failed to open Service Control Manager")?;

    // Check if service already exists
    if let Ok(_) = manager.open_service(SERVICE_NAME, ServiceAccess::QUERY_STATUS) {
        return Err(anyhow::anyhow!(
            "Service '{}' already exists. Uninstall it first.",
            SERVICE_NAME
        ));
    }

    // Create service info
    let service_info = ServiceInfo {
        name: OsString::from(SERVICE_NAME),
        display_name: OsString::from(display_name),
        service_type: ServiceType::OWN_PROCESS,
        start_type: ServiceStartType::AutoStart, // Start automatically on boot
        error_control: ServiceErrorControl::Normal,
        executable_path: exe_path.clone(),
        launch_arguments: vec![OsString::from("service")], // Run in service mode
        dependencies: vec![],                               // Could add PostgreSQL dependency
        account_name: None,                                 // Run as Local System
        account_password: None,
    };

    // Create service
    let service = manager
        .create_service(&service_info, ServiceAccess::CHANGE_CONFIG)
        .context("Failed to create service")?;

    // Set service description
    service
        .set_description(description)
        .context("Failed to set service description")?;

    // Note: Service recovery options (restart on failure) can be configured via sc.exe:
    // sc failure ProfessionalSMART reset= 86400 actions= restart/60000/restart/120000/restart/300000

    info!("Service '{}' installed successfully", SERVICE_NAME);

    Ok(())
}

/// Uninstall Windows service
pub fn uninstall_service() -> Result<()> {
    // Open service manager
    let manager = ServiceManager::local_computer(
        None::<&str>,
        ServiceManagerAccess::CONNECT,
    )
    .context("Failed to open Service Control Manager")?;

    // Open service
    let service = manager
        .open_service(SERVICE_NAME, ServiceAccess::QUERY_STATUS | ServiceAccess::STOP | ServiceAccess::DELETE)
        .context(format!("Failed to open service '{}'. Is it installed?", SERVICE_NAME))?;

    // Check if service is running
    let status = service
        .query_status()
        .context("Failed to query service status")?;

    // Stop service if running
    if status.current_state != ServiceState::Stopped {
        info!("Stopping service before uninstall...");

        service
            .stop()
            .context("Failed to stop service")?;

        // Wait for service to stop (max 30 seconds)
        let mut retries = 30;
        loop {
            std::thread::sleep(Duration::from_secs(1));

            let status = service
                .query_status()
                .context("Failed to query service status")?;

            if status.current_state == ServiceState::Stopped {
                break;
            }

            retries -= 1;
            if retries == 0 {
                return Err(anyhow::anyhow!("Service did not stop within 30 seconds"));
            }
        }

        info!("Service stopped");
    }

    // Delete service
    service
        .delete()
        .context("Failed to delete service")?;

    info!("Service '{}' uninstalled successfully", SERVICE_NAME);

    Ok(())
}

/// Start Windows service
pub fn start_service() -> Result<()> {
    // Open service manager
    let manager = ServiceManager::local_computer(
        None::<&str>,
        ServiceManagerAccess::CONNECT,
    )
    .context("Failed to open Service Control Manager")?;

    // Open service
    let service = manager
        .open_service(SERVICE_NAME, ServiceAccess::QUERY_STATUS | ServiceAccess::START)
        .context(format!("Failed to open service '{}'. Is it installed?", SERVICE_NAME))?;

    // Check current status
    let status = service
        .query_status()
        .context("Failed to query service status")?;

    match status.current_state {
        ServiceState::Running => {
            info!("Service is already running");
            return Ok(());
        }
        ServiceState::StartPending => {
            info!("Service is already starting");
            return Ok(());
        }
        _ => {}
    }

    // Start service
    service
        .start::<&str>(&[])
        .context("Failed to start service")?;

    info!("Service start command sent");

    Ok(())
}

/// Stop Windows service
pub fn stop_service() -> Result<()> {
    // Open service manager
    let manager = ServiceManager::local_computer(
        None::<&str>,
        ServiceManagerAccess::CONNECT,
    )
    .context("Failed to open Service Control Manager")?;

    // Open service
    let service = manager
        .open_service(SERVICE_NAME, ServiceAccess::QUERY_STATUS | ServiceAccess::STOP)
        .context(format!("Failed to open service '{}'. Is it installed?", SERVICE_NAME))?;

    // Check current status
    let status = service
        .query_status()
        .context("Failed to query service status")?;

    match status.current_state {
        ServiceState::Stopped => {
            info!("Service is already stopped");
            return Ok(());
        }
        ServiceState::StopPending => {
            info!("Service is already stopping");
            return Ok(());
        }
        _ => {}
    }

    // Stop service
    service
        .stop()
        .context("Failed to stop service")?;

    info!("Service stop command sent");

    Ok(())
}

/// Load environment configuration from .env file
/// This must be called BEFORE init_service_logging so LOG_LEVEL is available
fn load_env_config() {
    let env_path = std::path::Path::new("C:\\ProgramData\\Professional SMART\\config\\.env");

    // Try to load the .env file - errors are non-fatal since defaults will be used
    match std::fs::metadata(env_path) {
        Ok(metadata) => {
            if let Ok(content) = std::fs::read_to_string(env_path) {
                // Use eprintln since logging isn't initialized yet
                eprintln!("[PRE-LOG] .env file found - {} bytes, {} lines",
                    metadata.len(), content.lines().count());

                match dotenvy::from_path(env_path) {
                    Ok(_) => {
                        eprintln!("[PRE-LOG] Successfully loaded configuration from {}", env_path.display());
                    }
                    Err(e) => {
                        eprintln!("[PRE-LOG] Failed to load .env file: {}", e);
                    }
                }
            }
        }
        Err(e) => {
            eprintln!("[PRE-LOG] .env file not found at {}: {}", env_path.display(), e);
        }
    }
}

/// Get the configured log level from environment
/// Checks LOG_LEVEL first, then RUST_LOG, defaults to "info"
fn get_log_level() -> String {
    // Check LOG_LEVEL first (simple user-friendly option)
    if let Ok(level) = std::env::var("LOG_LEVEL") {
        let level = level.to_lowercase();
        // Validate the level
        match level.as_str() {
            "trace" | "debug" | "info" | "warn" | "error" => return level,
            _ => {
                eprintln!("[PRE-LOG] Invalid LOG_LEVEL '{}', using 'info'. Valid values: trace, debug, info, warn, error", level);
            }
        }
    }

    // Fall back to RUST_LOG (more advanced option for per-module control)
    if let Ok(rust_log) = std::env::var("RUST_LOG") {
        return rust_log;
    }

    // Default to info
    "info".to_string()
}

/// Initialize logging for service mode
fn init_service_logging() -> Result<tracing_appender::non_blocking::WorkerGuard> {
    use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

    // Log to file in ProgramData directory
    let log_dir = std::path::Path::new("C:\\ProgramData\\Professional SMART\\logs");
    std::fs::create_dir_all(log_dir)
        .context("Failed to create log directory")?;

    let file_appender = tracing_appender::rolling::daily(log_dir, "service.log");
    let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);

    // Get configured log level (from LOG_LEVEL or RUST_LOG env vars)
    let log_level = get_log_level();
    eprintln!("[PRE-LOG] Initializing logging with level: {}", log_level);

    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_new(&log_level)
                .unwrap_or_else(|_| "info".into()),
        )
        .with(tracing_subscriber::fmt::layer().with_writer(non_blocking))
        .init();

    // Return guard so caller can keep it alive
    Ok(guard)
}

/// Move file to processed directory after successful Stage 1 ingestion
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

/// Move file to error directory after Stage 1 failure
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
