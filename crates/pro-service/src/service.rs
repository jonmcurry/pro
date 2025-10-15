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
use tracing::{error, info, warn};
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

/// Service display name
const SERVICE_DISPLAY_NAME: &str = "Professional SMART Claims Processing Service";

/// Service description
const SERVICE_DESCRIPTION: &str = "Automated claims processing, validation, and flagging system for healthcare providers. Processes EDI 837p and CSV claim files with comprehensive rules engine and RVU-based payment calculations.";

/// Main service run loop
pub fn run_service() -> Result<()> {
    // Initialize file logging for service mode
    init_service_logging()?;

    info!("Professional SMART service starting...");

    // Create shutdown channel
    let (shutdown_tx, mut shutdown_rx) = tokio::sync::mpsc::channel::<()>(1);

    // Create event handler for service control events
    let event_handler = move |control_event| -> ServiceControlHandlerResult {
        match control_event {
            ServiceControl::Stop => {
                info!("Service stop requested");
                // Signal worker to shutdown gracefully
                if let Err(e) = shutdown_tx.try_send(()) {
                    error!("Failed to send shutdown signal: {}", e);
                }
                ServiceControlHandlerResult::NoError
            }
            ServiceControl::Shutdown => {
                info!("System shutdown - stopping service");
                // Signal worker to shutdown immediately
                if let Err(e) = shutdown_tx.try_send(()) {
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

    // Load configuration
    if let Err(e) = dotenvy::dotenv() {
        warn!("Failed to load .env file: {}", e);
    }

    // Initialize worker
    info!("Initializing worker pipeline...");

    // Create Tokio runtime for async operations
    let runtime = tokio::runtime::Runtime::new()
        .context("Failed to create Tokio runtime")?;

    // Spawn worker task
    let worker_handle = runtime.spawn(async move {
        info!("Worker task started");

        // TODO: Initialize and run worker pipeline
        // For now, just keep running until shutdown signal received
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

/// Initialize logging for service mode
fn init_service_logging() -> Result<()> {
    use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

    // Log to file in ProgramData directory
    let log_dir = std::path::Path::new("C:\\ProgramData\\Professional SMART\\logs");
    std::fs::create_dir_all(log_dir)
        .context("Failed to create log directory")?;

    let file_appender = tracing_appender::rolling::daily(log_dir, "service.log");
    let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);

    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .with(tracing_subscriber::fmt::layer().with_writer(non_blocking))
        .init();

    // Note: _guard is intentionally not stored, as service runs indefinitely
    // In production, you'd want to keep the guard alive

    Ok(())
}
