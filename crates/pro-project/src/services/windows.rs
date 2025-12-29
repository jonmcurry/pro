use anyhow::{Context, Result, bail};
use std::time::Duration;
use std::thread;

#[cfg(windows)]
use windows_service::{
    service::{ServiceAccess, ServiceState},
    service_manager::{ServiceManager, ServiceManagerAccess},
};

const SERVICE_NAME: &str = "ProfessionalSMART";
const SERVICE_TIMEOUT: Duration = Duration::from_secs(30);

/// Service for managing the ProfessionalSMART Windows service
pub struct WindowsServiceManager;

impl WindowsServiceManager {
    /// Get the current service status
    #[cfg(windows)]
    pub fn get_status() -> Result<ServiceStatus> {
        let manager = ServiceManager::local_computer(
            None::<&str>,
            ServiceManagerAccess::CONNECT,
        )
        .context("Failed to connect to service manager")?;

        let service = match manager.open_service(
            SERVICE_NAME,
            ServiceAccess::QUERY_STATUS,
        ) {
            Ok(s) => s,
            Err(_) => return Ok(ServiceStatus::NotInstalled),
        };

        let status = service
            .query_status()
            .context("Failed to query service status")?;

        Ok(match status.current_state {
            ServiceState::Running => ServiceStatus::Running,
            ServiceState::Stopped => ServiceStatus::Stopped,
            ServiceState::StartPending => ServiceStatus::Starting,
            ServiceState::StopPending => ServiceStatus::Stopping,
            ServiceState::ContinuePending => ServiceStatus::Starting,
            ServiceState::PausePending => ServiceStatus::Stopping,
            ServiceState::Paused => ServiceStatus::Stopped,
        })
    }

    #[cfg(not(windows))]
    pub fn get_status() -> Result<ServiceStatus> {
        Ok(ServiceStatus::NotInstalled)
    }

    /// Stop the service
    #[cfg(windows)]
    pub fn stop() -> Result<()> {
        let manager = ServiceManager::local_computer(
            None::<&str>,
            ServiceManagerAccess::CONNECT,
        )
        .context("Failed to connect to service manager")?;

        let service = manager
            .open_service(
                SERVICE_NAME,
                ServiceAccess::STOP | ServiceAccess::QUERY_STATUS,
            )
            .context("Failed to open service")?;

        let status = service.query_status()?;
        if status.current_state == ServiceState::Stopped {
            return Ok(());
        }

        service.stop().context("Failed to stop service")?;

        // Wait for stopped state
        Self::wait_for_state(&service, ServiceState::Stopped, SERVICE_TIMEOUT)?;

        Ok(())
    }

    #[cfg(not(windows))]
    pub fn stop() -> Result<()> {
        println!("Service control not available on this platform");
        Ok(())
    }

    /// Start the service
    #[cfg(windows)]
    pub fn start() -> Result<()> {
        let manager = ServiceManager::local_computer(
            None::<&str>,
            ServiceManagerAccess::CONNECT,
        )
        .context("Failed to connect to service manager")?;

        let service = manager
            .open_service(
                SERVICE_NAME,
                ServiceAccess::START | ServiceAccess::QUERY_STATUS,
            )
            .context("Failed to open service")?;

        let status = service.query_status()?;
        if status.current_state == ServiceState::Running {
            return Ok(());
        }

        service
            .start::<&str>(&[])
            .context("Failed to start service")?;

        // Wait for running state
        Self::wait_for_state(&service, ServiceState::Running, SERVICE_TIMEOUT)?;

        Ok(())
    }

    #[cfg(not(windows))]
    pub fn start() -> Result<()> {
        println!("Service control not available on this platform");
        Ok(())
    }

    /// Restart the service (stop then start)
    /// Reserved for future service management UI
    #[allow(dead_code)]
    pub fn restart() -> Result<()> {
        Self::stop()?;
        thread::sleep(Duration::from_secs(2));
        Self::start()?;
        Ok(())
    }

    /// Wait for a specific service state
    #[cfg(windows)]
    fn wait_for_state(
        service: &windows_service::service::Service,
        target_state: ServiceState,
        timeout: Duration,
    ) -> Result<()> {
        let start = std::time::Instant::now();
        let poll_interval = Duration::from_millis(500);

        loop {
            let status = service.query_status()?;
            if status.current_state == target_state {
                return Ok(());
            }

            if start.elapsed() > timeout {
                bail!(
                    "Timeout waiting for service to reach {:?} state",
                    target_state
                );
            }

            thread::sleep(poll_interval);
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServiceStatus {
    Running,
    Stopped,
    Starting,
    Stopping,
    NotInstalled,
}

impl std::fmt::Display for ServiceStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ServiceStatus::Running => write!(f, "Running"),
            ServiceStatus::Stopped => write!(f, "Stopped"),
            ServiceStatus::Starting => write!(f, "Starting"),
            ServiceStatus::Stopping => write!(f, "Stopping"),
            ServiceStatus::NotInstalled => write!(f, "Not Installed"),
        }
    }
}
