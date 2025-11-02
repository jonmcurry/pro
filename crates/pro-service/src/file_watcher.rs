//! File watcher for automatic file processing
//!
//! Monitors the input directory for new CSV and EDI files and processes them automatically.
//! - CSV files: Master data (organizations, facilities, providers)
//! - EDI files: 837p claims data (.edi and .837p extensions)

use anyhow::{Context, Result};
use notify::{Config, Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use std::path::{Path, PathBuf};
use std::time::Duration;
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver};
use tokio::time::sleep;
use tracing::{debug, error, info, warn};

/// File watcher that monitors a directory for new files
pub struct FileWatcher {
    input_dir: PathBuf,
    processed_dir: PathBuf,
    error_dir: PathBuf,
    _watcher: RecommendedWatcher,
    receiver: UnboundedReceiver<Result<Event, notify::Error>>,
}

impl FileWatcher {
    /// Create a new file watcher
    pub fn new(input_dir: impl AsRef<Path>) -> Result<Self> {
        let input_dir = input_dir.as_ref().to_path_buf();

        // Derive processed and error directories from input directory
        let parent = input_dir.parent()
            .context("Input directory must have a parent")?;
        let processed_dir = parent.join("processed");
        let error_dir = parent.join("error");

        // Ensure directories exist
        std::fs::create_dir_all(&input_dir)
            .context("Failed to create input directory")?;
        std::fs::create_dir_all(&processed_dir)
            .context("Failed to create processed directory")?;
        std::fs::create_dir_all(&error_dir)
            .context("Failed to create error directory")?;

        info!("Initializing file watcher");
        info!("  Input directory: '{}'", input_dir.display());
        info!("  Processed directory: '{}'", processed_dir.display());
        info!("  Error directory: '{}'", error_dir.display());

        // Create channel for file system events (using tokio's unbounded channel)
        let (tx, rx) = unbounded_channel();

        // Create watcher
        let mut watcher = RecommendedWatcher::new(
            move |res| {
                if let Err(e) = tx.send(res) {
                    error!("Failed to send file watcher event: {}", e);
                }
            },
            Config::default()
                .with_poll_interval(Duration::from_secs(2))
        )?;

        // Start watching the input directory
        watcher.watch(&input_dir, RecursiveMode::NonRecursive)
            .context("Failed to start watching input directory")?;

        info!("File watcher started successfully");

        Ok(Self {
            input_dir,
            processed_dir,
            error_dir,
            _watcher: watcher,
            receiver: rx,
        })
    }

    /// Run the file watcher loop
    pub async fn run<F, Fut>(&mut self, mut process_file: F) -> Result<()>
    where
        F: FnMut(PathBuf) -> Fut + Send,
        Fut: std::future::Future<Output = Result<()>> + Send,
    {
        info!("File watcher is now monitoring for new files...");

        // Process any existing files in the directory first
        self.process_existing_files(&mut process_file).await?;

        // Then start monitoring for new files
        loop {
            match self.receiver.try_recv() {
                Ok(Ok(event)) => {
                    self.handle_event(event, &mut process_file).await?;
                }
                Ok(Err(e)) => {
                    error!("File watcher error: {}", e);
                }
                Err(tokio::sync::mpsc::error::TryRecvError::Empty) => {
                    // No events, sleep briefly
                    sleep(Duration::from_millis(500)).await;
                }
                Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => {
                    error!("File watcher channel disconnected");
                    return Err(anyhow::anyhow!("File watcher channel disconnected"));
                }
            }
        }
    }

    /// Process any files that already exist in the input directory
    async fn process_existing_files<F, Fut>(&self, process_file: &mut F) -> Result<()>
    where
        F: FnMut(PathBuf) -> Fut,
        Fut: std::future::Future<Output = Result<()>>,
    {
        info!("Scanning for existing files in input directory...");

        let entries = std::fs::read_dir(&self.input_dir)
            .context("Failed to read input directory")?;

        let mut file_count = 0;
        for entry in entries {
            let entry = entry?;
            let path = entry.path();

            if path.is_file() && self.is_processable_file(&path) {
                info!("Found existing file: {}", path.display());
                file_count += 1;

                match process_file(path.clone()).await {
                    Ok(_) => {
                        info!("Successfully processed existing file: {}", path.display());
                        self.move_to_processed(&path)?;
                    }
                    Err(e) => {
                        // Check if this is a special SKIP_MOVE error
                        if e.to_string().contains("SKIP_MOVE") {
                            info!("Existing file enqueued, staying in place for processor: {}", path.display());
                        } else {
                            error!("Failed to process existing file {}: {}", path.display(), e);
                            self.move_to_error(&path, &e.to_string())?;
                        }
                    }
                }
            }
        }

        if file_count > 0 {
            info!("Processed {} existing file(s)", file_count);
        } else {
            info!("No existing files found to process");
        }

        Ok(())
    }

    /// Handle a file system event
    async fn handle_event<F, Fut>(&self, event: Event, process_file: &mut F) -> Result<()>
    where
        F: FnMut(PathBuf) -> Fut,
        Fut: std::future::Future<Output = Result<()>>,
    {
        match event.kind {
            EventKind::Create(_) | EventKind::Modify(_) => {
                for path in event.paths {
                    if path.is_file() && self.is_processable_file(&path) {
                        debug!("Detected file: {}", path.display());

                        // Wait briefly to ensure file is fully written
                        sleep(Duration::from_millis(500)).await;

                        info!("Processing new file: {}", path.display());

                        match process_file(path.clone()).await {
                            Ok(_) => {
                                info!("Successfully processed file: {}", path.display());
                                self.move_to_processed(&path)?;
                            }
                            Err(e) => {
                                // Check if this is a special SKIP_MOVE error
                                // (used when file is enqueued but should stay in place for later processing)
                                if e.to_string().contains("SKIP_MOVE") {
                                    info!("File enqueued, staying in place for processor: {}", path.display());
                                } else {
                                    error!("Failed to process file {}: {}", path.display(), e);
                                    self.move_to_error(&path, &e.to_string())?;
                                }
                            }
                        }
                    }
                }
            }
            _ => {
                // Ignore other event types (delete, remove, etc.)
            }
        }

        Ok(())
    }

    /// Check if a file should be processed
    fn is_processable_file(&self, path: &Path) -> bool {
        if let Some(ext) = path.extension() {
            let ext_lower = ext.to_string_lossy().to_lowercase();
            // Process CSV files (master data) and EDI files (837p claims with .edi or .837p extension)
            if ext_lower == "csv" || ext_lower == "edi" || ext_lower == "837p" {
                return true;
            }
        }
        false
    }

    /// Move file to processed directory
    fn move_to_processed(&self, file_path: &Path) -> Result<()> {
        let filename = file_path.file_name()
            .context("Failed to get filename")?;

        let dest = self.processed_dir.join(filename);

        debug!("Moving {} to {}", file_path.display(), dest.display());

        std::fs::rename(file_path, &dest)
            .context("Failed to move file to processed directory")?;

        info!("Moved file to processed directory: {}", dest.display());

        Ok(())
    }

    /// Move file to error directory with error details
    fn move_to_error(&self, file_path: &Path, error_message: &str) -> Result<()> {
        let filename = file_path.file_name()
            .context("Failed to get filename")?;

        let dest = self.error_dir.join(filename);

        warn!("Moving failed file {} to {}", file_path.display(), dest.display());

        std::fs::rename(file_path, &dest)
            .context("Failed to move file to error directory")?;

        // Write error message to a companion .error file
        let error_file = dest.with_extension("error");
        std::fs::write(&error_file, error_message)
            .context("Failed to write error file")?;

        error!("Moved file to error directory: {}", dest.display());
        error!("Error details written to: {}", error_file.display());

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_file_watcher_creation() {
        let temp_dir = TempDir::new().unwrap();
        let input_dir = temp_dir.path().join("input");

        let watcher = FileWatcher::new(&input_dir);
        assert!(watcher.is_ok());

        // Verify directories were created
        assert!(input_dir.exists());
        assert!(temp_dir.path().join("processed").exists());
        assert!(temp_dir.path().join("error").exists());
    }

    #[test]
    fn test_is_processable_file() {
        let temp_dir = TempDir::new().unwrap();
        let input_dir = temp_dir.path().join("input");
        let watcher = FileWatcher::new(&input_dir).unwrap();

        // CSV files (master data)
        assert!(watcher.is_processable_file(Path::new("test.csv")));
        assert!(watcher.is_processable_file(Path::new("test.CSV")));

        // EDI files (837p claims - both .edi and .837p extensions)
        assert!(watcher.is_processable_file(Path::new("claims.edi")));
        assert!(watcher.is_processable_file(Path::new("claims.EDI")));
        assert!(watcher.is_processable_file(Path::new("claims.837p")));
        assert!(watcher.is_processable_file(Path::new("claims.837P")));

        // Not processable
        assert!(!watcher.is_processable_file(Path::new("test.txt")));
        assert!(!watcher.is_processable_file(Path::new("test.xlsx")));
    }
}
