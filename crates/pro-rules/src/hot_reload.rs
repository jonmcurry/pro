// Hot Reload Infrastructure for Rules Engine
//
// Provides mechanisms to reload rules from the database without restarting the service.
// Includes signal handling (SIGHUP), atomic rule engine swapping, and cache invalidation.

use crate::loader::{load_rules_from_database, query_global_rules, instantiate_rule};
use crate::rule_engine::RuleEngine;
use pro_common::{Error, Result};
use sqlx::{PgPool, Row};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, error};

/// Reload coordinator that manages rule engine hot reloading
pub struct ReloadCoordinator {
    pool: PgPool,
    rule_engine: Arc<RwLock<RuleEngine>>,
    encryption_key: String,
}

impl ReloadCoordinator {
    /// Create a new reload coordinator
    pub fn new(pool: PgPool, rule_engine: Arc<RwLock<RuleEngine>>, encryption_key: String) -> Self {
        Self {
            pool,
            rule_engine,
            encryption_key,
        }
    }

    /// Reload rules from the database
    ///
    /// This operation:
    /// 1. Loads new rules from database
    /// 2. Invalidates the rule cache
    /// 3. Atomically replaces the rules in the engine
    ///
    /// Returns the number of rules loaded, or an error if reload failed.
    pub async fn reload_rules(&self) -> Result<usize> {
        info!("Starting rule engine reload...");

        // Set encryption key temporarily for loading
        std::env::set_var("RULE_ENCRYPTION_KEY", &self.encryption_key);

        // Load new rules from database (engine is rebuilt below with fresh instantiation)
        let (_new_engine, loaded_rules) = match load_rules_from_database(&self.pool, None).await {
            Ok(result) => result,
            Err(e) => {
                error!("Failed to load rules from database: {}", e);
                return Err(Error::Config(format!("Failed to load rules: {}", e)));
            }
        };

        let rule_count = loaded_rules.len();
        info!("Loaded {} rule(s) from database", rule_count);

        // Extract rules from the new engine
        // Since we can't easily extract rules from RuleEngine, we'll reload them directly
        // This is a bit inefficient but ensures proper instantiation
        let mut new_rules = Vec::new();

        // Re-query and instantiate rules
        let encryption_key_clone = self.encryption_key.clone();
        let rows = query_global_rules(&self.pool, &encryption_key_clone).await?;

        for row in rows {
            let rule_code: String = row.get("rule_code");
            let template_code: Option<String> = row.get("template_code");

            match instantiate_rule(&rule_code, &template_code, &row) {
                Ok(rule) => new_rules.push(rule),
                Err(e) => {
                    warn!("Failed to instantiate rule {}: {} - skipping", rule_code, e);
                }
            }
        }

        // Atomically replace rules in the engine (write lock ensures no rules are executing)
        {
            let mut engine = self.rule_engine.write().await;
            engine.replace_rules(new_rules);
            info!("Rules replaced successfully");
        }

        // Invalidate cache after swap
        self.invalidate_cache().await?;

        info!("Rule engine reload complete - {} rule(s) loaded", rule_count);
        Ok(rule_count)
    }

    /// Invalidate the rule result cache
    ///
    /// Clears all cached rule results to ensure new rules are executed
    /// on the next evaluation cycle.
    async fn invalidate_cache(&self) -> Result<()> {
        // Get write lock to clear cache
        let mut engine = self.rule_engine.write().await;
        engine.clear_cache();
        info!("Rule cache invalidated");
        Ok(())
    }

    /// Get current rule count (for monitoring/logging)
    pub async fn get_rule_count(&self) -> usize {
        let engine = self.rule_engine.read().await;
        engine.rule_count()
    }
}

/// Signal handler setup for hot reload
///
/// On Windows, we can't use SIGHUP, so we'll use a file-based trigger instead.
/// On Unix, we use SIGHUP signal.
#[cfg(unix)]
pub async fn setup_reload_signal(coordinator: Arc<ReloadCoordinator>) -> Result<()> {
    use tokio::signal::unix::{signal, SignalKind};

    let mut sighup = signal(SignalKind::hangup())
        .map_err(|e| Error::Config(format!("Failed to setup SIGHUP handler: {}", e)))?;

    tokio::spawn(async move {
        loop {
            sighup.recv().await;
            info!("Received SIGHUP signal - reloading rules...");

            match coordinator.reload_rules().await {
                Ok(count) => info!("Successfully reloaded {} rules", count),
                Err(e) => error!("Failed to reload rules: {}", e),
            }
        }
    });

    info!("SIGHUP signal handler installed - send SIGHUP to reload rules");
    Ok(())
}

/// Windows file-based reload trigger
///
/// Watches for a file named "reload_rules.trigger" in the data directory.
/// When detected, reloads rules and deletes the trigger file.
#[cfg(windows)]
pub async fn setup_reload_signal(coordinator: Arc<ReloadCoordinator>) -> Result<()> {
    use std::path::PathBuf;
    use tokio::fs;
    use tokio::time::{interval, Duration};

    // Get data directory from environment or use default
    let data_dir = std::env::var("PROFESSIONAL_SMART_DATA_DIR")
        .unwrap_or_else(|_| r"C:\ProgramData\Professional SMART".to_string());

    let trigger_path = PathBuf::from(data_dir).join("reload_rules.trigger");

    info!("File-based reload trigger enabled - create '{}' to reload rules", trigger_path.display());

    tokio::spawn(async move {
        let mut check_interval = interval(Duration::from_secs(5));

        loop {
            check_interval.tick().await;

            // Check if trigger file exists
            if trigger_path.exists() {
                info!("Reload trigger file detected - reloading rules...");

                match coordinator.reload_rules().await {
                    Ok(count) => {
                        info!("Successfully reloaded {} rules", count);

                        // Delete trigger file
                        if let Err(e) = fs::remove_file(&trigger_path).await {
                            warn!("Failed to delete trigger file: {}", e);
                        }
                    }
                    Err(e) => {
                        error!("Failed to reload rules: {}", e);

                        // Still delete trigger file to avoid repeated failures
                        if let Err(e) = fs::remove_file(&trigger_path).await {
                            warn!("Failed to delete trigger file: {}", e);
                        }
                    }
                }
            }
        }
    });

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reload_coordinator_creation() {
        // This is a basic compilation test
        // Full integration tests would require a database connection
    }
}
