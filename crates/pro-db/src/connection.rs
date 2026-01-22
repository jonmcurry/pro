use sqlx::postgres::{PgPool, PgPoolOptions};
use sqlx::Error as SqlxError;
use std::time::Duration;

pub type DbPool = PgPool;

/// Database connection configuration
#[derive(Debug, Clone)]
pub struct DbConfig {
    pub database_url: String,
    pub max_connections: u32,
    pub min_connections: u32,
    pub connection_timeout_seconds: u64,
    pub idle_timeout_seconds: u64,
    pub max_lifetime_seconds: u64,

    // PHASE 5: Performance tuning
    /// Number of prepared statements to cache per connection (default: 100)
    /// Higher values improve performance for repeated queries but use more memory
    pub statement_cache_capacity: u32,

    /// Statement timeout in seconds (default: 30)
    /// Prevents slow queries from blocking the connection pool
    pub statement_timeout_seconds: u64,

    /// Test connections before acquiring from pool (default: true)
    /// Ensures connections are valid, adds small overhead but prevents errors
    pub test_before_acquire: bool,
}

impl Default for DbConfig {
    fn default() -> Self {
        Self {
            database_url: std::env::var("DATABASE_URL")
                .unwrap_or_else(|_| "postgresql://localhost/professional_smart".to_string())
                .trim()
                .to_string(),
            // Connection pool sizing - balance between throughput and resource usage
            // 75 connections supports 12 workers with MAX_CONCURRENT_ENCOUNTERS=24
            max_connections: std::env::var("DB_MAX_CONNECTIONS")
                .ok()
                .and_then(|s| s.parse::<u32>().ok())
                .unwrap_or(75),  // Default: 75 for high parallel throughput
            // PERFORMANCE: min_connections=0 allows pool to fully shrink when idle
            // This releases connections back to PostgreSQL for web app usage
            // Connections are created on-demand when processing starts
            min_connections: std::env::var("DB_MIN_CONNECTIONS")
                .ok()
                .and_then(|s| s.parse::<u32>().ok())
                .unwrap_or(0),  // Default: 0 (allow full shrinkage after batch processing)
            connection_timeout_seconds: 30,
            // Idle timeout - how long to keep unused connections open
            // Balance between releasing connections and avoiding reconnection overhead
            // 60s allows connections to stay warm during batch processing
            // but still releases them reasonably quickly after processing completes
            idle_timeout_seconds: std::env::var("DB_IDLE_TIMEOUT")
                .ok()
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(60),  // Default: 60 seconds (balance between performance and release)
            max_lifetime_seconds: 1800,

            // PHASE 5: Performance defaults
            statement_cache_capacity: 100,  // Cache up to 100 prepared statements per connection
            statement_timeout_seconds: 30,   // 30 second timeout for slow queries
            // PERFORMANCE: Disabled test_before_acquire to reduce connection overhead
            // PostgreSQL connections are reliable; testing adds latency per query
            test_before_acquire: false,
        }
    }
}

/// Create a database connection pool with the given configuration
pub async fn create_pool(config: &DbConfig) -> Result<DbPool, SqlxError> {
    // Build connection URL with performance settings
    let mut conn_url = config.database_url.clone();

    // Add statement_cache_size parameter for prepared statement caching
    let separator = if conn_url.contains('?') { '&' } else { '?' };
    conn_url.push_str(&format!(
        "{}statement_cache_size={}&statement_timeout={}s&application_name=pro-smart",
        separator,
        config.statement_cache_capacity,
        config.statement_timeout_seconds
    ));

    // Create pool with configuration
    let pool = PgPoolOptions::new()
        .max_connections(config.max_connections)
        .min_connections(config.min_connections)
        .acquire_timeout(Duration::from_secs(config.connection_timeout_seconds))
        .idle_timeout(Duration::from_secs(config.idle_timeout_seconds))
        .max_lifetime(Duration::from_secs(config.max_lifetime_seconds))
        .test_before_acquire(config.test_before_acquire)
        .connect(&conn_url)
        .await?;

    // PHASE 5: Warm up minimum connections with test queries
    // This ensures connections are ready and the prepared statement cache is initialized
    let warmup_tasks: Vec<_> = (0..config.min_connections)
        .map(|_| {
            let pool_clone = pool.clone();
            async move {
                if let Ok(mut conn) = pool_clone.acquire().await {
                    // Execute simple query to warm up connection
                    let _ = sqlx::query("SELECT 1").fetch_one(&mut *conn).await;
                }
            }
        })
        .collect();

    // Wait for all warmup tasks to complete (with timeout)
    let warmup_timeout = Duration::from_secs(5);
    let _ = tokio::time::timeout(
        warmup_timeout,
        futures::future::join_all(warmup_tasks)
    ).await;

    Ok(pool)
}

/// Create a database connection pool with default configuration
pub async fn create_pool_default() -> Result<DbPool, SqlxError> {
    let config = DbConfig::default();
    create_pool(&config).await
}

/// Test database connection
pub async fn test_connection(pool: &DbPool) -> Result<(), SqlxError> {
    sqlx::query("SELECT 1")
        .fetch_one(pool)
        .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore] // Requires database to be running
    async fn test_create_pool() {
        let config = DbConfig::default();
        let pool = create_pool(&config).await;
        assert!(pool.is_ok());
    }

    #[tokio::test]
    #[ignore] // Requires database to be running
    async fn test_connection_works() {
        let pool = create_pool_default().await.unwrap();
        let result = test_connection(&pool).await;
        assert!(result.is_ok());
    }
}
